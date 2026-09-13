"""Small CUDA/CPU-kinematics differential checks, with zero CPU physics steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from puffysics_backend import PuffysicsBackend
from puffysics_semantic_physics import PuffysicsSemanticPhysics


def verify(args):
    import mujoco

    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    kwargs = dict(arenas=4, device="cuda:0", solver_mode=args.solver_mode, export_path=args.model_export)
    controls = None
    original = PuffysicsBackend(config, library=args.original_library, **kwargs)
    traces = []
    try:
        initial = original.qpos.clone()
        joint_indices = torch.as_tensor(np.stack(original.joint_qpos), device="cuda:0")
        controls = original.qpos[:, joint_indices].reshape(4, 58).clone()
        perturb = torch.sin(torch.arange(58, device="cuda:0", dtype=torch.float32)) * 0.001
        for tick in range(12):
            original.ctrl.copy_(controls + perturb * tick)
            original.step()
            traces.append((original.qpos.clone(), original.qvel.clone()))
        original_status = original.stats()
    finally:
        original.close()

    physics = PuffysicsSemanticPhysics(config, library=args.library, **kwargs)
    result = {"schema": "rek.puffysics_semantic_adapter_verification.v1", "cpu_physics_steps": 0,
              "solver_mode": args.solver_mode,
              "native_step_function": "b3_step" if args.solver_mode == 0 else "rp_art_step",
              "engine_parity_required": False}
    try:
        errors = []
        bitwise = True
        for tick, (expected_q, expected_v) in enumerate(traces):
            physics.ctrl.copy_(controls + perturb * tick)
            physics.step()
            bitwise &= torch.equal(expected_q, physics.qpos) and torch.equal(expected_v, physics.qvel)
            errors.append(max(float((expected_q-physics.qpos).abs().max().item()),
                              float((expected_v-physics.qvel).abs().max().item())))
        result["original_v8_step_bitwise_equal"] = bitwise
        result["original_v8_step_max_abs_error"] = max(errors)
        result["original_status"] = original_status
        if not bitwise:
            raise AssertionError("semantic export changed pinned solver stepping")
        snapshots = {name: value.clone() for name, value in physics._fields.items()}
        physics.forward_selected(torch.zeros(4, dtype=torch.bool, device="cuda:0"))
        unchanged = {name: torch.equal(value, physics._fields[name]) for name, value in snapshots.items()}
        result["zero_mask_fields_unchanged"] = unchanged
        if not all(unchanged.values()):
            raise AssertionError("zero-mask forward changed published fields")

        mask = torch.tensor([True, False, True, False], device="cuda:0")
        requested_q = initial.clone()
        requested_q[:, joint_indices] += torch.sin(torch.arange(58, device="cuda:0")).reshape(2, 29) * 0.015
        requested_q[:, 0] += 0.015
        requested_q[:, 36] -= 0.013
        requested_v = torch.sin(torch.arange(70, device="cuda:0", dtype=torch.float32))[None].expand(4,-1).contiguous() * 0.03
        physics.qpos[mask] = requested_q[mask]
        physics.qvel[mask] = requested_v[mask]
        before_q, before_v = physics.qpos.clone(), physics.qvel.clone()
        before_time = physics.time.clone()
        physics.forward_selected(mask)
        result["masked_reset_unselected_qpos_unchanged"] = torch.equal(before_q[~mask], physics.qpos[~mask])
        result["masked_reset_unselected_qvel_unchanged"] = torch.equal(before_v[~mask], physics.qvel[~mask])
        result["masked_reset_clock_unchanged"] = torch.equal(before_time, physics.time)
        result["generalized_qpos_roundtrip_max_abs_error"] = float((physics.qpos[mask]-requested_q[mask]).abs().max().item())
        result["generalized_qvel_roundtrip_max_abs_error"] = float((physics.qvel[mask]-requested_v[mask]).abs().max().item())
        source = mujoco.MjData(physics.host_model)
        source.qpos[:] = requested_q[0].cpu().numpy()
        source.qvel[:] = requested_v[0].cpu().numpy()
        mujoco.mj_kinematics(physics.host_model, source)
        mujoco.mj_comPos(physics.host_model, source)
        mujoco.mj_comVel(physics.host_model, source)
        differential = {}
        for native, host in (("xpos", "xpos"), ("xipos", "xipos"), ("xmat", "xmat"),
                             ("ximat", "ximat"), ("com", "subtree_com"), ("cvel", "cvel"),
                             ("geom_xpos", "geom_xpos"), ("geom_xmat", "geom_xmat")):
            actual = physics._fields[native][0].cpu().numpy().reshape(-1)
            expected = np.asarray(getattr(source, host)).reshape(-1)
            differential[native] = float(np.max(np.abs(actual-expected)))
        result["mujoco_kinematics_max_abs_error"] = differential
        # Isolate original counted-fall sequencing without running combat.
        from gpu_duel_reset import GpuDuelReset
        from gpu_motion_assets import GpuMotionAssets
        assets = GpuMotionAssets(Path(config["assets"]), config["assets_sha256"], "cuda:0")
        reset = GpuDuelReset(physics, assets)
        reset.full(torch.ones(4,dtype=torch.bool,device="cuda:0"), reset_clock=True)
        result["full_idle_reset_max_abs_error"] = float((physics.qpos-initial).abs().max().item())
        physics.qvel.copy_(requested_v)
        physics.forward_selected(torch.ones(4,dtype=torch.bool,device="cuda:0"))
        roots_before = physics.qvel.reshape(4,2,35)[...,:6].clone()
        reset.begin(mask)
        reset.complete(mask)
        result["counted_reset_preserves_free_velocity_max_abs_error"] = float(
            (physics.qvel.reshape(4,2,35)[...,:6]-roots_before).abs().max().item())
        result["counted_reset_joint_qpos_max_abs"] = float(physics.qpos[mask][:,joint_indices].abs().max().item())
        result["counted_reset_joint_qvel_max_abs"] = float(
            physics.qvel[mask][:,torch.as_tensor(np.stack(physics.joint_qvel),device="cuda:0")].abs().max().item())
        from gpu_observation import GpuObservationAssembler
        from gpu_combat_measurement import RekG1GpuCombatMeasurement
        observer = GpuObservationAssembler(physics)
        observer.gather_kinematics()
        measurement = RekG1GpuCombatMeasurement.from_physics(physics)
        measurement.reset()
        measurement.sample(0, torch.zeros(8,dtype=torch.uint8,device="cuda:0"))
        result["unchanged_observation_and_measurement_constructed"] = True
        if config.get("fused_combat_library"):
            from gpu_combat_measurement_fused import FusedGpuCombatMeasurement
            fused = FusedGpuCombatMeasurement.from_physics(physics, library=config["fused_combat_library"])
            fused.reset()
            fused.sample(0, torch.zeros(8,dtype=torch.uint8,device="cuda:0"))
            result["unchanged_fused_measurement_constructed"] = True
        result["native_status"] = physics.check_status()
        count = int(physics.data.nacon.item())
        result["real_native_manifold_contact_points"] = count
        result["real_contact_counts_per_arena"] = physics._fields["counts"].cpu().tolist()
        result["passed"] = (
            all(unchanged.values()) and all(differential[name] < 2e-5 for name in differential)
            and result["generalized_qpos_roundtrip_max_abs_error"] < 2e-5
            and result["generalized_qvel_roundtrip_max_abs_error"] < 2e-5
            and result["full_idle_reset_max_abs_error"] < 2e-5
            and result["counted_reset_preserves_free_velocity_max_abs_error"] < 2e-5
            and result["counted_reset_joint_qpos_max_abs"] < 2e-5
            and result["counted_reset_joint_qvel_max_abs"] < 2e-5
            and all(result[name] for name in ("masked_reset_unselected_qpos_unchanged",
                                              "masked_reset_unselected_qvel_unchanged", "masked_reset_clock_unchanged"))
        )
    except Exception as exc:
        result["passed"] = False
        result["error"] = repr(exc)
        raise
    finally:
        physics.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--original-library", type=Path, required=True)
    parser.add_argument("--model-export", type=Path, required=True)
    parser.add_argument("--solver-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if verify(parser.parse_args())["passed"] else 1)
