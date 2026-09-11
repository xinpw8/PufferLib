"""Isolated fixed-reference G1 controller/physics experiment.

Both backends run physics and the unchanged Sonic controller on CUDA. This
probe does not implement semantic combat, rewards, an opponent, or training.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

import gear_sonic_candidate as candidate
from gpu_actuator_drive import GpuActuatorDrive
from gpu_controller import GearSonicGpuController
from gpu_motion_assets import GpuMotionAssets
from gpu_robot_state import G1GpuControllerState


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MujocoBackend:
    def __init__(self, config, assets, *, arenas, device):
        from gpu_duel_physics import GpuDuelPhysics
        from gpu_duel_reset import GpuDuelReset
        from gpu_observation import GpuObservationAssembler

        self.physics = GpuDuelPhysics(
            Path(config["model"]), config["model_sha256"], arenas=arenas, device=device,
        )
        for name in ("qpos", "qvel", "ctrl", "time", "host_model", "actuator_ids",
                     "joint_qpos", "joint_qvel", "arenas"):
            setattr(self, name, getattr(self.physics, name))
        self._reset = GpuDuelReset(self.physics, assets)
        self._observer = GpuObservationAssembler(self.physics)
        self._all = torch.ones(arenas, device=device, dtype=torch.bool)
        self.initial_heading = self._reset.initial_heading
        self.base = self.qpos.reshape(-1, 36)[:, 3:7]
        self.angular_local = torch.zeros((arenas * 2, 3), device=device)
        self.metadata = {
            "backend": "mujoco_warp",
            "model_sha256": config["model_sha256"],
            "physics_timestep_seconds": 0.002,
        }

    def reset(self):
        self._reset.full(self._all, reset_clock=True)
        self.refresh_kinematics()

    def refresh_kinematics(self):
        self.angular_local.copy_(self._observer.gather_kinematics()[:, 10:13])

    def step(self):
        self.physics.step()

    def capture_context(self):
        return self.physics.wp.ScopedCapture(stream=self.physics.stream, external=True)


class FixedReferenceProbe:
    def __init__(self, config, args):
        self.controller = GearSonicGpuController.from_manifest(
            Path(config["controller_manifest"]), Path(config["controller_source"]),
            device=args.device,
        )
        self.rows = self.controller.batch_size
        if self.rows < 2 or self.rows % 2:
            raise ValueError("the explicit controller batch must contain paired fighters")
        self.arenas = self.rows // 2
        self.assets = GpuMotionAssets(
            Path(config["assets"]), config["assets_sha256"], args.device,
        )
        if args.role not in self.assets.roles:
            raise ValueError(f"unknown fixed-reference role {args.role!r}; available: {sorted(self.assets.roles)}")
        self.role = args.role
        self.loop = args.loop
        if args.backend == "puffysics":
            from puffysics_backend import PuffysicsBackend
            self.physics = PuffysicsBackend(
                config, arenas=self.arenas, device=args.device,
                solver_mode=args.solver_mode, library=args.library,
                export_path=args.model_export,
            )
        else:
            self.physics = MujocoBackend(
                config, self.assets, arenas=self.arenas, device=args.device,
            )
        self.state = G1GpuControllerState(self.rows, args.device)
        self.drive = GpuActuatorDrive.from_physics(self.physics)
        self.active = torch.ones(self.rows, device=args.device, dtype=torch.bool)
        self.cursor = torch.zeros(self.rows, device=args.device, dtype=torch.long)
        self.qindices = torch.as_tensor(np.stack(self.physics.joint_qpos), device=args.device)
        self.dqindices = torch.as_tensor(np.stack(self.physics.joint_qvel), device=args.device)
        self.initial_heading = self.physics.initial_heading
        expected_shapes = {
            "qpos": (self.arenas, 72), "qvel": (self.arenas, 70),
            "ctrl": (self.arenas, 58), "base": (self.rows, 4),
            "angular_local": (self.rows, 3), "time": (self.arenas,),
        }
        for name, shape in expected_shapes.items():
            value = getattr(self.physics, name)
            if tuple(value.shape) != shape or value.device != self.state.last_actions.device:
                raise ValueError(f"native {name} must be CUDA {shape} on the controller device")
            if value.dtype != torch.float32:
                raise ValueError(f"native {name} must be float32")
        self.reset()

    def reset(self):
        self.physics.reset()
        self.state.reset(self.active)
        self.drive.complete_reset(self.active)
        self.cursor.zero_()

    def step(self):
        if hasattr(self.physics, "refresh_kinematics"):
            self.physics.refresh_kinematics()
        q = self.physics.qpos[:, self.qindices].reshape(self.rows, 29)
        dq = self.physics.qvel[:, self.dqindices].reshape(self.rows, 29)
        reference, next_reference, roots = self.assets.fixed_reference(
            self.role, self.cursor, loop=self.loop,
        )
        observation = self.state.prepare(
            self.physics.base, self.physics.angular_local, q, dq,
            self.initial_heading, reference, next_reference, roots, self.active,
        )
        tokens = self.controller.encode(observation)
        actions = self.controller.decode(self.state.decoder_input(tokens))
        targets = self.state.apply_actions(actions, self.active)
        for substep in range(10):
            q = self.physics.qpos[:, self.qindices].reshape(self.rows, 29)
            dq = self.physics.qvel[:, self.dqindices].reshape(self.rows, 29)
            self.physics.ctrl.copy_(
                self.drive.prepare(targets, q, dq, substep=substep).reshape(self.arenas, 58)
            )
            self.physics.step()
        self.cursor.add_(1)

    def initial_mapping(self):
        model = self.physics.host_model
        expected = model.qpos0.copy()
        idle = self.assets.roles["idle"]
        reference = self.assets.host_arrays[idle["files"]["mujoco_joint_order"]][0]
        reference_root = self.assets.host_arrays[idle["files"]["xyzw"]][0][[3, 0, 1, 2]]
        headings = []
        for side in range(2):
            joints = model.actuator_trnid[self.physics.actuator_ids[side], 0]
            limited = model.jnt_limited[joints].astype(bool)
            limits = model.jnt_range[joints]
            initial = reference.copy()
            initial[limited] = np.clip(initial[limited], limits[limited, 0], limits[limited, 1])
            expected[self.physics.joint_qpos[side]] = initial
            headings.append(candidate.reference_heading_delta(
                expected[side * 36 + 3:side * 36 + 7], reference_root,
            ))
        actual = self.physics.qpos.detach().cpu().numpy()
        expected = np.tile(expected.astype(np.float32), (self.arenas, 1))
        expected_heading = np.tile(headings, (self.arenas, 1))
        actual_heading = self.initial_heading.detach().cpu().numpy()
        def max_error(left, right=0.0):
            difference = np.asarray(left) - right
            return float(np.max(np.abs(difference))) if np.isfinite(difference).all() else None

        qpos_error = max_error(actual, expected)
        heading_error = max_error(actual_heading, expected_heading)
        base = self.physics.base.detach().cpu().numpy()
        base_error = max_error(base, actual.reshape(self.rows, 36)[:, 3:7])
        velocity_error = max_error(self.physics.qvel.detach().cpu().numpy())
        angular_error = max_error(self.physics.angular_local.detach().cpu().numpy())
        result = {
            "qpos_max_abs_error": qpos_error, "heading_max_abs_error": heading_error,
            "base_wxyz_max_abs_error": base_error,
            "initial_qvel_max_abs": velocity_error,
            "initial_angular_local_max_abs": angular_error,
            "expected_qpos_sha256_float32": hashlib.sha256(expected.tobytes()).hexdigest(),
            "actual_qpos_sha256_float32": hashlib.sha256(actual.tobytes()).hexdigest(),
            "joint_qpos": np.asarray(self.physics.joint_qpos).tolist(),
            "joint_qvel": np.asarray(self.physics.joint_qvel).tolist(),
            "within_1e_minus_6": all(error is not None and error <= 1e-6 for error in (
                qpos_error, heading_error, base_error, velocity_error, angular_error,
            )),
        }
        return result


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = torch.cuda.Stream(device=args.device)
    stream.wait_stream(torch.cuda.current_stream(args.device))
    setup_start = time.perf_counter()
    with torch.cuda.stream(stream):
        runtime = FixedReferenceProbe(config, args)
        stream.synchronize()
        initial_mapping = runtime.initial_mapping()
        with (args.out / "initial_mapping.json").open("x", encoding="utf-8") as destination:
            json.dump(initial_mapping, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
        if not initial_mapping["within_1e_minus_6"]:
            print(json.dumps({"stage": "initial_mapping_failed", "physics_steps": 0,
                              "initial_mapping": initial_mapping}, indent=2, allow_nan=False))
            return 2
        for _ in range(args.warmup):
            runtime.step()
        stream.synchronize()
        graph = None
        if not args.eager:
            graph = torch.cuda.CUDAGraph()
            context = (runtime.physics.capture_context() if hasattr(runtime.physics, "capture_context")
                       else nullcontext())
            with torch.cuda.graph(graph, stream=stream):
                with context as native_capture:
                    runtime.step()
            runtime.native_capture = native_capture
        runtime.reset()
        stream.synchronize()
        mapping = runtime.initial_mapping()
        qpos_trace = torch.empty((args.steps + 1, runtime.arenas, 72), device=args.device)
        qvel_trace = torch.empty((args.steps + 1, runtime.arenas, 70), device=args.device)
        action_trace = torch.empty((args.steps, runtime.rows, 29), device=args.device)
        target_trace = torch.empty_like(action_trace)
        qpos_trace[0].copy_(runtime.physics.qpos)
        qvel_trace[0].copy_(runtime.physics.qvel)
        before_time = runtime.physics.time.clone()
        stream.synchronize()
        setup_seconds = time.perf_counter() - setup_start
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True, blocking=True)
        cpu_before = time.process_time()
        wall_before = time.perf_counter()
        start_event.record(stream)
        for tick in range(args.steps):
            runtime.step() if graph is None else graph.replay()
            qpos_trace[tick + 1].copy_(runtime.physics.qpos)
            qvel_trace[tick + 1].copy_(runtime.physics.qvel)
            action_trace[tick].copy_(runtime.state.last_actions)
            target_trace[tick].copy_(runtime.state.targets)
        stop_event.record(stream)
        stop_event.synchronize()
        wall_seconds = time.perf_counter() - wall_before
        host_cpu_seconds = time.process_time() - cpu_before
        gpu_seconds = start_event.elapsed_time(stop_event) / 1000.0
        traces = {
            "qpos": qpos_trace.cpu().numpy(), "qvel": qvel_trace.cpu().numpy(),
            "raw_policy_actions": action_trace.cpu().numpy(),
            "position_targets": target_trace.cpu().numpy(),
            "elapsed_simulation_time": (runtime.physics.time - before_time).cpu().numpy(),
        }
    trace_path = args.out / "trace.npz"
    with trace_path.open("xb") as destination:
        np.savez(destination, **traces)
    root = traces["qpos"].reshape(args.steps + 1, runtime.rows, 36)[..., :7]
    finite = {name: bool(np.isfinite(value).all()) for name, value in traces.items()}
    clock_error = (float(np.max(np.abs(traces["elapsed_simulation_time"] - args.steps * 0.02)))
                   if finite["elapsed_simulation_time"] else None)
    report = {
        "schema": "rek.puffysics_fixed_reference_prototype.v1",
        "host": socket.gethostname(), "gpu_name": torch.cuda.get_device_name(args.device),
        "backend": args.backend, "backend_metadata": runtime.physics.metadata,
        "arenas": runtime.arenas, "robot_rows": runtime.rows,
        "fixed_reference_role": args.role, "reference_loop": args.loop,
        "controller_period_seconds": 0.02, "target_filter_period_seconds": 0.004,
        "physics_timestep_seconds": 0.002, "physics_steps_per_control_tick": 10,
        "control_ticks_per_robot": args.steps, "simulated_seconds": args.steps * 0.02,
        "setup_seconds": setup_seconds, "wall_seconds": wall_seconds,
        "host_cpu_seconds": host_cpu_seconds, "gpu_stream_seconds": gpu_seconds,
        "robot_control_steps_per_wall_second": args.steps * runtime.rows / wall_seconds,
        "env_control_steps_per_wall_second": args.steps * runtime.arenas / wall_seconds,
        "robot_control_steps_per_gpu_second": args.steps * runtime.rows / gpu_seconds,
        "timed_trace_device_copies_per_tick": 4,
        "cuda_graph_replay": graph is not None, "initial_mapping": mapping,
        "finite": finite, "clock_max_abs_error_seconds": clock_error,
        "config_path": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "model_sha256": config["model_sha256"], "assets_sha256": config["assets_sha256"],
        "encoder_sha256": runtime.controller.encoder_identity.sha256,
        "decoder_sha256": runtime.controller.decoder_identity.sha256,
        "trace_path": str(trace_path.resolve()), "trace_sha256": sha256(trace_path),
        "cpu_physics_steps": 0, "cpu_controller_inferences": 0,
        "semantic_combat_integrated": False, "training_sps_measured": False,
        "authentic_rek_parity_established": False,
    }
    if hasattr(runtime.physics, "stats"):
        report["native_stats"] = runtime.physics.stats()
    if all(finite.values()):
        report.update({
            "root_height_min_m": float(root[..., 2].min()),
            "root_height_final_m": root[-1, :, 2].tolist(),
            "root_up_z_min": float((1 - 2 * (root[..., 4] ** 2 + root[..., 5] ** 2)).min()),
            "root_translation_max_m": float(np.linalg.norm(root[..., :3] - root[0, :, :3], axis=-1).max()),
        })
    report["finite_mapping_clock_checks_passed"] = (
        all(finite.values()) and mapping["within_1e_minus_6"]
        and clock_error is not None and clock_error <= max(1e-5, args.steps * 0.02 * 1e-4)
    )
    native_stats = report.get("native_stats", {})
    report["native_solver_checks_passed"] = all(native_stats.get(name, 0) == 0 for name in (
        "nonfinite_arenas", "articulated_solver_failure_arenas", "contact_capacity_reached_arenas",
    ))
    report["smoke_test_passed"] = (
        report["finite_mapping_clock_checks_passed"] and report["native_solver_checks_passed"]
    )
    with (args.out / "report.json").open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["smoke_test_passed"] else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("puffysics", "mujoco"), required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--model-export", type=Path)
    parser.add_argument("--solver-mode", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--role", default="idle")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup must be nonnegative")
    if args.backend == "puffysics" and (args.library is None or args.model_export is None):
        parser.error("Puffysics requires --library and --model-export")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
