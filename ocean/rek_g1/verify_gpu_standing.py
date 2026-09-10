"""Exercise the pinned low-level controller and two-G1 physics on CUDA.

This is a fixed-idle integration test, not the semantic fighting environment.
It does not infer move, hit, knockout, or authentic REK parity from standing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from gpu_controller import GearSonicGpuController
from gpu_actuator_drive import GpuActuatorDrive
from gpu_duel_physics import GpuDuelPhysics
from gpu_duel_reset import GpuDuelReset
from gpu_motion_assets import GpuMotionAssets
from gpu_observation import GpuObservationAssembler
from gpu_robot_state import G1GpuControllerState


class FixedIdleGpuProbe:
    def __init__(self, args):
        self.controller = GearSonicGpuController.from_manifest(
            args.controller_manifest, args.source_bundle, device=args.device,
        )
        rows = self.controller.batch_size
        if rows % 2:
            raise ValueError("two fighters are required per arena")
        self.physics = GpuDuelPhysics(
            args.model, args.model_sha256, arenas=rows // 2, device=args.device,
        )
        self.assets = GpuMotionAssets(args.assets, args.assets_sha256, args.device)
        self.state = G1GpuControllerState(rows, args.device)
        self.active = torch.ones(rows, device=args.device, dtype=torch.bool)
        self.cursor = torch.zeros(rows, device=args.device, dtype=torch.long)
        physics = self.physics
        self.drive = GpuActuatorDrive.from_physics(self.physics)
        self.reset = GpuDuelReset(physics, self.assets)
        self.observation = GpuObservationAssembler(physics)
        self.reset.full(
            torch.ones(physics.arenas, device=args.device, dtype=torch.bool),
            reset_clock=True,
        )
        self.heading = self.reset.initial_heading

    def step(self):
        physics = self.physics
        rows = self.controller.batch_size
        kinematics = self.observation.gather_kinematics()
        base = kinematics[:, 3:7]
        angular_local = kinematics[:, 10:13]
        q = kinematics[:, 13:42]
        dq = kinematics[:, 42:71]
        position, next_position, root = self.assets.fixed_reference(
            "idle", self.cursor, loop=True,
        )
        observation = self.state.prepare(
            base, angular_local, q, dq, self.heading,
            position, next_position, root, self.active,
        )
        tokens = self.controller.encode(observation)
        actions = self.controller.decode(self.state.decoder_input(tokens))
        targets = self.state.apply_actions(actions, self.active)
        for substep in range(10):
            q = physics.qpos[:, self.reset.qindices].reshape(rows, 29)
            dq = physics.qvel[:, self.reset.dqindices].reshape(rows, 29)
            controls = self.drive.prepare(targets, q, dq, substep=substep)
            physics.ctrl.copy_(controls.reshape(-1, 58))
            physics.step()
        self.cursor.add_(1)


def verify_reset_isolation(runtime):
    physics = runtime.physics
    if physics.arenas < 2:
        raise ValueError("reset isolation requires at least two arenas")
    for _ in range(3):
        runtime.step()
    mask = torch.arange(physics.arenas, device=physics.qpos.device) == 0
    for operation in (runtime.reset.begin, runtime.reset.complete):
        before = {name: value.clone() for name, value in physics.reset_reader_fields.items()}
        qpos, qvel, clock = physics.qpos.clone(), physics.qvel.clone(), physics.time.clone()
        operation(mask)
        for name, values in physics.reset_reader_fields.items():
            torch.testing.assert_close(values[~mask], before[name][~mask], rtol=0, atol=0)
        torch.testing.assert_close(physics.qpos[~mask], qpos[~mask], rtol=0, atol=0)
        torch.testing.assert_close(physics.qvel[~mask], qvel[~mask], rtol=0, atol=0)
        torch.testing.assert_close(physics.time, clock, rtol=0, atol=0)
        torch.testing.assert_close(
            physics.qvel.reshape(-1, 2, 35)[..., :6],
            qvel.reshape(-1, 2, 35)[..., :6], rtol=0, atol=0,
        )
        if operation == runtime.reset.begin:
            physics.step()
    all_arenas = torch.ones_like(mask)
    all_rows = all_arenas.repeat_interleave(2)
    runtime.reset.full(all_arenas, reset_clock=True)
    runtime.state.reset(all_rows)
    runtime.drive.complete_reset(all_rows)
    runtime.cursor.zero_()
    return True


def _probe_current_stream(args):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    setup_start = time.perf_counter()
    runtime = FixedIdleGpuProbe(args)
    reset_isolation_verified = verify_reset_isolation(runtime) if args.reset_isolation else False
    for _ in range(3):
        runtime.step()
    torch.cuda.synchronize()
    graph = None
    if not args.eager:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream(args.device)):
            # Register Torch's external capture with Warp so temporary solver
            # allocations are retained for the captured graph's lifetime.
            with runtime.physics.wp.ScopedCapture(
                stream=runtime.physics.stream, external=True,
            ) as warp_capture:
                runtime.step()
        runtime.warp_capture = warp_capture
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    qpos_trace = torch.empty(
        (args.steps, runtime.physics.arenas, 72), device=args.device,
    )
    before_time = runtime.physics.time.clone()
    completion = torch.cuda.Event(blocking=True)
    cpu_before = time.process_time()
    start = time.perf_counter()
    for step in range(args.steps):
        if graph is None:
            runtime.step()
        else:
            graph.replay()
        qpos_trace[step].copy_(runtime.physics.qpos)
    completion.record()
    completion.synchronize()
    elapsed = time.perf_counter() - start
    host_cpu = time.process_time() - cpu_before
    # Downloads are confined to the completed measurement interval.
    traces = qpos_trace.cpu().numpy().reshape(args.steps, -1, 36)
    final_time = (runtime.physics.time - before_time).cpu().numpy()
    finite = bool(np.isfinite(traces).all())
    if not finite:
        raise RuntimeError("nonfinite GPU standing trajectory")
    clock_error = float(np.max(np.abs(final_time - args.steps * 0.02)))
    if clock_error > max(1e-5, args.steps * 0.02 * 1e-4):
        raise RuntimeError("CUDA physics clock did not advance by the requested steps")
    rotation = traces[..., 3:7]
    up_z = 1 - 2 * (rotation[..., 1] ** 2 + rotation[..., 2] ** 2)
    report = {
        "schema": "rek.g1_gpu_fixed_idle_probe.v1",
        "host": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(),
        "device": str(runtime.physics.qpos.device),
        "arenas": runtime.physics.arenas,
        "robot_rows": runtime.controller.batch_size,
        "timed_control_steps_per_robot": args.steps,
        "simulated_seconds": args.steps * 0.02,
        "wall_seconds": elapsed,
        "host_cpu_seconds": host_cpu,
        "setup_and_compilation_seconds": setup_seconds,
        "robot_control_steps_per_second": args.steps * runtime.controller.batch_size / elapsed,
        "root_z_min_m": float(traces[..., 2].min()),
        "root_z_final_m": traces[-1, :, 2].tolist(),
        "root_up_z_min": float(up_z.min()),
        "root_xy_displacement_max_m": float(np.linalg.norm(
            traces[-1, :, :2] - traces[0, :, :2], axis=-1,
        ).max()),
        "clock_max_abs_error_seconds": clock_error,
        "model_mutable_field_shapes": {
            name: list(getattr(runtime.physics.model, name).shape)
            for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
        },
        "model_sha256": args.model_sha256,
        "assets_manifest_sha256": args.assets_sha256,
        "encoder_sha256": runtime.controller.encoder_identity.sha256,
        "decoder_sha256": runtime.controller.decoder_identity.sha256,
        "model_cuda_bytes": runtime.controller.resident_bytes(),
        "cpu_physics_steps": 0,
        "cpu_controller_inferences": 0,
        "cuda_graph_replay": graph is not None,
        "reset_isolation_verified": reset_isolation_verified,
        "reference_mode": "fixed_idle_loop",
        "semantic_moves_integrated": False,
        "combat_integrated": False,
        "training_sps_measured": False,
        "rek_parity_claim": False,
    }
    return report


def probe(args):
    stream = torch.cuda.Stream(device=args.device)
    stream.wait_stream(torch.cuda.current_stream(args.device))
    with torch.cuda.stream(stream):
        return _probe_current_stream(args)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "assets", "controller-manifest", "source-bundle", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("model-sha256", "assets-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--reset-isolation", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    result = probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
