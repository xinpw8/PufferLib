"""CUDA physics for the existing two-G1 model using MuJoCo Warp.

Model parsing and the initial upload happen on the host. Physics steps, state,
contacts, and actuator commands remain on CUDA. This component does not yet
implement the semantic move controller or establish parity with authentic REK.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import numpy as np

import gear_sonic_candidate as candidate
from g1_two_fighter_arena import EXPECTED_MODEL_DIMENSIONS, EXPECTED_SPAWN_QPOS


class GpuDuelPhysics:
    """One CUDA world per arena, with two fighters sharing its contacts."""

    def __init__(
        self,
        model_path: Path,
        model_sha256: str,
        *,
        arenas: int,
        device: str = "cuda:0",
        contacts_per_arena: int = 128,
        constraints_per_arena: int = 1024,
    ):
        import mujoco
        import mujoco_warp as mjw
        import torch
        import warp as wp

        if arenas < 1 or contacts_per_arena < 1 or constraints_per_arena < 1:
            raise ValueError("arena and capacity counts must be positive")
        payload = model_path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != model_sha256:
            raise ValueError(f"model SHA-256 mismatch: {actual_hash}")
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("REK GPU physics requires a CUDA device")
        self.wp = wp
        self.mjw = mjw
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        self.arenas = arenas
        self.model_sha256 = actual_hash
        self.host_model = mujoco.MjModel.from_xml_string(payload.decode("utf-8"))
        model = self.host_model
        for name, expected in EXPECTED_MODEL_DIMENSIONS.items():
            if getattr(model, name) != expected:
                raise ValueError(f"model {name} differs from the two-G1 contract")
        self.actuator_ids = []
        self.joint_qpos = []
        self.joint_qvel = []
        self.root_bodies = []
        for side, role in enumerate(("player", "opponent")):
            prefix = role + "__"
            root_joint = model.joint(prefix + "joint__floating_base_joint_3081")
            root_body = model.body(prefix + "pelvis_3266")
            if int(root_joint.qposadr[0]) != side * 36:
                raise ValueError("root qpos mapping differs from the native runtime")
            if int(root_joint.dofadr[0]) != side * 35:
                raise ValueError("root qvel mapping differs from the native runtime")
            np.testing.assert_allclose(
                model.qpos0[side * 36 : side * 36 + 7],
                EXPECTED_SPAWN_QPOS[role], atol=1e-12, rtol=0,
            )
            ids = np.arange(side * 29, (side + 1) * 29)
            for actuator_id in ids:
                if not model.actuator(int(actuator_id)).name.startswith(prefix):
                    raise ValueError("actuator role ordering differs from native runtime")
            joints = model.actuator_trnid[ids, 0]
            self.actuator_ids.append(ids)
            self.joint_qpos.append(model.jnt_qposadr[joints].copy())
            self.joint_qvel.append(model.jnt_dofadr[joints].copy())
            self.root_bodies.append(root_body.id)
            model.actuator_gainprm[ids] = 0.0
            model.actuator_biasprm[ids] = 0.0
            candidate.configure_native_position_actuators(
                mujoco, model, SimpleNamespace(actuator_ids=ids),
                candidate.PUBLIC_EFFORT_LIMIT_MUJOCO,
            )
        model.opt.timestep = 0.002
        initial = mujoco.MjData(model)
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self.model = mjw.put_model(model)
            self._validate_selected_forward_backend()
            self.data = mjw.put_data(
                model, initial, nworld=arenas,
                nconmax=contacts_per_arena, njmax=constraints_per_arena,
            )
            mjw.forward(self.model, self.data)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.time = wp.to_torch(self.data.time)
        if not all(value.is_cuda for value in (self.qpos, self.qvel, self.ctrl)):
            raise RuntimeError("physics buffers were not allocated on CUDA")
        # mjw.forward is batched. Preserve unselected worlds' externally read
        # post-step fields and solver warm starts when refreshing reset worlds.
        self.reset_reader_fields = {
            name: wp.to_torch(getattr(self.data, name))
            for name in (
                "qacc", "qacc_warmstart", "xpos", "xquat", "xmat", "xipos",
                "ximat", "subtree_com", "cvel", "geom_xpos", "geom_xmat",
                "actuator_force",
            )
        }
        self._reset_reader_backup = {
            name: torch.empty_like(value) for name, value in self.reset_reader_fields.items()
        }

    def _validate_selected_forward_backend(self):
        """Selected reset isolation relies on these checked backend conditions."""
        if not self.model.opt.run_collision_detection:
            raise ValueError("selected resets require contact reconstruction on every forward")
        callbacks = vars(self.model.callback)
        active = [name for name, callback in callbacks.items() if callback is not None]
        if active:
            raise ValueError(f"selected resets do not support mutating backend callbacks: {active}")

    def step(self) -> None:
        """Advance one 2 ms physics step without downloading simulation state."""
        # Torch and Warp share this stream. Importing work from Warp's previous
        # stream would invalidate an enclosing Torch CUDA graph capture.
        with self.wp.ScopedDevice(self.device), self.wp.ScopedStream(self.stream, sync_enter=False):
            self.mjw.step(self.model, self.data)

    def forward(self) -> None:
        """Refresh derived quantities after an explicit state reset."""
        with self.wp.ScopedDevice(self.device), self.wp.ScopedStream(self.stream, sync_enter=False):
            self.mjw.forward(self.model, self.data)

    def forward_selected(self, mask) -> None:
        """Refresh reset readers without advancing unaffected worlds' readers.

        Contact/constraint scratch is rebuilt globally and is consumed only
        after the next physics step. Callers must consume this substep's contact
        measurement before invoking any reset operation.
        """
        import torch

        if mask.shape != (self.arenas,) or mask.device != self.qpos.device or mask.dtype != torch.bool:
            raise ValueError("forward mask must be one CUDA Boolean per arena")
        self._validate_selected_forward_backend()
        for name, values in self.reset_reader_fields.items():
            self._reset_reader_backup[name].copy_(values)
        self.forward()
        for name, values in self.reset_reader_fields.items():
            selected = mask.reshape(-1, *([1] * (values.ndim - 1)))
            values.copy_(torch.where(selected, values, self._reset_reader_backup[name]))

    def capture_steps(self, count: int = 10):
        """Capture physics launches. Semantic callbacks must be added by caller."""
        if count < 1:
            raise ValueError("step count must be positive")
        with self.wp.ScopedDevice(self.device), self.wp.ScopedStream(self.stream):
            self.wp.synchronize_device(self.device)
            with self.wp.ScopedCapture(stream=self.stream) as capture:
                for _ in range(count):
                    self.mjw.step(self.model, self.data)
        return capture.graph


def probe(args: argparse.Namespace) -> dict:
    import torch

    started = time.perf_counter()
    physics = GpuDuelPhysics(
        args.model, args.model_sha256, arenas=args.arenas, device=args.device,
    )
    physics.step()
    physics.wp.synchronize_device(physics.device)
    graph = physics.capture_steps(args.graph_steps)
    physics.wp.capture_launch(graph, stream=physics.stream)
    physics.wp.synchronize_device(physics.device)
    setup_seconds = time.perf_counter() - started
    before = physics.time.detach().cpu().numpy().copy()
    host_cpu_before = time.process_time()
    start = time.perf_counter()
    for _ in range(args.launches):
        physics.wp.capture_launch(graph, stream=physics.stream)
    physics.wp.synchronize_device(physics.device)
    elapsed = time.perf_counter() - start
    host_cpu_seconds = time.process_time() - host_cpu_before
    after = physics.time.detach().cpu().numpy().copy()
    expected_advance = args.launches * args.graph_steps * 0.002
    time_error = float(np.max(np.abs((after - before) - expected_advance)))
    if not bool(torch.isfinite(physics.qpos).all().item()):
        raise RuntimeError("CUDA physics produced nonfinite qpos")
    if not bool(torch.isfinite(physics.qvel).all().item()):
        raise RuntimeError("CUDA physics produced nonfinite qvel")
    if time_error > max(1e-5, expected_advance * 1e-4):
        raise RuntimeError(f"CUDA clock advance mismatch: {time_error}")
    return {
        "schema": "rek.g1_cuda_physics_probe.v1",
        "host": socket.gethostname(),
        "device": str(physics.device),
        "gpu_name": torch.cuda.get_device_name(0),
        "model_path": str(args.model),
        "model_sha256": physics.model_sha256,
        "arenas": args.arenas,
        "robot_rows": args.arenas * 2,
        "physics_timestep_seconds": 0.002,
        "timed_steps_per_arena": args.launches * args.graph_steps,
        "wall_seconds": elapsed,
        "host_cpu_seconds": host_cpu_seconds,
        "setup_and_compilation_seconds": setup_seconds,
        "aggregate_arena_physics_steps_per_second":
            args.arenas * args.launches * args.graph_steps / elapsed,
        "clock_max_abs_error_seconds": time_error,
        "simulation_buffers_device": str(physics.qpos.device),
        "physics_backend": "mujoco_warp",
        "cpu_physics_steps": 0,
        "semantic_controller_integrated": False,
        "training_sps_measured": False,
        "rek_parity_claim": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "mujoco-warp", "warp-lang", "torch")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--arenas", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-steps", type=int, default=10)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.launches < 1:
        parser.error("--launches must be positive")
    result = probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
