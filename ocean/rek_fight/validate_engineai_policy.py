#!/usr/bin/env python3
"""Run the official EngineAI walking policy in the official T800 MJCF."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from engineai_t800_policy import CONTROL_DT, DEFAULT_Q, T800WalkingController


EXPECTED_POLICY_SHA256 = "cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"missing sensor {name}")
    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])
    return data.sensordata[address : address + dimension].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--forward-seconds", type=float, default=5.0)
    args = parser.parse_args()

    policy_hash = sha256(args.policy)
    if policy_hash != EXPECTED_POLICY_SHA256:
        raise SystemExit(f"walking policy SHA-256 mismatch: {policy_hash}")

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    if model.nq != 32 or model.nv != 31 or model.nu != 25:
        raise SystemExit(f"unexpected T800 dimensions: nq={model.nq} nv={model.nv} nu={model.nu}")
    ratio = CONTROL_DT / float(model.opt.timestep)
    substeps = int(round(ratio))
    if abs(ratio - substeps) > 1e-9:
        raise SystemExit(f"control_dt {CONTROL_DT} is not divisible by physics dt {model.opt.timestep}")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[7:32] = DEFAULT_Q
    mujoco.mj_forward(model, data)
    controller = T800WalkingController(args.policy)
    initial_position = data.qpos[:3].copy()
    samples = []
    total_control_steps = int(round((args.idle_seconds + args.forward_seconds) / CONTROL_DT))
    started = time.perf_counter()
    target_q = DEFAULT_Q.copy()

    for control_step in range(total_control_steps):
        normalized_command = np.array([1.0, 0.0, 0.0]) if control_step * CONTROL_DT >= args.idle_seconds else np.zeros(3)
        command = controller.scale_command(normalized_command)
        controller.observe(
            data.qpos[7:32],
            data.qvel[6:31],
            sensor(model, data, "imu_quaternion"),
            sensor(model, data, "imu_angular_velocity"),
        )
        action, target_q = controller.act(command)
        for _ in range(substeps):
            data.ctrl[:] = controller.pd_torque(data.qpos[7:32], data.qvel[6:31], target_q)
            mujoco.mj_step(model, data)
        if control_step % 10 == 0 or control_step == total_control_steps - 1:
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            samples.append(
                {
                    "t": float(data.time),
                    "x": float(data.qpos[0]),
                    "y": float(data.qpos[1]),
                    "height": float(data.qpos[2]),
                    "up_z": float(rotation[8]),
                    "command": command.tolist(),
                    "action_linf": float(np.max(np.abs(action))),
                    "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
                }
            )

    elapsed = time.perf_counter() - started
    result = {
        "schema": "engineai.t800.walking_policy_validation.v1",
        "source_commit": "335c60e88772c26c7852d0abd6b3c7439037dd8f",
        "mjcf": str(args.mjcf.resolve()),
        "mjcf_sha256": sha256(args.mjcf),
        "policy": str(args.policy.resolve()),
        "policy_sha256": policy_hash,
        "physics_dt": float(model.opt.timestep),
        "control_dt": CONTROL_DT,
        "physics_substeps": substeps,
        "simulated_seconds": float(data.time),
        "wall_seconds": elapsed,
        "realtime_factor": float(data.time / elapsed),
        "initial_position": initial_position.tolist(),
        "final_position": data.qpos[:3].tolist(),
        "forward_displacement": float(data.qpos[0] - initial_position[0]),
        "min_height": min(sample["height"] for sample in samples),
        "min_up_z": min(sample["up_z"] for sample in samples),
        "all_finite": all(sample["finite"] for sample in samples),
        "samples": samples,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["all_finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
