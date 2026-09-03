#!/usr/bin/env python3
"""Run the official T800 supine-to-stance controller in the official MJCF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from engineai_t800_policy import CONTROL_DT, T800SupineRecoveryController


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    args = parser.parse_args()

    raw_trajectory = np.load(args.trajectory, allow_pickle=False).astype(np.float64)
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = raw_trajectory[90, :3]
    data.qpos[3:7] = raw_trajectory[90, 3:7]
    data.qpos[7:32] = raw_trajectory[90, 7:32]
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)

    controller = T800SupineRecoveryController(args.policy, args.trajectory)
    controller.reset(data.qpos[7:32])
    substeps = int(round(CONTROL_DT / float(model.opt.timestep)))
    minimum_height = float(data.qpos[2])
    samples = []
    started = time.perf_counter()

    while not controller.finished:
        target, _ = controller.step(
            data.qpos[7:32],
            data.qvel[6:31],
            data.qpos[3:7],
            data.qvel[3:6],
        )
        for _ in range(substeps):
            data.ctrl[:] = controller.pd_torque(
                data.qpos[7:32],
                data.qvel[6:31],
                target,
            )
            mujoco.mj_step(model, data)
        minimum_height = min(minimum_height, float(data.qpos[2]))
        if controller.trajectory_index % 25 == 0 or controller.finished:
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            samples.append(
                {
                    "index": controller.trajectory_index,
                    "time": float(data.time),
                    "height": float(data.qpos[2]),
                    "up_z": float(rotation[8]),
                    "target_linf": float(np.max(np.abs(target))),
                    "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
                }
            )
    elapsed = time.perf_counter() - started
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    result = {
        "schema": "engineai.t800.supine_recovery_validation.v1",
        "trajectory_steps": len(controller.trajectory),
        "simulated_seconds": float(data.time),
        "wall_seconds": elapsed,
        "realtime_factor": float(data.time / elapsed),
        "minimum_height": minimum_height,
        "final_height": float(data.qpos[2]),
        "final_up_z": float(rotation[8]),
        "all_finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "samples": samples,
    }
    print(json.dumps(result, indent=2))
    passed = result["all_finite"] and result["final_height"] > 0.8 and result["final_up_z"] > 0.9
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
