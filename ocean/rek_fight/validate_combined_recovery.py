#!/usr/bin/env python3
"""Validate the official get-up controller in the recovered two-T800 plant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from engineai_t800_policy import CONTROL_DT, T800MuJoCoBinding, T800SupineRecoveryController


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    args = parser.parse_args()

    raw = np.load(args.trajectory, allow_pickle=False).astype(np.float64)
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    bindings = [T800MuJoCoBinding(mujoco, model, fighter=agent) for agent in range(2)]
    tested = bindings[0]
    tested.set_default_pose(data)
    bindings[1].set_default_pose(data)
    spawn = tested.root_position(data)
    data.qpos[tested.root_qpos_address : tested.root_qpos_address + 3] = [spawn[0], spawn[1], raw[90, 2]]
    data.qpos[tested.root_qpos_address + 3 : tested.root_qpos_address + 7] = raw[90, 3:7]
    data.qpos[tested.qpos_addresses] = raw[90, 7:32]
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)

    recovery = T800SupineRecoveryController(args.policy, args.trajectory)
    joint_q, _, _, _ = tested.state(data)
    recovery.reset(joint_q)
    substeps = int(round(CONTROL_DT / float(model.opt.timestep)))
    minimum_height = float(tested.root_position(data)[2])

    while not recovery.finished:
        joint_q, joint_qd, quaternion, angular_velocity = tested.state(data)
        target, _ = recovery.step(joint_q, joint_qd, quaternion, angular_velocity)
        for _ in range(substeps):
            joint_q, joint_qd, _, _ = tested.state(data)
            tested.apply_torque(data, recovery.pd_torque(joint_q, joint_qd, target))
            mujoco.mj_step(model, data)
        minimum_height = min(minimum_height, float(tested.root_position(data)[2]))

    result = {
        "schema": "engineai.t800.combined_supine_recovery_validation.v1",
        "simulated_seconds": float(data.time),
        "minimum_height": minimum_height,
        "final_height": float(tested.root_position(data)[2]),
        "final_up_z": tested.root_up_z(data),
        "all_finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }
    print(json.dumps(result, indent=2))
    passed = result["all_finite"] and result["final_height"] > 0.8 and result["final_up_z"] > 0.9
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
