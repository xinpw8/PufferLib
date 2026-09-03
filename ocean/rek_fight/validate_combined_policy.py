#!/usr/bin/env python3
"""Validate two official walking controllers in the recovered two-T800 plant."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from engineai_t800_policy import CONTROL_DT, T800MuJoCoBinding, T800WalkingController


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    source_timestep = float(model.opt.timestep)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    bindings = [T800MuJoCoBinding(mujoco, model, fighter=agent) for agent in range(2)]
    controllers = [T800WalkingController(args.policy) for _ in range(2)]
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for binding in bindings:
        binding.set_default_pose(data)
    mujoco.mj_forward(model, data)
    initial = [binding.root_position(data) for binding in bindings]
    substeps = int(round(CONTROL_DT / float(model.opt.timestep)))
    minima = [[float("inf"), float("inf")] for _ in range(2)]
    initial_time = float(data.time)
    started = time.perf_counter()
    total_steps = int(round(args.seconds / CONTROL_DT))

    for step in range(total_steps):
        commands = [np.array([1.0, 0.0, 0.0]) if step * CONTROL_DT >= args.seconds / 2 else np.zeros(3), np.zeros(3)]
        targets = []
        for binding, controller, normalized in zip(bindings, controllers, commands):
            q, qd, quaternion, angular_velocity = binding.state(data)
            controller.observe(q, qd, quaternion, angular_velocity)
            _, target = controller.act(controller.scale_command(normalized))
            targets.append(target)
        for _ in range(substeps):
            for binding, controller, target in zip(bindings, controllers, targets):
                q, qd, _, _ = binding.state(data)
                binding.apply_torque(data, controller.pd_torque(q, qd, target))
            mujoco.mj_step(model, data)
        for agent, binding in enumerate(bindings):
            minima[agent][0] = min(minima[agent][0], float(binding.root_position(data)[2]))
            minima[agent][1] = min(minima[agent][1], binding.root_up_z(data))

    elapsed = time.perf_counter() - started
    final = [binding.root_position(data) for binding in bindings]
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    result = {
        "schema": "engineai.t800.combined_policy_validation.v1",
        "source_physics_timestep": source_timestep,
        "physics_timestep": float(model.opt.timestep),
        "physics_substeps": substeps,
        "initial_model_time": initial_time,
        "final_model_time": float(data.time),
        "simulated_seconds": float(data.time - initial_time),
        "wall_seconds": elapsed,
        "realtime_factor": float((data.time - initial_time) / elapsed),
        "initial_positions": [value.tolist() for value in initial],
        "final_positions": [value.tolist() for value in final],
        "agent_0_forward_displacement": float(final[0][0] - initial[0][0]),
        "minimum_height_up_z": minima,
        "all_finite": finite,
    }
    print(json.dumps(result, indent=2))
    stable = finite and all(height > 0.8 and up_z > 0.9 for height, up_z in minima)
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
