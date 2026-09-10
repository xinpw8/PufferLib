"""Run actual CUDA semantic-duel input/contact probes and save measured traces."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from gpu_metrics import RekG1GpuMetricCollector
from gpu_semantic_duel import GpuDuelConfig, GpuSemanticDuel


def load_config(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("model", "assets", "controller_manifest", "controller_source",
                "motion_features", "motion_library", "combat_library"):
        values[key] = Path(values[key])
    values["move_duration_ticks"] = tuple(values["move_duration_ticks"])
    return GpuDuelConfig(**values)


def probe(config, steps, scenario, eager, spawn_distance=None):
    setup = time.perf_counter()
    env = GpuSemanticDuel(config)
    if spawn_distance is not None:
        with torch.cuda.stream(env.stream):
            env.physical_reset.initial_qpos[0] = -spawn_distance / 2
            env.physical_reset.initial_qpos[36] = spawn_distance / 2
            env.physical_reset.spawn_roots[0, 0] = -spawn_distance / 2
            env.physical_reset.spawn_roots[1, 0] = spawn_distance / 2
        env.reset()
    if not eager:
        env.capture_step()
    setup = time.perf_counter() - setup
    metrics = RekG1GpuMetricCollector(env.rows, config.device)
    trace = torch.empty((steps, env.rows, 223), device=config.device)
    masks = torch.empty((steps, env.rows, 33), device=config.device, dtype=torch.uint8)
    actions = torch.ones((env.rows, 1), device=config.device, dtype=torch.int64)
    action_trace = torch.empty((steps, env.rows), device=config.device, dtype=torch.int64)
    requested_trace = torch.empty_like(action_trace)
    move_starts = torch.empty((steps, env.rows), device=config.device, dtype=torch.uint8)
    clock_trace = torch.empty((steps, env.arenas), device=config.device)
    reference_roots = torch.empty((steps, env.rows, 4), device=config.device)
    bodies = env.physics.wp.to_torch(env.physics.data.xpos)
    body_trace = torch.empty((steps, *bodies.shape), device=config.device)
    cpu_before, start = time.process_time(), time.perf_counter()
    for tick in range(steps):
        actions.fill_(1)
        if scenario == "turn-kick":
            if tick < 10:
                actions[0::2, 0] = 6
            elif tick == 10:
                actions[0::2, 0] = 17
            else:
                actions[0::2, 0] = torch.where(env.action_mask[0::2, 0] != 0, 0, 1)
        elif scenario == "approach-kick":
            position = env.observations[0::2, :3]
            opponent = env.observations[0::2, 86:89]
            distance = (position[:, :2] - opponent[:, :2]).square().sum(-1).sqrt()
            kick_ready = env.action_mask[0::2, 17] != 0
            busy = env.observations[0::2, 183] != 0
            command = torch.where(distance > 0.80, 2, torch.where(kick_ready, 17, 1))
            actions[0::2, 0] = torch.where(busy, 0, command)
        elif scenario == "front-kick":
            if tick == 50:
                actions[0::2, 0] = 17
            elif tick > 50:
                actions[0::2, 0] = torch.where(env.action_mask[0::2, 0] != 0, 0, 1)
        requested_trace[tick].copy_(actions[:, 0])
        # The diagnostic controller must respect the same mask as a policy.
        # A fall can clear a command while the frozen compositor still reads
        # busy, so compositor busy alone cannot choose CONTINUE legally.
        legal = env.action_mask.gather(1, actions) != 0
        neutral_or_continue = torch.where(env.action_mask[:, :1] != 0, 0, 1)
        actions.copy_(torch.where(legal, actions, neutral_or_continue))
        if eager:
            env.stream.wait_stream(torch.cuda.current_stream(config.device))
            with torch.cuda.stream(env.stream):
                env.actions.copy_(actions[:, 0])
                env._step_impl()
            torch.cuda.current_stream(config.device).wait_stream(env.stream)
        else:
            env.step(actions)
        metrics.update(env.observations, env.terminals)
        trace[tick].copy_(env.observations)
        masks[tick].copy_(env.action_mask)
        action_trace[tick].copy_(actions[:, 0])
        move_starts[tick].copy_(env.scheduler.move_start_edge)
        clock_trace[tick].copy_(env.physics.time)
        reference_roots[tick].copy_(env.motion.rotations[:, 0])
        body_trace[tick].copy_(bodies)
    finished = torch.cuda.Event(blocking=True)
    finished.record()
    finished.synchronize()
    elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_before
    status_error = None
    try:
        env.check_status()
    except RuntimeError as error:
        status_error = str(error)
    arrays = {
        "observations": trace.cpu().numpy(), "action_mask": masks.cpu().numpy(),
        "actions": action_trace.cpu().numpy(), "move_start_edge": move_starts.cpu().numpy(),
        "requested_actions": requested_trace.cpu().numpy(),
        "arena_clock_seconds": clock_trace.cpu().numpy(),
        "reference_root_xyzw": reference_roots.cpu().numpy(),
        "body_xyz": body_trace.cpu().numpy(),
    }
    values = arrays["observations"]
    report = {
        "schema": "rek.g1_cuda_semantic_duel_probe.v1",
        "status_error": status_error,
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
        "scenario": scenario, "robots": env.rows, "arenas": env.arenas,
        "fixture_spawn_distance_m": spawn_distance,
        "steps_per_robot": steps, "simulated_seconds": steps * 0.02,
        "setup_seconds": setup, "wall_seconds": elapsed, "host_cpu_seconds": cpu,
        "robot_control_steps_per_second": env.rows * steps / elapsed,
        "move_start_count": arrays["move_start_edge"].sum(axis=0).tolist(),
        "script_mask_substitutions": (arrays["requested_actions"] != arrays["actions"]).sum(axis=0).tolist(),
        "attributed_contacts": values[:, 0::2, 221].sum(axis=0).tolist(),
        "scored_hits": values[:, 0::2, 222].sum(axis=0).tolist(),
        "root_z_min_m": values[..., 2].min(axis=0).tolist(),
        "root_xyz_final_m": values[-1, :, :3].tolist(),
        "final_points": values[-1, 0::2, 190:192].tolist(),
        "round_metrics": metrics.snapshot(),
        "cuda_graph": not eager, "cpu_physics_steps": 0,
        "cpu_controller_inferences": 0, "training_sps_measured": False,
        "authentic_rek_parity_established": False,
    }
    env.close()
    return report, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--controller-manifest", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--scenario", choices=("idle", "turn-kick", "approach-kick", "front-kick"), default="idle")
    parser.add_argument("--spawn-distance", type=float)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.spawn_distance is not None and not 0.5 <= args.spawn_distance <= 2.0:
        parser.error("fixture spawn distance must be between 0.5 and 2.0 metres")
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)
    config = load_config(args.config)
    if args.controller_manifest is not None:
        config = replace(config, controller_manifest=args.controller_manifest)
    report, arrays = probe(config, args.steps, args.scenario, args.eager, args.spawn_distance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.out.with_suffix(".npz")
    with trace_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["trace_path"] = str(trace_path)
    report["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status_error"]:
        raise RuntimeError(report["status_error"])


if __name__ == "__main__":
    main()
