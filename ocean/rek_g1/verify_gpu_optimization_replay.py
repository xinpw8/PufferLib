"""Bounded original/repeat/optimized full-environment CUDA replay diagnostic.

Four arenas share one scripted trace. No production class is modified. The
fourth arena substitutes explicitly synthetic fall measurements to exercise
the unchanged native counted-fall and mixed physical-reset path. All remaining
measurements, controllers, physics, rewards and event transitions are live.
The deferred-combat mode keeps the configured reset and measurement options
identical in both branches and changes only intermediate observation packing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from verify_gpu_reset_forward_gate import paired_error


SCENARIOS = (
    "idle",
    "held_translation_and_yaw_to_kick",
    "close_range_front_and_side_kicks",
    "synthetic_fall_native_counted_reset",
)
EXACT_FIELDS = (
    "actions", "input_action_mask", "action_mask", "illegal_input",
    "rewards", "terminals", "move_start_edge", "move_active",
    "active_route_ids", "semantic_kind", "move_registry_index",
    "scheduler_status", "combat_status", "fall_phase", "points",
    "tick_fall_events", "tick_signals", "tick_referee_calls",
    "tick_score_delta", "tick_attributed_contacts", "tick_scored_contacts",
    "substep_begin_reset", "substep_complete_reset", "substep_combat_status",
    "episode_reset", "completed_reset_in_tick",
)
NUMERIC_FIELDS = ("observations", "qpos", "qvel", "qacc", "body_xyz", "ctrl", "clock")


def requested_actions(steps):
    """Categories from g1_semantic_action_table.c; one row per fighter."""
    trace = np.ones((steps, 8), dtype=np.int64)
    # The replay horizon includes two separate yaw/attack interventions. A
    # request rejected by the original mask is recorded, never called an input.
    trace[8:28, 2] = 8                 # held forward + yaw left
    trace[32:48, 2] = 6                # held yaw left
    trace[48, 2] = 17                  # front kick interrupts held yaw
    trace[140:156, 2] = 7              # held yaw right
    trace[156, 2] = 18                 # next original discrete kick category
    trace[200:220, 2] = 5              # held strafe right
    trace[16, 4] = 17
    trace[148, 4] = 18
    return trace


def exact_difference(a, b):
    different = np.not_equal(a, b)
    indices = np.argwhere(different)
    return {
        "exact": not bool(len(indices)),
        "different_elements": int(different.sum()),
        "first_index_repeat_tick_and_field": indices[0].tolist() if len(indices) else None,
    }


def pairwise_exact(a, b, c):
    return {
        "original_repeat": exact_difference(a, b),
        "gated_vs_original": exact_difference(a, c),
        "gated_vs_original_b": exact_difference(b, c),
    }


def replay_configs(config, *, conditional_reset_forward, deferred_combat_library=None):
    """Select one explicit intervention without changing the existing default."""
    if deferred_combat_library is not None:
        original = dataclasses.replace(
            config, combat_library=deferred_combat_library,
            defer_substep_combat_observations=False)
        return original, dataclasses.replace(original, defer_substep_combat_observations=True)
    original = dataclasses.replace(config, conditional_reset_forward=False, fused_combat_library=None)
    candidate = dataclasses.replace(config, conditional_reset_forward=conditional_reset_forward)
    return original, candidate


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest(config, config_path):
    here = Path(__file__).resolve().parent
    runtime_root = Path(importlib.util.find_spec("gpu_semantic_duel").origin).parent
    sources = ("verify_gpu_optimization_replay.py", "verify_gpu_reset_forward_gate.py",
               "gpu_reset_forward_gate.py", "gpu_semantic_duel.py", "gpu_duel_physics.py",
               "gpu_duel_reset.py", "gpu_combat_measurement.py", "gpu_native_combat.py",
               "gpu_combat_measurement_fused.py", "gpu_combat_measurement_fused.cu",
               "gpu_semantic_scheduler.py", "gpu_robot_state.py", "gpu_actuator_drive.py",
               "gpu_observation.py", "gpu_native_motion.py", "gpu_controller.py",
               "g1_semantic_action_table.c", "g1_fall_state.c", "g1_fight_state.c",
               "g1_native_combat_cuda.cu", "g1_native_combat_cuda.h")
    paths = [config_path]
    for name in sources:
        if name.endswith(".py"):
            spec = importlib.util.find_spec(name[:-3])
            paths.append(Path(spec.origin) if spec is not None else here / name)
        else:
            paths.append(runtime_root / name)
    paths.extend(getattr(config, name) for name in (
        "model", "assets", "controller_manifest", "controller_source",
        "motion_features", "motion_library", "combat_library"))
    if config.fused_combat_library is not None:
        paths.append(config.fused_combat_library)
    # Directory identities are already pinned by the runtime asset/manifest
    # validators. Avoid recursively copying or publishing private assets.
    files = [{"path": str(path), "sha256": digest(path)} for path in paths if path.is_file()]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"git_revision_if_available": revision, "files": files,
            "runtime_validated_model_sha256": config.model_sha256,
            "runtime_validated_assets_sha256": config.assets_sha256}


class ReplayHarness:
    """Graph-compatible fixture and ten-substep event recording, test-only."""

    def __init__(self, env):
        import torch

        self.env = env
        device = env.actions.device
        self.synthetic_fall = torch.zeros(env.rows, device=device, dtype=torch.bool)
        self.fall_floats = torch.tensor([90.0, .2, .002], device=device)
        self.fall_integers = torch.tensor([1, 1, 0, 1, 0, 0, 0], device=device, dtype=torch.int64)
        self.begin = torch.zeros((10, env.arenas), device=device, dtype=torch.bool)
        self.complete = torch.zeros_like(self.begin)
        self.status = torch.zeros((10, env.arenas), device=device, dtype=torch.int32)
        original_sample = env.measurement.sample
        original_post = env.combat.post_step
        self.substep = None

        def sample(substep, can_get_up):
            self.substep = substep  # setup/capture-time index; no hot-path Python
            batch = original_sample(substep, can_get_up)
            selected = self.synthetic_fall[:, None]
            batch.fall.floats[:, :3].copy_(torch.where(
                selected, self.fall_floats, batch.fall.floats[:, :3]))
            batch.fall.integers.copy_(torch.where(
                selected, self.fall_integers, batch.fall.integers))
            return batch

        def post(batch, **kwargs):
            output = original_post(batch, **kwargs)
            self.begin[self.substep].copy_(output.begin_reset)
            self.complete[self.substep].copy_(output.complete_reset)
            self.status[self.substep].copy_(output.statuses)
            # Stop imposing the synthetic fall after its native reset begins.
            self.synthetic_fall.logical_and_(~output.begin_reset.repeat_interleave(2))
            return output

        env.measurement.sample = sample
        env.combat.post_step = post

    def reset(self, common_physics=None):
        import torch

        env = self.env
        with torch.cuda.stream(env.stream):
            self.synthetic_fall.zero_()
        env.reset()
        with torch.cuda.stream(env.stream):
            # Close-range fixture only changes arena 2's starting separation.
            env.physics.qpos[2, 0] = -.4
            env.physics.qpos[2, 36] = .4
            selected = torch.zeros(env.arenas, dtype=torch.bool, device=env.actions.device)
            selected[2] = True
            env.physics.forward_selected(selected)
            if common_physics is not None:
                arrays = physics_arrays(env.physics)
                if set(arrays) != set(common_physics):
                    raise ValueError("original and candidate Data layouts differ")
                for name, values in arrays.items():
                    values.copy_(common_physics[name])
            env._gather()
            env.observer.pack(env.combat.fall, env.scheduler.observation12, env.combat.fight)
            self.synthetic_fall[6] = True
            self.begin.zero_()
            self.complete.zero_()
            self.status.zero_()
        env.stream.synchronize()


def physics_arrays(physics):
    """Restore Data, without transferring native structs containing pointers."""
    arrays = {}

    def visit(value, prefix=""):
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            name = prefix + field.name
            if isinstance(item, physics.wp.array):
                tensor = physics.wp.to_torch(item)
                if tensor.numel():
                    arrays[name] = tensor
            elif dataclasses.is_dataclass(item):
                visit(item, name + ".")

    visit(physics.data)
    return arrays


def trace_sources(harness):
    env = harness.env
    combat, scheduler, physics = env.combat, env.scheduler, env.physics
    fields = {
        "observations": env.observations, "action_mask": env.action_mask,
        "rewards": env.rewards, "terminals": env.terminals,
        "scheduler_status": scheduler.statuses, "combat_status": combat.statuses,
        "fall_phase": combat.fall[:, 8], "points": combat.fight[:, 6:8],
        "qpos": physics.qpos, "qvel": physics.qvel,
        "qacc": physics.wp.to_torch(physics.data.qacc),
        "body_xyz": physics.wp.to_torch(physics.data.xpos),
        "ctrl": physics.ctrl, "clock": physics.time,
        "substep_begin_reset": harness.begin, "substep_complete_reset": harness.complete,
        "substep_combat_status": harness.status,
        "completed_reset_in_tick": env.completed_reset_in_tick,
    }
    for name in ("move_start_edge", "move_active", "active_route_ids", "semantic_kind", "move_registry_index"):
        fields[name] = getattr(scheduler, name)
    for name in ("tick_fall_events", "tick_signals", "tick_referee_calls", "tick_score_delta",
                 "tick_attributed_contacts", "tick_scored_contacts", "episode_reset"):
        fields[name] = getattr(combat, name)
    return fields


def replay(harness, requested, common_physics, recorded_actions=None):
    import torch

    env = harness.env
    harness.reset(common_physics)
    sources = trace_sources(harness)
    steps = len(requested)
    arrays = {name: torch.empty((steps, *value.shape), device=value.device, dtype=value.dtype)
              for name, value in sources.items()}
    arrays["input_action_mask"] = torch.empty_like(arrays["action_mask"])
    arrays["actions"] = torch.empty((steps, env.rows), device=env.actions.device, dtype=torch.int64)
    arrays["illegal_input"] = torch.empty((steps, env.rows), device=env.actions.device, dtype=torch.bool)
    commands = torch.as_tensor(requested if recorded_actions is None else recorded_actions,
                               device=env.actions.device, dtype=torch.int64)
    status_error = None
    completed = 0
    for tick in range(steps):
        mask = env.action_mask
        desired = commands[tick, :, None]
        if recorded_actions is None:
            legal = mask.gather(1, desired) != 0
            substitute = torch.where(mask[:, :1] != 0, 0, 1)
            actual = torch.where(legal, desired, substitute)
        else:
            actual = desired
        arrays["input_action_mask"][tick].copy_(mask)
        arrays["actions"][tick].copy_(actual[:, 0])
        arrays["illegal_input"][tick].copy_(mask.gather(1, actual)[:, 0] == 0)
        env.step(actual)
        for name, value in sources.items():
            arrays[name][tick].copy_(value)
        completed += 1
        try:
            env.check_status()
        except RuntimeError as error:
            status_error = {"tick": tick, "error": str(error)}
            break
    torch.cuda.synchronize(env.actions.device)
    return {name: values[:completed].cpu().numpy() for name, values in arrays.items()}, status_error


def coverage(arrays):
    begins = arrays["substep_begin_reset"]
    completes = arrays["substep_complete_reset"]
    return {
        "ticks": len(arrays["actions"]),
        "move_starts_by_fighter": arrays["move_start_edge"].sum(axis=0).tolist(),
        "scored_contacts_by_arena": arrays["tick_scored_contacts"].sum(axis=0).tolist(),
        "score_delta_by_fighter": arrays["tick_score_delta"].sum(axis=0).tolist(),
        "begin_reset_by_arena": begins.sum(axis=(0, 1)).tolist(),
        "complete_reset_by_arena": completes.sum(axis=(0, 1)).tolist(),
        "mixed_reset_substeps": int(((begins | completes).any(axis=-1)
                                      & ~(begins | completes).all(axis=-1)).sum()),
        "synthetic_arena_knockout_referee_observed": bool(np.any(arrays["tick_referee_calls"][:, 3] & 16)),
        "any_terminal_observed": bool(np.any(arrays["terminals"])),
        "illegal_input_count": int(arrays["illegal_input"].sum()),
        "q_to_front_kick_accepted": bool(len(arrays["actions"]) > 48
            and arrays["actions"][47, 2] == 6 and arrays["actions"][48, 2] == 17),
        "e_to_kick_accepted": bool(len(arrays["actions"]) > 156
            and arrays["actions"][155, 2] == 7 and arrays["actions"][156, 2] == 18),
    }


def compare(raw, repeats):
    report = {"exact": {}, "numeric": {}}
    failures = []
    for field in EXACT_FIELDS:
        values = [np.stack([raw[f"{branch}/{repeat}/{field}"] for repeat in range(repeats)])
                  for branch in ("original_a", "original_b", "gated")]
        pairs = pairwise_exact(*values)
        baseline, candidate = pairs["original_repeat"], pairs["gated_vs_original"]
        report["exact"][field] = pairs
        if not baseline["exact"] or not candidate["exact"]:
            failures.append("exact/" + field)
    for field in NUMERIC_FIELDS:
        values = [np.stack([raw[f"{branch}/{repeat}/{field}"] for repeat in range(repeats)])
                  for branch in ("original_a", "original_b", "gated")]
        report["numeric"][field] = paired_error(*values)
        if not report["numeric"][field]["accepted"]:
            failures.append("numeric/" + field)
    return report, failures


def run(args):
    import torch
    from gpu_semantic_duel import GpuSemanticDuel
    from verify_gpu_duel import load_config

    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)
    config = load_config(args.config)
    if args.fused_combat_library is not None:
        config = dataclasses.replace(config, fused_combat_library=args.fused_combat_library)
    original_config, candidate_config = replay_configs(
        config, conditional_reset_forward=args.conditional_reset_forward,
        deferred_combat_library=args.deferred_combat_library)
    torch.manual_seed(args.seed)
    order_rng = np.random.default_rng(args.seed)
    manifest = source_manifest(candidate_config, args.config)
    setup_start = time.perf_counter()
    harnesses = {}
    branches = [("original_a", False)]
    if args.independent_original_graphs:
        branches.append(("original_b", False))
    branches.append(("gated", True))
    for label, enabled in branches:
        env = GpuSemanticDuel(candidate_config if enabled else original_config)
        if env.arenas != 4:
            raise ValueError("this bounded four-scenario replay requires exactly eight fighters")
        harness = ReplayHarness(env)
        env.capture_step()
        harnesses[label] = harness
    original = harnesses["original_a"]
    original.reset()
    with torch.cuda.stream(original.env.stream):
        common = {name: values.clone() for name, values in physics_arrays(original.env.physics).items()}
    original.env.stream.synchronize()
    requested = requested_actions(args.steps)
    supplied_actions = None
    if args.actions_from is not None:
        with np.load(args.actions_from, allow_pickle=False) as supplied:
            supplied_actions = supplied["recorded_actions"].copy()
        if supplied_actions.shape != requested.shape or supplied_actions.dtype != np.int64:
            raise ValueError("recorded action trace layout differs from this probe")
    pilot, pilot_error = replay(original, requested, common, supplied_actions)
    action_trace = pilot["actions"]
    raw = {"requested_actions": requested, "recorded_actions": action_trace}
    for name, value in pilot.items():
        raw["pilot/" + name] = value
    statuses, coverage_reports, order = {"pilot": pilot_error}, {"pilot": coverage(pilot)}, []
    setup_seconds = time.perf_counter() - setup_start
    start = time.perf_counter()
    complete_runs = pilot_error is None and len(action_trace) == args.steps
    if complete_runs:
        for repeat in range(args.repeats):
            for branch in order_rng.permutation(("original_a", "original_b", "gated")):
                key_env = branch if branch in harnesses else "original_a"
                harness = harnesses[key_env]
                arrays, error = replay(harness, requested, common, action_trace)
                key = f"{branch}/{repeat}"
                order.append(key)
                statuses[key] = error
                coverage_reports[key] = coverage(arrays)
                for name, value in arrays.items():
                    raw[f"{key}/{name}"] = value
                complete_runs &= error is None and len(arrays["actions"]) == args.steps
                print(json.dumps({"completed": key, "status_error": error, "coverage": coverage_reports[key]}), flush=True)
    comparison, failures = compare(raw, args.repeats) if complete_runs else ({}, ["incomplete_or_failed_replay"])
    if any(value["illegal_input_count"] for value in coverage_reports.values()):
        failures.append("recorded_action_mask_invalid")
    reset_coverage = all(value["begin_reset_by_arena"][3] and value["complete_reset_by_arena"][3]
                         and value["mixed_reset_substeps"] for value in coverage_reports.values())
    if not reset_coverage:
        failures.append("mixed_native_reset_not_observed")
    result = {
        "schema": "rek.gpu_optimization_full_replay.v1", "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(config.device), "seed": args.seed,
        "repeats_per_branch": args.repeats, "ticks_per_replay": args.steps,
        "simulated_seconds_per_replay": args.steps * .02, "arenas": 4,
        "scenario_by_arena": SCENARIOS,
        "candidate_conditional_reset_forward": candidate_config.conditional_reset_forward,
        "candidate_fused_combat_library": str(candidate_config.fused_combat_library) if candidate_config.fused_combat_library else None,
        "candidate_defer_substep_combat_observations": candidate_config.defer_substep_combat_observations,
        "original_conditional_reset_forward": original_config.conditional_reset_forward,
        "original_fused_combat_library": str(original_config.fused_combat_library) if original_config.fused_combat_library else None,
        "original_defer_substep_combat_observations": original_config.defer_substep_combat_observations,
        "intervention": "deferred_combat_observation_only" if args.deferred_combat_library else "reset_forward_and_measurement_options",
        "actions_from": str(args.actions_from) if args.actions_from else None,
        "actions_from_sha256": digest(args.actions_from) if args.actions_from else None,
        "independent_original_graphs": args.independent_original_graphs,
        "comparison": comparison, "failures": failures, "criterion_passed": not failures,
        "statuses": statuses, "coverage": coverage_reports, "execution_order": order,
        "manifest": manifest, "common_restored_physics_arrays": len(common),
        "recorded_actions_sha256": hashlib.sha256(action_trace.tobytes()).hexdigest(),
        "requested_mask_substitutions_by_fighter": (action_trace != requested[:len(action_trace)]).sum(axis=0).tolist(),
        "setup_and_pilot_seconds": setup_seconds, "replays_wall_seconds": time.perf_counter() - start,
        "training_sps_measured": False, "authentic_rek_parity_established": False,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "warp-lang", "mujoco-warp", "mujoco")},
        "limits": [
            "Synthetic arena 3 fall geometry is not evidence of a physically generated KO or a scoring kick.",
            "Original and candidate use separate production-style CUDA graphs and allocators.",
            "Native/controller reset uses existing reset methods; all explicit MuJoCo Data arrays start from one snapshot.",
            "Graph allocator scratch is not copied across distinct graphs; original repeats measure its numerical variation.",
            "Exact semantic differences are reported even if original repeated runs also differ.",
            "Numerical criterion is the existing paired 95% bootstrap upper-extra-RMS rule, with no added epsilon.",
            "Per-tick synchronization and recording are diagnostic overhead; this run is not a throughput benchmark.",
            "Zero terminal values are compared exactly, but nonzero terminal coverage requires its explicit report flag.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.out.with_suffix(".npz")
    with trace_path.open("xb") as stream:
        np.savez_compressed(stream, **raw)
    result["raw_arrays_path"], result["raw_arrays_sha256"] = str(trace_path), digest(trace_path)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for harness in harnesses.values():
        harness.env.close()
    print(json.dumps({"report": str(args.out), "criterion_passed": not failures, "failures": failures}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fused-combat-library", type=Path)
    parser.add_argument("--deferred-combat-library", type=Path,
                        help="Compare only deferred observation packing; preserve configured reset/fusion in both branches")
    parser.add_argument("--conditional-reset-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actions-from", type=Path)
    parser.add_argument("--independent-original-graphs", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73019)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 224 or args.repeats < 2:
        parser.error("steps must be at least 224 and repeats at least two")
    raise SystemExit(0 if run(args)["criterion_passed"] else 1)
