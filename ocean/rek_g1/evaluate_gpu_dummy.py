"""Frozen native Puffer policy evaluation against the candidate scripted dummy.

The native rollout, action sampling, horizon RNN reset and environment reset
paths are shared with training. No PPO update is performed. A one-horizon CUDA
journal preserves every completed round at explicit reporting boundaries.
This measures the semantic candidate, not authentic REK or human performance.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch

from gpu_metrics import (
    CONTROL_DELTA_SECONDS, CURRENT_ROUND_IS_REDO, KNOCKOUT_OCCURRED,
    ROUND_RESULT, ROUND_WINNER, SIDE0_FALLS, SIDE0_POINTS, SIDE1_FALLS,
    SIDE1_POINTS, TICK_ATTRIBUTED_CONTACTS, TICK_SCORED_HITS,
)


def file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def pinned_checkpoint(path, expected):
    expected = expected.lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("checkpoint SHA256 must contain 64 hexadecimal digits")
    record = file_record(path)
    if record["sha256"] != expected:
        raise ValueError("checkpoint SHA256 mismatch")
    return record


def evaluation_config(native, *, fighters, horizon, ticks, seed, minibatch_size=0):
    if fighters < 2 or fighters % 2:
        raise ValueError("fighters must be a positive even count")
    if horizon <= 1 or ticks <= 0 or ticks % horizon:
        raise ValueError("ticks must be positive and divisible by horizon > 1")
    if seed < 0 or seed > 2147483647:
        raise ValueError("native seed must fit a nonnegative signed 32-bit integer")
    learners = fighters // 2
    batch = learners * horizon
    minibatch = minibatch_size or batch
    if minibatch <= 0 or minibatch > batch or minibatch % horizon:
        raise ValueError("minibatch must be horizon-divisible and fit one rollout")
    result = deepcopy(native)
    result["vec"].update(total_agents=learners, num_buffers=1, num_threads=0)
    result["train"].update(total_timesteps=learners * ticks, horizon=horizon,
                           minibatch_size=minibatch, gpus=1)
    # Native bindings read the base seed, not train.seed.
    result.update(seed=seed, world_size=1, rank=0, nccl_id=b"")
    return result


class RoundJournal:
    """Bounded device journal; CPU is supported only for regression fixtures."""

    columns = (ROUND_RESULT, ROUND_WINNER, SIDE0_POINTS, SIDE1_POINTS,
               SIDE0_FALLS, SIDE1_FALLS, KNOCKOUT_OCCURRED, CURRENT_ROUND_IS_REDO)
    fields = ("result", "winner_side", "learner_points", "opponent_points",
              "learner_falls", "opponent_falls", "knockout", "redo_round",
              "control_ticks", "arena_scored_hits", "arena_attributed_contacts")

    def __init__(self, arenas, horizon, device):
        if arenas < 1 or horizon < 2:
            raise ValueError("journal needs positive arenas and horizon > 1")
        self.rows = torch.empty((horizon, arenas, 12), dtype=torch.float64, device=device)
        self.indices = torch.tensor(self.columns, dtype=torch.int64, device=device)
        self.ongoing = torch.zeros((arenas, 3), dtype=torch.float64, device=device)
        self.slot = 0

    def record(self, observations, terminals):
        if self.slot >= self.rows.shape[0]:
            raise RuntimeError("rollout exceeded the configured journal horizon")
        even = observations[0::2]
        terminal = terminals[0::2] != 0
        self.ongoing[:, 0].add_(1)
        self.ongoing[:, 1].add_(even[:, TICK_SCORED_HITS])
        self.ongoing[:, 2].add_(even[:, TICK_ATTRIBUTED_CONTACTS])
        destination = self.rows[self.slot]
        destination[:, 0].copy_(terminal)
        destination[:, 1:9].copy_(even.index_select(1, self.indices))
        destination[:, 9:12].copy_(self.ongoing)
        self.ongoing.mul_((~terminal)[:, None])
        self.slot += 1

    def drain(self, previous_ticks):
        if self.slot != self.rows.shape[0]:
            raise RuntimeError("native rollout did not fill the configured horizon")
        values = self.rows.detach().cpu().numpy()
        events = []
        for tick, arena in np.argwhere(values[:, :, 0] != 0):
            data = values[tick, arena, 1:]
            if not np.isfinite(data).all():
                raise RuntimeError("nonfinite completed-round metadata")
            if not np.equal(data, np.floor(data)).all():
                raise RuntimeError("nonintegral completed-round metadata")
            event = dict(zip(self.fields, (int(value) for value in data)))
            result, winner = event["result"], event["winner_side"]
            if not ((result in (1, 2) and winner in (0, 1))
                    or (result in (3, 4) and winner == -1)):
                raise RuntimeError("unclassified completed-round outcome")
            event.update(arena=int(arena), end_control_tick=previous_ticks + int(tick) + 1,
                         outcome=("learner_win" if winner == 0 else "opponent_win")
                         if result in (1, 2) else ("tie" if result == 3 else "redo"))
            events.append(event)
        self.slot = 0
        return events


def frozen_rollouts(trainer, duel, journal, *, ticks, horizon, on_horizon):
    """Use the unmodified native rollout while appending outcome observations."""
    original_step = trainer.env.step

    def recorded_step(actions):
        original_step(actions)
        journal.record(duel.observations, duel.terminals)

    trainer.env.step = recorded_step
    start_steps = trainer.global_step
    try:
        for previous in range(0, ticks, horizon):
            trainer.rollouts()
            events = journal.drain(previous)
            expected_steps = (previous + horizon) * (duel.rows // 2)
            if trainer.global_step - start_steps != expected_steps:
                raise RuntimeError("native learner-step count disagrees with executed ticks")
            on_horizon(previous + horizon, events)
    finally:
        trainer.env.step = original_step


def save_frozen_weights(trainer, path, expected_sha256):
    trainer.save_weights(path)
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise RuntimeError("evaluation policy weights differ from the pinned checkpoint")
    return record


def ongoing_snapshot(duel, journal):
    observations = duel.observations[0::2].detach().cpu().numpy()
    terminal = duel.terminals[0::2].detach().cpu().numpy() != 0
    counts = journal.ongoing.detach().cpu().numpy()
    result = []
    for arena in range(len(observations)):
        if terminal[arena]:
            continue
        row, count = observations[arena], counts[arena]
        if not np.isfinite(row).all() or not np.isfinite(count).all():
            raise RuntimeError("nonfinite ongoing evaluation state")
        result.append({
            "arena": arena, "phase": int(row[185]),
            "learner_points": float(row[SIDE0_POINTS]),
            "opponent_points": float(row[SIDE1_POINTS]),
            "learner_falls": int(row[SIDE0_FALLS]),
            "opponent_falls": int(row[SIDE1_FALLS]),
            "control_ticks": int(count[0]), "arena_scored_hits": int(count[1]),
            "arena_attributed_contacts": int(count[2]),
        })
    return result


def evaluate(args):
    # Load runtime modules lazily so pure CPU tests need no native CUDA library.
    from gpu_candidate_dummy import DUMMY_LABEL, GpuCandidateDummyDuel
    from gpu_native_puffer import NativeExternalGpuPuffer
    from gpu_puffer_env import CudaTensorEnvAdapter
    from train_gpu_duel import load_native_config
    from verify_gpu_duel import load_config

    checkpoint = pinned_checkpoint(args.checkpoint, args.checkpoint_sha256)
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if args.log_every < 1:
        raise ValueError("log_every must be positive")
    native = load_native_config(args.default_config, args.native_config)
    native = evaluation_config(native, fighters=args.total_agents, horizon=args.horizon,
                               ticks=args.ticks, seed=args.seed,
                               minibatch_size=args.minibatch_size)
    config = load_config(args.gpu_duel_config)
    native["gpu_id"] = torch.device(config.device).index or 0
    module_name, separator, attribute = args.environment_factory.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("environment_factory must be module:callable")
    factory_module = importlib.import_module(module_name)
    factory = getattr(factory_module, attribute)
    args.run_dir.mkdir(parents=True)
    trainer = duel = None
    actual_ticks, event_count = 0, 0
    try:
        setup_start = time.perf_counter()
        duel = factory(config)
        if duel.rows != args.total_agents:
            raise ValueError("configured fighter count disagrees with CUDA controller batch")
        duel.capture_step()
        candidate = GpuCandidateDummyDuel(duel)
        adapter = CudaTensorEnvAdapter(candidate, (33,), metric_plugins=(candidate.metric_plugin,))
        trainer = NativeExternalGpuPuffer(native, adapter, reward_clip=0.0)
        trainer.load_weights(args.checkpoint)
        initial = save_frozen_weights(trainer, args.run_dir / "policy-before.bin", checkpoint["sha256"])
        journal = RoundJournal(duel.rows // 2, args.horizon, config.device)
        initial_state = duel.physics.qpos.detach().cpu().contiguous().numpy()
        initial_state_sha = hashlib.sha256(initial_state.tobytes()).hexdigest()
        torch.cuda.synchronize(config.device)
        setup_seconds = time.perf_counter() - setup_start
        start, cpu_start = time.perf_counter(), time.process_time()
        with (args.run_dir / "rounds.jsonl").open("x", encoding="utf-8") as outcomes:
            def on_horizon(ticks, events):
                nonlocal actual_ticks, event_count
                actual_ticks = ticks
                event_count += len(events)
                for event in events:
                    outcomes.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                outcomes.flush()
                if ticks // args.horizon % args.log_every == 0 or ticks == args.ticks:
                    duel.check_status()
                    candidate.dummy.check_status()
                    print(json.dumps({"control_ticks_per_arena": ticks,
                                      "completed_rounds": event_count}), flush=True)
            frozen_rollouts(trainer, duel, journal, ticks=args.ticks,
                            horizon=args.horizon, on_horizon=on_horizon)
        torch.cuda.synchronize(config.device)
        wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
        final_log = trainer.eval_log(clear_metrics=False)
        ongoing = ongoing_snapshot(duel, journal)
        final = save_frozen_weights(trainer, args.run_dir / "policy-after.bin", checkpoint["sha256"])
        pinned_checkpoint(args.checkpoint, checkpoint["sha256"])
        behavior = final_log["env"]["behavior"]
        if event_count != behavior["completed_rounds"] or event_count != final_log["env"]["n"]:
            raise RuntimeError("round journal disagrees with existing training metrics")
        expected_steps = args.ticks * (duel.rows // 2)
        if behavior["learner_control_steps"] != expected_steps:
            raise RuntimeError("behavior sample count disagrees with executed ticks")
        sources = {}
        for name in ("gpu_native_puffer", "gpu_semantic_duel", "gpu_candidate_dummy",
                     "gpu_metrics", "gpu_behavior_metrics", "gpu_combat_measurement",
                     "gpu_duel_physics", "gpu_controller", module_name):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "__file__", None):
                sources[name] = file_record(module.__file__)
        sources["evaluate_gpu_dummy"] = file_record(__file__)
        report = {
            "schema": "rek.g1_frozen_native_candidate_dummy_evaluation.v1",
            "status": "completed", "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(config.device), "sources": sources,
            "native_extension": file_record(trainer.backend.__file__),
            "inputs": {"checkpoint": checkpoint, "gpu_duel_config": file_record(args.gpu_duel_config),
                       "default_config": file_record(args.default_config),
                       "native_config": file_record(args.native_config),
                       "environment_factory": args.environment_factory},
            "evaluation": {
                "opponent": DUMMY_LABEL, "opponent_is_authentic_bot_1": False,
                "policy_side": 0, "dummy_side": 1, "sampling": "native stochastic categorical",
                "seed": args.seed, "horizon": args.horizon,
                "reset_state_each_horizon": bool(native["reset_state"]),
                "native_policy_configuration": native["policy"],
                "native_graph_capture_epoch": native["cudagraphs"],
                "native_setup_minibatch_size": native["train"]["minibatch_size"],
                "initial_qpos_sha256": initial_state_sha,
                "arenas": duel.rows // 2, "physical_fighters": duel.rows,
                "requested_control_ticks_per_arena": args.ticks,
                "actual_control_ticks_per_arena": actual_ticks,
                "simulated_seconds_per_arena": actual_ticks * CONTROL_DELTA_SECONDS,
                "learner_control_steps": trainer.global_step,
                "physical_fighter_control_steps": trainer.global_step * 2,
                "policy_updates": 0, "weights_unchanged": initial["sha256"] == final["sha256"],
                "setup_seconds": setup_seconds, "instrumented_evaluation_wall_seconds": wall,
                "host_cpu_seconds": cpu,
                "instrumented_evaluation_learner_steps_per_second": trainer.global_step / wall,
                "cpu_physics_steps": 0, "cpu_controller_inferences": 0,
                "native_cpu_environment_count": int(trainer.pufferl.native_env_count),
                "native_environment_threads": bool(trainer.pufferl.has_env_threads),
            },
            "checkpoints": {"before": initial, "after": final},
            "completed_rounds": event_count, "round_events": file_record(args.run_dir / "rounds.jsonl"),
            "cutoff_ongoing_rounds": ongoing, "final_native_log": final_log,
            "claim_limits": [
                "The opponent is the human-eval candidate script, not authentic REK Bot 1.",
                "Round results do not establish full-match, human, or authentic REK performance.",
                "Points include referee awards; scored-hit metadata is arena-wide.",
                "Identical seed, horizon, arena count and initial state are required for checkpoint comparisons.",
                "Outcome journaling adds reporting cost; this throughput is not training SPS.",
            ],
        }
        with (args.run_dir / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return report
    except Exception as error:
        failure = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                   "completed_control_ticks_per_arena": actual_ticks, "completed_rounds": event_count}
        with (args.run_dir / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump(failure, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        raise
    finally:
        if trainer is not None:
            trainer.close()
        elif duel is not None:
            duel.close()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--gpu-duel-config", type=Path, required=True)
    parser.add_argument("--default-config", type=Path, default=root / "config/default.ini")
    parser.add_argument("--native-config", type=Path, default=root / "config/rek_g1.ini")
    parser.add_argument("--total-agents", type=int, required=True, help="total physical fighter rows")
    parser.add_argument("--ticks", type=int, required=True, help="control ticks per arena, horizon-divisible")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=0, help="native setup only; no PPO updates")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--environment-factory", default="gpu_semantic_duel:GpuSemanticDuel")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
