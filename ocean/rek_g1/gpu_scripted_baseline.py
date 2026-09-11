"""Observation-only scripted controls against the unchanged candidate dummy.

These are explicitly scripted baselines, never learned policies or authentic
REK parity results. Selection and counters remain on CUDA during an actual
probe. CPU tensors are accepted only by the independent regression fixtures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import socket
import time
import traceback

import torch

from evaluate_gpu_dummy import RoundJournal, file_record, ongoing_snapshot
from gpu_candidate_dummy import DUMMY_LABEL, GpuCandidateApproachDummy


# semantic_duel_runtime.h / gpu_observation.py, schema 4 (223 floats).
SELF_QUATERNION = slice(3, 7)  # wxyz
OPPONENT_XY = slice(86, 88)
SELF_FALL_PHASE, OPPONENT_FALL_PHASE = 79, 165
COMPOSER_BUSY, FIGHT_PHASE = 183, 185
# g1_semantic_action_table.c and human_eval_server.py MOVE_METADATA.
CONTINUE, NEUTRAL, FORWARD, BACKWARD, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3, 6, 7
FORWARD_LEFT, FORWARD_RIGHT = 8, 9
LEFT_SIDE_KICK, LEFT_FRONT_KICK, RIGHT_SIDE_KICK = 16, 17, 18
KICK_CATEGORIES = (LEFT_SIDE_KICK, LEFT_FRONT_KICK, RIGHT_SIDE_KICK)


@dataclass(frozen=True)
class Strategy:
    name: str
    attack: int | None
    face: bool = True
    approach: bool = True
    minimum_range_m: float = 0.72
    maximum_range_m: float = 1.25
    bearing_tolerance_radians: float = 0.16
    cycle_kicks: bool = False
    combined_approach_yaw: bool = False


STRATEGIES = (
    Strategy("neutral", None, face=False, approach=False),
    Strategy("face_only", None, approach=False),
    Strategy("face_front_kick", LEFT_FRONT_KICK),
    Strategy("face_left_side_kick", LEFT_SIDE_KICK),
    Strategy("face_right_side_kick", RIGHT_SIDE_KICK),
    Strategy("face_cycle_kicks", LEFT_SIDE_KICK, cycle_kicks=True),
    Strategy("face_front_kick_close", LEFT_FRONT_KICK,
             minimum_range_m=0.60, maximum_range_m=0.90),
    Strategy("approach_yaw_front_kick", LEFT_FRONT_KICK, combined_approach_yaw=True),
)


def bearing_and_range(observations):
    """Root local +X bearing, using exactly learner-visible wxyz and XY."""
    q = observations[:, SELF_QUATERNION].double()
    norm = torch.linalg.vector_norm(q, dim=1)
    w, x, y, z = (q / norm[:, None]).unbind(1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    dx, dy = (observations[:, OPPONENT_XY].double() - observations[:, :2].double()).unbind(1)
    distance = torch.sqrt(dx*dx + dy*dy)
    forward = torch.cos(yaw)*dx + torch.sin(yaw)*dy
    lateral = -torch.sin(yaw)*dx + torch.cos(yaw)*dy
    return torch.atan2(lateral, forward), distance, norm


class GpuScriptedPolicies:
    """One statically assigned strategy per arena, no privileged state input.

    Range bands are declared policy choices, not recovered strike reach. A
    legal kick request is issued only when aligned and both robots are upright.
    A blocked request is released to neutral, permitting translation to settle.
    The legal mask remains authoritative even when composer_busy is true:
    held yaw can be interrupted by an allowed kick. A continue-only mask keeps
    its current command. No additional move is queued. The cycle advances only
    when its legal kick category is actually selected.
    """

    def __init__(self, arenas, device, strategies=STRATEGIES):
        self.strategies = tuple(strategies)
        if not self.strategies or arenas < len(self.strategies) or arenas % len(self.strategies):
            raise ValueError("arena count must be a positive multiple of strategy count")
        for value in self.strategies:
            if value.attack not in (None, *KICK_CATEGORIES):
                raise ValueError("scripted attack must be one of the three named kicks")
            if not 0 <= value.minimum_range_m < value.maximum_range_m < math.inf:
                raise ValueError("invalid strategy range band")
            if not 0 < value.bearing_tolerance_radians < math.pi:
                raise ValueError("invalid facing tolerance")
        self.arenas = arenas
        self.per_strategy = arenas // len(self.strategies)
        self.strategy_ids = torch.arange(len(self.strategies), device=device).repeat_interleave(self.per_strategy)
        tensor = lambda values, dtype: torch.tensor(values, device=device, dtype=dtype).repeat_interleave(self.per_strategy)
        self.attack = tensor([value.attack if value.attack is not None else NEUTRAL for value in self.strategies], torch.int64)
        self.face = tensor([value.face for value in self.strategies], torch.bool)
        self.approach = tensor([value.approach for value in self.strategies], torch.bool)
        self.near = tensor([value.minimum_range_m for value in self.strategies], torch.float64)
        self.far = tensor([value.maximum_range_m for value in self.strategies], torch.float64)
        self.tolerance = tensor([value.bearing_tolerance_radians for value in self.strategies], torch.float64)
        self.cycle = tensor([value.cycle_kicks for value in self.strategies], torch.bool)
        self.combined = tensor([value.combined_approach_yaw for value in self.strategies], torch.bool)
        self.cycle_offset = torch.zeros(arenas, device=device, dtype=torch.int64)
        self.actions = torch.ones((arenas, 1), device=device, dtype=torch.int32)
        self.preferred = torch.ones(arenas, device=device, dtype=torch.int64)
        self.substitutions = torch.zeros(arenas, device=device, dtype=torch.int64)
        self.status = torch.zeros(arenas, device=device, dtype=torch.int32)

    def reset(self, terminals=None):
        if terminals is None:
            self.cycle_offset.zero_()
            self.actions.fill_(NEUTRAL)
            self.preferred.fill_(NEUTRAL)
            self.substitutions.zero_()
            self.status.zero_()
        else:
            self.cycle_offset.masked_fill_(terminals != 0, 0)

    def select(self, observations, masks):
        if observations.shape != (self.arenas, 223) or masks.shape != (self.arenas, 33):
            raise ValueError("scripted policy requires learner observations and native masks")
        if observations.device != self.actions.device or masks.device != self.actions.device:
            raise ValueError("scripted inputs must remain on the policy device")
        bearing, distance, norm = bearing_and_range(observations)
        turning = self.face & (bearing.abs() > self.tolerance)
        yaw = torch.where(bearing > 0, YAW_LEFT, YAW_RIGHT)
        attack = torch.where(self.cycle, self.cycle_offset + LEFT_SIDE_KICK, self.attack)
        preferred = torch.where(self.approach & (distance > self.far), FORWARD, attack)
        preferred = torch.where(self.approach & (distance < self.near), BACKWARD, preferred)
        turn_action = torch.where(
            self.combined & (distance > self.far) & (bearing.abs() < math.pi / 2),
            torch.where(bearing > 0, FORWARD_LEFT, FORWARD_RIGHT), yaw,
        )
        preferred = torch.where(turning, turn_action, preferred)
        both_upright = (observations[:, SELF_FALL_PHASE] == 0) & (observations[:, OPPONENT_FALL_PHASE] == 0)
        active = observations[:, FIGHT_PHASE] == 2
        preferred = torch.where(active & both_upright, preferred, NEUTRAL)
        preferred = torch.where(self.face | self.approach, preferred, NEUTRAL)
        allowed = masks.gather(1, preferred[:, None]).squeeze(1) != 0
        selected = torch.where(allowed, preferred, torch.where(masks[:, NEUTRAL] != 0, NEUTRAL, CONTINUE))
        selected_allowed = masks.gather(1, selected[:, None]).squeeze(1) != 0
        self.preferred.copy_(preferred)
        self.actions[:, 0].copy_(selected)
        self.substitutions.add_((selected != preferred).to(torch.int64))
        self.cycle_offset.copy_(torch.where(
            self.cycle & (selected >= LEFT_SIDE_KICK) & (selected <= RIGHT_SIDE_KICK),
            (self.cycle_offset + 1).remainder(3), self.cycle_offset,
        ))
        self.status.bitwise_or_(
            (~torch.isfinite(observations).all(1)).to(torch.int32)
            | ((~torch.isfinite(norm) | (norm <= 1e-9)).to(torch.int32) * 2)
            | ((~((masks == 0) | (masks == 1)).all(1)).to(torch.int32) * 4)
            | ((~selected_allowed).to(torch.int32) * 8)
        )
        return self.actions

    def check_status(self):
        values = self.status.detach().cpu().tolist()
        if any(values):
            raise RuntimeError(f"invalid scripted-policy observation/mask: {values}")

    def assignments(self):
        return [dict(asdict(value), arenas=list(range(index * self.per_strategy, (index + 1) * self.per_strategy)))
                for index, value in enumerate(self.strategies)]


class PerArenaDiagnostics:
    """CUDA counters without merging strategies or attributing arena hits."""

    columns = ("control_steps", "active_samples", "facing_samples", "attack_requests",
               "facing_attack_requests", "native_move_starts", "learner_points_all_steps",
               "opponent_points_all_steps", "arena_attributed_contacts_all_steps",
               "arena_scored_hits_all_steps", "learner_down_samples", "opponent_down_samples",
               "range_sum_m", "learner_reward_sum", "terminal_pair_mismatches")

    def __init__(self, arenas, device):
        self.counts = torch.zeros((arenas, len(self.columns)), device=device, dtype=torch.float64)
        self.action_counts = torch.zeros((arenas, 33), device=device, dtype=torch.int64)
        self.categories = torch.arange(33, device=device)

    def reset(self):
        self.counts.zero_()
        self.action_counts.zero_()

    def update(self, full_observations, terminals, actions, move_starts, rewards):
        obs = full_observations[0::2]
        bearing, distance, _ = bearing_and_range(obs)
        active = obs[:, FIGHT_PHASE] == 2
        facing = active & (bearing.abs() <= math.pi / 6)
        attack = (actions[:, 0] >= LEFT_SIDE_KICK) & (actions[:, 0] <= RIGHT_SIDE_KICK)
        self.action_counts.add_((actions == self.categories[None, :]).to(torch.int64))
        self.counts.add_(torch.stack((
            torch.ones_like(distance), active, facing, attack, facing & attack,
            move_starts[0::2] != 0, obs[:, 217], obs[:, 218], obs[:, 221], obs[:, 222],
            active & (obs[:, SELF_FALL_PHASE] != 0), active & (obs[:, OPPONENT_FALL_PHASE] != 0),
            torch.where(active, distance, 0), rewards[0::2], terminals[0::2] != terminals[1::2],
        ), dim=1).double())

    def snapshot(self):
        counts, actions = self.counts.cpu().tolist(), self.action_counts.cpu().tolist()
        return [dict(zip(self.columns, row), arena=arena, action_category_counts=actions[arena])
                for arena, row in enumerate(counts)]


def summarize(assignments, outcomes, diagnostics, substitutions):
    summaries = []
    for assignment in assignments:
        arenas = set(assignment["arenas"])
        rounds = [event for event in outcomes if event["arena"] in arenas]
        counters = {key: sum(row[key] for row in diagnostics if row["arena"] in arenas)
                    for key in PerArenaDiagnostics.columns}
        total = len(rounds)
        wins = sum(event["outcome"] == "learner_win" for event in rounds)
        mean = lambda field: sum(event[field] for event in rounds) / total if total else None
        percent = lambda a, b: 100 * a / b if b else None
        summaries.append({
            "strategy": assignment, "completed_rounds": total, "wins": wins,
            "losses": sum(event["outcome"] == "opponent_win" for event in rounds),
            "ties": sum(event["outcome"] == "tie" for event in rounds),
            "redos": sum(event["outcome"] == "redo" for event in rounds),
            "win_percent": percent(wins, total),
            "learner_points_per_completed_round": mean("learner_points"),
            "opponent_points_per_completed_round": mean("opponent_points"),
            "learner_falls_per_completed_round": mean("learner_falls"),
            "opponent_falls_per_completed_round": mean("opponent_falls"),
            "arena_scored_hits_per_completed_round": mean("arena_scored_hits"),
            "learner_scored_hits": None,
            "counters": counters,
            "action_mask_substitutions": sum(substitutions[arena] for arena in arenas),
            "facing_percent": percent(counters["facing_samples"], counters["active_samples"]),
            "attack_facing_percent": percent(counters["facing_attack_requests"], counters["attack_requests"]),
            "mean_range_m": counters["range_sum_m"] / counters["active_samples"] if counters["active_samples"] else None,
        })
    return summaries


def probe(args):
    from gpu_semantic_duel import GpuSemanticDuel
    from verify_gpu_duel import load_config

    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if args.ticks < 1 or args.horizon < 2 or args.ticks % args.horizon:
        raise ValueError("ticks must be a positive multiple of journal horizon >= 2")
    names = args.strategies.split(",") if args.strategies else [value.name for value in STRATEGIES]
    by_name = {value.name: value for value in STRATEGIES}
    if len(names) != len(set(names)) or any(name not in by_name for name in names):
        raise ValueError("unknown or duplicate strategy name")
    strategies = tuple(by_name[name] for name in names)
    config = load_config(args.config)
    if torch.device(config.device).type != "cuda":
        raise ValueError("actual scripted baseline requires CUDA simulation")
    args.run_dir.mkdir(parents=True)
    duel = None
    actual_ticks = 0
    outcomes = []
    try:
        setup_start = time.perf_counter()
        duel = GpuSemanticDuel(config)
        duel.capture_step()
        learner = GpuScriptedPolicies(duel.arenas, config.device, strategies)
        dummy = GpuCandidateApproachDummy(duel.arenas, config.device)
        diagnostics = PerArenaDiagnostics(duel.arenas, config.device)
        journal = RoundJournal(duel.arenas, args.horizon, config.device)
        full_actions = torch.ones((duel.rows, 1), dtype=torch.int32, device=config.device)

        def select():
            learner.reset(duel.terminals[0::2])
            dummy.reset(duel.terminals[1::2])
            full_actions[0::2].copy_(learner.select(duel.observations[0::2], duel.action_mask[0::2]))
            full_actions[1::2, 0].copy_(dummy.select(duel.observations[1::2], duel.action_mask[1::2]))

        def measure():
            diagnostics.update(duel.observations, duel.terminals, learner.actions,
                               duel.scheduler.move_start_edge, duel.rewards)

        stream = torch.cuda.Stream(device=config.device)
        stream.wait_stream(torch.cuda.current_stream(config.device))
        with torch.cuda.stream(stream):
            select()
            measure()
            select_graph, measurement_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(select_graph, stream=stream):
                select()
            with torch.cuda.graph(measurement_graph, stream=stream):
                measure()
        stream.synchronize()
        duel.reset()
        learner.reset()
        dummy.reset()
        diagnostics.reset()
        initial_observations = duel.observations.cpu().contiguous().numpy()
        initial_sha = hashlib.sha256(initial_observations.tobytes()).hexdigest()
        torch.cuda.synchronize(config.device)
        setup_seconds = time.perf_counter() - setup_start
        start, cpu_start = time.perf_counter(), time.process_time()
        with (args.run_dir / "rounds.jsonl").open("x", encoding="utf-8") as stream:
            for previous in range(0, args.ticks, args.horizon):
                for _ in range(args.horizon):
                    select_graph.replay()
                    duel.step(full_actions)
                    measurement_graph.replay()
                    journal.record(duel.observations, duel.terminals)
                    actual_ticks += 1
                events = journal.drain(previous)
                outcomes.extend(events)
                for event in events:
                    event["strategy"] = strategies[event["arena"] // learner.per_strategy].name
                    stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                if (previous // args.horizon + 1) % args.log_every == 0 or actual_ticks == args.ticks:
                    duel.check_status()
                    learner.check_status()
                    dummy.check_status()
                    print(json.dumps({"control_ticks_per_arena": actual_ticks, "completed_rounds": len(outcomes)}), flush=True)
        torch.cuda.synchronize(config.device)
        elapsed, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
        duel.check_status()
        learner.check_status()
        dummy.check_status()
        arena_diagnostics = diagnostics.snapshot()
        substitutions = learner.substitutions.cpu().tolist()
        assignments = learner.assignments()
        report = {
            "schema": "rek.scripted_policy_baseline.v1", "status": "completed",
            "policy_kind": "scripted_baseline", "policy_updates": 0,
            "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(config.device),
            "opponent": DUMMY_LABEL, "opponent_is_authentic_bot_1": False,
            "arenas": duel.arenas, "physical_fighters": duel.rows,
            "control_ticks_per_arena": actual_ticks, "simulated_seconds_per_arena": actual_ticks * 0.02,
            "learner_control_steps": actual_ticks * duel.arenas,
            "instrumented_wall_seconds": elapsed, "host_cpu_seconds": cpu_seconds,
            "setup_seconds": setup_seconds, "training_sps_measured": False,
            "cpu_physics_steps": 0, "cpu_controller_inferences": 0,
            "initial_observations_sha256": initial_sha,
            "configuration": file_record(args.config),
            "sources": {name: file_record(Path(__file__).with_name(name)) for name in (
                "gpu_scripted_baseline.py", "gpu_candidate_dummy.py", "gpu_semantic_duel.py",
                "evaluate_gpu_dummy.py", "gpu_observation.py",
            )},
            "round_events": file_record(args.run_dir / "rounds.jsonl"),
            "summaries": summarize(assignments, outcomes, arena_diagnostics, substitutions),
            "per_arena_diagnostics": arena_diagnostics,
            "cutoff_ongoing_rounds": ongoing_snapshot(duel, journal),
            "claim_limits": [
                "Deterministic scripted strategies, not trained policies or a human comparison.",
                "Unchanged candidate dummy, model, reset, reward, masks and simulation timing.",
                "Policy decisions use learner observations, native legal masks and its own cycle state only.",
                "Range bands and facing tolerance are explicit policy choices, not measured strike zones.",
                "Parallel arenas share the fixed initial condition; outcomes are not independent random scenario trials.",
                "Arena scored hits cannot be attributed to the learner. Points include referee awards.",
                "Completed rounds are not best-of-three fight wins; cutoff ongoing rounds are separate.",
                "Instrumented scripted evaluation measures no training throughput or authentic REK parity.",
            ],
        }
        with (args.run_dir / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "completed", "report": str(args.run_dir / "report.json"),
                          "results": [{"strategy": row["strategy"]["name"], "rounds": row["completed_rounds"],
                                       "wins": row["wins"], "win_percent": row["win_percent"]} for row in report["summaries"]]}), flush=True)
        return report
    except Exception as error:
        with (args.run_dir / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "failed", "actual_ticks": actual_ticks,
                       "error": str(error), "traceback": traceback.format_exc()}, stream, indent=2)
        raise
    finally:
        if duel is not None:
            duel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=6400)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--strategies", default=None, help="comma-separated declared names, default all eight")
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    probe(args)


if __name__ == "__main__":
    main()
