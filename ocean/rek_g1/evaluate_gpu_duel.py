"""Role-balanced CUDA evaluation for two native Puffer REK G1 checkpoints.

This module performs inference only. It loads the verified PyTorch realization
of the native linear-MinGRU checkpoint layout and drives the same CUDA semantic
duel used by training. Checkpoint A controls side 0 in half of the arenas and
side 1 in the other half; checkpoint B receives the complementary rows.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import socket
import time
from typing import Any, Sequence

import torch

try:
    from .gpu_metrics import (
        CURRENT_ROUND_IS_REDO,
        KNOCKOUT_OCCURRED,
        OBSERVATION_FLOATS,
        ROUND_REDO,
        ROUND_RESULT,
        ROUND_TIE,
        ROUND_WINNER,
        ROUND_WON_BY_KO,
        ROUND_WON_BY_POINTS,
        SIDE0_FALLS,
        SIDE0_POINTS,
        SIDE1_FALLS,
        SIDE1_POINTS,
        TICK_ATTRIBUTED_CONTACTS,
        TICK_SCORED_HITS,
    )
    from .gpu_puffer_env import CudaTensorEnvAdapter
    from .gpu_puffer_policy import NativePufferPolicy
    from .gpu_semantic_duel import GpuSemanticDuel
    from .verify_gpu_duel import load_config as load_gpu_duel_config
except ImportError:
    from gpu_metrics import (
        CURRENT_ROUND_IS_REDO,
        KNOCKOUT_OCCURRED,
        OBSERVATION_FLOATS,
        ROUND_REDO,
        ROUND_RESULT,
        ROUND_TIE,
        ROUND_WINNER,
        ROUND_WON_BY_KO,
        ROUND_WON_BY_POINTS,
        SIDE0_FALLS,
        SIDE0_POINTS,
        SIDE1_FALLS,
        SIDE1_POINTS,
        TICK_ATTRIBUTED_CONTACTS,
        TICK_SCORED_HITS,
    )
    from gpu_puffer_env import CudaTensorEnvAdapter
    from gpu_puffer_policy import NativePufferPolicy
    from gpu_semantic_duel import GpuSemanticDuel
    from verify_gpu_duel import load_config as load_gpu_duel_config


CONTROL_DELTA_SECONDS = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    expected = expected_sha256.lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex digits")
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(
            f"checkpoint hash mismatch for {resolved}: expected {expected}, got {actual}"
        )
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual}


def _policy_shape(default_config: Path, native_config: Path) -> tuple[int, int]:
    parser = configparser.ConfigParser()
    loaded = parser.read((default_config, native_config))
    if loaded != [str(default_config), str(native_config)]:
        raise FileNotFoundError(
            f"failed to load policy configs: {default_config}, {native_config}"
        )
    hidden_size = parser.getint("policy", "hidden_size")
    num_layers = parser.getint("policy", "num_layers")
    if hidden_size <= 0 or num_layers <= 0:
        raise ValueError("native policy dimensions must be positive")
    return hidden_size, num_layers


def _masked_actions(
    logits: torch.Tensor,
    act_sizes: Sequence[int],
    action_mask: torch.Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Select categorical actions on CUDA while respecting every head mask."""
    sizes = tuple(int(size) for size in act_sizes)
    if logits.shape != action_mask.shape or logits.shape[-1] != sum(sizes):
        raise ValueError("policy logits and action mask have incompatible shapes")
    selections = []
    offset = 0
    for size in sizes:
        head_logits = logits[:, offset : offset + size]
        head_mask = action_mask[:, offset : offset + size].to(dtype=torch.bool)
        masked_logits = head_logits.masked_fill(~head_mask, -torch.inf)
        if deterministic:
            action = masked_logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(masked_logits, dim=-1)
            action = torch.multinomial(
                probabilities,
                1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
        selections.append(action)
        offset += size
    return torch.stack(selections, dim=-1)


class RoleBalancedMatchAccumulator:
    """Accumulate completed and cutoff-truncated arena outcomes on CUDA."""

    _SCALAR_NAMES = (
        "completed",
        "checkpoint_a_wins",
        "checkpoint_b_wins",
        "ties",
        "redo_results",
        "unclassified_results",
        "knockouts",
        "checkpoint_a_points",
        "checkpoint_b_points",
        "checkpoint_a_falls",
        "checkpoint_b_falls",
        "completed_scored_hits",
        "completed_attributed_contacts",
        "completed_semantic_steps",
        "completed_redo_rounds",
        "side0_wins",
        "side1_wins",
        "a_as_side0_completed",
        "a_as_side0_wins",
        "a_as_side0_points",
        "a_as_side1_completed",
        "a_as_side1_wins",
        "a_as_side1_points",
        "b_as_side0_completed",
        "b_as_side0_wins",
        "b_as_side0_points",
        "b_as_side1_completed",
        "b_as_side1_wins",
        "b_as_side1_points",
        "terminal_pair_mismatches",
        "action_mask_violations",
        "terminal_state_reset_max_abs",
    )

    def __init__(self, total_agents: int, device: torch.device | str) -> None:
        if total_agents < 4 or total_agents % 4:
            raise ValueError(
                "role-balanced evaluation requires a multiple of four fighter rows"
            )
        self.total_agents = int(total_agents)
        self.arena_count = self.total_agents // 2
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("checkpoint match accounting requires a CUDA device")
        self.a_is_side0 = torch.arange(
            self.arena_count, device=self.device
        ) < (self.arena_count // 2)
        self._scalars = torch.zeros(
            len(self._SCALAR_NAMES), dtype=torch.float64, device=self.device
        )
        self._index = {name: index for index, name in enumerate(self._SCALAR_NAMES)}
        self._ongoing_steps = torch.zeros(
            self.arena_count, dtype=torch.float64, device=self.device
        )
        self._ongoing_scored_hits = torch.zeros_like(self._ongoing_steps)
        self._ongoing_attributed_contacts = torch.zeros_like(self._ongoing_steps)
        self._latest_a_points = torch.zeros_like(self._ongoing_steps)
        self._latest_b_points = torch.zeros_like(self._ongoing_steps)
        self._last_terminal = torch.zeros(
            self.arena_count, dtype=torch.bool, device=self.device
        )

    def reset(self) -> None:
        self._scalars.zero_()
        self._ongoing_steps.zero_()
        self._ongoing_scored_hits.zero_()
        self._ongoing_attributed_contacts.zero_()
        self._latest_a_points.zero_()
        self._latest_b_points.zero_()
        self._last_terminal.zero_()

    def _add(self, name: str, value: torch.Tensor) -> None:
        self._scalars[self._index[name]].add_(value.to(dtype=torch.float64).sum())

    def record_action_validity(
        self,
        masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        offset = 0
        for head, size in enumerate((33,)):
            selected = actions[:, head].to(torch.int64).unsqueeze(-1)
            valid = masks[:, offset : offset + size].to(torch.bool).gather(1, selected)
            self._add("action_mask_violations", ~valid)
            offset += size

    def record_terminal_state_residual(
        self,
        state_a: torch.Tensor,
        terminal_a: torch.Tensor,
        state_b: torch.Tensor,
        terminal_b: torch.Tensor,
    ) -> None:
        a_residual = torch.where(
            terminal_a.view(1, -1, 1), state_a.abs(), 0.0
        ).amax()
        b_residual = torch.where(
            terminal_b.view(1, -1, 1), state_b.abs(), 0.0
        ).amax()
        index = self._index["terminal_state_reset_max_abs"]
        self._scalars[index].copy_(
            torch.maximum(self._scalars[index], torch.maximum(a_residual, b_residual))
        )

    def update(self, observations: torch.Tensor, terminals: torch.Tensor) -> None:
        if observations.shape != (self.total_agents, OBSERVATION_FLOATS):
            raise ValueError("unexpected REK G1 observation shape")
        if terminals.shape != (self.total_agents,):
            raise ValueError("unexpected REK G1 terminal shape")
        if observations.device != self.device or terminals.device != self.device:
            raise ValueError("match accounting tensors must remain on one CUDA device")

        even = observations[0::2]
        even_terminal = terminals[0::2] != 0
        odd_terminal = terminals[1::2] != 0
        self._last_terminal.copy_(even_terminal)
        self._add("terminal_pair_mismatches", even_terminal != odd_terminal)

        self._ongoing_steps.add_(1.0)
        self._ongoing_scored_hits.add_(even[:, TICK_SCORED_HITS].to(torch.float64))
        self._ongoing_attributed_contacts.add_(
            even[:, TICK_ATTRIBUTED_CONTACTS].to(torch.float64)
        )

        side0_points = even[:, SIDE0_POINTS].to(torch.float64)
        side1_points = even[:, SIDE1_POINTS].to(torch.float64)
        side0_falls = even[:, SIDE0_FALLS].to(torch.float64)
        side1_falls = even[:, SIDE1_FALLS].to(torch.float64)
        a_points = torch.where(self.a_is_side0, side0_points, side1_points)
        b_points = torch.where(self.a_is_side0, side1_points, side0_points)
        a_falls = torch.where(self.a_is_side0, side0_falls, side1_falls)
        b_falls = torch.where(self.a_is_side0, side1_falls, side0_falls)
        self._latest_a_points.copy_(a_points)
        self._latest_b_points.copy_(b_points)

        result = even[:, ROUND_RESULT]
        winner = even[:, ROUND_WINNER]
        decisive = (result == ROUND_WON_BY_POINTS) | (result == ROUND_WON_BY_KO)
        valid_decisive = decisive & ((winner == 0) | (winner == 1))
        ties = (result == ROUND_TIE) & (winner == -1)
        redo_results = (result == ROUND_REDO) & (winner == -1)
        classified = valid_decisive | ties | redo_results
        a_won = valid_decisive & torch.where(
            self.a_is_side0, winner == 0, winner == 1
        )
        b_won = valid_decisive & torch.where(
            self.a_is_side0, winner == 1, winner == 0
        )
        side0_won = valid_decisive & (winner == 0)
        side1_won = valid_decisive & (winner == 1)
        terminal = even_terminal

        self._add("completed", terminal)
        self._add("checkpoint_a_wins", terminal & a_won)
        self._add("checkpoint_b_wins", terminal & b_won)
        self._add("ties", terminal & ties)
        self._add("redo_results", terminal & redo_results)
        self._add("unclassified_results", terminal & ~classified)
        self._add("knockouts", terminal & (even[:, KNOCKOUT_OCCURRED] != 0))
        self._add("checkpoint_a_points", terminal * a_points)
        self._add("checkpoint_b_points", terminal * b_points)
        self._add("checkpoint_a_falls", terminal * a_falls)
        self._add("checkpoint_b_falls", terminal * b_falls)
        self._add("completed_scored_hits", terminal * self._ongoing_scored_hits)
        self._add(
            "completed_attributed_contacts",
            terminal * self._ongoing_attributed_contacts,
        )
        self._add("completed_semantic_steps", terminal * self._ongoing_steps)
        self._add("completed_redo_rounds", terminal & (even[:, CURRENT_ROUND_IS_REDO] != 0))
        self._add("side0_wins", terminal & side0_won)
        self._add("side1_wins", terminal & side1_won)

        a_side0_terminal = terminal & self.a_is_side0
        a_side1_terminal = terminal & ~self.a_is_side0
        self._add("a_as_side0_completed", a_side0_terminal)
        self._add("a_as_side0_wins", a_side0_terminal & a_won)
        self._add("a_as_side0_points", a_side0_terminal * a_points)
        self._add("a_as_side1_completed", a_side1_terminal)
        self._add("a_as_side1_wins", a_side1_terminal & a_won)
        self._add("a_as_side1_points", a_side1_terminal * a_points)
        self._add("b_as_side0_completed", a_side1_terminal)
        self._add("b_as_side0_wins", a_side1_terminal & b_won)
        self._add("b_as_side0_points", a_side1_terminal * b_points)
        self._add("b_as_side1_completed", a_side0_terminal)
        self._add("b_as_side1_wins", a_side0_terminal & b_won)
        self._add("b_as_side1_points", a_side0_terminal * b_points)

        keep = (~terminal).to(torch.float64)
        self._ongoing_steps.mul_(keep)
        self._ongoing_scored_hits.mul_(keep)
        self._ongoing_attributed_contacts.mul_(keep)

    @staticmethod
    def _rate(numerator: float, denominator: float) -> float | None:
        return None if denominator == 0.0 else numerator / denominator

    def snapshot(self, *, expected_arena_steps: int) -> dict[str, Any]:
        active = ~self._last_terminal
        tail = torch.stack(
            (
                active.to(torch.float64).sum(),
                (active * self._latest_a_points).sum(),
                (active * self._latest_b_points).sum(),
                self._ongoing_scored_hits.sum(),
                self._ongoing_attributed_contacts.sum(),
                self._ongoing_steps.sum(),
            )
        )
        packed = torch.cat((self._scalars, tail)).detach().to(device="cpu").tolist()
        values = dict(zip(self._SCALAR_NAMES, packed[: len(self._SCALAR_NAMES)]))
        (
            truncated,
            truncated_a_points,
            truncated_b_points,
            truncated_hits,
            truncated_contacts,
            truncated_steps,
        ) = packed[len(self._SCALAR_NAMES) :]
        completed = values["completed"]
        classified = (
            values["checkpoint_a_wins"]
            + values["checkpoint_b_wins"]
            + values["ties"]
        )
        accounted_steps = values["completed_semantic_steps"] + truncated_steps
        failures = []
        if values["terminal_pair_mismatches"] != 0.0:
            failures.append("terminal pairs disagreed")
        if values["action_mask_violations"] != 0.0:
            failures.append("a selected action was masked")
        if values["terminal_state_reset_max_abs"] != 0.0:
            failures.append("terminal recurrent state was not cleared")
        if values["unclassified_results"] != 0.0:
            failures.append("a terminal result was not classified")
        if accounted_steps != float(expected_arena_steps):
            failures.append(
                f"arena-step accounting was {accounted_steps}, expected {expected_arena_steps}"
            )
        if failures:
            raise RuntimeError("invalid checkpoint evaluation: " + "; ".join(failures))

        def role(completed_rounds: float, wins: float, points: float) -> dict[str, Any]:
            return {
                "completed": int(completed_rounds),
                "wins": int(wins),
                "win_rate": self._rate(wins, completed_rounds),
                "points": points,
                "points_per_round": self._rate(points, completed_rounds),
            }

        return {
            "episode_boundary": "round_terminal",
            "completed_environment_episodes": int(completed),
            "truncated_environment_episodes_at_cutoff": int(truncated),
            "attempted_environment_episodes": int(completed + truncated),
            "terminal_pair_mismatches": int(values["terminal_pair_mismatches"]),
            "action_mask_violations": int(values["action_mask_violations"]),
            "terminal_state_reset_max_abs": values["terminal_state_reset_max_abs"],
            "completed": {
                "checkpoint_a_wins": int(values["checkpoint_a_wins"]),
                "checkpoint_b_wins": int(values["checkpoint_b_wins"]),
                "ties": int(values["ties"]),
                "redo_results": int(values["redo_results"]),
                "unclassified_results": int(values["unclassified_results"]),
                "knockouts": int(values["knockouts"]),
                "checkpoint_a_round_win_rate": self._rate(
                    values["checkpoint_a_wins"], completed
                ),
                "checkpoint_b_round_win_rate": self._rate(
                    values["checkpoint_b_wins"], completed
                ),
                "checkpoint_a_score_rate_excluding_redo_and_unclassified": self._rate(
                    values["checkpoint_a_wins"] + 0.5 * values["ties"], classified
                ),
                "checkpoint_b_score_rate_excluding_redo_and_unclassified": self._rate(
                    values["checkpoint_b_wins"] + 0.5 * values["ties"], classified
                ),
                "checkpoint_a_points": values["checkpoint_a_points"],
                "checkpoint_b_points": values["checkpoint_b_points"],
                "checkpoint_a_points_per_round": self._rate(
                    values["checkpoint_a_points"], completed
                ),
                "checkpoint_b_points_per_round": self._rate(
                    values["checkpoint_b_points"], completed
                ),
                "checkpoint_a_falls": values["checkpoint_a_falls"],
                "checkpoint_b_falls": values["checkpoint_b_falls"],
                "checkpoint_a_falls_per_round": self._rate(
                    values["checkpoint_a_falls"], completed
                ),
                "checkpoint_b_falls_per_round": self._rate(
                    values["checkpoint_b_falls"], completed
                ),
                "scored_hits": values["completed_scored_hits"],
                "scored_hits_per_round": self._rate(
                    values["completed_scored_hits"], completed
                ),
                "attributed_contacts": values["completed_attributed_contacts"],
                "attributed_contacts_per_round": self._rate(
                    values["completed_attributed_contacts"], completed
                ),
                "semantic_steps": values["completed_semantic_steps"],
                "simulated_seconds": values["completed_semantic_steps"]
                * CONTROL_DELTA_SECONDS,
                "redo_rounds": int(values["completed_redo_rounds"]),
                "side0_wins": int(values["side0_wins"]),
                "side1_wins": int(values["side1_wins"]),
            },
            "role_balance": {
                "checkpoint_a_as_side0": role(
                    values["a_as_side0_completed"],
                    values["a_as_side0_wins"],
                    values["a_as_side0_points"],
                ),
                "checkpoint_a_as_side1": role(
                    values["a_as_side1_completed"],
                    values["a_as_side1_wins"],
                    values["a_as_side1_points"],
                ),
                "checkpoint_b_as_side0": role(
                    values["b_as_side0_completed"],
                    values["b_as_side0_wins"],
                    values["b_as_side0_points"],
                ),
                "checkpoint_b_as_side1": role(
                    values["b_as_side1_completed"],
                    values["b_as_side1_wins"],
                    values["b_as_side1_points"],
                ),
            },
            "truncated_at_cutoff_excluded_from_completed_statistics": {
                "episodes": int(truncated),
                "checkpoint_a_points": truncated_a_points,
                "checkpoint_b_points": truncated_b_points,
                "scored_hits": truncated_hits,
                "attributed_contacts": truncated_contacts,
                "semantic_steps": truncated_steps,
                "simulated_seconds": truncated_steps * CONTROL_DELTA_SECONDS,
            },
        }


class SwappedCheckpointEvaluator:
    """Run two fixed policies with complementary interleaved fighter rows."""

    def __init__(
        self,
        env: CudaTensorEnvAdapter,
        checkpoint_a: NativePufferPolicy,
        checkpoint_b: NativePufferPolicy,
    ) -> None:
        if env.total_agents < 4 or env.total_agents % 4:
            raise ValueError("evaluation requires an even number of paired arenas")
        if env.act_sizes != (33,):
            raise ValueError("REK G1 evaluation requires one 33-way action head")
        for policy in (checkpoint_a, checkpoint_b):
            if policy.device != env.device:
                raise ValueError("policies and environment must use one CUDA device")
            if policy.observation_size != env.obs_size or policy.act_sizes != env.act_sizes:
                raise ValueError("policy and environment layouts do not match")
        self.env = env
        self.checkpoint_a = checkpoint_a
        self.checkpoint_b = checkpoint_b
        self.arena_count = env.total_agents // 2
        side0 = torch.arange(0, env.total_agents, 2, device=env.device)
        side1 = side0 + 1
        first_half = torch.arange(self.arena_count, device=env.device) < (
            self.arena_count // 2
        )
        self.checkpoint_a_rows = torch.where(first_half, side0, side1)
        self.checkpoint_b_rows = torch.where(first_half, side1, side0)
        self.actions = torch.empty(
            env.total_agents, env.num_atns, dtype=torch.int64, device=env.device
        )
        self.metrics = RoleBalancedMatchAccumulator(env.total_agents, env.device)

    def run(
        self,
        steps: int,
        *,
        deterministic: bool = True,
        seed: int = 0,
    ) -> dict[str, Any]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.env.reset(reset_metrics=True)
        self.metrics.reset()
        state_a = self.checkpoint_a.initial_state(self.arena_count)
        state_b = self.checkpoint_b.initial_state(self.arena_count)
        generator = None
        if not deterministic:
            generator = torch.Generator(device=self.env.device)
            generator.manual_seed(int(seed))

        start = time.perf_counter()
        for _ in range(steps):
            with torch.no_grad():
                observations_a = self.env.observations.index_select(
                    0, self.checkpoint_a_rows
                )
                observations_b = self.env.observations.index_select(
                    0, self.checkpoint_b_rows
                )
                masks_a = self.env.action_mask.index_select(0, self.checkpoint_a_rows)
                masks_b = self.env.action_mask.index_select(0, self.checkpoint_b_rows)
                logits_a, _, state_a = self.checkpoint_a.forward_eval(
                    observations_a, state_a
                )
                logits_b, _, state_b = self.checkpoint_b.forward_eval(
                    observations_b, state_b
                )
                actions_a = _masked_actions(
                    logits_a,
                    self.env.act_sizes,
                    masks_a,
                    deterministic=deterministic,
                    generator=generator,
                )
                actions_b = _masked_actions(
                    logits_b,
                    self.env.act_sizes,
                    masks_b,
                    deterministic=deterministic,
                    generator=generator,
                )
                self.actions.index_copy_(0, self.checkpoint_a_rows, actions_a)
                self.actions.index_copy_(0, self.checkpoint_b_rows, actions_b)
                self.metrics.record_action_validity(self.env.action_mask, self.actions)

            self.env.step(self.actions)
            self.metrics.update(self.env.observations, self.env.terminals)
            with torch.no_grad():
                terminal_a = self.env.terminals.index_select(
                    0, self.checkpoint_a_rows
                ) != 0
                terminal_b = self.env.terminals.index_select(
                    0, self.checkpoint_b_rows
                ) != 0
                state_a = (
                    state_a[0].masked_fill(terminal_a.view(1, -1, 1), 0.0),
                )
                state_b = (
                    state_b[0].masked_fill(terminal_b.view(1, -1, 1), 0.0),
                )
                self.metrics.record_terminal_state_residual(
                    state_a[0], terminal_a, state_b[0], terminal_b
                )

        torch.cuda.synchronize(self.env.device)
        elapsed = time.perf_counter() - start
        return {
            "semantic_steps": steps,
            "agent_steps": steps * self.env.total_agents,
            "wall_seconds": elapsed,
            "agent_steps_per_second": steps * self.env.total_agents / elapsed,
            "deterministic": deterministic,
            "seed": int(seed),
            "metrics": self.metrics.snapshot(
                expected_arena_steps=steps * self.arena_count
            ),
        }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    checkpoint_a_record = _pinned_checkpoint(
        args.checkpoint_a, args.checkpoint_a_sha256
    )
    checkpoint_b_record = _pinned_checkpoint(
        args.checkpoint_b, args.checkpoint_b_sha256
    )
    hidden_size, num_layers = _policy_shape(
        args.default_config, args.native_config
    )
    duel_config = load_gpu_duel_config(args.gpu_duel_config)
    environment = GpuSemanticDuel(duel_config)
    adapter: CudaTensorEnvAdapter | None = None
    try:
        if environment.rows < 4 or environment.rows % 4:
            raise ValueError(
                "GPU duel must contain an even number of arenas for role swapping"
            )
        environment.capture_step()
        adapter = CudaTensorEnvAdapter(environment, (33,))
        policy_a = NativePufferPolicy(
            adapter.obs_size,
            adapter.act_sizes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=adapter.device,
        )
        policy_b = NativePufferPolicy(
            adapter.obs_size,
            adapter.act_sizes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=adapter.device,
        )
        loaded_a = policy_a.load_native_checkpoint(checkpoint_a_record["path"])
        loaded_b = policy_b.load_native_checkpoint(checkpoint_b_record["path"])
        if loaded_a != checkpoint_a_record["sha256"] or loaded_b != checkpoint_b_record["sha256"]:
            raise RuntimeError("loaded checkpoint digest changed unexpectedly")
        if not torch.isfinite(policy_a.flat_native()).all().item():
            raise ValueError("checkpoint A contains nonfinite policy weights")
        if not torch.isfinite(policy_b.flat_native()).all().item():
            raise ValueError("checkpoint B contains nonfinite policy weights")

        evaluator = SwappedCheckpointEvaluator(adapter, policy_a, policy_b)
        result = evaluator.run(
            args.steps,
            deterministic=not args.stochastic,
            seed=args.seed,
        )
        environment.check_status()
        post_a = _sha256(Path(checkpoint_a_record["path"]))
        post_b = _sha256(Path(checkpoint_b_record["path"]))
        if post_a != loaded_a or post_b != loaded_b:
            raise RuntimeError("a pinned checkpoint changed during evaluation")

        report = {
            "schema": "rek.g1_gpu_checkpoint_match.v1",
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(adapter.device),
            "cuda_device": str(adapter.device),
            "checkpoints": {
                "a": {"label": args.label_a, **checkpoint_a_record},
                "b": {"label": args.label_b, **checkpoint_b_record},
                "same_checkpoint": loaded_a == loaded_b,
                "native_flat_layout": True,
                "policy_implementation": str(
                    Path(__file__).with_name("gpu_puffer_policy.py").resolve()
                ),
                "policy_implementation_sha256": _sha256(
                    Path(__file__).with_name("gpu_puffer_policy.py")
                ),
                "hidden_size": hidden_size,
                "num_layers": num_layers,
            },
            "environment": {
                "gpu_duel_config": str(args.gpu_duel_config.resolve()),
                "gpu_duel_config_sha256": _sha256(args.gpu_duel_config),
                "fighter_rows": adapter.total_agents,
                "arenas": adapter.total_agents // 2,
                "physics": "mujoco-warp-cuda",
                "controller": "torch-cuda",
                "cpu_physics_steps": 0,
                "cpu_controller_inferences": 0,
                "checkpoint_a_side0_arenas": adapter.total_agents // 4,
                "checkpoint_a_side1_arenas": adapter.total_agents // 4,
                "interleaved_fighter_rows_routed_explicitly": True,
            },
            "evaluation": result,
            "claim_limits": {
                "fixed_checkpoint_comparison_only": True,
                "human_baseline_measured": False,
                "authentic_rek_parity_established": False,
                "superhuman_claim_supported": False,
                "truncated_episodes_excluded_from_completed_outcomes": True,
                "scored_hits_are_arena_totals_not_attributed_per_checkpoint": True,
                "pooled_rates_are_completion_duration_weighted": True,
                "deterministic_cloned_arenas_are_not_independent_trials": (
                    not args.stochastic
                ),
            },
        }
    finally:
        if adapter is not None:
            adapter.close()
        else:
            environment.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-duel-config", required=True, type=Path)
    parser.add_argument(
        "--default-config", type=Path, default=repo_root / "config" / "default.ini"
    )
    parser.add_argument(
        "--native-config", type=Path, default=repo_root / "config" / "rek_g1.ini"
    )
    parser.add_argument("--checkpoint-a", required=True, type=Path)
    parser.add_argument("--checkpoint-a-sha256", required=True)
    parser.add_argument("--label-a", default="checkpoint-a")
    parser.add_argument("--checkpoint-b", required=True, type=Path)
    parser.add_argument("--checkpoint-b-sha256", required=True)
    parser.add_argument("--label-b", default="checkpoint-b")
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
