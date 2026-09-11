"""Opt-in training reward whose complete-round return targets win probability.

Require native gamma=1 and unclipped rewards. The base reward is one only for a
published terminal points/KO win. Dense shaping is Phi(next)-Phi(current), with
Phi=scale*tanh((own_points-opponent_points)/margin_points) and both terminal
potentials zero. Complete returns equal terminal_win-Phi(initial), irrespective
of episode duration; a zero-score initial round therefore returns exactly its
win indicator in real arithmetic. Finite PPO and bootstrapping remain approximate.

This module does not modify game scores, observations, masks, opponent or eval.
It deliberately omits facing shaping so this objective can be tested alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch

SIDE, OWN_POINTS, OPPONENT_POINTS, ROUND_RESULT, ROUND_WINNER = 184, 190, 191, 210, 211


@dataclass(frozen=True)
class RoundWinRewardConfig:
    gamma: float
    margin_potential_scale: float
    margin_points: float

    def __post_init__(self):
        if self.gamma != 1.0:
            raise ValueError("win-probability reward requires native gamma exactly 1")
        if not math.isfinite(self.margin_potential_scale) or self.margin_potential_scale < 0:
            raise ValueError("margin potential scale must be finite and nonnegative")
        if not math.isfinite(self.margin_points) or self.margin_points <= 0:
            raise ValueError("margin_points must be finite and positive")

    def validate_training_discount(self, gamma, *, reward_clip):
        if gamma != self.gamma or reward_clip != 0:
            raise ValueError("native gamma 1 and zero reward clipping are required")

    def metadata(self):
        return {
            "name": "round_win_probability_bounded_margin_v1", "training_only": True,
            "default_activation": False, "gamma": self.gamma,
            "margin_potential_scale": self.margin_potential_scale, "margin_points": self.margin_points,
            "base_reward": "1 iff next_terminal and result in {1,2} and winner==published_side; else 0",
            "potential": "margin_potential_scale*tanh((own_points-opponent_points)/margin_points)",
            "formula": "terminal_win + gamma*next_potential - current_potential",
            "terminal_potential": 0, "both_current_and_next_terminal_masks_required": True,
            "complete_undiscounted_return": "terminal_win - initial_potential",
            "initial_zero_score_return": "terminal_win",
            "tie_redo_loss_reward": 0, "horizon_is_terminal": False,
            "bootstrap_value": "V_shaped(s)=P(eventual_round_win|s)-Phi(s)",
            "source_columns": {"published_side": SIDE, "own_points": OWN_POINTS,
                               "opponent_points": OPPONENT_POINTS, "round_result": ROUND_RESULT,
                               "round_winner": ROUND_WINNER},
            "calculation": "float64 potential/combined reward, float32 learner reward storage",
            "raw_score_delta_reward_used": False, "game_rules_or_scores_changed": False,
            "evaluation_changed": False, "facing_potential_included": False,
            "scale_and_margin_are_training_choices_not_game_rules": True,
        }


def resolve_round_win_reward(native_args, *, margin_potential_scale, margin_points, reward_clip):
    configured = float(native_args["train"]["gamma"])
    if not math.isfinite(configured) or not 0 <= configured <= 1:
        raise ValueError("configured native gamma must be finite and in [0, 1]")
    native_gamma = struct.unpack("f", struct.pack("f", configured))[0]
    config = RoundWinRewardConfig(native_gamma, float(margin_potential_scale), float(margin_points))
    config.validate_training_discount(native_gamma, reward_clip=reward_clip)
    return config, {**config.metadata(), "gamma_configured": configured, "native_reward_clip": reward_clip,
                    "gamma_source": "native_args.train.gamma converted to HypersT float32"}


def resolve_reward_objective(native_args, *, objective="score-delta", facing_scale=0.0,
                             margin_potential_scale=0.5, margin_points=5.0, reward_clip=0.0):
    """Return mutually exclusive facing/win configurations and provenance."""
    if objective == "score-delta":
        from gpu_facing_potential import resolve_training_reward
        facing, descriptor = resolve_training_reward(native_args, scale=facing_scale, reward_clip=reward_clip)
        return facing, None, descriptor
    if objective != "round-win":
        raise ValueError("unknown reward objective")
    if facing_scale != 0:
        raise ValueError("round-win reward and nonzero facing potential are mutually exclusive")
    config, descriptor = resolve_round_win_reward(native_args,
        margin_potential_scale=margin_potential_scale, margin_points=margin_points, reward_clip=reward_clip)
    return None, config, descriptor


class GpuRoundWinReward:
    """Same begin/finish transition protocol as GpuFacingPotential; CUDA owned buffers."""

    def __init__(self, rows, device, config, *, allow_cpu_for_tests=False):
        if rows < 1 or not isinstance(config, RoundWinRewardConfig):
            raise ValueError("positive rows and an explicit round-win configuration are required")
        requested = torch.device(device)
        if requested.type != "cuda" and not (allow_cpu_for_tests and requested.type == "cpu"):
            raise ValueError("round-win reward requires CUDA; CPU is test-only")
        self.rows, self.config = rows, config
        self.rewards = torch.zeros(rows, dtype=torch.float32, device=requested)
        self.device = self.rewards.device
        self.current_potential = torch.zeros(rows, dtype=torch.float64, device=self.device)
        self.next_potential = torch.zeros_like(self.current_potential)
        self.shaping_reward = torch.zeros_like(self.current_potential)
        self.terminal_win = torch.zeros_like(self.current_potential)
        self.status = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self.pending = torch.zeros(rows, dtype=torch.bool, device=self.device)

    def reset(self):
        for tensor in (self.rewards, self.current_potential, self.next_potential,
                       self.shaping_reward, self.terminal_win, self.status, self.pending):
            tensor.zero_()

    def _require(self, value, name, observations=False):
        shape = (self.rows, 223) if observations else (self.rows,)
        if value.shape != shape or value.dtype != torch.float32 or value.device != self.device:
            raise ValueError(f"{name} must be float32 {shape} on the reward device")

    def _potential(self, observations, terminals):
        self._require(observations, "raw observations", True)
        self._require(terminals, "terminals")
        own, opponent = observations[:, OWN_POINTS].double(), observations[:, OPPONENT_POINTS].double()
        side, result, winner = observations[:, SIDE], observations[:, ROUND_RESULT], observations[:, ROUND_WINNER]
        decisive, tied = (result == 1) | (result == 2), (result == 3) | (result == 4)
        terminal = terminals != 0
        valid_outcome = (decisive & ((winner == 0) | (winner == 1))) | (tied & (winner == -1))
        potential = self.config.margin_potential_scale * torch.tanh((own - opponent) / self.config.margin_points)
        self.status.bitwise_or_(
            (~torch.isfinite(own) | ~torch.isfinite(opponent)).to(torch.int32)
            | (((own < 0) | (opponent < 0) | (own != own.floor()) | (opponent != opponent.floor())).to(torch.int32) * 2)
            | ((~((terminals == 0) | (terminals == 1))).to(torch.int32) * 4)
            | ((~((side == 0) | (side == 1))).to(torch.int32) * 8)
            | ((terminal & ~valid_outcome).to(torch.int32) * 16)
        )
        return torch.where(terminal, 0.0, potential)

    def begin_transition(self, raw_current, current_terminals):
        self.status.bitwise_or_(self.pending.to(torch.int32) * 32)
        self.current_potential.copy_(self._potential(raw_current, current_terminals))
        self.pending.fill_(True)

    def finish_transition(self, raw_next, raw_reward, next_terminals):
        self._require(raw_reward, "raw rewards")
        if raw_reward.data_ptr() == self.rewards.data_ptr():
            raise ValueError("round-win output cannot be supplied as raw game reward")
        self.status.bitwise_or_((~self.pending).to(torch.int32) * 64)
        self.next_potential.copy_(self._potential(raw_next, next_terminals))
        decisive = (raw_next[:, ROUND_RESULT] == 1) | (raw_next[:, ROUND_RESULT] == 2)
        self.terminal_win.copy_((next_terminals != 0) & decisive & (raw_next[:, ROUND_WINNER] == raw_next[:, SIDE]))
        self.shaping_reward.copy_(self.config.gamma * self.next_potential - self.current_potential)
        self.rewards.copy_(self.terminal_win + self.shaping_reward)
        self.status.bitwise_or_((~torch.isfinite(raw_reward) | ~torch.isfinite(self.rewards)).to(torch.int32) * 128)
        self.pending.zero_()
        return self.rewards

    def check_status(self):
        values = self.status.detach().cpu().tolist()
        if any(values):
            raise RuntimeError(f"invalid round-win reward transition: {values}")
