"""Device-resident learner behavior measurements for the G1 candidate.

The update path uses only tensor operations. A CPU device is supported for
cross-backend regression fixtures; production CUDA wrappers keep every input
and accumulator on their CUDA device. Only snapshot downloads the report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class GpuBehaviorMetricCollector:
    """Measure one selected learner per arena, with explicit denominators."""

    def __init__(self, total_fighters: int, device: str | torch.device, *,
                 learner_rows: Sequence[int], facing_degrees: float = 30.0,
                 range_edges_m: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)):
        rows = tuple(int(row) for row in learner_rows)
        if total_fighters < 2 or total_fighters % 2:
            raise ValueError("total_fighters must contain complete pairs")
        if len(rows) != total_fighters // 2 or set(row // 2 for row in rows) != set(range(total_fighters // 2)):
            raise ValueError("select exactly one learner per arena")
        if any(row < 0 or row >= total_fighters for row in rows):
            raise ValueError("learner row out of range")
        if not 0 < facing_degrees < 180:
            raise ValueError("facing_degrees must be in (0, 180)")
        edges = tuple(float(value) for value in range_edges_m)
        if not edges or any(not math.isfinite(value) or value <= 0 for value in edges) or tuple(sorted(set(edges))) != edges:
            raise ValueError("range edges must be finite, positive and increasing")
        self.total_fighters = total_fighters
        self.learner_rows = rows
        self.facing_degrees = float(facing_degrees)
        self.range_edges_m = edges
        self.rows = torch.tensor(rows, dtype=torch.int64, device=device)
        self.device = self.rows.device
        self.sides = self.rows.remainder(2)
        self.edges = torch.tensor(edges, device=self.device)
        self.categories = torch.arange(33, device=self.device)
        self.histogram_bins = torch.arange(len(edges) + 1, device=self.device)
        self.cos_threshold = math.cos(math.radians(facing_degrees))
        self.counts = torch.zeros(19, dtype=torch.float64, device=self.device)
        self.action_counts = torch.zeros(33, dtype=torch.float64, device=self.device)
        self.range_counts = torch.zeros(len(edges) + 1, dtype=torch.float64, device=self.device)
        self._move_starts_available = True

    def reset(self):
        self.counts.zero_()
        self.action_counts.zero_()
        self.range_counts.zero_()
        self._move_starts_available = True

    def update(self, observations, terminals, actions, move_start=None, score_delta=None):
        """Sample post-step geometry and native events without a host read.

        Coordinates are pelvis/root separation in the horizontal XY plane.
        Facing uses the root's projected local +X direction, not desired yaw.
        Native points include referee awards and therefore are not hit counts.
        """
        if observations.shape != (self.total_fighters, 223):
            raise ValueError("observations must be full fighter rows with 223 fields")
        if terminals.shape != (self.total_fighters,):
            raise ValueError("terminals must contain every fighter")
        if actions.shape not in ((self.total_fighters,), (self.total_fighters, 1)):
            raise ValueError("actions must contain one category per fighter")
        for value in (observations, terminals, actions, move_start, score_delta):
            if value is not None and value.device != self.device:
                raise ValueError("all metric inputs must share the collector device")
        if move_start is not None and move_start.shape != (self.total_fighters,):
            raise ValueError("move_start must contain every fighter")
        if score_delta is not None and score_delta.shape != (self.total_fighters,):
            raise ValueError("score_delta must contain every fighter")
        obs = observations.index_select(0, self.rows)
        terminal = terminals.index_select(0, self.rows) != 0
        action = actions.reshape(-1).index_select(0, self.rows)
        category_valid = torch.isfinite(action) & (action == action.to(torch.int64)) & (action >= 0) & (action < 33)
        self.action_counts.add_(((action[:, None] == self.categories) & category_valid[:, None]).sum(dim=0))
        active = obs[:, 185] == 2
        delta = obs[:, 86:88] - obs[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=1)
        range_valid = active & torch.isfinite(delta).all(dim=1) & torch.isfinite(distance)
        safe_range = torch.where(range_valid, distance, 0.0)
        bins = (safe_range[:, None] >= self.edges).sum(dim=1)
        self.range_counts.add_(((bins[:, None] == self.histogram_bins) & range_valid[:, None]).sum(dim=0))
        w, x, y, z = obs[:, 3:7].unbind(dim=1)
        forward = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), dim=1)
        forward_norm = torch.linalg.vector_norm(forward, dim=1)
        quaternion_norm = obs[:, 3:7].square().sum(dim=1)
        facing_valid = range_valid & (distance > 1e-8) & torch.isfinite(forward).all(dim=1) & (forward_norm > 1e-8) & ((quaternion_norm - 1).abs() < 1e-3)
        cosine = (forward * delta).sum(dim=1) / (forward_norm * distance).clamp_min(1e-8)
        facing = facing_valid & (cosine >= self.cos_threshold)
        attack = category_valid & (action >= 16)
        decisive = (obs[:, 210] == 1) | (obs[:, 210] == 2)
        won = terminal & decisive & (obs[:, 211] == self.sides)
        lost = terminal & decisive & (obs[:, 211] != self.sides)
        if move_start is None:
            self._move_starts_available = False
            starts = torch.zeros((), device=self.device)
        else:
            starts = (move_start.index_select(0, self.rows) != 0).sum()
        points_delta = obs[:, 217] if score_delta is None else score_delta.index_select(0, self.rows)
        values = (
            torch.ones_like(action).sum(), (~category_valid).sum(), starts,
            active.sum(), range_valid.sum(), safe_range.sum(), facing_valid.sum(), facing.sum(),
            (facing_valid & attack).sum(), (facing & attack).sum(), terminal.sum(), won.sum(), lost.sum(),
            (terminal & (obs[:, 210] == 3)).sum(),
            torch.where(terminal, obs[:, 190], 0).sum(),
            torch.where(terminal, obs[:, 191], 0).sum(), points_delta.sum(),
            obs[:, 221].sum(), obs[:, 222].sum(),
        )
        self.counts.add_(torch.stack(values).to(torch.float64))

    def snapshot(self, *, clear: bool = True) -> dict:
        values = torch.cat((self.counts, self.action_counts, self.range_counts)).detach().cpu().tolist()
        c, action, ranges = values[:19], values[19:52], values[52:]
        percent = lambda numerator, denominator: None if not denominator else 100 * numerator / denominator
        per_round = lambda numerator: None if not c[10] else numerator / c[10]
        result = {
            "schema": "rek.g1_learner_behavior.v1",
            "learner_rows": list(self.learner_rows),
            "learner_control_steps": int(c[0]),
            "invalid_action_categories": int(c[1]),
            "action_category_counts": [int(value) for value in action],
            "discrete_move_starts": int(c[2]) if self._move_starts_available else None,
            "completed_rounds": int(c[10]), "learner_round_wins": int(c[11]),
            "learner_round_losses": int(c[12]), "round_ties": int(c[13]),
            "learner_round_win_percent": percent(c[11], c[10]),
            "learner_points_per_completed_round": per_round(c[14]),
            "opponent_points_per_completed_round": per_round(c[15]),
            "learner_native_points_awarded_all_steps": c[16],
            "arena_attributed_contacts_all_steps": int(c[17]),
            "arena_scored_hits_all_steps": int(c[18]),
            "learner_scored_hits": None,
            "facing": {
                "threshold_degrees": self.facing_degrees,
                "definition": "projected root local +X within threshold of opponent root XY bearing",
                "sampling": "post-control-step, active round only",
                "eligible_samples": int(c[6]), "facing_samples": int(c[7]),
                "percent": percent(c[7], c[6]),
                "attack_requested_eligible_samples": int(c[8]),
                "attack_requested_facing_percent": percent(c[9], c[8]),
            },
            "range": {
                "definition": "horizontal root-to-root separation in metres, active round only",
                "samples": int(c[4]), "mean_m": None if not c[4] else c[5] / c[4],
                "edges_m": list(self.range_edges_m),
                "intervals": "[0, edge0), [edge0, edge1), ..., [last_edge, infinity)",
                "counts": [int(value) for value in ranges],
                "percent": [percent(value, c[4]) for value in ranges],
            },
            "limits": [
                "Points include referee awards and are not scored-hit counts.",
                "Scored-hit and contact metadata are arena totals; learner attribution is unavailable.",
                "Attributed contacts qualify for knockdown attribution; scoring is independent and scored hits may exceed attributed contacts.",
                "Discrete starts are sampled native move_start_edge flags; input requests are counted separately.",
                "Completed rounds are not full best-of-three fights or human-performance comparisons.",
            ],
        }
        if clear:
            self.reset()
        return result
