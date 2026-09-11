"""Deferred GPU accounting for completed REK G1 arena rounds.

`update` performs tensor operations on the source device and does not call
``cpu()``, ``item()``, or otherwise synchronize it. `snapshot` is the explicit
reporting boundary: it transfers one small stacked tensor to the host. Clearing
a snapshot never clears an unfinished arena round.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


OBSERVATION_FLOATS = 223
FIGHT_OFFSET = 184
CURRENT_ROUND_IS_REDO = FIGHT_OFFSET + 3
SIDE0_POINTS = FIGHT_OFFSET + 6
SIDE1_POINTS = FIGHT_OFFSET + 7
SIDE0_FALLS = FIGHT_OFFSET + 8
SIDE1_FALLS = FIGHT_OFFSET + 9
ROUND_RESULT = FIGHT_OFFSET + 26
ROUND_WINNER = FIGHT_OFFSET + 27
KNOCKOUT_OCCURRED = FIGHT_OFFSET + 28
TICK_ATTRIBUTED_CONTACTS = FIGHT_OFFSET + 37
TICK_SCORED_HITS = FIGHT_OFFSET + 38

ROUND_WON_BY_POINTS = 1
ROUND_WON_BY_KO = 2
ROUND_TIE = 3
ROUND_REDO = 4

CONTROL_DELTA_SECONDS = 0.02

METRIC_KEYS = (
    "side0_round_win_rate",
    "side1_round_win_rate",
    "round_tie_rate",
    "round_redo_result_rate",
    "redo_round_rate",
    "ko_round_rate",
    "side0_points_per_round",
    "side1_points_per_round",
    "side0_falls_per_round",
    "side1_falls_per_round",
    "scored_hits_per_round",
    "attributed_contacts_per_round",
    "elapsed_seconds_per_round",
    "semantic_steps_per_round",
)


class RekG1GpuMetricCollector:
    """Accumulate arena metrics without synchronizing the execution device."""

    def __init__(
        self,
        total_agents: int,
        device: torch.device | str,
        *,
        control_delta_seconds: float = CONTROL_DELTA_SECONDS,
    ) -> None:
        if total_agents < 2 or total_agents % 2:
            raise ValueError("total_agents must contain complete fighter pairs")
        if not 0.0 < control_delta_seconds < float("inf"):
            raise ValueError("control_delta_seconds must be finite and positive")
        self.total_agents = total_agents
        self.arena_count = total_agents // 2
        requested_device = torch.device(device)
        self.control_delta_seconds = float(control_delta_seconds)
        self._ongoing_steps = torch.zeros(
            self.arena_count, dtype=torch.float64, device=requested_device)
        self.device = self._ongoing_steps.device
        self._ongoing_scored_hits = torch.zeros_like(self._ongoing_steps)
        self._ongoing_attributed_contacts = torch.zeros_like(
            self._ongoing_steps)
        self._ongoing_elapsed_seconds = torch.zeros_like(self._ongoing_steps)
        self._completed = torch.zeros(
            len(METRIC_KEYS), dtype=torch.float64, device=self.device)
        self._completed_rounds = torch.zeros(
            (), dtype=torch.float64, device=self.device)
        self._terminal_pair_mismatches = torch.zeros(
            (), dtype=torch.float64, device=self.device)

    def reset(self) -> None:
        """Clear both unfinished rounds and completed reporting totals."""
        self._ongoing_steps.zero_()
        self._ongoing_scored_hits.zero_()
        self._ongoing_attributed_contacts.zero_()
        self._ongoing_elapsed_seconds.zero_()
        self._completed.zero_()
        self._completed_rounds.zero_()
        self._terminal_pair_mismatches.zero_()

    def update(
        self,
        observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> None:
        """Record one successful semantic step for every arena.

        Final points and falls are sampled only where the canonical even row is
        terminal. Tick scored-hit and attribution counts are arena totals in
        that same row and are accumulated throughout the unfinished round.
        """
        if observations.ndim != 2 or observations.shape != (
            self.total_agents,
            OBSERVATION_FLOATS,
        ):
            raise ValueError(
                f"observations must have shape "
                f"({self.total_agents}, {OBSERVATION_FLOATS})"
            )
        if terminals.ndim != 1 or terminals.shape[0] != self.total_agents:
            raise ValueError(
                f"terminals must have shape ({self.total_agents},)"
            )
        if observations.device != self.device or terminals.device != self.device:
            raise ValueError("observations and terminals must use collector device")
        if not observations.dtype.is_floating_point:
            raise TypeError("observations must be floating point")

        even = observations[0::2]
        even_terminal = terminals[0::2] != 0
        odd_terminal = terminals[1::2] != 0
        terminal = even_terminal.to(dtype=torch.float64)
        self._terminal_pair_mismatches.add_(
            torch.count_nonzero(even_terminal != odd_terminal).to(torch.float64)
        )

        self._ongoing_steps.add_(1.0)
        self._ongoing_elapsed_seconds.add_(self.control_delta_seconds)
        self._ongoing_scored_hits.add_(
            even[:, TICK_SCORED_HITS].to(dtype=torch.float64)
        )
        self._ongoing_attributed_contacts.add_(
            even[:, TICK_ATTRIBUTED_CONTACTS].to(dtype=torch.float64)
        )

        result = even[:, ROUND_RESULT]
        winner = even[:, ROUND_WINNER]
        decisive = (result == ROUND_WON_BY_POINTS) | (result == ROUND_WON_BY_KO)
        completed_values = torch.stack(
            (
                terminal * decisive * (winner == 0),
                terminal * decisive * (winner == 1),
                terminal * (result == ROUND_TIE),
                terminal * (result == ROUND_REDO),
                terminal * (even[:, CURRENT_ROUND_IS_REDO] != 0),
                terminal * (even[:, KNOCKOUT_OCCURRED] != 0),
                terminal * even[:, SIDE0_POINTS],
                terminal * even[:, SIDE1_POINTS],
                terminal * even[:, SIDE0_FALLS],
                terminal * even[:, SIDE1_FALLS],
                terminal * self._ongoing_scored_hits,
                terminal * self._ongoing_attributed_contacts,
                terminal * self._ongoing_elapsed_seconds,
                terminal * self._ongoing_steps,
            ),
            dim=0,
        ).to(dtype=torch.float64)
        self._completed.add_(completed_values.sum(dim=1))
        self._completed_rounds.add_(terminal.sum())

        keep = (~even_terminal).to(dtype=torch.float64)
        self._ongoing_steps.mul_(keep)
        self._ongoing_scored_hits.mul_(keep)
        self._ongoing_attributed_contacts.mul_(keep)
        self._ongoing_elapsed_seconds.mul_(keep)

    def snapshot(self, *, clear: bool = True) -> dict[str, float]:
        """Synchronize once, return completed-round averages, and optionally clear.

        ``clear=True`` clears completed reporting totals and mismatch diagnostics.
        It deliberately preserves every unfinished arena accumulator.
        """
        packed = torch.cat(
            (
                self._completed,
                self._completed_rounds.reshape(1),
                self._terminal_pair_mismatches.reshape(1),
            )
        )
        host_values = packed.detach().to(device="cpu").tolist()
        completed_rounds = host_values[len(METRIC_KEYS)]
        result = {
            "n": completed_rounds,
            "terminal_pair_mismatches": host_values[-1],
        }
        if completed_rounds > 0.0:
            result.update(
                {
                    key: host_values[index] / completed_rounds
                    for index, key in enumerate(METRIC_KEYS)
                }
            )
        if clear:
            self._completed.zero_()
            self._completed_rounds.zero_()
            self._terminal_pair_mismatches.zero_()
        return result

    def ongoing_state(self) -> Mapping[str, torch.Tensor]:
        """Return device tensors for diagnostics without copying or synchronizing."""
        return {
            "semantic_steps": self._ongoing_steps,
            "scored_hits": self._ongoing_scored_hits,
            "attributed_contacts": self._ongoing_attributed_contacts,
            "elapsed_seconds": self._ongoing_elapsed_seconds,
        }
