"""CUDA version of the human evaluator's CandidateApproachDummy.

The opponent is the deterministic candidate script, not authentic REK Bot 1.
Only even learner rows are exported to Puffer. Physics and combat still run
for both fighters in every arena. No opponent samples enter PPO.
"""

from __future__ import annotations

import torch


DUMMY_LABEL = "deterministic_state_based_approach_facing_16_combat_move_candidate_dummy"


class GpuCandidateApproachDummy:
    """Batched exact decision rules of the native-mask evaluator path.

Geometry uses float64, as the original Python script does. Invalid input is
latched on device and reported by check_status at reporting boundaries.
"""

    def __init__(self, arenas: int, device: str | torch.device):
        if arenas < 1:
            raise ValueError("at least one arena is required")
        self.next_move_offset = torch.zeros(arenas, dtype=torch.int64, device=device)
        self.actions = torch.ones(arenas, dtype=torch.int32, device=device)
        self.preferred = torch.ones_like(self.actions)
        self.status = torch.zeros(arenas, dtype=torch.int32, device=device)

    def reset(self, terminals: torch.Tensor | None = None) -> None:
        if terminals is None:
            self.next_move_offset.zero_()
            self.actions.fill_(1)
            self.preferred.fill_(1)
            self.status.zero_()
        else:
            self.next_move_offset.masked_fill_(terminals != 0, 0)

    def select(self, observations: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        rows = self.next_move_offset.numel()
        if observations.shape != (rows, 223) or masks.shape != (rows, 33):
            raise ValueError("dummy requires paired opponent observations and native masks")
        if observations.device != self.actions.device or masks.device != self.actions.device:
            raise ValueError("dummy inputs must remain on its execution device")
        q = observations[:, 3:7].to(torch.float64)
        w, x, y, z = q.unbind(1)
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        yaw = torch.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
        delta = observations[:, 86:88].to(torch.float64) - observations[:, :2].to(torch.float64)
        dx, dy = delta.unbind(1)
        distance = torch.sqrt(dx*dx + dy*dy)
        forward = torch.cos(yaw)*dx + torch.sin(yaw)*dy
        lateral = -torch.sin(yaw)*dx + torch.cos(yaw)*dy
        bearing = torch.atan2(lateral, forward)
        preferred = self.next_move_offset + 16
        preferred = torch.where(distance < 0.72, 3, preferred)
        preferred = torch.where(distance > 1.25, 2, preferred)
        preferred = torch.where(torch.abs(bearing) > 0.16,
                                torch.where(bearing > 0.0, 6, 7), preferred)
        upright = observations[:, 79] == 0.0
        preferred = torch.where(upright, preferred, 1)
        self.preferred.copy_(preferred)

        # ConservativeActionPlanner.select(preferred, 1, row, native_mask).
        allowed = masks.gather(1, preferred[:, None]).squeeze(1) != 0
        selected = torch.where(allowed, preferred,
                               torch.where(masks[:, 1] != 0, 1, 0))
        conservative = allowed | (masks[:, 1] != 0) | (masks[:, 0] != 0)
        self.actions.copy_(selected)
        self.next_move_offset.copy_(torch.where(
            (selected >= 16) & (selected < 32),
            (selected - 15).remainder(16), self.next_move_offset,
        ))
        flags = observations[:, 180:184]
        self.status.bitwise_or_(
            (~torch.isfinite(observations).all(1)).to(torch.int32)
            | ((upright & (~torch.isfinite(norm) | (norm <= 1e-9))).to(torch.int32) * 2)
            | ((~((masks == 0) | (masks == 1)).all(1)).to(torch.int32) * 4)
            | ((~conservative).to(torch.int32) * 8)
            | ((~((flags == 0) | (flags == 1)).all(1)).to(torch.int32) * 16)
        )
        return self.actions

    def check_status(self) -> None:
        statuses = self.status.detach().cpu().tolist()
        if any(statuses):
            raise RuntimeError(f"candidate dummy received invalid observations/masks: {statuses}")


class _FullDuelMetricPlugin:
    """Adapter hook for metrics already updated inside the wrapper's graph."""

    def __init__(self, owner):
        self.owner = owner
        self.device = owner.observations.device

    def reset(self):
        self.owner.combat_metrics.reset()
        self.owner.behavior_metrics.reset()

    def update(self, observations, terminals):
        pass

    def snapshot(self, *, clear=True):
        result = self.owner.combat_metrics.snapshot(clear=clear)
        result["behavior"] = self.owner.behavior_metrics.snapshot(clear=clear)
        return result


class GpuCandidateDummyDuel:
    """Stable contiguous learner-only boundary around complete GPU duels."""

    def __init__(self, duel, *, capture=True):
        from gpu_behavior_metrics import GpuBehaviorMetricCollector
        from gpu_metrics import RekG1GpuMetricCollector

        if duel.rows < 2 or duel.rows % 2:
            raise ValueError("full duel must contain interleaved fighter pairs")
        self.duel = duel
        self.rows = duel.rows // 2
        device = duel.observations.device
        if device.type != "cuda":
            raise ValueError("candidate dummy training requires CUDA physics")
        self.dummy = GpuCandidateApproachDummy(self.rows, device)
        self.learner_actions = torch.ones((self.rows, 1), dtype=torch.int32, device=device)
        self.full_actions = torch.ones((duel.rows, 1), dtype=torch.int32, device=device)
        self.observations = torch.empty((self.rows, 223), dtype=torch.float32, device=device)
        self.rewards = torch.empty(self.rows, dtype=torch.float32, device=device)
        self.terminals = torch.empty_like(self.rewards)
        self.action_mask = torch.empty((self.rows, 33), dtype=torch.uint8, device=device)
        self.combat_metrics = RekG1GpuMetricCollector(duel.rows, device)
        self.behavior_metrics = GpuBehaviorMetricCollector(
            duel.rows, device, learner_rows=tuple(range(0, duel.rows, 2)),
        )
        self.metric_plugin = _FullDuelMetricPlugin(self)
        self._select_graph = None
        self._publish_graph = None
        self.reset()
        if capture:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                self._select()
                self._publish()
                stream.synchronize()
                self._select_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._select_graph, stream=stream):
                    self._select()
                self._publish_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._publish_graph, stream=stream):
                    self._publish()
            stream.synchronize()
            self.reset()

    def _copy_learner_buffers(self):
        self.observations.copy_(self.duel.observations[0::2])
        self.rewards.copy_(self.duel.rewards[0::2])
        self.terminals.copy_(self.duel.terminals[0::2])
        self.action_mask.copy_(self.duel.action_mask[0::2])

    def _select(self):
        self.full_actions[0::2].copy_(self.learner_actions)
        self.full_actions[1::2, 0].copy_(self.dummy.select(
            self.duel.observations[1::2], self.duel.action_mask[1::2],
        ))

    def _publish(self):
        self.combat_metrics.update(self.duel.observations, self.duel.terminals)
        self.behavior_metrics.update(
            self.duel.observations, self.duel.terminals, self.full_actions,
            self.duel.scheduler.move_start_edge, self.duel.combat.tick_score_delta,
        )
        self.dummy.reset(self.duel.terminals[1::2])
        self._copy_learner_buffers()

    def reset(self):
        self.duel.reset()
        self.dummy.reset()
        self.combat_metrics.reset()
        self.behavior_metrics.reset()
        self.learner_actions.fill_(1)
        self.full_actions.fill_(1)
        self._copy_learner_buffers()

    def step(self, actions):
        if actions.shape != (self.rows, 1) or actions.device != self.learner_actions.device:
            raise ValueError("learner actions must be CUDA [arenas, 1]")
        self.learner_actions.copy_(actions)
        if self._select_graph is None:
            self._select()
        else:
            self._select_graph.replay()
        self.duel.step(self.full_actions)
        if self._publish_graph is None:
            self._publish()
        else:
            self._publish_graph.replay()

    def log(self):
        self.duel.check_status()
        self.dummy.check_status()
        return {}

    def close(self):
        self.duel.close()
