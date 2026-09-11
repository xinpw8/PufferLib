"""Explicitly configured, default-disabled training-only potential shaping.

The candidate training wrapper activates this only at positive explicit scale.
Evaluation never activates it. It changes no game score, opponent, action or
observation. CPU execution is restricted to tests. A separate CUDA fixture
checks capture correctness; a training benefit must be measured separately.

For the actual wrapper-visible transition (s, a, r, s_next), use
    shaped_r = r + gamma * Phi(s_next) - Phi(s)
with Phi = scale*cos(opponent ego-yaw bearing), and Phi=0 when the associated
terminal flag is set. The scale is a training choice, not a measured REK rule.

The current GPU runtime publishes a terminal observation, then applies the
episode reset at the start of the NEXT step (gpu_semantic_duel._step_impl and
g1_native_combat_cuda.begin_tick_kernel). Consequently BOTH terminal masks
are required. A terminal pre-step observation has zero potential, including
when it is the input to the following delayed-reset transition. Never read
the pre-step potential from the mutable observation buffer after stepping.

For T transitions the discounted shaping sum is exactly
    -Phi(s_0) + gamma**T * Phi(s_T)
in real arithmetic. A completed episode has zero final potential. A rollout
horizon is not a terminal: bootstrapping must use the transformed value
V_shaped(s)=V_base(s)-Phi(s). The learner must use the SAME gamma and must not
clip the combined reward. PPO approximation, finite precision and optimization
still prevent a guarantee of learning improvement or identical learned weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch


@dataclass(frozen=True)
class FacingPotentialConfig:
    gamma: float
    scale: float

    def __post_init__(self):
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("scale must be finite and nonnegative")

    def validate_training_discount(self, gamma, *, reward_clip):
        if gamma != self.gamma:
            raise ValueError("shaping and learner discount must match exactly")
        if reward_clip != 0:
            raise ValueError("reward clipping breaks the potential identity")

    def metadata(self):
        return {
            "name": "facing_cosine_potential_v1",
            "gamma": self.gamma,
            "scale": self.scale,
            "training_only": True,
            "default_activation": False,
            "formula": "raw_reward + gamma*next_potential - current_potential",
            "raw_reward": "unchanged own-minus-opponent score delta",
            "potential": "scale*cos(egocentric planar opponent bearing)",
            "terminal_potential": 0,
            "both_current_and_next_terminal_masks_required": True,
            "undefined_direction_extension": "zero potential at coincident XY or exactly vertical projected forward",
            "source_columns": {"self_xy": [0, 1], "self_quaternion_wxyz": [3, 4, 5, 6], "opponent_xy": [86, 87]},
            "calculation": "float64 potential and shaping; float32 learner reward storage",
            "raw_simulator_buffers_mutated": False,
        }


def resolve_training_reward(native_args, *, scale, reward_clip):
    """Use the actual native gamma source and its C++ float representation.

    bindings.cu assigns train.gamma to HypersT.gamma, a float. The caller also
    verifies this value against the constructed native trainer before running.
    Scale zero returns no shaper, retaining the original device hot path.
    """
    configured_gamma = float(native_args["train"]["gamma"])
    if not math.isfinite(configured_gamma) or not 0 <= configured_gamma <= 1:
        raise ValueError("configured native gamma must be finite and in [0, 1]")
    if not math.isfinite(reward_clip) or reward_clip < 0:
        raise ValueError("reward_clip must be finite and nonnegative")
    native_gamma = struct.unpack("f", struct.pack("f", configured_gamma))[0]
    config = FacingPotentialConfig(native_gamma, float(scale))
    if config.scale > 0:
        config.validate_training_discount(native_gamma, reward_clip=reward_clip)
    descriptor = {
        "name": ("facing_cosine_potential_v1" if config.scale > 0 else
                 "none" if reward_clip == 0 else "symmetric_clamp"),
        "facing_potential_scale": config.scale,
        "gamma": native_gamma,
        "gamma_configured": configured_gamma,
        "gamma_source": "native_args.train.gamma converted to HypersT float32",
        "native_reward_clip": reward_clip,
        "potential": config.metadata() if config.scale > 0 else None,
        "raw_score_reward_unchanged": True,
        "potential_applied_to": "learner reward buffer only" if config.scale > 0 else None,
        "evaluation_shaping": False,
    }
    return (config if config.scale > 0 else None), descriptor


class GpuFacingPotential:
    """Two-phase potential snapshot around a single environment step.

    ``begin_transition(raw_current, current_terminals)`` must run before the
    environment mutates its buffers. ``finish_transition(raw_next, raw_reward,
    next_terminals)`` must run after that step, once per begin. All hot-path
    operations are device tensor operations; check_status is an explicit
    reporting boundary. Only the owned reward buffer goes to a future learner.

    A zero-range or vertical-forward orientation has no planar bearing. Zero
    potential there is an explicit extension of the TRAINING function, not a
    guessed simulator direction. Invalid quaternions/nonfinite data are errors.
    """

    def __init__(self, rows, device, config: FacingPotentialConfig, *, allow_cpu_for_tests=False):
        if rows < 1 or not isinstance(config, FacingPotentialConfig):
            raise ValueError("positive rows and an explicit shaping config are required")
        requested = torch.device(device)
        if requested.type != "cuda" and not (requested.type == "cpu" and allow_cpu_for_tests):
            raise ValueError("potential shaping requires CUDA; CPU is test-only")
        self.rows, self.config = rows, config
        self.rewards = torch.empty(rows, device=requested, dtype=torch.float32)
        self.device = self.rewards.device
        self.current_potential = torch.zeros(rows, device=self.device, dtype=torch.float64)
        self.next_potential = torch.zeros_like(self.current_potential)
        self.shaping_reward = torch.zeros_like(self.current_potential)
        self.status = torch.zeros(rows, device=self.device, dtype=torch.int32)
        self.pending = torch.zeros(rows, device=self.device, dtype=torch.bool)

    def reset(self):
        self.status.zero_()
        self.pending.zero_()
        self.current_potential.zero_()
        self.next_potential.zero_()
        self.shaping_reward.zero_()
        self.rewards.zero_()

    def _require_rows(self, values, name, *, observations=False):
        shape = (self.rows, 223) if observations else (self.rows,)
        if values.shape != shape or values.device != self.device or values.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 {shape} on the shaping device")

    def _potential(self, observations, terminals):
        self._require_rows(observations, "raw observations", observations=True)
        self._require_rows(terminals, "terminals")
        q = observations[:, 3:7].double()
        norm = torch.linalg.vector_norm(q, dim=1)
        w, x, y, z = (q / norm[:, None]).unbind(1)
        fx, fy = 1 - 2*(y*y + z*z), 2*(w*z + x*y)
        dx, dy = (observations[:, 86:88].double() - observations[:, :2].double()).unbind(1)
        forward, lateral = fx*dx + fy*dy, -fy*dx + fx*dy
        potential = self.config.scale * torch.cos(torch.atan2(lateral, forward))
        undefined = ((dx == 0) & (dy == 0)) | ((fx == 0) & (fy == 0))
        potential = torch.where(undefined | (terminals != 0), 0.0, potential)
        self.status.bitwise_or_(
            (~torch.isfinite(observations).all(1)).to(torch.int32)
            | ((~torch.isfinite(norm) | (norm <= 0)).to(torch.int32) * 2)
            | ((~((terminals == 0) | (terminals == 1))).to(torch.int32) * 4)
            | ((~torch.isfinite(potential)).to(torch.int32) * 8)
        )
        return potential

    def begin_transition(self, raw_current, current_terminals):
        self.status.bitwise_or_(self.pending.to(torch.int32) * 16)
        self.current_potential.copy_(self._potential(raw_current, current_terminals))
        self.pending.fill_(True)

    def finish_transition(self, raw_next, raw_reward, next_terminals):
        self._require_rows(raw_reward, "raw rewards")
        if raw_reward.data_ptr() == self.rewards.data_ptr():
            raise ValueError("shaped rewards cannot be reused as raw rewards")
        self.status.bitwise_or_((~self.pending).to(torch.int32) * 32)
        self.next_potential.copy_(self._potential(raw_next, next_terminals))
        self.shaping_reward.copy_(self.config.gamma*self.next_potential - self.current_potential)
        self.rewards.copy_(raw_reward.double() + self.shaping_reward)
        self.status.bitwise_or_((~torch.isfinite(raw_reward) | ~torch.isfinite(self.rewards)).to(torch.int32) * 64)
        self.pending.zero_()
        return self.rewards

    def check_status(self):
        statuses = self.status.detach().cpu().tolist()
        if any(statuses):
            raise RuntimeError(f"invalid potential shaping transition: {statuses}")
