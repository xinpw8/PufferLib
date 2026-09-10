"""CUDA physical reset operations matching gear_sonic_native_duel.c.

Round initialization uses the idle pose. A counted-fall reset instead restores
roots first and zeroes articulated joints after the next physics boundary.
Free-root velocities survive that counted-fall reset, as in the native code.
"""

from __future__ import annotations

import numpy as np
import torch

import gear_sonic_candidate as candidate


class GpuDuelReset:
    def __init__(self, physics, assets):
        self.physics = physics
        device = physics.qpos.device
        self.pending = torch.zeros(physics.arenas, dtype=torch.bool, device=device)
        pose = physics.host_model.qpos0.copy()
        idle = assets.roles["idle"]
        reference = assets.host_arrays[idle["files"]["mujoco_joint_order"]][0]
        reference_root = assets.host_arrays[idle["files"]["xyzw"]][0][[3, 0, 1, 2]]
        headings = []
        for side in range(2):
            joints = physics.host_model.actuator_trnid[physics.actuator_ids[side], 0]
            limited = physics.host_model.jnt_limited[joints].astype(bool)
            limits = physics.host_model.jnt_range[joints]
            initial = reference.copy()
            initial[limited] = np.clip(initial[limited], limits[limited, 0], limits[limited, 1])
            pose[physics.joint_qpos[side]] = initial
            headings.append(candidate.reference_heading_delta(
                pose[side * 36 + 3:side * 36 + 7], reference_root,
            ))
        self.initial_qpos = torch.as_tensor(pose, dtype=physics.qpos.dtype, device=device)
        self.spawn_roots = torch.as_tensor(
            physics.host_model.qpos0.reshape(2, 36)[:, :7].copy(),
            dtype=physics.qpos.dtype, device=device,
        )
        self.initial_heading = torch.as_tensor(np.tile(headings, (physics.arenas, 1)), device=device)
        self.qindices = torch.as_tensor(np.stack(physics.joint_qpos), device=device)
        self.dqindices = torch.as_tensor(np.stack(physics.joint_qvel), device=device)
        self.reset_buffers = {
            name: physics.wp.to_torch(getattr(physics.data, name))
            for name in ("qacc", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "act")
        }

    def _validate(self, mask):
        if mask.shape != self.pending.shape or mask.device != self.pending.device or mask.dtype != torch.bool:
            raise ValueError("reset mask must have one CUDA Boolean per arena")

    def _restore_roots(self, mask):
        roots = self.physics.qpos.reshape(-1, 2, 36)[..., :7]
        roots.copy_(torch.where(mask[:, None, None], self.spawn_roots, roots))

    def full(self, mask, *, reset_clock=False):
        """Initialize selected arenas; caller resets controller and game state."""
        self._validate(mask)
        physics = self.physics
        physics.qpos.copy_(torch.where(mask[:, None], self.initial_qpos, physics.qpos))
        physics.qvel.masked_fill_(mask[:, None], 0)
        physics.ctrl.masked_fill_(mask[:, None], 0)
        for values in self.reset_buffers.values():
            values.masked_fill_(mask.reshape(-1, *([1] * (values.ndim - 1))), 0)
        if reset_clock:
            physics.time.masked_fill_(mask, 0)
        self.pending.logical_and_(~mask)
        physics.forward_selected(mask)

    def begin(self, mask):
        """Called after physics; only root poses change at this boundary."""
        self._validate(mask)
        self._restore_roots(mask)
        self.pending.logical_or_(mask)
        self.physics.forward_selected(mask)

    def complete(self, mask):
        """Caller supplies resets pending before the just-finished 2 ms step."""
        self._validate(mask)
        physics = self.physics
        joints = physics.qpos[:, self.qindices]
        velocities = physics.qvel[:, self.dqindices]
        physics.qpos[:, self.qindices] = torch.where(mask[:, None, None], 0.0, joints)
        physics.qvel[:, self.dqindices] = torch.where(mask[:, None, None], 0.0, velocities)
        physics.ctrl.masked_fill_(mask[:, None], 0)
        self._restore_roots(mask)
        self.pending.logical_and_(~mask)
        physics.forward_selected(mask)
