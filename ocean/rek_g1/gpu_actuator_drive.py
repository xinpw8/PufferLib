"""CUDA implementation of the native duel target filter and retained drives."""

from __future__ import annotations

import torch
import numpy as np

from gear_sonic_candidate import KP_MUJOCO, KD_MUJOCO, PUBLIC_EFFORT_LIMIT_MUJOCO


class GpuActuatorDrive:
    def __init__(self, rows: int, device: str = "cuda:0", *, joint_limited, joint_ranges):
        if rows < 2 or rows % 2 or torch.device(device).type != "cuda":
            raise ValueError("actuator drives require paired CUDA robot rows")
        self.rows = rows
        self.device = torch.device(device)
        self.filtered = torch.zeros((rows, 29), device=device)
        self.initialized = torch.zeros(rows, device=device, dtype=torch.bool)
        self.dampened = torch.zeros(rows, device=device, dtype=torch.bool)
        self.resetting = torch.zeros(rows, device=device, dtype=torch.bool)
        self.retained_targets = torch.zeros_like(self.filtered)
        self.controls = torch.zeros_like(self.filtered)
        limited = np.asarray(joint_limited, dtype=bool)
        ranges = np.asarray(joint_ranges, dtype=np.float32)
        if limited.shape != (2, 29) or ranges.shape != (2, 29, 2):
            raise ValueError("joint limit metadata must describe both 29-joint fighters")
        if not np.isfinite(ranges[limited]).all() or np.any(
                ranges[..., 0][limited] > ranges[..., 1][limited]):
            raise ValueError("limited joints require finite, ordered float32 ranges")
        self.joint_limited = torch.as_tensor(np.tile(limited, (rows // 2, 1)), device=device)
        bounds = torch.as_tensor(np.tile(ranges, (rows // 2, 1, 1)), device=device)
        self.joint_lower, self.joint_upper = bounds.unbind(-1)
        tensor = lambda value: torch.as_tensor(value.copy(), device=device, dtype=torch.float32)
        self.kp = tensor(KP_MUJOCO)
        self.kd = tensor(KD_MUJOCO)
        self.force_limit = tensor(PUBLIC_EFFORT_LIMIT_MUJOCO)
        # Native storage rounds the scaled gains and limits to float32 before
        # preparing each subsequent force in double precision.
        self.retained_kp = (self.kp * 0.10000000149011612).double()
        self.retained_kd = (self.kd * 0.10000000149011612).double()
        self.retained_force_limit = (self.force_limit * 0.10000000149011612).double()

    @classmethod
    def from_physics(cls, physics):
        model = physics.host_model
        joints = np.stack([model.actuator_trnid[ids, 0] for ids in physics.actuator_ids])
        return cls(physics.arenas * 2, str(physics.qpos.device),
            joint_limited=model.jnt_limited[joints], joint_ranges=model.jnt_range[joints])

    @property
    def active(self):
        return ~(self.dampened | self.resetting)

    def set_dampened(self, desired, live_controls):
        entering = desired & ~self.dampened
        self.retained_targets.copy_(torch.where(
            entering[:, None], live_controls, self.retained_targets,
        ))
        self.dampened.copy_(desired)

    def begin_reset(self, reset_rows, live_controls):
        entering = reset_rows & self.active
        self.retained_targets.copy_(torch.where(
            entering[:, None], live_controls, self.retained_targets,
        ))
        self.resetting.logical_or_(reset_rows)

    def complete_reset(self, reset_rows):
        self.filtered.masked_fill_(reset_rows[:, None], 0)
        self.retained_targets.masked_fill_(reset_rows[:, None], 0)
        self.controls.masked_fill_(reset_rows[:, None], 0)
        self.initialized.logical_and_(~reset_rows)
        self.dampened.logical_and_(~reset_rows)
        self.resetting.logical_and_(~reset_rows)

    def prepare(self, targets, joint_position, joint_velocity, *, substep: int):
        if substep not in range(10):
            raise ValueError("physics substep must be in 0..9")
        active = self.active
        if substep % 2 == 0:
            smoothed = self.filtered + (targets - self.filtered) * 0.5568627119064331
            filtered = torch.where(self.initialized[:, None], smoothed, targets)
            self.filtered.copy_(torch.where(active[:, None], filtered, self.filtered))
            self.initialized.logical_or_(active)
        position, velocity = joint_position.double(), joint_velocity.double()
        force = self.retained_kp * (self.retained_targets.double() - position)
        force = force - self.retained_kd * velocity
        force = torch.maximum(torch.minimum(force, self.retained_force_limit), -self.retained_force_limit)
        base_bias = -self.kp.double() * position - self.kd.double() * velocity
        # Keep the shared model's position gains untouched, exactly as the
        # native multi-arena implementation does. Encode retained force as ctrl.
        dampened_control = ((force - base_bias) / self.kp.double()).float()
        # Native prepare_controls clips only the final active drive target.
        # Filtering retains the unbounded target, and suspended drives follow
        # their separate retained-target/retained-force branches unchanged.
        limited_control = torch.maximum(torch.minimum(self.filtered, self.joint_upper), self.joint_lower)
        active_control = torch.where(self.joint_limited, limited_control, self.filtered)
        output = torch.where(self.resetting[:, None], self.retained_targets, active_control)
        self.controls.copy_(torch.where(self.dampened[:, None], dampened_control, output))
        return self.controls
