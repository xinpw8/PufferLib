"""CUDA tensor implementation of gear_sonic_native_batch observation assembly.

The low-level controller keeps the existing 1762/994 feature layouts, ten
history frames, joint permutations, float64 quaternion intermediates, and
float32 targets. No simulator state is copied to the host by these operations.
"""

from __future__ import annotations

import torch

from gear_sonic_candidate import (
    ACTION_SCALE_MUJOCO, DEFAULT_ANGLES_MUJOCO,
    ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB,
)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def rotation_six(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
    ), dim=-1).float()


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
    # The native history first stores q as binary32, then promotes to double.
    w, x, y, z = quaternion_conjugate(q.float().double()).unbind(-1)
    return torch.stack((
        (-y) * w * 2.0 + x * (-z) * 2.0,
        x * w * 2.0 + y * (-z) * 2.0,
        -(2.0 * w * w - 1.0) + z * (-z) * 2.0,
    ), dim=-1).float()


class G1GpuControllerState:
    def __init__(self, rows: int, device: str = "cuda:0"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("robot controller state requires CUDA")
        self.rows = rows
        self.to_policy = torch.as_tensor(
            MUJOCO_TO_ISAACLAB.copy(), device=device, dtype=torch.long,
        )
        self.to_mujoco = torch.as_tensor(
            ISAACLAB_TO_MUJOCO.copy(), device=device, dtype=torch.long,
        )
        self.default = torch.as_tensor(
            DEFAULT_ANGLES_MUJOCO.copy(), device=device, dtype=torch.float32,
        )
        self.scale = torch.as_tensor(
            ACTION_SCALE_MUJOCO.copy(), device=device, dtype=torch.float32,
        )
        self.encoder_observations = torch.zeros((rows, 1762), device=device)
        self.decoder_observations = torch.zeros((rows, 994), device=device)
        self.history = {
            name: torch.zeros((rows, 10, width), device=device)
            for name, width in (
                ("angular", 3), ("position", 29), ("velocity", 29),
                ("action", 29), ("gravity", 3),
            )
        }
        self.last_actions = torch.zeros((rows, 29), device=device)
        self.targets = torch.zeros((rows, 29), device=device)

    def _check_device(self, *values: torch.Tensor) -> None:
        if any(value.device != self.last_actions.device for value in values):
            raise ValueError("all robot controller inputs must share its CUDA device")

    def reset(self, reset_rows: torch.Tensor) -> None:
        self._check_device(reset_rows)
        for values in self.history.values():
            values.masked_fill_(reset_rows[:, None, None], 0.0)
        self.last_actions.masked_fill_(reset_rows[:, None], 0.0)
        self.targets.masked_fill_(reset_rows[:, None], 0.0)

    def prepare(
        self,
        base_quaternion: torch.Tensor,
        angular_velocity: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        heading_delta: torch.Tensor,
        reference_position: torch.Tensor,
        reference_next_position: torch.Tensor,
        reference_root_xyzw: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        self._check_device(
            base_quaternion, angular_velocity, joint_position, joint_velocity,
            heading_delta, reference_position, reference_next_position,
            reference_root_xyzw, active,
        )
        entries = {
            "angular": angular_velocity.float(),
            "position": (
                joint_position.double() - self.default.double()
            )[:, self.to_policy].float(),
            "velocity": joint_velocity[:, self.to_policy].float(),
            "action": self.last_actions,
            "gravity": projected_gravity(base_quaternion),
        }
        for name, values in self.history.items():
            shifted = torch.cat((values[:, 1:], entries[name][:, None]), dim=1)
            values.copy_(torch.where(active[:, None, None], shifted, values))
        obs = self.encoder_observations
        obs.zero_()
        obs[:, 4:294] = reference_position[:, :, self.to_policy].flatten(1)
        velocity = (
            reference_next_position.double() - reference_position.double()
        ) / 0.02
        obs[:, 294:584] = velocity[:, :, self.to_policy].float().flatten(1)
        reference_wxyz = torch.cat((
            reference_root_xyzw[..., 3:4], reference_root_xyzw[..., :3],
        ), dim=-1).double()
        aligned = quaternion_multiply(heading_delta.double()[:, None], reference_wxyz)
        relative = quaternion_multiply(
            quaternion_conjugate(base_quaternion.double())[:, None], aligned,
        )
        obs[:, 601:661] = rotation_six(relative).flatten(1)
        return obs

    def decoder_input(self, tokens: torch.Tensor) -> torch.Tensor:
        self._check_device(tokens)
        self.decoder_observations.copy_(torch.cat((
            tokens,
            self.history["angular"].flatten(1),
            self.history["position"].flatten(1),
            self.history["velocity"].flatten(1),
            self.history["action"].flatten(1),
            self.history["gravity"].flatten(1),
        ), dim=1))
        return self.decoder_observations

    def apply_actions(self, raw_actions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        self._check_device(raw_actions, active)
        clipped = raw_actions.clamp(-100.0, 100.0)
        # Separate operations preserve the native float32 multiply then add.
        scaled = clipped[:, self.to_mujoco] * self.scale
        targets = self.default + scaled
        self.targets.copy_(torch.where(active[:, None], targets, self.targets))
        self.last_actions.copy_(torch.where(active[:, None], clipped, self.last_actions))
        return self.targets
