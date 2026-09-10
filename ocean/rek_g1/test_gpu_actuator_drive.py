"""Compare GPU actuator filtering, suspension and retained-force arithmetic."""

import unittest
import numpy as np
import torch

from gear_sonic_candidate import KP_MUJOCO, KD_MUJOCO, PUBLIC_EFFORT_LIMIT_MUJOCO
from gpu_actuator_drive import GpuActuatorDrive


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GpuActuatorDriveTests(unittest.TestCase):
    def test_filter_dampening_and_deferred_reset(self):
        rng = np.random.default_rng(19791)
        rows = 8
        limited = np.ones((2, 29), bool)
        limited[:, ::4] = False
        ranges = np.empty((2, 29, 2), np.float64)
        ranges[:, :, 0] = -0.312345678
        ranges[:, :, 1] = 0.587654321
        drive = GpuActuatorDrive(rows, joint_limited=limited, joint_ranges=ranges)
        lower, upper = ranges.astype(np.float32).transpose(2, 0, 1)
        tensor = lambda value: torch.as_tensor(value, device="cuda")
        filtered = np.zeros((rows, 29), np.float32)
        retained = filtered.copy()
        control = filtered.copy()
        initialized = np.zeros(rows, bool)
        dampened = initialized.copy()
        resetting = initialized.copy()
        retention = np.float32(0.1)
        kp = KP_MUJOCO.astype(np.float32)
        kd = KD_MUJOCO.astype(np.float32)
        force_limit = PUBLIC_EFFORT_LIMIT_MUJOCO.astype(np.float32)
        for tick in range(25):
            if tick in (3, 9):
                desired = dampened.copy()
                desired[[tick % rows, (tick + 2) % rows]] = True
                entering = desired & ~dampened
                retained[entering] = control[entering]
                drive.set_dampened(tensor(desired), tensor(control))
                dampened = desired
            if tick == 12:
                reset = np.asarray([True, True, False, False, False, False, True, True])
                entering = reset & ~(dampened | resetting)
                retained[entering] = control[entering]
                resetting |= reset
                drive.begin_reset(tensor(reset), tensor(control))
            if tick == 13:
                reset = resetting.copy()
                filtered[reset] = retained[reset] = control[reset] = 0
                initialized[reset] = dampened[reset] = resetting[reset] = False
                drive.complete_reset(tensor(reset))
            targets = rng.normal(size=(rows, 29)).astype(np.float32)
            for substep in range(10):
                position = rng.normal(size=(rows, 29)).astype(np.float32)
                velocity = rng.normal(0, 10, (rows, 29)).astype(np.float32)
                active = ~(dampened | resetting)
                if substep % 2 == 0:
                    delta = (targets - filtered) * np.float32(0.5568627119064331)
                    next_filtered = filtered + delta
                    next_filtered[~initialized] = targets[~initialized]
                    filtered[active] = next_filtered[active]
                    initialized[active] = True
                for row in range(rows):
                    if dampened[row]:
                        force = (kp * retention).astype(np.float64) * (retained[row].astype(np.float64) - position[row])
                        force -= (kd * retention).astype(np.float64) * velocity[row]
                        limit = (force_limit * retention).astype(np.float64)
                        force = np.clip(force, -limit, limit)
                        bias = -kp.astype(np.float64) * position[row] - kd.astype(np.float64) * velocity[row]
                        control[row] = ((force - bias) / kp.astype(np.float64)).astype(np.float32)
                    elif resetting[row]:
                        control[row] = retained[row]
                    else:
                        side = row % 2
                        control[row] = np.where(limited[side],
                            np.clip(filtered[row], lower[side], upper[side]), filtered[row])
                actual = drive.prepare(tensor(targets), tensor(position), tensor(velocity), substep=substep)
                np.testing.assert_array_equal(actual.cpu().numpy(), control)
                np.testing.assert_array_equal(drive.filtered.cpu().numpy(), filtered)

    def test_active_clamp_preserves_filter_and_suspended_branches(self):
        limits = np.ones((2, 29), bool)
        limits[:, 1] = False
        ranges = np.tile(np.asarray([-0.2, 0.3]), (2, 29, 1))
        drive = GpuActuatorDrive(2, joint_limited=limits, joint_ranges=ranges)
        targets = torch.full((2, 29), 2.0, device='cuda')
        zero = torch.zeros_like(targets)
        actual = drive.prepare(targets, zero, zero, substep=0)
        self.assertEqual(actual[0, 0].item(), np.float32(0.3))
        self.assertEqual(actual[0, 1].item(), 2.0)
        self.assertTrue(torch.equal(drive.filtered, targets))
        retained = torch.full_like(targets, 1.5)
        drive.begin_reset(torch.tensor([True, False], device='cuda'), retained)
        drive.set_dampened(torch.tensor([False, True], device='cuda'), retained)
        actual = drive.prepare(targets, zero, zero, substep=1)
        self.assertTrue(torch.equal(actual[0], retained[0]))
        self.assertTrue(torch.equal(drive.retained_targets, retained))
        expected = (torch.minimum(drive.retained_kp * 1.5,
            drive.retained_force_limit) / drive.kp.double()).float()
        torch.testing.assert_close(actual[1], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
