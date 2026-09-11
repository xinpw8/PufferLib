"""CPU-only geometry, transition-contract and discounted-telescoping proofs."""

import math
import unittest

import torch

from gpu_facing_potential import FacingPotentialConfig, GpuFacingPotential


def observations(bearings, yaws=None):
    angle = torch.as_tensor(bearings, dtype=torch.float64)
    yaw = torch.zeros_like(angle) if yaws is None else torch.as_tensor(yaws, dtype=torch.float64)
    values = torch.zeros((len(angle), 223), dtype=torch.float32)
    values[:, 3], values[:, 6] = torch.cos(yaw/2), torch.sin(yaw/2)
    values[:, 86], values[:, 87] = torch.cos(yaw+angle), torch.sin(yaw+angle)
    return values


def shaper(rows=1, gamma=0.99, scale=0.25):
    return GpuFacingPotential(rows, "cpu", FacingPotentialConfig(gamma, scale), allow_cpu_for_tests=True)


class FacingPotentialTests(unittest.TestCase):
    def test_explicit_config_and_matching_learner_gamma_without_clipping(self):
        config = FacingPotentialConfig(0.99, 0.25)
        config.validate_training_discount(0.99, reward_clip=0)
        for gamma, clip in ((0.98, 0), (0.99, 1), (0.99, float("nan"))):
            with self.assertRaises(ValueError):
                config.validate_training_discount(gamma, reward_clip=clip)
        for gamma, scale in ((-1, 1), (1.01, 1), (float("nan"), 1), (0.99, -1), (0.99, float("inf"))):
            with self.assertRaises(ValueError):
                FacingPotentialConfig(gamma, scale)
        with self.assertRaises(ValueError):
            GpuFacingPotential(1, "cpu", config)
        self.assertFalse(config.metadata()["default_activation"])

    def test_known_geometry_and_global_yaw(self):
        obj = shaper(4, scale=0.4)
        values = observations([0, math.pi/2, math.pi, -math.pi/2], [0.3, -0.7, 1.8, 2.5])
        obj.begin_transition(values, torch.zeros(4))
        torch.testing.assert_close(obj.current_potential, torch.tensor([0.4, 0, -0.4, 0], dtype=torch.float64), atol=4e-8, rtol=0)
        obj.check_status()

    def test_translation_quaternion_sign_and_scale(self):
        obj = shaper(3)
        values = observations([0.2, -1.1, 2.8], [0.4, -0.6, 1.9])
        obj.begin_transition(values, torch.zeros(3))
        first = obj.current_potential.clone()
        obj.finish_transition(values, torch.zeros(3), torch.zeros(3))
        values[:, 0:2] += torch.tensor([1.0, 2.0])
        values[:, 86:88] += torch.tensor([1.0, 2.0])
        values[:, 3:7] *= -2
        obj.begin_transition(values, torch.zeros(3))
        torch.testing.assert_close(first, obj.current_potential, atol=3e-8, rtol=0)
        obj.check_status()

    def test_snapshot_prevents_mutable_next_observation_aliasing(self):
        obj = shaper(gamma=0.9, scale=0.25)
        live = observations([0])
        obj.begin_transition(live, torch.zeros(1))
        live.copy_(observations([math.pi]))
        result = obj.finish_transition(live, torch.tensor([5.0]), torch.zeros(1))
        self.assertAlmostEqual(result.item(), 5 - 0.225 - 0.25, places=6)
        self.assertEqual(obj.current_potential.item(), 0.25)
        obj.check_status()

    def test_terminal_zero_and_delayed_reset_do_not_leak_old_orientation(self):
        obj = shaper(2, gamma=0.9, scale=0.25)
        start = observations([0, 0])
        terminal = observations([0, math.pi])
        obj.begin_transition(start, torch.zeros(2))
        terminal_reward = obj.finish_transition(terminal, torch.tensor([5.0, 5.0]), torch.ones(2)).clone()
        torch.testing.assert_close(terminal_reward, torch.full((2,), 4.75), atol=0, rtol=0)
        obj.begin_transition(terminal, torch.ones(2))
        first_live = observations([math.pi/3, math.pi/3])
        after_reset = obj.finish_transition(first_live, torch.zeros(2), torch.zeros(2))
        torch.testing.assert_close(after_reset[0], after_reset[1], atol=0, rtol=0)
        self.assertAlmostEqual(after_reset[0].item(), 0.9*0.25*0.5, places=7)
        self.assertEqual(obj.current_potential.abs().max().item(), 0)
        # Negative control: unmasked pre-step potential leaks the old terminal
        # pose into the new round by 2*scale in this exact pair.
        naive = obj.config.gamma*obj.next_potential - torch.tensor([0.25, -0.25])
        self.assertAlmostEqual((naive[1] - naive[0]).item(), 0.5)
        obj.check_status()

    def test_variable_length_completed_returns_telescope(self):
        generator = torch.Generator().manual_seed(7123)
        for gamma in (0.0, 0.9, 0.999, 1.0):
            for length in (1, 2, 7, 31, 128):
                obj = shaper(3, gamma=gamma, scale=0.37)
                angles = torch.rand((length+1, 3), generator=generator)*6.2-3.1
                rewards = torch.randint(-5, 6, (length, 3), generator=generator).float()
                base = torch.zeros(3, dtype=torch.float64)
                shaped = torch.zeros_like(base)
                exact_extra = torch.zeros_like(base)
                initial = None
                for tick in range(length):
                    obj.begin_transition(observations(angles[tick]), torch.zeros(3))
                    if initial is None:
                        initial = obj.current_potential.clone()
                    terminal = torch.full((3,), float(tick == length-1))
                    value = obj.finish_transition(observations(angles[tick+1]), rewards[tick], terminal)
                    base += gamma**tick * rewards[tick].double()
                    shaped += gamma**tick * value.double()
                    exact_extra += gamma**tick * obj.shaping_reward
                torch.testing.assert_close(exact_extra, -initial, atol=5e-15, rtol=0)
                torch.testing.assert_close(shaped-base, -initial, atol=4e-6, rtol=0)
                obj.check_status()

    def test_completed_return_ordering_same_start_different_lengths(self):
        totals = []
        for angles, raw in (([0.4, 1.7, -1.2], [0, 3]), ([0.4, 0.1], [2])):
            obj = shaper(gamma=0.9, scale=0.4)
            base = transformed = 0.0
            for tick, reward in enumerate(raw):
                obj.begin_transition(observations([angles[tick]]), torch.zeros(1))
                value = obj.finish_transition(observations([angles[tick+1]]), torch.tensor([float(reward)]), torch.tensor([float(tick == len(raw)-1)]))
                base += 0.9**tick * reward
                transformed += 0.9**tick * value.item()
            totals.append((base, transformed))
        self.assertGreater(totals[0][0], totals[1][0])
        self.assertGreater(totals[0][1], totals[1][1])
        self.assertAlmostEqual(totals[0][0]-totals[1][0], totals[0][1]-totals[1][1], places=6)

    def test_nonterminal_horizon_retains_bootstrap_potential(self):
        obj = shaper(gamma=0.9, scale=0.5)
        exact_extra = 0.0
        angles = [0.2, 1.1, 2.7]
        for tick in range(2):
            obj.begin_transition(observations([angles[tick]]), torch.zeros(1))
            if tick == 0:
                initial = obj.current_potential.item()
            obj.finish_transition(observations([angles[tick+1]]), torch.zeros(1), torch.zeros(1))
            exact_extra += 0.9**tick*obj.shaping_reward.item()
        final = obj.next_potential.item()
        self.assertAlmostEqual(exact_extra, -initial + 0.9**2*final, places=14)
        base_value = 3.2
        self.assertAlmostEqual(exact_extra + 0.9**2*(base_value-final), 0.9**2*base_value-initial, places=14)

    def test_raw_buffers_remain_unchanged_and_output_pointer_stable(self):
        obj = shaper(2)
        raw = observations([0.2, 1.5])
        reward, terminal = torch.tensor([3.0, -5.0]), torch.tensor([0.0, 1.0])
        originals = [value.clone() for value in (raw, reward, terminal)]
        pointer = obj.rewards.data_ptr()
        for _ in range(3):
            obj.begin_transition(raw, torch.zeros(2))
            obj.finish_transition(raw, reward, terminal)
            self.assertEqual(obj.rewards.data_ptr(), pointer)
        for value, original in zip((raw, reward, terminal), originals):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
        obj.check_status()

    def test_undefined_planar_direction_has_declared_zero_extension(self):
        obj = shaper(2)
        raw = observations([0, 0])
        raw[0, 86:88] = 0
        # Exactly vertical local +X for the rational unit quaternion below.
        raw[1, 3:7] = torch.tensor([0.5, 0.5, 0.5, -0.5])
        obj.begin_transition(raw, torch.zeros(2))
        torch.testing.assert_close(obj.current_potential, torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)
        obj.check_status()

    def test_invalid_inputs_ordering_and_alias_fail_explicitly(self):
        for case in ("quaternion", "nonfinite", "terminal", "reward", "without_begin", "double_begin"):
            obj = shaper()
            raw, terminal, reward = observations([0]), torch.zeros(1), torch.zeros(1)
            if case == "quaternion": raw[:, 3:7] = 0
            if case == "nonfinite": raw[:, 86] = float("nan")
            if case == "terminal": terminal.fill_(0.5)
            if case == "reward": reward.fill_(float("inf"))
            if case != "without_begin": obj.begin_transition(raw, terminal)
            if case == "double_begin": obj.begin_transition(raw, terminal)
            obj.finish_transition(raw, reward, terminal)
            with self.assertRaises(RuntimeError): obj.check_status()
            obj.reset()
            obj.check_status()
        obj = shaper()
        with self.assertRaises(ValueError): obj.begin_transition(torch.zeros((1, 222)), torch.zeros(1))
        with self.assertRaises(ValueError): obj.begin_transition(observations([0]).double(), torch.zeros(1))
        with self.assertRaises(ValueError): obj.finish_transition(observations([0]), obj.rewards, torch.zeros(1))

    def test_zero_scale_exact_ablation(self):
        obj = shaper(2, scale=0)
        reward = torch.tensor([3.5, -5.0])
        obj.begin_transition(observations([0.1, 1.3]), torch.zeros(2))
        result = obj.finish_transition(observations([2.8, 0.1]), reward, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(result, reward, atol=0, rtol=0)
        obj.check_status()


if __name__ == "__main__":
    unittest.main()
