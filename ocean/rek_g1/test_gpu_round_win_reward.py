"""CPU contract, complete-return identity and recorded outcome classification."""
import json
import math
from pathlib import Path
import unittest

import torch

from gpu_round_win_reward import (
    GpuRoundWinReward, RoundWinRewardConfig, resolve_round_win_reward,
    SIDE, OWN_POINTS, OPPONENT_POINTS, ROUND_RESULT, ROUND_WINNER,
)


def observations(own, opponent=None, *, sides=None, results=None, winners=None):
    n = len(own)
    values = torch.zeros((n, 223), dtype=torch.float32)
    values[:, OWN_POINTS] = torch.as_tensor(own)
    values[:, OPPONENT_POINTS] = torch.as_tensor(opponent if opponent is not None else [0]*n)
    values[:, SIDE] = torch.tensor(sides if sides is not None else [0]*n)
    values[:, ROUND_RESULT] = torch.tensor(results if results is not None else [0]*n)
    values[:, ROUND_WINNER] = torch.tensor(winners if winners is not None else [-1]*n)
    return values


def rewarder(rows=1, scale=0.5):
    return GpuRoundWinReward(rows, "cpu", RoundWinRewardConfig(1, scale, 5), allow_cpu_for_tests=True)


class RoundWinRewardTests(unittest.TestCase):
    def test_configuration_requires_undiscounted_unclipped_native_reward(self):
        obj = RoundWinRewardConfig(1, 0.5, 5)
        obj.validate_training_discount(1, reward_clip=0)
        for gamma, scale, points in ((0.999, .5, 5), (0, .5, 5), (math.nan, .5, 5),
                                     (1, -.1, 5), (1, math.inf, 5), (1, .5, 0), (1, .5, math.nan)):
            with self.assertRaises(ValueError): RoundWinRewardConfig(gamma, scale, points)
        for gamma, clip in ((.999, 0), (1, 1), (1, math.nan)):
            with self.assertRaises(ValueError): obj.validate_training_discount(gamma, reward_clip=clip)
        with self.assertRaises(ValueError): GpuRoundWinReward(1, "cpu", obj)
        resolved, metadata = resolve_round_win_reward({"train": {"gamma": 1}},
            margin_potential_scale=.5, margin_points=5, reward_clip=0)
        self.assertEqual(resolved, obj)
        self.assertFalse(metadata["default_activation"])
        self.assertFalse(metadata["evaluation_changed"])
        with self.assertRaises(ValueError): resolve_round_win_reward({"train": {"gamma": .999}},
            margin_potential_scale=.5, margin_points=5, reward_clip=0)

    def test_native_outcome_classes_and_published_side_identity(self):
        obj = rewarder(8)
        initial = observations([0]*8, sides=[0, 0, 0, 0, 0, 1, 1, 0])
        final = observations([1, 18, 4, 3, 7, 9, 6, 100], [0, 0, 2, 3, 7, 1, 2, 0],
            sides=[0, 0, 0, 0, 0, 1, 1, 0], results=[1, 1, 2, 3, 4, 2, 1, 0],
            winners=[0, 0, 1, -1, -1, 1, 0, -1])
        obj.begin_transition(initial, torch.zeros(8))
        obj.finish_transition(final, torch.arange(8).float(), torch.tensor([1.]*7+[0.]))
        torch.testing.assert_close(obj.terminal_win, torch.tensor([1., 1, 0, 0, 0, 1, 0, 0], dtype=torch.float64))
        torch.testing.assert_close(obj.rewards[:7], torch.tensor([1., 1, 0, 0, 0, 1, 0]), atol=0, rtol=0)
        self.assertAlmostEqual(obj.rewards[-1].item(), .5)
        obj.check_status()

    def test_complete_returns_equal_win_indicator_across_durations_and_margins(self):
        generator = torch.Generator().manual_seed(7311)
        for nonzero_initial in (False, True):
            for length in (1, 2, 7, 31, 128):
                obj = rewarder(4)
                points = torch.randint(0, 30, (length+1, 4, 2), generator=generator)
                if not nonzero_initial: points[0].zero_()
                total = torch.zeros(4, dtype=torch.float64)
                exact = torch.zeros_like(total)
                initial = None
                for tick in range(length):
                    obj.begin_transition(observations(points[tick, :, 0], points[tick, :, 1]), torch.zeros(4))
                    if initial is None: initial = obj.current_potential.clone()
                    final = tick == length-1
                    nxt = observations(points[tick+1, :, 0], points[tick+1, :, 1],
                        results=[1, 2, 1, 3] if final else None,
                        winners=[0, 0, 1, -1] if final else None)
                    total += obj.finish_transition(nxt, torch.randn(4, generator=generator), torch.full((4,), float(final))).double()
                    exact += obj.terminal_win + obj.shaping_reward
                expected = torch.tensor([1., 1, 0, 0], dtype=torch.float64)-initial
                torch.testing.assert_close(exact, expected, atol=2e-14, rtol=0)
                torch.testing.assert_close(total, expected, atol=2e-6, rtol=0)
                obj.check_status()

    def test_nonterminal_horizon_has_bootstrap_potential(self):
        obj = rewarder()
        total = 0.0
        for tick, (before, after) in enumerate(((2, 7), (7, 4))):
            obj.begin_transition(observations([before]), torch.zeros(1))
            if tick == 0: initial = obj.current_potential.item()
            obj.finish_transition(observations([after]), torch.zeros(1), torch.zeros(1))
            total += obj.shaping_reward.item()
        final = obj.next_potential.item()
        self.assertAlmostEqual(total, final-initial, places=14)
        self.assertAlmostEqual(total + (.7-final), .7-initial, places=14)
        obj.check_status()

    def test_both_terminal_masks_prevent_previous_round_margin_leak(self):
        obj = rewarder(2)
        terminal = observations([20, 0], [0, 20], results=[1, 1], winners=[0, 1])
        obj.begin_transition(observations([5, 5]), torch.zeros(2))
        obj.finish_transition(terminal, torch.zeros(2), torch.ones(2))
        self.assertEqual(obj.next_potential.abs().max().item(), 0)
        obj.begin_transition(terminal, torch.ones(2))
        self.assertEqual(obj.current_potential.abs().max().item(), 0)
        after_reset = obj.finish_transition(observations([0, 0]), torch.zeros(2), torch.zeros(2))
        torch.testing.assert_close(after_reset, torch.zeros(2), atol=0, rtol=0)
        obj.check_status()

    def test_snapshot_preserves_previous_margin_before_mutable_buffer_step(self):
        obj = rewarder()
        raw = observations([5])
        obj.begin_transition(raw, torch.zeros(1))
        expected = .5*math.tanh(1)
        raw[:, OWN_POINTS] = 0
        raw[:, OPPONENT_POINTS] = 5
        value = obj.finish_transition(raw, torch.zeros(1), torch.zeros(1))
        self.assertAlmostEqual(obj.current_potential.item(), expected, places=14)
        self.assertAlmostEqual(value.item(), -2*expected, places=7)
        obj.check_status()

    def test_zero_scale_is_sparse_win_ablation_and_raw_buffers_unchanged(self):
        obj = rewarder(3, scale=0)
        current = observations([0, 4, 7])
        following = observations([2, 100, 1000], [1, 0, 2], results=[1, 1, 0], winners=[0, 1, -1])
        raw, terminal = torch.tensor([-100., 100, 1000]), torch.tensor([1., 1, 0])
        tensors = (current, following, raw, terminal)
        copies = [x.clone() for x in tensors]
        pointer = obj.rewards.data_ptr()
        for _ in range(3):
            obj.begin_transition(current, torch.zeros(3))
            result = obj.finish_transition(following, raw, terminal)
            torch.testing.assert_close(result, torch.tensor([1., 0, 0]), atol=0, rtol=0)
            self.assertEqual(pointer, result.data_ptr())
        for actual, original in zip(tensors, copies): torch.testing.assert_close(actual, original, atol=0, rtol=0)
        obj.check_status()

    def test_invalid_native_flags_scores_ordering_and_reward_latch_errors(self):
        cases = ("terminal", "missing_result", "missing_winner", "tie_winner", "score_negative",
                 "score_fractional", "score_nan", "side", "raw_reward", "double_begin", "no_begin")
        for case in cases:
            with self.subTest(case=case):
                obj = rewarder()
                raw = observations([0], results=[1], winners=[0])
                terminal, reward = torch.ones(1), torch.zeros(1)
                if case == "terminal": terminal.fill_(.5)
                if case == "missing_result": raw[:, ROUND_RESULT] = 0
                if case == "missing_winner": raw[:, ROUND_WINNER] = -1
                if case == "tie_winner": raw[:, ROUND_RESULT] = 3
                if case == "score_negative": raw[:, OWN_POINTS] = -1
                if case == "score_fractional": raw[:, OPPONENT_POINTS] = .5
                if case == "score_nan": raw[:, OWN_POINTS] = math.nan
                if case == "side": raw[:, SIDE] = 2
                if case == "raw_reward": reward.fill_(math.nan)
                if case != "no_begin": obj.begin_transition(raw, terminal)
                if case == "double_begin": obj.begin_transition(raw, terminal)
                obj.finish_transition(raw, reward, terminal)
                with self.assertRaises(RuntimeError): obj.check_status()
                obj.reset()
                obj.check_status()
        with self.assertRaises(ValueError): obj.begin_transition(torch.zeros((1, 222)), terminal)
        with self.assertRaises(ValueError): obj.begin_transition(raw.double(), terminal)
        with self.assertRaises(ValueError): obj.finish_transition(raw, obj.rewards, terminal)

    def test_recorded_frozen_round_classification_from_published_native_flags(self):
        root = Path("C:/rekagent/evidence/rek-training-opt-20260911")
        paths = [root/f"eval-{name}256-ticks6400-v1/evaluations/eval-{name}256-ticks6400-v1/rounds.jsonl"
                 for name in ("initial", "trained", "corrected")]
        if not all(path.is_file() for path in paths): self.skipTest("local recorded frozen round journals unavailable")
        rounds = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
        obj = rewarder(len(rounds), scale=0)
        final = observations([r["learner_points"] for r in rounds], [r["opponent_points"] for r in rounds],
            results=[r["result"] for r in rounds], winners=[r["winner_side"] for r in rounds])
        obj.begin_transition(observations([0]*len(rounds)), torch.zeros(len(rounds)))
        result = obj.finish_transition(final, torch.zeros(len(rounds)), torch.ones(len(rounds)))
        expected = torch.tensor([float(r["outcome"] == "learner_win") for r in rounds])
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        self.assertEqual(len(rounds), 384)
        self.assertEqual(int(result[:128].sum()), 91)
        obj.check_status()


if __name__ == "__main__":
    unittest.main()
