"""CPU-only scripted-baseline policy and separated outcome accounting tests."""

import ast
import math
from pathlib import Path
import unittest

import torch

from gpu_scripted_baseline import (
    COMPOSER_BUSY, FIGHT_PHASE, GpuScriptedPolicies, KICK_CATEGORIES,
    LEFT_FRONT_KICK, OPPONENT_FALL_PHASE, PerArenaDiagnostics, SELF_FALL_PHASE,
    STRATEGIES, Strategy, bearing_and_range, summarize,
)


def fixture(arenas, *, distance=1.0, bearing=0.0):
    obs = torch.zeros((arenas, 223), dtype=torch.float32)
    obs[:, 3] = 1
    obs[:, 86] = distance * math.cos(bearing)
    obs[:, 87] = distance * math.sin(bearing)
    obs[:, 89] = 1
    obs[:, FIGHT_PHASE] = 2
    mask = torch.ones((arenas, 33), dtype=torch.uint8)
    return obs, mask


class ScriptedPolicyTests(unittest.TestCase):
    def test_bindings_match_evaluator_metadata(self):
        module = ast.parse(Path(__file__).with_name("human_eval_server.py").read_text())
        declaration = next(node for node in module.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "MOVE_METADATA" for target in node.targets))
        identities = {row["category"]: row["identity"] for row in ast.literal_eval(declaration.value)}
        self.assertEqual([identities[value] for value in KICK_CATEGORIES],
                         ["left_side_kick", "left_front_kick", "right_side_kick"])

    def test_all_strategies_and_no_punches(self):
        policy = GpuScriptedPolicies(8, "cpu")
        obs, masks = fixture(8)
        self.assertEqual(policy.select(obs, masks)[:, 0].tolist(), [1, 1, 17, 16, 18, 16, 2, 17])
        policy.check_status()
        self.assertEqual([row["arenas"] for row in policy.assignments()], [[i] for i in range(8)])

    def test_turn_sign_and_world_translation_invariance(self):
        policy = GpuScriptedPolicies(2, "cpu", (Strategy("front", LEFT_FRONT_KICK),))
        obs, masks = fixture(2, bearing=0.5)
        obs[1, 87] *= -1
        expected = policy.select(obs, masks).clone()
        self.assertEqual(expected[:, 0].tolist(), [6, 7])
        obs[:, :2] += torch.tensor([24., -31.])
        obs[:, 86:88] += torch.tensor([24., -31.])
        self.assertTrue(torch.equal(policy.select(obs, masks), expected))

    def test_quaternion_layout_and_sign_invariance(self):
        obs, _ = fixture(1)
        obs[:, 3] = math.cos(math.pi / 4)
        obs[:, 6] = math.sin(math.pi / 4)
        bearing, distance, norm = bearing_and_range(obs)
        self.assertAlmostEqual(bearing.item(), -math.pi / 2, places=6)
        obs[:, 3:7] *= -1
        self.assertTrue(torch.equal(bearing_and_range(obs)[0], bearing))
        self.assertAlmostEqual(distance.item(), 1)
        self.assertAlmostEqual(norm.item(), 1, places=6)

    def test_motion_holds_release_masked_attack_and_native_mask_is_authority(self):
        policy = GpuScriptedPolicies(1, "cpu", (Strategy("front", LEFT_FRONT_KICK),))
        obs, masks = fixture(1, distance=2)
        for _ in range(10):
            self.assertEqual(policy.select(obs, masks).item(), 2)
        obs[:, 86] = 1
        masks[:, 17] = 0
        self.assertEqual(policy.select(obs, masks).item(), 1)
        masks[:, 17] = 1
        self.assertEqual(policy.select(obs, masks).item(), 17)
        obs[:, COMPOSER_BUSY] = 1
        self.assertEqual(policy.select(obs, masks).item(), 17)
        masks.zero_()
        masks[:, 0] = 1
        self.assertEqual(policy.select(obs, masks).item(), 0)
        masks[:, 1] = 1
        self.assertEqual(policy.select(obs, masks).item(), 1)
        policy.check_status()

    def test_combined_turn_only_when_approaching_front_hemisphere(self):
        strategy = Strategy("combined", LEFT_FRONT_KICK, combined_approach_yaw=True)
        policy = GpuScriptedPolicies(1, "cpu", (strategy,))
        for angle, distance, expected in ((0.5, 2, 8), (-0.5, 2, 9), (2, 2, 6), (0.5, 1, 6)):
            obs, masks = fixture(1, distance=distance, bearing=angle)
            self.assertEqual(policy.select(obs, masks).item(), expected)

    def test_fall_and_inactive_are_neutral(self):
        policy = GpuScriptedPolicies(3, "cpu", (Strategy("front", LEFT_FRONT_KICK),))
        obs, masks = fixture(3)
        obs[0, SELF_FALL_PHASE] = 1
        obs[1, OPPONENT_FALL_PHASE] = 1
        obs[2, FIGHT_PHASE] = 1
        self.assertEqual(policy.select(obs, masks)[:, 0].tolist(), [1, 1, 1])

    def test_cycle_only_advances_legal_kicks_and_terminal_clears_it(self):
        policy = GpuScriptedPolicies(1, "cpu", (Strategy("cycle", 16, cycle_kicks=True),))
        obs, masks = fixture(1)
        masks[:, 16] = 0
        self.assertEqual(policy.select(obs, masks).item(), 1)
        masks[:, 16] = 1
        self.assertEqual([policy.select(obs, masks).item() for _ in range(4)], [16, 17, 18, 16])
        policy.reset(torch.ones(1))
        self.assertEqual(policy.select(obs, masks).item(), 16)

    def test_invalid_mask_or_nonfinite_observation_latches_failure(self):
        for kind in ("empty", "nonbinary", "nonfinite", "quaternion"):
            with self.subTest(kind=kind):
                policy = GpuScriptedPolicies(1, "cpu", (Strategy("front", LEFT_FRONT_KICK),))
                obs, masks = fixture(1)
                if kind == "empty":
                    masks.zero_()
                elif kind == "nonbinary":
                    masks[:, 4] = 2
                elif kind == "nonfinite":
                    obs[:, 100] = float("nan")
                else:
                    obs[:, 3:7] = 0
                policy.select(obs, masks)
                with self.assertRaises(RuntimeError):
                    policy.check_status()

    def test_invalid_policy_configuration_fails(self):
        for args in ((4, STRATEGIES), (1, (Strategy("punch", 20),)),
                     (1, (Strategy("bad_range", 17, minimum_range_m=2),))):
            with self.assertRaises(ValueError):
                GpuScriptedPolicies(args[0], "cpu", args[1])


class BaselineAccountingTests(unittest.TestCase):
    def test_per_arena_counters_and_completed_outcomes_stay_separate(self):
        obs, _ = fixture(4)
        obs[0::2, 217] = torch.tensor([2., 5.])
        obs[0::2, 218] = torch.tensor([1., 3.])
        obs[0::2, 222] = torch.tensor([1., 0.])
        counters = PerArenaDiagnostics(2, "cpu")
        counters.update(obs, torch.tensor([1., 1., 0., 0.]), torch.tensor([[17], [1]]),
                        torch.tensor([1, 0, 0, 0]), torch.tensor([1., -1., 2., -2.]))
        rows = counters.snapshot()
        self.assertEqual(rows[0]["learner_points_all_steps"], 2)
        self.assertEqual(rows[1]["learner_points_all_steps"], 5)
        self.assertEqual(rows[0]["arena_scored_hits_all_steps"], 1)
        self.assertEqual(rows[1]["arena_scored_hits_all_steps"], 0)
        self.assertEqual(rows[0]["native_move_starts"], 1)
        self.assertEqual(rows[0]["action_category_counts"][17], 1)
        assignments = [{"name": "a", "arenas": [0]}, {"name": "b", "arenas": [1]}]
        events = [{"arena": 0, "outcome": "learner_win", "learner_points": 9,
                   "opponent_points": 4, "learner_falls": 0, "opponent_falls": 1,
                   "arena_scored_hits": 3}]
        reports = summarize(assignments, events, rows, [3, 7])
        self.assertEqual(reports[0]["win_percent"], 100)
        self.assertEqual(reports[0]["learner_points_per_completed_round"], 9)
        self.assertEqual(reports[0]["action_mask_substitutions"], 3)
        self.assertIsNone(reports[1]["win_percent"])
        self.assertEqual(reports[1]["completed_rounds"], 0)
        self.assertIsNone(reports[0]["learner_scored_hits"])


if __name__ == "__main__":
    unittest.main()
