"""Host-only formatting of already-collected training progress."""
from copy import deepcopy
import json
import unittest

from train_gpu_duel import training_progress


class TrainingProgressTests(unittest.TestCase):
    def test_available_metrics_keep_original_progress_fields(self):
        source = {"env": {"n": 10.0, "behavior": {
            "learner_round_wins": 7, "learner_round_losses": 2, "round_ties": 1,
            "learner_round_win_percent": 70.0, "learner_points_per_completed_round": 11.2,
            "opponent_points_per_completed_round": 7.4,
            "facing": {"percent": 34.0, "attack_requested_facing_percent": 51.0}}},
            "loss": {"kl": .012, "entropy": .9}}
        before = deepcopy(source)
        report = training_progress(epoch=5, agent_steps=10240, elapsed=2.0, latest_log=source)
        self.assertEqual(report["measured_cumulative_agent_steps_per_second"], 5120)
        self.assertEqual(report["completed_rounds"], 10)
        self.assertEqual(report["online"]["wins"], 7)
        self.assertEqual(report["online"]["losses"], 2)
        self.assertEqual(report["online"]["ties"], 1)
        self.assertEqual(report["online"]["attack_facing_percent"], 51)
        self.assertEqual(report["native_loss"], {"kl": .012, "entropy": .9})
        self.assertEqual(report["metrics_scope"], "cumulative_online_changing_policy")
        self.assertEqual(source, before)

    def test_missing_behavior_or_loss_stays_null(self):
        report = training_progress(epoch=1, agent_steps=64, elapsed=1.0, latest_log={"env": {"n": 0}})
        self.assertEqual(report["completed_rounds"], 0)
        self.assertTrue(all(value is None for value in report["online"].values()))
        self.assertEqual(report["native_loss"], {"kl": None, "entropy": None})

    def test_nonfinite_and_tensorlike_values_are_not_read_or_serialized(self):
        class ForbiddenRead:
            def __float__(self):
                raise AssertionError("must not coerce a potential device value")

            def item(self):
                raise AssertionError("must not read a device value")

        report = training_progress(epoch=1, agent_steps=64, elapsed=0.0,
                                   latest_log={"env": {"n": ForbiddenRead()},
                                               "loss": {"kl": float("nan"), "entropy": float("inf")}})
        self.assertIsNone(report["completed_rounds"])
        self.assertIsNone(report["measured_cumulative_agent_steps_per_second"])
        self.assertEqual(report["native_loss"], {"kl": None, "entropy": None})
        self.assertIsInstance(json.dumps(report, allow_nan=False), str)


if __name__ == "__main__":
    unittest.main()
