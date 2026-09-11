import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verify_gpu_optimization_replay import (
    EXACT_FIELDS, NUMERIC_FIELDS, compare, exact_difference, pairwise_exact,
    requested_actions, replay_configs,
)


class OptimizationReplayTests(unittest.TestCase):
    def test_deferred_intervention_preserves_other_configured_options(self):
        @dataclass
        class Config:
            combat_library: Path = Path("original.so")
            conditional_reset_forward: bool = True
            fused_combat_library: Path = Path("fused.so")
            defer_substep_combat_observations: bool = False

        source = Config()
        original, candidate = replay_configs(
            source, conditional_reset_forward=False, deferred_combat_library=Path("new.so"))
        self.assertEqual(original.combat_library, Path("new.so"))
        self.assertEqual(candidate.combat_library, original.combat_library)
        self.assertTrue(original.conditional_reset_forward)
        self.assertTrue(candidate.conditional_reset_forward)
        self.assertEqual(original.fused_combat_library, source.fused_combat_library)
        self.assertEqual(candidate.fused_combat_library, source.fused_combat_library)
        self.assertFalse(original.defer_substep_combat_observations)
        self.assertTrue(candidate.defer_substep_combat_observations)
        self.assertEqual(source.combat_library, Path("original.so"))
        legacy_original, legacy_candidate = replay_configs(source, conditional_reset_forward=True)
        self.assertFalse(legacy_original.conditional_reset_forward)
        self.assertIsNone(legacy_original.fused_combat_library)
        self.assertTrue(legacy_candidate.conditional_reset_forward)
        self.assertFalse(legacy_candidate.defer_substep_combat_observations)

    def test_script_has_eight_fighters_and_declared_yaw_requests(self):
        actions = requested_actions(256)
        self.assertEqual(actions.shape, (256, 8))
        self.assertEqual(actions.dtype, np.int64)
        self.assertTrue(np.all(actions[:, :2] == 1))
        self.assertEqual(actions[47:49, 2].tolist(), [6, 17])
        self.assertEqual(actions[155:157, 2].tolist(), [7, 18])
        self.assertTrue(np.all((actions >= 0) & (actions < 33)))

    def test_pairwise_control_identifies_candidate_matching_other_original(self):
        a = np.zeros((2, 3, 2), dtype=np.uint32)
        b = a.copy()
        b[1, 2, 0] = 2
        pairs = pairwise_exact(a, b, b.copy())
        self.assertEqual(pairs["original_repeat"]["different_elements"], 1)
        self.assertEqual(pairs["gated_vs_original"]["different_elements"], 1)
        self.assertEqual(pairs["original_repeat"]["first_index_repeat_tick_and_field"], [1, 2, 0])
        self.assertTrue(pairs["gated_vs_original_b"]["exact"])

    def test_control_diagnostic_does_not_relax_existing_failure_criterion(self):
        raw = {f"{branch}/{repeat}/{field}": np.zeros((3, 2), dtype=np.float32)
               for branch in ("original_a", "original_b", "gated")
               for repeat in range(2) for field in (*EXACT_FIELDS, *NUMERIC_FIELDS)}
        raw["original_b/0/fall_phase"][1, 0] = 1
        raw["gated/0/fall_phase"][1, 0] = 1
        report, failures = compare(raw, 2)
        self.assertEqual(failures, ["exact/fall_phase"])
        self.assertTrue(report["exact"]["fall_phase"]["gated_vs_original_b"]["exact"])

    def test_nonfinite_values_are_not_exact(self):
        values = np.array([np.nan])
        self.assertFalse(exact_difference(values, values)["exact"])


if __name__ == "__main__":
    unittest.main()
