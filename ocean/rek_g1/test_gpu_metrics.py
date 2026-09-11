import unittest

import torch

import gpu_metrics as metrics


def step_tensors(device, arenas=1):
    observations = torch.zeros(
        (arenas * 2, metrics.OBSERVATION_FLOATS),
        dtype=torch.float32,
        device=device,
    )
    terminals = torch.zeros(arenas * 2, dtype=torch.float32, device=device)
    return observations, terminals


def terminal_round(
        observations, terminals, arena, *, result, winner=-1, redo=False,
        ko=False, side0_points=0, side1_points=0, side0_falls=0,
        side1_falls=0, scored_hits=0, attributed_contacts=0):
    row = arena * 2
    observations[row, metrics.ROUND_RESULT] = result
    observations[row, metrics.ROUND_WINNER] = winner
    observations[row, metrics.CURRENT_ROUND_IS_REDO] = float(redo)
    observations[row, metrics.KNOCKOUT_OCCURRED] = float(ko)
    observations[row, metrics.SIDE0_POINTS] = side0_points
    observations[row, metrics.SIDE1_POINTS] = side1_points
    observations[row, metrics.SIDE0_FALLS] = side0_falls
    observations[row, metrics.SIDE1_FALLS] = side1_falls
    observations[row, metrics.TICK_SCORED_HITS] = scored_hits
    observations[row, metrics.TICK_ATTRIBUTED_CONTACTS] = attributed_contacts
    terminals[row:row + 2] = 1.0


class GpuMetricAccountingTests(unittest.TestCase):
    def test_partial_snapshot_preserves_unfinished_accumulator(self):
        collector = metrics.RekG1GpuMetricCollector(2, "cpu")
        observations, terminals = step_tensors("cpu")
        observations[0, metrics.TICK_SCORED_HITS] = 1
        observations[0, metrics.TICK_ATTRIBUTED_CONTACTS] = 2
        collector.update(observations, terminals)

        empty = collector.snapshot(clear=True)
        self.assertEqual(empty["n"], 0.0)
        self.assertEqual(empty["terminal_pair_mismatches"], 0.0)
        for key in metrics.METRIC_KEYS:
            self.assertNotIn(key, empty)

        observations.zero_()
        terminal_round(
            observations,
            terminals,
            0,
            result=metrics.ROUND_WON_BY_KO,
            winner=0,
            ko=True,
            side0_points=7,
            side1_points=2,
            side0_falls=1,
            side1_falls=3,
            scored_hits=2,
            attributed_contacts=5,
        )
        collector.update(observations, terminals)
        result = collector.snapshot()

        self.assertEqual(result["n"], 1.0)
        self.assertEqual(result["side0_round_win_rate"], 1.0)
        self.assertEqual(result["ko_round_rate"], 1.0)
        self.assertEqual(result["side0_points_per_round"], 7.0)
        self.assertEqual(result["side1_points_per_round"], 2.0)
        self.assertEqual(result["side0_falls_per_round"], 1.0)
        self.assertEqual(result["side1_falls_per_round"], 3.0)
        self.assertEqual(result["scored_hits_per_round"], 3.0)
        self.assertEqual(result["attributed_contacts_per_round"], 7.0)
        self.assertAlmostEqual(result["elapsed_seconds_per_round"], 0.04)
        self.assertEqual(result["semantic_steps_per_round"], 2.0)
        self.assertNotEqual(
            result["scored_hits_per_round"],
            result["side0_points_per_round"],
        )

    def test_two_arenas_aggregate_side_outcomes_without_row_double_count(self):
        collector = metrics.RekG1GpuMetricCollector(4, "cpu")
        observations, terminals = step_tensors("cpu", arenas=2)
        terminal_round(
            observations,
            terminals,
            0,
            result=metrics.ROUND_WON_BY_POINTS,
            winner=0,
            side0_points=4,
            side1_points=1,
        )
        terminal_round(
            observations,
            terminals,
            1,
            result=metrics.ROUND_WON_BY_POINTS,
            winner=1,
            redo=True,
            side0_points=2,
            side1_points=6,
        )
        collector.update(observations, terminals)
        result = collector.snapshot()

        self.assertEqual(result["n"], 2.0)
        self.assertEqual(result["side0_round_win_rate"], 0.5)
        self.assertEqual(result["side1_round_win_rate"], 0.5)
        self.assertEqual(result["side0_points_per_round"], 3.0)
        self.assertEqual(result["side1_points_per_round"], 3.5)
        self.assertEqual(result["redo_round_rate"], 0.5)

    def test_tie_redo_result_and_completed_redo_round_are_distinct(self):
        collector = metrics.RekG1GpuMetricCollector(6, "cpu")
        observations, terminals = step_tensors("cpu", arenas=3)
        terminal_round(
            observations, terminals, 0, result=metrics.ROUND_TIE, ko=True)
        terminal_round(
            observations,
            terminals,
            1,
            result=metrics.ROUND_REDO,
            redo=True,
        )
        terminal_round(
            observations,
            terminals,
            2,
            result=metrics.ROUND_WON_BY_POINTS,
            winner=1,
            redo=True,
        )
        collector.update(observations, terminals)
        result = collector.snapshot()

        self.assertAlmostEqual(result["round_tie_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(result["round_redo_result_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(result["redo_round_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(result["side1_round_win_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(result["ko_round_rate"], 1.0 / 3.0)

    def test_clear_does_not_erase_other_arena_in_progress(self):
        collector = metrics.RekG1GpuMetricCollector(4, "cpu")
        observations, terminals = step_tensors("cpu", arenas=2)
        observations[2, metrics.TICK_SCORED_HITS] = 4
        terminal_round(
            observations,
            terminals,
            0,
            result=metrics.ROUND_WON_BY_POINTS,
            winner=0,
        )
        collector.update(observations, terminals)
        self.assertEqual(collector.snapshot(clear=True)["n"], 1.0)

        observations.zero_()
        terminals.zero_()
        terminal_round(
            observations,
            terminals,
            1,
            result=metrics.ROUND_WON_BY_POINTS,
            winner=1,
            scored_hits=2,
        )
        collector.update(observations, terminals)
        result = collector.snapshot(clear=True)
        self.assertEqual(result["n"], 1.0)
        self.assertEqual(result["scored_hits_per_round"], 6.0)
        self.assertEqual(result["semantic_steps_per_round"], 2.0)

    def test_terminal_pair_mismatch_is_deferred_diagnostic(self):
        collector = metrics.RekG1GpuMetricCollector(2, "cpu")
        observations, terminals = step_tensors("cpu")
        terminal_round(
            observations,
            terminals,
            0,
            result=metrics.ROUND_TIE,
        )
        terminals[1] = 0.0
        collector.update(observations, terminals)
        result = collector.snapshot()
        self.assertEqual(result["n"], 1.0)
        self.assertEqual(result["terminal_pair_mismatches"], 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_tensor_accounting_matches_cpu_fixture(self):
        outputs = []
        for device in ("cpu", "cuda"):
            collector = metrics.RekG1GpuMetricCollector(2, device)
            observations, terminals = step_tensors(device)
            observations[0, metrics.TICK_SCORED_HITS] = 1
            collector.update(observations, terminals)
            observations.zero_()
            terminal_round(
                observations,
                terminals,
                0,
                result=metrics.ROUND_WON_BY_POINTS,
                winner=1,
                side0_points=1,
                side1_points=4,
                scored_hits=2,
                attributed_contacts=5,
            )
            collector.update(observations, terminals)
            outputs.append(collector.snapshot())
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
