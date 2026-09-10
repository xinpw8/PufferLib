import math
import unittest
from unittest.mock import patch

import torch

from gpu_behavior_metrics import GpuBehaviorMetricCollector


def fixture(device="cpu"):
    obs = torch.zeros((4, 223), device=device)
    obs[:, 3] = 1
    obs[:, 185] = 2
    obs[0, 86] = 1.0
    obs[2, 86] = -2.0
    terminals = torch.zeros(4, device=device)
    actions = torch.tensor([[17], [0], [6], [1]], device=device)
    starts = torch.tensor([1, 0, 0, 0], device=device)
    delta = torch.tensor([2, 0, 5, 0], device=device)
    return obs, terminals, actions, starts, delta


class BehaviorMetricTests(unittest.TestCase):
    def test_active_geometry_actions_points_are_distinct(self):
        collector = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(0, 2))
        collector.update(*fixture())
        report = collector.snapshot()
        self.assertEqual(report["learner_control_steps"], 2)
        self.assertEqual(report["discrete_move_starts"], 1)
        self.assertEqual(report["action_category_counts"][17], 1)
        self.assertEqual(report["action_category_counts"][6], 1)
        self.assertEqual(report["facing"]["percent"], 50)
        self.assertEqual(report["facing"]["attack_requested_facing_percent"], 100)
        self.assertEqual(report["range"]["mean_m"], 1.5)
        self.assertEqual(report["range"]["counts"], [0, 0, 1, 0, 1, 0, 0])
        self.assertEqual(report["learner_native_points_awarded_all_steps"], 7)
        self.assertIsNone(report["learner_scored_hits"])
        self.assertIsNone(report["learner_round_win_percent"])
        self.assertIsNone(report["learner_points_per_completed_round"])

    def test_side_relative_points_and_absolute_winner(self):
        collector = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(1, 2))
        obs, terminal, action, starts, delta = fixture()
        terminal[:] = 1
        obs[1, 210], obs[2, 210] = 1, 2
        obs[1, 211], obs[2, 211] = 1, 1
        obs[1, 190:192] = torch.tensor([4, 2])
        obs[2, 190:192] = torch.tensor([8, 7])
        obs[1, 222], obs[2, 222] = 1, 2
        collector.update(obs, terminal, action, starts, delta)
        report = collector.snapshot(clear=False)
        self.assertEqual(report["completed_rounds"], 2)
        self.assertEqual(report["learner_round_wins"], 1)
        self.assertEqual(report["learner_round_losses"], 1)
        self.assertEqual(report["learner_points_per_completed_round"], 6)
        self.assertEqual(report["opponent_points_per_completed_round"], 4.5)
        self.assertEqual(report["arena_scored_hits_all_steps"], 3)
        self.assertEqual(report, collector.snapshot(clear=True))
        self.assertEqual(collector.snapshot()["learner_control_steps"], 0)

    def test_invalid_geometry_is_excluded_without_nan_contamination(self):
        collector = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(0, 2))
        obs, terminal, action, starts, delta = fixture()
        obs[0, 86] = float("nan")
        obs[2, 3:7] = 0
        collector.update(obs, terminal, action, starts, delta)
        report = collector.snapshot()
        self.assertEqual(report["range"]["samples"], 1)
        self.assertEqual(report["range"]["mean_m"], 2)
        self.assertEqual(report["facing"]["eligible_samples"], 0)
        self.assertIsNone(report["facing"]["percent"])

    def test_only_active_round_geometry_and_declared_angle_threshold(self):
        collector = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(0, 2))
        obs, terminal, action, starts, delta = fixture()
        obs[0, 3] = math.cos(math.radians(40) / 2)
        obs[0, 6] = math.sin(math.radians(40) / 2)
        obs[2, 185] = 3
        collector.update(obs, terminal, action, starts, delta)
        report = collector.snapshot()
        self.assertEqual(report["range"]["samples"], 1)
        self.assertEqual(report["facing"]["percent"], 0)

    def test_unknown_move_starts_and_observation_native_score_delta(self):
        collector = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(0, 2))
        obs, terminal, action, _, _ = fixture()
        obs[0, 217] = 3
        collector.update(obs, terminal, action)
        report = collector.snapshot()
        self.assertIsNone(report["discrete_move_starts"])
        self.assertEqual(report["learner_native_points_awarded_all_steps"], 3)

    def test_update_never_calls_scalar_or_host_conversion(self):
        collector = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(0, 2))
        inputs = fixture()
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("item")), \
             patch.object(torch.Tensor, "cpu", side_effect=AssertionError("cpu")), \
             patch.object(torch.Tensor, "tolist", side_effect=AssertionError("tolist")):
            collector.update(*inputs)
        self.assertEqual(collector.snapshot()["learner_control_steps"], 2)

    def test_requires_one_row_per_arena(self):
        for rows in ((0,), (0, 1), (0, 4), (-1, 2)):
            with self.assertRaises(ValueError):
                GpuBehaviorMetricCollector(4, "cpu", learner_rows=rows)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_graph_replay_matches_cpu(self):
        cpu = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(0, 2))
        gpu = GpuBehaviorMetricCollector(4, "cuda", learner_rows=(0, 2))
        inputs = fixture("cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gpu.update(*inputs)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                gpu.update(*inputs)
            gpu.reset()
            for _ in range(3):
                graph.replay()
        stream.synchronize()
        for _ in range(3):
            cpu.update(*fixture())
        self.assertEqual(gpu.snapshot(), cpu.snapshot())


if __name__ == "__main__":
    unittest.main()
