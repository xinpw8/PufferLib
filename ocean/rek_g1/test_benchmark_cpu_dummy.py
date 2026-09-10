"""CPU fixtures for the explicitly CPU-resident baseline adapter."""

import unittest

import numpy as np
import torch

from benchmark_cpu_dummy import CpuBaselineUploadBoundary
from gpu_metrics import RekG1GpuMetricCollector
from human_eval_server import CandidateApproachDummy, ConservativeActionPlanner


class FakeBoundary:
    def __init__(self):
        self.observations = np.zeros((8, 223), dtype=np.float32)
        self.rewards = np.zeros(8, dtype=np.float32)
        self.terminals = np.zeros(8, dtype=np.float32)
        self.action_masks = np.ones((8, 33), dtype=np.uint8)
        self.terminal_next = False
        self.closed = False
        self.reset()

    def reset(self):
        self.observations.fill(0)
        self.observations[:, 3] = 1
        self.observations[:, 86] = 1
        self.rewards.fill(0)
        self.terminals.fill(0)

    def step(self, actions):
        self.actions = actions.copy()
        self.rewards[:] = np.arange(8)
        self.terminals[:] = float(self.terminal_next)

    def close(self):
        self.closed = True


def fixture():
    env = CpuBaselineUploadBoundary.__new__(CpuBaselineUploadBoundary)
    env.boundary = FakeBoundary()
    env.learning_agents = 4
    env.device = torch.device("cpu")
    env.observations = torch.empty((4, 223))
    env.rewards = torch.empty(4)
    env.terminals = torch.empty(4)
    env.action_mask = torch.empty((4, 33), dtype=torch.uint8)
    env.full_actions = np.ones((8, 1), dtype=np.float32)
    env.dummies = [CandidateApproachDummy() for _ in range(4)]
    env.planners = [ConservativeActionPlanner() for _ in range(4)]
    env.metrics = RekG1GpuMetricCollector(8, "cpu")
    env.behavior = None
    env.clear_timing()
    env.reset()
    return env


class CpuFixedDummyTests(unittest.TestCase):
    def test_only_even_rows_receive_learner_actions(self):
        env = fixture()
        learned = torch.tensor([[8], [1], [17], [0]], dtype=torch.int32)
        env.step(learned)
        np.testing.assert_array_equal(env.boundary.actions[0::2], learned.numpy())
        np.testing.assert_array_equal(env.boundary.actions[1::2, 0], [16] * 4)
        self.assertEqual([dummy.next_move_offset for dummy in env.dummies], [1] * 4)
        self.assertEqual(env.rewards.tolist(), [0, 2, 4, 6])
        self.assertEqual(env.control_ticks, 1)

    def test_dummy_mask_rejects_move_without_advancing_cycle(self):
        env = fixture()
        env.boundary.action_masks[1::2, 16] = 0
        env.step(torch.ones((4, 1), dtype=torch.int32))
        np.testing.assert_array_equal(env.boundary.actions[1::2, 0], [1] * 4)
        self.assertEqual([dummy.next_move_offset for dummy in env.dummies], [0] * 4)

    def test_terminal_resets_each_dummy(self):
        env = fixture()
        env.step(torch.ones((4, 1), dtype=torch.int32))
        env.boundary.terminal_next = True
        env.step(torch.ones((4, 1), dtype=torch.int32))
        self.assertEqual([dummy.next_move_offset for dummy in env.dummies], [0] * 4)
        self.assertEqual(env.terminals.tolist(), [1] * 4)

    def test_exposed_buffers_stay_stable(self):
        env = fixture()
        pointers = tuple(value.data_ptr() for value in (
            env.observations, env.rewards, env.terminals, env.action_mask))
        env.step(torch.ones((4, 1), dtype=torch.int32))
        env.reset()
        self.assertEqual(pointers, tuple(value.data_ptr() for value in (
            env.observations, env.rewards, env.terminals, env.action_mask)))
        env.close()
        self.assertTrue(env.boundary.closed)


if __name__ == "__main__":
    unittest.main()
