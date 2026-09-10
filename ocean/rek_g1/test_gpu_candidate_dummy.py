from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gpu_candidate_dummy import (
    DUMMY_LABEL, GpuCandidateApproachDummy, GpuCandidateDummyDuel,
)
from human_eval_server import (
    CandidateApproachDummy, ConservativeActionPlanner, HumanEvalFailure,
)


def _rows(count):
    result = np.zeros((count, 223), dtype=np.float32)
    result[:, 3] = 1.0
    result[:, 86] = 1.0
    return result


def _reference(observations, masks, offsets):
    actions = []
    next_offsets = []
    for row, mask, offset in zip(observations, masks, offsets):
        dummy = CandidateApproachDummy()
        dummy.next_move_offset = int(offset)
        preferred = dummy.preferred_action(row)
        action = ConservativeActionPlanner().select(preferred, 1, row, mask).category
        dummy.note_selected(action)
        actions.append(action)
        next_offsets.append(dummy.next_move_offset)
    return np.asarray(actions), np.asarray(next_offsets)


class DummyDifferentialTests(unittest.TestCase):
    DEVICE = "cpu"

    def test_label_and_random_geometry_masks_state(self):
        self.assertEqual(DUMMY_LABEL, CandidateApproachDummy.LABEL)
        rng = np.random.default_rng(319)
        count = 8192
        rows = _rows(count)
        rows[:, :3] = rng.uniform(-100, 100, (count, 3))
        rows[:, 86:89] = rows[:, :3] + rng.uniform(-3, 3, (count, 3))
        rows[:, 3:7] = rng.normal(size=(count, 4))
        rows[:, 79] = rng.choice([0, 0, 0, 1, 2, 3], count)
        masks = rng.integers(0, 2, (count, 33), dtype=np.uint8)
        masks[:, 0] = 1
        offsets = rng.integers(0, 16, count)
        expected, next_offsets = _reference(rows, masks, offsets)
        dummy = GpuCandidateApproachDummy(count, self.DEVICE)
        dummy.next_move_offset.copy_(torch.as_tensor(offsets, device=self.DEVICE))
        actions = dummy.select(torch.as_tensor(rows, device=self.DEVICE),
                               torch.as_tensor(masks, device=self.DEVICE))
        np.testing.assert_array_equal(actions.cpu().numpy(), expected)
        np.testing.assert_array_equal(dummy.next_move_offset.cpu().numpy(), next_offsets)
        dummy.check_status()

    def test_distance_bearing_and_quaternion_boundaries(self):
        fixtures = []
        for distance in (0.0, 0.72, 1.25):
            value = np.float32(distance)
            for x in (np.nextafter(value, np.float32(-np.inf)), value,
                      np.nextafter(value, np.float32(np.inf))):
                row = _rows(1)[0]
                row[86] = x
                fixtures.append(row)
        for yaw in (-math.pi, -2.0, -0.16, 0.0, 0.16, 2.0, math.pi):
            for bearing in (-math.pi, -0.16000002, -0.16, -0.15999998,
                            0.15999998, 0.16, 0.16000002, math.pi):
                for scale in (0.01, 1.0, 10.0):
                    row = _rows(1)[0]
                    row[3] = math.cos(yaw/2)*scale
                    row[6] = math.sin(yaw/2)*scale
                    row[86] = math.cos(yaw+bearing)
                    row[87] = math.sin(yaw+bearing)
                    fixtures.append(row)
        rows = np.asarray(fixtures)
        masks = np.ones((len(rows), 33), dtype=np.uint8)
        offsets = np.arange(len(rows)) % 16
        expected, next_offsets = _reference(rows, masks, offsets)
        dummy = GpuCandidateApproachDummy(len(rows), self.DEVICE)
        dummy.next_move_offset.copy_(torch.as_tensor(offsets, device=self.DEVICE))
        actions = dummy.select(torch.as_tensor(rows, device=self.DEVICE),
                               torch.as_tensor(masks, device=self.DEVICE))
        np.testing.assert_array_equal(actions.cpu().numpy(), expected)
        np.testing.assert_array_equal(dummy.next_move_offset.cpu().numpy(), next_offsets)
        dummy.check_status()

    def test_cycles_masks_and_per_arena_terminal_reset(self):
        count = 17
        rows = _rows(count)
        masks = np.ones((count, 33), dtype=np.uint8)
        dummy = GpuCandidateApproachDummy(count, self.DEVICE)
        offsets = np.zeros(count, dtype=np.int64)
        for tick in range(80):
            masks.fill(1)
            masks[(tick % count), 16:32] = 0
            masks[(tick + 1) % count, 1:] = 0
            masks[(tick + 2) % count, :] = 0
            masks[(tick + 2) % count, 1] = 1
            expected, offsets = _reference(rows, masks, offsets)
            result = dummy.select(torch.as_tensor(rows, device=self.DEVICE),
                                  torch.as_tensor(masks, device=self.DEVICE))
            np.testing.assert_array_equal(result.cpu().numpy(), expected)
            terminal = np.arange(count) == tick % count
            offsets[terminal] = 0
            dummy.reset(torch.as_tensor(terminal, device=self.DEVICE))
            np.testing.assert_array_equal(dummy.next_move_offset.cpu().numpy(), offsets)
        dummy.check_status()

    def test_invalid_input_latches_instead_of_host_read_in_select(self):
        rows = _rows(1)
        masks = np.zeros((1, 33), dtype=np.uint8)
        with self.assertRaises(HumanEvalFailure):
            _reference(rows, masks, [0])
        dummy = GpuCandidateApproachDummy(1, self.DEVICE)
        dummy.select(torch.as_tensor(rows, device=self.DEVICE),
                     torch.as_tensor(masks, device=self.DEVICE))
        with self.assertRaises(RuntimeError):
            dummy.check_status()
        dummy.reset()
        dummy.check_status()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CudaDummyDifferentialTests(DummyDifferentialTests):
    DEVICE = "cuda"

    def test_graph_replays_match_reference_state(self):
        rows = _rows(32)
        masks = np.ones((32, 33), dtype=np.uint8)
        gpu_rows = torch.as_tensor(rows, device=self.DEVICE)
        gpu_masks = torch.as_tensor(masks, device=self.DEVICE)
        dummy = GpuCandidateApproachDummy(32, self.DEVICE)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            dummy.select(gpu_rows, gpu_masks)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                dummy.select(gpu_rows, gpu_masks)
        stream.synchronize()
        dummy.reset()
        offsets = np.zeros(32, dtype=np.int64)
        for tick in range(40):
            rows[tick % 32, 86] = 1.5 if tick % 2 else 1.0
            gpu_rows.copy_(torch.as_tensor(rows, device=self.DEVICE))
            expected, offsets = _reference(rows, masks, offsets)
            graph.replay()
            np.testing.assert_array_equal(dummy.actions.cpu().numpy(), expected)
            np.testing.assert_array_equal(dummy.next_move_offset.cpu().numpy(), offsets)
        dummy.check_status()


class _FullCudaDuelFixture:
    """Controlled pair fixture, not simulated fight evidence."""

    def __init__(self, rows=16):
        self.rows = rows
        self.actions = torch.ones(rows, device="cuda")
        self.observations = torch.as_tensor(_rows(rows), device="cuda")
        self.rewards = torch.zeros(rows, device="cuda")
        self.terminals = torch.zeros_like(self.rewards)
        self.action_mask = torch.ones((rows, 33), dtype=torch.uint8, device="cuda")
        self.scheduler = SimpleNamespace(move_start_edge=torch.zeros(rows, device="cuda"))
        self.combat = SimpleNamespace(tick_score_delta=torch.zeros(rows, device="cuda"))
        self.steps = torch.zeros((), dtype=torch.int32, device="cuda")

    def reset(self):
        self.actions.fill_(1)
        self.rewards.zero_()
        self.terminals.zero_()
        self.action_mask.fill_(1)
        self.steps.zero_()

    def step(self, actions):
        self.actions.copy_(actions[:, 0])
        self.steps.add_(1)
        self.observations[:, 10].copy_(self.actions)
        self.rewards.copy_(torch.arange(self.rows, device="cuda", dtype=torch.float32))
        self.terminals.copy_((self.steps.remainder(4) == 0).expand(self.rows))

    def check_status(self):
        pass

    def close(self):
        pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LearnerOnlyBoundaryTests(unittest.TestCase):
    def test_rows_stable_contiguous_pair_actions_and_metrics(self):
        from gpu_puffer_env import CudaTensorEnvAdapter

        raw = _FullCudaDuelFixture()
        env = GpuCandidateDummyDuel(raw)
        adapter = CudaTensorEnvAdapter(env, (33,), metric_plugins=(env.metric_plugin,))
        pointers = [x.data_ptr() for x in (env.observations, env.rewards,
                                          env.terminals, env.action_mask)]
        self.assertEqual(adapter.total_agents, 8)
        actions = torch.full((8, 1), 2, dtype=torch.int32, device="cuda")
        for tick in range(4):
            adapter.step(actions)
            self.assertEqual(raw.actions[0::2].cpu().tolist(), [2]*8)
            self.assertEqual(raw.actions[1::2].cpu().tolist(), [16+tick]*8)
        self.assertEqual(env.dummy.next_move_offset.cpu().tolist(), [0]*8)
        self.assertEqual(env.rewards.cpu().tolist(), list(range(0, 16, 2)))
        self.assertEqual(env.observations[:, 10].cpu().tolist(), [2]*8)
        self.assertEqual(adapter.log(clear_metrics=False)["n"], 8.0)
        self.assertEqual(adapter.log()["n"], 8.0)
        self.assertEqual(adapter.log()["n"], 0.0)
        adapter.reset()
        self.assertEqual(pointers, [x.data_ptr() for x in (
            env.observations, env.rewards, env.terminals, env.action_mask)])

    def test_native_trainer_only_contains_learner_rows(self):
        from gpu_native_puffer import NativeExternalGpuPuffer
        from gpu_puffer_env import CudaTensorEnvAdapter
        from test_gpu_native_puffer import _args

        env = GpuCandidateDummyDuel(_FullCudaDuelFixture())
        adapter = CudaTensorEnvAdapter(env, (33,), metric_plugins=(env.metric_plugin,))
        trainer = NativeExternalGpuPuffer(_args(), adapter)
        try:
            self.assertEqual(trainer.pufferl.total_agents, 8)
            self.assertEqual(trainer.pufferl.native_env_count, 0)
            self.assertFalse(trainer.pufferl.has_env_threads)
            trainer.rollouts()
            trainer.train()
            self.assertEqual(trainer.global_step, 32)
            self.assertEqual(trainer.log()["env"]["n"], 8.0)
        finally:
            trainer.close()


if __name__ == "__main__":
    unittest.main()
