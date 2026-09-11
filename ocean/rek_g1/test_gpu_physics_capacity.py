"""CUDA fixtures for sticky capacity diagnostics; run explicitly on Spark."""

from types import SimpleNamespace
import unittest

import torch
import warp as wp

from gpu_physics_capacity import GpuPhysicsCapacityMonitor


@unittest.skipUnless(torch.cuda.is_available(), "CUDA diagnostic fixture")
class CapacityTests(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = torch.device("cuda:0")
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        self.nefc = torch.tensor([3, 4], dtype=torch.int32, device=self.device)
        self.nnz = torch.tensor([6, 8], dtype=torch.int32, device=self.device)
        self.nacon = torch.tensor([7], dtype=torch.int32, device=self.device)
        self.ncollision = torch.tensor([8], dtype=torch.int32, device=self.device)
        self.data = SimpleNamespace(
            nworld=2, njmax=4, njmax_nnz=8, naconmax=8,
            nefc=wp.from_torch(self.nefc), nacon=wp.from_torch(self.nacon),
            ncollision=wp.from_torch(self.ncollision),
        )
        self.physics = SimpleNamespace(
            data=self.data, model=SimpleNamespace(is_sparse=True), stream=self.stream,
            wp=wp, qpos=torch.zeros((2, 1), device=self.device),
        )
        self.monitor = GpuPhysicsCapacityMonitor(self.physics)

    def record(self):
        self.monitor.record_constraints(self.physics.model, self.data, wp.from_torch(self.nnz))

    def test_exact_capacity_and_sticky_overflow(self):
        self.record()
        first = self.monitor.snapshot()
        self.assertEqual(first["overflow_flags"], 0)
        self.nefc[0] = 5
        self.nnz[0] = 9
        self.nacon.fill_(9)
        self.ncollision.fill_(10)
        self.record()
        self.nefc.zero_()
        self.nnz.zero_()
        self.nacon.zero_()
        self.ncollision.zero_()
        self.record()
        with self.assertRaisesRegex(RuntimeError, "capacity overflow"):
            self.monitor.snapshot()
        result = self.monitor.snapshot(fail_on_overflow=False)
        self.assertEqual(result["overflow_flags"], 15)
        self.assertEqual(result["high_water"], {"constraints_per_arena": 5, "contacts_aggregate": 9,
                                                "broadphase_pairs_aggregate": 10, "sparse_nnz_per_arena": 9})

    def test_graph_replay_updates_and_reset(self):
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        self.stream = wp.stream_from_torch(capture_stream)
        self.physics.stream = self.stream
        with torch.cuda.stream(capture_stream):
            self.record()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                with wp.ScopedCapture(stream=self.stream, external=True):
                    self.record()
            self.monitor.reset()
            graph.replay()
            graph.replay()
        self.assertEqual(self.monitor.snapshot()["observed_constraint_builds"], 2)

    def test_global_function_restored_on_exception(self):
        original = self.monitor.constraint_module.make_constraint
        with self.assertRaisesRegex(ValueError, "fixture"):
            with self.monitor.capture_hook():
                self.assertIsNot(self.monitor.constraint_module.make_constraint, original)
                raise ValueError("fixture")
        self.assertIs(self.monitor.constraint_module.make_constraint, original)


if __name__ == "__main__":
    unittest.main()
