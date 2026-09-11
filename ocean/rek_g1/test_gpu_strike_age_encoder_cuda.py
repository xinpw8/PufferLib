"""Bounded CUDA graph fixture; skipped with CUDA_VISIBLE_DEVICES=-1."""
import unittest

import torch

from gpu_policy_observation_encoder import GpuScaledPolarXYPolicyEncoder, GpuStrikeAgeScaledPolarXYPolicyEncoder


@unittest.skipUnless(torch.cuda.is_available(), "CUDA encoder fixture requires an explicit GPU test slot")
class StrikeAgeCudaGraphTests(unittest.TestCase):
    def test_captured_scaling_changes_only_ages_and_preserves_stable_buffers(self):
        rows = 1024
        cpu = torch.randn((rows,223), generator=torch.Generator().manual_seed(9731))
        cpu[:,3:7] = torch.tensor([1.,0.,0.,0.])
        cpu[:,198:200] = torch.linspace(0.,240.,rows)[:,None]
        raw = cpu.cuda()
        old = GpuScaledPolarXYPolicyEncoder(rows, raw.device, initialization="fresh-random")
        new = GpuStrikeAgeScaledPolarXYPolicyEncoder(rows, raw.device, initialization="fresh-random")
        ptr = new.observations.data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            old.encode(raw)
            new.encode(raw)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                old.encode(raw)
                new.encode(raw)
        torch.cuda.current_stream().wait_stream(stream)
        preserved = [i for i in range(223) if i not in (198,199)]
        for index in range(3):
            raw[:,198:200].copy_(cpu[:,198:200].to(raw.device) + index*.02)
            before = raw.clone()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(new.observations.data_ptr(), ptr)
            self.assertTrue(torch.equal(raw, before))
            self.assertTrue(torch.equal(new.observations[:,preserved], old.observations[:,preserved]))
            self.assertTrue(torch.equal(new.observations[:,198:200], (raw[:,198:200].double()/120).float()))
            old.check_status()
            new.check_status()


if __name__ == "__main__":
    unittest.main()
