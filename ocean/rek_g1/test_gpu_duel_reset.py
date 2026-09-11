"""Check selected-arena reset state and the two-boundary counted-fall reset."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gpu_duel_reset import GpuDuelReset


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GpuDuelResetTests(unittest.TestCase):
    def fixture(self):
        tensor = lambda values: torch.as_tensor(values, device="cuda:0")
        spawn = np.zeros(72)
        spawn[[2, 38]] = 0.8
        spawn[[3, 39]] = 1
        model = SimpleNamespace(
            qpos0=spawn, actuator_trnid=np.column_stack((np.arange(58), np.zeros(58))).astype(int),
            jnt_limited=np.ones(58, bool), jnt_range=np.tile([-.5, .5], (58, 1)),
        )
        physics = SimpleNamespace(
            arenas=4, host_model=model,
            actuator_ids=[np.arange(29), np.arange(29, 58)],
            joint_qpos=[np.arange(7, 36), np.arange(43, 72)],
            joint_qvel=[np.arange(6, 35), np.arange(41, 70)],
            qpos=tensor(np.arange(288, dtype=np.float32).reshape(4, 72)),
            qvel=tensor(np.arange(280, dtype=np.float32).reshape(4, 70)),
            ctrl=tensor(np.arange(232, dtype=np.float32).reshape(4, 58)),
            time=tensor(np.arange(4, dtype=np.float32)),
            wp=SimpleNamespace(to_torch=lambda value: value),
            data=SimpleNamespace(**{
                name: tensor(np.ones((4, width), np.float32))
                for name, width in (("qacc", 70), ("qacc_warmstart", 70),
                                    ("qfrc_applied", 70), ("xfrc_applied", 378), ("act", 0))
            }),
            forward_selected=lambda mask: None,
        )
        assets = SimpleNamespace(
            roles={"idle": {"files": {"mujoco_joint_order": "joints", "xyzw": "rotations"}}},
            host_arrays={"joints": np.ones((1, 29), np.float32),
                         "rotations": np.array([[0, 0, 0, 1]], np.float32)},
        )
        return physics, GpuDuelReset(physics, assets)

    def test_full_reset_preserves_other_worlds_and_optional_clock(self):
        physics, reset = self.fixture()
        before_q = physics.qpos.clone()
        before_dq = physics.qvel.clone()
        before_clock = physics.time.clone()
        mask = torch.tensor([True, False, False, True], device="cuda")
        reset.full(mask)
        torch.testing.assert_close(physics.qpos[~mask], before_q[~mask], rtol=0, atol=0)
        torch.testing.assert_close(physics.qvel[~mask], before_dq[~mask], rtol=0, atol=0)
        torch.testing.assert_close(physics.qpos[mask], reset.initial_qpos.repeat(2, 1), rtol=0, atol=0)
        torch.testing.assert_close(physics.time, before_clock, rtol=0, atol=0)
        self.assertTrue((physics.qpos[mask][:, reset.qindices] == .5).all().item())
        for values in reset.reset_buffers.values():
            self.assertTrue((values[mask] == 0).all().item())
        reset.full(mask, reset_clock=True)
        self.assertTrue((physics.time[mask] == 0).all().item())

    def test_deferred_reset_preserves_root_velocity_and_waits_for_caller_boundary(self):
        physics, reset = self.fixture()
        mask = torch.tensor([False, True, True, False], device="cuda")
        before_q, before_dq = physics.qpos.clone(), physics.qvel.clone()
        reset.begin(mask)
        torch.testing.assert_close(physics.qvel, before_dq, rtol=0, atol=0)
        torch.testing.assert_close(
            physics.qpos[:, reset.qindices], before_q[:, reset.qindices], rtol=0, atol=0,
        )
        self.assertTrue(torch.equal(reset.pending, mask))
        # Represent the next fixed physics boundary without simulating on CPU.
        physics.qpos[mask] += .125
        physics.qvel[mask] += .25
        root_velocity = physics.qvel.reshape(4, 2, 35)[..., :6].clone()
        before_clock = physics.time.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                reset.complete(mask)
            graph.replay()
            stream.synchronize()
        torch.testing.assert_close(physics.qpos[~mask], before_q[~mask], rtol=0, atol=0)
        torch.testing.assert_close(physics.qvel[~mask], before_dq[~mask], rtol=0, atol=0)
        torch.testing.assert_close(
            physics.qvel.reshape(4, 2, 35)[..., :6], root_velocity, rtol=0, atol=0,
        )
        self.assertTrue((physics.qpos[mask][:, reset.qindices] == 0).all().item())
        self.assertTrue((physics.qvel[mask][:, reset.dqindices] == 0).all().item())
        self.assertTrue((physics.ctrl[mask] == 0).all().item())
        self.assertFalse(reset.pending.any().item())
        torch.testing.assert_close(physics.time, before_clock, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
