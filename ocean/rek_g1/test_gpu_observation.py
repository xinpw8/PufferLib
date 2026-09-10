"""Check the CUDA observation ABI and MuJoCo body-inertial velocity transform."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from gpu_observation import GpuObservationAssembler, inertial_body_velocity


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GpuObservationTests(unittest.TestCase):
    def test_body_inertial_velocity_matches_mujoco_without_physics_steps(self):
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/>'
            '</body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        rng = np.random.default_rng(2719)
        expected, cvel, matrices, origins, centers = [], [], [], [], []
        for _ in range(64):
            spatial = rng.normal(size=6).astype(np.float32)
            matrix = np.linalg.qr(rng.normal(size=(3, 3)))[0].astype(np.float32)
            origin = rng.normal(size=3).astype(np.float32)
            center = rng.normal(size=3).astype(np.float32)
            data.cvel[1] = spatial
            data.ximat[1] = matrix.reshape(9)
            data.xipos[1] = origin
            data.subtree_com[model.body_rootid[1]] = center
            value = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, value, 1)
            expected.append(value)
            cvel.append(spatial)
            matrices.append(matrix)
            origins.append(origin)
            centers.append(center)
        tensor = lambda value: torch.as_tensor(np.asarray(value), device="cuda")
        actual = inertial_body_velocity(
            tensor(cvel), tensor(matrices), tensor(origins), tensor(centers),
        )
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=1e-12)

    def test_paired_223_float_layout_and_graph_capture(self):
        tensor = lambda value: torch.as_tensor(np.asarray(value), device="cuda:0")
        roots = np.asarray([1, 2])
        physics = SimpleNamespace(
            arenas=2,
            qpos=tensor(np.arange(144, dtype=np.float32).reshape(2, 72)),
            qvel=tensor(np.arange(140, dtype=np.float32).reshape(2, 70)),
            root_bodies=[1, 2],
            joint_qpos=[np.arange(7, 36), np.arange(43, 72)],
            joint_qvel=[np.arange(6, 35), np.arange(41, 70)],
            host_model=SimpleNamespace(body_rootid=np.array([0, 1, 2])),
            wp=SimpleNamespace(to_torch=lambda value: value),
            data=SimpleNamespace(
                cvel=tensor(np.arange(36, dtype=np.float32).reshape(2, 3, 6)),
                ximat=tensor(np.tile(np.eye(3, dtype=np.float32), (2, 3, 1, 1))),
                xipos=tensor(np.zeros((2, 3, 3), np.float32)),
                subtree_com=tensor(np.zeros((2, 3, 3), np.float32)),
            ),
        )
        assembler = GpuObservationAssembler(physics)
        fall = tensor(np.arange(60, dtype=np.float32).reshape(4, 15))
        semantic = tensor(np.arange(48, dtype=np.float32).reshape(4, 12))
        fight = tensor(np.arange(156, dtype=np.float32).reshape(4, 39))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            assembler.gather_kinematics()
            assembler.pack(fall, semantic, fight)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                assembler.gather_kinematics()
                assembler.pack(fall, semantic, fight)
            graph.replay()
            stream.synchronize()
            output = assembler.observations.cpu().numpy()
        qpos = physics.qpos.cpu().numpy()
        velocity = physics.data.cvel.cpu().numpy()[:, roots].reshape(4, 6)
        entity = np.concatenate((
            qpos.reshape(4, 36)[:, :7], velocity[:, 3:], velocity[:, :3],
            qpos[:, np.stack(physics.joint_qpos)].reshape(4, 29),
            physics.qvel.cpu().numpy()[:, np.stack(physics.joint_qvel)].reshape(4, 29),
            fall.cpu().numpy(),
        ), axis=1)
        np.testing.assert_array_equal(output[:, :86], entity)
        np.testing.assert_array_equal(output[:, 86:172], entity.reshape(2, 2, 86)[:, ::-1].reshape(4, 86))
        np.testing.assert_array_equal(output[:, 172:184], semantic.cpu().numpy())
        np.testing.assert_array_equal(output[:, 184:], fight.cpu().numpy())
        with self.assertRaisesRegex(ValueError, "CUDA float32"):
            assembler.pack(fall.cpu(), semantic, fight)


if __name__ == "__main__":
    unittest.main()
