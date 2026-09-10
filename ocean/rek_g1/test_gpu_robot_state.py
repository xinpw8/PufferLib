"""Compare CUDA controller features to the existing NumPy reference."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

import gear_sonic_candidate as reference
from gpu_robot_state import G1GpuControllerState


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GpuRobotStateTests(unittest.TestCase):
    def test_features_history_suspension_reset_and_target_transform(self):
        rng = np.random.default_rng(7301)
        rows = 8
        state = G1GpuControllerState(rows)
        histories = [reference.StateHistory() for _ in range(rows)]
        last = np.zeros((rows, 29), np.float32)
        targets = np.zeros_like(last)
        dof = rng.normal(0, 0.2, (64, 29)).astype(np.float32)
        roots = rng.normal(size=(64, 4)).astype(np.float32)
        roots /= np.linalg.norm(roots, axis=1, keepdims=True)
        motion = SimpleNamespace(dof_pos=dof, dof_vel=np.zeros_like(dof), root_rot_xyzw=roots)
        token = rng.normal(size=(rows, 64)).astype(np.float32)
        tensor = lambda value: torch.as_tensor(value, device="cuda")
        for tick in range(18):
            active = np.asarray([True, tick % 2 == 0, tick < 7, True] * 2)
            reset = np.zeros(rows, dtype=bool)
            if tick == 12:
                reset[[0, 1, 6]] = True
                state.reset(tensor(reset))
                for row in np.flatnonzero(reset):
                    histories[row].reset()
                    last[row] = 0
                    targets[row] = 0
            base = rng.normal(size=(rows, 4))
            base /= np.linalg.norm(base, axis=1, keepdims=True)
            heading = rng.normal(size=(rows, 4))
            heading /= np.linalg.norm(heading, axis=1, keepdims=True)
            q = rng.normal(size=(rows, 29))
            dq = rng.normal(size=(rows, 29))
            angular = rng.normal(size=(rows, 3))
            indices = tick + np.arange(10) * 5
            expected_encoder = []
            for row in range(rows):
                expected, _ = reference.build_encoder_observation(motion, tick, base[row], heading[row])
                expected_encoder.append(expected)
                if active[row]:
                    histories[row].append(reference.HistoryEntry(
                        base_quat_wxyz=base[row].astype(np.float32),
                        base_ang_vel=angular[row].astype(np.float32),
                        body_q_policy=(q[row] - reference.DEFAULT_ANGLES_MUJOCO)[reference.MUJOCO_TO_ISAACLAB].astype(np.float32),
                        body_dq_policy=dq[row, reference.MUJOCO_TO_ISAACLAB].astype(np.float32),
                        last_action_policy=last[row].copy(),
                    ))
            actual = state.prepare(
                tensor(base), tensor(angular), tensor(q), tensor(dq), tensor(heading),
                tensor(np.broadcast_to(dof[indices], (rows, 10, 29)).copy()),
                tensor(np.broadcast_to(dof[indices + 1], (rows, 10, 29)).copy()),
                tensor(np.broadcast_to(roots[indices], (rows, 10, 4)).copy()),
                tensor(active),
            )
            np.testing.assert_allclose(actual.cpu().numpy(), expected_encoder, atol=1e-7, rtol=0)
            expected_decoder = np.stack([
                reference.build_decoder_observation(token[row], histories[row])
                for row in range(rows)
            ])
            actual_decoder = state.decoder_input(tensor(token))
            np.testing.assert_array_equal(actual_decoder.cpu().numpy(), expected_decoder)
            raw = rng.normal(0, 120, (rows, 29)).astype(np.float32)
            for row in np.flatnonzero(active):
                last[row], targets[row] = reference.action_to_targets(raw[row])
            actual_targets = state.apply_actions(tensor(raw), tensor(active))
            np.testing.assert_array_equal(actual_targets.cpu().numpy(), targets)
            np.testing.assert_array_equal(state.last_actions.cpu().numpy(), last)

    def test_cpu_storage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            G1GpuControllerState(8, device="cpu")

    def test_feature_history_and_targets_allow_cuda_graph_capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            state = G1GpuControllerState(8)
            base = torch.zeros((8, 4), device="cuda")
            base[:, 0] = 1
            angular = torch.zeros((8, 3), device="cuda")
            joints = state.default.repeat(8, 1)
            velocity = torch.zeros_like(joints)
            position = joints[:, None].repeat(1, 10, 1)
            roots = torch.zeros((8, 10, 4), device="cuda")
            roots[:, :, 3] = 1
            active = torch.ones(8, device="cuda", dtype=torch.bool)
            tokens = torch.ones((8, 64), device="cuda")
            raw = torch.ones((8, 29), device="cuda")

            def step():
                state.prepare(
                    base, angular, joints, velocity, base,
                    position, position, roots, active,
                )
                state.decoder_input(tokens)
                state.apply_actions(raw, active)

            step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                step()
            for _ in range(15):
                graph.replay()
            stream.synchronize()
            self.assertTrue(torch.isfinite(state.decoder_observations).all().item())
            np.testing.assert_array_equal(
                state.history["action"].cpu().numpy(), np.ones((8, 10, 29), np.float32),
            )


if __name__ == "__main__":
    unittest.main()
