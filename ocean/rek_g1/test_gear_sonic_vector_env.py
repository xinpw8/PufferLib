import sys
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gear_sonic_bundle as bundle_contract
import gear_sonic_candidate as candidate
import gear_sonic_vector_env as vector_env
import sonic_candidate as plant


class FakeNode:
    def __init__(self, width: int, batch):
        self.name = "obs_dict"
        self.shape = [batch, width]


class FakeSession:
    def __init__(self, kind: str, batch):
        self.kind = kind
        self.width = (
            candidate.ENCODER_DIM if kind == "encoder" else candidate.DECODER_DIM
        )
        self.input = FakeNode(self.width, batch)
        self.batch_calls: list[int] = []

    def get_inputs(self):
        return [self.input]

    def run(self, output_names, feeds):
        value = np.asarray(feeds["obs_dict"])
        self.batch_calls.append(int(value.shape[0]))
        self.assert_contract(output_names, value)
        if self.kind == "encoder":
            result = np.zeros((value.shape[0], candidate.TOKEN_DIM), dtype=np.float32)
            result[:, 0] = value[:, 0]
            result[:, 1] = value[:, 4:294].mean(axis=1)
            result[:, 2] = value[:, 601:661].mean(axis=1)
            return [result]

        latest_history_start = 94 + 9 * candidate.ACTION_DIM
        latest_history_end = latest_history_start + candidate.ACTION_DIM
        latest_positions = value[:, latest_history_start:latest_history_end]
        result = np.asarray(
            np.float32(0.25) * latest_positions + np.float32(0.01) * value[:, :1],
            dtype=np.float32,
        )
        return [result]

    def assert_contract(self, output_names, value):
        expected_output = "encoded_tokens" if self.kind == "encoder" else "action"
        if output_names != [expected_output]:
            raise AssertionError(output_names)
        if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != self.width:
            raise AssertionError((value.dtype, value.shape))
        declared_batch = self.input.shape[0]
        if isinstance(declared_batch, int) and value.shape[0] != declared_batch:
            raise AssertionError((declared_batch, value.shape))


class FakeData:
    def __init__(self):
        self.qpos = np.zeros(36, dtype=np.float64)
        self.qvel = np.zeros(35, dtype=np.float64)
        self.ctrl = np.zeros(candidate.ACTION_DIM, dtype=np.float64)
        self.actuator_force = np.zeros(candidate.ACTION_DIM, dtype=np.float64)
        self.time = 0.0


class FakeMujoco:
    def __init__(self):
        self.mjtObj = SimpleNamespace(mjOBJ_BODY=1)
        self.step_calls: Counter[int] = Counter()

    def mj_resetData(self, _model, data):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.actuator_force[:] = 0.0
        data.time = 0.0

    def mj_forward(self, _model, _data):
        return None

    def mj_objectVelocity(
        self, _model, data, _object_type, _object_id, result, _local
    ):
        result[:] = 0.0
        result[:3] = data.qvel[3:6]

    def mj_step(self, model, data):
        dt = float(model.opt.timestep)
        q = data.qpos[7:36]
        dq = data.qvel[6:35]
        acceleration = np.float64(4.0) * (data.ctrl - q) - np.float64(0.2) * dq
        dq += acceleration * dt
        q += dq * dt
        data.actuator_force[:] = np.float64(3.0) * (data.ctrl - q)
        data.time += dt
        self.step_calls[id(data)] += 1


def synthetic_motion(frames: int = 80) -> plant.MotionData:
    positions = np.broadcast_to(
        candidate.DEFAULT_ANGLES_MUJOCO, (frames, candidate.ACTION_DIM)
    ).copy()
    positions += np.arange(frames, dtype=np.float32)[:, None] * np.float32(1e-4)
    rotations = np.zeros((frames, 4), dtype=np.float32)
    rotations[:, 3] = 1.0
    return plant.MotionData(
        role="idle",
        filename="idle_processed.npz",
        size=1,
        sha256="1" * 64,
        fps=50.0,
        dof_pos=positions,
        dof_vel=plant.compute_dof_velocities(positions, candidate.CONTROL_DT),
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_rot_xyzw=rotations,
        manifest_sha256="2" * 64,
        inventory_sha256="3" * 64,
    )


def build_environment(
    count: int, batch=1, reference_resolver=None, physics_workers: int = 1
):
    mujoco = FakeMujoco()
    model = SimpleNamespace(
        opt=SimpleNamespace(timestep=candidate.PHYSICS_DT),
        jnt_limited=np.zeros(candidate.ACTION_DIM, dtype=np.int32),
        jnt_range=np.column_stack(
            (
                np.full(candidate.ACTION_DIM, -10.0),
                np.full(candidate.ACTION_DIM, 10.0),
            )
        ),
    )
    runtime_map = SimpleNamespace(
        joint_ids=np.arange(candidate.ACTION_DIM, dtype=np.int64),
        qpos_addresses=np.arange(7, 36, dtype=np.int64),
        qvel_addresses=np.arange(6, 35, dtype=np.int64),
        actuator_ids=np.arange(candidate.ACTION_DIM, dtype=np.int64),
        root_qpos_address=0,
        root_body_id=1,
    )
    motion = synthetic_motion()
    data = [FakeData() for _ in range(count)]
    initializations = [
        plant.initialize_simulation(
            mujoco, model, item, runtime_map, motion, root_z_offset=0.0
        )
        for item in data
    ]
    encoder = FakeSession("encoder", batch)
    decoder = FakeSession("decoder", batch)
    environment = vector_env.GearSonicVectorEnv(
        mujoco=mujoco,
        model=model,
        runtime_map=runtime_map,
        motion=motion,
        controller=vector_env.GearSonicController(encoder, decoder),
        data=data,
        force_limits=np.full(candidate.ACTION_DIM, 1_000.0),
        frame_mode="clamp",
        physics_workers=physics_workers,
        reference_resolver=reference_resolver,
        initializations=initializations,
    )
    return environment, encoder, decoder, mujoco


class GearSonicVectorEnvTests(unittest.TestCase):
    def test_public_fixed_batch_one_sessions_loop_under_one_controller(self):
        self.assertEqual(
            bundle_contract.EXPECTED_ENCODER_GRAPH["inputs"][0]["shape"], [1, 1762]
        )
        self.assertEqual(
            bundle_contract.EXPECTED_DECODER_GRAPH["inputs"][0]["shape"], [1, 994]
        )
        environment, encoder, decoder, mujoco = build_environment(3, batch=1)

        results = environment.step()

        self.assertEqual(encoder.batch_calls, [1, 1, 1])
        self.assertEqual(decoder.batch_calls, [1, 1, 1])
        self.assertEqual(
            environment.controller.encoder_batch_mode, "per_env_fixed_batch_1"
        )
        self.assertEqual(
            environment.controller.decoder_batch_mode, "per_env_fixed_batch_1"
        )
        self.assertEqual(len({id(item) for item in environment.data}), 3)
        self.assertEqual(environment.policy_ticks, (1, 1, 1))
        for data, result in zip(environment.data, results):
            self.assertEqual(result.physics_steps, 10)
            self.assertEqual(result.command_lpf_updates, 5)
            self.assertAlmostEqual(data.time, candidate.CONTROL_DT, places=15)
            self.assertEqual(mujoco.step_calls[id(data)], 10)

    def test_compatible_graph_uses_one_batched_call_with_identical_results(self):
        for compatible_batch in (None, "batch", 3):
            with self.subTest(declared_batch=compatible_batch):
                fixed, fixed_encoder, fixed_decoder, _ = build_environment(3, batch=1)
                compatible, compatible_encoder, compatible_decoder, _ = (
                    build_environment(3, batch=compatible_batch)
                )

                for _tick in range(3):
                    fixed_results = fixed.step()
                    compatible_results = compatible.step()
                    for fixed_result, compatible_result in zip(
                        fixed_results, compatible_results
                    ):
                        for field in (
                            "token",
                            "raw_action_policy",
                            "applied_targets_mujoco",
                            "q_after_mujoco",
                        ):
                            np.testing.assert_array_equal(
                                getattr(fixed_result, field),
                                getattr(compatible_result, field),
                            )

                self.assertEqual(fixed_encoder.batch_calls, [1] * 9)
                self.assertEqual(fixed_decoder.batch_calls, [1] * 9)
                self.assertEqual(compatible_encoder.batch_calls, [3] * 3)
                self.assertEqual(compatible_decoder.batch_calls, [3] * 3)
                self.assertEqual(compatible.controller.encoder_batch_mode, "batched")
                self.assertEqual(compatible.controller.decoder_batch_mode, "batched")

    def test_encoder_and_decoder_choose_batch_modes_independently(self):
        environment, encoder, decoder, _ = build_environment(2, batch=1)
        decoder.input.shape[0] = None

        environment.step()

        self.assertEqual(encoder.batch_calls, [1, 1])
        self.assertEqual(decoder.batch_calls, [2])
        self.assertEqual(
            environment.controller.encoder_batch_mode, "per_env_fixed_batch_1"
        )
        self.assertEqual(environment.controller.decoder_batch_mode, "batched")

    def test_incompatible_static_batch_and_invalid_output_fail_closed(self):
        observations = np.zeros((3, candidate.ENCODER_DIM), dtype=np.float32)
        controller = vector_env.GearSonicController(
            FakeSession("encoder", 4), FakeSession("decoder", 1)
        )
        with self.assertRaisesRegex(candidate.GearSonicError, "requires batch 4"):
            controller.encode(observations)

        class InvalidOutputSession(FakeSession):
            def __init__(self, invalid_output):
                super().__init__("encoder", None)
                self.invalid_output = invalid_output

            def run(self, output_names, feeds):
                value = np.asarray(feeds["obs_dict"])
                self.batch_calls.append(int(value.shape[0]))
                self.assert_contract(output_names, value)
                return [self.invalid_output(value.shape[0])]

        invalid_outputs = (
            lambda count: np.zeros((count, candidate.TOKEN_DIM - 1), dtype=np.float32),
            lambda count: np.zeros((count, candidate.TOKEN_DIM), dtype=np.float64),
            lambda count: np.full(
                (count, candidate.TOKEN_DIM), np.nan, dtype=np.float32
            ),
        )
        for invalid_output in invalid_outputs:
            with self.subTest(invalid_output=invalid_output):
                controller = vector_env.GearSonicController(
                    InvalidOutputSession(invalid_output), FakeSession("decoder", 1)
                )
                with self.assertRaisesRegex(candidate.GearSonicError, "invalid dtype"):
                    controller.encode(observations)

    def test_environment_state_is_independent_and_matches_one_environment(self):
        vector, _, _, _ = build_environment(2, batch=1)
        scalar, _, _, _ = build_environment(1, batch=1)
        vector.data[0].qpos[7] += 0.4

        first_vector = vector.step()
        first_scalar = scalar.step()[0]
        second_vector = vector.step()
        second_scalar = scalar.step()[0]

        self.assertFalse(
            np.array_equal(first_vector[0].q_after_mujoco, first_vector[1].q_after_mujoco)
        )
        for vector_result, scalar_result in (
            (first_vector[1], first_scalar),
            (second_vector[1], second_scalar),
        ):
            for field in (
                "encoder_observation",
                "token",
                "decoder_observation",
                "raw_action_policy",
                "command_lpf_state_mujoco",
                "q_after_mujoco",
            ):
                np.testing.assert_array_equal(
                    getattr(vector_result, field), getattr(scalar_result, field)
                )
        np.testing.assert_array_equal(vector.data[1].qpos, scalar.data[0].qpos)
        np.testing.assert_array_equal(vector.data[1].qvel, scalar.data[0].qvel)

    def test_selected_reset_does_not_mutate_other_environment(self):
        environment, _, _, _ = build_environment(2, batch=1)
        environment.step()
        environment.step()
        env_zero_qpos = environment.data[0].qpos.copy()
        env_zero_qvel = environment.data[0].qvel.copy()

        records = environment.reset(indices=[1], root_z_offsets=[0.25])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["root_z_offset_metres"], 0.25)
        self.assertEqual(environment.policy_ticks, (2, 0))
        np.testing.assert_array_equal(environment.data[0].qpos, env_zero_qpos)
        np.testing.assert_array_equal(environment.data[0].qvel, env_zero_qvel)
        self.assertEqual(environment.data[1].time, 0.0)
        self.assertEqual(environment.data[1].qpos[2], 0.25)

    def test_reference_resolver_metadata_is_preserved(self):
        calls = []

        def resolver(env_index, policy_tick):
            calls.append((env_index, policy_tick))
            return 0, {"cursor_before": 0.0, "completed": False}

        environment, _, _, _ = build_environment(
            1, batch=1, reference_resolver=resolver
        )

        result = environment.step()[0]

        self.assertEqual(calls, [(0, 0)])
        self.assertEqual(result.current_frame, 0)
        self.assertEqual(
            result.reference_metadata,
            {"cursor_before": 0.0, "completed": False},
        )

    def test_failed_step_requires_full_reset_before_retry(self):
        class Resolver:
            def __init__(self):
                self.fail = True
                self.reset_calls = []

            def __call__(self, env_index, policy_tick):
                if self.fail and env_index == 1:
                    raise ValueError("injected resolver failure")
                return policy_tick, {}

            def reset(self, indices):
                self.reset_calls.append(tuple(indices))
                self.fail = False

        resolver = Resolver()
        environment, _, _, _ = build_environment(
            3, batch=1, reference_resolver=resolver
        )

        with self.assertRaisesRegex(candidate.GearSonicError, "resolver failed"):
            environment.step()
        self.assertTrue(environment.failed)
        self.assertEqual(environment.policy_ticks, (0, 0, 0))
        with self.assertRaisesRegex(candidate.GearSonicError, "unusable"):
            environment.step()
        with self.assertRaisesRegex(candidate.GearSonicError, "every environment"):
            environment.reset(indices=[0])

        environment.reset()
        self.assertEqual(resolver.reset_calls, [(0, 1, 2)])
        self.assertFalse(environment.failed)
        results = environment.step()
        self.assertEqual(environment.policy_ticks, (1, 1, 1))
        for result in results[1:]:
            np.testing.assert_array_equal(
                result.decoder_observation, results[0].decoder_observation
            )
            np.testing.assert_array_equal(
                result.q_after_mujoco, results[0].q_after_mujoco
            )

    def test_partial_physics_failure_poison_requires_full_reset(self):
        environment, _, _, mujoco = build_environment(
            3, batch=1, physics_workers=3
        )
        normal_step = mujoco.mj_step
        first_steps = set()
        first_step_barrier = threading.Barrier(3)

        def injected_step(model, data):
            data_id = id(data)
            if data_id not in first_steps:
                first_steps.add(data_id)
                first_step_barrier.wait(timeout=5.0)
                if data is environment.data[1]:
                    raise RuntimeError("injected physics failure")
            normal_step(model, data)

        failed_executor = environment._executor
        mujoco.mj_step = injected_step
        with self.assertRaisesRegex(RuntimeError, "injected physics failure"):
            environment.step()

        self.assertTrue(environment.failed)
        self.assertAlmostEqual(environment.data[0].time, candidate.CONTROL_DT)
        self.assertEqual(environment.data[1].time, 0.0)
        self.assertAlmostEqual(environment.data[2].time, candidate.CONTROL_DT)
        self.assertIsNone(environment._executor)
        with self.assertRaisesRegex(candidate.GearSonicError, "unusable"):
            environment.step()

        mujoco.mj_step = normal_step
        environment.reset()
        self.assertFalse(environment.failed)
        self.assertIsNotNone(environment._executor)
        self.assertIsNot(environment._executor, failed_executor)
        results = environment.step()
        self.assertEqual([result.env_index for result in results], [0, 1, 2])
        for result in results[1:]:
            np.testing.assert_array_equal(
                result.q_after_mujoco, results[0].q_after_mujoco
            )
        environment.close()

    def test_parallel_apply_preserves_order_and_matches_serial(self):
        serial, _, _, _ = build_environment(4, batch=1, physics_workers=1)
        parallel, _, _, _ = build_environment(4, batch=1, physics_workers=4)
        for env_index in range(4):
            qpos_index = 7 + env_index
            offset = np.float64(0.05 * (env_index + 1))
            serial.data[env_index].qpos[qpos_index] += offset
            parallel.data[env_index].qpos[qpos_index] += offset

        persistent_executor = parallel._executor
        try:
            for _tick in range(3):
                serial_results = serial.step()
                parallel_results = parallel.step()
                self.assertEqual(
                    [result.env_index for result in parallel_results], [0, 1, 2, 3]
                )
                self.assertIs(parallel._executor, persistent_executor)
                for serial_result, parallel_result in zip(
                    serial_results, parallel_results
                ):
                    for field in (
                        "decoder_observation",
                        "raw_action_policy",
                        "command_lpf_state_mujoco",
                        "actuator_force_after_mujoco",
                        "q_after_mujoco",
                    ):
                        np.testing.assert_array_equal(
                            getattr(serial_result, field),
                            getattr(parallel_result, field),
                        )
        finally:
            serial.close()
            parallel.close()

    def test_physics_worker_validation_and_close_lifecycle(self):
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    candidate.GearSonicError, "physics_workers must be a positive integer"
                ):
                    build_environment(2, physics_workers=invalid)

        environment, _, _, _ = build_environment(2, physics_workers=2)
        with environment as entered:
            self.assertIs(entered, environment)
            entered.step()
            self.assertFalse(entered.closed)
        self.assertTrue(environment.closed)
        self.assertIsNone(environment._executor)
        environment.close()
        with self.assertRaisesRegex(candidate.GearSonicError, "closed"):
            environment.step()
        with self.assertRaisesRegex(candidate.GearSonicError, "closed"):
            environment.reset()
        with self.assertRaisesRegex(candidate.GearSonicError, "closed"):
            environment.__enter__()

    def test_custom_resolver_reset_without_lifecycle_hook_fails_before_mutation(self):
        environment, _, _, _ = build_environment(
            1, batch=1, reference_resolver=lambda _env, tick: (tick, {})
        )
        environment.step()
        qpos = environment.data[0].qpos.copy()
        tick = environment.policy_ticks

        with self.assertRaisesRegex(
            candidate.GearSonicError, r"reset\(indices\) support"
        ):
            environment.reset()

        self.assertFalse(environment.failed)
        self.assertEqual(environment.policy_ticks, tick)
        np.testing.assert_array_equal(environment.data[0].qpos, qpos)

    def test_reference_resolver_must_start_at_heading_alignment_frame(self):
        environment, _, _, _ = build_environment(
            1, batch=1, reference_resolver=lambda _env, _tick: (7, {})
        )

        with self.assertRaisesRegex(candidate.GearSonicError, "begin at frame zero"):
            environment.step()

        self.assertTrue(environment.failed)
        self.assertEqual(environment.policy_ticks, (0,))

    def test_duplicate_mjdata_is_rejected(self):
        environment, encoder, decoder, _ = build_environment(1, batch=1)
        with self.assertRaisesRegex(candidate.GearSonicError, "distinct MjData"):
            vector_env.GearSonicVectorEnv(
                mujoco=environment.mujoco,
                model=environment.model,
                runtime_map=environment.runtime_map,
                motion=environment.motion,
                controller=vector_env.GearSonicController(encoder, decoder),
                data=[environment.data[0], environment.data[0]],
                force_limits=environment.force_limits,
                frame_mode="clamp",
            )


if __name__ == "__main__":
    unittest.main()
