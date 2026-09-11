import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gear_sonic_candidate as candidate
import sonic_candidate as plant


def synthetic_motion(frames: int = 60) -> plant.MotionData:
    q = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 1000.0
    dq = -q.copy()
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    return plant.MotionData(
        role="idle",
        filename="idle_processed.npz",
        size=1,
        sha256="1" * 64,
        fps=50.0,
        dof_pos=q,
        dof_vel=dq,
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_rot_xyzw=root_rot,
        manifest_sha256="2" * 64,
        inventory_sha256="3" * 64,
    )


def contract_motion(role: str, frames: int, filename: str, sha256: str) -> plant.MotionData:
    motion = synthetic_motion(frames)
    return plant.MotionData(
        role=role,
        filename=filename,
        size=motion.size,
        sha256=sha256,
        fps=motion.fps,
        dof_pos=motion.dof_pos,
        dof_vel=motion.dof_vel,
        root_pos=motion.root_pos,
        root_rot_xyzw=motion.root_rot_xyzw,
        manifest_sha256=motion.manifest_sha256,
        inventory_sha256=motion.inventory_sha256,
    )


def history_entry(value: float, quaternion=None) -> candidate.HistoryEntry:
    if quaternion is None:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return candidate.HistoryEntry(
        base_quat_wxyz=np.asarray(quaternion, dtype=np.float32),
        base_ang_vel=np.full(3, value, dtype=np.float32),
        body_q_policy=np.full(29, value + 1.0, dtype=np.float32),
        body_dq_policy=np.full(29, value + 2.0, dtype=np.float32),
        last_action_policy=np.full(29, value + 3.0, dtype=np.float32),
    )


class GearSonicCandidateTests(unittest.TestCase):
    def test_composer_provider_keeps_measured_idle_loop_stable(self):
        motion = contract_motion(
            "idle",
            39,
            "idle_processed.npz",
            "2fd9ca753183566cceaff3ff10ae54b1a7167986297524a9d24d970d57aee969",
        )
        provider = candidate.MotionComposerCursorProvider(motion, "idle", "loop")
        records = [provider(0, tick)[1] for tick in range(40)]

        self.assertEqual(records[0]["composer_cursor_before"], 0.0)
        self.assertEqual(records[38]["composer_cursor_before"], 38.0)
        self.assertTrue(records[38]["composer_wrapped"])
        self.assertEqual(records[38]["composer_cursor_after"], 0.0)
        self.assertEqual(records[38]["composer_next_frame"]["f0"], 0)
        self.assertEqual(records[39]["composer_frame_f0"], 0)
        self.assertTrue(all(item["composer_interpolation_t"] == 0.0 for item in records))

    def test_composer_provider_emits_one_non_loop_completion_event(self):
        motion = contract_motion(
            "kick_left_front",
            146,
            "left_front_kick_processed.npz",
            "3db29034517acc376c16ce5bc04f923803a48367ad58a5c65a314123639815c1",
        )
        provider = candidate.MotionComposerCursorProvider(
            motion, "kick_left_front", "clamp"
        )
        records = [provider(0, tick)[1] for tick in range(146)]

        events = [item for item in records if item["composer_completion_event"]]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["composer_cursor_after"], 145.0)
        self.assertEqual(records[-1]["composer_frame_f0"], 145)
        self.assertTrue(records[-1]["composer_endpoint_condition"])

    def test_composer_provider_fails_closed_on_contract_mode_or_transition(self):
        motion = contract_motion(
            "idle",
            39,
            "idle_processed.npz",
            "2fd9ca753183566cceaff3ff10ae54b1a7167986297524a9d24d970d57aee969",
        )
        with self.assertRaises(candidate.GearSonicError):
            candidate.MotionComposerCursorProvider(motion, "idle", "clamp")
        provider = candidate.MotionComposerCursorProvider(motion, "idle", "loop")
        with self.assertRaises(candidate.GearSonicError):
            provider.transition("kick_left_front")

    def test_composer_provider_fails_closed_on_wrong_motion_hash(self):
        motion = contract_motion("idle", 39, "idle_processed.npz", "0" * 64)
        with self.assertRaises(candidate.GearSonicError):
            candidate.MotionComposerCursorProvider(motion, "idle", "loop")

    def test_run_records_composer_resolution_in_trace_and_report(self):
        motion = contract_motion(
            "idle",
            39,
            "idle_processed.npz",
            "2fd9ca753183566cceaff3ff10ae54b1a7167986297524a9d24d970d57aee969",
        )

        class FakeEnvironment:
            def __init__(self):
                qpos = np.zeros(7, dtype=np.float64)
                qpos[2] = 1.0
                qpos[3] = 1.0
                self.motion = motion
                self.bundle_report = {"status": "ok"}
                self.xml_contract = SimpleNamespace(source_sha256="1" * 64)
                self.arena_contract = SimpleNamespace(source_sha256="2" * 64)
                self.mujoco = object()
                self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
                self.runtime_map = SimpleNamespace(root_qpos_address=0)
                self.data = [SimpleNamespace(qpos=qpos, time=0.02)]
                self.force_limits = np.ones(29, dtype=np.float64)
                self.serialized_timestep = 0.02
                self.mujoco_version = "test"
                self.onnxruntime_version = "test"
                self.physics_steps_per_control = 10
                self.command_lpf_interval = 2
                self.command_lpf_dt = 0.004
                self.command_lpf_alpha = 0.5
                self.initializations = [{"status": "ok"}]
                self.reference_resolver = None

            def step(self):
                frame, metadata = self.reference_resolver(0, 0)
                future = np.asarray(
                    [item["f0"] for item in metadata["composer_future_frames"]],
                    dtype=np.int64,
                )
                zeros29 = np.zeros(29, dtype=np.float64)
                return (
                    SimpleNamespace(
                        tick=0,
                        current_frame=frame,
                        reference_metadata=metadata,
                        future_indices=future,
                        encoder_observation=np.zeros(1762, dtype=np.float32),
                        token=np.zeros(64, dtype=np.float32),
                        decoder_observation=np.zeros(994, dtype=np.float32),
                        raw_action_policy=np.zeros(29, dtype=np.float32),
                        clipped_action_policy=np.zeros(29, dtype=np.float32),
                        raw_targets_mujoco=zeros29,
                        command_lpf_state_mujoco=zeros29,
                        applied_targets_mujoco=zeros29,
                        joint_target_clipped=np.zeros(29, dtype=np.bool_),
                        q_before_mujoco=zeros29,
                        dq_before_mujoco=zeros29,
                        torque_raw_mujoco=zeros29,
                        torque_predicted_mujoco=zeros29,
                        actuator_force_after_mujoco=zeros29,
                        torque_saturated=np.zeros(29, dtype=np.bool_),
                        q_after_mujoco=zeros29,
                        projected_gravity_after=np.asarray([0.0, 0.0, -1.0]),
                        physics_steps=10,
                        command_lpf_updates=5,
                        actuator_force_squared_sum=0.0,
                        torque_saturation_count=0,
                        torque_value_count=29,
                    ),
                )

        fake_environment = FakeEnvironment()
        fake_vector_module = SimpleNamespace(
            GearSonicVectorEnv=SimpleNamespace(
                from_artifacts=lambda **_kwargs: fake_environment
            )
        )
        with patch.dict(sys.modules, {"gear_sonic_vector_env": fake_vector_module}), patch.object(
            candidate.plant_contract, "read_body_pose", return_value={}
        ):
            report, trace_lines = candidate.run(
                bundle=Path("bundle"),
                assets_dir=Path("assets"),
                motion_role="idle",
                manifest=Path("manifest"),
                xml=Path("plant.xml"),
                arena=Path("arena.json"),
                steps=1,
                control_boundary="native-position-actuator",
                frame_mode="loop",
                force_limit_source="public-model-config",
            )

        trace = json.loads(trace_lines[0])
        self.assertEqual(trace["composer_cursor_before"], 0.0)
        self.assertEqual(trace["composer_cursor_after"], 1.0)
        self.assertEqual(trace["composer_frame_f0"], 0)
        self.assertEqual(trace["composer_interpolation_t"], 0.0)
        self.assertEqual(trace["future_reference_frames"], [0, 5, 10, 15, 20, 25, 30, 35, 1, 6])
        self.assertEqual(
            report["controller"]["motion_composer"]["transition_support"],
            "fail_closed_crossfade_and_feature_matching",
        )
        self.assertEqual(report["runtime"]["composer_final_cursor"], 1.0)

    def test_joint_permutations_are_exact_inverses(self):
        np.testing.assert_array_equal(
            candidate.ISAACLAB_TO_MUJOCO[candidate.MUJOCO_TO_ISAACLAB],
            np.arange(29),
        )
        np.testing.assert_array_equal(
            candidate.MUJOCO_TO_ISAACLAB[candidate.ISAACLAB_TO_MUJOCO],
            np.arange(29),
        )
        self.assertEqual(len(set(candidate.ISAACLAB_TO_MUJOCO.tolist())), 29)

    def test_public_parameter_vectors_cover_29_mujoco_joints(self):
        for value in (
            candidate.DEFAULT_ANGLES_MUJOCO,
            candidate.ACTION_SCALE_MUJOCO,
            candidate.KP_MUJOCO,
            candidate.KD_MUJOCO,
            candidate.PUBLIC_EFFORT_LIMIT_MUJOCO,
        ):
            self.assertEqual(value.shape, (29,))
            self.assertTrue(np.isfinite(value).all())
        self.assertTrue(np.all(candidate.ACTION_SCALE_MUJOCO > 0.0))
        self.assertTrue(np.all(candidate.KP_MUJOCO > 0.0))
        self.assertTrue(np.all(candidate.KD_MUJOCO > 0.0))

    def test_observation_offsets_cover_exact_graph_dimensions(self):
        self.assertEqual(list(candidate.ENCODER_OFFSETS.values())[0][0], 0)
        self.assertEqual(list(candidate.ENCODER_OFFSETS.values())[-1][1], 1762)
        previous = 0
        for start, end in candidate.ENCODER_OFFSETS.values():
            self.assertEqual(start, previous)
            self.assertGreater(end, start)
            previous = end
        previous = 0
        for start, end in candidate.DECODER_OFFSETS.values():
            self.assertEqual(start, previous)
            self.assertGreater(end, start)
            previous = end
        self.assertEqual(previous, 994)

    def test_encoder_mode_zero_uses_current_plus_five_tick_references(self):
        motion = synthetic_motion()
        observation, indices = candidate.build_encoder_observation(
            motion,
            2,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(indices, [2, 7, 12, 17, 22, 27, 32, 37, 42, 47])
        np.testing.assert_array_equal(
            observation[4:294].reshape(10, 29),
            motion.dof_pos[indices][:, candidate.MUJOCO_TO_ISAACLAB],
        )
        expected_velocity = (
            motion.dof_pos[indices + 1] - motion.dof_pos[indices]
        ) / candidate.CONTROL_DT
        np.testing.assert_allclose(
            observation[294:584].reshape(10, 29),
            expected_velocity[:, candidate.MUJOCO_TO_ISAACLAB],
            rtol=0.0,
            atol=1e-5,
        )
        expected_rotation = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            observation[601:661].reshape(10, 6),
            np.broadcast_to(expected_rotation, (10, 6)),
            rtol=0.0,
            atol=0.0,
        )

    def test_encoder_mode_zero_is_scalar_zero_and_leaves_inactive_channels_zero(self):
        observation, _ = candidate.build_encoder_observation(
            synthetic_motion(),
            0,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_array_equal(observation[0:4], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(observation[584:601], np.zeros(17))
        np.testing.assert_array_equal(observation[661:1762], np.zeros(1101))

    def test_encoder_future_window_clamps_to_final_frame(self):
        motion = synthetic_motion(12)
        observation, indices = candidate.build_encoder_observation(
            motion,
            11,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(indices, np.full(10, 11))
        np.testing.assert_array_equal(
            observation[4:294].reshape(10, 29),
            np.broadcast_to(
                motion.dof_pos[11][candidate.MUJOCO_TO_ISAACLAB], (10, 29)
            ),
        )
        np.testing.assert_array_equal(
            observation[294:584].reshape(10, 29), np.zeros((10, 29))
        )

    def test_encoder_loop_wraps_future_frames_and_seam_velocity(self):
        motion = synthetic_motion(12)
        observation, indices = candidate.build_encoder_observation(
            motion,
            11,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            loop=True,
        )
        np.testing.assert_array_equal(indices, [11, 4, 9, 2, 7, 0, 5, 10, 3, 8])
        expected_position = motion.dof_pos[indices][:, candidate.MUJOCO_TO_ISAACLAB]
        np.testing.assert_array_equal(
            observation[4:294].reshape(10, 29), expected_position
        )
        next_indices = (indices + 1) % motion.dof_pos.shape[0]
        expected_velocity = (
            motion.dof_pos[next_indices] - motion.dof_pos[indices]
        ) / candidate.CONTROL_DT
        np.testing.assert_allclose(
            observation[294:584].reshape(10, 29),
            expected_velocity[:, candidate.MUJOCO_TO_ISAACLAB],
            rtol=0.0,
            atol=1e-5,
        )

    def test_heading_delta_aligns_reference_yaw_to_initial_base_yaw(self):
        half = np.pi / 4.0
        base = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        reference = np.array([1.0, 0.0, 0.0, 0.0])
        delta = candidate.reference_heading_delta(base, reference)
        np.testing.assert_allclose(delta, base, rtol=0.0, atol=1e-12)

    def test_empty_history_matches_native_zero_padding_behavior(self):
        history = candidate.StateHistory()
        token = np.arange(64, dtype=np.float32)
        observation = candidate.build_decoder_observation(token, history)
        np.testing.assert_array_equal(observation[:64], token)
        np.testing.assert_array_equal(observation[64:964], np.zeros(900))
        np.testing.assert_array_equal(
            observation[964:994].reshape(10, 3), np.zeros((10, 3))
        )

    def test_history_is_oldest_first_with_zero_left_padding(self):
        history = candidate.StateHistory()
        history.append(history_entry(4.0))
        history.append(history_entry(5.0))
        observation = candidate.build_decoder_observation(np.zeros(64), history)
        angular = observation[64:94].reshape(10, 3)
        positions = observation[94:384].reshape(10, 29)
        actions = observation[674:964].reshape(10, 29)
        np.testing.assert_array_equal(angular[:8], np.zeros((8, 3)))
        np.testing.assert_array_equal(angular[8], np.full(3, 4.0))
        np.testing.assert_array_equal(angular[9], np.full(3, 5.0))
        np.testing.assert_array_equal(positions[8], np.full(29, 5.0))
        np.testing.assert_array_equal(actions[9], np.full(29, 8.0))
        gravity = observation[964:994].reshape(10, 3)
        np.testing.assert_array_equal(gravity[:8], np.zeros((8, 3)))
        np.testing.assert_array_equal(gravity[8:], np.tile([0.0, 0.0, -1.0], (2, 1)))

    def test_history_capacity_and_reset(self):
        history = candidate.StateHistory()
        for value in range(12):
            history.append(history_entry(float(value)))
        entries = history.oldest_first()
        self.assertEqual([entry.base_ang_vel[0] for entry in entries], list(range(2, 12)))
        history.reset()
        self.assertEqual(history.oldest_first(), [None] * 10)

    def test_history_uses_native_local_body_object_angular_velocity(self):
        qpos = np.zeros(36, dtype=np.float64)
        qpos[3] = 1.0
        qvel = np.zeros(35, dtype=np.float64)
        qvel[3:6] = [91.0, 92.0, 93.0]
        data = SimpleNamespace(qpos=qpos, qvel=qvel)
        model = object()
        runtime_map = SimpleNamespace(
            qpos_addresses=np.arange(7, 36, dtype=np.int64),
            qvel_addresses=np.arange(6, 35, dtype=np.int64),
            root_qpos_address=0,
            root_body_id=17,
        )
        calls = []

        def mj_object_velocity(
            actual_model, actual_data, object_type, object_id, result, local
        ):
            calls.append(
                (actual_model, actual_data, object_type, object_id, local)
            )
            result[:] = [1.25, -2.5, 3.75, 40.0, 50.0, 60.0]

        mujoco = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_BODY=1),
            mj_objectVelocity=mj_object_velocity,
        )
        entry = candidate.state_to_history_entry(
            mujoco, model, data, runtime_map, np.zeros(29, dtype=np.float32)
        )

        self.assertEqual(calls, [(model, data, 1, 17, 1)])
        np.testing.assert_array_equal(
            entry.base_ang_vel, np.array([1.25, -2.5, 3.75], dtype=np.float32)
        )
        self.assertFalse(np.array_equal(entry.base_ang_vel, qvel[3:6]))

    def test_policy_action_clips_then_maps_to_mujoco_targets(self):
        action = np.arange(29, dtype=np.float32) - 14.0
        action[0] = 101.0
        clipped, targets = candidate.action_to_targets(action)
        self.assertEqual(clipped.dtype, np.float32)
        self.assertEqual(targets.dtype, np.float32)
        self.assertEqual(clipped[0], 100.0)
        expected = (
            candidate.DEFAULT_ANGLES_MUJOCO
            + clipped[candidate.ISAACLAB_TO_MUJOCO] * candidate.ACTION_SCALE_MUJOCO
        )
        np.testing.assert_allclose(targets, expected, rtol=0.0, atol=0.0)

    def test_action_target_pipeline_matches_system_single_rounding(self):
        action = np.linspace(
            -3.1415926535, 3.1415926535, 29, dtype=np.float64
        ) + 1e-8
        action[0] = 100.000003
        action[1] = -100.000003

        clipped, targets = candidate.action_to_targets(action)

        expected_clipped = np.empty(29, dtype=np.float32)
        for index, value in enumerate(action):
            expected_clipped[index] = np.minimum(
                np.maximum(np.float32(value), np.float32(-100.0)),
                np.float32(100.0),
            )
        expected_targets = np.empty(29, dtype=np.float32)
        for mujoco_index, policy_index in enumerate(candidate.ISAACLAB_TO_MUJOCO):
            scaled = np.float32(
                expected_clipped[policy_index]
                * np.float32(candidate.ACTION_SCALE_MUJOCO[mujoco_index])
            )
            expected_targets[mujoco_index] = np.float32(
                np.float32(candidate.DEFAULT_ANGLES_MUJOCO[mujoco_index]) + scaled
            )

        self.assertEqual(candidate.DEFAULT_ANGLES_MUJOCO.dtype, np.float32)
        self.assertEqual(candidate.ACTION_SCALE_MUJOCO.dtype, np.float32)
        np.testing.assert_array_equal(clipped.view(np.uint32), expected_clipped.view(np.uint32))
        np.testing.assert_array_equal(targets.view(np.uint32), expected_targets.view(np.uint32))

    def test_pd_torque_supports_explicit_declared_and_unbounded_modes(self):
        q = np.zeros(29)
        dq = np.zeros(29)
        targets = np.ones(29)
        low = np.full(29, -1.0)
        high = np.full(29, 1.0)
        raw, bounded, saturated = candidate.pd_torque(
            q, dq, targets, low, high, "declared-effort"
        )
        np.testing.assert_array_equal(bounded, np.ones(29))
        self.assertTrue(saturated.all())
        raw_again, unbounded, saturated_again = candidate.pd_torque(
            q, dq, targets, low, high, "unbounded"
        )
        np.testing.assert_array_equal(raw_again, raw)
        np.testing.assert_array_equal(unbounded, raw)
        self.assertFalse(saturated_again.any())

    def test_native_position_actuator_configuration_matches_rek_arrays(self):
        count = 31
        selected = np.arange(1, 30, dtype=np.int64)
        ctrl_min = -np.arange(1, 30, dtype=np.float64)
        ctrl_max = np.arange(2, 31, dtype=np.float64)
        runtime_map = SimpleNamespace(
            actuator_ids=selected,
            ctrl_min=ctrl_min,
            ctrl_max=ctrl_max,
        )
        model = SimpleNamespace(
            actuator_dyntype=np.full(count, -1, dtype=np.int32),
            actuator_gaintype=np.full(count, -1, dtype=np.int32),
            actuator_biastype=np.full(count, -1, dtype=np.int32),
            actuator_gainprm=np.full((count, 10), np.nan, dtype=np.float64),
            actuator_biasprm=np.full((count, 10), np.nan, dtype=np.float64),
            actuator_ctrllimited=np.full(count, 7, dtype=np.int32),
            actuator_forcelimited=np.full(count, 7, dtype=np.int32),
            actuator_forcerange=np.full((count, 2), np.nan, dtype=np.float64),
        )
        mujoco = SimpleNamespace(
            mjtDyn=SimpleNamespace(mjDYN_NONE=0),
            mjtGain=SimpleNamespace(mjGAIN_FIXED=2),
            mjtBias=SimpleNamespace(mjBIAS_AFFINE=3),
        )

        force_limits = candidate.configure_native_position_actuators(
            mujoco, model, runtime_map, ctrl_max
        )

        np.testing.assert_array_equal(force_limits, ctrl_max)
        np.testing.assert_array_equal(model.actuator_dyntype[selected], 0)
        np.testing.assert_array_equal(model.actuator_gaintype[selected], 2)
        np.testing.assert_array_equal(model.actuator_biastype[selected], 3)
        np.testing.assert_allclose(
            model.actuator_gainprm[selected, 0], candidate.KP_MUJOCO, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            model.actuator_biasprm[selected, 0], 0.0, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            model.actuator_biasprm[selected, 1], -candidate.KP_MUJOCO, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            model.actuator_biasprm[selected, 2], -candidate.KD_MUJOCO, rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(model.actuator_ctrllimited[selected], 0)
        np.testing.assert_array_equal(model.actuator_forcelimited[selected], 1)
        np.testing.assert_array_equal(model.actuator_forcerange[selected, 0], -ctrl_max)
        np.testing.assert_array_equal(model.actuator_forcerange[selected, 1], ctrl_max)
        self.assertEqual(model.actuator_dyntype[0], -1)
        self.assertEqual(model.actuator_dyntype[30], -1)

    def test_native_position_actuator_configuration_rejects_bad_limits(self):
        runtime_map = SimpleNamespace(
            actuator_ids=np.arange(29),
            ctrl_min=np.zeros(29),
            ctrl_max=np.zeros(29),
        )
        with self.assertRaisesRegex(candidate.GearSonicError, "must be positive"):
            candidate.configure_native_position_actuators(
                None, None, runtime_map, np.zeros(29, dtype=np.float64)
            )

    def test_force_limit_sources_are_explicit(self):
        runtime_map = SimpleNamespace(
            ctrl_min=-np.arange(1, 30, dtype=np.float64),
            ctrl_max=np.arange(2, 31, dtype=np.float64),
        )
        np.testing.assert_array_equal(
            candidate.resolve_force_limits(runtime_map, "public-model-config"),
            candidate.PUBLIC_EFFORT_LIMIT_MUJOCO,
        )
        np.testing.assert_array_equal(
            candidate.resolve_force_limits(runtime_map, "recovered-import-fallback"),
            np.arange(2, 31, dtype=np.float64),
        )
        with self.assertRaisesRegex(candidate.GearSonicError, "unsupported"):
            candidate.resolve_force_limits(runtime_map, "implicit")

    def test_native_command_lpf_and_joint_range_clamp(self):
        lpf_dt = 2 * candidate.PHYSICS_DT
        expected = lpf_dt / (1.0 / (2.0 * np.pi * 50.0) + lpf_dt)
        self.assertAlmostEqual(candidate.command_lpf_alpha(50.0, lpf_dt), expected)
        self.assertIsInstance(candidate.command_lpf_alpha(50.0, lpf_dt), np.float32)
        self.assertEqual(candidate.command_lpf_alpha(0.0, 0.02), 1.0)
        self.assertEqual(
            candidate.native_scheduler_interval(candidate.PHYSICS_DT, 50.0), 10
        )
        self.assertEqual(
            candidate.native_scheduler_interval(
                candidate.PHYSICS_DT, candidate.NATIVE_WORK_RATE_HZ
            ),
            2,
        )
        runtime_map = SimpleNamespace(joint_ids=np.arange(29, dtype=np.int64))
        model = SimpleNamespace(
            jnt_limited=np.ones(29, dtype=np.int32),
            jnt_range=np.column_stack(
                (np.full(29, -0.5), np.full(29, 0.5))
            ),
        )
        targets = np.linspace(-1.0, 1.0, 29)
        bounded, clipped = candidate.clip_targets_to_joint_ranges(
            model, runtime_map, targets
        )
        expected_bounded = np.clip(
            targets.astype(np.float32), np.float32(-0.5), np.float32(0.5)
        )
        self.assertEqual(bounded.dtype, np.float32)
        np.testing.assert_array_equal(bounded, expected_bounded)
        np.testing.assert_array_equal(clipped, targets.astype(np.float32) != bounded)

    def test_command_lpf_update_matches_system_single_rounding(self):
        alpha = candidate.command_lpf_alpha(50.0, 2 * candidate.PHYSICS_DT)
        initial = np.linspace(-2.7, 3.1, 29, dtype=np.float32)
        targets = np.linspace(4.3, -1.9, 29, dtype=np.float32)
        expected = np.empty(29, dtype=np.float32)
        for index in range(29):
            delta = np.float32(targets[index] - initial[index])
            increment = np.float32(alpha * delta)
            expected[index] = np.float32(initial[index] + increment)
        double_then_cast = (
            initial.astype(np.float64)
            + float(alpha)
            * (targets.astype(np.float64) - initial.astype(np.float64))
        ).astype(np.float32)
        self.assertTrue(
            np.any(expected.view(np.uint32) != double_then_cast.view(np.uint32))
        )

        state = initial.copy()
        updated = candidate.update_command_lpf(state, targets, alpha)

        self.assertIs(updated, state)
        self.assertEqual(updated.dtype, np.float32)
        np.testing.assert_array_equal(updated.view(np.uint32), expected.view(np.uint32))

    def test_float_hash_is_little_endian_float32(self):
        value = np.array([1.0, 2.0], dtype=np.float64)
        self.assertEqual(
            candidate.sha256_float32(value),
            __import__("hashlib").sha256(np.array([1.0, 2.0], dtype="<f4").tobytes()).hexdigest(),
        )

    def test_invalid_history_entry_is_rejected_before_mutating_history(self):
        history = candidate.StateHistory()
        invalid = history_entry(0.0)
        object.__setattr__(invalid, "body_q_policy", np.zeros(28))
        with self.assertRaisesRegex(candidate.GearSonicError, "length 29"):
            history.append(invalid)
        self.assertEqual(history.oldest_first(), [None] * 10)


if __name__ == "__main__":
    unittest.main()
