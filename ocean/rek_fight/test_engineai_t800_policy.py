from types import SimpleNamespace

import numpy as np

from engineai_t800_policy import (
    ACTIVE_JOINTS,
    DEFAULT_Q,
    FirstOrderLowPass,
    OFFICIAL_DEFAULT_ROOT_CLEARANCE_M,
    RECOVERY_TORQUE_LIMIT,
    SINGLE_OBSERVATION_SIZE,
    T800MuJoCoBinding,
    T800SupineRecoveryController,
    T800WalkingController,
    quaternion_matrix_wxyz,
    upright_quaternion_at_heading_wxyz,
    world_box_top_height,
)


def test_upright_orientation_projects_gravity_down():
    rotation = quaternion_matrix_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(-rotation.T @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])


def test_first_filter_value_is_unmodified():
    filter_ = FirstOrderLowPass(sample_rate=40.0, cutoff_frequency=0.1)
    value = np.array([1.0, -0.5, 0.25])
    np.testing.assert_allclose(filter_.update(value), value)
    second = filter_.update(np.zeros(3))
    assert np.all(second != 0.0)
    assert np.all(np.abs(second) < np.abs(value))


def test_observation_shape_and_active_joint_map_without_mnn():
    controller = object.__new__(T800WalkingController)
    controller.command_filter_enabled = False
    controller.command_filter = FirstOrderLowPass()
    controller.reset()
    single = controller.observe(DEFAULT_Q, np.zeros(25), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))
    assert single.shape == (SINGLE_OBSERVATION_SIZE,)
    np.testing.assert_allclose(single[:22], 0.0)
    np.testing.assert_allclose(single[-3:], [0.0, 0.0, -1.0])
    assert ACTIVE_JOINTS.tolist() == list(range(12)) + list(range(13, 23))


def test_sdk_standing_state_is_complete_and_preserves_only_arena_heading():
    binding = object.__new__(T800MuJoCoBinding)
    binding.root_qpos_address = 0
    binding.root_dof_address = 0
    binding.qpos_addresses = np.arange(7, 32)
    binding.dof_addresses = np.arange(6, 31)
    measured_quaternion = np.array(
        [
            0.9992214614231879,
            -0.010719439586156,
            0.015103461416902643,
            0.03483461065514475,
        ]
    )
    data = SimpleNamespace(qpos=np.zeros(32), qvel=np.ones(31))
    data.qpos[:3] = [-0.98583424, 0.097520486, 0.9382012]
    data.qpos[3:7] = measured_quaternion
    data.qpos[7:] = np.linspace(-1.0, 1.0, 25)

    binding.set_sdk_standing_state(data, support_height_m=0.01)

    np.testing.assert_allclose(data.qpos[:2], [-0.98583424, 0.097520486])
    assert data.qpos[2] == 0.01 + OFFICIAL_DEFAULT_ROOT_CLEARANCE_M
    np.testing.assert_allclose(
        data.qpos[3:7],
        upright_quaternion_at_heading_wxyz(measured_quaternion),
    )
    np.testing.assert_allclose(data.qpos[4:6], 0.0)
    np.testing.assert_allclose(data.qpos[7:], DEFAULT_Q)
    np.testing.assert_allclose(data.qvel, 0.0)


def test_static_rotated_box_top_height_is_computed_from_model_geometry():
    class FakeMujoco:
        mjtObj = SimpleNamespace(mjOBJ_GEOM=5)
        mjtGeom = SimpleNamespace(mjGEOM_BOX=6)

        @staticmethod
        def mj_name2id(model, kind, name):
            assert kind == 5
            return 0 if name == "floor" else -1

    model = SimpleNamespace(
        geom_bodyid=np.array([0]),
        geom_type=np.array([6]),
        geom_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_size=np.array([[2.3, 2.3, 0.06]]),
        geom_pos=np.array([[0.0, 0.0, -0.05]]),
    )
    np.testing.assert_allclose(world_box_top_height(model, FakeMujoco, "floor"), 0.01)


def test_recovery_torque_limit_is_encoded_once_in_commanded_position():
    joint_q = np.zeros(25)
    joint_qd = np.zeros(25)
    raw_target = np.zeros(25)
    raw_target[:12] = 100.0

    limited_target, initial_torque = (
        T800SupineRecoveryController.torque_limited_target(
            joint_q,
            joint_qd,
            raw_target,
        )
    )

    np.testing.assert_allclose(
        T800SupineRecoveryController.pd_torque(
            joint_q,
            joint_qd,
            limited_target,
        ),
        initial_torque,
    )
    assert np.all(np.abs(initial_torque) <= RECOVERY_TORQUE_LIMIT)
    np.testing.assert_allclose(np.abs(initial_torque[:12]).sum(), 1700.0)
    later_torque = T800SupineRecoveryController.pd_torque(
        np.full(25, -10.0),
        joint_qd,
        limited_target,
    )
    assert np.any(np.abs(later_torque) > RECOVERY_TORQUE_LIMIT)
