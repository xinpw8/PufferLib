import numpy as np

from engineai_t800_policy import (
    ACTIVE_JOINTS,
    DEFAULT_Q,
    FirstOrderLowPass,
    SINGLE_OBSERVATION_SIZE,
    T800WalkingController,
    quaternion_matrix_wxyz,
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
