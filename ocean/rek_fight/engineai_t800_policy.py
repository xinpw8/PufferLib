"""EngineAI T800 walking policy adapter.

The constants and observation order in this module are transcribed from the
BSD-3-Clause EngineAI native SDK at commit 335c60e88772c26c7852d0abd6b3c7439037dd8f.
The MNN weight file is fetched from that repository and is deliberately not
vendored here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


ACTIVE_JOINTS = np.array(list(range(12)) + list(range(13, 23)), dtype=np.int32)
DEFAULT_Q = np.array(
    [
        -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
        -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
        0.0,
        0.0, 0.15, 0.0, -0.25, 0.0,
        0.0, -0.15, 0.0, -0.25, 0.0,
        0.0, 0.0,
    ],
    dtype=np.float64,
)
KP = np.array(
    [
        180, 100, 100, 180, 40, 40,
        180, 100, 100, 180, 40, 40,
        100,
        60, 50, 50, 60, 50,
        60, 50, 50, 60, 50,
        100, 100,
    ],
    dtype=np.float64,
)
KD = np.array(
    [
        5, 3, 3, 5, 0.3, 0.3,
        5, 3, 3, 5, 0.3, 0.3,
        5,
        0.3, 0.3, 0.3, 0.3, 0.3,
        0.3, 0.3, 0.3, 0.3, 0.3,
        1, 1,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.array(
    [
        0.5, 0.2, 0.2, 0.5, 0.5, 0.2,
        0.5, 0.2, 0.2, 0.5, 0.5, 0.2,
        0.2, 0.2, 0.05, 0.2, 0.05,
        0.2, 0.2, 0.05, 0.2, 0.05,
    ],
    dtype=np.float64,
)
OBSERVATION_SCALE = np.concatenate(
    (
        np.ones(22),
        np.full(22, 0.05),
        np.ones(22),
        np.ones(3),
        np.ones(3),
    )
).astype(np.float64)
COMMAND_OBSERVATION_SCALE = np.array([2.0, 2.0, 1.0], dtype=np.float64)
COMMAND_SCALE_POS = np.array([1.0, 0.4, 1.0], dtype=np.float64)
COMMAND_SCALE_NEG = np.array([0.6, 0.4, 1.0], dtype=np.float64)
OBSERVATION_CLIP = 100.0
ACTION_CLIP = 100.0
CONTROL_DT = 0.01
HISTORY_STEPS = 15
SINGLE_OBSERVATION_SIZE = 72
POLICY_INPUT_SIZE = HISTORY_STEPS * SINGLE_OBSERVATION_SIZE + 3
POLICY_OUTPUT_SIZE = 22


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return the 3x3 local-to-world rotation for a MuJoCo wxyz quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class FirstOrderLowPass:
    """Exact EngineAI first-order command filter."""

    def __init__(self, sample_rate: float = 40.0, cutoff_frequency: float = 0.1):
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * math.pi * cutoff_frequency)
        self.alpha = dt / (dt + rc)
        self.output: np.ndarray | None = None

    def reset(self) -> None:
        self.output = None

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.output is None:
            self.output = value.copy()
        else:
            self.output = self.alpha * value + (1.0 - self.alpha) * self.output
        return self.output.copy()


class MNNPolicy:
    def __init__(self, model_path: Path, threads: int = 1, input_size: int = POLICY_INPUT_SIZE, output_size: int = POLICY_OUTPUT_SIZE):
        import MNN

        self.MNN = MNN
        self.interpreter = MNN.Interpreter(str(model_path))
        self.session = self.interpreter.createSession({"numThread": threads})
        inputs = self.interpreter.getSessionInputAll(self.session)
        outputs = self.interpreter.getSessionOutputAll(self.session)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"expected one MNN input and output, got {list(inputs)} and {list(outputs)}")
        self.input = next(iter(inputs.values()))
        self.output = next(iter(outputs.values()))
        self.input_size = input_size
        self.output_size = output_size
        if list(self.input.getShape()) != [1, input_size]:
            raise ValueError(f"unexpected policy input shape: {self.input.getShape()}")
        if list(self.output.getShape()) != [1, output_size]:
            raise ValueError(f"unexpected policy output shape: {self.output.getShape()}")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, self.input_size)
        host_input = self.MNN.Tensor(
            [1, self.input_size],
            self.MNN.Halide_Type_Float,
            tuple(float(value) for value in observation.ravel()),
            self.MNN.Tensor_DimensionType_Caffe,
        )
        self.input.copyFrom(host_input)
        self.interpreter.runSession(self.session)
        host_output = self.MNN.Tensor(
            [1, self.output_size],
            self.MNN.Halide_Type_Float,
            tuple(0.0 for _ in range(self.output_size)),
            self.MNN.Tensor_DimensionType_Caffe,
        )
        self.output.copyToHostTensor(host_output)
        return np.asarray(host_output.getData(), dtype=np.float64).reshape(self.output_size)


class T800WalkingController:
    """Stateful adapter for the official T800 walking MNN policy."""

    def __init__(self, model_path: Path, *, command_filter: bool = True):
        self.policy = MNNPolicy(model_path)
        self.command_filter_enabled = command_filter
        self.command_filter = FirstOrderLowPass()
        self.reset()

    def reset(self) -> None:
        self.history = np.zeros((HISTORY_STEPS, SINGLE_OBSERVATION_SIZE), dtype=np.float64)
        self.previous_action = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float64)
        self.first_observation = True
        self.command_filter.reset()

    def scale_command(self, normalized_command: np.ndarray) -> np.ndarray:
        raw = np.clip(np.asarray(normalized_command, dtype=np.float64), -1.0, 1.0)
        command = raw * np.where(raw >= 0.0, COMMAND_SCALE_POS, COMMAND_SCALE_NEG)
        if self.command_filter_enabled:
            command = self.command_filter.update(command)
        return command

    def observe(
        self,
        joint_q: np.ndarray,
        joint_qd: np.ndarray,
        body_quaternion_wxyz: np.ndarray,
        body_angular_velocity: np.ndarray,
    ) -> np.ndarray:
        rotation = quaternion_matrix_wxyz(body_quaternion_wxyz)
        projected_gravity = -rotation.T @ np.array([0.0, 0.0, 1.0])
        single = np.concatenate(
            (
                np.asarray(joint_q)[ACTIVE_JOINTS] - DEFAULT_Q[ACTIVE_JOINTS],
                np.asarray(joint_qd)[ACTIVE_JOINTS],
                self.previous_action,
                np.asarray(body_angular_velocity, dtype=np.float64),
                projected_gravity,
            )
        )
        if single.shape != (SINGLE_OBSERVATION_SIZE,):
            raise ValueError(f"unexpected single observation shape: {single.shape}")
        single = np.clip(single * OBSERVATION_SCALE, -OBSERVATION_CLIP, OBSERVATION_CLIP)
        if self.first_observation:
            self.history[:] = single
            self.previous_action[:] = 0.0
            self.first_observation = False
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = single
        return single

    def act(self, command_metres_radians_per_second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy_input = np.concatenate(
            (self.history.reshape(-1), np.asarray(command_metres_radians_per_second) * COMMAND_OBSERVATION_SCALE)
        )
        action = np.clip(self.policy(policy_input), -ACTION_CLIP, ACTION_CLIP)
        self.previous_action = action
        target_q = DEFAULT_Q.copy()
        target_q[ACTIVE_JOINTS] += action * ACTION_SCALE
        return action, target_q

    @staticmethod
    def pd_torque(joint_q: np.ndarray, joint_qd: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        return KP * (np.asarray(target_q) - np.asarray(joint_q)) - KD * np.asarray(joint_qd)


class T800MuJoCoBinding:
    """Name-derived state and actuator binding for one T800 in a MuJoCo model."""

    def __init__(self, mujoco_module, model, fighter: int | None = None):
        self.mujoco = mujoco_module
        self.model = model
        prefix = "" if fighter is None else f"fighter_{fighter}__"

        def matching_id(kind, count: int, fragment: str) -> int:
            matches = []
            for index in range(count):
                name = mujoco_module.mj_id2name(model, kind, index) or ""
                if name.startswith(prefix) and fragment in name:
                    matches.append(index)
            if len(matches) != 1:
                raise ValueError(f"expected one {prefix}{fragment} match, got {matches}")
            return matches[0]

        root_joint = matching_id(mujoco_module.mjtObj.mjOBJ_JOINT, model.njnt, "LINK_BASE_freejoint")
        self.root_qpos_address = int(model.jnt_qposadr[root_joint])
        self.root_dof_address = int(model.jnt_dofadr[root_joint])
        self.root_body = matching_id(mujoco_module.mjtObj.mjOBJ_BODY, model.nbody, "LINK_BASE")
        joint_ids = [
            matching_id(mujoco_module.mjtObj.mjOBJ_JOINT, model.njnt, f"J{index:02d}_")
            for index in range(25)
        ]
        self.qpos_addresses = np.array([model.jnt_qposadr[index] for index in joint_ids], dtype=np.int32)
        self.dof_addresses = np.array([model.jnt_dofadr[index] for index in joint_ids], dtype=np.int32)
        self.actuator_ids = np.array(
            [
                matching_id(mujoco_module.mjtObj.mjOBJ_ACTUATOR, model.nu, f"motor_J{index:02d}_")
                for index in range(25)
            ],
            dtype=np.int32,
        )

    def state(self, data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            data.qpos[self.qpos_addresses].copy(),
            data.qvel[self.dof_addresses].copy(),
            data.qpos[self.root_qpos_address + 3 : self.root_qpos_address + 7].copy(),
            data.qvel[self.root_dof_address + 3 : self.root_dof_address + 6].copy(),
        )

    def set_default_pose(self, data) -> None:
        data.qpos[self.qpos_addresses] = DEFAULT_Q

    def apply_torque(self, data, torque: np.ndarray) -> None:
        data.ctrl[self.actuator_ids] = torque

    def root_position(self, data) -> np.ndarray:
        return data.qpos[self.root_qpos_address : self.root_qpos_address + 3].copy()

    def root_up_z(self, data) -> float:
        return float(data.xmat[self.root_body, 8])


RECOVERY_KP = np.array(
    [
        280.4, 95.3, 34.3, 280.4, 12.6, 12.6,
        280.4, 95.3, 34.3, 280.4, 12.6, 12.6,
        34.3,
        22.4, 22.4, 22.4, 22.4, 4.2,
        22.4, 22.4, 22.4, 22.4, 4.2,
        2.4, 2.4,
    ],
    dtype=np.float64,
)
RECOVERY_KD = np.array(
    [
        41.1, 21.1, 4.0, 41.1, 1.5, 1.5,
        41.1, 21.1, 4.0, 41.1, 1.5, 1.5,
        4.0,
        2.0, 2.0, 2.0, 2.0, 0.4,
        2.0, 2.0, 2.0, 2.0, 0.4,
        0.3, 0.3,
    ],
    dtype=np.float64,
)
RECOVERY_QD_MASK = np.array(
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    dtype=np.float64,
)
RECOVERY_TORQUE_LIMIT = np.array(
    [
        415, 370, 222, 415, 160, 160,
        415, 370, 222, 415, 160, 160,
        222,
        160, 160, 160, 160, 52,
        160, 160, 160, 160, 52,
        52, 52,
    ],
    dtype=np.float64,
)
RECOVERY_HISTORY_STEPS = 5
RECOVERY_INPUT_SIZE = 430
RECOVERY_TRAJECTORY_SHA256 = "c2f19c164093701311634024eb27999fed4631a00d38d507f8aa306ee138c161"
RECOVERY_POLICY_SHA256 = "deb9974b1f4f4a7e77801f8c9c6e77f599caab0ca4dd7709fe0bae55870e0e86"


def load_supine_recovery_trajectory(path: Path) -> np.ndarray:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != RECOVERY_TRAJECTORY_SHA256:
        raise ValueError(f"unexpected recovery trajectory SHA-256: {digest}")
    source = np.load(path, allow_pickle=False)
    if source.shape != (197, 34):
        raise ValueError(f"unexpected recovery trajectory shape: {source.shape}")
    clipped = source[90:181, 7:32].astype(np.float64)
    source_dt = 0.033333
    target_dt = CONTROL_DT
    target_count = int(round((len(clipped) - 1) * source_dt / target_dt)) + 1
    target = np.empty((target_count, 25), dtype=np.float64)
    for frame in range(target_count):
        source_position = frame * target_dt / source_dt
        lower = min(int(math.floor(source_position)), len(clipped) - 1)
        upper = min(lower + 1, len(clipped) - 1)
        alpha = source_position - lower
        target[frame] = (1.0 - alpha) * clipped[lower] + alpha * clipped[upper]
    return target


class T800SupineRecoveryController:
    """Official EngineAI supine-to-stance residual policy and reference motion."""

    def __init__(self, policy_path: Path, trajectory_path: Path):
        import hashlib

        digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        if digest != RECOVERY_POLICY_SHA256:
            raise ValueError(f"unexpected recovery policy SHA-256: {digest}")
        self.policy = MNNPolicy(policy_path, input_size=RECOVERY_INPUT_SIZE, output_size=25)
        self.trajectory = load_supine_recovery_trajectory(trajectory_path)
        self.reset(np.zeros(25, dtype=np.float64))

    def reset(self, initial_joint_q: np.ndarray) -> None:
        self.histories = [
            np.zeros((25, RECOVERY_HISTORY_STEPS), dtype=np.float64),
            np.zeros((25, RECOVERY_HISTORY_STEPS), dtype=np.float64),
            np.zeros((25, RECOVERY_HISTORY_STEPS), dtype=np.float64),
            np.zeros((3, RECOVERY_HISTORY_STEPS), dtype=np.float64),
            np.zeros((3, RECOVERY_HISTORY_STEPS), dtype=np.float64),
        ]
        self.previous_action = np.zeros(25, dtype=np.float64)
        self.trajectory_index = 0
        self.transition_iteration = 0
        self.initial_joint_q = np.asarray(initial_joint_q, dtype=np.float64).copy()

    @property
    def finished(self) -> bool:
        return self.trajectory_index >= len(self.trajectory) - 1

    def step(
        self,
        joint_q: np.ndarray,
        joint_qd: np.ndarray,
        body_quaternion_wxyz: np.ndarray,
        body_angular_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_q = np.asarray(joint_q, dtype=np.float64)
        joint_qd = np.asarray(joint_qd, dtype=np.float64)
        rotation = quaternion_matrix_wxyz(body_quaternion_wxyz)
        gravity = -rotation.T @ np.array([0.0, 0.0, 1.0])
        updates = [joint_q, joint_qd * RECOVERY_QD_MASK, self.previous_action, body_angular_velocity, gravity]
        for history, update in zip(self.histories, updates):
            history[:, :-1] = history[:, 1:]
            history[:, -1] = update
        proprioception = np.concatenate([history.T.reshape(-1) for history in self.histories])
        proprioception[125:250] *= 0.05
        reference = self.trajectory[self.trajectory_index]
        goal = joint_q - reference
        observation = np.clip(np.concatenate((proprioception, goal)), -100.0, 100.0)
        action = np.clip(self.policy(observation), -100.0, 100.0)
        self.previous_action = action
        if self.trajectory_index < len(self.trajectory) - 1:
            self.trajectory_index += 1
        target = self.trajectory[self.trajectory_index] + action * 0.25
        ratio = min(1.0, self.transition_iteration * CONTROL_DT / 0.3)
        if ratio < 1.0:
            target = ratio * target + (1.0 - ratio) * self.initial_joint_q
            self.transition_iteration += 1
        torque = RECOVERY_KP * (target - joint_q) - RECOVERY_KD * joint_qd
        torque = np.clip(torque, -RECOVERY_TORQUE_LIMIT, RECOVERY_TORQUE_LIMIT)
        lower_sum = float(np.abs(torque[:12]).sum())
        if lower_sum > 1700.0:
            torque[:12] *= 1700.0 / lower_sum
        return target, torque

    @staticmethod
    def pd_torque(joint_q: np.ndarray, joint_qd: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        torque = RECOVERY_KP * (np.asarray(target_q) - np.asarray(joint_q)) - RECOVERY_KD * np.asarray(joint_qd)
        torque = np.clip(torque, -RECOVERY_TORQUE_LIMIT, RECOVERY_TORQUE_LIMIT)
        lower_sum = float(np.abs(torque[:12]).sum())
        if lower_sum > 1700.0:
            torque[:12] *= 1700.0 / lower_sum
        return torque
