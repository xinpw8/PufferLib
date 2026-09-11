"""Run the pinned public GEAR-SONIC controller against the recovered REK G1 plant.

This is a controller-family candidate, not a REK parity result.  The public
encoder and decoder match the model family and observation names recovered
from the pinned REK build.  Exact packaged-weight identity and trajectory
parity still require live REK controller traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gear_sonic_bundle as bundle_contract
import sonic_candidate as plant_contract
import sonic_motion_composer as motion_composer


RUN_SCHEMA = "rek.g1_gear_sonic_candidate.run.v1"
TRACE_SCHEMA = "rek.g1_gear_sonic_candidate.trace.v1"
CLASSIFICATION = "public_family_candidate"
CONTROL_HZ = 50
CONTROL_DT = 0.02
PHYSICS_HZ = 500
PHYSICS_DT = 0.002
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ
NATIVE_WORK_RATE_HZ = 200.0
HISTORY_FRAMES = 10
ENCODER_DIM = 1762
DECODER_DIM = 994
TOKEN_DIM = 64
ACTION_DIM = 29
ACTION_CLIP = 100.0
FUTURE_STEP = 5
COMMAND_LPF_CUTOFF_HZ = 50.0

# Each value at MuJoCo index i selects the corresponding IsaacLab policy index.
ISAACLAB_TO_MUJOCO = np.asarray(
    [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
        11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int64,
)

# Each value at IsaacLab policy index i selects the corresponding MuJoCo index.
MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)

DEFAULT_ANGLES_MUJOCO = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

PUBLIC_EFFORT_LIMIT_MUJOCO = np.asarray(
    [
        139.0, 139.0, 88.0, 139.0, 25.0, 25.0,
        139.0, 139.0, 88.0, 139.0, 25.0, 25.0,
        88.0, 25.0, 25.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ],
    dtype=np.float64,
)

ACTION_SCALE_MUJOCO = np.asarray(
    [
        0.350661, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
        0.350661, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
        0.547546, 0.438577, 0.438577,
        0.438577, 0.438577, 0.438577, 0.438577, 0.438577, 0.074501, 0.074501,
        0.438577, 0.438577, 0.438577, 0.438577, 0.438577, 0.074501, 0.074501,
    ],
    dtype=np.float32,
)

KP_MUJOCO = np.asarray(
    [
        99.098428, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
        99.098428, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
        40.179238, 28.501246, 28.501246,
        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
    ],
    dtype=np.float64,
)

KD_MUJOCO = np.asarray(
    [
        6.308802, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
        6.308802, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
        2.55789, 1.814446, 1.814446,
        0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
        0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
    ],
    dtype=np.float64,
)

ENCODER_OFFSETS = {
    "encoder_mode_4": (0, 4),
    "motion_joint_positions_10frame_step5": (4, 294),
    "motion_joint_velocities_10frame_step5": (294, 584),
    "motion_root_z_position_10frame_step5": (584, 594),
    "motion_root_z_position": (594, 595),
    "motion_anchor_orientation": (595, 601),
    "motion_anchor_orientation_10frame_step5": (601, 661),
    "motion_joint_positions_lowerbody_10frame_step5": (661, 781),
    "motion_joint_velocities_lowerbody_10frame_step5": (781, 901),
    "vr_3point_local_target": (901, 910),
    "vr_3point_local_orn_target": (910, 922),
    "smpl_joints_10frame_step1": (922, 1642),
    "smpl_anchor_orientation_10frame_step1": (1642, 1702),
    "motion_joint_positions_wrists_10frame_step1": (1702, 1762),
}

DECODER_OFFSETS = {
    "token_state": (0, 64),
    "his_base_angular_velocity_10frame_step1": (64, 94),
    "his_body_joint_positions_10frame_step1": (94, 384),
    "his_body_joint_velocities_10frame_step1": (384, 674),
    "his_last_actions_10frame_step1": (674, 964),
    "his_gravity_dir_10frame_step1": (964, 994),
}


class GearSonicError(RuntimeError):
    """A pinned artifact or candidate runtime invariant failed."""


@dataclass(frozen=True)
class HistoryEntry:
    base_quat_wxyz: np.ndarray
    base_ang_vel: np.ndarray
    body_q_policy: np.ndarray
    body_dq_policy: np.ndarray
    last_action_policy: np.ndarray


class StateHistory:
    """REK/public deployment history with oldest-first, zero-left padding."""

    def __init__(self, capacity: int = HISTORY_FRAMES):
        if capacity < HISTORY_FRAMES:
            raise GearSonicError("history capacity must be at least 10")
        self.capacity = capacity
        self._entries: list[HistoryEntry] = []

    def reset(self) -> None:
        self._entries.clear()

    def append(self, entry: HistoryEntry) -> None:
        _validate_history_entry(entry)
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            del self._entries[: len(self._entries) - self.capacity]

    def oldest_first(self, count: int = HISTORY_FRAMES) -> list[HistoryEntry | None]:
        if count < 1:
            raise GearSonicError("history request must be positive")
        present = self._entries[-count:]
        return [None] * (count - len(present)) + present


def _validate_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,) or not np.issubdtype(array.dtype, np.number):
        raise GearSonicError(f"{label} must be a numeric vector of length {size}")
    if not np.isfinite(array).all():
        raise GearSonicError(f"{label} contains non-finite values")
    return array


def _validate_history_entry(entry: HistoryEntry) -> None:
    _validate_vector(entry.base_quat_wxyz, 4, "history base quaternion")
    _validate_vector(entry.base_ang_vel, 3, "history base angular velocity")
    _validate_vector(entry.body_q_policy, ACTION_DIM, "history joint position")
    _validate_vector(entry.body_dq_policy, ACTION_DIM, "history joint velocity")
    _validate_vector(entry.last_action_policy, ACTION_DIM, "history action")


def sha256_float32(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _quat_conjugate_wxyz(value: np.ndarray) -> np.ndarray:
    q = _validate_vector(value, 4, "quaternion").astype(np.float64, copy=True)
    q[1:] *= -1.0
    return q


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    l = _validate_vector(left, 4, "left quaternion").astype(np.float64, copy=False)
    r = _validate_vector(right, 4, "right quaternion").astype(np.float64, copy=False)
    lw, lx, ly, lz = l
    rw, rx, ry, rz = r
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_rotate_wxyz(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Match the public C++ helper, including its zero-quaternion behavior."""
    q = _validate_vector(q, 4, "rotation quaternion").astype(np.float64, copy=False)
    vector = _validate_vector(vector, 3, "rotation vector").astype(
        np.float64, copy=False
    )
    qw = q[0]
    qv = q[1:]
    return (
        vector * (2.0 * qw * qw - 1.0)
        + np.cross(qv, vector) * qw * 2.0
        + qv * np.dot(qv, vector) * 2.0
    )


def _heading_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = _validate_vector(q, 4, "heading quaternion").astype(np.float64, copy=False)
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def _rotation_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    q = _validate_vector(q, 4, "matrix quaternion").astype(np.float64, copy=False)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    return _quat_rotate_wxyz(
        _quat_conjugate_wxyz(base_quat_wxyz),
        np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
    )


def reference_heading_delta(
    initial_base_quat_wxyz: np.ndarray,
    initial_reference_quat_wxyz: np.ndarray,
) -> np.ndarray:
    return _quat_multiply_wxyz(
        _heading_quat_wxyz(initial_base_quat_wxyz),
        _quat_conjugate_wxyz(_heading_quat_wxyz(initial_reference_quat_wxyz)),
    )


def build_encoder_observation(
    motion: plant_contract.MotionData,
    current_frame: int,
    base_quat_wxyz: np.ndarray,
    heading_delta_wxyz: np.ndarray,
    *,
    loop: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mode-0 G1 encoder input in the pinned 1762-value layout."""
    if isinstance(current_frame, bool) or not isinstance(current_frame, int):
        raise GearSonicError("current frame must be an integer")
    frames = int(motion.dof_pos.shape[0])
    if current_frame < 0 or current_frame >= frames:
        raise GearSonicError("current frame is outside the motion clip")
    if motion.dof_pos.shape != (frames, ACTION_DIM):
        raise GearSonicError("motion joint positions have the wrong shape")
    if motion.dof_vel.shape != (frames, ACTION_DIM):
        raise GearSonicError("motion joint velocities have the wrong shape")
    if motion.root_rot_xyzw.shape != (frames, 4):
        raise GearSonicError("motion root rotations have the wrong shape")
    base = _validate_vector(base_quat_wxyz, 4, "base quaternion")
    delta = _validate_vector(heading_delta_wxyz, 4, "heading delta")
    raw_indices = current_frame + np.arange(
        HISTORY_FRAMES, dtype=np.int64
    ) * FUTURE_STEP
    if loop:
        indices = raw_indices % frames
        next_indices = (indices + 1) % frames
    else:
        indices = np.minimum(raw_indices, frames - 1)
        next_indices = np.minimum(indices + 1, frames - 1)
    reference_velocity_mujoco = (
        motion.dof_pos[next_indices].astype(np.float64)
        - motion.dof_pos[indices].astype(np.float64)
    ) / CONTROL_DT

    observation = np.zeros(ENCODER_DIM, dtype=np.float32)
    # The pinned public GEAR-SONIC deployment stores GetEncodeMode() as one
    # scalar followed by three zeros. G1 is mode zero, so the complete
    # encoder_mode_4 field remains zero. This is not a one-hot field.
    # REK's packaged NPZ rows follow the native motor/MuJoCo order.  The
    # encoder model consumes IsaacLab policy order, just like the public
    # deployment's MotionSequence.  Initialization remains in native order.
    observation[4:294] = motion.dof_pos[indices][:, MUJOCO_TO_ISAACLAB].reshape(-1)
    observation[294:584] = reference_velocity_mujoco[
        :, MUJOCO_TO_ISAACLAB
    ].reshape(-1)

    rotations = np.empty((HISTORY_FRAMES, 6), dtype=np.float32)
    for output_index, frame in enumerate(indices):
        reference_wxyz = plant_contract._xyzw_to_wxyz(
            motion.root_rot_xyzw[int(frame)]
        ).astype(np.float64)
        aligned_reference = _quat_multiply_wxyz(delta, reference_wxyz)
        relative = _quat_multiply_wxyz(
            _quat_conjugate_wxyz(base), aligned_reference
        )
        rotations[output_index] = _rotation_matrix_wxyz(relative)[:, :2].reshape(-1)
    observation[601:661] = rotations.reshape(-1)
    if not np.isfinite(observation).all():
        raise GearSonicError("encoder observation contains non-finite values")
    return observation, indices


def build_decoder_observation(
    token: np.ndarray,
    history: StateHistory,
) -> np.ndarray:
    token = np.asarray(token)
    if token.shape == (1, TOKEN_DIM):
        token = token[0]
    token = _validate_vector(token, TOKEN_DIM, "encoder token")
    entries = history.oldest_first(HISTORY_FRAMES)
    observation = np.zeros(DECODER_DIM, dtype=np.float32)
    observation[0:64] = token
    angular = np.zeros((HISTORY_FRAMES, 3), dtype=np.float32)
    positions = np.zeros((HISTORY_FRAMES, ACTION_DIM), dtype=np.float32)
    velocities = np.zeros((HISTORY_FRAMES, ACTION_DIM), dtype=np.float32)
    actions = np.zeros((HISTORY_FRAMES, ACTION_DIM), dtype=np.float32)
    gravity = np.zeros((HISTORY_FRAMES, 3), dtype=np.float32)
    for index, entry in enumerate(entries):
        if entry is None:
            continue
        angular[index] = entry.base_ang_vel
        positions[index] = entry.body_q_policy
        velocities[index] = entry.body_dq_policy
        actions[index] = entry.last_action_policy
        gravity[index] = projected_gravity(entry.base_quat_wxyz)
    observation[64:94] = angular.reshape(-1)
    observation[94:384] = positions.reshape(-1)
    observation[384:674] = velocities.reshape(-1)
    observation[674:964] = actions.reshape(-1)
    observation[964:994] = gravity.reshape(-1)
    if not np.isfinite(observation).all():
        raise GearSonicError("decoder observation contains non-finite values")
    return observation


def state_to_history_entry(
    mujoco: Any,
    model: Any,
    data: Any,
    runtime_map: plant_contract.RuntimeMap,
    last_action_policy: np.ndarray,
) -> HistoryEntry:
    q_mujoco = np.asarray(
        data.qpos[runtime_map.qpos_addresses], dtype=np.float64
    )
    dq_mujoco = np.asarray(
        data.qvel[runtime_map.qvel_addresses], dtype=np.float64
    )
    root = runtime_map.root_qpos_address
    object_velocity_local = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        int(runtime_map.root_body_id),
        object_velocity_local,
        1,
    )
    entry = HistoryEntry(
        base_quat_wxyz=np.asarray(data.qpos[root + 3 : root + 7], dtype=np.float32).copy(),
        base_ang_vel=np.asarray(object_velocity_local[:3], dtype=np.float32).copy(),
        body_q_policy=np.asarray(
            (q_mujoco - DEFAULT_ANGLES_MUJOCO)[MUJOCO_TO_ISAACLAB],
            dtype=np.float32,
        ),
        body_dq_policy=np.asarray(
            dq_mujoco[MUJOCO_TO_ISAACLAB], dtype=np.float32
        ),
        last_action_policy=np.asarray(last_action_policy, dtype=np.float32).copy(),
    )
    _validate_history_entry(entry)
    return entry


def action_to_targets(raw_action_policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = _validate_vector(raw_action_policy, ACTION_DIM, "raw policy action").astype(
        np.float32, copy=False
    )
    clip = np.float32(ACTION_CLIP)
    clipped = np.clip(raw, -clip, clip)
    scaled = np.empty(ACTION_DIM, dtype=np.float32)
    np.multiply(
        clipped[ISAACLAB_TO_MUJOCO], ACTION_SCALE_MUJOCO, out=scaled
    )
    targets = np.empty(ACTION_DIM, dtype=np.float32)
    np.add(DEFAULT_ANGLES_MUJOCO, scaled, out=targets)
    return clipped, targets


def pd_torque(
    q_mujoco: np.ndarray,
    dq_mujoco: np.ndarray,
    targets_mujoco: np.ndarray,
    ctrl_min: np.ndarray,
    ctrl_max: np.ndarray,
    torque_limit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = _validate_vector(q_mujoco, ACTION_DIM, "joint position")
    dq = _validate_vector(dq_mujoco, ACTION_DIM, "joint velocity")
    targets = _validate_vector(targets_mujoco, ACTION_DIM, "joint target")
    raw = KP_MUJOCO * (targets - q) - KD_MUJOCO * dq
    if torque_limit == "declared-effort":
        applied = np.clip(raw, ctrl_min, ctrl_max)
    elif torque_limit == "unbounded":
        applied = raw.copy()
    else:
        raise GearSonicError(f"unsupported torque limit mode {torque_limit!r}")
    return raw, applied, raw != applied


def configure_native_position_actuators(
    mujoco: Any,
    model: Any,
    runtime_map: plant_contract.RuntimeMap,
    force_limits: np.ndarray,
) -> np.ndarray:
    """Recreate REK Robot.WritePositionActuatorGains on the recovered model.

    The recovered XML serializes the runtime-managed actuators as motors.  REK
    mutates the compiled MuJoCo gain, bias, and force-limit arrays before each
    step, then writes a position target to data.ctrl.  Reconstructing those
    arrays is required; treating data.ctrl as torque is not the native boundary.
    """
    force_limits = np.asarray(force_limits, dtype=np.float64)
    if force_limits.shape != (ACTION_DIM,) or not np.all(np.isfinite(force_limits)):
        raise GearSonicError("recovered actuator force limits are invalid")
    if np.any(force_limits <= 0.0):
        raise GearSonicError("recovered actuator force limits must be positive")
    for index, actuator_id in enumerate(runtime_map.actuator_ids):
        actuator_id = int(actuator_id)
        model.actuator_dyntype[actuator_id] = int(mujoco.mjtDyn.mjDYN_NONE)
        model.actuator_gaintype[actuator_id] = int(mujoco.mjtGain.mjGAIN_FIXED)
        model.actuator_biastype[actuator_id] = int(mujoco.mjtBias.mjBIAS_AFFINE)
        model.actuator_gainprm[actuator_id, 0] = KP_MUJOCO[index]
        model.actuator_biasprm[actuator_id, 0] = 0.0
        model.actuator_biasprm[actuator_id, 1] = -KP_MUJOCO[index]
        model.actuator_biasprm[actuator_id, 2] = -KD_MUJOCO[index]
        model.actuator_ctrllimited[actuator_id] = 0
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id, 0] = -force_limits[index]
        model.actuator_forcerange[actuator_id, 1] = force_limits[index]
    return force_limits


def resolve_force_limits(
    runtime_map: plant_contract.RuntimeMap, source: str
) -> np.ndarray:
    if source == "public-model-config":
        return PUBLIC_EFFORT_LIMIT_MUJOCO.copy()
    if source == "recovered-import-fallback":
        return np.maximum(
            np.abs(np.asarray(runtime_map.ctrl_min, dtype=np.float64)),
            np.abs(np.asarray(runtime_map.ctrl_max, dtype=np.float64)),
        )
    raise GearSonicError(f"unsupported force limit source {source!r}")


def command_lpf_alpha(cutoff_hz: float, dt: float) -> np.float32:
    if not math.isfinite(cutoff_hz) or not math.isfinite(dt):
        raise GearSonicError("command LPF arguments must be finite")
    cutoff = np.float32(cutoff_hz)
    step_dt = np.float32(dt)
    if not np.isfinite(cutoff) or not np.isfinite(step_dt):
        raise GearSonicError("command LPF arguments exceed System.Single range")
    if cutoff <= np.float32(0.0) or step_dt <= np.float32(0.0):
        return np.float32(1.0)
    two_pi = np.float32(np.float32(2.0) * np.float32(math.pi))
    tau = np.float32(np.float32(1.0) / np.float32(two_pi * cutoff))
    return np.float32(step_dt / np.float32(tau + step_dt))


def update_command_lpf(
    state: np.ndarray, targets: np.ndarray, alpha: float
) -> np.ndarray:
    """Apply one native System.Single command-filter update in place."""
    current = _validate_vector(state, ACTION_DIM, "command LPF state").astype(
        np.float32, copy=False
    )
    target = _validate_vector(targets, ACTION_DIM, "command LPF target").astype(
        np.float32, copy=False
    )
    coefficient = np.float32(alpha)
    if not np.isfinite(coefficient):
        raise GearSonicError("command LPF coefficient must be finite")
    delta = np.empty(ACTION_DIM, dtype=np.float32)
    np.subtract(target, current, out=delta)
    np.multiply(delta, coefficient, out=delta)
    np.add(current, delta, out=current)
    return current


def native_scheduler_interval(fixed_dt: float, work_rate_hz: float) -> int:
    """Match SonicPolicyRunner's ties-to-even Convert.ToInt32 scheduler."""
    if not math.isfinite(fixed_dt) or not math.isfinite(work_rate_hz):
        raise GearSonicError("native scheduler arguments must be finite")
    if fixed_dt <= 0.0 or work_rate_hz <= 0.0:
        raise GearSonicError("native scheduler arguments must be positive")
    fixed_rate_hz = round(1.0 / fixed_dt)
    return max(1, round(fixed_rate_hz / work_rate_hz))


def clip_targets_to_joint_ranges(
    model: Any,
    runtime_map: plant_contract.RuntimeMap,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = _validate_vector(targets, ACTION_DIM, "filtered joint target").astype(
        np.float32, copy=True
    )
    clipped = np.zeros(ACTION_DIM, dtype=np.bool_)
    for index, joint_id in enumerate(runtime_map.joint_ids):
        joint_id = int(joint_id)
        if int(model.jnt_limited[joint_id]) == 0:
            continue
        low = np.float32(model.jnt_range[joint_id, 0])
        high = np.float32(model.jnt_range[joint_id, 1])
        bounded = np.minimum(np.maximum(target[index], low), high)
        clipped[index] = bounded != target[index]
        target[index] = bounded
    return target, clipped


def _create_session(path: Path) -> tuple[Any, str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise GearSonicError("onnxruntime is required") from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    if hasattr(options, "use_deterministic_compute"):
        options.use_deterministic_compute = True
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session, str(getattr(ort, "__version__", "unknown"))


def _session_output(session: Any, output_name: str, observation: np.ndarray) -> np.ndarray:
    result = session.run(
        [output_name], {"obs_dict": np.ascontiguousarray(observation[None], dtype=np.float32)}
    )[0]
    array = np.asarray(result)
    expected = (1, TOKEN_DIM if output_name == "encoded_tokens" else ACTION_DIM)
    if array.shape != expected or array.dtype != np.float32 or not np.isfinite(array).all():
        raise GearSonicError(
            f"{output_name} has invalid dtype, shape, or finite values: {array.dtype} {array.shape}"
        )
    return np.ascontiguousarray(array[0])


def _write_new(path: Path, text: str, label: str) -> Path:
    resolved = Path(os.path.abspath(path))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise GearSonicError(f"{label} already exists; refusing to overwrite") from exc
    return resolved


def _default_manifest() -> Path:
    return Path(__file__).resolve().parents[1] / "rek" / "evidence" / "g1_runtime_assets.v1.json"


def _default_motion_contract() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rek"
        / "evidence"
        / "g1_motion_clip_contract.v1.json"
    )


def _default_xml() -> Path:
    return Path(__file__).resolve().parents[1] / "rek" / "evidence" / "evidence_out" / "g1_29dof.recovered.xml"


def _default_arena() -> Path:
    return Path(__file__).resolve().parents[1] / "rek" / "evidence" / "evidence_out" / "g1_arena_physics_contract.v1.json"


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


class MotionComposerCursorProvider:
    """Resolve one exact contract clip at the native 50 Hz composer boundary."""

    def __init__(
        self,
        motion: plant_contract.MotionData,
        motion_role: str,
        frame_mode: str,
        contract_path: Path | None = None,
    ) -> None:
        self.contract_path = (
            _default_motion_contract()
            if contract_path is None
            else Path(contract_path).resolve()
        )
        try:
            measured = motion_composer.load_contract_clip(
                self.contract_path, motion_role
            )
        except (KeyError, OSError, ValueError) as exc:
            raise GearSonicError(
                f"could not load exact motion contract for role {motion_role!r}"
            ) from exc

        actual_frames = int(motion.dof_pos.shape[0])
        expected = measured.clip
        if motion.role != measured.role:
            raise GearSonicError("loaded motion role differs from composer contract")
        if motion.filename != expected.filename:
            raise GearSonicError("loaded motion filename differs from composer contract")
        if motion.sha256.lower() != expected.sha256.lower():
            raise GearSonicError("loaded motion hash differs from composer contract")
        if actual_frames != expected.frame_count:
            raise GearSonicError("loaded motion frame count differs from composer contract")
        if np.float32(motion.fps) != np.float32(expected.fps):
            raise GearSonicError("loaded motion rate differs from composer contract")

        expected_mode = "loop" if measured.config.loop else "clamp"
        if frame_mode != expected_mode:
            raise GearSonicError(
                f"frame mode {frame_mode!r} conflicts with exact clip mode "
                f"{expected_mode!r}"
            )

        layer = motion_composer.Layer()
        motion_composer.install_contract_clip(
            layer, measured, controller_rate_hz=CONTROL_HZ
        )
        if layer.mirror:
            raise GearSonicError(
                "mirrored composer sampling is unavailable in vector sampling"
            )
        if layer.start_frame != 0 or layer.end_frame != actual_frames - 1:
            raise GearSonicError(
                "partial composer frame windows are unavailable in vector sampling"
            )
        if layer.per_tick != np.float32(1.0):
            raise GearSonicError(
                "vector reference sampling only supports exact one-frame composer ticks"
            )

        self.measured = measured
        self.layer = layer
        self.runtime = motion_composer.SingleLayerComposer(
            current_layer=layer,
            action_playing=not layer.loop,
        )
        self._expected_policy_tick = 0
        self._completion_seen = False

    @property
    def loop(self) -> bool:
        return self.layer.loop

    def __call__(
        self, env_index: int, policy_tick: int
    ) -> tuple[int, Mapping[str, Any]]:
        if env_index != 0:
            raise GearSonicError("single-candidate composer only supports env index zero")
        if policy_tick != self._expected_policy_tick:
            raise GearSonicError(
                "composer reference resolver was called out of policy-tick order"
            )

        cursor_before = self.layer.cursor
        current = motion_composer.resolve_frames(self.layer, 0)
        next_frame = motion_composer.resolve_frames(self.layer, 1)
        future = tuple(
            motion_composer.resolve_frames(self.layer, offset)
            for offset in range(0, HISTORY_FRAMES * FUTURE_STEP, FUTURE_STEP)
        )
        if (
            current.t != 0.0
            or next_frame.t != 0.0
            or any(item.t != 0.0 for item in future)
        ):
            raise GearSonicError(
                "fractional composer interpolation is unavailable in vector sampling"
            )

        advance = self.runtime.advance()
        completion_event = advance.completed and not self._completion_seen
        self._completion_seen |= advance.completed
        self._expected_policy_tick += 1
        metadata: dict[str, Any] = {
            "composer_contract_schema": motion_composer.CONTRACT_SCHEMA,
            "composer_role": self.measured.role,
            "composer_cursor_before": cursor_before,
            "composer_frame_f0": current.f0,
            "composer_frame_f1": current.f1,
            "composer_interpolation_t": current.t,
            "composer_next_frame": {
                "f0": next_frame.f0,
                "f1": next_frame.f1,
                "t": next_frame.t,
            },
            "composer_future_frames": [
                {
                    "frames_ahead": offset,
                    "f0": resolved.f0,
                    "f1": resolved.f1,
                    "t": resolved.t,
                }
                for offset, resolved in zip(
                    range(0, HISTORY_FRAMES * FUTURE_STEP, FUTURE_STEP), future
                )
            ],
            "composer_cursor_after": self.layer.cursor,
            "composer_wrapped": advance.wrapped,
            "composer_endpoint_condition": advance.completed,
            "composer_completion_event": completion_event,
        }
        return current.f0, metadata

    def transition(self, *_args: Any, **_kwargs: Any) -> None:
        raise GearSonicError(
            "crossfade and feature-matched composer transitions are unsupported"
        )


def run(
    *,
    bundle: Path,
    assets_dir: Path,
    motion_role: str,
    manifest: Path,
    xml: Path,
    arena: Path,
    steps: int | None,
    control_boundary: str,
    frame_mode: str,
    force_limit_source: str,
) -> tuple[Mapping[str, Any], list[str]]:
    import gear_sonic_vector_env as vector_runtime

    environment = vector_runtime.GearSonicVectorEnv.from_artifacts(
        bundle=bundle,
        assets_dir=assets_dir,
        motion_role=motion_role,
        manifest=manifest,
        xml=xml,
        arena=arena,
        num_envs=1,
        control_boundary=control_boundary,
        frame_mode=frame_mode,
        force_limit_source=force_limit_source,
    )
    motion = environment.motion
    composer_provider = MotionComposerCursorProvider(
        motion, motion_role, frame_mode
    )
    environment.reference_resolver = composer_provider
    bundle_report = environment.bundle_report
    xml_contract = environment.xml_contract
    arena_contract = environment.arena_contract
    if bundle_report is None or xml_contract is None or arena_contract is None:
        raise GearSonicError("vector artifact metadata was not retained")
    mujoco = environment.mujoco
    model = environment.model
    runtime_map = environment.runtime_map
    data = environment.data[0]
    force_limits = environment.force_limits
    serialized_timestep = environment.serialized_timestep
    if serialized_timestep is None:
        raise GearSonicError("vector serialized timestep metadata is missing")
    mujoco_version = environment.mujoco_version
    ort_version = environment.onnxruntime_version
    physics_steps_per_control = environment.physics_steps_per_control
    command_lpf_interval = environment.command_lpf_interval
    command_lpf_dt = environment.command_lpf_dt
    command_lpf_alpha_value = environment.command_lpf_alpha
    initialization = environment.initializations[0]
    frame_count = int(motion.dof_pos.shape[0])
    run_steps = frame_count if steps is None else steps
    if isinstance(run_steps, bool) or run_steps < 1:
        raise GearSonicError("steps must be positive")

    root = runtime_map.root_qpos_address
    initial_base = np.asarray(data.qpos[root + 3 : root + 7], dtype=np.float64).copy()
    trace_lines: list[str] = []
    root_heights: list[float] = [float(data.qpos[root + 2])]
    upright_cosines: list[float] = [float(-projected_gravity(initial_base)[2])]
    torque_squared = 0.0
    target_error_squared = 0.0
    reference_error_squared = 0.0
    saturation_count = 0
    total_torque_values = 0

    for tick in range(run_steps):
        step = environment.step()[0]
        if step.tick != tick:
            raise GearSonicError("vector policy tick diverged from scalar run")
        current_frame = step.current_frame
        if current_frame != step.reference_metadata.get("composer_frame_f0"):
            raise GearSonicError("vector current frame diverged from composer resolution")
        composer_future = step.reference_metadata.get("composer_future_frames")
        if not isinstance(composer_future, list) or len(composer_future) != HISTORY_FRAMES:
            raise GearSonicError("composer future-frame metadata is incomplete")
        composer_future_indices = np.asarray(
            [item["f0"] for item in composer_future], dtype=np.int64
        )
        if not np.array_equal(step.future_indices, composer_future_indices):
            raise GearSonicError("vector future frames diverged from composer resolution")
        q_after = step.q_after_mujoco
        next_frame = step.reference_metadata.get("composer_next_frame")
        if not isinstance(next_frame, Mapping):
            raise GearSonicError("composer next-frame metadata is missing")
        next_f0 = int(next_frame["f0"])
        next_f1 = int(next_frame["f1"])
        next_t = float(next_frame["t"])
        next_reference = (
            motion.dof_pos[next_f0].astype(np.float64) * (1.0 - next_t)
            + motion.dof_pos[next_f1].astype(np.float64) * next_t
        )
        target_error_squared += float(
            np.dot(
                q_after - step.applied_targets_mujoco,
                q_after - step.applied_targets_mujoco,
            )
        )
        reference_error_squared += float(
            np.dot(q_after - next_reference, q_after - next_reference)
        )
        torque_squared += step.actuator_force_squared_sum
        saturation_count += step.torque_saturation_count
        total_torque_values += step.torque_value_count
        root_height = float(data.qpos[root + 2])
        root_heights.append(root_height)
        upright_cosines.append(float(-step.projected_gravity_after[2]))

        trace = {
            "type": "step",
            "tick": tick,
            "simulation_time_seconds": float(data.time),
            "physics_steps": step.physics_steps,
            "command_lpf_updates": step.command_lpf_updates,
            "reference_frame": current_frame,
            "future_reference_frames": composer_future_indices.tolist(),
            "encoder_observation_sha256_float32_le": sha256_float32(
                step.encoder_observation
            ),
            "encoded_token": step.token.tolist(),
            "decoder_observation_sha256_float32_le": sha256_float32(
                step.decoder_observation
            ),
            "raw_action_policy_order": step.raw_action_policy.tolist(),
            "clipped_action_policy_order": step.clipped_action_policy.tolist(),
            "raw_target_position_mujoco_order_rad": step.raw_targets_mujoco.tolist(),
            "lpf_target_position_mujoco_order_rad": (
                step.command_lpf_state_mujoco.tolist()
            ),
            "applied_target_position_mujoco_order_rad": (
                step.applied_targets_mujoco.tolist()
            ),
            "joint_range_clipped_mujoco_indices": np.flatnonzero(
                step.joint_target_clipped
            ).tolist(),
            "joint_position_before_mujoco_order_rad": step.q_before_mujoco.tolist(),
            "joint_velocity_before_mujoco_order_rad_s": step.dq_before_mujoco.tolist(),
            "torque_raw_mujoco_order_nm": step.torque_raw_mujoco.tolist(),
            "torque_predicted_clipped_mujoco_order_nm": (
                step.torque_predicted_mujoco.tolist()
            ),
            "actuator_force_after_step_mujoco_order_nm": (
                step.actuator_force_after_mujoco.tolist()
            ),
            "torque_saturated_mujoco_indices": np.flatnonzero(
                step.torque_saturated
            ).tolist(),
            "joint_position_after_mujoco_order_rad": q_after.tolist(),
            "root_position_after_m": np.asarray(data.qpos[root : root + 3]).tolist(),
            "root_quaternion_after_wxyz": np.asarray(data.qpos[root + 3 : root + 7]).tolist(),
            "projected_gravity_after": step.projected_gravity_after.tolist(),
            "body_pose_after": plant_contract.read_body_pose(data, runtime_map),
        }
        trace.update(step.reference_metadata)
        trace_lines.append(
            json.dumps(trace, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )

    joint_values = run_steps * ACTION_DIM
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "classification": CLASSIFICATION,
        "rek_parity_claim": False,
        "status": "ok",
        "artifacts": {
            "public_bundle": bundle_report,
            "motion": {
                "role": motion.role,
                "file": motion.filename,
                "sha256": motion.sha256,
                "frames": frame_count,
            },
            "motion_composer_contract": {
                "path": str(composer_provider.contract_path),
                "sha256": hashlib.sha256(
                    composer_provider.contract_path.read_bytes()
                ).hexdigest(),
                "schema": motion_composer.CONTRACT_SCHEMA,
            },
            "recovered_xml": {
                "path": str(Path(os.path.abspath(xml))),
                "sha256": xml_contract.source_sha256,
            },
            "arena_contract": {
                "path": str(Path(os.path.abspath(arena))),
                "sha256": arena_contract.source_sha256,
            },
        },
        "controller": {
            "family": "NVIDIA GEAR-SONIC",
            "source_commit": bundle_contract.SOURCE_COMMIT,
            "model_revision": bundle_contract.MODEL_REVISION,
            "control_hz": CONTROL_HZ,
            "action_clip": ACTION_CLIP,
            "control_boundary": control_boundary,
            "force_limit_source": force_limit_source,
            "force_limits_mujoco_order_nm": force_limits.tolist(),
            "command_lpf_cutoff_hz": COMMAND_LPF_CUTOFF_HZ,
            "command_lpf_work_rate_hz": NATIVE_WORK_RATE_HZ,
            "command_lpf_interval_physics_steps": command_lpf_interval,
            "command_lpf_dt_seconds": command_lpf_dt,
            "command_lpf_alpha": float(command_lpf_alpha_value),
            "joint_target_limit": "native_hinge_range_before_data_ctrl",
            "joint_history_order": "isaaclab_default_relative",
            "reference_motion_source_order": "mujoco_absolute",
            "reference_motion_encoder_order": "isaaclab_absolute",
            "output_order": "isaaclab_raw_then_remapped_to_mujoco_targets",
            "history": "10 ticks oldest first with zero left padding",
            "future_reference_offsets_ticks": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            "frame_mode": frame_mode,
            "motion_composer": {
                "role": composer_provider.measured.role,
                "loop": composer_provider.layer.loop,
                "speed": composer_provider.layer.speed,
                "per_tick": composer_provider.layer.per_tick,
                "start_frame": composer_provider.layer.start_frame,
                "end_frame": composer_provider.layer.end_frame,
                "clip_path_id": composer_provider.measured.clip.path_id,
                "config_path_id": composer_provider.measured.config.path_id,
                "transition_support": "fail_closed_crossfade_and_feature_matching",
            },
        },
        "runtime": {
            "onnxruntime": ort_version,
            "mujoco": mujoco_version,
            "serialized_xml_timestep_seconds": serialized_timestep,
            "plant_timestep_seconds": float(model.opt.timestep),
            "physics_steps_per_controller_tick": physics_steps_per_control,
            "steps": run_steps,
            "simulated_seconds": run_steps * CONTROL_DT,
            "initialization": initialization,
            "composer_final_cursor": composer_provider.layer.cursor,
        },
        "metrics": {
            "joint_reference_rmse_rad": math.sqrt(reference_error_squared / joint_values),
            "joint_target_rmse_rad": math.sqrt(target_error_squared / joint_values),
            "applied_torque_rms_nm": math.sqrt(torque_squared / total_torque_values),
            "torque_saturation_count": saturation_count,
            "torque_saturation_fraction": saturation_count / total_torque_values,
            "root_height_min_m": min(root_heights),
            "root_height_max_m": max(root_heights),
            "root_height_final_m": root_heights[-1],
            "upright_cosine_min": min(upright_cosines),
            "upright_cosine_final": upright_cosines[-1],
            "step_records_sha256": hashlib.sha256(
                ("\n".join(trace_lines) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "limits": [
            "Public model byte identity with the REK-packaged model is unknown.",
            "The recovered XML contract marks the plant control-equivalent flag false.",
            "Motion velocity uses REK's forward finite difference because the packaged NPZ omits velocity.",
            "Position-actuator gain mutation is native REK code; exact packaged gain values remain gated on the unavailable local Sonic config asset.",
            "The public-model-config effort limits match the public weights, not an extracted live REK SonicConfig.",
            "Recovery orchestration, fight contact events, damage, rewards, and opponent logic are not implemented here.",
            "Composer crossfade, feature-matched entry, mirrored sampling, and partial frame windows fail closed.",
            "No authoritative held-out REK trajectory was compared by this run.",
        ],
    }
    return report, trace_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--motion-role", required=True)
    parser.add_argument("--manifest", type=Path, default=_default_manifest())
    parser.add_argument("--xml", type=Path, default=_default_xml())
    parser.add_argument("--arena-contract", type=Path, default=_default_arena())
    parser.add_argument("--steps", type=_positive_int)
    parser.add_argument(
        "--control-boundary",
        required=True,
        choices=("native-position-actuator",),
        help="explicit REK Robot control boundary; no implicit fallback is used",
    )
    parser.add_argument(
        "--frame-mode",
        required=True,
        choices=("clamp", "loop"),
        help="explicit clip cursor behavior; supplied clip metadata decides this in REK",
    )
    parser.add_argument(
        "--force-limit-source",
        required=True,
        choices=("public-model-config", "recovered-import-fallback"),
        help="explicit actuator effort source; no implicit fallback is used",
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--metrics-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.trace is not None and args.metrics_out is not None:
            if Path(os.path.abspath(args.trace)) == Path(os.path.abspath(args.metrics_out)):
                raise GearSonicError("trace and metrics paths must differ")
        report, trace_lines = run(
            bundle=args.bundle,
            assets_dir=args.assets_dir,
            motion_role=args.motion_role,
            manifest=args.manifest,
            xml=args.xml,
            arena=args.arena_contract,
            steps=args.steps,
            control_boundary=args.control_boundary,
            frame_mode=args.frame_mode,
            force_limit_source=args.force_limit_source,
        )
        if args.trace is not None:
            header = {
                "type": "header",
                "schema": TRACE_SCHEMA,
                "classification": CLASSIFICATION,
                "rek_parity_claim": False,
                "artifacts": report["artifacts"],
                "controller": report["controller"],
                "runtime": report["runtime"],
                "limits": report["limits"],
            }
            trace_text = json.dumps(
                header, sort_keys=True, separators=(",", ":"), allow_nan=False
            ) + "\n" + "\n".join(trace_lines) + "\n"
            trace_path = _write_new(args.trace, trace_text, "trace output")
            report["trace"] = {
                "path": str(trace_path),
                "sha256": hashlib.sha256(trace_text.encode("utf-8")).hexdigest(),
                "records": len(trace_lines) + 1,
            }
        if args.metrics_out is not None:
            report["metrics_output_path"] = str(Path(os.path.abspath(args.metrics_out)))
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.metrics_out is not None:
            _write_new(args.metrics_out, output, "metrics output")
        sys.stdout.write(output)
        return 0
    except (
        GearSonicError,
        bundle_contract.BundleError,
        plant_contract.CandidateError,
        OSError,
        ValueError,
    ) as exc:
        sys.stderr.write(
            json.dumps(
                {
                    "schema": RUN_SCHEMA,
                    "classification": CLASSIFICATION,
                    "rek_parity_claim": False,
                    "status": "rejected",
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
