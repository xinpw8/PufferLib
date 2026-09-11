#!/usr/bin/env python3
"""Independent Spark cross-check for the native G1 foot-feature bake."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np


MANIFEST_SHA256 = "b305abade35ffc01ccf284a5d81d88fa9c2bcd0204d8e7b98b5778289197e0b6"
CLIPS = ((370, 36), (371, 146), (372, 159), (375, 47),
         (377, 39), (380, 140), (388, 35), (392, 158))
JOINT_NAMES = (
    "joint__left_hip_pitch_joint_3047",
    "joint__left_hip_roll_joint_3248",
    "joint__left_hip_yaw_joint_3267",
    "joint__left_knee_joint_3137",
    "joint__left_ankle_pitch_joint_2982",
    "joint__left_ankle_roll_joint_2905",
    "joint__right_hip_pitch_joint_3298",
    "joint__right_hip_roll_joint_3059",
    "joint__right_hip_yaw_joint_3071",
    "joint__right_knee_joint_3412",
    "joint__right_ankle_pitch_joint_3312",
    "joint__right_ankle_roll_joint_3474",
    "joint__waist_yaw_joint_3441",
    "joint__waist_roll_joint_3341",
    "joint__waist_pitch_joint_3233",
    "joint__left_shoulder_pitch_joint_3340",
    "joint__left_shoulder_roll_joint_3184",
    "joint__left_shoulder_yaw_joint_2923",
    "joint__left_elbow_joint_3144",
    "joint__left_wrist_roll_joint_3260",
    "joint__left_wrist_pitch_joint_3007",
    "joint__left_wrist_yaw_joint_3398",
    "joint__right_shoulder_pitch_joint_3242",
    "joint__right_shoulder_roll_joint_3044",
    "joint__right_shoulder_yaw_joint_3176",
    "joint__right_elbow_joint_3407",
    "joint__right_wrist_roll_joint_3378",
    "joint__right_wrist_pitch_joint_3437",
    "joint__right_wrist_yaw_joint_3226",
)
ROOT_BODY = "player__pelvis_3266"
LEFT_BODY = "player__left_ankle_roll_link_3045"
RIGHT_BODY = "player__right_ankle_roll_link_3090"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, kind, name))
    if identifier < 0:
        raise ValueError(f"missing MuJoCo object: {name}")
    return identifier


def inverse_transform_point(
    root_position: np.ndarray,
    root_matrix: np.ndarray,
    world_position: np.ndarray,
) -> tuple[float, float, float]:
    delta = tuple(float(world_position[index]) - float(root_position[index])
                  for index in range(3))
    result = []
    for axis in range(3):
        first = float(root_matrix[0, axis]) * delta[0]
        second = float(root_matrix[1, axis]) * delta[1]
        third = float(root_matrix[2, axis]) * delta[2]
        result.append((first + second) + third)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("non-finite inverse-transform result")
    return result[0], result[1], result[2]


def bake(asset_dir: Path) -> np.ndarray:
    manifest = asset_dir / "semantic_duel_assets_manifest.json"
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("asset manifest SHA-256 mismatch")
    model = mujoco.MjModel.from_xml_path(
        str(asset_dir / "model.two_fighter_arena.xml"))
    data = mujoco.MjData(model)
    root_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    left_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_BODY)
    right_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_BODY)
    qpos_addresses = []
    for suffix in JOINT_NAMES:
        joint_id = object_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"player__{suffix}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    if len(set(qpos_addresses)) != 29:
        raise ValueError("29-DOF qpos mapping is not unique")

    output = np.empty((sum(frames for _, frames in CLIPS), 6), dtype="<f4")
    output_row = 0
    for clip_id, frame_count in CLIPS:
        path = asset_dir / f"motion_{clip_id}_dof_position.f32le"
        positions = np.fromfile(path, dtype="<f4")
        if positions.shape != (frame_count * 29,):
            raise ValueError(f"clip {clip_id} size mismatch")
        positions = positions.reshape(frame_count, 29)
        if not np.isfinite(positions).all():
            raise ValueError(f"clip {clip_id} contains non-finite values")
        for frame in positions:
            mujoco.mj_resetData(model, data)
            data.qpos[qpos_addresses] = frame.astype(np.float64)
            mujoco.mj_kinematics(model, data)
            root_position = data.xpos[root_id]
            root_matrix = data.xmat[root_id].reshape(3, 3)
            left_mujoco = inverse_transform_point(
                root_position, root_matrix, data.xpos[left_id])
            right_mujoco = inverse_transform_point(
                root_position, root_matrix, data.xpos[right_id])
            output[output_row] = (
                left_mujoco[0], left_mujoco[2], left_mujoco[1],
                right_mujoco[0], right_mujoco[2], right_mujoco[1],
            )
            output_row += 1
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_dir", type=Path)
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    features = bake(args.asset_dir.resolve())
    actual = hashlib.sha256(features.tobytes(order="C")).hexdigest()
    if actual != args.expected_sha256.lower():
        raise ValueError(
            f"native/Python feature SHA-256 mismatch: {actual} != "
            f"{args.expected_sha256.lower()}")
    print(json.dumps({
        "schema": "rek.g1_mujoco_feature_registry_python_crosscheck.v1",
        "clips": len(CLIPS),
        "frames": int(features.shape[0]),
        "features": int(features.size),
        "feature_sha256": actual,
        "first_feature_unity_xyz": features[0].tolist(),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
