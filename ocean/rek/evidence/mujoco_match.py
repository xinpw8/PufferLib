#!/usr/bin/env python3
"""Build a two-T800 diagnostic MJCF from measured client evidence.

This composes two copies of the recovered T800 plant into the recovered static
arena and reconstructs one client-observed first-active articulated pose.  It
does not recover REK control, policy, damage, reward, opponent, or round
semantics, and it does not establish a reset distribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "rek.mujoco_two_t800_diagnostic.v1"
EXPECTED_TRACE_NAME = (
    "rek-private-ai-client-fixed-20260902T024258.5184763Z-pid108872-"
    "3f965094791f4dc4a76ae5ec2ea8a54e.jsonl"
)
EXPECTED_TRACE_SHA256 = "6f61645fe650b4de199abbeb25ec581aa23215d853da1f1e3dbea8ddf3477057"
EXPECTED_BASE_MJCF_SHA256 = "0a5fb688156fb57474056470c78e2209ebfff4e09e3935b73e3375c28d33ba93"
EXPECTED_BUILD_FINGERPRINT = "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
EXPECTED_GAME_ASSEMBLY_SHA256 = "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412"
EXPECTED_GLOBAL_METADATA_SHA256 = "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd"
EXPECTED_COMMAND_RESULTS = {
    "491": {
        "result_sha256": "a6da184f7050c67b7af1b866f44acff2f2840b659e7464fcd3fc6deda2d0b635",
        "stdout_sha256": "7cfaa103a55df2086e32b5020f7d72b443950565f8bb92b1d4c347df4c2fdcdb",
    },
    "493": {
        "result_sha256": "a9b3e02457599259adc321d54dbeb1c4555302cd5479d38f01255ca5ceb0123b",
        "stdout_sha256": "623020b43737b037ebd96e7a804b9dfe0223ddefc163b7e71d7b15ab07bbab13",
    },
}
EXPECTED_BASE_DIMENSIONS = {
    "nbody": 31,
    "njnt": 26,
    "ngeom": 54,
    "nq": 32,
    "nv": 31,
    "nu": 25,
}
EXPECTED_MATCH_DIMENSIONS = {
    "nbody": 61,
    "njnt": 52,
    "ngeom": 91,
    "nq": 64,
    "nv": 62,
    "nu": 50,
}
EXPECTED_T800_BONES = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
]
FIGHTER_PREFIXES = ("fighter_0__", "fighter_1__")
KEYFRAME_NAME = "client_observed_round2_first_active"
REFERENCE_ATTRIBUTES = {
    "actuator",
    "body",
    "geom",
    "joint",
    "site",
    "tendon",
}
POSITION_FIT_LIMIT_METERS = 1e-5
ORIENTATION_FIT_LIMIT_RADIANS = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: float | int) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite MJCF value: {result}")
    if result == 0:
        result = 0.0
    return format(result, ".17g")


def vector_text(values: Iterable[float]) -> str:
    return " ".join(number(value) for value in values)


def vector_to_mujoco(unity_xyz: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in unity_xyz)
    return x, z, y


def normalize_quaternion(values: Iterable[float]) -> tuple[float, float, float, float]:
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4:
        raise ValueError("quaternion must contain four components")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("zero-length quaternion")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def quaternion_xyzw_to_mujoco_wxyz(
    unity_xyzw: Iterable[float],
) -> tuple[float, float, float, float]:
    x, y, z, w = (float(value) for value in unity_xyzw)
    return normalize_quaternion((-w, x, z, y))


def quaternion_conjugate(
    value: Iterable[float],
) -> tuple[float, float, float, float]:
    w, x, y, z = (float(component) for component in value)
    return w, -x, -y, -z


def quaternion_multiply(
    lhs: Iterable[float], rhs: Iterable[float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = (float(component) for component in lhs)
    rw, rx, ry, rz = (float(component) for component in rhs)
    return normalize_quaternion((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def relative_quaternion(
    parent: Iterable[float], child: Iterable[float],
) -> tuple[float, float, float, float]:
    return quaternion_multiply(quaternion_conjugate(parent), child)


def quaternion_distance(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    a = normalize_quaternion(lhs)
    b = normalize_quaternion(rhs)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def source_body_name(compiled_name: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]+)", compiled_name)
    if match is None:
        raise ValueError(f"body name lacks a numeric path-ID suffix: {compiled_name!r}")
    return match.group(1)


def _finite_vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} components")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{field} contains a non-finite component")
    return result


def _bone_names(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            result.append(item["name"])
        else:
            raise ValueError(f"{field} contains an invalid bone record")
    return result


def read_first_active(trace_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if trace_path.name != EXPECTED_TRACE_NAME:
        raise ValueError(f"unexpected trace basename: {trace_path.name}")
    actual_hash = sha256_file(trace_path)
    if actual_hash != EXPECTED_TRACE_SHA256:
        raise ValueError(
            f"trace SHA-256 mismatch: expected={EXPECTED_TRACE_SHA256} actual={actual_hash}"
        )

    start = None
    sample = None
    nonblank_index = 0
    with trace_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if nonblank_index == 0 and record.get("event") != "capture_start":
                raise ValueError("capture_start is not the first nonblank record")
            nonblank_index += 1
            if record.get("event") == "capture_start":
                if start is not None:
                    raise ValueError("multiple capture_start records precede the first sample")
                start = record
            elif record.get("event") == "sample":
                sample = record
                break
    if start is None or sample is None:
        raise ValueError("trace lacks capture_start or sample evidence")
    validate_capture(start, sample)
    return start, sample


def validate_capture(start: dict[str, Any], sample: dict[str, Any]) -> None:
    if start.get("schema") != "rek.private_ai.client_fixed.v3":
        raise ValueError("unsupported recorder schema")
    if start.get("scene") != "Arena" or sample.get("scene") != "Arena":
        raise ValueError("evidence was not observed in the Arena scene")
    if start.get("game_assembly_sha256") != EXPECTED_GAME_ASSEMBLY_SHA256:
        raise ValueError("trace GameAssembly identity mismatch")
    if start.get("global_metadata_sha256") != EXPECTED_GLOBAL_METADATA_SHA256:
        raise ValueError("trace global-metadata identity mismatch")
    if start.get("authority_semantics") != (
        "client_observation_of_remote_authoritative_private_AI_mode"
    ):
        raise ValueError("trace authority semantics mismatch")

    scope = start.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("trace lacks a scope record")
    expected_scope = {
        "allowed": True,
        "network_is_client": True,
        "network_is_server": False,
        "opponent_is_ai": True,
        "opponent_slot_is_ai": True,
        "human_in_opponent_slot": False,
        "opponent_slot_has_client": False,
        "opponent_human_bit_set": False,
        "fighter_0_visual_only": True,
        "fighter_1_visual_only": True,
        "sparring_bot_number": 1,
    }
    for field, expected in expected_scope.items():
        if scope.get(field) != expected:
            raise ValueError(f"scope field {field!r} differs from required Bot 1 evidence")

    for slot in (0, 1):
        names = _bone_names(start.get(f"fighter_{slot}_bones"), f"fighter_{slot}_bones")
        if names != EXPECTED_T800_BONES:
            raise ValueError(f"fighter {slot} is not the expected ordered T800 bone layout")
        fighter = sample.get(f"fighter_{slot}")
        if not isinstance(fighter, dict):
            raise ValueError(f"sample lacks fighter_{slot}")
        if fighter.get("visual_only") is not True or fighter.get("player_controlled") is not False:
            raise ValueError(f"fighter {slot} is not a remote visual-only observation")
        _finite_vector(fighter.get("root_position"), 3, f"fighter_{slot}.root_position")
        _finite_vector(fighter.get("root_rotation"), 4, f"fighter_{slot}.root_rotation")
        _finite_vector(fighter.get("root_linear_velocity"), 3,
                       f"fighter_{slot}.root_linear_velocity")
        _finite_vector(fighter.get("root_angular_velocity"), 3,
                       f"fighter_{slot}.root_angular_velocity")
        bones = fighter.get("bones")
        if not isinstance(bones, dict) or bones.get("count") != len(EXPECTED_T800_BONES):
            raise ValueError(f"fighter {slot} bone pose count mismatch")
        _finite_vector(bones.get("world_positions_xyz"), 3 * len(EXPECTED_T800_BONES),
                       f"fighter_{slot}.bones.world_positions_xyz")
        _finite_vector(bones.get("world_rotations_xyzw"), 4 * len(EXPECTED_T800_BONES),
                       f"fighter_{slot}.bones.world_rotations_xyzw")

    if sample.get("sample_index") != 0 or sample.get("client_fixed_tick") != 0:
        raise ValueError("selected evidence is not the first captured active sample")
    if sample.get("phase") != "RoundActive":
        raise ValueError("selected sample is not round-active")
    round_record = sample.get("round")
    if not isinstance(round_record, dict) or round_record.get("active") is not True:
        raise ValueError("selected sample lacks an active round record")
    if round_record.get("number") != 2 or round_record.get("time_remaining") != 120:
        raise ValueError("selected sample is not the exact audited round-2 boundary")


def validate_command_result(path: Path, command_id: str) -> dict[str, Any]:
    expected = EXPECTED_COMMAND_RESULTS[command_id]
    actual_hash = sha256_file(path)
    if actual_hash != expected["result_sha256"]:
        raise ValueError(f"command {command_id} result SHA-256 mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    if str(result.get("id")) != command_id or result.get("exit_code") != 0:
        raise ValueError(f"command {command_id} did not record a successful execution")
    if str(result.get("stdout_sha256", "")).lower() != expected["stdout_sha256"]:
        raise ValueError(f"command {command_id} stdout identity mismatch")
    return {
        "id": command_id,
        "result_path": str(path.resolve()),
        "result_sha256": actual_hash,
        "stdout_path": result.get("stdout_path"),
        "stdout_sha256": str(result["stdout_sha256"]).lower(),
        "host": result.get("host"),
        "exit_code": result.get("exit_code"),
    }


def validate_sources(
    base_mjcf: Path,
    base_report_path: Path,
    inventory_path: Path,
    start: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_hash = sha256_file(base_mjcf)
    if base_hash != EXPECTED_BASE_MJCF_SHA256:
        raise ValueError("base T800 arena MJCF SHA-256 mismatch")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if base_report.get("schema") != "rek.mujoco_arena_composition.v1":
        raise ValueError("unsupported base arena report")
    if base_report.get("mjcf_sha256") != base_hash:
        raise ValueError("base report does not bind the input MJCF")
    if str(base_report.get("inventory_sha256", "")).lower() != sha256_file(inventory_path):
        raise ValueError("base report does not bind the input inventory")
    fingerprints = {
        base_report.get("build_fingerprint"),
        inventory.get("build_fingerprint"),
    }
    if fingerprints != {EXPECTED_BUILD_FINGERPRINT}:
        raise ValueError("base model and inventory build identities differ")

    file_hashes = {
        str(record.get("path", "")).replace("\\", "/"): str(record.get("sha256", "")).lower()
        for record in inventory.get("files", ())
    }
    if file_hashes.get("GameAssembly.dll") != start["game_assembly_sha256"]:
        raise ValueError("inventory does not bind the trace GameAssembly")
    metadata_path = "REK_Data/il2cpp_data/Metadata/global-metadata.dat"
    if file_hashes.get(metadata_path) != start["global_metadata_sha256"]:
        raise ValueError("inventory does not bind the trace global metadata")
    return base_report, inventory


def model_dimensions(model: Any) -> dict[str, int]:
    return {
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
    }


def captured_targets(sample: dict[str, Any], slot: int) -> dict[str, dict[str, Any]]:
    fighter = sample[f"fighter_{slot}"]
    bones = fighter["bones"]
    positions = bones["world_positions_xyz"]
    rotations = bones["world_rotations_xyzw"]
    result = {}
    for index, name in enumerate(EXPECTED_T800_BONES):
        result[name] = {
            "position": vector_to_mujoco(positions[3 * index:3 * index + 3]),
            "quaternion": quaternion_xyzw_to_mujoco_wxyz(
                rotations[4 * index:4 * index + 4]
            ),
        }
    return result


def model_body_map(mujoco: Any, model: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for body_id in range(1, model.nbody):
        compiled_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if compiled_name is None:
            raise ValueError(f"unnamed body ID {body_id}")
        source_name = source_body_name(compiled_name)
        if source_name in result:
            raise ValueError(f"duplicate source body name: {source_name}")
        result[source_name] = body_id
    missing = [name for name in EXPECTED_T800_BONES if name not in result]
    if missing:
        raise ValueError(f"captured T800 bones are absent from model: {missing}")
    return result


def _model_relative_quaternion(data: Any, parent_id: int, child_id: int) -> tuple[float, ...]:
    return relative_quaternion(data.xquat[parent_id], data.xquat[child_id])


def reconstruct_slot(mujoco: Any, model: Any, sample: dict[str, Any], slot: int) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    data = mujoco.MjData(model)
    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("first T800 joint is not free")
    if int(model.jnt_qposadr[0]) != 0 or int(model.jnt_dofadr[0]) != 0:
        raise ValueError("unexpected T800 free-joint address")

    fighter = sample[f"fighter_{slot}"]
    data.qpos[0:3] = vector_to_mujoco(fighter["root_position"])
    data.qpos[3:7] = quaternion_xyzw_to_mujoco_wxyz(fighter["root_rotation"])
    targets = captured_targets(sample, slot)
    bodies = model_body_map(mujoco, model)
    names_by_body = {body_id: name for name, body_id in bodies.items()}
    fits = []

    for joint_id in range(1, model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"joint ID {joint_id} is not a hinge")
        child_id = int(model.jnt_bodyid[joint_id])
        parent_id = int(model.body_parentid[child_id])
        child_name = names_by_body.get(child_id)
        parent_name = names_by_body.get(parent_id)
        if child_name not in targets or parent_name not in targets:
            raise ValueError(
                f"joint {joint_id} lacks captured parent/child orientations: "
                f"parent={parent_name!r} child={child_name!r}"
            )
        target_relative = relative_quaternion(
            targets[parent_name]["quaternion"], targets[child_name]["quaternion"]
        )
        qpos_address = int(model.jnt_qposadr[joint_id])
        if not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"hinge joint {joint_id} lacks a serialized range")
        lower = float(model.jnt_range[joint_id, 0])
        upper = float(model.jnt_range[joint_id, 1])
        if not lower < upper:
            raise ValueError(f"hinge joint {joint_id} has an invalid range")

        def objective(value: float) -> float:
            data.qpos[qpos_address] = value
            mujoco.mj_forward(model, data)
            return quaternion_distance(
                _model_relative_quaternion(data, parent_id, child_id), target_relative
            )

        grid = np.linspace(lower, upper, 721)
        errors = [objective(float(value)) for value in grid]
        best_index = int(np.argmin(errors))
        if best_index == 0 or best_index == len(grid) - 1:
            raise ValueError(f"joint {joint_id} reconstruction reaches a range boundary")
        left = float(grid[best_index - 1])
        right = float(grid[best_index + 1])
        for _ in range(45):
            lower_third = left + (right - left) / 3.0
            upper_third = right - (right - left) / 3.0
            if objective(lower_third) < objective(upper_third):
                right = upper_third
            else:
                left = lower_third
        value = (left + right) * 0.5
        fit_error = objective(value)
        data.qpos[qpos_address] = value
        fits.append({
            "joint_id": joint_id,
            "joint_name": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            ),
            "qpos_address": qpos_address,
            "value": value,
            "range": [lower, upper],
            "relative_orientation_error_radians": fit_error,
        })

    mujoco.mj_forward(model, data)
    residuals = []
    for name in EXPECTED_T800_BONES:
        body_id = bodies[name]
        target = targets[name]
        position_error = math.sqrt(sum(
            (float(data.xpos[body_id, index]) - target["position"][index]) ** 2
            for index in range(3)
        ))
        orientation_error = quaternion_distance(
            data.xquat[body_id], target["quaternion"]
        )
        residuals.append({
            "bone": name,
            "body_id": body_id,
            "compiled_body_name": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ),
            "position_error_meters": position_error,
            "orientation_error_radians": orientation_error,
        })
    max_position_error = max(record["position_error_meters"] for record in residuals)
    max_orientation_error = max(record["orientation_error_radians"] for record in residuals)
    if max_position_error > POSITION_FIT_LIMIT_METERS:
        raise ValueError(f"slot {slot} position fit exceeds the numerical gate")
    if max_orientation_error > ORIENTATION_FIT_LIMIT_RADIANS:
        raise ValueError(f"slot {slot} orientation fit exceeds the numerical gate")

    qpos = data.qpos.copy()
    return qpos, {
        "slot": slot,
        "qpos": [float(value) for value in qpos],
        "root_source_unity_position_xyz": fighter["root_position"],
        "root_source_unity_quaternion_xyzw": fighter["root_rotation"],
        "root_mujoco_position_xyz": [float(value) for value in qpos[0:3]],
        "root_mujoco_quaternion_wxyz": [float(value) for value in qpos[3:7]],
        "source_root_linear_velocity_xyz": fighter["root_linear_velocity"],
        "source_root_angular_velocity_xyz": fighter["root_angular_velocity"],
        "hinge_fits": fits,
        "bone_residuals": residuals,
        "max_position_error_meters": max_position_error,
        "max_orientation_error_radians": max_orientation_error,
    }


def prefixed_copy(
    element: ET.Element,
    prefix: str,
    source_names: Iterable[str] | None = None,
) -> ET.Element:
    result = copy.deepcopy(element)
    old_names = list(source_names) if source_names is not None else [
        node.get("name") for node in result.iter() if node.get("name")
    ]
    if len(old_names) != len(set(old_names)):
        raise ValueError("source subtree has duplicate names")
    name_map = {name: prefix + name for name in old_names}
    for node in result.iter():
        name = node.get("name")
        if name is not None:
            node.set("name", name_map[name])
        for attribute in REFERENCE_ATTRIBUTES:
            reference = node.get(attribute)
            if reference is not None:
                if reference not in name_map:
                    raise ValueError(
                        f"unresolved source reference {attribute}={reference!r}"
                    )
                node.set(attribute, name_map[reference])
    return result


def compose_xml(base_mjcf: Path, qpos_by_slot: list[Any]) -> ET.ElementTree:
    tree = ET.parse(base_mjcf)
    root = tree.getroot()
    worldbodies = root.findall("worldbody")
    actuators = root.findall("actuator")
    if len(worldbodies) != 1 or len(actuators) != 1:
        raise ValueError("base model must contain one worldbody and one actuator section")
    worldbody = worldbodies[0]
    actuator_section = actuators[0]
    robot_roots = [element for element in list(worldbody) if element.tag == "body"]
    arena_geoms = [element for element in list(worldbody) if element.tag == "geom"]
    if len(robot_roots) != 1 or len(arena_geoms) != 17:
        raise ValueError("base model does not contain one robot and 17 arena geoms")
    source_robot = robot_roots[0]
    source_actuators = list(actuator_section)
    if len(source_actuators) != 25 or any(element.tag != "motor" for element in source_actuators):
        raise ValueError("base model does not contain 25 motor actuators")

    worldbody.remove(source_robot)
    actuator_section.clear()
    source_names = [
        node.get("name")
        for element in (source_robot, *source_actuators)
        for node in element.iter()
        if node.get("name")
    ]
    for prefix in FIGHTER_PREFIXES:
        worldbody.append(prefixed_copy(source_robot, prefix, source_names))
        for actuator in source_actuators:
            actuator_section.append(prefixed_copy(actuator, prefix, source_names))

    if root.find("keyframe") is not None:
        raise ValueError("base model unexpectedly contains a keyframe section")
    keyframes = ET.SubElement(root, "keyframe")
    combined_qpos = [float(value) for slot in qpos_by_slot for value in slot]
    ET.SubElement(
        keyframes,
        "key",
        name=KEYFRAME_NAME,
        qpos=vector_text(combined_qpos),
    )
    root.set("model", "rek_t800_t800_arena_diagnostic")

    names = [node.get("name") for node in root.iter() if node.get("name")]
    if len(names) != len(set(names)):
        raise ValueError("composed MJCF names are not globally unique")
    joints = {node.get("name") for node in root.iter("joint")}
    joints.update(node.get("name") for node in root.iter("freejoint"))
    for actuator in actuator_section:
        if actuator.get("joint") not in joints:
            raise ValueError(f"unresolved composed actuator joint: {actuator.get('joint')}")
    return tree


def classify_contact(mujoco: Any, model: Any, contact: Any) -> str:
    categories = []
    for geom_id in (int(contact.geom1), int(contact.geom2)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(FIGHTER_PREFIXES[0]):
            categories.append("fighter_0")
        elif name.startswith(FIGHTER_PREFIXES[1]):
            categories.append("fighter_1")
        else:
            categories.append("arena")
    return "--".join(sorted(categories))


def validate_composed(
    mujoco: Any,
    tree: ET.ElementTree,
    sample: dict[str, Any],
    steps: int,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    xml_text = ET.tostring(tree.getroot(), encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_text)
    dimensions = model_dimensions(model)
    if dimensions != EXPECTED_MATCH_DIMENSIONS:
        raise ValueError(f"composed dimensions differ: {dimensions}")
    compiled_model_name = bytes(model.names).split(b"\0", 1)[0].decode("utf-8")
    if compiled_model_name != "rek_t800_t800_arena_diagnostic":
        raise ValueError(f"compiled model name differs: {compiled_model_name!r}")
    if int(model.nkey) != 1:
        raise ValueError(f"composed model has an unexpected key count: {model.nkey}")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    if key_id < 0:
        raise ValueError("diagnostic reset keyframe did not compile")

    free_joints = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_joints.append({
                "joint_id": joint_id,
                "joint_name": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                ),
                "body_id": int(model.jnt_bodyid[joint_id]),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
            })
    expected_addresses = [(0, 0, 0), (26, 32, 31)]
    actual_addresses = [
        (record["joint_id"], record["qpos_address"], record["dof_address"])
        for record in free_joints
    ]
    if actual_addresses != expected_addresses:
        raise ValueError(f"unexpected free-joint addresses: {actual_addresses}")

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    initial_finite = bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(data.qacc).all()
    )
    if not initial_finite:
        raise ValueError("composed keyframe is not finite")

    residuals = []
    for slot, prefix in enumerate(FIGHTER_PREFIXES):
        targets = captured_targets(sample, slot)
        for source_name in EXPECTED_T800_BONES:
            matches = []
            for body_id in range(1, model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name is not None and name.startswith(prefix):
                    if source_body_name(name[len(prefix):]) == source_name:
                        matches.append(body_id)
            if len(matches) != 1:
                raise ValueError(f"composed body mapping is not unique: {prefix}{source_name}")
            body_id = matches[0]
            target = targets[source_name]
            position_error = math.sqrt(sum(
                (float(data.xpos[body_id, index]) - target["position"][index]) ** 2
                for index in range(3)
            ))
            orientation_error = quaternion_distance(
                data.xquat[body_id], target["quaternion"]
            )
            residuals.append({
                "slot": slot,
                "bone": source_name,
                "position_error_meters": position_error,
                "orientation_error_radians": orientation_error,
            })
    max_position_error = max(item["position_error_meters"] for item in residuals)
    max_orientation_error = max(item["orientation_error_radians"] for item in residuals)
    if max_position_error > POSITION_FIT_LIMIT_METERS:
        raise ValueError("composed position fit exceeds the numerical gate")
    if max_orientation_error > ORIENTATION_FIT_LIMIT_RADIANS:
        raise ValueError("composed orientation fit exceeds the numerical gate")

    contact_categories: dict[str, int] = {}
    for index in range(data.ncon):
        category = classify_contact(mujoco, model, data.contact[index])
        contact_categories[category] = contact_categories.get(category, 0) + 1
    if contact_categories.get("fighter_0--fighter_1", 0) != 0:
        raise ValueError("fighters intersect at the reconstructed observation")

    initial = {
        "finite": initial_finite,
        "contact_count": int(data.ncon),
        "contact_categories": contact_categories,
        "max_position_error_meters": max_position_error,
        "max_orientation_error_radians": max_orientation_error,
    }
    max_contact_count = int(data.ncon)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        max_contact_count = max(max_contact_count, int(data.ncon))
    final_finite = bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(data.qacc).all()
        and math.isfinite(float(data.time))
    )
    if not final_finite:
        raise ValueError("composed state becomes non-finite in zero-control validation")
    validation = {
        "mujoco_version": mujoco.__version__,
        "model_name": compiled_model_name,
        "dimensions": dimensions,
        "keyframe_count": int(model.nkey),
        "keyframe_name": KEYFRAME_NAME,
        "keyframe_id": int(key_id),
        "free_joints": free_joints,
        "qpos_blocks": [[0, 32], [32, 64]],
        "qvel_blocks": [[0, 31], [31, 62]],
        "control_blocks": [[0, 25], [25, 50]],
        "initial": initial,
        "zero_control_steps": steps,
        "zero_control_final_time_seconds": float(data.time),
        "zero_control_final_finite": final_finite,
        "zero_control_final_contact_count": int(data.ncon),
        "zero_control_max_contact_count": max_contact_count,
        "zero_control_final_root_heights_meters": [
            float(data.qpos[2]), float(data.qpos[34])
        ],
    }
    return model, validation


def generate(
    base_mjcf: Path,
    base_report_path: Path,
    inventory_path: Path,
    trace_path: Path,
    pose_command_result: Path,
    bone_command_result: Path,
    steps: int,
) -> tuple[ET.ElementTree, dict[str, Any]]:
    import mujoco

    start, sample = read_first_active(trace_path)
    base_report, _ = validate_sources(
        base_mjcf, base_report_path, inventory_path, start
    )
    commands = [
        validate_command_result(pose_command_result, "491"),
        validate_command_result(bone_command_result, "493"),
    ]

    base_model = mujoco.MjModel.from_xml_path(str(base_mjcf))
    dimensions = model_dimensions(base_model)
    if dimensions != EXPECTED_BASE_DIMENSIONS:
        raise ValueError(f"base model dimensions differ: {dimensions}")
    qpos_by_slot = []
    reconstruction = []
    for slot in (0, 1):
        qpos, slot_report = reconstruct_slot(mujoco, base_model, sample, slot)
        qpos_by_slot.append(qpos)
        reconstruction.append(slot_report)

    tree = compose_xml(base_mjcf, qpos_by_slot)
    _, validation = validate_composed(mujoco, tree, sample, steps)
    report = {
        "schema": SCHEMA,
        "build_fingerprint": EXPECTED_BUILD_FINGERPRINT,
        "control_equivalent": False,
        "behavioral_clone": False,
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "base_mjcf_path": str(base_mjcf.resolve()),
        "base_mjcf_sha256": sha256_file(base_mjcf),
        "base_report_path": str(base_report_path.resolve()),
        "base_report_sha256": sha256_file(base_report_path),
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "arena_geometry_signature_sha256": base_report[
            "arena_geometry_signature_sha256"
        ],
        "source_trace_path": str(trace_path.resolve()),
        "source_trace_name": trace_path.name,
        "source_trace_sha256": EXPECTED_TRACE_SHA256,
        "source_trace_bytes": trace_path.stat().st_size,
        "source_observation": {
            "schema": start["schema"],
            "scene": start["scene"],
            "authority_semantics": start["authority_semantics"],
            "sample_index": sample["sample_index"],
            "client_fixed_tick": sample["client_fixed_tick"],
            "round_number": sample["round"]["number"],
            "round_time_remaining": sample["round"]["time_remaining"],
            "fighter_visual_only": [
                sample["fighter_0"]["visual_only"],
                sample["fighter_1"]["visual_only"],
            ],
            "sparring_bot_number": start["scope"]["sparring_bot_number"],
        },
        "audit_commands": commands,
        "bone_layout": EXPECTED_T800_BONES,
        "unity_mapping": {
            "source_position_order": "xyz",
            "mujoco_position_xyz": ["x", "z", "y"],
            "source_quaternion_order": "xyzw",
            "mujoco_quaternion_wxyz": ["-w", "x", "z", "y"],
            "quaternion_normalized": True,
            "quaternion_sign_equivalent": True,
            "source": {
                "repository": "https://github.com/google-deepmind/mujoco",
                "tag": "3.7.0",
                "commit": "72cb2b210da666617924de709406d6aadbe60c71",
                "path": "unity/Runtime/Tools/MjEngineTool.cs",
                "source_blob_sha1": "bf27a1d950099b11a38fcb83d3e03763a5592166",
            },
        },
        "reconstruction_method": {
            "root": "direct coordinate conversion of the recorded Robot.RootTransform world pose",
            "hinges": (
                "independent one-dimensional sign-invariant relative-quaternion fit "
                "over each serialized MuJoCo hinge range"
            ),
            "grid_samples_per_hinge": 721,
            "ternary_refinement_iterations": 45,
            "position_fit_limit_meters": POSITION_FIT_LIMIT_METERS,
            "orientation_fit_limit_radians": ORIENTATION_FIT_LIMIT_RADIANS,
        },
        "reconstructed_slots": reconstruction,
        "composition": {
            "fighter_prefixes": list(FIGHTER_PREFIXES),
            "arena_geom_count": 17,
            "robot_copies": 2,
            "actuators_per_robot": 25,
            "keyframe": KEYFRAME_NAME,
        },
        "validation": validation,
        "claims": {
            "two_t800_actor_identity_observed": True,
            "client_visual_pose_reconstructed": True,
            "static_arena_composed": True,
            "controller_recovered": False,
            "opponent_policy_recovered": False,
            "reward_recovered": False,
            "reset_distribution_recovered": False,
            "held_out_parity_established": False,
        },
        "unknowns_and_limits": [
            "the source is a remote-authoritative client visual observation, not server qpos",
            "one first-active frame does not establish a reset or randomization distribution",
            "hinge coordinates are kinematically reconstructed from named client bone orientations",
            "root linear and angular velocities are recorded as zero at this frame",
            "joint velocities are not recorded; the diagnostic keyframe defaults all qvel to zero",
            "the active ONNX policy/configuration and runtime controller targets remain unavailable",
            "the current direct-motor interface is diagnostic and is not a REK action contract",
            "damage, hits, score, falls, recovery, rewards, round transitions, and Bot 1 logic are absent",
            "zero-control stepping is a finite-integration check; both uncontrolled robots fall",
            "exact server build and server-side physics equivalence remain unknown",
        ],
    }
    return tree, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mjcf", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--pose-command-result", type=Path, required=True)
    parser.add_argument("--bone-command-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("steps must be non-negative")

    tree, report = generate(
        args.base_mjcf.resolve(),
        args.base_report.resolve(),
        args.inventory.resolve(),
        args.trace.resolve(),
        args.pose_command_result.resolve(),
        args.bone_command_result.resolve(),
        args.steps,
    )
    ET.indent(tree, space="  ")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    report["mjcf_path"] = str(args.out.resolve())
    report["mjcf_sha256"] = sha256_file(args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "schema": report["schema"],
        "build_fingerprint": report["build_fingerprint"],
        "source_trace_sha256": report["source_trace_sha256"],
        "dimensions": report["validation"]["dimensions"],
        "initial": report["validation"]["initial"],
        "zero_control_steps": report["validation"]["zero_control_steps"],
        "zero_control_final_finite": report["validation"]["zero_control_final_finite"],
        "control_equivalent": report["control_equivalent"],
        "mjcf_sha256": report["mjcf_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
