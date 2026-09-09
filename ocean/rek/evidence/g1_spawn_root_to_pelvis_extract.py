"""Extract the build-pinned G1 prefab-root-to-pelvis spawn contract.

The extractor joins two independently generated Unity asset probes, the
arena spawn contract, and the recovered G1 MJCF. It emits the exact initial
free-joint pelvis pose implied by Unity's position/rotation Instantiate
boundary. Runtime control, post-spawn dynamics, and server build equality are
outside this contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import mujoco_plant as plant
except ImportError:
    import mujoco_plant as plant


HERE = Path(__file__).resolve().parent
EVIDENCE_OUT = HERE / "evidence_out"
DEFAULT_PROBE_V7_PATH = EVIDENCE_OUT / "mujoco_asset_probe_v7.json"
DEFAULT_PROBE_V8_PATH = EVIDENCE_OUT / "mujoco_asset_probe_v8.json"
DEFAULT_INVENTORY_PATH = EVIDENCE_OUT / "inventory.json"
DEFAULT_ARENA_CONTRACT_PATH = EVIDENCE_OUT / "g1_arena_physics_contract.v1.json"
DEFAULT_RECOVERY_REPORT_PATH = EVIDENCE_OUT / "g1_29dof.recovered.report.json"
DEFAULT_RECOVERED_MJCF_PATH = EVIDENCE_OUT / "g1_29dof.recovered.xml"
DEFAULT_CONTRACT_PATH = HERE / "g1_spawn_root_to_pelvis_contract.v1.json"

CONTRACT_SCHEMA = "rek.g1_spawn_root_to_pelvis_contract.v1"
PROBE_SCHEMA = "rek.mujoco_asset_probe.v1"
ARENA_SCHEMA = "rek.g1_arena_physics_contract.v1"
RECOVERY_SCHEMA = "rek.mujoco_plant_generation.v1"
BUILD_FINGERPRINT = "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
UNITY_VERSION = "6000.5.8f1"
CONTAINER_NAME = "sharedassets0.assets"
CONTAINER_PATH = "REK_Data/sharedassets0.assets"
PREFAB_ROOT_NAME = "g1_29dof_Prefab_SONIC"
PELVIS_PATH = f"{PREFAB_ROOT_NAME}/pelvis"
FREE_JOINT_PATH = f"{PELVIS_PATH}/joint__floating_base_joint"

PINNED_FILES = {
    "probe_v7": {
        "bytes": 4_468_385,
        "sha256": "2e94ebda205da7445c0767722250ddc1e1966a9a951a3a405e41822ac818ef36",
    },
    "probe_v8": {
        "bytes": 4_568_314,
        "sha256": "b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686",
    },
    "inventory": {
        "bytes": 17_991,
        "sha256": "ea932824c7f1fa9781ab816716d4bfca9ec22b14e754466941c8c157910eff79",
    },
    "arena_contract": {
        "bytes": 41_098,
        "sha256": "67128d45f8b5995d57b5ca925a2db7b2d613ede15bfa19f8df46c4e01c60e3e8",
    },
    "recovery_report": {
        "bytes": 2_546,
        "sha256": "5c8e3490cf9b5ba8f5a4a1277d516ed0c59b19162928f5497a29cd643257cea1",
    },
    "recovered_mjcf": {
        "bytes": 60_654,
        "sha256": "811fdc1e5bee74026b780974207cbcd628cdd83a249d3f76b75a668d71aad835",
    },
}
INVENTORY_SHA256 = PINNED_FILES["inventory"]["sha256"]
PROBE_V7_SHA256 = PINNED_FILES["probe_v7"]["sha256"]
PROBE_V8_SHA256 = PINNED_FILES["probe_v8"]["sha256"]
RECOVERED_MJCF_SHA256 = PINNED_FILES["recovered_mjcf"]["sha256"]
MATH_SOURCE_SHA256 = "ac8fc221c11daabf610facc341002563928d4700511354c8ea7f3d00334a8ad3"
PROBE_GENERATOR_SHA256 = {
    "v7": "4d0a0d0f8397b884a094e2d36b82b5730997f3003b1ee2933c32626342159212",
    "v8": "e75e174c73c3567641a710bd6c4e294c315b17ab1e3e4356c60768e54e44e8e4",
}

EXPECTED_BODY = {
    "assembly": "Mujoco.Runtime",
    "namespace": "Mujoco",
    "class": "MjBody",
    "owner": "pelvis",
    "path_id": 3266,
    "serialized_bytes": 36,
    "serialized_sha256": "e5718a6e839d42ef5192fe77cfd18940b973b5d22e678b6de35e0de10035a28c",
}
EXPECTED_FREE_JOINT = {
    "assembly": "Mujoco.Runtime",
    "namespace": "Mujoco",
    "class": "MjFreeJoint",
    "owner": "joint__floating_base_joint",
    "path_id": 3081,
    "serialized_bytes": 32,
    "serialized_sha256": "3c394b416d9b69ce3ae1ecba11d65537c3829985414516233b57ccbb579e14b2",
}
EXPECTED_NATIVE_METHODS = {
    "REKApp.FightCoordinator.SpawnSlot.MoveNext": {
        "length_bytes": 1459,
        "rva": "0x23A2FB0",
        "sha256": "e53c79ed8308cab002c5b348c5f5423887130d189c73b38dd5f2074ec7279596",
        "verified": True,
    },
    "REKApp.RobotSpawner.SpawnLive.MoveNext": {
        "length_bytes": 654,
        "rva": "0x23A2CE0",
        "sha256": "8028333e49b2045204706ca2be9d54f248c27a26b2d9cc293cd8b2d849fff5c7",
        "verified": True,
    },
    "REKApp.RobotSpawner.TryInstantiate": {
        "length_bytes": 757,
        "rva": "0x239DB40",
        "sha256": "1cbcc27d3d1a2ea44f2f694eb236b0fa55e58765fc64e3e0708726c11b78ef14",
        "verified": True,
    },
}
EXPECTED_NATIVE_FLOW = [
    "FightCoordinator.SpawnSlot selects the Transform by slot",
    "RobotSpawner.SpawnLive passes the selected Transform to TryInstantiate",
    "TryInstantiate reads Transform.position and Transform.rotation",
    "TryInstantiate passes both values to UnityEngine.Object.Instantiate",
]


class ExtractionError(RuntimeError):
    """A pinned source, hierarchy, transform, or composition check failed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"value cannot be canonicalized: {exc}") from exc


def _reject_constant(token: str) -> None:
    raise ExtractionError(f"JSON contains nonfinite constant {token!r}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_json_bytes(raw: bytes, label: str) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{label} is not UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ExtractionError:
        raise
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"{label} is invalid JSON: {exc}") from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ExtractionError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ExtractionError(f"{label} must be an array")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExtractionError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtractionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExtractionError(f"{label} must be finite")
    return result


def _vector(value: object, names: Sequence[str], label: str) -> list[float]:
    record = _mapping(value, label)
    if set(record) != set(names):
        raise ExtractionError(f"{label} component names mismatch")
    return [_number(record.get(name), f"{label}.{name}") for name in names]


def _numeric_array(value: object, count: int, label: str) -> list[float]:
    source = _sequence(value, label)
    if len(source) != count:
        raise ExtractionError(f"{label} must contain {count} numbers")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(source)]


def _pinned_bytes(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    expected = PINNED_FILES[label]
    if path.is_symlink() or not path.is_file():
        raise ExtractionError(f"{label} must be a regular, non-symlink file")
    raw = path.read_bytes()
    if len(raw) != expected["bytes"]:
        raise ExtractionError(
            f"{label} byte count mismatch: expected {expected['bytes']}, got {len(raw)}"
        )
    digest = sha256_bytes(raw)
    if digest != expected["sha256"]:
        raise ExtractionError(
            f"{label} SHA-256 mismatch: expected {expected['sha256']}, got {digest}"
        )
    return raw, {"bytes": len(raw), "name": path.name, "sha256": digest}


def load_pinned_json(path: Path, label: str) -> tuple[Mapping[str, object], dict[str, object]]:
    raw, source = _pinned_bytes(path, label)
    return _mapping(load_json_bytes(raw, label), label), source


def _transform_node(value: object, label: str) -> dict[str, object]:
    node = _mapping(value, label)
    expected_keys = {
        "active",
        "container",
        "local_position",
        "local_rotation",
        "local_scale",
        "name",
        "sibling_index",
        "transform_path_id",
    }
    if set(node) != expected_keys:
        raise ExtractionError(f"{label} keys mismatch")
    if not isinstance(node.get("active"), bool):
        raise ExtractionError(f"{label}.active must be Boolean")
    sibling = node.get("sibling_index")
    if sibling is not None:
        sibling = _integer(sibling, f"{label}.sibling_index")
    return {
        "active": node["active"],
        "container": str(node.get("container")),
        "local_position_xyz_m": _vector(
            node.get("local_position"), ("x", "y", "z"), f"{label}.local_position"
        ),
        "local_rotation_wxyz": _vector(
            node.get("local_rotation"), ("w", "x", "y", "z"), f"{label}.local_rotation"
        ),
        "local_scale_xyz": _vector(
            node.get("local_scale"), ("x", "y", "z"), f"{label}.local_scale"
        ),
        "name": str(node.get("name")),
        "sibling_index": sibling,
        "transform_path_id": _integer(
            node.get("transform_path_id"), f"{label}.transform_path_id"
        ),
    }


def _component_projection(
    target: Mapping[str, object], expected: Mapping[str, object], expected_path: str
) -> dict[str, object]:
    label = str(expected["class"])
    for key, wanted in expected.items():
        if target.get(key) != wanted:
            raise ExtractionError(f"{label} {key} mismatch")
    if target.get("enabled") is not True or target.get("parsed") is not True:
        raise ExtractionError(f"{label} is not enabled and parsed")
    if target.get("container") != CONTAINER_NAME:
        raise ExtractionError(f"{label} container mismatch")
    hierarchy = _mapping(target.get("hierarchy"), f"{label}.hierarchy")
    if hierarchy.get("game_object_path") != expected_path:
        raise ExtractionError(f"{label} hierarchy path mismatch")
    chain = [
        _transform_node(item, f"{label}.transform_chain[{index}]")
        for index, item in enumerate(
            _sequence(hierarchy.get("transform_chain"), f"{label}.transform_chain")
        )
    ]
    local_position = _vector(
        hierarchy.get("local_position"), ("x", "y", "z"), f"{label}.local_position"
    )
    local_rotation = _vector(
        hierarchy.get("local_rotation"), ("w", "x", "y", "z"), f"{label}.local_rotation"
    )
    local_scale = _vector(
        hierarchy.get("local_scale"), ("x", "y", "z"), f"{label}.local_scale"
    )
    if not chain or hierarchy.get("transform_path_id") != chain[-1]["transform_path_id"]:
        raise ExtractionError(f"{label} terminal Transform mismatch")
    if (
        local_position != chain[-1]["local_position_xyz_m"]
        or local_rotation != chain[-1]["local_rotation_wxyz"]
        or local_scale != chain[-1]["local_scale_xyz"]
    ):
        raise ExtractionError(f"{label} local transform disagrees with its chain")
    return {
        **{key: target[key] for key in expected},
        "container": target["container"],
        "game_object_path": hierarchy["game_object_path"],
        "transform_chain": chain,
        "transform_path_id": hierarchy["transform_path_id"],
    }


def source_projection(probe: Mapping[str, object]) -> dict[str, object]:
    targets = _sequence(probe.get("targets"), "probe.targets")

    def select(expected: Mapping[str, object], path: str) -> Mapping[str, object]:
        matches = [
            _mapping(item, "probe target")
            for item in targets
            if isinstance(item, dict)
            and item.get("container") == CONTAINER_NAME
            and item.get("class") == expected["class"]
            and item.get("path_id") == expected["path_id"]
        ]
        if len(matches) != 1:
            raise ExtractionError(
                f"expected one {expected['class']}:{expected['path_id']}, got {len(matches)}"
            )
        return matches[0]

    body = _component_projection(select(EXPECTED_BODY, PELVIS_PATH), EXPECTED_BODY, PELVIS_PATH)
    free_joint = _component_projection(
        select(EXPECTED_FREE_JOINT, FREE_JOINT_PATH),
        EXPECTED_FREE_JOINT,
        FREE_JOINT_PATH,
    )
    body_chain = body["transform_chain"]
    joint_chain = free_joint["transform_chain"]
    if len(body_chain) != 2:
        raise ExtractionError("pelvis must be a direct child of the prefab root")
    if len(joint_chain) != 3 or joint_chain[:2] != body_chain:
        raise ExtractionError("free-joint chain does not extend the pelvis chain")
    root_node, pelvis_node = body_chain
    joint_node = joint_chain[-1]
    if (
        root_node["name"] != PREFAB_ROOT_NAME
        or root_node["transform_path_id"] != 1650
        or root_node["sibling_index"] is not None
        or pelvis_node["name"] != "pelvis"
        or pelvis_node["transform_path_id"] != 1759
        or pelvis_node["sibling_index"] != 0
        or joint_node["name"] != "joint__floating_base_joint"
        or joint_node["transform_path_id"] != 1437
        or joint_node["sibling_index"] != 4
    ):
        raise ExtractionError("G1 root, pelvis, or free-joint Transform identity mismatch")
    for index, node in enumerate(joint_chain):
        if node["container"] != CONTAINER_NAME or node["active"] is not True:
            raise ExtractionError(f"Transform chain node {index} is inactive or external")
    if root_node["local_scale_xyz"] != [1.0, 1.0, 1.0]:
        raise ExtractionError("prefab root scale is not identity")
    if pelvis_node["local_scale_xyz"] != [1.0, 1.0, 1.0]:
        raise ExtractionError("pelvis local scale is not identity")
    if joint_node["local_position_xyz_m"] != [0.0, 0.0, 0.0]:
        raise ExtractionError("free-joint object is offset from the pelvis")
    if joint_node["local_rotation_wxyz"] != [1.0, 0.0, 0.0, 0.0]:
        raise ExtractionError("free-joint object is rotated from the pelvis")
    if joint_node["local_scale_xyz"] != [1.0, 1.0, 1.0]:
        raise ExtractionError("free-joint object scale is not identity")
    return {"pelvis_mjbody": body, "pelvis_free_joint": free_joint}


def _validate_probe(probe: Mapping[str, object], target_count: int, label: str) -> None:
    if probe.get("schema") != PROBE_SCHEMA:
        raise ExtractionError(f"{label} schema mismatch")
    if probe.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise ExtractionError(f"{label} build fingerprint mismatch")
    if probe.get("unity_version") != UNITY_VERSION:
        raise ExtractionError(f"{label} Unity version mismatch")
    if probe.get("inventory_sha256") != INVENTORY_SHA256:
        raise ExtractionError(f"{label} inventory hash mismatch")
    targets = _sequence(probe.get("targets"), f"{label}.targets")
    if (
        probe.get("target_count") != target_count
        or probe.get("parsed_target_count") != target_count
        or probe.get("unparsed_target_count") != 0
        or len(targets) != target_count
    ):
        raise ExtractionError(f"{label} target counts mismatch")


def _validate_inventory(inventory: Mapping[str, object]) -> Mapping[str, object]:
    if inventory.get("schema") != 1 or inventory.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise ExtractionError("inventory identity mismatch")
    matches = [
        _mapping(item, "inventory file")
        for item in _sequence(inventory.get("files"), "inventory.files")
        if isinstance(item, dict) and item.get("path") == CONTAINER_PATH
    ]
    if len(matches) != 1:
        raise ExtractionError("inventory must contain one sharedassets0.assets record")
    record = matches[0]
    expected = {
        "path": CONTAINER_PATH,
        "kind": "asset_container",
        "size": 31_315_504,
        "sha256": "37f7a476c56caae37f5a04d4fa1acf5954fdc2b90f20f521830369ecff05f355",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ExtractionError("sharedassets0.assets inventory record mismatch")
    return expected


def _quat_norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _pose_from_arena(spawn: Mapping[str, object], label: str) -> dict[str, list[float]]:
    unity_position = _numeric_array(spawn.get("unity_world_position_m"), 3, f"{label}.unity_position")
    unity_quaternion = _numeric_array(
        spawn.get("unity_world_quaternion_wxyz"), 4, f"{label}.unity_quaternion"
    )
    mujoco_position = _numeric_array(spawn.get("mujoco_world_position_m"), 3, f"{label}.mujoco_position")
    mujoco_quaternion = _numeric_array(
        spawn.get("mujoco_world_quaternion_wxyz"), 4, f"{label}.mujoco_quaternion"
    )
    if not math.isclose(_quat_norm(unity_quaternion), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ExtractionError(f"{label} Unity quaternion is not normalized")
    if list(plant.mj_vector(tuple(unity_position))) != mujoco_position:
        raise ExtractionError(f"{label} position mapping mismatch")
    if list(plant.mj_quaternion(tuple(unity_quaternion))) != mujoco_quaternion:
        raise ExtractionError(f"{label} quaternion mapping mismatch")
    return {
        "mujoco_position_xyz_m": mujoco_position,
        "mujoco_quaternion_wxyz": mujoco_quaternion,
        "unity_position_xyz_m": unity_position,
        "unity_quaternion_wxyz": unity_quaternion,
    }


def _validate_arena(arena: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if arena.get("schema") != ARENA_SCHEMA or arena.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise ExtractionError("arena contract identity mismatch")
    mapping = _mapping(arena.get("coordinate_mapping"), "arena.coordinate_mapping")
    if mapping.get("units") != "metres":
        raise ExtractionError("arena units mismatch")
    if mapping.get("unity_position_xyz_to_mujoco_xyz") != ["x", "z", "y"]:
        raise ExtractionError("arena position convention mismatch")
    if mapping.get("unity_quaternion_wxyz_to_mujoco_wxyz") != ["-w", "x", "z", "y"]:
        raise ExtractionError("arena quaternion convention mismatch")
    spawn_points = _mapping(arena.get("spawn_points"), "arena.spawn_points")
    if spawn_points.get("runtime_observation") is not False:
        raise ExtractionError("arena spawn source must remain classified static")
    if spawn_points.get("identical_pose_across_level_containers") is not True:
        raise ExtractionError("arena spawn transforms differ across levels")
    native = _mapping(spawn_points.get("native_application"), "arena.native_application")
    if native.get("proven_for_pinned_client_build") is not True:
        raise ExtractionError("native spawn application is not build-pinned")
    if native.get("flow") != EXPECTED_NATIVE_FLOW:
        raise ExtractionError("native spawn application flow mismatch")
    sources = _mapping(arena.get("sources"), "arena.sources")
    native_source = _mapping(sources.get("native"), "arena.sources.native")
    methods = _mapping(native_source.get("methods"), "arena.sources.native.methods")
    for name, expected in EXPECTED_NATIVE_METHODS.items():
        if methods.get(name) != expected:
            raise ExtractionError(f"native method pin mismatch for {name}")
    result: dict[str, dict[str, object]] = {}
    for role, slot in (("player", 0), ("opponent", 1)):
        spawn = _mapping(spawn_points.get(role), f"arena.spawn_points.{role}")
        if spawn.get("slot") != slot or spawn.get("fallback_robot_id") != "g1":
            raise ExtractionError(f"{role} spawn identity mismatch")
        result[role] = {"slot": slot, **_pose_from_arena(spawn, role)}
    return result


def _validate_report(report: Mapping[str, object]) -> None:
    expected_mapping = {
        "unity_vector_to_mujoco": ["x", "z", "y"],
        "unity_quaternion_to_mujoco_wxyz": ["-w", "x", "z", "y"],
    }
    if report.get("schema") != RECOVERY_SCHEMA or report.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise ExtractionError("recovery report identity mismatch")
    if report.get("container") != CONTAINER_NAME or report.get("root_name") != PREFAB_ROOT_NAME:
        raise ExtractionError("recovery hierarchy identity mismatch")
    if report.get("root_body_count") != 1:
        raise ExtractionError("recovered plant must contain one root body")
    if report.get("probe_sha256") != PROBE_V7_SHA256:
        raise ExtractionError("recovery report probe hash mismatch")
    if report.get("mjcf_sha256") != RECOVERED_MJCF_SHA256:
        raise ExtractionError("recovery report MJCF hash mismatch")
    mapping = _mapping(report.get("unity_mapping_source"), "report.unity_mapping_source")
    for key, value in expected_mapping.items():
        if mapping.get(key) != value:
            raise ExtractionError(f"recovery mapping mismatch for {key}")


def _node_for_plant(node: Mapping[str, object]) -> dict[str, object]:
    position = node["local_position_xyz_m"]
    rotation = node["local_rotation_wxyz"]
    scale = node["local_scale_xyz"]
    return {
        "local_position": dict(zip(("x", "y", "z"), position)),
        "local_rotation": dict(zip(("w", "x", "y", "z"), rotation)),
        "local_scale": dict(zip(("x", "y", "z"), scale)),
    }


def _xml_pose(root: ET.Element) -> dict[str, list[float]]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ExtractionError("recovered MJCF omits worldbody")
    bodies = [child for child in worldbody if child.tag == "body"]
    if len(bodies) != 1 or bodies[0].get("name") != "pelvis_3266":
        raise ExtractionError("recovered MJCF root body identity mismatch")
    def attribute_numbers(name: str, count: int) -> list[float]:
        fields = str(bodies[0].get(name, "")).split()
        if len(fields) != count:
            raise ExtractionError(f"recovered pelvis {name} must contain {count} numbers")
        try:
            result = [float(field) for field in fields]
        except ValueError as exc:
            raise ExtractionError(f"recovered pelvis {name} contains invalid numbers") from exc
        if not all(math.isfinite(value) for value in result):
            raise ExtractionError(f"recovered pelvis {name} contains nonfinite numbers")
        return result

    return {
        "position_xyz_m": attribute_numbers("pos", 3),
        "quaternion_wxyz": attribute_numbers("quat", 4),
    }


def _rotation_error_rad(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(lhs, rhs)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _compose_spawn(
    spawn: Mapping[str, object], root_node: Mapping[str, object], pelvis_node: Mapping[str, object]
) -> dict[str, object]:
    spawn_node = {
        "local_position": dict(
            zip(("x", "y", "z"), spawn["unity_position_xyz_m"])
        ),
        "local_rotation": dict(
            zip(("w", "x", "y", "z"), spawn["unity_quaternion_wxyz"])
        ),
        "local_scale": dict(zip(("x", "y", "z"), root_node["local_scale_xyz"])),
    }
    world = plant.world_transform([spawn_node, _node_for_plant(pelvis_node)])
    unity_position = list(world.position)
    unity_quaternion = list(world.rotation)
    mujoco_position = list(plant.mj_vector(world.position))
    mujoco_quaternion = list(plant.mj_quaternion(world.rotation))
    spawn_mujoco_quaternion = spawn["mujoco_quaternion_wxyz"]
    if _rotation_error_rad(mujoco_quaternion, spawn_mujoco_quaternion) > 1e-12:
        raise ExtractionError("pelvis local rotation changes the physical spawn orientation")
    return {
        "free_joint_qpos_prefix": [*mujoco_position, *mujoco_quaternion],
        "pelvis_mujoco_world_pose": {
            "position_xyz_m": mujoco_position,
            "quaternion_wxyz": mujoco_quaternion,
        },
        "pelvis_unity_world_pose": {
            "position_xyz_m": unity_position,
            "quaternion_wxyz": unity_quaternion,
        },
        "slot": spawn["slot"],
        "spawn_root_mujoco_world_pose": {
            "position_xyz_m": spawn["mujoco_position_xyz_m"],
            "quaternion_wxyz": spawn_mujoco_quaternion,
        },
        "spawn_root_unity_world_pose": {
            "position_xyz_m": spawn["unity_position_xyz_m"],
            "quaternion_wxyz": spawn["unity_quaternion_wxyz"],
        },
    }


def build_contract(
    probe_v7: Mapping[str, object],
    probe_v8: Mapping[str, object],
    inventory: Mapping[str, object],
    arena: Mapping[str, object],
    recovery_report: Mapping[str, object],
    recovered_mjcf_root: ET.Element,
    source_files: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    _validate_probe(probe_v7, 670, "probe_v7")
    _validate_probe(probe_v8, 715, "probe_v8")
    projection_v7 = source_projection(probe_v7)
    projection_v8 = source_projection(probe_v8)
    if projection_v7 != projection_v8:
        raise ExtractionError("v7 and v8 G1 source projections differ")
    container = _validate_inventory(inventory)
    spawns = _validate_arena(arena)
    _validate_report(recovery_report)

    body = projection_v7["pelvis_mjbody"]
    root_node, pelvis_node = body["transform_chain"]
    serialized_world = plant.world_transform(
        [_node_for_plant(root_node), _node_for_plant(pelvis_node)]
    )
    derived_recovered_pose = {
        "position_xyz_m": list(plant.mj_vector(serialized_world.position)),
        "quaternion_wxyz": list(plant.mj_quaternion(serialized_world.rotation)),
    }
    observed_recovered_pose = _xml_pose(recovered_mjcf_root)
    position_residual = max(
        abs(a - b)
        for a, b in zip(
            derived_recovered_pose["position_xyz_m"],
            observed_recovered_pose["position_xyz_m"],
        )
    )
    rotation_residual = _rotation_error_rad(
        derived_recovered_pose["quaternion_wxyz"],
        observed_recovered_pose["quaternion_wxyz"],
    )
    if position_residual > 1e-12 or rotation_residual > 1e-12:
        raise ExtractionError("serialized hierarchy does not reproduce the recovered MJCF root")

    root_to_pelvis_unity = {
        "position_xyz_m": pelvis_node["local_position_xyz_m"],
        "quaternion_wxyz": pelvis_node["local_rotation_wxyz"],
        "scale_xyz": pelvis_node["local_scale_xyz"],
    }
    root_to_pelvis_mujoco = {
        "position_xyz_m": list(
            plant.mj_vector(tuple(root_to_pelvis_unity["position_xyz_m"]))
        ),
        "quaternion_wxyz": list(
            plant.mj_quaternion(tuple(root_to_pelvis_unity["quaternion_wxyz"]))
        ),
        "scale_xyz": [
            root_to_pelvis_unity["scale_xyz"][0],
            root_to_pelvis_unity["scale_xyz"][2],
            root_to_pelvis_unity["scale_xyz"][1],
        ],
    }
    spawn_poses = {
        role: _compose_spawn(spawn, root_node, pelvis_node)
        for role, spawn in spawns.items()
    }
    evidence_projection = {
        "build_fingerprint": BUILD_FINGERPRINT,
        "source_identity": projection_v7,
        "prefab_root_to_pelvis": {
            "unity": root_to_pelvis_unity,
            "mujoco": root_to_pelvis_mujoco,
        },
        "arena_spawn_to_free_joint_pelvis": spawn_poses,
    }

    source = {
        key: dict(value) for key, value in source_files.items()
    }
    source["probe_v7"].update(
        {
            "generator_sha256": PROBE_GENERATOR_SHA256["v7"],
            "inventory_sha256": INVENTORY_SHA256,
            "schema": PROBE_SCHEMA,
            "unity_version": UNITY_VERSION,
        }
    )
    source["probe_v8"].update(
        {
            "generator_sha256": PROBE_GENERATOR_SHA256["v8"],
            "inventory_sha256": INVENTORY_SHA256,
            "schema": PROBE_SCHEMA,
            "unity_version": UNITY_VERSION,
        }
    )
    source["unity_asset_container"] = {
        "bytes": container["size"],
        "name": container["path"],
        "sha256": container["sha256"],
    }
    source["transform_math"] = {
        "name": "mujoco_plant.py",
        "sha256": MATH_SOURCE_SHA256,
    }

    return {
        "arena_spawn_to_free_joint_pelvis": spawn_poses,
        "build_fingerprint": BUILD_FINGERPRINT,
        "classification": "exact build-pinned static initial-spawn transform",
        "control_equivalent": False,
        "evidence_projection_sha256": sha256_bytes(canonical_json(evidence_projection)),
        "exact_initial_spawn_pose_available": True,
        "math_conventions": {
            "composition": "T_world_pelvis = T_world_spawn_root * T_prefab_root_pelvis",
            "quaternion_component_order": "wxyz",
            "quaternion_equivalence": "q and -q encode the same physical rotation; emitted signs preserve serialized multiplication and mapping",
            "transform_chain_order": "parent_to_child",
            "unity_handedness": "Unity native Transform coordinates",
            "unity_to_mujoco_position_xyz": ["x", "z", "y"],
            "unity_to_mujoco_quaternion_wxyz": ["-w", "x", "z", "y"],
            "units": "metres",
        },
        "prefab_root_to_pelvis": {
            "direct_child": True,
            "mujoco": root_to_pelvis_mujoco,
            "unity": root_to_pelvis_unity,
        },
        "recovered_mjcf_crosscheck": {
            "derived_from_serialized_hierarchy": derived_recovered_pose,
            "maximum_position_abs_error_m": position_residual,
            "observed_root_body": observed_recovered_pose,
            "physical_rotation_error_rad": rotation_residual,
        },
        "schema": CONTRACT_SCHEMA,
        "source": source,
        "source_identity": projection_v7,
        "spawn_gate_status": "closed_for_pinned_client_initial_instantiation",
        "uncertainty": {
            "passive_runtime_capture": "existing G1 traces expose Robot.RootTransform and replicated pelvis bones, not the prefab clone root Transform; they do not independently measure this parent-child transform",
            "post_spawn_scope": "later physics, resets, networking, and controller mutations are outside this initial pose contract",
            "runtime_clone_observation": False,
            "serialized_numeric_uncertainty": "zero at the source float32 round-trip values carried by both hash-pinned probes",
            "server_build_equality_observed": False,
        },
    }


def render_contract(contract: object) -> bytes:
    try:
        return (
            json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExtractionError(f"contract cannot be serialized: {exc}") from exc


def extract_contract(
    probe_v7_path: Path = DEFAULT_PROBE_V7_PATH,
    probe_v8_path: Path = DEFAULT_PROBE_V8_PATH,
    inventory_path: Path = DEFAULT_INVENTORY_PATH,
    arena_contract_path: Path = DEFAULT_ARENA_CONTRACT_PATH,
    recovery_report_path: Path = DEFAULT_RECOVERY_REPORT_PATH,
    recovered_mjcf_path: Path = DEFAULT_RECOVERED_MJCF_PATH,
) -> bytes:
    math_source = (HERE / "mujoco_plant.py").read_bytes()
    if sha256_bytes(math_source) != MATH_SOURCE_SHA256:
        raise ExtractionError("mujoco_plant.py transform math hash mismatch")
    probe_v7, source_v7 = load_pinned_json(probe_v7_path, "probe_v7")
    probe_v8, source_v8 = load_pinned_json(probe_v8_path, "probe_v8")
    inventory, source_inventory = load_pinned_json(inventory_path, "inventory")
    arena, source_arena = load_pinned_json(arena_contract_path, "arena_contract")
    report, source_report = load_pinned_json(recovery_report_path, "recovery_report")
    mjcf_raw, source_mjcf = _pinned_bytes(recovered_mjcf_path, "recovered_mjcf")
    try:
        mjcf_root = ET.fromstring(mjcf_raw)
    except ET.ParseError as exc:
        raise ExtractionError(f"recovered MJCF is invalid XML: {exc}") from exc
    return render_contract(
        build_contract(
            probe_v7,
            probe_v8,
            inventory,
            arena,
            report,
            mjcf_root,
            {
                "arena_contract": source_arena,
                "inventory": source_inventory,
                "probe_v7": source_v7,
                "probe_v8": source_v8,
                "recovered_mjcf": source_mjcf,
                "recovery_report": source_report,
            },
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-v7", type=Path, default=DEFAULT_PROBE_V7_PATH)
    parser.add_argument("--probe-v8", type=Path, default=DEFAULT_PROBE_V8_PATH)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument(
        "--arena-contract", type=Path, default=DEFAULT_ARENA_CONTRACT_PATH
    )
    parser.add_argument(
        "--recovery-report", type=Path, default=DEFAULT_RECOVERY_REPORT_PATH
    )
    parser.add_argument(
        "--recovered-mjcf", type=Path, default=DEFAULT_RECOVERED_MJCF_PATH
    )
    parser.add_argument(
        "--check", type=Path, help="require this existing contract to match exactly"
    )
    arguments = parser.parse_args(argv)
    try:
        rendered = extract_contract(
            arguments.probe_v7,
            arguments.probe_v8,
            arguments.inventory,
            arguments.arena_contract,
            arguments.recovery_report,
            arguments.recovered_mjcf,
        )
        if arguments.check is None:
            sys.stdout.buffer.write(rendered)
        else:
            if arguments.check.is_symlink() or not arguments.check.is_file():
                raise ExtractionError("checked contract must be a regular file")
            if arguments.check.read_bytes() != rendered:
                raise ExtractionError("checked contract differs from pinned extraction")
    except (ExtractionError, OSError) as exc:
        print(f"G1 spawn transform extraction failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
