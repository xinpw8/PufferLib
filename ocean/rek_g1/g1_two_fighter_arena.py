"""Build-pinned two-G1 initial-spawn arena without combat semantics.

The exact static prefab-root-to-pelvis contract composes each arena spawn root
with the G1 pelvis local transform.  The resulting player and opponent pelvis
poses initialize their respective MuJoCo free joints.  Static world-relative
spawn-root frames remain in the model as inspectable source references.

Names, name-valued references, and each cloned root body's spawn ``pos`` and
``quat`` are the only robot XML changes.  Every other numeric and Boolean
robot XML attribute is retained byte-for-byte.  The 17 arena boxes come from
the hash-pinned arena contract through ``sonic_candidate.add_arena_geoms``.
This reference makes no claim about later dynamics, control parity, server
build equality, networking, or combat behavior.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

import sonic_candidate as plant_contract


SCHEMA = "rek.g1_two_fighter_arena_reference.v2"
CLASSIFICATION = "build_pinned_static_initial_spawn_reference"
ROLES = ("player", "opponent")
NAMESPACE_SEPARATOR = "__"
MODEL_NAME = "rek_g1_two_fighter_arena_reference"
EXPECTED_MODEL_DIMENSIONS = {
    "nbody": 63,
    "njnt": 60,
    "nq": 72,
    "nv": 70,
    "nu": 58,
    "ngeom": 91,
}
SPAWN_FRAME_NAMES = {
    "player": "arena_spawn_reference__player",
    "opponent": "arena_spawn_reference__opponent",
}
SPAWN_CONTRACT_SCHEMA = "rek.g1_spawn_root_to_pelvis_contract.v1"
SPAWN_CONTRACT_BYTES = 11_442
SPAWN_CONTRACT_SHA256 = (
    "18f6444cfcb6d9972dc237c4e9f78cb449302a5440c830d0acba6aef2ad8087d"
)
SPAWN_EVIDENCE_PROJECTION_SHA256 = (
    "b0213e7a154ca55ab59cab6f05b5fb708ceca5549404380cd32b695793b98978"
)
SPAWN_REBASE_STATUS = (
    "applied_from_exact_build_pinned_static_initial_spawn_contract"
)
STALE_ARENA_SPAWN_UNKNOWN = (
    "the serialized spawn references establish arena anchors; this contract "
    "does not infer a replacement free-joint pelvis height"
)
SPAWN_SOURCE_SHA256 = {
    "arena_contract": "67128d45f8b5995d57b5ca925a2db7b2d613ede15bfa19f8df46c4e01c60e3e8",
    "inventory": "ea932824c7f1fa9781ab816716d4bfca9ec22b14e754466941c8c157910eff79",
    "probe_v7": "2e94ebda205da7445c0767722250ddc1e1966a9a951a3a405e41822ac818ef36",
    "probe_v8": "b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686",
    "recovered_mjcf": "811fdc1e5bee74026b780974207cbcd628cdd83a249d3f76b75a668d71aad835",
    "recovery_report": "5c8e3490cf9b5ba8f5a4a1277d516ed0c59b19162928f5497a29cd643257cea1",
    "transform_math": "ac8fc221c11daabf610facc341002563928d4700511354c8ea7f3d00334a8ad3",
    "unity_asset_container": "37f7a476c56caae37f5a04d4fa1acf5954fdc2b90f20f521830369ecff05f355",
}
EXPECTED_SPAWN_QPOS = {
    "player": (
        -0.9000000357627869,
        0.0,
        0.8029999826103449,
        0.9999999999998863,
        0.0,
        0.0,
        -4.7683710135965193e-07,
    ),
    "opponent": (
        0.8999999761581421,
        0.0,
        0.8029999826103449,
        -1.1026858146572184e-06,
        -0.0,
        0.0,
        0.999999999999392,
    ),
}
EXCLUDED_BEHAVIORS = (
    "damage",
    "hit_zones",
    "rewards",
    "networking",
    "opponent_policy",
    "runtime_clip_selection",
)

_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_SUPPORTED_NAMED_TAGS = frozenset({"body", "freejoint", "joint", "geom", "motor"})
_KNOWN_REFERENCE_ATTRIBUTES = frozenset(
    {
        "body",
        "body1",
        "body2",
        "childclass",
        "class",
        "geom",
        "geom1",
        "geom2",
        "hfield",
        "joint",
        "joint1",
        "joint2",
        "jointinparent",
        "material",
        "mesh",
        "name1",
        "name2",
        "objname",
        "refname",
        "site",
        "site1",
        "site2",
        "target",
        "tendon",
        "texture",
    }
)
_SUPPORTED_REFERENCES = {("motor", "joint"): frozenset({"joint"})}


class TwoFighterArenaError(RuntimeError):
    """A fail-closed source, namespace, compilation, or mapping error."""


@dataclass(frozen=True)
class SpawnAnchorReference:
    role: str
    slot: int
    frame_body_name: str
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    runtime_observation: bool


@dataclass(frozen=True)
class SpawnRebaseReference:
    role: str
    slot: int
    spawn_root_position_m: tuple[float, float, float]
    spawn_root_quaternion_wxyz: tuple[float, float, float, float]
    pelvis_position_m: tuple[float, float, float]
    pelvis_quaternion_wxyz: tuple[float, float, float, float]
    free_joint_qpos_prefix: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class FighterNamespace:
    role: str
    namespace: str
    root_body_name: str
    free_joint_name: str
    anchor_body_name: str
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    geom_names: tuple[str, ...]


@dataclass(frozen=True)
class TwoFighterArenaContract:
    schema: str
    classification: str
    build_fingerprint: str
    control_equivalent: bool
    xml_text: str
    xml_sha256: str
    recovered_xml_sha256: str
    arena_contract_sha256: str
    arena_geometry_sha256: str
    spawn_contract_sha256: str
    spawn_evidence_projection_sha256: str
    runtime_manifest_file_sha256: str
    runtime_manifest_canonical_sha256: str
    runtime_asset_roles: tuple[str, ...]
    robot_template_canonical_sha256: str
    serialized_root_position_m: tuple[float, float, float]
    serialized_root_quaternion_wxyz: tuple[float, float, float, float]
    fighters: tuple[FighterNamespace, FighterNamespace]
    spawn_anchors: tuple[SpawnAnchorReference, SpawnAnchorReference]
    spawn_rebases: tuple[SpawnRebaseReference, SpawnRebaseReference]
    spawn_rebase_applied: bool
    spawn_rebase_status: str
    excluded_behaviors: tuple[str, ...]
    unknowns: tuple[str, ...]

    def fighter(self, role: str) -> FighterNamespace:
        matches = [fighter for fighter in self.fighters if fighter.role == role]
        if len(matches) != 1:
            raise TwoFighterArenaError(f"unknown or duplicate fighter role {role!r}")
        return matches[0]

    def spawn_anchor(self, role: str) -> SpawnAnchorReference:
        matches = [anchor for anchor in self.spawn_anchors if anchor.role == role]
        if len(matches) != 1:
            raise TwoFighterArenaError(f"unknown or duplicate spawn role {role!r}")
        return matches[0]

    def spawn_rebase(self, role: str) -> SpawnRebaseReference:
        matches = [rebase for rebase in self.spawn_rebases if rebase.role == role]
        if len(matches) != 1:
            raise TwoFighterArenaError(f"unknown or duplicate rebase role {role!r}")
        return matches[0]


@dataclass(frozen=True)
class TwoFighterArenaReference:
    contract: TwoFighterArenaContract
    mujoco: Any
    model: Any
    mujoco_version: str
    runtime_maps: Mapping[str, plant_contract.RuntimeMap]

    def runtime_map(self, role: str) -> plant_contract.RuntimeMap:
        try:
            return self.runtime_maps[role]
        except KeyError as exc:
            raise TwoFighterArenaError(f"unknown fighter role {role!r}") from exc


@dataclass(frozen=True)
class _RuntimeManifestFacts:
    file_sha256: str
    canonical_sha256: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class _SpawnContractFacts:
    file_sha256: str
    evidence_projection_sha256: str
    rebases: tuple[SpawnRebaseReference, SpawnRebaseReference]


def _default_xml_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rek"
        / "evidence"
        / "evidence_out"
        / "g1_29dof.recovered.xml"
    )


def _default_arena_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rek"
        / "evidence"
        / "evidence_out"
        / "g1_arena_physics_contract.v1.json"
    )


def _default_spawn_contract_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rek"
        / "evidence"
        / "g1_spawn_root_to_pelvis_contract.v1.json"
    )


def _default_runtime_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rek"
        / "evidence"
        / "g1_runtime_assets.v1.json"
    )


def _regular_file(path: Path, label: str) -> Path:
    resolved = Path(os.path.abspath(path))
    if resolved.is_symlink() or not resolved.is_file():
        raise TwoFighterArenaError(f"{label} must be a regular, non-symlink file")
    return resolved


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_numbers(value: Sequence[Any], length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise TwoFighterArenaError(f"{label} must contain exactly {length} numbers")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TwoFighterArenaError(f"{label}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise TwoFighterArenaError(f"{label}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TwoFighterArenaError(f"{label} must be an object")
    return value


def _reject_json_constant(value: str) -> None:
    raise TwoFighterArenaError(f"JSON contains nonfinite constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TwoFighterArenaError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_spawn_component_identities(value: Any) -> None:
    identities = _object(value, "spawn contract source_identity")
    if set(identities) != {"pelvis_mjbody", "pelvis_free_joint"}:
        raise TwoFighterArenaError("spawn contract component identity set mismatch")
    expected_components = {
        "pelvis_mjbody": {
            "assembly": "Mujoco.Runtime",
            "class": "MjBody",
            "container": "sharedassets0.assets",
            "game_object_path": "g1_29dof_Prefab_SONIC/pelvis",
            "namespace": "Mujoco",
            "owner": "pelvis",
            "path_id": 3266,
            "serialized_bytes": 36,
            "serialized_sha256": "e5718a6e839d42ef5192fe77cfd18940b973b5d22e678b6de35e0de10035a28c",
            "transform_path_id": 1759,
        },
        "pelvis_free_joint": {
            "assembly": "Mujoco.Runtime",
            "class": "MjFreeJoint",
            "container": "sharedassets0.assets",
            "game_object_path": (
                "g1_29dof_Prefab_SONIC/pelvis/joint__floating_base_joint"
            ),
            "namespace": "Mujoco",
            "owner": "joint__floating_base_joint",
            "path_id": 3081,
            "serialized_bytes": 32,
            "serialized_sha256": "3c394b416d9b69ce3ae1ecba11d65537c3829985414516233b57ccbb579e14b2",
            "transform_path_id": 1437,
        },
    }
    root_node = {
        "active": True,
        "container": "sharedassets0.assets",
        "local_position_xyz_m": [
            18.2549991607666,
            -9.392000198364258,
            0.36000001430511475,
        ],
        "local_rotation_wxyz": [
            0.5054116249084473,
            -0.0,
            -0.8628783822059631,
            -0.0,
        ],
        "local_scale_xyz": [1.0, 1.0, 1.0],
        "name": "g1_29dof_Prefab_SONIC",
        "sibling_index": None,
        "transform_path_id": 1650,
    }
    pelvis_node = {
        "active": True,
        "container": "sharedassets0.assets",
        "local_position_xyz_m": [0.0, 0.7929999828338623, 0.0],
        "local_rotation_wxyz": [-1.0, 0.0, 0.0, 0.0],
        "local_scale_xyz": [1.0, 1.0, 1.0],
        "name": "pelvis",
        "sibling_index": 0,
        "transform_path_id": 1759,
    }
    free_joint_node = {
        "active": True,
        "container": "sharedassets0.assets",
        "local_position_xyz_m": [0.0, 0.0, 0.0],
        "local_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "local_scale_xyz": [1.0, 1.0, 1.0],
        "name": "joint__floating_base_joint",
        "sibling_index": 4,
        "transform_path_id": 1437,
    }
    expected_chains = {
        "pelvis_mjbody": [root_node, pelvis_node],
        "pelvis_free_joint": [root_node, pelvis_node, free_joint_node],
    }
    for name, expected in expected_components.items():
        component = _object(identities.get(name), f"spawn contract {name}")
        if set(component) != {*expected, "transform_chain"}:
            raise TwoFighterArenaError(f"spawn contract {name} keys mismatch")
        if any(component.get(key) != item for key, item in expected.items()):
            raise TwoFighterArenaError(f"spawn contract {name} identity mismatch")
        if component.get("transform_chain") != expected_chains[name]:
            raise TwoFighterArenaError(f"spawn contract {name} hierarchy mismatch")


def _validate_spawn_contract_value(
    value: Any,
    file_sha256: str,
) -> _SpawnContractFacts:
    if file_sha256 != SPAWN_CONTRACT_SHA256:
        raise TwoFighterArenaError("G1 spawn transform contract file identity mismatch")
    contract = _object(value, "G1 spawn transform contract")
    expected_keys = {
        "arena_spawn_to_free_joint_pelvis",
        "build_fingerprint",
        "classification",
        "control_equivalent",
        "evidence_projection_sha256",
        "exact_initial_spawn_pose_available",
        "math_conventions",
        "prefab_root_to_pelvis",
        "recovered_mjcf_crosscheck",
        "schema",
        "source",
        "source_identity",
        "spawn_gate_status",
        "uncertainty",
    }
    if set(contract) != expected_keys:
        raise TwoFighterArenaError("G1 spawn transform contract keys mismatch")
    if contract.get("schema") != SPAWN_CONTRACT_SCHEMA:
        raise TwoFighterArenaError("G1 spawn transform contract schema mismatch")
    if contract.get("build_fingerprint") != plant_contract.BUILD_FINGERPRINT:
        raise TwoFighterArenaError(
            "G1 spawn transform contract build fingerprint mismatch"
        )
    if (
        contract.get("classification")
        != "exact build-pinned static initial-spawn transform"
        or contract.get("control_equivalent") is not False
        or contract.get("exact_initial_spawn_pose_available") is not True
        or contract.get("spawn_gate_status")
        != "closed_for_pinned_client_initial_instantiation"
    ):
        raise TwoFighterArenaError("G1 spawn transform contract scope mismatch")
    if contract.get("evidence_projection_sha256") != SPAWN_EVIDENCE_PROJECTION_SHA256:
        raise TwoFighterArenaError("G1 spawn evidence projection mismatch")
    expected_math = {
        "composition": "T_world_pelvis = T_world_spawn_root * T_prefab_root_pelvis",
        "quaternion_component_order": "wxyz",
        "quaternion_equivalence": (
            "q and -q encode the same physical rotation; emitted signs preserve "
            "serialized multiplication and mapping"
        ),
        "transform_chain_order": "parent_to_child",
        "units": "metres",
        "unity_handedness": "Unity native Transform coordinates",
        "unity_to_mujoco_position_xyz": ["x", "z", "y"],
        "unity_to_mujoco_quaternion_wxyz": ["-w", "x", "z", "y"],
    }
    if contract.get("math_conventions") != expected_math:
        raise TwoFighterArenaError("G1 spawn transform math conventions mismatch")
    expected_local_transform = {
        "direct_child": True,
        "mujoco": {
            "position_xyz_m": [0.0, 0.0, 0.7929999828338623],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "scale_xyz": [1.0, 1.0, 1.0],
        },
        "unity": {
            "position_xyz_m": [0.0, 0.7929999828338623, 0.0],
            "quaternion_wxyz": [-1.0, 0.0, 0.0, 0.0],
            "scale_xyz": [1.0, 1.0, 1.0],
        },
    }
    if contract.get("prefab_root_to_pelvis") != expected_local_transform:
        raise TwoFighterArenaError("G1 prefab-root-to-pelvis transform mismatch")
    sources = _object(contract.get("source"), "G1 spawn transform sources")
    if set(sources) != set(SPAWN_SOURCE_SHA256):
        raise TwoFighterArenaError("G1 spawn transform source set mismatch")
    for name, expected_sha256 in SPAWN_SOURCE_SHA256.items():
        source = _object(sources.get(name), f"G1 spawn source {name}")
        if source.get("sha256") != expected_sha256:
            raise TwoFighterArenaError(f"G1 spawn source {name} SHA-256 mismatch")
    expected_source_files = {
        "arena_contract": ("g1_arena_physics_contract.v1.json", 41_098),
        "inventory": ("inventory.json", 17_991),
        "probe_v7": ("mujoco_asset_probe_v7.json", 4_468_385),
        "probe_v8": ("mujoco_asset_probe_v8.json", 4_568_314),
        "recovered_mjcf": ("g1_29dof.recovered.xml", 60_654),
        "recovery_report": ("g1_29dof.recovered.report.json", 2_546),
        "unity_asset_container": ("REK_Data/sharedassets0.assets", 31_315_504),
    }
    for name, (expected_name, expected_bytes) in expected_source_files.items():
        source = sources[name]
        if source.get("name") != expected_name or source.get("bytes") != expected_bytes:
            raise TwoFighterArenaError(f"G1 spawn source {name} file identity mismatch")
    if sources["transform_math"] != {
        "name": "mujoco_plant.py",
        "sha256": SPAWN_SOURCE_SHA256["transform_math"],
    }:
        raise TwoFighterArenaError("G1 spawn transform math identity mismatch")
    expected_probe_metadata = {
        "probe_v7": "4d0a0d0f8397b884a094e2d36b82b5730997f3003b1ee2933c32626342159212",
        "probe_v8": "e75e174c73c3567641a710bd6c4e294c315b17ab1e3e4356c60768e54e44e8e4",
    }
    for name, generator_sha256 in expected_probe_metadata.items():
        source = sources[name]
        if (
            source.get("generator_sha256") != generator_sha256
            or source.get("schema") != "rek.mujoco_asset_probe.v1"
            or source.get("unity_version") != "6000.5.8f1"
        ):
            raise TwoFighterArenaError(f"G1 spawn source {name} metadata mismatch")
    if sources["probe_v7"].get("inventory_sha256") != SPAWN_SOURCE_SHA256["inventory"]:
        raise TwoFighterArenaError("G1 spawn probe v7 inventory identity mismatch")
    if sources["probe_v8"].get("inventory_sha256") != SPAWN_SOURCE_SHA256["inventory"]:
        raise TwoFighterArenaError("G1 spawn probe v8 inventory identity mismatch")
    _validate_spawn_component_identities(contract.get("source_identity"))
    crosscheck = _object(
        contract.get("recovered_mjcf_crosscheck"),
        "G1 spawn recovered MJCF crosscheck",
    )
    if (
        crosscheck.get("maximum_position_abs_error_m") != 0.0
        or crosscheck.get("physical_rotation_error_rad") != 0.0
        or crosscheck.get("derived_from_serialized_hierarchy")
        != crosscheck.get("observed_root_body")
    ):
        raise TwoFighterArenaError("G1 recovered MJCF crosscheck mismatch")
    uncertainty = _object(contract.get("uncertainty"), "G1 spawn uncertainty")
    if (
        uncertainty.get("runtime_clone_observation") is not False
        or uncertainty.get("server_build_equality_observed") is not False
    ):
        raise TwoFighterArenaError("G1 spawn uncertainty classification mismatch")

    spawn_values = _object(
        contract.get("arena_spawn_to_free_joint_pelvis"),
        "G1 arena spawn pelvis poses",
    )
    if set(spawn_values) != set(ROLES):
        raise TwoFighterArenaError("G1 spawn role set mismatch")
    rebases: list[SpawnRebaseReference] = []
    for role, expected_slot in (("player", 0), ("opponent", 1)):
        item = _object(spawn_values.get(role), f"G1 {role} spawn rebase")
        if item.get("slot") != expected_slot:
            raise TwoFighterArenaError(f"G1 {role} spawn slot mismatch")
        spawn_root = _object(
            item.get("spawn_root_mujoco_world_pose"),
            f"G1 {role} spawn root pose",
        )
        pelvis = _object(
            item.get("pelvis_mujoco_world_pose"),
            f"G1 {role} pelvis pose",
        )
        spawn_root_position = _finite_numbers(
            spawn_root.get("position_xyz_m"), 3, f"G1 {role} spawn root position"
        )
        spawn_root_quaternion = _finite_numbers(
            spawn_root.get("quaternion_wxyz"),
            4,
            f"G1 {role} spawn root quaternion",
        )
        pelvis_position = _finite_numbers(
            pelvis.get("position_xyz_m"), 3, f"G1 {role} pelvis position"
        )
        pelvis_quaternion = _finite_numbers(
            pelvis.get("quaternion_wxyz"), 4, f"G1 {role} pelvis quaternion"
        )
        qpos = _finite_numbers(
            item.get("free_joint_qpos_prefix"), 7, f"G1 {role} free-joint qpos"
        )
        if qpos != pelvis_position + pelvis_quaternion:
            raise TwoFighterArenaError(f"G1 {role} pelvis pose and qpos disagree")
        if qpos != EXPECTED_SPAWN_QPOS[role]:
            raise TwoFighterArenaError(f"G1 {role} free-joint qpos mismatch")
        for label, quaternion in (
            ("spawn root", spawn_root_quaternion),
            ("pelvis", pelvis_quaternion),
        ):
            if not math.isclose(
                math.sqrt(sum(component * component for component in quaternion)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise TwoFighterArenaError(
                    f"G1 {role} {label} quaternion is not normalized"
                )
        rebases.append(
            SpawnRebaseReference(
                role=role,
                slot=expected_slot,
                spawn_root_position_m=spawn_root_position,
                spawn_root_quaternion_wxyz=spawn_root_quaternion,
                pelvis_position_m=pelvis_position,
                pelvis_quaternion_wxyz=pelvis_quaternion,
                free_joint_qpos_prefix=qpos,
            )
        )
    return _SpawnContractFacts(
        file_sha256=file_sha256,
        evidence_projection_sha256=SPAWN_EVIDENCE_PROJECTION_SHA256,
        rebases=tuple(rebases),  # type: ignore[arg-type]
    )


def _load_spawn_contract(path: Path) -> _SpawnContractFacts:
    path = _regular_file(path, "G1 spawn transform contract")
    raw = path.read_bytes()
    if len(raw) != SPAWN_CONTRACT_BYTES:
        raise TwoFighterArenaError("G1 spawn transform contract byte count mismatch")
    file_sha256 = _sha256_bytes(raw)
    if file_sha256 != SPAWN_CONTRACT_SHA256:
        raise TwoFighterArenaError("G1 spawn transform contract SHA-256 mismatch")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except TwoFighterArenaError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoFighterArenaError(
            f"G1 spawn transform contract is invalid JSON: {exc}"
        ) from exc
    return _validate_spawn_contract_value(value, file_sha256)


def _mjcf_numbers(values: Sequence[float]) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def _load_runtime_manifest(path: Path) -> _RuntimeManifestFacts:
    path = _regular_file(path, "G1 runtime asset manifest")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoFighterArenaError(f"G1 runtime asset manifest is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TwoFighterArenaError("G1 runtime asset manifest must be an object")
    canonical_sha256 = plant_contract.canonical_json_sha256(value)
    if canonical_sha256 != plant_contract.EXPECTED_MANIFEST_CANONICAL_SHA256:
        raise TwoFighterArenaError("G1 runtime asset manifest canonical SHA-256 mismatch")
    if value.get("schema") != plant_contract.MANIFEST_SCHEMA:
        raise TwoFighterArenaError("G1 runtime asset manifest schema mismatch")
    if value.get("build_fingerprint") != plant_contract.BUILD_FINGERPRINT:
        raise TwoFighterArenaError("G1 runtime asset manifest build fingerprint mismatch")
    assets = value.get("assets")
    if not isinstance(assets, list) or len(assets) != 8:
        raise TwoFighterArenaError("G1 runtime asset manifest must contain eight assets")
    roles: list[str] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise TwoFighterArenaError(f"runtime asset {index} must be an object")
        role = asset.get("role")
        if not isinstance(role, str) or not role:
            raise TwoFighterArenaError(f"runtime asset {index} role is invalid")
        if asset.get("dof") != 29 or float(asset.get("fps", math.nan)) != 50.0:
            raise TwoFighterArenaError(f"runtime asset {role!r} G1 dimensions mismatch")
        roles.append(role)
    if len(set(roles)) != len(roles):
        raise TwoFighterArenaError("G1 runtime asset roles are not unique")
    return _RuntimeManifestFacts(
        file_sha256=_sha256_bytes(raw),
        canonical_sha256=canonical_sha256,
        roles=tuple(roles),
    )


def _source_robot_elements(
    source_bytes: bytes,
) -> tuple[ET.Element, ET.Element, tuple[ET.Element, ...]]:
    try:
        root = ET.fromstring(source_bytes)
    except ET.ParseError as exc:
        raise TwoFighterArenaError(f"recovered XML parse failed: {exc}") from exc
    expected_top_level = ("compiler", "option", "size", "custom", "worldbody", "actuator")
    if tuple(child.tag for child in root) != expected_top_level:
        raise TwoFighterArenaError("recovered XML top-level section layout mismatch")
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    if worldbody is None or actuator is None:
        raise TwoFighterArenaError("recovered XML omits worldbody or actuator")
    world_children = list(worldbody)
    motors = tuple(list(actuator))
    if len(world_children) != 1 or world_children[0].tag != "body":
        raise TwoFighterArenaError("recovered XML must contain one top-level robot body")
    if len(motors) != 29 or any(motor.tag != "motor" for motor in motors):
        raise TwoFighterArenaError("recovered XML must contain 29 direct motors")
    return root, world_children[0], motors


def _robot_elements(root_body: ET.Element, motors: Sequence[ET.Element]) -> list[ET.Element]:
    result = list(root_body.iter())
    for motor in motors:
        result.extend(list(motor.iter()))
    return result


def _namespace_name(namespace: str, source_name: str) -> str:
    return f"{namespace}{NAMESPACE_SEPARATOR}{source_name}"


def _namespace_robot(
    root_body: ET.Element,
    motors: Sequence[ET.Element],
    namespace: str,
) -> None:
    """Namespace one robot and reject every unsupported or unresolved reference."""
    if _SAFE_NAME.fullmatch(namespace) is None or NAMESPACE_SEPARATOR in namespace:
        raise TwoFighterArenaError(f"unsafe fighter namespace {namespace!r}")
    elements = _robot_elements(root_body, motors)
    name_owners: dict[str, str] = {}
    for element in elements:
        source_name = element.get("name")
        if source_name is None:
            continue
        if element.tag not in _SUPPORTED_NAMED_TAGS:
            raise TwoFighterArenaError(
                f"unsupported named MJCF element {element.tag!r} in robot copy"
            )
        if _SAFE_NAME.fullmatch(source_name) is None:
            raise TwoFighterArenaError(f"unsafe recovered MJCF name {source_name!r}")
        if source_name in name_owners:
            raise TwoFighterArenaError(f"duplicate recovered MJCF name {source_name!r}")
        name_owners[source_name] = element.tag
    if len(name_owners) != 126:
        raise TwoFighterArenaError(
            f"recovered robot must expose 126 unique names, got {len(name_owners)}"
        )
    renamed = {name: _namespace_name(namespace, name) for name in name_owners}

    for element in elements:
        for attribute, value in tuple(element.attrib.items()):
            if attribute == "name":
                element.set(attribute, renamed[value])
                continue
            if attribute in _KNOWN_REFERENCE_ATTRIBUTES:
                expected_target_tags = _SUPPORTED_REFERENCES.get((element.tag, attribute))
                if expected_target_tags is None:
                    raise TwoFighterArenaError(
                        f"unsupported robot name reference {element.tag}.{attribute}"
                    )
                target_tag = name_owners.get(value)
                if target_tag is None:
                    raise TwoFighterArenaError(
                        f"unresolved robot name reference {element.tag}.{attribute}={value!r}"
                    )
                if target_tag not in expected_target_tags:
                    raise TwoFighterArenaError(
                        f"robot name reference {element.tag}.{attribute} targets {target_tag!r}"
                    )
                element.set(attribute, renamed[value])
                continue
            if value in name_owners:
                raise TwoFighterArenaError(
                    f"undeclared robot name reference {element.tag}.{attribute}={value!r}"
                )

    namespaced_owners: dict[str, str] = {}
    for element in elements:
        name = element.get("name")
        if name is not None:
            if not name.startswith(f"{namespace}{NAMESPACE_SEPARATOR}"):
                raise TwoFighterArenaError(f"robot name escaped namespace {namespace!r}")
            namespaced_owners[name] = element.tag
        for attribute, value in element.attrib.items():
            if attribute not in _KNOWN_REFERENCE_ATTRIBUTES:
                continue
            expected_target_tags = _SUPPORTED_REFERENCES.get((element.tag, attribute))
            if expected_target_tags is None:
                raise TwoFighterArenaError(
                    f"unsupported namespaced reference {element.tag}.{attribute}"
                )
            target_tag = namespaced_owners.get(value)
            if target_tag is None:
                # Later elements may own the reference target.  Resolve after collection.
                continue
            if target_tag not in expected_target_tags:
                raise TwoFighterArenaError(
                    f"namespaced reference {element.tag}.{attribute} targets {target_tag!r}"
                )
    for element in elements:
        for attribute, value in element.attrib.items():
            if attribute in _KNOWN_REFERENCE_ATTRIBUTES:
                expected_target_tags = _SUPPORTED_REFERENCES.get((element.tag, attribute))
                target_tag = namespaced_owners.get(value)
                if expected_target_tags is None or target_tag not in expected_target_tags:
                    raise TwoFighterArenaError(
                        f"unresolved namespaced reference {element.tag}.{attribute}={value!r}"
                    )


def _element_projection(element: ET.Element, namespace: str | None) -> Mapping[str, Any]:
    prefix = None if namespace is None else f"{namespace}{NAMESPACE_SEPARATOR}"
    attributes: list[tuple[str, str]] = []
    for key, raw_value in element.attrib.items():
        value = raw_value
        if key == "name" or key in _KNOWN_REFERENCE_ATTRIBUTES:
            if prefix is not None:
                if not value.startswith(prefix):
                    raise TwoFighterArenaError(
                        f"projected {element.tag}.{key} escaped namespace {namespace!r}"
                    )
                value = value[len(prefix) :]
        attributes.append((key, value))
    return {
        "tag": element.tag,
        "attributes": sorted(attributes),
        "children": [_element_projection(child, namespace) for child in element],
    }


def _robot_projection_sha256(
    root_body: ET.Element,
    motors: Sequence[ET.Element],
    namespace: str | None,
) -> str:
    projection = {
        "root_body": _element_projection(root_body, namespace),
        "motors": [_element_projection(motor, namespace) for motor in motors],
    }
    raw = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _robot_nonspawn_projection_sha256(
    root_body: ET.Element,
    motors: Sequence[ET.Element],
    namespace: str | None,
) -> str:
    projected_body = copy.deepcopy(root_body)
    if "pos" not in projected_body.attrib or "quat" not in projected_body.attrib:
        raise TwoFighterArenaError("recovered root body omits its serialized pose")
    del projected_body.attrib["pos"]
    del projected_body.attrib["quat"]
    return _robot_projection_sha256(projected_body, motors, namespace)


def _apply_spawn_rebase(
    root_body: ET.Element,
    rebase: SpawnRebaseReference,
) -> None:
    before = dict(root_body.attrib)
    if "pos" not in before or "quat" not in before:
        raise TwoFighterArenaError("fighter root body omits pos or quat")
    root_body.set("pos", _mjcf_numbers(rebase.pelvis_position_m))
    root_body.set("quat", _mjcf_numbers(rebase.pelvis_quaternion_wxyz))
    expected = {
        **before,
        "pos": _mjcf_numbers(rebase.pelvis_position_m),
        "quat": _mjcf_numbers(rebase.pelvis_quaternion_wxyz),
    }
    if root_body.attrib != expected:
        raise TwoFighterArenaError(
            f"{rebase.role} spawn rebase changed unexpected root attributes"
        )


def _canonical_body_name(xml_name: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]+)", xml_name)
    if match is None:
        raise TwoFighterArenaError(f"unrecognized recovered body name {xml_name!r}")
    return match.group(1)


def _fighter_namespace(
    role: str,
    namespace: str,
    source_body: ET.Element,
    source_motors: Sequence[ET.Element],
    xml_contract: plant_contract.XmlContract,
) -> FighterNamespace:
    source_bodies = tuple(body.get("name") for body in source_body.iter("body"))
    source_joints = tuple(joint.get("name") for joint in source_body.iter("joint"))
    source_geoms = tuple(geom.get("name") for geom in source_body.iter("geom"))
    source_actuators = tuple(motor.get("name") for motor in source_motors)
    if any(name is None for names in (source_bodies, source_joints, source_geoms, source_actuators) for name in names):
        raise TwoFighterArenaError("every recovered robot object must be named")
    if tuple(_canonical_body_name(str(name)) for name in source_bodies) != plant_contract.BODY_NAMES:
        raise TwoFighterArenaError("recovered body order differs from the pinned G1 layout")
    return FighterNamespace(
        role=role,
        namespace=namespace,
        root_body_name=_namespace_name(namespace, xml_contract.root_body_xml_name),
        free_joint_name=_namespace_name(namespace, xml_contract.free_joint_name),
        anchor_body_name=_namespace_name(namespace, xml_contract.anchor_body_xml_name),
        body_names=tuple(_namespace_name(namespace, str(name)) for name in source_bodies),
        joint_names=tuple(_namespace_name(namespace, str(name)) for name in source_joints),
        actuator_names=tuple(
            _namespace_name(namespace, str(name)) for name in source_actuators
        ),
        geom_names=tuple(_namespace_name(namespace, str(name)) for name in source_geoms),
    )


def _spawn_anchor_references(
    arena_contract: plant_contract.ArenaContract,
) -> tuple[SpawnAnchorReference, SpawnAnchorReference]:
    source = arena_contract.spawn_points
    if source.get("runtime_observation") is not False:
        raise TwoFighterArenaError("spawn anchors must remain classified runtime_observation false")
    if source.get("identical_pose_across_level_containers") is not True:
        raise TwoFighterArenaError("spawn anchors differ across shipped level containers")
    native = source.get("native_application")
    if not isinstance(native, dict) or native.get("proven_for_pinned_client_build") is not True:
        raise TwoFighterArenaError("spawn native application path is not build-pinned")
    anchors = []
    for role, expected_slot in (("player", 0), ("opponent", 1)):
        value = source.get(role)
        if not isinstance(value, dict):
            raise TwoFighterArenaError(f"arena contract omits {role} spawn anchor")
        if value.get("slot") != expected_slot or value.get("fallback_robot_id") != "g1":
            raise TwoFighterArenaError(f"{role} spawn slot or robot identity mismatch")
        position = _finite_numbers(value.get("mujoco_world_position_m"), 3, f"{role} spawn position")
        quaternion = _finite_numbers(
            value.get("mujoco_world_quaternion_wxyz"),
            4,
            f"{role} spawn quaternion",
        )
        if not math.isclose(
            math.sqrt(sum(component * component for component in quaternion)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TwoFighterArenaError(f"{role} spawn quaternion is not normalized")
        anchors.append(
            SpawnAnchorReference(
                role=role,
                slot=expected_slot,
                frame_body_name=SPAWN_FRAME_NAMES[role],
                position_m=position,
                quaternion_wxyz=quaternion,
                runtime_observation=False,
            )
        )
    return tuple(anchors)  # type: ignore[return-value]


def _verify_composed_names(
    root: ET.Element,
    source_names: set[str],
    fighters: Sequence[FighterNamespace],
) -> None:
    named = [(element.tag, element.get("name")) for element in root.iter() if element.get("name")]
    names = [str(name) for _tag, name in named]
    if len(names) != len(set(names)):
        raise TwoFighterArenaError("composed MJCF contains duplicate names")
    escaped = sorted(source_names.intersection(names))
    if escaped:
        raise TwoFighterArenaError(f"un-namespaced robot names escaped composition: {escaped}")
    joint_names = {str(joint.get("name")) for joint in root.findall(".//joint")}
    motors = root.findall("./actuator/motor")
    if len(motors) != 58:
        raise TwoFighterArenaError("composed MJCF must contain 58 motors")
    for motor in motors:
        target = motor.get("joint")
        if target not in joint_names:
            raise TwoFighterArenaError(
                f"composed motor {motor.get('name')!r} has unresolved joint {target!r}"
            )
    for fighter in fighters:
        prefix = f"{fighter.namespace}{NAMESPACE_SEPARATOR}"
        expected = {
            *fighter.body_names,
            fighter.free_joint_name,
            *fighter.joint_names,
            *fighter.actuator_names,
            *fighter.geom_names,
        }
        observed = {name for name in names if name.startswith(prefix)}
        if observed != expected:
            raise TwoFighterArenaError(f"{fighter.role} namespace object set mismatch")


def compose_two_fighter_mjcf(
    *,
    xml: Path | None = None,
    arena: Path | None = None,
    spawn_contract: Path | None = None,
    runtime_manifest: Path | None = None,
) -> TwoFighterArenaContract:
    """Compose two namespaced G1s at their exact pinned initial pelvis poses."""
    xml_path = _default_xml_path() if xml is None else Path(xml)
    arena_path = _default_arena_path() if arena is None else Path(arena)
    spawn_contract_path = (
        _default_spawn_contract_path()
        if spawn_contract is None
        else Path(spawn_contract)
    )
    manifest_path = (
        _default_runtime_manifest_path()
        if runtime_manifest is None
        else Path(runtime_manifest)
    )
    try:
        xml_contract = plant_contract.inspect_xml_contract(xml_path)
        arena_contract = plant_contract.load_arena_contract(arena_path)
    except plant_contract.CandidateError as exc:
        raise TwoFighterArenaError(str(exc)) from exc
    spawn_facts = _load_spawn_contract(spawn_contract_path)
    manifest = _load_runtime_manifest(manifest_path)
    if spawn_facts.file_sha256 != SPAWN_CONTRACT_SHA256:
        raise TwoFighterArenaError("G1 spawn transform contract file identity mismatch")
    if SPAWN_SOURCE_SHA256["arena_contract"] != arena_contract.source_sha256:
        raise TwoFighterArenaError("spawn and arena contract source identities disagree")
    if SPAWN_SOURCE_SHA256["recovered_mjcf"] != xml_contract.source_sha256:
        raise TwoFighterArenaError("spawn and recovered MJCF source identities disagree")
    if not math.isclose(
        xml_contract.source_timestep,
        arena_contract.source_timestep,
        rel_tol=0.0,
        abs_tol=1e-17,
    ):
        raise TwoFighterArenaError("arena and recovered XML source timesteps disagree")

    _source_root, source_body, source_motors = _source_robot_elements(
        xml_contract.source_bytes
    )
    source_names = {
        str(element.get("name"))
        for element in _robot_elements(source_body, source_motors)
        if element.get("name")
    }
    source_projection_sha256 = _robot_projection_sha256(
        source_body, source_motors, None
    )
    source_nonspawn_projection_sha256 = _robot_nonspawn_projection_sha256(
        source_body, source_motors, None
    )
    try:
        arena_xml = plant_contract.add_arena_geoms(
            xml_contract.source_bytes, arena_contract
        )
    except plant_contract.CandidateError as exc:
        raise TwoFighterArenaError(str(exc)) from exc
    root = ET.fromstring(arena_xml)
    root.set("model", MODEL_NAME)
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    if worldbody is None or actuator is None:
        raise TwoFighterArenaError("arena composition omitted worldbody or actuator")
    direct_geoms = [child for child in worldbody if child.tag == "geom"]
    direct_bodies = [child for child in worldbody if child.tag == "body"]
    if len(direct_geoms) != 17 or len(direct_bodies) != 1:
        raise TwoFighterArenaError("single-plant arena template layout mismatch")
    for element, expected in zip(direct_geoms, arena_contract.geoms):
        if element.get("name") != expected["name"]:
            raise TwoFighterArenaError("arena geom order changed during composition")
    worldbody.remove(direct_bodies[0])
    for motor in tuple(list(actuator)):
        actuator.remove(motor)

    spawn_anchors = _spawn_anchor_references(arena_contract)
    for anchor, rebase in zip(spawn_anchors, spawn_facts.rebases):
        if (
            anchor.role != rebase.role
            or anchor.slot != rebase.slot
            or anchor.position_m != rebase.spawn_root_position_m
            or anchor.quaternion_wxyz != rebase.spawn_root_quaternion_wxyz
        ):
            raise TwoFighterArenaError(
                f"{anchor.role} arena and spawn-root source poses disagree"
            )
    for anchor in spawn_anchors:
        worldbody.append(
            ET.Element(
                "body",
                {
                    "name": anchor.frame_body_name,
                    "pos": _mjcf_numbers(anchor.position_m),
                    "quat": _mjcf_numbers(anchor.quaternion_wxyz),
                },
            )
        )

    fighters: list[FighterNamespace] = []
    for role in ROLES:
        namespace = role
        rebase = next(item for item in spawn_facts.rebases if item.role == role)
        cloned_body = copy.deepcopy(source_body)
        cloned_motors = tuple(copy.deepcopy(motor) for motor in source_motors)
        _namespace_robot(cloned_body, cloned_motors, namespace)
        if (
            _robot_projection_sha256(cloned_body, cloned_motors, namespace)
            != source_projection_sha256
        ):
            raise TwoFighterArenaError(
                f"{role} robot copy changed outside names and name references"
            )
        _apply_spawn_rebase(cloned_body, rebase)
        if (
            _robot_nonspawn_projection_sha256(
                cloned_body, cloned_motors, namespace
            )
            != source_nonspawn_projection_sha256
        ):
            raise TwoFighterArenaError(
                f"{role} robot copy changed outside names, references, and root pose"
            )
        fighter = _fighter_namespace(
            role, namespace, source_body, source_motors, xml_contract
        )
        fighters.append(fighter)
        worldbody.append(cloned_body)
        actuator.extend(cloned_motors)

    _verify_composed_names(root, source_names, fighters)
    xml_text = ET.tostring(root, encoding="unicode")
    root_position = _finite_numbers(
        [float(item) for item in str(source_body.get("pos", "")).split()],
        3,
        "serialized root position",
    )
    root_quaternion = _finite_numbers(
        [float(item) for item in str(source_body.get("quat", "")).split()],
        4,
        "serialized root quaternion",
    )
    if tuple(arena_contract.unknowns).count(STALE_ARENA_SPAWN_UNKNOWN) != 1:
        raise TwoFighterArenaError("arena spawn-rebase unknown identity mismatch")
    retained_arena_unknowns = tuple(
        item
        for item in arena_contract.unknowns
        if item != STALE_ARENA_SPAWN_UNKNOWN
    )
    unknowns = retained_arena_unknowns + (
        "the prefab clone root transform lacks an independent passive runtime observation",
        "server build equality is unobserved",
        "post-spawn physics, resets, networking, and controller mutations are outside this reference",
    )
    return TwoFighterArenaContract(
        schema=SCHEMA,
        classification=CLASSIFICATION,
        build_fingerprint=plant_contract.BUILD_FINGERPRINT,
        control_equivalent=False,
        xml_text=xml_text,
        xml_sha256=_sha256_bytes(xml_text.encode("utf-8")),
        recovered_xml_sha256=xml_contract.source_sha256,
        arena_contract_sha256=arena_contract.source_sha256,
        arena_geometry_sha256=arena_contract.derived_geometry_sha256,
        spawn_contract_sha256=spawn_facts.file_sha256,
        spawn_evidence_projection_sha256=spawn_facts.evidence_projection_sha256,
        runtime_manifest_file_sha256=manifest.file_sha256,
        runtime_manifest_canonical_sha256=manifest.canonical_sha256,
        runtime_asset_roles=manifest.roles,
        robot_template_canonical_sha256=source_projection_sha256,
        serialized_root_position_m=root_position,
        serialized_root_quaternion_wxyz=root_quaternion,
        fighters=tuple(fighters),  # type: ignore[arg-type]
        spawn_anchors=spawn_anchors,
        spawn_rebases=spawn_facts.rebases,
        spawn_rebase_applied=True,
        spawn_rebase_status=SPAWN_REBASE_STATUS,
        excluded_behaviors=EXCLUDED_BEHAVIORS,
        unknowns=unknowns,
    )


def assert_spawn_rebase_applied(
    value: TwoFighterArenaContract | TwoFighterArenaReference,
) -> None:
    """Assert that both exact pinned pelvis spawn poses were applied."""
    contract = value.contract if isinstance(value, TwoFighterArenaReference) else value
    if not contract.spawn_rebase_applied:
        raise TwoFighterArenaError("two-fighter spawn rebase is not applied")
    if contract.spawn_rebase_status != SPAWN_REBASE_STATUS:
        raise TwoFighterArenaError("two-fighter spawn rebase status mismatch")
    if contract.spawn_contract_sha256 != SPAWN_CONTRACT_SHA256:
        raise TwoFighterArenaError("two-fighter spawn contract identity mismatch")
    if tuple(rebase.role for rebase in contract.spawn_rebases) != ROLES:
        raise TwoFighterArenaError("two-fighter spawn rebase role order mismatch")
    for role in ROLES:
        if contract.spawn_rebase(role).free_joint_qpos_prefix != EXPECTED_SPAWN_QPOS[role]:
            raise TwoFighterArenaError(f"{role} spawn rebase qpos mismatch")


def assert_spawn_rebase_available(
    value: TwoFighterArenaContract | TwoFighterArenaReference,
) -> None:
    """Compatibility assertion for callers checking the former spawn gate."""
    assert_spawn_rebase_applied(value)


def _validate_model_dimensions(model: Any) -> None:
    observed = {name: int(getattr(model, name)) for name in EXPECTED_MODEL_DIMENSIONS}
    if observed != EXPECTED_MODEL_DIMENSIONS:
        raise TwoFighterArenaError(
            f"two-fighter MuJoCo dimensions mismatch: expected "
            f"{EXPECTED_MODEL_DIMENSIONS}, got {observed}"
        )


def _validate_compiled_arena(
    mujoco: Any,
    model: Any,
    arena_contract: plant_contract.ArenaContract,
) -> None:
    expected_box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    observed_ids: list[int] = []
    for geom in arena_contract.geoms:
        name = str(geom["name"])
        try:
            geom_id = int(model.geom(name).id)
        except Exception as exc:
            raise TwoFighterArenaError(
                f"compiled arena geom mapping failed for {name!r}: {exc}"
            ) from exc
        observed_ids.append(geom_id)
        contact = geom["contact"]
        scalar_checks = (
            ("type", int(model.geom_type[geom_id]), expected_box_type),
            ("body_id", int(model.geom_bodyid[geom_id]), 0),
            ("priority", int(model.geom_priority[geom_id]), int(contact["priority"])),
            ("contype", int(model.geom_contype[geom_id]), int(contact["contype"])),
            (
                "conaffinity",
                int(model.geom_conaffinity[geom_id]),
                int(contact["conaffinity"]),
            ),
            ("group", int(model.geom_group[geom_id]), int(contact["group"])),
            ("condim", int(model.geom_condim[geom_id]), int(contact["condim"])),
        )
        for field, observed, expected in scalar_checks:
            if observed != expected:
                raise TwoFighterArenaError(
                    f"compiled arena geom {name!r} {field} mismatch"
                )
        vector_checks = (
            ("position", model.geom_pos[geom_id], geom["position_m"]),
            ("quaternion", model.geom_quat[geom_id], geom["quaternion_wxyz"]),
            ("half_extents", model.geom_size[geom_id], geom["half_extents_m"]),
            ("friction", model.geom_friction[geom_id], contact["friction"]),
            ("solref", model.geom_solref[geom_id], contact["solref"]),
            ("solimp", model.geom_solimp[geom_id], contact["solimp"]),
        )
        for field, observed, expected in vector_checks:
            if not np.allclose(
                np.asarray(observed),
                np.asarray(expected, dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise TwoFighterArenaError(
                    f"compiled arena geom {name!r} {field} mismatch"
                )
        float_checks = (
            ("solmix", float(model.geom_solmix[geom_id]), float(contact["solmix"])),
            ("margin", float(model.geom_margin[geom_id]), float(contact["margin_m"])),
            ("gap", float(model.geom_gap[geom_id]), float(contact["gap_m"])),
        )
        for field, observed, expected in float_checks:
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise TwoFighterArenaError(
                    f"compiled arena geom {name!r} {field} mismatch"
                )
    if len(set(observed_ids)) != 17:
        raise TwoFighterArenaError("compiled arena geom mapping is not one-to-one")


def _validate_spawn_frames(model: Any, contract: TwoFighterArenaContract) -> None:
    for anchor in contract.spawn_anchors:
        try:
            body_id = int(model.body(anchor.frame_body_name).id)
        except Exception as exc:
            raise TwoFighterArenaError(
                f"compiled spawn frame mapping failed for {anchor.role}: {exc}"
            ) from exc
        if int(model.body_parentid[body_id]) != 0:
            raise TwoFighterArenaError(f"{anchor.role} spawn frame is not world-relative")
        if int(model.body_jntnum[body_id]) != 0 or int(model.body_geomnum[body_id]) != 0:
            raise TwoFighterArenaError(f"{anchor.role} spawn frame became physical")
        if not np.allclose(
            np.asarray(model.body_pos[body_id]),
            np.asarray(anchor.position_m),
            rtol=0.0,
            atol=1e-12,
        ):
            raise TwoFighterArenaError(f"{anchor.role} spawn frame position mismatch")
        if not np.allclose(
            np.asarray(model.body_quat[body_id]),
            np.asarray(anchor.quaternion_wxyz),
            rtol=0.0,
            atol=1e-12,
        ):
            raise TwoFighterArenaError(f"{anchor.role} spawn frame quaternion mismatch")


def _build_runtime_map(
    model: Any,
    xml_contract: plant_contract.XmlContract,
    fighter: FighterNamespace,
) -> plant_contract.RuntimeMap:
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    actuator_ids: list[int] = []
    ctrl_min: list[float] = []
    ctrl_max: list[float] = []
    for spec in xml_contract.joints:
        joint_name = _namespace_name(fighter.namespace, spec.xml_joint_name)
        actuator_name = _namespace_name(fighter.namespace, spec.actuator_name)
        try:
            joint_id = int(model.joint(joint_name).id)
            actuator_id = int(model.actuator(actuator_name).id)
        except Exception as exc:
            raise TwoFighterArenaError(
                f"{fighter.role} named mapping failed for {spec.canonical_name}: {exc}"
            ) from exc
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise TwoFighterArenaError(
                f"{fighter.role} actuator transmission mismatch for {spec.canonical_name}"
            )
        if not math.isclose(
            float(model.actuator_gear[actuator_id, 0]),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise TwoFighterArenaError(
                f"{fighter.role} actuator gear mismatch for {spec.canonical_name}"
            )
        if not np.allclose(
            np.asarray(model.actuator_ctrlrange[actuator_id]),
            np.asarray([spec.ctrl_min, spec.ctrl_max]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise TwoFighterArenaError(
                f"{fighter.role} actuator ctrlrange mismatch for {spec.canonical_name}"
            )
        if bool(model.actuator_ctrllimited[actuator_id]) or bool(
            model.actuator_forcelimited[actuator_id]
        ):
            raise TwoFighterArenaError(
                f"{fighter.role} actuator limit flag mismatch for {spec.canonical_name}"
            )
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
        ctrl_min.append(spec.ctrl_min)
        ctrl_max.append(spec.ctrl_max)
    if len(set(joint_ids)) != 29 or len(set(actuator_ids)) != 29:
        raise TwoFighterArenaError(f"{fighter.role} joint or actuator map is not one-to-one")
    if len(set(qpos_addresses)) != 29 or len(set(qvel_addresses)) != 29:
        raise TwoFighterArenaError(f"{fighter.role} state address map is not one-to-one")
    try:
        free_joint_id = int(model.joint(fighter.free_joint_name).id)
        root_body_id = int(model.body(fighter.root_body_name).id)
        anchor_body_id = int(model.body(fighter.anchor_body_name).id)
        body_ids = np.asarray(
            [int(model.body(name).id) for name in fighter.body_names],
            dtype=np.int32,
        )
    except Exception as exc:
        raise TwoFighterArenaError(f"{fighter.role} body or free-joint mapping failed: {exc}") from exc
    if len(set(body_ids.tolist())) != 30:
        raise TwoFighterArenaError(f"{fighter.role} body map is not one-to-one")
    root_qpos_address = int(model.jnt_qposadr[free_joint_id])
    root_qvel_address = int(model.jnt_dofadr[free_joint_id])
    return plant_contract.RuntimeMap(
        joint_ids=np.asarray(joint_ids, dtype=np.int32),
        qpos_addresses=np.asarray(qpos_addresses, dtype=np.int32),
        qvel_addresses=np.asarray(qvel_addresses, dtype=np.int32),
        actuator_ids=np.asarray(actuator_ids, dtype=np.int32),
        ctrl_min=np.asarray(ctrl_min, dtype=np.float64),
        ctrl_max=np.asarray(ctrl_max, dtype=np.float64),
        root_qpos_address=root_qpos_address,
        root_qvel_address=root_qvel_address,
        root_body_id=root_body_id,
        anchor_body_id=anchor_body_id,
        body_ids=body_ids,
    )


def _validate_disjoint_runtime_maps(
    maps: Mapping[str, plant_contract.RuntimeMap],
) -> None:
    if set(maps) != set(ROLES):
        raise TwoFighterArenaError("two-fighter runtime map roles mismatch")
    player = maps["player"]
    opponent = maps["opponent"]
    pairs = (
        ("joint ids", player.joint_ids, opponent.joint_ids),
        ("qpos addresses", player.qpos_addresses, opponent.qpos_addresses),
        ("qvel addresses", player.qvel_addresses, opponent.qvel_addresses),
        ("actuator ids", player.actuator_ids, opponent.actuator_ids),
        ("body ids", player.body_ids, opponent.body_ids),
    )
    for label, left, right in pairs:
        if set(np.asarray(left).tolist()).intersection(np.asarray(right).tolist()):
            raise TwoFighterArenaError(f"fighter {label} overlap")
    player_qpos = {
        *range(player.root_qpos_address, player.root_qpos_address + 7),
        *player.qpos_addresses.tolist(),
    }
    opponent_qpos = {
        *range(opponent.root_qpos_address, opponent.root_qpos_address + 7),
        *opponent.qpos_addresses.tolist(),
    }
    player_qvel = {
        *range(player.root_qvel_address, player.root_qvel_address + 6),
        *player.qvel_addresses.tolist(),
    }
    opponent_qvel = {
        *range(opponent.root_qvel_address, opponent.root_qvel_address + 6),
        *opponent.qvel_addresses.tolist(),
    }
    if len(player_qpos) != 36 or len(opponent_qpos) != 36 or player_qpos & opponent_qpos:
        raise TwoFighterArenaError("fighter complete qpos state maps overlap or are incomplete")
    if len(player_qvel) != 35 or len(opponent_qvel) != 35 or player_qvel & opponent_qvel:
        raise TwoFighterArenaError("fighter complete qvel state maps overlap or are incomplete")


def _validate_spawn_rebased_root_poses(
    model: Any,
    contract: TwoFighterArenaContract,
    maps: Mapping[str, plant_contract.RuntimeMap],
) -> None:
    for role in ROLES:
        rebase = contract.spawn_rebase(role)
        expected = np.asarray(rebase.free_joint_qpos_prefix, dtype=np.float64)
        start = maps[role].root_qpos_address
        observed = np.asarray(model.qpos0[start : start + 7], dtype=np.float64)
        if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
            raise TwoFighterArenaError(f"{role} initial free-joint pelvis pose mismatch")
        body_id = maps[role].root_body_id
        if not np.allclose(
            np.asarray(model.body_pos[body_id]),
            np.asarray(rebase.pelvis_position_m),
            rtol=0.0,
            atol=1e-12,
        ):
            raise TwoFighterArenaError(f"{role} root body spawn position mismatch")
        if not np.allclose(
            np.asarray(model.body_quat[body_id]),
            np.asarray(rebase.pelvis_quaternion_wxyz),
            rtol=0.0,
            atol=1e-12,
        ):
            raise TwoFighterArenaError(f"{role} root body spawn quaternion mismatch")


def create_two_fighter_arena_reference(
    *,
    xml: Path | None = None,
    arena: Path | None = None,
    spawn_contract: Path | None = None,
    runtime_manifest: Path | None = None,
) -> TwoFighterArenaReference:
    """Compile the initial-spawn reference and build one map per robot."""
    contract = compose_two_fighter_mjcf(
        xml=xml,
        arena=arena,
        spawn_contract=spawn_contract,
        runtime_manifest=runtime_manifest,
    )
    try:
        import mujoco
    except ImportError as exc:
        raise TwoFighterArenaError("mujoco is required to compile the two-fighter arena") from exc
    try:
        model = mujoco.MjModel.from_xml_string(contract.xml_text)
    except Exception as exc:
        raise TwoFighterArenaError(f"two-fighter MuJoCo compilation failed: {exc}") from exc
    _validate_model_dimensions(model)
    if not math.isclose(
        float(model.opt.timestep),
        2_822_399 / 141_120_000,
        rel_tol=0.0,
        abs_tol=1e-17,
    ):
        raise TwoFighterArenaError("compiled model did not preserve the measured timestep")
    try:
        arena_contract = plant_contract.load_arena_contract(
            _default_arena_path() if arena is None else Path(arena)
        )
        xml_contract = plant_contract.inspect_xml_contract(
            _default_xml_path() if xml is None else Path(xml)
        )
    except plant_contract.CandidateError as exc:
        raise TwoFighterArenaError(str(exc)) from exc
    _validate_compiled_arena(mujoco, model, arena_contract)
    _validate_spawn_frames(model, contract)
    runtime_maps = {
        fighter.role: _build_runtime_map(model, xml_contract, fighter)
        for fighter in contract.fighters
    }
    _validate_disjoint_runtime_maps(runtime_maps)
    _validate_spawn_rebased_root_poses(model, contract, runtime_maps)
    assert_spawn_rebase_applied(contract)
    return TwoFighterArenaReference(
        contract=contract,
        mujoco=mujoco,
        model=model,
        mujoco_version=str(getattr(mujoco, "__version__", "unknown")),
        runtime_maps=MappingProxyType(runtime_maps),
    )
