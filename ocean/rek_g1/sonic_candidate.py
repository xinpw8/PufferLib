"""Evidence-gated G1 motion-tracker candidate using ONNX Runtime and MuJoCo.

This tool evaluates an official NVlabs ProtoMotions G1 tracker against the
build-pinned REK motion references and recovered G1 plant.  It is a diagnostic
candidate.  Its output never constitutes a REK controller-parity claim.

Model files and extracted game payloads remain external.  The tool reads them
from caller-supplied paths, verifies pinned hashes and schemas, and writes only
optional metrics or trace files selected by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RUN_SCHEMA = "rek.g1_sonic_candidate.run.v1"
TRACE_SCHEMA = "rek.g1_sonic_candidate.trace.v1"
SEGMENT_SCHEMA = "rek.g1_sonic_candidate.segment.v1"
CLASSIFICATION = "diagnostic_candidate"
BUILD_FINGERPRINT = (
    "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
)
PROTOMOTIONS_COMMIT = "607ca7a0bb92e261120bcab8d9f97f28b3130ffc"
EXPECTED_ONNX_SHA256 = (
    "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c"
)
EXPECTED_YAML_SHA256 = (
    "9b7896f3355a9d9d5e7d3139b83924eeb2e45c62c30bfda44afe996cfc6cf01c"
)
EXPECTED_XML_SHA256 = (
    "811fdc1e5bee74026b780974207cbcd628cdd83a249d3f76b75a668d71aad835"
)
EXPECTED_MANIFEST_CANONICAL_SHA256 = (
    "09973b2793f5b3e546a4f32cbf6128a13100c2332e3ed18c7e3eb46398618367"
)
EXPECTED_ARENA_SHA256 = (
    "67128d45f8b5995d57b5ca925a2db7b2d613ede15bfa19f8df46c4e01c60e3e8"
)
EXPECTED_ARENA_GEOMETRY_SHA256 = (
    "39aa951681f929f2580e723ecf133cabbfe310aa6a344627c639f9db8fc03e5b"
)
MANIFEST_SCHEMA = "rek.g1_runtime_assets.manifest.v1"
INVENTORY_SCHEMA = "rek.g1_runtime_assets.inventory.v1"
ARENA_SCHEMA = "rek.g1_arena_physics_contract.v1"
FUTURE_STEPS = (1, 2, 4, 8)
CONTROL_DT = 0.02
PHYSICS_DT = 0.001
DECIMATION = 20

UPSTREAM_DEPLOYMENT_EVIDENCE = {
    "repository": "https://github.com/NVlabs/ProtoMotions",
    "commit": PROTOMOTIONS_COMMIT,
    "references": [
        {
            "path": "deployment/test_tracker_mujoco.py",
            "lines": "341-379",
            "url": "https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/test_tracker_mujoco.py#L341-L379",
            "claim": "MuJoCo state slicing, wxyz-to-xyzw conversion, and direct local free-joint angular velocity",
        },
        {
            "path": "deployment/tracker_inputs.py",
            "lines": "156-200",
            "url": "https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/tracker_inputs.py#L156-L200",
            "claim": "semantic ONNX input assembly and zero initial action history",
        },
        {
            "path": "deployment/test_tracker_mujoco.py",
            "lines": "720-775",
            "url": "https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/test_tracker_mujoco.py#L720-L775",
            "claim": "runtime feed construction and processed PD target feedback",
        },
        {
            "path": "deployment/state_utils.py",
            "lines": "296-350",
            "url": "https://github.com/NVlabs/ProtoMotions/blob/607ca7a0bb92e261120bcab8d9f97f28b3130ffc/deployment/state_utils.py#L296-L350",
            "claim": "yaw-only reference heading alignment",
        },
    ],
}

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

INPUT_CONTRACT = (
    ("current_anchor_rot", ("batch_size", 4), "current.anchor_rot"),
    ("current_dof_pos", ("batch_size", 29), "current.dof_pos"),
    ("current_dof_vel", ("batch_size", 29), "current.dof_vel"),
    (
        "current_root_local_ang_vel",
        ("batch_size", 3),
        "current.root_local_ang_vel",
    ),
    (
        "historical_processed_actions",
        ("batch_size", 1, 29),
        "historical.processed_actions",
    ),
    (
        "mimic_future_anchor_rot",
        ("batch_size", 4, 4),
        "mimic.future_anchor_rot",
    ),
    (
        "mimic_future_dof_pos",
        ("batch_size", 4, 29),
        "mimic.future_dof_pos",
    ),
    (
        "mimic_future_dof_vel",
        ("batch_size", 4, 29),
        "mimic.future_dof_vel",
    ),
)

OUTPUT_CONTRACT = (
    ("actions", ("batch_size", 29)),
    ("joint_pos_targets", ("batch_size", 29)),
    ("stiffness_targets", ("batch_size", 29)),
    ("damping_targets", ("batch_size", 29)),
)

LIMITS = (
    "The official NVlabs tracker is not the recovered REK Sonic encoder and decoder.",
    "The recovered XML report marks the plant control_equivalent false.",
    "The arena contract covers static collision geometry and physics settings, not game logic or controller behavior.",
    "The diagnostic-floor mode is synthetic and is not a recovered REK arena surface.",
    "Motion joint velocities are REK's forward finite difference because the pinned NPZ files do not contain velocities.",
    "Reference torso rotations are recovered-plant FK results from NPZ pelvis rotation and joint position.",
    "Reference torso FK is an explicit inference because the pinned NPZ files omit per-body rotations used by the official deployment interface.",
    "Recovered direct-motor and passive-joint semantics differ from the official deployment runner's implicit-PD plant mutation, so controller trajectory compatibility is unverified.",
    "Manual torque clipping to declared actuator ctrlrange is a candidate safety assumption because the recovered ctrllimited and forcelimited flags are false.",
    "The arena spawn anchors do not establish a replacement free-joint pelvis height, so this candidate does not apply a REK spawn rebase.",
    "No authoritative REK target, torque, or controller-state trace is available for parity comparison.",
)

_JOINT_NAME_RE = re.compile(r"joint__(.+)_([0-9]+)\Z")
_BODY_NAME_RE = re.compile(r"(.+)_([0-9]+)\Z")
_ACTUATOR_NAME_RE = re.compile(r"(.+)_([0-9]+)\Z")


class CandidateError(RuntimeError):
    """A fail-closed artifact, schema, mapping, or runtime error."""


@dataclass(frozen=True)
class PolicyContract:
    joint_names: tuple[str, ...]
    stiffness: np.ndarray
    damping: np.ndarray
    anchor_body_name: str
    root_body_name: str
    control_dt: float
    physics_dt: float
    decimation: int
    metadata: Mapping[str, Any]
    yaml_size: int
    yaml_sha256: str


@dataclass(frozen=True)
class XmlJoint:
    canonical_name: str
    xml_joint_name: str
    actuator_name: str
    ctrl_min: float
    ctrl_max: float


@dataclass(frozen=True)
class XmlContract:
    source_bytes: bytes
    source_size: int
    source_sha256: str
    source_timestep: float
    free_joint_name: str
    root_body_xml_name: str
    anchor_body_xml_name: str
    joints: tuple[XmlJoint, ...]
    passive_damping_range: tuple[float, float]
    frictionloss_range: tuple[float, float]
    source_has_floor: bool


@dataclass(frozen=True)
class MotionData:
    role: str
    filename: str
    size: int
    sha256: str
    fps: float
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_pos: np.ndarray
    root_rot_xyzw: np.ndarray
    manifest_sha256: str
    inventory_sha256: str


@dataclass(frozen=True)
class ArenaContract:
    source_size: int
    source_sha256: str
    source_timestep: float
    derived_geometry_sha256: str
    probe_geometry_signature_sha256: str
    geoms: tuple[Mapping[str, Any], ...]
    floor_top_z_m: float
    composed_ngeom: int
    spawn_points: Mapping[str, Any]
    unknowns: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeMap:
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    qvel_addresses: np.ndarray
    actuator_ids: np.ndarray
    ctrl_min: np.ndarray
    ctrl_max: np.ndarray
    root_qpos_address: int
    root_qvel_address: int
    root_body_id: int
    anchor_body_id: int
    body_ids: np.ndarray


@dataclass(frozen=True)
class SegmentRequest:
    source: str
    motion_role: str
    duration_ticks: int | None
    semantic_command: Mapping[str, Any] | None
    source_sha256: str | None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def _regular_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(path))
    if path.is_symlink() or not path.is_file():
        raise CandidateError(f"{label} must be a regular, non-symlink file")
    return path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CandidateError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise CandidateError(f"{label} must be an array")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateError(f"{label} must be finite")
    return result


def _xml_float(value: str | None, label: str) -> float:
    try:
        result = float(value) if value is not None else math.nan
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"{label} must be a finite XML number") from exc
    if not math.isfinite(result):
        raise CandidateError(f"{label} must be a finite XML number")
    return result


def _canonical_joint_name(xml_name: str) -> str:
    match = _JOINT_NAME_RE.fullmatch(xml_name)
    if match is None:
        raise CandidateError(f"unrecognized recovered joint name {xml_name!r}")
    return match.group(1)


def _canonical_body_name(xml_name: str) -> str:
    match = _BODY_NAME_RE.fullmatch(xml_name)
    if match is None:
        raise CandidateError(f"unrecognized recovered body name {xml_name!r}")
    return match.group(1)


def _canonical_actuator_name(xml_name: str) -> str:
    match = _ACTUATOR_NAME_RE.fullmatch(xml_name)
    if match is None:
        raise CandidateError(f"unrecognized recovered actuator name {xml_name!r}")
    return match.group(1)


def _shape_tuple(value: Any, label: str) -> tuple[Any, ...]:
    return tuple(_sequence(value, label))


def validate_policy_metadata(metadata: Mapping[str, Any]) -> None:
    root = _mapping(metadata, "policy metadata")
    if root.get("type") != "unified_pipeline":
        raise CandidateError("policy metadata type must be unified_pipeline")
    if not math.isclose(_finite_float(root.get("dt"), "dt"), CONTROL_DT):
        raise CandidateError("policy metadata dt must be 0.02")

    joint_names = tuple(_sequence(root.get("joint_names"), "joint_names"))
    if joint_names != JOINT_NAMES:
        raise CandidateError("policy metadata joint order does not match the pinned G1 order")

    policy_inputs = _sequence(root.get("policy_inputs"), "policy_inputs")
    observed_inputs = []
    for index, raw in enumerate(policy_inputs):
        entry = _mapping(raw, f"policy_inputs[{index}]")
        observed_inputs.append(
            (
                entry.get("name"),
                _shape_tuple(entry.get("shape"), f"policy_inputs[{index}].shape"),
                entry.get("key"),
            )
        )
    expected_yaml_inputs = [
        (name, tuple(1 if dim == "batch_size" else dim for dim in shape), key)
        for name, shape, key in INPUT_CONTRACT
    ]
    if observed_inputs != expected_yaml_inputs:
        raise CandidateError("policy input schema mismatch")

    policy_outputs = _sequence(root.get("policy_outputs"), "policy_outputs")
    observed_outputs = []
    for index, raw in enumerate(policy_outputs):
        entry = _mapping(raw, f"policy_outputs[{index}]")
        observed_outputs.append(
            (
                entry.get("name"),
                _shape_tuple(entry.get("shape"), f"policy_outputs[{index}].shape"),
            )
        )
    expected_yaml_outputs = [
        (name, tuple(1 if dim == "batch_size" else dim for dim in shape))
        for name, shape in OUTPUT_CONTRACT
    ]
    if observed_outputs != expected_yaml_outputs:
        raise CandidateError("policy output schema mismatch")

    runtime = _mapping(root.get("_runtime"), "_runtime")
    if tuple(runtime.get("onnx_in_names", ())) != tuple(x[0] for x in INPUT_CONTRACT):
        raise CandidateError("_runtime.onnx_in_names mismatch")
    if tuple(runtime.get("onnx_out_names", ())) != tuple(x[0] for x in OUTPUT_CONTRACT):
        raise CandidateError("_runtime.onnx_out_names mismatch")
    expected_name_to_key = {name: key for name, _shape, key in INPUT_CONTRACT}
    if runtime.get("onnx_name_to_in_key") != expected_name_to_key:
        raise CandidateError("_runtime.onnx_name_to_in_key mismatch")

    robot = _mapping(root.get("robot"), "robot")
    if robot.get("num_dofs") != 29:
        raise CandidateError("robot.num_dofs must be 29")
    if tuple(robot.get("joint_names", ())) != JOINT_NAMES:
        raise CandidateError("robot.joint_names mismatch")
    if robot.get("anchor_body_name") != "torso_link":
        raise CandidateError("robot.anchor_body_name must be torso_link")
    if robot.get("root_body_name") != "pelvis":
        raise CandidateError("robot.root_body_name must be pelvis")

    timing = _mapping(root.get("timing"), "timing")
    if not math.isclose(_finite_float(timing.get("control_dt"), "timing.control_dt"), CONTROL_DT):
        raise CandidateError("timing.control_dt must be 0.02")
    if not math.isclose(_finite_float(timing.get("physics_dt"), "timing.physics_dt"), PHYSICS_DT):
        raise CandidateError("timing.physics_dt must be 0.001")
    if timing.get("decimation") != DECIMATION:
        raise CandidateError("timing.decimation must be 20")
    if not math.isclose(CONTROL_DT, PHYSICS_DT * DECIMATION, abs_tol=1e-12):
        raise CandidateError("policy and physics timing are inconsistent")

    motion = _mapping(root.get("motion"), "motion")
    if tuple(motion.get("future_step_indices", ())) != FUTURE_STEPS:
        raise CandidateError("motion.future_step_indices must be [1, 2, 4, 8]")
    expected_future_dt = tuple(step * CONTROL_DT for step in FUTURE_STEPS)
    observed_future_dt = tuple(motion.get("future_dt_seconds", ()))
    if len(observed_future_dt) != len(expected_future_dt) or not np.allclose(
        observed_future_dt, expected_future_dt, rtol=0.0, atol=1e-12
    ):
        raise CandidateError("motion.future_dt_seconds mismatch")

    model_metadata = _mapping(root.get("metadata"), "metadata")
    if model_metadata.get("control_type") != "BUILT_IN_PD":
        raise CandidateError("metadata.control_type must be BUILT_IN_PD")

    control = _mapping(root.get("control"), "control")
    for label in ("stiffness", "damping"):
        values = np.asarray(control.get(label), dtype=np.float64)
        if values.shape != (29,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise CandidateError(f"control.{label} must contain 29 positive finite values")
    if control.get("pd_target_max_accel") is not None:
        raise CandidateError("the pinned policy must not request a PD acceleration clamp")
    if not math.isclose(
        _finite_float(control.get("action_ema_alpha"), "control.action_ema_alpha"),
        1.0,
    ):
        raise CandidateError("the pinned policy action_ema_alpha must be 1.0")


def load_policy_contract(path: Path) -> PolicyContract:
    path = _regular_file(path, "policy YAML")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_YAML_SHA256:
        raise CandidateError(
            f"policy YAML SHA-256 mismatch: expected {EXPECTED_YAML_SHA256}, got {digest}"
        )
    try:
        import yaml
    except ImportError as exc:
        raise CandidateError("PyYAML is required") from exc
    try:
        metadata = yaml.safe_load(raw)
    except Exception as exc:
        raise CandidateError(f"policy YAML parse failed: {exc}") from exc
    validate_policy_metadata(metadata)
    control = metadata["control"]
    timing = metadata["timing"]
    robot = metadata["robot"]
    return PolicyContract(
        joint_names=JOINT_NAMES,
        stiffness=np.asarray(control["stiffness"], dtype=np.float32),
        damping=np.asarray(control["damping"], dtype=np.float32),
        anchor_body_name=str(robot["anchor_body_name"]),
        root_body_name=str(robot["root_body_name"]),
        control_dt=float(timing["control_dt"]),
        physics_dt=float(timing["physics_dt"]),
        decimation=int(timing["decimation"]),
        metadata=metadata,
        yaml_size=len(raw),
        yaml_sha256=digest,
    )


def inspect_xml_contract(path: Path, joint_names: Sequence[str] = JOINT_NAMES) -> XmlContract:
    path = _regular_file(path, "recovered XML")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_XML_SHA256:
        raise CandidateError(
            f"recovered XML SHA-256 mismatch: expected {EXPECTED_XML_SHA256}, got {digest}"
        )
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise CandidateError(f"recovered XML parse failed: {exc}") from exc
    if root.tag != "mujoco":
        raise CandidateError("recovered XML root must be mujoco")
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle", "degree") != "degree":
        raise CandidateError("recovered XML must use degree-authored joint ranges")
    option = root.find("option")
    if option is None:
        raise CandidateError("recovered XML has no option element")
    source_timestep = _xml_float(option.get("timestep"), "XML option timestep")

    free_joints = root.findall(".//freejoint")
    if len(free_joints) != 1 or not free_joints[0].get("name"):
        raise CandidateError("recovered XML must contain exactly one named free joint")
    free_joint_name = str(free_joints[0].get("name"))

    body_by_canonical: dict[str, str] = {}
    free_joint_parent = None
    for body in root.findall(".//body"):
        xml_name = body.get("name")
        if not xml_name:
            raise CandidateError("every recovered body must be named")
        canonical = _canonical_body_name(xml_name)
        if canonical in body_by_canonical:
            raise CandidateError(f"duplicate canonical body name {canonical!r}")
        body_by_canonical[canonical] = xml_name
        if any(child is free_joints[0] for child in list(body)):
            free_joint_parent = canonical
    if free_joint_parent != "pelvis":
        raise CandidateError("the recovered free joint must belong to pelvis")
    for required_body in ("pelvis", "torso_link"):
        if required_body not in body_by_canonical:
            raise CandidateError(f"recovered XML has no {required_body} body")

    xml_joint_elements = root.findall(".//joint")
    if len(xml_joint_elements) != 29:
        raise CandidateError("recovered XML must contain exactly 29 hinge joints")
    joint_elements_by_name: dict[str, ET.Element] = {}
    canonical_by_xml: dict[str, str] = {}
    passive_damping: list[float] = []
    frictionloss: list[float] = []
    for element in xml_joint_elements:
        xml_name = element.get("name")
        if not xml_name or element.get("type", "hinge") != "hinge":
            raise CandidateError("all recovered controlled joints must be named hinges")
        canonical = _canonical_joint_name(xml_name)
        if canonical in canonical_by_xml.values():
            raise CandidateError(f"duplicate canonical joint name {canonical!r}")
        joint_elements_by_name[xml_name] = element
        canonical_by_xml[xml_name] = canonical
        passive_damping.append(_xml_float(element.get("damping", "0"), f"{xml_name}.damping"))
        frictionloss.append(
            _xml_float(element.get("frictionloss", "0"), f"{xml_name}.frictionloss")
        )

    actuator_root = root.find("actuator")
    if actuator_root is None or len(list(actuator_root)) != 29:
        raise CandidateError("recovered XML must contain exactly 29 actuators")
    joint_specs: dict[str, XmlJoint] = {}
    for actuator in list(actuator_root):
        if actuator.tag != "motor":
            raise CandidateError("the recovered controller requires direct motor actuators")
        actuator_name = actuator.get("name")
        xml_joint_name = actuator.get("joint")
        if not actuator_name or xml_joint_name not in joint_elements_by_name:
            raise CandidateError("each recovered motor must reference a named hinge")
        canonical_joint = canonical_by_xml[xml_joint_name]
        if _canonical_actuator_name(actuator_name) + "_joint" != canonical_joint:
            raise CandidateError(f"actuator and joint name mismatch for {actuator_name!r}")
        if actuator.get("ctrllimited") != "false" or actuator.get("forcelimited") != "false":
            raise CandidateError("recovered motor limit flags do not match the pinned semantics")
        gear = tuple(float(value) for value in actuator.get("gear", "").split())
        if gear != (1.0,):
            raise CandidateError("recovered direct motors must use scalar gear 1")
        ctrlrange = tuple(float(value) for value in actuator.get("ctrlrange", "").split())
        if len(ctrlrange) != 2 or not all(math.isfinite(value) for value in ctrlrange):
            raise CandidateError(f"invalid ctrlrange for {actuator_name!r}")
        if not ctrlrange[0] < 0.0 < ctrlrange[1]:
            raise CandidateError(f"ctrlrange must straddle zero for {actuator_name!r}")
        if canonical_joint in joint_specs:
            raise CandidateError(f"joint {canonical_joint!r} has multiple actuators")
        joint_specs[canonical_joint] = XmlJoint(
            canonical_name=canonical_joint,
            xml_joint_name=xml_joint_name,
            actuator_name=actuator_name,
            ctrl_min=ctrlrange[0],
            ctrl_max=ctrlrange[1],
        )
    expected_joint_names = tuple(joint_names)
    if set(joint_specs) != set(expected_joint_names) or len(joint_specs) != len(expected_joint_names):
        missing = sorted(set(expected_joint_names) - set(joint_specs))
        extra = sorted(set(joint_specs) - set(expected_joint_names))
        raise CandidateError(f"recovered XML joint set mismatch: missing={missing}, extra={extra}")

    source_has_floor = any(
        geom.get("type", "sphere") == "plane" for geom in root.findall(".//geom")
    )
    if source_has_floor:
        raise CandidateError("the pinned recovered XML unexpectedly contains a floor")
    return XmlContract(
        source_bytes=raw,
        source_size=len(raw),
        source_sha256=digest,
        source_timestep=source_timestep,
        free_joint_name=free_joint_name,
        root_body_xml_name=body_by_canonical["pelvis"],
        anchor_body_xml_name=body_by_canonical["torso_link"],
        joints=tuple(joint_specs[name] for name in expected_joint_names),
        passive_damping_range=(min(passive_damping), max(passive_damping)),
        frictionloss_range=(min(frictionloss), max(frictionloss)),
        source_has_floor=source_has_floor,
    )


def validate_onnx_session_schema(session: Any) -> None:
    observed_inputs = [
        (node.name, node.type, tuple(node.shape)) for node in session.get_inputs()
    ]
    expected_inputs = [(name, "tensor(float)", shape) for name, shape, _key in INPUT_CONTRACT]
    if observed_inputs != expected_inputs:
        raise CandidateError(
            f"ONNX input schema mismatch: expected {expected_inputs}, got {observed_inputs}"
        )
    observed_outputs = [
        (node.name, node.type, tuple(node.shape)) for node in session.get_outputs()
    ]
    expected_outputs = [(name, "tensor(float)", shape) for name, shape in OUTPUT_CONTRACT]
    if observed_outputs != expected_outputs:
        raise CandidateError(
            f"ONNX output schema mismatch: expected {expected_outputs}, got {observed_outputs}"
        )
    providers = list(session.get_providers())
    if providers != ["CPUExecutionProvider"]:
        raise CandidateError(f"ONNX session must use only CPUExecutionProvider, got {providers}")


def create_onnx_session(path: Path) -> tuple[Any, str, int, str]:
    path = _regular_file(path, "ONNX model")
    size = path.stat().st_size
    digest = sha256_file(path)
    if digest != EXPECTED_ONNX_SHA256:
        raise CandidateError(
            f"ONNX SHA-256 mismatch: expected {EXPECTED_ONNX_SHA256}, got {digest}"
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise CandidateError("onnxruntime is required") from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_mem_pattern = False
    if hasattr(options, "use_deterministic_compute"):
        options.use_deterministic_compute = True
    try:
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise CandidateError(f"ONNX session creation failed: {exc}") from exc
    validate_onnx_session_schema(session)
    return session, digest, size, str(getattr(ort, "__version__", "unknown"))


def _load_json_file(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    path = _regular_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{label} is invalid JSON: {exc}") from exc
    return _mapping(value, label), raw


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    items = _sequence(value, label)
    if len(items) != length:
        raise CandidateError(f"{label} must contain exactly {length} numbers")
    return tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(items))


def _arena_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateError(f"{label} must be an integer")
    return value


def _validate_arena_contract_schema(
    value: Mapping[str, Any], source_size: int, source_sha256: str
) -> ArenaContract:
    if value.get("schema") != ARENA_SCHEMA:
        raise CandidateError("arena contract schema mismatch")
    if value.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise CandidateError("arena contract build fingerprint mismatch")
    if value.get("control_equivalent") is not False:
        raise CandidateError("arena contract must remain classified control_equivalent false")

    sources = _mapping(value.get("sources"), "arena.sources")
    input_hashes = _mapping(sources.get("input_sha256"), "arena.sources.input_sha256")
    if input_hashes.get("g1_base_mjcf") != EXPECTED_XML_SHA256:
        raise CandidateError("arena contract recovered XML hash mismatch")

    coordinate = _mapping(value.get("coordinate_mapping"), "arena.coordinate_mapping")
    if coordinate.get("units") != "metres":
        raise CandidateError("arena coordinate units must be metres")
    if coordinate.get("unity_position_xyz_to_mujoco_xyz") != ["x", "z", "y"]:
        raise CandidateError("arena position coordinate mapping mismatch")
    if coordinate.get("unity_quaternion_wxyz_to_mujoco_wxyz") != ["-w", "x", "z", "y"]:
        raise CandidateError("arena quaternion coordinate mapping mismatch")

    timestep = _mapping(value.get("timestep"), "arena.timestep")
    exact = _mapping(timestep.get("seconds_exact"), "arena.timestep.seconds_exact")
    numerator = _arena_integer(exact.get("numerator"), "arena timestep numerator")
    denominator = _arena_integer(exact.get("denominator"), "arena timestep denominator")
    if (numerator, denominator, exact.get("fraction")) != (
        2822399,
        141120000,
        "2822399/141120000",
    ):
        raise CandidateError("arena exact timestep evidence mismatch")
    source_timestep = _finite_float(timestep.get("seconds_float"), "arena timestep")
    if not math.isclose(
        source_timestep, numerator / denominator, rel_tol=0.0, abs_tol=1e-17
    ):
        raise CandidateError("arena float and exact timesteps disagree")
    if timestep.get("g1_mjcf_option_matches") is not True:
        raise CandidateError("arena timestep is not bound to the recovered G1 MJCF")
    step_execution = _mapping(value.get("step_execution"), "arena.step_execution")
    if (
        step_execution.get("mujoco_steps_per_fixed_update") != 1
        or step_execution.get("application_level_substeps") != 1
        or step_execution.get("step_scene_calls_per_fixed_update") != 1
        or step_execution.get("mj_step1_and_mj_step2_are_phases_of_one_step") is not True
    ):
        raise CandidateError("arena step execution evidence mismatch")

    global_options = _mapping(value.get("global_options"), "arena.global_options")
    expected_option_values = {
        "impratio": 1.0,
        "iterations": 100,
        "noslip_iterations": 0,
        "ccd_iterations": 50,
        "density": 0.0,
        "viscosity": 0.0,
    }
    for key, expected in expected_option_values.items():
        observed = global_options.get(key)
        if isinstance(expected, int):
            if _arena_integer(observed, f"arena.global_options.{key}") != expected:
                raise CandidateError(f"arena global option {key} mismatch")
        elif _finite_float(observed, f"arena.global_options.{key}") != expected:
            raise CandidateError(f"arena global option {key} mismatch")
    if (
        _mapping(global_options.get("integrator"), "arena integrator").get("mujoco")
        != "implicitfast"
        or _mapping(global_options.get("cone"), "arena cone").get("mujoco")
        != "elliptic"
        or _mapping(global_options.get("solver"), "arena solver").get("mujoco")
        != "Newton"
    ):
        raise CandidateError("arena global solver options mismatch")

    plant = _mapping(value.get("g1_plant_boundary"), "arena.g1_plant_boundary")
    if (
        plant.get("separate_scene_geometry_was_omitted") is not True
        or plant.get("spawn_rebase_required") is not True
        or plant.get("arena_named_geom_count") != 0
        or plant.get("direct_world_geom_count") != 0
        or plant.get("composed_ngeom") != 54
    ):
        raise CandidateError("arena G1 plant boundary mismatch")
    composed_ngeom = 54

    arena = _mapping(value.get("arena"), "arena")
    if arena.get("geom_count") != 17 or arena.get("shape") != "box":
        raise CandidateError("arena contract must contain exactly 17 box geoms")
    if arena.get("role_counts") != {"floor": 1, "pillar": 8, "wall": 8}:
        raise CandidateError("arena geom role counts mismatch")
    if arena.get("geometry_identical_across_level_containers") is not True:
        raise CandidateError("arena geometry is not identical across level containers")
    derived_geometry_sha256 = arena.get("derived_geometry_sha256")
    if derived_geometry_sha256 != EXPECTED_ARENA_GEOMETRY_SHA256:
        raise CandidateError("arena derived geometry hash mismatch")
    probe_geometry_signature_sha256 = arena.get("probe_geometry_signature_sha256")
    if not isinstance(probe_geometry_signature_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", probe_geometry_signature_sha256
    ):
        raise CandidateError("arena probe geometry signature hash is invalid")
    shared_contact = _mapping(arena.get("shared_contact"), "arena.shared_contact")
    raw_geoms = _sequence(arena.get("geoms"), "arena.geoms")
    if len(raw_geoms) != 17:
        raise CandidateError("arena geoms array must contain exactly 17 entries")

    names: set[str] = set()
    roles: list[str] = []
    geoms: list[Mapping[str, Any]] = []
    floor_top_derived: float | None = None
    for index, raw_geom in enumerate(raw_geoms):
        geom = _mapping(raw_geom, f"arena.geoms[{index}]")
        name = geom.get("name")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) is None:
            raise CandidateError(f"arena.geoms[{index}].name is not a safe MJCF name")
        if name in names:
            raise CandidateError(f"duplicate arena geom name {name!r}")
        names.add(name)
        if geom.get("type") != "box":
            raise CandidateError(f"arena geom {name!r} must be a box")
        role = geom.get("role")
        if role not in ("floor", "pillar", "wall"):
            raise CandidateError(f"arena geom {name!r} has an invalid role")
        roles.append(str(role))
        position = _finite_vector(geom.get("position_m"), 3, f"{name}.position_m")
        quaternion = _finite_vector(
            geom.get("quaternion_wxyz"), 4, f"{name}.quaternion_wxyz"
        )
        if not math.isclose(
            math.sqrt(sum(component * component for component in quaternion)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CandidateError(f"arena geom {name!r} quaternion is not normalized")
        half_extents = _finite_vector(
            geom.get("half_extents_m"), 3, f"{name}.half_extents_m"
        )
        if any(value <= 0.0 for value in half_extents):
            raise CandidateError(f"arena geom {name!r} half extents must be positive")
        contact = _mapping(geom.get("contact"), f"{name}.contact")
        if contact != shared_contact:
            raise CandidateError(f"arena geom {name!r} contact settings mismatch")
        for key in ("priority", "contype", "conaffinity", "group", "condim"):
            _arena_integer(contact.get(key), f"{name}.contact.{key}")
        for key, length in (("solref", 2), ("solimp", 5), ("friction", 3), ("fluidcoef", 5)):
            _finite_vector(contact.get(key), length, f"{name}.contact.{key}")
        for key in ("solmix", "margin_m", "gap_m"):
            _finite_float(contact.get(key), f"{name}.contact.{key}")
        if contact.get("fluidshape") != "none":
            raise CandidateError(f"arena geom {name!r} fluidshape mismatch")
        if role == "floor":
            floor_top_derived = position[2] + half_extents[2]
        geoms.append(geom)
    if roles.count("floor") != 1 or roles.count("pillar") != 8 or roles.count("wall") != 8:
        raise CandidateError("arena geom roles do not match one floor, eight pillars, eight walls")
    floor_top_z_m = _finite_float(arena.get("floor_top_z_m"), "arena.floor_top_z_m")
    if floor_top_derived is None or not math.isclose(
        floor_top_z_m, floor_top_derived, rel_tol=0.0, abs_tol=1e-15
    ):
        raise CandidateError("arena reported and derived floor top disagree")

    masks = _mapping(value.get("contact_mask_check"), "arena.contact_mask_check")
    if masks.get("all_g1_pairs_can_contact_arena") is not True:
        raise CandidateError("arena contact masks do not cover all G1 geom pairs")
    if masks.get("arena") != {"contype": 2, "conaffinity": 1}:
        raise CandidateError("arena contact mask summary mismatch")

    spawn_points = _mapping(value.get("spawn_points"), "arena.spawn_points")
    for slot in ("player", "opponent"):
        spawn = _mapping(spawn_points.get(slot), f"arena.spawn_points.{slot}")
        _finite_vector(spawn.get("mujoco_world_position_m"), 3, f"{slot} spawn position")
        spawn_quaternion = _finite_vector(
            spawn.get("mujoco_world_quaternion_wxyz"), 4, f"{slot} spawn quaternion"
        )
        if not math.isclose(
            math.sqrt(sum(component * component for component in spawn_quaternion)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CandidateError(f"{slot} spawn quaternion is not normalized")
    unknown_items = _sequence(value.get("unknowns"), "arena.unknowns")
    if not unknown_items or any(not isinstance(item, str) or not item for item in unknown_items):
        raise CandidateError("arena unknowns must be nonempty strings")
    return ArenaContract(
        source_size=source_size,
        source_sha256=source_sha256,
        source_timestep=source_timestep,
        derived_geometry_sha256=str(derived_geometry_sha256),
        probe_geometry_signature_sha256=probe_geometry_signature_sha256,
        geoms=tuple(geoms),
        floor_top_z_m=floor_top_z_m,
        composed_ngeom=composed_ngeom,
        spawn_points=spawn_points,
        unknowns=tuple(unknown_items),
    )


def load_arena_contract(path: Path) -> ArenaContract:
    path = _regular_file(path, "arena contract")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_ARENA_SHA256:
        raise CandidateError(
            f"arena contract SHA-256 mismatch: expected {EXPECTED_ARENA_SHA256}, got {digest}"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"arena contract is invalid JSON: {exc}") from exc
    return _validate_arena_contract_schema(
        _mapping(value, "arena contract"), len(raw), digest
    )


def load_segment_request(path: Path) -> SegmentRequest:
    """Load one held-duration envelope compatible with semantic_action.h.

    The envelope binds a validated ``RekG1SemanticCommand`` representation to
    one diagnostic motion role.  The role binding is caller supplied because
    the recovered evidence does not establish REK's runtime clip selection.
    """
    value, raw = _load_json_file(path, "segment request")
    if value.get("schema") != SEGMENT_SCHEMA:
        raise CandidateError("segment request schema mismatch")
    if value.get("classification") != CLASSIFICATION:
        raise CandidateError("segment request must be classified diagnostic_candidate")
    motion_role = value.get("motion_role")
    if not isinstance(motion_role, str) or not motion_role:
        raise CandidateError("segment request motion_role must be a nonempty string")
    duration_ticks = value.get("duration_ticks")
    if (
        isinstance(duration_ticks, bool)
        or not isinstance(duration_ticks, int)
        or duration_ticks < 1
    ):
        raise CandidateError("segment request duration_ticks must be a positive integer")
    command = _mapping(value.get("semantic_command"), "segment semantic_command")
    kind = command.get("kind")
    held_code = command.get("held_code")
    command_duration = command.get("duration_ticks")
    kick_index = command.get("kick_registry_index")
    if kind not in (0, 1):
        raise CandidateError("semantic command kind must be 0 locomotion or 1 kick")
    if isinstance(held_code, bool) or not isinstance(held_code, int) or not 0 <= held_code < 27:
        raise CandidateError("semantic command held_code must be an integer in [0, 26]")
    if command_duration != duration_ticks:
        raise CandidateError("semantic command duration_ticks must match the segment duration")
    if isinstance(kick_index, bool) or not isinstance(kick_index, int) or not 0 <= kick_index <= 65535:
        raise CandidateError("semantic command kick_registry_index must be a uint16")
    if kind == 0 and kick_index != 65535:
        raise CandidateError("locomotion semantic command must use kick_registry_index 65535")
    if kind == 1:
        if kick_index == 65535:
            raise CandidateError("kick semantic command must name a kick registry index")
        forward_axis = held_code % 3
        strafe_axis = (held_code // 3) % 3
        if forward_axis != 0 or strafe_axis != 0:
            raise CandidateError("kick semantic command cannot contain translation")
    normalized_command = {
        "kind": kind,
        "held_code": held_code,
        "duration_ticks": duration_ticks,
        "kick_registry_index": kick_index,
    }
    return SegmentRequest(
        source="semantic_adapter_envelope",
        motion_role=motion_role,
        duration_ticks=duration_ticks,
        semantic_command=normalized_command,
        source_sha256=sha256_bytes(raw),
    )


def _validate_npz_members(path: Path, asset: Mapping[str, Any]) -> None:
    expected_members = {
        str(member["name"]): member for member in _sequence(asset.get("members"), "asset.members")
    }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != set(expected_members):
                raise CandidateError("motion NPZ member set mismatch")
            for info in infos:
                if info.is_dir() or Path(info.filename).name != info.filename:
                    raise CandidateError("motion NPZ must contain flat files only")
                expected = expected_members[info.filename]
                content = archive.read(info)
                if len(content) != expected.get("bytes"):
                    raise CandidateError(f"motion member size mismatch for {info.filename}")
                digest = sha256_bytes(content)
                if digest != expected.get("sha256"):
                    raise CandidateError(f"motion member SHA-256 mismatch for {info.filename}")
    except CandidateError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise CandidateError(f"motion NPZ archive validation failed: {exc}") from exc


def validate_motion_arrays(
    *,
    fps: np.ndarray,
    dof_pos: np.ndarray,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    expected_frames: int,
) -> None:
    expected = {
        "fps": ((1,), fps),
        "dof_pos": ((expected_frames, 29), dof_pos),
        "root_pos": ((expected_frames, 3), root_pos),
        "root_rot": ((expected_frames, 4), root_rot),
    }
    for name, (shape, value) in expected.items():
        if value.shape != shape or value.dtype != np.float32:
            raise CandidateError(
                f"motion {name} must have shape {shape} and dtype float32, got {value.shape} {value.dtype}"
            )
        if not np.all(np.isfinite(value)):
            raise CandidateError(f"motion {name} contains non-finite values")
    if float(fps[0]) != 50.0:
        raise CandidateError("motion fps must be exactly 50")
    norms = np.linalg.norm(root_rot.astype(np.float64), axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 1e-4:
        raise CandidateError(
            f"motion root_rot is not unit normalized; max norm error {max_norm_error}"
        )


def compute_dof_velocities(dof_pos: np.ndarray, dt: float) -> np.ndarray:
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29 or dof_pos.shape[0] < 3:
        raise CandidateError("dof_pos must have shape [frames>=3, 29]")
    if not math.isfinite(dt) or dt <= 0.0:
        raise CandidateError("motion dt must be positive and finite")
    result = np.zeros_like(dof_pos, dtype=np.float32)
    result[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt
    if not np.all(np.isfinite(result)):
        raise CandidateError("derived motion joint velocities are non-finite")
    return np.ascontiguousarray(result, dtype=np.float32)


def load_motion(
    assets_dir: Path,
    role: str,
    manifest_path: Path,
) -> MotionData:
    manifest, _manifest_raw = _load_json_file(manifest_path, "pinned asset manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CandidateError("pinned asset manifest schema mismatch")
    manifest_digest = canonical_json_sha256(manifest)
    if manifest_digest != EXPECTED_MANIFEST_CANONICAL_SHA256:
        raise CandidateError("pinned asset manifest canonical SHA-256 mismatch")
    if manifest.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise CandidateError("pinned asset manifest build fingerprint mismatch")
    assets = _sequence(manifest.get("assets"), "manifest.assets")
    matching = [asset for asset in assets if isinstance(asset, dict) and asset.get("role") == role]
    if len(matching) != 1:
        raise CandidateError(f"motion role {role!r} is not uniquely pinned")
    asset = matching[0]

    assets_dir = assets_dir.resolve()
    if not assets_dir.is_dir() or assets_dir.is_symlink():
        raise CandidateError("assets directory must be a regular, non-symlink directory")
    inventory_path = _regular_file(
        assets_dir / "g1_runtime_assets.inventory.json", "extracted asset inventory"
    )
    inventory, inventory_raw = _load_json_file(inventory_path, "extracted asset inventory")
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise CandidateError("extracted asset inventory schema mismatch")
    if inventory.get("build_fingerprint") != BUILD_FINGERPRINT:
        raise CandidateError("extracted asset inventory build fingerprint mismatch")
    if inventory.get("manifest_sha256") != manifest_digest:
        raise CandidateError("extracted asset inventory manifest hash mismatch")
    if inventory.get("source") != manifest.get("source"):
        raise CandidateError("extracted asset inventory source record mismatch")
    if inventory.get("assets") != assets:
        raise CandidateError("extracted asset inventory asset records mismatch")

    filename = str(asset.get("output"))
    if Path(filename).name != filename or not filename.endswith(".npz"):
        raise CandidateError("pinned motion filename is unsafe")
    motion_path = _regular_file(assets_dir / filename, "motion NPZ")
    size = motion_path.stat().st_size
    if size != asset.get("bytes"):
        raise CandidateError("motion NPZ byte count mismatch")
    digest = sha256_file(motion_path)
    if digest != asset.get("sha256"):
        raise CandidateError("motion NPZ SHA-256 mismatch")
    _validate_npz_members(motion_path, asset)
    try:
        with np.load(motion_path, allow_pickle=False) as archive:
            if set(archive.files) != {"fps", "dof_pos", "root_pos", "root_rot"}:
                raise CandidateError("motion NPZ array set mismatch")
            fps = np.array(archive["fps"], copy=True)
            dof_pos = np.array(archive["dof_pos"], copy=True)
            root_pos = np.array(archive["root_pos"], copy=True)
            root_rot = np.array(archive["root_rot"], copy=True)
    except CandidateError:
        raise
    except Exception as exc:
        raise CandidateError(f"motion NPZ load failed: {exc}") from exc
    expected_frames = int(asset.get("frames", -1))
    validate_motion_arrays(
        fps=fps,
        dof_pos=dof_pos,
        root_pos=root_pos,
        root_rot=root_rot,
        expected_frames=expected_frames,
    )
    if asset.get("dof") != 29 or float(asset.get("fps", -1.0)) != 50.0:
        raise CandidateError("pinned motion metadata mismatch")
    dof_vel = compute_dof_velocities(dof_pos, 1.0 / float(fps[0]))
    return MotionData(
        role=role,
        filename=filename,
        size=size,
        sha256=digest,
        fps=float(fps[0]),
        dof_pos=np.ascontiguousarray(dof_pos),
        dof_vel=dof_vel,
        root_pos=np.ascontiguousarray(root_pos),
        root_rot_xyzw=np.ascontiguousarray(root_rot),
        manifest_sha256=manifest_digest,
        inventory_sha256=sha256_bytes(inventory_raw),
    )


def add_diagnostic_floor(
    source_xml: bytes,
    floor_z: float,
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
) -> str:
    if not math.isfinite(floor_z):
        raise CandidateError("diagnostic floor height must be finite")
    if len(friction) != 3 or any(not math.isfinite(value) or value < 0.0 for value in friction):
        raise CandidateError("diagnostic floor friction must be three finite nonnegative values")
    root = ET.fromstring(source_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise CandidateError("recovered XML has no worldbody")
    if any(geom.get("type", "sphere") == "plane" for geom in root.findall(".//geom")):
        raise CandidateError("refusing to add a diagnostic floor to a model that already has a plane")
    floor = ET.Element(
        "geom",
        {
            "name": "sonic_candidate_diagnostic_floor",
            "type": "plane",
            "pos": f"0 0 {floor_z:.17g}",
            "size": "0 0 0.05",
            "friction": " ".join(f"{value:.17g}" for value in friction),
            "condim": "3",
            "rgba": "0.65 0.65 0.65 1",
        },
    )
    worldbody.insert(0, floor)
    return ET.tostring(root, encoding="unicode")


def _mjcf_numbers(values: Sequence[Any]) -> str:
    return " ".join(f"{_finite_float(value, 'MJCF numeric value'):.17g}" for value in values)


def add_arena_geoms(source_xml: bytes, arena_contract: ArenaContract) -> str:
    """Compose the hash-pinned static arena geoms into the recovered plant."""
    try:
        root = ET.fromstring(source_xml)
    except ET.ParseError as exc:
        raise CandidateError(f"recovered XML parse failed during arena composition: {exc}") from exc
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise CandidateError("recovered XML has no worldbody")
    existing_names = {
        geom.get("name") for geom in root.findall(".//geom") if geom.get("name")
    }
    for index, geom in enumerate(arena_contract.geoms):
        name = str(geom["name"])
        if name in existing_names:
            raise CandidateError(f"arena geom name conflicts with recovered plant: {name!r}")
        contact = _mapping(geom["contact"], f"{name}.contact")
        attributes = {
            "name": name,
            "type": "box",
            "pos": _mjcf_numbers(_sequence(geom["position_m"], f"{name}.position_m")),
            "quat": _mjcf_numbers(
                _sequence(geom["quaternion_wxyz"], f"{name}.quaternion_wxyz")
            ),
            "size": _mjcf_numbers(
                _sequence(geom["half_extents_m"], f"{name}.half_extents_m")
            ),
            "priority": str(_arena_integer(contact["priority"], f"{name}.priority")),
            "contype": str(_arena_integer(contact["contype"], f"{name}.contype")),
            "conaffinity": str(
                _arena_integer(contact["conaffinity"], f"{name}.conaffinity")
            ),
            "group": str(_arena_integer(contact["group"], f"{name}.group")),
            "condim": str(_arena_integer(contact["condim"], f"{name}.condim")),
            "solmix": f"{_finite_float(contact['solmix'], f'{name}.solmix'):.17g}",
            "solref": _mjcf_numbers(_sequence(contact["solref"], f"{name}.solref")),
            "solimp": _mjcf_numbers(_sequence(contact["solimp"], f"{name}.solimp")),
            "margin": f"{_finite_float(contact['margin_m'], f'{name}.margin_m'):.17g}",
            "gap": f"{_finite_float(contact['gap_m'], f'{name}.gap_m'):.17g}",
            "friction": _mjcf_numbers(
                _sequence(contact["friction"], f"{name}.friction")
            ),
            "fluidshape": str(contact["fluidshape"]),
            "fluidcoef": _mjcf_numbers(
                _sequence(contact["fluidcoef"], f"{name}.fluidcoef")
            ),
        }
        worldbody.insert(index, ET.Element("geom", attributes))
        existing_names.add(name)
    return ET.tostring(root, encoding="unicode")


def validate_compiled_arena(
    mujoco: Any, model: Any, arena_contract: ArenaContract
) -> None:
    if int(model.ngeom) != arena_contract.composed_ngeom:
        raise CandidateError(
            f"composed arena ngeom mismatch: expected {arena_contract.composed_ngeom}, got {model.ngeom}"
        )
    expected_box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    observed_ids: list[int] = []
    for geom in arena_contract.geoms:
        name = str(geom["name"])
        try:
            geom_id = int(model.geom(name).id)
        except Exception as exc:
            raise CandidateError(f"compiled arena geom mapping failed for {name!r}: {exc}") from exc
        observed_ids.append(geom_id)
        contact = _mapping(geom["contact"], f"{name}.contact")
        scalar_integer_checks = (
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
        for field, observed, expected in scalar_integer_checks:
            if observed != expected:
                raise CandidateError(
                    f"compiled arena geom {name!r} {field} mismatch: expected {expected}, got {observed}"
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
                raise CandidateError(f"compiled arena geom {name!r} {field} mismatch")
        scalar_float_checks = (
            ("solmix", float(model.geom_solmix[geom_id]), float(contact["solmix"])),
            ("margin", float(model.geom_margin[geom_id]), float(contact["margin_m"])),
            ("gap", float(model.geom_gap[geom_id]), float(contact["gap_m"])),
        )
        for field, observed, expected in scalar_float_checks:
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise CandidateError(f"compiled arena geom {name!r} {field} mismatch")
    if len(set(observed_ids)) != 17:
        raise CandidateError("compiled arena geom mapping is not one-to-one")


def create_mujoco_model(
    xml_contract: XmlContract,
    floor_z: float | None,
    arena_contract: ArenaContract | None = None,
) -> tuple[Any, Any, str]:
    try:
        import mujoco
    except ImportError as exc:
        raise CandidateError("mujoco is required") from exc
    if floor_z is not None and arena_contract is not None:
        raise CandidateError("diagnostic floor and arena contract are mutually exclusive")
    if arena_contract is not None:
        if not math.isclose(
            arena_contract.source_timestep,
            xml_contract.source_timestep,
            rel_tol=0.0,
            abs_tol=1e-17,
        ):
            raise CandidateError("arena and recovered XML source timesteps disagree")
        xml_text = add_arena_geoms(xml_contract.source_bytes, arena_contract)
    elif floor_z is not None:
        xml_text = add_diagnostic_floor(xml_contract.source_bytes, floor_z)
    else:
        xml_text = xml_contract.source_bytes.decode("utf-8")
    try:
        model = mujoco.MjModel.from_xml_string(xml_text)
    except Exception as exc:
        raise CandidateError(f"MuJoCo model compilation failed: {exc}") from exc
    expected_ngeom = 54 if arena_contract is not None else 38 if floor_z is not None else 37
    if int(model.ngeom) != expected_ngeom:
        raise CandidateError(
            f"compiled MuJoCo geom count mismatch: expected {expected_ngeom}, got {model.ngeom}"
        )
    if arena_contract is not None:
        validate_compiled_arena(mujoco, model, arena_contract)
    return mujoco, model, str(getattr(mujoco, "__version__", "unknown"))


def build_runtime_map(model: Any, xml_contract: XmlContract) -> RuntimeMap:
    if (model.nbody, model.njnt, model.nq, model.nv, model.nu) != (31, 30, 36, 35, 29):
        raise CandidateError(
            "MuJoCo dimensions mismatch; expected nbody=31 njnt=30 nq=36 nv=35 nu=29"
        )
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    actuator_ids: list[int] = []
    ctrl_min: list[float] = []
    ctrl_max: list[float] = []
    for spec in xml_contract.joints:
        try:
            joint_id = int(model.joint(spec.xml_joint_name).id)
            actuator_id = int(model.actuator(spec.actuator_name).id)
        except Exception as exc:
            raise CandidateError(f"MuJoCo named mapping failed for {spec.canonical_name}: {exc}") from exc
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise CandidateError(f"actuator transmission mismatch for {spec.canonical_name}")
        if not math.isclose(float(model.actuator_gear[actuator_id, 0]), 1.0):
            raise CandidateError(f"actuator gear mismatch for {spec.canonical_name}")
        model_ctrl = model.actuator_ctrlrange[actuator_id]
        if not np.allclose(
            model_ctrl,
            np.array([spec.ctrl_min, spec.ctrl_max]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise CandidateError(f"compiled actuator ctrlrange mismatch for {spec.canonical_name}")
        if bool(model.actuator_ctrllimited[actuator_id]) or bool(
            model.actuator_forcelimited[actuator_id]
        ):
            raise CandidateError(f"compiled actuator limit flag mismatch for {spec.canonical_name}")
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
        ctrl_min.append(spec.ctrl_min)
        ctrl_max.append(spec.ctrl_max)
    if len(set(joint_ids)) != 29 or len(set(actuator_ids)) != 29:
        raise CandidateError("MuJoCo joint or actuator mapping is not one-to-one")
    if len(set(qpos_addresses)) != 29 or len(set(qvel_addresses)) != 29:
        raise CandidateError("MuJoCo state address mapping is not one-to-one")
    try:
        free_joint_id = int(model.joint(xml_contract.free_joint_name).id)
        root_body_id = int(model.body(xml_contract.root_body_xml_name).id)
        anchor_body_id = int(model.body(xml_contract.anchor_body_xml_name).id)
    except Exception as exc:
        raise CandidateError(f"MuJoCo root or anchor mapping failed: {exc}") from exc
    root_qpos_address = int(model.jnt_qposadr[free_joint_id])
    root_qvel_address = int(model.jnt_dofadr[free_joint_id])
    if root_qpos_address != 0 or root_qvel_address != 0:
        raise CandidateError("recovered free-joint state is not the leading MuJoCo state")
    canonical_body_ids: dict[str, int] = {}
    for body_id in range(1, int(model.nbody)):
        xml_name = model.body(body_id).name
        canonical_name = _canonical_body_name(xml_name)
        if canonical_name in canonical_body_ids:
            raise CandidateError(
                f"duplicate canonical MuJoCo body name {canonical_name!r}"
            )
        canonical_body_ids[canonical_name] = body_id
    if tuple(canonical_body_ids) != BODY_NAMES:
        raise CandidateError(
            "recovered MuJoCo body order does not match the pinned G1 bone layout"
        )
    return RuntimeMap(
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
        body_ids=np.asarray(
            [canonical_body_ids[name] for name in BODY_NAMES], dtype=np.int32
        ),
    )


def _xyzw_to_wxyz(value: np.ndarray) -> np.ndarray:
    return np.asarray(value)[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    return np.asarray(value)[..., [1, 2, 3, 0]]


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    lx, ly, lz, lw = (left[..., index] for index in range(4))
    rx, ry, rz, rw = (right[..., index] for index in range(4))
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    ).astype(np.float32)


def _yaw_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise CandidateError("heading input must be one finite xyzw quaternion")
    x, y, z, w = value
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.asarray([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def align_reference_anchor_rotations(
    robot_anchor_start_xyzw: np.ndarray,
    reference_anchor_rot_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official deployment runner's fixed yaw-only start alignment."""
    references = np.asarray(reference_anchor_rot_xyzw, dtype=np.float32)
    if references.ndim != 2 or references.shape[1] != 4 or references.shape[0] < 1:
        raise CandidateError("reference anchor rotations must have shape [frames>=1, 4]")
    robot_yaw = _yaw_quaternion_xyzw(robot_anchor_start_xyzw)
    motion_yaw = _yaw_quaternion_xyzw(references[0])
    inverse_motion_yaw = motion_yaw.copy()
    inverse_motion_yaw[:3] *= -1.0
    offset = _quat_multiply_xyzw(robot_yaw, inverse_motion_yaw)
    aligned = _quat_multiply_xyzw(
        np.broadcast_to(offset, references.shape), references
    )
    if not np.all(np.isfinite(aligned)):
        raise CandidateError("aligned reference anchor rotations are non-finite")
    norm_error = float(np.max(np.abs(np.linalg.norm(aligned, axis=1) - 1.0)))
    if norm_error > 1e-4:
        raise CandidateError("aligned reference anchor rotations are not unit normalized")
    return np.ascontiguousarray(aligned, dtype=np.float32), offset


def derive_reference_anchor_rotations(
    mujoco: Any,
    model: Any,
    runtime_map: RuntimeMap,
    motion: MotionData,
) -> np.ndarray:
    data = mujoco.MjData(model)
    result = np.empty((motion.dof_pos.shape[0], 4), dtype=np.float32)
    for frame in range(motion.dof_pos.shape[0]):
        mujoco.mj_resetData(model, data)
        start = runtime_map.root_qpos_address
        data.qpos[start : start + 3] = motion.root_pos[frame]
        data.qpos[start + 3 : start + 7] = _xyzw_to_wxyz(motion.root_rot_xyzw[frame])
        data.qpos[runtime_map.qpos_addresses] = motion.dof_pos[frame]
        mujoco.mj_forward(model, data)
        result[frame] = _wxyz_to_xyzw(data.xquat[runtime_map.anchor_body_id]).astype(
            np.float32
        )
    if not np.all(np.isfinite(result)):
        raise CandidateError("reference anchor FK produced non-finite rotations")
    norm_error = float(np.max(np.abs(np.linalg.norm(result, axis=1) - 1.0)))
    if norm_error > 1e-4:
        raise CandidateError("reference anchor FK produced non-unit rotations")
    return np.ascontiguousarray(result)


def initialize_simulation(
    mujoco: Any,
    model: Any,
    data: Any,
    runtime_map: RuntimeMap,
    motion: MotionData,
    root_z_offset: float,
) -> dict[str, Any]:
    if not math.isfinite(root_z_offset):
        raise CandidateError("initial root z offset must be finite")
    mujoco.mj_resetData(model, data)
    root = runtime_map.root_qpos_address
    root_pos = motion.root_pos[0].astype(np.float64).copy()
    root_pos[2] += root_z_offset
    data.qpos[root : root + 3] = root_pos
    data.qpos[root + 3 : root + 7] = _xyzw_to_wxyz(motion.root_rot_xyzw[0])

    initial = motion.dof_pos[0].astype(np.float64)
    applied = initial.copy()
    clipped_names: list[str] = []
    max_correction = 0.0
    for policy_index, joint_id in enumerate(runtime_map.joint_ids):
        if bool(model.jnt_limited[joint_id]):
            lower, upper = model.jnt_range[joint_id]
            clipped = float(np.clip(applied[policy_index], lower, upper))
            correction = abs(clipped - applied[policy_index])
            if correction > 0.0:
                clipped_names.append(JOINT_NAMES[policy_index])
                max_correction = max(max_correction, correction)
            applied[policy_index] = clipped
    data.qpos[runtime_map.qpos_addresses] = applied
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise CandidateError("initial MuJoCo state is non-finite")
    return {
        "root_position": root_pos.tolist(),
        "root_z_offset_metres": root_z_offset,
        "joint_limit_clip_count": len(clipped_names),
        "joint_limit_clipped_names": clipped_names,
        "joint_limit_max_correction_rad": max_correction,
    }


def read_robot_state(data: Any, runtime_map: RuntimeMap) -> dict[str, np.ndarray]:
    result = {
        "dof_pos": np.asarray(data.qpos[runtime_map.qpos_addresses], dtype=np.float32).copy(),
        "dof_vel": np.asarray(data.qvel[runtime_map.qvel_addresses], dtype=np.float32).copy(),
        "anchor_rot": _wxyz_to_xyzw(data.xquat[runtime_map.anchor_body_id])
        .astype(np.float32)
        .copy(),
        "root_local_ang_vel": np.asarray(
            data.qvel[
                runtime_map.root_qvel_address + 3 : runtime_map.root_qvel_address + 6
            ],
            dtype=np.float32,
        ).copy(),
    }
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise CandidateError("MuJoCo state contains non-finite values")
    return result


def read_body_pose(data: Any, runtime_map: RuntimeMap) -> dict[str, Any]:
    positions = np.asarray(data.xpos[runtime_map.body_ids], dtype=np.float64)
    rotations = np.asarray(data.xquat[runtime_map.body_ids], dtype=np.float64)
    if positions.shape != (len(BODY_NAMES), 3) or rotations.shape != (
        len(BODY_NAMES),
        4,
    ):
        raise CandidateError("MuJoCo body pose shape does not match the G1 layout")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        raise CandidateError("MuJoCo body pose contains non-finite values")
    return {
        "layout": "g1_30",
        "ordered_names": list(BODY_NAMES),
        "world_positions_xyz": positions.reshape(-1).tolist(),
        "world_rotations_wxyz": rotations.reshape(-1).tolist(),
    }


def build_onnx_inputs(
    state: Mapping[str, np.ndarray],
    motion: MotionData,
    reference_anchor_rot: np.ndarray,
    current_frame: int,
    previous_actions: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    expected_state_shapes = {
        "dof_pos": (29,),
        "dof_vel": (29,),
        "anchor_rot": (4,),
        "root_local_ang_vel": (3,),
    }
    for name, shape in expected_state_shapes.items():
        if name not in state or np.asarray(state[name]).shape != shape:
            observed = None if name not in state else np.asarray(state[name]).shape
            raise CandidateError(
                f"state {name} must have shape {shape}, got {observed}"
            )
    if previous_actions.shape != (29,) or previous_actions.dtype != np.float32:
        raise CandidateError("previous_actions must be float32 [29]")
    if reference_anchor_rot.shape != (motion.dof_pos.shape[0], 4):
        raise CandidateError("reference anchor rotation array shape mismatch")
    indices = np.asarray([current_frame + step for step in FUTURE_STEPS], dtype=np.int64)
    if indices[-1] >= motion.dof_pos.shape[0]:
        raise CandidateError("future motion request exceeds the validated clip")
    feeds = {
        "current_anchor_rot": state["anchor_rot"][None],
        "current_dof_pos": state["dof_pos"][None],
        "current_dof_vel": state["dof_vel"][None],
        "current_root_local_ang_vel": state["root_local_ang_vel"][None],
        "historical_processed_actions": previous_actions[None, None],
        "mimic_future_anchor_rot": reference_anchor_rot[indices][None],
        "mimic_future_dof_pos": motion.dof_pos[indices][None],
        "mimic_future_dof_vel": motion.dof_vel[indices][None],
    }
    for name, value in feeds.items():
        feeds[name] = np.ascontiguousarray(value, dtype=np.float32)
    return feeds, indices


def validate_policy_outputs(
    outputs: Sequence[Any],
    policy: PolicyContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(outputs) != 4:
        raise CandidateError("ONNX inference returned the wrong output count")
    result: list[np.ndarray] = []
    for raw, (name, _shape) in zip(outputs, OUTPUT_CONTRACT):
        value = np.asarray(raw)
        if value.shape != (1, 29) or value.dtype != np.float32:
            raise CandidateError(
                f"ONNX output {name} must be float32 [1,29], got {value.dtype} {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise CandidateError(f"ONNX output {name} contains non-finite values")
        result.append(np.ascontiguousarray(value[0]))
    if not np.allclose(result[2], policy.stiffness, rtol=1e-5, atol=1e-5):
        raise CandidateError("ONNX stiffness output does not match pinned policy metadata")
    if not np.allclose(result[3], policy.damping, rtol=1e-5, atol=1e-5):
        raise CandidateError("ONNX damping output does not match pinned policy metadata")
    return result[0], result[1], result[2], result[3]


def compute_pd_torque(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_targets: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    ctrl_min: np.ndarray,
    ctrl_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [joint_pos, joint_vel, joint_targets, stiffness, damping, ctrl_min, ctrl_max]
    if any(np.asarray(value).shape != (29,) for value in arrays):
        raise CandidateError("PD inputs must all have shape [29]")
    raw = (
        np.asarray(stiffness, dtype=np.float64)
        * (np.asarray(joint_targets, dtype=np.float64) - np.asarray(joint_pos, dtype=np.float64))
        - np.asarray(damping, dtype=np.float64) * np.asarray(joint_vel, dtype=np.float64)
    )
    if not np.all(np.isfinite(raw)):
        raise CandidateError("PD torque is non-finite")
    clipped = np.clip(raw, ctrl_min, ctrl_max)
    saturated = raw != clipped
    return raw, clipped, saturated


def _joint_mapping_report(runtime_map: RuntimeMap, xml_contract: XmlContract) -> list[dict[str, Any]]:
    report = []
    for index, spec in enumerate(xml_contract.joints):
        report.append(
            {
                "policy_index": index,
                "joint": spec.canonical_name,
                "xml_joint": spec.xml_joint_name,
                "mujoco_joint_id": int(runtime_map.joint_ids[index]),
                "qpos_address": int(runtime_map.qpos_addresses[index]),
                "qvel_address": int(runtime_map.qvel_addresses[index]),
                "actuator": spec.actuator_name,
                "mujoco_actuator_id": int(runtime_map.actuator_ids[index]),
                "manual_torque_clip_nm": [spec.ctrl_min, spec.ctrl_max],
            }
        )
    return report


def _artifact_report(
    *,
    onnx_digest: str,
    onnx_size: int,
    policy: PolicyContract,
    xml_contract: XmlContract,
    motion: MotionData,
    arena_contract: ArenaContract | None = None,
) -> dict[str, Any]:
    report = {
        "protomotions_commit": PROTOMOTIONS_COMMIT,
        "onnx": {"sha256": onnx_digest, "bytes": onnx_size},
        "yaml": {"sha256": policy.yaml_sha256, "bytes": policy.yaml_size},
        "recovered_xml": {
            "sha256": xml_contract.source_sha256,
            "bytes": xml_contract.source_size,
        },
        "motion_manifest": {"canonical_sha256": motion.manifest_sha256},
        "motion_inventory": {"sha256": motion.inventory_sha256},
        "motion_npz": {
            "role": motion.role,
            "name": motion.filename,
            "sha256": motion.sha256,
            "bytes": motion.size,
        },
        "rek_build_fingerprint": BUILD_FINGERPRINT,
    }
    if arena_contract is not None:
        report["arena_contract"] = {
            "schema": ARENA_SCHEMA,
            "sha256": arena_contract.source_sha256,
            "bytes": arena_contract.source_size,
            "derived_geometry_sha256": arena_contract.derived_geometry_sha256,
            "probe_geometry_signature_sha256": arena_contract.probe_geometry_signature_sha256,
        }
    return report


def _segment_report(
    segment: SegmentRequest,
    duration_ticks: int | None,
    control_dt: float,
) -> dict[str, Any]:
    return {
        "schema": SEGMENT_SCHEMA,
        "source": segment.source,
        "source_sha256": segment.source_sha256,
        "motion_role": segment.motion_role,
        "duration_ticks": duration_ticks,
        "duration_seconds": None
        if duration_ticks is None
        else duration_ticks * control_dt,
        "semantic_command": segment.semantic_command,
        "role_binding_authority": "caller_supplied_not_recovered",
    }


def _arena_environment_report(arena_contract: ArenaContract) -> dict[str, Any]:
    spawn_summary: dict[str, Any] = {}
    for slot in ("player", "opponent"):
        spawn = _mapping(
            arena_contract.spawn_points.get(slot), f"arena.spawn_points.{slot}"
        )
        spawn_summary[slot] = {
            "position_m": spawn["mujoco_world_position_m"],
            "quaternion_wxyz": spawn["mujoco_world_quaternion_wxyz"],
        }
    return {
        "mode": "recovered_static_arena_contract",
        "schema": ARENA_SCHEMA,
        "sha256": arena_contract.source_sha256,
        "geom_count": len(arena_contract.geoms),
        "composed_ngeom": arena_contract.composed_ngeom,
        "roles": {"floor": 1, "pillar": 8, "wall": 8},
        "shape": "box",
        "control_equivalent": False,
        "derived_geometry_sha256": arena_contract.derived_geometry_sha256,
        "floor_top_z_metres": arena_contract.floor_top_z_m,
        "source_timestep_seconds": arena_contract.source_timestep,
        "source_mujoco_steps_per_fixed_update": 1,
        "global_options_source": "hash_pinned_recovered_xml_bound_by_arena_contract",
        "contact_settings_source": "per_geom_hash_pinned_arena_contract",
        "coordinate_mapping_validated": True,
        "spawn_anchors": spawn_summary,
        "spawn_rebase_applied": False,
        "spawn_rebase_status": "not_inferred_missing_free_joint_pelvis_height",
        "contract_unknowns": list(arena_contract.unknowns),
    }


def _diagnostic_floor_environment_report(floor_z: float) -> dict[str, Any]:
    return {
        "mode": "synthetic_diagnostic_floor",
        "name": "sonic_candidate_diagnostic_floor",
        "z_metres": floor_z,
        "friction": [1.0, 0.005, 0.0001],
        "source": "caller_explicit_parameter",
        "recovered_arena": False,
    }


def _base_report(
    *,
    artifacts: Mapping[str, Any],
    policy: PolicyContract,
    xml_contract: XmlContract,
    runtime_map: RuntimeMap,
    motion: MotionData,
    onnxruntime_version: str,
    mujoco_version: str,
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "classification": CLASSIFICATION,
        "rek_parity_claim": False,
        "artifacts": artifacts,
        "runtime_versions": {
            "numpy": np.__version__,
            "onnxruntime": onnxruntime_version,
            "mujoco": mujoco_version,
        },
        "policy_contract": {
            "joint_count": 29,
            "policy_hz": 1.0 / policy.control_dt,
            "physics_hz": 1.0 / policy.physics_dt,
            "pd_torque_update_hz": 1.0 / policy.physics_dt,
            "decimation": policy.decimation,
            "future_step_indices": list(FUTURE_STEPS),
            "future_dt_seconds": [step * policy.control_dt for step in FUTURE_STEPS],
            "onnx_provider": "CPUExecutionProvider",
            "onnx_intra_op_threads": 1,
            "onnx_inter_op_threads": 1,
            "onnx_execution": "sequential",
            "historical_processed_actions_feedback": "previous_joint_pos_targets",
            "quaternion_order": "xyzw_at_onnx_boundary",
            "quaternion_sign_canonicalization": "none",
        },
        "upstream_deployment_evidence": UPSTREAM_DEPLOYMENT_EVIDENCE,
        "onnx_input_provenance": {
            "current_dof_pos": "upstream_exact_named_mujoco_qpos_mapping",
            "current_dof_vel": "upstream_exact_named_mujoco_qvel_mapping",
            "current_anchor_rot": "upstream_exact_named_torso_xquat_wxyz_reordered_to_xyzw",
            "current_root_local_ang_vel": "upstream_exact_free_joint_qvel_angular_slice_already_local_no_rotation",
            "historical_processed_actions": "upstream_exact_previous_joint_pos_targets_zero_initialized",
            "mimic_future_dof_pos": "build_pinned_rek_npz_in_policy_joint_order",
            "mimic_future_dof_vel": "native_rek_forward_finite_difference_from_build_pinned_npz",
            "mimic_future_anchor_rot": "inferred_recovered_plant_fk_because_npz_omits_per_body_rotation",
            "reference_heading_alignment": "upstream_exact_yaw_only_start_alignment_applied_at_runtime",
        },
        "motion_contract": {
            "frames": int(motion.dof_pos.shape[0]),
            "fps": motion.fps,
            "velocity_derivation": "native_rek_forward_finite_difference_with_zero_terminal_frame",
            "reference_anchor_rotation_derivation": "recovered_plant_fk_from_npz_pelvis_rotation_and_joint_position",
            "npz_quaternion_order": "xyzw",
        },
        "recovered_xml_semantics": {
            "source_timestep_seconds": xml_contract.source_timestep,
            "source_has_floor": xml_contract.source_has_floor,
            "actuator_type": "direct_motor",
            "actuator_gear": 1.0,
            "source_ctrllimited": False,
            "source_forcelimited": False,
            "pd_torque_equation": "stiffness*(target-position)-damping*velocity",
            "pd_torque_update_rate_hz": 1.0 / policy.physics_dt,
            "manual_runtime_torque_clip": "declared_source_ctrlrange",
            "manual_runtime_torque_clip_authority": "candidate_safety_assumption_not_enforced_by_source_limit_flags",
            "passive_joint_damping_preserved": True,
            "passive_joint_damping_range": list(xml_contract.passive_damping_range),
            "joint_frictionloss_preserved": True,
            "joint_frictionloss_range": list(xml_contract.frictionloss_range),
        },
        "joint_mapping": _joint_mapping_report(runtime_map, xml_contract),
        "trajectory_acceptance": {
            "rek_comparison": "not_run",
            "trajectory_tests_passed": False,
            "rek_parity_eligible": False,
        },
        "limits": list(LIMITS),
    }


def validate_only(
    *,
    policy: PolicyContract,
    xml_contract: XmlContract,
    motion: MotionData,
    onnx_digest: str,
    onnx_size: int,
    onnxruntime_version: str,
    segment: SegmentRequest,
    arena_contract: ArenaContract | None,
) -> dict[str, Any]:
    mujoco, model, mujoco_version = create_mujoco_model(
        xml_contract, None, arena_contract
    )
    runtime_map = build_runtime_map(model, xml_contract)
    derive_reference_anchor_rotations(mujoco, model, runtime_map, motion)
    report = _base_report(
        artifacts=_artifact_report(
            onnx_digest=onnx_digest,
            onnx_size=onnx_size,
            policy=policy,
            xml_contract=xml_contract,
            motion=motion,
            arena_contract=arena_contract,
        ),
        policy=policy,
        xml_contract=xml_contract,
        runtime_map=runtime_map,
        motion=motion,
        onnxruntime_version=onnxruntime_version,
        mujoco_version=mujoco_version,
    )
    report.update(
        {
            "status": "validated",
            "execution": {"mode": "validation_only", "simulation_ran": False},
            "segment": _segment_report(
                segment, segment.duration_ticks, policy.control_dt
            ),
            "planned_runtime_modifications": {
                "physics_timestep_seconds": policy.physics_dt,
                "source_timestep_will_be_overridden": True,
                "simulation_environment_requirement": "exactly_one_of_arena_contract_or_diagnostic_floor",
                "validated_environment": None
                if arena_contract is None
                else _arena_environment_report(arena_contract),
            },
            "gate": {
                "passed": True,
                "scope": "candidate_artifact_schema_and_mapping_only",
                "checks": [
                    "artifact_hashes",
                    "yaml_schema",
                    "onnx_schema",
                    "motion_inventory",
                    "motion_schema",
                    "xml_actuator_semantics",
                    "joint_name_mapping",
                    "reference_fk",
                ]
                + ([] if arena_contract is None else ["arena_contract_hash_and_schema", "arena_17_box_composition"]),
            },
        }
    )
    return report


def run_candidate(
    *,
    session: Any,
    policy: PolicyContract,
    xml_contract: XmlContract,
    motion: MotionData,
    onnx_digest: str,
    onnx_size: int,
    onnxruntime_version: str,
    diagnostic_floor_z: float | None,
    arena_contract: ArenaContract | None,
    requested_steps: int | None,
    segment: SegmentRequest,
) -> tuple[dict[str, Any], list[str]]:
    if (diagnostic_floor_z is None) == (arena_contract is None):
        raise CandidateError(
            "simulation requires exactly one of --arena-contract or --diagnostic-floor-z"
        )
    if diagnostic_floor_z is not None and not math.isfinite(diagnostic_floor_z):
        raise CandidateError("diagnostic floor height must be finite")
    max_steps = motion.dof_pos.shape[0] - max(FUTURE_STEPS)
    if max_steps <= 0:
        raise CandidateError("motion clip is too short for the pinned future offsets")
    steps = max_steps if requested_steps is None else requested_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or steps > max_steps:
        raise CandidateError(f"steps must be in [1, {max_steps}]")

    mujoco, model, mujoco_version = create_mujoco_model(
        xml_contract, diagnostic_floor_z, arena_contract
    )
    runtime_map = build_runtime_map(model, xml_contract)
    original_timestep = float(model.opt.timestep)
    if not math.isclose(
        original_timestep,
        xml_contract.source_timestep,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise CandidateError("compiled MuJoCo source timestep mismatch")
    model.opt.timestep = policy.physics_dt
    if not math.isclose(float(model.opt.timestep), PHYSICS_DT, abs_tol=1e-15):
        raise CandidateError("MuJoCo physics timestep override failed")
    data = mujoco.MjData(model)
    reference_anchor_rot = derive_reference_anchor_rotations(
        mujoco, model, runtime_map, motion
    )
    initialization = initialize_simulation(
        mujoco,
        model,
        data,
        runtime_map,
        motion,
        0.0 if arena_contract is not None else float(diagnostic_floor_z),
    )
    initial_state = read_robot_state(data, runtime_map)
    reference_anchor_rot, heading_offset = align_reference_anchor_rotations(
        initial_state["anchor_rot"], reference_anchor_rot
    )
    initialization["reference_heading_alignment"] = {
        "method": "upstream_yaw_only_start_alignment",
        "offset_quaternion_xyzw": heading_offset.tolist(),
        "quaternion_sign_canonicalization": "none",
    }

    previous_actions = np.zeros(29, dtype=np.float32)
    tracking_sq_sum = 0.0
    target_sq_sum = 0.0
    torque_sq_sum = 0.0
    tracking_max = 0.0
    target_max = 0.0
    torque_max = 0.0
    saturation_count = 0
    joint_limit_violation_count = 0
    joint_limit_violation_max_rad = 0.0
    root_heights = [float(data.qpos[runtime_map.root_qpos_address + 2])]
    step_lines: list[str] = []

    for control_step in range(steps):
        state_before = read_robot_state(data, runtime_map)
        feeds, future_indices = build_onnx_inputs(
            state_before,
            motion,
            reference_anchor_rot,
            control_step,
            previous_actions,
        )
        try:
            raw_outputs = session.run([name for name, _shape in OUTPUT_CONTRACT], feeds)
        except Exception as exc:
            raise CandidateError(f"ONNX inference failed at control step {control_step}: {exc}") from exc
        actions, targets, stiffness, damping = validate_policy_outputs(
            raw_outputs, policy
        )
        substep_torque_raw: list[list[float]] = []
        substep_torque_applied: list[list[float]] = []
        substep_saturated_indices: list[list[int]] = []
        for _physics_substep in range(policy.decimation):
            torque_raw, torque_applied, saturated = compute_pd_torque(
                np.asarray(data.qpos[runtime_map.qpos_addresses]),
                np.asarray(data.qvel[runtime_map.qvel_addresses]),
                targets,
                stiffness,
                damping,
                runtime_map.ctrl_min,
                runtime_map.ctrl_max,
            )
            data.ctrl[:] = 0.0
            data.ctrl[runtime_map.actuator_ids] = torque_applied
            torque_sq_sum += float(np.dot(torque_applied, torque_applied))
            torque_max = max(torque_max, float(np.max(np.abs(torque_applied))))
            saturation_count += int(np.count_nonzero(saturated))
            substep_torque_raw.append(torque_raw.tolist())
            substep_torque_applied.append(torque_applied.tolist())
            substep_saturated_indices.append(np.flatnonzero(saturated).tolist())
            mujoco.mj_step(model, data)
        state_after = read_robot_state(data, runtime_map)

        reference_error = state_after["dof_pos"].astype(np.float64) - motion.dof_pos[
            control_step + 1
        ].astype(np.float64)
        target_error = state_after["dof_pos"].astype(np.float64) - targets.astype(
            np.float64
        )
        tracking_sq_sum += float(np.dot(reference_error, reference_error))
        target_sq_sum += float(np.dot(target_error, target_error))
        tracking_max = max(tracking_max, float(np.max(np.abs(reference_error))))
        target_max = max(target_max, float(np.max(np.abs(target_error))))
        for policy_index, joint_id in enumerate(runtime_map.joint_ids):
            if bool(model.jnt_limited[joint_id]):
                lower, upper = model.jnt_range[joint_id]
                value = float(state_after["dof_pos"][policy_index])
                if value < lower - 1e-8 or value > upper + 1e-8:
                    joint_limit_violation_count += 1
                    joint_limit_violation_max_rad = max(
                        joint_limit_violation_max_rad,
                        float(lower - value) if value < lower else float(value - upper),
                    )
        root_height = float(data.qpos[runtime_map.root_qpos_address + 2])
        root_heights.append(root_height)

        trace_record = {
            "type": "step",
            "control_step": control_step,
            "simulation_time_seconds": (control_step + 1) * policy.control_dt,
            "future_reference_indices": future_indices.tolist(),
            "state_before": {name: value.tolist() for name, value in state_before.items()},
            "raw_actions": actions.tolist(),
            "joint_pos_targets": targets.tolist(),
            "stiffness": stiffness.tolist(),
            "damping": damping.tolist(),
            "substep_torque_raw_nm": substep_torque_raw,
            "substep_torque_applied_nm": substep_torque_applied,
            "substep_saturated_policy_indices": substep_saturated_indices,
            "state_after": {name: value.tolist() for name, value in state_after.items()},
            "root_position_after": np.asarray(
                data.qpos[
                    runtime_map.root_qpos_address : runtime_map.root_qpos_address + 3
                ]
            ).tolist(),
            "root_quaternion_wxyz_after": np.asarray(
                data.qpos[
                    runtime_map.root_qpos_address
                    + 3 : runtime_map.root_qpos_address
                    + 7
                ]
            ).tolist(),
            "body_pose_after": read_body_pose(data, runtime_map),
        }
        step_lines.append(
            json.dumps(
                trace_record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        previous_actions = targets.copy()

    control_element_count = steps * 29
    torque_element_count = steps * policy.decimation * 29
    trace_digest = sha256_bytes(("\n".join(step_lines) + "\n").encode("utf-8"))
    report = _base_report(
        artifacts=_artifact_report(
            onnx_digest=onnx_digest,
            onnx_size=onnx_size,
            policy=policy,
            xml_contract=xml_contract,
            motion=motion,
            arena_contract=arena_contract,
        ),
        policy=policy,
        xml_contract=xml_contract,
        runtime_map=runtime_map,
        motion=motion,
        onnxruntime_version=onnxruntime_version,
        mujoco_version=mujoco_version,
    )
    report.update(
        {
            "status": "ok",
            "execution": {
                "mode": "simulation",
                "control_steps": steps,
                "physics_steps": steps * policy.decimation,
                "simulation_seconds": steps * policy.control_dt,
            },
            "segment": _segment_report(segment, steps, policy.control_dt),
            "runtime_modifications": {
                "physics_timestep": {
                    "source_seconds": original_timestep,
                    "applied_seconds": policy.physics_dt,
                    "authority": "pinned_official_policy_yaml",
                },
                "environment": _arena_environment_report(arena_contract)
                if arena_contract is not None
                else _diagnostic_floor_environment_report(float(diagnostic_floor_z)),
                "initialization": initialization,
            },
            "metrics": {
                "reference_joint_rmse_rad": math.sqrt(
                    tracking_sq_sum / control_element_count
                ),
                "reference_joint_max_abs_error_rad": tracking_max,
                "pd_target_rmse_rad": math.sqrt(
                    target_sq_sum / control_element_count
                ),
                "pd_target_max_abs_error_rad": target_max,
                "applied_torque_rms_nm": math.sqrt(
                    torque_sq_sum / torque_element_count
                ),
                "applied_torque_max_abs_nm": torque_max,
                "torque_saturation_count": saturation_count,
                "torque_saturation_fraction": saturation_count
                / torque_element_count,
                "joint_limit_violation_count": joint_limit_violation_count,
                "joint_limit_violation_max_rad": joint_limit_violation_max_rad,
                "root_height_min_metres": min(root_heights),
                "root_height_max_metres": max(root_heights),
                "root_height_final_metres": root_heights[-1],
                "deterministic_step_trace_sha256": trace_digest,
            },
            "gate": {
                "passed": True,
                "scope": "candidate_artifact_schema_mapping_and_runtime_health_only",
                "checks": [
                    "artifact_hashes",
                    "yaml_schema",
                    "onnx_schema",
                    "motion_inventory",
                    "motion_schema",
                    "xml_actuator_semantics",
                    "joint_name_mapping",
                    "hash_pinned_arena_contract"
                    if arena_contract is not None
                    else "explicit_synthetic_diagnostic_floor",
                    "physics_rate_1000_hz",
                    "finite_policy_outputs",
                    "policy_gain_outputs",
                    "finite_simulation_state",
                ],
            },
        }
    )
    return report, step_lines


def _write_new_text(path: Path, text: str, label: str) -> Path:
    path = Path(os.path.abspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise CandidateError(f"{label} already exists; refusing to overwrite it") from exc
    return path


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "rek" / "evidence" / "g1_runtime_assets.v1.json"


def _default_xml_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rek"
        / "evidence"
        / "evidence_out"
        / "g1_29dof.recovered.xml"
    )


def _parse_positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="pinned unified_pipeline.onnx")
    parser.add_argument("--yaml", required=True, type=Path, help="pinned unified_pipeline.yaml")
    parser.add_argument("--assets-dir", required=True, type=Path, help="external extracted G1 asset directory")
    segment_source = parser.add_mutually_exclusive_group(required=True)
    segment_source.add_argument(
        "--motion-role",
        help="build-pinned motion role held for --steps ticks or the maximum valid clip span",
    )
    segment_source.add_argument(
        "--segment-json",
        type=Path,
        help="held-duration semantic adapter envelope using rek.g1_sonic_candidate.segment.v1",
    )
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--xml", type=Path, default=_default_xml_path())
    environment = parser.add_mutually_exclusive_group()
    environment.add_argument(
        "--arena-contract",
        type=Path,
        help="hash-pinned recovered 17-box arena contract; recommended for simulation",
    )
    environment.add_argument(
        "--diagnostic-floor-z",
        type=float,
        help="explicit synthetic floor height in metres for isolated diagnostics only",
    )
    parser.add_argument("--steps", type=_parse_positive_int, help="control steps, bounded by the clip")
    parser.add_argument("--validate-only", action="store_true", help="validate all artifacts and mappings without stepping")
    parser.add_argument("--trace", type=Path, help="new JSONL trace path; simulation only")
    parser.add_argument("--metrics-out", type=Path, help="new JSON metrics path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_only and args.trace is not None:
            raise CandidateError("--trace cannot be used with --validate-only")
        if args.validate_only and args.diagnostic_floor_z is not None:
            raise CandidateError(
                "--diagnostic-floor-z cannot be used with --validate-only"
            )
        if not args.validate_only and args.arena_contract is None and args.diagnostic_floor_z is None:
            raise CandidateError(
                "simulation requires exactly one of --arena-contract or --diagnostic-floor-z"
            )
        if (
            args.trace is not None
            and args.metrics_out is not None
            and Path(os.path.abspath(args.trace)) == Path(os.path.abspath(args.metrics_out))
        ):
            raise CandidateError("--trace and --metrics-out must be different paths")
        if args.segment_json is not None:
            if args.steps is not None:
                raise CandidateError("--steps cannot override a semantic segment envelope")
            segment = load_segment_request(args.segment_json)
        else:
            segment = SegmentRequest(
                source="cli_motion_role_and_steps",
                motion_role=args.motion_role,
                duration_ticks=args.steps,
                semantic_command=None,
                source_sha256=None,
            )
        policy = load_policy_contract(args.yaml)
        xml_contract = inspect_xml_contract(args.xml, policy.joint_names)
        arena_contract = (
            None
            if args.arena_contract is None
            else load_arena_contract(args.arena_contract)
        )
        motion = load_motion(args.assets_dir, segment.motion_role, args.manifest)
        max_segment_ticks = motion.dof_pos.shape[0] - max(FUTURE_STEPS)
        if segment.duration_ticks is not None and segment.duration_ticks > max_segment_ticks:
            raise CandidateError(
                f"segment duration exceeds the motion future-reference bound {max_segment_ticks}"
            )
        session, onnx_digest, onnx_size, ort_version = create_onnx_session(args.onnx)
        if args.validate_only:
            report = validate_only(
                policy=policy,
                xml_contract=xml_contract,
                motion=motion,
                onnx_digest=onnx_digest,
                onnx_size=onnx_size,
                onnxruntime_version=ort_version,
                segment=segment,
                arena_contract=arena_contract,
            )
            step_lines: list[str] = []
        else:
            report, step_lines = run_candidate(
                session=session,
                policy=policy,
                xml_contract=xml_contract,
                motion=motion,
                onnx_digest=onnx_digest,
                onnx_size=onnx_size,
                onnxruntime_version=ort_version,
                diagnostic_floor_z=args.diagnostic_floor_z,
                arena_contract=arena_contract,
                requested_steps=segment.duration_ticks,
                segment=segment,
            )
        if args.trace is not None:
            header = {
                "type": "header",
                "schema": TRACE_SCHEMA,
                "classification": CLASSIFICATION,
                "rek_parity_claim": False,
                "artifacts": report["artifacts"],
                "policy_contract": report["policy_contract"],
                "runtime_modifications": report["runtime_modifications"],
                "segment": report["segment"],
                "joint_mapping": report["joint_mapping"],
                "limits": report["limits"],
            }
            trace_text = json.dumps(
                header, sort_keys=True, separators=(",", ":"), allow_nan=False
            ) + "\n"
            trace_text += "\n".join(step_lines) + "\n"
            trace_path = _write_new_text(args.trace, trace_text, "trace output")
            report["trace"] = {
                "path": str(trace_path),
                "sha256": sha256_bytes(trace_text.encode("utf-8")),
                "records": len(step_lines) + 1,
            }
        if args.metrics_out is not None:
            report["metrics_output_path"] = str(Path(os.path.abspath(args.metrics_out)))
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.metrics_out is not None:
            _write_new_text(args.metrics_out, output, "metrics output")
        sys.stdout.write(output)
        return 0
    except (CandidateError, OSError, ValueError) as exc:
        error = {
            "schema": RUN_SCHEMA,
            "classification": CLASSIFICATION,
            "rek_parity_claim": False,
            "status": "rejected",
            "gate": {"passed": False},
            "error": str(exc),
            "limits": list(LIMITS),
        }
        sys.stderr.write(json.dumps(error, indent=2, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
