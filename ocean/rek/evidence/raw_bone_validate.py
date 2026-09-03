#!/usr/bin/env python3
"""Validate a passive REK protocol/root-motion capture without invention.

The v5 recorder projects exact REK_Input and REK_Move request bodies, copies
fight and bone FastBufferReader bodies before REK consumes them, and observes
the corresponding post-receive state. V6 preserves those boundaries and adds
a fail-closed private T800-vs-T800 scope plus a 500 Hz root/camera stream with
UTC and Stopwatch timestamps. This validator checks those measurements. It
does not infer send completion, a server tick, a server timestamp, move
acceptance, execution, or request-to-pose causality.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path
from typing import Any


SCHEMA_V5 = "rek.private_ai.protocol.v5"
SCHEMA_V6 = "rek.private_ai.protocol.v6"
SCHEMA = SCHEMA_V5
EXPECTED_PLUGIN_VERSION_V5 = "0.5.1"
EXPECTED_PLUGIN_SHA256_V5 = (
    "f9848c17a83ae011f046aa2baa1fdfd2377dfcfc6d287728265d2b28ea3ce0a2"
)
EXPECTED_PLUGIN_VERSION_V6 = "0.6.1"
EXPECTED_PLUGIN_SHA256_V6 = (
    "24cbea0a149589b71c093e989f43b8dac4862e73d103c323f0f9472a38355e0b"
)
EXPECTED_PLUGIN_VERSION = EXPECTED_PLUGIN_VERSION_V5
EXPECTED_PLUGIN_SHA256 = EXPECTED_PLUGIN_SHA256_V5
EXPECTED_GAME_ASSEMBLY_SHA256 = (
    "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412"
)
EXPECTED_METADATA_SHA256 = (
    "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd"
)
T800_BONE_NAMES = (
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
)
G1_BONE_NAMES = (
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
T800_RUNTIME_OBJECT_NAME = "engineai_t800_FactoryPolicy(Clone)"
G1_RUNTIME_OBJECT_NAME = "g1_29dof_Prefab_SONIC(Clone)"
T800_BONE_COUNT = len(T800_BONE_NAMES)
G1_BONE_COUNT = len(G1_BONE_NAMES)
T800_BODY_BYTES = 2 + 28 * T800_BONE_COUNT
G1_BODY_BYTES = 2 + 28 * G1_BONE_COUNT
T800_BONE_SIGNATURE_SHA256 = (
    "ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab"
)
EXPECTED_BONE_LAYOUTS = {
    "t800_26": {
        "bone_names": T800_BONE_NAMES,
        "bone_count": T800_BONE_COUNT,
        "body_bytes": T800_BODY_BYTES,
        "runtime_object_name": T800_RUNTIME_OBJECT_NAME,
    },
    "g1_30": {
        "bone_names": G1_BONE_NAMES,
        "bone_count": G1_BONE_COUNT,
        "body_bytes": G1_BODY_BYTES,
        "runtime_object_name": G1_RUNTIME_OBJECT_NAME,
    },
}
EXPECTED_PROVENANCE = (
    "read-only copy of FastBufferReader at "
    "REKApp.Robot.OnBoneMessageReceived prefix"
)
EXPECTED_FIGHT_STATE_PROVENANCE = (
    "read-only copy of FastBufferReader at "
    "REKApp.FightCoordinator.ApplyFightStateSnapshot prefix"
)
EXPECTED_SCORE_PROVENANCE = (
    "read-only copy of FastBufferReader at "
    "REKApp.FightCoordinator.OnScoreReceived prefix"
)
EXPECTED_HIT_PROVENANCE = (
    "read-only copy of FastBufferReader at "
    "REKApp.FightCoordinator.OnHitReceived prefix"
)
EXPECTED_BONE_PROTOCOL = {
    "message": "REK_Bones",
    "body_layout": "uint8 networkIndex; uint8 boneCount; repeated float32 little-endian worldPosition.xyz and worldRotation.xyzw",
    "body_size_formula_bytes": "2 + 28 * boneCount",
    # Recorder v0.5.1 emitted these mislabelled literals. They are checked as
    # pinned capture-format bytes, never used to classify either fighter.
    "t800_bone_count": 30,
    "t800_body_bytes": 842,
    "intended_send_interval_seconds": 0.02,
    "intended_send_rate_hz": 50,
    "delivery": "unreliable",
    "native_method": "REKApp.Robot.ServerSendBones RVA 0x23D7D00",
    "native_method_source_sha256": "4f61233092542b15773e49d8404790a8ed89352d3b656fa41b75bab9c8283ded",
}
EXPECTED_BONE_PROTOCOL_V6 = {
    **EXPECTED_BONE_PROTOCOL,
    "t800_bone_count": T800_BONE_COUNT,
    "t800_body_bytes": T800_BODY_BYTES,
    "t800_ordered_bone_signature_sha256": T800_BONE_SIGNATURE_SHA256,
}
EXPECTED_FIGHT_PROTOCOL = {
    "fight_state": "REK_FightState: packed 33-byte little-endian memcpy; reliable; nominal 0.1 s interval",
    "score": "REK_Score: packed 7-byte little-endian memcpy; reliable; emitted per scoring event",
    "hit": "REK_Hit: packed 29-byte little-endian memcpy; unreliable; effects telemetry without fighter identity",
    "server_tick_available": False,
    "native_source_sha256": "9e847da90b34d96db852cf06c3dcd04760d92f4b4f84f525a2de140bbe6d42f5",
}
EXPECTED_OUTBOUND_PROTOCOL = {
    "observation_boundary": "RobotInputController.Send* prefix calls reached from RobotInputController.LateUpdate",
    "input": "REK_Input: uint8-truncated networkIndex plus velocity xyz as three float32 little-endian values; 13 bytes; unreliable",
    "move": "REK_Move: uint8-truncated networkIndex plus uint8-truncated pendingMoveIndex; 2 bytes; reliable",
    "special_and_estop": "invocation-only observations; wire layout and delivery deliberately unclaimed",
    "server_tick_available": False,
    "server_acceptance_available": False,
    "acknowledgement_observed": False,
    "native_source_sha256": "f248df08449e3ff0706ce15ea07e4d58517f2fc9ed3f3143473fa48c4323bc21",
}
EXPECTED_HOOKS = [
    "REKApp.RobotInputController.SendVelocityCommand:prefix_exact_REK_Input_request_projection",
    "REKApp.RobotInputController.SendMoveEvent:prefix_exact_REK_Move_request_projection",
    "REKApp.RobotInputController.SendSpecialEvent:prefix_observation",
    "REKApp.RobotInputController.SendEStopToggle:prefix_observation",
    "REKApp.FightCoordinator.ApplyFightStateSnapshot:prefix_raw_packet_copy_and_postfix_applied_state_correlation",
    "REKApp.FightCoordinator.OnScoreReceived:prefix_raw_packet_copy",
    "REKApp.FightCoordinator.OnHitReceived:prefix_raw_packet_copy",
    "REKApp.Robot.OnBoneMessageReceived:prefix_raw_packet_copy_and_postfix_decoded_snapshot_observation",
]
FIGHT_PHASE_NAMES = {
    0: "Idle", 1: "RoundActive", 2: "RoundEnd", 3: "BetweenRounds",
    4: "FightOver", 5: "Setup", 6: "Sandbox",
}
ROUND_RESULT_NAMES = {
    0: "InProgress", 1: "WonByPoints", 2: "WonByKO", 3: "Tie", 4: "Redo",
}
FIGHT_RESULT_NAMES = {0: "InProgress", 1: "WonByRounds", 2: "WonByTKO"}
FIGHT_FORMAT_NAMES = {0: "BestOf3", 1: "BestOf5"}
REFEREE_CALL_NAMES = {
    0: "Slip", 1: "SlipEStop", 2: "Knockdown", 3: "BeatCount",
    4: "Knockout", 5: "DoubleKnockdown", 6: "DoubleKnockout",
}
FLOAT7 = struct.Struct("<7f")
HEX64 = re.compile(r"[0-9a-f]{64}")


class EvidenceError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_json_integer(text: str) -> int | float:
    # System.Text.Json serializes Single negative zero as the valid JSON integer
    # token -0. Python's default decoder turns that token into integer zero and
    # loses the IEEE-754 sign bit needed for byte-exact redundant-field checks.
    return -0.0 if text == "-0" else int(text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_int=_parse_json_integer)
            except json.JSONDecodeError as exc:
                raise EvidenceError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise EvidenceError(f"{path}:{line_number}: record is not an object")
            records.append(record)
    if not records:
        raise EvidenceError(f"{path}: no records")
    return records


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{label} is not an integer")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"{label} is below {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite")
    return result


def _float32_bytes(value: Any, label: str) -> bytes:
    number = _finite(value, label)
    try:
        return struct.pack("<f", number)
    except (OverflowError, struct.error) as exc:
        raise EvidenceError(f"{label} is outside finite float32 range") from exc


def _require_float32_wire(value: Any, wire: bytes, label: str) -> None:
    if _float32_bytes(value, label) != wire:
        raise EvidenceError(
            f"{label} does not round-trip to the authoritative wire float32"
        )


def _float_list(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise EvidenceError(f"{label} must contain exactly {length} values")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _require_hex64(value: Any, label: str) -> str:
    text = str(value).lower()
    if not HEX64.fullmatch(text):
        raise EvidenceError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _int32(value: Any, label: str) -> int:
    result = _integer(value, label)
    if not -(1 << 31) <= result < (1 << 31):
        raise EvidenceError(f"{label} is not int32")
    return result


def _sbyte(value: int) -> int:
    return value if value < 0x80 else value - 0x100


def _wire_body(record: dict[str, Any], expected_length: int, label: str) -> bytes:
    if _integer(record.get("wire_body_bytes"), f"{label} byte count") != expected_length:
        raise EvidenceError(f"{label} byte count is not {expected_length}")
    try:
        body = base64.b64decode(record.get("wire_body_base64", ""), validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise EvidenceError(f"{label} body is not valid base64") from exc
    if len(body) != expected_length:
        raise EvidenceError(f"{label} decoded body length is not {expected_length}")
    observed_hash = _require_hex64(record.get("wire_body_sha256"), f"{label} body hash")
    if hashlib.sha256(body).hexdigest() != observed_hash:
        raise EvidenceError(f"{label} body hash mismatch")
    return body


def _exact_decoded(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise EvidenceError(f"{label} fields disagree with the audited decoder")
        for key, value in expected.items():
            _exact_decoded(observed[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise EvidenceError(f"{label} length disagrees with the audited decoder")
        for index, value in enumerate(expected):
            _exact_decoded(observed[index], value, f"{label}[{index}]")
        return
    if expected is None:
        if observed is not None:
            raise EvidenceError(f"{label} must be null")
        return
    if isinstance(expected, bool):
        if observed is not expected:
            raise EvidenceError(f"{label} disagrees with the audited decoder")
        return
    if isinstance(expected, int):
        if _integer(observed, label) != expected:
            raise EvidenceError(f"{label} disagrees with the audited decoder")
        return
    if isinstance(expected, float):
        _require_float32_wire(observed, struct.pack("<f", expected), label)
        return
    if observed != expected:
        raise EvidenceError(f"{label} disagrees with the audited decoder")


def _contains_client_send_frame(value: Any) -> bool:
    if isinstance(value, str):
        return "ClientSendFrame" in value
    if isinstance(value, dict):
        return any(
            _contains_client_send_frame(key) or _contains_client_send_frame(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_client_send_frame(item) for item in value)
    return False


def _validate_observation_clock(record: dict[str, Any], label: str,
                                 realtime_field: str) -> float:
    _integer(record.get("client_fixed_tick_at_observation"), f"{label} client tick", 0)
    _integer(record.get("unity_frame"), f"{label} Unity frame", 0)
    return _finite(record.get(realtime_field), f"{label} {realtime_field}")


def _utc_timestamp(value: Any, label: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} UTC timestamp is absent")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = re.sub(r"(\.\d{6})\d(?=[+-]\d{2}:\d{2}$)", r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EvidenceError(f"{label} UTC timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceError(f"{label} timestamp is not explicitly UTC")
    return parsed.timestamp()


def _validate_stopwatch_clock(
    record: dict[str, Any], label: str, start_ticks: int, end_ticks: int
) -> int:
    _utc_timestamp(record.get("utc"), label)
    ticks = _integer(
        record.get("stopwatch_timestamp_ticks"), f"{label} Stopwatch ticks", 0
    )
    if ticks < start_ticks or ticks > end_ticks:
        raise EvidenceError(f"{label} Stopwatch ticks fall outside capture bounds")
    return ticks


def _validate_pairing(start: dict[str, Any]) -> None:
    pairing = start.get("pairing")
    if not isinstance(pairing, dict):
        raise EvidenceError("v6 capture has no measured fighter pairing")
    expected_signature = EXPECTED_BONE_PROTOCOL_V6[
        "t800_ordered_bone_signature_sha256"
    ]
    expected = {
        "required_pairing": "t800_vs_t800",
        "required_robot_id": "t800",
        "required_t800_bone_count": T800_BONE_COUNT,
        "required_t800_bone_signature_sha256": expected_signature,
        "semantic_identity_source": "FightCoordinator.fighterIdentities[slot].RobotID",
        "bone_signature_source": (
            "FightCoordinator.Fighters[slot].boneTransforms[index].name"
        ),
        "exact_t800_vs_t800": True,
        "reason": "exact_t800_vs_t800_pairing_proven",
    }
    for field, value in expected.items():
        if pairing.get(field) != value:
            raise EvidenceError(
                f"v6 pairing field {field} is {pairing.get(field)!r}, expected {value!r}"
            )
    for slot in (0, 1):
        fighter = pairing.get(f"fighter_{slot}")
        if not isinstance(fighter, dict):
            raise EvidenceError(f"v6 pairing fighter {slot} is absent")
        fighter_expected = {
            "semantic_robot_id": "t800",
            "semantic_t800": True,
            "bone_count": T800_BONE_COUNT,
            "ordered_bone_signature_sha256": expected_signature,
            "exact_t800_bone_signature": True,
        }
        for field, value in fighter_expected.items():
            if fighter.get(field) != value:
                raise EvidenceError(
                    f"v6 pairing fighter {slot} field {field} is not exact"
                )


def _validate_camera(camera: Any, label: str) -> dict[str, Any]:
    if not isinstance(camera, dict):
        raise EvidenceError(f"{label} camera record is absent")
    if camera.get("selection") != "UnityEngine.Camera.main":
        raise EvidenceError(f"{label} camera selection is not Camera.main")
    if camera.get("enabled") is not True or camera.get("active_in_hierarchy") is not True:
        raise EvidenceError(f"{label} main camera is not active and enabled")
    instance_id = _integer(camera.get("instance_id"), f"{label} camera instance id")
    if not isinstance(camera.get("name"), str) or not camera["name"]:
        raise EvidenceError(f"{label} camera name is absent")
    if not isinstance(camera.get("camera_type"), str) or not camera["camera_type"]:
        raise EvidenceError(f"{label} camera type is absent")

    _float_list(camera.get("world_position_xyz"), 3, f"{label} camera position")
    _float_list(
        camera.get("world_rotation_xyzw"), 4, f"{label} camera rotation"
    )
    for field in (
        "world_to_camera_matrix_row_major",
        "projection_matrix_row_major",
        "gpu_projection_matrix_row_major",
    ):
        _float_list(camera.get(field), 16, f"{label} {field}")
    viewport = _float_list(
        camera.get("normalized_viewport_rect_xywh"),
        4,
        f"{label} normalized viewport",
    )
    pixel_rect = _float_list(
        camera.get("pixel_rect_xywh"), 4, f"{label} pixel rect"
    )
    if viewport[2] <= 0.0 or viewport[3] <= 0.0:
        raise EvidenceError(f"{label} normalized viewport has no area")
    if pixel_rect[2] <= 0.0 or pixel_rect[3] <= 0.0:
        raise EvidenceError(f"{label} pixel rect has no area")
    for field in (
        "pixel_width", "pixel_height", "scaled_pixel_width", "scaled_pixel_height",
        "screen_width", "screen_height",
    ):
        if _integer(camera.get(field), f"{label} {field}", 1) <= 0:
            raise EvidenceError(f"{label} {field} is not positive")
    _integer(camera.get("target_display"), f"{label} target display", 0)
    if not isinstance(camera.get("orthographic"), bool):
        raise EvidenceError(f"{label} orthographic flag is not Boolean")
    for field in (
        "orthographic_size", "field_of_view_degrees", "aspect",
        "near_clip_plane", "far_clip_plane", "screen_dpi",
    ):
        _finite(camera.get(field), f"{label} {field}")
    if _finite(camera.get("aspect"), f"{label} aspect") <= 0.0:
        raise EvidenceError(f"{label} camera aspect is not positive")
    near = _finite(camera.get("near_clip_plane"), f"{label} near clip")
    far = _finite(camera.get("far_clip_plane"), f"{label} far clip")
    if near <= 0.0 or far <= near:
        raise EvidenceError(f"{label} camera clip planes are invalid")
    for field in ("allow_hdr", "allow_msaa", "render_into_texture"):
        if not isinstance(camera.get(field), bool):
            raise EvidenceError(f"{label} {field} is not Boolean")
    if not isinstance(camera.get("screen_full_screen_mode"), str):
        raise EvidenceError(f"{label} screen mode is absent")
    render_scale = _float_list(
        camera.get("render_scale_xy"), 2, f"{label} render scale"
    )
    if any(value <= 0.0 for value in render_scale):
        raise EvidenceError(f"{label} render scale is not positive")
    target_texture = camera.get("target_texture")
    if camera["render_into_texture"]:
        if not isinstance(target_texture, dict):
            raise EvidenceError(f"{label} render target metadata is absent")
        for field in ("width", "height", "anti_aliasing"):
            _integer(target_texture.get(field), f"{label} target texture {field}", 1)
    elif target_texture is not None:
        raise EvidenceError(f"{label} declares a target texture without rendering to it")
    return {
        "instance_id": instance_id,
        "pixel_rect": pixel_rect,
        "pixel_width": camera["pixel_width"],
        "pixel_height": camera["pixel_height"],
        "screen_width": camera["screen_width"],
        "screen_height": camera["screen_height"],
    }


def _validate_root_pose(
    root: Any, camera_summary: dict[str, Any], label: str
) -> None:
    if not isinstance(root, dict):
        raise EvidenceError(f"{label} root record is absent")
    _float_list(root.get("world_position_xyz"), 3, f"{label} world position")
    _float_list(root.get("world_rotation_xyzw"), 4, f"{label} world rotation")
    screen = _float_list(
        root.get("screen_position_xyz"), 3, f"{label} screen position"
    )
    expected_front = screen[2] > 0.0
    if root.get("screen_in_front_of_camera") is not expected_front:
        raise EvidenceError(f"{label} screen-front flag disagrees with measured z")
    rect = camera_summary["pixel_rect"]
    expected_inside = (
        expected_front
        and rect[0] <= screen[0] < rect[0] + rect[2]
        and rect[1] <= screen[1] < rect[1] + rect[3]
    )
    if root.get("screen_inside_camera_pixel_rect") is not expected_inside:
        raise EvidenceError(f"{label} screen-rect flag disagrees with coordinates")


def _validate_v6_root_stream(
    records: list[dict[str, Any]],
    start: dict[str, Any],
    end: dict[str, Any],
    scope: dict[str, Any],
    start_ticks: int,
    end_ticks: int,
) -> dict[str, Any]:
    root_samples = [
        record for record in records if record.get("event") == "root_pose_sample"
    ]
    end_tick = _integer(end.get("client_fixed_tick_at_end"), "capture end tick", 1)
    declared_count = _integer(
        end.get("root_pose_sample_count"), "capture_end root pose count", 1
    )
    if declared_count != len(root_samples):
        raise EvidenceError("capture_end root pose count disagrees with records")
    if len(root_samples) != end_tick:
        raise EvidenceError("v6 root stream does not cover every captured client fixed tick")
    indices = [record.get("root_pose_sample_index") for record in root_samples]
    ticks = [record.get("client_fixed_tick") for record in root_samples]
    expected_ticks = list(range(end_tick))
    if indices != expected_ticks or ticks != expected_ticks:
        raise EvidenceError("v6 root samples are not contiguous from fixed tick zero")

    initial_camera = _validate_camera(start.get("initial_camera"), "capture start")
    expected_scene = start.get("scene")
    if not isinstance(expected_scene, str) or not expected_scene:
        raise EvidenceError("v6 capture start scene is absent")
    previous_stopwatch: int | None = None
    previous_fixed_time: float | None = None
    expected_fight_epoch: int | None = None
    expected_round_number: int | None = None
    fixed_delta = _finite(start.get("fixed_delta_time"), "fixed delta time")
    for index, record in enumerate(root_samples):
        label = f"root pose sample {index}"
        stopwatch = _validate_stopwatch_clock(
            record, label, start_ticks, end_ticks
        )
        if previous_stopwatch is not None and stopwatch <= previous_stopwatch:
            raise EvidenceError("v6 root sample Stopwatch ticks are not strictly increasing")
        previous_stopwatch = stopwatch
        _integer(record.get("unity_frame"), f"{label} Unity frame", 0)
        _finite(record.get("unity_time"), f"{label} Unity time")
        fixed_time = _finite(record.get("unity_fixed_time"), f"{label} fixed time")
        _finite(record.get("unity_unscaled_time"), f"{label} unscaled time")
        if previous_fixed_time is not None and not math.isclose(
            fixed_time - previous_fixed_time,
            fixed_delta,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise EvidenceError("v6 root sample fixed-time cadence is not 500 Hz")
        previous_fixed_time = fixed_time
        if record.get("scene") != expected_scene:
            raise EvidenceError(f"{label} scene changed")
        if record.get("local_fighter_index") != scope.get("local_fighter_index"):
            raise EvidenceError(f"{label} local fighter slot changed")
        if record.get("opponent_slot") != scope.get("opponent_slot"):
            raise EvidenceError(f"{label} opponent slot changed")
        fight_epoch = _integer(record.get("fight_epoch"), f"{label} fight epoch", 0)
        round_number = _integer(record.get("round_number"), f"{label} round number", 0)
        if expected_fight_epoch is None:
            expected_fight_epoch = fight_epoch
            expected_round_number = round_number
        elif fight_epoch != expected_fight_epoch or round_number != expected_round_number:
            raise EvidenceError(f"{label} fight or round identity changed")
        camera_summary = _validate_camera(record.get("camera"), label)
        if camera_summary["instance_id"] != initial_camera["instance_id"]:
            raise EvidenceError(f"{label} Camera.main instance changed")
        for field in (
            "pixel_rect", "pixel_width", "pixel_height", "screen_width", "screen_height"
        ):
            if camera_summary[field] != initial_camera[field]:
                raise EvidenceError(f"{label} camera render geometry changed")
        _validate_root_pose(record.get("fighter_0_root"), camera_summary, f"{label} fighter 0")
        _validate_root_pose(record.get("fighter_1_root"), camera_summary, f"{label} fighter 1")
    return {
        "samples": len(root_samples),
        "first_stopwatch_tick": root_samples[0]["stopwatch_timestamp_ticks"],
        "last_stopwatch_tick": root_samples[-1]["stopwatch_timestamp_ticks"],
        "stopwatch_frequency_hz": start["stopwatch_frequency_hz"],
        "camera_instance_id": initial_camera["instance_id"],
        "pixel_width": initial_camera["pixel_width"],
        "pixel_height": initial_camera["pixel_height"],
    }


def _validate_request_epistemics(record: dict[str, Any], label: str) -> None:
    for field in ("server_tick", "server_acceptance"):
        if field not in record or record[field] is not None:
            raise EvidenceError(f"{label} {field} must remain explicitly null")
    if record.get("ack_observed") is not False:
        raise EvidenceError(f"{label} must not claim an acknowledgement")
    if record.get("request_only") is not True:
        raise EvidenceError(f"{label} must remain request-only")


def _validate_request_projection(record: dict[str, Any], label: str) -> str:
    message = record.get("message")
    _validate_observation_clock(record, label, "unity_realtime_since_startup")
    _validate_request_epistemics(record, label)
    if message == "REK_Input":
        body = _wire_body(record, 13, label)
        source_index = _int32(record.get("network_index_source_int32"),
                              f"{label} source network index")
        wire_index = _integer(record.get("network_index_wire_uint8"),
                              f"{label} wire network index")
        if not 0 <= wire_index <= 255 or wire_index != (source_index & 0xFF):
            raise EvidenceError(f"{label} network-index uint8 projection mismatch")
        velocity = _float_list(record.get("velocity_command_xyz"), 3,
                               f"{label} velocity")
        body_bits = struct.unpack("<3I", body[1:])
        bit_patterns = [f"0x{value:08x}" for value in body_bits]
        if record.get("velocity_float32_bit_patterns") != bit_patterns:
            raise EvidenceError(f"{label} velocity bit patterns disagree with body")
        for index, value in enumerate(velocity):
            _require_float32_wire(
                value,
                body[1 + 4 * index:1 + 4 * (index + 1)],
                f"{label} velocity[{index}]",
            )
        if record.get("wire_delivery") != "unreliable":
            raise EvidenceError(f"{label} REK_Input delivery semantics drifted")
        if record.get("native_method") != (
            "REKApp.RobotInputController.SendVelocityCommand RVA 0x226F110"
        ):
            raise EvidenceError(f"{label} REK_Input native method mismatch")
        if record.get("provenance") != (
            "exact packed request projection from source fields at "
            "REKApp.RobotInputController.SendVelocityCommand prefix"
        ):
            raise EvidenceError(f"{label} REK_Input provenance mismatch")
        return "SendVelocityCommand"
    if message == "REK_Move":
        body = _wire_body(record, 2, label)
        source_index = _int32(record.get("network_index_source_int32"),
                              f"{label} source network index")
        wire_index = _integer(record.get("network_index_wire_uint8"),
                              f"{label} wire network index")
        source_move = _int32(record.get("move_index_source_int32"),
                             f"{label} source move index")
        wire_move = _integer(record.get("move_index_wire_uint8"),
                             f"{label} wire move index")
        if not 0 <= wire_index <= 255 or wire_index != (source_index & 0xFF):
            raise EvidenceError(f"{label} network-index uint8 projection mismatch")
        if not 0 <= wire_move <= 255 or wire_move != (source_move & 0xFF):
            raise EvidenceError(f"{label} move-index uint8 projection mismatch")
        if body != bytes((wire_index, wire_move)):
            raise EvidenceError(f"{label} REK_Move body disagrees with source fields")
        if record.get("wire_delivery") != "reliable":
            raise EvidenceError(f"{label} REK_Move delivery semantics drifted")
        if record.get("native_method") != (
            "REKApp.RobotInputController.SendMoveEvent RVA 0x226ECB0"
        ):
            raise EvidenceError(f"{label} REK_Move native method mismatch")
        if record.get("provenance") != (
            "exact packed request projection from source fields at "
            "REKApp.RobotInputController.SendMoveEvent prefix"
        ):
            raise EvidenceError(f"{label} REK_Move provenance mismatch")
        return "SendMoveEvent"
    raise EvidenceError(f"{label} has unsupported projected message {message!r}")


def _validate_invocation_only(record: dict[str, Any], label: str) -> str:
    _validate_observation_clock(record, label, "unity_realtime_since_startup")
    _validate_request_epistemics(record, label)
    method = record.get("method")
    messages = {"SendSpecialEvent": "REK_Special", "SendEStopToggle": "REK_EStop"}
    if method not in messages or record.get("message") != messages[method]:
        raise EvidenceError(f"{label} is not an audited invocation-only method")
    for field in ("wire_body_bytes", "wire_body_sha256", "wire_body_base64",
                  "wire_delivery"):
        if field not in record or record[field] is not None:
            raise EvidenceError(f"{label} {field} must remain explicitly null")
    if record.get("provenance") != (
        f"REKApp.RobotInputController.{method} prefix invocation observation"
    ):
        raise EvidenceError(f"{label} provenance mismatch")
    return method


def _match_bone_layout(value: Any, label: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceError(f"{label} is not an ordered bone-name list")
    names = tuple(value)
    for layout_id, layout in EXPECTED_BONE_LAYOUTS.items():
        if names == layout["bone_names"]:
            return layout_id, layout
    raise EvidenceError(f"{label} does not match a pinned bone layout")


def _decode_wire_body(
    record: dict[str, Any], bone_layouts: dict[int, tuple[str, dict[str, Any]]]
) -> None:
    sequence = _integer(
        record.get("raw_bone_packet_sequence"), "raw packet sequence", 1
    )
    slot = _integer(record.get("fighter_slot"), f"raw packet {sequence} fighter slot")
    if slot not in (0, 1):
        raise EvidenceError(f"raw packet {sequence} has invalid fighter slot {slot}")
    network_index = _integer(
        record.get("network_index"), f"raw packet {sequence} network index"
    )
    if not 0 <= network_index <= 255:
        raise EvidenceError(f"raw packet {sequence} network index is not uint8")
    layout_id, layout = bone_layouts[slot]
    bone_count = _integer(
        record.get("bone_count"), f"raw packet {sequence} bone count", 1
    )
    if bone_count != layout["bone_count"]:
        raise EvidenceError(
            f"raw packet {sequence} has {bone_count} bones; fighter {slot} "
            f"declares {layout_id} with {layout['bone_count']}"
        )
    body = _wire_body(
        record, layout["body_bytes"], f"raw bone packet {sequence}"
    )
    if body[0] != network_index or body[1] != bone_count:
        raise EvidenceError(f"raw packet {sequence} binary header disagrees with JSON")

    offset = 2
    for _ in range(bone_count):
        values = FLOAT7.unpack_from(body, offset)
        if not all(math.isfinite(value) for value in values):
            raise EvidenceError(f"raw packet {sequence} binary body has non-finite float")
        offset += FLOAT7.size
    recorded_positions = _float_list(
        record.get("world_positions_xyz"), bone_count * 3,
        f"raw packet {sequence} world positions",
    )
    recorded_rotations = _float_list(
        record.get("world_rotations_xyzw"), bone_count * 4,
        f"raw packet {sequence} world rotations",
    )
    reconstructed = bytearray((network_index, bone_count))
    try:
        for index in range(bone_count):
            reconstructed.extend(FLOAT7.pack(
                *recorded_positions[index * 3:(index + 1) * 3],
                *recorded_rotations[index * 4:(index + 1) * 4],
            ))
    except (OverflowError, struct.error) as exc:
        raise EvidenceError(
            f"raw packet {sequence} decoded transforms are not float32"
        ) from exc
    if bytes(reconstructed) != body:
        raise EvidenceError(f"raw packet {sequence} decoded transforms disagree with body")
    if record.get("bone_names") != list(layout["bone_names"]):
        raise EvidenceError(
            f"raw packet {sequence} bone names disagree with fighter {slot} "
            f"{layout_id} capture header"
        )
    if _finite(record.get("intended_wire_interval_seconds"), "wire interval") != 0.02:
        raise EvidenceError(f"raw packet {sequence} intended interval drifted")
    if _finite(record.get("intended_wire_rate_hz"), "wire rate") != 50.0:
        raise EvidenceError(f"raw packet {sequence} intended rate drifted")
    if record.get("wire_delivery") != "unreliable":
        raise EvidenceError(f"raw packet {sequence} delivery semantics drifted")
    if record.get("provenance") != EXPECTED_PROVENANCE:
        raise EvidenceError(f"raw packet {sequence} provenance mismatch")
    for field in (
        "client_fixed_tick_at_observation", "unity_frame",
    ):
        _integer(record.get(field), f"raw packet {sequence} {field}", 0)
    for field in (
        "unity_time", "unity_unscaled_time", "monotonic_receipt_time",
    ):
        _finite(record.get(field), f"raw packet {sequence} {field}")


def _validate_raw_protocol_record(record: dict[str, Any]) -> tuple[str, int, float]:
    event = record.get("event")
    protocol_sequence = _integer(
        record.get("raw_protocol_sequence"), f"{event} protocol sequence", 1
    )
    label = f"{event} protocol packet {protocol_sequence}"
    _validate_observation_clock(record, label, "monotonic_receipt_time")
    _finite(record.get("unity_time"), f"{label} unity_time")
    _finite(record.get("unity_unscaled_time"), f"{label} unity_unscaled_time")
    receipt = _finite(record.get("monotonic_receipt_time"), f"{label} receipt time")

    if event == "raw_fight_state_packet":
        type_sequence = _integer(
            record.get("raw_fight_state_sequence"), f"{label} fight-state sequence", 1
        )
        body = _wire_body(record, 33, label)
        time_remaining = struct.unpack_from("<f", body, 0x04)[0]
        if not math.isfinite(time_remaining):
            raise EvidenceError(f"{label} time_remaining is not finite")
        expected = {
            "phase": body[0x00],
            "phase_name": FIGHT_PHASE_NAMES.get(body[0x00]),
            "round_number": body[0x01],
            "round_active": body[0x02],
            "is_redo": body[0x03],
            "time_remaining": time_remaining,
            "hits_0": struct.unpack_from("<h", body, 0x08)[0],
            "hits_1": struct.unpack_from("<h", body, 0x0A)[0],
            "knockout_occurred": body[0x0C],
            "round_result": body[0x0D],
            "round_result_name": ROUND_RESULT_NAMES.get(body[0x0D]),
            "round_winner": _sbyte(body[0x0E]),
            "rounds_won_0": body[0x0F],
            "rounds_won_1": body[0x10],
            "fight_result": body[0x11],
            "fight_result_name": FIGHT_RESULT_NAMES.get(body[0x11]),
            "fight_winner": _sbyte(body[0x12]),
            "format": body[0x13],
            "format_name": FIGHT_FORMAT_NAMES.get(body[0x13]),
            "human_slot_mask": body[0x14],
            "champion_slot": _sbyte(body[0x15]),
            "fault_mask": body[0x16],
            "fault_stress_0": body[0x17],
            "fault_stress_1": body[0x18],
            "referee_count_mask": body[0x19],
            "referee_count_seconds": body[0x1A],
            "referee_call_sequence": body[0x1B],
            "referee_call_type": body[0x1C],
            "referee_call_name": REFEREE_CALL_NAMES.get(body[0x1C]),
            "referee_call_faller": _sbyte(body[0x1D]),
            "referee_call_points": body[0x1E],
            "ai_level": body[0x1F],
            "decided_winner_bits": body[0x20],
        }
        _exact_decoded(record.get("decoded"), expected, f"{label} decoded")
        if record.get("wire_delivery") != "reliable":
            raise EvidenceError(f"{label} delivery semantics drifted")
        if _finite(record.get("nominal_wire_interval_seconds"),
                   f"{label} nominal interval") != 0.1:
            raise EvidenceError(f"{label} nominal interval drifted")
        if record.get("native_sender") != (
            "REKApp.FightCoordinator.ServerSendFightState RVA 0x238BFA0"
        ) or record.get("native_receiver") != (
            "REKApp.FightCoordinator.ApplyFightStateSnapshot RVA 0x2379E00"
        ):
            raise EvidenceError(f"{label} native method identity mismatch")
        if record.get("provenance") != EXPECTED_FIGHT_STATE_PROVENANCE:
            raise EvidenceError(f"{label} provenance mismatch")
        return "fight_state", type_sequence, receipt

    if event == "raw_score_packet":
        type_sequence = _integer(
            record.get("raw_score_sequence"), f"{label} score sequence", 1
        )
        body = _wire_body(record, 7, label)
        points = struct.unpack_from("<f", body, 0x03)[0]
        if not math.isfinite(points):
            raise EvidenceError(f"{label} points_awarded is not finite")
        expected = {
            "fighter_index": body[0x00],
            "new_hit_count": struct.unpack_from("<h", body, 0x01)[0],
            "points_awarded": points,
        }
        _exact_decoded(record.get("decoded"), expected, f"{label} decoded")
        if record.get("wire_delivery") != "reliable":
            raise EvidenceError(f"{label} delivery semantics drifted")
        if record.get("native_sender") != (
            "REKApp.FightCoordinator.OnPointScoredNetwork RVA 0x23867D0"
        ) or record.get("native_receiver") != (
            "REKApp.FightCoordinator.OnScoreReceived RVA 0x2387010"
        ):
            raise EvidenceError(f"{label} native method identity mismatch")
        if record.get("provenance") != EXPECTED_SCORE_PROVENANCE:
            raise EvidenceError(f"{label} provenance mismatch")
        return "score", type_sequence, receipt

    if event == "raw_hit_packet":
        type_sequence = _integer(
            record.get("raw_hit_sequence"), f"{label} hit sequence", 1
        )
        body = _wire_body(record, 29, label)
        values = list(struct.unpack_from("<7f", body, 0x00))
        if not all(math.isfinite(value) for value in values):
            raise EvidenceError(f"{label} has a non-finite decoded float")
        expected = {
            "position_xyz": values[0:3],
            "surface_normal_xyz": values[3:6],
            "relative_speed": values[6],
            "is_kick": body[0x1C],
        }
        _exact_decoded(record.get("decoded"), expected, f"{label} decoded")
        if record.get("wire_delivery") != "unreliable":
            raise EvidenceError(f"{label} delivery semantics drifted")
        if record.get("native_sender") != (
            "REKApp.FightCoordinator.OnHitDetectedNetwork RVA 0x2385500"
        ) or record.get("native_receiver") != (
            "REKApp.FightCoordinator.OnHitReceived RVA 0x2385810"
        ):
            raise EvidenceError(f"{label} native method identity mismatch")
        if record.get("provenance") != EXPECTED_HIT_PROVENANCE:
            raise EvidenceError(f"{label} provenance mismatch")
        return "hit", type_sequence, receipt

    raise EvidenceError(f"unsupported raw protocol event {event!r}")


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None,
                "p95": None, "maximum": None}
    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "maximum": max(values),
    }


def validate(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.name.endswith(".partial"):
        raise EvidenceError("refusing an unfinalized .partial capture")
    records = _read_jsonl(path)
    if _contains_client_send_frame(records):
        raise EvidenceError("capture declares or records the obsolete ClientSendFrame surface")
    starts = [record for record in records if record.get("event") == "capture_start"]
    ends = [record for record in records if record.get("event") == "capture_end"]
    if len(starts) != 1 or records[0] is not starts[0]:
        raise EvidenceError("capture must begin with exactly one capture_start")
    if len(ends) != 1 or records[-1] is not ends[0]:
        raise EvidenceError("capture must end with exactly one capture_end")
    start = starts[0]
    end = ends[0]

    recorder_schema = start.get("schema")
    if recorder_schema not in {SCHEMA_V5, SCHEMA_V6}:
        raise EvidenceError(f"unsupported recorder schema {recorder_schema!r}")
    is_v6 = recorder_schema == SCHEMA_V6
    expected_plugin_version = (
        EXPECTED_PLUGIN_VERSION_V6 if is_v6 else EXPECTED_PLUGIN_VERSION_V5
    )
    expected_plugin_sha256 = (
        EXPECTED_PLUGIN_SHA256_V6 if is_v6 else EXPECTED_PLUGIN_SHA256_V5
    )
    if start.get("plugin_version") != expected_plugin_version:
        raise EvidenceError("recorder plugin version mismatch")
    if _require_hex64(start.get("plugin_sha256"), "plugin hash") != expected_plugin_sha256:
        raise EvidenceError("recorder plugin hash mismatch")
    if str(start.get("game_assembly_sha256", "")).lower() != EXPECTED_GAME_ASSEMBLY_SHA256:
        raise EvidenceError("GameAssembly hash mismatch")
    if str(start.get("global_metadata_sha256", "")).lower() != EXPECTED_METADATA_SHA256:
        raise EvidenceError("global metadata hash mismatch")
    if start.get("tick_level_claim") is not False:
        raise EvidenceError("compact client samples must not claim tick completeness")
    if start.get("tick_domain") != "client_fixed_update":
        raise EvidenceError("client tick domain is not declared")
    if start.get("server_tick_available") is not False:
        raise EvidenceError("capture must preserve absent server tick")
    if not str(start.get("server_tick_reason", "")).strip():
        raise EvidenceError("absent server tick has no reason")
    stride = _integer(start.get("client_sample_stride_ticks"), "sample stride", 1)
    if stride != 10:
        raise EvidenceError("compact client sample stride is not ten fixed substeps")

    start_ticks = 0
    end_ticks = (1 << 63) - 1
    if is_v6:
        start_ticks = _integer(
            start.get("stopwatch_timestamp_ticks"), "capture start Stopwatch ticks", 0
        )
        _utc_timestamp(start.get("utc"), "capture start")
        frequency = _integer(
            start.get("stopwatch_frequency_hz"), "Stopwatch frequency", 1
        )
        if frequency <= 0 or start.get("stopwatch_is_high_resolution") is not True:
            raise EvidenceError("v6 Stopwatch clock is not high resolution")
        if start.get("stopwatch_clock_semantics") != (
            "System.Diagnostics.Stopwatch.GetTimestamp; QueryPerformanceCounter-backed "
            "on Windows when Stopwatch.IsHighResolution is true"
        ):
            raise EvidenceError("v6 Stopwatch clock semantics are not pinned")
        end_ticks = _integer(
            end.get("stopwatch_timestamp_ticks"), "capture end Stopwatch ticks", 0
        )
        _utc_timestamp(end.get("utc"), "capture end")
        if end_ticks <= start_ticks:
            raise EvidenceError("capture end Stopwatch tick does not follow capture start")
        if _integer(
            start.get("root_pose_sample_stride_ticks"), "root pose sample stride", 1
        ) != 1:
            raise EvidenceError("v6 root pose sample stride is not one fixed substep")
        if _integer(
            start.get("root_pose_sample_rate_hz"), "root pose sample rate", 1
        ) != 500:
            raise EvidenceError("v6 root pose sample rate is not 500 Hz")
        if start.get("root_pose_tick_level_claim") is not True:
            raise EvidenceError("v6 root pose stream does not declare tick completeness")
        expected_root_metadata = {
            "root_pose_fields": (
                "world root position/rotation plus Camera.WorldToScreenPoint only; "
                "no inferred joints, velocities, contacts, or server state"
            ),
            "root_screen_coordinate_semantics": (
                "Unity Camera.WorldToScreenPoint pixels; origin bottom-left; "
                "z is world-unit distance from camera plane"
            ),
            "camera_selection_semantics": (
                "UnityEngine.Camera.main; capture is denied when absent, inactive, "
                "or without a positive pixel extent"
            ),
            "camera_matrix_semantics": (
                "16 float values in row-major m[row,column] order; world_to_camera "
                "is Unity view matrix; gpu_projection uses GL.GetGPUProjectionMatrix "
                "with render_into_texture"
            ),
        }
        for field, value in expected_root_metadata.items():
            if start.get(field) != value:
                raise EvidenceError(f"v6 {field} metadata is not pinned")
        if not math.isclose(
            _finite(start.get("fixed_delta_time"), "fixed delta time"),
            0.002,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise EvidenceError("v6 fixed delta time is not 0.002 seconds")

    expected_bone_protocol = (
        EXPECTED_BONE_PROTOCOL_V6 if is_v6 else EXPECTED_BONE_PROTOCOL
    )
    _exact_decoded(start.get("bone_wire_protocol"), expected_bone_protocol,
                   "capture bone protocol declaration")
    _exact_decoded(start.get("fight_wire_protocol"), EXPECTED_FIGHT_PROTOCOL,
                   "capture fight protocol declaration")
    _exact_decoded(start.get("outbound_request_protocol"), EXPECTED_OUTBOUND_PROTOCOL,
                   "capture outbound protocol declaration")
    if start.get("instrumentation_hooks") != EXPECTED_HOOKS:
        raise EvidenceError("capture instrumentation-hook declaration mismatch")
    if is_v6:
        _validate_pairing(start)

    server = start.get("server") or {}
    if not server.get("endpoint"):
        raise EvidenceError("capture has no server endpoint identity")
    forbidden_server_fields = {"session_id", "session_token", "arena_id"}
    if forbidden_server_fields.intersection(key.lower() for key in server):
        raise EvidenceError("capture persisted a raw session identifier")
    if server.get("session_identifier_recorded") is not False:
        raise EvidenceError("capture did not declare raw session identifier omission")
    _require_hex64(server.get("session_id_sha256"), "hashed session identity")

    target_status = start.get("harmony_target_status") or {}
    required_targets = {
        "REKApp.RobotInputController.SendVelocityCommand",
        "REKApp.RobotInputController.SendMoveEvent",
        "REKApp.RobotInputController.SendSpecialEvent",
        "REKApp.RobotInputController.SendEStopToggle",
        "REKApp.FightCoordinator.ApplyFightStateSnapshot",
        "REKApp.FightCoordinator.OnScoreReceived",
        "REKApp.FightCoordinator.OnHitReceived",
        "REKApp.Robot.OnBoneMessageReceived",
    }
    unverified = sorted(
        target for target in required_targets if target_status.get(target) is not True
    )
    if unverified:
        raise EvidenceError(f"Harmony ownership was not verified for {unverified}")

    scope = start.get("scope") or {}
    expected_scope = {
        "allowed": True,
        "network_connected": True,
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
    if is_v6:
        expected_scope.update({
            "context_is_solo": True,
            "context_is_ranked": False,
            "context_auto_find_match": False,
            "arena_id_present": True,
            "multiplayer_session_privacy_known": True,
            "multiplayer_session_is_private": True,
            "coordinator_is_ranked_arena": False,
            "client_ai_difficulty": 0,
            "exact_t800_vs_t800": True,
        })
    wrong_scope = {
        key: {"expected": value, "observed": scope.get(key)}
        for key, value in expected_scope.items() if scope.get(key) != value
    }
    if wrong_scope:
        raise EvidenceError(f"capture is outside private Bot 1 scope: {wrong_scope}")

    request_records = [
        record for record in records
        if record.get("event") in {
            "outbound_request_projection", "client_transport_method_invoked"
        }
    ]
    request_sequences = [
        _integer(record.get("request_sequence"), "outbound request sequence", 1)
        for record in request_records
    ]
    if request_sequences != list(range(1, len(request_records) + 1)):
        raise EvidenceError("outbound request sequence is not contiguous from one")
    request_counts = {
        "SendVelocityCommand": 0,
        "SendMoveEvent": 0,
        "SendSpecialEvent": 0,
        "SendEStopToggle": 0,
    }
    message_counts = {"REK_Input": 0, "REK_Move": 0}
    previous_request_time: float | None = None
    previous_request_stopwatch: int | None = None
    for record in request_records:
        sequence = record["request_sequence"]
        label = f"outbound request {sequence}"
        if record.get("event") == "outbound_request_projection":
            method = _validate_request_projection(record, label)
            message = record["message"]
            message_counts[message] += 1
            observed_method_sequence = _integer(
                record.get("message_request_sequence"),
                f"{label} message request sequence", 1,
            )
            if observed_method_sequence != message_counts[message]:
                raise EvidenceError(f"{label} message request sequence is not contiguous")
        else:
            method = _validate_invocation_only(record, label)
            observed_method_sequence = _integer(
                record.get("method_request_sequence"),
                f"{label} method request sequence", 1,
            )
            if observed_method_sequence != request_counts[method] + 1:
                raise EvidenceError(f"{label} method request sequence is not contiguous")
        request_counts[method] += 1
        request_time = _finite(
            record.get("unity_realtime_since_startup"), f"{label} realtime"
        )
        if previous_request_time is not None and request_time < previous_request_time:
            raise EvidenceError("outbound request realtime decreased")
        previous_request_time = request_time
        if is_v6:
            request_stopwatch = _validate_stopwatch_clock(
                record, label, start_ticks, end_ticks
            )
            if (previous_request_stopwatch is not None
                    and request_stopwatch < previous_request_stopwatch):
                raise EvidenceError("outbound request Stopwatch ticks decreased")
            previous_request_stopwatch = request_stopwatch

    bone_layouts = {
        slot: _match_bone_layout(
            start.get(f"fighter_{slot}_bones"), f"fighter {slot} capture header"
        )
        for slot in (0, 1)
    }
    if is_v6 and any(layout_id != "t800_26" for layout_id, _ in bone_layouts.values()):
        raise EvidenceError("v6 capture is not an exact T800-vs-T800 bone pairing")

    samples = [record for record in records if record.get("event") == "sample"]
    if len(samples) < 2:
        raise EvidenceError("capture has fewer than two compact client samples")
    expected_sample_indices = list(range(len(samples)))
    sample_indices = [record.get("sample_index") for record in samples]
    if sample_indices != expected_sample_indices:
        raise EvidenceError("compact sample indices are not contiguous from zero")
    expected_client_ticks = [index * stride for index in expected_sample_indices]
    observed_client_ticks = [record.get("client_fixed_tick") for record in samples]
    if observed_client_ticks != expected_client_ticks:
        raise EvidenceError("compact samples do not follow the declared client tick stride")
    if is_v6:
        previous_sample_stopwatch: int | None = None
        for index, record in enumerate(samples):
            sample_stopwatch = _validate_stopwatch_clock(
                record, f"compact sample {index}", start_ticks, end_ticks
            )
            if (previous_sample_stopwatch is not None
                    and sample_stopwatch <= previous_sample_stopwatch):
                raise EvidenceError(
                    "v6 compact sample Stopwatch ticks are not strictly increasing"
                )
            previous_sample_stopwatch = sample_stopwatch

        root_stream = _validate_v6_root_stream(
            records, start, end, scope, start_ticks, end_ticks
        )
    else:
        root_stream = None

    raw_packets = [
        record for record in records if record.get("event") == "raw_bone_packet"
    ]
    if not raw_packets:
        raise EvidenceError("capture has no raw REK_Bones packets")
    raw_sequences = [record.get("raw_bone_packet_sequence") for record in raw_packets]
    if raw_sequences != list(range(1, len(raw_packets) + 1)):
        raise EvidenceError("raw packet sequence is not contiguous from one")
    previous_receipt: float | None = None
    receipt_by_slot: dict[int, list[float]] = {0: [], 1: []}
    raw_by_sequence: dict[int, dict[str, Any]] = {}
    for record in raw_packets:
        _decode_wire_body(record, bone_layouts)
        sequence = record["raw_bone_packet_sequence"]
        receipt = _finite(record["monotonic_receipt_time"], "monotonic receipt time")
        if previous_receipt is not None and receipt < previous_receipt:
            raise EvidenceError("raw packet monotonic receipt time decreased")
        previous_receipt = receipt
        receipt_by_slot[record["fighter_slot"]].append(receipt)
        raw_by_sequence[sequence] = record
    if any(not values for values in receipt_by_slot.values()):
        raise EvidenceError("raw capture does not contain both fighters")

    decoded = [
        record for record in records if record.get("event") == "decoded_bone_snapshot"
    ]
    decoded_sequences = [record.get("bone_snapshot_sequence") for record in decoded]
    if decoded_sequences != list(range(1, len(decoded) + 1)):
        raise EvidenceError("decoded snapshot sequence is not contiguous from one")
    correlated_raw = []
    for record in decoded:
        raw_sequence = _integer(
            record.get("raw_bone_packet_sequence"),
            "decoded snapshot raw packet sequence", 1,
        )
        raw = raw_by_sequence.get(raw_sequence)
        if raw is None:
            raise EvidenceError("decoded snapshot cites an absent raw packet")
        if record.get("fighter_slot") != raw.get("fighter_slot"):
            raise EvidenceError("decoded snapshot fighter disagrees with raw packet")
        if record.get("network_index") != raw.get("network_index"):
            raise EvidenceError("decoded snapshot network index disagrees with raw packet")
        slot = raw["fighter_slot"]
        layout_id, layout = bone_layouts[slot]
        if record.get("bone_names") != list(layout["bone_names"]):
            raise EvidenceError(
                f"decoded snapshot bone names disagree with fighter {slot} "
                f"{layout_id} capture header"
            )
        _float_list(record.get("root_world_position"), 3, "decoded root position")
        _float_list(record.get("root_world_rotation_xyzw"), 4, "decoded root rotation")
        _float_list(
            record.get("child_local_rotations_xyzw"), layout["bone_count"] * 4,
            "decoded child local rotations",
        )
        correlated_raw.append(raw_sequence)
    if sorted(correlated_raw) != list(range(1, len(raw_packets) + 1)):
        raise EvidenceError("raw packets and decoded snapshots are not one-to-one")

    raw_protocol_records = [
        record for record in records if record.get("event") in {
            "raw_fight_state_packet", "raw_score_packet", "raw_hit_packet"
        }
    ]
    raw_protocol_sequences = [
        _integer(record.get("raw_protocol_sequence"), "raw protocol sequence", 1)
        for record in raw_protocol_records
    ]
    if raw_protocol_sequences != list(range(1, len(raw_protocol_records) + 1)):
        raise EvidenceError("raw fight-protocol sequence is not contiguous from one")
    raw_type_sequences: dict[str, list[int]] = {
        "fight_state": [], "score": [], "hit": [],
    }
    raw_fight_by_protocol_sequence: dict[int, dict[str, Any]] = {}
    previous_protocol_receipt: float | None = None
    for record in raw_protocol_records:
        kind, type_sequence, receipt = _validate_raw_protocol_record(record)
        raw_type_sequences[kind].append(type_sequence)
        if previous_protocol_receipt is not None and receipt < previous_protocol_receipt:
            raise EvidenceError("raw fight-protocol receipt time decreased")
        previous_protocol_receipt = receipt
        if kind == "fight_state":
            raw_fight_by_protocol_sequence[record["raw_protocol_sequence"]] = record
    for kind, sequences in raw_type_sequences.items():
        if sequences != list(range(1, len(sequences) + 1)):
            raise EvidenceError(f"raw {kind} sequence is not contiguous from one")
    if not raw_type_sequences["fight_state"]:
        raise EvidenceError("capture has no raw REK_FightState packets")

    fight_snapshots = [
        record for record in records
        if record.get("event") == "fight_state_snapshot_applied"
    ]
    fight_snapshot_sequences = [
        _integer(record.get("fight_state_snapshot_sequence"),
                 "fight-state snapshot sequence", 1)
        for record in fight_snapshots
    ]
    if fight_snapshot_sequences != list(range(1, len(fight_snapshots) + 1)):
        raise EvidenceError("fight-state snapshot sequence is not contiguous from one")
    correlated_fight_packets: list[int] = []
    for record in fight_snapshots:
        snapshot_sequence = record["fight_state_snapshot_sequence"]
        label = f"fight-state snapshot {snapshot_sequence}"
        raw_sequence = _integer(
            record.get("raw_protocol_sequence"), f"{label} raw protocol sequence", 1
        )
        raw = raw_fight_by_protocol_sequence.get(raw_sequence)
        if raw is None:
            raise EvidenceError(f"{label} cites an absent raw fight-state packet")
        _validate_observation_clock(record, label, "unity_unscaled_time")
        phase_value = _integer(record.get("phase_value"), f"{label} phase value")
        if phase_value != raw["decoded"]["phase"]:
            raise EvidenceError(f"{label} phase disagrees with its raw packet")
        phase_name = FIGHT_PHASE_NAMES.get(phase_value)
        if phase_name is not None and record.get("phase") != phase_name:
            raise EvidenceError(f"{label} phase name disagrees with its raw packet")
        if record.get("provenance") != (
            "REKApp.FightCoordinator.ApplyFightStateSnapshot postfix"
        ):
            raise EvidenceError(f"{label} provenance mismatch")
        correlated_fight_packets.append(raw_sequence)
    if sorted(correlated_fight_packets) != sorted(raw_fight_by_protocol_sequence):
        raise EvidenceError("raw fight-state packets and applied snapshots are not one-to-one")

    declared = {
        "sample_count": len(samples),
        "raw_bone_packet_count": len(raw_packets),
        "decoded_bone_snapshot_count": len(decoded),
        "client_transport_invocation_count": len(request_records),
        "fight_state_snapshot_count": len(fight_snapshots),
        "raw_protocol_packet_count": len(raw_protocol_records),
        "raw_fight_state_packet_count": len(raw_type_sequences["fight_state"]),
        "raw_score_packet_count": len(raw_type_sequences["score"]),
        "raw_hit_packet_count": len(raw_type_sequences["hit"]),
    }
    if is_v6:
        declared["root_pose_sample_count"] = root_stream["samples"]
    for field, observed in declared.items():
        if _integer(end.get(field), f"capture_end {field}", 0) != observed:
            raise EvidenceError(f"capture_end {field} disagrees with records")
    observed_method_counts = end.get("client_transport_method_counts")
    if not isinstance(observed_method_counts, dict):
        raise EvidenceError("capture_end client transport method counts are absent")
    expected_method_counts = {
        method: count for method, count in request_counts.items() if count
    }
    if set(observed_method_counts) != set(expected_method_counts):
        raise EvidenceError("capture_end client transport method names disagree with records")
    for method, expected_count in expected_method_counts.items():
        if _integer(observed_method_counts.get(method),
                    f"capture_end {method} count", 1) != expected_count:
            raise EvidenceError(
                f"capture_end {method} count disagrees with records"
            )
    if _integer(end.get("capture_error_count"), "capture error count", 0) != 0:
        raise EvidenceError("capture contains recorder errors")
    end_tick = _integer(end.get("client_fixed_tick_at_end"), "capture end tick", 1)
    if end_tick <= expected_client_ticks[-1]:
        raise EvidenceError("capture end tick does not follow the last compact sample")

    per_fighter: dict[str, Any] = {}
    for slot, receipts in receipt_by_slot.items():
        intervals = [right - left for left, right in zip(receipts, receipts[1:])]
        layout_id, layout = bone_layouts[slot]
        per_fighter[str(slot)] = {
            "packets": len(receipts),
            "client_receipt_interval_seconds": _quantiles(intervals),
            "bone_layout": {
                "layout_id": layout_id,
                "bone_count": layout["bone_count"],
                "wire_body_bytes": layout["body_bytes"],
                "identity_claimed": True,
                "runtime_object_name": layout["runtime_object_name"],
                "identity_basis": (
                    "exact ordered bone-name layout mapped to the measured "
                    "scoped Robot GameObject.name in the pinned Windows runtime"
                ),
            },
        }
    return {
        "schema": 1,
        "raw_path": str(path.resolve()),
        "raw_sha256": _sha256(path),
        "recorder_schema": recorder_schema,
        "recorder_plugin_sha256": expected_plugin_sha256,
        "game_assembly_sha256": EXPECTED_GAME_ASSEMBLY_SHA256,
        "global_metadata_sha256": EXPECTED_METADATA_SHA256,
        "raw_bone_packets": len(raw_packets),
        "decoded_bone_snapshots": len(decoded),
        "outbound_requests": {
            "REK_Input": message_counts["REK_Input"],
            "REK_Move": message_counts["REK_Move"],
            "REK_Special_invocations": request_counts["SendSpecialEvent"],
            "REK_EStop_invocations": request_counts["SendEStopToggle"],
        },
        "fight_protocol": {
            "raw_protocol_packets": len(raw_protocol_records),
            "fight_state_packets": len(raw_type_sequences["fight_state"]),
            "fight_state_snapshots": len(fight_snapshots),
            "score_packets": len(raw_type_sequences["score"]),
            "hit_packets": len(raw_type_sequences["hit"]),
        },
        "compact_samples": len(samples),
        "client_sample_stride_ticks": stride,
        "root_pose_stream": root_stream,
        "fighters": per_fighter,
        "wire": {
            "intended_interval_seconds": 0.02,
            "intended_rate_hz": 50,
            "delivery": "unreliable",
            "body_size_formula_bytes": "2 + 28 * bone_count",
        },
        "server_identity": {
            "endpoint_present": True,
            "hashed_session_identity_present": True,
            "raw_session_identity_recorded": False,
        },
        "claims": {
            "server_tick_available": False,
            "server_send_timestamp_available": False,
            "server_move_acceptance_available": False,
            "request_to_snapshot_causality_established": False,
            "request_projection_is_send_completion": False,
            "rek_input_request_projection_validated": message_counts["REK_Input"] > 0,
            "rek_move_request_projection_validated": message_counts["REK_Move"] > 0,
            "raw_fight_protocol_payloads_validated": all(
                raw_type_sequences[kind] for kind in ("fight_state", "score", "hit")
            ),
            "raw_fight_state_payloads_validated": bool(
                raw_type_sequences["fight_state"]
            ),
            "raw_score_payloads_validated": bool(raw_type_sequences["score"]),
            "raw_hit_payloads_validated": bool(raw_type_sequences["hit"]),
            "raw_wire_pose_payload_validated": True,
            "exact_private_session_validated": is_v6,
            "exact_t800_vs_t800_validated": is_v6,
            "root_pose_500hz_validated": is_v6,
            "root_world_to_screen_validated": is_v6,
            "client_send_frame_observed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = validate(args.raw)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
