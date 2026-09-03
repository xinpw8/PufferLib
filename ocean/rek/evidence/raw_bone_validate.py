#!/usr/bin/env python3
"""Validate a passive REK protocol capture without inventing server semantics.

The v5 recorder projects exact REK_Input and REK_Move request bodies, copies
fight and bone FastBufferReader bodies before REK consumes them, and observes
the corresponding post-receive state. This validator checks those bytes and
their decoded fields. It does not infer send completion, a server tick, a
server timestamp, move acceptance, execution, or request-to-pose causality.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path
from typing import Any


SCHEMA = "rek.private_ai.protocol.v5"
EXPECTED_PLUGIN_VERSION = "0.5.1"
EXPECTED_PLUGIN_SHA256 = (
    "f9848c17a83ae011f046aa2baa1fdfd2377dfcfc6d287728265d2b28ea3ce0a2"
)
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

    if start.get("schema") != SCHEMA:
        raise EvidenceError(f"unsupported recorder schema {start.get('schema')!r}")
    if start.get("plugin_version") != EXPECTED_PLUGIN_VERSION:
        raise EvidenceError("recorder plugin version mismatch")
    if _require_hex64(start.get("plugin_sha256"), "plugin hash") != EXPECTED_PLUGIN_SHA256:
        raise EvidenceError("recorder plugin hash mismatch")
    if str(start.get("game_assembly_sha256", "")).lower() != EXPECTED_GAME_ASSEMBLY_SHA256:
        raise EvidenceError("GameAssembly hash mismatch")
    if str(start.get("global_metadata_sha256", "")).lower() != EXPECTED_METADATA_SHA256:
        raise EvidenceError("global metadata hash mismatch")
    if start.get("tick_level_claim") is not False:
        raise EvidenceError("v5 capture must not claim client samples are tick-complete")
    if start.get("tick_domain") != "client_fixed_update":
        raise EvidenceError("client tick domain is not declared")
    if start.get("server_tick_available") is not False:
        raise EvidenceError("v5 capture must preserve absent server tick")
    if not str(start.get("server_tick_reason", "")).strip():
        raise EvidenceError("absent server tick has no reason")
    stride = _integer(start.get("client_sample_stride_ticks"), "sample stride", 1)

    _exact_decoded(start.get("bone_wire_protocol"), EXPECTED_BONE_PROTOCOL,
                   "capture bone protocol declaration")
    _exact_decoded(start.get("fight_wire_protocol"), EXPECTED_FIGHT_PROTOCOL,
                   "capture fight protocol declaration")
    _exact_decoded(start.get("outbound_request_protocol"), EXPECTED_OUTBOUND_PROTOCOL,
                   "capture outbound protocol declaration")
    if start.get("instrumentation_hooks") != EXPECTED_HOOKS:
        raise EvidenceError("capture instrumentation-hook declaration mismatch")

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

    bone_layouts = {
        slot: _match_bone_layout(
            start.get(f"fighter_{slot}_bones"), f"fighter {slot} capture header"
        )
        for slot in (0, 1)
    }

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
        "recorder_schema": SCHEMA,
        "recorder_plugin_sha256": EXPECTED_PLUGIN_SHA256,
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
