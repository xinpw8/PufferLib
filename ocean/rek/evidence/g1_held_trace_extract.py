#!/usr/bin/env python3
"""Extract a fail-closed 50 Hz G1 held-input trace from recorder v7.

The recorder measures roots at 500 Hz and REK_Bones packets at their receive
boundary. Bone delivery is unreliable and need not be 50 Hz. This extractor
places exact root samples on a deterministic 50 Hz client-fixed grid and
attaches the latest received full bone packet for each fighter. Every attached
packet retains its source tick, age, and freshness. No interpolation is used.

REK_Input and REK_Move records prove client request projection only. The
coverage report consequently leaves attack acceptance unknown, even when a
post-request pose response is visible. Held W/S/A/D/Q/E conditions are labels
for the measured request vector and are never reconstructed from keyboard
events.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


RECORDER_SCHEMA = "rek.private_ai.protocol.v7"
TRACE_SCHEMA = "rek.g1_held_motion_trace.v1"
COVERAGE_SCHEMA = "rek.g1_held_motion_trace.coverage.v1"
EXPECTED_PLUGIN_VERSION = "0.7.2"
EXPECTED_PLUGIN_SHA256 = (
    "a19f619c83eeecf9c6ccf79adf339be1f7f1cca8e3cd622f80616f268aaffa95"
)
EXPECTED_RECOVERY_PLUGIN_VERSION = "0.7.3"
EXPECTED_RECOVERY_PLUGIN_SHA256 = (
    "842ed03d2028c1e67275e9a533bfe3e11126c5b4d93a43c672b8a6d97b60113b"
)
EXPECTED_PLUGIN_IDENTITIES = {
    (EXPECTED_PLUGIN_VERSION, EXPECTED_PLUGIN_SHA256),
    (EXPECTED_RECOVERY_PLUGIN_VERSION, EXPECTED_RECOVERY_PLUGIN_SHA256),
}
EXPECTED_GAME_ASSEMBLY_SHA256 = (
    "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412"
)
EXPECTED_METADATA_SHA256 = (
    "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd"
)
G1_BONE_SIGNATURE_SHA256 = (
    "9d18e697233d9578b398fbe849cd59d65cb27a5c2223b2602db66a82a410e987"
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

UNITY_FIXED_RATE_HZ = 500
TRACE_RATE_HZ = 50
GRID_STRIDE_TICKS = UNITY_FIXED_RATE_HZ // TRACE_RATE_HZ
DEFAULT_PRE_ROLL_TICKS = UNITY_FIXED_RATE_HZ // 2
DEFAULT_POST_ROLL_TICKS = UNITY_FIXED_RATE_HZ // 2
DEFAULT_ATTACK_WINDOW_TICKS = UNITY_FIXED_RATE_HZ * 5
DEFAULT_MAX_BONE_AGE_TICKS = UNITY_FIXED_RATE_HZ // 2
DEFAULT_SETTLE_WINDOW_TICKS = UNITY_FIXED_RATE_HZ // 5
DEFAULT_SETTLE_SPEED_M_S = 0.03
DEFAULT_MINIMUM_HELD_SAMPLES = TRACE_RATE_HZ // 10
ZERO_EPSILON = 1e-6

EXPECTED_CONDITIONS = (
    "W",
    "S",
    "A",
    "D",
    "Q",
    "E",
    "W+Q",
    "W+E",
    "S+Q",
    "S+E",
    "A+Q",
    "A+E",
    "D+Q",
    "D+E",
)

G1_KICK_PROFILES = {
    6: "left_side_kick_processed",
    7: "left_front_kick_processed",
    8: "right_side_kick_processed",
    9: "right_knee_processed",
}


class HeldTraceError(ValueError):
    """The source cannot support the requested fail-closed trace."""


@dataclass
class Capture:
    source_name: str
    raw_sha256: str
    start: dict[str, Any]
    end: dict[str, Any]
    roots: dict[int, dict[str, Any]]
    bone_packets: dict[int, list[dict[str, Any]]]
    decoded_snapshots: dict[int, dict[str, Any]]
    input_requests: list[dict[str, Any]]
    move_requests: list[dict[str, Any]]
    forbidden_requests: list[dict[str, Any]]
    local_slot: int
    opponent_slot: int


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HeldTraceError(reason)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_vector(value: Any, length: int, context: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == length, f"{context}_shape")
    output = []
    for component in value:
        _require(
            isinstance(component, (int, float))
            and not isinstance(component, bool)
            and math.isfinite(float(component)),
            f"{context}_nonfinite",
        )
        output.append(float(component))
    return output


def _require_bool(container: dict[str, Any], name: str, expected: bool, context: str) -> None:
    value = container.get(name)
    _require(value is expected, f"{context}_{name}_expected_{str(expected).lower()}")


def _validate_start(start: dict[str, Any]) -> tuple[int, int]:
    _require(start.get("event") == "capture_start", "first_record_not_capture_start")
    _require(start.get("schema") == RECORDER_SCHEMA, "unsupported_recorder_schema")
    plugin_version = start.get("plugin_version")
    plugin_sha256 = start.get("plugin_sha256")
    _require(isinstance(plugin_version, str), "recorder_version_malformed")
    _require(isinstance(plugin_sha256, str), "recorder_sha256_malformed")
    plugin_identity = (plugin_version, plugin_sha256)
    _require(plugin_identity in EXPECTED_PLUGIN_IDENTITIES, "recorder_identity_mismatch")
    _require(
        start.get("game_assembly_sha256") == EXPECTED_GAME_ASSEMBLY_SHA256,
        "game_assembly_sha256_mismatch",
    )
    _require(
        start.get("global_metadata_sha256") == EXPECTED_METADATA_SHA256,
        "global_metadata_sha256_mismatch",
    )
    _require(start.get("client_sample_stride_ticks") == 10, "unexpected_compact_stride")
    _require(start.get("root_pose_sample_stride_ticks") == 1, "unexpected_root_stride")
    _require(start.get("root_pose_sample_rate_hz") == 500, "unexpected_root_rate")
    fixed_delta = start.get("fixed_delta_time")
    _require(
        isinstance(fixed_delta, (int, float))
        and not isinstance(fixed_delta, bool)
        and math.isclose(float(fixed_delta), 0.002, rel_tol=0.0, abs_tol=1e-12),
        "unexpected_fixed_delta_time",
    )

    scope = start.get("scope")
    _require(isinstance(scope, dict), "scope_missing")
    for name, expected in (
        ("allowed", True),
        ("network_connected", True),
        ("network_is_client", True),
        ("network_is_server", False),
        ("context_is_solo", True),
        ("context_is_ranked", False),
        ("context_auto_find_match", False),
        ("solo_route_hooks_verified", True),
        ("solo_route_proven", True),
        ("solo_route_connect_to_arena_observed", True),
        ("solo_route_enter_championship_observed", True),
        ("solo_route_enter_championship_koth", False),
        ("solo_route_enter_championship_solo", True),
        ("solo_route_arena_identity_consistent", True),
        ("solo_route_runtime_session_identity_consistent", True),
        ("coordinator_is_ranked_arena", False),
        ("opponent_is_ai", True),
        ("opponent_slot_is_ai", True),
        ("human_in_opponent_slot", False),
        ("opponent_slot_has_client", False),
        ("opponent_human_bit_set", False),
        ("fighter_0_visual_only", True),
        ("fighter_1_visual_only", True),
        ("exact_supported_runtime_pairing", True),
        ("exact_t800_vs_t800", False),
        ("exact_g1_vs_g1", True),
    ):
        _require_bool(scope, name, expected, "scope")
    _require(scope.get("solo_route_flow") == "solo", "scope_solo_route_flow_mismatch")
    _require(scope.get("solo_route_reason") == "solo_route_proven", "scope_route_not_proven")
    _require(scope.get("server_private_proven") is False, "unsupported_server_privacy_claim")
    _require(scope.get("server_private_status") == "unknown", "server_privacy_status_mismatch")
    _require(scope.get("sparring_bot_number") == 1, "scope_not_sparring_bot_1")
    _require(scope.get("client_ai_difficulty") == 0, "scope_not_sparring_bot_1_difficulty")
    _require(scope.get("runtime_model") == "g1", "scope_runtime_not_g1")

    local_slot = scope.get("local_fighter_index")
    opponent_slot = scope.get("opponent_slot")
    _require(local_slot in (0, 1), "invalid_local_slot")
    _require(opponent_slot == 1 - local_slot, "invalid_opponent_slot")

    pairing = start.get("pairing")
    _require(isinstance(pairing, dict), "pairing_missing")
    _require(
        pairing.get("required_pairing") == "exact_homogeneous_supported_runtime_pair",
        "pairing_requirement_mismatch",
    )
    _require(pairing.get("runtime_model") == "g1", "pairing_runtime_not_g1")
    for name, expected in (
        ("exact_supported_runtime_pairing", True),
        ("exact_t800_vs_t800", False),
        ("exact_g1_vs_g1", True),
        ("semantic_robot_id_required_for_acceptance", False),
    ):
        _require_bool(pairing, name, expected, "pairing")
    _require(
        pairing.get("required_g1_bone_count") == len(G1_BONE_NAMES),
        "pairing_g1_bone_count_mismatch",
    )
    _require(
        pairing.get("required_g1_bone_signature_sha256")
        == G1_BONE_SIGNATURE_SHA256,
        "pairing_g1_signature_mismatch",
    )
    for slot in (0, 1):
        fighter = pairing.get(f"fighter_{slot}")
        _require(isinstance(fighter, dict), f"pairing_fighter_{slot}_missing")
        _require(fighter.get("bone_count") == len(G1_BONE_NAMES), f"fighter_{slot}_bone_count")
        _require(
            fighter.get("ordered_bone_signature_sha256") == G1_BONE_SIGNATURE_SHA256,
            f"fighter_{slot}_bone_signature",
        )
        _require_bool(fighter, "exact_g1_bone_signature", True, f"fighter_{slot}")

    for slot in (0, 1):
        _require(
            tuple(start.get(f"fighter_{slot}_bones") or ()) == G1_BONE_NAMES,
            f"fighter_{slot}_bone_names_mismatch",
        )

    wire = start.get("bone_wire_protocol")
    _require(isinstance(wire, dict), "bone_wire_protocol_missing")
    _require(wire.get("g1_bone_count") == 30, "wire_g1_bone_count_mismatch")
    _require(wire.get("g1_body_bytes") == 842, "wire_g1_body_size_mismatch")
    _require(
        wire.get("g1_ordered_bone_signature_sha256") == G1_BONE_SIGNATURE_SHA256,
        "wire_g1_signature_mismatch",
    )
    _require(wire.get("delivery") == "unreliable", "wire_bone_delivery_mismatch")

    outbound = start.get("outbound_request_protocol")
    _require(isinstance(outbound, dict), "outbound_protocol_missing")
    _require_bool(outbound, "server_tick_available", False, "outbound")
    _require_bool(outbound, "server_acceptance_available", False, "outbound")
    _require_bool(outbound, "acknowledgement_observed", False, "outbound")

    hooks = start.get("harmony_target_status")
    _require(isinstance(hooks, dict), "harmony_target_status_missing")
    for name in (
        "REKApp.RobotInputController.SendVelocityCommand",
        "REKApp.RobotInputController.SendMoveEvent",
        "REKApp.RobotInputController.SendSpecialEvent",
        "REKApp.RobotInputController.SendEStopToggle",
        "REKApp.Robot.OnBoneMessageReceived",
    ):
        _require_bool(hooks, name, True, "hook")

    server = start.get("server")
    _require(isinstance(server, dict), "server_record_missing")
    _require_bool(server, "endpoint_present", True, "server")
    _require_bool(server, "endpoint_recorded", False, "server")
    _require(_is_sha256(server.get("session_id_sha256")), "session_hash_missing")
    return int(local_slot), int(opponent_slot)


def _open_source(raw_path: str | os.PathLike[str]) -> tuple[BinaryIO, bool, str]:
    if str(raw_path) == "-":
        return sys.stdin.buffer, False, "stdin"
    path = Path(raw_path)
    return path.open("rb"), True, str(path.resolve())


def _root_record(record: dict[str, Any], local_slot: int, opponent_slot: int) -> dict[str, Any]:
    tick = record.get("client_fixed_tick")
    _require(isinstance(tick, int) and not isinstance(tick, bool), "root_tick_missing")
    index = record.get("root_pose_sample_index")
    _require(isinstance(index, int) and not isinstance(index, bool), "root_index_missing")
    output: dict[str, Any] = {
        "tick": tick,
        "index": index,
        "utc": record.get("utc"),
        "stopwatch_timestamp_ticks": record.get("stopwatch_timestamp_ticks"),
        "unity_frame": record.get("unity_frame"),
        "unity_time": record.get("unity_time"),
        "unity_fixed_time": record.get("unity_fixed_time"),
        "unity_unscaled_time": record.get("unity_unscaled_time"),
        "fight_epoch": record.get("fight_epoch"),
        "round_number": record.get("round_number"),
        "local_fighter_index": record.get("local_fighter_index"),
        "opponent_slot": record.get("opponent_slot"),
    }
    _require(record.get("local_fighter_index") == local_slot, "root_local_slot_changed")
    _require(record.get("opponent_slot") == opponent_slot, "root_opponent_slot_changed")
    for slot in (0, 1):
        root = record.get(f"fighter_{slot}_root")
        _require(isinstance(root, dict), f"root_fighter_{slot}_missing")
        output[f"fighter_{slot}_root"] = {
            "world_position_xyz": _finite_vector(
                root.get("world_position_xyz"), 3, f"root_fighter_{slot}_position"
            ),
            "world_rotation_xyzw": _finite_vector(
                root.get("world_rotation_xyzw"), 4, f"root_fighter_{slot}_rotation"
            ),
        }
    return output


def _bone_record(record: dict[str, Any]) -> dict[str, Any]:
    slot = record.get("fighter_slot")
    _require(slot in (0, 1), "bone_invalid_fighter_slot")
    tick = record.get("client_fixed_tick_at_observation")
    sequence = record.get("raw_bone_packet_sequence")
    _require(isinstance(tick, int) and not isinstance(tick, bool), "bone_tick_missing")
    _require(isinstance(sequence, int) and not isinstance(sequence, bool), "bone_sequence_missing")
    _require(record.get("network_index") == slot, "bone_network_index_slot_mismatch")
    _require(record.get("bone_count") == len(G1_BONE_NAMES), "bone_count_mismatch")
    _require(record.get("wire_body_bytes") == 842, "bone_wire_size_mismatch")
    _require(_is_sha256(record.get("wire_body_sha256")), "bone_wire_sha256_missing")
    _require(tuple(record.get("bone_names") or ()) == G1_BONE_NAMES, "bone_names_mismatch")
    return {
        "slot": slot,
        "tick": tick,
        "sequence": sequence,
        "unity_frame": record.get("unity_frame"),
        "unity_time": record.get("unity_time"),
        "unity_unscaled_time": record.get("unity_unscaled_time"),
        "network_index": record.get("network_index"),
        "wire_body_sha256": record.get("wire_body_sha256"),
        "world_positions_xyz": _finite_vector(
            record.get("world_positions_xyz"), 3 * len(G1_BONE_NAMES), "bone_positions"
        ),
        "world_rotations_xyzw": _finite_vector(
            record.get("world_rotations_xyzw"), 4 * len(G1_BONE_NAMES), "bone_rotations"
        ),
    }


def _decoded_bone_record(record: dict[str, Any]) -> dict[str, Any]:
    slot = record.get("fighter_slot")
    tick = record.get("client_fixed_tick_at_observation")
    raw_sequence = record.get("raw_bone_packet_sequence")
    snapshot_sequence = record.get("bone_snapshot_sequence")
    _require(slot in (0, 1), "decoded_bone_invalid_fighter_slot")
    _require(isinstance(tick, int) and not isinstance(tick, bool), "decoded_bone_tick_missing")
    _require(
        isinstance(raw_sequence, int) and not isinstance(raw_sequence, bool),
        "decoded_bone_raw_sequence_missing",
    )
    _require(
        isinstance(snapshot_sequence, int) and not isinstance(snapshot_sequence, bool),
        "decoded_bone_snapshot_sequence_missing",
    )
    _require(
        tuple(record.get("bone_names") or ()) == G1_BONE_NAMES,
        "decoded_bone_names_mismatch",
    )
    return {
        "slot": slot,
        "tick": tick,
        "raw_sequence": raw_sequence,
        "snapshot_sequence": snapshot_sequence,
        "snapshot_received_at_client_time": record.get("snapshot_received_at_client_time"),
        "root_world_position": _finite_vector(
            record.get("root_world_position"), 3, "decoded_root_position"
        ),
        "root_world_rotation_xyzw": _finite_vector(
            record.get("root_world_rotation_xyzw"), 4, "decoded_root_rotation"
        ),
        "child_local_rotations_xyzw": _finite_vector(
            record.get("child_local_rotations_xyzw"),
            4 * len(G1_BONE_NAMES),
            "decoded_child_local_rotations",
        ),
    }


def _request_record(record: dict[str, Any], local_slot: int) -> dict[str, Any]:
    tick = record.get("client_fixed_tick_at_observation")
    sequence = record.get("request_sequence")
    _require(isinstance(tick, int) and not isinstance(tick, bool), "request_tick_missing")
    _require(isinstance(sequence, int) and not isinstance(sequence, bool), "request_sequence_missing")
    _require(record.get("network_index_source_int32") == local_slot, "request_wrong_fighter")
    _require_bool(record, "request_only", True, "request")
    _require(record.get("server_acceptance") is None, "request_claims_server_acceptance")
    _require_bool(record, "ack_observed", False, "request")
    output = {
        "tick": tick,
        "request_sequence": sequence,
        "message_request_sequence": record.get("message_request_sequence"),
        "utc": record.get("utc"),
        "stopwatch_timestamp_ticks": record.get("stopwatch_timestamp_ticks"),
        "unity_frame": record.get("unity_frame"),
        "network_index_source_int32": record.get("network_index_source_int32"),
        "request_only": True,
        "server_acceptance": None,
        "ack_observed": False,
    }
    if record.get("message") == "REK_Input":
        output["message"] = "REK_Input"
        output["velocity_command_xyz"] = _finite_vector(
            record.get("velocity_command_xyz"), 3, "request_velocity"
        )
        _require(record.get("wire_delivery") == "unreliable", "input_delivery_mismatch")
    elif record.get("message") == "REK_Move":
        move_index = record.get("move_index_source_int32")
        _require(
            isinstance(move_index, int) and not isinstance(move_index, bool),
            "move_index_missing",
        )
        output["message"] = "REK_Move"
        output["move_index"] = move_index
        _require(record.get("wire_delivery") == "reliable", "move_delivery_mismatch")
    else:
        raise HeldTraceError("unsupported_outbound_projection")
    return output


def read_capture(raw_path: str | os.PathLike[str]) -> Capture:
    stream, close_stream, source_name = _open_source(raw_path)
    digest = hashlib.sha256()
    start: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    roots: dict[int, dict[str, Any]] = {}
    bone_packets: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    decoded_snapshots: dict[int, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    local_slot: int | None = None
    opponent_slot: int | None = None
    decoded_bone_count = 0
    nonblank_records = 0
    last_event: str | None = None
    last_root_tick: int | None = None
    request_sequences: list[int] = []
    try:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            nonblank_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HeldTraceError(
                    f"invalid_json_line_{line_number}:{exc.msg}"
                ) from exc
            _require(isinstance(record, dict), f"record_{line_number}_not_object")
            event = record.get("event")
            _require(isinstance(event, str), f"record_{line_number}_event_missing")
            last_event = event
            if nonblank_records == 1:
                start = record
                local_slot, opponent_slot = _validate_start(start)
                continue
            _require(start is not None and local_slot is not None, "capture_start_missing")
            _require(end is None, "records_after_capture_end")

            if event == "capture_start":
                raise HeldTraceError("duplicate_capture_start")
            if event == "capture_end":
                end = record
                continue
            if event == "root_pose_sample":
                root = _root_record(record, local_slot, int(opponent_slot))
                tick = root["tick"]
                _require(tick not in roots, "duplicate_root_tick")
                _require(root["index"] == len(roots), "root_indices_not_contiguous")
                if last_root_tick is not None:
                    _require(tick == last_root_tick + 1, "root_ticks_not_contiguous")
                roots[tick] = root
                last_root_tick = tick
                continue
            if event == "raw_bone_packet":
                bone = _bone_record(record)
                slot_packets = bone_packets[bone["slot"]]
                if slot_packets:
                    _require(
                        bone["tick"] >= slot_packets[-1]["tick"],
                        f"fighter_{bone['slot']}_bone_ticks_decreased",
                    )
                slot_packets.append(bone)
                continue
            if event == "decoded_bone_snapshot":
                decoded_bone_count += 1
                decoded = _decoded_bone_record(record)
                _require(
                    decoded["raw_sequence"] not in decoded_snapshots,
                    "duplicate_decoded_raw_bone_sequence",
                )
                decoded_snapshots[decoded["raw_sequence"]] = decoded
                continue
            if event == "outbound_request_projection":
                request = _request_record(record, local_slot)
                request_sequences.append(request["request_sequence"])
                if request["message"] == "REK_Input":
                    inputs.append(request)
                else:
                    moves.append(request)
                continue
            if event == "client_transport_method_invoked" and record.get("method") in {
                "SendSpecialEvent",
                "SendEStopToggle",
            }:
                forbidden.append(
                    {
                        "tick": record.get("client_fixed_tick_at_observation"),
                        "method": record.get("method"),
                    }
                )
    finally:
        if close_stream:
            stream.close()

    _require(start is not None, "capture_start_missing")
    _require(end is not None and last_event == "capture_end", "completed_capture_end_missing")
    _require(roots, "root_stream_missing")
    _require(inputs, "input_request_stream_missing")
    _require(bone_packets[0] and bone_packets[1], "both_bone_streams_required")
    _require(end.get("capture_error_count") == 0, "capture_contains_errors")
    _require(end.get("root_pose_sample_count") == len(roots), "root_count_mismatch")
    observed_bone_count = len(bone_packets[0]) + len(bone_packets[1])
    _require(end.get("raw_bone_packet_count") == observed_bone_count, "raw_bone_count_mismatch")
    _require(end.get("decoded_bone_snapshot_count") == decoded_bone_count, "decoded_bone_count_mismatch")
    _require(decoded_bone_count == observed_bone_count, "raw_decoded_bone_count_mismatch")
    for slot in (0, 1):
        for packet in bone_packets[slot]:
            decoded = decoded_snapshots.get(packet["sequence"])
            _require(decoded is not None, "raw_bone_packet_has_no_decoded_snapshot")
            _require(decoded["slot"] == slot, "raw_decoded_bone_slot_mismatch")
            _require(decoded["tick"] == packet["tick"], "raw_decoded_bone_tick_mismatch")
    _require(
        end.get("client_fixed_tick_at_end") == max(roots) + 1,
        "capture_end_tick_mismatch",
    )
    _require(
        request_sequences == list(range(1, len(request_sequences) + 1)),
        "request_sequences_not_complete",
    )
    method_counts = end.get("client_transport_method_counts")
    _require(isinstance(method_counts, dict), "transport_method_counts_missing")
    _require(method_counts.get("SendVelocityCommand") == len(inputs), "input_count_mismatch")
    _require(method_counts.get("SendMoveEvent", 0) == len(moves), "move_count_mismatch")
    _require(
        end.get("client_transport_invocation_count") == sum(method_counts.values()),
        "transport_invocation_count_mismatch",
    )

    first_root = roots[min(roots)]
    for root in roots.values():
        _require(root["fight_epoch"] == first_root["fight_epoch"], "fight_epoch_changed")
        _require(root["round_number"] == first_root["round_number"], "round_number_changed")

    return Capture(
        source_name=source_name,
        raw_sha256=digest.hexdigest(),
        start=start,
        end=end,
        roots=roots,
        bone_packets=bone_packets,
        decoded_snapshots=decoded_snapshots,
        input_requests=inputs,
        move_requests=moves,
        forbidden_requests=forbidden,
        local_slot=int(local_slot),
        opponent_slot=int(opponent_slot),
    )


def classify_velocity(velocity: list[float]) -> str:
    forward, strafe, yaw = velocity
    translation: str | None = None
    if abs(forward) > ZERO_EPSILON and abs(strafe) > ZERO_EPSILON:
        return "unsupported_mixed_translation"
    if forward > ZERO_EPSILON:
        translation = "W"
    elif forward < -ZERO_EPSILON:
        translation = "S"
    elif strafe > ZERO_EPSILON:
        translation = "A"
    elif strafe < -ZERO_EPSILON:
        translation = "D"
    yaw_key = "Q" if yaw > ZERO_EPSILON else "E" if yaw < -ZERO_EPSILON else None
    if translation and yaw_key:
        return f"{translation}+{yaw_key}"
    return translation or yaw_key or "neutral"


def _auto_window(
    capture: Capture,
    start_tick: int | None,
    end_tick: int | None,
    pre_roll_ticks: int,
    post_roll_ticks: int,
    attack_window_ticks: int,
) -> tuple[int, int]:
    root_start = min(capture.roots)
    root_end = max(capture.roots)
    nonneutral = [
        request["tick"]
        for request in capture.input_requests
        if classify_velocity(request["velocity_command_xyz"]) != "neutral"
    ]
    activity = nonneutral + [request["tick"] for request in capture.move_requests]
    _require(activity, "no_non_neutral_or_move_requests")
    if start_tick is None:
        start_tick = max(root_start, min(activity) - pre_roll_ticks)
        start_tick -= start_tick % GRID_STRIDE_TICKS
    if end_tick is None:
        candidates = []
        if nonneutral:
            candidates.append(max(nonneutral) + post_roll_ticks)
        if capture.move_requests:
            candidates.append(max(request["tick"] for request in capture.move_requests) + attack_window_ticks)
        end_tick = min(root_end, max(candidates))
    _require(isinstance(start_tick, int) and not isinstance(start_tick, bool), "invalid_start_tick")
    _require(isinstance(end_tick, int) and not isinstance(end_tick, bool), "invalid_end_tick")
    _require(root_start <= start_tick <= end_tick <= root_end, "window_outside_root_stream")
    end_tick = start_tick + ((end_tick - start_tick) // GRID_STRIDE_TICKS) * GRID_STRIDE_TICKS
    _require(end_tick > start_tick, "trace_window_too_short")
    return start_tick, end_tick


def _latest(records: list[dict[str, Any]], ticks: list[int], tick: int, context: str) -> dict[str, Any]:
    index = bisect.bisect_right(ticks, tick) - 1
    _require(index >= 0, f"{context}_unavailable_at_grid_start")
    return records[index]


def _request_payload(request: dict[str, Any], grid_tick: int) -> dict[str, Any]:
    return {
        "source_client_fixed_tick": request["tick"],
        "source_age_ticks": grid_tick - request["tick"],
        "source_age_seconds": (grid_tick - request["tick"]) / UNITY_FIXED_RATE_HZ,
        "request_sequence": request["request_sequence"],
        "velocity_command_xyz": request["velocity_command_xyz"],
        "request_only": True,
        "server_acceptance": None,
        "ack_observed": False,
    }


def _bone_payload(
    packet: dict[str, Any],
    decoded: dict[str, Any],
    grid_tick: int,
    fresh: bool,
) -> dict[str, Any]:
    return {
        "layout": "g1_30",
        "source_client_fixed_tick": packet["tick"],
        "source_age_ticks": grid_tick - packet["tick"],
        "source_age_seconds": (grid_tick - packet["tick"]) / UNITY_FIXED_RATE_HZ,
        "fresh_since_previous_grid_sample": fresh,
        "raw_bone_packet_sequence": packet["sequence"],
        "unity_frame_at_receive": packet["unity_frame"],
        "unity_time_at_receive": packet["unity_time"],
        "unity_unscaled_time_at_receive": packet["unity_unscaled_time"],
        "wire_body_sha256": packet["wire_body_sha256"],
        "world_positions_xyz": packet["world_positions_xyz"],
        "world_rotations_xyzw": packet["world_rotations_xyzw"],
        "decoded_snapshot_sequence": decoded["snapshot_sequence"],
        "decoded_source_client_fixed_tick": decoded["tick"],
        "decoded_source_age_ticks": grid_tick - decoded["tick"],
        "decoded_source_age_seconds": (grid_tick - decoded["tick"])
        / UNITY_FIXED_RATE_HZ,
        "decoded_snapshot_received_at_client_time": decoded[
            "snapshot_received_at_client_time"
        ],
        "decoded_root_world_position": decoded["root_world_position"],
        "decoded_root_world_rotation_xyzw": decoded["root_world_rotation_xyzw"],
        "decoded_child_local_rotations_xyzw": decoded[
            "child_local_rotations_xyzw"
        ],
        "reference_alignment_status": "measured_joint_transform_required",
    }


def build_trace_samples(
    capture: Capture,
    start_tick: int,
    end_tick: int,
    max_bone_age_ticks: int,
) -> list[dict[str, Any]]:
    input_ticks = [request["tick"] for request in capture.input_requests]
    bone_ticks = {
        slot: [packet["tick"] for packet in capture.bone_packets[slot]]
        for slot in (0, 1)
    }
    previous_bone_sequence: dict[int, int | None] = {0: None, 1: None}
    previous_grid_tick = start_tick - 1
    samples = []
    for trace_index, tick in enumerate(range(start_tick, end_tick + 1, GRID_STRIDE_TICKS)):
        root = capture.roots.get(tick)
        _require(root is not None, f"root_missing_at_grid_tick_{tick}")
        request = _latest(capture.input_requests, input_ticks, tick, "input_request")
        bones = {}
        for slot in (0, 1):
            packet = _latest(capture.bone_packets[slot], bone_ticks[slot], tick, f"fighter_{slot}_bone")
            age = tick - packet["tick"]
            _require(age >= 0, f"fighter_{slot}_bone_from_future")
            _require(age <= max_bone_age_ticks, f"fighter_{slot}_bone_age_exceeded_at_tick_{tick}")
            fresh = packet["sequence"] != previous_bone_sequence[slot]
            previous_bone_sequence[slot] = packet["sequence"]
            decoded = capture.decoded_snapshots.get(packet["sequence"])
            _require(decoded is not None, f"fighter_{slot}_decoded_bone_unavailable")
            bones[slot] = _bone_payload(packet, decoded, tick, fresh)

        moves = []
        for move in capture.move_requests:
            if previous_grid_tick < move["tick"] <= tick:
                moves.append(
                    {
                        "source_client_fixed_tick": move["tick"],
                        "source_age_ticks": tick - move["tick"],
                        "request_sequence": move["request_sequence"],
                        "move_index": move["move_index"],
                        "move_profile": G1_KICK_PROFILES.get(move["move_index"]),
                        "request_only": True,
                        "server_acceptance": None,
                        "ack_observed": False,
                    }
                )
        velocity = request["velocity_command_xyz"]
        sample = {
            "event": "trace_sample",
            "trace_index": trace_index,
            "client_fixed_tick": tick,
            "time_from_trace_start_seconds": (tick - start_tick) / UNITY_FIXED_RATE_HZ,
            "root_source_age_ticks": 0,
            "root_pose_sample_index": root["index"],
            "utc": root["utc"],
            "stopwatch_timestamp_ticks": root["stopwatch_timestamp_ticks"],
            "unity_frame": root["unity_frame"],
            "unity_time": root["unity_time"],
            "unity_fixed_time": root["unity_fixed_time"],
            "unity_unscaled_time": root["unity_unscaled_time"],
            "fight_epoch": root["fight_epoch"],
            "round_number": root["round_number"],
            "held_condition": classify_velocity(velocity),
            "request_state": _request_payload(request, tick),
            "move_requests_since_previous_grid_sample": moves,
            "fighter_0_root": root["fighter_0_root"],
            "fighter_1_root": root["fighter_1_root"],
            "fighter_0_bones": bones[0],
            "fighter_1_bones": bones[1],
        }
        samples.append(sample)
        previous_grid_tick = tick
    return samples


def _condition_coverage(samples: list[dict[str, Any]], minimum_held_samples: int) -> dict[str, Any]:
    labels = [sample["held_condition"] for sample in samples]
    runs: dict[str, list[int]] = {}
    if labels:
        current = labels[0]
        length = 1
        for label in labels[1:]:
            if label == current:
                length += 1
            else:
                runs.setdefault(current, []).append(length)
                current = label
                length = 1
        runs.setdefault(current, []).append(length)

    conditions = {}
    for label in EXPECTED_CONDITIONS:
        lengths = runs.get(label, [])
        total = sum(lengths)
        longest = max(lengths, default=0)
        observed = longest >= minimum_held_samples
        vectors = [
            sample["request_state"]["velocity_command_xyz"]
            for sample in samples
            if sample["held_condition"] == label
        ]
        axis_ranges = None
        if vectors:
            axis_ranges = {
                "forward": [min(value[0] for value in vectors), max(value[0] for value in vectors)],
                "strafe": [min(value[1] for value in vectors), max(value[1] for value in vectors)],
                "yaw": [min(value[2] for value in vectors), max(value[2] for value in vectors)],
            }
        conditions[label] = {
            "status": "observed" if observed else "absent",
            "observed": observed,
            "grid_samples": total,
            "total_duration_seconds": total / TRACE_RATE_HZ,
            "maximum_contiguous_samples": longest,
            "maximum_contiguous_duration_seconds": longest / TRACE_RATE_HZ,
            "request_axis_ranges": axis_ranges,
        }
    unexpected = sorted(set(labels) - set(EXPECTED_CONDITIONS) - {"neutral"})
    return {
        "minimum_held_samples": minimum_held_samples,
        "minimum_held_duration_seconds": minimum_held_samples / TRACE_RATE_HZ,
        "conditions": conditions,
        "missing_conditions": [
            label for label in EXPECTED_CONDITIONS if not conditions[label]["observed"]
        ],
        "unexpected_conditions": {
            label: labels.count(label) for label in unexpected
        },
        "complete": all(conditions[label]["observed"] for label in EXPECTED_CONDITIONS),
    }


def _root_settle_measurement(
    capture: Capture,
    move_tick: int,
    settle_window_ticks: int,
    settle_speed_m_s: float,
    translation_was_observed: bool,
    translation_release_tick: int | None,
) -> dict[str, Any]:
    window_start = move_tick - settle_window_ticks
    local_key = f"fighter_{capture.local_slot}_root"
    roots = [capture.roots.get(tick) for tick in range(window_start, move_tick + 1)]
    if window_start < min(capture.roots) or any(root is None for root in roots):
        return {
            "status": "unavailable",
            "settled": None,
            "reason": "complete_pre_request_root_window_unavailable",
            "window_ticks": settle_window_ticks,
            "threshold_m_s": settle_speed_m_s,
        }
    positions = [root[local_key]["world_position_xyz"] for root in roots if root is not None]
    path = sum(
        math.hypot(right[0] - left[0], right[2] - left[2])
        for left, right in zip(positions, positions[1:])
    )
    net = math.hypot(
        positions[-1][0] - positions[0][0],
        positions[-1][2] - positions[0][2],
    )
    duration = settle_window_ticks / UNITY_FIXED_RATE_HZ
    path_speed = path / duration
    net_speed = net / duration
    neutral_dwell_complete = (
        not translation_was_observed
        or translation_release_tick is not None
        and translation_release_tick <= window_start
    )
    settled = neutral_dwell_complete and path_speed < settle_speed_m_s
    return {
        "status": "settled" if settled else "not_settled",
        "settled": settled,
        "reason": (
            "neutral_dwell_and_observed_planar_root_path_below_threshold"
            if settled
            else "neutral_dwell_incomplete"
            if not neutral_dwell_complete
            else "observed_planar_root_path_not_below_threshold"
        ),
            "source": "exact_500hz_client_root_pose_samples",
        "window_start_client_fixed_tick": window_start,
        "window_end_client_fixed_tick": move_tick,
        "window_ticks": settle_window_ticks,
        "window_seconds": duration,
            "threshold_m_s": settle_speed_m_s,
            "threshold_provenance": "provisional_G1_config_base_velocity_threshold_applied_to_observed_root_path_not_runtime_calibrated",
        "planar_path_m": path,
        "planar_net_displacement_m": net,
        "planar_path_speed_m_s": path_speed,
        "planar_net_speed_m_s": net_speed,
        "translation_request_neutral_for_full_window": neutral_dwell_complete,
    }


def _attack_coverage(
    capture: Capture,
    start_tick: int,
    end_tick: int,
    settle_window_ticks: int,
    settle_speed_m_s: float,
    attack_window_ticks: int,
) -> dict[str, Any]:
    moves = [
        move for move in capture.move_requests if start_tick <= move["tick"] <= end_tick
    ]
    reports = []
    for move_index, move in enumerate(moves):
        prior_inputs = [
            request
            for request in capture.input_requests
            if request["request_sequence"] < move["request_sequence"]
        ]
        _require(prior_inputs, "move_has_no_preceding_input_projection")
        input_at_request = prior_inputs[-1]
        translation_indices = [
            index
            for index, request in enumerate(prior_inputs)
            if abs(request["velocity_command_xyz"][0]) > ZERO_EPSILON
            or abs(request["velocity_command_xyz"][1]) > ZERO_EPSILON
        ]
        last_translation_index = translation_indices[-1] if translation_indices else None
        last_translation_tick = (
            prior_inputs[last_translation_index]["tick"]
            if last_translation_index is not None
            else None
        )
        translation_release_tick = None
        if last_translation_index is not None:
            for request in prior_inputs[last_translation_index + 1 :]:
                velocity = request["velocity_command_xyz"]
                if abs(velocity[0]) <= ZERO_EPSILON and abs(velocity[1]) <= ZERO_EPSILON:
                    translation_release_tick = request["tick"]
                    break
        next_move_tick = moves[move_index + 1]["tick"] if move_index + 1 < len(moves) else end_tick
        yaw_window_end = min(end_tick, move["tick"] + attack_window_ticks, next_move_tick)
        yaw_inputs = [input_at_request] + [
            request
            for request in capture.input_requests
            if move["request_sequence"] < request["request_sequence"]
            and request["tick"] <= yaw_window_end
        ]
        yaw_zero = bool(yaw_inputs) and all(
            abs(request["velocity_command_xyz"][2]) <= ZERO_EPSILON
            for request in yaw_inputs
        )
        settle = _root_settle_measurement(
            capture,
            move["tick"],
            settle_window_ticks,
            settle_speed_m_s,
            bool(translation_indices),
            translation_release_tick,
        )
        reports.append(
            {
                "move_index": move["move_index"],
                "move_profile": G1_KICK_PROFILES.get(move["move_index"]),
                "recognized_g1_kick_profile": move["move_index"] in G1_KICK_PROFILES,
                "client_fixed_tick": move["tick"],
                "request_sequence": move["request_sequence"],
                "request_only": True,
                "server_acceptance": {
                    "status": "unknown",
                    "value": None,
                    "ack_observed": False,
                    "reason": "recorder_protocol_exposes_no_server_acceptance_or_acknowledgement",
                },
                "input_at_request": {
                    "source_client_fixed_tick": input_at_request["tick"],
                    "source_age_ticks": move["tick"] - input_at_request["tick"],
                    "velocity_command_xyz": input_at_request["velocity_command_xyz"],
                    "held_condition": classify_velocity(input_at_request["velocity_command_xyz"]),
                },
                "last_nonzero_translation_request_tick": last_translation_tick,
                "translation_release_request_tick": translation_release_tick,
                "translation_neutral_dwell_seconds": (
                    None
                    if translation_release_tick is None
                    else (move["tick"] - translation_release_tick) / UNITY_FIXED_RATE_HZ
                ),
                "translation_settle_observation": settle,
                "post_request_yaw_projection": {
                    "status": "all_zero" if yaw_zero else "nonzero_observed",
                    "all_zero": yaw_zero,
                    "observation_end_client_fixed_tick": yaw_window_end,
                    "request_projection_count": len(yaw_inputs),
                    "authority": "client_request_projection_only",
                },
                "accepted_kick_yaw_suppression": {
                    "status": "unknown",
                    "value": None,
                    "reason": "kick_acceptance_is_unobservable;_only_request_time_yaw_suppression_can_be_reported",
                },
                "reference_alignment_window": {
                    "start_client_fixed_tick": max(start_tick, move["tick"] - settle_window_ticks),
                    "end_client_fixed_tick": yaw_window_end,
                    "observation": "decoded_child_local_quaternions_with_per_sample_source_age",
                    "comparison_method": "DTW_after_measured_calibrated_NPZ_joint_transform",
                    "direct_angle_identity_allowed": False,
                },
            }
        )
    return {
        "request_count": len(reports),
        "requests": reports,
        "acceptance_channel": {
            "status": "unknown",
            "available": False,
            "reason": "REK_Move_has_no_observed_server_acknowledgement_or_acceptance_field",
        },
    }


def _bone_source_summary(
    capture: Capture, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    output = {}
    trace_start = samples[0]["client_fixed_tick"]
    trace_end = samples[-1]["client_fixed_tick"]
    for slot in (0, 1):
        selected_sequences = {
            sample[f"fighter_{slot}_bones"]["raw_bone_packet_sequence"]
            for sample in samples
        }
        selected_packets = [
            packet
            for packet in capture.bone_packets[slot]
            if packet["sequence"] in selected_sequences
        ]
        received_packets = [
            packet
            for packet in capture.bone_packets[slot]
            if trace_start <= packet["tick"] <= trace_end
        ]
        rate = None
        if len(received_packets) > 1:
            elapsed_ticks = received_packets[-1]["tick"] - received_packets[0]["tick"]
            if elapsed_ticks > 0:
                rate = (len(received_packets) - 1) * UNITY_FIXED_RATE_HZ / elapsed_ticks
        ages = [sample[f"fighter_{slot}_bones"]["source_age_ticks"] for sample in samples]
        fresh = sum(
            1
            for sample in samples
            if sample[f"fighter_{slot}_bones"]["fresh_since_previous_grid_sample"]
        )
        output[str(slot)] = {
            "unique_source_packets_on_grid": len(selected_sequences),
            "received_source_packets_in_trace_window": len(received_packets),
            "observed_source_rate_hz": rate,
            "maximum_source_age_ticks": max(ages),
            "maximum_source_age_seconds": max(ages) / UNITY_FIXED_RATE_HZ,
            "fresh_grid_samples": fresh,
            "reused_grid_samples": len(samples) - fresh,
            "first_selected_packet_tick": selected_packets[0]["tick"],
            "last_selected_packet_tick": selected_packets[-1]["tick"],
        }
    return output


def build_coverage_report(
    capture: Capture,
    samples: list[dict[str, Any]],
    start_tick: int,
    end_tick: int,
    minimum_held_samples: int,
    settle_window_ticks: int,
    settle_speed_m_s: float,
    attack_window_ticks: int,
) -> dict[str, Any]:
    scope = capture.start["scope"]
    coverage = _condition_coverage(samples, minimum_held_samples)
    forbidden = [
        request
        for request in capture.forbidden_requests
        if isinstance(request.get("tick"), int)
        and start_tick <= request["tick"] <= end_tick
    ]
    _require(not forbidden, "special_or_estop_request_in_trace_window")
    return {
        "schema": COVERAGE_SCHEMA,
        "source": {
            "path": capture.source_name,
            "sha256": capture.raw_sha256,
            "recorder_schema": RECORDER_SCHEMA,
            "recorder_plugin_version": capture.start.get("plugin_version"),
            "recorder_plugin_sha256": capture.start.get("plugin_sha256"),
        },
        "scope": {
            "allowed": True,
            "solo_route_proven": True,
            "context_is_solo": True,
            "context_is_ranked": False,
            "sparring_bot_number": 1,
            "opponent_is_ai": True,
            "human_in_opponent_slot": False,
            "runtime_model": "g1",
            "exact_g1_vs_g1": True,
            "local_fighter_index": capture.local_slot,
            "opponent_slot": capture.opponent_slot,
            "session_id_sha256": capture.start["server"]["session_id_sha256"],
            "server_private_proven": scope.get("server_private_proven"),
            "server_private_status": scope.get("server_private_status"),
        },
        "trace_grid": {
            "start_client_fixed_tick": start_tick,
            "end_client_fixed_tick": end_tick,
            "sample_count": len(samples),
            "rate_hz": TRACE_RATE_HZ,
            "stride_client_fixed_ticks": GRID_STRIDE_TICKS,
            "root_sampling": "exact_500hz_root_sample_at_each_50hz_grid_tick",
            "bone_sampling": "latest_received_raw_REK_Bones_packet_at_or_before_grid_tick_no_interpolation",
            "bone_source": _bone_source_summary(capture, samples),
        },
        "request_authority": {
            "scope": "client_request_projection_only",
            "server_acceptance_available": False,
            "acknowledgement_available": False,
            "authoritative_execution_claimed": False,
        },
        "axis_mapping": {
            "W": "forward_positive",
            "S": "forward_negative",
            "A": "strafe_negative",
            "D": "strafe_positive",
            "Q": "yaw_negative",
            "E": "yaw_positive",
        },
        "held_condition_coverage": coverage,
        "attacks": _attack_coverage(
            capture,
            start_tick,
            end_tick,
            settle_window_ticks,
            settle_speed_m_s,
            attack_window_ticks,
        ),
        "runtime_integration_gate": {
            "status": "not_implemented_by_extractor",
            "required_bridge_mode": "fixed_50hz_G1_held_schedule_request_edges_only",
            "required_conditions": list(EXPECTED_CONDITIONS),
            "translation_gate": "neutral_request_then_network_root_settle_for_configured_consecutive_window",
            "attack_yaw_rule": "force_yaw_request_zero_from_move_arm_through_bounded_network_pose_completion_observation",
            "acceptance_label_rule": "remain_unknown_until_an_independent_server_acceptance_signal_exists",
            "pose_collection_rule": "retain_v7_recorder_as_pose_authority_and_do_not_duplicate_bridge_pose_payloads",
        },
        "reference_alignment": {
            "status": "calibration_required",
            "observed_signal": "Robot.BoneSnapshot_child_local_rotations_xyzw",
            "observed_link_count": len(G1_BONE_NAMES) - 1,
            "reference_signal": "extracted_G1_kick_NPZ_29_DOF_joint_trajectory",
            "required_transform": "measured_calibrated_NPZ_joint_to_REK_child_quaternion_transform",
            "direct_angle_identity_allowed": False,
            "recommended_comparison": "source_age_aware_DTW_on_the_attack_reference_alignment_window",
        },
    }


def _trace_start(capture: Capture, start_tick: int, end_tick: int, max_bone_age_ticks: int) -> dict[str, Any]:
    return {
        "event": "trace_start",
        "schema": TRACE_SCHEMA,
        "source_raw_sha256": capture.raw_sha256,
        "source_recorder_schema": RECORDER_SCHEMA,
        "authority": "client_request_projections_plus_client_observed_network_bones_and_roots",
        "server_acceptance_available": False,
        "server_tick_available": False,
        "root_tick_domain": "client_fixed_update",
        "trace_rate_hz": TRACE_RATE_HZ,
        "trace_grid_stride_client_fixed_ticks": GRID_STRIDE_TICKS,
        "start_client_fixed_tick": start_tick,
        "end_client_fixed_tick": end_tick,
        "maximum_bone_source_age_ticks": max_bone_age_ticks,
        "bone_layout": {
            "id": "g1_30",
            "count": len(G1_BONE_NAMES),
            "ordered_names": list(G1_BONE_NAMES),
            "ordered_signature_sha256": G1_BONE_SIGNATURE_SHA256,
        },
        "decoded_bone_observation": {
            "field": "decoded_child_local_rotations_xyzw",
            "stored_quaternion_count": len(G1_BONE_NAMES),
            "articulated_child_count": len(G1_BONE_NAMES) - 1,
            "root_quaternion_source": "decoded_root_world_rotation_xyzw",
            "reference_alignment_status": "measured_joint_transform_required",
            "direct_NPZ_angle_identity_allowed": False,
        },
    }


def _json_bytes(record: dict[str, Any], pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic(path: Path, chunks: Iterable[bytes]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary path {temporary}")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            for chunk in chunks:
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


def extract(
    raw_path: str | os.PathLike[str],
    trace_out: str | os.PathLike[str] | None = None,
    coverage_out: str | os.PathLike[str] | None = None,
    *,
    start_tick: int | None = None,
    end_tick: int | None = None,
    pre_roll_ticks: int = DEFAULT_PRE_ROLL_TICKS,
    post_roll_ticks: int = DEFAULT_POST_ROLL_TICKS,
    attack_window_ticks: int = DEFAULT_ATTACK_WINDOW_TICKS,
    max_bone_age_ticks: int = DEFAULT_MAX_BONE_AGE_TICKS,
    settle_window_ticks: int = DEFAULT_SETTLE_WINDOW_TICKS,
    settle_speed_m_s: float = DEFAULT_SETTLE_SPEED_M_S,
    minimum_held_samples: int = DEFAULT_MINIMUM_HELD_SAMPLES,
) -> dict[str, Any]:
    _require(pre_roll_ticks >= 0, "invalid_pre_roll_ticks")
    _require(post_roll_ticks >= 0, "invalid_post_roll_ticks")
    _require(attack_window_ticks > 0, "invalid_attack_window_ticks")
    _require(max_bone_age_ticks >= 0, "invalid_max_bone_age_ticks")
    _require(settle_window_ticks > 0, "invalid_settle_window_ticks")
    _require(
        math.isfinite(settle_speed_m_s) and settle_speed_m_s > 0,
        "invalid_settle_speed",
    )
    _require(minimum_held_samples > 0, "invalid_minimum_held_samples")
    capture = read_capture(raw_path)
    selected_start, selected_end = _auto_window(
        capture,
        start_tick,
        end_tick,
        pre_roll_ticks,
        post_roll_ticks,
        attack_window_ticks,
    )
    samples = build_trace_samples(
        capture,
        selected_start,
        selected_end,
        max_bone_age_ticks,
    )
    coverage = build_coverage_report(
        capture,
        samples,
        selected_start,
        selected_end,
        minimum_held_samples,
        settle_window_ticks,
        settle_speed_m_s,
        attack_window_ticks,
    )

    trace_sha256 = None
    if trace_out is not None:
        _require(str(trace_out) != "-", "trace_stdout_not_supported")
        trace_path = Path(trace_out)
        records = [_trace_start(capture, selected_start, selected_end, max_bone_age_ticks)]
        records.extend(samples)
        records.append(
            {
                "event": "trace_end",
                "schema": TRACE_SCHEMA,
                "sample_count": len(samples),
                "complete": True,
                "source_raw_sha256": capture.raw_sha256,
            }
        )
        trace_sha256 = _write_atomic(trace_path, (_json_bytes(record) for record in records))
        coverage["trace_artifact"] = {
            "path": str(trace_path.resolve()),
            "sha256": trace_sha256,
            "schema": TRACE_SCHEMA,
        }

    if coverage_out is not None:
        if str(coverage_out) == "-":
            sys.stdout.buffer.write(_json_bytes(coverage, pretty=True))
            sys.stdout.buffer.flush()
        else:
            _write_atomic(Path(coverage_out), [_json_bytes(coverage, pretty=True)])
    return coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="completed recorder v7 JSONL, or - for stdin")
    parser.add_argument("--trace-out", help="output JSONL trace; omit for coverage-only analysis")
    parser.add_argument("--coverage-out", required=True, help="coverage JSON path, or - for stdout")
    parser.add_argument("--start-tick", type=int)
    parser.add_argument("--end-tick", type=int)
    parser.add_argument("--pre-roll-ticks", type=int, default=DEFAULT_PRE_ROLL_TICKS)
    parser.add_argument("--post-roll-ticks", type=int, default=DEFAULT_POST_ROLL_TICKS)
    parser.add_argument("--attack-window-ticks", type=int, default=DEFAULT_ATTACK_WINDOW_TICKS)
    parser.add_argument("--max-bone-age-ticks", type=int, default=DEFAULT_MAX_BONE_AGE_TICKS)
    parser.add_argument("--settle-window-ticks", type=int, default=DEFAULT_SETTLE_WINDOW_TICKS)
    parser.add_argument("--settle-speed-m-s", type=float, default=DEFAULT_SETTLE_SPEED_M_S)
    parser.add_argument("--minimum-held-samples", type=int, default=DEFAULT_MINIMUM_HELD_SAMPLES)
    arguments = parser.parse_args(argv)
    try:
        extract(
            arguments.raw,
            arguments.trace_out,
            arguments.coverage_out,
            start_tick=arguments.start_tick,
            end_tick=arguments.end_tick,
            pre_roll_ticks=arguments.pre_roll_ticks,
            post_roll_ticks=arguments.post_roll_ticks,
            attack_window_ticks=arguments.attack_window_ticks,
            max_bone_age_ticks=arguments.max_bone_age_ticks,
            settle_window_ticks=arguments.settle_window_ticks,
            settle_speed_m_s=arguments.settle_speed_m_s,
            minimum_held_samples=arguments.minimum_held_samples,
        )
    except (HeldTraceError, FileExistsError, OSError) as exc:
        print(f"g1 held trace extraction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
