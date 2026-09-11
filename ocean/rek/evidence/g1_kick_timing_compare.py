#!/usr/bin/env python3
"""Compare G1 kick request and received-pose timing across isolated captures.

The analyzer keeps three clocks and authorities separate:

* the bridge schedule edge and local ``SendMoveEvent`` prefix;
* the recorder's outbound ``REK_Move`` projection;
* local-client receipt of ``REK_Bones`` packets used by the provisional pose
  departure detector in ``g1_schedule_pose_response.py``.

The last item is only a client-receipt-domain candidate signal. The packet has
no source timestamp, server tick, acknowledgement, acceptance, or move
identity. This tool therefore never emits a server-acceptance value, an
input-to-physical-response estimate, or a canonical move duration. NPZ frame
counts are retained as requested-asset metadata and are never converted into
timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OUTPUT_SCHEMA = "rek.g1_kick_timing_comparison.v1"
POSE_REPORT_SCHEMA = "rek.g1_schedule_pose_response.v1"
TRANSCRIPT_SCHEMA = "rek.g1_held_input_schedule.v2"
COVERAGE_SCHEMA = "rek.g1_held_motion_trace.coverage.v1"
TRACE_SCHEMA = "rek.g1_held_motion_trace.v1"
UNITY_FIXED_RATE_HZ = 500
EXPECTED_MOVES = (6, 7, 8, 9)
EXPECTED_PROBE_KINDS = ("translation_held", "yaw_preempted")
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RAW_BONE_SEQUENCE_PATTERN = re.compile(
    br'"raw_bone_packet_sequence"\s*:\s*([0-9]+)'
)


class KickTimingError(ValueError):
    """The supplied evidence cannot support a fail-closed timing report."""


@dataclass(frozen=True)
class RunSpec:
    label: str
    report_path: Path
    transcript_path: Path | None = None
    raw_path: Path | None = None
    coverage_path: Path | None = None


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise KickTimingError(reason)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{context}_not_finite",
    )
    return float(value)


def _finite_vector(value: Any, length: int, context: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == length, f"{context}_shape")
    return [_finite(component, f"{context}_{index}") for index, component in enumerate(value)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    _require(path.is_file(), f"{context}_not_a_file")
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise KickTimingError(f"{context}_invalid_json:{exc.msg}") from exc
    _require(isinstance(value, dict), f"{context}_not_object")
    return value


def _path_from_source(
    override: Path | None, source: dict[str, Any], context: str
) -> Path:
    if override is not None:
        return override.resolve()
    embedded = source.get("path")
    _require(isinstance(embedded, str) and embedded, f"{context}_path_missing")
    return Path(embedded).resolve()


def _verify_source_hash(path: Path, expected: Any, context: str) -> str:
    _require(path.is_file(), f"{context}_not_a_file")
    _require(
        isinstance(expected, str)
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected),
        f"{context}_expected_sha256_invalid",
    )
    observed = _sha256(path)
    _require(observed == expected, f"{context}_sha256_mismatch")
    return observed


def _parse_json_line(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KickTimingError(f"{context}_invalid_json:{exc.msg}") from exc
    _require(isinstance(value, dict), f"{context}_not_object")
    return value


def _scan_transcript(
    path: Path, expected_sha256: str
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    digest = hashlib.sha256()
    edges: dict[int, dict[str, Any]] = {}
    sends: dict[int, dict[str, Any]] = {}
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            if b"g1_held_schedule_tick" in raw and b"kick_edge" in raw:
                record = _parse_json_line(raw, f"transcript_line_{line_number}")
                detail = record.get("detail")
                if not isinstance(detail, dict) or detail.get("kick_edge") is not True:
                    continue
                _require(
                    record.get("g1_held_schedule_schema") == TRANSCRIPT_SCHEMA,
                    "transcript_edge_schema_mismatch",
                )
                ordinal = detail.get("kick_probe_ordinal")
                _require(_is_int(ordinal) and ordinal >= 0, "transcript_edge_ordinal_invalid")
                _require(ordinal not in edges, "transcript_duplicate_kick_edge")
                _require(record.get("request_only") is True, "transcript_edge_not_request_only")
                _require(
                    record.get("server_acceptance") == "unknown",
                    "transcript_edge_claims_server_acceptance",
                )
                _require(
                    record.get("server_acceptance_observed") is False,
                    "transcript_edge_claims_server_ack",
                )
                edges[int(ordinal)] = record
            elif b"g1_kick_request_lifecycle" in raw and b"send_move_invoked" in raw:
                record = _parse_json_line(raw, f"transcript_line_{line_number}")
                if record.get("event") != "g1_kick_request_lifecycle":
                    continue
                detail = record.get("detail")
                if not isinstance(detail, dict) or detail.get("lifecycle_stage") != "send_move_invoked":
                    continue
                ordinal = detail.get("probe_ordinal")
                _require(_is_int(ordinal) and ordinal >= 0, "transcript_send_ordinal_invalid")
                _require(ordinal not in sends, "transcript_duplicate_send_prefix")
                _require(record.get("request_only") is True, "transcript_send_not_request_only")
                _require(
                    record.get("server_acceptance") == "unknown",
                    "transcript_send_claims_server_acceptance",
                )
                sends[int(ordinal)] = record
    _require(digest.hexdigest() == expected_sha256, "transcript_sha256_mismatch")
    _require(edges, "transcript_kick_edges_missing")
    return edges, sends


def _change_points(pose: dict[str, Any]) -> dict[str, Any]:
    observed = pose.get("pose_departure_observed")
    if observed is None:
        return {
            "status": "unavailable",
            "reason": pose.get("status"),
            "onset_index": None,
            "return_index": None,
            "target_sequences": [],
        }
    _require(isinstance(observed, bool), "pose_departure_flag_invalid")
    samples = pose.get("samples")
    detector = pose.get("detector")
    _require(isinstance(samples, list), "pose_samples_missing")
    _require(isinstance(detector, dict), "pose_detector_missing")
    consecutive = detector.get("required_consecutive_received_packets")
    _require(_is_int(consecutive) and consecutive >= 1, "pose_consecutive_count_invalid")
    flags: list[bool] = []
    previous_tick: int | None = None
    sequences: list[int] = []
    for index, sample in enumerate(samples):
        _require(isinstance(sample, dict), f"pose_sample_{index}_not_object")
        flag = sample.get("above_departure_threshold")
        sequence = sample.get("raw_bone_packet_sequence")
        tick = sample.get("recorder_client_fixed_tick")
        _require(isinstance(flag, bool), f"pose_sample_{index}_flag_invalid")
        _require(_is_int(sequence) and sequence > 0, f"pose_sample_{index}_sequence_invalid")
        _require(_is_int(tick) and tick >= 0, f"pose_sample_{index}_tick_invalid")
        if previous_tick is not None:
            _require(tick >= previous_tick, "pose_sample_ticks_decreased")
        previous_tick = int(tick)
        flags.append(flag)
        sequences.append(int(sequence))
    _require(len(set(sequences)) == len(sequences), "pose_sample_sequences_not_unique")

    onset: int | None = None
    streak = 0
    for index, flag in enumerate(flags):
        streak = streak + 1 if flag else 0
        if streak >= consecutive:
            onset = index - consecutive + 1
            break
    _require((onset is not None) is observed, "pose_departure_flag_change_point_mismatch")

    returned: int | None = None
    if onset is not None:
        streak = 0
        for index in range(onset + consecutive, len(flags)):
            streak = streak + 1 if not flags[index] else 0
            if streak >= consecutive:
                returned = index - consecutive + 1
                break

    target_indices: set[int] = set()
    if onset is not None:
        target_indices.add(onset)
        if onset > 0:
            target_indices.add(onset - 1)
        if returned is None:
            above_after_onset = [
                index for index in range(onset, len(flags)) if flags[index]
            ]
            if above_after_onset:
                target_indices.add(above_after_onset[-1])
        else:
            target_indices.add(returned)
            prior_above = [
                index for index in range(onset, returned) if flags[index]
            ]
            _require(prior_above, "pose_return_has_no_prior_departed_packet")
            target_indices.add(prior_above[-1])

    return {
        "status": "observed" if onset is not None else "not_observed",
        "reason": pose.get("status"),
        "onset_index": onset,
        "return_index": returned,
        "last_confirmed_departed_index": (
            max(
                (index for index in range(onset, len(flags)) if flags[index]),
                default=None,
            )
            if onset is not None and returned is None
            else max(
                (index for index in range(onset or 0, returned or 0) if flags[index]),
                default=None,
            )
        ),
        "target_sequences": [sequences[index] for index in sorted(target_indices)],
    }


def _scan_raw(
    path: Path,
    expected_sha256: str,
    requested_sequences: set[int],
    bone_sequences: set[int],
    local_slot: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    digest = hashlib.sha256()
    moves: dict[int, dict[str, Any]] = {}
    bones: dict[int, dict[str, Any]] = {}
    capture_start: dict[str, Any] | None = None
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            if capture_start is None:
                capture_start = _parse_json_line(raw, "raw_capture_start")
                _require(capture_start.get("event") == "capture_start", "raw_first_event_not_start")
                continue
            if b"outbound_request_projection" in raw and b"REK_Move" in raw:
                record = _parse_json_line(raw, f"raw_line_{line_number}")
                sequence = record.get("request_sequence")
                if sequence not in requested_sequences:
                    continue
                _require(_is_int(sequence) and sequence > 0, "raw_move_sequence_invalid")
                _require(sequence not in moves, "raw_duplicate_move_sequence")
                _require(record.get("message") == "REK_Move", "raw_move_message_mismatch")
                _require(record.get("request_only") is True, "raw_move_not_request_only")
                _require(record.get("server_acceptance") is None, "raw_move_claims_acceptance")
                _require(record.get("ack_observed") is False, "raw_move_claims_ack")
                moves[int(sequence)] = record
            elif (
                b'"event":"raw_bone_packet"' in raw
                or b'"event": "raw_bone_packet"' in raw
            ):
                match = RAW_BONE_SEQUENCE_PATTERN.search(raw)
                _require(match is not None, "raw_bone_sequence_missing")
                sequence = int(match.group(1))
                if sequence not in bone_sequences:
                    continue
                record = _parse_json_line(raw, f"raw_line_{line_number}")
                _require(record.get("event") == "raw_bone_packet", "raw_bone_event_mismatch")
                _require(record.get("fighter_slot") == local_slot, "raw_bone_wrong_fighter_slot")
                _require(sequence not in bones, "raw_duplicate_bone_sequence")
                _finite(record.get("monotonic_receipt_time"), "raw_bone_receipt_time")
                bones[sequence] = record
    _require(digest.hexdigest() == expected_sha256, "raw_sha256_mismatch")
    _require(capture_start is not None, "raw_capture_empty")
    missing_moves = sorted(requested_sequences - set(moves))
    missing_bones = sorted(bone_sequences - set(bones))
    _require(not missing_moves, f"raw_move_sequences_missing:{missing_moves}")
    _require(not missing_bones, f"raw_bone_sequences_missing:{missing_bones}")
    return moves, bones, capture_start


def _pose_receipt_timing(
    pose: dict[str, Any],
    change: dict[str, Any],
    bones: dict[int, dict[str, Any]],
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    common_limits = {
        "authority": "local_client_receipt_of_local_slot_REK_Bones",
        "packet_source_timestamp_available": False,
        "server_tick_available": False,
        "server_action_acceptance": "unknown",
        "executed_move_identity": "unknown",
        "causal_attribution_to_request": "unknown",
    }
    if change["status"] == "unavailable":
        return {
            "status": "insufficient_packet_coverage",
            "candidate_departure_observed": None,
            "request_to_first_candidate_receipt_bracket_seconds": None,
            "candidate_departure_duration_receipt_domain_seconds": None,
            **common_limits,
        }
    if change["status"] == "not_observed":
        return {
            "status": "no_candidate_departure_observed",
            "candidate_departure_observed": False,
            "request_to_first_candidate_receipt_bracket_seconds": None,
            "candidate_departure_duration_receipt_domain_seconds": None,
            **common_limits,
        }
    if request is None:
        return {
            "status": "candidate_departure_without_dual_observed_request",
            "candidate_departure_observed": True,
            "request_to_first_candidate_receipt_bracket_seconds": None,
            "candidate_departure_duration_receipt_domain_seconds": None,
            **common_limits,
        }

    samples = pose["samples"]
    onset_index = int(change["onset_index"])
    onset_sequence = int(samples[onset_index]["raw_bone_packet_sequence"])
    onset_time = _finite(
        bones[onset_sequence].get("monotonic_receipt_time"), "onset_receipt_time"
    )
    request_time = _finite(
        request.get("unity_realtime_since_startup"), "request_projection_realtime"
    )
    if onset_index > 0:
        previous_sequence = int(samples[onset_index - 1]["raw_bone_packet_sequence"])
        previous_time = _finite(
            bones[previous_sequence].get("monotonic_receipt_time"),
            "previous_receipt_time",
        )
    else:
        previous_sequence = None
        previous_time = request_time
    _require(previous_time <= onset_time, "onset_receipt_times_decreased")

    signed_upper = onset_time - request_time
    if signed_upper < 0:
        onset_status = "candidate_pose_already_received_before_request_projection"
        onset_bracket = {
            "status": onset_status,
            "lower": None,
            "upper": None,
            "signed_first_candidate_receipt_age": signed_upper,
        }
    else:
        onset_status = "bounded_in_client_receipt_domain"
        onset_bracket = {
            "status": onset_status,
            "lower": max(0.0, previous_time - request_time),
            "upper": signed_upper,
            "signed_first_candidate_receipt_age": signed_upper,
        }

    returned_index = change.get("return_index")
    last_departed_index = change.get("last_confirmed_departed_index")
    duration: dict[str, Any]
    if returned_index is not None and last_departed_index is not None:
        returned_sequence = int(samples[int(returned_index)]["raw_bone_packet_sequence"])
        last_departed_sequence = int(
            samples[int(last_departed_index)]["raw_bone_packet_sequence"]
        )
        returned_time = _finite(
            bones[returned_sequence].get("monotonic_receipt_time"), "return_receipt_time"
        )
        last_departed_time = _finite(
            bones[last_departed_sequence].get("monotonic_receipt_time"),
            "last_departed_receipt_time",
        )
        _require(onset_time <= last_departed_time <= returned_time, "return_receipt_order_invalid")
        duration = {
            "status": "bounded_in_client_receipt_domain",
            "lower": max(0.0, last_departed_time - onset_time),
            "upper": max(0.0, returned_time - max(previous_time, request_time)),
            "last_confirmed_departed_packet_sequence": last_departed_sequence,
            "first_returned_packet_sequence": returned_sequence,
        }
    elif last_departed_index is not None:
        last_departed_sequence = int(
            samples[int(last_departed_index)]["raw_bone_packet_sequence"]
        )
        last_departed_time = _finite(
            bones[last_departed_sequence].get("monotonic_receipt_time"),
            "last_departed_receipt_time",
        )
        _require(onset_time <= last_departed_time, "censored_receipt_order_invalid")
        duration = {
            "status": "right_censored_in_client_receipt_domain",
            "lower": max(0.0, last_departed_time - onset_time),
            "upper": None,
            "last_confirmed_departed_packet_sequence": last_departed_sequence,
            "first_returned_packet_sequence": None,
        }
    else:
        duration = {"status": "unknown", "lower": None, "upper": None}

    return {
        "status": "candidate_departure_receipt_observed",
        "candidate_departure_observed": True,
        "first_candidate_packet_sequence": onset_sequence,
        "preceding_non_departed_packet_sequence": previous_sequence,
        "request_to_first_candidate_receipt_bracket_seconds": onset_bracket,
        "candidate_departure_duration_receipt_domain_seconds": duration,
        "source_report_fixed_tick_onset_bracket": pose.get("onset_receipt_bracket"),
        "source_report_fixed_tick_return_bracket": pose.get("return_receipt_bracket"),
        "detector": pose.get("detector"),
        **common_limits,
    }


def _trace_summary(
    coverage_path: Path | None,
    raw_sha256: str,
    local_slot: int,
) -> dict[str, Any]:
    if coverage_path is None:
        return {"status": "not_supplied"}
    coverage = _read_json(coverage_path, "coverage")
    _require(coverage.get("schema") == COVERAGE_SCHEMA, "coverage_schema_mismatch")
    source = coverage.get("source")
    _require(isinstance(source, dict), "coverage_source_missing")
    _require(source.get("sha256") == raw_sha256, "coverage_raw_sha256_mismatch")
    grid = coverage.get("trace_grid")
    artifact = coverage.get("trace_artifact")
    alignment = coverage.get("reference_alignment")
    _require(isinstance(grid, dict), "coverage_trace_grid_missing")
    _require(isinstance(artifact, dict), "coverage_trace_artifact_missing")
    _require(isinstance(alignment, dict), "coverage_alignment_missing")
    _require(artifact.get("schema") == TRACE_SCHEMA, "coverage_trace_schema_mismatch")
    trace_path_text = artifact.get("path")
    _require(isinstance(trace_path_text, str) and trace_path_text, "coverage_trace_path_missing")
    trace_path = Path(trace_path_text).resolve()
    trace_sha = artifact.get("sha256")
    _verify_source_hash(trace_path, trace_sha, "trace")
    bone_sources = grid.get("bone_source")
    _require(isinstance(bone_sources, dict), "coverage_bone_source_missing")
    local = bone_sources.get(str(local_slot))
    _require(isinstance(local, dict), "coverage_local_bone_source_missing")
    return {
        "status": "verified",
        "coverage_path": str(coverage_path),
        "coverage_sha256": _sha256(coverage_path),
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha,
        "grid_rate_hz": grid.get("rate_hz"),
        "bone_sampling": grid.get("bone_sampling"),
        "local_slot": local_slot,
        "local_bone_packet_age_at_grid": {
            "maximum_source_age_ticks": local.get("maximum_source_age_ticks"),
            "maximum_source_age_seconds": local.get("maximum_source_age_seconds"),
            "observed_source_rate_hz": local.get("observed_source_rate_hz"),
            "fresh_grid_samples": local.get("fresh_grid_samples"),
            "reused_grid_samples": local.get("reused_grid_samples"),
            "semantic_limit": (
                "age from local packet receipt to the 50 Hz trace grid; not network age, "
                "server age, or action latency"
            ),
        },
        "npz_reference_alignment": {
            "status": alignment.get("status"),
            "direct_angle_identity_allowed": alignment.get("direct_angle_identity_allowed"),
            "required_transform": alignment.get("required_transform"),
        },
    }


def _asset_metadata(probe: dict[str, Any]) -> dict[str, Any]:
    move_identity = probe.get("move_identity")
    _require(isinstance(move_identity, dict), "probe_move_identity_missing")
    _require(
        move_identity.get("executed_move_identity") == "unknown",
        "probe_claims_executed_move_identity",
    )
    _require(
        move_identity.get("requested_asset_to_observed_pose_identity_proven") is False,
        "probe_claims_requested_asset_pose_identity",
    )
    config = move_identity.get("requested_asset_configuration")
    _require(isinstance(config, dict), "requested_asset_configuration_missing")
    runtime_name = config.get("runtime_name")
    npz_sha256 = config.get("npz_sha256")
    controller_ticks = config.get("recovered_controller_ticks")
    _require(isinstance(runtime_name, str) and runtime_name, "requested_asset_name_missing")
    _require(
        isinstance(npz_sha256, str)
        and len(npz_sha256) == 64
        and all(character in "0123456789abcdef" for character in npz_sha256),
        "requested_asset_npz_sha256_invalid",
    )
    _require(
        _is_int(controller_ticks) and controller_ticks > 0,
        "requested_asset_controller_ticks_invalid",
    )
    return {
        "runtime_name": runtime_name,
        "npz_sha256": npz_sha256,
        "recovered_controller_ticks": controller_ticks,
        "frame_count_timing_use": "forbidden",
        "executed_move_identity": "unknown",
        "alignment": {
            "status": "not_performed",
            "reasons": [
                "server acceptance is unknown",
                "executed move identity is unknown",
                "a measured NPZ-joint to received-bone transform is unavailable",
                "opponent motion and contact are not excluded",
            ],
        },
    }


def _build_probe(
    label: str,
    probe: dict[str, Any],
    edge: dict[str, Any],
    lifecycle_send: dict[str, Any] | None,
    raw_moves: dict[int, dict[str, Any]],
    raw_bones: dict[int, dict[str, Any]],
    change: dict[str, Any],
) -> dict[str, Any]:
    ordinal = probe.get("probe_ordinal")
    move_index = probe.get("requested_move_index")
    probe_kind = probe.get("probe_kind")
    _require(_is_int(ordinal) and ordinal >= 0, "probe_ordinal_invalid")
    _require(move_index in EXPECTED_MOVES, "probe_move_index_not_g1_kick")
    _require(probe_kind in EXPECTED_PROBE_KINDS, "probe_kind_invalid")

    local = probe.get("local_execute_move_observation")
    server = probe.get("server_action_acceptance")
    request_sent = probe.get("request_sent")
    physical = probe.get("physical_response")
    _require(isinstance(local, dict), "probe_local_observation_missing")
    _require(isinstance(server, dict), "probe_server_acceptance_missing")
    _require(isinstance(request_sent, dict), "probe_request_sent_missing")
    _require(isinstance(physical, dict), "probe_physical_response_missing")
    _require(local.get("server_acceptance") == "unknown", "local_claims_server_acceptance")
    _require(
        server.get("status") == "unknown"
        and server.get("value") is None
        and server.get("acknowledgement_observed") is False,
        "probe_claims_server_acceptance",
    )
    _require(
        physical.get("server_action_acceptance") == "unknown",
        "physical_response_claims_server_acceptance",
    )
    _require(
        physical.get("executed_move_identity") == "unknown",
        "physical_response_claims_move_identity",
    )

    edge_detail = edge.get("detail")
    _require(isinstance(edge_detail, dict), "transcript_edge_detail_missing")
    _require(edge_detail.get("kick_probe_ordinal") == ordinal, "edge_ordinal_mismatch")
    effective_vector = _finite_vector(
        edge_detail.get("effective_controller_vector_xyz"), 3, "effective_controller_vector"
    )
    desired_vector = _finite_vector(
        edge_detail.get("desired_raw_controller_target_xyz"), 3, "desired_controller_vector"
    )
    if probe_kind == "yaw_preempted":
        _require(edge_detail.get("yaw_preempted") is True, "yaw_probe_not_preempted")
        _require(effective_vector == [0.0, 0.0, 0.0], "yaw_probe_not_neutral_at_edge")
        normalized_condition = "neutral_after_yaw_preemption"
    else:
        _require(edge_detail.get("yaw_preempted") is False, "translation_probe_preempted")
        _require(
            abs(effective_vector[0]) > 0.0 or abs(effective_vector[1]) > 0.0,
            "translation_probe_not_held_at_edge",
        )
        normalized_condition = "translation_held_at_edge"

    schedule_edge = probe.get("schedule_edge")
    _require(isinstance(schedule_edge, dict), "probe_schedule_edge_missing")
    edge_substep = edge.get("client_fixed_substep")
    edge_fixed_time = _finite(edge.get("unity_fixed_time"), "edge_unity_fixed_time")
    _require(_is_int(edge_substep), "edge_fixed_substep_invalid")
    _require(schedule_edge.get("client_fixed_substep") == edge_substep, "report_edge_substep_mismatch")

    send_anchor = request_sent.get("send_prefix_anchor")
    raw_request: dict[str, Any] | None = None
    dispatch: dict[str, Any]
    recorder_sequence = request_sent.get("recorder_request_sequence")
    if isinstance(send_anchor, dict):
        send_substep = send_anchor.get("client_fixed_substep")
        _require(_is_int(send_substep) and send_substep >= edge_substep, "send_substep_invalid")
        send_fixed_time = _finite(
            send_anchor.get("unity_fixed_time"), "send_unity_fixed_time"
        )
        _require(send_fixed_time >= edge_fixed_time, "send_unity_fixed_time_precedes_edge")
        dispatch = {
            "status": "observed",
            "edge_client_fixed_substep": edge_substep,
            "send_prefix_client_fixed_substep": send_substep,
            "fixed_substeps": int(send_substep) - int(edge_substep),
            "nominal_seconds_at_500hz": (int(send_substep) - int(edge_substep))
            / UNITY_FIXED_RATE_HZ,
            "unity_fixed_time_delta_seconds": send_fixed_time - edge_fixed_time,
            "unity_frames": (
                int(send_anchor["unity_frame"]) - int(edge["unity_frame"])
                if _is_int(send_anchor.get("unity_frame")) and _is_int(edge.get("unity_frame"))
                else None
            ),
            "authority": "local_visual_client_schedule_and_SendMoveEvent_prefix",
            "physical_response": "not_observed_by_this_measurement",
        }
        if lifecycle_send is not None:
            lifecycle_detail = lifecycle_send.get("detail")
            _require(isinstance(lifecycle_detail, dict), "lifecycle_send_detail_missing")
            _require(
                lifecycle_detail.get("send_prefix_fixed_substep") == send_substep,
                "lifecycle_report_send_substep_mismatch",
            )
            dispatch["late_update_opportunities"] = lifecycle_detail.get(
                "late_update_opportunities"
            )
    else:
        dispatch = {
            "status": "send_prefix_not_observed",
            "fixed_substeps": None,
            "nominal_seconds_at_500hz": None,
            "unity_fixed_time_delta_seconds": None,
        }

    projection: dict[str, Any]
    if _is_int(recorder_sequence):
        raw_request = raw_moves[int(recorder_sequence)]
        _require(
            raw_request.get("move_index_source_int32") == move_index,
            "raw_request_move_index_mismatch",
        )
        projection = {
            "status": "observed_in_bridge_and_recorder",
            "recorder_request_sequence": recorder_sequence,
            "recorder_client_fixed_tick": raw_request.get(
                "client_fixed_tick_at_observation"
            ),
            "request_only": True,
            "wire_delivery": raw_request.get("wire_delivery"),
            "server_action_acceptance": "unknown",
            "acknowledgement_observed": False,
        }
        if isinstance(send_anchor, dict):
            frequency = send_anchor.get("qpc_frequency_hz")
            send_qpc = send_anchor.get("qpc_ticks")
            raw_qpc = raw_request.get("stopwatch_timestamp_ticks")
            _require(
                _is_int(frequency) and frequency > 0 and _is_int(send_qpc) and _is_int(raw_qpc),
                "projection_qpc_fields_invalid",
            )
            signed = (int(raw_qpc) - int(send_qpc)) / int(frequency)
            projection["recorder_minus_bridge_observer_qpc_seconds"] = signed
            projection["absolute_observer_skew_seconds"] = abs(signed)
            reported_delta = request_sent.get("qpc_anchor_delta_seconds")
            if reported_delta is not None:
                _require(
                    math.isclose(abs(signed), _finite(reported_delta, "reported_qpc_delta"), abs_tol=1e-12),
                    "projection_qpc_delta_mismatch",
                )
    else:
        projection = {
            "status": request_sent.get("status"),
            "recorder_request_sequence": None,
            "server_action_acceptance": "unknown",
            "acknowledgement_observed": False,
        }

    pose = physical.get("pose_departure")
    _require(isinstance(pose, dict), "pose_departure_missing")
    receipt = _pose_receipt_timing(pose, change, raw_bones, raw_request)
    return {
        "run": label,
        "probe_ordinal": ordinal,
        "probe_label": probe.get("probe_label"),
        "requested_move_index": move_index,
        "probe_kind": probe_kind,
        "normalized_control_condition": normalized_condition,
        "control_at_request_edge": {
            "phase": edge_detail.get("phase"),
            "desired_held": edge_detail.get("desired_held"),
            "effective_held": edge_detail.get("effective_held"),
            "desired_controller_vector_xyz": desired_vector,
            "effective_controller_vector_xyz": effective_vector,
            "yaw_preempted": edge_detail.get("yaw_preempted"),
        },
        "local_execute_move": {
            "returned": local.get("returned"),
            "classification": local.get("classification"),
            "authority": local.get("authority"),
            "server_action_acceptance": "unknown",
        },
        "local_arm_to_send_prefix": dispatch,
        "outbound_request_projection": projection,
        "received_pose_candidate": receipt,
        "combat_context": probe.get("combat_context"),
        "requested_asset": _asset_metadata(probe),
        "canonical_input_to_physical_response_latency": {
            "status": "unknown",
            "seconds": None,
        },
        "canonical_move_duration": {"status": "unknown", "seconds": None},
        "server_action_acceptance": {"status": "unknown", "value": None},
    }


def _load_run(spec: RunSpec) -> dict[str, Any]:
    _require(LABEL_PATTERN.fullmatch(spec.label) is not None, "run_label_invalid")
    report_path = spec.report_path.resolve()
    report = _read_json(report_path, "pose_report")
    _require(report.get("schema") == POSE_REPORT_SCHEMA, "pose_report_schema_mismatch")
    sources = report.get("sources")
    scope = report.get("scope")
    summary = report.get("summary")
    probes = report.get("kick_probes")
    limits = report.get("evidence_limits")
    _require(isinstance(sources, dict), "pose_report_sources_missing")
    _require(isinstance(scope, dict), "pose_report_scope_missing")
    _require(isinstance(summary, dict), "pose_report_summary_missing")
    _require(isinstance(probes, list), "pose_report_probes_missing")
    _require(isinstance(limits, dict), "pose_report_limits_missing")
    _require(limits.get("server_action_acceptance") == "unknown", "report_claims_acceptance")
    _require(limits.get("executed_move_identity") == "unknown", "report_claims_move_identity")
    _require(limits.get("canonical_move_duration") == "unknown", "report_claims_duration")
    _require(scope.get("runtime_model") == "g1", "report_scope_not_g1")
    _require(scope.get("exact_g1_vs_g1") is True, "report_scope_not_g1_vs_g1")
    _require(scope.get("solo_route_proven") is True, "report_scope_not_solo")
    local_slot = scope.get("local_fighter_index")
    _require(local_slot in (0, 1), "report_local_slot_invalid")

    transcript_source = sources.get("transcript")
    raw_source = sources.get("recorder")
    _require(isinstance(transcript_source, dict), "transcript_source_missing")
    _require(isinstance(raw_source, dict), "raw_source_missing")
    transcript_path = _path_from_source(
        spec.transcript_path, transcript_source, "transcript"
    )
    raw_path = _path_from_source(spec.raw_path, raw_source, "raw")
    transcript_sha = _verify_source_hash(
        transcript_path, transcript_source.get("sha256"), "transcript"
    )
    raw_sha = _verify_source_hash(raw_path, raw_source.get("sha256"), "raw")

    edges, sends = _scan_transcript(transcript_path, transcript_sha)
    changes: dict[int, dict[str, Any]] = {}
    request_sequences: set[int] = set()
    bone_sequences: set[int] = set()
    seen_ordinals: set[int] = set()
    for probe in probes:
        _require(isinstance(probe, dict), "probe_not_object")
        ordinal = probe.get("probe_ordinal")
        _require(_is_int(ordinal) and ordinal >= 0, "probe_ordinal_invalid")
        _require(ordinal not in seen_ordinals, "duplicate_probe_ordinal")
        seen_ordinals.add(int(ordinal))
        request = probe.get("request_sent")
        physical = probe.get("physical_response")
        _require(isinstance(request, dict), "probe_request_missing")
        _require(isinstance(physical, dict), "probe_physical_missing")
        sequence = request.get("recorder_request_sequence")
        if _is_int(sequence):
            request_sequences.add(int(sequence))
        pose = physical.get("pose_departure")
        _require(isinstance(pose, dict), "probe_pose_missing")
        change = _change_points(pose)
        changes[int(ordinal)] = change
        bone_sequences.update(change["target_sequences"])
    _require(set(edges) == seen_ordinals, "transcript_report_edge_set_mismatch")

    raw_moves, raw_bones, capture_start = _scan_raw(
        raw_path,
        raw_sha,
        request_sequences,
        bone_sequences,
        int(local_slot),
    )
    built = [
        _build_probe(
            spec.label,
            probe,
            edges[int(probe["probe_ordinal"])],
            sends.get(int(probe["probe_ordinal"])),
            raw_moves,
            raw_bones,
            changes[int(probe["probe_ordinal"])],
        )
        for probe in probes
    ]
    return {
        "label": spec.label,
        "complete": summary.get("transcript_complete") is True,
        "completion_reason": summary.get("transcript_reason"),
        "schedule_sha256": transcript_source.get("schedule_sha256"),
        "schedule_run_id": transcript_source.get("schedule_run_id"),
        "machine": raw_source.get("machine"),
        "pid": raw_source.get("pid"),
        "sources": {
            "pose_report": {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            },
            "transcript": {"path": str(transcript_path), "sha256": transcript_sha},
            "raw_recorder": {
                "path": str(raw_path),
                "sha256": raw_sha,
                "schema": raw_source.get("schema"),
                "stopwatch_frequency_hz": capture_start.get("stopwatch_frequency_hz"),
            },
        },
        "trace_resampling": _trace_summary(
            spec.coverage_path.resolve() if spec.coverage_path is not None else None,
            raw_sha,
            int(local_slot),
        ),
        "probes": built,
    }


def _interval_comparison(observations: list[dict[str, Any]]) -> dict[str, Any]:
    bounded = [
        observation
        for observation in observations
        if observation.get("lower") is not None and observation.get("upper") is not None
    ]
    if not bounded:
        return {
            "observation_count": len(observations),
            "bounded_observation_count": 0,
            "right_censored_observation_count": sum(
                1
                for observation in observations
                if observation.get("status")
                == "right_censored_in_client_receipt_domain"
            ),
            "hull": None,
            "intersection": None,
        }
    lowers = [_finite(value["lower"], "interval_lower") for value in bounded]
    uppers = [_finite(value["upper"], "interval_upper") for value in bounded]
    _require(all(lower <= upper for lower, upper in zip(lowers, uppers)), "interval_inverted")
    intersection_lower = max(lowers)
    intersection_upper = min(uppers)
    return {
        "observation_count": len(observations),
        "bounded_observation_count": len(bounded),
        "right_censored_observation_count": sum(
            1
            for observation in observations
            if observation.get("status") == "right_censored_in_client_receipt_domain"
        ),
        "hull": {"lower": min(lowers), "upper": max(uppers)},
        "intersection": {
            "status": (
                "nonempty" if intersection_lower <= intersection_upper else "disjoint"
            ),
            "lower": intersection_lower,
            "upper": intersection_upper,
        },
        "semantic_limit": "descriptive comparison of receipt-domain candidates, not an estimator",
    }


def _repeat_groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for run in runs:
        for probe in run["probes"]:
            vector = tuple(probe["control_at_request_edge"]["effective_controller_vector_xyz"])
            key = (
                run["schedule_sha256"],
                probe["requested_move_index"],
                probe["probe_kind"],
                vector,
                probe["requested_asset"].get("npz_sha256"),
                probe["requested_asset"].get("recovered_controller_ticks"),
            )
            grouped.setdefault(key, []).append((run, probe))

    output = []
    for key in sorted(grouped, key=lambda value: (value[1], value[2], value[3])):
        schedule_sha, move_index, probe_kind, vector, npz_sha, controller_ticks = key
        members = grouped[key]
        dispatch_values = [
            member["local_arm_to_send_prefix"].get("fixed_substeps")
            for _, member in members
            if member["local_arm_to_send_prefix"].get("fixed_substeps") is not None
        ]
        onset_observations = []
        duration_observations = []
        references = []
        for run, probe in members:
            received = probe["received_pose_candidate"]
            onset = received.get("request_to_first_candidate_receipt_bracket_seconds")
            duration = received.get(
                "candidate_departure_duration_receipt_domain_seconds"
            )
            references.append(
                {
                    "run": run["label"],
                    "complete_run": run["complete"],
                    "probe_ordinal": probe["probe_ordinal"],
                    "local_dispatch_fixed_substeps": probe[
                        "local_arm_to_send_prefix"
                    ].get("fixed_substeps"),
                    "candidate_onset": onset,
                    "candidate_duration": duration,
                    "combat_confounding_not_excluded": (
                        probe.get("combat_context") or {}
                    ).get("confounding_not_excluded"),
                }
            )
            if isinstance(onset, dict):
                onset_observations.append(onset)
            if isinstance(duration, dict):
                duration_observations.append(duration)
        output.append(
            {
                "schedule_sha256": schedule_sha,
                "requested_move_index": move_index,
                "probe_kind": probe_kind,
                "effective_controller_vector_xyz": list(vector),
                "requested_asset_npz_sha256": npz_sha,
                "requested_asset_recovered_controller_ticks": controller_ticks,
                "run_count": len(members),
                "complete_run_count": sum(1 for run, _ in members if run["complete"]),
                "observations": references,
                "local_arm_to_send_prefix_fixed_substeps": {
                    "status": "observed" if dispatch_values else "unavailable",
                    "values": dispatch_values,
                    "minimum": min(dispatch_values) if dispatch_values else None,
                    "maximum": max(dispatch_values) if dispatch_values else None,
                    "authority": "local visual-client dispatch only",
                },
                "first_candidate_pose_receipt": _interval_comparison(onset_observations),
                "candidate_pose_departure_duration": _interval_comparison(
                    duration_observations
                ),
                "canonical_timing": {
                    "input_to_physical_response_latency": "unknown",
                    "move_duration": "unknown",
                    "server_action_acceptance": "unknown",
                },
            }
        )
    return output


def analyze_runs(specs: Iterable[RunSpec]) -> dict[str, Any]:
    selected = list(specs)
    _require(selected, "no_runs_supplied")
    labels = [spec.label for spec in selected]
    _require(len(set(labels)) == len(labels), "duplicate_run_label")
    runs = [_load_run(spec) for spec in selected]
    schedules = sorted({run["schedule_sha256"] for run in runs})
    machines = sorted({str(run["machine"]) for run in runs})
    return {
        "schema": OUTPUT_SCHEMA,
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "cohort": {
            "run_count": len(runs),
            "complete_run_count": sum(1 for run in runs if run["complete"]),
            "schedule_sha256_values": schedules,
            "schedule_identity_consistent": len(schedules) == 1,
            "machines": machines,
            "repeat_comparability": (
                "same_schedule_request_and_client_receipt_domains_only"
                if len(schedules) == 1
                else "not_comparable_schedule_identity_differs"
            ),
        },
        "runs": runs,
        "repeat_groups": _repeat_groups(runs),
        "measurement_contract": {
            "local_arm_to_send_prefix": {
                "status": "measured_when_send_prefix_is_present",
                "authority": "bridge local schedule and SendMoveEvent prefix",
            },
            "recorder_request_projection": {
                "status": "measured_when_dual_observed",
                "authority": "recorder outbound REK_Move projection",
                "server_acceptance": "unknown",
            },
            "client_received_pose_candidate": {
                "status": "provisional_threshold_departure_only",
                "clock": "Unity Time.realtimeSinceStartupAsDouble at local receipt",
                "packet_source_timestamp_available": False,
                "causal_attribution": "unknown",
            },
            "input_to_first_physical_response_latency": {
                "status": "unknown",
                "value_seconds": None,
            },
            "canonical_move_duration": {"status": "unknown", "value_seconds": None},
            "server_action_acceptance": {"status": "unknown", "value": None},
            "npz_frame_count_as_duration": {
                "allowed": False,
                "reason": (
                    "requested asset identity is not observed at execution and frame count "
                    "does not establish acceptance, playback rate, onset, or completion"
                ),
            },
            "combat_confounding_excluded": False,
        },
    }


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary path {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _mapping(values: list[str], option: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        label, separator, path = value.partition("=")
        _require(bool(separator) and bool(path), f"{option}_requires_LABEL_PATH")
        _require(LABEL_PATTERN.fullmatch(label) is not None, f"{option}_label_invalid")
        _require(label not in output, f"{option}_duplicate_label")
        output[label] = Path(path)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare exact local G1 kick dispatch and client-receipt candidate timing"
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=POSE_REPORT.json",
        help="repeatable run label and g1_schedule_pose_response report",
    )
    parser.add_argument(
        "--transcript",
        action="append",
        default=[],
        metavar="LABEL=TRANSCRIPT.jsonl",
        help="optional source-path override for a run",
    )
    parser.add_argument(
        "--raw",
        action="append",
        default=[],
        metavar="LABEL=RECORDER.jsonl",
        help="optional source-path override for a run",
    )
    parser.add_argument(
        "--coverage",
        action="append",
        default=[],
        metavar="LABEL=COVERAGE.json",
        help="optional held-trace coverage artifact for a run",
    )
    parser.add_argument("--out", required=True, help="new compact JSON report path, or -")
    args = parser.parse_args(argv)
    try:
        runs = _mapping(args.run, "run")
        transcripts = _mapping(args.transcript, "transcript")
        raws = _mapping(args.raw, "raw")
        coverages = _mapping(args.coverage, "coverage")
        _require(runs, "no_runs_supplied")
        _require(set(transcripts) <= set(runs), "transcript_override_has_unknown_run")
        _require(set(raws) <= set(runs), "raw_override_has_unknown_run")
        _require(set(coverages) <= set(runs), "coverage_has_unknown_run")
        specs = [
            RunSpec(
                label=label,
                report_path=path,
                transcript_path=transcripts.get(label),
                raw_path=raws.get(label),
                coverage_path=coverages.get(label),
            )
            for label, path in runs.items()
        ]
        report = analyze_runs(specs)
        payload = _json_bytes(report)
        if args.out == "-":
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
        else:
            output = Path(args.out).resolve()
            digest = _write_atomic(output, payload)
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "output_sha256": digest,
                        "runs": len(report["runs"]),
                        "complete_runs": report["cohort"]["complete_run_count"],
                        "canonical_move_duration": "unknown",
                        "server_action_acceptance": "unknown",
                    },
                    sort_keys=True,
                )
            )
    except (KickTimingError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
