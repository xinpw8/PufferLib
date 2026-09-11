#!/usr/bin/env python3
"""Correlate a G1 held-input schedule transcript with recorder-v7 pose data.

This is an evidence analyzer, not a move recognizer.  It reports separately:

* the schedule's local ExecuteMove result;
* observation of the client SendMoveEvent prefix in the pipe transcript;
* an independently observed REK_Move projection in the recorder stream; and
* a post-anchor departure of the local fighter's received bone pose from its
  probe-local pre-anchor pose envelope.

None of those observations exposes a server acknowledgement.  Consequently
``server_action_acceptance`` and executed move identity always remain unknown.
Packet timestamps are client receipt timestamps, so detected onset/end brackets
are receipt-domain bounds and are never emitted as a canonical move duration.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import g1_held_trace_extract


REPORT_SCHEMA = "rek.g1_schedule_pose_response.v1"
SCHEDULE_SCHEMA = "rek.g1_held_input_schedule.v2"
UNITY_FIXED_RATE_HZ = 500
SCHEDULE_RATE_HZ = 50
FIXED_SUBSTEPS_PER_SCHEDULE_TICK = 10
EXPECTED_COMPLETE_SCHEDULE_TICKS = 4551
EXPECTED_HELD_CONDITIONS = 14
EXPECTED_HELD_TICKS_PER_CONDITION = 100
EXPECTED_KICK_PROBES = 8
EXPECTED_TRANSLATION_RELEASES = 4
DEFAULT_CLOCK_TOLERANCE_SECONDS = 0.0011
DEFAULT_SEND_MATCH_TOLERANCE_SECONDS = 0.050
DEFAULT_BASELINE_SECONDS = 0.8
DEFAULT_MINIMUM_BASELINE_PACKETS = 6
DEFAULT_MINIMUM_POST_PACKETS = 6
DEFAULT_MINIMUM_RMS_DEPARTURE_RADIANS = 0.08
DEFAULT_MINIMUM_MAX_JOINT_DEPARTURE_RADIANS = 0.25
DEFAULT_THRESHOLD_MAD_MULTIPLIER = 6.0
DEFAULT_THRESHOLD_MARGIN_RADIANS = 0.03
DEFAULT_CONSECUTIVE_PACKETS = 2


class PoseResponseError(ValueError):
    """Inputs cannot support a fail-closed correlation report."""


@dataclass
class Transcript:
    source_name: str
    sha256: str
    pipe_server_pid: int
    schema: str
    schedule_id: str
    schedule_sha256: str
    run_id: str
    ticks: list[dict[str, Any]]
    edges: dict[int, dict[str, Any]]
    releases: dict[int, dict[str, Any]]
    summaries: dict[int, dict[str, Any]]
    lifecycle: dict[int, list[dict[str, Any]]]
    end: dict[str, Any]
    event_count: int


@dataclass(frozen=True)
class DetectionConfig:
    baseline_ticks: int
    minimum_baseline_packets: int
    minimum_post_packets: int
    minimum_rms_radians: float
    minimum_max_joint_radians: float
    mad_multiplier: float
    threshold_margin_radians: float
    consecutive_packets: int


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PoseResponseError(reason)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _detail(record: dict[str, Any], context: str) -> dict[str, Any]:
    value = record.get("detail")
    _require(isinstance(value, dict), f"{context}_detail_missing")
    return value


def _read_jsonl(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], str, str]:
    source = Path(path)
    _require(source.is_file(), "transcript_not_a_file")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PoseResponseError(
                    f"transcript_invalid_json_line_{line_number}:{exc.msg}"
                ) from exc
            _require(isinstance(record, dict), f"transcript_line_{line_number}_not_object")
            records.append(record)
    _require(records, "transcript_empty")
    return records, digest.hexdigest(), str(source.resolve())


def read_transcript(path: str | os.PathLike[str]) -> Transcript:
    records, digest, source_name = _read_jsonl(path)
    pipe_proofs = [
        record for record in records if record.get("event") == "client_pipe_server_proof"
    ]
    _require(len(pipe_proofs) == 1, "client_pipe_server_proof_count_mismatch")
    pipe_server_pid = pipe_proofs[0].get("process_id")
    _require(_is_int(pipe_server_pid) and pipe_server_pid > 0, "pipe_server_pid_invalid")
    emitted: list[dict[str, Any]] = []
    end: dict[str, Any] | None = None
    identity: tuple[str, str, str, str] | None = None
    expected_sequence = 1
    ticks: list[dict[str, Any]] = []
    edges: dict[int, dict[str, Any]] = {}
    releases: dict[int, dict[str, Any]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    lifecycle: dict[int, list[dict[str, Any]]] = {}

    for record in records:
        event = record.get("event")
        if event == "g1_held_schedule_end":
            _require(end is None, "duplicate_schedule_end")
            end = record
            continue
        if not _is_int(record.get("event_sequence")):
            if (
                isinstance(event, str)
                and event.startswith("g1_")
                and event not in {
                    "g1_held_input_schedule_started",
                    "g1_held_input_schedule_stopped",
                }
            ):
                raise PoseResponseError("sequenced_schedule_event_missing_sequence")
            continue
        _require(isinstance(event, str) and event.startswith("g1_"), "sequenced_non_g1_event")
        schema = record.get("g1_held_schedule_schema")
        schedule_id = record.get("g1_held_schedule_id")
        schedule_sha = record.get("g1_held_schedule_sha256")
        run_id = record.get("g1_held_schedule_run_id")
        _require(schema == SCHEDULE_SCHEMA, "unsupported_schedule_schema")
        _require(isinstance(schedule_id, str) and schedule_id, "schedule_id_missing")
        _require(_is_sha256(schedule_sha), "schedule_sha256_invalid")
        _require(schedule_sha != "0" * 64, "schedule_sha256_unsealed")
        _require(_is_hex(run_id, 32), "schedule_run_id_invalid")
        current_identity = (schema, schedule_id, schedule_sha, run_id)
        if identity is None:
            identity = current_identity
        _require(current_identity == identity, "schedule_identity_changed")
        _require(record["event_sequence"] == expected_sequence, "event_sequence_not_contiguous")
        expected_sequence += 1
        _require(record.get("request_only") is True, "schedule_event_not_request_only")
        _require(record.get("server_acceptance") == "unknown", "schedule_claims_server_acceptance")
        _require(record.get("server_acceptance_observed") is False, "schedule_claims_server_ack")
        _require(
            record.get("authoritative_execution_observed") is False,
            "schedule_claims_authoritative_execution",
        )
        _require(record.get("global_input_emitted") is False, "schedule_emitted_global_input")
        _require(record.get("unity_fixed_rate_hz") == UNITY_FIXED_RATE_HZ, "fixed_rate_mismatch")
        _require(record.get("schedule_rate_hz") == SCHEDULE_RATE_HZ, "schedule_rate_mismatch")
        _require(
            record.get("fixed_substeps_per_schedule_tick")
            == FIXED_SUBSTEPS_PER_SCHEDULE_TICK,
            "schedule_stride_mismatch",
        )
        _require(_is_number(record.get("unity_fixed_time")), "schedule_unity_fixed_time_missing")
        _require(_is_int(record.get("client_fixed_substep")), "schedule_substep_missing")
        emitted.append(record)

        if event == "g1_held_schedule_tick":
            detail = _detail(record, "schedule_tick")
            expected_tick = len(ticks)
            _require(record.get("schedule_tick") == expected_tick, "schedule_ticks_not_contiguous")
            _require(
                record.get("client_fixed_substep")
                == expected_tick * FIXED_SUBSTEPS_PER_SCHEDULE_TICK,
                "schedule_tick_substep_mismatch",
            )
            ticks.append(record)
            if detail.get("kick_edge") is True:
                ordinal = detail.get("kick_probe_ordinal")
                _require(_is_int(ordinal) and ordinal >= 0, "kick_edge_ordinal_missing")
                _require(ordinal not in edges, "duplicate_kick_edge")
                edges[ordinal] = record
        elif event == "g1_kick_measurement_summary":
            detail = _detail(record, "kick_summary")
            ordinal = detail.get("probe_ordinal")
            _require(_is_int(ordinal) and ordinal >= 0, "kick_summary_ordinal_missing")
            _require(ordinal not in summaries, "duplicate_kick_summary")
            summaries[ordinal] = record
        elif event == "g1_translation_release":
            detail = _detail(record, "translation_release")
            ordinal = detail.get("probe_ordinal")
            _require(_is_int(ordinal) and ordinal >= 0, "translation_release_ordinal_missing")
            _require(ordinal not in releases, "duplicate_translation_release")
            releases[ordinal] = record
        elif event == "g1_kick_request_lifecycle":
            detail = _detail(record, "kick_lifecycle")
            ordinal = detail.get("probe_ordinal")
            _require(_is_int(ordinal) and ordinal >= 0, "kick_lifecycle_ordinal_missing")
            lifecycle.setdefault(ordinal, []).append(record)

    _require(identity is not None, "no_sequenced_schedule_events")
    _require(ticks, "schedule_tick_stream_missing")
    _require(end is not None, "schedule_end_missing")
    schema, schedule_id, schedule_sha, run_id = identity
    _require(end.get("g1_held_schedule_schema") == schema, "end_schema_mismatch")
    _require(end.get("g1_held_schedule_id") == schedule_id, "end_schedule_id_mismatch")
    _require(end.get("g1_held_schedule_sha256") == schedule_sha, "end_schedule_sha_mismatch")
    _require(end.get("g1_held_schedule_run_id") == run_id, "end_run_id_mismatch")
    _require(end.get("request_only") is True, "end_not_request_only")
    _require(end.get("server_acceptance") == "unknown", "end_claims_server_acceptance")
    _require(end.get("server_acceptance_observed") is False, "end_claims_server_ack")
    _require(end.get("authoritative_execution_observed") is False, "end_claims_execution")
    _require(end.get("global_input_emitted") is False, "end_global_input_emitted")
    _require(end.get("complete") in (True, False), "end_complete_missing")
    _require(end.get("partial_coverage") is (not end["complete"]), "end_partial_flag_mismatch")
    if end["complete"]:
        _require(end.get("experiment_coverage_complete") is True, "complete_end_coverage_false")
        _require(end.get("reason") == "complete", "complete_end_reason_mismatch")
        _require(end.get("schedule_tick") == len(ticks) - 1, "complete_end_tick_mismatch")
        _require(
            len(ticks) == EXPECTED_COMPLETE_SCHEDULE_TICKS,
            "complete_schedule_tick_count_mismatch",
        )
        _require(
            sorted(edges) == list(range(EXPECTED_KICK_PROBES)),
            "complete_schedule_kick_edge_count_mismatch",
        )
        _require(
            sorted(summaries) == list(range(EXPECTED_KICK_PROBES)),
            "complete_schedule_summary_count_mismatch",
        )
        _require(
            sorted(releases) == list(range(EXPECTED_TRANSLATION_RELEASES)),
            "complete_schedule_translation_release_count_mismatch",
        )
        held_counts: dict[int, int] = {}
        for record in ticks:
            ordinal = _detail(record, "complete_schedule_tick").get(
                "held_condition_ordinal"
            )
            if _is_int(ordinal):
                held_counts[int(ordinal)] = held_counts.get(int(ordinal), 0) + 1
        _require(
            held_counts
            == {
                ordinal: EXPECTED_HELD_TICKS_PER_CONDITION
                for ordinal in range(EXPECTED_HELD_CONDITIONS)
            },
            "complete_schedule_held_condition_counts_mismatch",
        )

    return Transcript(
        source_name=source_name,
        sha256=digest,
        pipe_server_pid=int(pipe_server_pid),
        schema=schema,
        schedule_id=schedule_id,
        schedule_sha256=schedule_sha,
        run_id=run_id,
        ticks=ticks,
        edges=edges,
        releases=releases,
        summaries=summaries,
        lifecycle=lifecycle,
        end=end,
        event_count=len(emitted),
    )


class RootClock:
    def __init__(self, roots: dict[int, dict[str, Any]], tolerance_seconds: float):
        _require(roots, "root_clock_empty")
        self.roots = roots
        self.ticks = sorted(roots)
        self.times = [float(roots[tick]["unity_fixed_time"]) for tick in self.ticks]
        _require(all(math.isfinite(value) for value in self.times), "root_clock_nonfinite")
        _require(
            all(right > left for left, right in zip(self.times, self.times[1:])),
            "root_fixed_times_not_strictly_increasing",
        )
        self.tolerance_seconds = tolerance_seconds

    def nearest(self, fixed_time: Any) -> tuple[int, float]:
        _require(_is_number(fixed_time), "event_fixed_time_missing")
        target = float(fixed_time)
        insertion = bisect.bisect_left(self.times, target)
        candidates = []
        if insertion < len(self.times):
            candidates.append(insertion)
        if insertion > 0:
            candidates.append(insertion - 1)
        _require(candidates, "event_outside_root_clock")
        selected = min(candidates, key=lambda index: abs(self.times[index] - target))
        residual = abs(self.times[selected] - target)
        _require(residual <= self.tolerance_seconds, "event_root_clock_residual_exceeded")
        return self.ticks[selected], residual


def correlate_clock(
    transcript: Transcript,
    capture: g1_held_trace_extract.Capture,
    tolerance_seconds: float,
) -> tuple[RootClock, dict[int, int], dict[str, Any]]:
    clock = RootClock(capture.roots, tolerance_seconds)
    mapped: dict[int, int] = {}
    residuals: list[float] = []
    offsets: list[int] = []
    previous_raw_tick: int | None = None
    for record in transcript.ticks:
        schedule_tick = int(record["schedule_tick"])
        raw_tick, residual = clock.nearest(record["unity_fixed_time"])
        if previous_raw_tick is not None:
            _require(raw_tick > previous_raw_tick, "mapped_schedule_ticks_not_increasing")
        previous_raw_tick = raw_tick
        mapped[schedule_tick] = raw_tick
        residuals.append(residual)
        offsets.append(raw_tick - int(record["client_fixed_substep"]))
    _require(max(offsets) - min(offsets) <= 1, "schedule_recorder_fixed_clock_offset_drift")
    return clock, mapped, {
        "basis": "shared_Unity_Time.fixedTimeAsDouble",
        "schedule_clock": "client_fixed_substep_500hz_relative_to_schedule_start",
        "recorder_clock": "client_fixed_tick_500hz_relative_to_capture_start",
        "mapped_schedule_tick_count": len(mapped),
        "raw_tick_minus_schedule_substep_min": min(offsets),
        "raw_tick_minus_schedule_substep_max": max(offsets),
        "maximum_fixed_time_residual_seconds": max(residuals),
        "configured_maximum_residual_seconds": tolerance_seconds,
        "status": "correlated",
    }


def _normalize_quaternion(values: list[float]) -> tuple[float, float, float, float]:
    _require(len(values) == 4, "quaternion_shape")
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    _require(math.isfinite(norm) and norm > 1e-9, "quaternion_not_normalizable")
    return tuple(float(value) / norm for value in values)  # type: ignore[return-value]


def _yaw_radians(values: list[float]) -> float:
    x, y, z, w = _normalize_quaternion(values)
    forward_x = 2.0 * (x * z + w * y)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(forward_x, forward_z)


def _unwrap(values: Iterable[float]) -> list[float]:
    output: list[float] = []
    for value in values:
        if not output:
            output.append(value)
            continue
        candidate = value
        while candidate - output[-1] > math.pi:
            candidate -= 2.0 * math.pi
        while candidate - output[-1] < -math.pi:
            candidate += 2.0 * math.pi
        output.append(candidate)
    return output


def _planar_path(roots: list[dict[str, Any]], slot: int) -> float:
    positions = [root[f"fighter_{slot}_root"]["world_position_xyz"] for root in roots]
    return sum(
        math.hypot(right[0] - left[0], right[2] - left[2])
        for left, right in zip(positions, positions[1:])
    )


def _held_kind(names: list[str]) -> str:
    translation = any(name in {"W", "S", "A", "D"} for name in names)
    yaw = any(name in {"Q", "E"} for name in names)
    return (
        "translation_plus_yaw"
        if translation and yaw
        else "translation" if translation else "yaw"
    )


def build_held_trajectories(
    transcript: Transcript,
    capture: g1_held_trace_extract.Capture,
    mapped_ticks: dict[int, int],
) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in transcript.ticks:
        ordinal = _detail(record, "schedule_tick").get("held_condition_ordinal")
        if _is_int(ordinal):
            groups.setdefault(int(ordinal), []).append(record)

    reports: list[dict[str, Any]] = []
    for ordinal in sorted(groups):
        records = groups[ordinal]
        first_detail = _detail(records[0], "held_condition")
        names = first_detail.get("desired_held")
        _require(
            isinstance(names, list) and all(isinstance(v, str) for v in names),
            "held_names_invalid",
        )
        first_schedule_tick = int(records[0]["schedule_tick"])
        last_schedule_tick = int(records[-1]["schedule_tick"])
        start_raw_tick = mapped_ticks[first_schedule_tick]
        boundary_schedule_tick = last_schedule_tick + 1
        stop_raw_tick = mapped_ticks.get(boundary_schedule_tick, mapped_ticks[last_schedule_tick])
        dense_roots = [
            capture.roots[tick]
            for tick in range(start_raw_tick, stop_raw_tick + 1)
            if tick in capture.roots
        ]
        _require(len(dense_roots) >= 2, "held_dense_root_window_too_short")
        slot = capture.local_slot
        positions = [root[f"fighter_{slot}_root"]["world_position_xyz"] for root in dense_roots]
        yaws = _unwrap(
            _yaw_radians(root[f"fighter_{slot}_root"]["world_rotation_xyzw"])
            for root in dense_roots
        )
        elapsed = float(dense_roots[-1]["unity_fixed_time"]) - float(
            dense_roots[0]["unity_fixed_time"]
        )
        _require(elapsed > 0.0, "held_elapsed_not_positive")
        dx = positions[-1][0] - positions[0][0]
        dz = positions[-1][2] - positions[0][2]
        heading = yaws[0]
        unity_forward_displacement = dx * math.sin(heading) + dz * math.cos(heading)
        unity_right_displacement = dx * math.cos(heading) - dz * math.sin(heading)
        input_requests = [
            request
            for request in capture.input_requests
            if start_raw_tick <= request["tick"] <= stop_raw_tick
        ]
        grid_samples = []
        for record in records:
            schedule_tick = int(record["schedule_tick"])
            root = capture.roots[mapped_ticks[schedule_tick]]
            pose = root[f"fighter_{slot}_root"]
            detail = _detail(record, "held_condition_tick")
            grid_samples.append(
                {
                    "schedule_tick": schedule_tick,
                    "client_fixed_substep": record["client_fixed_substep"],
                    "recorder_client_fixed_tick": mapped_ticks[schedule_tick],
                    "unity_fixed_time": root["unity_fixed_time"],
                    "effective_controller_vector_xyz": detail.get(
                        "effective_controller_vector_xyz"
                    ),
                    "desired_raw_controller_target_xyz": detail.get(
                        "desired_raw_controller_target_xyz"
                    ),
                    "root_world_position_xyz": pose["world_position_xyz"],
                    "root_yaw_radians": _yaw_radians(pose["world_rotation_xyzw"]),
                }
            )
        reports.append(
            {
                "ordinal": ordinal,
                "label": "+".join(names),
                "kind": _held_kind(names),
                "desired_held": names,
                "schedule_tick_start_inclusive": first_schedule_tick,
                "schedule_tick_stop_exclusive": boundary_schedule_tick,
                "observed_schedule_ticks": len(records),
                "recorder_tick_start": start_raw_tick,
                "recorder_tick_stop": stop_raw_tick,
                "elapsed_seconds": elapsed,
                "root_trajectory": {
                    "authority": "client_observed_visual_root_transform",
                    "dense_root_sample_count": len(dense_roots),
                    "planar_path_m": _planar_path(dense_roots, slot),
                    "planar_net_displacement_xyz_m": [dx, positions[-1][1] - positions[0][1], dz],
                    "planar_net_speed_m_s": math.hypot(dx, dz) / elapsed,
                    "unity_local_plus_z_projection_m": unity_forward_displacement,
                    "unity_local_plus_x_projection_m": unity_right_displacement,
                    "projection_semantics": (
                        "mathematical projection onto the initial root quaternion's "
                        "Unity +Z/+X axes; "
                        "it is not an inferred controller-axis calibration"
                    ),
                    "root_yaw_change_radians": yaws[-1] - yaws[0],
                    "root_y_min_m": min(position[1] for position in positions),
                    "root_y_max_m": max(position[1] for position in positions),
                },
                "outbound_input_projection": {
                    "request_count": len(input_requests),
                    "request_only": True,
                    "server_acceptance_available": False,
                    "first_velocity_command_xyz": (
                        input_requests[0]["velocity_command_xyz"] if input_requests else None
                    ),
                    "last_velocity_command_xyz": (
                        input_requests[-1]["velocity_command_xyz"] if input_requests else None
                    ),
                    "source_client_fixed_ticks": [request["tick"] for request in input_requests],
                    "requests": [
                        {
                            "request_sequence": request["request_sequence"],
                            "recorder_client_fixed_tick": request["tick"],
                            "velocity_command_xyz": request["velocity_command_xyz"],
                        }
                        for request in input_requests
                    ],
                },
                "schedule_grid_samples": grid_samples,
            }
        )
    return reports


def _quaternion_angle(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _pose(decoded: dict[str, Any]) -> tuple[tuple[float, float, float, float], ...]:
    values = decoded["child_local_rotations_xyzw"]
    _require(len(values) == 4 * len(g1_held_trace_extract.G1_BONE_NAMES), "decoded_pose_shape")
    output = []
    # The pelvis occupies index zero and is represented by a zero local quaternion.
    for bone_index in range(1, len(g1_held_trace_extract.G1_BONE_NAMES)):
        offset = 4 * bone_index
        output.append(_normalize_quaternion(values[offset : offset + 4]))
    return tuple(output)


def _pose_distance(
    left: tuple[tuple[float, float, float, float], ...],
    right: tuple[tuple[float, float, float, float], ...],
) -> tuple[float, float]:
    _require(len(left) == len(right) and left, "pose_distance_shape")
    angles = [_quaternion_angle(a, b) for a, b in zip(left, right)]
    return math.sqrt(sum(value * value for value in angles) / len(angles)), max(angles)


def _percentile(values: list[float], fraction: float) -> float:
    _require(values, "percentile_empty")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _nearest_pose_distance(
    pose: tuple[tuple[float, float, float, float], ...],
    references: list[tuple[tuple[float, float, float, float], ...]],
) -> tuple[float, float, int]:
    _require(references, "pose_reference_empty")
    distances = [_pose_distance(pose, reference) for reference in references]
    index = min(range(len(distances)), key=lambda value: distances[value][0])
    return distances[index][0], distances[index][1], index


def detect_pose_departure(
    capture: g1_held_trace_extract.Capture,
    anchor_tick: int,
    stop_tick: int,
    config: DetectionConfig,
) -> dict[str, Any]:
    packets = capture.bone_packets[capture.local_slot]
    baseline_packets = [
        packet
        for packet in packets
        if anchor_tick - config.baseline_ticks <= packet["tick"] < anchor_tick
    ]
    post_packets = [packet for packet in packets if anchor_tick <= packet["tick"] <= stop_tick]
    if (
        len(baseline_packets) < config.minimum_baseline_packets
        or len(post_packets) < config.minimum_post_packets
    ):
        return {
            "status": "insufficient_bone_packet_coverage",
            "pose_departure_observed": None,
            "baseline_packet_count": len(baseline_packets),
            "post_anchor_packet_count": len(post_packets),
            "required_baseline_packets": config.minimum_baseline_packets,
            "required_post_anchor_packets": config.minimum_post_packets,
            "server_action_acceptance": "unknown",
            "executed_move_identity": "unknown",
            "canonical_move_duration": {"status": "unknown", "seconds": None},
        }

    baseline_poses = [
        _pose(capture.decoded_snapshots[packet["sequence"]]) for packet in baseline_packets
    ]
    baseline_nn_rms: list[float] = []
    for index, pose in enumerate(baseline_poses):
        other = baseline_poses[:index] + baseline_poses[index + 1 :]
        rms, _, _ = _nearest_pose_distance(pose, other)
        baseline_nn_rms.append(rms)
    median = statistics.median(baseline_nn_rms)
    mad = statistics.median(abs(value - median) for value in baseline_nn_rms)
    baseline_p95 = _percentile(baseline_nn_rms, 0.95)
    rms_threshold = max(
        config.minimum_rms_radians,
        baseline_p95
        + max(config.threshold_margin_radians, config.mad_multiplier * 1.4826 * mad),
    )

    samples = []
    above_flags = []
    for packet in post_packets:
        pose = _pose(capture.decoded_snapshots[packet["sequence"]])
        rms, maximum, reference_index = _nearest_pose_distance(pose, baseline_poses)
        above = rms >= rms_threshold and maximum >= config.minimum_max_joint_radians
        above_flags.append(above)
        samples.append(
            {
                "recorder_client_fixed_tick": packet["tick"],
                "seconds_from_anchor": (packet["tick"] - anchor_tick) / UNITY_FIXED_RATE_HZ,
                "raw_bone_packet_sequence": packet["sequence"],
                "nearest_baseline_packet_sequence": baseline_packets[reference_index]["sequence"],
                "pose_novelty_rms_radians": rms,
                "pose_novelty_max_joint_radians": maximum,
                "above_departure_threshold": above,
            }
        )

    onset_index: int | None = None
    streak = 0
    for index, above in enumerate(above_flags):
        streak = streak + 1 if above else 0
        if streak >= config.consecutive_packets:
            onset_index = index - config.consecutive_packets + 1
            break

    end_index: int | None = None
    if onset_index is not None:
        streak = 0
        for index in range(onset_index + config.consecutive_packets, len(above_flags)):
            streak = streak + 1 if not above_flags[index] else 0
            if streak >= config.consecutive_packets:
                end_index = index - config.consecutive_packets + 1
                break

    onset_bracket = None
    return_bracket = None
    receipt_duration = {"status": "unknown", "lower_seconds": None, "upper_seconds": None}
    if onset_index is not None:
        onset_tick = post_packets[onset_index]["tick"]
        previous_tick = (
            baseline_packets[-1]["tick"]
            if onset_index == 0
            else post_packets[onset_index - 1]["tick"]
        )
        onset_bracket = {
            "last_non_departed_receipt_tick": previous_tick,
            "first_departed_receipt_tick": onset_tick,
            "lower_seconds_from_anchor": max(0, previous_tick - anchor_tick) / UNITY_FIXED_RATE_HZ,
            "upper_seconds_from_anchor": (onset_tick - anchor_tick) / UNITY_FIXED_RATE_HZ,
        }
        if end_index is not None:
            return_tick = post_packets[end_index]["tick"]
            prior_tick = post_packets[end_index - 1]["tick"]
            return_bracket = {
                "last_departed_receipt_tick": prior_tick,
                "first_returned_receipt_tick": return_tick,
                "lower_seconds_from_anchor": (prior_tick - anchor_tick) / UNITY_FIXED_RATE_HZ,
                "upper_seconds_from_anchor": (return_tick - anchor_tick) / UNITY_FIXED_RATE_HZ,
            }
            receipt_duration = {
                "status": "bounded_in_client_receipt_domain",
                "lower_seconds": max(0, prior_tick - onset_tick) / UNITY_FIXED_RATE_HZ,
                "upper_seconds": max(0, return_tick - max(anchor_tick, previous_tick))
                / UNITY_FIXED_RATE_HZ,
            }
        else:
            receipt_duration = {
                "status": "right_censored_at_observation_stop",
                "lower_seconds": max(0, post_packets[-1]["tick"] - onset_tick)
                / UNITY_FIXED_RATE_HZ,
                "upper_seconds": None,
            }

    tick_gaps = [
        right["tick"] - left["tick"] for left, right in zip(post_packets, post_packets[1:])
    ]
    return {
        "status": (
            "observed_pose_departure"
            if onset_index is not None
            else "no_pose_departure_observed"
        ),
        "pose_departure_observed": onset_index is not None,
        "authority": "local_slot_decoded_REK_Bones_received_from_server",
        "baseline_window": {
            "start_recorder_client_fixed_tick": anchor_tick - config.baseline_ticks,
            "stop_recorder_client_fixed_tick_exclusive": anchor_tick,
            "packet_count": len(baseline_packets),
            "nearest_neighbor_rms_median_radians": median,
            "nearest_neighbor_rms_p95_radians": baseline_p95,
            "nearest_neighbor_rms_mad_radians": mad,
        },
        "detector": {
            "method": "nearest_pre_anchor_pose_envelope_quaternion_geodesic",
            "articulated_joint_count": len(g1_held_trace_extract.G1_BONE_NAMES) - 1,
            "rms_departure_threshold_radians": rms_threshold,
            "minimum_rms_floor_radians": config.minimum_rms_radians,
            "minimum_max_joint_departure_radians": config.minimum_max_joint_radians,
            "required_consecutive_received_packets": config.consecutive_packets,
            "baseline_threshold_rule": "max(floor,p95+max(margin,6_scaled_MAD))",
            "classifier_status": "provisional_conservative_detector_not_a_move_identity_model",
        },
        "packet_coverage": {
            "post_anchor_packet_count": len(post_packets),
            "maximum_post_anchor_receipt_gap_ticks": max(tick_gaps) if tick_gaps else None,
            "maximum_post_anchor_receipt_gap_seconds": (
                max(tick_gaps) / UNITY_FIXED_RATE_HZ if tick_gaps else None
            ),
            "wire_delivery": "unreliable",
            "source_timestamps_available": False,
        },
        "onset_receipt_bracket": onset_bracket,
        "return_receipt_bracket": return_bracket,
        "pose_departure_window_receipt_domain": receipt_duration,
        "samples": samples,
        "server_action_acceptance": "unknown",
        "causal_attribution_to_requested_move": "unknown",
        "executed_move_identity": "unknown",
        "canonical_move_duration": {
            "status": "unknown",
            "seconds": None,
            "reason": (
                "REK_Bones is unreliable and contains no source timestamp, move identity, "
                "server tick, acknowledgement, or acceptance field"
            ),
        },
    }


def _send_anchor(
    summary: dict[str, Any] | None,
    lifecycle: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if summary is not None:
        detail = _detail(summary, "kick_summary")
        anchor = detail.get("send_prefix_anchor")
        if isinstance(anchor, dict):
            return anchor
    for record in lifecycle:
        detail = _detail(record, "kick_lifecycle")
        if detail.get("lifecycle_stage") == "send_move_invoked":
            return {
                "client_fixed_substep": detail.get("send_prefix_fixed_substep"),
                "schedule_tick": detail.get("send_prefix_schedule_tick"),
                "unity_frame": detail.get("send_prefix_unity_frame"),
                "unity_fixed_time": detail.get("send_prefix_unity_fixed_time"),
                "qpc_ticks": detail.get("send_prefix_qpc_ticks"),
                "qpc_frequency_hz": detail.get("send_prefix_qpc_frequency_hz"),
            }
    return None


def _transcript_send_observed(
    summary: dict[str, Any] | None,
    lifecycle: list[dict[str, Any]],
) -> bool:
    if summary is not None:
        value = _detail(summary, "kick_summary").get("move_send_invoked")
        if isinstance(value, bool):
            return value
    return any(
        _detail(record, "kick_lifecycle").get("lifecycle_stage") == "send_move_invoked"
        for record in lifecycle
    )


def _match_raw_move(
    capture: g1_held_trace_extract.Capture,
    move_index: int,
    anchor: dict[str, Any] | None,
    used_sequences: set[int],
    tolerance_seconds: float,
) -> tuple[dict[str, Any] | None, float | None]:
    if anchor is None or not _is_int(anchor.get("qpc_ticks")):
        return None, None
    frequency = anchor.get("qpc_frequency_hz")
    capture_frequency = capture.start.get("stopwatch_frequency_hz")
    if frequency is None:
        # A partial transcript may end after the lifecycle send-prefix event but
        # before its summary copies Stopwatch.Frequency into send_prefix_anchor.
        # Both instruments are already bound to the same REK PID, so the
        # recorder's capture-start frequency is the only supported clock source
        # for that partial-run correlation.
        frequency = capture_frequency
    _require(_is_int(frequency) and frequency > 0, "send_anchor_qpc_frequency_invalid")
    _require(
        capture_frequency == frequency,
        "schedule_recorder_qpc_frequency_mismatch",
    )
    candidates = [
        request
        for request in capture.move_requests
        if request["move_index"] == move_index
        and request["request_sequence"] not in used_sequences
        and _is_int(request.get("stopwatch_timestamp_ticks"))
    ]
    if not candidates:
        return None, None
    selected = min(
        candidates,
        key=lambda request: abs(request["stopwatch_timestamp_ticks"] - anchor["qpc_ticks"]),
    )
    delta = abs(selected["stopwatch_timestamp_ticks"] - anchor["qpc_ticks"]) / frequency
    if delta > tolerance_seconds:
        return None, delta
    used_sequences.add(selected["request_sequence"])
    return selected, delta


def _read_combat_context(raw_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    events = []
    with Path(raw_path).open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PoseResponseError(f"raw_context_invalid_json_line_{line_number}") from exc
            if record.get("event") not in {
                "raw_fight_state_packet",
                "raw_score_packet",
                "raw_hit_packet",
            }:
                continue
            tick = record.get("client_fixed_tick_at_observation")
            _require(_is_int(tick), "combat_context_tick_missing")
            events.append(
                {
                    "event": record["event"],
                    "recorder_client_fixed_tick": tick,
                    "decoded": record.get("decoded"),
                }
            )
    return events


def build_attack_reports(
    transcript: Transcript,
    capture: g1_held_trace_extract.Capture,
    clock: RootClock,
    mapped_ticks: dict[int, int],
    combat_events: list[dict[str, Any]],
    config: DetectionConfig,
    send_match_tolerance_seconds: float,
) -> list[dict[str, Any]]:
    reports = []
    used_sequences: set[int] = set()
    for ordinal in sorted(transcript.edges):
        edge = transcript.edges[ordinal]
        edge_detail = _detail(edge, "kick_edge")
        summary = transcript.summaries.get(ordinal)
        release = transcript.releases.get(ordinal)
        lifecycle = transcript.lifecycle.get(ordinal, [])
        summary_detail = _detail(summary, "kick_summary") if summary is not None else {}
        release_detail = _detail(release, "translation_release") if release is not None else {}
        move_index = summary_detail.get("move_index", edge_detail.get("move_index"))
        if not _is_int(move_index):
            for record in lifecycle:
                candidate = _detail(record, "kick_lifecycle").get("move_index")
                if _is_int(candidate):
                    move_index = candidate
                    break
        _require(_is_int(move_index), "kick_move_index_missing")
        anchor = _send_anchor(summary, lifecycle)
        transcript_send = _transcript_send_observed(summary, lifecycle)
        if transcript_send:
            _require(anchor is not None, "transcript_send_has_no_prefix_anchor")
        raw_move, qpc_delta = _match_raw_move(
            capture,
            int(move_index),
            anchor,
            used_sequences,
            send_match_tolerance_seconds,
        )
        if transcript_send and raw_move is not None:
            sent_status = "observed_in_transcript_and_recorder"
            request_sent: bool | None = True
            analysis_anchor_tick = raw_move["tick"]
        elif transcript_send:
            sent_status = "transcript_send_unmatched_in_recorder"
            request_sent = None
            analysis_anchor_tick = mapped_ticks[int(edge["schedule_tick"])]
        else:
            sent_status = "not_observed_in_transcript"
            request_sent = False
            analysis_anchor_tick = mapped_ticks[int(edge["schedule_tick"])]

        if summary is not None:
            stop_tick, _ = clock.nearest(summary["unity_fixed_time"])
        else:
            stop_tick = min(max(capture.roots), analysis_anchor_tick + 4 * UNITY_FIXED_RATE_HZ)
        _require(stop_tick > analysis_anchor_tick, "kick_observation_window_not_positive")
        pose = detect_pose_departure(capture, analysis_anchor_tick, stop_tick, config)
        context = [
            event
            for event in combat_events
            if analysis_anchor_tick <= event["recorder_client_fixed_tick"] <= stop_tick
        ]
        context_counts = {
            name: sum(1 for event in context if event["event"] == name)
            for name in ("raw_fight_state_packet", "raw_score_packet", "raw_hit_packet")
        }
        if request_sent is not True:
            response_status = "not_attributable_without_dual_observed_send"
        elif pose.get("pose_departure_observed") is True:
            response_status = "candidate_post_send_pose_departure_observed"
        elif pose.get("pose_departure_observed") is False:
            response_status = "no_post_send_pose_departure_observed"
        else:
            response_status = "insufficient_pose_coverage"

        execute_return = summary_detail.get("execute_move_returned")
        local_classification = summary_detail.get("local_classification")
        if summary is None:
            for record in lifecycle:
                detail = _detail(record, "kick_lifecycle")
                if detail.get("lifecycle_stage") == "execute_move_returned":
                    execute_return = detail.get("execute_move_returned")
                    local_classification = detail.get("local_classification")
                    break
        reports.append(
            {
                "probe_ordinal": ordinal,
                "probe_label": summary_detail.get("probe_label", edge_detail.get("probe_label")),
                "probe_kind": summary_detail.get("probe_kind"),
                "requested_move_index": move_index,
                "schedule_edge": {
                    "schedule_tick": edge["schedule_tick"],
                    "client_fixed_substep": edge["client_fixed_substep"],
                    "recorder_client_fixed_tick": mapped_ticks[int(edge["schedule_tick"])],
                    "unity_fixed_time": edge["unity_fixed_time"],
                },
                "local_execute_move_observation": {
                    "returned": execute_return,
                    "classification": local_classification,
                    "authority": "visual_only_client_local_diagnostic",
                    "server_acceptance": "unknown",
                },
                "translation_gate_diagnostics": {
                    "applicable": summary_detail.get("probe_kind") == "translation_held",
                    "release_event_observed": release is not None,
                    "translation_release_tick": summary_detail.get(
                        "translation_release_tick", release_detail.get("release_tick")
                    ),
                    "translation_release_fixed_substep": summary_detail.get(
                        "translation_release_fixed_substep",
                        release_detail.get("release_fixed_substep"),
                    ),
                    "translation_release_qpc_ticks": summary_detail.get(
                        "translation_release_qpc_ticks",
                        release_detail.get("release_qpc_ticks"),
                    ),
                    "translation_first_transition_settled_fixed_substep": (
                        summary_detail.get(
                            "translation_first_transition_settled_fixed_substep"
                        )
                    ),
                    "translation_fixed_substeps_release_to_settled": (
                        summary_detail.get(
                            "translation_fixed_substeps_release_to_settled"
                        )
                    ),
                    "request_send_timing_classification": summary_detail.get(
                        "translation_request_send_timing_classification"
                    ),
                    "release_settle_diagnostic": summary_detail.get(
                        "translation_release_settle_diagnostic",
                        release_detail.get("transition_settled_diagnostic"),
                    ),
                    "last_settle_diagnostic": summary_detail.get(
                        "translation_last_settle_diagnostic"
                    ),
                    "post_release_settled_kick_control_included": summary_detail.get(
                        "translation_post_release_settled_kick_control_included"
                    ),
                    "remaining_unknown": summary_detail.get(
                        "translation_post_release_settled_kick_remaining_unknown"
                    ),
                    "authority": "visual_only_client_local_diagnostic",
                    "physical_behavior": "unknown",
                },
                "request_sent": {
                    "status": sent_status,
                    "value": request_sent,
                    "transcript_send_prefix_observed": transcript_send,
                    "send_prefix_anchor": anchor,
                    "recorder_REK_Move_projection_observed": raw_move is not None,
                    "recorder_request_sequence": (
                        raw_move["request_sequence"] if raw_move is not None else None
                    ),
                    "recorder_client_fixed_tick": (
                        raw_move["tick"] if raw_move is not None else None
                    ),
                    "qpc_anchor_delta_seconds": qpc_delta,
                    "match_tolerance_seconds": send_match_tolerance_seconds,
                    "wire_delivery": "reliable" if raw_move is not None else None,
                    "semantic_limit": (
                        "method invocation and projected request body only; delivery, "
                        "server acceptance, "
                        "and execution are not observed"
                    ),
                },
                "physical_response": {
                    "status": response_status,
                    "pose_departure": pose,
                    "causal_attribution_to_requested_move": "unknown",
                    "server_action_acceptance": "unknown",
                    "executed_move_identity": "unknown",
                },
                "combat_context": {
                    "window_start_recorder_client_fixed_tick": analysis_anchor_tick,
                    "window_stop_recorder_client_fixed_tick": stop_tick,
                    "event_counts": context_counts,
                    "hit_fighter_identity_available": False,
                    "confounding_not_excluded": True,
                    "reason": (
                        "Bot motion and contact continue during the probe; REK_Hit has "
                        "no fighter identity "
                        "and unreliable delivery"
                    ),
                },
                "server_action_acceptance": {
                    "status": "unknown",
                    "value": None,
                    "acknowledgement_observed": False,
                    "reason": "neither schedule nor recorder exposes a server acceptance field",
                },
                "move_identity": {
                    "requested_move_index": move_index,
                    "requested_asset_configuration": summary_detail.get(
                        "requested_move_asset"
                    ),
                    "executed_move_identity": "unknown",
                    "requested_asset_to_observed_pose_identity_proven": False,
                },
                "duration": {
                    "canonical_move_duration_status": "unknown",
                    "canonical_move_duration_seconds": None,
                    "receipt_domain_pose_departure_window": pose.get(
                        "pose_departure_window_receipt_domain"
                    ),
                },
            }
        )
    return reports


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


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
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_data(
    transcript: Transcript,
    capture: g1_held_trace_extract.Capture,
    combat_events: list[dict[str, Any]],
    *,
    clock_tolerance_seconds: float = DEFAULT_CLOCK_TOLERANCE_SECONDS,
    send_match_tolerance_seconds: float = DEFAULT_SEND_MATCH_TOLERANCE_SECONDS,
    config: DetectionConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = DetectionConfig(
            baseline_ticks=round(DEFAULT_BASELINE_SECONDS * UNITY_FIXED_RATE_HZ),
            minimum_baseline_packets=DEFAULT_MINIMUM_BASELINE_PACKETS,
            minimum_post_packets=DEFAULT_MINIMUM_POST_PACKETS,
            minimum_rms_radians=DEFAULT_MINIMUM_RMS_DEPARTURE_RADIANS,
            minimum_max_joint_radians=DEFAULT_MINIMUM_MAX_JOINT_DEPARTURE_RADIANS,
            mad_multiplier=DEFAULT_THRESHOLD_MAD_MULTIPLIER,
            threshold_margin_radians=DEFAULT_THRESHOLD_MARGIN_RADIANS,
            consecutive_packets=DEFAULT_CONSECUTIVE_PACKETS,
        )
    _require(clock_tolerance_seconds > 0, "invalid_clock_tolerance")
    _require(send_match_tolerance_seconds > 0, "invalid_send_tolerance")
    _require(config.baseline_ticks > 0, "invalid_baseline_ticks")
    _require(config.minimum_baseline_packets >= 2, "invalid_minimum_baseline_packets")
    _require(config.minimum_post_packets >= 1, "invalid_minimum_post_packets")
    _require(config.minimum_rms_radians > 0, "invalid_minimum_rms")
    _require(config.minimum_max_joint_radians > 0, "invalid_minimum_max_joint")
    _require(config.mad_multiplier >= 0, "invalid_mad_multiplier")
    _require(config.threshold_margin_radians >= 0, "invalid_threshold_margin")
    _require(config.consecutive_packets >= 1, "invalid_consecutive_packets")
    capture_pid = capture.start.get("pid")
    _require(_is_int(capture_pid) and capture_pid > 0, "recorder_pid_invalid")
    _require(capture_pid == transcript.pipe_server_pid, "transcript_recorder_pid_mismatch")

    clock, mapped, clock_report = correlate_clock(
        transcript, capture, clock_tolerance_seconds
    )
    held = build_held_trajectories(transcript, capture, mapped)
    attacks = build_attack_reports(
        transcript,
        capture,
        clock,
        mapped,
        combat_events,
        config,
        send_match_tolerance_seconds,
    )
    dual_send_count = sum(
        1 for report in attacks if report["request_sent"]["value"] is True
    )
    unmatched_send_count = sum(
        1
        for report in attacks
        if report["request_sent"]["status"] == "transcript_send_unmatched_in_recorder"
    )
    pose_observed_count = sum(
        1
        for report in attacks
        if report["physical_response"]["pose_departure"].get("pose_departure_observed") is True
    )
    no_send_departure_ordinals = [
        report["probe_ordinal"]
        for report in attacks
        if report["request_sent"]["value"] is not True
        and report["physical_response"]["pose_departure"].get(
            "pose_departure_observed"
        )
        is True
    ]
    matched_raw_request_sequences = {
        report["request_sent"]["recorder_request_sequence"]
        for report in attacks
        if report["request_sent"]["recorder_request_sequence"] is not None
    }
    schedule_raw_start = mapped[min(mapped)]
    schedule_raw_stop = mapped[max(mapped)]
    unmatched_raw_moves = [
        {
            "request_sequence": request["request_sequence"],
            "move_index": request["move_index"],
            "recorder_client_fixed_tick": request["tick"],
        }
        for request in capture.move_requests
        if schedule_raw_start <= request["tick"] <= schedule_raw_stop
        and request["request_sequence"] not in matched_raw_request_sequences
    ]
    pose_coverage_complete = all(
        report["physical_response"]["pose_departure"].get("pose_departure_observed")
        is not None
        for report in attacks
    )
    structural_correlation_complete = (
        transcript.end["complete"]
        and len(held) == EXPECTED_HELD_CONDITIONS
        and len(attacks) == EXPECTED_KICK_PROBES
        and unmatched_send_count == 0
        and not unmatched_raw_moves
        and pose_coverage_complete
    )
    return {
        "schema": REPORT_SCHEMA,
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(__file__),
            "pose_detector": "probe_local_nearest_pose_envelope_v1",
        },
        "sources": {
            "transcript": {
                "path": transcript.source_name,
                "sha256": transcript.sha256,
                "schedule_schema": transcript.schema,
                "schedule_id": transcript.schedule_id,
                "schedule_sha256": transcript.schedule_sha256,
                "schedule_run_id": transcript.run_id,
                "pipe_server_pid": transcript.pipe_server_pid,
            },
            "recorder": {
                "path": capture.source_name,
                "sha256": capture.raw_sha256,
                "schema": capture.start.get("schema"),
                "plugin_version": capture.start.get("plugin_version"),
                "plugin_sha256": capture.start.get("plugin_sha256"),
                "machine": capture.start.get("machine"),
                "pid": capture_pid,
            },
        },
        "scope": {
            "runtime_model": "g1",
            "exact_g1_vs_g1": True,
            "solo_route_proven": True,
            "sparring_bot_number": 1,
            "human_in_opponent_slot": False,
            "local_fighter_index": capture.local_slot,
            "opponent_slot": capture.opponent_slot,
        },
        "clock_correlation": clock_report,
        "held_input_trajectories": held,
        "kick_probes": attacks,
        "summary": {
            "transcript_complete": transcript.end["complete"],
            "transcript_reason": transcript.end.get("reason"),
            "sequenced_schedule_events": transcript.event_count,
            "schedule_ticks": len(transcript.ticks),
            "held_conditions_observed": len(held),
            "kick_edges_observed": len(transcript.edges),
            "kick_summaries_observed": len(transcript.summaries),
            "translation_release_events_observed": len(transcript.releases),
            "dual_instrument_send_observations": dual_send_count,
            "transcript_sends_unmatched_in_recorder": unmatched_send_count,
            "post_anchor_pose_departures_observed": pose_observed_count,
            "pose_departures_without_dual_observed_send": len(
                no_send_departure_ordinals
            ),
            "raw_move_projections_not_matched_to_transcript": len(unmatched_raw_moves),
        },
        "negative_control_diagnostics": {
            "probe_ordinals_with_pose_departure_but_no_dual_observed_send": (
                no_send_departure_ordinals
            ),
            "count": len(no_send_departure_ordinals),
            "interpretation": (
                "any listed probe demonstrates that this pose-departure detector is not a "
                "causal move-response classifier"
            ),
        },
        "unmatched_raw_move_projections": unmatched_raw_moves,
        "evidence_limits": {
            "request_sent_requires": (
                "schedule SendMoveEvent prefix plus matching recorder REK_Move "
                "projection on shared QPC"
            ),
            "server_action_acceptance": "unknown",
            "server_acknowledgement_available": False,
            "authoritative_execution_observed": False,
            "executed_move_identity": "unknown",
            "canonical_move_duration": "unknown",
            "pose_timing_domain": "client_receive_boundary_only",
            "pose_response_is_causal_proof": False,
            "negative_control_result": (
                "pose departures can occur without REK_Move because locomotion, Bot motion, "
                "contact, and yaw-preemption transitions remain in the observed stream"
            ),
            "joint_angle_identity_assumed": False,
        },
        "analysis_gate": {
            "status": (
                "structurally_complete_with_authority_limits"
                if structural_correlation_complete
                else "incomplete"
            ),
            "structural_correlation_complete": structural_correlation_complete,
            "pose_packet_coverage_complete": pose_coverage_complete,
            "suitable_for_held_root_trajectory_measurement": bool(held),
            "suitable_for_request_pose_correlation": (
                dual_send_count > 0 and pose_coverage_complete
            ),
            "suitable_for_server_acceptance_claim": False,
            "suitable_for_canonical_move_identity_claim": False,
            "suitable_for_canonical_move_duration_claim": False,
        },
    }


def analyze(
    transcript_path: str | os.PathLike[str],
    raw_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    transcript = read_transcript(transcript_path)
    capture = g1_held_trace_extract.read_capture(raw_path)
    combat_events = _read_combat_context(raw_path)
    report = analyze_data(transcript, capture, combat_events, **kwargs)
    if output_path is not None:
        payload = _json_bytes(report)
        if str(output_path) == "-":
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
        else:
            _write_atomic(Path(output_path), payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correlate a G1 held schedule with recorder-v7 root/bone evidence"
    )
    parser.add_argument(
        "--transcript", required=True, help="completed g1-held pipe transcript JSONL"
    )
    parser.add_argument("--raw", required=True, help="completed recorder-v7 JSONL")
    parser.add_argument("--out", required=True, help="new report JSON path, or - for stdout")
    parser.add_argument(
        "--clock-tolerance-seconds",
        type=float,
        default=DEFAULT_CLOCK_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--send-match-tolerance-seconds",
        type=float,
        default=DEFAULT_SEND_MATCH_TOLERANCE_SECONDS,
    )
    parser.add_argument("--baseline-seconds", type=float, default=DEFAULT_BASELINE_SECONDS)
    parser.add_argument(
        "--minimum-baseline-packets",
        type=int,
        default=DEFAULT_MINIMUM_BASELINE_PACKETS,
    )
    parser.add_argument(
        "--minimum-post-packets", type=int, default=DEFAULT_MINIMUM_POST_PACKETS
    )
    parser.add_argument(
        "--minimum-rms-radians",
        type=float,
        default=DEFAULT_MINIMUM_RMS_DEPARTURE_RADIANS,
    )
    parser.add_argument(
        "--minimum-max-joint-radians",
        type=float,
        default=DEFAULT_MINIMUM_MAX_JOINT_DEPARTURE_RADIANS,
    )
    parser.add_argument(
        "--consecutive-packets", type=int, default=DEFAULT_CONSECUTIVE_PACKETS
    )
    args = parser.parse_args(argv)
    if not _is_number(args.baseline_seconds) or args.baseline_seconds <= 0:
        parser.error("--baseline-seconds must be finite and positive")
    config = DetectionConfig(
        baseline_ticks=round(args.baseline_seconds * UNITY_FIXED_RATE_HZ),
        minimum_baseline_packets=args.minimum_baseline_packets,
        minimum_post_packets=args.minimum_post_packets,
        minimum_rms_radians=args.minimum_rms_radians,
        minimum_max_joint_radians=args.minimum_max_joint_radians,
        mad_multiplier=DEFAULT_THRESHOLD_MAD_MULTIPLIER,
        threshold_margin_radians=DEFAULT_THRESHOLD_MARGIN_RADIANS,
        consecutive_packets=args.consecutive_packets,
    )
    try:
        report = analyze(
            args.transcript,
            args.raw,
            args.out,
            clock_tolerance_seconds=args.clock_tolerance_seconds,
            send_match_tolerance_seconds=args.send_match_tolerance_seconds,
            config=config,
        )
        if args.out != "-":
            output = Path(args.out).resolve()
            print(
                json.dumps(
                    {
                        "analysis_gate": report["analysis_gate"]["status"],
                        "output": str(output),
                        "output_sha256": _file_sha256(output),
                    },
                    sort_keys=True,
                )
            )
    except (PoseResponseError, g1_held_trace_extract.HeldTraceError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
