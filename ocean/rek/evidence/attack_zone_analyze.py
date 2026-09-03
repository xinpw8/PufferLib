#!/usr/bin/env python3
"""Deterministic, fail-closed analysis for REK attack-zone trial evidence.

This program analyzes client request edges and local observations. It never
converts those observations into server acceptance, authoritative execution,
or confirmed collision causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ANALYZER_SCHEMA = "rek.attack_zone_analyzer.v1"
PROTOCOL_SCHEMA = "rek.t800_private_sparring_bot_1.attack_zone_mapping_protocol.v1"
ANALYSIS_SPEC_SCHEMA = "rek.t800_private_sparring_bot_1.attack_zone_analysis_spec.v1"
RUNNER_SCHEMA = "rek.attack_zone_trial.v1"
SCHEDULE_SCHEMA = "rek.attack_zone_schedule.v1"
RECORDER_SCHEMA = "rek.private_ai.protocol.v6"
RECORDER_VERSION = "0.6.1"
RECORDER_SHA256 = "24cbea0a149589b71c093e989f43b8dac4862e73d103c323f0f9472a38355e0b"
AUTHORITY_SCOPE = "client_request_edges_and_local_observations_only"
REQUIRED_ISOLATION_PROOF = "wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98"
T800_BONE_SIGNATURE_SHA256 = "ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab"
ALLOWED_MOVES = (2, 3, 4, 5, 9, 10)
HEX = frozenset("0123456789abcdef")
SETTLE_TICKS = 15
FIXED_SUBSTEPS_PER_CONTROL_TICK = 10
MAX_SETTLE_STOPWATCH_INTERVAL_S = 0.040
LOCAL_SPEED_LIMIT_M_S = 0.15
YAW_RATE_LIMIT_RAD_S = 0.30
BEARING_ERROR_LIMIT_DEG = 3.0
MIN_RUNS = 5
MIN_TRIALS = 20
MAX_TRIALS = 40
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED_HEX = "e42ae683b98b29af5f3ab8997b7e8eb450f2a07425b4c7047d814563de17e20f"
BOOTSTRAP_ALGORITHM = "sha256_counter_two_stage_run_then_trial_resample_rejection_v1"
MOTION_STRATA = (
    "stationary", "closing", "receding", "tangential", "turning",
    "compound_or_unknown", "unresolved",
)

BODY_REGIONS: dict[str, tuple[str, ...]] = {
    "head": ("LINK_HEAD_PITCH", "LINK_HEAD_YAW"),
    "torso": ("LINK_BASE", "LINK_WAIST_YAW"),
    "left_arm": (
        "LINK_SHOULDER_PITCH_L", "LINK_SHOULDER_ROLL_L",
        "LINK_SHOULDER_YAW_L", "LINK_ELBOW_PITCH_L", "LINK_ELBOW_YAW_L",
    ),
    "right_arm": (
        "LINK_SHOULDER_PITCH_R", "LINK_SHOULDER_ROLL_R",
        "LINK_SHOULDER_YAW_R", "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R",
    ),
    "left_leg": (
        "LINK_HIP_PITCH_L", "LINK_HIP_ROLL_L", "LINK_HIP_YAW_L",
        "LINK_KNEE_PITCH_L", "LINK_ANKLE_PITCH_L", "LINK_ANKLE_ROLL_L",
    ),
    "right_leg": (
        "LINK_HIP_PITCH_R", "LINK_HIP_ROLL_R", "LINK_HIP_YAW_R",
        "LINK_KNEE_PITCH_R", "LINK_ANKLE_PITCH_R", "LINK_ANKLE_ROLL_R",
    ),
}
T800_BONE_NAMES = (
    "LINK_BASE",
    "LINK_HIP_PITCH_L", "LINK_HIP_ROLL_L", "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L", "LINK_ANKLE_PITCH_L", "LINK_ANKLE_ROLL_L",
    "LINK_HIP_PITCH_R", "LINK_HIP_ROLL_R", "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R", "LINK_ANKLE_PITCH_R", "LINK_ANKLE_ROLL_R",
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_PITCH_L", "LINK_SHOULDER_ROLL_L", "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L", "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_PITCH_R", "LINK_SHOULDER_ROLL_R", "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R",
    "LINK_HEAD_PITCH", "LINK_HEAD_YAW",
)
BONE_TO_REGION = {
    bone: region for region, bones in BODY_REGIONS.items() for bone in bones
}


class AnalysisFailure(ValueError):
    """A stable fail-closed rejection with a machine-readable code."""

    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}:{detail}")


def fail(code: str, detail: str | None = None) -> None:
    raise AnalysisFailure(code, detail)


def require(condition: bool, code: str, detail: str | None = None) -> None:
    if not condition:
        fail(code, detail)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        fail("non_canonical_json_value", type(exc).__name__)
    return (rendered + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in HEX for c in value)


def finite_number(value: Any, code: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    require(math.isfinite(number), code)
    return number


def integer(value: Any, code: str, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), code)
    if minimum is not None:
        require(value >= minimum, code)
    return value


def mapping(value: Any, code: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), code)
    return value


def sequence(value: Any, code: str) -> Sequence[Any]:
    require(isinstance(value, list), code)
    return value


def string(value: Any, code: str) -> str:
    require(isinstance(value, str) and value != "", code)
    return value


def required(parent: Mapping[str, Any], name: str, code: str | None = None) -> Any:
    require(name in parent, code or f"missing_{name}")
    return parent[name]


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail("json_read_failed", f"{path}:{type(exc).__name__}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                require(isinstance(value, dict), "jsonl_record_not_object", f"{path}:{line_number}")
                copied = dict(value)
                copied["_analysis_source_path"] = str(path.resolve())
                copied["_analysis_source_line"] = line_number
                records.append(copied)
    except AnalysisFailure:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail("jsonl_read_failed", f"{path}:{type(exc).__name__}")
    return records


def verify_file_hash(path: Path, expected: str) -> dict[str, Any]:
    require(path.is_file(), "input_file_missing", str(path))
    require(valid_sha256(expected), "expected_sha256_invalid", str(path))
    actual = sha256_path(path)
    require(actual == expected, "input_sha256_mismatch", str(path))
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def scan_authority_flags(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"server_acceptance_observed", "authoritative_execution_observed"}:
                require(child is False, "authority_flag_not_false", child_path)
            scan_authority_flags(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_authority_flags(child, f"{path}[{index}]")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol_schema_mismatch")
    require(protocol.get("live_rek_interaction_performed") is False, "protocol_live_flag_not_false")
    require(protocol.get("bridge_source_modified") is False, "protocol_bridge_edit_flag_not_false")
    boundary = mapping(required(protocol, "claim_boundary"), "claim_boundary_invalid")
    for name in (
        "server_acceptance_observed",
        "authoritative_execution_observed",
        "authoritative_hit_attribution_available",
        "authoritative_completion_available",
    ):
        require(boundary.get(name) is False, "protocol_authority_boundary_not_false", name)
    profiles = sequence(required(protocol, "move_profiles"), "move_profiles_invalid")
    moves = tuple(integer(mapping(item, "move_profile_invalid").get("move_index"), "move_index_invalid") for item in profiles)
    require(moves == ALLOWED_MOVES, "move_profile_set_mismatch")
    _protocol_grid(protocol)
    scan_authority_flags(protocol)


def _protocol_grid(protocol: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    grid = mapping(required(protocol, "grid"), "protocol_grid_missing")
    distances = [mapping(item, "protocol_distance_bin_invalid") for item in sequence(
        required(grid, "distance_bins"), "protocol_distance_bins_missing",
    )]
    bearings = [mapping(item, "protocol_bearing_bin_invalid") for item in sequence(
        required(grid, "bearing_bins"), "protocol_bearing_bins_missing",
    )]
    require(len(distances) == 9 and len(bearings) == 11, "protocol_grid_shape_mismatch")
    for kind, bins, suffix in (("distance", distances, "m"), ("bearing", bearings, "deg")):
        ids: set[str] = set()
        for value in bins:
            identifier = string(required(value, "id"), f"protocol_{kind}_bin_id_invalid")
            require(identifier not in ids, f"protocol_{kind}_bin_id_duplicate", identifier)
            ids.add(identifier)
            lower = finite_number(required(value, f"lower_{suffix}"), f"protocol_{kind}_lower_invalid")
            upper = finite_number(required(value, f"upper_{suffix}"), f"protocol_{kind}_upper_invalid")
            require(lower < upper, f"protocol_{kind}_bounds_invalid", identifier)
            require(isinstance(value.get("lower_inclusive"), bool), f"protocol_{kind}_inclusivity_invalid")
            require(isinstance(value.get("upper_inclusive"), bool), f"protocol_{kind}_inclusivity_invalid")
    assignments = sequence(required(grid, "per_move_grid_assignments"), "protocol_grid_assignments_missing")
    assignment_moves = tuple(mapping(item, "protocol_grid_assignment_invalid").get("move_index") for item in assignments)
    require(assignment_moves == ALLOWED_MOVES, "protocol_grid_assignment_moves_mismatch")
    return distances, bearings


def _protocol_cells(protocol: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    distances, bearings = _protocol_grid(protocol)
    return [
        (move, str(distance["id"]), str(bearing["id"]))
        for move in ALLOWED_MOVES
        for distance in distances
        for bearing in bearings
    ]


def validate_analysis_spec(spec: Mapping[str, Any]) -> None:
    require(spec.get("schema") == ANALYSIS_SPEC_SCHEMA, "analysis_spec_schema_mismatch")
    tests = sequence(required(spec, "conformance_tests_required_before_real_analysis"), "analysis_spec_tests_invalid")
    require(len(tests) == 19, "analysis_spec_test_count_mismatch")
    final_status = mapping(required(spec, "final_status_rule"), "analysis_spec_status_invalid")
    require(final_status.get("empirical_map_available") is False, "analysis_spec_premature_map_flag")
    scan_authority_flags(spec)


def wrap_to_180(degrees: float) -> float:
    require(math.isfinite(degrees), "bearing_nonfinite")
    wrapped = math.fmod(degrees, 360.0)
    if wrapped >= 180.0:
        wrapped -= 360.0
    if wrapped < -180.0:
        wrapped += 360.0
    return wrapped


def native_facing_yaw(bearing_degrees: float) -> float:
    bearing = finite_number(bearing_degrees, "bearing_nonfinite")
    absolute = abs(bearing)
    if absolute <= 17.5:
        return 0.0
    magnitude = min(absolute / 45.0, 1.0) * 1.5
    return -magnitude if bearing > 0.0 else magnitude


def bin_contains(bin_value: Mapping[str, Any], value: float, suffix: str) -> bool:
    lower = finite_number(required(bin_value, f"lower_{suffix}"), "bin_lower_invalid")
    upper = finite_number(required(bin_value, f"upper_{suffix}"), "bin_upper_invalid")
    require(lower < upper, "bin_bounds_invalid")
    lower_inc = required(bin_value, "lower_inclusive")
    upper_inc = required(bin_value, "upper_inclusive")
    require(isinstance(lower_inc, bool) and isinstance(upper_inc, bool), "bin_inclusivity_invalid")
    return (value > lower or lower_inc and value == lower) and (
        value < upper or upper_inc and value == upper
    )


def central_interval(bin_value: Mapping[str, Any], suffix: str) -> tuple[float, float]:
    lower = finite_number(required(bin_value, f"lower_{suffix}"), "bin_lower_invalid")
    upper = finite_number(required(bin_value, f"upper_{suffix}"), "bin_upper_invalid")
    require(lower < upper, "bin_bounds_invalid")
    width = upper - lower
    return lower + width / 4.0, upper - width / 4.0


def normalize_requested_bin(value: Mapping[str, Any], kind: str) -> dict[str, Any]:
    require(kind in {"distance", "bearing"}, "bin_kind_invalid")
    suffix = "m" if kind == "distance" else "deg"
    lower_name = f"lower_{suffix}"
    upper_name = f"upper_{suffix}"
    center_name = f"center_{suffix}"
    snake_keys = {"id", lower_name, upper_name, "lower_inclusive", "upper_inclusive", center_name}
    pascal_keys = {"Id", "Lower", "Upper", "LowerInclusive", "UpperInclusive", "Center"}
    keys = set(value)
    if keys == snake_keys:
        normalized = dict(value)
    elif keys == pascal_keys:
        normalized = {
            "id": value["Id"],
            lower_name: value["Lower"],
            upper_name: value["Upper"],
            "lower_inclusive": value["LowerInclusive"],
            "upper_inclusive": value["UpperInclusive"],
            center_name: value["Center"],
        }
    else:
        fail("requested_bin_shape_invalid", kind)
    string(normalized["id"], "bin_id_invalid")
    lower = finite_number(normalized[lower_name], "bin_lower_invalid")
    upper = finite_number(normalized[upper_name], "bin_upper_invalid")
    center = finite_number(normalized[center_name], "bin_center_invalid")
    require(lower < upper and center == lower + (upper - lower) / 2.0, "bin_center_or_bounds_invalid")
    require(isinstance(normalized["lower_inclusive"], bool), "bin_inclusivity_invalid")
    require(isinstance(normalized["upper_inclusive"], bool), "bin_inclusivity_invalid")
    return normalized


def validate_clock(clock: Mapping[str, Any]) -> None:
    integer(required(clock, "stopwatch_timestamp_ticks"), "clock_stopwatch_invalid", 0)
    integer(required(clock, "stopwatch_frequency_hz"), "clock_frequency_invalid", 1)
    string(required(clock, "utc"), "clock_utc_invalid")
    integer(required(clock, "unity_frame"), "clock_unity_frame_invalid", 0)
    finite_number(required(clock, "unity_time"), "clock_unity_time_invalid")
    finite_number(required(clock, "unity_fixed_time"), "clock_unity_fixed_time_invalid")
    integer(required(clock, "client_control_tick"), "clock_control_tick_invalid", 0)
    integer(required(clock, "client_fixed_substep"), "clock_substep_invalid", 0)


def event_clock(event: Mapping[str, Any]) -> dict[str, Any]:
    clock = {
        "stopwatch_timestamp_ticks": required(event, "stopwatch_timestamp_ticks"),
        "stopwatch_frequency_hz": required(event, "stopwatch_frequency_hz"),
        "utc": required(event, "utc"),
        "unity_frame": required(event, "unity_frame"),
        "unity_time": required(event, "unity_time"),
        "unity_fixed_time": required(event, "unity_fixed_time"),
        "client_control_tick": required(event, "client_control_tick"),
        "client_fixed_substep": required(event, "client_fixed_substep"),
    }
    validate_clock(clock)
    return clock


def seconds_between(start: Mapping[str, Any], end: Mapping[str, Any]) -> float:
    sf = integer(required(start, "stopwatch_frequency_hz"), "clock_frequency_invalid", 1)
    ef = integer(required(end, "stopwatch_frequency_hz"), "clock_frequency_invalid", 1)
    require(sf == ef, "clock_frequency_changed")
    delta = integer(required(end, "stopwatch_timestamp_ticks"), "clock_stopwatch_invalid", 0) - integer(
        required(start, "stopwatch_timestamp_ticks"), "clock_stopwatch_invalid", 0
    )
    require(delta >= 0, "negative_duration")
    return delta / sf


def validate_consecutive_clocks(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    validate_clock(previous)
    validate_clock(current)
    require(current["client_control_tick"] == previous["client_control_tick"] + 1, "settle_control_tick_gap")
    require(
        current["client_fixed_substep"] == previous["client_fixed_substep"] + FIXED_SUBSTEPS_PER_CONTROL_TICK,
        "settle_fixed_substep_gap",
    )
    interval = seconds_between(previous, current)
    require(0.0 < interval <= MAX_SETTLE_STOPWATCH_INTERVAL_S, "settle_stopwatch_interval_invalid")
    require(current["unity_frame"] >= previous["unity_frame"], "settle_unity_frame_reversed")
    require(current["unity_fixed_time"] > previous["unity_fixed_time"], "settle_unity_fixed_time_not_advanced")


def validate_event_envelope(
    event: Mapping[str, Any], expected_runner_hash: str, expected_controller_hash: str
) -> None:
    string(required(event, "event"), "event_name_invalid")
    require(event.get("attack_zone_schema") == RUNNER_SCHEMA, "runner_event_schema_mismatch")
    require(event.get("attack_zone_protocol_sha256") == expected_runner_hash, "runner_contract_hash_mismatch")
    require(event.get("continuous_controller_sha256") == expected_controller_hash, "controller_contract_hash_mismatch")
    require(event.get("authority_scope") == AUTHORITY_SCOPE, "authority_scope_mismatch")
    string(event.get("authority_caveat"), "authority_caveat_missing")
    require(event.get("isolated_spark_proof") == REQUIRED_ISOLATION_PROOF, "isolation_proof_mismatch")
    require(event.get("global_input_used") is False, "global_input_observed")
    require(event.get("client_request_observation_only") is True, "request_observation_scope_missing")
    require(event.get("server_acceptance_observed") is False, "server_acceptance_not_false")
    require(event.get("authoritative_execution_observed") is False, "authoritative_execution_not_false")
    require(event.get("move_index") in ALLOWED_MOVES, "event_move_invalid")
    require(valid_sha256(event.get("serialized_asset_sha256")), "event_asset_sha256_invalid")
    for name in ("schedule_sha256", "session_identity_sha256", "round_identity_sha256"):
        require(valid_sha256(event.get(name)), f"event_{name}_invalid")
    string(event.get("independent_run_id"), "event_run_id_invalid")
    string(event.get("trial_id"), "event_trial_id_invalid")
    integer(event.get("action_sequence"), "event_action_sequence_invalid", 1)
    event_clock(event)
    scan_authority_flags(event)


def validate_runtime_identity(measured_state: Mapping[str, Any]) -> dict[str, Any]:
    local_identity = mapping(required(measured_state, "local_identity"), "local_identity_missing")
    opponent_identity = mapping(required(measured_state, "opponent_identity"), "opponent_identity_missing")
    require(local_identity.get("exact_local_t800_proven") is True, "local_runtime_t800_not_proven")
    require(local_identity.get("semantic_robot_id") == "t800", "local_semantic_t800_not_proven")
    require(local_identity.get("runtime_bone_count") == 26, "local_bone_count_not_26")
    require(opponent_identity.get("runtime_bone_count") == 26, "opponent_bone_count_not_26")
    require(
        local_identity.get("runtime_bone_signature_sha256") == T800_BONE_SIGNATURE_SHA256,
        "local_bone_signature_not_exact_t800",
    )
    require(
        opponent_identity.get("runtime_bone_signature_sha256") == T800_BONE_SIGNATURE_SHA256,
        "opponent_bone_signature_not_exact_t800",
    )
    string(local_identity.get("runtime_object_name"), "local_runtime_object_name_missing")
    string(opponent_identity.get("runtime_object_name"), "opponent_runtime_object_name_missing")
    require(valid_sha256(opponent_identity.get("runtime_identity_sha256")), "opponent_runtime_identity_invalid")
    require(opponent_identity.get("semantic_robot_id_used_for_acceptance") is False, "opponent_semantic_id_trusted")
    mismatch = opponent_identity.get("semantic_runtime_mismatch")
    require(isinstance(mismatch, bool), "opponent_semantic_mismatch_flag_invalid")
    return {
        "accepted": True,
        "opponent_semantic_runtime_mismatch": mismatch,
        "reason": "exact_runtime_t800_with_semantic_mismatch_recorded" if mismatch else "exact_runtime_t800",
    }


def classify_measured_opponent_motion(measured_state: Mapping[str, Any]) -> dict[str, Any] | None:
    geometry = measured_state.get("geometry")
    local_root = measured_state.get("local_root")
    opponent_root = measured_state.get("opponent_root")
    if not all(isinstance(value, dict) for value in (geometry, local_root, opponent_root)):
        return None
    local_position = sequence(required(local_root, "position_xyz_m"), "motion_local_position_missing")
    opponent_position = sequence(required(opponent_root, "position_xyz_m"), "motion_opponent_position_missing")
    local_velocity = sequence(required(local_root, "linear_velocity_xyz_m_s"), "motion_local_velocity_missing")
    opponent_velocity = sequence(required(opponent_root, "linear_velocity_xyz_m_s"), "motion_opponent_velocity_missing")
    opponent_angular = sequence(
        required(opponent_root, "angular_velocity_xyz_rad_s"), "motion_opponent_angular_velocity_missing",
    )
    require(
        all(len(value) == 3 for value in (local_position, opponent_position, local_velocity, opponent_velocity, opponent_angular)),
        "motion_vector_length_invalid",
    )
    separation_x = finite_number(opponent_position[0], "motion_position_nonfinite") - finite_number(
        local_position[0], "motion_position_nonfinite",
    )
    separation_z = finite_number(opponent_position[2], "motion_position_nonfinite") - finite_number(
        local_position[2], "motion_position_nonfinite",
    )
    separation_norm = math.hypot(separation_x, separation_z)
    require(separation_norm > 1e-12, "motion_root_separation_zero")
    unit_x, unit_z = separation_x / separation_norm, separation_z / separation_norm
    relative_x = finite_number(opponent_velocity[0], "motion_velocity_nonfinite") - finite_number(
        local_velocity[0], "motion_velocity_nonfinite",
    )
    relative_z = finite_number(opponent_velocity[2], "motion_velocity_nonfinite") - finite_number(
        local_velocity[2], "motion_velocity_nonfinite",
    )
    opponent_speed = math.hypot(
        finite_number(opponent_velocity[0], "motion_velocity_nonfinite"),
        finite_number(opponent_velocity[2], "motion_velocity_nonfinite"),
    )
    yaw_rate = abs(finite_number(opponent_angular[1], "motion_yaw_rate_nonfinite"))
    radial_closing = -(relative_x * unit_x + relative_z * unit_z)
    tangential = abs(relative_x * unit_z - relative_z * unit_x)
    stationary = opponent_speed <= LOCAL_SPEED_LIMIT_M_S and yaw_rate <= YAW_RATE_LIMIT_RAD_S
    if stationary:
        motion = "stationary"
    else:
        predicates = (
            (radial_closing > LOCAL_SPEED_LIMIT_M_S, "closing"),
            (radial_closing < -LOCAL_SPEED_LIMIT_M_S, "receding"),
            (abs(radial_closing) <= LOCAL_SPEED_LIMIT_M_S and tangential > LOCAL_SPEED_LIMIT_M_S, "tangential"),
            (yaw_rate > YAW_RATE_LIMIT_RAD_S, "turning"),
        )
        active = [name for passed, name in predicates if passed]
        motion = active[0] if len(active) == 1 else "compound_or_unknown"
    opponent_bearing = abs(wrap_to_180(finite_number(
        required(geometry, "opponent_bearing_to_local_deg"), "motion_opponent_bearing_missing",
    )))
    facing = (
        "opponent_face_on" if opponent_bearing <= 35.0 else
        "opponent_oblique" if opponent_bearing <= 90.0 else
        "opponent_back_turned"
    )
    return {
        "motion_stratum": motion,
        "facing_stratum": facing,
        "stationary": stationary,
        "opponent_planar_speed_m_s": opponent_speed,
        "opponent_yaw_rate_rad_s": yaw_rate,
        "radial_closing_speed_m_s": radial_closing,
        "tangential_speed_m_s": tangential,
    }


def opponent_motion_path(
    events: Sequence[Mapping[str, Any]], settle_digest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if settle_digest is not None:
        entries.append({
            "source": "validated_15_tick_settle",
            "event": "target_acquired",
            "client_control_tick": settle_digest["last_clock"]["client_control_tick"],
            "motion_stratum": settle_digest["opponent_motion_stratum"],
            "facing_stratum": settle_digest["opponent_facing_stratum"],
        })
    action_started = False
    for event in events:
        if event.get("event") == "local_command_edge_set":
            action_started = True
        if not action_started or not isinstance(event.get("measured_state"), dict):
            continue
        derived = classify_measured_opponent_motion(event["measured_state"])
        if derived is None:
            continue
        entries.append({
            "source": "runner_measured_state",
            "event": event["event"],
            "client_control_tick": event["client_control_tick"],
            **derived,
        })
    motions = [entry["motion_stratum"] for entry in entries]
    facings = [entry["facing_stratum"] for entry in entries]
    motion = motions[0] if motions and len(set(motions)) == 1 else "compound_or_unknown"
    facing = facings[0] if facings and len(set(facings)) == 1 else "opponent_facing_changed_or_unresolved"
    return {
        "entries": entries,
        "motion_stratum": motion if entries else "unresolved",
        "facing_stratum": facing if entries else "unresolved",
        "time_varying_motion": len(set(motions)) > 1,
        "time_varying_facing": len(set(facings)) > 1,
    }


def validate_settle_samples(
    samples: Sequence[Mapping[str, Any]],
    distance_bin: Mapping[str, Any],
    bearing_bin: Mapping[str, Any],
) -> dict[str, Any]:
    require(len(samples) == SETTLE_TICKS, "settle_sample_count_mismatch")
    d_lower, d_upper = central_interval(distance_bin, "m")
    bearing_center = finite_number(required(bearing_bin, "center_deg"), "bearing_center_invalid")
    clocks: list[Mapping[str, Any]] = []
    opponent_motion: list[str] = []
    opponent_facing: list[str] = []
    distances: list[float] = []
    bearings: list[float] = []
    local_speeds: list[float] = []
    local_yaw_rates: list[float] = []
    opponent_speeds: list[float] = []
    opponent_yaw_rates: list[float] = []
    for index, sample in enumerate(samples):
        clock = mapping(required(sample, "clock"), "settle_clock_missing")
        validate_clock(clock)
        if clocks:
            validate_consecutive_clocks(clocks[-1], clock)
        clocks.append(clock)
        distance = finite_number(required(sample, "distance_m"), "settle_distance_invalid")
        bearing = wrap_to_180(finite_number(required(sample, "bearing_deg"), "settle_bearing_invalid"))
        local_speed = finite_number(required(sample, "local_planar_speed_m_s"), "settle_local_speed_invalid")
        local_yaw = abs(finite_number(required(sample, "local_yaw_rate_rad_s"), "settle_local_yaw_invalid"))
        opponent_speed = finite_number(required(sample, "opponent_planar_speed_m_s"), "settle_opponent_speed_invalid")
        opponent_yaw = abs(finite_number(required(sample, "opponent_yaw_rate_rad_s"), "settle_opponent_yaw_invalid"))
        require(d_lower <= distance <= d_upper, "settle_distance_outside_central_half", str(index))
        require(bin_contains(bearing_bin, bearing, "deg"), "settle_bearing_outside_bin", str(index))
        require(abs(wrap_to_180(bearing - bearing_center)) <= BEARING_ERROR_LIMIT_DEG, "settle_bearing_error_exceeded", str(index))
        require(local_speed <= LOCAL_SPEED_LIMIT_M_S, "settle_local_speed_exceeded", str(index))
        require(local_yaw <= YAW_RATE_LIMIT_RAD_S, "settle_local_yaw_exceeded", str(index))
        for flag in (
            "neutral_request_method_returned", "velocity_command_exact_neutral",
            "local_action_ready", "no_pending_requests", "local_healthy", "opponent_healthy",
        ):
            require(sample.get(flag) is True, "settle_predicate_failed", f"{index}:{flag}")
        motion = string(required(sample, "opponent_motion_stratum"), "motion_stratum_missing")
        facing = string(required(sample, "opponent_facing_stratum"), "facing_stratum_missing")
        opponent_motion.append(motion)
        opponent_facing.append(facing)
        distances.append(distance)
        bearings.append(bearing)
        local_speeds.append(local_speed)
        local_yaw_rates.append(local_yaw)
        opponent_speeds.append(opponent_speed)
        opponent_yaw_rates.append(opponent_yaw)
    all_stationary = all(
        speed <= LOCAL_SPEED_LIMIT_M_S and yaw <= YAW_RATE_LIMIT_RAD_S and motion == "stationary"
        for speed, yaw, motion in zip(opponent_speeds, opponent_yaw_rates, opponent_motion)
    )
    motion_stratum = opponent_motion[0] if len(set(opponent_motion)) == 1 else "compound_or_unknown"
    facing_stratum = opponent_facing[0] if len(set(opponent_facing)) == 1 else "opponent_facing_changed"
    return {
        "sample_count": SETTLE_TICKS,
        "first_clock": clocks[0],
        "last_clock": clocks[-1],
        "distance_m": {"min": min(distances), "max": max(distances)},
        "local_bearing_deg": {"min": min(bearings), "max": max(bearings)},
        "local_planar_speed_m_s": {"min": min(local_speeds), "max": max(local_speeds)},
        "local_yaw_rate_rad_s": {"min": min(local_yaw_rates), "max": max(local_yaw_rates)},
        "opponent_planar_speed_m_s": {"min": min(opponent_speeds), "max": max(opponent_speeds)},
        "opponent_yaw_rate_rad_s": {"min": min(opponent_yaw_rates), "max": max(opponent_yaw_rates)},
        "all_opponent_stationary": all_stationary,
        "opponent_motion_stratum": motion_stratum,
        "opponent_facing_stratum": facing_stratum,
        "opponent_motion_path": opponent_motion,
        "opponent_facing_path": opponent_facing,
    }


def normalized_settle_sample(event: Mapping[str, Any]) -> dict[str, Any]:
    detail = mapping(required(event, "detail"), "acquisition_detail_missing")
    settle = mapping(required(detail, "settle"), "settle_update_missing")
    evaluation = mapping(required(settle, "current_evaluation"), "settle_evaluation_missing")
    geometry = mapping(required(evaluation, "Geometry"), "settle_geometry_missing")
    motion = mapping(required(evaluation, "Motion"), "settle_motion_missing")
    return {
        "clock": event_clock(event),
        "distance_m": required(geometry, "DistanceMeters"),
        "bearing_deg": required(geometry, "LocalBearingToOpponentDegrees"),
        "local_planar_speed_m_s": required(evaluation, "LocalPlanarSpeedMetersPerSecond"),
        "local_yaw_rate_rad_s": required(evaluation, "LocalYawRateRadiansPerSecond"),
        "opponent_planar_speed_m_s": required(motion, "OpponentPlanarSpeedMetersPerSecond"),
        "opponent_yaw_rate_rad_s": required(motion, "OpponentYawRateRadiansPerSecond"),
        "opponent_motion_stratum": required(motion, "MotionStratum"),
        "opponent_facing_stratum": required(motion, "FacingStratum"),
        "neutral_request_method_returned": evaluation.get("NeutralRequestMethodReturned"),
        "velocity_command_exact_neutral": evaluation.get("VelocityCommandExactNeutral"),
        "local_action_ready": evaluation.get("LocalActionReady"),
        "no_pending_requests": evaluation.get("NoPendingRequests"),
        "local_healthy": evaluation.get("LocalHealthy"),
        "opponent_healthy": evaluation.get("OpponentHealthy"),
    }


def validate_lifecycle(events: Sequence[Mapping[str, Any]], expected_move_name: str) -> dict[str, Any]:
    by_stage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_stage[str(event.get("event"))].append(event)
    terminal = sorted(
        by_stage.get("trial_censored", []) + by_stage.get("trial_interrupted", []),
        key=lambda value: event_clock(value)["stopwatch_timestamp_ticks"],
    )
    require(len(terminal) <= 1, "duplicate_trial_terminal_event")
    for stage in (
        "local_command_edge_set", "client_request_method_returned", "local_motion_start_observed",
        "local_motion_completion_and_readiness_observed",
    ):
        require(len(by_stage.get(stage, [])) <= 1, "duplicate_lifecycle_stage", stage)
    command = by_stage.get("local_command_edge_set", [None])[0]
    returned = by_stage.get("client_request_method_returned", [None])[0]
    started = by_stage.get("local_motion_start_observed", [None])[0]
    completed = by_stage.get("local_motion_completion_and_readiness_observed", [None])[0]
    observed = [event for event in (command, returned, started, completed) if event is not None]
    ticks = [event_clock(event)["stopwatch_timestamp_ticks"] for event in observed]
    require(ticks == sorted(ticks), "lifecycle_order_invalid")
    if terminal and observed:
        require(
            event_clock(terminal[0])["stopwatch_timestamp_ticks"] >= ticks[-1],
            "terminal_before_observed_lifecycle",
        )
    if returned is not None:
        detail = mapping(required(returned, "detail"), "method_return_detail_missing")
        require(detail.get("send_method") == "RobotInputController.SendMoveEvent", "move_send_method_mismatch")
        require(detail.get("send_method_returned") is True, "move_send_return_not_observed")
    if started is not None:
        state = mapping(required(started, "measured_state"), "start_measured_state_missing")
        local_motion = mapping(required(state, "local_motion"), "start_composer_state_missing")
        require(local_motion.get("action_playing") is True, "start_action_playing_not_true")
        require(local_motion.get("active_action_clip") == expected_move_name, "start_clip_identity_mismatch")
        finite_number(required(local_motion, "action_clip_frame"), "start_clip_frame_invalid")
        finite_number(required(local_motion, "action_clip_fps"), "start_clip_fps_invalid")
    if completed is not None:
        state = mapping(required(completed, "measured_state"), "completion_measured_state_missing")
        motion = mapping(required(state, "local_motion"), "completion_composer_state_missing")
        input_state = mapping(required(state, "input_state"), "completion_input_state_missing")
        require(motion.get("action_playing") is False, "completion_action_still_playing")
        require(motion.get("busy") is False, "completion_composer_still_busy")
        require(input_state.get("punching") is False, "completion_input_punching")
        require(input_state.get("recovering") is False, "completion_input_recovering")
        for name in ("pending_move", "pending_special", "pending_estop"):
            require(input_state.get(name) is False, "completion_pending_request", name)
    if not terminal:
        require(
            all(value is not None for value in (command, returned, started, completed)),
            "trial_terminal_event_missing",
        )
    clocks = {name: event_clock(value) if value is not None else None for name, value in (
        ("command", command), ("returned", returned), ("started", started), ("completed", completed)
    )}
    terminal_clock = event_clock(terminal[0]) if terminal else None
    censor_durations = {
        "request_method_duration_s": _optional_duration(clocks["command"], terminal_clock)
        if returned is None else None,
        "request_to_local_composer_start_s": _optional_duration(clocks["command"], terminal_clock)
        if started is None else None,
        "method_return_to_local_composer_start_s": _optional_duration(clocks["returned"], terminal_clock)
        if started is None else None,
        "local_composer_start_to_completion_readiness_s": _optional_duration(clocks["started"], terminal_clock)
        if completed is None else None,
        "request_to_completion_readiness_s": _optional_duration(clocks["command"], terminal_clock)
        if completed is None else None,
    }
    return {
        "request_method_duration_s": _optional_duration(clocks["command"], clocks["returned"]),
        "request_to_local_composer_start_s": _optional_duration(clocks["command"], clocks["started"]),
        "method_return_to_local_composer_start_s": _optional_duration(clocks["returned"], clocks["started"]),
        "local_composer_start_to_completion_readiness_s": _optional_duration(clocks["started"], clocks["completed"]),
        "request_to_completion_readiness_s": _optional_duration(clocks["command"], clocks["completed"]),
        "censored": completed is None,
        "censor_reason": terminal[-1].get("controller_reason") if completed is None and terminal else None,
        "censor_durations_s": censor_durations,
    }


def _optional_duration(start: Mapping[str, Any] | None, end: Mapping[str, Any] | None) -> float | None:
    return None if start is None or end is None else seconds_between(start, end)


def derive_auxiliary_timings(
    events: Sequence[Mapping[str, Any]], joined_raw_hit_sequences: set[int],
) -> dict[str, Any]:
    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_name[str(event.get("event"))].append(event)
    command = by_name.get("local_command_edge_set", [None])[0]
    started = by_name.get("local_motion_start_observed", [None])[0]
    command_clock = event_clock(command) if command is not None else None
    start_clock = event_clock(started) if started is not None else None
    marker_values: list[dict[str, Any]] = []
    for ordinal, event in enumerate(sorted(
        by_name.get("configured_asset_marker_projected", []),
        key=lambda value: (
            mapping(value.get("detail"), "configured_marker_detail_missing").get("ImpactTimeSeconds"),
            value["stopwatch_timestamp_ticks"],
        ),
    ), start=1):
        validate_configured_marker_event(event)
        detail = mapping(event["detail"], "configured_marker_detail_missing")
        projected_tick = integer(
            required(detail, "projected_stopwatch_timestamp_ticks"), "configured_marker_clock_invalid", 0,
        )
        if start_clock is None:
            duration = None
        else:
            require(
                start_clock["stopwatch_frequency_hz"] == event_clock(event)["stopwatch_frequency_hz"],
                "configured_marker_clock_frequency_changed",
            )
            duration = (projected_tick - start_clock["stopwatch_timestamp_ticks"]) / start_clock["stopwatch_frequency_hz"]
            require(duration >= 0.0, "configured_marker_before_local_start")
        marker_values.append({
            "marker_ordinal": ordinal,
            "limb": detail.get("Limb"),
            "configured_impact_time_s": finite_number(
                required(detail, "ImpactTimeSeconds"), "configured_marker_time_invalid",
            ),
            "local_start_to_configured_marker_s": duration,
            "class": "static_asset_projection",
            "contact": False,
        })
    raw_values: list[dict[str, Any]] = []
    for event in sorted(by_name.get("raw_rek_hit_observed", []), key=lambda value: (
        value["stopwatch_timestamp_ticks"],
        mapping(value.get("detail"), "raw_hit_detail_missing").get("raw_hit_sequence"),
    )):
        detail = mapping(event["detail"], "raw_hit_detail_missing")
        sequence_id = integer(required(detail, "raw_hit_sequence"), "raw_hit_sequence_invalid", 1)
        clock = event_clock(event)
        raw_values.append({
            "raw_hit_sequence": sequence_id,
            "request_to_raw_hit_observation_s": _optional_duration(command_clock, clock),
            "local_start_to_raw_hit_observation_s": _optional_duration(start_clock, clock),
            "joined_local_scoring_candidate": sequence_id in joined_raw_hit_sequences,
            "request_to_isolated_local_scoring_hit_candidate_s": (
                _optional_duration(command_clock, clock) if sequence_id in joined_raw_hit_sequences else None
            ),
            "local_start_to_isolated_local_scoring_hit_candidate_s": (
                _optional_duration(start_clock, clock) if sequence_id in joined_raw_hit_sequences else None
            ),
            "raw_class": "unattributed_client_packet",
        })
    score_values = []
    for event in sorted(by_name.get("round_score_delta_observed", []), key=lambda value: value["stopwatch_timestamp_ticks"]):
        detail = mapping(event.get("detail"), "score_detail_missing")
        if integer(required(detail, "local_clean_hit_delta"), "local_clean_hit_delta_invalid") > 0:
            score_values.append(_optional_duration(command_clock, event_clock(event)))
    pose_entry = by_name.get("analysis_pose_entry_observed", [None])[0]
    pose_return = by_name.get("analysis_pose_return_observed", [None])[0]
    pose_entry_clock = event_clock(pose_entry) if pose_entry is not None else None
    pose_return_clock = event_clock(pose_return) if pose_return is not None else None
    return {
        "configured_markers": marker_values,
        "raw_hits": raw_values,
        "request_to_first_temporal_local_score_delta_s": score_values[0] if score_values else None,
        "request_to_pose_entry_s": _optional_duration(command_clock, pose_entry_clock),
        "pose_entry_to_pose_return_s": _optional_duration(pose_entry_clock, pose_return_clock),
    }


def _normalize_quaternion(q: Sequence[Any]) -> tuple[float, float, float, float]:
    require(len(q) == 4, "quaternion_length_invalid")
    values = tuple(finite_number(value, "quaternion_nonfinite") for value in q)
    norm = math.sqrt(sum(value * value for value in values))
    require(norm > 1e-12, "quaternion_zero_norm")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def rotate_vector_by_inverse_quaternion(vector: Sequence[Any], quaternion: Sequence[Any]) -> tuple[float, float, float]:
    require(len(vector) == 3, "vector_length_invalid")
    vx, vy, vz = (finite_number(value, "vector_nonfinite") for value in vector)
    x, y, z, w = _normalize_quaternion(quaternion)
    tx = 2.0 * (-y * vz + z * vy)
    ty = 2.0 * (-z * vx + x * vz)
    tz = 2.0 * (-x * vy + y * vx)
    return (
        vx + w * tx + (-y * tz + z * ty),
        vy + w * ty + (-z * tx + x * tz),
        vz + w * tz + (-x * ty + y * tx),
    )


def opponent_root_transform(
    hit_world: Sequence[Any], root_world: Sequence[Any], root_rotation: Sequence[Any]
) -> tuple[float, float, float]:
    require(len(hit_world) == 3 and len(root_world) == 3, "position_length_invalid")
    delta = [
        finite_number(hit_world[index], "hit_position_nonfinite") -
        finite_number(root_world[index], "root_position_nonfinite")
        for index in range(3)
    ]
    return rotate_vector_by_inverse_quaternion(delta, root_rotation)


def quaternion_slerp(left: Sequence[Any], right: Sequence[Any], weight: float) -> tuple[float, float, float, float]:
    require(0.0 <= weight <= 1.0 and math.isfinite(weight), "slerp_weight_invalid")
    q0 = _normalize_quaternion(left)
    q1 = _normalize_quaternion(right)
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        values = tuple(a + weight * (b - a) for a, b in zip(q0, q1))
        return _normalize_quaternion(values)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    left_weight = math.sin((1.0 - weight) * theta) / sin_theta
    right_weight = math.sin(weight * theta) / sin_theta
    return tuple(left_weight * a + right_weight * b for a, b in zip(q0, q1))  # type: ignore[return-value]


def interpolate_root_pose(samples: Sequence[Mapping[str, Any]], target_substep: int) -> dict[str, Any]:
    require(samples, "root_samples_missing")
    ordered = sorted(samples, key=lambda value: integer(value.get("client_fixed_substep"), "root_substep_invalid", 0))
    exact = [sample for sample in ordered if sample["client_fixed_substep"] == target_substep]
    require(len(exact) <= 1, "duplicate_root_sample_tick")
    if exact:
        return {
            "position_xyz_m": list(sequence(required(exact[0], "position_xyz_m"), "root_position_invalid")),
            "rotation_xyzw": list(sequence(required(exact[0], "rotation_xyzw"), "root_rotation_invalid")),
            "method": "exact_client_fixed_substep",
            "source_substeps": [target_substep],
            "weight": 0.0,
        }
    lower = [sample for sample in ordered if sample["client_fixed_substep"] < target_substep]
    upper = [sample for sample in ordered if sample["client_fixed_substep"] > target_substep]
    require(lower and upper, "root_interpolation_bracket_missing")
    before, after = lower[-1], upper[0]
    lo = integer(before["client_fixed_substep"], "root_substep_invalid", 0)
    hi = integer(after["client_fixed_substep"], "root_substep_invalid", 0)
    require(lo < target_substep < hi, "root_interpolation_bracket_invalid")
    weight = (target_substep - lo) / (hi - lo)
    p0 = sequence(required(before, "position_xyz_m"), "root_position_invalid")
    p1 = sequence(required(after, "position_xyz_m"), "root_position_invalid")
    require(len(p0) == 3 and len(p1) == 3, "root_position_length_invalid")
    position = [
        finite_number(p0[index], "root_position_nonfinite") + weight * (
            finite_number(p1[index], "root_position_nonfinite") - finite_number(p0[index], "root_position_nonfinite")
        ) for index in range(3)
    ]
    rotation = quaternion_slerp(
        sequence(required(before, "rotation_xyzw"), "root_rotation_invalid"),
        sequence(required(after, "rotation_xyzw"), "root_rotation_invalid"),
        weight,
    )
    return {
        "position_xyz_m": position,
        "rotation_xyzw": list(rotation),
        "method": "bracketed_client_fixed_substep_interpolation",
        "source_substeps": [lo, hi],
        "weight": weight,
    }


def nearest_bone_candidate(
    hit_opponent_root_xyz: Sequence[Any], bones: Sequence[Mapping[str, Any]], tolerance: float = 1e-12
) -> dict[str, Any]:
    require(len(hit_opponent_root_xyz) == 3, "hit_root_position_invalid")
    hit = tuple(finite_number(value, "hit_root_position_nonfinite") for value in hit_opponent_root_xyz)
    require(len(bones) == 26, "opponent_bone_count_not_26")
    candidates: list[tuple[float, int, str]] = []
    seen_names: set[str] = set()
    for bone in bones:
        index = integer(required(bone, "index"), "bone_index_invalid", 0)
        name = string(required(bone, "name"), "bone_name_invalid")
        require(index < len(T800_BONE_NAMES), "bone_index_invalid")
        require(name == T800_BONE_NAMES[index], "bone_index_name_signature_mismatch", f"{index}:{name}")
        require(name not in seen_names, "duplicate_bone_name", name)
        seen_names.add(name)
        position = sequence(required(bone, "position_opponent_root_xyz_m"), "bone_position_missing")
        require(len(position) == 3, "bone_position_length_invalid")
        distance = math.sqrt(sum(
            (finite_number(position[axis], "bone_position_nonfinite") - hit[axis]) ** 2 for axis in range(3)
        ))
        candidates.append((distance, index, name))
    require(seen_names == set(BONE_TO_REGION), "t800_bone_signature_mismatch")
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    best = candidates[0]
    tied = [candidate for candidate in candidates if abs(candidate[0] - best[0]) <= tolerance]
    require(len(tied) == 1, "nearest_bone_tie")
    return {
        "candidate_nearest_opponent_bone_index": best[1],
        "candidate_nearest_opponent_bone_name": best[2],
        "candidate_nearest_opponent_bone_distance_m": best[0],
        "candidate_opponent_body_region": BONE_TO_REGION[best[2]],
        "candidate_distances_to_all_opponent_bones_m": [
            {"index": index, "name": name, "distance_m": distance}
            for distance, index, name in sorted(candidates, key=lambda value: (value[1], value[2]))
        ],
        "nearest_bone_tie_status": "unique_minimum",
        "candidate_only": True,
    }


def select_nearest_bone_sample(
    samples: Sequence[Mapping[str, Any]], hit_time_s: float, maximum_abs_offset_s: float = 0.01
) -> dict[str, Any]:
    target = finite_number(hit_time_s, "hit_time_invalid")
    limit = finite_number(maximum_abs_offset_s, "bone_offset_limit_invalid")
    require(limit >= 0.0, "bone_offset_limit_invalid")
    require(samples, "bone_samples_missing")
    candidates: list[tuple[float, float, int, Mapping[str, Any]]] = []
    for index, sample in enumerate(samples):
        time = finite_number(required(sample, "time_s"), "bone_sample_time_invalid")
        candidates.append((abs(time - target), time, index, sample))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    require(candidates[0][0] <= limit, "bone_sample_time_offset_exceeded")
    require(
        len(candidates) == 1 or abs(candidates[1][0] - candidates[0][0]) > 1e-15,
        "bone_sample_time_tie",
    )
    return {
        "sample": dict(candidates[0][3]),
        "bone_sample_time_offset_s": candidates[0][1] - target,
    }


def classify_fall_recovery(
    request_tick: int,
    start_tick: int | None,
    contamination_tick: int,
) -> dict[str, Any]:
    request = integer(request_tick, "fall_request_tick_invalid", 0)
    contamination = integer(contamination_tick, "fall_contamination_tick_invalid", 0)
    start = None if start_tick is None else integer(start_tick, "fall_start_tick_invalid", 0)
    if contamination <= request or start is None or contamination <= start:
        return {
            "whole_trial_primary_excluded": True,
            "completion_right_censored": start is not None,
            "remainder_of_round_excluded": True,
            "pre_fall_evidence_table_only": False,
            "classification": "pre_request_or_pre_start_contamination",
        }
    return {
        "whole_trial_primary_excluded": True,
        "completion_right_censored": True,
        "remainder_of_round_excluded": True,
        "pre_fall_evidence_table_only": True,
        "classification": "post_start_fall_associated_censored",
    }


def validate_configured_marker_event(event: Mapping[str, Any]) -> dict[str, Any]:
    require(event.get("event") == "configured_asset_marker_projected", "not_configured_marker_event")
    detail = mapping(required(event, "detail"), "configured_marker_detail_missing")
    require(detail.get("observed_contact") is False, "configured_marker_claims_contact")
    require(detail.get("observed_hit_ownership") is False, "configured_marker_claims_ownership")
    finite_number(required(detail, "ImpactTimeSeconds"), "configured_marker_time_invalid")
    integer(required(detail, "projected_stopwatch_timestamp_ticks"), "configured_marker_clock_invalid", 0)
    return {
        "class": "static_asset_projection",
        "contact": False,
        "authority": "configured_asset_marker_not_observed_contact",
    }


def raw_hit_location_candidate(raw_event: Mapping[str, Any]) -> dict[str, Any]:
    require(raw_event.get("event") == "raw_rek_hit_observed", "not_raw_hit_event")
    clock = event_clock(raw_event)
    detail = mapping(required(raw_event, "detail"), "raw_hit_detail_missing")
    require(detail.get("raw_packet_contains_fighter_identity") is False, "raw_hit_identity_flag_invalid")
    require(detail.get("raw_packet_contains_move_identity") is False, "raw_hit_move_flag_invalid")
    decoded = mapping(required(detail, "decoded"), "raw_hit_decoded_missing")
    hit_world = sequence(required(decoded, "world_position_xyz_m"), "raw_hit_world_position_missing")
    context = mapping(
        required(detail, "contemporaneous_opponent_root_bones_and_colliders"),
        "raw_hit_opponent_context_missing",
    )
    root_position = sequence(required(context, "root_position_xyz_m"), "raw_hit_root_position_missing")
    root_rotation = sequence(required(context, "root_rotation_xyzw"), "raw_hit_root_rotation_missing")
    hit_root = opponent_root_transform(hit_world, root_position, root_rotation)
    bones_world = sequence(required(context, "bones"), "raw_hit_bones_missing")
    bones_root: list[dict[str, Any]] = []
    for value in bones_world:
        bone = mapping(value, "raw_hit_bone_invalid")
        bones_root.append({
            "index": required(bone, "index"),
            "name": required(bone, "name"),
            "position_opponent_root_xyz_m": list(opponent_root_transform(
                sequence(required(bone, "world_position_xyz_m"), "bone_world_position_missing"),
                root_position,
                root_rotation,
            )),
        })
    nearest = nearest_bone_candidate(hit_root, bones_root)
    wire_hash = required(detail, "wire_body_sha256")
    require(valid_sha256(wire_hash), "raw_hit_wire_hash_invalid")
    return {
        "raw_hit_sequence": integer(required(detail, "raw_hit_sequence"), "raw_hit_sequence_invalid", 1),
        "wire_body_sha256": wire_hash,
        "world_position_xyz_m": list(hit_world),
        "candidate_hit_opponent_root_xyz_m": list(hit_root),
        "root_alignment_method": "exact_client_fixed_substep_runner_context",
        "root_alignment_status": "aligned",
        "root_alignment_client_fixed_substep": clock["client_fixed_substep"],
        "bone_alignment_method": "exact_client_fixed_substep_runner_context",
        "bone_alignment_status": "aligned",
        "bone_alignment_client_fixed_substep": clock["client_fixed_substep"],
        "location_status": "resolved_candidate",
        **nearest,
        "association_status": "unattributed_rek_hit_location_candidate",
    }


def join_local_scoring_candidate(
    raw_hits: Sequence[Mapping[str, Any]],
    score_event: Mapping[str, Any],
    lifecycle_events: Sequence[Mapping[str, Any]],
    contamination_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(score_event.get("event") == "round_score_delta_observed", "not_score_event")
    detail = mapping(required(score_event, "detail"), "score_detail_missing")
    local_delta = integer(required(detail, "local_clean_hit_delta"), "local_clean_hit_delta_invalid")
    explicit_sequences = sequence(required(detail, "raw_hit_sequences"), "score_raw_hit_sequences_missing")
    if local_delta <= 0:
        return {"passed": False, "label": "unattributed_rek_hit_location_candidate", "reason": "no_positive_local_score_delta"}
    if detail.get("isolated_selected_local_action_interval") is not True:
        return {"passed": False, "label": "unattributed_rek_hit_location_candidate", "reason": "local_action_window_not_isolated"}
    if len(explicit_sequences) != 1:
        return {"passed": False, "label": "unattributed_rek_hit_location_candidate", "reason": "raw_hit_pairing_not_unique"}
    sequence_id = integer(explicit_sequences[0], "score_raw_hit_sequence_invalid", 1)
    matched = [event for event in raw_hits if mapping(event.get("detail"), "raw_hit_detail_missing").get("raw_hit_sequence") == sequence_id]
    if len(matched) != 1:
        return {"passed": False, "label": "unattributed_rek_hit_location_candidate", "reason": "explicit_raw_hit_not_unique"}
    starts = [event for event in lifecycle_events if event.get("event") == "local_motion_start_observed"]
    commands = [event for event in lifecycle_events if event.get("event") == "local_command_edge_set"]
    require(len(starts) == 1 and len(commands) == 1, "isolated_lifecycle_not_unique")
    score_clock = event_clock(score_event)
    start_clock = event_clock(starts[0])
    hit_clock = event_clock(matched[0])
    require(seconds_between(start_clock, hit_clock) >= 0.0, "raw_hit_before_local_start")
    require(seconds_between(hit_clock, score_clock) >= 0.0, "score_before_raw_hit")
    for event in contamination_events:
        if event_clock(event)["stopwatch_timestamp_ticks"] <= score_clock["stopwatch_timestamp_ticks"]:
            return {"passed": False, "label": "unattributed_rek_hit_location_candidate", "reason": "fall_recovery_or_interrupt_contamination"}
    return {
        "passed": True,
        "label": "client_isolated_local_scoring_zone_candidate",
        "reason": "positive_local_score_unique_explicit_raw_hit_isolated_local_action",
        "raw_hit_sequence": sequence_id,
        "still_not_authoritative": True,
    }


def quantile(values: Sequence[float], probability: float) -> float:
    require(values, "quantile_empty")
    require(0.0 <= probability <= 1.0, "quantile_probability_invalid")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [finite_number(value, "summary_nonfinite") for value in values]
    if not clean:
        return {name: None for name in (
            "count", "median", "mad", "mean", "sample_standard_deviation", "p05", "p95", "minimum", "maximum"
        )} | {"count": 0}
    median = statistics.median(clean)
    return {
        "count": len(clean),
        "median": median,
        "mad": statistics.median(abs(value - median) for value in clean),
        "mean": statistics.fmean(clean),
        "sample_standard_deviation": statistics.stdev(clean) if len(clean) > 1 else None,
        "p05": quantile(clean, 0.05),
        "p95": quantile(clean, 0.95),
        "minimum": min(clean),
        "maximum": max(clean),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    integer(successes, "wilson_successes_invalid", 0)
    integer(trials, "wilson_trials_invalid", 0)
    require(successes <= trials and trials > 0, "wilson_counts_invalid")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _bootstrap_statistic(values: Sequence[float], statistic: str) -> float:
    require(values, "bootstrap_values_empty")
    if statistic in {"mean", "proportion"}:
        return statistics.fmean(values)
    if statistic == "median":
        return statistics.median(values)
    fail("bootstrap_statistic_unknown", statistic)


def _bounded_sha256_index(context_digest: bytes, replicate: int, draw: int, bound: int) -> int:
    require(len(context_digest) == 32, "bootstrap_context_digest_invalid")
    integer(replicate, "bootstrap_replicate_invalid", 0)
    integer(draw, "bootstrap_draw_invalid", 0)
    integer(bound, "bootstrap_bound_invalid", 1)
    modulus = 1 << 256
    acceptance_limit = modulus - modulus % bound
    rejection = 0
    while True:
        digest = hashlib.sha256(
            context_digest + replicate.to_bytes(8, "big") + draw.to_bytes(8, "big") +
            rejection.to_bytes(8, "big")
        ).digest()
        candidate = int.from_bytes(digest, "big")
        if candidate < acceptance_limit:
            return candidate % bound
        rejection += 1


def run_cluster_bootstrap(
    groups: Mapping[str, Sequence[float]], statistic: str, context: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(run_id): [finite_number(value, "bootstrap_value_nonfinite") for value in values]
        for run_id, values in sorted(groups.items()) if values
    }
    base = {
        "algorithm": BOOTSTRAP_ALGORITHM,
        "seed_hex": BOOTSTRAP_SEED_HEX,
        "statistic": statistic,
        "independent_run_count": len(prepared),
    }
    if len(prepared) < MIN_RUNS:
        return {
            **base,
            "status": "unknown_insufficient_independent_runs",
            "replicate_count": 0,
            "interval_95": [None, None],
        }
    run_ids = sorted(prepared)
    context_digest = hashlib.sha256(
        bytes.fromhex(BOOTSTRAP_SEED_HEX) + canonical_json_bytes({
            "context": dict(context),
            "run_ids": run_ids,
            "statistic": statistic,
        })
    ).digest()
    estimates: list[float] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled: list[float] = []
        random_draw = 0
        for _ in range(len(run_ids)):
            selected_run = run_ids[_bounded_sha256_index(
                context_digest, replicate, random_draw, len(run_ids),
            )]
            random_draw += 1
            within_run = prepared[selected_run]
            for _ in range(len(within_run)):
                sampled.append(within_run[_bounded_sha256_index(
                    context_digest, replicate, random_draw, len(within_run),
                )])
                random_draw += 1
        estimates.append(_bootstrap_statistic(sampled, statistic))
    return {
        **base,
        "status": "estimated_run_cluster_bootstrap",
        "replicate_count": BOOTSTRAP_REPLICATES,
        "interval_95": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
    }


def beta_binomial_run_variance(groups: Mapping[str, Sequence[int | bool]]) -> dict[str, Any]:
    prepared: dict[str, list[int]] = {}
    for run_id, values in sorted(groups.items()):
        clean: list[int] = []
        for value in values:
            require(value in (0, 1, False, True), "binary_variance_value_invalid")
            clean.append(int(value))
        if clean:
            prepared[str(run_id)] = clean
    if len(prepared) < MIN_RUNS:
        return {
            "status": "unknown_insufficient_independent_runs",
            "model": "beta_binomial_method_of_moments",
            "independent_run_count": len(prepared),
            "observation_count": sum(len(values) for values in prepared.values()),
            "mean_probability": None,
            "between_run_intraclass_correlation": None,
            "between_run_probability_variance": None,
            "alpha": None,
            "beta": None,
            "leave_one_run_out_probability": [],
        }
    total = sum(len(values) for values in prepared.values())
    successes = sum(sum(values) for values in prepared.values())
    probability = successes / total
    run_probabilities = [statistics.fmean(values) for values in prepared.values()]
    if probability in (0.0, 1.0):
        rho = 0.0
        alpha = None
        beta = None
        status = "estimated_beta_binomial_boundary_no_binary_variation"
    else:
        observed_run_variance = statistics.variance(run_probabilities)
        average_inverse_n = statistics.fmean(1.0 / len(values) for values in prepared.values())
        denominator = 1.0 - average_inverse_n
        require(denominator > 0.0, "binary_variance_no_within_run_replication")
        rho = max(0.0, min(1.0, (
            observed_run_variance / (probability * (1.0 - probability)) - average_inverse_n
        ) / denominator))
        if rho == 0.0:
            alpha = None
            beta = None
        else:
            concentration = 1.0 / rho - 1.0
            alpha = probability * concentration
            beta = (1.0 - probability) * concentration
        status = "estimated_beta_binomial_method_of_moments"
    leave_one_out = []
    for omitted in sorted(prepared):
        retained = [
            value for run_id, values in prepared.items() if run_id != omitted for value in values
        ]
        leave_one_out.append({
            "omitted_run_id": omitted,
            "probability": statistics.fmean(retained) if retained else None,
        })
    return {
        "status": status,
        "model": "beta_binomial_method_of_moments",
        "independent_run_count": len(prepared),
        "observation_count": total,
        "success_count": successes,
        "mean_probability": probability,
        "between_run_intraclass_correlation": rho,
        "between_run_probability_variance": probability * (1.0 - probability) * rho,
        "alpha": alpha,
        "beta": beta,
        "per_run_probability": {
            run_id: statistics.fmean(values) for run_id, values in sorted(prepared.items())
        },
        "leave_one_run_out_probability": leave_one_out,
    }


def classify_cell(
    successes: int, trials: int, run_count: int, uncensored_timing_count: int | None = None,
) -> str:
    if trials < MIN_TRIALS or run_count < MIN_RUNS:
        return "unresolved"
    low, high = wilson_interval(successes, trials)
    if uncensored_timing_count is not None:
        half_width = (high - low) / 2.0
        if trials > MAX_TRIALS or uncensored_timing_count < 10 or half_width > 0.15:
            return "unresolved"
    if low >= 0.50:
        return "supported_client_temporal_zone"
    if high <= 0.10:
        return "supported_client_temporal_miss"
    return "transition"


def kaplan_meier(observations: Sequence[tuple[float, bool]]) -> list[dict[str, Any]]:
    clean = [(finite_number(time, "km_time_invalid"), bool(observed)) for time, observed in observations]
    require(all(time >= 0.0 for time, _ in clean), "km_negative_time")
    grouped: dict[float, list[bool]] = defaultdict(list)
    for time, observed in clean:
        grouped[time].append(observed)
    at_risk = len(clean)
    survival = 1.0
    result: list[dict[str, Any]] = []
    for time in sorted(grouped):
        flags = grouped[time]
        events = sum(flags)
        censored = len(flags) - events
        if events:
            survival *= 1.0 - events / at_risk
        result.append({
            "time_s": time,
            "at_risk": at_risk,
            "events": events,
            "censored": censored,
            "survival": survival,
        })
        at_risk -= len(flags)
    return result


def one_way_variance_components(groups: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    prepared = {str(key): [finite_number(value, "variance_nonfinite") for value in values] for key, values in groups.items() if values}
    if len(prepared) < MIN_RUNS:
        return {
            "status": "unknown_insufficient_independent_runs",
            "independent_run_count": len(prepared),
            "between_run_variance_tau2": None,
            "within_run_variance_sigma2": None,
            "leave_one_run_out_grand_mean": [],
        }
    observations = [value for values in prepared.values() for value in values]
    if len(observations) <= len(prepared):
        return {
            "status": "unknown_no_within_run_replication",
            "independent_run_count": len(prepared),
            "observation_count": len(observations),
            "between_run_variance_tau2": None,
            "within_run_variance_sigma2": None,
            "leave_one_run_out_grand_mean": [{
                "omitted_run_id": omitted,
                "mean": statistics.fmean(
                    value for run_id, values in prepared.items() if run_id != omitted for value in values
                ),
            } for omitted in sorted(prepared)],
        }
    overall = statistics.fmean(observations)
    within_ss = sum(sum((value - statistics.fmean(values)) ** 2 for value in values) for values in prepared.values())
    within_df = len(observations) - len(prepared)
    sigma2_mom = within_ss / within_df
    means = {key: statistics.fmean(values) for key, values in prepared.items()}
    between_ss = sum(len(prepared[key]) * (means[key] - overall) ** 2 for key in prepared)
    between_ms = between_ss / (len(prepared) - 1)
    n_total = len(observations)
    k0 = (n_total - sum(len(values) ** 2 for values in prepared.values()) / n_total) / (len(prepared) - 1)
    tau2_mom = max(0.0, (between_ms - sigma2_mom) / k0)
    scale = statistics.variance(observations) if len(observations) > 1 else 1.0
    scale = max(scale, 1e-18)
    sigma2, tau2 = _reml_pattern_search(prepared, sigma2_mom, tau2_mom, scale)
    total = sigma2 + tau2
    return {
        "status": "estimated_reml_random_run_intercept",
        "algorithm": "deterministic_log_variance_pattern_search_with_zero_tau_boundary",
        "independent_run_count": len(prepared),
        "observation_count": len(observations),
        "between_run_variance_tau2": tau2,
        "within_run_variance_sigma2": sigma2,
        "total_variance_tau2_plus_sigma2": total,
        "intraclass_correlation": tau2 / total if total > 0.0 else 0.0,
        "between_run_standard_deviation": math.sqrt(tau2),
        "within_run_standard_deviation": math.sqrt(sigma2),
        "repeatability_coefficient": 2.77 * math.sqrt(total),
        "method_of_moments": {"tau2": tau2_mom, "sigma2": sigma2_mom},
        "per_run_medians": {key: statistics.median(values) for key, values in sorted(prepared.items())},
        "mad_of_run_medians": statistics.median(
            abs(value - statistics.median(means.values())) for value in means.values()
        ),
        "leave_one_run_out_grand_mean": [{
            "omitted_run_id": omitted,
            "mean": statistics.fmean(
                value for run_id, values in prepared.items() if run_id != omitted for value in values
            ),
        } for omitted in sorted(prepared)],
    }


def _reml_objective(groups: Mapping[str, Sequence[float]], sigma2: float, tau2: float) -> float:
    if not math.isfinite(sigma2) or not math.isfinite(tau2) or sigma2 <= 0.0 or tau2 < 0.0:
        return math.inf
    logdet = 0.0
    xvx = 0.0
    xvy = 0.0
    for values in groups.values():
        n = len(values)
        denom = sigma2 + n * tau2
        logdet += (n - 1) * math.log(sigma2) + math.log(denom)
        xvx += n / denom
        xvy += sum(values) / denom
    if xvx <= 0.0:
        return math.inf
    mean = xvy / xvx
    quadratic = 0.0
    for values in groups.values():
        group_mean = statistics.fmean(values)
        quadratic += sum((value - group_mean) ** 2 for value in values) / sigma2
        quadratic += len(values) * (group_mean - mean) ** 2 / (sigma2 + len(values) * tau2)
    return 0.5 * (logdet + math.log(xvx) + quadratic)


def _reml_pattern_search(
    groups: Mapping[str, Sequence[float]], sigma_start: float, tau_start: float, scale: float
) -> tuple[float, float]:
    floor = max(scale * 1e-12, 1e-24)
    starts = [
        (max(sigma_start, floor), max(tau_start, floor)),
        (max(scale, floor), max(scale, floor)),
        (max(scale / 10.0, floor), max(scale * 10.0, floor)),
    ]
    best_pair = starts[0]
    best_value = math.inf
    for start_sigma, start_tau in starts:
        logs = [math.log(start_sigma), math.log(start_tau)]
        step = 4.0
        current = _reml_objective(groups, math.exp(logs[0]), math.exp(logs[1]))
        for _ in range(120):
            choices: list[tuple[float, float, float]] = []
            for ds in (-step, 0.0, step):
                for dt in (-step, 0.0, step):
                    sigma = max(math.exp(logs[0] + ds), floor)
                    tau = max(math.exp(logs[1] + dt), floor)
                    choices.append((_reml_objective(groups, sigma, tau), sigma, tau))
            choices.sort(key=lambda value: (value[0], value[1], value[2]))
            if choices[0][0] + 1e-14 < current:
                current, sigma, tau = choices[0]
                logs = [math.log(sigma), math.log(tau)]
            else:
                step *= 0.5
            if step < 1e-9:
                break
        if current < best_value:
            best_value = current
            best_pair = (math.exp(logs[0]), math.exp(logs[1]))
    sigma_zero = sum(
        (value - statistics.fmean(v for values in groups.values() for v in values)) ** 2
        for values in groups.values() for value in values
    ) / (sum(len(values) for values in groups.values()) - 1)
    zero_value = _reml_objective(groups, max(sigma_zero, floor), 0.0)
    if zero_value <= best_value:
        return max(sigma_zero, floor), 0.0
    return best_pair


def validate_schedule(
    schedule: Mapping[str, Any], expected_runner_hash: str,
    protocol: Mapping[str, Any] | None = None,
) -> dict[int, Mapping[str, Any]]:
    require(schedule.get("attack_zone_trial_schema") == RUNNER_SCHEMA, "schedule_runner_schema_mismatch")
    require(schedule.get("protocol_sha256") == expected_runner_hash, "schedule_runner_hash_mismatch")
    require(schedule.get("schedule_schema") == SCHEDULE_SCHEMA, "schedule_schema_mismatch")
    require(schedule.get("randomization_algorithm") == "sha256_counter_fisher_yates_rejection_v1", "schedule_randomization_mismatch")
    require(valid_sha256(schedule.get("randomization_seed_hex")), "schedule_seed_invalid")
    entries = sequence(required(schedule, "entries"), "schedule_entries_missing")
    protocol_cells = set(_protocol_cells(protocol)) if protocol is not None else None
    profile_assets = {
        item["move_index"]: item["serialized_asset_sha256"]
        for item in protocol["move_profiles"]
    } if protocol is not None else None
    result: dict[int, Mapping[str, Any]] = {}
    for raw in entries:
        entry = mapping(raw, "schedule_entry_invalid")
        ordinal = integer(required(entry, "schedule_ordinal"), "schedule_ordinal_invalid", 0)
        require(ordinal not in result, "duplicate_schedule_ordinal", str(ordinal))
        require(entry.get("move_index") in ALLOWED_MOVES, "schedule_move_invalid")
        require(valid_sha256(entry.get("serialized_asset_sha256")), "schedule_asset_hash_invalid")
        distance = normalize_requested_bin(
            mapping(required(entry, "distance_bin"), "schedule_distance_bin_missing"), "distance",
        )
        bearing = normalize_requested_bin(
            mapping(required(entry, "bearing_bin"), "schedule_bearing_bin_missing"), "bearing",
        )
        if protocol_cells is not None and profile_assets is not None:
            cell = (entry["move_index"], distance["id"], bearing["id"])
            require(cell in protocol_cells, "schedule_cell_not_in_protocol_grid", repr(cell))
            require(
                entry["serialized_asset_sha256"] == profile_assets[entry["move_index"]],
                "schedule_asset_profile_mismatch", str(entry["move_index"]),
            )
        result[ordinal] = entry
    require(sorted(result) == list(range(len(result))), "schedule_ordinals_not_dense")
    return result


def _event_identity_tuple(event: Mapping[str, Any]) -> tuple[Any, ...]:
    detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
    return (
        event.get("trial_id"), event.get("event"), event.get("stopwatch_timestamp_ticks"),
        event.get("client_control_tick"), detail.get("raw_hit_sequence"),
    )


def reject_conflicting_duplicates(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_identity: dict[tuple[Any, ...], bytes] = {}
    accepted: list[Mapping[str, Any]] = []
    for event in events:
        identity = _event_identity_tuple(event)
        encoded = canonical_json_bytes({key: value for key, value in event.items() if not key.startswith("_analysis_")})
        if identity in by_identity:
            require(by_identity[identity] == encoded, "conflicting_duplicate_event", repr(identity))
            continue
        by_identity[identity] = encoded
        accepted.append(event)
    return accepted


def analyze_event_bundle(
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    schedules: Sequence[tuple[str, Mapping[str, Any]]],
    events: Sequence[Mapping[str, Any]],
    input_kind: str,
) -> dict[str, Any]:
    validate_protocol(protocol)
    validate_analysis_spec(spec)
    require(input_kind in {"fixture", "real"}, "input_kind_invalid")
    runner_contract = next(
        (item for item in spec["input_contracts"] if item.get("role") == "attack_zone_runner_contract"), None
    )
    controller_contract = next(
        (item for item in spec["input_contracts"] if item.get("role") == "continuous_lifecycle_contract"), None
    )
    require(isinstance(runner_contract, dict) and isinstance(controller_contract, dict), "analysis_contract_pins_missing")
    runner_hash = string(runner_contract.get("embedded_canonical_sha256"), "runner_hash_missing")
    controller_hash = string(controller_contract.get("embedded_canonical_sha256"), "controller_hash_missing")
    require(valid_sha256(runner_hash) and valid_sha256(controller_hash), "analysis_contract_hash_invalid")
    schedule_maps: dict[str, dict[int, Mapping[str, Any]]] = {}
    for schedule_sha, schedule in schedules:
        require(valid_sha256(schedule_sha), "schedule_file_hash_invalid")
        require(schedule_sha not in schedule_maps, "duplicate_schedule_hash")
        schedule_maps[schedule_sha] = validate_schedule(schedule, runner_hash, protocol)
        if input_kind == "real":
            scheduled_cells = {
                (
                    entry["move_index"],
                    normalize_requested_bin(mapping(entry["distance_bin"], "schedule_distance_bin_missing"), "distance")["id"],
                    normalize_requested_bin(mapping(entry["bearing_bin"], "schedule_bearing_bin_missing"), "bearing")["id"],
                )
                for entry in schedule_maps[schedule_sha].values()
            }
            require(
                scheduled_cells == set(_protocol_cells(protocol)),
                "real_schedule_grid_incomplete",
                schedule_sha,
            )
    cleaned = reject_conflicting_duplicates(events)
    relevant = [event for event in cleaned if event.get("attack_zone_schema") == RUNNER_SCHEMA]
    require(relevant, "no_attack_zone_events")
    for event in relevant:
        validate_event_envelope(event, runner_hash, controller_hash)
        require(event["schedule_sha256"] in schedule_maps, "event_schedule_not_supplied")
        entry = schedule_maps[event["schedule_sha256"]].get(event["schedule_ordinal"])
        require(entry is not None, "event_schedule_ordinal_missing")
        require(entry["move_index"] == event["move_index"], "event_schedule_move_mismatch")
        require(entry["serialized_asset_sha256"] == event["serialized_asset_sha256"], "event_schedule_asset_mismatch")
        event_distance = normalize_requested_bin(
            mapping(event["requested_distance_bin"], "requested_distance_bin_missing"), "distance",
        )
        event_bearing = normalize_requested_bin(
            mapping(event["requested_bearing_bin"], "requested_bearing_bin_missing"), "bearing",
        )
        schedule_distance = normalize_requested_bin(
            mapping(entry["distance_bin"], "schedule_distance_bin_missing"), "distance",
        )
        schedule_bearing = normalize_requested_bin(
            mapping(entry["bearing_bin"], "schedule_bearing_bin_missing"), "bearing",
        )
        require(event_distance == schedule_distance, "event_schedule_distance_bin_mismatch")
        require(event_bearing == schedule_bearing, "event_schedule_bearing_bin_mismatch")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in relevant:
        grouped[event["trial_id"]].append(event)
    trial_ledgers: list[dict[str, Any]] = []
    hit_candidates: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for trial_id in sorted(grouped):
        trial_events = sorted(grouped[trial_id], key=lambda event: (
            event["stopwatch_timestamp_ticks"], event["client_control_tick"], event["event"]
        ))
        first = trial_events[0]
        invariant_names = (
            "schedule_sha256", "schedule_ordinal", "independent_run_id", "session_identity_sha256",
            "round_identity_sha256", "trial_id", "action_sequence", "move_index", "serialized_asset_sha256",
        )
        for name in invariant_names:
            require(all(event.get(name) == first.get(name) for event in trial_events), "trial_identity_changed", name)
        state_events = [event for event in trial_events if isinstance(event.get("measured_state"), dict)]
        require(state_events, "trial_measured_state_missing")
        identity = validate_runtime_identity(mapping(state_events[0]["measured_state"], "trial_measured_state_invalid"))
        target_events = [event for event in trial_events if event.get("event") == "target_requested"]
        require(len(target_events) == 1, "target_requested_count_invalid")
        distance_bin = normalize_requested_bin(
            mapping(required(first, "requested_distance_bin"), "requested_distance_bin_missing"),
            "distance",
        )
        bearing_bin = normalize_requested_bin(
            mapping(required(first, "requested_bearing_bin"), "requested_bearing_bin_missing"),
            "bearing",
        )
        acquisition_events = [event for event in trial_events if event.get("event") == "acquisition_sample"]
        passing_samples: list[dict[str, Any]] = []
        for event in acquisition_events:
            try:
                candidate = normalized_settle_sample(event)
            except AnalysisFailure:
                continue
            if all(candidate.get(flag) is True for flag in (
                "neutral_request_method_returned", "velocity_command_exact_neutral", "local_action_ready",
                "no_pending_requests", "local_healthy", "opponent_healthy",
            )):
                passing_samples.append(candidate)
        acquired_events = [event for event in trial_events if event.get("event") == "target_acquired"]
        settle_digest = None
        if acquired_events:
            require(len(acquired_events) == 1, "target_acquired_count_invalid")
            require(len(passing_samples) >= SETTLE_TICKS, "target_acquired_without_15_samples")
            settle_digest = validate_settle_samples(passing_samples[-SETTLE_TICKS:], distance_bin, bearing_bin)
        move_name = _move_name(protocol, first["move_index"])
        lifecycle = validate_lifecycle(trial_events, move_name)
        raw_events = [event for event in trial_events if event.get("event") == "raw_rek_hit_observed"]
        score_events = [event for event in trial_events if event.get("event") == "round_score_delta_observed"]
        contamination = [event for event in trial_events if event.get("event") in {
            "fall_observed", "recovery_request_observed", "recovery_state_observed", "trial_interrupted"
        }]
        candidates_by_sequence: dict[int, dict[str, Any]] = {}
        for raw_event in raw_events:
            try:
                candidate = raw_hit_location_candidate(raw_event)
            except AnalysisFailure as exc:
                candidate = {
                    "raw_hit_sequence": mapping(raw_event.get("detail"), "raw_hit_detail_missing").get("raw_hit_sequence"),
                    "association_status": "unattributed_rek_hit_location_candidate",
                    "location_status": "unresolved",
                    "failure_code": exc.code,
                }
            candidate.update({
                "trial_id": trial_id,
                "independent_run_id": first["independent_run_id"],
                "move_index": first["move_index"],
                "entry_distance_bin_id": distance_bin.get("id"),
                "entry_bearing_bin_id": bearing_bin.get("id"),
            })
            if isinstance(candidate.get("raw_hit_sequence"), int):
                candidates_by_sequence[candidate["raw_hit_sequence"]] = candidate
            hit_candidates.append(candidate)
        join_passed = False
        joined_sequences: set[int] = set()
        join_failure_reasons: list[str] = []
        for score_event in score_events:
            join = join_local_scoring_candidate(raw_events, score_event, trial_events, contamination)
            if join["passed"]:
                join_passed = True
                joined_sequences.add(join["raw_hit_sequence"])
                candidate = candidates_by_sequence.get(join["raw_hit_sequence"])
                if candidate is not None:
                    candidate["association_status"] = join["label"]
                    candidate["association_reason"] = join["reason"]
            else:
                join_failure_reasons.append(str(join["reason"]))
        for candidate in candidates_by_sequence.values():
            if "association_reason" not in candidate:
                candidate["association_reason"] = (
                    sorted(set(join_failure_reasons)) if join_failure_reasons else
                    ["no_positive_local_score_join_observed"]
                )
        auxiliary = derive_auxiliary_timings(trial_events, joined_sequences)
        terminal_censored = bool([event for event in trial_events if event.get("event") in {"trial_censored", "trial_interrupted"}])
        motion_path = opponent_motion_path(trial_events, settle_digest)
        command_events = [event for event in trial_events if event.get("event") == "local_command_edge_set"]
        return_events = [event for event in trial_events if event.get("event") == "client_request_method_returned"]
        start_events = [event for event in trial_events if event.get("event") == "local_motion_start_observed"]
        completion_events = [
            event for event in trial_events
            if event.get("event") == "local_motion_completion_and_readiness_observed"
        ]
        command_tick = command_events[0]["stopwatch_timestamp_ticks"] if command_events else None
        pre_request_contamination = bool(
            contamination and (
                command_tick is None or min(event["stopwatch_timestamp_ticks"] for event in contamination) <= command_tick
            )
        )
        eligible_any_motion = bool(settle_digest and not pre_request_contamination)
        stationary_through_observed_endpoint = (
            motion_path["motion_stratum"] == "stationary" and not motion_path["time_varying_motion"]
        )
        primary = bool(
            eligible_any_motion and settle_digest["all_opponent_stationary"] and
            stationary_through_observed_endpoint and not contamination
        )
        exclusion_reasons: list[str] = []
        if not acquired_events:
            exclusion_reasons.append("target_not_acquired")
        if settle_digest is None:
            exclusion_reasons.append("settle_not_validated")
        elif not settle_digest["all_opponent_stationary"]:
            exclusion_reasons.append("opponent_not_stationary_during_settle")
        if motion_path["time_varying_motion"]:
            exclusion_reasons.append("opponent_motion_stratum_changed")
        if motion_path["time_varying_facing"]:
            exclusion_reasons.append("opponent_facing_stratum_changed")
        exclusion_reasons.extend(f"contamination:{event['event']}" for event in contamination)
        if not start_events:
            exclusion_reasons.append("local_start_not_observed")
        if terminal_censored or lifecycle["censored"]:
            exclusion_reasons.append(f"censored:{lifecycle['censor_reason'] or 'terminal'}")
        ledger = {
            "trial_id": trial_id,
            "independent_run_id": first["independent_run_id"],
            "session_identity_sha256": first["session_identity_sha256"],
            "round_identity_sha256": first["round_identity_sha256"],
            "action_sequence": first["action_sequence"],
            "move_index": first["move_index"],
            "serialized_asset_sha256": first["serialized_asset_sha256"],
            "entry_distance_bin_id": distance_bin.get("id"),
            "entry_bearing_bin_id": bearing_bin.get("id"),
            "scheduled": True,
            "target_acquired": bool(acquired_events),
            "settle_complete": settle_digest is not None,
            "eligible_any_motion_stratum": eligible_any_motion,
            "eligible_stationary_primary": bool(settle_digest and settle_digest["all_opponent_stationary"]),
            "request_edge_observed": len(command_events) == 1,
            "request_method_return_observed": len(return_events) == 1,
            "local_start_observed": lifecycle["request_to_local_composer_start_s"] is not None,
            "local_completion_observed": lifecycle["request_to_completion_readiness_s"] is not None,
            "uncensored_timing": not terminal_censored and not lifecycle["censored"],
            "raw_hit_observed": bool(raw_events),
            "local_score_observed": any(mapping(event.get("detail"), "score_detail_missing").get("local_clean_hit_delta", 0) > 0 for event in score_events),
            "isolated_local_scoring_join_passed": join_passed,
            "fall_recovery_contaminated": bool(contamination),
            "primary_analysis_eligible": primary and lifecycle["request_to_local_composer_start_s"] is not None,
            "timing_analysis_eligible": eligible_any_motion,
            "censored": terminal_censored or lifecycle["censored"],
            "exclusion_reasons": sorted(exclusion_reasons),
            "runtime_identity": identity,
            "settle_digest": settle_digest,
            "opponent_motion_path": motion_path,
        }
        trial_ledgers.append(ledger)
        timing_rows.append({
            "trial_id": trial_id,
            "independent_run_id": first["independent_run_id"],
            "move_index": first["move_index"],
            "entry_distance_bin_id": distance_bin.get("id"),
            "entry_bearing_bin_id": bearing_bin.get("id"),
            "timing_analysis_eligible": eligible_any_motion,
            "primary_analysis_eligible": ledger["primary_analysis_eligible"],
            "opponent_motion_stratum": motion_path["motion_stratum"],
            "opponent_facing_stratum": motion_path["facing_stratum"],
            "opponent_motion_path": motion_path["entries"],
            **lifecycle,
            **auxiliary,
        })
    cell_outcomes = _cell_outcomes(trial_ledgers, protocol)
    timing_distributions = _timing_distributions(timing_rows, protocol)
    impact_maps = _impact_maps(trial_ledgers, hit_candidates, protocol)
    run_variance = _run_variance(timing_rows, trial_ledgers, hit_candidates, protocol)
    full_cell_count = len(_protocol_cells(protocol))
    complete_cells = sum(value["label"] != "unresolved" for value in cell_outcomes)
    empirical_map_available = mapping_completed_from_cells(
        input_kind, cell_outcomes, full_cell_count,
    )
    return {
        "schema": ANALYZER_SCHEMA,
        "mapping_completed": empirical_map_available,
        "input_kind": input_kind,
        "authority": {
            "server_acceptance_observed": False,
            "authoritative_execution_observed": False,
            "confirmed_command_hit_causality": False,
        },
        "trial_ledger": trial_ledgers,
        "action_lifecycle_timings": timing_rows,
        "hit_location_candidates": hit_candidates,
        "per_move_cell_outcomes": cell_outcomes,
        "per_move_timing_distributions": timing_distributions,
        "per_move_impact_zone_maps": impact_maps,
        "opponent_motion_strata": _opponent_motion_output(trial_ledgers, protocol),
        "repeated_run_variance": run_variance,
        "mapping_gate": {
            "required_cell_count": full_cell_count,
            "resolved_cell_count": complete_cells,
            "real_input_required": True,
            "statement": "No authoritative or confirmed-causal map is produced.",
        },
    }


def mapping_completed_from_cells(
    input_kind: str, cell_outcomes: Sequence[Mapping[str, Any]], required_cell_count: int,
) -> bool:
    require(input_kind in {"fixture", "real"}, "input_kind_invalid")
    integer(required_cell_count, "required_cell_count_invalid", 1)
    require(len(cell_outcomes) == required_cell_count, "cell_outcome_count_mismatch")
    return input_kind == "real" and all(
        value.get("label") in {
            "supported_client_temporal_zone", "supported_client_temporal_miss", "transition",
        }
        for value in cell_outcomes
    )


def _move_name(protocol: Mapping[str, Any], move_index: int) -> str:
    matches = [item for item in protocol["move_profiles"] if item["move_index"] == move_index]
    require(len(matches) == 1, "move_profile_lookup_failed", str(move_index))
    return string(matches[0].get("move_name"), "move_name_missing")


TIMING_METRICS = (
    "request_method_duration_s",
    "request_to_local_composer_start_s",
    "method_return_to_local_composer_start_s",
    "local_composer_start_to_completion_readiness_s",
    "request_to_completion_readiness_s",
    "local_start_to_configured_marker_s",
    "request_to_raw_hit_observation_s",
    "local_start_to_raw_hit_observation_s",
    "request_to_isolated_local_scoring_hit_candidate_s",
    "local_start_to_isolated_local_scoring_hit_candidate_s",
    "request_to_first_temporal_local_score_delta_s",
    "request_to_pose_entry_s",
    "pose_entry_to_pose_return_s",
)


def _cell_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        integer(row["move_index"], "cell_move_invalid"),
        string(row["entry_distance_bin_id"], "cell_distance_bin_invalid"),
        string(row["entry_bearing_bin_id"], "cell_bearing_bin_invalid"),
    )


def _cell_outcomes(
    ledgers: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    valid_cells = set(_protocol_cells(protocol))
    for row in ledgers:
        key = _cell_key(row)
        require(key in valid_cells, "ledger_cell_not_in_protocol_grid", repr(key))
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key in _protocol_cells(protocol):
        attempts = groups.get(key, [])
        eligible = [row for row in attempts if row["primary_analysis_eligible"]]
        successes = sum(bool(row["isolated_local_scoring_join_passed"]) for row in eligible)
        runs = len({row["independent_run_id"] for row in eligible})
        uncensored = sum(bool(row["uncensored_timing"]) for row in eligible)
        interval = wilson_interval(successes, len(eligible)) if eligible else (None, None)
        by_run: dict[str, list[float]] = defaultdict(list)
        for row in eligible:
            by_run[str(row["independent_run_id"])].append(
                1.0 if row["isolated_local_scoring_join_passed"] else 0.0,
            )
        bootstrap = run_cluster_bootstrap(by_run, "proportion", {
            "output": "per_move_cell_outcomes",
            "move_index": key[0],
            "distance_bin_id": key[1],
            "bearing_bin_id": key[2],
        })
        half_width = (interval[1] - interval[0]) / 2.0 if interval[0] is not None else None
        output.append({
            "move_index": key[0],
            "entry_distance_bin_id": key[1],
            "entry_bearing_bin_id": key[2],
            "scheduled_attempts": len(attempts),
            "target_acquired_trials": sum(bool(row["target_acquired"]) for row in attempts),
            "settle_eligible_trials": sum(bool(row["settle_complete"]) for row in attempts),
            "request_edge_observed_trials": sum(bool(row["request_edge_observed"]) for row in attempts),
            "request_method_return_observed_trials": sum(
                bool(row["request_method_return_observed"]) for row in attempts
            ),
            "local_start_observed_trials": sum(bool(row["local_start_observed"]) for row in attempts),
            "local_completion_observed_trials": sum(
                bool(row["local_completion_observed"]) for row in attempts
            ),
            "uncensored_timing_trials": sum(bool(row["uncensored_timing"]) for row in attempts),
            "local_score_observed_trials": sum(bool(row["local_score_observed"]) for row in attempts),
            "joined_candidate_trials": sum(
                bool(row["isolated_local_scoring_join_passed"]) for row in attempts
            ),
            "primary_eligible_trials": len(eligible),
            "primary_eligible_uncensored_timing_trials": uncensored,
            "independent_run_count": runs,
            "successes": successes,
            "probability": successes / len(eligible) if eligible else None,
            "wilson_95": list(interval),
            "wilson_half_width": half_width,
            "run_cluster_bootstrap_95": bootstrap,
            "sampling_status": (
                "no_observations" if not attempts else
                "protocol_max_exceeded" if len(eligible) > MAX_TRIALS else
                "sufficient_for_discrete_label" if classify_cell(
                    successes, len(eligible), runs, uncensored,
                ) != "unresolved" else
                "insufficient_or_imprecise"
            ),
            "label": classify_cell(successes, len(eligible), runs, uncensored),
        })
    return output


def _timing_metric_variants(
    protocol: Mapping[str, Any], move_index: int, metric: str,
) -> list[dict[str, Any]]:
    if metric != "local_start_to_configured_marker_s":
        return [{}]
    profiles = [value for value in protocol["move_profiles"] if value["move_index"] == move_index]
    require(len(profiles) == 1, "timing_move_profile_lookup_failed", str(move_index))
    markers = profiles[0].get("configured_impact_markers", [])
    if not markers:
        return [{"configured_marker_ordinal": None, "configured_marker_limb": None}]
    result = []
    for marker in markers:
        value = mapping(marker, "configured_marker_profile_invalid")
        result.append({
            "configured_marker_ordinal": integer(
                required(value, "ordinal"), "configured_marker_ordinal_invalid", 1,
            ),
            "configured_marker_limb": required(value, "limb"),
        })
    require(len({value["configured_marker_ordinal"] for value in result}) == len(result), "configured_marker_ordinal_duplicate")
    return sorted(result, key=lambda value: value["configured_marker_ordinal"])


def _timing_metric_values(
    row: Mapping[str, Any], metric: str, variant: Mapping[str, Any] | None = None,
) -> list[float]:
    require(metric in TIMING_METRICS, "timing_metric_unknown", metric)
    if metric == "local_start_to_configured_marker_s":
        ordinal = None if variant is None else variant.get("configured_marker_ordinal")
        return [
            finite_number(item[metric], "timing_value_nonfinite")
            for item in sequence(row.get("configured_markers", []), "configured_markers_invalid")
            if mapping(item, "configured_marker_timing_invalid").get(metric) is not None and
            (ordinal is None or item.get("marker_ordinal") == ordinal)
        ]
    if metric in {
        "request_to_raw_hit_observation_s",
        "local_start_to_raw_hit_observation_s",
        "request_to_isolated_local_scoring_hit_candidate_s",
        "local_start_to_isolated_local_scoring_hit_candidate_s",
    }:
        return [
            finite_number(item[metric], "timing_value_nonfinite")
            for item in sequence(row.get("raw_hits", []), "raw_hit_timings_invalid")
            if mapping(item, "raw_hit_timing_invalid").get(metric) is not None
        ]
    value = row.get(metric)
    return [] if value is None else [finite_number(value, "timing_value_nonfinite")]


def _timing_distributions(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    valid_cells = set(_protocol_cells(protocol))
    for row in rows:
        key = _cell_key(row)
        require(key in valid_cells, "timing_cell_not_in_protocol_grid", repr(key))
        groups[key].append(row)
    result: list[dict[str, Any]] = []
    for key in _protocol_cells(protocol):
        cell_rows = groups.get(key, [])
        eligible = [
            row for row in cell_rows
            if row.get("timing_analysis_eligible") is True and
            row.get("opponent_motion_stratum") == "stationary"
        ]
        for metric in TIMING_METRICS:
            for variant in _timing_metric_variants(protocol, key[0], metric):
                values = [
                    value for row in eligible for value in _timing_metric_values(row, metric, variant)
                ]
                by_run: dict[str, list[float]] = defaultdict(list)
                for row in eligible:
                    by_run[str(row["independent_run_id"])].extend(
                        _timing_metric_values(row, metric, variant),
                    )
                censor_observations: list[tuple[float, bool]] = []
                censored_reasons: Counter[str] = Counter()
                censored_count = 0
                if metric in {
                    "request_method_duration_s", "request_to_local_composer_start_s",
                    "method_return_to_local_composer_start_s",
                    "local_composer_start_to_completion_readiness_s",
                    "request_to_completion_readiness_s",
                }:
                    for row in eligible:
                        observed = _timing_metric_values(row, metric, variant)
                        if observed:
                            censor_observations.append((observed[0], True))
                            continue
                        censor_map = mapping(row.get("censor_durations_s", {}), "censor_durations_invalid")
                        censor_value = censor_map.get(metric)
                        if censor_value is not None:
                            censor_observations.append((
                                finite_number(censor_value, "censor_duration_nonfinite"), False,
                            ))
                            censored_count += 1
                            censored_reasons[str(row.get("censor_reason") or "unspecified_terminal")] += 1
                observed_trials = sum(bool(_timing_metric_values(row, metric, variant)) for row in eligible)
                missing = len(eligible) - observed_trials - censored_count
                result.append({
                    "move_index": key[0],
                    "entry_distance_bin_id": key[1],
                    "entry_bearing_bin_id": key[2],
                    "opponent_motion_stratum": "stationary",
                    "metric": metric,
                    "endpoint_variant": dict(variant),
                    "scheduled_attempt_count": len(cell_rows),
                    "eligible_trial_count": len(eligible),
                    "observed_trial_count": observed_trials,
                    "observed_endpoint_count": len(values),
                    "censored_count": censored_count,
                    "missing_unaligned_endpoint_count": max(0, missing),
                    "censored_by_reason": dict(sorted(censored_reasons.items())),
                    "summary": numeric_summary(values),
                    "run_cluster_bootstrap_95": run_cluster_bootstrap(by_run, "median", {
                        "output": "per_move_timing_distributions",
                        "move_index": key[0],
                        "distance_bin_id": key[1],
                        "bearing_bin_id": key[2],
                        "metric": metric,
                        "endpoint_variant": dict(variant),
                    }),
                    "kaplan_meier": {
                        "status": (
                            "estimated_with_right_censoring" if censored_count else
                            "complete_observations_no_right_censoring" if censor_observations else
                            "not_estimable_no_aligned_observations"
                        ),
                        "curve": kaplan_meier(censor_observations) if censor_observations else [],
                    },
                })
    return result


def _impact_failure_class(candidate: Mapping[str, Any]) -> str | None:
    code = str(candidate.get("failure_code") or "")
    if code.startswith(("root_", "quaternion_", "position_", "hit_position_")):
        return "root_alignment"
    if code and (
        code.startswith(("bone_", "nearest_bone_", "opponent_bone_", "t800_bone_")) or
        "bone" in code
    ):
        return "bone_alignment"
    return None


def _impact_maps(
    ledgers: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    candidate_groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledgers:
        ledger_groups[_cell_key(row)].append(row)
    for candidate in candidates:
        candidate_groups[_cell_key(candidate)].append(candidate)
    region_groups: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate.get("association_status") != "client_isolated_local_scoring_zone_candidate":
            continue
        region = candidate.get("candidate_opponent_body_region")
        if region is None:
            continue
        region_groups[(*_cell_key(candidate), str(region))].append(candidate)
    cell_failures: list[dict[str, Any]] = []
    failure_by_cell: dict[tuple[int, str, str], dict[str, Any]] = {}
    for key in _protocol_cells(protocol):
        cell_candidates = candidate_groups.get(key, [])
        failure_codes = Counter(str(value.get("failure_code")) for value in cell_candidates if value.get("failure_code"))
        association_reasons: Counter[str] = Counter()
        for value in cell_candidates:
            reasons = value.get("association_reason")
            if isinstance(reasons, list):
                association_reasons.update(str(reason) for reason in reasons)
            elif reasons is not None:
                association_reasons[str(reasons)] += 1
        summary = {
            "move_index": key[0],
            "entry_distance_bin_id": key[1],
            "entry_bearing_bin_id": key[2],
            "raw_hit_count": len(cell_candidates),
            "resolved_opponent_root_and_bone_candidate_count": sum(
                value.get("location_status") == "resolved_candidate" for value in cell_candidates
            ),
            "root_alignment_failure_count": sum(
                _impact_failure_class(value) == "root_alignment" for value in cell_candidates
            ),
            "bone_alignment_failure_count": sum(
                _impact_failure_class(value) == "bone_alignment" for value in cell_candidates
            ),
            "temporal_join_failure_count": sum(
                value.get("association_status") != "client_isolated_local_scoring_zone_candidate"
                for value in cell_candidates
            ),
            "failure_codes": dict(sorted(failure_codes.items())),
            "temporal_join_failure_reasons": dict(sorted(association_reasons.items())),
        }
        failure_by_cell[key] = summary
        cell_failures.append(summary)
    region_maps: list[dict[str, Any]] = []
    for key in sorted(region_groups):
        base = key[:3]
        eligible_ledgers = [row for row in ledger_groups.get(base, []) if row["primary_analysis_eligible"]]
        denominator = len(eligible_ledgers)
        values = region_groups[key]
        unique_trials = len({item["trial_id"] for item in values})
        interval = wilson_interval(unique_trials, denominator) if denominator else (None, None)
        coordinates = [item["candidate_hit_opponent_root_xyz_m"] for item in values]
        nearest_distances = [item["candidate_nearest_opponent_bone_distance_m"] for item in values]
        region_maps.append({
            "move_index": key[0],
            "entry_distance_bin_id": key[1],
            "entry_bearing_bin_id": key[2],
            "candidate_opponent_body_region": key[3],
            "eligible_trial_count": denominator,
            "trials_with_candidate": unique_trials,
            "candidate_count": len(values),
            "trial_level_incidence": unique_trials / denominator if denominator else None,
            "wilson_95": list(interval),
            "candidate_opponent_root_xyz_m": {
                axis: numeric_summary([value[index] for value in coordinates])
                for index, axis in enumerate(("x", "y", "z"))
            },
            "nearest_opponent_bone_counts": dict(sorted(Counter(
                str(item["candidate_nearest_opponent_bone_name"]) for item in values
            ).items())),
            "nearest_opponent_bone_distance_m": numeric_summary(nearest_distances),
            "alignment_and_join_failures_for_cell": failure_by_cell[base],
            "label": "client-isolated candidate target-location map",
            "authoritative": False,
        })
    return {
        "schema": ANALYZER_SCHEMA,
        "region_maps": region_maps,
        "cell_alignment_and_join_failure_counts": cell_failures,
    }


def _run_variance(
    timing_rows: Sequence[Mapping[str, Any]], ledgers: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
) -> dict[str, Any]:
    timing_by_cell: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    ledger_by_cell: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in timing_rows:
        timing_by_cell[_cell_key(row)].append(row)
    for row in ledgers:
        ledger_by_cell[_cell_key(row)].append(row)
    continuous: list[dict[str, Any]] = []
    for cell in _protocol_cells(protocol):
        eligible_timing = [
            row for row in timing_by_cell.get(cell, [])
            if row.get("timing_analysis_eligible") is True and
            row.get("opponent_motion_stratum") == "stationary"
        ]
        for metric in TIMING_METRICS:
            for variant in _timing_metric_variants(protocol, cell[0], metric):
                groups: dict[str, list[float]] = defaultdict(list)
                for row in eligible_timing:
                    groups[str(row["independent_run_id"])].extend(
                        _timing_metric_values(row, metric, variant),
                    )
                continuous.append({
                    "move_index": cell[0],
                    "entry_distance_bin_id": cell[1],
                    "entry_bearing_bin_id": cell[2],
                    "metric": metric,
                    "endpoint_variant": dict(variant),
                    "variance": one_way_variance_components(groups),
                })
    binary_success: list[dict[str, Any]] = []
    for cell in _protocol_cells(protocol):
        groups: dict[str, list[int]] = defaultdict(list)
        for row in ledger_by_cell.get(cell, []):
            if row["primary_analysis_eligible"]:
                groups[str(row["independent_run_id"])].append(
                    1 if row["isolated_local_scoring_join_passed"] else 0,
                )
        binary_success.append({
            "move_index": cell[0],
            "entry_distance_bin_id": cell[1],
            "entry_bearing_bin_id": cell[2],
            "metric": "client_isolated_local_scoring_success",
            "variance": beta_binomial_run_variance(groups),
        })
    regions_by_trial: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        if (
            candidate.get("association_status") == "client_isolated_local_scoring_zone_candidate" and
            candidate.get("candidate_opponent_body_region") is not None
        ):
            regions_by_trial[str(candidate["trial_id"])].add(str(candidate["candidate_opponent_body_region"]))
    binary_region: list[dict[str, Any]] = []
    for cell in _protocol_cells(protocol):
        eligible = [row for row in ledger_by_cell.get(cell, []) if row["primary_analysis_eligible"]]
        if not eligible:
            continue
        for region in BODY_REGIONS:
            groups = defaultdict(list)
            for row in eligible:
                groups[str(row["independent_run_id"])].append(
                    1 if region in regions_by_trial.get(str(row["trial_id"]), set()) else 0,
                )
            binary_region.append({
                "move_index": cell[0],
                "entry_distance_bin_id": cell[1],
                "entry_bearing_bin_id": cell[2],
                "candidate_opponent_body_region": region,
                "metric": "client_isolated_candidate_region_incidence",
                "variance": beta_binomial_run_variance(groups),
            })
    return {
        "schema": ANALYZER_SCHEMA,
        "continuous_timing": continuous,
        "binary_trial_success": binary_success,
        "binary_region_incidence": binary_region,
    }


def write_outputs(output: Path, result: Mapping[str, Any], input_audit: Sequence[Mapping[str, Any]]) -> None:
    require(not output.exists() or not any(output.iterdir()), "output_directory_not_empty", str(output))
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {
        "analysis-input-audit.json": {"schema": ANALYZER_SCHEMA, "inputs": list(input_audit)},
        "trial-ledger.jsonl": result["trial_ledger"],
        "action-lifecycle-timings.jsonl": result["action_lifecycle_timings"],
        "hit-location-candidates.jsonl": result["hit_location_candidates"],
        "per-move-cell-outcomes.json": result["per_move_cell_outcomes"],
        "per-move-timing-distributions.json": result["per_move_timing_distributions"],
        "per-move-impact-zone-maps.json": result["per_move_impact_zone_maps"],
        "opponent-motion-strata.json": result["opponent_motion_strata"],
        "repeated-run-variance.json": result["repeated_run_variance"],
        "analysis-audit.json": {
            "schema": ANALYZER_SCHEMA,
            "mapping_completed": result["mapping_completed"],
            "authority": result["authority"],
            "mapping_gate": result["mapping_gate"],
        },
    }
    manifest_entries: list[dict[str, Any]] = []
    for item in sorted(input_audit, key=lambda value: (str(value.get("path")), str(value.get("role")))):
        path = string(item.get("path"), "manifest_input_path_invalid")
        size = integer(item.get("bytes"), "manifest_input_size_invalid", 0)
        digest = item.get("sha256")
        require(valid_sha256(digest), "manifest_input_sha256_invalid", path)
        manifest_entries.append({
            "role": "input",
            "input_role": item.get("role"),
            "path": path,
            "bytes": size,
            "sha256": digest,
        })
    for name in sorted(files):
        path = output / name
        value = files[name]
        if name.endswith(".jsonl"):
            content = b"".join(canonical_json_bytes(item) for item in value)
        else:
            content = canonical_json_bytes(value)
        with path.open("xb") as handle:
            handle.write(content)
        manifest_entries.append({
            "role": "output", "path": name, "bytes": len(content), "sha256": sha256_bytes(content),
        })
    manifest = {
        "schema": "rek.attack_zone_analysis_manifest.v1",
        "hash_algorithm": "SHA-256",
        "self_reference_excluded": True,
        "mapping_completed": result["mapping_completed"],
        "files": manifest_entries,
    }
    with (output / "sha256-manifest.json").open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))


def _stratum_flow(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    excluded: Counter[str] = Counter()
    for row in rows:
        excluded.update(str(reason) for reason in row.get("exclusion_reasons", []))
    return {
        "scheduled": len(rows),
        "acquired": sum(bool(row["target_acquired"]) for row in rows),
        "eligible": sum(bool(row["eligible_any_motion_stratum"]) for row in rows),
        "started": sum(bool(row["local_start_observed"]) for row in rows),
        "completed": sum(bool(row["local_completion_observed"]) for row in rows),
        "raw_hit": sum(bool(row["raw_hit_observed"]) for row in rows),
        "joined_local_scoring_candidate": sum(
            bool(row["isolated_local_scoring_join_passed"]) for row in rows
        ),
        "excluded_by_reason": dict(sorted(excluded.items())),
    }


def _stratum_estimate(
    rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["eligible_any_motion_stratum"] and row["local_start_observed"] and
        not row["fall_recovery_contaminated"] and
        not mapping(row["opponent_motion_path"], "motion_path_invalid")["time_varying_motion"] and
        not mapping(row["opponent_motion_path"], "motion_path_invalid")["time_varying_facing"]
    ]
    successes = sum(bool(row["isolated_local_scoring_join_passed"]) for row in eligible)
    by_run: dict[str, list[float]] = defaultdict(list)
    for row in eligible:
        by_run[str(row["independent_run_id"])].append(
            1.0 if row["isolated_local_scoring_join_passed"] else 0.0,
        )
    interval = wilson_interval(successes, len(eligible)) if eligible else (None, None)
    return {
        "eligible_started_trial_count": len(eligible),
        "independent_run_count": len(by_run),
        "success_count": successes,
        "probability": successes / len(eligible) if eligible else None,
        "wilson_95": list(interval),
        "run_cluster_bootstrap_95": run_cluster_bootstrap(by_run, "proportion", context),
    }


def _opponent_motion_output(
    ledgers: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
) -> dict[str, Any]:
    confounds = mapping(required(protocol, "opponent_motion_confounds"), "opponent_motion_confounds_missing")
    configured_motion = tuple(
        string(mapping(value, "motion_stratum_invalid").get("id"), "motion_stratum_id_invalid")
        for value in sequence(required(confounds, "motion_strata"), "motion_strata_missing")
    )
    require(configured_motion == MOTION_STRATA[:-1], "motion_strata_contract_mismatch")
    configured_facing = tuple(
        string(mapping(value, "facing_stratum_invalid").get("id"), "facing_stratum_id_invalid")
        for value in sequence(required(confounds, "opponent_facing_strata"), "facing_strata_missing")
    )
    motion_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    facing_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledgers:
        path = mapping(row.get("opponent_motion_path"), "motion_path_missing")
        motion_rows[str(path.get("motion_stratum") or "unresolved")].append(row)
        facing_rows[str(path.get("facing_stratum") or "unresolved")].append(row)
    motion_output: list[dict[str, Any]] = []
    for stratum in MOTION_STRATA:
        rows = motion_rows.get(stratum, [])
        cell_groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            cell_groups[_cell_key(row)].append(row)
        motion_output.append({
            "opponent_motion_stratum": stratum,
            "analysis_role": "stationary_primary" if stratum == "stationary" else "separate_secondary",
            "flow_counts": _stratum_flow(rows),
            "overall_candidate_incidence": _stratum_estimate(rows, {
                "output": "opponent_motion_strata",
                "stratum": stratum,
                "scope": "overall",
            }),
            "per_cell_candidate_incidence": [{
                "move_index": key[0],
                "entry_distance_bin_id": key[1],
                "entry_bearing_bin_id": key[2],
                **_stratum_estimate(cell_rows, {
                    "output": "opponent_motion_strata",
                    "stratum": stratum,
                    "move_index": key[0],
                    "distance_bin_id": key[1],
                    "bearing_bin_id": key[2],
                }),
            } for key, cell_rows in sorted(cell_groups.items())],
        })
    facing_output: list[dict[str, Any]] = []
    for stratum in (*configured_facing, "opponent_facing_changed_or_unresolved", "unresolved"):
        rows = facing_rows.get(stratum, [])
        facing_output.append({
            "opponent_facing_stratum": stratum,
            "flow_counts": _stratum_flow(rows),
            "overall_candidate_incidence": _stratum_estimate(rows, {
                "output": "opponent_facing_strata",
                "stratum": stratum,
                "scope": "overall",
            }),
        })
    return {
        "schema": ANALYZER_SCHEMA,
        "primary_motion_stratum": "stationary",
        "motion_strata": motion_output,
        "facing_strata": facing_output,
        "time_varying_paths": [
            {
                "trial_id": row["trial_id"],
                "path": row["opponent_motion_path"],
                "excluded_from_single_stratum_estimates": True,
            }
            for row in sorted(ledgers, key=lambda value: str(value["trial_id"]))
            if mapping(row["opponent_motion_path"], "motion_path_invalid")["time_varying_motion"] or
            mapping(row["opponent_motion_path"], "motion_path_invalid")["time_varying_facing"]
        ],
    }


def _collect_jsonl_inputs(
    paths: Sequence[Path], role: str = "event_jsonl",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for path in paths:
        require(path.is_file(), "event_file_missing", str(path))
        values = read_jsonl(path)
        records.extend(values)
        audit.append({
            "role": role,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
            "records": len(values),
        })
    return records, audit


def _collect_recorder_roots(
    roots: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        require(root.is_dir(), "recorder_root_missing", str(root))
        for path in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: str(value.resolve())):
            resolved = path.resolve()
            require(resolved not in seen, "recorder_root_file_duplicate", str(resolved))
            seen.add(resolved)
            values = read_jsonl(path) if path.suffix.lower() == ".jsonl" else []
            records.extend(values)
            audit.append({
                "role": "recorder_root_file",
                "recorder_root": str(root.resolve()),
                "path": str(resolved),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "records_parsed": len(values),
            })
    return records, audit


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, action="append", required=True)
    parser.add_argument("--runner-events", type=Path, action="append", required=True)
    parser.add_argument("--bridge-events", type=Path, action="append", default=[])
    parser.add_argument("--recorder-events", type=Path, action="append", default=[])
    parser.add_argument("--recorder-roots", type=Path, action="append", default=[])
    parser.add_argument("--input-kind", choices=("fixture", "real"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    protocol = mapping(read_json(args.protocol), "protocol_not_object")
    spec = mapping(read_json(args.analysis_spec), "analysis_spec_not_object")
    schedules: list[tuple[str, Mapping[str, Any]]] = []
    audit: list[dict[str, Any]] = []
    for path in args.schedule:
        value = mapping(read_json(path), "schedule_not_object")
        digest = sha256_path(path)
        schedules.append((digest, value))
        audit.append({"role": "schedule", "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest})
    runner_events, runner_audit = _collect_jsonl_inputs(args.runner_events, "runner_events")
    bridge_events, bridge_audit = _collect_jsonl_inputs(args.bridge_events, "bridge_events")
    recorder_events, recorder_audit = _collect_jsonl_inputs(args.recorder_events, "recorder_events")
    recorder_root_events, recorder_root_audit = _collect_recorder_roots(args.recorder_roots)
    events = runner_events + bridge_events + recorder_events + recorder_root_events
    audit.extend(runner_audit + bridge_audit + recorder_audit + recorder_root_audit)
    audit.extend([
        {"role": "protocol", "path": str(args.protocol.resolve()), "bytes": args.protocol.stat().st_size, "sha256": sha256_path(args.protocol)},
        {"role": "analysis_spec", "path": str(args.analysis_spec.resolve()), "bytes": args.analysis_spec.stat().st_size, "sha256": sha256_path(args.analysis_spec)},
    ])
    result = analyze_event_bundle(protocol, spec, schedules, events, args.input_kind)
    write_outputs(args.output, result, audit)
    print(json.dumps({
        "schema": ANALYZER_SCHEMA,
        "mapping_completed": result["mapping_completed"],
        "output": str(args.output.resolve()),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisFailure as exc:
        print(json.dumps({"schema": ANALYZER_SCHEMA, "status": "rejected", "reason": exc.code, "detail": exc.detail}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
