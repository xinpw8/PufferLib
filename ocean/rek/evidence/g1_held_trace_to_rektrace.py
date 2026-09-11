#!/usr/bin/env python3
"""Convert a finalized G1 held-motion JSONL trace to binary REKTRACE.

The input trace contains client-observed visual transforms and client request
projections.  It contains no server tick, acknowledgement, or acceptance
signal.  The binary output therefore deliberately uses ``authority='unknown'``
and carries explicit authority limits in its header.

Frames are numbered by the JSONL ``trace_index`` rather than by the recorder's
absolute client fixed tick.  This makes independently captured repeats of the
same 50 Hz schedule directly align in ``differ.py`` without relabelling or
interpolating any measured value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import g1_held_trace_extract as held
import g1_schedule_pose_response as schedule
from trace import Trace, TraceWriter


OUTPUT_SCHEMA = "rek.g1_held_motion_rektrace.v1"
EXPECTED_SCHEDULE_SCHEMA = "rek.g1_held_input_schedule.v2"
EXPECTED_SCHEDULE_ID = "rek.private_bot1.g1_held_input.v2"
EXPECTED_SCHEDULE_SHA256 = (
    "0e28e089c73e603c7ce1d9cd5e6de4dd7f6f6017ce3bf4bcae2a90366b1adc2b"
)
EXPECTED_AUTHORITY_SCOPE = (
    "client_request_edges_local_returns_and_visual_only_client_diagnostics_only"
)
EXPECTED_AUTHORITY_CAVEAT = (
    "server acceptance and authoritative execution are unknown; physical response "
    "and kick timing require separately captured recorder bone-trajectory correlation"
)
EXPECTED_PAIRING_REASON = (
    "exact_g1_vs_g1_runtime_pairing_proven_semantic_ids_recorded_not_trusted"
)
EXPECTED_SCHEDULE_TICKS = 4551
FINAL_SCHEDULE_TICK = EXPECTED_SCHEDULE_TICKS - 1
FINAL_SCHEDULE_SUBSTEP = EXPECTED_SCHEDULE_TICKS * 10 - 1
UNITY_FIXED_DELTA_SECONDS = 1.0 / held.UNITY_FIXED_RATE_HZ
TRACE_DELTA_SECONDS = 1.0 / held.TRACE_RATE_HZ
CLOCK_RESIDUAL_TOLERANCE_SECONDS = 1e-6
QUATERNION_MIN_NORM = 1e-12
COMPONENTS_XYZ = "xyz"
COMPONENTS_XYZW = "xyzw"


class ConversionError(ValueError):
    """The supplied artifacts cannot support an honest binary trace."""


@dataclass(frozen=True)
class InventoryIdentity:
    sha256: str
    build_fingerprint: str
    buildid: str | None
    game_assembly_sha256: str
    global_metadata_sha256: str


@dataclass(frozen=True)
class JsonlTrace:
    path: str
    sha256: str
    source_raw_sha256: str
    start: dict[str, Any]
    end: dict[str, Any]
    samples: list[dict[str, Any]]
    fight_epoch: int
    round_number: int
    decoded_child_local_complete: bool
    decoded_child_local_missing: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleIdentity:
    path: str
    transcript_sha256: str
    schedule_id: str
    schedule_sha256: str
    run_id: str
    fresh_round_request_id: str
    round_identity_sha256: str
    bridge_plugin_version: str
    bridge_plugin_sha256: str
    local_slot: int
    opponent_slot: int
    ticks: list[dict[str, Any]]


@dataclass(frozen=True)
class ClockAlignment:
    phase_substeps: int
    schedule_start_client_fixed_tick: int
    first_schedule_tick: int
    last_schedule_tick: int
    maximum_residual_seconds: float


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ConversionError(reason)


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


def _same_number(observed: Any, expected: float, tolerance: float = 1e-12) -> bool:
    return _is_number(observed) and math.isclose(
        float(observed), expected, rel_tol=0.0, abs_tol=tolerance
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in pairs:
        if name in output:
            raise ConversionError(f"duplicate_json_key:{name}")
        output[name] = value
    return output


def _loads_json(text: str, context: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"{context}_invalid_json:{exc.msg}") from exc


def _validate_finite_tree(value: Any, context: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{context}_nonfinite")
    elif isinstance(value, dict):
        for name, child in value.items():
            _validate_finite_tree(child, f"{context}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, f"{context}[{index}]")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _partial_path(path: Path) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if (
            lowered.endswith(".partial")
            or ".partial-" in lowered
            or lowered.startswith("partial-")
        ):
            return True
    return False


def _final_input_path(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value)
    _require(not _partial_path(path), f"{label}_partial_path_rejected")
    _require(path.is_file(), f"{label}_not_a_file")
    return path


def _new_output_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    _require(str(path) != "-", "output_stdout_not_supported")
    _require(not _partial_path(path), "output_partial_path_rejected")
    _require(not os.path.lexists(path), f"output_exists:{path}")
    return path


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConversionError(f"{label}_line_{line_number}_not_utf8") from exc
        record = _loads_json(text, f"{label}_line_{line_number}")
        _require(isinstance(record, dict), f"{label}_line_{line_number}_not_object")
        _validate_finite_tree(record, f"{label}_line_{line_number}")
        records.append(record)
    _require(records, f"{label}_empty")
    return records, _sha256_bytes(data)


def _finite_vector(value: Any, length: int, context: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == length, f"{context}_shape")
    output: list[float] = []
    for component in value:
        _require(_is_number(component), f"{context}_nonfinite")
        output.append(float(component))
    return output


def _validate_quaternion(value: Iterable[float], context: str) -> None:
    components = tuple(float(component) for component in value)
    _require(len(components) == 4, f"{context}_shape")
    norm = math.sqrt(sum(component * component for component in components))
    _require(
        math.isfinite(norm) and norm > QUATERNION_MIN_NORM,
        f"{context}_not_normalizable",
    )


def _canonical_quaternion(value: Iterable[float]) -> tuple[float, float, float, float]:
    """Map q and -q to one component representation without renormalizing q."""
    components = tuple(float(component) for component in value)
    _validate_quaternion(components, "quaternion")
    sign = 1.0
    for index in (3, 2, 1, 0):
        if components[index] != 0.0:
            sign = -1.0 if components[index] < 0.0 else 1.0
            break
    canonical = tuple(component * sign for component in components)
    return tuple(0.0 if component == 0.0 else component for component in canonical)  # type: ignore[return-value]


def _validate_inventory(path: Path) -> InventoryIdentity:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConversionError("inventory_not_utf8") from exc
    inventory = _loads_json(text, "inventory")
    _require(isinstance(inventory, dict), "inventory_not_object")
    _validate_finite_tree(inventory, "inventory")
    _require(inventory.get("schema") == 1, "inventory_schema_mismatch")
    fingerprint = inventory.get("build_fingerprint")
    _require(_is_sha256(fingerprint), "inventory_build_fingerprint_invalid")
    _require(not inventory.get("errors"), "inventory_contains_errors")

    files = inventory.get("files")
    _require(isinstance(files, list), "inventory_files_missing")
    hashes: dict[str, str] = {}
    for index, record in enumerate(files):
        _require(isinstance(record, dict), f"inventory_file_{index}_not_object")
        name = str(record.get("path", "")).replace("\\", "/")
        digest = record.get("sha256")
        _require(name and name not in hashes, f"inventory_file_{index}_path_invalid")
        _require(_is_sha256(digest), f"inventory_file_{index}_sha256_invalid")
        hashes[name] = digest

    expected = {
        "GameAssembly.dll": held.EXPECTED_GAME_ASSEMBLY_SHA256,
        "REK_Data/il2cpp_data/Metadata/global-metadata.dat": held.EXPECTED_METADATA_SHA256,
    }
    for name, digest in expected.items():
        _require(hashes.get(name) == digest, f"inventory_expected_file_mismatch:{name}")

    steam = inventory.get("steam")
    _require(isinstance(steam, dict), "inventory_steam_missing")
    buildid = steam.get("buildid")
    _require(buildid is None or isinstance(buildid, str), "inventory_buildid_invalid")
    return InventoryIdentity(
        sha256=_sha256_bytes(data),
        build_fingerprint=str(fingerprint),
        buildid=buildid,
        game_assembly_sha256=hashes["GameAssembly.dll"],
        global_metadata_sha256=hashes[
            "REK_Data/il2cpp_data/Metadata/global-metadata.dat"
        ],
    )


def _validate_bone_layout(start: dict[str, Any]) -> None:
    layout = start.get("bone_layout")
    _require(isinstance(layout, dict), "trace_bone_layout_missing")
    _require(layout.get("id") == "g1_30", "trace_bone_layout_id_mismatch")
    _require(layout.get("count") == len(held.G1_BONE_NAMES), "trace_bone_count_mismatch")
    names = layout.get("ordered_names")
    _require(tuple(names or ()) == held.G1_BONE_NAMES, "trace_bone_names_mismatch")
    signature = layout.get("ordered_signature_sha256")
    _require(
        signature == held.G1_BONE_SIGNATURE_SHA256,
        "trace_bone_signature_mismatch",
    )
    computed = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    _require(computed == signature, "trace_bone_layout_hash_mismatch")

    decoded = start.get("decoded_bone_observation")
    _require(isinstance(decoded, dict), "trace_decoded_bone_observation_missing")
    _require(
        decoded.get("field") == "decoded_child_local_rotations_xyzw",
        "trace_decoded_bone_field_mismatch",
    )
    _require(decoded.get("stored_quaternion_count") == 30, "trace_stored_quaternion_count")
    _require(decoded.get("articulated_child_count") == 29, "trace_child_count_mismatch")
    _require(
        decoded.get("root_quaternion_source") == "decoded_root_world_rotation_xyzw",
        "trace_decoded_root_source_mismatch",
    )
    _require(
        decoded.get("reference_alignment_status")
        == "measured_joint_transform_required",
        "trace_reference_alignment_status_mismatch",
    )
    _require(
        decoded.get("direct_NPZ_angle_identity_allowed") is False,
        "trace_claims_direct_npz_angle_identity",
    )


def _validate_root(root: Any, context: str) -> None:
    _require(isinstance(root, dict), f"{context}_missing")
    _finite_vector(root.get("world_position_xyz"), 3, f"{context}_position")
    quaternion = _finite_vector(root.get("world_rotation_xyzw"), 4, f"{context}_rotation")
    _validate_quaternion(quaternion, f"{context}_rotation")


def _validate_bones(
    bones: Any,
    sample_tick: int,
    context: str,
) -> None:
    _require(isinstance(bones, dict), f"{context}_missing")
    _require(bones.get("layout") == "g1_30", f"{context}_layout_mismatch")
    source_tick = bones.get("source_client_fixed_tick")
    source_age = bones.get("source_age_ticks")
    _require(_is_int(source_tick), f"{context}_source_tick_invalid")
    _require(_is_int(source_age), f"{context}_source_age_invalid")
    _require(0 <= source_age == sample_tick - source_tick, f"{context}_source_age_mismatch")
    _require(
        _same_number(bones.get("source_age_seconds"), source_age / held.UNITY_FIXED_RATE_HZ),
        f"{context}_source_age_seconds_mismatch",
    )
    _require(
        isinstance(bones.get("fresh_since_previous_grid_sample"), bool),
        f"{context}_fresh_flag_invalid",
    )
    _require(_is_int(bones.get("raw_bone_packet_sequence")), f"{context}_raw_sequence_invalid")
    _require(_is_sha256(bones.get("wire_body_sha256")), f"{context}_wire_sha256_invalid")
    _finite_vector(bones.get("world_positions_xyz"), 90, f"{context}_world_positions")
    world_rotations = _finite_vector(
        bones.get("world_rotations_xyzw"), 120, f"{context}_world_rotations"
    )
    for bone_index in range(30):
        offset = bone_index * 4
        _validate_quaternion(
            world_rotations[offset : offset + 4],
            f"{context}_world_rotation_{bone_index}",
        )

    decoded_tick = bones.get("decoded_source_client_fixed_tick")
    decoded_age = bones.get("decoded_source_age_ticks")
    _require(_is_int(decoded_tick), f"{context}_decoded_tick_invalid")
    _require(_is_int(decoded_age), f"{context}_decoded_age_invalid")
    _require(decoded_tick == source_tick, f"{context}_decoded_raw_tick_mismatch")
    _require(0 <= decoded_age == sample_tick - decoded_tick, f"{context}_decoded_age_mismatch")
    _require(
        _same_number(
            bones.get("decoded_source_age_seconds"),
            decoded_age / held.UNITY_FIXED_RATE_HZ,
        ),
        f"{context}_decoded_age_seconds_mismatch",
    )
    _require(
        _is_int(bones.get("decoded_snapshot_sequence")),
        f"{context}_decoded_sequence_invalid",
    )
    _finite_vector(
        bones.get("decoded_root_world_position"), 3, f"{context}_decoded_root_position"
    )
    decoded_root = _finite_vector(
        bones.get("decoded_root_world_rotation_xyzw"),
        4,
        f"{context}_decoded_root_rotation",
    )
    _validate_quaternion(decoded_root, f"{context}_decoded_root_rotation")
    _require(
        bones.get("reference_alignment_status") == "measured_joint_transform_required",
        f"{context}_reference_alignment_mismatch",
    )

    child_local = bones.get("decoded_child_local_rotations_xyzw")
    if child_local is not None:
        rotations = _finite_vector(
            child_local,
            120,
            f"{context}_decoded_child_local_rotations",
        )
        # Index zero is the root placeholder.  The extractor explicitly names
        # the separate decoded root quaternion, so only indices 1..29 are
        # child-local rotation channels.
        for bone_index in range(1, 30):
            offset = bone_index * 4
            _validate_quaternion(
                rotations[offset : offset + 4],
                f"{context}_decoded_child_local_rotation_{bone_index}",
            )


def _validate_request_state(value: Any, sample_tick: int, context: str) -> None:
    _require(isinstance(value, dict), f"{context}_missing")
    source_tick = value.get("source_client_fixed_tick")
    age = value.get("source_age_ticks")
    sequence = value.get("request_sequence")
    _require(_is_int(source_tick), f"{context}_source_tick_invalid")
    _require(_is_int(age), f"{context}_source_age_invalid")
    _require(0 <= age == sample_tick - source_tick, f"{context}_source_age_mismatch")
    _require(
        _same_number(value.get("source_age_seconds"), age / held.UNITY_FIXED_RATE_HZ),
        f"{context}_source_age_seconds_mismatch",
    )
    _require(_is_int(sequence) and sequence > 0, f"{context}_sequence_invalid")
    _finite_vector(value.get("velocity_command_xyz"), 3, f"{context}_velocity")
    _require(value.get("request_only") is True, f"{context}_not_request_only")
    _require(value.get("server_acceptance") is None, f"{context}_claims_acceptance")
    _require(value.get("ack_observed") is False, f"{context}_claims_ack")


def _validate_move(value: Any, sample_tick: int, previous_tick: int, context: str) -> None:
    _require(isinstance(value, dict), f"{context}_not_object")
    source_tick = value.get("source_client_fixed_tick")
    age = value.get("source_age_ticks")
    sequence = value.get("request_sequence")
    move_index = value.get("move_index")
    _require(_is_int(source_tick), f"{context}_source_tick_invalid")
    _require(previous_tick < source_tick <= sample_tick, f"{context}_source_tick_outside_bin")
    _require(_is_int(age) and age == sample_tick - source_tick, f"{context}_source_age_mismatch")
    _require(_is_int(sequence) and sequence > 0, f"{context}_sequence_invalid")
    _require(move_index in held.G1_KICK_PROFILES, f"{context}_move_index_not_g1_kick")
    _require(
        value.get("move_profile") == held.G1_KICK_PROFILES[move_index],
        f"{context}_move_profile_mismatch",
    )
    _require(value.get("request_only") is True, f"{context}_not_request_only")
    _require(value.get("server_acceptance") is None, f"{context}_claims_acceptance")
    _require(value.get("ack_observed") is False, f"{context}_claims_ack")


def _validate_trace(path: Path) -> JsonlTrace:
    records, digest = _read_jsonl(path, "trace")
    start = records[0]
    end = records[-1]
    _require(start.get("event") == "trace_start", "trace_start_missing")
    _require(start.get("schema") == held.TRACE_SCHEMA, "trace_start_schema_mismatch")
    _require(end.get("event") == "trace_end", "trace_end_missing_or_not_last")
    _require(end.get("schema") == held.TRACE_SCHEMA, "trace_end_schema_mismatch")
    _require(end.get("complete") is True, "trace_end_not_complete")
    _require(
        start.get("source_recorder_schema") == held.RECORDER_SCHEMA,
        "trace_source_recorder_schema_mismatch",
    )
    source_raw_sha256 = start.get("source_raw_sha256")
    _require(
        _is_sha256(source_raw_sha256) and source_raw_sha256 != "0" * 64,
        "trace_source_raw_sha256_invalid",
    )
    _require(
        end.get("source_raw_sha256") == source_raw_sha256,
        "trace_source_raw_sha256_end_mismatch",
    )
    _require(
        start.get("authority")
        == "client_request_projections_plus_client_observed_network_bones_and_roots",
        "trace_authority_declaration_mismatch",
    )
    _require(start.get("server_acceptance_available") is False, "trace_claims_acceptance")
    _require(start.get("server_tick_available") is False, "trace_claims_server_tick")
    _require(start.get("root_tick_domain") == "client_fixed_update", "trace_tick_domain")
    _require(start.get("trace_rate_hz") == 50, "trace_rate_mismatch")
    _require(
        start.get("trace_grid_stride_client_fixed_ticks") == 10,
        "trace_grid_stride_mismatch",
    )
    _validate_bone_layout(start)
    maximum_bone_age = start.get("maximum_bone_source_age_ticks")
    _require(
        _is_int(maximum_bone_age) and maximum_bone_age >= 0,
        "trace_maximum_bone_age_invalid",
    )

    samples = records[1:-1]
    _require(samples, "trace_samples_missing")
    _require(end.get("sample_count") == len(samples), "trace_end_sample_count_mismatch")
    start_tick = start.get("start_client_fixed_tick")
    end_tick = start.get("end_client_fixed_tick")
    _require(_is_int(start_tick) and _is_int(end_tick), "trace_grid_bounds_invalid")
    _require(end_tick >= start_tick, "trace_grid_bounds_reversed")
    _require(
        end_tick - start_tick == (len(samples) - 1) * held.GRID_STRIDE_TICKS,
        "trace_grid_bounds_count_mismatch",
    )

    missing_child_local: list[str] = []
    for sample_index, sample in enumerate(samples):
        for slot in (0, 1):
            bones = sample.get(f"fighter_{slot}_bones")
            if not isinstance(bones, dict) or bones.get(
                "decoded_child_local_rotations_xyzw"
            ) is None:
                missing_child_local.append(f"sample_{sample_index}_fighter_{slot}")
    include_child_local = not missing_child_local

    fight_epoch: int | None = None
    round_number: int | None = None
    previous_root_index: int | None = None
    previous_bone_ticks: dict[int, int | None] = {0: None, 1: None}
    previous_bone_sequences: dict[int, int | None] = {0: None, 1: None}
    first_fixed_time: float | None = None
    for index, sample in enumerate(samples):
        context = f"trace_sample_{index}"
        _require(sample.get("event") == "trace_sample", f"{context}_event_mismatch")
        _require(sample.get("trace_index") == index, f"{context}_index_mismatch")
        expected_tick = start_tick + index * held.GRID_STRIDE_TICKS
        _require(sample.get("client_fixed_tick") == expected_tick, f"{context}_grid_tick_mismatch")
        _require(
            _same_number(
                sample.get("time_from_trace_start_seconds"),
                index / held.TRACE_RATE_HZ,
            ),
            f"{context}_relative_time_mismatch",
        )
        _require(sample.get("root_source_age_ticks") == 0, f"{context}_root_not_exact")
        root_index = sample.get("root_pose_sample_index")
        _require(_is_int(root_index), f"{context}_root_index_invalid")
        if previous_root_index is not None:
            _require(
                root_index == previous_root_index + held.GRID_STRIDE_TICKS,
                f"{context}_root_indices_not_50hz",
            )
        previous_root_index = root_index

        fixed_time = sample.get("unity_fixed_time")
        _require(_is_number(fixed_time), f"{context}_unity_fixed_time_invalid")
        if first_fixed_time is None:
            first_fixed_time = float(fixed_time)
        _require(
            math.isclose(
                float(fixed_time),
                first_fixed_time + index * TRACE_DELTA_SECONDS,
                rel_tol=0.0,
                abs_tol=CLOCK_RESIDUAL_TOLERANCE_SECONDS,
            ),
            f"{context}_unity_fixed_time_not_50hz",
        )

        current_epoch = sample.get("fight_epoch")
        current_round = sample.get("round_number")
        _require(_is_int(current_epoch), f"{context}_fight_epoch_invalid")
        _require(_is_int(current_round), f"{context}_round_number_invalid")
        if fight_epoch is None:
            fight_epoch = int(current_epoch)
            round_number = int(current_round)
        _require(current_epoch == fight_epoch, f"{context}_fight_epoch_changed")
        _require(current_round == round_number, f"{context}_round_number_changed")

        for slot in (0, 1):
            _validate_root(sample.get(f"fighter_{slot}_root"), f"{context}_fighter_{slot}_root")
            bones = sample.get(f"fighter_{slot}_bones")
            _validate_bones(
                bones,
                expected_tick,
                f"{context}_fighter_{slot}_bones",
            )
            assert isinstance(bones, dict)
            bone_tick = int(bones["source_client_fixed_tick"])
            bone_sequence = int(bones["raw_bone_packet_sequence"])
            _require(
                int(bones["source_age_ticks"]) <= maximum_bone_age,
                f"{context}_fighter_{slot}_bone_age_exceeds_header",
            )
            if previous_bone_ticks[slot] is not None:
                _require(
                    bone_tick >= previous_bone_ticks[slot],
                    f"{context}_fighter_{slot}_bone_tick_decreased",
                )
                _require(
                    bone_sequence >= previous_bone_sequences[slot],
                    f"{context}_fighter_{slot}_bone_sequence_decreased",
                )
            previous_bone_ticks[slot] = bone_tick
            previous_bone_sequences[slot] = bone_sequence

        request = sample.get("request_state")
        _validate_request_state(request, expected_tick, f"{context}_request_state")
        assert isinstance(request, dict)
        velocity = [float(component) for component in request["velocity_command_xyz"]]
        _require(
            sample.get("held_condition") == held.classify_velocity(velocity),
            f"{context}_held_condition_mismatch",
        )
        moves = sample.get("move_requests_since_previous_grid_sample")
        _require(isinstance(moves, list), f"{context}_move_requests_not_list")
        previous_tick = expected_tick - held.GRID_STRIDE_TICKS if index else expected_tick - 1
        for move_index, move in enumerate(moves):
            _validate_move(move, expected_tick, previous_tick, f"{context}_move_{move_index}")

    assert fight_epoch is not None and round_number is not None
    return JsonlTrace(
        path=str(path.resolve()),
        sha256=digest,
        source_raw_sha256=str(source_raw_sha256),
        start=start,
        end=end,
        samples=samples,
        fight_epoch=fight_epoch,
        round_number=round_number,
        decoded_child_local_complete=include_child_local,
        decoded_child_local_missing=tuple(missing_child_local),
    )


def _validate_measured_fighter(value: Any, slot: int, context: str) -> None:
    _require(isinstance(value, dict), f"{context}_missing")
    _require(value.get("slot") == slot, f"{context}_slot_mismatch")
    _require(value.get("bone_count") == 30, f"{context}_bone_count_mismatch")
    names = value.get("bone_names")
    _require(tuple(names or ()) == held.G1_BONE_NAMES, f"{context}_bone_names_mismatch")
    signature = value.get("runtime_bone_signature_sha256")
    _require(signature == held.G1_BONE_SIGNATURE_SHA256, f"{context}_signature_mismatch")
    _require(
        hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest() == signature,
        f"{context}_computed_signature_mismatch",
    )
    _require(value.get("exact_t800_bone_signature") is False, f"{context}_claims_t800")
    _require(value.get("exact_g1_bone_signature") is True, f"{context}_not_exact_g1")
    _require(
        value.get("semantic_robot_id_used_for_continuous_acceptance") is False,
        f"{context}_semantic_id_used_for_acceptance",
    )


def _validate_measured_pairing(value: Any) -> tuple[int, int]:
    _require(isinstance(value, dict), "schedule_measured_pairing_missing")
    required = {
        "required_pairing": "exact_homogeneous_supported_runtime_pair",
        "required_robot_id": None,
        "semantic_robot_id_required_for_acceptance": False,
        "required_g1_bone_count": 30,
        "required_g1_bone_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
        "exact_supported_runtime_pairing": True,
        "runtime_model": "g1",
        "exact_t800_vs_t800": False,
        "exact_g1_vs_g1": True,
        "reason": EXPECTED_PAIRING_REASON,
    }
    for name, expected in required.items():
        _require(value.get(name) == expected, f"schedule_pairing_{name}_mismatch")
    local_slot = value.get("local_slot")
    opponent_slot = value.get("opponent_slot")
    _require(local_slot in (0, 1), "schedule_pairing_local_slot_invalid")
    _require(opponent_slot == 1 - local_slot, "schedule_pairing_opponent_slot_invalid")
    _validate_measured_fighter(value.get("local_fighter"), local_slot, "schedule_local_fighter")
    _validate_measured_fighter(
        value.get("opponent_fighter"), opponent_slot, "schedule_opponent_fighter"
    )
    return int(local_slot), int(opponent_slot)


def _validate_schedule(path: Path, inventory: InventoryIdentity) -> ScheduleIdentity:
    records, digest = _read_jsonl(path, "schedule_transcript")
    try:
        parsed = schedule.read_transcript(path)
    except schedule.PoseResponseError as exc:
        raise ConversionError(f"schedule_transcript_invalid:{exc}") from exc
    _require(parsed.sha256 == digest, "schedule_transcript_changed_during_validation")
    _require(parsed.schema == EXPECTED_SCHEDULE_SCHEMA, "schedule_schema_mismatch")
    _require(parsed.schedule_id == EXPECTED_SCHEDULE_ID, "schedule_id_mismatch")
    _require(parsed.schedule_sha256 == EXPECTED_SCHEDULE_SHA256, "schedule_sha256_mismatch")
    _require(parsed.end.get("complete") is True, "schedule_end_not_complete")
    _require(parsed.end.get("partial_coverage") is False, "schedule_end_is_partial")
    _require(parsed.end.get("reason") == "complete", "schedule_end_reason_mismatch")
    _require(len(parsed.ticks) == EXPECTED_SCHEDULE_TICKS, "schedule_tick_count_mismatch")
    _require(
        parsed.end.get("client_fixed_substep") == FINAL_SCHEDULE_SUBSTEP,
        "schedule_end_substep_mismatch",
    )

    results = [record for record in records if record.get("event") == "client_result"]
    _require(len(results) == 1, "schedule_client_result_count_mismatch")
    _require(records[-1] is results[0], "schedule_client_result_not_last")
    result = results[0]
    _require(result.get("mode") == "g1-held", "schedule_client_result_mode_mismatch")
    _require(result.get("status") == "complete", "schedule_client_result_not_complete")
    _require(result.get("error") is None, "schedule_client_result_has_error")
    _require(result.get("lease_held") is False, "schedule_client_result_lease_held")

    starts = [
        record
        for record in records
        if record.get("event") == "ack"
        and record.get("command") == "StartG1HeldInputSchedule"
        and record.get("reason") == "g1_held_input_schedule_started"
    ]
    _require(len(starts) == 1, "schedule_start_ack_count_mismatch")
    start = starts[0]
    for name, expected in (
        ("protocol", "rek.ui_bridge.v1"),
        ("status", "accepted"),
        ("applied", True),
        ("client_request_issued", False),
        ("server_acceptance_observed", False),
        ("authoritative_execution_observed", False),
        ("g1_held_schedule_schema", EXPECTED_SCHEDULE_SCHEMA),
        ("g1_held_schedule_id", EXPECTED_SCHEDULE_ID),
        ("g1_held_schedule_sha256", EXPECTED_SCHEDULE_SHA256),
        ("g1_held_schedule_authority_scope", EXPECTED_AUTHORITY_SCOPE),
        ("g1_held_schedule_authority_caveat", EXPECTED_AUTHORITY_CAVEAT),
        ("g1_held_schedule_running", True),
        ("g1_held_schedule_tick", 0),
        ("g1_held_schedule_client_fixed_substep", 0),
        ("g1_held_schedule_round_capacity_proven", True),
    ):
        _require(start.get(name) == expected, f"schedule_start_{name}_mismatch")

    run_id = start.get("g1_held_schedule_run_id")
    fresh_round_request_id = start.get("g1_held_schedule_fresh_round_request_id")
    round_identity = start.get("g1_held_schedule_round_identity_sha256")
    _require(_is_hex(run_id, 32), "schedule_run_id_invalid")
    _require(
        isinstance(fresh_round_request_id, str) and fresh_round_request_id,
        "schedule_fresh_round_request_id_invalid",
    )
    _require(_is_sha256(round_identity), "schedule_round_identity_sha256_invalid")
    _require(parsed.run_id == run_id, "schedule_start_run_id_mismatch")
    local_slot, opponent_slot = _validate_measured_pairing(start.get("measured_pairing"))

    build = start.get("build")
    _require(isinstance(build, dict), "schedule_start_build_missing")
    _require(
        build.get("game_assembly_sha256") == inventory.game_assembly_sha256,
        "schedule_inventory_game_assembly_mismatch",
    )
    _require(
        build.get("global_metadata_sha256") == inventory.global_metadata_sha256,
        "schedule_inventory_metadata_mismatch",
    )
    bridge_plugin_version = build.get("plugin_version")
    bridge_plugin_sha256 = build.get("plugin_sha256")
    _require(
        isinstance(bridge_plugin_version, str) and bridge_plugin_version,
        "schedule_plugin_version_invalid",
    )
    _require(
        _is_sha256(bridge_plugin_sha256) and bridge_plugin_sha256 != "0" * 64,
        "schedule_plugin_sha256_invalid",
    )

    end = parsed.end
    _require(end.get("fresh_round_request_id") == fresh_round_request_id, "schedule_end_round_request_mismatch")
    _require(end.get("round_identity_sha256") == round_identity, "schedule_end_round_identity_mismatch")
    _require(end.get("g1_held_schedule_run_id") == run_id, "schedule_end_run_id_mismatch")

    schedule_end_index = next(
        index
        for index, record in enumerate(records)
        if record.get("event") == "g1_held_schedule_end"
    )
    _require(starts[0] in records[:schedule_end_index], "schedule_start_after_end")
    for record_index, record in enumerate(records[schedule_end_index + 1 :], schedule_end_index + 1):
        event = record.get("event")
        _require(
            not (isinstance(event, str) and event.startswith("g1_")),
            f"schedule_g1_event_after_end:{record_index}",
        )

    identity_fields = {
        "g1_held_schedule_schema": EXPECTED_SCHEDULE_SCHEMA,
        "g1_held_schedule_id": EXPECTED_SCHEDULE_ID,
        "g1_held_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    }
    for record_index, record in enumerate(records):
        for name, expected in identity_fields.items():
            if name in record:
                _require(
                    record.get(name) == expected,
                    f"schedule_record_{record_index}_{name}_mismatch",
                )
        if record.get("g1_held_schedule_run_id") is not None:
            _require(
                record.get("g1_held_schedule_run_id") == run_id,
                f"schedule_record_{record_index}_run_id_mismatch",
            )
        if record.get("fresh_round_request_id") is not None:
            _require(
                record.get("fresh_round_request_id") == fresh_round_request_id,
                f"schedule_record_{record_index}_round_request_mismatch",
            )
        if record.get("round_identity_sha256") is not None:
            _require(
                record.get("round_identity_sha256") == round_identity,
                f"schedule_record_{record_index}_round_identity_mismatch",
            )
        if record.get("g1_held_schedule_fresh_round_request_id") is not None:
            _require(
                record.get("g1_held_schedule_fresh_round_request_id")
                == fresh_round_request_id,
                f"schedule_record_{record_index}_ack_round_request_mismatch",
            )
        if record.get("g1_held_schedule_round_identity_sha256") is not None:
            _require(
                record.get("g1_held_schedule_round_identity_sha256") == round_identity,
                f"schedule_record_{record_index}_ack_round_identity_mismatch",
            )

    first_time = float(parsed.ticks[0]["unity_fixed_time"])
    for tick_index, record in enumerate(parsed.ticks):
        _require(record.get("schedule_tick") == tick_index, "schedule_ticks_not_contiguous")
        _require(
            record.get("client_fixed_substep") == tick_index * 10,
            f"schedule_tick_{tick_index}_substep_mismatch",
        )
        _require(
            math.isclose(
                float(record["unity_fixed_time"]),
                first_time + tick_index * TRACE_DELTA_SECONDS,
                rel_tol=0.0,
                abs_tol=CLOCK_RESIDUAL_TOLERANCE_SECONDS,
            ),
            f"schedule_tick_{tick_index}_fixed_time_not_50hz",
        )
        _require(
            record.get("authority_scope") == EXPECTED_AUTHORITY_SCOPE,
            f"schedule_tick_{tick_index}_authority_scope_mismatch",
        )
        _require(
            record.get("authority_caveat") == EXPECTED_AUTHORITY_CAVEAT,
            f"schedule_tick_{tick_index}_authority_caveat_mismatch",
        )
        detail = record.get("detail")
        _require(isinstance(detail, dict), f"schedule_tick_{tick_index}_detail_missing")
        _finite_vector(
            detail.get("effective_controller_vector_xyz"),
            3,
            f"schedule_tick_{tick_index}_effective_velocity",
        )
        _require(
            detail.get("velocity_property_write_returned") is True,
            f"schedule_tick_{tick_index}_velocity_write_failed",
        )
        _require(
            detail.get("velocity_readback_exact") is True,
            f"schedule_tick_{tick_index}_velocity_readback_failed",
        )

    return ScheduleIdentity(
        path=str(path.resolve()),
        transcript_sha256=digest,
        schedule_id=str(parsed.schedule_id),
        schedule_sha256=str(parsed.schedule_sha256),
        run_id=str(run_id),
        fresh_round_request_id=str(fresh_round_request_id),
        round_identity_sha256=str(round_identity),
        bridge_plugin_version=str(bridge_plugin_version),
        bridge_plugin_sha256=str(bridge_plugin_sha256),
        local_slot=local_slot,
        opponent_slot=opponent_slot,
        ticks=parsed.ticks,
    )


def _align_clocks(trace_input: JsonlTrace, schedule_input: ScheduleIdentity) -> ClockAlignment:
    schedule_zero_time = float(schedule_input.ticks[0]["unity_fixed_time"])
    relative_substeps: list[int] = []
    residuals: list[float] = []
    phases: list[int] = []
    schedule_start_ticks: list[int] = []
    schedule_ticks: list[int] = []

    for index, sample in enumerate(trace_input.samples):
        sample_time = float(sample["unity_fixed_time"])
        relative_float = (sample_time - schedule_zero_time) / UNITY_FIXED_DELTA_SECONDS
        relative = int(round(relative_float))
        residual = abs(sample_time - (schedule_zero_time + relative * UNITY_FIXED_DELTA_SECONDS))
        _require(
            residual <= CLOCK_RESIDUAL_TOLERANCE_SECONDS,
            f"trace_sample_{index}_schedule_clock_residual_exceeded",
        )
        _require(
            0 <= relative <= FINAL_SCHEDULE_SUBSTEP,
            f"trace_sample_{index}_outside_completed_schedule",
        )
        relative_substeps.append(relative)
        residuals.append(residual)
        phases.append(relative % held.GRID_STRIDE_TICKS)
        schedule_start_ticks.append(int(sample["client_fixed_tick"]) - relative)

    _require(len(set(phases)) == 1, "trace_schedule_fixed_substep_phase_changed")
    _require(len(set(schedule_start_ticks)) == 1, "trace_schedule_client_tick_offset_changed")
    phase = phases[0]
    for relative in relative_substeps:
        _require((relative - phase) % held.GRID_STRIDE_TICKS == 0, "trace_schedule_phase_invalid")
        schedule_ticks.append((relative - phase) // held.GRID_STRIDE_TICKS)
    _require(
        schedule_ticks == list(range(schedule_ticks[0], schedule_ticks[0] + len(schedule_ticks))),
        "trace_schedule_ticks_not_contiguous",
    )
    _require(schedule_ticks[-1] <= FINAL_SCHEDULE_TICK, "trace_schedule_tick_out_of_range")

    for index, (sample, schedule_tick) in enumerate(zip(trace_input.samples, schedule_ticks)):
        expected_time = (
            float(schedule_input.ticks[schedule_tick]["unity_fixed_time"])
            + phase * UNITY_FIXED_DELTA_SECONDS
        )
        _require(
            math.isclose(
                float(sample["unity_fixed_time"]),
                expected_time,
                rel_tol=0.0,
                abs_tol=CLOCK_RESIDUAL_TOLERANCE_SECONDS,
            ),
            f"trace_sample_{index}_schedule_tick_time_mismatch",
        )

    schedule_start_client_tick = schedule_start_ticks[0]
    for index, sample in enumerate(trace_input.samples):
        request_tick = int(sample["request_state"]["source_client_fixed_tick"])
        relative_request_tick = request_tick - schedule_start_client_tick
        _require(
            0 <= relative_request_tick <= FINAL_SCHEDULE_SUBSTEP,
            f"trace_sample_{index}_request_outside_schedule",
        )
        for move_index, move in enumerate(sample["move_requests_since_previous_grid_sample"]):
            relative_move_tick = int(move["source_client_fixed_tick"]) - schedule_start_client_tick
            _require(
                0 <= relative_move_tick <= FINAL_SCHEDULE_SUBSTEP,
                f"trace_sample_{index}_move_{move_index}_outside_schedule",
            )

    return ClockAlignment(
        phase_substeps=phase,
        schedule_start_client_fixed_tick=schedule_start_client_tick,
        first_schedule_tick=schedule_ticks[0],
        last_schedule_tick=schedule_ticks[-1],
        maximum_residual_seconds=max(residuals),
    )


def _joint_identity(index: int) -> str:
    return f"{index:02d}_{held.G1_BONE_NAMES[index]}"


def _channels(local_slot: int, include_child_local: bool) -> list[str]:
    channels: list[str] = []
    for slot in (0, 1):
        channels.extend(f"root.{slot}.pos.{axis}" for axis in COMPONENTS_XYZ)
        channels.extend(f"root.{slot}.quat.{axis}" for axis in COMPONENTS_XYZW)
    if include_child_local:
        for slot in (0, 1):
            for bone_index in range(1, len(held.G1_BONE_NAMES)):
                identity = _joint_identity(bone_index)
                channels.extend(
                    f"joint.{slot}.{identity}.local.quat.{axis}"
                    for axis in COMPONENTS_XYZW
                )
    channels.extend(f"cmd.{local_slot}.velocity.{axis}" for axis in COMPONENTS_XYZ)
    return channels


def _provenance(channels: list[str], local_slot: int) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for name in channels:
        pieces = name.split(".")
        if pieces[0] == "root":
            slot = int(pieces[1])
            kind = pieces[2]
            axis = pieces[3]
            raw_name, axes = (
                ("world_position_xyz", COMPONENTS_XYZ)
                if kind == "pos"
                else ("world_rotation_xyzw", COMPONENTS_XYZW)
            )
            raw_field = f"trace_sample.fighter_{slot}_root.{raw_name}[{axes.index(axis)}]"
            output[name] = {
                "kind": "class",
                "ref": (
                    f"RekEvidenceRecorder rek.private_ai.protocol.v7 {raw_field}; "
                    "REKApp.Robot RootTransform measured on the visual-only client robot"
                ),
                "raw_field": raw_field,
            }
        elif pieces[0] == "joint":
            slot = int(pieces[1])
            identity = pieces[2]
            bone_index = int(identity.split("_", 1)[0])
            axis = pieces[-1]
            offset = bone_index * 4 + COMPONENTS_XYZW.index(axis)
            raw_field = (
                f"trace_sample.fighter_{slot}_bones."
                f"decoded_child_local_rotations_xyzw[{offset}]"
            )
            output[name] = {
                "kind": "class",
                "ref": (
                    f"RekEvidenceRecorder decoded_bone_snapshot {raw_field}; "
                    "REKApp.Robot.BoneSnapshot child-local rotation decoded after "
                    "OnBoneMessageReceived"
                ),
                "raw_field": raw_field,
            }
        else:
            axis = pieces[-1]
            raw_field = (
                "trace_sample.request_state.velocity_command_xyz"
                f"[{COMPONENTS_XYZ.index(axis)}]"
            )
            output[name] = {
                "kind": "transport_message",
                "ref": (
                    "RekEvidenceRecorder outbound_request_projection REK_Input "
                    f"for local fighter slot {local_slot}, retained in {raw_field}"
                ),
                "raw_field": raw_field,
            }
    return output


def _frames(trace_input: JsonlTrace, local_slot: int) -> list[dict[str, float]]:
    frames: list[dict[str, float]] = []
    for sample in trace_input.samples:
        frame: dict[str, float] = {}
        for slot in (0, 1):
            root = sample[f"fighter_{slot}_root"]
            for axis, value in zip(COMPONENTS_XYZ, root["world_position_xyz"]):
                frame[f"root.{slot}.pos.{axis}"] = float(value)
            root_quaternion = _canonical_quaternion(root["world_rotation_xyzw"])
            for axis, value in zip(COMPONENTS_XYZW, root_quaternion):
                frame[f"root.{slot}.quat.{axis}"] = value

        if trace_input.decoded_child_local_complete:
            for slot in (0, 1):
                rotations = sample[f"fighter_{slot}_bones"][
                    "decoded_child_local_rotations_xyzw"
                ]
                for bone_index in range(1, len(held.G1_BONE_NAMES)):
                    offset = bone_index * 4
                    quaternion = _canonical_quaternion(rotations[offset : offset + 4])
                    identity = _joint_identity(bone_index)
                    for axis, value in zip(COMPONENTS_XYZW, quaternion):
                        frame[f"joint.{slot}.{identity}.local.quat.{axis}"] = value

        velocity = sample["request_state"]["velocity_command_xyz"]
        for axis, value in zip(COMPONENTS_XYZ, velocity):
            frame[f"cmd.{local_slot}.velocity.{axis}"] = float(value)
        _require(all(math.isfinite(value) for value in frame.values()), "output_frame_nonfinite")
        frames.append(frame)
    return frames


def _request_events(trace_input: JsonlTrace) -> list[dict[str, Any]]:
    by_sequence: dict[int, dict[str, Any]] = {}

    def remember(sequence: int, record: dict[str, Any]) -> None:
        previous = by_sequence.get(sequence)
        if previous is None:
            by_sequence[sequence] = record
            return
        comparable = {
            name: value for name, value in record.items() if name != "observed_trace_index"
        }
        previous_comparable = {
            name: value for name, value in previous.items() if name != "observed_trace_index"
        }
        _require(previous_comparable == comparable, f"request_sequence_{sequence}_conflict")
        if record["observed_trace_index"] < previous["observed_trace_index"]:
            previous["observed_trace_index"] = record["observed_trace_index"]

    for trace_index, sample in enumerate(trace_input.samples):
        request = sample["request_state"]
        sequence = int(request["request_sequence"])
        remember(
            sequence,
            {
                "observed_trace_index": trace_index,
                "kind": "outbound_request_projection",
                "message": "REK_Input",
                "request_sequence": sequence,
                "source_client_fixed_tick": int(request["source_client_fixed_tick"]),
                "velocity_command_xyz": [
                    float(value) for value in request["velocity_command_xyz"]
                ],
                "request_only": True,
                "server_acceptance": None,
                "ack_observed": False,
            },
        )
        for move in sample["move_requests_since_previous_grid_sample"]:
            move_sequence = int(move["request_sequence"])
            remember(
                move_sequence,
                {
                    "observed_trace_index": trace_index,
                    "kind": "outbound_request_projection",
                    "message": "REK_Move",
                    "request_sequence": move_sequence,
                    "source_client_fixed_tick": int(move["source_client_fixed_tick"]),
                    "move_index": int(move["move_index"]),
                    "move_profile": move["move_profile"],
                    "request_only": True,
                    "server_acceptance": None,
                    "ack_observed": False,
                },
            )
    return sorted(
        by_sequence.values(),
        key=lambda record: (record["observed_trace_index"], record["request_sequence"]),
    )


def _publish_atomic_no_overwrite(temporary: Path, output: Path) -> None:
    # A same-directory hard link atomically creates the destination and fails if
    # it already exists.  Unlike os.replace, it cannot overwrite a path won by a
    # concurrent producer between validation and publication.
    os.link(temporary, output)
    temporary.unlink()


def _verify_binary(
    path: Path,
    channels: list[str],
    frames: list[dict[str, float]],
    events: list[dict[str, Any]],
) -> None:
    loaded = Trace.load(path)
    _require(loaded.source == "rek", "binary_verify_source_mismatch")
    _require(loaded.authority == "unknown", "binary_verify_authority_mismatch")
    _require(loaded.header.get("channels") == channels, "binary_verify_channels_mismatch")
    _require(loaded.ticks == list(range(len(frames))), "binary_verify_ticks_mismatch")
    _require(len(loaded) == len(frames), "binary_verify_frame_count_mismatch")
    for channel in channels:
        expected = [frame[channel] for frame in frames]
        observed = loaded.channels.get(channel)
        _require(observed == expected, f"binary_verify_channel_values_mismatch:{channel}")
        _require(
            all(math.isfinite(value) for value in observed),
            f"binary_verify_channel_nonfinite:{channel}",
        )
    expected_events = []
    for event in events:
        expected = dict(event)
        expected["tick"] = expected.pop("observed_trace_index")
        expected_events.append(expected)
    _require(loaded.events == expected_events, "binary_verify_events_mismatch")


def convert(
    trace_jsonl: str | os.PathLike[str],
    schedule_transcript: str | os.PathLike[str],
    inventory_json: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    trace_path = _final_input_path(trace_jsonl, "trace")
    transcript_path = _final_input_path(schedule_transcript, "schedule_transcript")
    inventory_path = _final_input_path(inventory_json, "inventory")
    output = _new_output_path(output_path)

    resolved_inputs = {
        os.path.normcase(str(path.resolve()))
        for path in (trace_path, transcript_path, inventory_path)
    }
    _require(len(resolved_inputs) == 3, "input_paths_not_distinct")
    _require(
        os.path.normcase(str(output.resolve())) not in resolved_inputs,
        "output_path_is_input",
    )

    inventory = _validate_inventory(inventory_path)
    trace_input = _validate_trace(trace_path)
    schedule_input = _validate_schedule(transcript_path, inventory)
    alignment = _align_clocks(trace_input, schedule_input)

    channels = _channels(
        schedule_input.local_slot, trace_input.decoded_child_local_complete
    )
    provenance = _provenance(channels, schedule_input.local_slot)
    frames = _frames(trace_input, schedule_input.local_slot)
    events = _request_events(trace_input)
    _require(
        all(set(frame) == set(channels) for frame in frames),
        "output_channel_family_not_uniform",
    )

    optional_family = {
        "name": "decoded_child_local_quaternions",
        "status": "included" if trace_input.decoded_child_local_complete else "omitted",
        "channel_count": 232 if trace_input.decoded_child_local_complete else 0,
        "required_source_field": "fighter_[01]_bones.decoded_child_local_rotations_xyzw",
        "omission_reason": (
            None
            if trace_input.decoded_child_local_complete
            else "source_field_missing_or_null_in_at_least_one_sample; whole_family_omitted"
        ),
        "missing_locations": list(trace_input.decoded_child_local_missing),
    }
    fighter_pairing = {
        "runtime_model": "g1",
        "exact_g1_vs_g1": True,
        "local_fighter_index": schedule_input.local_slot,
        "opponent_fighter_index": schedule_input.opponent_slot,
        "bone_layout": {
            "id": "g1_30",
            "count": 30,
            "ordered_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
        },
    }
    authority_limits = {
        "observation": "client_observed_visual_transforms_and_client_request_projections",
        "server_tick_available": False,
        "server_acceptance_available": False,
        "server_acknowledgement_available": False,
        "authoritative_execution_observed": False,
        "request_events_are_acceptance_events": False,
        "round_identity_cross_artifact_limit": (
            "schedule transcript round identity is internally validated; the extracted "
            "motion JSONL does not carry that hash, so cross-artifact identity rests on "
            "the validated shared fixed clock, pinned build, and exact G1-vs-G1 layout"
        ),
        "parity_claim_supported": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    _require(not os.path.lexists(temporary), "temporary_output_exists")
    try:
        with TraceWriter(
            temporary,
            channels,
            inventory.build_fingerprint,
            "rek",
            authority="unknown",
            provenance=provenance,
            artifact_schema=OUTPUT_SCHEMA,
            source_trace_schema=held.TRACE_SCHEMA,
            source_trace_path=trace_input.path,
            source_trace_sha256=trace_input.sha256,
            source_raw_sha256=trace_input.source_raw_sha256,
            source_raw_hash_validation=(
                "validated_equal_nonzero_sha256_declarations_in_trace_start_and_trace_end; "
                "raw_recorder_capture_not_supplied_to_this_converter"
            ),
            schedule_transcript_path=schedule_input.path,
            schedule_transcript_sha256=schedule_input.transcript_sha256,
            schedule_bridge_plugin_version=schedule_input.bridge_plugin_version,
            schedule_bridge_plugin_sha256=schedule_input.bridge_plugin_sha256,
            inventory_path=str(inventory_path.resolve()),
            inventory_sha256=inventory.sha256,
            client_buildid=inventory.buildid,
            tick_domain="g1_held_trace_index_50hz",
            tick_rate_hz=held.TRACE_RATE_HZ,
            tick_normalization=(
                "binary frame tick equals source trace_index; absolute client_fixed_tick "
                "is retained only in source-grid metadata and event provenance"
            ),
            source_start_client_fixed_tick=int(
                trace_input.start["start_client_fixed_tick"]
            ),
            source_end_client_fixed_tick=int(trace_input.start["end_client_fixed_tick"]),
            source_fight_epoch=trace_input.fight_epoch,
            source_round_number=trace_input.round_number,
            command_sequence_sha256=schedule_input.schedule_sha256,
            command_sequence_schema=EXPECTED_SCHEDULE_SCHEMA,
            command_sample_phase_substeps=alignment.phase_substeps,
            schedule_id=schedule_input.schedule_id,
            schedule_manifest_sha256=schedule_input.schedule_sha256,
            schedule_run_id=schedule_input.run_id,
            schedule_fresh_round_request_id=schedule_input.fresh_round_request_id,
            schedule_round_identity_sha256=schedule_input.round_identity_sha256,
            schedule_start_client_fixed_tick=alignment.schedule_start_client_fixed_tick,
            observed_schedule_tick_start=alignment.first_schedule_tick,
            observed_schedule_tick_end=alignment.last_schedule_tick,
            fixed_clock_alignment={
                "basis": "shared_UnityEngine.Time.fixedTimeAsDouble",
                "status": "correlated",
                "observed_phase_substeps": alignment.phase_substeps,
                "maximum_residual_seconds": alignment.maximum_residual_seconds,
                "configured_maximum_residual_seconds": CLOCK_RESIDUAL_TOLERANCE_SECONDS,
            },
            complete_schedule=True,
            complete_round=False,
            fighter_pairing=fighter_pairing,
            bone_layout=trace_input.start["bone_layout"],
            optional_channel_families={
                "decoded_child_local_quaternions": optional_family
            },
            request_event_deduplication={
                "key": "request_sequence",
                "event_count": len(events),
                "event_tick": "first source trace_index carrying the measured request",
            },
            authority_limits=authority_limits,
        ) as writer:
            for trace_index, frame in enumerate(frames):
                writer.append(trace_index, frame)
            for event in events:
                payload = dict(event)
                event_tick = int(payload.pop("observed_trace_index"))
                event_kind = str(payload.pop("kind"))
                writer.event(event_tick, event_kind, **payload)
        _verify_binary(temporary, channels, frames, events)
        _publish_atomic_no_overwrite(temporary, output)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()

    return {
        "schema": OUTPUT_SCHEMA,
        "output_path": str(output.resolve()),
        "output_sha256": _sha256_path(output),
        "frame_count": len(frames),
        "channel_count": len(channels),
        "request_event_count": len(events),
        "command_sample_phase_substeps": alignment.phase_substeps,
        "decoded_child_local_quaternions": optional_family["status"],
        "authority": "unknown",
        "parity_claimed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--schedule-transcript", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = convert(
            arguments.trace_jsonl,
            arguments.schedule_transcript,
            arguments.inventory,
            arguments.out,
        )
    except (ConversionError, FileExistsError, OSError) as exc:
        print(f"G1 held trace conversion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
