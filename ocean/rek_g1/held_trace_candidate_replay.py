"""Replay a measured REK G1 held-input timeline through the Spark candidate.

The extractor correlates the exact 50 Hz trace with the schedule analysis and
emits a compact, portable contract. The replayer consumes that contract through
the native Puffer extension. The result is a diagnostic comparison only. The
source capture has no server action acknowledgements, no opponent action
timeline, no repeated-run variance envelope, and no proof that the public
policy candidate is the current REK server policy.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


TRACE_SCHEMA = "rek.g1_held_motion_trace.v1"
POSE_SCHEMA = "rek.g1_schedule_pose_response.v1"
COVERAGE_SCHEMA = "rek.g1_held_motion_trace.coverage.v1"
REAL_CONTRACT_SCHEMA = "rek.g1_held_candidate_replay_input.v1"
REPORT_SCHEMA = "rek.g1_held_candidate_replay_report.v1"
TRACE_RATE_HZ = 50
ROBOT_ROWS = 8
OBSERVATION_FLOATS = 223
ACTION_HEADS = 1
ACTION_CATEGORIES = 20
KICK_DURATION_TICKS = (157, 145, 158, 139)

HELD_CATEGORY_BY_LABEL = {
    "neutral": 1,
    "W": 2,
    "S": 3,
    "A": 4,
    "D": 5,
    "Q": 6,
    "E": 7,
    "W+Q": 8,
    "W+E": 9,
    "S+Q": 10,
    "S+E": 11,
    "A+Q": 12,
    "A+E": 13,
    "D+Q": 14,
    "D+E": 15,
}
HELD_LABELS = tuple(label for label in HELD_CATEGORY_BY_LABEL if label != "neutral")
EXPECTED_RUN_LABELS = tuple(
    item
    for pair in ((label, "neutral") for label in HELD_LABELS)
    for item in pair
)


class ReplayFailure(RuntimeError):
    """An input or runtime violated the diagnostic replay contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayFailure(message)


def _regular_file(path: Path, description: str) -> Path:
    _require(not path.is_symlink(), f"{description} must not be a symlink")
    resolved = path.resolve()
    _require(resolved.is_file(), f"{description} is not a regular file")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category_sequence_sha256(categories: Iterable[int]) -> str:
    values = bytes(categories)
    return hashlib.sha256(values).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayFailure(f"failed to read {description}: {error}") from error
    _require(isinstance(value, dict), f"{description} root must be an object")
    return value


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(isinstance(value, dict), f"trace line {line_number} is not an object")
                yield line_number, value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayFailure(f"failed to parse trace: {error}") from error


def _finite_vector(value: Any, length: int, description: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == length, f"{description} shape")
    output = [float(item) for item in value]
    _require(all(math.isfinite(item) for item in output), f"{description} nonfinite")
    return output


def _normalized_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    _require(len(values) == 4, "quaternion shape")
    converted = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in converted))
    _require(math.isfinite(norm) and norm > 1e-9, "quaternion is not normalizable")
    return tuple(value / norm for value in converted)  # type: ignore[return-value]


def _unity_yaw_xyzw(values: Sequence[float]) -> float:
    x, y, z, w = _normalized_quaternion(values)
    forward_x = 2.0 * (x * z + w * y)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(forward_x, forward_z)


def _mujoco_yaw_wxyz(values: Sequence[float]) -> float:
    w, x, y, z = _normalized_quaternion(values)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _unwrap(values: Iterable[float]) -> list[float]:
    output: list[float] = []
    for value in values:
        candidate = float(value)
        if output:
            while candidate - output[-1] > math.pi:
                candidate -= 2.0 * math.pi
            while candidate - output[-1] < -math.pi:
                candidate += 2.0 * math.pi
        output.append(candidate)
    return output


def _wrapped_abs_error(left: float, right: float) -> float:
    difference = left - right
    return abs((difference + math.pi) % (2.0 * math.pi) - math.pi)


def _runs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _require(samples, "trace has no samples")
    output: list[dict[str, Any]] = []
    for sample_offset, sample in enumerate(samples):
        label = sample["held_condition"]
        if not output or output[-1]["label"] != label:
            output.append(
                {
                    "label": label,
                    "sample_start": sample_offset,
                    "sample_stop_exclusive": sample_offset + 1,
                }
            )
        else:
            output[-1]["sample_stop_exclusive"] = sample_offset + 1
    return output


def _select_canonical_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pattern = ("neutral",) + EXPECTED_RUN_LABELS
    for start in range(len(runs) - len(pattern) + 1):
        labels = tuple(run["label"] for run in runs[start : start + len(pattern)])
        if labels == pattern:
            return runs[start : start + len(pattern)]
    raise ReplayFailure("trace does not contain the ordered 14-condition held schedule")


def _pose_segments(pose: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = pose.get("held_input_trajectories")
    _require(isinstance(values, list), "pose held_input_trajectories missing")
    output: dict[str, dict[str, Any]] = {}
    for ordinal, expected_label in enumerate(HELD_LABELS):
        _require(ordinal < len(values), "pose held trajectory count is incomplete")
        value = values[ordinal]
        _require(isinstance(value, dict), "pose held trajectory is not an object")
        _require(value.get("ordinal") == ordinal, "pose held trajectory ordinal mismatch")
        _require(value.get("label") == expected_label, "pose held trajectory order mismatch")
        output[expected_label] = value
    return output


def _trace_samples(
    trace_path: Path,
    local_slot: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    root_key = f"fighter_{local_slot}_root"
    expected_trace_index = 0
    for line_number, value in _jsonl(trace_path):
        event = value.get("event")
        if header is None:
            _require(event == "trace_start", "trace first record is not trace_start")
            _require(value.get("schema") == TRACE_SCHEMA, "trace schema mismatch")
            header = value
            continue
        if event != "trace_sample":
            continue
        _require(value.get("trace_index") == expected_trace_index, "trace index is not contiguous")
        expected_trace_index += 1
        label = value.get("held_condition")
        _require(label in HELD_CATEGORY_BY_LABEL, f"unsupported held condition at line {line_number}")
        root = value.get(root_key)
        _require(isinstance(root, dict), f"missing {root_key} at line {line_number}")
        position = _finite_vector(root.get("world_position_xyz"), 3, "real root position")
        quaternion = _finite_vector(root.get("world_rotation_xyzw"), 4, "real root rotation")
        client_tick = value.get("client_fixed_tick")
        trace_time = value.get("time_from_trace_start_seconds")
        _require(isinstance(client_tick, int), "trace client_fixed_tick is not an integer")
        _require(isinstance(trace_time, (int, float)) and math.isfinite(trace_time), "trace time invalid")
        samples.append(
            {
                "trace_index": value["trace_index"],
                "client_fixed_tick": client_tick,
                "time_from_trace_start_seconds": float(trace_time),
                "held_condition": label,
                "position_xyz": position,
                "yaw_radians": _unity_yaw_xyzw(quaternion),
            }
        )
    _require(header is not None, "trace header missing")
    _require(header.get("trace_rate_hz") == TRACE_RATE_HZ, "trace rate is not 50 Hz")
    return header, samples


def extract_real_contract(
    trace_path: Path,
    pose_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    trace_path = _regular_file(trace_path, "real trace")
    pose_path = _regular_file(pose_path, "pose response")
    coverage_path = _regular_file(coverage_path, "trace coverage")
    pose = _load_json(pose_path, "pose response")
    coverage = _load_json(coverage_path, "trace coverage")
    _require(pose.get("schema") == POSE_SCHEMA, "pose response schema mismatch")
    _require(coverage.get("schema") == COVERAGE_SCHEMA, "trace coverage schema mismatch")

    trace_sha256 = _sha256(trace_path)
    coverage_trace = coverage.get("trace_artifact")
    _require(isinstance(coverage_trace, dict), "coverage trace_artifact missing")
    _require(coverage_trace.get("sha256") == trace_sha256, "coverage trace hash mismatch")
    _require(coverage_trace.get("schema") == TRACE_SCHEMA, "coverage trace schema mismatch")

    pose_scope = pose.get("scope")
    coverage_scope = coverage.get("scope")
    _require(isinstance(pose_scope, dict), "pose scope missing")
    _require(isinstance(coverage_scope, dict), "coverage scope missing")
    local_slot = pose_scope.get("local_fighter_index")
    _require(local_slot in (0, 1), "pose local fighter slot invalid")
    _require(coverage_scope.get("local_fighter_index") == local_slot, "scope fighter slot mismatch")
    _require(pose_scope.get("exact_g1_vs_g1") is True, "pose is not exact G1 versus G1")
    _require(coverage_scope.get("exact_g1_vs_g1") is True, "coverage is not exact G1 versus G1")

    header, all_samples = _trace_samples(trace_path, int(local_slot))
    sources = pose.get("sources")
    _require(isinstance(sources, dict), "pose sources missing")
    recorder = sources.get("recorder")
    transcript = sources.get("transcript")
    _require(isinstance(recorder, dict), "pose recorder source missing")
    _require(isinstance(transcript, dict), "pose transcript source missing")
    _require(
        header.get("source_raw_sha256") == recorder.get("sha256"),
        "trace raw source does not match pose recorder source",
    )

    pose_by_label = _pose_segments(pose)
    selected_runs = _select_canonical_runs(_runs(all_samples))
    timeline_start = int(selected_runs[0]["sample_start"])
    timeline_stop = int(selected_runs[-1]["sample_stop_exclusive"])
    timeline_samples = all_samples[timeline_start:timeline_stop]
    stride = header.get("trace_grid_stride_client_fixed_ticks")
    _require(isinstance(stride, int) and stride > 0, "trace grid stride invalid")
    _require(
        all(
            right["client_fixed_tick"] - left["client_fixed_tick"] == stride
            for left, right in zip(timeline_samples, timeline_samples[1:])
        ),
        "selected trace timeline is not uniform",
    )
    categories = [HELD_CATEGORY_BY_LABEL[sample["held_condition"]] for sample in timeline_samples]

    compact_runs: list[dict[str, Any]] = []
    real_segments: list[dict[str, Any]] = []
    for run in selected_runs:
        absolute_start = int(run["sample_start"])
        absolute_stop = int(run["sample_stop_exclusive"])
        relative_start = absolute_start - timeline_start
        relative_stop = absolute_stop - timeline_start
        label = str(run["label"])
        run_samples = all_samples[absolute_start:absolute_stop]
        compact_runs.append(
            {
                "label": label,
                "category": HELD_CATEGORY_BY_LABEL[label],
                "tick_start_inclusive": relative_start,
                "tick_stop_exclusive": relative_stop,
                "sample_count": relative_stop - relative_start,
                "real_client_fixed_tick_start": run_samples[0]["client_fixed_tick"],
                "real_client_fixed_tick_stop_exclusive": (
                    run_samples[-1]["client_fixed_tick"] + stride
                ),
            }
        )
        if label == "neutral":
            continue
        pose_segment = pose_by_label[label]
        nominal_start = pose_segment.get("recorder_tick_start")
        nominal_stop = pose_segment.get("recorder_tick_stop")
        _require(isinstance(nominal_start, int) and isinstance(nominal_stop, int), "pose recorder interval invalid")
        observed_start = run_samples[0]["client_fixed_tick"]
        observed_stop = run_samples[-1]["client_fixed_tick"] + stride
        _require(observed_start <= nominal_stop and observed_stop >= nominal_start, "trace run does not overlap pose schedule interval")
        real_segments.append(
            {
                "ordinal": len(real_segments),
                "label": label,
                "category": HELD_CATEGORY_BY_LABEL[label],
                "tick_start_inclusive": relative_start,
                "tick_stop_exclusive": relative_stop,
                "sample_count": len(run_samples),
                "nominal_schedule": {
                    "schedule_tick_start_inclusive": pose_segment.get("schedule_tick_start_inclusive"),
                    "schedule_tick_stop_exclusive": pose_segment.get("schedule_tick_stop_exclusive"),
                    "recorder_tick_start": nominal_start,
                    "recorder_tick_stop": nominal_stop,
                    "observed_run_start_minus_nominal_seconds": (observed_start - nominal_start) / 500.0,
                    "observed_run_stop_minus_nominal_seconds": (observed_stop - nominal_stop) / 500.0,
                },
                "samples": [
                    {
                        "tick": sample["trace_index"] - timeline_start,
                        "real_client_fixed_tick": sample["client_fixed_tick"],
                        "real_time_seconds": (
                            sample["time_from_trace_start_seconds"]
                            - timeline_samples[0]["time_from_trace_start_seconds"]
                        ),
                        "world_position_xyz_m": sample["position_xyz"],
                        "world_yaw_radians": sample["yaw_radians"],
                    }
                    for sample in run_samples
                ],
            }
        )

    analysis_gate = pose.get("analysis_gate")
    request_authority = coverage.get("request_authority")
    _require(isinstance(analysis_gate, dict), "pose analysis gate missing")
    _require(isinstance(request_authority, dict), "coverage request authority missing")
    return {
        "schema": REAL_CONTRACT_SCHEMA,
        "classification": "single_continuous_contact_confounded_real_trace",
        "rek_parity_claim": False,
        "trace_rate_hz": TRACE_RATE_HZ,
        "coordinate_contract": {
            "real": "Unity world XYZ, Y up, root yaw from local +Z projected into world XZ",
            "candidate": "MuJoCo world XYZ, Z up, root yaw about world +Z",
            "comparison": (
                "per-segment initial-heading frame: candidate +X maps to REK forward +Z, "
                "candidate -Y maps to REK right +X, and candidate yaw sign is negated"
            ),
        },
        "action_contract": {
            "category_by_label": HELD_CATEGORY_BY_LABEL,
            "timeline_tick_start": 0,
            "timeline_tick_stop_exclusive": len(timeline_samples),
            "category_sequence_sha256": _category_sequence_sha256(categories),
            "trace_grid_uniform": True,
            "runs": compact_runs,
        },
        "real_segments": real_segments,
        "provenance": {
            "trace": {"path": str(trace_path), "sha256": trace_sha256, "header": header},
            "pose_response": {"path": str(pose_path), "sha256": _sha256(pose_path)},
            "coverage": {"path": str(coverage_path), "sha256": _sha256(coverage_path)},
            "raw_recorder_sha256": recorder.get("sha256"),
            "schedule_transcript_sha256": transcript.get("sha256"),
            "schedule_definition_sha256": transcript.get("schedule_sha256"),
            "schedule_id": transcript.get("schedule_id"),
            "schedule_run_id": transcript.get("schedule_run_id"),
            "scope": coverage_scope,
        },
        "gates": {
            "held_root_trajectory_measurement_supported": analysis_gate.get(
                "suitable_for_held_root_trajectory_measurement"
            )
            is True,
            "server_action_acceptance_available": request_authority.get(
                "server_acceptance_available"
            )
            is True,
            "opponent_action_timeline_available": False,
            "matching_initial_physical_state_available": False,
            "repeated_run_variance_available": False,
            "current_server_policy_identity_proven": False,
            "same_execution_runtime": False,
            "parity_acceptance_evaluable": False,
        },
        "limitations": [
            "The trace is one continuous fight, so prior motion, Bot motion, contact, and falls can affect every segment.",
            "Held state and move projections are request-side observations without server acknowledgements.",
            "The Bot 1 action sequence was not observed and is replaced by neutral actions in candidate replay.",
            "The real trace begins from an in-progress arena state while candidate replay begins from its deterministic reset.",
            "The candidate uses the public GEAR-SONIC family and MuJoCo on Spark, not the identified current REK server runtime.",
            "No repeated-run REK variance envelope is available, so these errors cannot accept or reject parity.",
        ],
    }


def _expand_categories(action_contract: dict[str, Any]) -> list[int]:
    stop = action_contract.get("timeline_tick_stop_exclusive")
    _require(isinstance(stop, int) and stop > 0, "action timeline stop invalid")
    categories = [0] * stop
    cursor = 0
    runs = action_contract.get("runs")
    _require(isinstance(runs, list), "action runs missing")
    for run in runs:
        _require(isinstance(run, dict), "action run is not an object")
        start = run.get("tick_start_inclusive")
        end = run.get("tick_stop_exclusive")
        category = run.get("category")
        label = run.get("label")
        _require(start == cursor and isinstance(end, int) and end > start, "action runs are not contiguous")
        _require(HELD_CATEGORY_BY_LABEL.get(label) == category, "action run category mismatch")
        categories[start:end] = [category] * (end - start)
        cursor = end
    _require(cursor == stop and all(category > 0 for category in categories), "action timeline is incomplete")
    expected_hash = action_contract.get("category_sequence_sha256")
    _require(_category_sequence_sha256(categories) == expected_hash, "action timeline hash mismatch")
    return categories


def _load_extension(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("_C", path)
    _require(specification is not None and specification.loader is not None, "extension specification failed")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _float_view(pointer: int, rows: int, columns: int) -> np.ndarray:
    _require(pointer != 0, "native vector exposed a null buffer")
    storage = (ctypes.c_float * (rows * columns)).from_address(pointer)
    return np.ctypeslib.as_array(storage).reshape(rows, columns)


def _candidate_samples(
    contract: dict[str, Any],
    extension_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    native = _load_extension(extension_path)
    categories = _expand_categories(contract["action_contract"])
    arguments = {
        "vec": {"total_agents": ROBOT_ROWS, "num_buffers": 1},
        "env": {
            "max_steps": len(categories) + 1,
            "physics_workers": 4,
            "locomotion_segment_ticks": 1,
            "kick_move_6_duration_ticks": KICK_DURATION_TICKS[0],
            "kick_move_7_duration_ticks": KICK_DURATION_TICKS[1],
            "kick_move_8_duration_ticks": KICK_DURATION_TICKS[2],
            "kick_move_9_duration_ticks": KICK_DURATION_TICKS[3],
        },
    }
    vector = native.create_vec(arguments, 0)
    try:
        _require(native.env_name == "rek_g1", "extension environment mismatch")
        _require(vector.total_agents == ROBOT_ROWS, "extension row count mismatch")
        _require(vector.obs_size == OBSERVATION_FLOATS, "extension observation ABI mismatch")
        _require(vector.num_atns == ACTION_HEADS, "extension action head mismatch")
        _require(vector.act_sizes == [ACTION_CATEGORIES], "extension action ABI mismatch")
        observations = _float_view(int(vector.obs_ptr), ROBOT_ROWS, OBSERVATION_FLOATS)
        vector.reset()
        _require(bool(np.isfinite(observations).all()), "reset observations are nonfinite")
        output: dict[int, dict[str, Any]] = {}
        maximum_replica_linf = 0.0
        actions = np.full((ROBOT_ROWS, ACTION_HEADS), 1.0, dtype=np.float32)
        even_rows = (0, 2, 4, 6)
        start = time.perf_counter()
        for tick, category in enumerate(categories):
            actions.fill(1.0)
            actions[list(even_rows), 0] = float(category)
            vector.cpu_step(int(actions.ctypes.data))
            _require(bool(np.isfinite(observations).all()), f"candidate observations nonfinite at tick {tick}")
            replica_linf = max(
                float(np.max(np.abs(observations[row, :7] - observations[0, :7])))
                for row in even_rows[1:]
            )
            maximum_replica_linf = max(maximum_replica_linf, replica_linf)
            position = observations[0, :3].astype(np.float64).tolist()
            quaternion = observations[0, 3:7].astype(np.float64).tolist()
            output[tick] = {
                "tick": tick,
                "world_position_xyz_m": position,
                "world_yaw_radians": _mujoco_yaw_wxyz(quaternion),
            }
        elapsed = time.perf_counter() - start
        ordered_output = [output[tick] for tick in range(len(categories))]
        return output, {
            "host": {
                "hostname": platform.node(),
                "machine": platform.machine(),
                "platform": sys.platform,
                "python": platform.python_version(),
            },
            "timeline_steps": len(categories),
            "elapsed_seconds": elapsed,
            "vector_steps_per_second": len(categories) / elapsed,
            "robot_steps_per_second": ROBOT_ROWS * len(categories) / elapsed,
            "player_replica_root_pose_linf": maximum_replica_linf,
            "player_rows": list(even_rows),
            "opponent_rows": [1, 3, 5, 7],
            "opponent_action": "neutral category 1 for every tick",
            "category_sequence_sha256": _category_sequence_sha256(categories),
            "candidate_root_pose_sequence_sha256": _canonical_json_sha256(ordered_output),
        }
    finally:
        vector.close()


def _segment_frame(samples: list[dict[str, Any]], runtime: str) -> list[dict[str, float | int]]:
    _require(runtime in {"real", "candidate"}, "trajectory runtime invalid")
    _require(samples, "trajectory segment is empty")
    yaws = _unwrap(float(sample["world_yaw_radians"]) for sample in samples)
    origin = _finite_vector(samples[0]["world_position_xyz_m"], 3, "trajectory origin")
    initial_yaw = yaws[0]
    output: list[dict[str, float | int]] = []
    for sample, yaw in zip(samples, yaws):
        position = _finite_vector(sample["world_position_xyz_m"], 3, "trajectory position")
        if runtime == "real":
            dx = position[0] - origin[0]
            dz = position[2] - origin[2]
            right = dx * math.cos(initial_yaw) - dz * math.sin(initial_yaw)
            forward = dx * math.sin(initial_yaw) + dz * math.cos(initial_yaw)
            height = position[1] - origin[1]
            yaw_delta = yaw - initial_yaw
        else:
            dx = position[0] - origin[0]
            dy = position[1] - origin[1]
            forward = dx * math.cos(initial_yaw) + dy * math.sin(initial_yaw)
            left = -dx * math.sin(initial_yaw) + dy * math.cos(initial_yaw)
            right = -left
            height = position[2] - origin[2]
            yaw_delta = -(yaw - initial_yaw)
        output.append(
            {
                "tick": int(sample["tick"]),
                "right_m": right,
                "forward_m": forward,
                "height_delta_m": height,
                "yaw_delta_radians": yaw_delta,
            }
        )
    return output


def compare_segments(
    real_segments: list[dict[str, Any]],
    candidate_by_tick: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for real_segment in real_segments:
        real_samples = real_segment.get("samples")
        _require(isinstance(real_samples, list) and real_samples, "real segment samples missing")
        candidate_samples = []
        for sample in real_samples:
            _require(isinstance(sample, dict) and isinstance(sample.get("tick"), int), "real sample tick invalid")
            tick = int(sample["tick"])
            _require(tick in candidate_by_tick, "candidate timeline missing a comparison tick")
            candidate_samples.append(candidate_by_tick[tick])
        real_frame = _segment_frame(real_samples, "real")
        candidate_frame = _segment_frame(candidate_samples, "candidate")
        aligned_samples: list[dict[str, Any]] = []
        planar_squared = 0.0
        height_squared = 0.0
        yaw_squared = 0.0
        maximum_planar = 0.0
        maximum_height = 0.0
        maximum_yaw = 0.0
        for real_point, candidate_point in zip(real_frame, candidate_frame):
            _require(real_point["tick"] == candidate_point["tick"], "comparison tick mismatch")
            right_error = float(candidate_point["right_m"]) - float(real_point["right_m"])
            forward_error = float(candidate_point["forward_m"]) - float(real_point["forward_m"])
            planar_error = math.hypot(right_error, forward_error)
            height_error = abs(
                float(candidate_point["height_delta_m"]) - float(real_point["height_delta_m"])
            )
            yaw_error = _wrapped_abs_error(
                float(candidate_point["yaw_delta_radians"]),
                float(real_point["yaw_delta_radians"]),
            )
            planar_squared += planar_error * planar_error
            height_squared += height_error * height_error
            yaw_squared += yaw_error * yaw_error
            maximum_planar = max(maximum_planar, planar_error)
            maximum_height = max(maximum_height, height_error)
            maximum_yaw = max(maximum_yaw, yaw_error)
            aligned_samples.append(
                {
                    "tick": real_point["tick"],
                    "real": real_point,
                    "candidate": candidate_point,
                    "error": {
                        "planar_m": planar_error,
                        "height_delta_m": height_error,
                        "yaw_radians": yaw_error,
                    },
                }
            )
        count = len(aligned_samples)
        reports.append(
            {
                "ordinal": real_segment["ordinal"],
                "label": real_segment["label"],
                "category": real_segment["category"],
                "sample_count": count,
                "tick_start_inclusive": real_segment["tick_start_inclusive"],
                "tick_stop_exclusive": real_segment["tick_stop_exclusive"],
                "nominal_schedule": real_segment["nominal_schedule"],
                "diagnostics": {
                    "planar_rmse_m": math.sqrt(planar_squared / count),
                    "planar_max_m": maximum_planar,
                    "height_delta_rmse_m": math.sqrt(height_squared / count),
                    "height_delta_max_m": maximum_height,
                    "yaw_rmse_radians": math.sqrt(yaw_squared / count),
                    "yaw_max_radians": maximum_yaw,
                },
                "samples": aligned_samples,
            }
        )
    return reports


def replay_candidate(
    contract_path: Path,
    extension_path: Path,
    asset_root: Path,
    encoder_path: Path,
    decoder_path: Path,
) -> dict[str, Any]:
    contract_path = _regular_file(contract_path, "real replay contract")
    extension_path = _regular_file(extension_path, "Puffer extension")
    asset_root = asset_root.resolve()
    _require(asset_root.is_dir() and not asset_root.is_symlink(), "semantic asset root invalid")
    encoder_path = _regular_file(encoder_path, "encoder ONNX")
    decoder_path = _regular_file(decoder_path, "decoder ONNX")
    manifest_path = _regular_file(asset_root / "semantic_duel_assets_manifest.json", "asset manifest")
    contract = _load_json(contract_path, "real replay contract")
    _require(contract.get("schema") == REAL_CONTRACT_SCHEMA, "real replay contract schema mismatch")
    _require(contract.get("rek_parity_claim") is False, "real replay contract parity flag invalid")

    previous = {
        name: os.environ.get(name)
        for name in (
            "REK_G1_SEMANTIC_ASSETS_DIR",
            "REK_G1_ENCODER_ONNX",
            "REK_G1_DECODER_ONNX",
        )
    }
    os.environ["REK_G1_SEMANTIC_ASSETS_DIR"] = str(asset_root)
    os.environ["REK_G1_ENCODER_ONNX"] = str(encoder_path)
    os.environ["REK_G1_DECODER_ONNX"] = str(decoder_path)
    try:
        candidate_by_tick, runtime = _candidate_samples(contract, extension_path)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    segments = compare_segments(contract["real_segments"], candidate_by_tick)
    categories = _expand_categories(contract["action_contract"])
    action_hash_matches = runtime["category_sequence_sha256"] == contract["action_contract"][
        "category_sequence_sha256"
    ]
    return {
        "schema": REPORT_SCHEMA,
        "classification": "candidate_diagnostic_against_single_confounded_real_trace",
        "rek_parity_claim": False,
        "training_enabled": False,
        "checks": {
            "exact_recorded_category_timeline_replayed": action_hash_matches,
            "candidate_observations_finite": True,
            "all_14_held_categories_replayed": set(categories) == set(range(1, 16)),
            "candidate_player_replicas_binary32_equal": runtime[
                "player_replica_root_pose_linf"
            ]
            == 0.0,
        },
        "gates": {
            **contract["gates"],
            "parity_acceptance_evaluable": False,
        },
        "coordinate_contract": contract["coordinate_contract"],
        "runtime": runtime,
        "segments": segments,
        "provenance": {
            "real_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
            "real_sources": contract["provenance"],
            "candidate_inputs": {
                "extension": {"path": str(extension_path), "sha256": _sha256(extension_path)},
                "asset_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
                "encoder": {"path": str(encoder_path), "sha256": _sha256(encoder_path)},
                "decoder": {"path": str(decoder_path), "sha256": _sha256(decoder_path)},
            },
        },
        "limitations": contract["limitations"],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    parent = path.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path.resolve())
    finally:
        if temporary.exists():
            temporary.unlink()


def _argument_path(value: str | None, environment_name: str, description: str) -> Path:
    selected = value or os.environ.get(environment_name)
    if not selected:
        raise ReplayFailure(f"{description} is required by argument or {environment_name}")
    return Path(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-real")
    extract.add_argument("--trace", type=Path, required=True)
    extract.add_argument("--pose-response", type=Path, required=True)
    extract.add_argument("--coverage", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    replay = commands.add_parser("replay-candidate")
    replay.add_argument("--real-contract", type=Path, required=True)
    replay.add_argument("--extension")
    replay.add_argument("--asset-root")
    replay.add_argument("--encoder")
    replay.add_argument("--decoder")
    replay.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "extract-real":
        output = extract_real_contract(
            arguments.trace,
            arguments.pose_response,
            arguments.coverage,
        )
    else:
        output = replay_candidate(
            arguments.real_contract,
            _argument_path(arguments.extension, "REK_G1_PUFFER_EXTENSION", "Puffer extension"),
            _argument_path(arguments.asset_root, "REK_G1_SEMANTIC_ASSETS_DIR", "semantic assets"),
            _argument_path(arguments.encoder, "REK_G1_ENCODER_ONNX", "encoder ONNX"),
            _argument_path(arguments.decoder, "REK_G1_DECODER_ONNX", "decoder ONNX"),
        )
    _write_json(arguments.out, output)
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
