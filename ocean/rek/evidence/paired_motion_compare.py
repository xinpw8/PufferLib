#!/usr/bin/env python3
"""Command-edge-aligned motion parity diagnostics and synchronized video.

This tool compares one simulator motion against repeated measurements of the
same single command in REK.  It does not estimate a transform by matching the
two motions, use dynamic time warping, subtract their command-edge positions,
or widen a tolerance.  Each input is independently mapped into the arena frame
by an explicit, provenance-bearing similarity transform.  The complete
transform and its determinant are emitted in the report.

Accepted inputs are:

* ``rek.paired_motion_trace.v1`` JSON documents;
* ``rek.t800_canned_move_windows.v2`` JSON documents, with ``--*-window``;
* canonical binary ``REKTRACE`` files containing one ``command_edge`` event.

The JSON trace schema is intentionally small::

    {
      "schema": "rek.paired_motion_trace.v1",
      "source": "rek",
      "build_fingerprint": "...",
      "command": {
        "identity": "walk_forward:press:v1",
        "edge_time_s": 1.0,
        "execution_state": "measured_executed"
      },
      "screen_frame": {"id": "camera-a-1920x1080", "width_px": 1920,
                       "height_px": 1080},
      "calibration": {
        "schema": "rek.paired_motion_calibration.v1",
        "mode": "explicit_similarity_3d",
        "position_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "position_offset": [0, 0, 0],
        "source_forward_vector": [1, 0, 0],
        "direct_yaw_sign": 1,
        "direct_yaw_offset_rad": 0,
        "provenance": {"kind": "arena_landmarks", "artifact": "..."}
      },
      "samples": [
        {"time_s": 0.98, "root_position": [0, 0, 1],
         "root_yaw_rad": 0, "screen_root_px": [960, 700]}
      ]
    }

``root_quaternion_xyzw`` may replace ``root_yaw_rad``.  In that case yaw is
computed by rotating ``source_forward_vector`` into the arena frame and
projecting it onto arena XY.  Screen points are compared only when every input
declares the exact same screen-frame identity and dimensions.  No camera fit is
performed.

The comparison grid is the primary REK trace's measured sample timebase unless
``--fps`` is supplied.  Every input must contribute a distinct measured sample
to every grid point within the explicitly declared timestamp uncertainty.
Interpolation is not acceptance evidence.  The evaluation window is the
primary REK trace's complete relative-time range unless explicitly bounded.
Missing coverage never causes silent cropping.

At least three REK runs are required for a verdict.  At every frame the
candidate's root-position, yaw, and optional screen error is held to the chosen
quantile of all pairwise REK-repeat errors at that same frame.  The default is
``p99``.  A zero REK envelope therefore requires exact equality.

Rendered video is acceptance-bearing only when each source video has a
``rek.video_clock_anchor.v1`` sidecar.  The sidecar binds the video SHA-256,
trace SHA-256, command identity, measured command-edge video PTS, measurement
method, provenance, and uncertainty.  Manual video offsets can render a useful
diagnostic but can never support an acceptance or frame-perfect verdict.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


TRACE_SCHEMA = "rek.paired_motion_trace.v1"
CALIBRATION_SCHEMA = "rek.paired_motion_calibration.v1"
REPORT_SCHEMA = "rek.paired_motion_comparison.v1"
CANNED_SCHEMA = "rek.t800_canned_move_windows.v2"
VIDEO_CLOCK_ANCHOR_SCHEMA = "rek.video_clock_anchor.v1"
QUANTILES = {"median": 0.5, "p95": 0.95, "p99": 0.99, "max": 1.0}
EPSILON = 1e-12


class MotionComparisonError(ValueError):
    """An input cannot support the requested, unambiguous comparison."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionComparisonError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MotionComparisonError(f"{label} must be a finite number")
    return result


def _vector(value: Any, width: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise MotionComparisonError(f"{label} must contain {width} numbers")
    return tuple(_finite(component, f"{label}[{index}]")
                 for index, component in enumerate(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) \
        -> tuple[float, ...]:
    return tuple(_dot(row, vector) for row in matrix)


def _determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _quat_rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) \
        -> tuple[float, float, float]:
    x, y, z, w = quaternion
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= EPSILON:
        raise MotionComparisonError("root quaternion has zero norm")
    x, y, z, w = x / length, y / length, z / length, w / length
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise MotionComparisonError("a quantile needs measured values")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "rms": None,
        }
    numeric = [float(value) for value in values]
    return {
        "sample_count": len(numeric),
        "min": min(numeric),
        "median": _quantile(numeric, 0.5),
        "p95": _quantile(numeric, 0.95),
        "p99": _quantile(numeric, 0.99),
        "max": max(numeric),
        "rms": math.sqrt(sum(value * value for value in numeric) / len(numeric)),
    }


def _provenance(value: Any, label: str = "calibration provenance") -> Any:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict) and value:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if encoded != "{}":
            return value
    raise MotionComparisonError(
        f"{label} must name a measured artifact or landmark set"
    )


def _optional_identifier(label: str,
                         records: Sequence[tuple[str, dict[str, Any]]],
                         keys: Sequence[str]) -> str | None:
    """Return one consistent non-empty identifier from candidate records."""
    found: list[tuple[str, str, str]] = []
    for record_label, record in records:
        for key in keys:
            if key not in record or record[key] is None:
                continue
            value = record[key]
            if not isinstance(value, str) or not value.strip():
                raise MotionComparisonError(
                    f"{record_label}.{key} must be a non-empty string"
                )
            found.append((record_label, key, value.strip()))
    values = {value for _, _, value in found}
    if len(values) > 1:
        locations = [f"{record}.{key}={value!r}"
                     for record, key, value in found]
        raise MotionComparisonError(
            f"conflicting {label} identifiers: {locations}"
        )
    return next(iter(values)) if values else None


@dataclass(frozen=True)
class Calibration:
    matrix: tuple[tuple[float, float, float], ...]
    rotation: tuple[tuple[float, float, float], ...]
    offset: tuple[float, float, float]
    source_forward: tuple[float, float, float]
    direct_yaw_sign: int
    direct_yaw_offset: float
    scale: float
    determinant: float
    provenance: Any
    source_document: dict[str, Any]
    source_path: str | None
    source_sha256: str | None

    def position(self, value: Sequence[float]) -> tuple[float, float, float]:
        transformed = _mat_vec(self.matrix, value)
        return tuple(transformed[index] + self.offset[index]
                     for index in range(3))

    def direct_yaw(self, value: float) -> float:
        return _wrap_angle(self.direct_yaw_sign * value + self.direct_yaw_offset)

    def quaternion_yaw(self, value: Sequence[float]) -> float:
        source_heading = _quat_rotate_xyzw(value, self.source_forward)
        arena_heading = _mat_vec(self.rotation, source_heading)
        horizontal_length = math.hypot(arena_heading[0], arena_heading[1])
        if horizontal_length <= 1e-9:
            raise MotionComparisonError(
                "calibrated root heading is vertical; yaw is undefined"
            )
        return math.atan2(arena_heading[1], arena_heading[0])

    def report(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_SCHEMA,
            "mode": "explicit_similarity_3d",
            "position_matrix": [list(row) for row in self.matrix],
            "position_offset": list(self.offset),
            "source_forward_vector": list(self.source_forward),
            "direct_yaw_sign": self.direct_yaw_sign,
            "direct_yaw_offset_rad": self.direct_yaw_offset,
            "uniform_scale": self.scale,
            "orthogonal_determinant": self.determinant,
            "includes_reflection": self.determinant < 0.0,
            "provenance": self.provenance,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "trajectory_fit_used": False,
        }


def parse_calibration(document: dict[str, Any], *, source_path: Path | None = None,
                      source_sha256: str | None = None) -> Calibration:
    if not isinstance(document, dict):
        raise MotionComparisonError("calibration must be a JSON object")
    if document.get("schema") != CALIBRATION_SCHEMA:
        raise MotionComparisonError(
            f"calibration schema must be {CALIBRATION_SCHEMA!r}"
        )
    if document.get("mode") not in ("identity", "explicit_similarity_3d"):
        raise MotionComparisonError(
            "calibration mode must be identity or explicit_similarity_3d"
        )
    common_keys = {"schema", "mode", "source_forward_vector", "provenance"}
    explicit_keys = {
        "position_matrix", "position_offset", "direct_yaw_sign",
        "direct_yaw_offset_rad",
    }
    allowed_keys = common_keys | (explicit_keys if document["mode"] != "identity"
                                  else set())
    unknown_keys = sorted(set(document) - allowed_keys)
    if unknown_keys:
        raise MotionComparisonError(
            f"calibration contains unsupported fields: {unknown_keys}"
        )
    provenance = _provenance(document.get("provenance"))
    if document["mode"] == "identity":
        matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        offset = (0.0, 0.0, 0.0)
        source_forward = _vector(
            document.get("source_forward_vector", [1.0, 0.0, 0.0]), 3,
            "calibration.source_forward_vector",
        )
        yaw_sign = 1
        yaw_offset = 0.0
    else:
        raw_matrix = document.get("position_matrix")
        if not isinstance(raw_matrix, list) or len(raw_matrix) != 3:
            raise MotionComparisonError(
                "calibration.position_matrix must be a 3x3 matrix"
            )
        matrix = tuple(_vector(row, 3, f"calibration.position_matrix[{index}]")
                       for index, row in enumerate(raw_matrix))
        offset = _vector(document.get("position_offset"), 3,
                         "calibration.position_offset")
        source_forward = _vector(
            document.get("source_forward_vector"), 3,
            "calibration.source_forward_vector",
        )
        yaw_sign = document.get("direct_yaw_sign")
        if yaw_sign not in (-1, 1):
            raise MotionComparisonError(
                "calibration.direct_yaw_sign must be -1 or 1"
            )
        yaw_offset = _finite(document.get("direct_yaw_offset_rad"),
                             "calibration.direct_yaw_offset_rad")

    columns = [tuple(matrix[row][column] for row in range(3))
               for column in range(3)]
    scales = [_norm(column) for column in columns]
    if min(scales) <= EPSILON:
        raise MotionComparisonError("calibration position matrix is singular")
    scale = sum(scales) / 3.0
    tolerance = max(1e-10, scale * 1e-9)
    if any(abs(item - scale) > tolerance for item in scales):
        raise MotionComparisonError(
            "calibration position matrix has nonuniform scale"
        )
    for left, right in itertools.combinations(columns, 2):
        if abs(_dot(left, right)) > max(1e-10, scale * scale * 1e-9):
            raise MotionComparisonError(
                "calibration position matrix contains shear"
            )
    rotation = tuple(tuple(component / scale for component in row)
                     for row in matrix)
    determinant = _determinant_3x3(rotation)
    if abs(abs(determinant) - 1.0) > 1e-8:
        raise MotionComparisonError(
            "calibration position matrix is not an orthogonal similarity"
        )
    forward_norm = _norm(source_forward)
    if forward_norm <= EPSILON:
        raise MotionComparisonError("source_forward_vector has zero norm")
    source_forward = tuple(value / forward_norm for value in source_forward)
    return Calibration(
        matrix=matrix,
        rotation=rotation,
        offset=offset,
        source_forward=source_forward,
        direct_yaw_sign=int(yaw_sign),
        direct_yaw_offset=float(yaw_offset),
        scale=scale,
        determinant=determinant,
        provenance=provenance,
        source_document=document,
        source_path=str(source_path.resolve()) if source_path else None,
        source_sha256=source_sha256,
    )


@dataclass(frozen=True)
class RawSample:
    time_s: float
    position: tuple[float, float, float]
    yaw: float | None
    quaternion: tuple[float, float, float, float] | None
    screen: tuple[float, float] | None


@dataclass(frozen=True)
class MotionSample:
    time_s: float
    position: tuple[float, float, float]
    yaw: float
    screen: tuple[float, float] | None


@dataclass
class MotionTrace:
    path: Path
    sha256: str
    input_format: str
    source: str
    build_fingerprint: str | None
    capture_id: str | None
    run_id: str | None
    command_identity: str
    command_edge_time_s: float
    execution_state: str
    samples: list[MotionSample]
    calibration: Calibration
    screen_frame: dict[str, Any] | None
    video_edge_time_s: float | None
    adapter_notes: list[str]

    @property
    def relative_times(self) -> list[float]:
        return [sample.time_s - self.command_edge_time_s for sample in self.samples]

    def report_identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
            "format": self.input_format,
            "source": self.source,
            "build_fingerprint": self.build_fingerprint,
            "capture_id": self.capture_id,
            "run_id": self.run_id,
            "command_identity": self.command_identity,
            "command_edge_time_s": self.command_edge_time_s,
            "execution_state": self.execution_state,
            "sample_count": len(self.samples),
            "first_relative_time_s": self.relative_times[0],
            "last_relative_time_s": self.relative_times[-1],
            "screen_frame": self.screen_frame,
            "video_edge_time_s": self.video_edge_time_s,
            "calibration": self.calibration.report(),
            "adapter_notes": self.adapter_notes,
        }


@dataclass(frozen=True)
class RawTrace:
    input_format: str
    source: str
    build_fingerprint: str | None
    capture_id: str | None
    run_id: str | None
    command_identity: str
    command_edge_time_s: float
    execution_state: str
    samples: list[RawSample]
    screen_frame: dict[str, Any] | None
    video_edge_time_s: float | None
    inline_calibration: dict[str, Any] | None
    adapter_notes: list[str]


def _screen_frame(value: Any, has_screen: bool) -> dict[str, Any] | None:
    if not has_screen:
        return None
    if not isinstance(value, dict):
        raise MotionComparisonError(
            "screen samples require a measured screen_frame object"
        )
    identity = value.get("id")
    if not isinstance(identity, str) or not identity.strip():
        raise MotionComparisonError("screen_frame.id must be non-empty")
    width = value.get("width_px")
    height = value.get("height_px")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise MotionComparisonError("screen_frame.width_px must be positive")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise MotionComparisonError("screen_frame.height_px must be positive")
    return {"id": identity, "width_px": width, "height_px": height}


def _parse_json_trace(document: dict[str, Any]) -> RawTrace:
    source = document.get("source")
    if source != "rek" and not str(source).startswith("clone:"):
        raise MotionComparisonError("trace source must be rek or clone:<name>")
    fingerprint = document.get("build_fingerprint")
    if fingerprint is not None and (not isinstance(fingerprint, str)
                                    or not fingerprint.strip()):
        raise MotionComparisonError("build_fingerprint must be non-empty")
    command = document.get("command")
    if not isinstance(command, dict):
        raise MotionComparisonError("trace command object is absent")
    identifier_records = (("trace", document), ("trace.command", command))
    capture_id = _optional_identifier(
        "capture", identifier_records, ("capture_id",)
    )
    run_id = _optional_identifier(
        "run", identifier_records, ("schedule_run_id", "run_id")
    )
    identity = command.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise MotionComparisonError("command.identity must be non-empty")
    edge = _finite(command.get("edge_time_s"), "command.edge_time_s")
    execution = command.get("execution_state", "unknown")
    if not isinstance(execution, str) or not execution:
        raise MotionComparisonError("command.execution_state must be a string")
    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < 2:
        raise MotionComparisonError("trace needs at least two samples")
    samples = []
    any_screen = False
    all_screen = True
    for index, record in enumerate(raw_samples):
        if not isinstance(record, dict):
            raise MotionComparisonError(f"samples[{index}] must be an object")
        yaw_present = "root_yaw_rad" in record
        quaternion_present = "root_quaternion_xyzw" in record
        if yaw_present == quaternion_present:
            raise MotionComparisonError(
                f"samples[{index}] must contain exactly one root orientation"
            )
        screen = None
        if "screen_root_px" in record and record["screen_root_px"] is not None:
            screen = _vector(record["screen_root_px"], 2,
                             f"samples[{index}].screen_root_px")
            any_screen = True
        else:
            all_screen = False
        samples.append(RawSample(
            time_s=_finite(record.get("time_s"), f"samples[{index}].time_s"),
            position=_vector(record.get("root_position"), 3,
                             f"samples[{index}].root_position"),
            yaw=(_finite(record.get("root_yaw_rad"),
                         f"samples[{index}].root_yaw_rad") if yaw_present else None),
            quaternion=(_vector(record.get("root_quaternion_xyzw"), 4,
                                f"samples[{index}].root_quaternion_xyzw")
                        if quaternion_present else None),
            screen=screen,
        ))
    if any_screen and not all_screen:
        raise MotionComparisonError(
            "screen_root_px must be present in every sample or none"
        )
    video = document.get("video") or {}
    if not isinstance(video, dict):
        raise MotionComparisonError("video metadata must be an object")
    video_edge = video.get("command_edge_time_s")
    if video_edge is not None:
        video_edge = _finite(video_edge, "video.command_edge_time_s")
    return RawTrace(
        input_format=TRACE_SCHEMA,
        source=source,
        build_fingerprint=fingerprint,
        capture_id=capture_id,
        run_id=run_id,
        command_identity=identity,
        command_edge_time_s=edge,
        execution_state=execution,
        samples=samples,
        screen_frame=_screen_frame(document.get("screen_frame"), any_screen),
        video_edge_time_s=video_edge,
        inline_calibration=document.get("calibration"),
        adapter_notes=[],
    )


def _parse_canned(document: dict[str, Any], window_index: int | None) -> RawTrace:
    windows = document.get("windows")
    if not isinstance(windows, list) or not windows:
        raise MotionComparisonError("canned-move artifact has no windows")
    if window_index is None:
        if len(windows) != 1:
            raise MotionComparisonError(
                "canned-move artifact contains multiple windows; select one explicitly"
            )
        window_index = 0
    if window_index < 0 or window_index >= len(windows):
        raise MotionComparisonError(f"window index {window_index} is out of range")
    window = windows[window_index]
    if not isinstance(window, dict):
        raise MotionComparisonError("selected canned-move window is not an object")
    request = window.get("request") or {}
    if not isinstance(request, dict):
        raise MotionComparisonError("selected window request is not an object")
    identifier_records = (
        ("artifact", document), ("window", window), ("window.request", request),
    )
    capture_id = _optional_identifier(
        "capture", identifier_records, ("capture_id",)
    )
    run_id = _optional_identifier(
        "run", identifier_records, ("schedule_run_id", "run_id")
    )
    identity = request.get("move_name")
    if not isinstance(identity, str) or not identity:
        raise MotionComparisonError("selected window has no measured move_name")
    execution = (window.get("executed") or {}).get("state", "unknown")
    samples = []
    for index, record in enumerate(window.get("samples") or []):
        fighter = record.get("fighter") or {}
        samples.append(RawSample(
            time_s=_finite(record.get("relative_time_s"),
                           f"window.samples[{index}].relative_time_s"),
            position=_vector(fighter.get("root_position"), 3,
                             f"window.samples[{index}].fighter.root_position"),
            yaw=None,
            quaternion=_vector(
                fighter.get("root_rotation"), 4,
                f"window.samples[{index}].fighter.root_rotation",
            ),
            screen=None,
        ))
    if len(samples) < 2:
        raise MotionComparisonError("selected canned-move window has fewer than two samples")
    source = document.get("source") or {}
    binary_identity = ":".join(str(source.get(key) or "") for key in (
        "game_assembly_sha256", "global_metadata_sha256"))
    fingerprint = binary_identity if binary_identity.strip(":") else None
    return RawTrace(
        input_format=CANNED_SCHEMA,
        source="rek",
        build_fingerprint=fingerprint,
        capture_id=capture_id,
        run_id=run_id,
        command_identity=identity,
        command_edge_time_s=0.0,
        execution_state=str(execution),
        samples=samples,
        screen_frame=None,
        video_edge_time_s=None,
        inline_calibration=None,
        adapter_notes=[
            "request prefix proves invocation, not server acceptance or execution",
            "client FixedUpdate samples may interpolate remote snapshots",
        ],
    )


def _load_rektrace(path: Path, pilot: str, edge_event: str) -> RawTrace:
    module_path = Path(__file__).with_name("trace.py")
    specification = importlib.util.spec_from_file_location(
        "rek_evidence_trace_reader", module_path
    )
    if specification is None or specification.loader is None:
        raise MotionComparisonError("cannot load the canonical trace reader")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    Trace = module.Trace
    trace = Trace.load(path)
    fixed_delta = _finite(trace.header.get("fixed_delta_time"),
                          "trace fixed_delta_time")
    events = [event for event in trace.events if event.get("kind") == edge_event]
    if len(events) != 1:
        raise MotionComparisonError(
            f"REKTRACE needs exactly one {edge_event!r} event, got {len(events)}"
        )
    edge_record = events[0]
    identifier_records = (
        ("trace.header", trace.header), (f"event[{edge_event}]", edge_record),
    )
    capture_id = _optional_identifier(
        "capture", identifier_records, ("capture_id",)
    )
    run_id = _optional_identifier(
        "run", identifier_records, ("schedule_run_id", "run_id")
    )
    identity = (edge_record.get("command_identity") or edge_record.get("command")
                or edge_record.get("name"))
    if not isinstance(identity, str) or not identity:
        raise MotionComparisonError(
            f"{edge_event!r} event has no command_identity"
        )
    edge_tick = edge_record.get("tick")
    if not isinstance(edge_tick, int) or isinstance(edge_tick, bool):
        raise MotionComparisonError(f"{edge_event!r} event tick is not an integer")
    prefix = f"root.{pilot}"
    position_names = tuple(f"{prefix}.pos.{axis}" for axis in "xyz")
    if not all(name in trace.channels for name in position_names):
        raise MotionComparisonError(
            f"REKTRACE lacks complete {prefix}.pos.x/y/z channels"
        )
    yaw_name = f"{prefix}.yaw"
    quaternion_names = tuple(f"{prefix}.quat.{axis}" for axis in "xyzw")
    has_yaw = yaw_name in trace.channels
    has_quaternion = all(name in trace.channels for name in quaternion_names)
    if has_yaw == has_quaternion:
        raise MotionComparisonError(
            f"REKTRACE must expose exactly one of {yaw_name} or complete quaternion"
        )
    screen_names = (f"screen.{pilot}.root.x", f"screen.{pilot}.root.y")
    has_screen = all(name in trace.channels for name in screen_names)
    if any(name in trace.channels for name in screen_names) and not has_screen:
        raise MotionComparisonError("REKTRACE has an incomplete screen root pair")
    samples = []
    for index, tick in enumerate(trace.ticks):
        samples.append(RawSample(
            time_s=tick * fixed_delta,
            position=tuple(float(trace.channels[name][index])
                           for name in position_names),
            yaw=float(trace.channels[yaw_name][index]) if has_yaw else None,
            quaternion=(tuple(float(trace.channels[name][index])
                              for name in quaternion_names)
                        if has_quaternion else None),
            screen=(tuple(float(trace.channels[name][index])
                          for name in screen_names) if has_screen else None),
        ))
    video = trace.header.get("video") or {}
    video_edge = video.get("command_edge_time_s") if isinstance(video, dict) else None
    if video_edge is not None:
        video_edge = _finite(video_edge, "trace video.command_edge_time_s")
    return RawTrace(
        input_format="REKTRACE.v1",
        source=trace.source,
        build_fingerprint=trace.build_fingerprint,
        capture_id=capture_id,
        run_id=run_id,
        command_identity=identity,
        command_edge_time_s=edge_tick * fixed_delta,
        execution_state=trace.header.get("command_execution_state", "unknown"),
        samples=samples,
        screen_frame=_screen_frame(trace.header.get("screen_frame"), has_screen),
        video_edge_time_s=video_edge,
        inline_calibration=trace.header.get("arena_calibration"),
        adapter_notes=[],
    )


def _load_calibration(external_path: Path | None,
                      inline: dict[str, Any] | None) -> Calibration:
    if external_path is not None and inline is not None:
        raise MotionComparisonError(
            "calibration is present both inline and on the command line"
        )
    if external_path is not None:
        document = json.loads(external_path.read_text(encoding="utf-8"))
        return parse_calibration(document, source_path=external_path,
                                 source_sha256=_sha256(external_path))
    if inline is None:
        raise MotionComparisonError(
            "input has no coordinate calibration; supply a measured calibration"
        )
    return parse_calibration(inline)


def load_motion_trace(path: Path, *, calibration_path: Path | None = None,
                      window_index: int | None = None, pilot: str = "0",
                      edge_event: str = "command_edge") -> MotionTrace:
    path = path.resolve()
    digest = _sha256(path)
    with path.open("rb") as stream:
        head = stream.read(9)
    if head == b"REKTRACE\0":
        raw = _load_rektrace(path, pilot, edge_event)
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise MotionComparisonError("motion input must be a JSON object")
        if document.get("schema") == TRACE_SCHEMA:
            raw = _parse_json_trace(document)
        elif document.get("schema") == CANNED_SCHEMA:
            raw = _parse_canned(document, window_index)
        else:
            raise MotionComparisonError(
                f"unsupported motion schema {document.get('schema')!r}"
            )
    calibration = _load_calibration(calibration_path, raw.inline_calibration)
    samples = []
    last_time = None
    for index, sample in enumerate(raw.samples):
        if last_time is not None and sample.time_s <= last_time:
            raise MotionComparisonError(
                f"sample times must strictly increase at index {index}"
            )
        last_time = sample.time_s
        if sample.yaw is not None:
            yaw = calibration.direct_yaw(sample.yaw)
        elif sample.quaternion is not None:
            yaw = calibration.quaternion_yaw(sample.quaternion)
        else:
            raise AssertionError("validated sample has no orientation")
        samples.append(MotionSample(
            time_s=sample.time_s,
            position=calibration.position(sample.position),
            yaw=yaw,
            screen=sample.screen,
        ))
    if raw.command_edge_time_s < samples[0].time_s - EPSILON:
        raise MotionComparisonError("command edge precedes the first sample")
    if raw.command_edge_time_s > samples[-1].time_s + EPSILON:
        raise MotionComparisonError("command edge follows the last sample")
    return MotionTrace(
        path=path,
        sha256=digest,
        input_format=raw.input_format,
        source=raw.source,
        build_fingerprint=raw.build_fingerprint,
        capture_id=raw.capture_id,
        run_id=raw.run_id,
        command_identity=raw.command_identity,
        command_edge_time_s=raw.command_edge_time_s,
        execution_state=raw.execution_state,
        samples=samples,
        calibration=calibration,
        screen_frame=raw.screen_frame,
        video_edge_time_s=raw.video_edge_time_s,
        adapter_notes=raw.adapter_notes,
    )


@dataclass(frozen=True)
class MeasuredGridMatch:
    sample: MotionSample
    source_sample_index: int
    source_relative_time_s: float
    timestamp_error_s: float
    exact: bool


def _measured_sample_on_grid(trace: MotionTrace, relative_time_s: float,
                             maximum_timestamp_uncertainty_s: float) \
        -> MeasuredGridMatch | None:
    """Select the nearest measured sample without synthesizing a value."""
    times = trace.relative_times
    insertion = bisect.bisect_left(times, relative_time_s)
    candidate_indices = []
    if insertion < len(times):
        candidate_indices.append(insertion)
    if insertion > 0:
        candidate_indices.append(insertion - 1)
    if not candidate_indices:
        return None
    source_index = min(
        candidate_indices,
        key=lambda index: (abs(times[index] - relative_time_s), index),
    )
    error = abs(times[source_index] - relative_time_s)
    if error > maximum_timestamp_uncertainty_s + EPSILON:
        return None
    sample = trace.samples[source_index]
    return MeasuredGridMatch(
        sample=sample,
        source_sample_index=source_index,
        source_relative_time_s=times[source_index],
        timestamp_error_s=error,
        exact=error <= EPSILON,
    )


def _position_error(left: MotionSample, right: MotionSample) -> float:
    return _norm(tuple(right.position[index] - left.position[index]
                       for index in range(3)))


def _yaw_error(left: MotionSample, right: MotionSample) -> float:
    return abs(_wrap_angle(right.yaw - left.yaw))


def _screen_error(left: MotionSample, right: MotionSample) -> float:
    if left.screen is None or right.screen is None:
        raise MotionComparisonError("screen error requested without screen points")
    return _norm(tuple(right.screen[index] - left.screen[index]
                       for index in range(2)))


def _screen_key(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _grid(reference: MotionTrace, start_s: float | None, end_s: float | None,
          fps: float | None) -> tuple[list[float], dict[str, Any]]:
    available = reference.relative_times
    start = available[0] if start_s is None else _finite(start_s, "start_s")
    end = available[-1] if end_s is None else _finite(end_s, "end_s")
    if end < start:
        raise MotionComparisonError("end_s precedes start_s")
    if start < available[0] - EPSILON or end > available[-1] + EPSILON:
        raise MotionComparisonError(
            "requested window exceeds the primary REK trace coverage"
        )
    if fps is None:
        points = [value for value in available
                  if value >= start - EPSILON and value <= end + EPSILON]
        source = "primary_rek_sample_timebase"
        rate = None
    else:
        fps = _finite(fps, "fps")
        if fps <= 0.0:
            raise MotionComparisonError("fps must be positive")
        count = int(math.floor((end - start) * fps + EPSILON)) + 1
        points = [start + index / fps for index in range(count)]
        if points[-1] < end - EPSILON:
            points.append(end)
        source = "explicit_uniform_timebase"
        rate = fps
    if not points:
        raise MotionComparisonError("comparison grid has no frames")
    return points, {
        "source": source,
        "fps": rate,
        "start_relative_time_s": points[0],
        "end_relative_time_s": points[-1],
        "frame_count": len(points),
        "dynamic_time_warping_used": False,
        "edge_position_subtraction_used": False,
    }


def _metric_summary(name: str, unit: str, values: list[float],
                    allowances: list[float], passed: list[bool],
                    times: list[float]) -> dict[str, Any]:
    worst_index = max(range(len(values)), key=values.__getitem__) if values else None
    exceedances = [value - allowance
                   for value, allowance in zip(values, allowances)]
    largest_excess_index = (
        max(range(len(exceedances)), key=exceedances.__getitem__)
        if exceedances else None
    )
    return {
        "name": name,
        "unit": unit,
        "error": _distribution(values),
        "time_indexed_rek_allowance": _distribution(allowances),
        "failed_frame_count": sum(not value for value in passed),
        "passed": all(passed),
        "worst_error_relative_time_s": (
            times[worst_index] if worst_index is not None else None
        ),
        "largest_excess": (
            exceedances[largest_excess_index]
            if largest_excess_index is not None else None
        ),
        "largest_excess_relative_time_s": (
            times[largest_excess_index]
            if largest_excess_index is not None else None
        ),
    }


def _require_distinct_rek_captures(traces: Sequence[MotionTrace]) \
        -> dict[str, Any]:
    collisions: list[str] = []
    resolved_paths: dict[str, int] = {}
    content_hashes: dict[str, int] = {}
    capture_ids: dict[str, int] = {}
    run_ids: dict[str, int] = {}
    for index, trace in enumerate(traces):
        path_key = os.path.normcase(str(trace.path.resolve()))
        for label, value, seen in (
            ("resolved path", path_key, resolved_paths),
            ("SHA-256", trace.sha256.lower(), content_hashes),
            ("capture_id", trace.capture_id, capture_ids),
            ("run_id", trace.run_id, run_ids),
        ):
            if value is None:
                continue
            if value in seen:
                collisions.append(
                    f"rek[{seen[value]}] and rek[{index}] share {label} {value!r}"
                )
            else:
                seen[value] = index
    if collisions:
        raise MotionComparisonError(
            "REK repeat inputs are not distinct captures: " + "; ".join(collisions)
        )
    return {
        "distinct_resolved_paths": True,
        "distinct_sha256": True,
        "capture_id_available_count": len(capture_ids),
        "capture_ids_distinct_where_available": True,
        "run_id_available_count": len(run_ids),
        "run_ids_distinct_where_available": True,
    }


def compare_motion(reference: MotionTrace, repeats: Sequence[MotionTrace],
                   candidate: MotionTrace, *, accept_at: str = "p99",
                   require_screen: bool = True, start_s: float | None = None,
                   end_s: float | None = None, fps: float | None = None,
                   maximum_timestamp_uncertainty_s: float | None = None) \
        -> dict[str, Any]:
    if accept_at not in QUANTILES:
        raise MotionComparisonError(
            f"accept_at must be one of {tuple(QUANTILES)}"
        )
    rek_traces = [reference, *repeats]
    if reference.source != "rek" or any(trace.source != "rek" for trace in repeats):
        raise MotionComparisonError("reference and every repeat must be REK traces")
    distinct_capture_evidence = _require_distinct_rek_captures(rek_traces)
    if not candidate.source.startswith("clone:"):
        raise MotionComparisonError("candidate source must be clone:<name>")
    command_identities = {trace.command_identity for trace in [*rek_traces, candidate]}
    if len(command_identities) != 1:
        raise MotionComparisonError(
            f"motion inputs name different commands: {sorted(command_identities)}"
        )
    fingerprints = {trace.build_fingerprint for trace in [*rek_traces, candidate]}
    identity_complete = None not in fingerprints and "" not in fingerprints
    identity_matches = identity_complete and len(fingerprints) == 1
    if identity_complete and not identity_matches:
        raise MotionComparisonError("motion inputs identify different REK builds")

    if maximum_timestamp_uncertainty_s is None:
        raise MotionComparisonError(
            "maximum_timestamp_uncertainty_s must be declared explicitly"
        )
    maximum_timestamp_uncertainty_s = _finite(
        maximum_timestamp_uncertainty_s,
        "maximum_timestamp_uncertainty_s",
    )
    if maximum_timestamp_uncertainty_s < 0.0:
        raise MotionComparisonError(
            "maximum_timestamp_uncertainty_s cannot be negative"
        )
    times, alignment = _grid(reference, start_s, end_s, fps)
    grid_intervals = [right - left for left, right in zip(times, times[1:])]
    half_minimum_grid_interval = (
        min(grid_intervals) / 2.0 if grid_intervals else None
    )
    if (half_minimum_grid_interval is not None
            and maximum_timestamp_uncertainty_s
            > half_minimum_grid_interval + EPSILON):
        raise MotionComparisonError(
            "maximum_timestamp_uncertainty_s exceeds half the minimum "
            "comparison-grid interval"
        )
    alignment.update({
        "sample_selection": "nearest_distinct_measured_sample_no_interpolation",
        "declared_maximum_timestamp_uncertainty_s": (
            maximum_timestamp_uncertainty_s
        ),
        "half_minimum_grid_interval_s": half_minimum_grid_interval,
        "interpolation_used": False,
    })
    interpolation: dict[str, dict[str, Any]] = {}
    aligned: list[list[MotionSample] | None] = []
    coverage_failures = []
    missing_measurements = False
    reused_measurements = False
    for trace_index, trace in enumerate([*rek_traces, candidate]):
        rows = [
            _measured_sample_on_grid(
                trace, value, maximum_timestamp_uncertainty_s
            )
            for value in times
        ]
        missing = [times[index] for index, row in enumerate(rows) if row is None]
        label = "candidate" if trace_index == len(rek_traces) else f"rek[{trace_index}]"
        source_indices = [row.source_sample_index for row in rows if row is not None]
        reused_source_indices = sorted({
            index for index in source_indices if source_indices.count(index) > 1
        })
        interpolation[label] = {
            "sample_count": len(rows),
            "exact_sample_count": sum(row is not None and row.exact for row in rows),
            "measured_nearest_sample_count": sum(
                row is not None and not row.exact for row in rows
            ),
            "interpolated_sample_count": 0,
            "interpolation_used": False,
            "declared_maximum_timestamp_uncertainty_s": (
                maximum_timestamp_uncertainty_s
            ),
            "maximum_observed_timestamp_error_s": max(
                (row.timestamp_error_s for row in rows if row is not None),
                default=None,
            ),
            "source_sample_indices": source_indices,
            "source_relative_times_s": [
                row.source_relative_time_s for row in rows if row is not None
            ],
            "timestamp_errors_s": [
                row.timestamp_error_s for row in rows if row is not None
            ],
            "reused_source_sample_indices": reused_source_indices,
            "missing_relative_times_s": missing,
        }
        if missing or reused_source_indices:
            coverage_failures.append(label)
            aligned.append(None)
            missing_measurements = missing_measurements or bool(missing)
            reused_measurements = (
                reused_measurements or bool(reused_source_indices)
            )
        else:
            aligned.append([row.sample for row in rows if row is not None])

    repeat_count_sufficient = len(rek_traces) >= 3
    execution_measured = all(
        trace.execution_state == "measured_executed" for trace in rek_traces
    )
    screen_keys = {_screen_key(trace.screen_frame)
                   for trace in [*rek_traces, candidate]}
    screen_comparable = None not in screen_keys and len(screen_keys) == 1

    blockers = []
    if not repeat_count_sufficient:
        blockers.append("fewer_than_three_rek_runs")
    if not execution_measured:
        blockers.append("rek_command_execution_not_measured")
    if not identity_matches:
        blockers.append("build_identity_missing")
    if coverage_failures:
        blockers.append("samples_not_measured_on_common_grid")
    if missing_measurements:
        blockers.append("incomplete_time_coverage")
    if reused_measurements:
        blockers.append("source_sample_reused_across_grid_points")
    if require_screen and not screen_comparable:
        blockers.append("common_screen_frame_not_measured")

    probability = QUANTILES[accept_at]
    frames = []
    metric_values = {"root_position": [], "root_yaw": [], "screen_root": []}
    metric_allowances = {"root_position": [], "root_yaw": [], "screen_root": []}
    metric_passes = {"root_position": [], "root_yaw": [], "screen_root": []}
    if not coverage_failures and repeat_count_sufficient:
        rek_aligned = [rows for rows in aligned[:len(rek_traces)] if rows is not None]
        candidate_aligned = aligned[-1]
        assert candidate_aligned is not None
        for frame_index, relative_time in enumerate(times):
            reference_sample = rek_aligned[0][frame_index]
            candidate_sample = candidate_aligned[frame_index]
            pair_indices = itertools.combinations(range(len(rek_aligned)), 2)
            pairs = [(rek_aligned[left][frame_index], rek_aligned[right][frame_index])
                     for left, right in pair_indices]
            position_allowance = _quantile(
                (_position_error(left, right) for left, right in pairs), probability
            )
            pairs = [(rek_aligned[left][frame_index], rek_aligned[right][frame_index])
                     for left, right in itertools.combinations(
                         range(len(rek_aligned)), 2)]
            yaw_allowance = _quantile(
                (_yaw_error(left, right) for left, right in pairs), probability
            )
            position_error = _position_error(reference_sample, candidate_sample)
            yaw_signed = _wrap_angle(candidate_sample.yaw - reference_sample.yaw)
            yaw_error = abs(yaw_signed)
            position_delta = [candidate_sample.position[index]
                              - reference_sample.position[index]
                              for index in range(3)]
            record: dict[str, Any] = {
                "frame_index": frame_index,
                "relative_time_s": relative_time,
                "rek_root_position_m": list(reference_sample.position),
                "candidate_root_position_m": list(candidate_sample.position),
                "root_position_delta_m": position_delta,
                "root_position_error_m": position_error,
                "root_position_rek_allowance_m": position_allowance,
                "root_position_within_rek_variance": (
                    position_error <= position_allowance
                ),
                "rek_root_yaw_rad": reference_sample.yaw,
                "candidate_root_yaw_rad": candidate_sample.yaw,
                "root_yaw_signed_error_rad": yaw_signed,
                "root_yaw_error_rad": yaw_error,
                "root_yaw_rek_allowance_rad": yaw_allowance,
                "root_yaw_within_rek_variance": yaw_error <= yaw_allowance,
            }
            metric_values["root_position"].append(position_error)
            metric_allowances["root_position"].append(position_allowance)
            metric_passes["root_position"].append(position_error <= position_allowance)
            metric_values["root_yaw"].append(yaw_error)
            metric_allowances["root_yaw"].append(yaw_allowance)
            metric_passes["root_yaw"].append(yaw_error <= yaw_allowance)
            if screen_comparable:
                pairs = [(rek_aligned[left][frame_index],
                          rek_aligned[right][frame_index])
                         for left, right in itertools.combinations(
                             range(len(rek_aligned)), 2)]
                screen_allowance = _quantile(
                    (_screen_error(left, right) for left, right in pairs), probability
                )
                screen_error = _screen_error(reference_sample, candidate_sample)
                assert reference_sample.screen is not None
                assert candidate_sample.screen is not None
                screen_delta = [candidate_sample.screen[index]
                                - reference_sample.screen[index]
                                for index in range(2)]
                record.update({
                    "rek_screen_root_px": list(reference_sample.screen),
                    "candidate_screen_root_px": list(candidate_sample.screen),
                    "screen_root_delta_px": screen_delta,
                    "screen_root_error_px": screen_error,
                    "screen_root_rek_allowance_px": screen_allowance,
                    "screen_root_within_rek_variance": (
                        screen_error <= screen_allowance
                    ),
                })
                metric_values["screen_root"].append(screen_error)
                metric_allowances["screen_root"].append(screen_allowance)
                metric_passes["screen_root"].append(screen_error <= screen_allowance)
            frames.append(record)

    metrics: dict[str, Any] = {}
    if metric_values["root_position"]:
        metrics["root_position"] = _metric_summary(
            "root_position", "m", metric_values["root_position"],
            metric_allowances["root_position"], metric_passes["root_position"], times,
        )
        metrics["root_yaw"] = _metric_summary(
            "root_yaw", "rad", metric_values["root_yaw"],
            metric_allowances["root_yaw"], metric_passes["root_yaw"], times,
        )
        if screen_comparable:
            metrics["screen_root"] = _metric_summary(
                "screen_root", "px", metric_values["screen_root"],
                metric_allowances["screen_root"], metric_passes["screen_root"], times,
            )
    required_metric_names = ["root_position", "root_yaw"]
    if require_screen:
        required_metric_names.append("screen_root")
    if blockers:
        passed = None
        state = "insufficient_evidence"
    else:
        passed = all(metrics[name]["passed"] for name in required_metric_names)
        state = "passed" if passed else "failed"
    return {
        "schema": REPORT_SCHEMA,
        "verdict": {
            "state": state,
            "passed": passed,
            "blockers": blockers,
            "criterion": (
                "every evaluated frame and required metric is no greater than "
                f"the time-indexed REK {accept_at} repeat envelope"
            ),
        },
        "command_identity": reference.command_identity,
        "accept_at": accept_at,
        "rek_run_count": len(rek_traces),
        "rek_repeat_identity": distinct_capture_evidence,
        "alignment": alignment,
        "interpolation": interpolation,
        "screen_metric": {
            "required": require_screen,
            "comparable": screen_comparable,
            "reason": (None if screen_comparable else
                       "all inputs must carry the same screen frame and complete points"),
        },
        "inputs": {
            "reference": reference.report_identity(),
            "repeats": [trace.report_identity() for trace in repeats],
            "candidate": candidate.report_identity(),
        },
        "metrics": metrics,
        "frames": frames,
        "video": None,
    }


def _probe_video(path: Path, ffprobe: str, *, count_frames: bool = False) \
        -> dict[str, Any]:
    command = [
        ffprobe, "-v", "error",
    ]
    if count_frames:
        command.append("-count_frames")
    command.extend([
        "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,duration,nb_read_frames",
        "-of", "json", str(path),
    ])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MotionComparisonError(
            f"ffprobe failed for {path}: {completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise MotionComparisonError(f"{path} has no unambiguous video stream")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration_s": (float(stream["duration"])
                       if stream.get("duration") not in (None, "N/A") else None),
        "frame_count": (int(stream["nb_read_frames"])
                        if stream.get("nb_read_frames") not in (None, "N/A")
                        else None),
        "command": command,
    }


def _video_frame_rate_hz(probe: dict[str, Any], label: str) -> float:
    value = probe.get("r_frame_rate")
    if not isinstance(value, str) or not value:
        raise MotionComparisonError(f"{label} video frame rate is absent")
    numerator_text, separator, denominator_text = value.partition("/")
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text) if separator else 1.0
    except ValueError as error:
        raise MotionComparisonError(
            f"{label} video frame rate is invalid: {value!r}"
        ) from error
    if (not math.isfinite(numerator) or not math.isfinite(denominator)
            or numerator <= 0.0 or denominator <= 0.0):
        raise MotionComparisonError(
            f"{label} video frame rate is invalid: {value!r}"
        )
    return numerator / denominator


def _load_video_clock_anchor(path: Path, *, video_path: Path,
                             trace: MotionTrace,
                             source_frame_rate_hz: float) \
        -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise MotionComparisonError(f"video clock anchor does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise MotionComparisonError("video clock anchor must be a JSON object")
    if document.get("schema") != VIDEO_CLOCK_ANCHOR_SCHEMA:
        raise MotionComparisonError(
            f"video clock anchor schema must be {VIDEO_CLOCK_ANCHOR_SCHEMA!r}"
        )
    video_sha256 = document.get("video_sha256")
    actual_video_sha256 = _sha256(video_path)
    if (not isinstance(video_sha256, str)
            or video_sha256.lower() != actual_video_sha256):
        raise MotionComparisonError(
            "video clock anchor video_sha256 does not match the source video"
        )
    trace_sha256 = document.get("trace_sha256")
    if (not isinstance(trace_sha256, str)
            or trace_sha256.lower() != trace.sha256.lower()):
        raise MotionComparisonError(
            "video clock anchor trace_sha256 does not match the motion trace"
        )
    identity = document.get("command_identity")
    if identity != trace.command_identity:
        raise MotionComparisonError(
            "video clock anchor command_identity does not match the motion trace"
        )
    edge_pts = _finite(
        document.get("command_edge_video_pts_s"),
        "video clock anchor command_edge_video_pts_s",
    )
    if edge_pts < 0.0:
        raise MotionComparisonError(
            "video clock anchor command-edge PTS cannot be negative"
        )
    measurement = document.get("measurement")
    if not isinstance(measurement, dict):
        raise MotionComparisonError(
            "video clock anchor measurement must be an object"
        )
    if measurement.get("state") != "measured":
        raise MotionComparisonError(
            "video clock anchor measurement.state must be 'measured'"
        )
    method = measurement.get("method")
    if not isinstance(method, str) or not method.strip():
        raise MotionComparisonError(
            "video clock anchor measurement.method must be non-empty"
        )
    lowered_method = method.casefold()
    if any(term in lowered_method for term in (
            "manual", "assumed", "estimated", "inferred")):
        raise MotionComparisonError(
            "video clock anchor measurement.method is not a measured clock bind"
        )
    uncertainty = _finite(
        measurement.get("uncertainty_s"),
        "video clock anchor measurement.uncertainty_s",
    )
    if uncertainty < 0.0:
        raise MotionComparisonError(
            "video clock anchor uncertainty cannot be negative"
        )
    half_frame_s = 0.5 / source_frame_rate_hz
    if uncertainty > half_frame_s + EPSILON:
        raise MotionComparisonError(
            "video clock anchor uncertainty exceeds half a source-video frame"
        )
    provenance = _provenance(
        measurement.get("provenance"), "video clock anchor provenance"
    )
    for label, expected in (
        ("capture_id", trace.capture_id), ("run_id", trace.run_id),
    ):
        supplied = document.get(label)
        if supplied is not None and supplied != expected:
            raise MotionComparisonError(
                f"video clock anchor {label} does not match the motion trace"
            )
    return {
        "schema": VIDEO_CLOCK_ANCHOR_SCHEMA,
        "path": str(path),
        "sha256": _sha256(path),
        "video_sha256": actual_video_sha256,
        "trace_sha256": trace.sha256,
        "command_identity": trace.command_identity,
        "command_edge_video_pts_s": edge_pts,
        "measurement": {
            "state": "measured",
            "method": method,
            "uncertainty_s": uncertainty,
            "maximum_allowed_uncertainty_s": half_frame_s,
            "provenance": provenance,
        },
    }


def render_aligned_video(report: dict[str, Any], reference: MotionTrace,
                         candidate: MotionTrace, *, rek_video: Path,
                         candidate_video: Path, output: Path,
                         layout: str = "side-by-side", fps: float = 50.0,
                         rek_video_anchor: Path | None = None,
                         candidate_video_anchor: Path | None = None,
                         rek_video_edge_s: float | None = None,
                         candidate_video_edge_s: float | None = None) -> dict[str, Any]:
    if layout not in ("side-by-side", "overlay"):
        raise MotionComparisonError("video layout must be side-by-side or overlay")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not rek_video.is_file():
        raise MotionComparisonError(f"reference video does not exist: {rek_video}")
    if not candidate_video.is_file():
        raise MotionComparisonError(f"candidate video does not exist: {candidate_video}")
    anchors_supplied = (rek_video_anchor is not None,
                        candidate_video_anchor is not None)
    manual_edges_supplied = (rek_video_edge_s is not None,
                             candidate_video_edge_s is not None)
    if any(anchors_supplied) and not all(anchors_supplied):
        raise MotionComparisonError(
            "both videos need clock-anchor sidecars for an anchored pair"
        )
    if any(manual_edges_supplied) and not all(manual_edges_supplied):
        raise MotionComparisonError(
            "manual video offsets must be supplied for both videos"
        )
    if any(anchors_supplied) and any(manual_edges_supplied):
        raise MotionComparisonError(
            "clock-anchor sidecars and manual video offsets are mutually exclusive"
        )
    fps = _finite(fps, "video fps")
    if fps <= 0.0:
        raise MotionComparisonError("video fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        return {
            "status": "ffmpeg_unavailable",
            "evidence_grade": "diagnostic_only",
            "supports_parity_verdict": False,
            "verdict_blocker": "video_tools_unavailable",
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "reference_video": {
                "path": str(rek_video.resolve()),
                "sha256": _sha256(rek_video),
            },
            "candidate_video": {
                "path": str(candidate_video.resolve()),
                "sha256": _sha256(candidate_video),
            },
        }
    reference_probe = _probe_video(rek_video, ffprobe)
    candidate_probe = _probe_video(candidate_video, ffprobe)
    reference_rate = _video_frame_rate_hz(reference_probe, "reference")
    candidate_rate = _video_frame_rate_hz(candidate_probe, "candidate")
    reference_anchor = None
    candidate_anchor = None
    if all(anchors_supplied):
        assert rek_video_anchor is not None
        assert candidate_video_anchor is not None
        reference_anchor = _load_video_clock_anchor(
            rek_video_anchor, video_path=rek_video, trace=reference,
            source_frame_rate_hz=reference_rate,
        )
        candidate_anchor = _load_video_clock_anchor(
            candidate_video_anchor, video_path=candidate_video, trace=candidate,
            source_frame_rate_hz=candidate_rate,
        )
        reference_edge = reference_anchor["command_edge_video_pts_s"]
        candidate_edge = candidate_anchor["command_edge_video_pts_s"]
        evidence_grade = "acceptance"
        supports_parity_verdict = True
        verdict_blocker = None
        alignment_basis = "measured_video_clock_anchors"
    else:
        reference_edge = (
            _finite(rek_video_edge_s, "rek video edge")
            if rek_video_edge_s is not None else reference.video_edge_time_s
        )
        candidate_edge = (
            _finite(candidate_video_edge_s, "candidate video edge")
            if candidate_video_edge_s is not None else candidate.video_edge_time_s
        )
        if reference_edge is None or candidate_edge is None:
            raise MotionComparisonError(
                "diagnostic video needs two manual offsets or trace video metadata; "
                "acceptance video needs two measured clock-anchor sidecars"
            )
        evidence_grade = "diagnostic_only"
        supports_parity_verdict = False
        verdict_blocker = "video_clock_anchor_not_measured"
        alignment_basis = (
            "manual_video_offsets" if all(manual_edges_supplied)
            else "unverified_trace_video_metadata"
        )
    if reference_edge < 0.0 or candidate_edge < 0.0:
        raise MotionComparisonError("video command-edge PTS cannot be negative")
    reference_video_identity = {
        "path": str(rek_video.resolve()),
        "sha256": _sha256(rek_video),
        "command_edge_video_pts_s": reference_edge,
        "clock_anchor": reference_anchor,
    }
    candidate_video_identity = {
        "path": str(candidate_video.resolve()),
        "sha256": _sha256(candidate_video),
        "command_edge_video_pts_s": candidate_edge,
        "clock_anchor": candidate_anchor,
    }
    start = report["alignment"]["start_relative_time_s"]
    end = report["alignment"]["end_relative_time_s"]
    duration = end - start + 1.0 / fps
    video_frame_count = int(math.floor((end - start) * fps + EPSILON)) + 1
    reference_start = reference_edge + start
    candidate_start = candidate_edge + start
    if reference_start < 0.0 or candidate_start < 0.0:
        raise MotionComparisonError("aligned video window starts before its file")
    for label, start_time, probe in (
        ("reference", reference_start, reference_probe),
        ("candidate", candidate_start, candidate_probe),
    ):
        measured_duration = probe["duration_s"]
        if (measured_duration is not None
                and start_time + duration > measured_duration + 1.0 / fps):
            raise MotionComparisonError(
                f"{label} video does not cover the aligned comparison window"
            )
    target_height = min(reference_probe["height"], candidate_probe["height"])
    target_height -= target_height % 2
    common = (
        f"trim=start={{start:.12g}}:duration={duration:.12g},"
        f"setpts=PTS-STARTPTS,fps={fps:.12g},setsar=1"
    )
    if layout == "side-by-side":
        left = common.format(start=reference_start) + f",scale=-2:{target_height}[r]"
        right = common.format(start=candidate_start) + f",scale=-2:{target_height}[c]"
        filter_complex = f"[0:v]{left};[1:v]{right};[r][c]hstack=inputs=2[v]"
    else:
        if (reference_probe["width"], reference_probe["height"]) != (
                candidate_probe["width"], candidate_probe["height"]):
            raise MotionComparisonError(
                "overlay video requires identical source dimensions; refusing "
                "a visual scaling fit"
            )
        width = reference_probe["width"] - reference_probe["width"] % 2
        height = reference_probe["height"] - reference_probe["height"] % 2
        left = common.format(start=reference_start) + f",scale={width}:{height}[r]"
        right = common.format(start=candidate_start) + f",scale={width}:{height}[c]"
        filter_complex = (
            f"[0:v]{left};[1:v]{right};"
            "[r][c]blend=all_expr=A*0.5+B*0.5[v]"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary path {temporary}")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(rek_video), "-i", str(candidate_video),
        "-filter_complex", filter_complex, "-map", "[v]", "-an",
        "-frames:v", str(video_frame_count),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-threads", "1", "-map_metadata", "-1",
        str(temporary),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        if temporary.exists():
            temporary.unlink()
        raise MotionComparisonError(
            f"ffmpeg failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        rendered_probe = _probe_video(temporary, ffprobe, count_frames=True)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    if rendered_probe["frame_count"] != video_frame_count:
        temporary.unlink()
        raise MotionComparisonError(
            "rendered video frame count differs from the aligned timebase: "
            f"expected {video_frame_count}, got {rendered_probe['frame_count']}"
        )
    try:
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    version = subprocess.run(
        [ffmpeg, "-version"], check=False, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return {
        "status": "rendered",
        "evidence_grade": evidence_grade,
        "supports_parity_verdict": supports_parity_verdict,
        "verdict_blocker": verdict_blocker,
        "alignment_basis": alignment_basis,
        "layout": layout,
        "fps": fps,
        "frame_count": video_frame_count,
        "relative_start_s": start,
        "relative_end_s": end,
        "reference_video": {**reference_video_identity, "probe": reference_probe},
        "candidate_video": {**candidate_video_identity, "probe": candidate_probe},
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
            "probe": rendered_probe,
        },
        "ffmpeg_version": version,
        "command": command,
        "stderr": completed.stderr,
    }


def _write_json_exclusive(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary path {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _apply_video_verdict_gate(report: dict[str, Any]) -> None:
    video = report.get("video")
    if not isinstance(video, dict):
        return
    if video.get("supports_parity_verdict") is True:
        return
    blocker = video.get("verdict_blocker") or "video_not_acceptance_grade"
    blockers = report["verdict"]["blockers"]
    if blocker not in blockers:
        blockers.append(blocker)
    if report["verdict"]["passed"] is True:
        report["verdict"]["passed"] = None
        report["verdict"]["state"] = "insufficient_evidence"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rek", type=Path, required=True,
                        help="primary measured REK motion")
    parser.add_argument("--rek-repeat", type=Path, action="append", default=[],
                        help="another measured execution of the exact command")
    parser.add_argument("--sim", type=Path, required=True,
                        help="candidate simulator motion")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rek-calibration", type=Path)
    parser.add_argument("--sim-calibration", type=Path)
    parser.add_argument("--rek-window", type=int)
    parser.add_argument("--sim-window", type=int)
    parser.add_argument("--rek-pilot", default="0")
    parser.add_argument("--sim-pilot", default="0")
    parser.add_argument("--edge-event", default="command_edge")
    parser.add_argument("--accept-at", choices=tuple(QUANTILES), default="p99")
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--fps", type=float,
                        help="uniform comparison timebase; default uses REK frames")
    parser.add_argument(
        "--max-timestamp-uncertainty-s", type=float, required=True,
        help=("largest allowed distance from a grid point to a distinct measured "
              "sample; interpolation is never used"),
    )
    parser.add_argument("--require-screen", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--rek-video", type=Path)
    parser.add_argument("--sim-video", type=Path)
    parser.add_argument("--video-out", type=Path)
    parser.add_argument("--video-layout", choices=("side-by-side", "overlay"),
                        default="side-by-side")
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument("--rek-video-anchor", type=Path)
    parser.add_argument("--sim-video-anchor", type=Path)
    parser.add_argument("--rek-video-edge-s", type=float)
    parser.add_argument("--sim-video-edge-s", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reference = load_motion_trace(
        args.rek, calibration_path=args.rek_calibration,
        window_index=args.rek_window, pilot=args.rek_pilot,
        edge_event=args.edge_event,
    )
    repeats = [
        load_motion_trace(
            path, calibration_path=args.rek_calibration,
            window_index=args.rek_window, pilot=args.rek_pilot,
            edge_event=args.edge_event,
        )
        for path in args.rek_repeat
    ]
    candidate = load_motion_trace(
        args.sim, calibration_path=args.sim_calibration,
        window_index=args.sim_window, pilot=args.sim_pilot,
        edge_event=args.edge_event,
    )
    report = compare_motion(
        reference, repeats, candidate, accept_at=args.accept_at,
        require_screen=args.require_screen, start_s=args.start_s,
        end_s=args.end_s, fps=args.fps,
        maximum_timestamp_uncertainty_s=args.max_timestamp_uncertainty_s,
    )
    video_arguments = (args.rek_video, args.sim_video, args.video_out)
    video_related = (*video_arguments, args.rek_video_anchor,
                     args.sim_video_anchor, args.rek_video_edge_s,
                     args.sim_video_edge_s)
    if any(value is not None for value in video_related):
        if not all(value is not None for value in video_arguments):
            raise MotionComparisonError(
                "--rek-video, --sim-video, and --video-out are one atomic request"
            )
        report["video"] = render_aligned_video(
            report, reference, candidate,
            rek_video=args.rek_video.resolve(),
            candidate_video=args.sim_video.resolve(),
            output=args.video_out.resolve(),
            layout=args.video_layout,
            fps=args.video_fps,
            rek_video_anchor=(args.rek_video_anchor.resolve()
                              if args.rek_video_anchor else None),
            candidate_video_anchor=(args.sim_video_anchor.resolve()
                                    if args.sim_video_anchor else None),
            rek_video_edge_s=args.rek_video_edge_s,
            candidate_video_edge_s=args.sim_video_edge_s,
        )
        _apply_video_verdict_gate(report)
    _write_json_exclusive(args.out.resolve(), report)
    print(f"command: {report['command_identity']}")
    print(f"rek runs: {report['rek_run_count']}")
    print(f"frames: {report['alignment']['frame_count']}")
    print(f"verdict: {report['verdict']['state']}")
    if report["verdict"]["blockers"]:
        print(f"blockers: {json.dumps(report['verdict']['blockers'])}")
    for name, metric in report["metrics"].items():
        print(
            f"{name}: max={metric['error']['max']} {metric['unit']}, "
            f"failed_frames={metric['failed_frame_count']}"
        )
    print(f"report: {args.out.resolve()}")
    print(f"report sha256: {_sha256(args.out.resolve())}")
    if report["video"] is not None and report["video"]["status"] != "rendered":
        return 2
    if report["verdict"]["passed"] is True:
        return 0
    if report["verdict"]["passed"] is False:
        return 1
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MotionComparisonError, FileExistsError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
