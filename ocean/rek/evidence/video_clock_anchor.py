#!/usr/bin/env python3
"""Bind a rendered command marker to a published capture's video timeline.

``spark_x11_capture.py`` cannot observe the command edge inside the separate
Wine process.  Its capture is therefore diagnostic by itself.  This helper
creates an acceptance-grade ``rek.video_clock_anchor.v1`` sidecar only when it
can machine-detect one exact, persistent marker transition in the sealed PNG
sequence and verify the marker producer, command trace, capture manifest,
frames, and encoded video by SHA-256.

The marker contract must use this shape::

    {
      "schema": "rek.rendered_command_marker.v1",
      "command_identity": "walk_forward:press:v1",
      "schedule_run_id": "...",
      "trace": {"path": "trace.jsonl", "sha256": "..."},
      "producer": {
        "path": "RekUiBridgeAgent.dll",
        "sha256": "...",
        "render_binding":
          "first_post_marker_frame_is_first_rendered_frame_after_command_edge"
      },
      "marker": {
        "transition": "persistent_exact_rgb_rising_edge",
        "region_px": {"x": 0, "y": 0, "width": 8, "height": 8},
        "pre_rgb": [0, 0, 0],
        "post_rgb": [255, 0, 255]
      }
    }

Paths in the contract are resolved relative to the contract file.  The
producer must implement the stated render binding.  A human-selected frame,
manual time shift, lossy-color match, absent transition, or transition whose
measured interval exceeds one source-video period fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Sequence


ANCHOR_SCHEMA = "rek.video_clock_anchor.v1"
CAPTURE_SCHEMA = "rek.spark_x11_frame_capture.v2"
SIMULATOR_CAPTURE_SCHEMA = "rek.simulator_frame_capture.v1"
SUPPORTED_CAPTURE_SCHEMAS = {CAPTURE_SCHEMA, SIMULATOR_CAPTURE_SCHEMA}
CONTRACT_SCHEMA = "rek.rendered_command_marker.v1"
RENDER_BINDING = (
    "first_post_marker_frame_is_first_rendered_frame_after_command_edge"
)
TRANSITION = "persistent_exact_rgb_rising_edge"


class AnchorError(RuntimeError):
    """The supplied evidence cannot support a video clock anchor."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnchorError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnchorError(f"{label} must be a JSON object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\0"):
        raise AnchorError(f"{label} must be a non-empty single-line string")
    return value


def _sha_string(value: Any, label: str) -> str:
    text = _nonempty_string(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AnchorError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnchorError(f"{label} must be an integer >= {minimum}")
    return value


def _resolve_file(base: Path, raw_path: Any, label: str) -> Path:
    text = _nonempty_string(raw_path, f"{label}.path")
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    if path.is_symlink():
        raise AnchorError(f"{label} must not be a symlink: {path}")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise AnchorError(f"{label} file is unavailable: {path}") from exc
    if not path.is_file():
        raise AnchorError(f"{label} must be a regular file: {path}")
    return path


def _verified_contract_artifact(
    contract_path: Path, value: Any, label: str
) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise AnchorError(f"{label} must be an object")
    path = _resolve_file(contract_path.parent, value.get("path"), label)
    expected = _sha_string(value.get("sha256"), f"{label}.sha256")
    observed = _sha256(path)
    if observed != expected:
        raise AnchorError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return path, observed


def _capture_file(capture_dir: Path, raw_path: Any, label: str) -> Path:
    text = _nonempty_string(raw_path, label)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise AnchorError(f"{label} must stay inside the capture directory")
    root = capture_dir.resolve(strict=True)
    candidate = root / relative
    if candidate.is_symlink():
        raise AnchorError(f"{label} must not be a symlink")
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise AnchorError(f"capture artifact is unavailable: {text}") from exc
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AnchorError(f"{label} escapes the capture directory") from exc
    if not path.is_file():
        raise AnchorError(f"{label} is not a regular file")
    return path


def _rgb(value: Any, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(channel, bool) or not isinstance(channel, int) for channel in value)
        or any(channel < 0 or channel > 255 for channel in value)
    ):
        raise AnchorError(f"{label} must be three integer RGB channels in 0..255")
    return value[0], value[1], value[2]


def _classify_marker(
    frame_path: Path,
    region: tuple[int, int, int, int],
    pre_rgb: tuple[int, int, int],
    post_rgb: tuple[int, int, int],
) -> str:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise AnchorError("Pillow is required for rendered-marker detection") from exc
    x, y, width, height = region
    try:
        with Image.open(frame_path) as image:
            image.load()
            if x + width > image.width or y + height > image.height:
                raise AnchorError(
                    f"marker region exceeds frame dimensions in {frame_path.name}"
                )
            pixels = image.convert("RGB").crop(
                (x, y, x + width, y + height)
            ).tobytes()
    except AnchorError:
        raise
    except Exception as exc:
        raise AnchorError(f"cannot decode {frame_path.name}: {exc}") from exc
    if pixels == bytes(pre_rgb) * (width * height):
        return "pre"
    if pixels == bytes(post_rgb) * (width * height):
        return "post"
    raise AnchorError(
        f"marker region in {frame_path.name} is neither exact pre-RGB nor exact post-RGB"
    )


def _frame_set_sha256(records: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["index"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["timestamp_ns"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication provides no-replace semantics on both target hosts.
        os.link(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_manifest_binding(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    *,
    contract_sha256: str,
    trace_sha256: str,
    producer_sha256: str,
    command_identity: str,
    schedule_run_id: str,
) -> dict[str, Any]:
    binding = manifest.get("binding")
    if not isinstance(binding, dict):
        raise AnchorError("capture manifest has no marker/trace/run binding")
    expected = {
        "marker_contract_schema": CONTRACT_SCHEMA,
        "marker_contract_sha256": contract_sha256,
        "trace_sha256": trace_sha256,
        "producer_sha256": producer_sha256,
        "capture_id": manifest.get("capture_id"),
        "schedule_run_id": schedule_run_id,
        "command_identity": command_identity,
        "render_binding": RENDER_BINDING,
    }
    differences = {
        key: {"expected": value, "observed": binding.get(key)}
        for key, value in expected.items()
        if binding.get(key) != value
    }
    if differences:
        raise AnchorError(
            "capture manifest binding differs from the supplied marker contract: "
            + json.dumps(differences, sort_keys=True)
        )
    producer = contract.get("producer")
    if not isinstance(producer, dict) or producer.get("render_binding") != (
        binding.get("render_binding")
    ):
        raise AnchorError("capture binding and producer render binding differ")
    return {key: binding[key] for key in expected}


def _validate_simulator_frame_binding(
    trace_path: Path,
    manifest: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    states: Sequence[str],
    *,
    command_identity: str,
    schedule_run_id: str,
) -> dict[str, Any]:
    trace = _load_object(trace_path, "simulator motion trace")
    if trace.get("schema") != "rek.paired_motion_trace.v1":
        raise AnchorError("simulator marker trace has the wrong schema")
    if trace.get("source") != "clone:rek_fight_engineai":
        raise AnchorError("simulator marker trace has the wrong source")
    if trace.get("capture_id") != manifest.get("capture_id"):
        raise AnchorError("simulator trace and capture IDs differ")
    if trace.get("schedule_run_id") != schedule_run_id:
        raise AnchorError("simulator trace and marker schedule run IDs differ")
    command = trace.get("command")
    timing = trace.get("timing")
    samples = trace.get("samples")
    request = manifest.get("request")
    binding = manifest.get("binding")
    if not all(isinstance(item, dict) for item in (command, timing, request, binding)):
        raise AnchorError("simulator trace/capture timing records are incomplete")
    if not isinstance(samples, list) or len(samples) < 2:
        raise AnchorError("simulator marker trace has fewer than two samples")
    assert isinstance(command, dict)
    assert isinstance(timing, dict)
    assert isinstance(request, dict)
    assert isinstance(binding, dict)
    if (
        command.get("identity") != command_identity
        or command.get("execution_state") != "simulated"
    ):
        raise AnchorError("simulator trace command identity or execution state differs")
    source_rate = request.get("source_physics_rate_hz")
    stride = request.get("frame_selection_stride_physics_ticks")
    if (
        isinstance(source_rate, bool)
        or not isinstance(source_rate, int)
        or source_rate <= 0
        or isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride <= 0
        or timing.get("physics_rate_hz") != source_rate
        or source_rate * request["fps_denominator"]
        != request["fps_numerator"] * stride
    ):
        raise AnchorError("simulator video rate is not an exact physics-tick stride")
    if len(frames) != (len(samples) - 1) // stride + 1:
        raise AnchorError("simulator frames do not cover the complete trace stride")
    first_tick = samples[0].get("physics_tick") if isinstance(samples[0], dict) else None
    edge_tick = command.get("edge_physics_tick")
    if (
        isinstance(first_tick, bool)
        or not isinstance(first_tick, int)
        or isinstance(edge_tick, bool)
        or not isinstance(edge_tick, int)
        or edge_tick < first_tick
        or (edge_tick - first_tick) % stride
    ):
        raise AnchorError("simulator command edge is not on the captured frame grid")
    edge_frame = (edge_tick - first_tick) // stride
    first_post = edge_frame + 1
    if (
        binding.get("command_edge_physics_tick") != edge_tick
        or binding.get("command_edge_frame_index") != edge_frame
        or binding.get("first_post_marker_frame_index") != first_post
    ):
        raise AnchorError("simulator capture edge/frame binding differs from its trace")
    for index, (frame, state) in enumerate(zip(frames, states)):
        trace_index = index * stride
        sample = samples[trace_index]
        if not isinstance(sample, dict):
            raise AnchorError("simulator trace sample is not an object")
        physics_tick = sample.get("physics_tick")
        time_s = sample.get("time_s")
        expected_timestamp_ns = (
            index * 1_000_000_000 * request["fps_denominator"]
        ) // request["fps_numerator"]
        if (
            frame.get("trace_sample_index") != trace_index
            or frame.get("simulation_physics_tick") != physics_tick
            or isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or frame.get("simulation_time_ns") != round(float(time_s) * 1e9)
            or frame.get("timestamp_ns") != expected_timestamp_ns
            or frame.get("marker_state") != state
        ):
            raise AnchorError(
                f"simulator frame {index} is not bound to its exact trace sample"
            )
    return {
        "trace_sample_stride": stride,
        "source_physics_rate_hz": source_rate,
        "command_edge_physics_tick": edge_tick,
        "command_edge_frame_index": edge_frame,
        "first_post_marker_frame_index": first_post,
        "frame_count": len(frames),
    }


def create_anchor(
    capture_dir: Path, contract_path: Path, output_path: Path
) -> dict[str, Any]:
    capture_candidate = capture_dir.expanduser().absolute()
    contract_candidate = contract_path.expanduser().absolute()
    if capture_candidate.is_symlink():
        raise AnchorError(f"capture path must not be a symlink: {capture_candidate}")
    if contract_candidate.is_symlink():
        raise AnchorError(f"marker contract must not be a symlink: {contract_candidate}")
    capture_dir = capture_candidate.resolve(strict=True)
    contract_path = contract_candidate.resolve(strict=True)
    if not capture_dir.is_dir():
        raise AnchorError(f"capture path is not a directory: {capture_dir}")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise AnchorError(f"marker contract is not a regular file: {contract_path}")

    manifest_path = capture_dir / "capture.manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AnchorError("capture.manifest.json is missing or is a symlink")
    manifest = _load_object(manifest_path, "capture manifest")
    capture_schema = manifest.get("schema")
    if capture_schema not in SUPPORTED_CAPTURE_SCHEMAS or manifest.get("status") != "complete":
        raise AnchorError(
            "capture must be a complete supported sealed-frame artifact; "
            f"observed schema {capture_schema!r}"
        )
    contract = _load_object(contract_path, "marker contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise AnchorError(f"marker contract schema must be {CONTRACT_SCHEMA}")

    command_identity = _nonempty_string(
        contract.get("command_identity"), "command_identity"
    )
    schedule_run_id = _nonempty_string(
        contract.get("schedule_run_id"), "schedule_run_id"
    )
    trace_path, trace_sha256 = _verified_contract_artifact(
        contract_path, contract.get("trace"), "trace"
    )
    producer_value = contract.get("producer")
    producer_path, producer_sha256 = _verified_contract_artifact(
        contract_path, producer_value, "producer"
    )
    assert isinstance(producer_value, dict)
    if producer_value.get("render_binding") != RENDER_BINDING:
        raise AnchorError(
            "producer.render_binding does not assert the required same-render edge binding"
        )
    contract_sha256 = _sha256(contract_path)
    manifest_binding = _validate_manifest_binding(
        manifest,
        contract,
        contract_sha256=contract_sha256,
        trace_sha256=trace_sha256,
        producer_sha256=producer_sha256,
        command_identity=command_identity,
        schedule_run_id=schedule_run_id,
    )

    marker = contract.get("marker")
    if not isinstance(marker, dict) or marker.get("transition") != TRANSITION:
        raise AnchorError(f"marker.transition must be {TRANSITION}")
    region_value = marker.get("region_px")
    if not isinstance(region_value, dict):
        raise AnchorError("marker.region_px must be an object")
    region = (
        _integer(region_value.get("x"), "marker.region_px.x"),
        _integer(region_value.get("y"), "marker.region_px.y"),
        _integer(region_value.get("width"), "marker.region_px.width", minimum=1),
        _integer(region_value.get("height"), "marker.region_px.height", minimum=1),
    )
    pre_rgb = _rgb(marker.get("pre_rgb"), "marker.pre_rgb")
    post_rgb = _rgb(marker.get("post_rgb"), "marker.post_rgb")
    if pre_rgb == post_rgb:
        raise AnchorError("marker pre-RGB and post-RGB must differ")

    request = manifest.get("request")
    if not isinstance(request, dict):
        raise AnchorError("capture manifest request is missing")
    fps_numerator = _integer(request.get("fps_numerator"), "fps_numerator", minimum=1)
    fps_denominator = _integer(request.get("fps_denominator"), "fps_denominator", minimum=1)
    expected_count = _integer(
        request.get("expected_frame_count"), "expected_frame_count", minimum=2
    )
    if request.get("synthetic_frame_duplication") is not False:
        raise AnchorError("capture must explicitly forbid synthetic frame duplication")
    if capture_schema == SIMULATOR_CAPTURE_SCHEMA:
        if request.get("sample_interpolation_used") is not False:
            raise AnchorError("simulator capture must explicitly forbid interpolation")
        if request.get("frame_selection") != (
            "exact_measured_physics_tick_modulo_stride"
        ):
            raise AnchorError("simulator capture frame selection is not exact measured state")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_count:
        raise AnchorError("capture manifest does not contain the expected frames")
    capture_result = manifest.get("result")
    if (
        not isinstance(capture_result, dict)
        or capture_result.get("actual_frame_count") != expected_count
    ):
        raise AnchorError("capture result frame count is not exact")

    states: list[str] = []
    prior_timestamp: int | None = None
    for expected_index, record in enumerate(frames):
        if not isinstance(record, dict) or record.get("index") != expected_index:
            raise AnchorError("capture frames are not contiguous from index zero")
        frame_path = _capture_file(capture_dir, record.get("path"), "frame.path")
        expected_frame_sha = _sha_string(record.get("sha256"), "frame.sha256")
        observed_frame_sha = _sha256(frame_path)
        if observed_frame_sha != expected_frame_sha:
            raise AnchorError(f"captured frame {expected_index} SHA-256 mismatch")
        timestamp_ns = _integer(record.get("timestamp_ns"), "frame.timestamp_ns")
        if prior_timestamp is not None and timestamp_ns <= prior_timestamp:
            raise AnchorError("frame timestamps are not strictly increasing")
        prior_timestamp = timestamp_ns
        states.append(_classify_marker(frame_path, region, pre_rgb, post_rgb))

    if states[0] != "pre" or states[-1] != "post":
        raise AnchorError("marker sequence must start pre and end post")
    first_post = states.index("post")
    if first_post == 0 or any(state != "pre" for state in states[:first_post]):
        raise AnchorError("marker sequence has no unique exact rising edge")
    if any(state != "post" for state in states[first_post:]):
        raise AnchorError("marker reverted after its rising edge")
    simulator_frame_binding = None
    if capture_schema == SIMULATOR_CAPTURE_SCHEMA:
        simulator_frame_binding = _validate_simulator_frame_binding(
            trace_path,
            manifest,
            frames,
            states,
            command_identity=command_identity,
            schedule_run_id=schedule_run_id,
        )

    encoding = manifest.get("encoding")
    artifacts = manifest.get("artifacts")
    if not isinstance(encoding, dict) or not isinstance(artifacts, dict):
        raise AnchorError("capture has no encoded video")
    encoded_artifact = artifacts.get("encoded_video")
    if not isinstance(encoded_artifact, dict):
        raise AnchorError("capture has no encoded_video artifact")
    video_path = _capture_file(capture_dir, encoded_artifact.get("path"), "video.path")
    expected_video_sha = _sha_string(
        encoded_artifact.get("sha256"), "video.sha256"
    )
    video_sha256 = _sha256(video_path)
    if video_sha256 != expected_video_sha:
        raise AnchorError("published video SHA-256 does not match its capture manifest")
    result = manifest.get("result")
    if not isinstance(result, dict) or result.get("encoded_video_sha256") != video_sha256:
        raise AnchorError("capture result does not bind the encoded video SHA-256")

    prior = frames[first_post - 1]
    current = frames[first_post]
    prior_source_pts_ns = _integer(prior.get("timestamp_ns"), "prior timestamp")
    current_source_pts_ns = _integer(current.get("timestamp_ns"), "current timestamp")
    measured_source_gap_ns = current_source_pts_ns - prior_source_pts_ns
    measured_source_gap_s = measured_source_gap_ns / 1e9
    nominal_period_s = fps_denominator / fps_numerator
    # The encoded image sequence has a known constant-rate timeline.  The
    # rendered edge is bounded by two adjacent encoded frames, so its estimate
    # is their midpoint and its video-PTS uncertainty is exactly half that
    # encoded interval.  Source PTS uses integer nanoseconds; allow only the
    # mathematical ceiling of one rational source period to prove there was no
    # longer acquisition gap across the marker.
    maximum_source_gap_ns = math.ceil(
        1_000_000_000 * fps_denominator / fps_numerator
    )
    if measured_source_gap_ns > maximum_source_gap_ns:
        raise AnchorError(
            "measured marker bracket exceeds one encoded frame period; "
            "cannot meet the half-frame uncertainty gate"
        )
    half_encoded_period_s = nominal_period_s / 2.0
    uncertainty_s = half_encoded_period_s
    prior_video_pts_s = (first_post - 1) * nominal_period_s
    current_video_pts_s = first_post * nominal_period_s
    command_edge_video_pts_s = (prior_video_pts_s + current_video_pts_s) / 2.0

    capture_id = _nonempty_string(manifest.get("capture_id"), "capture_id")
    manifest_sha256 = _sha256(manifest_path)
    frame_set_sha256 = _sha_string(
        result.get("frame_set_sha256"), "frame_set_sha256"
    )
    observed_frame_set_sha256 = _frame_set_sha256(frames)
    if observed_frame_set_sha256 != frame_set_sha256:
        raise AnchorError("capture frame-set SHA-256 does not match its manifest")
    provenance = (
        f"machine-detected {TRANSITION} in sealed capture {capture_id}; "
        f"capture_manifest_sha256={manifest_sha256}; "
        f"marker_contract_sha256={contract_sha256}; "
        f"producer_sha256={producer_sha256}"
    )
    anchor: dict[str, Any] = {
        "schema": ANCHOR_SCHEMA,
        "video_sha256": video_sha256,
        "trace_sha256": trace_sha256,
        "command_identity": command_identity,
        "command_edge_video_pts_s": command_edge_video_pts_s,
        "run_id": schedule_run_id,
        "measurement": {
            "state": "measured",
            "method": "rendered_command_marker_transition_v1",
            "uncertainty_s": uncertainty_s,
            "provenance": provenance,
        },
        "schedule_run_id": schedule_run_id,
        "source_capture": {
            "capture_id": capture_id,
            "capture_schema": capture_schema,
            "capture_manifest_sha256": manifest_sha256,
            "frame_set_sha256": frame_set_sha256,
            "video_path": video_path.relative_to(capture_dir).as_posix(),
            "fps_numerator": fps_numerator,
            "fps_denominator": fps_denominator,
        },
        "marker_observation": {
            "contract_schema": CONTRACT_SCHEMA,
            "contract_sha256": contract_sha256,
            "transition": TRANSITION,
            "region_px": {
                "x": region[0],
                "y": region[1],
                "width": region[2],
                "height": region[3],
            },
            "pre_rgb": list(pre_rgb),
            "post_rgb": list(post_rgb),
            "last_pre_frame_index": first_post - 1,
            "first_post_frame_index": first_post,
            "last_pre_source_pts_ns": prior_source_pts_ns,
            "first_post_source_pts_ns": current_source_pts_ns,
            "measured_source_bracket_s": measured_source_gap_s,
            "maximum_allowed_source_bracket_ns": maximum_source_gap_ns,
            "last_pre_video_pts_s": prior_video_pts_s,
            "first_post_video_pts_s": current_video_pts_s,
        },
        "capture_binding": manifest_binding,
        "simulator_frame_binding": simulator_frame_binding,
        "verified_artifacts": {
            "trace": {
                "path": str(trace_path),
                "sha256": trace_sha256,
            },
            "marker_producer": {
                "path": str(producer_path),
                "sha256": producer_sha256,
                "render_binding": RENDER_BINDING,
            },
            "published_video": {
                "path": str(video_path),
                "sha256": video_sha256,
            },
        },
    }
    _atomic_json(output_path.expanduser().absolute(), anchor)
    return anchor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--marker-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        anchor = create_anchor(
            arguments.capture, arguments.marker_contract, arguments.out
        )
    except (AnchorError, FileExistsError, OSError) as exc:
        print(f"video clock anchor failed: {exc}", file=os.sys.stderr)
        return 1
    print("video clock anchor: measured")
    print(f"command: {anchor['command_identity']}")
    print(f"edge video PTS: {anchor['command_edge_video_pts_s']:.9f} s")
    print(f"uncertainty: {anchor['measurement']['uncertainty_s']:.9f} s")
    print(f"sidecar: {arguments.out.expanduser().absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
