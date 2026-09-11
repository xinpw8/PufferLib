#!/usr/bin/env python3
"""Render one isolated single-press simulator trace and sealed 50 fps video.

The trace and video are produced during the same ``EngineAIMotionRunner``
pass.  Video frames are selected only when a measured 500 Hz physics sample
lands on the 50 Hz grid.  No interpolation, duplicate-frame synthesis, or
trajectory fitting is used.

An exact 8 by 8 pixel marker is black through the command-edge frame and
magenta in every subsequent frame.  The first magenta frame is therefore the
first selected render strictly after the command edge.  The capture emits the
machine-readable contract consumed by ``video_clock_anchor.py``.  It does not
make a motion-parity claim.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import export_motion_trace as motion


CAPTURE_SCHEMA = "rek.simulator_frame_capture.v1"
MARKER_CONTRACT_SCHEMA = "rek.rendered_command_marker.v1"
VIDEO_CLOCK_ANCHOR_SCHEMA = "rek.video_clock_anchor.v1"
RENDER_BINDING = (
    "first_post_marker_frame_is_first_rendered_frame_after_command_edge"
)
MARKER_TRANSITION = "persistent_exact_rgb_rising_edge"
MARKER_PRE_RGB = (0, 0, 0)
MARKER_POST_RGB = (255, 0, 255)
MARKER_REGION = (0, 0, 8, 8)
VIDEO_FPS = 50
FRAME_STRIDE_PHYSICS_TICKS = 10
FRAME_PERIOD_NS = 20_000_000
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360


class CaptureError(RuntimeError):
    """The simulator capture cannot satisfy its evidence contract."""


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": motion.sha256_file(path),
    }


def _fsync_directory(path: Path) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return False
    try:
        os.fsync(descriptor)
    except OSError:
        return False
    finally:
        os.close(descriptor)
    return True


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def _rename_no_replace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise CaptureError("renameat2 is required for no-replace publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(f"refusing to overwrite {destination}")
            raise OSError(error, os.strerror(error), str(destination))
        return
    if os.name == "nt":
        os.rename(source, destination)
        return
    raise CaptureError("atomic no-replace publication is unsupported on this host")


def _frame_set_sha256(records: Sequence[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["index"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["timestamp_ns"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_png(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CaptureError("Pillow is required for PNG evidence frames") from exc
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CaptureError("renderer must return an HxWx3 uint8 RGB array")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}.png")
    try:
        Image.fromarray(rgb).save(
            temporary,
            format="PNG",
            compress_level=6,
            optimize=False,
        )
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DeterministicMujocoRenderer:
    """Pinned MuJoCo camera and coloring for evidence frames."""

    def __init__(self, runner: motion.EngineAIMotionRunner, width: int, height: int):
        if width < MARKER_REGION[2] or height < MARKER_REGION[3]:
            raise ValueError("render dimensions are smaller than the marker")
        if width % 2 or height % 2:
            raise ValueError("render dimensions must be even for yuv420p encoding")
        self.width = width
        self.height = height
        for geom in range(runner.model.ngeom):
            body = int(runner.model.geom_bodyid[geom])
            if 1 <= body <= 30:
                runner.model.geom_rgba[geom] = (0.10, 0.65, 0.95, 1.0)
            elif 31 <= body <= 60:
                runner.model.geom_rgba[geom] = (0.95, 0.40, 0.12, 1.0)
        self.renderer = runner.mujoco.Renderer(
            runner.model, height=height, width=width
        )
        self.camera = runner.mujoco.MjvCamera()
        runner.mujoco.mjv_defaultCamera(self.camera)
        roots = np.vstack(
            [binding.root_position(runner.data) for binding in runner.bindings]
        )
        lookat = roots.mean(axis=0)
        lookat[2] = max(0.8, lookat[2])
        self.camera.lookat[:] = lookat
        self.camera.distance = 6.2
        self.camera.azimuth = 90.0
        self.camera.elevation = -42.0
        self.camera_contract = {
            "type": "fixed_mujoco_free_camera",
            "lookat_xyz": [float(value) for value in lookat],
            "distance_m": 6.2,
            "azimuth_deg": 90.0,
            "elevation_deg": -42.0,
            "width_px": width,
            "height_px": height,
        }

    def render(self, runner: motion.EngineAIMotionRunner, marker_post: bool) -> np.ndarray:
        self.renderer.update_scene(runner.data, camera=self.camera)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
        color = MARKER_POST_RGB if marker_post else MARKER_PRE_RGB
        x, y, width, height = MARKER_REGION
        rgb[y : y + height, x : x + width] = color
        return rgb

    def close(self) -> None:
        self.renderer.close()


def _encode_video(
    stage: Path,
    frame_count: int,
    width: int,
    height: int,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise CaptureError("ffmpeg and ffprobe are required for sealed MP4 output")
    video = stage / "capture.mp4"
    stdout_path = stage / "encode.stdout.log"
    stderr_path = stage / "encode.stderr.log"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "info",
        "-nostdin",
        "-n",
        "-framerate",
        str(VIDEO_FPS),
        "-start_number",
        "0",
        "-i",
        str(stage / "frames" / "frame-%06d.png"),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        str(video),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=max(60.0, frame_count / VIDEO_FPS * 4.0),
    )
    _atomic_write(stdout_path, completed.stdout)
    _atomic_write(stderr_path, completed.stderr)
    if completed.returncode != 0 or not video.is_file() or not video.stat().st_size:
        raise CaptureError(f"ffmpeg failed with exit {completed.returncode}")
    probe_command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video),
    ]
    probe = subprocess.run(
        probe_command, check=False, capture_output=True, timeout=60.0
    )
    if probe.returncode != 0:
        raise CaptureError(f"ffprobe failed with exit {probe.returncode}")
    try:
        probe_document = json.loads(probe.stdout)
        stream = probe_document["streams"][0]
        probed_count = int(stream["nb_read_frames"])
        probed_width = int(stream["width"])
        probed_height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CaptureError("ffprobe did not return a complete video stream") from exc
    if (probed_count, probed_width, probed_height, stream.get("avg_frame_rate")) != (
        frame_count,
        width,
        height,
        f"{VIDEO_FPS}/1",
    ):
        raise CaptureError("encoded video geometry, rate, or frame count differs")
    return {
        "mode": "ffmpeg_libx264_single_thread_bitexact",
        "command": command,
        "probe_command": probe_command,
        "probe": stream,
        "returncode": completed.returncode,
        "stdout": _artifact(stdout_path, stage),
        "stderr": _artifact(stderr_path, stage),
    }


Encoder = Callable[[Path, int, int, int], dict[str, Any]]


def capture_motion(
    runner: Any,
    renderer: Any,
    command: motion.CommandSpec,
    *,
    output_dir: Path,
    build_fingerprint: str,
    capture_id: str,
    run_id: str,
    artifact_provenance: dict[str, Any],
    calibration: dict[str, Any],
    producer_path: Path,
    encoder: Encoder | None = None,
) -> dict[str, Any]:
    output = output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        frames_dir = stage / "frames"
        producer_dir = stage / "producer"
        frames_dir.mkdir()
        producer_dir.mkdir()
        records: list[dict[str, Any]] = []
        first_physics_tick = (
            motion.TRIAL_NEUTRAL_START_TICK * motion.TRIAL_FIXED_SUBSTEPS
        )

        def observe_frame(
            active_runner: Any,
            sample: dict[str, Any],
            trace_sample_index: int,
            marker_post: bool,
        ) -> None:
            relative_physics_tick = sample["physics_tick"] - first_physics_tick
            if relative_physics_tick % FRAME_STRIDE_PHYSICS_TICKS:
                return
            frame_index = relative_physics_tick // FRAME_STRIDE_PHYSICS_TICKS
            if frame_index != len(records):
                raise CaptureError("selected video frame indices are not contiguous")
            rgb = renderer.render(active_runner, marker_post)
            x, y, marker_width, marker_height = MARKER_REGION
            expected_rgb = MARKER_POST_RGB if marker_post else MARKER_PRE_RGB
            marker_pixels = rgb[
                y : y + marker_height, x : x + marker_width
            ]
            if not np.all(marker_pixels == np.asarray(expected_rgb, dtype=np.uint8)):
                raise CaptureError("renderer did not emit the exact marker RGB")
            frame_path = frames_dir / f"frame-{frame_index:06d}.png"
            _write_png(frame_path, rgb)
            records.append(
                {
                    "index": frame_index,
                    "path": f"frames/{frame_path.name}",
                    "sha256": motion.sha256_file(frame_path),
                    "timestamp_ns": frame_index * FRAME_PERIOD_NS,
                    "duration_ns": FRAME_PERIOD_NS,
                    "simulation_time_ns": sample["physics_tick"] * 2_000_000,
                    "simulation_physics_tick": sample["physics_tick"],
                    "trace_sample_index": trace_sample_index,
                    "marker_state": "post" if marker_post else "pre",
                }
            )

        trace = motion.generate_trace(
            runner,
            command,
            build_fingerprint=build_fingerprint,
            capture_id=capture_id,
            run_id=run_id,
            artifact_provenance=artifact_provenance,
            calibration=calibration,
            sample_observer=observe_frame,
        )
        expected_frame_count = (len(trace["samples"]) - 1) // 10 + 1
        if len(records) != expected_frame_count:
            raise CaptureError(
                f"expected {expected_frame_count} exact-grid frames, got {len(records)}"
            )
        command_edge_frame_index = (
            trace["command"]["edge_physics_tick"] - first_physics_tick
        ) // FRAME_STRIDE_PHYSICS_TICKS
        first_post_frame_index = command_edge_frame_index + 1
        if records[command_edge_frame_index]["marker_state"] != "pre":
            raise CaptureError("command-edge video frame is not pre-marker")
        if records[first_post_frame_index]["marker_state"] != "post":
            raise CaptureError("first post-edge video frame is not post-marker")
        if any(record["marker_state"] != "post" for record in records[first_post_frame_index:]):
            raise CaptureError("rendered command marker is not persistent")

        trace_path = stage / "motion.trace.json"
        motion.write_json_exclusive(trace_path, trace)
        sidecar_path = stage / "frames.jsonl"
        sidecar_payload = b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for record in records
        )
        _atomic_write(sidecar_path, sidecar_payload)

        encoder = encoder or _encode_video
        encoding = encoder(stage, len(records), renderer.width, renderer.height)
        video_path = stage / "capture.mp4"
        if not video_path.is_file() or not video_path.stat().st_size:
            raise CaptureError("encoder did not produce capture.mp4")

        producer_path = producer_path.resolve(strict=True)
        producer_copy = producer_dir / producer_path.name
        producer_payload = producer_path.read_bytes()
        _atomic_write(producer_copy, producer_payload)
        producer_sha256 = motion.sha256_file(producer_path)
        if motion.sha256_file(producer_copy) != producer_sha256:
            raise CaptureError("copied marker producer hash differs from executed source")
        trace_sha256 = motion.sha256_file(trace_path)
        video_sha256 = motion.sha256_file(video_path)

        contract = {
            "schema": MARKER_CONTRACT_SCHEMA,
            "command_identity": command.identity,
            "schedule_run_id": run_id,
            "trace": {"path": "motion.trace.json", "sha256": trace_sha256},
            "producer": {
                "path": f"producer/{producer_copy.name}",
                "sha256": producer_sha256,
                "render_binding": RENDER_BINDING,
            },
            "marker": {
                "transition": MARKER_TRANSITION,
                "region_px": {
                    "x": MARKER_REGION[0],
                    "y": MARKER_REGION[1],
                    "width": MARKER_REGION[2],
                    "height": MARKER_REGION[3],
                },
                "pre_rgb": list(MARKER_PRE_RGB),
                "post_rgb": list(MARKER_POST_RGB),
            },
        }
        contract_path = stage / "marker.contract.json"
        _atomic_json(contract_path, contract)
        contract_sha256 = motion.sha256_file(contract_path)
        frame_set_sha256 = _frame_set_sha256(records)
        manifest = {
            "schema": CAPTURE_SCHEMA,
            "status": "complete",
            "capture_id": capture_id,
            "request": {
                "fps_numerator": VIDEO_FPS,
                "fps_denominator": 1,
                "expected_frame_count": len(records),
                "source_physics_rate_hz": 500,
                "source_controller_rate_hz": 100,
                "frame_selection_stride_physics_ticks": (
                    FRAME_STRIDE_PHYSICS_TICKS
                ),
                "frame_selection": "exact_measured_physics_tick_modulo_stride",
                "sample_interpolation_used": False,
                "synthetic_frame_duplication": False,
            },
            "binding": {
                "marker_contract_schema": MARKER_CONTRACT_SCHEMA,
                "marker_contract_sha256": contract_sha256,
                "trace_sha256": trace_sha256,
                "producer_sha256": producer_sha256,
                "capture_id": capture_id,
                "schedule_run_id": run_id,
                "trial_run_id": run_id,
                "command_identity": command.identity,
                "trial_protocol_identity": motion.TRIAL_PROTOCOL_ID,
                "trial_protocol_sha256": motion.TRIAL_PROTOCOL_SHA256,
                "reset_identity_sha256": trace["reset"]["identity_sha256"],
                "runtime_provenance_sha256": trace["runtime_provenance"][
                    "identity_sha256"
                ],
                "edge_trial_tick": trace["command"]["edge_trial_tick"],
                "command_edge_physics_tick": trace["command"]["edge_physics_tick"],
                "command_edge_frame_index": command_edge_frame_index,
                "first_post_marker_frame_index": first_post_frame_index,
                "render_binding": RENDER_BINDING,
            },
            "result": {
                "actual_frame_count": len(records),
                "width_px": renderer.width,
                "height_px": renderer.height,
                "frame_set_sha256": frame_set_sha256,
                "first_timestamp_ns": records[0]["timestamp_ns"],
                "last_timestamp_ns": records[-1]["timestamp_ns"],
                "encoded_video_sha256": video_sha256,
            },
            "timebase": {
                "frame_timestamp_source": (
                    "exact frame index on the 50 Hz grid selected from the "
                    "measured 500 Hz MuJoCo physics tick"
                ),
                "frame_timestamp_unit": "nanosecond",
                "simulation_time_source": "physics_tick_times_exact_2000000_ns",
                "command_edge_to_video_pts": {
                    "state": "machine_observable_rendered_marker",
                    "required_anchor_schema": VIDEO_CLOCK_ANCHOR_SCHEMA,
                },
            },
            "render": {
                "backend": "mujoco.Renderer",
                "camera": renderer.camera_contract,
                "marker_region_px": contract["marker"]["region_px"],
                "marker_pre_rgb": list(MARKER_PRE_RGB),
                "marker_post_rgb": list(MARKER_POST_RGB),
            },
            "evidence_use": {
                "classification_without_video_clock_anchor": "diagnostic_only",
                "acceptance_bearing_without_video_clock_anchor": False,
                "required_external_anchor_schema": VIDEO_CLOCK_ANCHOR_SCHEMA,
                "marker_contract_path": "marker.contract.json",
                "parity_demonstrated": False,
            },
            "host": {
                "hostname": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "user": getpass.getuser(),
                "python": sys.version,
            },
            "publication": {
                "evidence_directory": (
                    "Linux renameat2(RENAME_NOREPLACE)"
                    if sys.platform.startswith("linux")
                    else "Windows no-replace directory rename"
                ),
                "overwrite_allowed": False,
            },
            "artifacts": {
                "motion_trace": _artifact(trace_path, stage),
                "frame_timestamp_sidecar": _artifact(sidecar_path, stage),
                "marker_contract": _artifact(contract_path, stage),
                "marker_producer": _artifact(producer_copy, stage),
                "encoded_video": _artifact(video_path, stage),
            },
            "encoding": encoding,
            "frames": records,
        }
        manifest_path = stage / "capture.manifest.json"
        _atomic_json(manifest_path, manifest)
        _fsync_directory(frames_dir)
        _fsync_directory(producer_dir)
        _fsync_directory(stage)
        _rename_no_replace(stage, output)
        _fsync_directory(output.parent)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--walking-policy", type=Path, required=True)
    parser.add_argument("--recovery-policy", type=Path, required=True)
    parser.add_argument("--recovery-trajectory", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--command", choices=tuple(motion.COMMANDS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rek-build-fingerprint", required=True)
    parser.add_argument("--capture-id")
    parser.add_argument("--run-id")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = [
        args.model.resolve(),
        args.walking_policy.resolve(),
        args.recovery_policy.resolve(),
        args.recovery_trajectory.resolve(),
        args.calibration.resolve(),
    ]
    provenance = motion.build_artifact_provenance(*paths)
    calibration = json.loads(paths[4].read_text(encoding="utf-8"))
    runner = motion.EngineAIMotionRunner(*paths[:4])
    renderer = DeterministicMujocoRenderer(runner, args.width, args.height)
    capture_id = args.capture_id or f"clone-capture-{uuid.uuid4()}"
    run_id = args.run_id or f"clone-run-{uuid.uuid4()}"
    try:
        manifest = capture_motion(
            runner,
            renderer,
            motion.COMMANDS[args.command],
            output_dir=args.out,
            build_fingerprint=args.rek_build_fingerprint,
            capture_id=capture_id,
            run_id=run_id,
            artifact_provenance=provenance,
            calibration=calibration,
            producer_path=Path(__file__),
        )
    finally:
        renderer.close()
    output = args.out.expanduser().absolute()
    print(f"command: {manifest['binding']['command_identity']}")
    print(f"capture_id: {capture_id}")
    print(f"run_id: {run_id}")
    print(f"frames: {manifest['result']['actual_frame_count']}")
    print(f"trace: {output / 'motion.trace.json'}")
    print(f"video: {output / 'capture.mp4'}")
    print(f"manifest: {output / 'capture.manifest.json'}")
    print(f"marker_contract: {output / 'marker.contract.json'}")
    print("parity_demonstrated: false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CaptureError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
