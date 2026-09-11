#!/usr/bin/env python3
"""Capture a fixed-rate, cursor-free X11 frame sequence as sealed evidence.

The intended source is the isolated REK display on Spark (normally ``:98``).
This program never sends input and never discovers or activates windows.  It
captures the complete X display through GStreamer's ``ximagesrc`` element.

Successful output is published only after all expected frames, per-frame
GStreamer timestamps, PNG dimensions, hashes, and optional video encoding have
validated.  The destination must not already exist.  A failed capture is kept
under a sibling ``.failed-*`` name and is never published at the requested
destination.

The collector cannot observe a command applied inside the separate Wine
process.  A capture is therefore marked diagnostic-only until
``video_clock_anchor.py`` machine-detects a producer-bound rendered command
marker and emits a hash-bound ``rek.video_clock_anchor.v1`` sidecar.

Example on Spark::

    python3 spark_x11_capture.py --out evidence/forward-01-video \
        --duration 5 --fps 60 --display :98
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "rek.spark_x11_frame_capture.v2"
SIDECAR_SCHEMA = "rek.spark_x11_frame_timestamps.v1"
FAILURE_SCHEMA = "rek.spark_x11_frame_capture_failure.v1"
VIDEO_CLOCK_ANCHOR_SCHEMA = "rek.video_clock_anchor.v1"
MARKER_CONTRACT_SCHEMA = "rek.rendered_command_marker.v1"
MARKER_TRANSITION = "persistent_exact_rgb_rising_edge"
RENDER_BINDING = (
    "first_post_marker_frame_is_first_rendered_frame_after_command_edge"
)
UINT64_MAX = (1 << 64) - 1
GST_GRID_TOLERANCE_NS = 2
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FRAME_NAME = re.compile(r"^frame-(\d{6})\.png$")
GST_FRAME_MESSAGE = re.compile(
    r"GstMultiFileSink,\s+filename=\(string\)(?P<filename>.*),\s+"
    r"index=\(int\)(?P<index>\d+),\s+"
    r"timestamp=\(guint64\)(?P<timestamp>\d+),\s+"
    r"stream-time=\(guint64\)(?P<stream_time>\d+),\s+"
    r"running-time=\(guint64\)(?P<running_time>\d+),\s+"
    r"duration=\(guint64\)(?P<duration>\d+)"
)


class CaptureError(RuntimeError):
    """The requested capture could not produce valid evidence."""


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path
    duration_s: Decimal
    fps: int
    display: str = ":98"
    encode: str = "none"
    gst_command_prefix: tuple[str, ...] = ("gst-launch-1.0",)
    ffmpeg: str | None = None
    marker_contract: Path | None = None

    @property
    def expected_frames(self) -> int:
        value = self.duration_s * self.fps
        integral = value.to_integral_value()
        if value != integral:
            raise CaptureError(
                "duration multiplied by fps must be an integer frame count"
            )
        return int(integral)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    start_monotonic_ns: int
    end_monotonic_ns: int
    start_utc: str
    end_utc: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
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


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing target."""
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
        # Windows os.rename does not replace an existing directory.  Spark is
        # Linux; this branch exists to make the validation suite portable.
        os.rename(source, destination)
        return
    raise CaptureError("atomic no-replace publication is unsupported here")


def _validate_config(config: CaptureConfig) -> None:
    if not config.gst_command_prefix or any(
        not isinstance(part, str) or not part for part in config.gst_command_prefix
    ):
        raise CaptureError("gst command prefix must contain non-empty arguments")
    if not config.duration_s.is_finite() or config.duration_s <= 0:
        raise CaptureError("duration must be positive")
    if not 1 <= config.fps <= 240:
        raise CaptureError("fps must be between 1 and 240")
    if not 1 <= config.expected_frames <= 1_000_000:
        raise CaptureError("requested frame count is outside 1..1000000")
    if not config.display or any(char in config.display for char in "\r\n\0"):
        raise CaptureError("display must be a non-empty single-line value")
    if config.encode not in {"none", "auto", "ffmpeg", "opencv"}:
        raise CaptureError("encode must be none, auto, ffmpeg, or opencv")
    if any(char in str(config.output_dir) for char in "\r\n\0"):
        raise CaptureError("output path must be a single-line value")
    if config.marker_contract is None:
        raise CaptureError("a rendered marker contract path is required")
    if any(char in str(config.marker_contract) for char in "\r\n\0"):
        raise CaptureError("marker contract path must be a single-line value")


def _contract_file(base: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise CaptureError(f"marker contract {label} must be an object")
    raw_path = value.get("path")
    expected_sha = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise CaptureError(f"marker contract {label}.path is absent")
    if (
        not isinstance(expected_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
    ):
        raise CaptureError(f"marker contract {label}.sha256 is invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    if path.is_symlink():
        raise CaptureError(f"marker contract {label} must not be a symlink")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise CaptureError(f"marker contract {label} is unavailable") from exc
    if not path.is_file():
        raise CaptureError(f"marker contract {label} is not a regular file")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise CaptureError(f"marker contract {label} SHA-256 mismatch")
    return path, actual_sha


def _verified_marker_contract(path: Path) -> dict[str, Any]:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise CaptureError("marker contract must not be a symlink")
    try:
        contract_path = candidate.resolve(strict=True)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read marker contract {candidate}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("schema") != MARKER_CONTRACT_SCHEMA:
        raise CaptureError(f"marker contract schema must be {MARKER_CONTRACT_SCHEMA}")
    command_identity = contract.get("command_identity")
    schedule_run_id = contract.get("schedule_run_id")
    if not isinstance(command_identity, str) or not command_identity:
        raise CaptureError("marker contract command_identity is absent")
    if not isinstance(schedule_run_id, str) or not schedule_run_id:
        raise CaptureError("marker contract schedule_run_id is absent")
    trace_path, trace_sha = _contract_file(
        contract_path.parent, contract.get("trace"), "trace"
    )
    producer_value = contract.get("producer")
    producer_path, producer_sha = _contract_file(
        contract_path.parent, producer_value, "producer"
    )
    assert isinstance(producer_value, dict)
    if producer_value.get("render_binding") != RENDER_BINDING:
        raise CaptureError("marker producer does not assert the required render binding")
    marker = contract.get("marker")
    if not isinstance(marker, dict) or marker.get("transition") != MARKER_TRANSITION:
        raise CaptureError("marker contract does not require the exact persistent transition")
    return {
        "marker_contract_path": str(contract_path),
        "marker_contract_schema": MARKER_CONTRACT_SCHEMA,
        "marker_contract_sha256": _sha256(contract_path),
        "trace_path": str(trace_path),
        "trace_sha256": trace_sha,
        "producer_path": str(producer_path),
        "producer_sha256": producer_sha,
        "schedule_run_id": schedule_run_id,
        "command_identity": command_identity,
        "render_binding": RENDER_BINDING,
    }


def build_gstreamer_command(config: CaptureConfig, frame_dir: Path) -> list[str]:
    pattern = frame_dir / "frame-%06d.png"
    return [
        *config.gst_command_prefix,
        "-m",
        "-e",
        "ximagesrc",
        f"display-name={config.display}",
        "use-damage=false",
        "show-pointer=false",
        f"num-buffers={config.expected_frames}",
        "!",
        f"video/x-raw,framerate={config.fps}/1",
        "!",
        "queue",
        "max-size-buffers=120",
        "max-size-bytes=0",
        "max-size-time=0",
        "!",
        "videoconvert",
        "!",
        "pngenc",
        "compression-level=1",
        "!",
        "multifilesink",
        "name=framesink",
        f"location={pattern}",
        "post-messages=true",
    ]


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _run_logged(
    argv: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    timeout_s: float,
    environment: dict[str, str] | None = None,
) -> ProcessResult:
    start_utc = _utc_now()
    start_ns = time.monotonic_ns()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            start_new_session=(os.name == "posix"),
        )
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _terminate(process)
            raise CaptureError(
                f"process exceeded {timeout_s:.3f} second timeout"
            ) from exc
        except BaseException:
            _terminate(process)
            raise
        finally:
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    end_ns = time.monotonic_ns()
    return ProcessResult(
        returncode=returncode,
        start_monotonic_ns=start_ns,
        end_monotonic_ns=end_ns,
        start_utc=start_utc,
        end_utc=_utc_now(),
    )


def _version(
    command_prefix: Sequence[str], version_args: Sequence[str] = ("--version",)
) -> str:
    try:
        result = subprocess.run(
            [*command_prefix, *version_args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaptureError(f"cannot query the version of {command_prefix[0]}") from exc
    text = result.stdout.decode("utf-8", errors="replace").strip()
    if result.returncode != 0 or not text:
        raise CaptureError(
            f"version query for {command_prefix[0]} failed with exit {result.returncode}"
        )
    return text


def _decode_gst_string(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        escaped = False
        decoded = []
        controls = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
        for character in value[1:-1]:
            if escaped:
                decoded.append(controls.get(character, character))
                escaped = False
            elif character == "\\":
                escaped = True
            else:
                decoded.append(character)
        if escaped:
            raise CaptureError("GStreamer emitted an incomplete filename escape")
        return "".join(decoded)
    return value


def parse_gstreamer_frame_messages(text: str) -> list[dict[str, Any]]:
    records = []
    for match in GST_FRAME_MESSAGE.finditer(text):
        records.append(
            {
                "filename": _decode_gst_string(match.group("filename")),
                "index": int(match.group("index")),
                "timestamp_ns": int(match.group("timestamp")),
                "stream_time_ns": int(match.group("stream_time")),
                "running_time_ns": int(match.group("running_time")),
                "duration_ns": int(match.group("duration")),
            }
        )
    return records


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != PNG_SIGNATURE
        or header[12:16] != b"IHDR"
    ):
        raise CaptureError(f"{path.name} is not a PNG with an IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise CaptureError(f"{path.name} has invalid PNG dimensions")
    return width, height


def _validate_frames(
    frame_dir: Path,
    messages: list[dict[str, Any]],
    expected_count: int,
    fps: int,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    paths = sorted(frame_dir.glob("frame-*.png"))
    if len(paths) != expected_count:
        raise CaptureError(
            f"expected {expected_count} PNG frames, found {len(paths)}"
        )
    if len(messages) != expected_count:
        raise CaptureError(
            f"expected {expected_count} timestamp messages, found {len(messages)}"
        )
    by_index: dict[int, dict[str, Any]] = {}
    for message in messages:
        index = message["index"]
        if index in by_index:
            raise CaptureError(f"duplicate GStreamer frame index {index}")
        by_index[index] = message
    if sorted(by_index) != list(range(expected_count)):
        raise CaptureError("GStreamer frame indices are not contiguous from zero")

    result = []
    dimensions: tuple[int, int] | None = None
    prior_timestamp = prior_stream = prior_running = None
    for index, path in enumerate(paths):
        match = FRAME_NAME.fullmatch(path.name)
        if match is None or int(match.group(1)) != index:
            raise CaptureError("PNG frame names are not contiguous from zero")
        if path.is_symlink() or not path.is_file():
            raise CaptureError(f"{path.name} is not a regular captured frame")
        message = by_index[index]
        message_path = Path(message["filename"])
        try:
            path_matches = message_path.resolve() == path.resolve()
        except OSError:
            path_matches = False
        if not path_matches:
            raise CaptureError(
                f"timestamp message {index} names a different frame path"
            )
        width, height = _png_dimensions(path)
        if dimensions is None:
            dimensions = (width, height)
        elif dimensions != (width, height):
            raise CaptureError("captured PNG dimensions changed during capture")

        timestamp = message["timestamp_ns"]
        stream_time = message["stream_time_ns"]
        running_time = message["running_time_ns"]
        duration = message["duration_ns"]
        if UINT64_MAX in (timestamp, stream_time, running_time, duration):
            raise CaptureError(f"frame {index} has an unknown GStreamer time")
        if duration <= 0:
            raise CaptureError(f"frame {index} has no positive duration")
        if prior_timestamp is not None and timestamp <= prior_timestamp:
            raise CaptureError("GStreamer frame timestamps are not strictly monotonic")
        if prior_stream is not None and stream_time <= prior_stream:
            raise CaptureError("GStreamer stream times are not strictly monotonic")
        if prior_running is not None and running_time <= prior_running:
            raise CaptureError("GStreamer running times are not strictly monotonic")
        # ximagesrc negotiates a fixed-rate sequence directly.  As a live
        # source, it can acquire within a nominal slot rather than exactly at
        # the slot boundary.  Its measured PTS and shortened duration must
        # remain inside that slot and end on the next boundary.  We preserve
        # those measurements rather than fabricating duplicate frames with
        # videorate.
        slot_start = (index * 1_000_000_000) // fps
        slot_end = ((index + 1) * 1_000_000_000) // fps
        tolerance = GST_GRID_TOLERANCE_NS * fps
        if timestamp * fps < index * 1_000_000_000 - tolerance:
            raise CaptureError(f"frame {index} begins before its fixed-rate slot")
        if timestamp * fps >= (index + 1) * 1_000_000_000 + tolerance:
            raise CaptureError(f"frame {index} begins after its fixed-rate slot")
        if (
            abs((timestamp + duration) * fps - (index + 1) * 1_000_000_000)
            > tolerance
        ):
            raise CaptureError(
                f"frame {index} does not end on its fixed-rate slot boundary"
            )
        prior_timestamp = timestamp
        prior_stream = stream_time
        prior_running = running_time
        result.append(
            {
                "schema": SIDECAR_SCHEMA,
                "index": index,
                "path": f"frames/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "width_px": width,
                "height_px": height,
                "nominal_slot_start_ns": slot_start,
                "nominal_slot_end_ns": slot_end,
                "acquisition_lateness_ns": max(0, timestamp - slot_start),
                **{key: value for key, value in message.items() if key != "filename"},
            }
        )
    assert dimensions is not None
    return result, dimensions


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


def _encode_ffmpeg(
    executable: str,
    stage: Path,
    fps: int,
    frame_count: int,
) -> dict[str, Any]:
    version = _version((executable,), ("-version",))
    video = stage / "capture.mp4"
    if video.exists() or video.is_symlink():
        raise FileExistsError(f"refusing to overwrite {video}")
    argv = [
        executable,
        "-hide_banner",
        "-loglevel",
        "info",
        "-nostdin",
        "-n",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(stage / "frames" / "frame-%06d.png"),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    result = _run_logged(
        argv,
        stage / "encode.stdout.log",
        stage / "encode.stderr.log",
        timeout_s=max(60.0, frame_count / fps * 4.0),
    )
    if result.returncode != 0 or not video.is_file() or video.stat().st_size == 0:
        raise CaptureError(f"ffmpeg failed with exit {result.returncode}")
    return {
        "mode": "ffmpeg",
        "version": version,
        "command": argv,
        "process": result.__dict__,
        "video": _artifact(video, stage),
        "stdout": _artifact(stage / "encode.stdout.log", stage),
        "stderr": _artifact(stage / "encode.stderr.log", stage),
    }


def _encode_opencv(
    stage: Path,
    fps: int,
    records: Sequence[dict[str, Any]],
    dimensions: tuple[int, int],
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise CaptureError("OpenCV encoding requested but cv2 is unavailable") from exc
    width, height = dimensions
    video = stage / "capture.mp4"
    if video.exists() or video.is_symlink():
        raise FileExistsError(f"refusing to overwrite {video}")
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise CaptureError("OpenCV could not open the MP4 writer")
    written = 0
    try:
        for record in records:
            frame = cv2.imread(str(stage / record["path"]), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[1] != width or frame.shape[0] != height:
                raise CaptureError(f"OpenCV could not decode {record['path']}")
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    if written != len(records) or not video.is_file() or video.stat().st_size == 0:
        raise CaptureError("OpenCV did not encode the complete frame sequence")
    operation_log = stage / "encode.opencv.json"
    _atomic_json(
        operation_log,
        {
            "backend": "opencv",
            "opencv_version": cv2.__version__,
            "codec_fourcc": "mp4v",
            "fps": fps,
            "frames_written": written,
            "width_px": width,
            "height_px": height,
        },
    )
    return {
        "mode": "opencv",
        "version": cv2.__version__,
        "codec_fourcc": "mp4v",
        "frames_written": written,
        "video": _artifact(video, stage),
        "operation_log": _artifact(operation_log, stage),
    }


def _encode(
    config: CaptureConfig,
    stage: Path,
    records: Sequence[dict[str, Any]],
    dimensions: tuple[int, int],
) -> dict[str, Any] | None:
    if config.encode == "none":
        return None
    executable = config.ffmpeg or shutil.which("ffmpeg")
    mode = config.encode
    if mode == "auto":
        mode = "ffmpeg" if executable else "opencv"
    if mode == "ffmpeg":
        if not executable:
            raise CaptureError("ffmpeg encoding requested but ffmpeg is unavailable")
        return _encode_ffmpeg(
            executable, stage, config.fps, config.expected_frames
        )
    return _encode_opencv(stage, config.fps, records, dimensions)


def _failure_destination(output: Path) -> Path:
    return output.with_name(f".{output.name}.failed-{uuid.uuid4().hex}")


def capture(config: CaptureConfig) -> dict[str, Any]:
    _validate_config(config)
    output = config.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    failure_path: Path | None = None
    try:
        frame_dir = stage / "frames"
        frame_dir.mkdir()
        gst_version = _version(config.gst_command_prefix)
        argv = build_gstreamer_command(config, frame_dir)
        environment = os.environ.copy()
        environment["DISPLAY"] = config.display
        environment["LC_ALL"] = "C"
        result = _run_logged(
            argv,
            stage / "gst.bus.log",
            stage / "gst.stderr.log",
            timeout_s=float(config.duration_s) + max(
                30.0, float(config.duration_s) * 0.25
            ),
            environment=environment,
        )
        if result.returncode != 0:
            raise CaptureError(
                f"GStreamer capture failed with exit {result.returncode}"
            )
        stdout_text = (stage / "gst.bus.log").read_text(
            encoding="utf-8", errors="replace"
        )
        stderr_text = (stage / "gst.stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )
        bus_text = stdout_text + "\n" + stderr_text
        if "(eos)" not in bus_text and "Got EOS" not in bus_text:
            raise CaptureError("GStreamer did not report end-of-stream")
        messages = parse_gstreamer_frame_messages(bus_text)
        records, dimensions = _validate_frames(
            frame_dir, messages, config.expected_frames, config.fps
        )

        sidecar = stage / "frames.jsonl"
        sidecar_payload = b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ) + b"\n"
            for record in records
        )
        _atomic_write(sidecar, sidecar_payload)
        encoding = _encode(config, stage, records, dimensions)
        assert config.marker_contract is not None
        marker_binding = _verified_marker_contract(config.marker_contract)

        clock = time.get_clock_info("monotonic")
        script_path = Path(__file__).resolve()
        artifacts: dict[str, Any] = {
            "frame_timestamp_sidecar": _artifact(sidecar, stage),
            "gstreamer_bus_log": _artifact(stage / "gst.bus.log", stage),
            "gstreamer_stderr_log": _artifact(stage / "gst.stderr.log", stage),
        }
        if encoding is not None:
            artifacts["encoded_video"] = encoding["video"]
        width, height = dimensions
        capture_id = uuid.uuid4().hex
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "complete",
            "capture_id": capture_id,
            "binding": {
                "marker_contract_path": marker_binding["marker_contract_path"],
                "marker_contract_schema": marker_binding["marker_contract_schema"],
                "marker_contract_sha256": marker_binding["marker_contract_sha256"],
                "trace_path": marker_binding["trace_path"],
                "trace_sha256": marker_binding["trace_sha256"],
                "producer_path": marker_binding["producer_path"],
                "producer_sha256": marker_binding["producer_sha256"],
                "capture_id": capture_id,
                "schedule_run_id": marker_binding["schedule_run_id"],
                "command_identity": marker_binding["command_identity"],
                "render_binding": marker_binding["render_binding"],
            },
            "request": {
                "display": config.display,
                "duration_s_decimal": str(config.duration_s),
                "fps_numerator": config.fps,
                "fps_denominator": 1,
                "expected_frame_count": config.expected_frames,
                "cursor_included": False,
                "xdamage_incremental_capture": False,
                "synthetic_frame_duplication": False,
                "encode": config.encode,
            },
            "result": {
                "actual_frame_count": len(records),
                "width_px": width,
                "height_px": height,
                "frame_set_sha256": _frame_set_sha256(records),
                "first_timestamp_ns": records[0]["timestamp_ns"],
                "last_timestamp_ns": records[-1]["timestamp_ns"],
                "last_duration_ns": records[-1]["duration_ns"],
                "first_frame_is_partial_rate_period": (
                    records[0]["timestamp_ns"] != 0
                ),
                "partial_rate_period_count": sum(
                    record["acquisition_lateness_ns"] != 0 for record in records
                ),
                "maximum_acquisition_lateness_ns": max(
                    record["acquisition_lateness_ns"] for record in records
                ),
                "encoded_video_sha256": (
                    encoding["video"]["sha256"] if encoding is not None else None
                ),
            },
            "evidence_use": {
                "classification_without_video_clock_anchor": "diagnostic_only",
                "acceptance_bearing_without_video_clock_anchor": False,
                "reason": (
                    "this capture process measures relative GStreamer frame PTS; "
                    "the supplied trace and marker producer are hash-bound, but "
                    "acceptance still requires machine-detecting the marker pixels"
                ),
                "required_external_anchor_schema": VIDEO_CLOCK_ANCHOR_SCHEMA,
                "anchor_tool": "video_clock_anchor.py",
                "acceptance_condition": (
                    "a machine-measured rendered-marker transition must bind the "
                    "published video SHA-256, trace SHA-256, command identity, and "
                    "command-edge video PTS with uncertainty no greater than half "
                    "one encoded frame period"
                ),
            },
            "timebase": {
                "frame_timestamp_source": (
                    "GstMultiFileSink element messages retained in gst.bus.log"
                ),
                "process_wall_clock_utc": {
                    "start": result.start_utc,
                    "end": result.end_utc,
                },
                "absolute_frame_acquisition_utc_measured": False,
                "absolute_frame_acquisition_utc_limitation": (
                    "GStreamer PTS is retained, but gst-launch does not expose "
                    "the pipeline clock-to-UTC offset; cross-process video edge "
                    "alignment therefore requires a separately measured anchor"
                ),
                "command_edge_to_video_pts": {
                    "state": "unmeasured",
                    "method": None,
                    "uncertainty_s": None,
                    "reason": (
                        "the command is applied by a separate Wine process; the "
                        "bound marker transition has not yet been classified in "
                        "the sealed frames"
                    ),
                    "required_anchor_schema": VIDEO_CLOCK_ANCHOR_SCHEMA,
                },
                "frame_timestamp_unit": "nanosecond",
                "frame_timestamp_fields": [
                    "timestamp_ns",
                    "stream_time_ns",
                    "running_time_ns",
                    "duration_ns",
                ],
                "process_monotonic_clock": {
                    "implementation": clock.implementation,
                    "resolution_s": clock.resolution,
                    "monotonic": clock.monotonic,
                    "adjustable": clock.adjustable,
                    "start_ns": result.start_monotonic_ns,
                    "end_ns": result.end_monotonic_ns,
                },
            },
            "host": {
                "hostname": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "user": getpass.getuser(),
                "python": sys.version,
            },
            "tools": {
                "capture_script": {
                    "path": str(script_path),
                    "sha256": _sha256(script_path),
                },
                "gstreamer": {
                    "command_prefix": list(config.gst_command_prefix),
                    "version_command": [
                        *config.gst_command_prefix,
                        "--version",
                    ],
                    "version_output": gst_version,
                },
            },
            "commands": {
                "gstreamer_version": [*config.gst_command_prefix, "--version"],
                "gstreamer_capture": argv,
            },
            "publication": {
                "manifest": "fsynced temporary file followed by atomic replace",
                "evidence_directory": (
                    "Linux renameat2(RENAME_NOREPLACE)"
                    if sys.platform.startswith("linux")
                    else "Windows no-replace directory rename"
                ),
            },
            "process": result.__dict__,
            "artifacts": artifacts,
            "encoding": encoding,
            "frames": records,
        }
        manifest_path = stage / "capture.manifest.json"
        _atomic_json(manifest_path, manifest)
        _fsync_directory(frame_dir)
        _fsync_directory(stage)
        _rename_no_replace(stage, output)
        _fsync_directory(output.parent)
        return manifest
    except BaseException as exc:
        if stage.exists():
            try:
                _atomic_json(
                    stage / "capture.failure.json",
                    {
                        "schema": FAILURE_SCHEMA,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "request": {
                            "output_dir": str(output),
                            "display": config.display,
                            "duration_s_decimal": str(config.duration_s),
                            "fps": config.fps,
                            "expected_frame_count": config.expected_frames,
                            "encode": config.encode,
                        },
                        "recorded_utc": _utc_now(),
                    },
                )
                failure_path = _failure_destination(output)
                _fsync_directory(stage)
                _rename_no_replace(stage, failure_path)
                _fsync_directory(output.parent)
            except BaseException:
                failure_path = stage
        if isinstance(exc, (CaptureError, FileExistsError)):
            suffix = f"; incomplete evidence: {failure_path}" if failure_path else ""
            raise type(exc)(f"{exc}{suffix}") from exc
        suffix = f"; incomplete evidence: {failure_path}" if failure_path else ""
        raise CaptureError(f"capture failed: {exc}{suffix}") from exc


def _decimal(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("duration must be a decimal number") from exc
    if not result.is_finite():
        raise argparse.ArgumentTypeError("duration must be finite")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=_decimal, required=True, metavar="SECONDS")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--display", default=":98")
    parser.add_argument(
        "--encode", choices=("none", "auto", "ffmpeg", "opencv"), default="none"
    )
    parser.add_argument("--gst-launch", default="gst-launch-1.0")
    parser.add_argument("--ffmpeg")
    parser.add_argument(
        "--marker-contract",
        type=Path,
        required=True,
        help=(
            "completed rendered-command marker contract; it may be published "
            "while capture is running but must validate before publication"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = CaptureConfig(
        output_dir=arguments.out,
        duration_s=arguments.duration,
        fps=arguments.fps,
        display=arguments.display,
        encode=arguments.encode,
        gst_command_prefix=(arguments.gst_launch,),
        ffmpeg=arguments.ffmpeg,
        marker_contract=arguments.marker_contract,
    )
    try:
        manifest = capture(config)
    except (CaptureError, FileExistsError) as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 1
    output = config.output_dir.expanduser().absolute()
    print(f"capture: complete")
    print(f"frames: {manifest['result']['actual_frame_count']}")
    print(f"dimensions: {manifest['result']['width_px']}x{manifest['result']['height_px']}")
    print(f"manifest: {output / 'capture.manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
