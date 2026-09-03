import base64
import binascii
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import textwrap
import unittest
import zlib
from decimal import Decimal
from pathlib import Path
from unittest import mock

import spark_x11_capture as capture


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Wl2n0AAAAAASUVORK5CYII="
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_marker_contract(root: Path, suffix: str = "fixture") -> Path:
    trace = root / f"trace-{suffix}.jsonl"
    producer = root / f"producer-{suffix}.dll"
    contract = root / f"marker-{suffix}.contract.json"
    trace.write_text('{"event":"command_edge"}\n', encoding="utf-8")
    producer.write_bytes(f"producer {suffix}".encode("utf-8"))
    contract.write_text(
        json.dumps(
            {
                "schema": capture.MARKER_CONTRACT_SCHEMA,
                "command_identity": "walk_forward:press:v1",
                "schedule_run_id": f"run-{suffix}",
                "trace": {"path": trace.name, "sha256": sha256(trace)},
                "producer": {
                    "path": producer.name,
                    "sha256": sha256(producer),
                    "render_binding": capture.RENDER_BINDING,
                },
                "marker": {"transition": capture.MARKER_TRANSITION},
            }
        ),
        encoding="utf-8",
    )
    return contract


def make_rgba_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(
        b"\0" + bytes((20, 40, 60, 255)) * width for _ in range(height)
    )
    return (
        capture.PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def write_fake_gstreamer(directory: Path) -> Path:
    script = directory / "fake_gstreamer.py"
    encoded_png = base64.b64encode(PNG_1X1).decode("ascii")
    script.write_text(
        textwrap.dedent(
            f"""
            import base64
            import pathlib
            import re
            import sys

            if "--version" in sys.argv:
                print("gst-launch-1.0 version 1.24.0-test")
                raise SystemExit(0)

            frame_count = int(next(
                value.split("=", 1)[1] for value in sys.argv
                if value.startswith("num-buffers=")
            ))
            fps = int(next(
                re.search(r"framerate=(\\d+)/1", value).group(1)
                for value in sys.argv if "framerate=" in value
            ))
            pattern = next(
                value.split("=", 1)[1] for value in sys.argv
                if value.startswith("location=")
            )
            missing = "--fake-missing-message" in sys.argv
            partial_first = "--fake-partial-first" in sys.argv
            for index in range(frame_count):
                path = pathlib.Path(pattern.replace("%06d", f"{{index:06d}}"))
                path.write_bytes(base64.b64decode({encoded_png!r}))
                if missing and index == frame_count - 1:
                    continue
                timestamp = (index * 1_000_000_000) // fps
                next_timestamp = ((index + 1) * 1_000_000_000) // fps
                duration = next_timestamp - timestamp
                if partial_first and index == 0:
                    timestamp = 1_000_000
                    duration = next_timestamp - timestamp
                print(
                    'Got message from element "framesink" (element): '
                    f'GstMultiFileSink, filename=(string){{path}}, index=(int){{index}}, '
                    f'timestamp=(guint64){{timestamp}}, stream-time=(guint64){{timestamp}}, '
                    f'running-time=(guint64){{timestamp}}, duration=(guint64){{duration}}, '
                    'offset=(guint64)18446744073709551615, '
                    'offset-end=(guint64)18446744073709551615;'
                )
            print('Got message from element "pipeline0" (eos): no message details')
            print('Got EOS from element "pipeline0".')
            """
        ),
        encoding="utf-8",
    )
    return script


class SparkX11CaptureTests(unittest.TestCase):
    def config(self, root: Path, fake: Path, *extra: str) -> capture.CaptureConfig:
        return capture.CaptureConfig(
            output_dir=root / "evidence",
            duration_s=Decimal("1.5"),
            fps=2,
            display=":test",
            gst_command_prefix=(sys.executable, str(fake), *extra),
            marker_contract=write_marker_contract(root),
        )

    def test_builds_cursor_free_fixed_rate_pipeline(self):
        config = capture.CaptureConfig(
            output_dir=Path("out"),
            duration_s=Decimal("2"),
            fps=60,
        )
        command = capture.build_gstreamer_command(config, Path("frames"))
        self.assertIn("show-pointer=false", command)
        self.assertIn("use-damage=false", command)
        self.assertIn("num-buffers=120", command)
        self.assertIn("video/x-raw,framerate=60/1", command)
        self.assertIn("post-messages=true", command)
        self.assertNotIn("videorate", command)
        self.assertIn("queue", command)
        self.assertIn("max-size-buffers=120", command)
        self.assertIn("compression-level=1", command)

    def test_parses_measured_multifilesink_timestamp(self):
        line = (
            'GstMultiFileSink, filename=(string)/tmp/frame-000001.png, '
            'index=(int)1, timestamp=(guint64)16666666, '
            'stream-time=(guint64)16666666, running-time=(guint64)16666666, '
            'duration=(guint64)16666667, offset=(guint64)0;'
        )
        records = capture.parse_gstreamer_frame_messages(line)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["index"], 1)
        self.assertEqual(records[0]["timestamp_ns"], 16_666_666)
        self.assertEqual(records[0]["duration_ns"], 16_666_667)

    def test_parses_gstreamer_escaped_output_path(self):
        line = (
            'GstMultiFileSink, filename=(string)"/tmp/rek\\ cap\\,one/'
            'frame-000000.png", index=(int)0, timestamp=(guint64)0, '
            'stream-time=(guint64)0, running-time=(guint64)0, '
            'duration=(guint64)500000000;'
        )
        records = capture.parse_gstreamer_frame_messages(line)
        self.assertEqual(records[0]["filename"], "/tmp/rek cap,one/frame-000000.png")

    def test_end_to_end_fake_capture_publishes_manifest_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = write_fake_gstreamer(root)
            config = self.config(root, fake)
            result = capture.capture(config)
            output = config.output_dir
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["result"]["actual_frame_count"], 3)
            self.assertEqual(result["result"]["width_px"], 1)
            self.assertEqual(result["result"]["height_px"], 1)
            self.assertFalse(result["request"]["cursor_included"])
            self.assertIn("start", result["timebase"]["process_wall_clock_utc"])
            self.assertIn("end", result["timebase"]["process_wall_clock_utc"])
            self.assertFalse(
                result["timebase"]["absolute_frame_acquisition_utc_measured"]
            )
            self.assertEqual(
                result["evidence_use"][
                    "classification_without_video_clock_anchor"
                ],
                "diagnostic_only",
            )
            self.assertFalse(
                result["evidence_use"][
                    "acceptance_bearing_without_video_clock_anchor"
                ]
            )
            self.assertEqual(
                result["evidence_use"]["required_external_anchor_schema"],
                capture.VIDEO_CLOCK_ANCHOR_SCHEMA,
            )
            self.assertEqual(
                result["timebase"]["command_edge_to_video_pts"]["state"],
                "unmeasured",
            )
            self.assertIsNone(result["result"]["encoded_video_sha256"])
            self.assertTrue((output / "capture.manifest.json").is_file())
            self.assertTrue((output / "frames.jsonl").is_file())
            self.assertEqual(len(list((output / "frames").glob("*.png"))), 3)
            manifest = json.loads(
                (output / "capture.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["frames"]), 3)
            self.assertEqual(len(manifest["frames"][0]["sha256"]), 64)
            self.assertEqual(
                manifest["artifacts"]["frame_timestamp_sidecar"]["sha256"],
                capture._sha256(output / "frames.jsonl"),
            )
            partials = list(root.glob(".evidence.partial-*"))
            self.assertEqual(partials, [])

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = write_fake_gstreamer(root)
            config = self.config(root, fake)
            config.output_dir.mkdir()
            marker = config.output_dir / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                capture.capture(config)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_published_manifest_binds_post_encode_video_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = write_fake_gstreamer(root)
            config = capture.CaptureConfig(
                output_dir=root / "evidence",
                duration_s=Decimal("1.5"),
                fps=2,
                display=":test",
                encode="auto",
                gst_command_prefix=(sys.executable, str(fake)),
                marker_contract=write_marker_contract(root),
            )

            def encode_fixture(_config, stage, _records, _dimensions):
                video = stage / "capture.mp4"
                video.write_bytes(b"post-encode fixture")
                return {
                    "mode": "fixture",
                    "video": capture._artifact(video, stage),
                }

            with mock.patch.object(capture, "_encode", side_effect=encode_fixture):
                result = capture.capture(config)
            video = config.output_dir / "capture.mp4"
            observed = capture._sha256(video)
            self.assertEqual(result["result"]["encoded_video_sha256"], observed)
            self.assertEqual(
                result["artifacts"]["encoded_video"]["sha256"], observed
            )
            published = json.loads(
                (config.output_dir / "capture.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                published["result"]["encoded_video_sha256"], observed
            )
            self.assertEqual(
                published["binding"]["marker_contract_sha256"],
                sha256(config.marker_contract),
            )
            self.assertEqual(
                published["binding"]["capture_id"], published["capture_id"]
            )

    def test_missing_marker_contract_fails_before_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = write_fake_gstreamer(root)
            config = capture.CaptureConfig(
                output_dir=root / "evidence",
                duration_s=Decimal("1.5"),
                fps=2,
                display=":test",
                gst_command_prefix=(sys.executable, str(fake)),
            )
            with self.assertRaisesRegex(capture.CaptureError, "contract path"):
                capture.capture(config)

    def test_marker_contract_trace_swap_fails_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = write_fake_gstreamer(root)
            contract = write_marker_contract(root)
            document = json.loads(contract.read_text(encoding="utf-8"))
            swapped_trace = root / "swapped.jsonl"
            swapped_trace.write_text("swapped\n", encoding="utf-8")
            document["trace"]["path"] = swapped_trace.name
            contract.write_text(json.dumps(document), encoding="utf-8")
            config = capture.CaptureConfig(
                output_dir=root / "evidence",
                duration_s=Decimal("1.5"),
                fps=2,
                display=":test",
                gst_command_prefix=(sys.executable, str(fake)),
                marker_contract=contract,
            )
            with self.assertRaisesRegex(capture.CaptureError, "trace SHA-256"):
                capture.capture(config)
            self.assertFalse(config.output_dir.exists())

    def test_atomic_publish_refuses_destination_created_after_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                capture._rename_no_replace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_accepts_measured_partial_first_live_source_period(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = write_fake_gstreamer(root)
            config = self.config(root, fake, "--fake-partial-first")
            result = capture.capture(config)
            self.assertTrue(result["result"]["first_frame_is_partial_rate_period"])
            self.assertEqual(result["frames"][0]["timestamp_ns"], 1_000_000)
            self.assertEqual(result["frames"][1]["timestamp_ns"], 500_000_000)

    def test_missing_timestamp_message_fails_closed_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = write_fake_gstreamer(root)
            config = self.config(root, fake, "--fake-missing-message")
            with self.assertRaises(capture.CaptureError):
                capture.capture(config)
            self.assertFalse(config.output_dir.exists())
            failures = list(root.glob(".evidence.failed-*"))
            self.assertEqual(len(failures), 1)
            failure = json.loads(
                (failures[0] / "capture.failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error_type"], "CaptureError")

    def test_rejects_fractional_frame_count(self):
        config = capture.CaptureConfig(
            output_dir=Path("out"), duration_s=Decimal("0.1"), fps=24
        )
        with self.assertRaises(capture.CaptureError):
            capture._validate_config(config)

    @unittest.skipIf(
        importlib.util.find_spec("cv2") is None, "optional OpenCV is unavailable"
    )
    def test_optional_opencv_encoder_writes_mp4(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            frames = stage / "frames"
            frames.mkdir()
            records = []
            for index in range(3):
                path = frames / f"frame-{index:06d}.png"
                path.write_bytes(make_rgba_png(16, 16))
                records.append({"path": f"frames/{path.name}"})
            result = capture._encode_opencv(stage, 2, records, (16, 16))
            self.assertEqual(result["mode"], "opencv")
            self.assertEqual(result["frames_written"], 3)
            self.assertGreater((stage / "capture.mp4").stat().st_size, 0)
            self.assertEqual(len(result["video"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
