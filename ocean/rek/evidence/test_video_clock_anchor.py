import binascii
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import video_clock_anchor as anchor


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solid_rgb_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\0" + bytes(rgb) * width for _ in range(height))
    signature = b"\x89PNG\r\n\x1a\n"
    return (
        signature
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


@unittest.skipIf(
    importlib.util.find_spec("PIL") is None, "Pillow is required by the marker tool"
)
class VideoClockAnchorTests(unittest.TestCase):
    def fixture(
        self,
        root: Path,
        states: tuple[str, ...] = ("pre", "pre", "post", "post"),
        timestamps_ns: tuple[int, ...] = (0, 500_000_000, 1_000_000_000, 1_500_000_000),
        fps: int = 2,
    ) -> tuple[Path, Path, Path]:
        capture_dir = root / "capture"
        frames_dir = capture_dir / "frames"
        frames_dir.mkdir(parents=True)
        colors = {"pre": (0, 0, 0), "post": (255, 0, 255), "other": (3, 4, 5)}
        records = []
        for index, (state, timestamp_ns) in enumerate(zip(states, timestamps_ns)):
            frame = frames_dir / f"frame-{index:06d}.png"
            frame.write_bytes(solid_rgb_png(4, 4, colors[state]))
            records.append(
                {
                    "index": index,
                    "path": f"frames/{frame.name}",
                    "sha256": sha256(frame),
                    "timestamp_ns": timestamp_ns,
                }
            )
        video = capture_dir / "capture.mp4"
        video.write_bytes(b"sealed synthetic MP4 fixture")
        video_sha = sha256(video)
        frame_set = hashlib.sha256()
        for record in records:
            frame_set.update(str(record["index"]).encode("ascii"))
            frame_set.update(b"\0")
            frame_set.update(record["sha256"].encode("ascii"))
            frame_set.update(b"\0")
            frame_set.update(str(record["timestamp_ns"]).encode("ascii"))
            frame_set.update(b"\n")
        manifest = {
            "schema": anchor.CAPTURE_SCHEMA,
            "status": "complete",
            "capture_id": "capture001",
            "request": {
                "fps_numerator": fps,
                "fps_denominator": 1,
                "expected_frame_count": len(records),
                "synthetic_frame_duplication": False,
            },
            "result": {
                "actual_frame_count": len(records),
                "frame_set_sha256": frame_set.hexdigest(),
                "encoded_video_sha256": video_sha,
            },
            "artifacts": {
                "encoded_video": {
                    "path": "capture.mp4",
                    "sha256": video_sha,
                }
            },
            "encoding": {"mode": "fixture"},
            "frames": records,
        }
        (capture_dir / "capture.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        trace = root / "trace.jsonl"
        trace.write_text('{"event":"command_edge"}\n', encoding="utf-8")
        producer = root / "marker-producer.dll"
        producer.write_bytes(b"fixture marker producer")
        contract = {
            "schema": anchor.CONTRACT_SCHEMA,
            "command_identity": "walk_forward:press:v1",
            "schedule_run_id": "0123456789abcdef0123456789abcdef",
            "trace": {"path": trace.name, "sha256": sha256(trace)},
            "producer": {
                "path": producer.name,
                "sha256": sha256(producer),
                "render_binding": anchor.RENDER_BINDING,
            },
            "marker": {
                "transition": anchor.TRANSITION,
                "region_px": {"x": 0, "y": 0, "width": 4, "height": 4},
                "pre_rgb": [0, 0, 0],
                "post_rgb": [255, 0, 255],
            },
        }
        contract_path = root / "marker-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        manifest["binding"] = {
            "marker_contract_schema": anchor.CONTRACT_SCHEMA,
            "marker_contract_sha256": sha256(contract_path),
            "trace_sha256": sha256(trace),
            "producer_sha256": sha256(producer),
            "capture_id": manifest["capture_id"],
            "schedule_run_id": contract["schedule_run_id"],
            "command_identity": contract["command_identity"],
            "render_binding": anchor.RENDER_BINDING,
        }
        (capture_dir / "capture.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return capture_dir, contract_path, root / "video-clock-anchor.json"

    def make_simulator_capture(self, capture_dir: Path, contract_path: Path) -> None:
        manifest_path = capture_dir / "capture.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        trace_path = contract_path.parent / contract["trace"]["path"]
        fps = manifest["request"]["fps_numerator"]
        frame_count = len(manifest["frames"])
        trace = {
            "schema": "rek.paired_motion_trace.v1",
            "source": "clone:rek_fight_engineai",
            "capture_id": manifest["capture_id"],
            "schedule_run_id": contract["schedule_run_id"],
            "command": {
                "identity": contract["command_identity"],
                "execution_state": "simulated",
                "edge_physics_tick": 1,
            },
            "timing": {"physics_rate_hz": fps},
            "samples": [
                {"physics_tick": index, "time_s": index / fps}
                for index in range(frame_count)
            ],
        }
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        contract["trace"]["sha256"] = sha256(trace_path)
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        manifest["schema"] = anchor.SIMULATOR_CAPTURE_SCHEMA
        manifest["request"].update(
            {
                "source_physics_rate_hz": fps,
                "frame_selection_stride_physics_ticks": 1,
                "sample_interpolation_used": False,
                "frame_selection": "exact_measured_physics_tick_modulo_stride",
            }
        )
        manifest["result"]["actual_frame_count"] = frame_count
        manifest["binding"].update(
            {
                "marker_contract_sha256": sha256(contract_path),
                "trace_sha256": sha256(trace_path),
                "command_edge_physics_tick": 1,
                "command_edge_frame_index": 1,
                "first_post_marker_frame_index": 2,
            }
        )
        for index, frame in enumerate(manifest["frames"]):
            frame.update(
                {
                    "trace_sample_index": index,
                    "simulation_physics_tick": index,
                    "simulation_time_ns": round(index / fps * 1e9),
                    "marker_state": "pre" if index < 2 else "post",
                }
            )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_machine_detected_marker_creates_hash_bound_half_frame_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir, contract, output = self.fixture(Path(temporary))
            result = anchor.create_anchor(capture_dir, contract, output)
            self.assertEqual(result["schema"], anchor.ANCHOR_SCHEMA)
            self.assertEqual(result["command_identity"], "walk_forward:press:v1")
            self.assertEqual(
                result["run_id"], "0123456789abcdef0123456789abcdef"
            )
            self.assertEqual(result["command_edge_video_pts_s"], 0.75)
            self.assertEqual(result["measurement"]["uncertainty_s"], 0.25)
            self.assertEqual(
                result["measurement"]["method"],
                "rendered_command_marker_transition_v1",
            )
            self.assertEqual(result["video_sha256"], sha256(capture_dir / "capture.mp4"))
            self.assertEqual(result["marker_observation"]["last_pre_frame_index"], 1)
            self.assertEqual(result["marker_observation"]["first_post_frame_index"], 2)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir, contract, output = self.fixture(Path(temporary))
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                anchor.create_anchor(capture_dir, contract, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_non_marker_pixels_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir, contract, output = self.fixture(
                Path(temporary), states=("pre", "other", "post", "post")
            )
            with self.assertRaisesRegex(anchor.AnchorError, "neither exact"):
                anchor.create_anchor(capture_dir, contract, output)
            self.assertFalse(output.exists())

    def test_marker_reversion_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir, contract, output = self.fixture(
                Path(temporary), states=("pre", "post", "pre", "post")
            )
            with self.assertRaisesRegex(anchor.AnchorError, "reverted"):
                anchor.create_anchor(capture_dir, contract, output)

    def test_measured_gap_over_one_video_period_fails_uncertainty_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir, contract, output = self.fixture(
                Path(temporary),
                timestamps_ns=(0, 500_000_000, 1_100_000_000, 1_500_000_000),
            )
            with self.assertRaisesRegex(anchor.AnchorError, "half-frame uncertainty"):
                anchor.create_anchor(capture_dir, contract, output)

    def test_integer_nanosecond_rounding_at_60_fps_meets_exact_half_frame_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir, contract, output = self.fixture(
                Path(temporary),
                timestamps_ns=(0, 16_666_666, 33_333_333, 50_000_000),
                fps=60,
            )
            result = anchor.create_anchor(capture_dir, contract, output)
            self.assertAlmostEqual(
                result["measurement"]["uncertainty_s"], 1.0 / 120.0
            )
            self.assertEqual(
                result["marker_observation"]["maximum_allowed_source_bracket_ns"],
                16_666_667,
            )

    def test_trace_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir, contract, output = self.fixture(root)
            trace = root / "trace.jsonl"
            trace.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(anchor.AnchorError, "trace SHA-256 mismatch"):
                anchor.create_anchor(capture_dir, contract, output)

    def test_capture_from_another_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir, _, output = self.fixture(root / "first")
            _, swapped_contract, _ = self.fixture(root / "second")
            swapped = json.loads(swapped_contract.read_text(encoding="utf-8"))
            swapped["schedule_run_id"] = "fedcba9876543210fedcba9876543210"
            swapped_contract.write_text(json.dumps(swapped), encoding="utf-8")
            with self.assertRaisesRegex(
                anchor.AnchorError, "manifest binding differs"
            ):
                anchor.create_anchor(capture_dir, swapped_contract, output)

    def test_manifest_trace_binding_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir, contract, output = self.fixture(root)
            manifest_path = capture_dir / "capture.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["binding"]["trace_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                anchor.AnchorError, "manifest binding differs"
            ):
                anchor.create_anchor(capture_dir, contract, output)

    def test_unsupported_capture_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir, contract, output = self.fixture(root)
            manifest_path = capture_dir / "capture.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema"] = "rek.simulator_frame_capture.v0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(anchor.AnchorError, "supported sealed-frame"):
                anchor.create_anchor(capture_dir, contract, output)

    def test_strict_simulator_capture_schema_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir, contract, output = self.fixture(root)
            self.make_simulator_capture(capture_dir, contract)
            result = anchor.create_anchor(capture_dir, contract, output)
            self.assertEqual(
                result["source_capture"]["capture_schema"],
                anchor.SIMULATOR_CAPTURE_SCHEMA,
            )

    def test_simulator_schema_without_exact_selection_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir, contract, output = self.fixture(root)
            self.make_simulator_capture(capture_dir, contract)
            manifest_path = capture_dir / "capture.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["request"]["sample_interpolation_used"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(anchor.AnchorError, "forbid interpolation"):
                anchor.create_anchor(capture_dir, contract, output)

    def test_simulator_frame_trace_index_swap_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture_dir, contract, output = self.fixture(root)
            self.make_simulator_capture(capture_dir, contract)
            manifest_path = capture_dir / "capture.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["frames"][2]["trace_sample_index"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(anchor.AnchorError, "exact trace sample"):
                anchor.create_anchor(capture_dir, contract, output)


if __name__ == "__main__":
    unittest.main()
