import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import paired_motion_compare as paired


BUILD = "synthetic-build-fingerprint"
TIMES = (-0.02, 0.0, 0.02, 0.04, 0.06)
SCREEN_FRAME = {"id": "synthetic-camera-640x480", "width_px": 640,
                "height_px": 480}


def identity_calibration():
    return {
        "schema": paired.CALIBRATION_SCHEMA,
        "mode": "identity",
        "source_forward_vector": [1, 0, 0],
        "provenance": {
            "kind": "synthetic_fixture",
            "description": "known identity frame",
        },
    }


def transformed_calibration():
    return {
        "schema": paired.CALIBRATION_SCHEMA,
        "mode": "explicit_similarity_3d",
        "position_matrix": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "position_offset": [10, -3, 0],
        "source_forward_vector": [1, 0, 0],
        "direct_yaw_sign": 1,
        "direct_yaw_offset_rad": math.pi / 2,
        "provenance": {
            "kind": "synthetic_fixture",
            "description": "known quarter-turn frame",
        },
    }


def arena_state(relative_time, *, offset=0.0, yaw_offset=0.0,
                screen_offset=0.0):
    progress = max(0.0, relative_time)
    return {
        "position": (2.0 + progress + offset, -1.0 + 0.5 * progress, 1.0),
        "yaw": 0.25 * progress + yaw_offset,
        "screen": (320.0 + 100.0 * progress + screen_offset,
                   260.0 - 20.0 * progress),
    }


def to_transformed_source(position):
    shifted = (position[0] - 10.0, position[1] + 3.0, position[2])
    return (shifted[1], -shifted[0], shifted[2])


def trace_document(source, edge, *, offset=0.0, yaw_offset=0.0,
                   screen_offset=0.0, calibration=None, include_screen=True,
                   transformed=False, execution_state=None, end_at=None,
                   capture_id=None, run_id=None):
    if execution_state is None:
        execution_state = "measured_executed" if source == "rek" else "simulated"
    samples = []
    for relative_time in TIMES:
        if end_at is not None and relative_time > end_at:
            continue
        state = arena_state(
            relative_time, offset=offset, yaw_offset=yaw_offset,
            screen_offset=screen_offset,
        )
        position = state["position"]
        yaw = state["yaw"]
        if transformed:
            position = to_transformed_source(position)
            yaw -= math.pi / 2
        sample = {
            "time_s": edge + relative_time,
            "root_position": list(position),
            "root_yaw_rad": yaw,
        }
        if include_screen:
            sample["screen_root_px"] = list(state["screen"])
        samples.append(sample)
    result = {
        "schema": paired.TRACE_SCHEMA,
        "source": source,
        "build_fingerprint": BUILD,
        "capture_id": capture_id or f"capture-{source}-{edge}",
        "schedule_run_id": run_id or f"run-{source}-{edge}",
        "command": {
            "identity": "walk_forward:press:v1",
            "edge_time_s": edge,
            "execution_state": execution_state,
        },
        "calibration": calibration or identity_calibration(),
        "samples": samples,
    }
    if include_screen:
        result["screen_frame"] = SCREEN_FRAME
    return result


class PairedMotionComparisonTests(unittest.TestCase):
    def load(self, directory, name, document):
        path = Path(directory) / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return paired.load_motion_trace(path)

    def compare(self, *args, **kwargs):
        kwargs.setdefault("maximum_timestamp_uncertainty_s", 1e-9)
        return paired.compare_motion(*args, **kwargs)

    def write_video_anchor(self, directory, name, video, trace, *, edge=0.2,
                           uncertainty=0.005):
        path = Path(directory) / name
        path.write_text(json.dumps({
            "schema": paired.VIDEO_CLOCK_ANCHOR_SCHEMA,
            "video_sha256": paired._sha256(video),
            "trace_sha256": trace.sha256,
            "command_identity": trace.command_identity,
            "command_edge_video_pts_s": edge,
            "measurement": {
                "state": "measured",
                "method": "synthetic_frame_marker_transition_v1",
                "uncertainty_s": uncertainty,
                "provenance": {
                    "kind": "synthetic_fixture",
                    "description": "known marker transition",
                },
            },
        }), encoding="utf-8")
        return path

    def exact_inputs(self, directory, include_screen=True):
        reference = self.load(
            directory, "rek-0.json",
            trace_document("rek", 1.0, include_screen=include_screen),
        )
        repeat_a = self.load(
            directory, "rek-1.json",
            trace_document("rek", 3.25, offset=1e-12, yaw_offset=1e-12,
                           include_screen=include_screen),
        )
        repeat_b = self.load(
            directory, "rek-2.json",
            trace_document("rek", 7.5, offset=-1e-12, yaw_offset=-1e-12,
                           include_screen=include_screen),
        )
        candidate = self.load(
            directory, "sim.json",
            trace_document(
                "clone:synthetic", 20.0, include_screen=include_screen,
                transformed=True, calibration=transformed_calibration(),
            ),
        )
        return reference, [repeat_a, repeat_b], candidate

    def test_aligns_command_edges_and_explicit_frames_without_fitting_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, candidate = self.exact_inputs(directory)
            report = self.compare(
                reference, repeats, candidate, accept_at="max",
                require_screen=True,
            )
        self.assertEqual(report["verdict"]["state"], "passed")
        self.assertTrue(report["verdict"]["passed"])
        self.assertEqual(report["alignment"]["source"],
                         "primary_rek_sample_timebase")
        self.assertFalse(report["alignment"]["dynamic_time_warping_used"])
        self.assertFalse(report["alignment"]["edge_position_subtraction_used"])
        self.assertEqual(report["alignment"]["frame_count"], len(TIMES))
        self.assertLess(report["metrics"]["root_position"]["error"]["max"], 1e-14)
        self.assertLess(report["metrics"]["root_yaw"]["error"]["max"], 1e-14)
        self.assertEqual(report["metrics"]["screen_root"]["error"]["max"], 0.0)
        self.assertFalse(
            report["inputs"]["candidate"]["calibration"]["trajectory_fit_used"]
        )
        self.assertAlmostEqual(
            report["inputs"]["candidate"]["calibration"]["orthogonal_determinant"],
            1.0,
        )

    def test_reports_every_frame_outside_measured_repeat_variance(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self.load(
                directory, "rek-0.json", trace_document("rek", 1.0)
            )
            repeat_a = self.load(
                directory, "rek-1.json",
                trace_document("rek", 2.0, offset=0.01, yaw_offset=0.005,
                               screen_offset=1.0),
            )
            repeat_b = self.load(
                directory, "rek-2.json",
                trace_document("rek", 3.0, offset=-0.01, yaw_offset=-0.005,
                               screen_offset=-1.0),
            )
            candidate = self.load(
                directory, "sim.json",
                trace_document("clone:synthetic", 4.0, offset=0.03,
                               yaw_offset=0.02, screen_offset=3.0),
            )
            report = self.compare(
                reference, [repeat_a, repeat_b], candidate,
                accept_at="max", require_screen=True,
            )
        self.assertEqual(report["verdict"]["state"], "failed")
        self.assertFalse(report["verdict"]["passed"])
        self.assertEqual(
            report["metrics"]["root_position"]["failed_frame_count"], len(TIMES)
        )
        self.assertEqual(
            report["metrics"]["root_yaw"]["failed_frame_count"], len(TIMES)
        )
        self.assertEqual(
            report["metrics"]["screen_root"]["failed_frame_count"], len(TIMES)
        )
        first = report["frames"][0]
        self.assertAlmostEqual(first["root_position_rek_allowance_m"], 0.02)
        self.assertAlmostEqual(first["root_position_error_m"], 0.03)
        self.assertAlmostEqual(first["screen_root_rek_allowance_px"], 2.0)
        self.assertAlmostEqual(first["screen_root_error_px"], 3.0)

    def test_missing_screen_measurement_blocks_required_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, candidate = self.exact_inputs(
                directory, include_screen=False
            )
            blocked = self.compare(
                reference, repeats, candidate, accept_at="max",
                require_screen=True,
            )
            world_only = self.compare(
                reference, repeats, candidate, accept_at="max",
                require_screen=False,
            )
        self.assertIsNone(blocked["verdict"]["passed"])
        self.assertIn("common_screen_frame_not_measured",
                      blocked["verdict"]["blockers"])
        self.assertTrue(world_only["verdict"]["passed"])
        self.assertNotIn("screen_root", world_only["metrics"])

    def test_unknown_rek_execution_never_becomes_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self.load(
                directory, "rek-0.json",
                trace_document("rek", 1.0, execution_state="unknown"),
            )
            repeat_a = self.load(
                directory, "rek-1.json",
                trace_document("rek", 2.0, execution_state="unknown"),
            )
            repeat_b = self.load(
                directory, "rek-2.json",
                trace_document("rek", 3.0, execution_state="unknown"),
            )
            candidate = self.load(
                directory, "sim.json",
                trace_document("clone:synthetic", 4.0),
            )
            report = self.compare(
                reference, [repeat_a, repeat_b], candidate,
                accept_at="max", require_screen=True,
            )
        self.assertIsNone(report["verdict"]["passed"])
        self.assertEqual(report["verdict"]["state"], "insufficient_evidence")
        self.assertIn("rek_command_execution_not_measured",
                      report["verdict"]["blockers"])
        self.assertEqual(report["metrics"]["root_position"]["error"]["max"], 0.0)

    def test_missing_candidate_coverage_is_reported_without_cropping(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, _ = self.exact_inputs(directory)
            candidate = self.load(
                directory, "short-sim.json",
                trace_document("clone:synthetic", 10.0, end_at=0.04),
            )
            report = self.compare(
                reference, repeats, candidate, accept_at="max",
                require_screen=True,
            )
        self.assertIsNone(report["verdict"]["passed"])
        self.assertIn("incomplete_time_coverage", report["verdict"]["blockers"])
        self.assertAlmostEqual(
            report["alignment"]["end_relative_time_s"], TIMES[-1]
        )
        missing = report["interpolation"]["candidate"]["missing_relative_times_s"]
        self.assertEqual(len(missing), 1)
        self.assertAlmostEqual(missing[0], TIMES[-1])

    def test_rejects_same_rek_capture_object_repeated_as_multiple_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, candidate = self.exact_inputs(directory)
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "not distinct captures"):
                self.compare(reference, [reference, repeats[0]], candidate)

    def test_rejects_byte_identical_rek_capture_at_distinct_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            document = trace_document("rek", 1.0)
            reference = self.load(directory, "rek-original.json", document)
            copy = self.load(directory, "rek-copy.json", document)
            other = self.load(
                directory, "rek-other.json", trace_document("rek", 2.0)
            )
            candidate = self.load(
                directory, "sim.json", trace_document("clone:synthetic", 3.0)
            )
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "share SHA-256"):
                self.compare(reference, [copy, other], candidate)

    def test_rejects_distinct_files_that_share_a_run_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self.load(
                directory, "rek-0.json",
                trace_document("rek", 1.0, run_id="same-server-run"),
            )
            repeat_a = self.load(
                directory, "rek-1.json",
                trace_document("rek", 2.0, run_id="same-server-run"),
            )
            repeat_b = self.load(
                directory, "rek-2.json", trace_document("rek", 3.0)
            )
            candidate = self.load(
                directory, "sim.json", trace_document("clone:synthetic", 4.0)
            )
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "share run_id"):
                self.compare(reference, [repeat_a, repeat_b], candidate)

    def test_sparse_trace_is_not_interpolated_onto_acceptance_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, _ = self.exact_inputs(directory)
            sparse = trace_document("clone:synthetic", 10.0)
            sparse["samples"] = sparse["samples"][::2]
            candidate = self.load(directory, "sparse-sim.json", sparse)
            report = self.compare(reference, repeats, candidate)
        self.assertIsNone(report["verdict"]["passed"])
        self.assertIn(
            "samples_not_measured_on_common_grid", report["verdict"]["blockers"]
        )
        self.assertEqual(
            report["interpolation"]["candidate"]["interpolated_sample_count"], 0
        )
        self.assertGreater(
            len(report["interpolation"]["candidate"]["missing_relative_times_s"]),
            0,
        )

    def test_acceptance_requires_predeclared_timestamp_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, candidate = self.exact_inputs(directory)
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "must be declared explicitly"):
                paired.compare_motion(reference, repeats, candidate)

    def test_rejects_uncertainty_larger_than_half_grid_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, candidate = self.exact_inputs(directory)
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "exceeds half"):
                paired.compare_motion(
                    reference, repeats, candidate,
                    maximum_timestamp_uncertainty_s=0.011,
                )

    def test_rejects_shear_as_coordinate_calibration(self):
        document = {
            "schema": paired.CALIBRATION_SCHEMA,
            "mode": "explicit_similarity_3d",
            "position_matrix": [[1, 0.25, 0], [0, 1, 0], [0, 0, 1]],
            "position_offset": [0, 0, 0],
            "source_forward_vector": [1, 0, 0],
            "direct_yaw_sign": 1,
            "direct_yaw_offset_rad": 0,
            "provenance": "synthetic invalid transform",
        }
        with self.assertRaisesRegex(paired.MotionComparisonError,
                                    "nonuniform scale|shear"):
            paired.parse_calibration(document)

    def test_loads_canonical_binary_trace_with_command_edge(self):
        from trace import TraceWriter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measured.trace"
            channels = [
                "root.0.pos.x", "root.0.pos.y", "root.0.pos.z",
                "root.0.quat.x", "root.0.quat.y", "root.0.quat.z",
                "root.0.quat.w", "screen.0.root.x", "screen.0.root.y",
            ]
            citation = {
                channel: {"kind": "controlled_experiment", "ref": "synthetic"}
                for channel in channels
            }
            with TraceWriter(
                path, channels, BUILD, "rek", provenance=citation,
                fixed_delta_time=0.02,
                command_execution_state="measured_executed",
                arena_calibration=identity_calibration(),
                screen_frame=SCREEN_FRAME,
            ) as writer:
                for tick in range(49, 54):
                    relative_time = (tick - 50) * 0.02
                    state = arena_state(relative_time)
                    writer.append(tick, {
                        "root.0.pos.x": state["position"][0],
                        "root.0.pos.y": state["position"][1],
                        "root.0.pos.z": state["position"][2],
                        "root.0.quat.x": 0,
                        "root.0.quat.y": 0,
                        "root.0.quat.z": 0,
                        "root.0.quat.w": 1,
                        "screen.0.root.x": state["screen"][0],
                        "screen.0.root.y": state["screen"][1],
                    })
                writer.event(
                    50, "command_edge",
                    command_identity="walk_forward:press:v1",
                )
            loaded = paired.load_motion_trace(path)
        self.assertEqual(loaded.input_format, "REKTRACE.v1")
        self.assertEqual(loaded.execution_state, "measured_executed")
        self.assertEqual(loaded.command_edge_time_s, 1.0)
        for actual, expected in zip(loaded.relative_times, TIMES):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(loaded.samples[0].screen, arena_state(TIMES[0])["screen"])

    def test_canned_window_adapter_preserves_unknown_execution_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(identity_calibration()), encoding="utf-8"
            )
            samples = []
            for relative_time in TIMES:
                state = arena_state(relative_time)
                samples.append({
                    "relative_time_s": relative_time,
                    "fighter": {
                        "root_position": list(state["position"]),
                        "root_rotation": [0, 0, 0, 1],
                    },
                })
            document = {
                "schema": paired.CANNED_SCHEMA,
                "source": {
                    "game_assembly_sha256": "a" * 64,
                    "global_metadata_sha256": "b" * 64,
                },
                "windows": [{
                    "request": {"move_name": "walk_forward:press:v1"},
                    "executed": {"state": "unknown"},
                    "samples": samples,
                }],
            }
            path = root / "window.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = paired.load_motion_trace(
                path, calibration_path=calibration_path, window_index=0
            )
        self.assertEqual(loaded.input_format, paired.CANNED_SCHEMA)
        self.assertEqual(loaded.execution_state, "unknown")
        self.assertEqual(loaded.command_edge_time_s, 0.0)
        self.assertIn("not server acceptance or execution", loaded.adapter_notes[0])
        self.assertIsNotNone(loaded.calibration.source_sha256)

    def test_report_writer_refuses_to_replace_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("owned", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                paired._write_json_exclusive(output, {"test": True})
            self.assertEqual(output.read_text(encoding="utf-8"), "owned")

    def test_report_serialization_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, repeats, candidate = self.exact_inputs(directory)
            report = self.compare(
                reference, repeats, candidate, accept_at="max",
                require_screen=True,
            )
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            paired._write_json_exclusive(first, report)
            paired._write_json_exclusive(second, report)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(paired._sha256(first), paired._sha256(second))

    def test_video_anchor_rejects_more_than_half_frame_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, _, _ = self.exact_inputs(directory)
            video = root / "synthetic-video.bin"
            video.write_text("synthetic", encoding="utf-8")
            anchor = self.write_video_anchor(
                root, "bad.anchor.json", video, reference,
                uncertainty=0.010001,
            )
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "exceeds half"):
                paired._load_video_clock_anchor(
                    anchor, video_path=video, trace=reference,
                    source_frame_rate_hz=50.0,
                )

    def test_video_anchor_binds_exact_video_and_trace_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, _, candidate = self.exact_inputs(directory)
            video = root / "synthetic-video.bin"
            video.write_text("synthetic", encoding="utf-8")
            anchor = self.write_video_anchor(
                root, "anchor.json", video, reference
            )
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "trace_sha256"):
                paired._load_video_clock_anchor(
                    anchor, video_path=video, trace=candidate,
                    source_frame_rate_hz=50.0,
                )
            video.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(
                    paired.MotionComparisonError, "video_sha256"):
                paired._load_video_clock_anchor(
                    anchor, video_path=video, trace=reference,
                    source_frame_rate_hz=50.0,
                )

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are not installed")
    def test_renders_command_edge_aligned_side_by_side_and_overlay_mp4s(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, repeats, candidate = self.exact_inputs(directory)
            report = self.compare(
                reference, repeats, candidate, accept_at="max",
                require_screen=True,
            )
            reference_video = root / "rek.mp4"
            candidate_video = root / "sim.mp4"
            for color, output in (("red", reference_video),
                                  ("blue", candidate_video)):
                command = [
                    shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i",
                    f"color=c={color}:s=64x48:r=50:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
                ]
                subprocess.run(command, check=True)
            reference_anchor = self.write_video_anchor(
                root, "rek.anchor.json", reference_video, reference
            )
            candidate_anchor = self.write_video_anchor(
                root, "sim.anchor.json", candidate_video, candidate
            )
            videos = []
            for layout in ("side-by-side", "overlay"):
                videos.append(paired.render_aligned_video(
                    report, reference, candidate,
                    rek_video=reference_video,
                    candidate_video=candidate_video,
                    output=root / f"paired-{layout}.mp4",
                    layout=layout,
                    fps=50.0,
                    rek_video_anchor=reference_anchor,
                    candidate_video_anchor=candidate_anchor,
                ))
        for video in videos:
            self.assertEqual(video["status"], "rendered")
            self.assertEqual(video["evidence_grade"], "acceptance")
            self.assertTrue(video["supports_parity_verdict"])
            self.assertEqual(video["alignment_basis"],
                             "measured_video_clock_anchors")
            self.assertEqual(video["frame_count"], len(TIMES))
            self.assertEqual(video["output"]["probe"]["frame_count"], len(TIMES))
            self.assertEqual(len(video["output"]["sha256"]), 64)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are not installed")
    def test_manual_video_offsets_are_diagnostic_and_cannot_support_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, repeats, candidate = self.exact_inputs(directory)
            report = self.compare(
                reference, repeats, candidate, accept_at="max",
                require_screen=True,
            )
            videos = []
            for color, name in (("red", "rek.mp4"), ("blue", "sim.mp4")):
                path = root / name
                subprocess.run([
                    shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c={color}:s=64x48:r=50:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
                ], check=True)
                videos.append(path)
            report["video"] = paired.render_aligned_video(
                report, reference, candidate,
                rek_video=videos[0], candidate_video=videos[1],
                output=root / "diagnostic.mp4", fps=50.0,
                rek_video_edge_s=0.2, candidate_video_edge_s=0.2,
            )
            paired._apply_video_verdict_gate(report)
        self.assertEqual(report["video"]["evidence_grade"], "diagnostic_only")
        self.assertFalse(report["video"]["supports_parity_verdict"])
        self.assertEqual(report["verdict"]["state"], "insufficient_evidence")
        self.assertIsNone(report["verdict"]["passed"])
        self.assertIn("video_clock_anchor_not_measured",
                      report["verdict"]["blockers"])


if __name__ == "__main__":
    unittest.main()
