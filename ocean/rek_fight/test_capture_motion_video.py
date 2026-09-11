import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import capture_motion_video as capture
import export_motion_trace as motion


BUILD = "measured-rek-build-fingerprint"
PROVENANCE = {
    "model": {"path": "/fixture/model.xml", "sha256": "a" * 64},
}
CALIBRATION = {
    "schema": "rek.paired_motion_calibration.v1",
    "mode": "identity",
    "source_forward_vector": [1.0, 0.0, 0.0],
    "provenance": {
        "kind": "pinned_mujoco_arena_coordinate_definition",
        "artifact": "evidence_out/t800_t800_factory_arena.diagnostic.xml",
        "artifact_sha256": (
            "01caa6ed90277a90fc71b4c16d7959fb"
            "70f7702fa15bea7c68d3eaa9d5f27b2c"
        ),
        "source_report": (
            "evidence_out/t800_t800_factory_arena.diagnostic.report.json"
        ),
        "source_report_sha256": (
            "fac3024f118eb3d8a61436b9a7c160a2"
            "1e9062ec5591011deb6260169531427e"
        ),
    },
}
RUNTIME = {
    "schema": "rek.clone.runtime_provenance.v1",
    "python": {"version": "fixture"},
    "numpy": {"versions": {"module_api": "fixture"}},
    "mujoco": {"versions": {"native_runtime_api": "fixture"}},
    "mnn": {"versions": {"python_distribution": "fixture"}},
}
RUNTIME["identity_sha256"] = motion._canonical_sha256(RUNTIME)
CANONICAL_STATE_AUTHORITY = {
    "kind": "engineai_native_sdk_model_and_walking_configuration",
    "sdk_commit": motion.ENGINEAI_SDK_COMMIT,
    "model_path": "assets/resource/t800.xml",
    "walking_configuration_path": (
        "assets/config/t800/rl_walking_example/default.yaml"
    ),
    "default_joint_q_sha256": motion._array_sha256(motion.DEFAULT_Q),
    "root_clearance_above_support_m": 1.03,
    "root_orientation": "sdk_upright_roll_pitch_with_retained_arena_heading",
    "root_and_joint_velocities": "zero",
    "support_geometry": motion.ARENA_SUPPORT_GEOM_NAME,
    "support_height_m": 0.01,
}
RESET_IDENTITY_MATERIAL = {
    "model_sha256": "a" * 64,
    "keyframe_id": 0,
    "keyframe_name": "client_observed_round2_first_active",
    "keyframe_qpos_sha256_before_joint_override": "b" * 64,
    "composite_qpos_sha256_after_joint_override": "c" * 64,
    "physics_timestep_s": 0.002,
    "procedure": (
        "mj_resetDataKeyframe_then_retain_arena_xy_heading_and_apply_"
        "per_fighter_engineai_sdk_standing_state_"
        "and_walking_controller_reset_then_mj_forward_v1"
    ),
    "aggregate_state_sha256": "d" * 64,
    "walking_controller_state_sha256": ["e" * 64, "e" * 64],
    "canonical_state_authority": CANONICAL_STATE_AUTHORITY,
}
RESET = {
    "schema": "rek.clone.composite_reset.v1",
    "identity_sha256": motion._canonical_sha256(RESET_IDENTITY_MATERIAL),
    "identity_material": RESET_IDENTITY_MATERIAL,
    "procedure_steps": json.loads(json.dumps(motion.RESET_PROCEDURE_STEPS)),
    "measured_state": {
        "aggregate_state_sha256": "d" * 64,
        "qpos_sha256": "c" * 64,
        "qvel_sha256": "f" * 64,
        "ctrl_sha256": "0" * 64,
    },
    "untouched_xml_keyframe": False,
    "keyframe_joint_pose_overridden_by_engineai_default_q": True,
    "keyframe_root_state_overridden_by_engineai_sdk_model": True,
    "keyframe_arena_xy_and_heading_retained": True,
    "canonical_state_authority": CANONICAL_STATE_AUTHORITY,
}


class FakeData:
    def __init__(self):
        self.time = 0.0


class FakeRunner:
    substeps = 5

    def __init__(self):
        self.data = FakeData()
        self.runtime_provenance = RUNTIME
        self.reset()

    def reset(self):
        self.data.time = 0.0
        self.physics_tick = 0
        self.control_tick = 0
        self.command_history = []
        self._reset_record = json.loads(json.dumps(RESET))

    def reset_record(self):
        return json.loads(json.dumps(self._reset_record))

    def snapshot(self, phase, normalized_command):
        command = tuple(float(value) for value in normalized_command)
        roots = [
            {
                "fighter_index": fighter,
                "role": "controlled" if fighter == 0 else "approach_dummy",
                "root_position": [
                    self.data.time + fighter,
                    command[1],
                    1.0,
                ],
                "root_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "recovering": False,
            }
            for fighter in range(2)
        ]
        return {
            "time_s": self.data.time,
            "physics_tick": self.physics_tick,
            "control_tick": self.control_tick,
            "control_phase": phase,
            "normalized_command": list(command),
            "root_position": roots[0]["root_position"],
            "root_quaternion_xyzw": roots[0]["root_quaternion_xyzw"],
            "fighter_roots": roots,
        }

    def step_control(self, command, phase, observe):
        self.command_history.append(
            (self.control_tick, tuple(float(value) for value in command), phase)
        )
        for _ in range(self.substeps):
            self.physics_tick += 1
            self.data.time = self.physics_tick * motion.PHYSICS_DT
            observe(self.snapshot(phase, command))
        self.control_tick += 1


class FakeRenderer:
    width = 16
    height = 12
    camera_contract = {
        "type": "fixed_fixture_camera",
        "lookat_xyz": [0.0, 0.0, 1.0],
        "distance_m": 6.2,
        "azimuth_deg": 90.0,
        "elevation_deg": -42.0,
        "width_px": width,
        "height_px": height,
    }

    def render(self, runner, marker_post):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:, :, 0] = runner.physics_tick % 251
        image[:, :, 1] = 17
        image[:, :, 2] = 29
        x, y, width, height = capture.MARKER_REGION
        image[y : y + height, x : x + width] = (
            capture.MARKER_POST_RGB
            if marker_post
            else capture.MARKER_PRE_RGB
        )
        return image


class StaticWorldRenderer:
    width = 16
    height = 12
    camera_contract = {
        "type": "fixed_static_world_fixture_camera",
        "lookat_xyz": [0.0, 0.0, 1.0],
        "distance_m": 6.2,
        "azimuth_deg": 90.0,
        "elevation_deg": -42.0,
        "width_px": width,
        "height_px": height,
    }
    static_world_region = (slice(0, height), slice(8, width))

    def render(self, runner, marker_post):
        del runner
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[self.static_world_region] = (47, 47, 47)
        image[2:10, 10:14] = (51, 51, 51)
        x, y, width, height = capture.MARKER_REGION
        image[y : y + height, x : x + width] = (
            capture.MARKER_POST_RGB
            if marker_post
            else capture.MARKER_PRE_RGB
        )
        return image


def fake_encoder(stage, frame_count, width, height):
    capture._atomic_write(
        stage / "capture.mp4",
        f"fixture MP4 {frame_count} {width} {height}\n".encode("ascii"),
    )
    return {
        "mode": "deterministic_test_fixture",
        "probe": {
            "width": width,
            "height": height,
            "avg_frame_rate": f"{capture.VIDEO_FPS}/1",
            "nb_read_frames": str(frame_count),
        },
        "returncode": 0,
    }


def load_video_clock_anchor():
    path = (
        Path(__file__).resolve().parents[1]
        / "rek"
        / "evidence"
        / "video_clock_anchor.py"
    )
    specification = importlib.util.spec_from_file_location(
        "video_clock_anchor_for_sim_capture_test", path
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class CaptureMotionVideoTests(unittest.TestCase):
    def capture_backward(self, root, renderer=None):
        runner = FakeRunner()
        output = root / "backward-capture"
        manifest = capture.capture_motion(
            runner,
            renderer or FakeRenderer(),
            motion.COMMANDS["backward"],
            output_dir=output,
            build_fingerprint=BUILD,
            capture_id="sim-capture-backward-001",
            run_id="trial-run-001",
            artifact_provenance=PROVENANCE,
            calibration=CALIBRATION,
            producer_path=Path(capture.__file__),
            encoder=fake_encoder,
        )
        return runner, output, manifest

    def test_static_world_pixels_cannot_alternate_between_frames(self):
        renderer = StaticWorldRenderer()
        with tempfile.TemporaryDirectory() as temporary:
            _, output, _ = self.capture_backward(Path(temporary), renderer)
            frames = []
            for index in range(48, 55):
                with Image.open(output / "frames" / f"frame-{index:06d}.png") as image:
                    frames.append(np.asarray(image.convert("RGB")).copy())

            static_world = [
                frame[renderer.static_world_region]
                for frame in frames
            ]
            for observed in static_world[1:]:
                np.testing.assert_array_equal(observed, static_world[0])
            self.assertTrue(np.all(static_world[0][:, :2] == 47))
            self.assertTrue(np.all(static_world[0][2:10, 2:6] == 51))

    def test_same_pass_trace_and_video_are_single_press_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner, output, manifest = self.capture_backward(Path(temporary))
            trace = json.loads(
                (output / "motion.trace.json").read_text(encoding="utf-8")
            )
            published = json.loads(
                (output / "capture.manifest.json").read_text(encoding="utf-8")
            )
            contract = json.loads(
                (output / "marker.contract.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest, published)
            self.assertEqual(
                published["schema"], "rek.simulator_frame_capture.v1"
            )
            self.assertEqual(trace["calibration"], CALIBRATION)
            self.assertFalse(trace["claims"]["parity_demonstrated"])
            self.assertTrue(
                trace["claims"]["selected_command_started_from_shared_pinned_reset"]
            )
            self.assertEqual(
                trace["claims"]["controlled_prior_non_neutral_command_count"],
                0,
            )
            self.assertEqual(trace["command"]["edge_trial_tick"], 50)
            self.assertEqual(trace["samples"][0]["physics_tick"], 0)
            self.assertEqual(trace["samples"][-1]["physics_tick"], 2000)
            self.assertEqual(len(trace["samples"]), 2001)
            self.assertTrue(
                all(
                    entry[1] == motion.NEUTRAL_COMMAND
                    for entry in runner.command_history[:100]
                )
            )
            self.assertTrue(
                all(
                    entry[1] == (-1.0, 0.0, 0.0)
                    for entry in runner.command_history[100:300]
                )
            )
            self.assertTrue(
                all(
                    entry[1] == motion.NEUTRAL_COMMAND
                    for entry in runner.command_history[300:]
                )
            )

            frames = published["frames"]
            self.assertEqual(len(frames), 201)
            self.assertEqual(
                [record["index"] for record in frames], list(range(201))
            )
            self.assertEqual(
                [record["timestamp_ns"] for record in frames],
                [index * 20_000_000 for index in range(201)],
            )
            self.assertEqual(
                [record["simulation_physics_tick"] for record in frames],
                list(range(0, 2001, 10)),
            )
            self.assertEqual(
                [record["trace_sample_index"] for record in frames],
                list(range(0, 2001, 10)),
            )
            self.assertEqual(frames[50]["marker_state"], "pre")
            self.assertTrue(
                all(record["marker_state"] == "post" for record in frames[51:])
            )
            self.assertEqual(
                contract["trace"]["sha256"],
                motion.sha256_file(output / "motion.trace.json"),
            )
            self.assertEqual(
                contract["producer"]["sha256"],
                published["binding"]["producer_sha256"],
            )
            self.assertEqual(
                published["binding"]["marker_contract_schema"],
                "rek.rendered_command_marker.v1",
            )
            self.assertEqual(
                published["binding"]["marker_contract_sha256"],
                motion.sha256_file(output / "marker.contract.json"),
            )
            self.assertEqual(
                published["binding"]["trial_protocol_sha256"],
                motion.TRIAL_PROTOCOL_SHA256,
            )
            self.assertEqual(
                published["binding"]["reset_identity_sha256"],
                trace["reset"]["identity_sha256"],
            )
            self.assertEqual(
                published["binding"]["runtime_provenance_sha256"],
                trace["runtime_provenance"]["identity_sha256"],
            )
            self.assertEqual(
                published["result"]["frame_set_sha256"],
                capture._frame_set_sha256(frames),
            )
            self.assertEqual(
                published["result"]["encoded_video_sha256"],
                motion.sha256_file(output / "capture.mp4"),
            )

    def test_strict_anchor_verifier_accepts_capture_at_half_frame_bound(self):
        anchor = load_video_clock_anchor()
        with tempfile.TemporaryDirectory() as temporary:
            _, output, _ = self.capture_backward(Path(temporary))
            anchor_path = output / "video-clock-anchor.json"
            result = anchor.create_anchor(
                output,
                output / "marker.contract.json",
                anchor_path,
            )
            self.assertEqual(result["schema"], "rek.video_clock_anchor.v1")
            self.assertEqual(
                result["source_capture"]["capture_schema"],
                "rek.simulator_frame_capture.v1",
            )
            self.assertEqual(
                result["command_identity"], "walk_backward:press:v1"
            )
            self.assertEqual(result["schedule_run_id"], "trial-run-001")
            self.assertAlmostEqual(result["command_edge_video_pts_s"], 1.01)
            self.assertAlmostEqual(result["measurement"]["uncertainty_s"], 0.01)
            self.assertEqual(
                result["marker_observation"]["last_pre_frame_index"], 50
            )
            self.assertEqual(
                result["marker_observation"]["first_post_frame_index"], 51
            )

    def test_capture_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.capture_backward(root)
            with self.assertRaises(FileExistsError):
                self.capture_backward(root)


if __name__ == "__main__":
    unittest.main()
