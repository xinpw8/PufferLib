import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import export_motion_trace as exporter


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
RUNTIME["identity_sha256"] = exporter._canonical_sha256(RUNTIME)
CANONICAL_STATE_AUTHORITY = {
    "kind": "engineai_native_sdk_model_and_walking_configuration",
    "sdk_commit": exporter.ENGINEAI_SDK_COMMIT,
    "model_path": "assets/resource/t800.xml",
    "walking_configuration_path": (
        "assets/config/t800/rl_walking_example/default.yaml"
    ),
    "default_joint_q_sha256": exporter._array_sha256(exporter.DEFAULT_Q),
    "root_clearance_above_support_m": 1.03,
    "root_orientation": "sdk_upright_roll_pitch_with_retained_arena_heading",
    "root_and_joint_velocities": "zero",
    "support_geometry": exporter.ARENA_SUPPORT_GEOM_NAME,
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
    "identity_sha256": exporter._canonical_sha256(RESET_IDENTITY_MATERIAL),
    "identity_material": RESET_IDENTITY_MATERIAL,
    "procedure_steps": json.loads(json.dumps(exporter.RESET_PROCEDURE_STEPS)),
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
            self.data.time = self.physics_tick * exporter.PHYSICS_DT
            observe(self.snapshot(phase, command))
        self.control_tick += 1


def make_trace(
    command="forward", capture_id="capture-1", run_id="run-1", runner=None
):
    runner = runner or FakeRunner()
    return exporter.generate_trace(
        runner,
        exporter.COMMANDS[command],
        build_fingerprint=BUILD,
        capture_id=capture_id,
        run_id=run_id,
        artifact_provenance=PROVENANCE,
        calibration=CALIBRATION,
    )


class ExportMotionTraceTests(unittest.TestCase):
    def test_explicit_simulator_calibration_is_exactly_hash_pinned(self):
        calibration_path = (
            Path(__file__).resolve().parents[1]
            / "rek"
            / "evidence"
            / "mujoco_arena_identity_calibration.json"
        )
        self.assertEqual(
            exporter.sha256_file(calibration_path),
            exporter.EXPECTED_CALIBRATION_SHA256,
        )
        self.assertEqual(
            json.loads(calibration_path.read_text(encoding="utf-8")),
            CALIBRATION,
        )

    def test_canonical_command_identities_and_signs(self):
        expected = {
            "forward": ("walk_forward:press:v1", (1.0, 0.0, 0.0)),
            "backward": ("walk_backward:press:v1", (-1.0, 0.0, 0.0)),
            "strafe-left": ("strafe_left:press:v1", (0.0, -1.0, 0.0)),
            "strafe-right": ("strafe_right:press:v1", (0.0, 1.0, 0.0)),
            "yaw-left": ("yaw_left:press:v1", (0.0, 0.0, -1.0)),
            "yaw-right": ("yaw_right:press:v1", (0.0, 0.0, 1.0)),
        }
        self.assertEqual(set(exporter.COMMANDS), set(expected))
        for name, (identity, command) in expected.items():
            self.assertEqual(exporter.COMMANDS[name].identity, identity)
            self.assertEqual(exporter.COMMANDS[name].normalized_command, command)

    def test_trace_has_exact_500_hz_samples_and_both_roots(self):
        trace = make_trace()
        self.assertEqual(trace["source"], "clone:rek_fight_engineai")
        self.assertEqual(trace["calibration"], CALIBRATION)
        self.assertEqual(trace["runtime_provenance"], RUNTIME)
        self.assertEqual(trace["command"]["edge_time_s"], 1.0)
        self.assertEqual(trace["command"]["edge_trial_tick"], 50)
        self.assertEqual(trace["command"]["edge_physics_tick"], 500)
        self.assertEqual(trace["command"]["release_time_s"], 3.0)
        self.assertEqual(trace["command"]["release_trial_tick"], 150)
        self.assertEqual(trace["command"]["release_physics_tick"], 1500)
        self.assertEqual(len(trace["samples"]), 2001)
        self.assertEqual(
            [sample["physics_tick"] for sample in trace["samples"]],
            list(range(2001)),
        )
        self.assertTrue(
            all(len(sample["fighter_roots"]) == 2 for sample in trace["samples"])
        )
        for left, right in zip(trace["samples"], trace["samples"][1:]):
            self.assertAlmostEqual(right["time_s"] - left["time_s"], 0.002)
        self.assertIn("neutral_pre_roll", trace["samples"][499]["control_phase"])
        self.assertIn("neutral_pre_roll", trace["samples"][500]["control_phase"])
        self.assertIn("selected_command_held", trace["samples"][501]["control_phase"])
        self.assertIn("selected_command_held", trace["samples"][1500]["control_phase"])
        self.assertIn("neutral_release", trace["samples"][1501]["control_phase"])
        self.assertFalse(trace["claims"]["parity_demonstrated"])
        self.assertFalse(trace["claims"]["combat_trajectory_present"])
        self.assertFalse(trace["claims"]["screen_coordinates_present"])
        self.assertTrue(
            trace["claims"]["selected_command_started_from_shared_pinned_reset"]
        )
        self.assertEqual(
            trace["claims"]["controlled_prior_non_neutral_command_count"], 0
        )

    def test_each_command_is_one_press_from_the_same_zero_history(self):
        reset_hashes = set()
        for name, spec in exporter.COMMANDS.items():
            with self.subTest(command=name):
                runner = FakeRunner()
                trace = make_trace(name, runner=runner)
                reset_hashes.add(trace["reset"]["identity_sha256"])
                self.assertEqual(trace["samples"][0]["physics_tick"], 0)
                self.assertEqual(trace["samples"][-1]["physics_tick"], 2000)
                self.assertEqual(len(runner.command_history), 400)
                self.assertTrue(
                    all(
                        entry[1] == exporter.NEUTRAL_COMMAND
                        for entry in runner.command_history[:100]
                    )
                )
                self.assertTrue(
                    all(
                        entry[1] == spec.normalized_command
                        for entry in runner.command_history[100:300]
                    )
                )
                self.assertTrue(
                    all(
                        entry[1] == exporter.NEUTRAL_COMMAND
                        for entry in runner.command_history[300:]
                    )
                )
                self.assertEqual(trace["trial"]["prior_non_neutral_command_count"], 0)
                self.assertEqual(trace["trial"]["non_neutral_press_count"], 1)
                self.assertEqual(trace["trial"]["attack_command_count"], 0)
        self.assertEqual(reset_hashes, {RESET["identity_sha256"]})

    def test_two_fresh_trials_have_identical_post_reset_state_hashes(self):
        first = make_trace("forward", runner=FakeRunner())
        second = make_trace("yaw-right", runner=FakeRunner())
        self.assertEqual(
            first["reset"]["measured_state"]["aggregate_state_sha256"],
            second["reset"]["measured_state"]["aggregate_state_sha256"],
        )
        self.assertEqual(
            first["reset"]["identity_sha256"],
            second["reset"]["identity_sha256"],
        )

    def test_trace_loads_in_paired_motion_comparator_without_screen(self):
        comparator_path = (
            Path(__file__).resolve().parents[1]
            / "rek"
            / "evidence"
            / "paired_motion_compare.py"
        )
        if not comparator_path.is_file():
            self.skipTest("paired_motion_compare.py is not present in this checkout")
        specification = importlib.util.spec_from_file_location(
            "paired_motion_compare_for_export_test", comparator_path
        )
        module = importlib.util.module_from_spec(specification)
        assert specification.loader is not None
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(make_trace()), encoding="utf-8")
            loaded = module.load_motion_trace(path)
        self.assertEqual(loaded.source, exporter.SOURCE)
        self.assertEqual(loaded.build_fingerprint, BUILD)
        self.assertEqual(loaded.capture_id, "capture-1")
        self.assertEqual(loaded.run_id, "run-1")
        self.assertEqual(loaded.command_identity, "walk_forward:press:v1")
        self.assertEqual(len(loaded.samples), 2001)
        self.assertIsNone(loaded.screen_frame)

    def test_exclusive_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            exporter.write_json_exclusive(path, make_trace())
            with self.assertRaises(FileExistsError):
                exporter.write_json_exclusive(path, make_trace())

    def test_ids_change_trace_content(self):
        left = json.dumps(
            make_trace(capture_id="capture-a", run_id="run-a"), sort_keys=True
        )
        right = json.dumps(
            make_trace(capture_id="capture-b", run_id="run-b"), sort_keys=True
        )
        self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()
