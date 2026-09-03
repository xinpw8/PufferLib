import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import motion_parity_suite as suite
import client_fixed_import as importer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.inputs = root / "inputs"
        self.inputs.mkdir()
        self.model = self.inputs / "arena.xml"
        self.model.write_bytes(b"measured pinned arena")
        mapping = {
            "mujoco_position_xyz": ["x", "z", "y"],
            "mujoco_quaternion_wxyz": ["-w", "x", "z", "y"],
        }
        self.mapping_report = self.inputs / "arena-report.json"
        write_json(
            self.mapping_report,
            {
                "schema": "fixture.measured_arena.v1",
                "unity_mapping": {**mapping, "source": {"fixture": True}},
                "mjcf_sha256": sha256(self.model),
            },
        )
        self.rek_calibration = self.inputs / "rek-calibration.json"
        write_json(
            self.rek_calibration,
            {
                "schema": suite.CALIBRATION_SCHEMA,
                "mode": "explicit_similarity_3d",
                "position_matrix": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                "position_offset": [0, 0, 0],
                "source_forward_vector": [1, 0, 0],
                "direct_yaw_sign": -1,
                "direct_yaw_offset_rad": 0,
                "provenance": {
                    "kind": "pinned_unity_to_mujoco_coordinate_conversion",
                    "artifact": self.mapping_report.name,
                    "artifact_sha256": sha256(self.mapping_report),
                    "mapping": mapping,
                },
            },
        )
        self.sim_calibration = self.inputs / "sim-calibration.json"
        self.sim_calibration_document = {
            "schema": suite.CALIBRATION_SCHEMA,
            "mode": "identity",
            "source_forward_vector": [1, 0, 0],
            "provenance": {
                "kind": "pinned_mujoco_arena_coordinate_definition",
                "artifact": self.model.name,
                "artifact_sha256": sha256(self.model),
                "source_report": self.mapping_report.name,
                "source_report_sha256": sha256(self.mapping_report),
            },
        }
        write_json(self.sim_calibration, self.sim_calibration_document)
        self.inventory = self.inputs / "inventory.json"
        self.schedule = self.inputs / "schedule.json"
        write_json(self.inventory, {"fixture": "inventory"})
        self.schedule_document = {
            "schedule_id": "rek.private_bot1.baseline.v1",
            "schedule_rate_hz": 50,
            "fixed_substeps_per_tick": 10,
            "velocity_segments": [
                {"start": 0, "stop": 50, "velocity_command": [0, 0, 0]},
                {"start": 50, "stop": 150, "velocity_command": [1, 0, 0]},
                {"start": 150, "stop": 200, "velocity_command": [0, 0, 0]},
                {"start": 200, "stop": 300, "velocity_command": [-1, 0, 0]},
                {"start": 300, "stop": 350, "velocity_command": [0, 0, 0]},
                {"start": 350, "stop": 450, "velocity_command": [0, -1, 0]},
                {"start": 450, "stop": 500, "velocity_command": [0, 0, 0]},
                {"start": 500, "stop": 600, "velocity_command": [0, 1, 0]},
                {"start": 600, "stop": 650, "velocity_command": [0, 0, 0]},
                {"start": 650, "stop": 750, "velocity_command": [0, 0, -1]},
                {"start": 750, "stop": 800, "velocity_command": [0, 0, 0]},
                {"start": 800, "stop": 900, "velocity_command": [0, 0, 1]},
                {"start": 900, "stop": 2601, "velocity_command": [0, 0, 0]},
            ],
        }
        write_json(self.schedule, self.schedule_document)
        self.rek_runs = []
        for index in range(3):
            raw = self.inputs / f"raw-{index}.jsonl"
            transcript = self.inputs / f"transcript-{index}.jsonl"
            raw.write_text(f'{{"fixture_raw":{index}}}\n', encoding="utf-8")
            transcript.write_text(
                f'{{"fixture_transcript":{index}}}\n', encoding="utf-8"
            )
            self.rek_runs.append(suite.RekRun(raw, transcript))

        self.walking = self.inputs / "walking.onnx"
        self.recovery = self.inputs / "recovery.onnx"
        self.trajectory = self.inputs / "recovery.csv"
        self.walking.write_bytes(b"fixture walking policy")
        self.recovery.write_bytes(b"fixture recovery policy")
        self.trajectory.write_bytes(b"fixture recovery trajectory")
        self.adapter = (
            Path(suite.__file__).resolve().parent.parent.parent
            / "rek_fight"
            / "engineai_t800_policy.py"
        )
        self.exporter = self.adapter.with_name("export_motion_trace.py")
        controller_states = []
        for fighter in (0, 1):
            state = {
                "fighter_index": fighter,
                "history_sha256": f"{fighter + 1:064x}",
                "previous_action_sha256": f"{fighter + 3:064x}",
                "first_observation": True,
                "command_filter_output": None,
            }
            state["state_sha256"] = suite._canonical_sha256(state)
            controller_states.append(state)
        measured_reset = {
            "time_s": 0.0,
            "qpos_count": 64,
            "qpos_sha256": suite.EXPECTED_COMPOSITE_RESET_QPOS_SHA256,
            "qvel_count": 62,
            "qvel_sha256": "5" * 64,
            "act_count": 0,
            "act_sha256": "6" * 64,
            "ctrl_count": 52,
            "ctrl_sha256": "7" * 64,
            "walking_controller_states": controller_states,
            "recovering_flags": [False, False],
            "aggregate_state_sha256": "8" * 64,
        }
        reset_identity = {
            "model_sha256": sha256(self.model),
            "keyframe_id": suite.RESET_KEYFRAME_ID,
            "keyframe_name": suite.RESET_KEYFRAME_NAME,
            "keyframe_qpos_sha256_before_joint_override": (
                suite.EXPECTED_KEYFRAME_QPOS_SHA256
            ),
            "composite_qpos_sha256_after_joint_override": (
                suite.EXPECTED_COMPOSITE_RESET_QPOS_SHA256
            ),
            "physics_timestep_s": 0.002,
            "procedure": (
                "mj_resetDataKeyframe_then_retain_arena_xy_heading_and_apply_"
                "per_fighter_engineai_sdk_standing_state_"
                "and_walking_controller_reset_then_mj_forward_v1"
            ),
            "aggregate_state_sha256": measured_reset["aggregate_state_sha256"],
            "walking_controller_state_sha256": [
                state["state_sha256"] for state in controller_states
            ],
            "canonical_state_authority": (
                suite.EXPECTED_CANONICAL_STATE_AUTHORITY
            ),
        }
        self.reset_document = {
            "schema": suite.RESET_SCHEMA,
            "identity_sha256": suite._canonical_sha256(reset_identity),
            "identity_material": reset_identity,
            "procedure_steps": [
                {"sequence": 1, "operation": "mj_resetDataKeyframe",
                 "keyframe_id": 0, "keyframe_name": suite.RESET_KEYFRAME_NAME},
                {"sequence": 2,
                 "operation": "T800MuJoCoBinding.set_sdk_standing_state",
                 "fighter_index": 0,
                 "arena_placement": "retain_keyframe_xy_and_heading",
                 "state_authority": "engineai_sdk_model"},
                {"sequence": 3, "operation": "T800WalkingController.reset",
                 "fighter_index": 0},
                {"sequence": 4,
                 "operation": "T800MuJoCoBinding.set_sdk_standing_state",
                 "fighter_index": 1,
                 "arena_placement": "retain_keyframe_xy_and_heading",
                 "state_authority": "engineai_sdk_model"},
                {"sequence": 5, "operation": "T800WalkingController.reset",
                 "fighter_index": 1},
                {"sequence": 6, "operation": "set_recovering_flags_false"},
                {"sequence": 7, "operation": "mj_forward"},
                {"sequence": 8, "operation": "set_trial_counters_zero"},
            ],
            "measured_state": measured_reset,
            "untouched_xml_keyframe": False,
            "keyframe_joint_pose_overridden_by_engineai_default_q": True,
            "keyframe_root_state_overridden_by_engineai_sdk_model": True,
            "keyframe_arena_xy_and_heading_retained": True,
            "canonical_state_authority": (
                suite.EXPECTED_CANONICAL_STATE_AUTHORITY
            ),
        }
        self.rek_captures = {}
        self.rek_videos = {}
        self.rek_contracts = {}
        self.sim_traces = {}
        self.sim_captures = {}
        self.sim_videos = {}
        self.sim_contracts = {}
        for motion in suite.MOTIONS:
            rek_contract = self.inputs / f"{motion.slug}-rek-contract.json"
            rek_bound_trace = self.inputs / f"{motion.slug}-rek-bound.rektrace"
            rek_bound_trace.write_bytes(
                f"raw-0.jsonl:{motion.selector}".encode("utf-8")
            )
            rek_producer = self.inputs / f"{motion.slug}-rek-producer.dll"
            rek_producer.write_bytes(f"rek producer {motion.selector}".encode())
            rek_run_id = f"rek-run-{motion.slug}"
            write_json(
                rek_contract,
                {
                    "schema": "rek.rendered_command_marker.v1",
                    "command_identity": motion.command_identity,
                    "schedule_run_id": rek_run_id,
                    "trace": {
                        "path": rek_bound_trace.name,
                        "sha256": sha256(rek_bound_trace),
                    },
                    "producer": {
                        "path": rek_producer.name,
                        "sha256": sha256(rek_producer),
                        "render_binding": (
                            "first_post_marker_frame_is_first_rendered_frame_after_command_edge"
                        ),
                    },
                    "marker": {
                        "transition": "persistent_exact_rgb_rising_edge",
                    },
                },
            )
            self.rek_contracts[motion.selector] = rek_contract
            rek_capture = self.inputs / f"{motion.slug}-rek-capture"
            rek_capture.mkdir()
            rek_video = rek_capture / "capture.mp4"
            rek_video.write_bytes(f"rek video {motion.selector}".encode("utf-8"))
            write_json(
                rek_capture / "capture.manifest.json",
                self.capture_manifest(
                    schema=suite.CAPTURE_SCHEMA,
                    capture_id=f"rek-capture-{motion.slug}",
                    run_id=rek_run_id,
                    command_identity=motion.command_identity,
                    contract=rek_contract,
                    trace=rek_bound_trace,
                    producer=rek_producer,
                    video=rek_video,
                    simulator=False,
                ),
            )
            self.rek_captures[motion.selector] = rek_capture
            self.rek_videos[motion.selector] = rek_video
            sim_trace = self.inputs / f"{motion.slug}-sim-trace.json"
            trial_segments = [
                {
                    "phase": "neutral_pre_roll",
                    "start_trial_tick": 0,
                    "end_trial_tick_exclusive": 50,
                    "normalized_command": [0.0, 0.0, 0.0],
                },
                {
                    "phase": "selected_command_held",
                    "start_trial_tick": 50,
                    "end_trial_tick_exclusive": 150,
                    "normalized_command": list(motion.normalized_command),
                },
                {
                    "phase": "neutral_release",
                    "start_trial_tick": 150,
                    "end_trial_tick_exclusive": 200,
                    "normalized_command": [0.0, 0.0, 0.0],
                },
            ]
            samples = []
            for tick in range(2001):
                if tick <= 500:
                    normalized = [0.0, 0.0, 0.0]
                    phase = "neutral_pre_roll:trial_tick_0"
                elif tick <= 1500:
                    normalized = list(motion.normalized_command)
                    phase = "selected_command_held:trial_tick_50"
                else:
                    normalized = [0.0, 0.0, 0.0]
                    phase = "neutral_release:trial_tick_150"
                samples.append({
                    "physics_tick": tick,
                    "control_tick": 0 if tick == 0 else (tick - 1) // 5,
                    "time_s": tick * 0.002,
                    "control_phase": phase,
                    "normalized_command": normalized,
                    "screen_root_px": [320.0, 180.0],
                })
            write_json(
                sim_trace,
                {
                    "schema": suite.TRACE_SCHEMA,
                    "source": "clone:rek_fight_engineai",
                    "build_fingerprint": "fixture-build-fingerprint",
                    "capture_id": f"sim-capture-{motion.slug}",
                    "schedule_run_id": f"sim-run-{motion.slug}",
                    "command": {
                        "identity": motion.command_identity,
                        "execution_state": "simulated",
                        "semantic_name": motion.simulator_command,
                        "normalized_command": list(motion.normalized_command),
                        "edge_time_s": 1.0,
                        "release_time_s": 3.0,
                        "duration_s": 2.0,
                        "trial_scope": "isolated_same_reset_single_press",
                        "edge_trial_tick": 50,
                        "release_trial_tick": 150,
                        "edge_physics_tick": 500,
                        "release_physics_tick": 1500,
                        "edge_control_tick": 100,
                        "release_control_tick": 300,
                        "identity_provenance": {
                            "kind": "pinned_isolated_single_press_selector",
                            "trial_protocol_identity": suite.TRIAL_PROTOCOL_ID,
                            "trial_protocol_sha256": suite._canonical_sha256(
                                suite.TRIAL_PROTOCOL_DEFINITION
                            ),
                        },
                    },
                    "trial": {
                        "schema": suite.TRIAL_SCHEMA,
                        "protocol": {
                            "identity": suite.TRIAL_PROTOCOL_ID,
                            "sha256": suite._canonical_sha256(
                                suite.TRIAL_PROTOCOL_DEFINITION
                            ),
                            "definition": suite.TRIAL_PROTOCOL_DEFINITION,
                        },
                        "selector": motion.simulator_command,
                        "command_identity": motion.command_identity,
                        "controlled_command_segments": trial_segments,
                        "prior_non_neutral_command_count": 0,
                        "non_neutral_press_count": 1,
                        "attack_command_count": 0,
                    },
                    "reset": self.reset_document,
                    "calibration": self.sim_calibration_document,
                    "timing": {
                        "physics_dt_s": 0.002,
                        "physics_rate_hz": 500.0,
                        "controller_dt_s": 0.01,
                        "controller_rate_hz": 100.0,
                        "physics_substeps_per_controller_step": 5,
                        "sample_timing": "post_mujoco_step_plus_initial_state",
                        "interpolation_used": False,
                        "realtime_pacing_used": False,
                        "trial_rate_hz": 50.0,
                        "trial_started_from_tick": 0,
                        "capture_first_physics_tick": 0,
                        "capture_last_physics_tick": 2000,
                        "capture_pre_edge_s": 1.0,
                        "capture_post_release_s": 1.0,
                    },
                    "claims": {
                        "combat_trajectory_present": False,
                        "screen_coordinates_present": True,
                        "samples_are_measured_simulator_state": True,
                        "sample_interpolation_used": False,
                        "selected_command_started_from_shared_pinned_reset": True,
                        "controlled_prior_non_neutral_command_count": 0,
                        "controlled_non_neutral_press_count": 1,
                        "controlled_post_release_command_is_neutral": True,
                        "rek_attack_trajectories_replayed": False,
                    },
                    "screen_frame": {
                        "id": "fixture-common-camera-640x360",
                        "width_px": 640,
                        "height_px": 360,
                    },
                    "controller": {
                        "controlled_fighter": "official_engineai_walking_policy",
                        "opponent": "official_engineai_walking_policy_approach_dummy",
                        "automatic_getup": "official_engineai_supine_to_stance_policy",
                        "combat_moves_enabled": False,
                    },
                    "artifact_provenance": {
                        "engineai_sdk_commit": suite.ENGINEAI_SDK_COMMIT,
                        "model": {
                            "path": str(self.model),
                            "sha256": sha256(self.model),
                        },
                        "walking_policy": {
                            "path": str(self.walking),
                            "sha256": sha256(self.walking),
                        },
                        "recovery_policy": {
                            "path": str(self.recovery),
                            "sha256": sha256(self.recovery),
                        },
                        "recovery_trajectory": {
                            "path": str(self.trajectory),
                            "sha256": sha256(self.trajectory),
                        },
                        "calibration": {
                            "path": str(self.sim_calibration),
                            "sha256": sha256(self.sim_calibration),
                        },
                        "controller_adapter": {
                            "path": str(self.adapter),
                            "sha256": sha256(self.adapter),
                        },
                        "exporter": {
                            "path": str(self.exporter),
                            "sha256": sha256(self.exporter),
                        },
                    },
                    "samples": samples,
                },
            )
            self.sim_traces[motion.selector] = sim_trace
            capture = self.inputs / f"{motion.slug}-sim-capture"
            capture.mkdir()
            video = capture / "sim.mp4"
            video.write_bytes(f"sim video {motion.selector}".encode("utf-8"))
            contract = self.inputs / f"{motion.slug}-sim-contract.json"
            sim_producer = self.exporter.with_name("capture_motion_video.py")
            write_json(
                contract,
                {
                    "schema": "rek.rendered_command_marker.v1",
                    "command_identity": motion.command_identity,
                    "schedule_run_id": f"sim-run-{motion.slug}",
                    "trace": {
                        "path": sim_trace.name,
                        "sha256": sha256(sim_trace),
                    },
                    "producer": {
                        "path": str(sim_producer),
                        "sha256": sha256(sim_producer),
                        "render_binding": (
                            "first_post_marker_frame_is_first_rendered_frame_after_command_edge"
                        ),
                    },
                    "marker": {
                        "transition": "persistent_exact_rgb_rising_edge",
                    },
                },
            )
            write_json(
                capture / "capture.manifest.json",
                self.capture_manifest(
                    schema=suite.SIM_CAPTURE_SCHEMA,
                    capture_id=f"sim-capture-{motion.slug}",
                    run_id=f"sim-run-{motion.slug}",
                    command_identity=motion.command_identity,
                    contract=contract,
                    trace=sim_trace,
                    producer=sim_producer,
                    video=video,
                    simulator=True,
                ),
            )
            self.sim_captures[motion.selector] = capture
            self.sim_videos[motion.selector] = video
            self.sim_contracts[motion.selector] = contract

    @staticmethod
    def capture_manifest(
        *, schema, capture_id, run_id, command_identity, contract,
        trace, producer, video, simulator
    ):
        request = {
            "fps_numerator": 50,
            "fps_denominator": 1,
            "expected_frame_count": 201,
            "synthetic_frame_duplication": False,
        }
        if simulator:
            request.update(
                {
                    "sample_interpolation_used": False,
                    "frame_selection": "exact_measured_physics_tick_modulo_stride",
                }
            )
        return {
            "schema": schema,
            "status": "complete",
            "capture_id": capture_id,
            "request": request,
            "result": {
                "actual_frame_count": 201,
                "width_px": 640,
                "height_px": 360,
                "encoded_video_sha256": sha256(video),
            },
            "binding": {
                "marker_contract_schema": "rek.rendered_command_marker.v1",
                "marker_contract_sha256": sha256(contract),
                "trace_sha256": sha256(trace),
                "producer_sha256": sha256(producer),
                "capture_id": capture_id,
                "schedule_run_id": run_id,
                "command_identity": command_identity,
                "render_binding": (
                    "first_post_marker_frame_is_first_rendered_frame_after_command_edge"
                ),
            },
            "artifacts": {
                "encoded_video": {
                    "path": video.name,
                    "sha256": sha256(video),
                }
            },
        }

    def config(self, output: Path) -> suite.SuiteConfig:
        return suite.SuiteConfig(
            rek_runs=tuple(self.rek_runs),
            inventory=self.inventory,
            schedule_manifest=self.schedule,
            rek_calibration=self.rek_calibration,
            sim_calibration=self.sim_calibration,
            rek_pilot="0",
            sim_pilot="0",
            rek_captures=self.rek_captures,
            rek_videos=self.rek_videos,
            rek_marker_contracts=self.rek_contracts,
            sim_traces=self.sim_traces,
            sim_captures=self.sim_captures,
            sim_videos=self.sim_videos,
            sim_marker_contracts=self.sim_contracts,
            output_dir=output,
            accept_at="p99",
            start_s=-1.0,
            end_s=3.0,
            comparison_fps=500.0,
            maximum_timestamp_uncertainty_s=0.0,
            screen_required=True,
            video_fps=50.0,
            video_layout="side-by-side",
        )

    def pin_patch(self):
        return mock.patch.multiple(
            suite,
            WALKING_POLICY_SHA256=sha256(self.walking),
            RECOVERY_POLICY_SHA256=sha256(self.recovery),
            RECOVERY_TRAJECTORY_SHA256=sha256(self.trajectory),
        )


class FakeToolchain:
    def __init__(self, fixture: Fixture, *, preflight="passed", anchor_method=None):
        self.fixture = fixture
        self.preflight = preflight
        self.anchor_method = anchor_method or suite.MEASURED_ANCHOR_METHOD
        self.commands = []

    @staticmethod
    def option(command, name):
        return Path(command[command.index(name) + 1])

    def completed(self, command, code=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(command, code, stdout, stderr)

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        tool = Path(command[1]).name
        if tool == "client_fixed_import.py":
            output = self.option(command, "--out")
            raw = self.option(command, "--raw")
            selector = command[command.index("--motion-edge") + 1]
            output.write_bytes(f"{raw.name}:{selector}".encode("utf-8"))
            return self.completed(command, stdout="fixture import complete\n")
        if tool == "video_clock_anchor.py":
            output = self.option(command, "--out")
            motion = next(item for item in suite.MOTIONS if item.slug in str(output))
            capture_dir = self.option(command, "--capture")
            contract_path = self.option(command, "--marker-contract")
            manifest_path = capture_dir / "capture.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if output.name.startswith("rek."):
                trace = output.parent / "rek" / "run-1.rektrace"
                video = self.fixture.rek_videos[motion.selector]
            else:
                trace = self.fixture.sim_traces[motion.selector]
                video = self.fixture.sim_videos[motion.selector]
            write_json(
                output,
                {
                    "schema": suite.ANCHOR_SCHEMA,
                    "trace_sha256": sha256(trace),
                    "video_sha256": sha256(video),
                    "command_identity": motion.command_identity,
                    "run_id": manifest["binding"]["schedule_run_id"],
                    "schedule_run_id": manifest["binding"]["schedule_run_id"],
                    "command_edge_video_pts_s": 1.0,
                    "measurement": {
                        "state": "measured",
                        "method": self.anchor_method,
                        "uncertainty_s": 1 / 120,
                        "provenance": "machine fixture",
                    },
                    "marker_observation": {
                        "transition": suite.MEASURED_MARKER_TRANSITION,
                    },
                    "source_capture": {
                        "capture_id": manifest["capture_id"],
                        "capture_schema": manifest["schema"],
                        "capture_manifest_sha256": sha256(manifest_path),
                    },
                    "capture_binding": {
                        **manifest["binding"],
                        "marker_contract_sha256": sha256(contract_path),
                        "trace_sha256": sha256(trace),
                    },
                    "verified_artifacts": {
                        "published_video": {
                            "path": str(video.resolve()),
                            "sha256": sha256(video),
                        }
                    },
                },
            )
            return self.completed(command, stdout="video clock anchor: measured\n")
        if tool != "paired_motion_compare.py":
            raise AssertionError(tool)
        output = self.option(command, "--out")
        sim_trace = self.option(command, "--sim")
        sim_document = json.loads(sim_trace.read_text(encoding="utf-8"))
        identity = sim_document["command"]["identity"]
        is_video = "--video-out" in command
        state = "passed" if is_video else self.preflight
        passed = {"passed": True, "failed": False, "insufficient_evidence": None}[state]
        blockers = [] if state == "passed" else ["fixture_blocker"]
        video_record = None
        if is_video:
            clip = self.option(command, "--video-out")
            rek_video = self.option(command, "--rek-video")
            sim_video = self.option(command, "--sim-video")
            clip.write_bytes(f"paired {identity}".encode("utf-8"))
            video_record = {
                "status": "rendered",
                "evidence_grade": "acceptance",
                "supports_parity_verdict": True,
                "alignment_basis": "measured_video_clock_anchors",
                "layout": "side-by-side",
                "fps": 50.0,
                "frame_count": 201,
                "reference_video": {
                    "path": str(rek_video.resolve()),
                    "sha256": sha256(rek_video),
                    "probe": {
                        "width": 640,
                        "height": 360,
                        "r_frame_rate": "50/1",
                    },
                },
                "candidate_video": {
                    "path": str(sim_video.resolve()),
                    "sha256": sha256(sim_video),
                    "probe": {
                        "width": 640,
                        "height": 360,
                        "r_frame_rate": "50/1",
                    },
                },
                "output": {
                    "path": str(clip.resolve()),
                    "sha256": sha256(clip),
                    "probe": {"frame_count": 201},
                },
            }
        metrics = {
            "root_position": {"passed": True, "failed_frame_count": 0},
            "root_yaw": {"passed": True, "failed_frame_count": 0},
            "screen_root": {"passed": True, "failed_frame_count": 0},
        }
        rek_paths = [self.option(command, "--rek")]
        for index, value in enumerate(command):
            if value == "--rek-repeat":
                rek_paths.append(Path(command[index + 1]))
        rek_identities = [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "format": "REKTRACE.v1",
                "source": "rek",
                "build_fingerprint": "fixture-build-fingerprint",
                "capture_id": f"rek-capture-{index}",
                "run_id": f"rek-run-{index}",
                "command_identity": identity,
                "execution_state": "measured_executed",
            }
            for index, path in enumerate(rek_paths)
        ]
        start_s = float(command[command.index("--start-s") + 1])
        end_s = float(command[command.index("--end-s") + 1])
        fps = float(command[command.index("--fps") + 1])
        write_json(
            output,
            {
                "schema": suite.COMPARISON_SCHEMA,
                "command_identity": identity,
                "verdict": {"state": state, "passed": passed, "blockers": blockers},
                "alignment": {
                    "source": "explicit_uniform_timebase",
                    "fps": fps,
                    "start_relative_time_s": start_s,
                    "end_relative_time_s": end_s,
                    "frame_count": round((end_s - start_s) * fps) + 1,
                    "interpolation_used": False,
                    "declared_maximum_timestamp_uncertainty_s": 0.0,
                    "sample_selection": (
                        "nearest_distinct_measured_sample_no_interpolation"
                    ),
                },
                "screen_metric": {"required": True, "comparable": True},
                "inputs": {
                    "reference": rek_identities[0],
                    "repeats": rek_identities[1:],
                    "candidate": {
                        "path": str(sim_trace.resolve()),
                        "sha256": sha256(sim_trace),
                        "format": suite.TRACE_SCHEMA,
                        "source": "clone:rek_fight_engineai",
                        "build_fingerprint": "fixture-build-fingerprint",
                        "capture_id": sim_document["capture_id"],
                        "run_id": sim_document["schedule_run_id"],
                        "command_identity": identity,
                        "execution_state": "simulated",
                    },
                },
                "metrics": metrics,
                "video": video_record,
            },
        )
        return self.completed(
            command,
            {"passed": 0, "failed": 1, "insufficient_evidence": 2}[state],
            stdout=f"verdict: {state}\n",
        )


class MotionParitySuiteTests(unittest.TestCase):
    def test_current_production_contracts_are_explicit_fail_closed_blockers(self):
        self.assertEqual(
            importer.V6_COMMAND_EXECUTION_STATE,
            "request_projected_server_execution_unknown",
        )
        self.assertNotEqual(
            importer.V6_COMMAND_EXECUTION_STATE, "measured_executed"
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            rek_fight = Path(suite.__file__).resolve().parents[2] / "rek_fight"
            module_path = rek_fight / "test_export_motion_trace.py"
            specification = importlib.util.spec_from_file_location(
                "production_export_contract_fixture", module_path
            )
            module = importlib.util.module_from_spec(specification)
            assert specification.loader is not None
            sys.path.insert(0, str(rek_fight))
            try:
                specification.loader.exec_module(module)
            finally:
                sys.path.remove(str(rek_fight))
            fixture_trace = json.loads(
                fixture.sim_traces[suite.MOTIONS[0].selector].read_text(
                    encoding="utf-8"
                )
            )
            module.RESET["identity_material"]["model_sha256"] = (
                fixture_trace["artifact_provenance"]["model"]["sha256"]
            )
            module.RESET["identity_sha256"] = (
                module.exporter._canonical_sha256(
                    module.RESET["identity_material"]
                )
            )
            runner = module.FakeRunner()
            produced = module.exporter.generate_trace(
                runner,
                module.exporter.COMMANDS["forward"],
                build_fingerprint="fixture-build-fingerprint",
                capture_id="production-contract-capture",
                run_id="production-contract-run",
                artifact_provenance=fixture_trace["artifact_provenance"],
                calibration=fixture.sim_calibration_document,
            )
            self.assertFalse(produced["claims"]["screen_coordinates_present"])
            trace_path = fixture.inputs / "production-contract.trace.json"
            write_json(trace_path, produced)
            with fixture.pin_patch(), self.assertRaisesRegex(
                suite.SuiteError, "measured screen coordinates"
            ):
                suite.validate_simulator_trace_calibration(
                    trace_path,
                    fixture.sim_calibration,
                    "walk_forward:press:v1",
                )

    def test_config_requires_per_motion_captures_exact_grid_and_screen_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            config = fixture.config(root / "out")
            selectors = list(suite.MOTION_SELECTORS)
            reused = dict(config.rek_captures)
            reused[selectors[0]] = reused[selectors[1]]
            with self.assertRaisesRegex(suite.SuiteError, "distinct per motion"):
                suite.validate_config(replace(config, rek_captures=reused))
            with self.assertRaisesRegex(suite.SuiteError, "final neutral sample"):
                suite.validate_config(replace(config, end_s=2.998))
            with self.assertRaisesRegex(suite.SuiteError, "screen-frame gate"):
                suite.validate_config(replace(config, screen_required=False))
            with self.assertRaisesRegex(suite.SuiteError, "must be zero"):
                suite.validate_config(
                    replace(config, maximum_timestamp_uncertainty_s=0.001)
                )

    def test_shared_calibration_requires_same_measured_report_and_mjcf(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            proof = suite.validate_shared_calibrations(
                fixture.rek_calibration, fixture.sim_calibration
            )
            self.assertEqual(proof["state"], "measured_shared_target_proven")
            self.assertIn(sha256(fixture.model), proof["target_arena_frame"])
            other = fixture.inputs / "other-report.json"
            write_json(other, {"different": True})
            sim = json.loads(fixture.sim_calibration.read_text(encoding="utf-8"))
            sim["provenance"]["source_report"] = other.name
            sim["provenance"]["source_report_sha256"] = sha256(other)
            write_json(fixture.inputs / "bad-sim-calibration.json", sim)
            with self.assertRaisesRegex(
                suite.SuiteError, "same measured arena mapping report"
            ):
                suite.validate_shared_calibrations(
                    fixture.rek_calibration,
                    fixture.inputs / "bad-sim-calibration.json",
                )

    def test_mapping_requires_every_explicit_motion_once(self):
        values = [(motion.selector, Path(motion.slug)) for motion in suite.MOTIONS]
        result = suite._mapping(values, "fixture")
        self.assertEqual(tuple(result), suite.MOTION_SELECTORS)
        with self.assertRaisesRegex(suite.SuiteError, "missing="):
            suite._mapping(values[:-1], "fixture")
        with self.assertRaisesRegex(suite.SuiteError, "duplicate"):
            suite._mapping([*values, values[0]], "fixture")

    def test_isolated_same_reset_single_press_and_full_window_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            motion = suite.MOTIONS[-1]
            trace = fixture.sim_traces[motion.selector]
            proof = suite.validate_simulator_isolated_trial(
                trace, motion, -1.0, 3.0
            )
            self.assertEqual(
                proof["state"], "exact_isolated_same_reset_single_press_verified"
            )
            self.assertEqual(proof["prior_non_neutral_command_count"], 0)
            self.assertEqual(proof["non_neutral_press_count"], 1)
            with self.assertRaisesRegex(suite.SuiteError, "omits part"):
                suite.validate_simulator_isolated_trial(
                    trace, motion, -1.0, 2.998
                )
            modified = json.loads(trace.read_text(encoding="utf-8"))
            modified["trial"]["prior_non_neutral_command_count"] = 1
            contaminated = fixture.inputs / "contaminated-trial.json"
            write_json(contaminated, modified)
            with self.assertRaisesRegex(suite.SuiteError, "exact isolated"):
                suite.validate_simulator_isolated_trial(
                    contaminated, motion, -1.0, 3.0
                )

    def test_six_selectors_must_share_the_exact_same_reset_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            selector = suite.MOTION_SELECTORS[-1]
            changed = json.loads(
                fixture.sim_traces[selector].read_text(encoding="utf-8")
            )
            changed["reset"]["measured_state"]["qvel_sha256"] = "9" * 64
            write_json(fixture.sim_traces[selector], changed)
            fake = FakeToolchain(fixture)
            with fixture.pin_patch(), mock.patch.object(
                suite.subprocess, "run", side_effect=fake
            ):
                code, report = suite.run_suite(fixture.config(root / "out"))
            self.assertEqual(code, 2)
            self.assertEqual(report["state"], "insufficient_evidence")
            self.assertTrue(
                all(
                    item["blockers"][0]["code"]
                    == "simulator_isolated_same_reset_single_press_not_proven"
                    for item in report["motions"]
                )
            )
            self.assertEqual(fake.commands, [])
            self.assertFalse(list((root / "out").rglob("*.mp4")))

    def test_preflight_insufficient_never_generates_anchors_or_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fake = FakeToolchain(fixture, preflight="insufficient_evidence")
            with fixture.pin_patch(), mock.patch.object(
                suite.subprocess, "run", side_effect=fake
            ):
                code, report = suite.run_suite(
                    fixture.config(Path(temporary) / "out")
                )
            self.assertEqual(code, 2)
            self.assertEqual(report["state"], "insufficient_evidence")
            self.assertEqual(len(report["motions"]), len(suite.MOTIONS))
            self.assertTrue(
                all(item["state"] == "insufficient_evidence" for item in report["motions"])
            )
            tools = [Path(command[1]).name for command in fake.commands]
            self.assertNotIn("video_clock_anchor.py", tools)
            self.assertFalse(list((Path(temporary) / "out").rglob("*.mp4")))

    def test_manual_anchor_output_is_refused_before_video_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fake = FakeToolchain(
                fixture, preflight="passed", anchor_method="manual_video_offset"
            )
            with fixture.pin_patch(), mock.patch.object(
                suite.subprocess, "run", side_effect=fake
            ):
                code, report = suite.run_suite(
                    fixture.config(Path(temporary) / "out")
                )
            self.assertEqual(code, 2)
            self.assertTrue(
                all(
                    item["blockers"][0]["code"]
                    == "diagnostic_manual_or_unbound_video_anchor_refused"
                    for item in report["motions"]
                )
            )
            video_comparisons = [
                command for command in fake.commands
                if Path(command[1]).name == "paired_motion_compare.py"
                and "--video-out" in command
            ]
            self.assertEqual(video_comparisons, [])
            self.assertFalse(list((Path(temporary) / "out").rglob("*.mp4")))

    def test_unit_only_hypothetical_six_pass_with_all_required_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fake = FakeToolchain(fixture)
            output = Path(temporary) / "out"
            with fixture.pin_patch(), mock.patch.object(
                suite.subprocess, "run", side_effect=fake
            ):
                code, report = suite.run_suite(fixture.config(output))
            self.assertEqual(code, 0)
            self.assertTrue(report["passed"])
            self.assertEqual(len(list(output.rglob("paired.mp4"))), len(suite.MOTIONS))
            for motion in report["motions"]:
                self.assertEqual(motion["state"], "passed")
                self.assertIsNotNone(motion["paired_video"])
            comparisons = [
                command for command in fake.commands
                if Path(command[1]).name == "paired_motion_compare.py"
            ]
            self.assertEqual(len(comparisons), len(suite.MOTIONS) * 2)
            for command in comparisons:
                self.assertEqual(command[command.index("--start-s") + 1], "-1.0")
                self.assertEqual(command[command.index("--end-s") + 1], "3.0")
                self.assertEqual(command[command.index("--fps") + 1], "500.0")
                self.assertNotIn("--rek-video-edge-s", command)
                self.assertNotIn("--sim-video-edge-s", command)
            suite_report = json.loads(
                (output / "suite.report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(suite_report["state"], "passed")
            self.assertFalse(
                suite_report["acceptance"]["manual_clock_offsets_allowed"]
            )


if __name__ == "__main__":
    unittest.main()
