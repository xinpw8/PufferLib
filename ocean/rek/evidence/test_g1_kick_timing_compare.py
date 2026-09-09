import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import g1_kick_timing_compare as timing


SCHEDULE_SHA256 = "1" * 64
NPZ_SHA256 = "2" * 64


def json_bytes(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def write_jsonl(path, records):
    with path.open("wb") as stream:
        for record in records:
            stream.write(json_bytes(record))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(
    root,
    label,
    *,
    move_index=6,
    probe_kind="yaw_preempted",
    effective_vector=None,
    desired_vector=None,
    send_delta_substeps=10,
    request_time=10.0,
    pose_status="bounded",
    complete=True,
    include_coverage=False,
    acceptance_status="unknown",
):
    if effective_vector is None:
        effective_vector = [0.0, 0.0, 0.0]
    if desired_vector is None:
        desired_vector = [0.0, 0.0, 0.0]
    run_root = root / label
    run_root.mkdir()
    transcript = run_root / "transcript.jsonl"
    raw = run_root / "raw.jsonl"
    report_path = run_root / "pose-response.json"
    coverage_path = run_root / "coverage.json"
    trace_path = run_root / "trace.jsonl"

    edge_substep = 1000
    send_substep = edge_substep + send_delta_substeps
    edge_fixed_time = 20.0
    send_fixed_time = edge_fixed_time + send_delta_substeps / 500.0
    send_qpc = 1_000_000
    raw_qpc = 1_005_000
    qpc_frequency = 10_000_000
    edge_detail = {
        "phase": (
            "yaw_kick_preempted"
            if probe_kind == "yaw_preempted"
            else "translation_kick_held"
        ),
        "kick_edge": True,
        "kick_probe_ordinal": 0,
        "desired_held": ["Q"] if probe_kind == "yaw_preempted" else ["W"],
        "effective_held": [] if probe_kind == "yaw_preempted" else ["W"],
        "desired_raw_controller_target_xyz": desired_vector,
        "effective_controller_vector_xyz": effective_vector,
        "yaw_preempted": probe_kind == "yaw_preempted",
    }
    transcript_records = [
        {
            "event": "g1_held_schedule_tick",
            "g1_held_schedule_schema": timing.TRANSCRIPT_SCHEMA,
            "client_fixed_substep": edge_substep,
            "schedule_tick": 100,
            "unity_frame": 50,
            "unity_fixed_time": edge_fixed_time,
            "request_only": True,
            "server_acceptance": "unknown",
            "server_acceptance_observed": False,
            "detail": edge_detail,
        },
        {
            "event": "g1_kick_request_lifecycle",
            "g1_held_schedule_schema": timing.TRANSCRIPT_SCHEMA,
            "client_fixed_substep": send_substep,
            "schedule_tick": 101,
            "unity_frame": 51,
            "unity_fixed_time": send_fixed_time,
            "request_only": True,
            "server_acceptance": "unknown",
            "server_acceptance_observed": False,
            "detail": {
                "probe_ordinal": 0,
                "move_index": move_index,
                "lifecycle_stage": "send_move_invoked",
                "send_prefix_fixed_substep": send_substep,
                "late_update_opportunities": 1,
            },
        },
    ]
    write_jsonl(transcript, transcript_records)

    flags = []
    receipt_times = []
    pose_observed = None
    pose_reason = "insufficient_bone_packet_coverage"
    if pose_status == "bounded":
        flags = [False, True, True, True, False, False]
        receipt_times = [10.05, 10.12, 10.14, 10.20, 11.00, 11.04]
        pose_observed = True
        pose_reason = "observed_pose_departure"
    elif pose_status == "censored":
        flags = [False, True, True, True, True]
        receipt_times = [10.05, 10.12, 10.14, 10.20, 10.50]
        pose_observed = True
        pose_reason = "observed_pose_departure"
    elif pose_status == "none":
        flags = [False, False, False]
        receipt_times = [10.05, 10.12, 10.20]
        pose_observed = False
        pose_reason = "no_pose_departure_observed"

    raw_records = [
        {
            "event": "capture_start",
            "schema": "rek.private_ai.protocol.v7",
            "stopwatch_frequency_hz": qpc_frequency,
        },
        {
            "event": "outbound_request_projection",
            "message": "REK_Move",
            "request_sequence": 10,
            "client_fixed_tick_at_observation": 1005,
            "stopwatch_timestamp_ticks": raw_qpc,
            "unity_realtime_since_startup": request_time,
            "move_index_source_int32": move_index,
            "wire_delivery": "reliable",
            "request_only": True,
            "server_acceptance": None,
            "ack_observed": False,
        },
    ]
    samples = []
    for index, (flag, receipt_time) in enumerate(zip(flags, receipt_times)):
        sequence = 100 + index
        tick = 1005 + index * 10
        raw_records.append(
            {
                "event": "raw_bone_packet",
                "raw_bone_packet_sequence": sequence,
                "fighter_slot": 0,
                "client_fixed_tick_at_observation": tick,
                "monotonic_receipt_time": receipt_time,
            }
        )
        samples.append(
            {
                "above_departure_threshold": flag,
                "raw_bone_packet_sequence": sequence,
                "recorder_client_fixed_tick": tick,
                "seconds_from_anchor": (tick - 1005) / 500.0,
            }
        )
    raw_records.append({"event": "capture_end"})
    write_jsonl(raw, raw_records)

    pose = {
        "status": pose_reason,
        "pose_departure_observed": pose_observed,
    }
    if pose_observed is not None:
        pose.update(
            {
                "detector": {
                    "method": "nearest_pre_anchor_pose_envelope_quaternion_geodesic",
                    "required_consecutive_received_packets": 2,
                    "classifier_status": "provisional_conservative_detector_not_a_move_identity_model",
                },
                "samples": samples,
                "onset_receipt_bracket": {"fixture": True},
                "return_receipt_bracket": (
                    {"fixture": True} if pose_status == "bounded" else None
                ),
            }
        )
    raw_sha = sha256(raw)
    transcript_sha = sha256(transcript)
    report = {
        "schema": timing.POSE_REPORT_SCHEMA,
        "sources": {
            "transcript": {
                "path": str(transcript),
                "sha256": transcript_sha,
                "schedule_sha256": SCHEDULE_SHA256,
                "schedule_run_id": label,
            },
            "recorder": {
                "path": str(raw),
                "sha256": raw_sha,
                "schema": "rek.private_ai.protocol.v7",
                "machine": "SPARK-TEST",
                "pid": 32,
            },
        },
        "scope": {
            "runtime_model": "g1",
            "exact_g1_vs_g1": True,
            "solo_route_proven": True,
            "local_fighter_index": 0,
        },
        "summary": {
            "transcript_complete": complete,
            "transcript_reason": "complete" if complete else "lease_released",
        },
        "evidence_limits": {
            "server_action_acceptance": "unknown",
            "executed_move_identity": "unknown",
            "canonical_move_duration": "unknown",
        },
        "kick_probes": [
            {
                "probe_ordinal": 0,
                "probe_label": f"fixture-kick-{move_index}",
                "probe_kind": probe_kind,
                "requested_move_index": move_index,
                "schedule_edge": {
                    "client_fixed_substep": edge_substep,
                    "recorder_client_fixed_tick": 1000,
                    "schedule_tick": 100,
                    "unity_fixed_time": edge_fixed_time,
                },
                "local_execute_move_observation": {
                    "returned": True,
                    "classification": "accepted_locally_and_armed",
                    "authority": "visual_only_client_local_diagnostic",
                    "server_acceptance": "unknown",
                },
                "request_sent": {
                    "status": "observed_in_transcript_and_recorder",
                    "value": True,
                    "recorder_request_sequence": 10,
                    "recorder_client_fixed_tick": 1005,
                    "qpc_anchor_delta_seconds": 0.0005,
                    "send_prefix_anchor": {
                        "client_fixed_substep": send_substep,
                        "schedule_tick": 101,
                        "unity_frame": 51,
                        "unity_fixed_time": send_fixed_time,
                        "qpc_ticks": send_qpc,
                        "qpc_frequency_hz": qpc_frequency,
                    },
                },
                "physical_response": {
                    "status": "fixture",
                    "pose_departure": pose,
                    "server_action_acceptance": "unknown",
                    "executed_move_identity": "unknown",
                },
                "combat_context": {
                    "confounding_not_excluded": True,
                    "event_counts": {
                        "raw_fight_state_packet": 2,
                        "raw_hit_packet": 0,
                        "raw_score_packet": 0,
                    },
                },
                "server_action_acceptance": {
                    "status": acceptance_status,
                    "value": None,
                    "acknowledgement_observed": False,
                },
                "move_identity": {
                    "requested_move_index": move_index,
                    "requested_asset_configuration": {
                        "runtime_name": "fixture_kick",
                        "npz_sha256": NPZ_SHA256,
                        "recovered_controller_ticks": 158,
                    },
                    "executed_move_identity": "unknown",
                    "requested_asset_to_observed_pose_identity_proven": False,
                },
            }
        ],
    }
    report_path.write_bytes(json_bytes(report))

    if include_coverage:
        write_jsonl(trace_path, [{"event": "trace_start"}, {"event": "trace_end"}])
        coverage = {
            "schema": timing.COVERAGE_SCHEMA,
            "source": {"sha256": raw_sha},
            "trace_artifact": {
                "path": str(trace_path),
                "sha256": sha256(trace_path),
                "schema": timing.TRACE_SCHEMA,
            },
            "trace_grid": {
                "rate_hz": 50,
                "bone_sampling": "latest_received_no_interpolation",
                "bone_source": {
                    "0": {
                        "maximum_source_age_ticks": 12,
                        "maximum_source_age_seconds": 0.024,
                        "observed_source_rate_hz": 40.0,
                        "fresh_grid_samples": 4,
                        "reused_grid_samples": 2,
                    }
                },
            },
            "reference_alignment": {
                "status": "calibration_required",
                "direct_angle_identity_allowed": False,
                "required_transform": "measured_transform",
            },
        }
        coverage_path.write_bytes(json_bytes(coverage))

    return timing.RunSpec(
        label=label,
        report_path=report_path,
        coverage_path=coverage_path if include_coverage else None,
    )


class G1KickTimingCompareTests(unittest.TestCase):
    def test_separates_local_dispatch_candidate_receipt_and_unknown_canonical_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_run(Path(temporary), "run-a", include_coverage=True)
            report = timing.analyze_runs([spec])

        self.assertEqual(report["schema"], timing.OUTPUT_SCHEMA)
        probe = report["runs"][0]["probes"][0]
        self.assertEqual(probe["local_arm_to_send_prefix"]["fixed_substeps"], 10)
        self.assertAlmostEqual(
            probe["local_arm_to_send_prefix"]["nominal_seconds_at_500hz"], 0.02
        )
        self.assertAlmostEqual(
            probe["outbound_request_projection"][
                "recorder_minus_bridge_observer_qpc_seconds"
            ],
            0.0005,
        )
        onset = probe["received_pose_candidate"][
            "request_to_first_candidate_receipt_bracket_seconds"
        ]
        self.assertAlmostEqual(onset["lower"], 0.05)
        self.assertAlmostEqual(onset["upper"], 0.12)
        duration = probe["received_pose_candidate"][
            "candidate_departure_duration_receipt_domain_seconds"
        ]
        self.assertAlmostEqual(duration["lower"], 0.08)
        self.assertAlmostEqual(duration["upper"], 0.95)
        self.assertEqual(
            probe["canonical_input_to_physical_response_latency"]["status"],
            "unknown",
        )
        self.assertEqual(probe["canonical_move_duration"]["status"], "unknown")
        self.assertEqual(probe["requested_asset"]["frame_count_timing_use"], "forbidden")
        self.assertEqual(probe["requested_asset"]["alignment"]["status"], "not_performed")
        trace = report["runs"][0]["trace_resampling"]
        self.assertEqual(trace["status"], "verified")
        self.assertEqual(
            trace["local_bone_packet_age_at_grid"]["maximum_source_age_ticks"], 12
        )

    def test_right_censored_interval_uses_last_confirmed_departed_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_run(Path(temporary), "run-a", pose_status="censored")
            report = timing.analyze_runs([spec])
        duration = report["runs"][0]["probes"][0]["received_pose_candidate"][
            "candidate_departure_duration_receipt_domain_seconds"
        ]
        self.assertEqual(duration["status"], "right_censored_in_client_receipt_domain")
        self.assertAlmostEqual(duration["lower"], 0.38)
        self.assertIsNone(duration["upper"])

    def test_insufficient_capture_preserves_unknown_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_run(
                Path(temporary), "run-partial", pose_status="insufficient", complete=False
            )
            report = timing.analyze_runs([spec])
        run = report["runs"][0]
        self.assertFalse(run["complete"])
        received = run["probes"][0]["received_pose_candidate"]
        self.assertEqual(received["status"], "insufficient_packet_coverage")
        self.assertIsNone(received["candidate_departure_observed"])
        self.assertEqual(
            report["measurement_contract"]["input_to_first_physical_response_latency"][
                "status"
            ],
            "unknown",
        )

    def test_repeat_comparison_is_descriptive_and_never_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_run(root, "run-a", send_delta_substeps=2, request_time=10.0)
            second = make_run(root, "run-b", send_delta_substeps=11, request_time=10.0)
            report = timing.analyze_runs([first, second])
        group = report["repeat_groups"][0]
        self.assertEqual(group["run_count"], 2)
        self.assertEqual(
            group["local_arm_to_send_prefix_fixed_substeps"]["values"], [2, 11]
        )
        self.assertEqual(
            group["first_candidate_pose_receipt"]["semantic_limit"],
            "descriptive comparison of receipt-domain candidates, not an estimator",
        )
        self.assertEqual(
            group["candidate_pose_departure_duration"][
                "right_censored_observation_count"
            ],
            0,
        )
        self.assertEqual(group["canonical_timing"]["move_duration"], "unknown")

    def test_translation_held_condition_remains_distinct_from_neutral_preempted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            held = make_run(
                root,
                "held",
                probe_kind="translation_held",
                effective_vector=[1.0, 0.0, 0.0],
                desired_vector=[1.0, 0.0, 0.0],
            )
            neutral = make_run(root, "neutral", probe_kind="yaw_preempted")
            report = timing.analyze_runs([held, neutral])
        conditions = {
            probe["normalized_control_condition"]
            for run in report["runs"]
            for probe in run["probes"]
        }
        self.assertEqual(
            conditions,
            {"translation_held_at_edge", "neutral_after_yaw_preemption"},
        )
        self.assertEqual(len(report["repeat_groups"]), 2)

    def test_rejects_any_server_acceptance_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_run(
                Path(temporary), "run-a", acceptance_status="accepted"
            )
            with self.assertRaisesRegex(
                timing.KickTimingError, "probe_claims_server_acceptance"
            ):
                timing.analyze_runs([spec])

    def test_rejects_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = make_run(Path(temporary), "run-a")
            source = json.loads(spec.report_path.read_text(encoding="utf-8"))
            raw_path = Path(source["sources"]["recorder"]["path"])
            with raw_path.open("ab") as stream:
                stream.write(b"{}\n")
            with self.assertRaisesRegex(timing.KickTimingError, "raw_sha256_mismatch"):
                timing.analyze_runs([spec])

    def test_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = make_run(root, "run-a")
            output = root / "comparison.json"
            arguments = ["--run", f"run-a={spec.report_path}", "--out", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(timing.main(arguments), 0)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(timing.main(arguments), 1)


if __name__ == "__main__":
    unittest.main()
