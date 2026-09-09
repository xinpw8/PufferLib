import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import g1_held_trace_extract
import g1_schedule_pose_response as response
import test_g1_held_trace_extract as held_fixture


SCHEDULE_SHA = "a" * 64
RUN_ID = "b" * 32


def identity_pose(rotated_radians=0.0):
    values = [0.0, 0.0, 0.0, 0.0]
    for bone in range(1, len(g1_held_trace_extract.G1_BONE_NAMES)):
        if bone == 1 and rotated_radians:
            values.extend([math.sin(rotated_radians / 2.0), 0.0, 0.0,
                           math.cos(rotated_radians / 2.0)])
        else:
            values.extend([0.0, 0.0, 0.0, 1.0])
    return values


def root_record(tick):
    return {
        "tick": tick,
        "index": tick,
        "utc": None,
        "stopwatch_timestamp_ticks": 1_000_000 + tick * 20_000,
        "unity_frame": tick // 8,
        "unity_time": 10.0 + tick * 0.002,
        "unity_fixed_time": 10.0 + tick * 0.002,
        "unity_unscaled_time": 10.0 + tick * 0.002,
        "fight_epoch": 0,
        "round_number": 1,
        "local_fighter_index": 0,
        "opponent_slot": 1,
        "fighter_0_root": {
            "world_position_xyz": [tick * 0.001, 0.72, tick * 0.0005],
            "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "fighter_1_root": {
            "world_position_xyz": [2.0, 0.72, 0.0],
            "world_rotation_xyzw": [0.0, 1.0, 0.0, 0.0],
        },
    }


def synthetic_capture(*, departure=True, single_spike=False, move_qpc=3_200_000):
    roots = {tick: root_record(tick) for tick in range(221)}
    local_ticks = [60, 70, 80, 90, 100, 115, 125, 135, 145, 155, 165, 175]
    rotations = {}
    for tick in local_ticks:
        angle = 0.0
        if departure and tick in ({125} if single_spike else {125, 135}):
            angle = 0.7
        rotations[tick] = angle
    packets = {0: [], 1: []}
    decoded = {}
    sequence = 1
    for slot, ticks in ((0, local_ticks), (1, local_ticks)):
        for tick in ticks:
            packet = {
                "slot": slot,
                "tick": tick,
                "sequence": sequence,
                "unity_frame": tick // 8,
                "unity_time": 10.0 + tick * 0.002,
                "unity_unscaled_time": 10.0 + tick * 0.002,
                "network_index": slot,
                "wire_body_sha256": "%064x" % sequence,
                "world_positions_xyz": [0.0] * 90,
                "world_rotations_xyzw": [0.0] * 120,
            }
            packets[slot].append(packet)
            decoded[sequence] = {
                "slot": slot,
                "tick": tick,
                "raw_sequence": sequence,
                "snapshot_sequence": sequence,
                "snapshot_received_at_client_time": 10.0 + tick * 0.002,
                "root_world_position": [0.0, 0.72, 0.0],
                "root_world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "child_local_rotations_xyzw": identity_pose(
                    rotations[tick] if slot == 0 else 0.0
                ),
            }
            sequence += 1
    return g1_held_trace_extract.Capture(
        source_name="synthetic-raw.jsonl",
        raw_sha256="c" * 64,
        start={
            "schema": g1_held_trace_extract.RECORDER_SCHEMA,
            "plugin_version": "0.7.2",
            "plugin_sha256": "d" * 64,
            "machine": "SPARK-4AE3",
            "pid": 860,
            "stopwatch_frequency_hz": 10_000_000,
        },
        end={},
        roots=roots,
        bone_packets=packets,
        decoded_snapshots=decoded,
        input_requests=[
            {
                "tick": 100,
                "request_sequence": 1,
                "message": "REK_Input",
                "velocity_command_xyz": [1.0, 0.0, 0.0],
            },
            {
                "tick": 120,
                "request_sequence": 3,
                "message": "REK_Input",
                "velocity_command_xyz": [0.0, 0.0, 0.0],
            },
        ],
        move_requests=[
            {
                "tick": 110,
                "request_sequence": 2,
                "message": "REK_Move",
                "move_index": 6,
                "stopwatch_timestamp_ticks": move_qpc,
            }
        ],
        forbidden_requests=[],
        local_slot=0,
        opponent_slot=1,
    )


def schedule_record(event, sequence, schedule_tick, substep, fixed_time, detail):
    return {
        "event": event,
        "protocol": "rek.ui_bridge.v1",
        "g1_held_schedule_schema": response.SCHEDULE_SCHEMA,
        "g1_held_schedule_id": "synthetic.schedule.v2",
        "g1_held_schedule_sha256": SCHEDULE_SHA,
        "g1_held_schedule_run_id": RUN_ID,
        "event_sequence": sequence,
        "schedule_tick": schedule_tick,
        "client_fixed_substep": substep,
        "fixed_substeps_per_schedule_tick": 10,
        "schedule_rate_hz": 50,
        "unity_fixed_rate_hz": 500,
        "detail": detail,
        "request_only": True,
        "server_acceptance": "unknown",
        "server_acceptance_observed": False,
        "authoritative_execution_observed": False,
        "global_input_emitted": False,
        "unity_frame": 1,
        "unity_fixed_time": fixed_time,
    }


def synthetic_transcript(*, qpc=3_200_000, complete=False):
    tick0 = schedule_record(
        "g1_held_schedule_tick",
        1,
        0,
        0,
        10.2,
        {
            "phase": "held_condition",
            "desired_held": ["W"],
            "held_condition_ordinal": 0,
            "kick_probe_ordinal": None,
            "kick_edge": False,
            "effective_controller_vector_xyz": [1.0, 0.0, 0.0],
        },
    )
    edge = schedule_record(
        "g1_held_schedule_tick",
        2,
        1,
        10,
        10.22,
        {
            "phase": "translation_kick_probe",
            "desired_held": ["W"],
            "held_condition_ordinal": None,
            "kick_probe_ordinal": 0,
            "kick_edge": True,
            "effective_controller_vector_xyz": [1.0, 0.0, 0.0],
        },
    )
    lifecycle = schedule_record(
        "g1_kick_request_lifecycle",
        3,
        1,
        10,
        10.22,
        {
            "probe_ordinal": 0,
            "probe_label": "W+kick-6",
            "probe_kind": "translation_held",
            "move_index": 6,
            "lifecycle_stage": "execute_move_returned",
            "execute_move_returned": True,
            "local_classification": "accepted_locally_and_armed",
        },
    )
    neutral = schedule_record(
        "g1_held_schedule_tick",
        4,
        2,
        20,
        10.24,
        {
            "phase": "neutral_gap",
            "desired_held": [],
            "held_condition_ordinal": None,
            "kick_probe_ordinal": 0,
            "kick_edge": False,
            "effective_controller_vector_xyz": [0.0, 0.0, 0.0],
        },
    )
    summary = schedule_record(
        "g1_kick_measurement_summary",
        6,
        8,
        80,
        10.36,
        {
            "probe_ordinal": 0,
            "probe_label": "W+kick-6",
            "probe_kind": "translation_held",
            "move_index": 6,
            "execute_move_returned": True,
            "local_classification": "accepted_locally_and_armed",
            "move_send_invoked": True,
            "translation_release_tick": 2,
            "translation_release_fixed_substep": 20,
            "translation_first_transition_settled_fixed_substep": 25,
            "translation_fixed_substeps_release_to_settled": 5,
            "translation_request_send_timing_classification": (
                "request_sent_before_translation_release"
            ),
            "translation_post_release_settled_kick_control_included": False,
            "translation_post_release_settled_kick_remaining_unknown": (
                "single_edge_no_retry_schedule_observes_the_original_request_only"
            ),
            "send_prefix_anchor": {
                "client_fixed_substep": 10,
                "schedule_tick": 1,
                "unity_frame": 1,
                "unity_fixed_time": 10.22,
                "qpc_ticks": qpc,
                "qpc_frequency_hz": 10_000_000,
            },
        },
    )
    release = schedule_record(
        "g1_translation_release",
        5,
        2,
        20,
        10.24,
        {
            "probe_ordinal": 0,
            "probe_label": "W+kick-6",
            "probe_kind": "translation_held",
            "move_index": 6,
            "release_tick": 2,
            "release_fixed_substep": 20,
            "release_qpc_ticks": 3_400_000,
            "qpc_frequency_hz": 10_000_000,
            "transition_settled_diagnostic": {
                "available": True,
                "transition_settled": False,
                "planar_speed_m_s": 0.10,
                "transition_settle_planar_speed_m_s": 0.03,
            },
        },
    )
    end = {
        "event": "g1_held_schedule_end",
        "g1_held_schedule_schema": response.SCHEDULE_SCHEMA,
        "g1_held_schedule_id": "synthetic.schedule.v2",
        "g1_held_schedule_sha256": SCHEDULE_SHA,
        "g1_held_schedule_run_id": RUN_ID,
        "schedule_tick": 2,
        "client_fixed_substep": 20,
        "reason": "complete" if complete else "synthetic_partial",
        "complete": complete,
        "experiment_coverage_complete": complete,
        "partial_coverage": not complete,
        "request_only": True,
        "server_acceptance": "unknown",
        "server_acceptance_observed": False,
        "authoritative_execution_observed": False,
        "global_input_emitted": False,
        "unity_fixed_time": 10.36,
    }
    records = [tick0, edge, lifecycle, neutral, release, summary]
    return response.Transcript(
        source_name="synthetic-transcript.jsonl",
        sha256="e" * 64,
        pipe_server_pid=860,
        schema=response.SCHEDULE_SCHEMA,
        schedule_id="synthetic.schedule.v2",
        schedule_sha256=SCHEDULE_SHA,
        run_id=RUN_ID,
        ticks=[tick0, edge, neutral],
        edges={0: edge},
        releases={0: release},
        summaries={0: summary},
        lifecycle={0: [lifecycle]},
        end=end,
        event_count=len(records),
    )


def detector_config():
    return response.DetectionConfig(
        baseline_ticks=60,
        minimum_baseline_packets=4,
        minimum_post_packets=5,
        minimum_rms_radians=0.08,
        minimum_max_joint_radians=0.25,
        mad_multiplier=6.0,
        threshold_margin_radians=0.03,
        consecutive_packets=2,
    )


class PoseDetectorTests(unittest.TestCase):
    def test_consecutive_departure_has_receipt_bounds_but_no_canonical_duration(self):
        result = response.detect_pose_departure(
            synthetic_capture(), 110, 180, detector_config()
        )
        self.assertEqual(result["status"], "observed_pose_departure")
        self.assertTrue(result["pose_departure_observed"])
        self.assertEqual(
            result["pose_departure_window_receipt_domain"]["status"],
            "bounded_in_client_receipt_domain",
        )
        self.assertEqual(result["server_action_acceptance"], "unknown")
        self.assertEqual(result["executed_move_identity"], "unknown")
        self.assertEqual(result["canonical_move_duration"]["status"], "unknown")

    def test_one_packet_spike_is_not_a_response(self):
        result = response.detect_pose_departure(
            synthetic_capture(single_spike=True), 110, 180, detector_config()
        )
        self.assertEqual(result["status"], "no_pose_departure_observed")
        self.assertFalse(result["pose_departure_observed"])

    def test_no_departure_is_reported_without_inference(self):
        result = response.detect_pose_departure(
            synthetic_capture(departure=False), 110, 180, detector_config()
        )
        self.assertFalse(result["pose_departure_observed"])
        self.assertIsNone(result["onset_receipt_bracket"])

    def test_insufficient_packets_remains_unknown(self):
        capture = synthetic_capture()
        capture.bone_packets[0] = capture.bone_packets[0][:3]
        result = response.detect_pose_departure(capture, 110, 180, detector_config())
        self.assertEqual(result["status"], "insufficient_bone_packet_coverage")
        self.assertIsNone(result["pose_departure_observed"])


class CorrelationTests(unittest.TestCase):
    def test_partial_send_anchor_uses_same_pid_recorder_qpc_frequency(self):
        capture = synthetic_capture()
        matched, delta = response._match_raw_move(
            capture,
            6,
            {"qpc_ticks": 3_200_000},
            set(),
            response.DEFAULT_SEND_MATCH_TOLERANCE_SECONDS,
        )
        self.assertEqual(matched["request_sequence"], 2)
        self.assertEqual(delta, 0.0)

    def test_dual_send_and_pose_response_remain_distinct_from_acceptance(self):
        report = response.analyze_data(
            synthetic_transcript(),
            synthetic_capture(),
            [],
            config=detector_config(),
        )
        probe = report["kick_probes"][0]
        self.assertTrue(probe["request_sent"]["value"])
        self.assertEqual(
            probe["request_sent"]["status"], "observed_in_transcript_and_recorder"
        )
        self.assertEqual(
            probe["physical_response"]["status"],
            "candidate_post_send_pose_departure_observed",
        )
        self.assertEqual(probe["server_action_acceptance"]["status"], "unknown")
        self.assertEqual(probe["move_identity"]["executed_move_identity"], "unknown")
        self.assertEqual(probe["duration"]["canonical_move_duration_status"], "unknown")
        self.assertTrue(probe["translation_gate_diagnostics"]["release_event_observed"])
        self.assertEqual(
            probe["translation_gate_diagnostics"][
                "request_send_timing_classification"
            ],
            "request_sent_before_translation_release",
        )
        self.assertEqual(report["clock_correlation"]["status"], "correlated")

    def test_qpc_mismatch_does_not_claim_send(self):
        report = response.analyze_data(
            synthetic_transcript(qpc=9_000_000),
            synthetic_capture(),
            [],
            config=detector_config(),
        )
        probe = report["kick_probes"][0]
        self.assertIsNone(probe["request_sent"]["value"])
        self.assertEqual(
            probe["request_sent"]["status"], "transcript_send_unmatched_in_recorder"
        )
        self.assertEqual(
            probe["physical_response"]["status"],
            "not_attributable_without_dual_observed_send",
        )
        self.assertEqual(report["summary"]["raw_move_projections_not_matched_to_transcript"], 1)
        self.assertEqual(
            report["negative_control_diagnostics"][
                "probe_ordinals_with_pose_departure_but_no_dual_observed_send"
            ],
            [0],
        )
        self.assertEqual(report["analysis_gate"]["status"], "incomplete")

    def test_held_trajectory_keeps_measured_grid_and_dense_root_path(self):
        report = response.analyze_data(
            synthetic_transcript(), synthetic_capture(), [], config=detector_config()
        )
        held = report["held_input_trajectories"][0]
        self.assertEqual(held["label"], "W")
        self.assertEqual(held["kind"], "translation")
        self.assertEqual(len(held["schedule_grid_samples"]), 1)
        self.assertGreater(held["root_trajectory"]["dense_root_sample_count"], 1)
        self.assertGreater(held["root_trajectory"]["planar_path_m"], 0.0)

    def test_clock_offset_drift_is_rejected(self):
        transcript = synthetic_transcript()
        transcript.ticks[2]["unity_fixed_time"] += 0.004
        with self.assertRaisesRegex(response.PoseResponseError, "offset_drift"):
            response.correlate_clock(transcript, synthetic_capture(), 0.0011)

    def test_end_to_end_reads_strict_recorder_and_writes_create_only_report(self):
        transcript = synthetic_transcript(qpc=6_000_000)
        fixed_times = {0: 0.58, 1: 0.60, 2: 0.62}
        for record in transcript.ticks:
            record["unity_fixed_time"] = fixed_times[record["schedule_tick"]]
        transcript.edges[0]["unity_fixed_time"] = 0.60
        transcript.summaries[0]["unity_fixed_time"] = 0.78
        transcript.summaries[0]["detail"]["send_prefix_anchor"].update(
            {"unity_fixed_time": 0.60, "qpc_ticks": 6_000_000}
        )
        transcript.end["unity_fixed_time"] = 0.78
        sequenced = {
            record["event_sequence"]: record
            for record in (
                transcript.ticks
                + transcript.lifecycle[0]
                + list(transcript.releases.values())
                + [transcript.summaries[0]]
            )
        }
        records = [
            {
                "event": "client_pipe_server_proof",
                "process_id": transcript.pipe_server_pid,
                "executable": "C:\\REK.exe",
            }
        ] + [sequenced[index] for index in sorted(sequenced)] + [transcript.end]
        raw_records = held_fixture.fixture()
        raw_records[0]["stopwatch_frequency_hz"] = 10_000_000
        raw_records[0]["pid"] = transcript.pipe_server_pid
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            transcript_path = directory / "schedule.jsonl"
            raw_path = directory / "raw.jsonl"
            output_path = directory / "report.json"
            transcript_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            held_fixture.write_jsonl(raw_path, raw_records)
            report = response.analyze(transcript_path, raw_path, output_path)
            stored = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["schema"], response.REPORT_SCHEMA)
            self.assertEqual(
                stored["sources"]["recorder"]["sha256"],
                report["sources"]["recorder"]["sha256"],
            )
            self.assertTrue(stored["kick_probes"][0]["request_sent"]["value"])
            self.assertEqual(
                stored["kick_probes"][0]["server_action_acceptance"]["status"],
                "unknown",
            )
            with self.assertRaises(FileExistsError):
                response.analyze(transcript_path, raw_path, output_path)


class TranscriptTests(unittest.TestCase):
    def write_records(self, records):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "transcript.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return directory, path

    def partial_records(self):
        transcript = synthetic_transcript()
        records = []
        by_sequence = {}
        for record in transcript.ticks:
            by_sequence[record["event_sequence"]] = record
        for records_for_probe in transcript.lifecycle.values():
            for record in records_for_probe:
                by_sequence[record["event_sequence"]] = record
        for record in transcript.releases.values():
            by_sequence[record["event_sequence"]] = record
        for record in transcript.summaries.values():
            by_sequence[record["event_sequence"]] = record
        records.append(
            {
                "event": "client_pipe_server_proof",
                "process_id": transcript.pipe_server_pid,
                "executable": "C:\\REK.exe",
            }
        )
        records.extend(by_sequence[index] for index in sorted(by_sequence))
        records.append(transcript.end)
        return records

    def test_reads_partial_transcript_with_contiguous_event_sequence(self):
        directory, path = self.write_records(self.partial_records())
        with directory:
            parsed = response.read_transcript(path)
        self.assertEqual(parsed.run_id, RUN_ID)
        self.assertEqual(len(parsed.ticks), 3)
        self.assertEqual(len(parsed.edges), 1)
        self.assertFalse(parsed.end["complete"])

    def test_rejects_claimed_server_acceptance(self):
        records = self.partial_records()
        records[1] = copy.deepcopy(records[1])
        records[1]["server_acceptance"] = "accepted"
        directory, path = self.write_records(records)
        with directory, self.assertRaisesRegex(
            response.PoseResponseError, "claims_server_acceptance"
        ):
            response.read_transcript(path)

    def test_rejects_missing_pipe_server_proof(self):
        records = self.partial_records()[1:]
        directory, path = self.write_records(records)
        with directory, self.assertRaisesRegex(
            response.PoseResponseError, "pipe_server_proof"
        ):
            response.read_transcript(path)

    def test_rejects_transcript_recorder_pid_mismatch(self):
        transcript = synthetic_transcript()
        transcript.pipe_server_pid = 999
        with self.assertRaisesRegex(response.PoseResponseError, "pid_mismatch"):
            response.analyze_data(
                transcript, synthetic_capture(), [], config=detector_config()
            )

    def test_rejects_unsealed_schedule_hash(self):
        records = self.partial_records()
        for record in records:
            record["g1_held_schedule_sha256"] = "0" * 64
        directory, path = self.write_records(records)
        with directory, self.assertRaisesRegex(
            response.PoseResponseError, "schedule_sha256_unsealed"
        ):
            response.read_transcript(path)

    def test_rejects_noncontiguous_event_sequence(self):
        records = self.partial_records()
        records[1] = copy.deepcopy(records[1])
        records[1]["event_sequence"] = 7
        directory, path = self.write_records(records)
        with directory, self.assertRaisesRegex(
            response.PoseResponseError, "event_sequence_not_contiguous"
        ):
            response.read_transcript(path)

    def test_rejects_schedule_event_missing_sequence(self):
        records = self.partial_records()
        records[1] = copy.deepcopy(records[1])
        del records[1]["event_sequence"]
        directory, path = self.write_records(records)
        with directory, self.assertRaisesRegex(
            response.PoseResponseError, "missing_sequence"
        ):
            response.read_transcript(path)


if __name__ == "__main__":
    unittest.main()
