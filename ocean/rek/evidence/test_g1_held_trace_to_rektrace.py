"""Focused tests for finalized G1 JSONL to binary REKTRACE conversion."""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import differ
import g1_held_trace_extract as held
import g1_held_trace_to_rektrace as converter
from trace import Trace


FINGERPRINT = "f" * 64
RUN_ID = "1" * 32
FRESH_ROUND_ID = "fresh-round-request-1"
ROUND_ID = "2" * 64
RAW_ID = "3" * 64
TRACE_FIRST_SCHEDULE_TICK = 2199
TRACE_PHASE_SUBSTEPS = 2


HELD_RANGES = (
    (50, 150, 0, [1.0, 0.0, 0.0]),
    (200, 300, 1, [-1.0, 0.0, 0.0]),
    (350, 450, 2, [0.0, 1.0, 0.0]),
    (500, 600, 3, [0.0, -1.0, 0.0]),
    (650, 750, 4, [0.0, 0.0, 1.0]),
    (800, 900, 5, [0.0, 0.0, -1.0]),
    (950, 1050, 6, [1.0, 0.0, 1.0]),
    (1100, 1200, 7, [1.0, 0.0, -1.0]),
    (1250, 1350, 8, [-1.0, 0.0, 1.0]),
    (1400, 1500, 9, [-1.0, 0.0, -1.0]),
    (1550, 1650, 10, [0.0, 1.0, 1.0]),
    (1700, 1800, 11, [0.0, 1.0, -1.0]),
    (1850, 1950, 12, [0.0, -1.0, 1.0]),
    (2000, 2100, 13, [0.0, -1.0, -1.0]),
)
KICK_EDGES = {2200: 0, 2500: 1, 2800: 2, 3100: 3, 3400: 4, 3700: 5, 4000: 6, 4300: 7}


def inventory_document():
    return {
        "schema": 1,
        "build_fingerprint": FINGERPRINT,
        "errors": [],
        "steam": {"buildid": "24969755"},
        "files": [
            {
                "path": "GameAssembly.dll",
                "sha256": held.EXPECTED_GAME_ASSEMBLY_SHA256,
            },
            {
                "path": "REK_Data/il2cpp_data/Metadata/global-metadata.dat",
                "sha256": held.EXPECTED_METADATA_SHA256,
            },
        ],
    }


def measured_fighter(slot):
    return {
        "slot": slot,
        "semantic_robot_id": "g1",
        "runtime_object_name": f"G1 fighter {slot}",
        "bone_count": 30,
        "bone_names": list(held.G1_BONE_NAMES),
        "runtime_bone_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
        "semantic_t800": False,
        "semantic_g1": True,
        "exact_t800_bone_signature": False,
        "exact_g1_bone_signature": True,
        "semantic_runtime_mismatch": False,
        "semantic_runtime_consistency": "semantic_robot_id_matches_runtime_model",
        "semantic_robot_id_used_for_continuous_acceptance": False,
    }


def measured_pairing(local_slot=0, opponent_slot=1):
    return {
        "required_pairing": "exact_homogeneous_supported_runtime_pair",
        "required_robot_id": None,
        "supported_runtime_models": ["t800", "g1"],
        "semantic_robot_id_required_for_acceptance": False,
        "required_t800_bone_count": 26,
        "required_t800_bone_signature_sha256": "4" * 64,
        "required_g1_bone_count": 30,
        "required_g1_bone_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
        "semantic_identity_source": "FightCoordinator.fighterIdentities[slot].RobotID",
        "bone_signature_source": "FightCoordinator.Fighters[slot].boneTransforms[index].name",
        "exact_supported_runtime_pairing": True,
        "runtime_model": "g1",
        "exact_t800_vs_t800": False,
        "exact_g1_vs_g1": True,
        "reason": converter.EXPECTED_PAIRING_REASON,
        "local_slot": local_slot,
        "opponent_slot": opponent_slot,
        "local_fighter": measured_fighter(local_slot),
        "opponent_fighter": measured_fighter(opponent_slot),
    }


def held_detail(tick):
    ordinal = None
    velocity = [0.0, 0.0, 0.0]
    for start, stop, candidate, vector in HELD_RANGES:
        if start <= tick < stop:
            ordinal = candidate
            velocity = list(vector)
            break
    if 2150 <= tick < 2205:
        velocity = [1.0, 0.0, 0.0]
    return {
        "effective_controller_vector_xyz": velocity,
        "velocity_property_write_returned": True,
        "velocity_readback_exact": True,
        "held_condition_ordinal": ordinal,
        "kick_edge": tick in KICK_EDGES,
        "kick_probe_ordinal": KICK_EDGES.get(tick),
    }


def sequenced_record(event, sequence, tick, fixed_time, detail, schedule_id):
    return {
        "event": event,
        "protocol": "rek.ui_bridge.v1",
        "g1_held_schedule_schema": converter.EXPECTED_SCHEDULE_SCHEMA,
        "g1_held_schedule_id": schedule_id,
        "g1_held_schedule_sha256": converter.EXPECTED_SCHEDULE_SHA256,
        "g1_held_schedule_run_id": RUN_ID,
        "fresh_round_request_id": FRESH_ROUND_ID,
        "round_identity_sha256": ROUND_ID,
        "event_sequence": sequence,
        "schedule_tick": tick,
        "client_fixed_substep": tick * 10,
        "fixed_substeps_per_schedule_tick": 10,
        "schedule_rate_hz": 50,
        "unity_fixed_rate_hz": 500,
        "detail": detail,
        "authority_scope": converter.EXPECTED_AUTHORITY_SCOPE,
        "authority_caveat": converter.EXPECTED_AUTHORITY_CAVEAT,
        "request_only": True,
        "server_acceptance": "unknown",
        "server_acceptance_observed": False,
        "authoritative_execution_observed": False,
        "global_input_emitted": False,
        "unity_frame": tick * 2,
        "unity_fixed_time": fixed_time,
    }


def transcript_records(base_time=10.0, schedule_id=converter.EXPECTED_SCHEDULE_ID):
    records = [
        {"event": "client_pipe_server_proof", "process_id": 1200},
        {
            "event": "ack",
            "protocol": "rek.ui_bridge.v1",
            "command": "StartG1HeldInputSchedule",
            "status": "accepted",
            "reason": "g1_held_input_schedule_started",
            "applied": True,
            "client_request_issued": False,
            "server_acceptance_observed": False,
            "authoritative_execution_observed": False,
            "g1_held_schedule_schema": converter.EXPECTED_SCHEDULE_SCHEMA,
            "g1_held_schedule_id": schedule_id,
            "g1_held_schedule_sha256": converter.EXPECTED_SCHEDULE_SHA256,
            "g1_held_schedule_authority_scope": converter.EXPECTED_AUTHORITY_SCOPE,
            "g1_held_schedule_authority_caveat": converter.EXPECTED_AUTHORITY_CAVEAT,
            "g1_held_schedule_run_id": RUN_ID,
            "g1_held_schedule_fresh_round_request_id": FRESH_ROUND_ID,
            "g1_held_schedule_round_identity_sha256": ROUND_ID,
            "g1_held_schedule_running": True,
            "g1_held_schedule_tick": 0,
            "g1_held_schedule_client_fixed_substep": 0,
            "g1_held_schedule_round_capacity_proven": True,
            "build": {
                "game_assembly_sha256": held.EXPECTED_GAME_ASSEMBLY_SHA256,
                "global_metadata_sha256": held.EXPECTED_METADATA_SHA256,
                "plugin_sha256": held.EXPECTED_PLUGIN_SHA256,
                "plugin_version": held.EXPECTED_PLUGIN_VERSION,
            },
            "measured_pairing": measured_pairing(),
        },
    ]
    sequence = 0
    for tick in range(converter.EXPECTED_SCHEDULE_TICKS):
        sequence += 1
        records.append(
            sequenced_record(
                "g1_held_schedule_tick",
                sequence,
                tick,
                base_time + tick / 50.0,
                held_detail(tick),
                schedule_id,
            )
        )
    for ordinal in range(8):
        sequence += 1
        records.append(
            sequenced_record(
                "g1_kick_measurement_summary",
                sequence,
                converter.FINAL_SCHEDULE_TICK,
                base_time + converter.FINAL_SCHEDULE_TICK / 50.0,
                {"probe_ordinal": ordinal},
                schedule_id,
            )
        )
    for ordinal in range(4):
        sequence += 1
        records.append(
            sequenced_record(
                "g1_translation_release",
                sequence,
                converter.FINAL_SCHEDULE_TICK,
                base_time + converter.FINAL_SCHEDULE_TICK / 50.0,
                {"probe_ordinal": ordinal},
                schedule_id,
            )
        )
    records.extend(
        [
            {
                "event": "g1_held_schedule_end",
                "g1_held_schedule_schema": converter.EXPECTED_SCHEDULE_SCHEMA,
                "g1_held_schedule_id": schedule_id,
                "g1_held_schedule_sha256": converter.EXPECTED_SCHEDULE_SHA256,
                "g1_held_schedule_run_id": RUN_ID,
                "fresh_round_request_id": FRESH_ROUND_ID,
                "round_identity_sha256": ROUND_ID,
                "schedule_tick": converter.FINAL_SCHEDULE_TICK,
                "client_fixed_substep": converter.FINAL_SCHEDULE_SUBSTEP,
                "reason": "complete",
                "complete": True,
                "experiment_coverage_complete": True,
                "partial_coverage": False,
                "request_only": True,
                "server_acceptance": "unknown",
                "server_acceptance_observed": False,
                "authoritative_execution_observed": False,
                "global_input_emitted": False,
            },
            {
                "event": "client_result",
                "mode": "g1-held",
                "status": "complete",
                "error": None,
                "lease_held": False,
            },
        ]
    )
    return records


def trace_bones(sample_index, slot, negative_quaternions=False):
    world_quaternion = [0.0, 0.0, 0.0, -1.0 if negative_quaternions else 1.0]
    child_quaternion = [0.1, -0.2, 0.3, -0.9] if negative_quaternions else [-0.1, 0.2, -0.3, 0.9]
    rotations = [0.0, 0.0, 0.0, 0.0]
    for _ in range(29):
        rotations.extend(child_quaternion)
    source_tick = 30000 + sample_index * 10
    sequence = sample_index * 2 + slot + 1
    return {
        "layout": "g1_30",
        "source_client_fixed_tick": source_tick,
        "source_age_ticks": 0,
        "source_age_seconds": 0.0,
        "fresh_since_previous_grid_sample": True,
        "raw_bone_packet_sequence": sequence,
        "unity_frame_at_receive": sample_index,
        "unity_time_at_receive": 1.0 + sample_index / 50.0,
        "unity_unscaled_time_at_receive": 1.0 + sample_index / 50.0,
        "wire_body_sha256": f"{slot + 5:x}" * 64,
        "world_positions_xyz": [float(slot), 0.7, 0.0] * 30,
        "world_rotations_xyzw": world_quaternion * 30,
        "decoded_snapshot_sequence": sequence,
        "decoded_source_client_fixed_tick": source_tick,
        "decoded_source_age_ticks": 0,
        "decoded_source_age_seconds": 0.0,
        "decoded_snapshot_received_at_client_time": 1.0 + sample_index / 50.0,
        "decoded_root_world_position": [float(slot), 0.7, 0.0],
        "decoded_root_world_rotation_xyzw": world_quaternion,
        "decoded_child_local_rotations_xyzw": rotations,
        "reference_alignment_status": "measured_joint_transform_required",
    }


def trace_records(base_time=10.0, negative_quaternions=False):
    start_client_tick = 30000
    start = {
        "event": "trace_start",
        "schema": held.TRACE_SCHEMA,
        "source_raw_sha256": RAW_ID,
        "source_recorder_schema": held.RECORDER_SCHEMA,
        "authority": "client_request_projections_plus_client_observed_network_bones_and_roots",
        "server_acceptance_available": False,
        "server_tick_available": False,
        "root_tick_domain": "client_fixed_update",
        "trace_rate_hz": 50,
        "trace_grid_stride_client_fixed_ticks": 10,
        "start_client_fixed_tick": start_client_tick,
        "end_client_fixed_tick": start_client_tick + 20,
        "maximum_bone_source_age_ticks": 250,
        "bone_layout": {
            "id": "g1_30",
            "count": 30,
            "ordered_names": list(held.G1_BONE_NAMES),
            "ordered_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
        },
        "decoded_bone_observation": {
            "field": "decoded_child_local_rotations_xyzw",
            "stored_quaternion_count": 30,
            "articulated_child_count": 29,
            "root_quaternion_source": "decoded_root_world_rotation_xyzw",
            "reference_alignment_status": "measured_joint_transform_required",
            "direct_NPZ_angle_identity_allowed": False,
        },
    }
    samples = []
    root_quaternion = [0.2, -0.3, 0.1, -0.8] if negative_quaternions else [-0.2, 0.3, -0.1, 0.8]
    for index in range(3):
        tick = start_client_tick + index * 10
        request_sequence = 1 if index < 2 else 3
        request_tick = start_client_tick if index < 2 else tick
        moves = []
        if index == 1:
            moves.append(
                {
                    "source_client_fixed_tick": tick - 2,
                    "source_age_ticks": 2,
                    "request_sequence": 2,
                    "move_index": 6,
                    "move_profile": held.G1_KICK_PROFILES[6],
                    "request_only": True,
                    "server_acceptance": None,
                    "ack_observed": False,
                }
            )
        samples.append(
            {
                "event": "trace_sample",
                "trace_index": index,
                "client_fixed_tick": tick,
                "time_from_trace_start_seconds": index / 50.0,
                "root_source_age_ticks": 0,
                "root_pose_sample_index": 100 + index * 10,
                "utc": f"2026-09-08T00:00:{index:02d}Z",
                "stopwatch_timestamp_ticks": 100000 + index * 20000,
                "unity_frame": 500 + index,
                "unity_time": base_time + (TRACE_FIRST_SCHEDULE_TICK + index) / 50.0,
                "unity_fixed_time": (
                    base_time
                    + (TRACE_FIRST_SCHEDULE_TICK + index) / 50.0
                    + TRACE_PHASE_SUBSTEPS / 500.0
                ),
                "unity_unscaled_time": base_time + (TRACE_FIRST_SCHEDULE_TICK + index) / 50.0,
                "fight_epoch": 7,
                "round_number": 1,
                "held_condition": "W",
                "request_state": {
                    "source_client_fixed_tick": request_tick,
                    "source_age_ticks": tick - request_tick,
                    "source_age_seconds": (tick - request_tick) / 500.0,
                    "request_sequence": request_sequence,
                    "velocity_command_xyz": [1.0, 0.0, 0.0],
                    "request_only": True,
                    "server_acceptance": None,
                    "ack_observed": False,
                },
                "move_requests_since_previous_grid_sample": moves,
                "fighter_0_root": {
                    "world_position_xyz": [index * 0.01, 0.7, 0.0],
                    "world_rotation_xyzw": root_quaternion,
                },
                "fighter_1_root": {
                    "world_position_xyz": [1.0 - index * 0.01, 0.7, 0.0],
                    "world_rotation_xyzw": root_quaternion,
                },
                "fighter_0_bones": trace_bones(index, 0, negative_quaternions),
                "fighter_1_bones": trace_bones(index, 1, negative_quaternions),
            }
        )
    end = {
        "event": "trace_end",
        "schema": held.TRACE_SCHEMA,
        "sample_count": len(samples),
        "complete": True,
        "source_raw_sha256": RAW_ID,
    }
    return [start, *samples, end]


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def write_inputs(root, name, *, base_time=10.0, negative_quaternions=False):
    inventory = root / f"{name}.inventory.json"
    transcript = root / f"{name}.schedule.jsonl"
    source = root / f"{name}.motion.jsonl"
    write_json(inventory, inventory_document())
    write_jsonl(transcript, transcript_records(base_time=base_time))
    write_jsonl(
        source,
        trace_records(base_time=base_time, negative_quaternions=negative_quaternions),
    )
    return source, transcript, inventory


class G1HeldTraceToRekTraceTests(unittest.TestCase):
    def test_loads_normalizes_sign_deduplicates_and_baselines_two_repeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_inputs = write_inputs(root, "first", base_time=10.0)
            second_inputs = write_inputs(
                root,
                "second",
                base_time=110.0,
                negative_quaternions=True,
            )
            first_output = root / "first.rektrace"
            second_output = root / "second.rektrace"
            first_result = converter.convert(*first_inputs, first_output)
            second_result = converter.convert(*second_inputs, second_output)

            self.assertEqual(first_result["frame_count"], 3)
            self.assertEqual(first_result["channel_count"], 249)
            self.assertEqual(first_result["request_event_count"], 3)
            self.assertEqual(first_result["command_sample_phase_substeps"], 2)
            self.assertEqual(second_result["decoded_child_local_quaternions"], "included")

            first = Trace.load(first_output)
            second = Trace.load(second_output)
            self.assertEqual(first.ticks, [0, 1, 2])
            self.assertEqual(second.ticks, [0, 1, 2])
            self.assertEqual(first.authority, "unknown")
            self.assertFalse(first.header["authority_limits"]["parity_claim_supported"])
            self.assertEqual(
                first.header["command_sequence_sha256"],
                converter.EXPECTED_SCHEDULE_SHA256,
            )
            self.assertEqual(
                first.header["command_sequence_schema"],
                converter.EXPECTED_SCHEDULE_SCHEMA,
            )
            self.assertEqual(first.header["command_sample_phase_substeps"], 2)
            self.assertEqual(len(first.events), 3)
            self.assertEqual([event["request_sequence"] for event in first.events], [1, 2, 3])
            self.assertEqual(first.channels, second.channels)
            self.assertEqual(first.channels["root.0.quat.w"], [0.8, 0.8, 0.8])
            child = "joint.0.01_left_hip_pitch_link.local.quat.w"
            self.assertEqual(first.channels[child], [0.9, 0.9, 0.9])
            self.assertEqual(first.channels[child], second.channels[child])

            envelope = root / "envelope.json"
            self.assertEqual(
                differ.baseline([first_output, second_output], envelope, "p99"),
                0,
            )
            baseline = json.loads(envelope.read_text(encoding="utf-8"))
            self.assertEqual(baseline["runs"], 2)
            self.assertEqual(baseline["ticks_compared"], 3)
            self.assertEqual(len(baseline["channels"]), 249)
            self.assertTrue(
                all(channel["max"] == 0.0 for channel in baseline["channels"].values())
            )

    def test_missing_child_local_value_omits_whole_optional_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, transcript, inventory = write_inputs(root, "missing")
            records = trace_records()
            del records[2]["fighter_1_bones"]["decoded_child_local_rotations_xyzw"]
            write_jsonl(source, records)
            output = root / "missing.rektrace"

            result = converter.convert(source, transcript, inventory, output)
            loaded = Trace.load(output)
            self.assertEqual(result["decoded_child_local_quaternions"], "omitted")
            self.assertEqual(result["channel_count"], 17)
            self.assertFalse(any(".local.quat." in name for name in loaded.channels))
            family = loaded.header["optional_channel_families"][
                "decoded_child_local_quaternions"
            ]
            self.assertEqual(family["status"], "omitted")
            self.assertIn("sample_1_fighter_1", family["missing_locations"])

    def test_rejects_nonfinite_consumed_or_unconsumed_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, transcript, inventory = write_inputs(root, "nonfinite")
            records = trace_records()
            records[1]["fighter_0_root"]["world_position_xyz"][0] = math.nan
            write_jsonl(source, records)
            with self.assertRaisesRegex(converter.ConversionError, "nonfinite"):
                converter.convert(source, transcript, inventory, root / "bad.rektrace")

    def test_rejects_source_hash_grid_and_bone_layout_mismatches(self):
        mutations = (
            (
                "source_hash",
                lambda records: records[-1].__setitem__("source_raw_sha256", "9" * 64),
                "source_raw_sha256_end_mismatch",
            ),
            (
                "grid",
                lambda records: records[2].__setitem__("client_fixed_tick", 30011),
                "grid_tick_mismatch",
            ),
            (
                "layout",
                lambda records: records[0]["bone_layout"].__setitem__(
                    "ordered_signature_sha256", "9" * 64
                ),
                "bone_signature_mismatch",
            ),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, transcript, inventory = write_inputs(root, name)
                records = trace_records()
                mutate(records)
                write_jsonl(source, records)
                with self.assertRaisesRegex(converter.ConversionError, error):
                    converter.convert(source, transcript, inventory, root / "bad.rektrace")

    def test_rejects_wrong_schedule_id_and_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, transcript, inventory = write_inputs(root, "schedule-id")
            write_jsonl(transcript, transcript_records(schedule_id="wrong.schedule"))
            with self.assertRaisesRegex(converter.ConversionError, "schedule_id_mismatch"):
                converter.convert(source, transcript, inventory, root / "bad-id.rektrace")

            write_jsonl(transcript, transcript_records())
            records = transcript_records()
            records[1]["measured_pairing"]["opponent_slot"] = 0
            write_jsonl(transcript, records)
            with self.assertRaisesRegex(converter.ConversionError, "opponent_slot_invalid"):
                converter.convert(source, transcript, inventory, root / "bad-pair.rektrace")

    def test_rejects_partial_paths_and_existing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, transcript, inventory = write_inputs(root, "atomic")
            partial = root / "capture.partial-7.jsonl"
            partial.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(converter.ConversionError, "partial_path_rejected"):
                converter.convert(partial, transcript, inventory, root / "partial.rektrace")

            output = root / "existing.rektrace"
            output.write_bytes(b"preserve me")
            with self.assertRaisesRegex(converter.ConversionError, "output_exists"):
                converter.convert(source, transcript, inventory, output)
            self.assertEqual(output.read_bytes(), b"preserve me")


if __name__ == "__main__":
    unittest.main()
