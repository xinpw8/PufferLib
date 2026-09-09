"""Focused tests for the fail-closed G1 held-motion trace extractor."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_held_trace_extract as held


def capture_start():
    fighter = {
        "bone_count": 30,
        "ordered_bone_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
        "exact_g1_bone_signature": True,
    }
    return {
        "event": "capture_start",
        "schema": held.RECORDER_SCHEMA,
        "plugin_version": held.EXPECTED_PLUGIN_VERSION,
        "plugin_sha256": held.EXPECTED_PLUGIN_SHA256,
        "game_assembly_sha256": held.EXPECTED_GAME_ASSEMBLY_SHA256,
        "global_metadata_sha256": held.EXPECTED_METADATA_SHA256,
        "client_sample_stride_ticks": 10,
        "root_pose_sample_stride_ticks": 1,
        "root_pose_sample_rate_hz": 500,
        "fixed_delta_time": 0.002,
        "scope": {
            "allowed": True,
            "network_connected": True,
            "network_is_client": True,
            "network_is_server": False,
            "context_is_solo": True,
            "context_is_ranked": False,
            "context_auto_find_match": False,
            "solo_route_hooks_verified": True,
            "solo_route_proven": True,
            "solo_route_flow": "solo",
            "solo_route_connect_to_arena_observed": True,
            "solo_route_enter_championship_observed": True,
            "solo_route_enter_championship_koth": False,
            "solo_route_enter_championship_solo": True,
            "solo_route_arena_identity_consistent": True,
            "solo_route_runtime_session_identity_consistent": True,
            "solo_route_reason": "solo_route_proven",
            "server_private_proven": False,
            "server_private_status": "unknown",
            "coordinator_is_ranked_arena": False,
            "local_fighter_index": 0,
            "opponent_slot": 1,
            "opponent_is_ai": True,
            "opponent_slot_is_ai": True,
            "human_in_opponent_slot": False,
            "opponent_slot_has_client": False,
            "opponent_human_bit_set": False,
            "fighter_0_visual_only": True,
            "fighter_1_visual_only": True,
            "sparring_bot_number": 1,
            "client_ai_difficulty": 0,
            "exact_supported_runtime_pairing": True,
            "runtime_model": "g1",
            "exact_t800_vs_t800": False,
            "exact_g1_vs_g1": True,
        },
        "pairing": {
            "required_pairing": "exact_homogeneous_supported_runtime_pair",
            "runtime_model": "g1",
            "exact_supported_runtime_pairing": True,
            "exact_t800_vs_t800": False,
            "exact_g1_vs_g1": True,
            "semantic_robot_id_required_for_acceptance": False,
            "required_g1_bone_count": 30,
            "required_g1_bone_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
            "fighter_0": copy.deepcopy(fighter),
            "fighter_1": copy.deepcopy(fighter),
        },
        "fighter_0_bones": list(held.G1_BONE_NAMES),
        "fighter_1_bones": list(held.G1_BONE_NAMES),
        "bone_wire_protocol": {
            "g1_bone_count": 30,
            "g1_body_bytes": 842,
            "g1_ordered_bone_signature_sha256": held.G1_BONE_SIGNATURE_SHA256,
            "delivery": "unreliable",
        },
        "outbound_request_protocol": {
            "server_tick_available": False,
            "server_acceptance_available": False,
            "acknowledgement_observed": False,
        },
        "harmony_target_status": {
            "REKApp.RobotInputController.SendVelocityCommand": True,
            "REKApp.RobotInputController.SendMoveEvent": True,
            "REKApp.RobotInputController.SendSpecialEvent": True,
            "REKApp.RobotInputController.SendEStopToggle": True,
            "REKApp.Robot.OnBoneMessageReceived": True,
        },
        "server": {
            "endpoint_present": True,
            "endpoint_recorded": False,
            "session_id_sha256": "c" * 64,
        },
    }


def root_sample(tick, keep_moving=False):
    if tick < 20:
        local_x = 0.0
    elif keep_moving:
        local_x = (tick - 20) * 0.001
    elif tick < 190:
        local_x = min(tick - 20, 170) * 0.001
    else:
        local_x = 0.17
    return {
        "event": "root_pose_sample",
        "root_pose_sample_index": tick,
        "client_fixed_tick": tick,
        "utc": f"2026-09-04T00:00:{tick / 500:09.6f}+00:00",
        "stopwatch_timestamp_ticks": tick * 20000,
        "unity_frame": tick // 8,
        "unity_time": tick / 500,
        "unity_fixed_time": tick / 500,
        "unity_unscaled_time": tick / 500,
        "fight_epoch": 7,
        "round_number": 1,
        "local_fighter_index": 0,
        "opponent_slot": 1,
        "fighter_0_root": {
            "world_position_xyz": [local_x, 0.7, 0.0],
            "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "fighter_1_root": {
            "world_position_xyz": [1.0, 0.7, 0.0],
            "world_rotation_xyzw": [0.0, 1.0, 0.0, 0.0],
        },
    }


def input_request(tick, sequence, velocity):
    return {
        "event": "outbound_request_projection",
        "message": "REK_Input",
        "request_sequence": sequence,
        "message_request_sequence": sequence,
        "client_fixed_tick_at_observation": tick,
        "utc": f"2026-09-04T00:00:{tick / 500:09.6f}+00:00",
        "stopwatch_timestamp_ticks": tick * 20000,
        "unity_frame": tick // 8,
        "network_index_source_int32": 0,
        "velocity_command_xyz": list(velocity),
        "wire_delivery": "unreliable",
        "request_only": True,
        "server_acceptance": None,
        "ack_observed": False,
    }


def move_request(tick, sequence, move_index=6):
    return {
        "event": "outbound_request_projection",
        "message": "REK_Move",
        "request_sequence": sequence,
        "message_request_sequence": 1,
        "client_fixed_tick_at_observation": tick,
        "utc": f"2026-09-04T00:00:{tick / 500:09.6f}+00:00",
        "stopwatch_timestamp_ticks": tick * 20000,
        "unity_frame": tick // 8,
        "network_index_source_int32": 0,
        "move_index_source_int32": move_index,
        "wire_delivery": "reliable",
        "request_only": True,
        "server_acceptance": None,
        "ack_observed": False,
    }


def bone_packet(tick, slot, sequence):
    positions = []
    rotations = []
    for index in range(30):
        positions.extend([slot + index * 0.01, 0.7, tick * 0.0001])
        rotations.extend([0.0, 0.0, 0.0, 1.0])
    return {
        "event": "raw_bone_packet",
        "raw_bone_packet_sequence": sequence,
        "client_fixed_tick_at_observation": tick,
        "unity_frame": tick // 8,
        "unity_time": tick / 500,
        "unity_unscaled_time": tick / 500,
        "fighter_slot": slot,
        "network_index": slot,
        "bone_count": 30,
        "wire_body_bytes": 842,
        "wire_body_sha256": "b" * 64,
        "bone_names": list(held.G1_BONE_NAMES),
        "world_positions_xyz": positions,
        "world_rotations_xyzw": rotations,
    }


def decoded_bone_snapshot(tick, slot, sequence):
    rotations = [0.0, 0.0, 0.0, 0.0]
    for _index in range(29):
        rotations.extend([0.0, 0.0, 0.0, 1.0])
    return {
        "event": "decoded_bone_snapshot",
        "bone_snapshot_sequence": sequence,
        "raw_bone_packet_sequence": sequence,
        "client_fixed_tick_at_observation": tick,
        "fighter_slot": slot,
        "snapshot_received_at_client_time": tick / 500,
        "root_world_position": [float(slot), 0.7, 0.0],
        "root_world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "child_local_rotations_xyzw": rotations,
        "bone_names": list(held.G1_BONE_NAMES),
    }


def fixture(keep_moving=False):
    records = [capture_start()]
    requests = {
        0: [input_request(0, 1, [0.0, 0.0, 0.0])],
        20: [input_request(20, 2, [0.8, 0.0, 0.0])],
        80: [input_request(80, 3, [0.8, 0.0, 1.5])],
        140: [input_request(140, 4, [0.0, 0.0, 0.0])],
        300: [move_request(300, 5)],
        310: [input_request(310, 6, [0.0, 0.0, 0.0])],
    }
    raw_bone_sequence = 0
    raw_bone_count = 0
    for tick in range(401):
        records.append(root_sample(tick, keep_moving=keep_moving))
        records.extend(requests.get(tick, []))
        if tick % 16 == 0:
            for slot in (0, 1):
                raw_bone_sequence += 1
                raw_bone_count += 1
                records.append(bone_packet(tick, slot, raw_bone_sequence))
                records.append(decoded_bone_snapshot(tick, slot, raw_bone_sequence))
    records.append(
        {
            "event": "capture_end",
            "reason": "scope_exit:round_not_active",
            "client_fixed_tick_at_end": 401,
            "root_pose_sample_count": 401,
            "capture_error_count": 0,
            "raw_bone_packet_count": raw_bone_count,
            "decoded_bone_snapshot_count": raw_bone_count,
            "client_transport_invocation_count": 6,
            "client_transport_method_counts": {
                "SendVelocityCommand": 5,
                "SendMoveEvent": 1,
            },
        }
    )
    return records


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


class G1HeldTraceExtractTests(unittest.TestCase):
    def test_native_controller_axis_signs_map_to_key_labels(self):
        self.assertEqual(held.classify_velocity([0.0, 1.0, 0.0]), "A")
        self.assertEqual(held.classify_velocity([0.0, -1.0, 0.0]), "D")
        self.assertEqual(held.classify_velocity([0.0, 0.0, 1.0]), "Q")
        self.assertEqual(held.classify_velocity([0.0, 0.0, -1.0]), "E")
        self.assertEqual(held.classify_velocity([1.0, 0.0, 1.0]), "W+Q")

    def test_extracts_grid_bones_coverage_and_request_only_attack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "capture.jsonl"
            trace = root / "trace.jsonl"
            coverage_path = root / "coverage.json"
            write_jsonl(raw, fixture())

            coverage = held.extract(raw, trace, coverage_path)

            self.assertEqual(coverage["schema"], held.COVERAGE_SCHEMA)
            self.assertEqual(coverage["trace_grid"]["rate_hz"], 50)
            self.assertEqual(coverage["trace_grid"]["sample_count"], 41)
            conditions = coverage["held_condition_coverage"]["conditions"]
            self.assertEqual(conditions["W"]["status"], "observed")
            self.assertEqual(conditions["W+Q"]["status"], "observed")
            self.assertEqual(conditions["S"]["status"], "absent")
            self.assertIn("D+E", coverage["held_condition_coverage"]["missing_conditions"])
            self.assertFalse(coverage["held_condition_coverage"]["complete"])

            attack = coverage["attacks"]["requests"][0]
            self.assertEqual(attack["move_profile"], "left_side_kick_processed")
            self.assertEqual(attack["server_acceptance"]["status"], "unknown")
            self.assertIsNone(attack["server_acceptance"]["value"])
            self.assertEqual(attack["translation_release_request_tick"], 140)
            self.assertTrue(attack["translation_settle_observation"]["settled"])
            self.assertTrue(attack["post_request_yaw_projection"]["all_zero"])
            self.assertEqual(attack["accepted_kick_yaw_suppression"]["status"], "unknown")

            lines = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines[0]["event"], "trace_start")
            self.assertEqual(lines[-1]["event"], "trace_end")
            samples = lines[1:-1]
            self.assertEqual(len(samples), 41)
            self.assertEqual([sample["client_fixed_tick"] for sample in samples], list(range(0, 401, 10)))
            self.assertTrue(all(sample["root_source_age_ticks"] == 0 for sample in samples))
            self.assertTrue(
                all(sample["fighter_0_bones"]["source_age_ticks"] <= 15 for sample in samples)
            )
            self.assertTrue(
                all(
                    len(sample["fighter_0_bones"]["decoded_child_local_rotations_xyzw"])
                    == 120
                    for sample in samples
                )
            )
            self.assertEqual(
                lines[0]["decoded_bone_observation"]["reference_alignment_status"],
                "measured_joint_transform_required",
            )
            self.assertTrue(
                any(not sample["fighter_0_bones"]["fresh_since_previous_grid_sample"] for sample in samples)
            )
            disk_coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            self.assertEqual(disk_coverage["source"]["sha256"], coverage["source"]["sha256"])

    def test_accepts_v0_7_3_recorder_identity(self):
        records = fixture()
        records[0]["plugin_version"] = held.EXPECTED_RECOVERY_PLUGIN_VERSION
        records[0]["plugin_sha256"] = held.EXPECTED_RECOVERY_PLUGIN_SHA256
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "capture.jsonl"
            write_jsonl(raw, records)
            coverage = held.extract(
                raw,
                coverage_out=Path(temporary) / "coverage.json",
            )
        self.assertEqual(
            coverage["source"]["recorder_plugin_version"],
            held.EXPECTED_RECOVERY_PLUGIN_VERSION,
        )

    def test_rejects_non_g1_scope(self):
        records = fixture()
        records[0]["scope"]["runtime_model"] = "t800"
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "capture.jsonl"
            write_jsonl(raw, records)
            with self.assertRaisesRegex(held.HeldTraceError, "scope_runtime_not_g1"):
                held.extract(raw, coverage_out=Path(temporary) / "coverage.json")

    def test_rejects_stale_bone_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "capture.jsonl"
            write_jsonl(raw, fixture())
            with self.assertRaisesRegex(held.HeldTraceError, "bone_age_exceeded"):
                held.extract(
                    raw,
                    coverage_out=Path(temporary) / "coverage.json",
                    max_bone_age_ticks=5,
                )

    def test_reports_observed_root_motion_as_not_settled(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "capture.jsonl"
            write_jsonl(raw, fixture(keep_moving=True))
            coverage = held.extract(raw, coverage_out=Path(temporary) / "coverage.json")
            settle = coverage["attacks"]["requests"][0]["translation_settle_observation"]
            self.assertFalse(settle["settled"])
            self.assertEqual(settle["status"], "not_settled")
            self.assertGreater(settle["planar_path_speed_m_s"], 0.03)

    def test_rejects_uninstrumented_acceptance_claim(self):
        records = fixture()
        move = next(
            record
            for record in records
            if record.get("event") == "outbound_request_projection"
            and record.get("message") == "REK_Move"
        )
        move["server_acceptance"] = True
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "capture.jsonl"
            write_jsonl(raw, records)
            with self.assertRaisesRegex(held.HeldTraceError, "claims_server_acceptance"):
                held.extract(raw, coverage_out=Path(temporary) / "coverage.json")


if __name__ == "__main__":
    unittest.main()
