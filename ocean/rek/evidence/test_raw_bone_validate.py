import base64
import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import raw_bone_validate


PINNED_TEST_LAYOUTS = {
    "t800_26": (
        "LINK_BASE",
        "LINK_HIP_PITCH_L",
        "LINK_HIP_ROLL_L",
        "LINK_HIP_YAW_L",
        "LINK_KNEE_PITCH_L",
        "LINK_ANKLE_PITCH_L",
        "LINK_ANKLE_ROLL_L",
        "LINK_HIP_PITCH_R",
        "LINK_HIP_ROLL_R",
        "LINK_HIP_YAW_R",
        "LINK_KNEE_PITCH_R",
        "LINK_ANKLE_PITCH_R",
        "LINK_ANKLE_ROLL_R",
        "LINK_WAIST_YAW",
        "LINK_SHOULDER_PITCH_L",
        "LINK_SHOULDER_ROLL_L",
        "LINK_SHOULDER_YAW_L",
        "LINK_ELBOW_PITCH_L",
        "LINK_ELBOW_YAW_L",
        "LINK_SHOULDER_PITCH_R",
        "LINK_SHOULDER_ROLL_R",
        "LINK_SHOULDER_YAW_R",
        "LINK_ELBOW_PITCH_R",
        "LINK_ELBOW_YAW_R",
        "LINK_HEAD_PITCH",
        "LINK_HEAD_YAW",
    ),
    "g1_30": (
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ),
}


def bone_names(layout_id="t800_26"):
    return list(PINNED_TEST_LAYOUTS[layout_id])


def bone_count(layout_id="t800_26"):
    return len(PINNED_TEST_LAYOUTS[layout_id])


def encoded_body(body):
    return {
        "wire_body_bytes": len(body),
        "wire_body_sha256": hashlib.sha256(body).hexdigest(),
        "wire_body_base64": base64.b64encode(body).decode("ascii"),
    }


def request_clock(sequence, timestamp):
    return {
        "request_sequence": sequence,
        "client_fixed_tick_at_observation": sequence,
        "unity_frame": 100 + sequence,
        "unity_realtime_since_startup": timestamp,
        "server_tick": None,
        "server_acceptance": None,
        "ack_observed": False,
        "request_only": True,
    }


def input_request(sequence=1, timestamp=0.10):
    source_index = 257
    velocity = [1.25, -2.5, 0.125]
    body = bytes((source_index & 0xFF,)) + struct.pack("<3f", *velocity)
    return {
        "event": "outbound_request_projection",
        "message": "REK_Input",
        **request_clock(sequence, timestamp),
        "message_request_sequence": 1,
        "wire_delivery": "unreliable",
        **encoded_body(body),
        "network_index_source_int32": source_index,
        "network_index_wire_uint8": source_index & 0xFF,
        "velocity_command_xyz": velocity,
        "velocity_float32_bit_patterns": [
            f"0x{value:08x}" for value in struct.unpack("<3I", body[1:])
        ],
        "native_method": "REKApp.RobotInputController.SendVelocityCommand RVA 0x226F110",
        "provenance": "exact packed request projection from source fields at REKApp.RobotInputController.SendVelocityCommand prefix",
        "semantic_limit": "fixture",
    }


def move_request(sequence=2, timestamp=0.20):
    source_index = 257
    source_move = 266
    body = bytes((source_index & 0xFF, source_move & 0xFF))
    return {
        "event": "outbound_request_projection",
        "message": "REK_Move",
        **request_clock(sequence, timestamp),
        "message_request_sequence": 1,
        "wire_delivery": "reliable",
        **encoded_body(body),
        "network_index_source_int32": source_index,
        "network_index_wire_uint8": source_index & 0xFF,
        "move_index_source_int32": source_move,
        "move_index_wire_uint8": source_move & 0xFF,
        "native_method": "REKApp.RobotInputController.SendMoveEvent RVA 0x226ECB0",
        "provenance": "exact packed request projection from source fields at REKApp.RobotInputController.SendMoveEvent prefix",
        "semantic_limit": "fixture",
    }


def invocation_request(method, sequence, timestamp):
    message = {
        "SendSpecialEvent": "REK_Special",
        "SendEStopToggle": "REK_EStop",
    }[method]
    return {
        "event": "client_transport_method_invoked",
        **request_clock(sequence, timestamp),
        "method_request_sequence": 1,
        "method": method,
        "message": message,
        "wire_body_bytes": None,
        "wire_body_sha256": None,
        "wire_body_base64": None,
        "wire_delivery": None,
        "provenance": f"REKApp.RobotInputController.{method} prefix invocation observation",
        "semantic_limit": "fixture",
    }


def raw_packet(sequence, slot, receipt, layout_id="t800_26"):
    count = bone_count(layout_id)
    body = bytearray((slot, count))
    positions = []
    rotations = []
    for index in range(count):
        values = (
            float(slot + index), float(index) / 10.0, -float(index),
            0.0, 0.0, 0.0, 1.0,
        )
        body.extend(struct.pack("<7f", *values))
        decoded = struct.unpack("<7f", body[-28:])
        positions.extend(decoded[:3])
        rotations.extend(decoded[3:])
    encoded = bytes(body)
    return {
        "event": "raw_bone_packet",
        "raw_bone_packet_sequence": sequence,
        "client_fixed_tick_at_observation": 5 + sequence,
        "unity_frame": 200 + sequence,
        "unity_time": receipt,
        "unity_unscaled_time": receipt,
        "monotonic_receipt_time": receipt,
        "fighter_slot": slot,
        "network_index": slot,
        "bone_count": count,
        **encoded_body(encoded),
        "bone_names": bone_names(layout_id),
        "world_positions_xyz": positions,
        "world_rotations_xyzw": rotations,
        "intended_wire_interval_seconds": 0.02,
        "intended_wire_rate_hz": 50,
        "wire_delivery": "unreliable",
        "provenance": raw_bone_validate.EXPECTED_PROVENANCE,
        "semantic_limit": "fixture",
    }


def decoded_snapshot(sequence, slot, receipt, layout_id="t800_26"):
    return {
        "event": "decoded_bone_snapshot",
        "bone_snapshot_sequence": sequence,
        "raw_bone_packet_sequence": sequence,
        "client_fixed_tick_at_observation": 5 + sequence,
        "unity_frame": 200 + sequence,
        "unity_time": receipt,
        "unity_unscaled_time": receipt,
        "fighter_slot": slot,
        "network_index": slot,
        "snapshot_ring_index": 0,
        "snapshot_ring_head_after_decode": 1,
        "snapshot_ring_count_after_decode": 1,
        "snapshot_received_at_client_time": receipt,
        "root_world_position": [float(slot), 0.0, 0.0],
        "root_world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "child_local_rotations_xyzw": (
            [0.0, 0.0, 0.0, 1.0]
            * bone_count(layout_id)
        ),
        "bone_names": bone_names(layout_id),
        "provenance": "fixture",
        "semantic_limit": "fixture",
    }


def raw_protocol_clock(protocol_sequence, type_sequence, event, receipt):
    sequence_field = {
        "raw_fight_state_packet": "raw_fight_state_sequence",
        "raw_score_packet": "raw_score_sequence",
        "raw_hit_packet": "raw_hit_sequence",
    }[event]
    return {
        "event": event,
        "raw_protocol_sequence": protocol_sequence,
        sequence_field: type_sequence,
        "client_fixed_tick_at_observation": 7 + protocol_sequence,
        "unity_frame": 300 + protocol_sequence,
        "unity_time": receipt,
        "unity_unscaled_time": receipt,
        "monotonic_receipt_time": receipt,
    }


def fight_state_packet(protocol_sequence=1, type_sequence=1, receipt=3.00):
    body = bytearray(33)
    body[0x00:0x04] = bytes((1, 2, 1, 0))
    struct.pack_into("<f", body, 0x04, 57.25)
    struct.pack_into("<hh", body, 0x08, 12, 7)
    body[0x0C:0x21] = bytes((
        0, 0, 0xFF, 1, 0, 0, 0xFF, 0, 1, 0xFF, 0, 2, 3, 4, 1, 8, 9, 2, 1, 2, 3,
    ))
    encoded = bytes(body)
    return {
        **raw_protocol_clock(
            protocol_sequence, type_sequence, "raw_fight_state_packet", receipt
        ),
        **encoded_body(encoded),
        "decoded": {
            "phase": 1,
            "phase_name": "RoundActive",
            "round_number": 2,
            "round_active": 1,
            "is_redo": 0,
            "time_remaining": 57.25,
            "hits_0": 12,
            "hits_1": 7,
            "knockout_occurred": 0,
            "round_result": 0,
            "round_result_name": "InProgress",
            "round_winner": -1,
            "rounds_won_0": 1,
            "rounds_won_1": 0,
            "fight_result": 0,
            "fight_result_name": "InProgress",
            "fight_winner": -1,
            "format": 0,
            "format_name": "BestOf3",
            "human_slot_mask": 1,
            "champion_slot": -1,
            "fault_mask": 0,
            "fault_stress_0": 2,
            "fault_stress_1": 3,
            "referee_count_mask": 4,
            "referee_count_seconds": 1,
            "referee_call_sequence": 8,
            "referee_call_type": 9,
            "referee_call_name": None,
            "referee_call_faller": 2,
            "referee_call_points": 1,
            "ai_level": 2,
            "decided_winner_bits": 3,
        },
        "wire_delivery": "reliable",
        "nominal_wire_interval_seconds": 0.1,
        "native_sender": "REKApp.FightCoordinator.ServerSendFightState RVA 0x238BFA0",
        "native_receiver": "REKApp.FightCoordinator.ApplyFightStateSnapshot RVA 0x2379E00",
        "provenance": raw_bone_validate.EXPECTED_FIGHT_STATE_PROVENANCE,
        "semantic_limit": "fixture",
    }


def fight_state_snapshot(snapshot_sequence=1, raw_protocol_sequence=1, receipt=3.00):
    return {
        "event": "fight_state_snapshot_applied",
        "fight_state_snapshot_sequence": snapshot_sequence,
        "raw_protocol_sequence": raw_protocol_sequence,
        "client_fixed_tick_at_observation": 8,
        "unity_frame": 301,
        "unity_unscaled_time": receipt,
        "phase": "RoundActive",
        "phase_value": 1,
        "round": {"number": 2},
        "fight": {"current_round": 2},
        "provenance": "REKApp.FightCoordinator.ApplyFightStateSnapshot postfix",
    }


def score_packet(protocol_sequence=2, type_sequence=1, receipt=3.01):
    body = struct.pack("<Bhf", 1, 14, 2.5)
    return {
        **raw_protocol_clock(protocol_sequence, type_sequence, "raw_score_packet", receipt),
        **encoded_body(body),
        "decoded": {
            "fighter_index": 1,
            "new_hit_count": 14,
            "points_awarded": 2.5,
        },
        "wire_delivery": "reliable",
        "native_sender": "REKApp.FightCoordinator.OnPointScoredNetwork RVA 0x23867D0",
        "native_receiver": "REKApp.FightCoordinator.OnScoreReceived RVA 0x2387010",
        "provenance": raw_bone_validate.EXPECTED_SCORE_PROVENANCE,
        "semantic_limit": "fixture",
    }


def hit_packet(protocol_sequence=3, type_sequence=1, receipt=3.02):
    values = [1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 4.5]
    body = struct.pack("<7fB", *values, 1)
    return {
        **raw_protocol_clock(protocol_sequence, type_sequence, "raw_hit_packet", receipt),
        **encoded_body(body),
        "decoded": {
            "position_xyz": values[:3],
            "surface_normal_xyz": values[3:6],
            "relative_speed": values[6],
            "is_kick": 1,
        },
        "wire_delivery": "unreliable",
        "native_sender": "REKApp.FightCoordinator.OnHitDetectedNetwork RVA 0x2385500",
        "native_receiver": "REKApp.FightCoordinator.OnHitReceived RVA 0x2385810",
        "provenance": raw_bone_validate.EXPECTED_HIT_PROVENANCE,
        "semantic_limit": "fixture",
    }


def fixture(fighter_layouts=("t800_26", "t800_26")):
    targets = {
        "REKApp.RobotInputController.SendVelocityCommand": True,
        "REKApp.RobotInputController.SendMoveEvent": True,
        "REKApp.RobotInputController.SendSpecialEvent": True,
        "REKApp.RobotInputController.SendEStopToggle": True,
        "REKApp.FightCoordinator.ApplyFightStateSnapshot": True,
        "REKApp.FightCoordinator.OnScoreReceived": True,
        "REKApp.FightCoordinator.OnHitReceived": True,
        "REKApp.Robot.OnBoneMessageReceived": True,
    }
    start = {
        "event": "capture_start",
        "schema": raw_bone_validate.SCHEMA,
        "plugin_version": raw_bone_validate.EXPECTED_PLUGIN_VERSION,
        "plugin_sha256": raw_bone_validate.EXPECTED_PLUGIN_SHA256,
        "game_assembly_sha256": raw_bone_validate.EXPECTED_GAME_ASSEMBLY_SHA256,
        "global_metadata_sha256": raw_bone_validate.EXPECTED_METADATA_SHA256,
        "tick_level_claim": False,
        "tick_domain": "client_fixed_update",
        "client_sample_stride_ticks": 10,
        "server_tick_available": False,
        "server_tick_reason": "recovered packets expose no server tick",
        "bone_wire_protocol": copy.deepcopy(raw_bone_validate.EXPECTED_BONE_PROTOCOL),
        "fight_wire_protocol": copy.deepcopy(raw_bone_validate.EXPECTED_FIGHT_PROTOCOL),
        "outbound_request_protocol": copy.deepcopy(
            raw_bone_validate.EXPECTED_OUTBOUND_PROTOCOL
        ),
        "instrumentation_hooks": list(raw_bone_validate.EXPECTED_HOOKS),
        "harmony_target_status": targets,
        "server": {
            "endpoint": "test.invalid:7777",
            "session_identifier_recorded": False,
            "session_identifier_reason": "omitted",
            "session_id_sha256": "a" * 64,
        },
        "scope": {
            "allowed": True,
            "network_connected": True,
            "network_is_client": True,
            "network_is_server": False,
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
        },
        "fighter_0_bones": bone_names(fighter_layouts[0]),
        "fighter_1_bones": bone_names(fighter_layouts[1]),
    }
    return [
        start,
        {"event": "sample", "sample_index": 0, "client_fixed_tick": 0},
        input_request(),
        move_request(),
        invocation_request("SendSpecialEvent", 3, 0.30),
        invocation_request("SendEStopToggle", 4, 0.40),
        raw_packet(1, 0, 2.00, fighter_layouts[0]),
        decoded_snapshot(1, 0, 2.00, fighter_layouts[0]),
        fight_state_packet(),
        fight_state_snapshot(),
        score_packet(),
        hit_packet(),
        {"event": "sample", "sample_index": 1, "client_fixed_tick": 10},
        raw_packet(2, 1, 3.03, fighter_layouts[1]),
        decoded_snapshot(2, 1, 3.03, fighter_layouts[1]),
        {
            "event": "capture_end",
            "sample_count": 2,
            "raw_bone_packet_count": 2,
            "decoded_bone_snapshot_count": 2,
            "client_transport_invocation_count": 4,
            "client_transport_method_counts": {
                "SendVelocityCommand": 1,
                "SendMoveEvent": 1,
                "SendSpecialEvent": 1,
                "SendEStopToggle": 1,
            },
            "fight_state_snapshot_count": 1,
            "raw_protocol_packet_count": 3,
            "raw_fight_state_packet_count": 1,
            "raw_score_packet_count": 1,
            "raw_hit_packet_count": 1,
            "capture_error_count": 0,
            "client_fixed_tick_at_end": 11,
        },
    ]


def event(records, name, message=None):
    for record in records:
        if record.get("event") == name and (
            message is None or record.get("message") == message
        ):
            return record
    raise AssertionError(f"missing fixture event {name}/{message}")


class RawBoneValidateTests(unittest.TestCase):
    def write_fixture(self, root, records):
        path = root / "capture.jsonl"
        path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n"
                    for record in records),
            encoding="utf-8",
        )
        return path

    def assert_rejected(self, records, pattern=None):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(Path(directory), records)
            context = (
                self.assertRaisesRegex(raw_bone_validate.EvidenceError, pattern)
                if pattern else self.assertRaises(raw_bone_validate.EvidenceError)
            )
            with context:
                raw_bone_validate.validate(path)

    def test_validates_v5_protocol_bytes_and_preserves_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(Path(directory), fixture())
            report = raw_bone_validate.validate(path)
        self.assertEqual(report["raw_bone_packets"], 2)
        self.assertEqual(report["outbound_requests"]["REK_Input"], 1)
        self.assertEqual(report["outbound_requests"]["REK_Move"], 1)
        self.assertEqual(report["fight_protocol"]["fight_state_packets"], 1)
        self.assertEqual(report["fight_protocol"]["score_packets"], 1)
        self.assertEqual(report["fight_protocol"]["hit_packets"], 1)
        self.assertTrue(report["claims"]["raw_wire_pose_payload_validated"])
        self.assertTrue(report["claims"]["raw_fight_protocol_payloads_validated"])
        self.assertFalse(report["claims"]["server_tick_available"])
        self.assertFalse(report["claims"]["request_projection_is_send_completion"])
        self.assertFalse(report["claims"]["request_to_snapshot_causality_established"])
        self.assertFalse(report["claims"]["client_send_frame_observed"])
        self.assertFalse(report["server_identity"]["raw_session_identity_recorded"])

    def test_pinned_wire_layout_contracts(self):
        self.assertEqual(
            raw_bone_validate.T800_BONE_NAMES, PINNED_TEST_LAYOUTS["t800_26"]
        )
        self.assertEqual(raw_bone_validate.G1_BONE_NAMES, PINNED_TEST_LAYOUTS["g1_30"])
        self.assertEqual(raw_bone_validate.T800_BONE_COUNT, 26)
        self.assertEqual(raw_bone_validate.T800_BODY_BYTES, 730)
        self.assertEqual(raw_bone_validate.G1_BONE_COUNT, 30)
        self.assertEqual(raw_bone_validate.G1_BODY_BYTES, 842)

    def test_validates_t800_26_layout_with_measured_runtime_object_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(Path(directory), fixture())
            report = raw_bone_validate.validate(path)
        for fighter in report["fighters"].values():
            self.assertEqual(fighter["bone_layout"]["layout_id"], "t800_26")
            self.assertEqual(fighter["bone_layout"]["bone_count"], 26)
            self.assertEqual(fighter["bone_layout"]["wire_body_bytes"], 730)
            self.assertTrue(fighter["bone_layout"]["identity_claimed"])
            self.assertEqual(
                fighter["bone_layout"]["runtime_object_name"],
                "engineai_t800_FactoryPolicy(Clone)",
            )
            self.assertIn("measured", fighter["bone_layout"]["identity_basis"])

    def test_validates_g1_30_layout_with_measured_runtime_object_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(
                Path(directory), fixture(("g1_30", "g1_30"))
            )
            report = raw_bone_validate.validate(path)
        for fighter in report["fighters"].values():
            self.assertEqual(fighter["bone_layout"]["layout_id"], "g1_30")
            self.assertEqual(fighter["bone_layout"]["bone_count"], 30)
            self.assertEqual(fighter["bone_layout"]["wire_body_bytes"], 842)
            self.assertTrue(fighter["bone_layout"]["identity_claimed"])
            self.assertEqual(
                fighter["bone_layout"]["runtime_object_name"],
                "g1_29dof_Prefab_SONIC(Clone)",
            )
            self.assertIn("measured", fighter["bone_layout"]["identity_basis"])

    def test_validates_mixed_26_and_30_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(
                Path(directory), fixture(("t800_26", "g1_30"))
            )
            report = raw_bone_validate.validate(path)
        self.assertEqual(
            report["fighters"]["0"]["bone_layout"]["layout_id"], "t800_26"
        )
        self.assertEqual(
            report["fighters"]["1"]["bone_layout"]["layout_id"], "g1_30"
        )

    def test_accepts_json_decimal_that_round_trips_to_same_float32(self):
        records = fixture()
        packet = event(records, "raw_bone_packet")
        self.assertNotEqual(packet["world_positions_xyz"][4], 0.1)
        packet["world_positions_xyz"][4] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(Path(directory), records)
            raw_bone_validate.validate(path)

    def test_accepts_shortest_decimal_for_fight_state_float32(self):
        records = fixture()
        packet = event(records, "raw_fight_state_packet")
        body = bytearray(base64.b64decode(packet["wire_body_base64"]))
        struct.pack_into("<f", body, 4, 119.83323)
        packet.update(encoded_body(bytes(body)))
        packet["decoded"]["time_remaining"] = 119.83323
        self.assertNotEqual(
            packet["decoded"]["time_remaining"], struct.unpack_from("<f", body, 4)[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(Path(directory), records)
            raw_bone_validate.validate(path)

    def test_preserves_system_text_json_integer_negative_zero(self):
        records = fixture()
        encoded = "".join(
            json.dumps(record, separators=(",", ":")) + "\n" for record in records
        )
        self.assertIn("-0.0", encoded)
        encoded = encoded.replace("-0.0", "-0", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(encoded, encoding="utf-8")
            raw_bone_validate.validate(path)

    def test_rejects_lost_negative_zero_sign(self):
        records = fixture()
        encoded = "".join(
            json.dumps(record, separators=(",", ":")) + "\n" for record in records
        )
        self.assertIn("-0.0", encoded)
        encoded = encoded.replace("-0.0", "0", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(encoded, encoding="utf-8")
            with self.assertRaisesRegex(
                raw_bone_validate.EvidenceError, "decoded transforms disagree with body"
            ):
                raw_bone_validate.validate(path)

    def test_rejects_json_decimal_that_maps_to_different_float32(self):
        records = fixture()
        event(records, "raw_fight_state_packet")["decoded"][
            "time_remaining"
        ] += 0.001
        self.assert_rejected(records, "authoritative wire float32")

    def test_rejects_unknown_header_layout(self):
        records = fixture()
        records[0]["fighter_0_bones"] = [f"unknown_{index}" for index in range(30)]
        self.assert_rejected(records, "does not match a pinned bone layout")

    def test_rejects_packet_layout_mismatched_to_fighter_header(self):
        records = fixture()
        packet_index = records.index(event(records, "raw_bone_packet"))
        records[packet_index] = raw_packet(1, 0, 2.00, "g1_30")
        self.assert_rejected(records, "has 30 bones.*declares t800_26 with 26")

    def test_rejects_packet_name_order_mismatched_to_fighter_header(self):
        records = fixture(("t800_26", "g1_30"))
        packet = event(records, "raw_bone_packet")
        packet["bone_names"][0], packet["bone_names"][1] = (
            packet["bone_names"][1], packet["bone_names"][0]
        )
        self.assert_rejected(records, "raw packet 1 bone names disagree")

    def test_rejects_packet_body_length_mismatched_to_layout(self):
        records = fixture(("t800_26", "g1_30"))
        packet = event(records, "raw_bone_packet")
        body = base64.b64decode(packet["wire_body_base64"]) + bytes(28)
        packet.update(encoded_body(body))
        self.assert_rejected(records, "raw bone packet 1 byte count is not 730")

    def test_rejects_packet_coordinate_length_mismatched_to_layout(self):
        records = fixture(("t800_26", "g1_30"))
        event(records, "raw_bone_packet")["world_positions_xyz"].extend(
            [0.0, 0.0, 0.0]
        )
        self.assert_rejected(
            records, "raw packet 1 world positions must contain exactly 78 values"
        )

    def test_rejects_decoded_layout_mismatched_to_fighter_header(self):
        records = fixture()
        snapshot = event(records, "decoded_bone_snapshot")
        snapshot["bone_names"] = bone_names("g1_30")
        snapshot["child_local_rotations_xyzw"] = [0.0, 0.0, 0.0, 1.0] * 30
        self.assert_rejected(records, "decoded snapshot bone names disagree")

    def test_rejects_decoded_coordinate_length_mismatch(self):
        records = fixture(("t800_26", "g1_30"))
        snapshot = event(records, "decoded_bone_snapshot")
        snapshot["child_local_rotations_xyzw"].extend([0.0, 0.0, 0.0, 1.0])
        self.assert_rejected(
            records, "decoded child local rotations must contain exactly 104 values"
        )

    def test_rejects_wrong_recorder_schema(self):
        records = fixture()
        records[0]["schema"] = "rek.private_ai.raw_snapshot.v4"
        self.assert_rejected(records, "unsupported recorder schema")

    def test_rejects_wrong_recorder_binary_hash(self):
        records = fixture()
        records[0]["plugin_sha256"] = "0" * 64
        self.assert_rejected(records, "plugin hash mismatch")

    def test_rejects_input_projection_drift(self):
        records = fixture()
        event(records, "outbound_request_projection", "REK_Input")[
            "velocity_command_xyz"
        ][0] += 1.0
        self.assert_rejected(records, "velocity\\[0\\].*authoritative wire float32")

    def test_rejects_input_bit_pattern_drift(self):
        records = fixture()
        event(records, "outbound_request_projection", "REK_Input")[
            "velocity_float32_bit_patterns"
        ][0] = "0x00000000"
        self.assert_rejected(records, "velocity bit patterns disagree")

    def test_rejects_move_uint8_projection_drift(self):
        records = fixture()
        event(records, "outbound_request_projection", "REK_Move")[
            "move_index_wire_uint8"
        ] = 11
        self.assert_rejected(records, "move-index uint8 projection mismatch")

    def test_rejects_claimed_server_acceptance(self):
        records = fixture()
        event(records, "outbound_request_projection", "REK_Move")[
            "server_acceptance"
        ] = True
        self.assert_rejected(records, "server_acceptance must remain explicitly null")

    def test_rejects_invocation_only_wire_claim(self):
        records = fixture()
        event(records, "client_transport_method_invoked", "REK_Special")[
            "wire_delivery"
        ] = "reliable"
        self.assert_rejected(records, "wire_delivery must remain explicitly null")

    def test_rejects_client_send_frame_target(self):
        records = fixture()
        records[0]["harmony_target_status"][
            "REKApp.RobotInputController.ClientSendFrame"
        ] = True
        self.assert_rejected(records, "ClientSendFrame")

    def test_rejects_unowned_score_receiver(self):
        records = fixture()
        records[0]["harmony_target_status"][
            "REKApp.FightCoordinator.OnScoreReceived"
        ] = False
        self.assert_rejected(records, "Harmony ownership")

    def test_rejects_client_send_frame_record(self):
        records = fixture()
        records.insert(-1, {
            "event": "client_transport_method_invoked",
            "method": "ClientSendFrame",
        })
        self.assert_rejected(records, "ClientSendFrame")

    def test_rejects_fight_state_decoded_drift(self):
        records = fixture()
        event(records, "raw_fight_state_packet")["decoded"]["round_number"] = 3
        self.assert_rejected(records, "round_number.*audited decoder")

    def test_rejects_score_body_length_drift(self):
        records = fixture()
        score = event(records, "raw_score_packet")
        short_body = base64.b64decode(score["wire_body_base64"])[:-1]
        score.update(encoded_body(short_body))
        self.assert_rejected(records, "byte count is not 7")

    def test_rejects_hit_decoded_drift(self):
        records = fixture()
        event(records, "raw_hit_packet")["decoded"]["relative_speed"] += 0.5
        self.assert_rejected(records, "relative_speed.*authoritative wire float32")

    def test_rejects_raw_protocol_sequence_drift(self):
        records = fixture()
        event(records, "raw_score_packet")["raw_protocol_sequence"] = 3
        self.assert_rejected(records, "raw fight-protocol sequence")

    def test_rejects_fight_state_snapshot_miscorrelation(self):
        records = fixture()
        event(records, "fight_state_snapshot_applied")["raw_protocol_sequence"] = 2
        self.assert_rejected(records, "absent raw fight-state packet")

    def test_rejects_protocol_count_drift(self):
        records = fixture()
        event(records, "capture_end")["raw_hit_packet_count"] = 2
        self.assert_rejected(records, "raw_hit_packet_count disagrees")

    def test_rejects_decoded_bone_float_drift(self):
        records = fixture()
        event(records, "raw_bone_packet")["world_positions_xyz"][0] += 1.0
        self.assert_rejected(records, "transforms disagree")

    def test_rejects_raw_session_identifier(self):
        records = fixture()
        records[0]["server"]["session_id"] = "must-not-persist"
        self.assert_rejected(records, "raw session")

    def test_rejects_missing_raw_to_decoded_bone_correlation(self):
        records = fixture()
        snapshots = [
            record for record in records
            if record.get("event") == "decoded_bone_snapshot"
        ]
        snapshots[1]["raw_bone_packet_sequence"] = 1
        self.assert_rejected(records)

    def test_rejects_undeclared_sample_stride(self):
        records = fixture()
        samples = [record for record in records if record.get("event") == "sample"]
        samples[1]["client_fixed_tick"] = 9
        self.assert_rejected(records, "tick stride")


if __name__ == "__main__":
    unittest.main()
