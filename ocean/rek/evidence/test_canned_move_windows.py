import json
import tempfile
import unittest
from pathlib import Path

import canned_move_windows


def sample(tick):
    robot = {
        "falling": False, "fallen": False, "dampened": False,
        "resetting": False, "tilt_angle": 0.0, "floor_contact_count": 0,
        "root_position": [0, 1, 0], "root_rotation": [0, 0, 0, 1],
        "root_linear_velocity": [0, 0, 0], "root_angular_velocity": [0, 0, 0],
        "bones": {
            "count": 1,
            "world_positions_xyz": [0, 1, 0],
            "world_rotations_xyzw": [0, 0, 0, 1],
            "local_rotations_xyzw": [0, 0, 0, 1],
        },
    }
    return {
        "event": "sample", "client_fixed_tick": tick,
        "local_fighter_index": 0,
        "unity_fixed_time": tick * 0.002,
        "transport_observation": {"fight_state_snapshot_sequence": tick // 50},
        "round": {"time_remaining": 10, "clean_hits": [0, 0],
                  "falls": [0, 0], "result": "InProgress"},
        "fighter_0": robot, "fighter_1": robot,
    }


def documents(with_end=True):
    start = {
        "event": "capture_start", "schema": canned_move_windows.RAW_SCHEMA,
        "machine": "D21", "fixed_delta_time": 0.002,
        "game_assembly_sha256": canned_move_windows.GAME_ASSEMBLY_SHA256,
        "global_metadata_sha256": canned_move_windows.GLOBAL_METADATA_SHA256,
        "plugin_sha256": "a" * 64, "server_tick_available": False,
        "scope": {
            "allowed": True, "local_fighter_index": 0, "opponent_slot": 1,
            "opponent_is_ai": True, "opponent_slot_is_ai": True,
            "human_in_opponent_slot": False, "sparring_bot_number": 1,
        },
        "fighter_0_bones": ["root"], "fighter_1_bones": ["root"],
    }
    records = [start]
    records.extend(sample(tick) for tick in range(5))
    records.append({
        "event": "client_transport_method_invoked", "method": "SendMoveEvent",
        "client_fixed_tick_at_observation": 4, "unity_frame": 50,
        "unity_unscaled_time": 2.5, "provenance": "prefix",
        "client_transport_invocation_sequence": 7,
        "method_invocation_sequence": 1,
        "input": {"network_index": 0, "pending_move_index": 3,
                  "velocity_command": [1, 0, 0],
                  "punching": False, "recovering": False},
    })
    records.extend(sample(tick) for tick in range(5, 11))
    if with_end:
        records.append({
            "event": "capture_end", "sample_count": 11,
            "capture_error_count": 0,
            "client_transport_method_counts": {"SendMoveEvent": 1},
            "reason": "scope_exit:round_not_active",
        })
    return records


class CannedMoveWindowTests(unittest.TestCase):
    def write_records(self, path, records):
        path.write_text("".join(json.dumps(row) + "\n" for row in records),
                        encoding="utf-8")

    def test_extracts_request_aligned_window_without_acceptance_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            out = Path(directory) / "windows.json"
            self.write_records(raw, documents())
            result = canned_move_windows.extract(
                raw, out, pre_ms=4, post_ms=14, stride_ms=2
            )
            window = result["windows"][0]
            self.assertEqual(window["request"]["move_index"], 3)
            self.assertEqual(window["request"]["move_name"], "youbiantui")
            self.assertEqual(window["accepted"], {"state": "unknown"})
            self.assertEqual(window["executed"], {"state": "unknown"})
            self.assertEqual(window["request"]["network_index"], 0)
            self.assertEqual(window["request"]["client_transport_invocation_sequence"], 7)
            self.assertEqual([row["tick"] for row in window["samples"]],
                             [2, 3, 4, 5, 6, 7, 8, 9, 10])
            self.assertTrue(window["coverage"]["pre_complete"])
            self.assertFalse(window["coverage"]["post_complete"])
            self.assertTrue(window["coverage"]["right_censored"])
            self.assertTrue(out.is_file())

    def test_rejects_fighter_mismatched_to_command_issuer(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            out = Path(directory) / "windows.json"
            self.write_records(raw, documents())
            with self.assertRaisesRegex(
                canned_move_windows.WindowError, "command issuer"
            ):
                canned_move_windows.extract(raw, out, fighter=1)

    def test_rejects_request_from_nonlocal_fighter(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            out = Path(directory) / "windows.json"
            records = documents()
            request = next(row for row in records if row.get("method") == "SendMoveEvent")
            request["input"]["network_index"] = 1
            self.write_records(raw, records)
            with self.assertRaisesRegex(
                canned_move_windows.WindowError, "network index"
            ):
                canned_move_windows.extract(raw, out)

    def test_rejects_malformed_bone_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            out = Path(directory) / "windows.json"
            records = documents()
            records[1]["fighter_0"]["bones"]["local_rotations_xyzw"] = [0, 0, 0]
            self.write_records(raw, records)
            with self.assertRaisesRegex(
                canned_move_windows.WindowError, "local_rotations_xyzw length"
            ):
                canned_move_windows.extract(raw, out)

    def test_rejects_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            out = Path(directory) / "windows.json"
            self.write_records(raw, documents(with_end=False))
            with self.assertRaisesRegex(canned_move_windows.WindowError, "incomplete"):
                canned_move_windows.extract(raw, out)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            out = Path(directory) / "windows.json"
            self.write_records(raw, documents())
            out.write_text("owned", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                canned_move_windows.extract(raw, out)


if __name__ == "__main__":
    unittest.main()
