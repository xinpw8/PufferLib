import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import raw_bone_validate
from test_raw_bone_validate import fixture


def camera_record():
    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    return {
        "selection": "UnityEngine.Camera.main",
        "instance_id": 17,
        "name": "Main Camera",
        "enabled": True,
        "active_in_hierarchy": True,
        "camera_type": "Game",
        "world_position_xyz": [0.0, 2.0, -5.0],
        "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "world_to_camera_matrix_row_major": identity,
        "projection_matrix_row_major": identity,
        "gpu_projection_matrix_row_major": identity,
        "normalized_viewport_rect_xywh": [0.0, 0.0, 1.0, 1.0],
        "pixel_rect_xywh": [0.0, 0.0, 1280.0, 720.0],
        "pixel_width": 1280,
        "pixel_height": 720,
        "scaled_pixel_width": 1280,
        "scaled_pixel_height": 720,
        "target_display": 0,
        "orthographic": False,
        "orthographic_size": 5.0,
        "field_of_view_degrees": 60.0,
        "aspect": 16.0 / 9.0,
        "near_clip_plane": 0.1,
        "far_clip_plane": 1000.0,
        "render_into_texture": False,
        "target_texture": None,
        "allow_hdr": True,
        "allow_msaa": True,
        "screen_width": 1280,
        "screen_height": 720,
        "screen_full_screen_mode": "Windowed",
        "screen_dpi": 96.0,
        "render_scale_xy": [1.0, 1.0],
    }


def root_pose_sample(tick):
    return {
        "event": "root_pose_sample",
        "root_pose_sample_index": tick,
        "client_fixed_tick": tick,
        "utc": f"2026-09-03T00:00:00.{tick:07d}+00:00",
        "stopwatch_timestamp_ticks": 200 + tick,
        "unity_frame": tick // 5,
        "unity_time": tick * 0.002,
        "unity_fixed_time": tick * 0.002,
        "unity_unscaled_time": tick * 0.002,
        "scene": "Arena",
        "fight_epoch": 1,
        "round_number": 1,
        "local_fighter_index": 0,
        "opponent_slot": 1,
        "camera": camera_record(),
        "fighter_0_root": {
            "world_position_xyz": [tick * 0.001, 1.0, 0.0],
            "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "screen_position_xyz": [600.0 + tick, 360.0, 5.0],
            "screen_in_front_of_camera": True,
            "screen_inside_camera_pixel_rect": True,
        },
        "fighter_1_root": {
            "world_position_xyz": [1.0, 1.0, 0.0],
            "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "screen_position_xyz": [700.0, 360.0, 5.0],
            "screen_in_front_of_camera": True,
            "screen_inside_camera_pixel_rect": True,
        },
    }


def v6_fixture():
    records = fixture()
    start = records[0]
    end = records[-1]
    start.update({
        "schema": raw_bone_validate.SCHEMA_V6,
        "plugin_version": raw_bone_validate.EXPECTED_PLUGIN_VERSION_V6,
        "plugin_sha256": raw_bone_validate.EXPECTED_PLUGIN_SHA256_V6,
        "utc": "2026-09-03T00:00:00+00:00",
        "stopwatch_timestamp_ticks": 100,
        "stopwatch_frequency_hz": 10_000_000,
        "stopwatch_is_high_resolution": True,
        "stopwatch_clock_semantics": (
            "System.Diagnostics.Stopwatch.GetTimestamp; QueryPerformanceCounter-backed "
            "on Windows when Stopwatch.IsHighResolution is true"
        ),
        "fixed_delta_time": 0.002,
        "scene": "Arena",
        "root_pose_sample_stride_ticks": 1,
        "root_pose_sample_rate_hz": 500,
        "root_pose_tick_level_claim": True,
        "root_pose_fields": (
            "world root position/rotation plus Camera.WorldToScreenPoint only; "
            "no inferred joints, velocities, contacts, or server state"
        ),
        "root_screen_coordinate_semantics": (
            "Unity Camera.WorldToScreenPoint pixels; origin bottom-left; "
            "z is world-unit distance from camera plane"
        ),
        "camera_selection_semantics": (
            "UnityEngine.Camera.main; capture is denied when absent, inactive, "
            "or without a positive pixel extent"
        ),
        "camera_matrix_semantics": (
            "16 float values in row-major m[row,column] order; world_to_camera "
            "is Unity view matrix; gpu_projection uses GL.GetGPUProjectionMatrix "
            "with render_into_texture"
        ),
        "bone_wire_protocol": copy.deepcopy(
            raw_bone_validate.EXPECTED_BONE_PROTOCOL_V6
        ),
        "initial_camera": camera_record(),
        "pairing": {
            "required_pairing": "t800_vs_t800",
            "required_robot_id": "t800",
            "required_t800_bone_count": 26,
            "required_t800_bone_signature_sha256": (
                raw_bone_validate.EXPECTED_BONE_PROTOCOL_V6[
                    "t800_ordered_bone_signature_sha256"
                ]
            ),
            "semantic_identity_source": (
                "FightCoordinator.fighterIdentities[slot].RobotID"
            ),
            "bone_signature_source": (
                "FightCoordinator.Fighters[slot].boneTransforms[index].name"
            ),
            "exact_t800_vs_t800": True,
            "reason": "exact_t800_vs_t800_pairing_proven",
            "fighter_0": {},
            "fighter_1": {},
        },
    })
    for slot in (0, 1):
        start["pairing"][f"fighter_{slot}"] = {
            "semantic_robot_id": "t800",
            "semantic_t800": True,
            "bone_count": 26,
            "ordered_bone_signature_sha256": (
                raw_bone_validate.EXPECTED_BONE_PROTOCOL_V6[
                    "t800_ordered_bone_signature_sha256"
                ]
            ),
            "exact_t800_bone_signature": True,
        }
    start["scope"].update({
        "context_is_solo": True,
        "context_is_ranked": False,
        "context_auto_find_match": False,
        "arena_id_present": True,
        "multiplayer_session_privacy_known": True,
        "multiplayer_session_is_private": True,
        "coordinator_is_ranked_arena": False,
        "client_ai_difficulty": 0,
        "exact_t800_vs_t800": True,
    })

    sample_stopwatch = 500
    request_stopwatch = 600
    for record in records:
        if record.get("event") == "sample":
            record["utc"] = "2026-09-03T00:00:00.100000+00:00"
            record["stopwatch_timestamp_ticks"] = sample_stopwatch
            sample_stopwatch += 10
        if record.get("event") in {
            "outbound_request_projection", "client_transport_method_invoked"
        }:
            record["utc"] = "2026-09-03T00:00:00.200000+00:00"
            record["stopwatch_timestamp_ticks"] = request_stopwatch
            request_stopwatch += 1

    end.update({
        "utc": "2026-09-03T00:00:01+00:00",
        "stopwatch_timestamp_ticks": 1000,
        "root_pose_sample_count": end["client_fixed_tick_at_end"],
    })
    roots = [root_pose_sample(tick) for tick in range(end["client_fixed_tick_at_end"])]
    return [start, *roots, *records[1:]]


class RawBoneValidateV6Tests(unittest.TestCase):
    def validate(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return raw_bone_validate.validate(path)

    def assert_rejected(self, records, pattern):
        with self.assertRaisesRegex(raw_bone_validate.EvidenceError, pattern):
            self.validate(records)

    def test_accepts_exact_private_t800_root_camera_stream(self):
        self.assertEqual(
            hashlib.sha256(
                "\n".join(raw_bone_validate.T800_BONE_NAMES).encode("utf-8")
            ).hexdigest(),
            raw_bone_validate.T800_BONE_SIGNATURE_SHA256,
        )
        report = self.validate(v6_fixture())
        self.assertEqual(report["recorder_schema"], raw_bone_validate.SCHEMA_V6)
        self.assertEqual(report["root_pose_stream"]["samples"], 11)
        self.assertTrue(report["claims"]["exact_private_session_validated"])
        self.assertTrue(report["claims"]["exact_t800_vs_t800_validated"])
        self.assertTrue(report["claims"]["root_pose_500hz_validated"])
        self.assertTrue(report["claims"]["root_world_to_screen_validated"])

    def test_rejects_public_session(self):
        records = v6_fixture()
        records[0]["scope"]["multiplayer_session_is_private"] = False
        self.assert_rejected(records, "outside private Bot 1 scope")

    def test_rejects_non_t800_semantic_identity(self):
        records = v6_fixture()
        records[0]["pairing"]["fighter_1"]["semantic_robot_id"] = "g1"
        self.assert_rejected(records, "fighter 1 field semantic_robot_id")

    def test_rejects_missing_fixed_tick_root_sample(self):
        records = v6_fixture()
        records.pop(5)
        self.assert_rejected(records, "root pose count disagrees")

    def test_rejects_duplicate_root_stopwatch_tick(self):
        records = v6_fixture()
        roots = [record for record in records if record.get("event") == "root_pose_sample"]
        roots[1]["stopwatch_timestamp_ticks"] = roots[0]["stopwatch_timestamp_ticks"]
        self.assert_rejected(records, "not strictly increasing")

    def test_rejects_incomplete_projection_matrix(self):
        records = v6_fixture()
        roots = [record for record in records if record.get("event") == "root_pose_sample"]
        roots[0]["camera"]["projection_matrix_row_major"].pop()
        self.assert_rejected(records, "must contain exactly 16")

    def test_rejects_command_without_stopwatch_edge(self):
        records = v6_fixture()
        command = next(
            record for record in records
            if record.get("event") == "outbound_request_projection"
        )
        del command["stopwatch_timestamp_ticks"]
        self.assert_rejected(records, "Stopwatch ticks is not an integer")


if __name__ == "__main__":
    unittest.main()
