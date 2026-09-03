import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import client_fixed_import
import controlled_schedule
import paired_motion_compare
import raw_bone_validate
from test_client_fixed_import import (
    control_records,
    inventory_record,
    schedule_manifest,
    v5_records,
)
from test_raw_bone_validate_v6 import camera_record
from trace import Trace


def _utc_for_tick(tick):
    elapsed_microseconds = tick * 2000
    seconds, microseconds = divmod(elapsed_microseconds, 1_000_000)
    return f"2026-09-03T00:00:{seconds:02d}.{microseconds:06d}+00:00"


def _root_pose_sample(tick, camera):
    return {
        "event": "root_pose_sample",
        "root_pose_sample_index": tick,
        "client_fixed_tick": tick,
        "utc": _utc_for_tick(tick),
        "stopwatch_timestamp_ticks": 1_000_000 + tick * 20_000,
        "unity_frame": tick // 5,
        "unity_time": tick * 0.002,
        "unity_fixed_time": tick * 0.002,
        "unity_unscaled_time": tick * 0.002,
        "scene": "Arena",
        "fight_epoch": 1,
        "round_number": 1,
        "local_fighter_index": 0,
        "opponent_slot": 1,
        "camera": camera,
        "fighter_0_root": {
            "world_position_xyz": [tick * 0.001, 1.0, 0.0],
            "world_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "screen_position_xyz": [600.0 + tick * 0.01, 360.0, 5.0],
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


def v6_schedule_records():
    records = v5_records()
    start = records[0]
    end = records[-1]
    camera = camera_record()
    start.update({
        "schema": client_fixed_import.SCHEMA_V6,
        "plugin_version": raw_bone_validate.EXPECTED_PLUGIN_VERSION_V6,
        "plugin_sha256": raw_bone_validate.EXPECTED_PLUGIN_SHA256_V6,
        "utc": "2026-09-03T00:00:00.000000+00:00",
        "stopwatch_timestamp_ticks": 1_000_000,
        "stopwatch_frequency_hz": 10_000_000,
        "stopwatch_is_high_resolution": True,
        "stopwatch_clock_semantics": (
            "System.Diagnostics.Stopwatch.GetTimestamp; QueryPerformanceCounter-backed "
            "on Windows when Stopwatch.IsHighResolution is true"
        ),
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
        "initial_camera": camera,
        "pairing": {
            "required_pairing": "t800_vs_t800",
            "required_robot_id": "t800",
            "required_t800_bone_count": 26,
            "required_t800_bone_signature_sha256": (
                raw_bone_validate.T800_BONE_SIGNATURE_SHA256
            ),
            "semantic_identity_source": (
                "FightCoordinator.fighterIdentities[slot].RobotID"
            ),
            "bone_signature_source": (
                "FightCoordinator.Fighters[slot].boneTransforms[index].name"
            ),
            "exact_t800_vs_t800": True,
            "reason": "exact_t800_vs_t800_pairing_proven",
        },
    })
    for slot in (0, 1):
        start["pairing"][f"fighter_{slot}"] = {
            "semantic_robot_id": "t800",
            "semantic_t800": True,
            "bone_count": 26,
            "ordered_bone_signature_sha256": (
                raw_bone_validate.T800_BONE_SIGNATURE_SHA256
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

    for record in records:
        if record.get("event") == "sample":
            tick = record["client_fixed_tick"]
            record["utc"] = _utc_for_tick(tick)
            record["stopwatch_timestamp_ticks"] = 2_000_000 + tick * 20_000
        if record.get("event") in {
            "outbound_request_projection", "client_transport_method_invoked"
        }:
            sequence = record["request_sequence"]
            record["utc"] = "2026-09-03T00:00:01.000000+00:00"
            record["stopwatch_timestamp_ticks"] = 3_000_000 + sequence * 20_000

    end.update({
        "utc": "2026-09-03T00:00:55.000000+00:00",
        "stopwatch_timestamp_ticks": 600_000_000,
        "root_pose_sample_count": end["client_fixed_tick_at_end"],
    })
    roots = [
        _root_pose_sample(tick, camera)
        for tick in range(end["client_fixed_tick_at_end"])
    ]
    return [start, *roots, *records[1:]]


class ClientFixedImportV6RoutingTests(unittest.TestCase):
    def test_v6_uses_strict_protocol_import_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            output = root / "trace.jsonl"
            raw.write_text(
                json.dumps({
                    "event": "capture_start",
                    "schema": client_fixed_import.SCHEMA_V6,
                }) + "\n",
                encoding="utf-8",
            )
            expected = {"raw_recorder_schema": client_fixed_import.SCHEMA_V6}
            with mock.patch.object(
                client_fixed_import, "_convert_v5", return_value=expected
            ) as protocol_import:
                observed = client_fixed_import.convert(
                    raw,
                    root / "inventory.json",
                    output,
                    control_log_path=root / "control.jsonl",
                    schedule_manifest_path=root / "schedule.json",
                    motion_edge="walk_forward.press.1",
                )
            self.assertEqual(observed, expected)
            protocol_import.assert_called_once()
            self.assertEqual(
                protocol_import.call_args.kwargs["motion_edge"],
                "walk_forward.press.1",
            )


class ClientFixedImportV6EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.raw = root / "capture.jsonl"
        cls.inventory = root / "inventory.json"
        cls.control = root / "control.jsonl"
        cls.manifest = root / "schedule.json"
        cls.output = root / "v6.trace"
        manifest = schedule_manifest()
        cls.raw.write_text(
            "".join(
                json.dumps(record, separators=(",", ":")) + "\n"
                for record in v6_schedule_records()
            ),
            encoding="utf-8",
        )
        cls.inventory.write_text(json.dumps(inventory_record()), encoding="utf-8")
        cls.control.write_text(
            "".join(
                json.dumps(record, separators=(",", ":")) + "\n"
                for record in control_records(manifest)
            ),
            encoding="utf-8",
        )
        cls.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        cls.validation = raw_bone_validate.validate(cls.raw)
        cls.result = client_fixed_import.convert(
            cls.raw,
            cls.inventory,
            cls.output,
            control_log_path=cls.control,
            schedule_manifest_path=cls.manifest,
            motion_edge="walk_forward.press.1",
        )
        cls.trace = Trace.load(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_validator_and_importer_preserve_complete_500hz_root_stream(self):
        self.assertTrue(
            self.validation["claims"]["root_pose_500hz_validated"]
        )
        self.assertEqual(
            self.trace.header["tick_domain"],
            "controlled_schedule_client_fixed_substep_500hz",
        )
        self.assertEqual(self.trace.header["tick_rate_hz"], 500)
        self.assertEqual(self.trace.header["fixed_delta_time"], 0.002)
        self.assertEqual(self.trace.header["client_fixed_delta_time"], 0.002)
        self.assertEqual(
            self.trace.ticks,
            list(range(controlled_schedule.DURATION_TICKS * 10)),
        )
        self.assertEqual(self.result["ticks"], 26010)
        self.assertAlmostEqual(self.trace.channels["root.0.pos.x"][0], 0.1)
        self.assertAlmostEqual(
            self.trace.channels["root.0.pos.x"][-1], 26.109
        )
        self.assertEqual(self.trace.channels["root.0.quat.w"][0], 1.0)
        self.assertEqual(self.trace.channels["screen.0.root.x"][0], 601.0)
        self.assertEqual(
            self.trace.channels["camera.world_to_camera.m00"][0], 1.0
        )
        self.assertEqual(
            self.trace.provenance["screen.0.root.x"]["raw_field"],
            "root_pose_sample.fighter_0_root.screen_position_xyz[0]",
        )
        self.assertIn("screen_frame", self.trace.header)
        self.assertEqual(self.trace.header["screen_frame"]["width_px"], 1280)
        self.assertEqual(self.trace.header["screen_frame"]["height_px"], 720)

    def test_each_schedule_motion_has_one_selectable_comparator_edge(self):
        catalog = self.trace.header["command_edge_catalog"]
        schedule_catalog = self.trace.header["schedule_step_edge_catalog"]
        self.assertEqual(len(schedule_catalog), 22)
        self.assertEqual(
            {edge["schedule_tick"] for edge in schedule_catalog},
            {tick for tick, _velocity, _move in controlled_schedule._expected_steps()},
        )
        for edge in schedule_catalog:
            matching = [
                event for event in self.trace.events
                if event["kind"] == edge["event_kind"]
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(
                matching[0]["command_identity"], edge["command_identity"]
            )
        self.assertEqual(len(catalog), 24)
        self.assertEqual(len({edge["selector"] for edge in catalog}), 24)
        self.assertEqual(len({edge["event_kind"] for edge in catalog}), 24)
        expected_identities = {
            *(f"{name}:{action}:v1"
              for name in client_fixed_import.V6_VELOCITY_IDENTITIES.values()
              for action in ("press", "release")),
            *(f"move_index_{index}:press:v1" for index in (2, 3, 4, 5, 9, 10)),
        }
        self.assertEqual(
            {edge["command_identity"] for edge in catalog},
            expected_identities,
        )
        expected_step_ticks = {
            tick for tick, _velocity, _move in controlled_schedule._expected_steps()
            if tick != 0
        }
        self.assertEqual(
            {edge["schedule_tick"] for edge in catalog}, expected_step_ticks
        )
        for edge in catalog:
            with self.subTest(selector=edge["selector"]):
                self.assertEqual(
                    client_fixed_import._v6_select_command_edge(
                        catalog, edge["selector"]
                    ),
                    edge,
                )
                matching = [
                    event for event in self.trace.events
                    if event["kind"] == edge["event_kind"]
                ]
                self.assertEqual(len(matching), 1)
                self.assertEqual(
                    matching[0]["command_identity"], edge["command_identity"]
                )

    def test_selected_edge_loads_through_paired_comparator_adapter(self):
        loaded = paired_motion_compare._load_rektrace(
            self.output, "0", "command_edge"
        )
        self.assertEqual(loaded.command_identity, "walk_forward:press:v1")
        self.assertEqual(
            loaded.execution_state,
            client_fixed_import.V6_COMMAND_EXECUTION_STATE,
        )
        self.assertAlmostEqual(loaded.command_edge_time_s, 1.0)
        self.assertEqual(len(loaded.samples), 26010)
        self.assertEqual(loaded.samples[0].position, (0.1, 1.0, 0.0))
        self.assertEqual(loaded.samples[0].quaternion, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(loaded.samples[0].screen, (601.0, 360.0))

    def test_no_motion_is_silently_selected(self):
        self.assertIsNone(
            client_fixed_import._v6_select_command_edge(
                self.trace.header["command_edge_catalog"], None
            )
        )
        with self.assertRaisesRegex(ValueError, "available selectors"):
            client_fixed_import._v6_select_command_edge(
                self.trace.header["command_edge_catalog"], "walk_forward.press"
            )


if __name__ == "__main__":
    unittest.main()
