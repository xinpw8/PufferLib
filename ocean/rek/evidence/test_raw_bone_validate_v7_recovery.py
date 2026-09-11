import json
import math
import tempfile
import unittest
from pathlib import Path

import raw_bone_validate
from test_raw_bone_validate_v7 import v7_fixture


def _recovery(runtime_model, slot, can_get_up):
    sonic = None
    if runtime_model == "g1":
        sonic = {
            "present": True,
            "assigned_to_robot_policy_runner": True,
            "can_get_up": can_get_up,
            "init_complete": True,
            "paused": False,
            "motion_composer_present": True,
            "get_up_prone_clip_present": can_get_up,
            "get_up_supine_clip_present": can_get_up,
        }
    return {
        "complete": True,
        "reason": "direct_runtime_can_get_up_authority_complete",
        "source": raw_bone_validate.RUNTIME_CAN_GET_UP_AUTHORITY_SOURCE,
        "runtime_model": runtime_model,
        "fighter_slot": slot,
        "can_get_up": can_get_up,
        "fight_coordinator_can_fighter_get_up": can_get_up,
        "robot_policy_runner_present": True,
        "robot_policy_runner_component_present": True,
        "robot_policy_runner_component_managed_type": (
            "REKApp.SonicPolicyRunner"
            if runtime_model == "g1"
            else "REKApp.EngineAIPolicyRunner"
        ),
        "robot_policy_runner_component_name": "fixture policy runner",
        "robot_policy_runner_can_get_up": can_get_up,
        "robot_policy_runner_is_initialized": True,
        "robot_policy_runner_is_paused": False,
        "robot_policy_runner_is_recovering": False,
        "robot_policy_runner_is_done": False,
        "robot_get_up_pending": False,
        "robot_recovery_armed": False,
        "robot_suggested_get_up_orientation": "Prone",
        "robot_suggested_get_up_orientation_value": 2,
        "g1_sonic_policy_runner": sonic,
        "server_acceptance_available": False,
        "server_acceptance": None,
        "server_authority_claim": False,
    }


def _fighter(runtime_model, slot, can_get_up):
    names = (
        list(raw_bone_validate.G1_BONE_NAMES)
        if runtime_model == "g1"
        else list(raw_bone_validate.T800_BONE_NAMES)
    )
    signature = (
        raw_bone_validate.G1_BONE_SIGNATURE_SHA256
        if runtime_model == "g1"
        else raw_bone_validate.T800_BONE_SIGNATURE_SHA256
    )
    count = len(names)
    return {
        "fighter_slot": slot,
        "network_index": slot,
        "visual_only": True,
        "player_controlled": False,
        "falling": False,
        "fallen": False,
        "dampened": False,
        "resetting": False,
        "motor_shutdown": False,
        "policy_suspended": False,
        "tilt_angle": 0.0,
        "pelvis_height_ratio": 1.0,
        "floor_contact_count": 2,
        "both_feet_off_floor": False,
        "root_position": [float(slot), 1.0, 0.0],
        "root_rotation": [0.0, 0.0, 0.0, 1.0],
        "root_linear_velocity": [0.0, 0.0, 0.0],
        "root_angular_velocity": [0.0, 0.0, 0.0],
        "bones": {
            "count": count,
            "ordered_names": names,
            "ordered_name_signature_sha256": signature,
            "world_positions_xyz": [0.0] * (count * 3),
            "world_rotations_xyzw": [0.0, 0.0, 0.0, 1.0] * count,
            "local_positions_xyz": [0.0] * (count * 3),
            "local_rotations_xyzw": [0.0, 0.0, 0.0, 1.0] * count,
        },
        "recovery_authority": _recovery(runtime_model, slot, can_get_up),
    }


def recovery_fixture(runtime_model="g1", can_get_up=(False, False)):
    records = v7_fixture(
        runtime_model,
        semantic_ids=(runtime_model, runtime_model),
    )
    start = records[0]
    start.update({
        "plugin_version": raw_bone_validate.EXPECTED_PLUGIN_VERSION_V7_RECOVERY,
        "plugin_sha256": raw_bone_validate.EXPECTED_PLUGIN_SHA256_V7_RECOVERY,
        "initial_state": {
            "atomic_both_fighters": True,
            "observation_boundary": (
                raw_bone_validate.INITIAL_STATE_OBSERVATION_BOUNDARY
            ),
            "publication_unit": raw_bone_validate.INITIAL_STATE_PUBLICATION_UNIT,
            "read_order": ["fighter_0", "fighter_1"],
            "simultaneous_hardware_sample_claim": False,
            "stopwatch_begin_timestamp_ticks": start["stopwatch_timestamp_ticks"],
            "stopwatch_end_timestamp_ticks": start["stopwatch_timestamp_ticks"] + 1,
            "utc_begin": start["utc"],
            "utc_end": start["utc"],
            "unity_frame": 0,
            "unity_fixed_time": 0.0,
            "client_fixed_tick": 0,
            "scene": "Arena",
            "fight_epoch": 1,
            "phase": "RoundActive",
            "phase_value": 1,
            "local_fighter_index": start["scope"]["local_fighter_index"],
            "opponent_slot": start["scope"]["opponent_slot"],
            "input": {},
            "round": {},
            "fight": {},
            "fighter_0": _fighter(runtime_model, 0, can_get_up[0]),
            "fighter_1": _fighter(runtime_model, 1, can_get_up[1]),
        },
    })
    return records


class RawBoneValidateV7RecoveryTests(unittest.TestCase):
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

    def test_accepts_atomic_g1_initial_state_with_direct_false_authority(self):
        report = self.validate(recovery_fixture())
        self.assertTrue(report["initial_state"]["validated"])
        self.assertFalse(report["initial_state"]["fighter_0_can_get_up"])
        self.assertFalse(report["initial_state"]["fighter_1_can_get_up"])
        self.assertFalse(report["initial_state"]["server_authority_claimed"])

    def test_accepts_direct_true_authority(self):
        report = self.validate(recovery_fixture(can_get_up=(True, True)))
        self.assertTrue(report["initial_state"]["fighter_0_can_get_up"])
        self.assertTrue(report["initial_state"]["fighter_1_can_get_up"])

    def test_accepts_legacy_v0_7_2_without_new_initial_state(self):
        report = self.validate(v7_fixture())
        self.assertIsNone(report["initial_state"])

    def test_rejects_missing_atomic_initial_state(self):
        records = recovery_fixture()
        del records[0]["initial_state"]
        self.assert_rejected(records, "no atomic initial-state record")

    def test_rejects_policy_and_coordinator_disagreement(self):
        records = recovery_fixture()
        records[0]["initial_state"]["fighter_0"]["recovery_authority"][
            "robot_policy_runner_can_get_up"
        ] = True
        self.assert_rejected(records, "direct CanGetUp values disagree")

    def test_rejects_missing_g1_sonic_runner(self):
        records = recovery_fixture()
        records[0]["initial_state"]["fighter_1"]["recovery_authority"][
            "g1_sonic_policy_runner"
        ] = None
        self.assert_rejected(records, "G1 SonicPolicyRunner record is absent")

    def test_rejects_unassigned_g1_sonic_runner(self):
        records = recovery_fixture()
        records[0]["initial_state"]["fighter_1"]["recovery_authority"][
            "g1_sonic_policy_runner"
        ]["assigned_to_robot_policy_runner"] = False
        self.assert_rejected(records, "not assigned to Robot.policyRunner")

    def test_rejects_nonfinite_initial_pose(self):
        records = recovery_fixture()
        records[0]["initial_state"]["fighter_0"]["bones"][
            "world_positions_xyz"
        ][0] = math.nan
        self.assert_rejected(records, r"bone world positions\[0\].*not finite")

    def test_accepts_t800_without_sonic_runner_claim(self):
        report = self.validate(recovery_fixture("t800", (True, True)))
        self.assertTrue(report["initial_state"]["fighter_0_can_get_up"])


if __name__ == "__main__":
    unittest.main()
