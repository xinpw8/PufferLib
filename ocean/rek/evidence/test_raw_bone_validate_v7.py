import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import raw_bone_validate
from test_raw_bone_validate_v6 import v6_fixture


def _semantic_consistency(semantic_robot_id, runtime_model):
    if semantic_robot_id is None or not semantic_robot_id.strip():
        return f"semantic_robot_id_unavailable_runtime_{runtime_model}_exact"
    if semantic_robot_id == runtime_model:
        return f"semantic_and_runtime_{runtime_model}_exact"
    return f"semantic_robot_id_mismatch_runtime_{runtime_model}_exact"


def upgrade_to_v7(
    records,
    runtime_model="g1",
    semantic_ids=(None, "t800"),
    copy_records=True,
):
    if copy_records:
        records = copy.deepcopy(records)
    start = records[0]
    start.update({
        "schema": raw_bone_validate.SCHEMA_V7,
        "plugin_version": raw_bone_validate.EXPECTED_PLUGIN_VERSION_V7,
        "plugin_sha256": raw_bone_validate.EXPECTED_PLUGIN_SHA256_V7,
        "instrumentation_hooks": list(raw_bone_validate.EXPECTED_HOOKS_V7),
        "bone_wire_protocol": copy.deepcopy(
            raw_bone_validate.EXPECTED_BONE_PROTOCOL_V7
        ),
    })
    start["server"].pop("endpoint", None)
    start["server"].update({
        "endpoint_present": True,
        "endpoint_recorded": False,
        "endpoint_reason": "omitted as connection-sensitive data",
    })

    local_slot = start["scope"]["local_fighter_index"]
    opponent_slot = start["scope"]["opponent_slot"]
    semantic_t800 = [value == "t800" for value in semantic_ids]
    semantic_g1 = [value == "g1" for value in semantic_ids]
    consistency = [
        _semantic_consistency(value, runtime_model) for value in semantic_ids
    ]
    mismatches = [
        value.startswith("semantic_robot_id_mismatch_")
        for value in consistency
    ]
    exact_t800 = runtime_model == "t800"
    exact_g1 = runtime_model == "g1"
    signature = (
        raw_bone_validate.T800_BONE_SIGNATURE_SHA256
        if exact_t800
        else raw_bone_validate.G1_BONE_SIGNATURE_SHA256
    )
    bone_count = (
        raw_bone_validate.T800_BONE_COUNT
        if exact_t800
        else raw_bone_validate.G1_BONE_COUNT
    )
    reason = (
        "exact_g1_vs_g1_runtime_pairing_proven_semantic_ids_recorded_not_trusted"
        if exact_g1
        else (
            "exact_t800_vs_t800_runtime_pairing_proven_semantic_mismatch_recorded"
            if any(mismatches)
            else "exact_t800_vs_t800_pairing_proven"
        )
    )
    pairing = {
        "required_pairing": "exact_homogeneous_supported_runtime_pair",
        "required_robot_id": None,
        "supported_runtime_models": ["t800", "g1"],
        "semantic_robot_id_required_for_acceptance": False,
        "required_t800_bone_count": raw_bone_validate.T800_BONE_COUNT,
        "required_t800_bone_signature_sha256": (
            raw_bone_validate.T800_BONE_SIGNATURE_SHA256
        ),
        "required_g1_bone_count": raw_bone_validate.G1_BONE_COUNT,
        "required_g1_bone_signature_sha256": (
            raw_bone_validate.G1_BONE_SIGNATURE_SHA256
        ),
        "semantic_identity_source": (
            "FightCoordinator.fighterIdentities[slot].RobotID"
        ),
        "bone_signature_source": (
            "FightCoordinator.Fighters[slot].boneTransforms[index].name"
        ),
        "exact_supported_runtime_pairing": True,
        "runtime_model": runtime_model,
        "exact_t800_vs_t800": exact_t800,
        "exact_g1_vs_g1": exact_g1,
        "reason": reason,
        "local_slot": local_slot,
        "local_semantic_t800": semantic_t800[local_slot],
        "local_semantic_g1": semantic_g1[local_slot],
        "opponent_semantic_t800": semantic_t800[opponent_slot],
        "opponent_semantic_g1": semantic_g1[opponent_slot],
        "local_semantic_runtime_mismatch": mismatches[local_slot],
        "local_semantic_runtime_consistency": consistency[local_slot],
        "opponent_semantic_runtime_mismatch": mismatches[opponent_slot],
        "opponent_semantic_runtime_consistency": consistency[opponent_slot],
    }
    for slot in (0, 1):
        pairing[f"fighter_{slot}"] = {
            "semantic_robot_id": semantic_ids[slot],
            "semantic_t800": semantic_t800[slot],
            "semantic_g1": semantic_g1[slot],
            "bone_count": bone_count,
            "ordered_bone_signature_sha256": signature,
            "exact_t800_bone_signature": exact_t800,
            "exact_g1_bone_signature": exact_g1,
        }
    start["pairing"] = pairing
    for legacy_key in (
        "multiplayer_session_privacy_known",
        "multiplayer_session_is_private",
        "multiplayer_session_privacy_reason",
        "multiplayer_session_absent_current_session_fallback_used",
    ):
        start["scope"].pop(legacy_key, None)
    start["scope"].update({
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
        "exact_supported_runtime_pairing": True,
        "runtime_model": runtime_model,
        "exact_t800_vs_t800": exact_t800,
        "exact_g1_vs_g1": exact_g1,
        "local_semantic_t800": semantic_t800[local_slot],
        "local_semantic_g1": semantic_g1[local_slot],
        "local_semantic_runtime_mismatch": mismatches[local_slot],
        "local_semantic_runtime_consistency": consistency[local_slot],
        "opponent_semantic_runtime_mismatch": mismatches[opponent_slot],
        "opponent_semantic_runtime_consistency": consistency[opponent_slot],
    })
    return records


def v7_fixture(runtime_model="g1", semantic_ids=(None, "t800")):
    layout = "t800_26" if runtime_model == "t800" else "g1_30"
    return upgrade_to_v7(
        v6_fixture((layout, layout)),
        runtime_model=runtime_model,
        semantic_ids=semantic_ids,
    )


class RawBoneValidateV7Tests(unittest.TestCase):
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

    def test_accepts_exact_private_g1_with_non_authoritative_semantic_ids(self):
        self.assertEqual(
            hashlib.sha256(
                "\n".join(raw_bone_validate.G1_BONE_NAMES).encode("utf-8")
            ).hexdigest(),
            raw_bone_validate.G1_BONE_SIGNATURE_SHA256,
        )
        report = self.validate(v7_fixture())
        self.assertEqual(report["recorder_schema"], raw_bone_validate.SCHEMA_V7)
        self.assertEqual(report["root_pose_stream"]["samples"], 11)
        self.assertEqual(
            report["claims"]["validated_runtime_model"], "g1"
        )
        self.assertTrue(report["claims"]["exact_g1_vs_g1_validated"])
        self.assertFalse(report["claims"]["exact_t800_vs_t800_validated"])
        self.assertEqual(
            report["fighters"]["0"]["bone_layout"]["layout_id"],
            "g1_30",
        )

    def test_accepts_exact_private_t800(self):
        report = self.validate(
            v7_fixture("t800", semantic_ids=("t800", "t800"))
        )
        self.assertEqual(
            report["claims"]["validated_runtime_model"], "t800"
        )
        self.assertTrue(report["claims"]["exact_t800_vs_t800_validated"])
        self.assertFalse(report["claims"]["exact_g1_vs_g1_validated"])

    def test_accepts_t800_runtime_with_recorded_semantic_mismatch(self):
        records = v7_fixture("t800", semantic_ids=("g1", "t800"))
        self.assertIn(
            "semantic_mismatch_recorded", records[0]["pairing"]["reason"]
        )
        self.assertEqual(
            self.validate(records)["claims"]["validated_runtime_model"],
            "t800",
        )

    def test_rejects_mixed_runtime_layouts(self):
        records = v7_fixture()
        records[0]["fighter_1_bones"] = list(
            raw_bone_validate.T800_BONE_NAMES
        )
        self.assert_rejected(records, "not an exact homogeneous runtime model")

    def test_rejects_reordered_g1_runtime_layout(self):
        records = v7_fixture()
        bones = records[0]["fighter_0_bones"]
        bones[0], bones[1] = bones[1], bones[0]
        self.assert_rejected(records, "does not match a pinned bone layout")

    def test_rejects_pairing_signature_disagreement(self):
        records = v7_fixture()
        records[0]["pairing"]["fighter_0"][
            "ordered_bone_signature_sha256"
        ] = "0" * 64
        self.assert_rejected(records, "ordered_bone_signature_sha256")

    def test_rejects_missing_required_robot_id_null_field(self):
        records = v7_fixture()
        del records[0]["pairing"]["required_robot_id"]
        self.assert_rejected(records, "required_robot_id is absent")

    def test_rejects_unproven_solo_route_fields(self):
        mutations = (
            ("solo_route_hooks_verified", False),
            ("solo_route_proven", False),
            ("solo_route_flow", "koth"),
            ("solo_route_connect_to_arena_observed", False),
            ("solo_route_enter_championship_observed", False),
            ("solo_route_enter_championship_koth", True),
            ("solo_route_enter_championship_solo", False),
            ("solo_route_arena_identity_consistent", False),
            ("solo_route_runtime_session_identity_consistent", False),
            ("solo_route_reason", "solo_route_not_observed"),
            ("server_private_proven", True),
            ("server_private_status", "proven"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                records = v7_fixture()
                records[0]["scope"][field] = value
                self.assert_rejected(records, "outside private Bot 1 scope")
        records = v7_fixture()
        del records[0]["scope"]["solo_route_reason"]
        self.assert_rejected(records, "outside private Bot 1 scope")

    def test_rejects_raw_server_endpoint(self):
        records = v7_fixture()
        records[0]["server"]["endpoint"] = "test.invalid:7777"
        self.assert_rejected(records, "persisted a raw server endpoint")

    def test_rejects_wrong_v7_binary_pin(self):
        records = v7_fixture()
        records[0]["plugin_sha256"] = "0" * 64
        self.assert_rejected(records, "recorder plugin hash mismatch")


if __name__ == "__main__":
    unittest.main()
