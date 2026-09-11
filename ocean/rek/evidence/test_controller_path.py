import copy
import struct
import tempfile
import unittest
from pathlib import Path

import controller_path


def valid_report_fixture():
    methods = []
    for item in controller_path.METHODS:
        methods.append({
            "id": item["id"],
            "metadata_token": item["token"],
            "rva": f"0x{item['rva']:X}",
            "size_bytes": item["size"],
            "native_body_sha256": item["native_sha256"],
            "isil_method_block_sha256": item["isil_sha256"],
            "evidence_role": "static_native_semantics",
        })
    moves = []
    for index in range(12):
        path_id = controller_path.EXPECTED_MOVE_PATH_IDS[index]
        moves.append({
            "index": index,
            "pointer": {"file_id": 0, "path_id": path_id},
            "referenced_object": (
                controller_path._expected_resolved_move_object(path_id)
                if path_id else None
            ),
        })
    sources = {
        "inventory": {"sha256": controller_path.SOURCE_HASHES["inventory"]},
        "il2cpp_recovery": {
            "sha256": controller_path.SOURCE_HASHES["recovery"],
            "dummy_method_bodies_consumed": False,
            "semantic_claims_allowed": False,
        },
        "asset_probe": {"sha256": controller_path.SOURCE_HASHES["probe"]},
        "game_assembly": {
            "sha256": controller_path.SOURCE_HASHES["game_assembly"]
        },
        "global_metadata": {
            "sha256": controller_path.SOURCE_HASHES["global_metadata"]
        },
        "isil": {
            filename: {"sha256": digest}
            for filename, digest in controller_path.SOURCE_HASHES.items()
            if filename.endswith(".txt")
        },
    }
    return {
        "schema": controller_path.SCHEMA,
        "build_fingerprint": controller_path.BUILD_FINGERPRINT,
        "scope": {
            "evidence_role": "static_native_semantics",
            "control_equivalent": False,
            "private_ai_runtime_active": {"state": "unknown"},
            "server_equivalent": {"state": "unknown"},
            "move_object_fields": {
                "state": "measured",
                "count": len(controller_path.MOVE_OBJECT_SOURCES),
                "evidence_role": "serialized_object_fields",
            },
        },
        "sources": sources,
        "native_methods": methods,
        "static_semantics": {
            "one_fact": {
                "evidence_role": "static_native_semantics",
                "activation": {"state": "unknown"},
                "citations": [controller_path.METHODS[0]["id"]],
            }
        },
        "serialized_t800": {
            "robot_config": {"move_map": moves},
            "derived_force_limit_vector": list(
                controller_path.EXPECTED_FORCE_LIMITS
            ),
        },
        "hard_unknowns": [
            {"id": unknown_id, "state": "unknown"}
            for unknown_id in sorted(controller_path.UNKNOWN_IDS)
        ],
    }


class ControllerPathTests(unittest.TestCase):
    def test_force_limit_precedence_and_unknown(self):
        self.assertEqual(
            controller_path.derive_force_limit((-3, 7), (-20, 20), 99),
            {"state": "measured", "source": "force_range", "value": 7.0},
        )
        self.assertEqual(
            controller_path.derive_force_limit((0, 0), (-20, 13), 99),
            {"state": "measured", "source": "control_range", "value": 20.0},
        )
        self.assertEqual(
            controller_path.derive_force_limit((0, 0), (0, 0), 9),
            {
                "state": "measured",
                "source": "joint_fingerprint_effort",
                "value": 9.0,
            },
        )
        unknown = controller_path.derive_force_limit((0, 0), (0, 0), None)
        self.assertEqual(unknown["state"], "unknown")
        self.assertNotIn("value", unknown)

    def test_isil_blocks_have_a_canonical_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "methods.txt"
            path.write_bytes(
                b"Type: Example\r\n\r\n"
                b"Method: System.Void A()\r\n\r\n"
                b"Disassembly:\r\n\tret\r\n\r\n"
                b"ISIL:\r\n\t0 Return\r\n\r\n"
                b"Method: System.Void B()\r\n\r\n"
                b"Disassembly:\r\n\tret\r\n\r\n"
                b"ISIL:\r\n\t0 Return\r\n"
            )
            blocks = controller_path._normalise_method_blocks(path)
        self.assertEqual(set(blocks), {"System.Void A()", "System.Void B()"})
        self.assertNotIn("\r", blocks["System.Void A()"])
        self.assertTrue(blocks["System.Void B()"].endswith("\n"))

    def test_duplicate_isil_signatures_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "methods.txt"
            body = "Method: System.Void A()\n\nDisassembly:\n\tret\n\nISIL:\n\t0 Return\n"
            path.write_text(body + "\n" + body, encoding="utf-8")
            with self.assertRaises(controller_path.EvidenceError):
                controller_path._normalise_method_blocks(path)

    def test_pe_extent_must_be_file_backed_and_executable(self):
        image = bytearray(0x400)
        image[:2] = b"MZ"
        struct.pack_into("<I", image, 0x3C, 0x80)
        image[0x80:0x84] = b"PE\0\0"
        struct.pack_into("<H", image, 0x86, 1)
        struct.pack_into("<H", image, 0x94, 0xE0)
        section = 0x80 + 24 + 0xE0
        image[section:section + 8] = b"code\0\0\0\0"
        struct.pack_into("<IIII", image, section + 8, 0x100, 0x1000, 0x100, 0x200)
        struct.pack_into("<I", image, section + 36, 0x20000000)
        image[0x210:0x214] = b"REK!"

        sections = controller_path._pe_sections(bytes(image))
        found, offset, body = controller_path._native_extent(
            bytes(image), sections, 0x1010, 4
        )
        self.assertEqual(found["name"], "code")
        self.assertEqual(offset, 0x210)
        self.assertEqual(body, b"REK!")
        with self.assertRaises(controller_path.EvidenceError):
            controller_path._native_extent(bytes(image), sections, 0x10FF, 2)

        sections[0]["executable"] = False
        with self.assertRaises(controller_path.EvidenceError):
            controller_path._native_extent(bytes(image), sections, 0x1010, 4)

    def test_fixed_build_report_rejects_overclaim_and_source_drift(self):
        report = valid_report_fixture()
        self.assertEqual(controller_path.validate_report(report), [])

        overclaim = copy.deepcopy(report)
        overclaim["scope"]["private_ai_runtime_active"]["state"] = "measured"
        self.assertTrue(any(
            "runtime activation" in error
            for error in controller_path.validate_report(overclaim)
        ))

        dummy = copy.deepcopy(report)
        dummy["sources"]["il2cpp_recovery"]["dummy_method_bodies_consumed"] = True
        self.assertTrue(any(
            "dummy method bodies" in error
            for error in controller_path.validate_report(dummy)
        ))

        drift = copy.deepcopy(report)
        drift["native_methods"][0]["native_body_sha256"] = "0" * 64
        self.assertTrue(any(
            "native hash mismatch" in error
            for error in controller_path.validate_report(drift)
        ))

    def test_move_nulls_and_bound_object_fields_cannot_drift(self):
        report = valid_report_fixture()
        invented_move = copy.deepcopy(report)
        invented_move["serialized_t800"]["robot_config"]["move_map"][0]["referenced_object"] = {
            "display_name": "invented"
        }
        self.assertTrue(any(
            "slot 0 null target" in error
            for error in controller_path.validate_report(invented_move)
        ))

        unbound_move = copy.deepcopy(report)
        unbound_move["serialized_t800"]["robot_config"]["move_map"][2][
            "referenced_object"
        ]["fields"]["displayName"] = "unverified"
        self.assertTrue(any(
            "slot 2 referenced object mismatch" in error
            for error in controller_path.validate_report(unbound_move)
        ))

        for row in report["serialized_t800"]["robot_config"]["move_map"]:
            if row["pointer"]["path_id"] == 0:
                continue
            self.assertEqual(row["referenced_object"]["state"], "measured")
            self.assertEqual(
                row["referenced_object"]["fields"]["npzFile"],
                {"m_FileID": 0, "m_PathID": 0},
            )

        invented_unknown = copy.deepcopy(report)
        invented_unknown["hard_unknowns"][0]["state"] = "assumed"
        self.assertTrue(any(
            "must remain unknown" in error
            for error in controller_path.validate_report(invented_unknown)
        ))

    def test_formula_transcription_retains_signs_and_thresholds(self):
        facts = controller_path._static_semantics()
        mimic = "\n".join(facts["engine_ai_mimic_torque_limit"]["equations"])
        self.assertIn("- joint_kd[i] * qd[i]", mimic)
        self.assertIn("1700", mimic)
        self.assertIn("1e-6", mimic)
        walk = "\n".join(facts["engine_ai_walk_target_clamp"]["equations"])
        self.assertIn("0.001", walk)
        pd_writes = facts["robot_implicit_pd_configuration"]["writes"]
        self.assertIn("biasprm[1] = -kp", pd_writes)
        self.assertIn("biasprm[2] = -kd", pd_writes)


if __name__ == "__main__":
    unittest.main()
