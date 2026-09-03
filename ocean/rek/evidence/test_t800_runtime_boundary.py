import copy
import json
import tempfile
import unittest
from pathlib import Path

import t800_runtime_boundary
from test_controller_path import valid_report_fixture


def controller_fixture():
    report = valid_report_fixture()
    null_pointer = {"m_FileID": 0, "m_PathID": 0}
    profiles = []
    for index in range(45):
        profiles.append({
            "name": (
                "switch_to_boxing_idle@default" if index == 0
                else f"profile_{index}"
            ),
            "onnxBytes": dict(null_pointer),
            "configJson": dict(null_pointer),
            "trajectoryCsv": dict(null_pointer),
        })
    report["serialized_t800"]["runner"] = {"values": {"profiles": profiles}}
    return report


def authority_fixture():
    return {
        "scope": {"build_fingerprint": t800_runtime_boundary.BUILD_FINGERPRINT},
        "verdict": {"verdict": "remote_authority"},
    }


def log_fixture():
    return "\n".join([
        "[EngineAIPolicyRunner] Profile 'switch_to_boxing_idle@default' missing onnxBytes or configJson.",
        "UnityEngine.Debug:LogError(Object)",
        "REKApp.EngineAIPolicyRunner:LoadProfile(ProfileAssets, SessionOptions)",
        "REKApp.EngineAIPolicyRunner:Init()",
        "[Robot] Visual-only mode: disabled 148 MuJoCo components on engineai_t800_FactoryPolicy(Clone); colliders preserved for local VFX.",
        "[Robot.Network] Initialized on engineai_t800_FactoryPolicy(Clone): 26 bones, role=Client",
        "",
    ])


class T800RuntimeBoundaryTests(unittest.TestCase):
    def write_inputs(self, root, controller=None, authority=None, log=None):
        controller_path = root / "controller.json"
        authority_path = root / "authority.json"
        log_path = root / "Player.log"
        controller_path.write_text(
            json.dumps(controller or controller_fixture()), encoding="utf-8"
        )
        authority_path.write_text(
            json.dumps(authority or authority_fixture()), encoding="utf-8"
        )
        log_path.write_text(log if log is not None else log_fixture(), encoding="utf-8")
        return log_path, controller_path, authority_path

    def test_binds_null_payloads_to_observed_visual_client(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory))
            report = t800_runtime_boundary.build_report(*paths)
        self.assertTrue(report["serialized_client"]["profile_payload_sets_all_null"])
        self.assertEqual(
            report["verdict"]["client_t800_runner_state"],
            "initialization_aborted_on_missing_profile_payload",
        )
        self.assertFalse(
            report["verdict"]["client_contains_authoritative_t800_policy_payload"]
        )
        self.assertEqual(
            report["verdict"]["authoritative_controller_payload_location"],
            "unknown",
        )

    def test_rejects_non_null_serialized_payload(self):
        controller = controller_fixture()
        controller["serialized_t800"]["runner"]["values"]["profiles"][0][
            "onnxBytes"
        ] = {"m_FileID": 0, "m_PathID": 99}
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory), controller=controller)
            with self.assertRaisesRegex(
                t800_runtime_boundary.EvidenceError, "45 null T800 payload"
            ):
                t800_runtime_boundary.build_report(*paths)

    def test_rejects_missing_init_stack_frame(self):
        log = log_fixture().replace(
            "REKApp.EngineAIPolicyRunner:Init()", "REKApp.Other:Init()"
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory), log=log)
            with self.assertRaisesRegex(
                t800_runtime_boundary.EvidenceError, "no Init stack frame"
            ):
                t800_runtime_boundary.build_report(*paths)

    def test_rejects_authority_drift(self):
        authority = copy.deepcopy(authority_fixture())
        authority["verdict"]["verdict"] = "inconclusive"
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory), authority=authority)
            with self.assertRaisesRegex(
                t800_runtime_boundary.EvidenceError, "remote authority"
            ):
                t800_runtime_boundary.build_report(*paths)


if __name__ == "__main__":
    unittest.main()
