import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_asset_extract as asset_extract
import g1_motion_clip_extract as extractor


class MotionClipContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.manifest_sha256 = asset_extract.load_pinned_manifest()
        cls.probe, cls.probe_bytes, cls.probe_sha256 = (
            extractor.validate_pinned_probe(extractor.DEFAULT_PROBE_PATH)
        )
        cls.expected_identities = {
            (asset.path_id, asset.name) for asset in cls.manifest.assets
        }

    def minimal_probe(self):
        root = {
            key: copy.deepcopy(value)
            for key, value in self.probe.items()
            if key != "targets"
        }
        targets = []
        for target in self.probe["targets"]:
            if target.get("class") != extractor.EXPECTED_MOCAP_CLASS:
                continue
            values = target["values"]
            identity = (values["npzFile"]["m_PathID"], values["m_Name"])
            if identity in self.expected_identities:
                targets.append(copy.deepcopy(target))
        self.assertEqual(len(targets), len(asset_extract.EXPECTED_ROLES))
        root["targets"] = targets
        self.sync_counts(root)
        return root

    @staticmethod
    def sync_counts(probe):
        probe["target_count"] = len(probe["targets"])
        probe["parsed_target_count"] = len(probe["targets"])
        probe["unparsed_target_count"] = 0

    def build(self, probe):
        return extractor.build_contract(
            probe,
            self.manifest,
            self.manifest_sha256,
            probe_bytes=self.probe_bytes,
            probe_sha256=self.probe_sha256,
        )

    def test_checked_in_contract_is_exact_pinned_extraction(self):
        rendered = extractor.extract_contract()
        self.assertEqual(extractor.DEFAULT_CONTRACT_PATH.read_bytes(), rendered)
        contract = json.loads(rendered)
        self.assertEqual(contract["schema"], extractor.CONTRACT_SCHEMA)
        self.assertEqual(
            contract["build_fingerprint"], self.manifest.build_fingerprint
        )
        self.assertEqual(
            contract["source"]["probe"],
            {
                "bytes": extractor.PINNED_PROBE_BYTES,
                "inventory_sha256": self.probe["inventory_sha256"],
                "name": extractor.PROBE_NAME,
                "schema": extractor.PROBE_SCHEMA,
                "sha256": extractor.PINNED_PROBE_SHA256,
                "unity_version": extractor.EXPECTED_UNITY_VERSION,
                "unitypy_version": self.manifest.parser_version,
            },
        )
        roles = [clip["role"] for clip in contract["clips"]]
        self.assertEqual(roles, sorted(asset_extract.EXPECTED_ROLES))
        required = {
            "blendInTime",
            "blendOutTime",
            "displayName",
            "endFrame",
            "impactEvents",
            "impactForgivenessDuration",
            "impactReversal",
            "impactYawForgiveness",
            "loop",
            "mirror",
            "path_id",
            "playbackSpeed",
            "serialized_bytes",
            "serialized_sha256",
            "startFrame",
            "yawBlend",
            "yawForgiveness",
        }
        for clip in contract["clips"]:
            self.assertEqual(set(clip["mocap_clip_config"]), required)
            asset = next(
                item for item in self.manifest.assets if item.role == clip["role"]
            )
            self.assertEqual(
                (clip["npz"]["path_id"], clip["npz"]["name"]),
                (asset.path_id, asset.name),
            )
            source_target = next(
                target
                for target in self.probe["targets"]
                if target.get("class") == extractor.EXPECTED_MOCAP_CLASS
                and target["values"]["npzFile"]["m_PathID"] == asset.path_id
                and target["values"]["m_Name"] == asset.name
            )
            config = clip["mocap_clip_config"]
            self.assertEqual(config["path_id"], source_target["path_id"])
            self.assertEqual(
                config["serialized_bytes"], source_target["serialized_bytes"]
            )
            self.assertEqual(
                config["serialized_sha256"], source_target["serialized_sha256"]
            )
            for field in required - {
                "path_id",
                "serialized_bytes",
                "serialized_sha256",
            }:
                self.assertEqual(config[field], source_target["values"][field])

        by_role = {clip["role"]: clip for clip in contract["clips"]}
        self.assertEqual(by_role["turn_right"]["mocap_clip_config"]["path_id"], 2718)
        self.assertEqual(by_role["walk"]["mocap_clip_config"]["path_id"], 2720)

    def test_contract_is_deterministic_under_target_reordering(self):
        probe = self.minimal_probe()
        forward = extractor.render_contract(self.build(probe))
        probe["targets"].reverse()
        reverse = extractor.render_contract(self.build(probe))
        self.assertEqual(forward, reverse)

    def test_missing_duplicate_and_identity_mismatch_fail_closed(self):
        missing = self.minimal_probe()
        missing["targets"].pop()
        self.sync_counts(missing)
        with self.assertRaisesRegex(extractor.ExtractionError, "missing MocapClipConfig"):
            self.build(missing)

        duplicate = self.minimal_probe()
        duplicate["targets"].append(copy.deepcopy(duplicate["targets"][0]))
        self.sync_counts(duplicate)
        with self.assertRaisesRegex(extractor.ExtractionError, "duplicate MocapClipConfig"):
            self.build(duplicate)

        wrong_name = self.minimal_probe()
        wrong_name["targets"][0]["values"]["m_Name"] += "_mismatch"
        with self.assertRaisesRegex(extractor.ExtractionError, "missing MocapClipConfig"):
            self.build(wrong_name)

        wrong_path = self.minimal_probe()
        wrong_path["targets"][0]["values"]["npzFile"]["m_PathID"] = 999_999
        with self.assertRaisesRegex(extractor.ExtractionError, "missing MocapClipConfig"):
            self.build(wrong_path)

    def test_unknown_and_missing_schema_fields_fail_closed(self):
        unknown_root = self.minimal_probe()
        unknown_root["unknown"] = True
        with self.assertRaisesRegex(extractor.ExtractionError, r"unknown=\['unknown'\]"):
            self.build(unknown_root)

        unknown_value = self.minimal_probe()
        unknown_value["targets"][0]["values"]["unknown"] = 1
        with self.assertRaisesRegex(extractor.ExtractionError, r"unknown=\['unknown'\]"):
            self.build(unknown_value)

        missing_value = self.minimal_probe()
        del missing_value["targets"][0]["values"]["blendInTime"]
        with self.assertRaisesRegex(extractor.ExtractionError, r"missing=\['blendInTime'\]"):
            self.build(missing_value)

        missing_event_value = self.minimal_probe()
        kick = next(
            target
            for target in missing_event_value["targets"]
            if target["values"]["impactEvents"]
        )
        del kick["values"]["impactEvents"][0]["limb"]
        with self.assertRaisesRegex(extractor.ExtractionError, r"missing=\['limb'\]"):
            self.build(missing_event_value)

    def test_invalid_types_nonfinite_values_and_counts_fail_closed(self):
        bad_flag = self.minimal_probe()
        bad_flag["targets"][0]["values"]["loop"] = True
        with self.assertRaisesRegex(extractor.ExtractionError, "loop must be an integer"):
            self.build(bad_flag)

        nonfinite = self.minimal_probe()
        nonfinite["targets"][0]["values"]["playbackSpeed"] = float("nan")
        with self.assertRaisesRegex(extractor.ExtractionError, "must be finite"):
            self.build(nonfinite)

        bad_frames = self.minimal_probe()
        bad_frames["targets"][0]["values"]["startFrame"] = 2
        bad_frames["targets"][0]["values"]["endFrame"] = 1
        with self.assertRaisesRegex(extractor.ExtractionError, "frame range is invalid"):
            self.build(bad_frames)

        bad_count = self.minimal_probe()
        bad_count["target_count"] += 1
        with self.assertRaisesRegex(extractor.ExtractionError, "target_count"):
            self.build(bad_count)

        unparsed = self.minimal_probe()
        unparsed["parsed_target_count"] -= 1
        unparsed["unparsed_target_count"] = 1
        with self.assertRaisesRegex(extractor.ExtractionError, "unparsed targets"):
            self.build(unparsed)

    def test_build_parser_container_and_manifest_roles_must_match(self):
        wrong_build = self.minimal_probe()
        wrong_build["build_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(extractor.ExtractionError, "fingerprints mismatch"):
            self.build(wrong_build)

        wrong_parser = self.minimal_probe()
        wrong_parser["unitypy_version"] = "0.0.0"
        with self.assertRaisesRegex(extractor.ExtractionError, "UnityPy version mismatch"):
            self.build(wrong_parser)

        wrong_container = self.minimal_probe()
        wrong_container["targets"][0]["container"] = "other.assets"
        with self.assertRaisesRegex(extractor.ExtractionError, "container mismatch"):
            self.build(wrong_container)

        bad_asset = replace(self.manifest.assets[0], role="unknown_role")
        bad_manifest = replace(
            self.manifest, assets=(bad_asset, *self.manifest.assets[1:])
        )
        with self.assertRaisesRegex(extractor.ExtractionError, "manifest roles mismatch"):
            extractor.build_contract(
                self.minimal_probe(),
                bad_manifest,
                self.manifest_sha256,
                probe_bytes=self.probe_bytes,
                probe_sha256=self.probe_sha256,
            )

    def test_probe_bytes_and_json_object_keys_are_strictly_pinned(self):
        self.assertEqual(
            extractor.validate_pinned_probe(extractor.DEFAULT_PROBE_PATH)[1:],
            (extractor.PINNED_PROBE_BYTES, extractor.PINNED_PROBE_SHA256),
        )
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / extractor.PROBE_NAME
            changed.write_bytes(extractor.DEFAULT_PROBE_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(extractor.ExtractionError, "byte count mismatch"):
                extractor.validate_pinned_probe(changed)

        with self.assertRaisesRegex(extractor.ExtractionError, "duplicate key 'a'"):
            extractor.load_json_bytes(b'{"a": 1, "a": 2}', "fixture")
        with self.assertRaisesRegex(extractor.ExtractionError, "nonfinite constant"):
            extractor.load_json_bytes(b'{"a": NaN}', "fixture")

        probe = self.minimal_probe()
        with self.assertRaisesRegex(extractor.ExtractionError, "byte count is not pinned"):
            extractor.build_contract(
                probe,
                self.manifest,
                self.manifest_sha256,
                probe_bytes=self.probe_bytes - 1,
                probe_sha256=self.probe_sha256,
            )
        with self.assertRaisesRegex(extractor.ExtractionError, "SHA-256 is not pinned"):
            extractor.build_contract(
                probe,
                self.manifest,
                self.manifest_sha256,
                probe_bytes=self.probe_bytes,
                probe_sha256="0" * 64,
            )
        with self.assertRaisesRegex(
            extractor.ExtractionError, "manifest canonical SHA-256 mismatch"
        ):
            extractor.build_contract(
                probe,
                self.manifest,
                "0" * 64,
                probe_bytes=self.probe_bytes,
                probe_sha256=self.probe_sha256,
            )


if __name__ == "__main__":
    unittest.main()
