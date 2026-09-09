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
import g1_motion_route_extract as extractor


class MotionRouteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.manifest_sha256 = asset_extract.load_pinned_manifest()
        cls.probe, cls.probe_bytes, cls.probe_sha256 = (
            extractor.clip_extract.validate_pinned_probe(extractor.DEFAULT_PROBE_PATH)
        )
        cls.config_path_ids = {
            spec.mocap_clip_config_path_id for spec in extractor.ROUTE_SPECS
        }

    @staticmethod
    def sync_counts(probe):
        probe["target_count"] = len(probe["targets"])
        probe["parsed_target_count"] = len(probe["targets"])
        probe["unparsed_target_count"] = 0

    def minimal_probe(self):
        root = {
            key: copy.deepcopy(value)
            for key, value in self.probe.items()
            if key != "targets"
        }
        root["targets"] = [
            copy.deepcopy(target)
            for target in self.probe["targets"]
            if (
                target.get("class") == "RobotConfig"
                and target.get("path_id") == extractor.ROBOT_CONFIG_PATH_ID
            )
            or (
                target.get("class") == extractor.clip_extract.EXPECTED_MOCAP_CLASS
                and target.get("path_id") in self.config_path_ids
            )
        ]
        self.assertEqual(len(root["targets"]), 12)
        self.sync_counts(root)
        return root

    def build(self, probe, manifest=None, manifest_sha256=None):
        return extractor.build_contract(
            probe,
            self.manifest if manifest is None else manifest,
            self.manifest_sha256 if manifest_sha256 is None else manifest_sha256,
            probe_bytes=self.probe_bytes,
            probe_sha256=self.probe_sha256,
        )

    def test_checked_in_contract_is_exact_pinned_extraction(self):
        rendered = extractor.extract_contract()
        self.assertEqual(extractor.DEFAULT_CONTRACT_PATH.read_bytes(), rendered)
        contract = json.loads(rendered)
        self.assertEqual(contract["schema"], extractor.CONTRACT_SCHEMA)
        self.assertFalse(contract["runtime_selection_observed"])
        self.assertFalse(contract["parity_validated"])
        self.assertEqual(contract["build_fingerprint"], self.manifest.build_fingerprint)
        self.assertEqual(
            contract["source"]["probe"]["sha256"],
            extractor.clip_extract.PINNED_PROBE_SHA256,
        )
        self.assertEqual(
            contract["source"]["runtime_assets_manifest"]["canonical_sha256"],
            asset_extract.PINNED_MANIFEST_SHA256,
        )
        self.assertEqual(
            contract["source"]["unity_container"]["sha256"],
            self.manifest.source_sha256,
        )

    def test_all_route_ids_bindings_configs_and_npz_identities_are_exact(self):
        contract = self.build(self.minimal_probe())
        routes = contract["routes"]
        self.assertEqual([route["route_id"] for route in routes], list(range(11)))
        self.assertEqual(
            [route["runtime_move_index"] for route in routes],
            [None, None, None, None, None, None, None, 6, 7, 8, 9],
        )
        assets = {asset.role: asset for asset in self.manifest.assets}
        source_targets = {target["path_id"]: target for target in self.probe["targets"]}
        for spec, route in zip(extractor.ROUTE_SPECS, routes):
            self.assertEqual(route["name"], spec.name)
            self.assertEqual(route["kind"], spec.kind)
            self.assertEqual(
                route["robot_config_binding"],
                {
                    "field": spec.robot_config_field,
                    "index": spec.runtime_move_index,
                },
            )
            config = route["mocap_clip_config"]
            source = source_targets[spec.mocap_clip_config_path_id]
            self.assertEqual(config["path_id"], spec.mocap_clip_config_path_id)
            self.assertEqual(config["m_Name"], source["values"]["m_Name"])
            self.assertEqual(
                config["serialized_sha256"], source["serialized_sha256"]
            )
            asset = assets[spec.npz_role]
            self.assertEqual(
                (route["npz"]["path_id"], route["npz"]["name"], route["npz"]["sha256"]),
                (asset.path_id, asset.name, asset.sha256),
            )

        by_name = {route["name"]: route for route in routes}
        self.assertEqual(
            by_name["backward"]["mocap_clip_config"]["playbackSpeed"], -1.0
        )
        self.assertEqual(
            by_name["strafe_right"]["mocap_clip_config"]["playbackSpeed"],
            -1.0,
        )
        self.assertEqual(
            by_name["turn_left"]["mocap_clip_config"]["playbackSpeed"], -1.0
        )
        self.assertEqual(
            by_name["forward"]["npz"], by_name["backward"]["npz"]
        )
        self.assertEqual(
            by_name["strafe_left"]["npz"], by_name["strafe_right"]["npz"]
        )
        self.assertEqual(
            by_name["turn_left"]["npz"], by_name["turn_right"]["npz"]
        )

    def test_robot_config_field_and_move_index_changes_fail_closed(self):
        wrong_field = self.minimal_probe()
        robot = next(
            target
            for target in wrong_field["targets"]
            if target["class"] == "RobotConfig"
        )
        robot["values"]["walkBackward"]["m_PathID"] = 2720
        with self.assertRaisesRegex(extractor.ExtractionError, "backward.*binding mismatch"):
            self.build(wrong_field)

        wrong_move = self.minimal_probe()
        robot = next(
            target
            for target in wrong_move["targets"]
            if target["class"] == "RobotConfig"
        )
        robot["values"]["moves"][8]["m_PathID"] = 2714
        with self.assertRaisesRegex(extractor.ExtractionError, "kick_move_8.*binding mismatch"):
            self.build(wrong_move)

        short_moves = self.minimal_probe()
        robot = next(
            target
            for target in short_moves["targets"]
            if target["class"] == "RobotConfig"
        )
        robot["values"]["moves"].pop()
        with self.assertRaisesRegex(extractor.ExtractionError, "exactly 17"):
            self.build(short_moves)

    def test_missing_duplicate_and_substituted_config_fail_closed(self):
        missing_robot = self.minimal_probe()
        missing_robot["targets"] = [
            target for target in missing_robot["targets"] if target["class"] != "RobotConfig"
        ]
        self.sync_counts(missing_robot)
        with self.assertRaisesRegex(extractor.ExtractionError, "expected one G1 RobotConfig"):
            self.build(missing_robot)

        duplicate_robot = self.minimal_probe()
        duplicate_robot["targets"].append(copy.deepcopy(next(
            target for target in duplicate_robot["targets"] if target["class"] == "RobotConfig"
        )))
        self.sync_counts(duplicate_robot)
        with self.assertRaisesRegex(extractor.ExtractionError, "got 2"):
            self.build(duplicate_robot)

        missing_config = self.minimal_probe()
        missing_config["targets"] = [
            target for target in missing_config["targets"] if target.get("path_id") != 2721
        ]
        self.sync_counts(missing_config)
        with self.assertRaisesRegex(extractor.ExtractionError, "missing MocapClipConfig"):
            self.build(missing_config)

        wrong_name = self.minimal_probe()
        target = next(target for target in wrong_name["targets"] if target.get("path_id") == 2721)
        target["values"]["m_Name"] = "walking_processed"
        with self.assertRaisesRegex(extractor.ExtractionError, "name mismatch"):
            self.build(wrong_name)

        wrong_npz = self.minimal_probe()
        target = next(target for target in wrong_npz["targets"] if target.get("path_id") == 2716)
        target["values"]["npzFile"]["m_PathID"] = 370
        with self.assertRaisesRegex(extractor.ExtractionError, "NPZ path ID mismatch"):
            self.build(wrong_npz)

    def test_schema_and_source_pins_fail_closed(self):
        unknown_config_field = self.minimal_probe()
        target = next(
            target
            for target in unknown_config_field["targets"]
            if target.get("path_id") == 2702
        )
        target["values"]["unknown"] = 1
        with self.assertRaisesRegex(extractor.ExtractionError, r"unknown=\['unknown'\]"):
            self.build(unknown_config_field)

        wrong_robot_hash = self.minimal_probe()
        robot = next(
            target
            for target in wrong_robot_hash["targets"]
            if target["class"] == "RobotConfig"
        )
        robot["serialized_sha256"] = "0" * 64
        with self.assertRaisesRegex(extractor.ExtractionError, "serialized_sha256 mismatch"):
            self.build(wrong_robot_hash)

        with self.assertRaisesRegex(
            extractor.ExtractionError, "manifest canonical SHA-256 mismatch"
        ):
            self.build(self.minimal_probe(), manifest_sha256="0" * 64)

        with self.assertRaisesRegex(extractor.ExtractionError, "probe byte count is not pinned"):
            extractor.build_contract(
                self.minimal_probe(),
                self.manifest,
                self.manifest_sha256,
                probe_bytes=self.probe_bytes - 1,
                probe_sha256=self.probe_sha256,
            )

        with self.assertRaisesRegex(extractor.ExtractionError, "probe SHA-256 is not pinned"):
            extractor.build_contract(
                self.minimal_probe(),
                self.manifest,
                self.manifest_sha256,
                probe_bytes=self.probe_bytes,
                probe_sha256="0" * 64,
            )

    def test_runtime_asset_role_and_identity_substitution_fail_closed(self):
        walk_index = next(
            index for index, asset in enumerate(self.manifest.assets) if asset.role == "walk"
        )
        assets = list(self.manifest.assets)
        assets[walk_index] = replace(assets[walk_index], path_id=999_999)
        bad_manifest = replace(self.manifest, assets=tuple(assets))
        with self.assertRaisesRegex(extractor.ExtractionError, "NPZ identity mismatch"):
            self.build(self.minimal_probe(), manifest=bad_manifest)

    def test_contract_is_deterministic_and_cli_check_is_fail_closed(self):
        probe = self.minimal_probe()
        forward = extractor.render_contract(self.build(probe))
        probe["targets"].reverse()
        self.assertEqual(forward, extractor.render_contract(self.build(probe)))

        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "contract.json"
            wrong.write_bytes(b"{}\n")
            self.assertEqual(extractor.main(["--check", str(wrong)]), 2)


if __name__ == "__main__":
    unittest.main()
