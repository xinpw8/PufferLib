import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_sonic_asset_extract as extractor


def synthetic_contract():
    contract = copy.deepcopy(extractor.load_contract())
    payloads = {
        "model_decoder.onnx": b"synthetic decoder hash-only payload",
        "sonic_config": json.dumps(
            contract["config"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        "model_encoder.onnx": b"synthetic encoder hash-only payload",
    }
    for spec in contract["text_assets"]:
        payload = payloads[spec["name"]]
        spec["bytes"] = len(payload)
        spec["sha256"] = extractor.sha256_bytes(payload)
    return contract, payloads


class ContractTests(unittest.TestCase):
    def test_tracked_contract_is_exact_and_redacts_model_payloads(self):
        contract = extractor.load_contract()
        self.assertEqual(contract["schema"], extractor.CONTRACT_SCHEMA)
        self.assertEqual(len(contract["config"]["joints"]), 29)
        assets = {item["name"]: item for item in contract["text_assets"]}
        self.assertEqual(
            assets["sonic_config"]["sha256"],
            "9e8e18d763adfdce094ece2061a42acf4b48e43f89b62ccea3d8b7ba25c2a355",
        )
        self.assertFalse(assets["model_encoder.onnx"]["payload_committed"])
        self.assertFalse(assets["model_decoder.onnx"]["payload_committed"])
        self.assertNotIn("payload", assets["model_encoder.onnx"])
        self.assertNotIn("payload", assets["model_decoder.onnx"])
        self.assertFalse(contract["scope"]["current_steam_authority"])
        self.assertFalse(
            contract["scope"]["native_isil_reference"]["same_as_contract_build"]
        )
        self.assertTrue(
            contract["scope"]["current_steam_build"]["same_as_native_isil_inputs"]
        )

    def test_contract_rejects_duplicate_or_missing_text_assets(self):
        contract = copy.deepcopy(extractor.load_contract())
        contract["text_assets"][1]["path_id"] = contract["text_assets"][0]["path_id"]
        with self.assertRaisesRegex(extractor.SonicAssetError, "unique"):
            extractor.validate_contract(contract)

        contract = copy.deepcopy(extractor.load_contract())
        contract["text_assets"].pop()
        with self.assertRaisesRegex(extractor.SonicAssetError, "must be exactly"):
            extractor.validate_contract(contract)


class BuildAndAssetValidationTests(unittest.TestCase):
    def test_build_validation_checks_every_file_size_and_hash(self):
        contract = copy.deepcopy(extractor.load_contract())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, spec in contract["files"].items():
                payload = (key + ":pinned").encode("ascii")
                path = root.joinpath(*spec["path"].split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                spec["bytes"] = len(payload)
                spec["sha256"] = extractor.sha256_bytes(payload)
            extractor.validate_build_files(root, contract)

            (root / "GameAssembly.dll").write_bytes(b"changed")
            with self.assertRaisesRegex(
                extractor.SonicAssetError, "game_assembly byte count mismatch"
            ):
                extractor.validate_build_files(root, contract)

    def test_selected_assets_validate_without_writing_model_bytes(self):
        contract, payloads = synthetic_contract()
        loaded = {
            item["path_id"]: (item["name"], payloads[item["name"]])
            for item in contract["text_assets"]
        }
        observed = extractor.validate_text_assets(loaded, contract)
        self.assertEqual(set(observed), set(payloads))
        self.assertTrue(all("payload" not in item for item in observed.values()))

        changed = dict(loaded)
        path_id = contract["text_assets"][0]["path_id"]
        changed[path_id] = (loaded[path_id][0], loaded[path_id][1] + b"x")
        with self.assertRaisesRegex(extractor.SonicAssetError, "byte count mismatch"):
            extractor.validate_text_assets(changed, contract)

    def test_surrogateescape_round_trips_binary_textasset(self):
        payload = bytes(range(256))
        value = payload.decode("utf-8", "surrogateescape")
        self.assertEqual(extractor._script_bytes(value), payload)


class CandidateComparisonTests(unittest.TestCase):
    def test_comparator_covers_all_semantic_constant_groups(self):
        contract = extractor.load_contract()
        config = contract["config"]
        joints = config["joints"]
        encoder_offsets = extractor._offsets(config, "encoder")
        decoder_offsets = extractor._offsets(config, "decoder")
        candidate = SimpleNamespace(
            CONTROL_HZ=50,
            CONTROL_DT=0.02,
            HISTORY_FRAMES=10,
            ENCODER_DIM=1762,
            DECODER_DIM=994,
            TOKEN_DIM=64,
            ACTION_DIM=29,
            ACTION_CLIP=100.0,
            FUTURE_STEP=5,
            COMMAND_LPF_CUTOFF_HZ=50.0,
            ISAACLAB_TO_MUJOCO=config["isaaclab_to_mujoco"],
            MUJOCO_TO_ISAACLAB=config["mujoco_to_isaaclab"],
            DEFAULT_ANGLES_MUJOCO=[item["default_pos"] for item in joints],
            PUBLIC_EFFORT_LIMIT_MUJOCO=[item["effort_limit"] for item in joints],
            ACTION_SCALE_MUJOCO=[item["action_scale"] for item in joints],
            KP_MUJOCO=[item["kp"] for item in joints],
            KD_MUJOCO=[item["kd"] for item in joints],
            ENCODER_OFFSETS=encoder_offsets,
            DECODER_OFFSETS=decoder_offsets,
        )
        report = extractor.compare_candidate_constants(contract, candidate)
        self.assertTrue(all(item["exact"] for item in report["scalars"].values()))
        self.assertTrue(all(item["exact"] for item in report["vectors"].values()))
        self.assertTrue(all(item["exact"] for item in report["offsets"].values()))
        self.assertEqual(
            set(report["internal_only"]),
            {"RUN_SCHEMA", "TRACE_SCHEMA", "CLASSIFICATION"},
        )

        candidate.KP_MUJOCO = list(candidate.KP_MUJOCO)
        candidate.KP_MUJOCO[3] += 0.25
        mismatch = extractor.compare_candidate_constants(contract, candidate)
        self.assertEqual(mismatch["vectors"]["KP_MUJOCO"]["mismatch_count"], 1)
        self.assertEqual(mismatch["vectors"]["KP_MUJOCO"]["mismatches"][0]["index"], 3)

    def test_current_candidate_has_no_material_constant_mismatch(self):
        candidate_dir = HERE.parent.parent / "rek_g1"
        sys.path.insert(0, str(candidate_dir))
        try:
            import gear_sonic_candidate as candidate
        finally:
            sys.path.pop(0)
        report = extractor.compare_candidate_constants(
            extractor.load_contract(), candidate
        )
        self.assertTrue(all(item["exact"] for item in report["scalars"].values()))
        self.assertTrue(report["vectors"]["ISAACLAB_TO_MUJOCO"]["exact"])
        self.assertTrue(report["vectors"]["MUJOCO_TO_ISAACLAB"]["exact"])
        self.assertTrue(report["vectors"]["DEFAULT_ANGLES_MUJOCO"]["exact"])
        self.assertTrue(report["vectors"]["PUBLIC_EFFORT_LIMIT_MUJOCO"]["exact"])
        for name in ("ACTION_SCALE_MUJOCO", "KP_MUJOCO", "KD_MUJOCO"):
            self.assertTrue(report["vectors"][name]["exact"], name)
        self.assertTrue(all(item["exact"] for item in report["offsets"].values()))


if __name__ == "__main__":
    unittest.main()
