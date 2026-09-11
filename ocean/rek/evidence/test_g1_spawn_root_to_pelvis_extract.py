import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_spawn_root_to_pelvis_extract as extractor


class SpawnTransformExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rendered = extractor.extract_contract()
        cls.contract = json.loads(cls.rendered)
        cls.probe_v7 = extractor.load_json_bytes(
            extractor.DEFAULT_PROBE_V7_PATH.read_bytes(), "probe_v7"
        )
        cls.probe_v8 = extractor.load_json_bytes(
            extractor.DEFAULT_PROBE_V8_PATH.read_bytes(), "probe_v8"
        )

    def test_checked_in_contract_is_exact_pinned_extraction(self):
        self.assertEqual(extractor.DEFAULT_CONTRACT_PATH.read_bytes(), self.rendered)
        self.assertEqual(self.contract["schema"], extractor.CONTRACT_SCHEMA)
        self.assertEqual(
            self.contract["spawn_gate_status"],
            "closed_for_pinned_client_initial_instantiation",
        )
        self.assertTrue(self.contract["exact_initial_spawn_pose_available"])
        self.assertFalse(self.contract["control_equivalent"])
        self.assertEqual(
            self.contract["evidence_projection_sha256"],
            "b0213e7a154ca55ab59cab6f05b5fb708ceca5549404380cd32b695793b98978",
        )

    def test_exact_transform_and_composed_qpos_are_stable(self):
        transform = self.contract["prefab_root_to_pelvis"]
        self.assertTrue(transform["direct_child"])
        self.assertEqual(
            transform["unity"],
            {
                "position_xyz_m": [0.0, 0.7929999828338623, 0.0],
                "quaternion_wxyz": [-1.0, 0.0, 0.0, 0.0],
                "scale_xyz": [1.0, 1.0, 1.0],
            },
        )
        self.assertEqual(
            transform["mujoco"],
            {
                "position_xyz_m": [0.0, 0.0, 0.7929999828338623],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale_xyz": [1.0, 1.0, 1.0],
            },
        )
        self.assertEqual(
            self.contract["arena_spawn_to_free_joint_pelvis"]["player"][
                "free_joint_qpos_prefix"
            ],
            [
                -0.9000000357627869,
                0.0,
                0.8029999826103449,
                0.9999999999998863,
                0.0,
                0.0,
                -4.7683710135965193e-07,
            ],
        )
        self.assertEqual(
            self.contract["arena_spawn_to_free_joint_pelvis"]["opponent"][
                "free_joint_qpos_prefix"
            ],
            [
                0.8999999761581421,
                0.0,
                0.8029999826103449,
                -1.1026858146572184e-06,
                -0.0,
                0.0,
                0.999999999999392,
            ],
        )

    def test_probe_generations_have_identical_source_projection(self):
        self.assertEqual(
            extractor.source_projection(self.probe_v7),
            extractor.source_projection(self.probe_v8),
        )

    def test_non_direct_pelvis_hierarchy_is_rejected(self):
        changed = copy.deepcopy(self.probe_v7)
        body = next(
            item
            for item in changed["targets"]
            if item.get("class") == "MjBody"
            and item.get("path_id") == extractor.EXPECTED_BODY["path_id"]
        )
        chain = body["hierarchy"]["transform_chain"]
        intermediate = copy.deepcopy(chain[0])
        intermediate["name"] = "unexpected_intermediate"
        intermediate["transform_path_id"] = 999_999
        intermediate["sibling_index"] = 0
        chain.insert(1, intermediate)
        with self.assertRaisesRegex(
            extractor.ExtractionError, "direct child of the prefab root"
        ):
            extractor.source_projection(changed)

    def test_offset_free_joint_is_rejected(self):
        changed = copy.deepcopy(self.probe_v7)
        joint = next(
            item
            for item in changed["targets"]
            if item.get("class") == "MjFreeJoint"
            and item.get("path_id") == extractor.EXPECTED_FREE_JOINT["path_id"]
        )
        joint["hierarchy"]["transform_chain"][-1]["local_position"]["x"] = 0.001
        joint["hierarchy"]["local_position"]["x"] = 0.001
        with self.assertRaisesRegex(extractor.ExtractionError, "offset from the pelvis"):
            extractor.source_projection(changed)

    def test_mutated_pinned_probe_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "mujoco_asset_probe_v7.json"
            changed.write_bytes(extractor.DEFAULT_PROBE_V7_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                extractor.ExtractionError, "probe_v7 byte count mismatch"
            ):
                extractor.extract_contract(probe_v7_path=changed)

    def test_uncertainty_does_not_claim_a_runtime_clone_observation(self):
        uncertainty = self.contract["uncertainty"]
        self.assertFalse(uncertainty["runtime_clone_observation"])
        self.assertFalse(uncertainty["server_build_equality_observed"])
        self.assertIn("not the prefab clone root", uncertainty["passive_runtime_capture"])


if __name__ == "__main__":
    unittest.main()
