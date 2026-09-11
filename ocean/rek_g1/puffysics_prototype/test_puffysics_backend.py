"""CPU-only C-ABI packing tests against an explicit private export fixture."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

import puffysics_backend as backend


class PackingTests(unittest.TestCase):
    fixture = None

    def setUp(self):
        self.adapter = deepcopy(self.fixture)

    def test_actual_export_layouts_without_cuda_initialization(self):
        initialized = backend.torch.cuda.is_initialized()
        packed = backend.pack_export(self.adapter)
        for name, shape in (("bodies", (60, 11)), ("shapes", (91, 17)),
                            ("joints", (58, 22)), ("roots", (2, 12))):
            self.assertEqual(packed[name].shape, shape)
            self.assertEqual(packed[name].dtype, np.float32)
            self.assertTrue(packed[name].flags.c_contiguous)
        for item, row in zip(self.adapter["bodies"], packed["bodies"], strict=True):
            np.testing.assert_array_equal(row[:3], np.asarray(item["world_pose"]["position_xyz"], np.float32))
            np.testing.assert_array_equal(row[3:7], np.asarray(item["world_pose"]["quaternion_xyzw"], np.float32))
            self.assertEqual(row[7], np.float32(item["mass_kg"]))
            np.testing.assert_array_equal(row[8:11], np.asarray(item["principal_inertia_kg_m2"], np.float32))
        for item, row in zip(self.adapter["hinges"], packed["joints"], strict=True):
            np.testing.assert_array_equal(row[2:4], [item["qposadr"], item["dofadr"]])
            np.testing.assert_array_equal(row[10:13], np.asarray(item["axis_parent_xyz"], np.float32))
            self.assertEqual(row[13], np.float32(item["mujoco_initial_angle_rad"]))
            self.assertEqual(row[19], np.float32(item["armature_kg_m2"]))
        for item, row in zip(self.adapter["shapes"], packed["shapes"], strict=True):
            self.assertEqual(row[1], backend.SHAPE_ENUM[item["kind"]])
            np.testing.assert_array_equal(row[5:9], np.asarray(item["target_local_pose"]["quaternion_xyzw"], np.float32))
            self.assertEqual(row[12], np.float32(item["friction"][0]))
            np.testing.assert_array_equal(row[13:15], [item["contype"], item["conaffinity"]])
            np.testing.assert_array_equal(row[15:17], [0, 0])
        self.assertEqual(backend.torch.cuda.is_initialized(), initialized)

    def test_signed_free_root_quaternions_match_source_qpos(self):
        packed = backend.pack_export(self.adapter)
        for root, row in zip(self.adapter["free_bases"], packed["roots"], strict=True):
            body_quaternion = packed["bodies"][root["rigid_body_id"], 3:7]
            composed = backend._multiply_xyzw(body_quaternion, row[6:10])
            expected = np.asarray(root["initial_qpos_xyz_wxyz"][3:7])[[1, 2, 3, 0]]
            np.testing.assert_allclose(composed, expected, rtol=0, atol=1e-7)

    def test_hinge_actuator_mismatch_rejected(self):
        self.adapter["actuators"][0]["hinge_id"] = 1
        with self.assertRaisesRegex(ValueError, "actuator/hinge"):
            backend.pack_export(self.adapter)

    def test_unsupported_contact_dimension_rejected(self):
        self.adapter["shapes"][0]["condim"] = 6
        with self.assertRaisesRegex(ValueError, "contact dimension"):
            backend.pack_export(self.adapter)

    def test_nonunit_gear_rejected(self):
        self.adapter["actuators"][0]["gear"][0] = 2
        with self.assertRaises(AssertionError):
            backend.pack_export(self.adapter)

    def test_nonfinite_state_rejected(self):
        self.adapter["bodies"][0]["world_pose"]["position_xyz"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            backend.pack_export(self.adapter)

    def test_positive_gravity_rejected(self):
        self.adapter["gravity_xyz_m_s2"][2] *= -1
        with self.assertRaises(AssertionError):
            backend.pack_export(self.adapter)

    def test_asymmetric_effort_rejected(self):
        self.adapter["actuators"][0]["forcerange"][0] *= 0.5
        with self.assertRaisesRegex(ValueError, "symmetric"):
            backend.pack_export(self.adapter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    PackingTests.fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    unittest.main(argv=[__file__, *remaining])
