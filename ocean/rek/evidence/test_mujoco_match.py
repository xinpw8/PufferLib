import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from mujoco_match import (
    EXPECTED_MATCH_DIMENSIONS,
    EXPECTED_T800_BONES,
    EXPECTED_TRACE_SHA256,
    FIGHTER_PREFIXES,
    KEYFRAME_NAME,
    ORIENTATION_FIT_LIMIT_RADIANS,
    POSITION_FIT_LIMIT_METERS,
    SCHEMA,
    prefixed_copy,
    quaternion_distance,
    quaternion_xyzw_to_mujoco_wxyz,
    source_body_name,
    validate_capture,
    vector_to_mujoco,
)


EVIDENCE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EVIDENCE_DIR / "evidence_out" / "t800_t800_factory_arena.diagnostic.xml"
REPORT_PATH = EVIDENCE_DIR / "evidence_out" / "t800_t800_factory_arena.diagnostic.report.json"


def capture_fixture():
    start = {
        "schema": "rek.private_ai.client_fixed.v3",
        "scene": "Arena",
        "game_assembly_sha256": (
            "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412"
        ),
        "global_metadata_sha256": (
            "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd"
        ),
        "authority_semantics": (
            "client_observation_of_remote_authoritative_private_AI_mode"
        ),
        "scope": {
            "allowed": True,
            "network_is_client": True,
            "network_is_server": False,
            "opponent_is_ai": True,
            "opponent_slot_is_ai": True,
            "human_in_opponent_slot": False,
            "opponent_slot_has_client": False,
            "opponent_human_bit_set": False,
            "fighter_0_visual_only": True,
            "fighter_1_visual_only": True,
            "sparring_bot_number": 1,
        },
        "fighter_0_bones": list(EXPECTED_T800_BONES),
        "fighter_1_bones": list(EXPECTED_T800_BONES),
    }
    fighter = {
        "visual_only": True,
        "player_controlled": False,
        "root_position": [0.0, 1.0, 0.0],
        "root_rotation": [0.0, 0.0, 0.0, 1.0],
        "root_linear_velocity": [0.0, 0.0, 0.0],
        "root_angular_velocity": [0.0, 0.0, 0.0],
        "bones": {
            "count": len(EXPECTED_T800_BONES),
            "world_positions_xyz": [0.0] * (3 * len(EXPECTED_T800_BONES)),
            "world_rotations_xyzw": [0.0, 0.0, 0.0, 1.0]
            * len(EXPECTED_T800_BONES),
        },
    }
    sample = {
        "scene": "Arena",
        "sample_index": 0,
        "client_fixed_tick": 0,
        "phase": "RoundActive",
        "round": {"active": True, "number": 2, "time_remaining": 120},
        "fighter_0": json.loads(json.dumps(fighter)),
        "fighter_1": json.loads(json.dumps(fighter)),
    }
    return start, sample


class ConversionTests(unittest.TestCase):
    def test_vector_mapping(self):
        self.assertEqual(vector_to_mujoco((1, 2, 3)), (1.0, 3.0, 2.0))

    def test_measured_quaternion_mapping(self):
        actual = quaternion_xyzw_to_mujoco_wxyz(
            (-0.01071944, 0.034834612, 0.015103462, -0.9992215)
        )
        expected = (
            0.9992214614231879,
            -0.010719439586156,
            0.015103461416902643,
            0.03483461065514475,
        )
        for got, wanted in zip(actual, expected):
            self.assertAlmostEqual(got, wanted, places=15)

    def test_quaternion_distance_is_sign_invariant(self):
        quaternion = (0.5, 0.5, 0.5, 0.5)
        self.assertEqual(quaternion_distance(quaternion, tuple(-x for x in quaternion)), 0.0)

    def test_source_body_name_removes_only_numeric_suffix(self):
        self.assertEqual(source_body_name("LINK_HIP_PITCH_L_3427"), "LINK_HIP_PITCH_L")
        with self.assertRaisesRegex(ValueError, "numeric path-ID"):
            source_body_name("LINK_BASE")


class CaptureValidationTests(unittest.TestCase):
    def test_exact_private_bot1_scope_passes(self):
        start, sample = capture_fixture()
        validate_capture(start, sample)

    def test_non_t800_slot_fails(self):
        start, sample = capture_fixture()
        start["fighter_1_bones"][0] = "pelvis"
        with self.assertRaisesRegex(ValueError, "ordered T800"):
            validate_capture(start, sample)

    def test_non_first_sample_fails(self):
        start, sample = capture_fixture()
        sample["client_fixed_tick"] = 1
        with self.assertRaisesRegex(ValueError, "first captured"):
            validate_capture(start, sample)


class PrefixTests(unittest.TestCase):
    def test_names_and_joint_reference_are_rewritten(self):
        motor = ET.fromstring('<motor name="motor_1" joint="joint_1"/>')
        result = prefixed_copy(motor, "fighter_0__", ["motor_1", "joint_1"])
        self.assertEqual(result.get("name"), "fighter_0__motor_1")
        self.assertEqual(result.get("joint"), "fighter_0__joint_1")

    def test_unresolved_reference_fails(self):
        motor = ET.fromstring('<motor name="motor_1" joint="joint_1"/>')
        with self.assertRaisesRegex(ValueError, "unresolved source reference"):
            prefixed_copy(motor, "fighter_0__", ["motor_1"])


@unittest.skipUnless(MODEL_PATH.exists() and REPORT_PATH.exists(), "generated artifact absent")
class GeneratedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        cls.tree = ET.parse(MODEL_PATH)

    def test_report_claim_boundary_and_source(self):
        self.assertEqual(self.report["schema"], SCHEMA)
        self.assertFalse(self.report["control_equivalent"])
        self.assertFalse(self.report["behavioral_clone"])
        self.assertEqual(self.report["source_trace_sha256"], EXPECTED_TRACE_SHA256)
        self.assertFalse(self.report["claims"]["held_out_parity_established"])
        self.assertFalse(self.report["claims"]["controller_recovered"])

    def test_report_dimensions_and_fit(self):
        validation = self.report["validation"]
        self.assertEqual(validation["model_name"], "rek_t800_t800_arena_diagnostic")
        self.assertEqual(validation["dimensions"], EXPECTED_MATCH_DIMENSIONS)
        self.assertEqual(validation["keyframe_count"], 1)
        self.assertLessEqual(
            validation["initial"]["max_position_error_meters"],
            POSITION_FIT_LIMIT_METERS,
        )
        self.assertLessEqual(
            validation["initial"]["max_orientation_error_radians"],
            ORIENTATION_FIT_LIMIT_RADIANS,
        )
        self.assertEqual(
            validation["initial"]["contact_categories"].get(
                "fighter_0--fighter_1", 0
            ),
            0,
        )
        self.assertTrue(validation["zero_control_final_finite"])

    def test_xml_structure_and_unique_names(self):
        root = self.tree.getroot()
        self.assertEqual(root.get("model"), "rek_t800_t800_arena_diagnostic")
        worldbody = root.find("worldbody")
        self.assertIsNotNone(worldbody)
        self.assertEqual(len([node for node in worldbody if node.tag == "geom"]), 17)
        self.assertEqual(len([node for node in worldbody if node.tag == "body"]), 2)
        self.assertEqual(len(root.findall("actuator/motor")), 50)
        key = root.find("keyframe/key")
        self.assertIsNotNone(key)
        self.assertEqual(key.get("name"), KEYFRAME_NAME)
        self.assertEqual(len(key.get("qpos").split()), 64)
        names = [node.get("name") for node in root.iter() if node.get("name")]
        self.assertEqual(len(names), len(set(names)))
        joints = {node.get("name") for node in root.iter("joint")}
        joints.update(node.get("name") for node in root.iter("freejoint"))
        for motor in root.findall("actuator/motor"):
            self.assertIn(motor.get("joint"), joints)
        for prefix in FIGHTER_PREFIXES:
            self.assertEqual(
                len([motor for motor in root.findall("actuator/motor")
                     if motor.get("name").startswith(prefix)]),
                25,
            )


try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipUnless(
    mujoco is not None and MODEL_PATH.exists(), "MuJoCo or generated artifact absent"
)
class MujocoArtifactTests(unittest.TestCase):
    def test_model_keyframe_and_addresses(self):
        import numpy as np

        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.assertEqual(
            {
                "nbody": int(model.nbody),
                "njnt": int(model.njnt),
                "ngeom": int(model.ngeom),
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
            },
            EXPECTED_MATCH_DIMENSIONS,
        )
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
        self.assertEqual(key_id, 0)
        free = []
        for joint_id in range(model.njnt):
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                free.append((
                    joint_id,
                    int(model.jnt_qposadr[joint_id]),
                    int(model.jnt_dofadr[joint_id]),
                ))
        self.assertEqual(free, [(0, 0, 0), (26, 32, 31)])
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        self.assertTrue(np.isfinite(data.qacc).all())
        self.assertTrue(np.array_equal(data.qvel, np.zeros(model.nv)))
        expected_roots = [
            (-0.98583424, 0.097520486, 0.9382012),
            (0.9858345, -0.0975202, 0.9382012),
        ]
        for joint_id, expected in zip((0, 26), expected_roots):
            body_id = int(model.jnt_bodyid[joint_id])
            for got, wanted in zip(data.xpos[body_id], expected):
                self.assertAlmostEqual(float(got), wanted, places=12)


if __name__ == "__main__":
    unittest.main()
