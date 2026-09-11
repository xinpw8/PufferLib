import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sonic_candidate as candidate


PINNED_XML = HERE.parent / "rek" / "evidence" / "evidence_out" / "g1_29dof.recovered.xml"
PINNED_ARENA = (
    HERE.parent
    / "rek"
    / "evidence"
    / "evidence_out"
    / "g1_arena_physics_contract.v1.json"
)


def policy_metadata():
    inputs = [
        {
            "name": name,
            "shape": [1 if dim == "batch_size" else dim for dim in shape],
            "key": key,
        }
        for name, shape, key in candidate.INPUT_CONTRACT
    ]
    outputs = [
        {
            "name": name,
            "shape": [1 if dim == "batch_size" else dim for dim in shape],
        }
        for name, shape in candidate.OUTPUT_CONTRACT
    ]
    return {
        "type": "unified_pipeline",
        "dt": 0.02,
        "joint_names": list(candidate.JOINT_NAMES),
        "policy_inputs": inputs,
        "policy_outputs": outputs,
        "_runtime": {
            "onnx_in_names": [entry[0] for entry in candidate.INPUT_CONTRACT],
            "onnx_out_names": [entry[0] for entry in candidate.OUTPUT_CONTRACT],
            "onnx_name_to_in_key": {
                name: key for name, _shape, key in candidate.INPUT_CONTRACT
            },
        },
        "robot": {
            "num_dofs": 29,
            "joint_names": list(candidate.JOINT_NAMES),
            "anchor_body_name": "torso_link",
            "root_body_name": "pelvis",
        },
        "timing": {"control_dt": 0.02, "physics_dt": 0.001, "decimation": 20},
        "motion": {
            "future_step_indices": [1, 2, 4, 8],
            "future_dt_seconds": [0.02, 0.04, 0.08, 0.16],
        },
        "metadata": {"control_type": "BUILT_IN_PD"},
        "control": {
            "stiffness": [2.0] * 29,
            "damping": [0.5] * 29,
            "pd_target_max_accel": None,
            "action_ema_alpha": 1.0,
        },
    }


def policy_contract():
    metadata = policy_metadata()
    return candidate.PolicyContract(
        joint_names=candidate.JOINT_NAMES,
        stiffness=np.full(29, 2.0, dtype=np.float32),
        damping=np.full(29, 0.5, dtype=np.float32),
        anchor_body_name="torso_link",
        root_body_name="pelvis",
        control_dt=0.02,
        physics_dt=0.001,
        decimation=20,
        metadata=metadata,
        yaml_size=10,
        yaml_sha256="0" * 64,
    )


def motion_data(frames=12):
    dof_pos = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 100.0
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    return candidate.MotionData(
        role="idle",
        filename="idle_processed.npz",
        size=1,
        sha256="1" * 64,
        fps=50.0,
        dof_pos=dof_pos,
        dof_vel=candidate.compute_dof_velocities(dof_pos, 0.02),
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_rot_xyzw=root_rot,
        manifest_sha256="2" * 64,
        inventory_sha256="3" * 64,
    )


class FakeNode:
    def __init__(self, name, shape, type_name="tensor(float)"):
        self.name = name
        self.shape = list(shape)
        self.type = type_name


class FakeSession:
    def __init__(self):
        self.inputs = [FakeNode(name, shape) for name, shape, _key in candidate.INPUT_CONTRACT]
        self.outputs = [FakeNode(name, shape) for name, shape in candidate.OUTPUT_CONTRACT]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]


class SonicCandidateTests(unittest.TestCase):
    def test_policy_metadata_contract_accepts_pinned_schema(self):
        candidate.validate_policy_metadata(policy_metadata())

    def test_policy_metadata_rejects_future_offset_change(self):
        metadata = policy_metadata()
        metadata["motion"]["future_step_indices"] = [1, 2, 3, 8]
        with self.assertRaisesRegex(candidate.CandidateError, "future_step_indices"):
            candidate.validate_policy_metadata(metadata)

    def test_onnx_schema_is_exact_and_cpu_only(self):
        session = FakeSession()
        candidate.validate_onnx_session_schema(session)
        session.inputs[0].shape = ["batch_size", 5]
        with self.assertRaisesRegex(candidate.CandidateError, "input schema mismatch"):
            candidate.validate_onnx_session_schema(session)

    def test_recovered_xml_maps_policy_order_by_name(self):
        contract = candidate.inspect_xml_contract(PINNED_XML)
        self.assertEqual([joint.canonical_name for joint in contract.joints], list(candidate.JOINT_NAMES))
        self.assertEqual(contract.root_body_xml_name, "pelvis_3266")
        self.assertEqual(contract.anchor_body_xml_name, "torso_link_3347")
        self.assertEqual(contract.free_joint_name, "joint__floating_base_joint_3081")
        self.assertAlmostEqual(contract.source_timestep, 0.02, places=7)
        self.assertFalse(contract.source_has_floor)
        self.assertEqual(contract.joints[0].actuator_name, "left_hip_pitch_3206")
        self.assertEqual(
            (contract.joints[0].ctrl_min, contract.joints[0].ctrl_max), (-88.0, 88.0)
        )
        xml = ET.fromstring(PINNED_XML.read_bytes())
        self.assertEqual(
            tuple(
                candidate._canonical_body_name(body.get("name"))
                for body in xml.findall(".//body")
            ),
            candidate.BODY_NAMES,
        )

    def test_diagnostic_floor_is_explicit_and_does_not_mutate_source(self):
        source = PINNED_XML.read_bytes()
        patched = candidate.add_diagnostic_floor(source, -0.25)
        root = ET.fromstring(patched)
        floors = [
            geom
            for geom in root.findall(".//geom")
            if geom.get("name") == "sonic_candidate_diagnostic_floor"
        ]
        self.assertEqual(len(floors), 1)
        self.assertEqual(floors[0].get("pos"), "0 0 -0.25")
        self.assertEqual(candidate.sha256_bytes(source), candidate.EXPECTED_XML_SHA256)

    def test_arena_contract_hash_schema_boxes_and_timestep(self):
        contract = candidate.load_arena_contract(PINNED_ARENA)
        self.assertEqual(contract.source_sha256, candidate.EXPECTED_ARENA_SHA256)
        self.assertEqual(contract.derived_geometry_sha256, candidate.EXPECTED_ARENA_GEOMETRY_SHA256)
        self.assertEqual(len(contract.geoms), 17)
        self.assertEqual([geom["role"] for geom in contract.geoms].count("floor"), 1)
        self.assertEqual([geom["role"] for geom in contract.geoms].count("pillar"), 8)
        self.assertEqual([geom["role"] for geom in contract.geoms].count("wall"), 8)
        self.assertTrue(all(geom["type"] == "box" for geom in contract.geoms))
        self.assertEqual(contract.composed_ngeom, 54)
        self.assertEqual(contract.source_timestep, 2822399 / 141120000)

    def test_arena_contract_injects_all_exact_box_and_contact_attributes(self):
        contract = candidate.load_arena_contract(PINNED_ARENA)
        patched = candidate.add_arena_geoms(PINNED_XML.read_bytes(), contract)
        root = ET.fromstring(patched)
        worldbody = root.find("worldbody")
        self.assertIsNotNone(worldbody)
        direct_geoms = [child for child in list(worldbody) if child.tag == "geom"]
        self.assertEqual(len(direct_geoms), 17)
        self.assertEqual([geom.get("name") for geom in direct_geoms], [geom["name"] for geom in contract.geoms])
        floor_xml = next(geom for geom in direct_geoms if geom.get("name") == "arena_Collider_Floor_Rektagon")
        floor_contract = next(geom for geom in contract.geoms if geom["role"] == "floor")
        self.assertEqual(floor_xml.get("type"), "box")
        np.testing.assert_allclose(
            [float(value) for value in floor_xml.get("pos").split()],
            floor_contract["position_m"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            [float(value) for value in floor_xml.get("size").split()],
            floor_contract["half_extents_m"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            [float(value) for value in floor_xml.get("quat").split()],
            floor_contract["quaternion_wxyz"],
            rtol=0.0,
            atol=0.0,
        )
        contact = floor_contract["contact"]
        for key in ("priority", "contype", "conaffinity", "group", "condim"):
            self.assertEqual(int(floor_xml.get(key)), contact[key])
        for xml_key, contract_key in (
            ("solref", "solref"),
            ("solimp", "solimp"),
            ("friction", "friction"),
            ("fluidcoef", "fluidcoef"),
        ):
            np.testing.assert_allclose(
                [float(value) for value in floor_xml.get(xml_key).split()],
                contact[contract_key],
                rtol=0.0,
                atol=0.0,
            )
        self.assertEqual(floor_xml.get("fluidshape"), "none")

    def test_arena_contract_rejects_non_pinned_bytes(self):
        raw = PINNED_ARENA.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena.json"
            path.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(candidate.CandidateError, "SHA-256 mismatch"):
                candidate.load_arena_contract(path)

    def test_arena_schema_validator_rejects_wrong_schema(self):
        value = json.loads(PINNED_ARENA.read_text(encoding="utf-8"))
        value["schema"] = "wrong"
        with self.assertRaisesRegex(candidate.CandidateError, "schema mismatch"):
            candidate._validate_arena_contract_schema(value, 1, "0" * 64)

    def test_native_forward_joint_velocity_derivation(self):
        positions = np.zeros((4, 29), dtype=np.float32)
        positions[:, 0] = [0.0, 1.0, 4.0, 9.0]
        velocity = candidate.compute_dof_velocities(positions, 1.0)
        np.testing.assert_array_equal(velocity[:, 0], [1.0, 3.0, 5.0, 0.0])
        self.assertEqual(velocity.dtype, np.float32)

    def test_motion_schema_requires_float32_unit_quaternions(self):
        frames = 9
        fps = np.array([50.0], dtype=np.float32)
        dof = np.zeros((frames, 29), dtype=np.float32)
        pos = np.zeros((frames, 3), dtype=np.float32)
        rot = np.zeros((frames, 4), dtype=np.float32)
        rot[:, 3] = 1.0
        candidate.validate_motion_arrays(
            fps=fps,
            dof_pos=dof,
            root_pos=pos,
            root_rot=rot,
            expected_frames=frames,
        )
        rot[4, 3] = 0.5
        with self.assertRaisesRegex(candidate.CandidateError, "unit normalized"):
            candidate.validate_motion_arrays(
                fps=fps,
                dof_pos=dof,
                root_pos=pos,
                root_rot=rot,
                expected_frames=frames,
            )

    def test_future_references_use_pinned_offsets(self):
        motion = motion_data()
        anchors = motion.root_rot_xyzw.copy()
        state = {
            "dof_pos": np.zeros(29, dtype=np.float32),
            "dof_vel": np.zeros(29, dtype=np.float32),
            "anchor_rot": np.array([0, 0, 0, 1], dtype=np.float32),
            "root_local_ang_vel": np.zeros(3, dtype=np.float32),
        }
        previous = np.ones(29, dtype=np.float32)
        feeds, indices = candidate.build_onnx_inputs(
            state, motion, anchors, current_frame=2, previous_actions=previous
        )
        np.testing.assert_array_equal(indices, [3, 4, 6, 10])
        np.testing.assert_array_equal(
            feeds["mimic_future_dof_pos"][0], motion.dof_pos[indices]
        )
        self.assertEqual(feeds["historical_processed_actions"].shape, (1, 1, 29))

    def test_quaternion_boundary_and_reference_heading_alignment_match_upstream(self):
        wxyz = np.array([0.5, 0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(
            candidate._wxyz_to_xyzw(wxyz),
            np.array([0.1, 0.2, 0.3, 0.5], dtype=np.float32),
        )
        np.testing.assert_array_equal(candidate._xyzw_to_wxyz(candidate._wxyz_to_xyzw(wxyz)), wxyz)
        yaw_90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)
        yaw_180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        references = np.stack([yaw_90, yaw_90])
        aligned, offset = candidate.align_reference_anchor_rotations(yaw_180, references)
        np.testing.assert_allclose(offset, yaw_90, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(np.abs(aligned[:, 2]), 1.0, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(aligned[:, 3], 0.0, rtol=0.0, atol=1e-6)

    def test_pd_torque_is_clipped_to_recovered_declared_range(self):
        zeros = np.zeros(29)
        targets = np.full(29, 10.0)
        stiffness = np.full(29, 20.0)
        damping = np.full(29, 1.0)
        lower = np.full(29, -50.0)
        upper = np.full(29, 50.0)
        raw, applied, saturated = candidate.compute_pd_torque(
            zeros, zeros, targets, stiffness, damping, lower, upper
        )
        np.testing.assert_array_equal(raw, np.full(29, 200.0))
        np.testing.assert_array_equal(applied, np.full(29, 50.0))
        self.assertTrue(np.all(saturated))

    def test_simulation_rejects_missing_explicit_floor(self):
        with self.assertRaisesRegex(candidate.CandidateError, "exactly one"):
            candidate.run_candidate(
                session=None,
                policy=policy_contract(),
                xml_contract=None,
                motion=motion_data(),
                onnx_digest="0" * 64,
                onnx_size=1,
                onnxruntime_version="test",
                diagnostic_floor_z=None,
                arena_contract=None,
                requested_steps=1,
                segment=candidate.SegmentRequest(
                    source="test",
                    motion_role="idle",
                    duration_ticks=1,
                    semantic_command=None,
                    source_sha256=None,
                ),
            )

    def test_policy_outputs_require_metadata_gains(self):
        policy = policy_contract()
        outputs = [
            np.zeros((1, 29), dtype=np.float32),
            np.zeros((1, 29), dtype=np.float32),
            np.full((1, 29), 2.0, dtype=np.float32),
            np.full((1, 29), 0.5, dtype=np.float32),
        ]
        candidate.validate_policy_outputs(outputs, policy)
        outputs[2][0, 7] = 3.0
        with self.assertRaisesRegex(candidate.CandidateError, "stiffness"):
            candidate.validate_policy_outputs(outputs, policy)

    def test_semantic_segment_envelope_matches_c_scheduler_fields(self):
        envelope = {
            "schema": candidate.SEGMENT_SCHEMA,
            "classification": candidate.CLASSIFICATION,
            "motion_role": "kick_left_front",
            "duration_ticks": 12,
            "semantic_command": {
                "kind": 1,
                "held_code": 9,
                "duration_ticks": 12,
                "kick_registry_index": 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segment.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            segment = candidate.load_segment_request(path)
        self.assertEqual(segment.motion_role, "kick_left_front")
        self.assertEqual(segment.duration_ticks, 12)
        self.assertEqual(segment.semantic_command["held_code"], 9)
        self.assertIsNotNone(segment.source_sha256)

    def test_semantic_kick_segment_rejects_translation(self):
        envelope = {
            "schema": candidate.SEGMENT_SCHEMA,
            "classification": candidate.CLASSIFICATION,
            "motion_role": "kick_left_front",
            "duration_ticks": 12,
            "semantic_command": {
                "kind": 1,
                "held_code": 1,
                "duration_ticks": 12,
                "kick_registry_index": 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segment.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(candidate.CandidateError, "cannot contain translation"):
                candidate.load_segment_request(path)

    def test_output_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            candidate._write_new_text(path, "first\n", "metrics")
            with self.assertRaisesRegex(candidate.CandidateError, "refusing to overwrite"):
                candidate._write_new_text(path, "second\n", "metrics")

    def test_classification_cannot_claim_rek_parity(self):
        self.assertEqual(candidate.CLASSIFICATION, "diagnostic_candidate")
        self.assertIn("No authoritative REK", candidate.LIMITS[-1])


if __name__ == "__main__":
    unittest.main()
