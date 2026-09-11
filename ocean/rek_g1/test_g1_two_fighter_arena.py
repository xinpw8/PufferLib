import copy
import importlib.util
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_two_fighter_arena as duel
import sonic_candidate as plant


PINNED_XML = HERE.parent / "rek" / "evidence" / "evidence_out" / "g1_29dof.recovered.xml"
PINNED_ARENA = (
    HERE.parent
    / "rek"
    / "evidence"
    / "evidence_out"
    / "g1_arena_physics_contract.v1.json"
)
PINNED_SPAWN = (
    HERE.parent
    / "rek"
    / "evidence"
    / "g1_spawn_root_to_pelvis_contract.v1.json"
)
PINNED_RUNTIME_MANIFEST = (
    HERE.parent / "rek" / "evidence" / "g1_runtime_assets.v1.json"
)


class TwoFighterCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = duel.compose_two_fighter_mjcf(
            xml=PINNED_XML,
            arena=PINNED_ARENA,
            spawn_contract=PINNED_SPAWN,
            runtime_manifest=PINNED_RUNTIME_MANIFEST,
        )
        cls.root = ET.fromstring(cls.contract.xml_text)
        cls.xml_contract = plant.inspect_xml_contract(PINNED_XML)
        cls.arena_contract = plant.load_arena_contract(PINNED_ARENA)

    def test_contract_is_build_pinned_and_excludes_unmeasured_behavior(self):
        contract = self.contract
        self.assertEqual(contract.schema, duel.SCHEMA)
        self.assertEqual(
            contract.classification,
            duel.CLASSIFICATION,
        )
        self.assertEqual(contract.build_fingerprint, plant.BUILD_FINGERPRINT)
        self.assertFalse(contract.control_equivalent)
        self.assertEqual(contract.recovered_xml_sha256, plant.EXPECTED_XML_SHA256)
        self.assertEqual(contract.arena_contract_sha256, plant.EXPECTED_ARENA_SHA256)
        self.assertEqual(
            contract.arena_geometry_sha256,
            plant.EXPECTED_ARENA_GEOMETRY_SHA256,
        )
        self.assertEqual(
            contract.runtime_manifest_canonical_sha256,
            plant.EXPECTED_MANIFEST_CANONICAL_SHA256,
        )
        self.assertEqual(contract.spawn_contract_sha256, duel.SPAWN_CONTRACT_SHA256)
        self.assertEqual(
            contract.spawn_evidence_projection_sha256,
            duel.SPAWN_EVIDENCE_PROJECTION_SHA256,
        )
        self.assertEqual(set(contract.excluded_behaviors), set(duel.EXCLUDED_BEHAVIORS))
        self.assertTrue(contract.spawn_rebase_applied)
        self.assertEqual(contract.spawn_rebase_status, duel.SPAWN_REBASE_STATUS)
        self.assertFalse(
            any("transform is absent" in unknown for unknown in contract.unknowns)
        )
        self.assertNotIn(duel.STALE_ARENA_SPAWN_UNKNOWN, contract.unknowns)

    def test_two_namespaced_robot_copies_only_change_spawn_and_contact_masks(self):
        _source_root, source_body, source_motors = duel._source_robot_elements(
            PINNED_XML.read_bytes()
        )
        expected = duel._robot_nonspawn_projection_without_contact_masks_sha256(
            source_body, source_motors, None
        )
        worldbody = self.root.find("worldbody")
        actuator = self.root.find("actuator")
        self.assertIsNotNone(worldbody)
        self.assertIsNotNone(actuator)
        motors = list(actuator)
        self.assertEqual(len(motors), 58)
        for index, role in enumerate(duel.ROLES):
            fighter = self.contract.fighter(role)
            body = next(
                child for child in worldbody if child.get("name") == fighter.root_body_name
            )
            fighter_motors = motors[index * 29 : (index + 1) * 29]
            self.assertEqual(
                duel._robot_nonspawn_projection_without_contact_masks_sha256(
                    body, fighter_motors, fighter.namespace
                ),
                expected,
            )
            rebase = self.contract.spawn_rebase(role)
            self.assertEqual(
                body.get("pos"),
                duel._mjcf_numbers(rebase.pelvis_position_m),
            )
            self.assertEqual(
                body.get("quat"),
                duel._mjcf_numbers(rebase.pelvis_quaternion_wxyz),
            )

    def test_all_names_and_motor_references_are_unique_and_resolved(self):
        named = [element.get("name") for element in self.root.iter() if element.get("name")]
        self.assertEqual(len(named), len(set(named)))
        source_root = ET.fromstring(PINNED_XML.read_bytes())
        source_names = {
            element.get("name") for element in source_root.iter() if element.get("name")
        }
        self.assertFalse(source_names.intersection(named))
        joints = {joint.get("name") for joint in self.root.findall(".//joint")}
        for motor in self.root.findall("./actuator/motor"):
            self.assertIn(motor.get("joint"), joints)
            role = motor.get("name").split(duel.NAMESPACE_SEPARATOR, 1)[0]
            self.assertTrue(motor.get("joint").startswith(f"{role}{duel.NAMESPACE_SEPARATOR}"))

    def test_arena_boxes_remain_exact_contract_xml(self):
        expected_root = ET.fromstring(
            plant.add_arena_geoms(PINNED_XML.read_bytes(), self.arena_contract)
        )
        expected = [
            child
            for child in expected_root.find("worldbody")
            if child.tag == "geom"
        ]
        observed = [
            child for child in self.root.find("worldbody") if child.tag == "geom"
        ]
        self.assertEqual(len(observed), 17)
        self.assertEqual(
            [(element.tag, element.attrib) for element in observed],
            [(element.tag, element.attrib) for element in expected],
        )

    def test_spawn_anchors_are_exact_nonphysical_reference_frames(self):
        worldbody = self.root.find("worldbody")
        for role in duel.ROLES:
            anchor = self.contract.spawn_anchor(role)
            frame = next(
                child
                for child in worldbody
                if child.get("name") == anchor.frame_body_name
            )
            self.assertEqual(frame.tag, "body")
            self.assertEqual(len(list(frame)), 0)
            np.testing.assert_allclose(
                [float(value) for value in frame.get("pos").split()],
                anchor.position_m,
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                [float(value) for value in frame.get("quat").split()],
                anchor.quaternion_wxyz,
                rtol=0.0,
                atol=0.0,
            )
            self.assertFalse(anchor.runtime_observation)

    def test_spawn_rebase_success_assertions_pass(self):
        self.assertIsNone(duel.assert_spawn_rebase_applied(self.contract))
        self.assertIsNone(duel.assert_spawn_rebase_available(self.contract))
        for role in duel.ROLES:
            self.assertEqual(
                self.contract.spawn_rebase(role).free_joint_qpos_prefix,
                duel.EXPECTED_SPAWN_QPOS[role],
            )
        unavailable = replace(self.contract, spawn_rebase_applied=False)
        with self.assertRaisesRegex(
            duel.TwoFighterArenaError,
            "spawn rebase is not applied",
        ):
            duel.assert_spawn_rebase_applied(unavailable)

    def test_spawn_contract_hash_tamper_is_rejected(self):
        raw = PINNED_SPAWN.read_bytes()
        changed = raw.replace(
            duel.SPAWN_CONTRACT_SCHEMA.encode("ascii"),
            duel.SPAWN_CONTRACT_SCHEMA[:-1].encode("ascii") + b"0",
            1,
        )
        self.assertEqual(len(changed), len(raw))
        self.assertNotEqual(changed, raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spawn.json"
            path.write_bytes(changed)
            with self.assertRaisesRegex(
                duel.TwoFighterArenaError,
                "SHA-256 mismatch",
            ):
                duel.compose_two_fighter_mjcf(
                    xml=PINNED_XML,
                    arena=PINNED_ARENA,
                    spawn_contract=path,
                    runtime_manifest=PINNED_RUNTIME_MANIFEST,
                )

    def test_spawn_contract_semantic_identities_fail_closed(self):
        source = json.loads(PINNED_SPAWN.read_text(encoding="utf-8"))
        mutations = (
            ("schema", lambda value: value.__setitem__("schema", "wrong")),
            (
                "build fingerprint",
                lambda value: value.__setitem__("build_fingerprint", "wrong"),
            ),
            (
                "source probe hash",
                lambda value: value["source"]["probe_v7"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "source file identity",
                lambda value: value["source"]["inventory"].__setitem__(
                    "name", "wrong.json"
                ),
            ),
            (
                "component identity",
                lambda value: value["source_identity"]["pelvis_mjbody"].__setitem__(
                    "path_id", 0
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(source)
                mutate(changed)
                with self.assertRaises(duel.TwoFighterArenaError):
                    duel._validate_spawn_contract_value(
                        changed, duel.SPAWN_CONTRACT_SHA256
                    )

    def test_namespacing_rejects_unresolved_motor_joint(self):
        _root, body, motors = duel._source_robot_elements(PINNED_XML.read_bytes())
        body = copy.deepcopy(body)
        motors = tuple(copy.deepcopy(motor) for motor in motors)
        motors[0].set("joint", "missing_joint")
        with self.assertRaisesRegex(duel.TwoFighterArenaError, "unresolved robot name reference"):
            duel._namespace_robot(body, motors, "testfighter")

    def test_namespacing_rejects_unsupported_named_reference(self):
        _root, body, motors = duel._source_robot_elements(PINNED_XML.read_bytes())
        body = copy.deepcopy(body)
        motors = tuple(copy.deepcopy(motor) for motor in motors)
        first_geom = next(body.iter("geom"))
        first_geom.set("material", "unresolved_material")
        with self.assertRaisesRegex(duel.TwoFighterArenaError, "unsupported robot name reference"):
            duel._namespace_robot(body, motors, "testfighter")

    def test_runtime_manifest_tamper_is_rejected(self):
        value = json.loads(PINNED_RUNTIME_MANIFEST.read_text(encoding="utf-8"))
        value["assets"][0]["dof"] = 28
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                duel.TwoFighterArenaError,
                "canonical SHA-256 mismatch",
            ):
                duel.compose_two_fighter_mjcf(
                    xml=PINNED_XML,
                    arena=PINNED_ARENA,
                    spawn_contract=PINNED_SPAWN,
                    runtime_manifest=path,
                )


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "mujoco is unavailable")
class TwoFighterCompiledModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = duel.create_two_fighter_arena_reference(
            xml=PINNED_XML,
            arena=PINNED_ARENA,
            spawn_contract=PINNED_SPAWN,
            runtime_manifest=PINNED_RUNTIME_MANIFEST,
        )
        cls.model = cls.reference.model
        cls.mujoco = cls.reference.mujoco
        cls.xml_contract = plant.inspect_xml_contract(PINNED_XML)
        cls.base_mujoco, cls.base_model, _version = plant.create_mujoco_model(
            cls.xml_contract,
            None,
            None,
        )
        cls.base_map = plant.build_runtime_map(cls.base_model, cls.xml_contract)

    def test_compiled_dimensions_and_disjoint_runtime_maps(self):
        observed = {
            name: int(getattr(self.model, name))
            for name in duel.EXPECTED_MODEL_DIMENSIONS
        }
        self.assertEqual(observed, duel.EXPECTED_MODEL_DIMENSIONS)
        player = self.reference.runtime_map("player")
        opponent = self.reference.runtime_map("opponent")
        for left, right in (
            (player.joint_ids, opponent.joint_ids),
            (player.qpos_addresses, opponent.qpos_addresses),
            (player.qvel_addresses, opponent.qvel_addresses),
            (player.actuator_ids, opponent.actuator_ids),
            (player.body_ids, opponent.body_ids),
        ):
            self.assertFalse(set(left.tolist()).intersection(right.tolist()))
        self.assertNotEqual(player.root_qpos_address, opponent.root_qpos_address)
        self.assertNotEqual(player.root_qvel_address, opponent.root_qvel_address)

    def test_default_free_joint_states_use_exact_spawn_pelvis_poses(self):
        for role in duel.ROLES:
            runtime_map = self.reference.runtime_map(role)
            rebase = self.reference.contract.spawn_rebase(role)
            expected = np.asarray(rebase.free_joint_qpos_prefix)
            np.testing.assert_allclose(
                self.model.qpos0[
                    runtime_map.root_qpos_address : runtime_map.root_qpos_address + 7
                ],
                expected,
                rtol=0.0,
                atol=1e-12,
            )
            anchor = self.reference.contract.spawn_anchor(role)
            self.assertFalse(
                np.allclose(
                    expected[:3],
                    np.asarray(anchor.position_m),
                    rtol=0.0,
                    atol=1e-12,
                )
            )

    def test_forward_kinematics_reads_both_runtime_maps(self):
        data = self.mujoco.MjData(self.model)
        self.mujoco.mj_forward(self.model, data)
        for role in duel.ROLES:
            state = plant.read_robot_state(data, self.reference.runtime_map(role))
            self.assertEqual(state["dof_pos"].shape, (29,))
            self.assertEqual(state["dof_vel"].shape, (29,))
            self.assertTrue(all(np.all(np.isfinite(value)) for value in state.values()))

    def test_each_compiled_robot_preserves_source_physical_arrays(self):
        base = self.base_model
        dual = self.model
        base_geom_names = [
            ET_name
            for ET_name in (
                element.get("name")
                for element in ET.fromstring(PINNED_XML.read_bytes()).findall(".//geom")
            )
            if ET_name is not None
        ]
        for role in duel.ROLES:
            fighter = self.reference.contract.fighter(role)
            runtime_map = self.reference.runtime_map(role)
            rebase = self.reference.contract.spawn_rebase(role)
            np.testing.assert_allclose(
                dual.body_pos[runtime_map.root_body_id],
                rebase.pelvis_position_m,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                dual.body_quat[runtime_map.root_body_id],
                rebase.pelvis_quaternion_wxyz,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                dual.body_pos[runtime_map.body_ids[1:]],
                base.body_pos[self.base_map.body_ids[1:]],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                dual.body_quat[runtime_map.body_ids[1:]],
                base.body_quat[self.base_map.body_ids[1:]],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                dual.body_mass[runtime_map.body_ids],
                base.body_mass[self.base_map.body_ids],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                dual.body_inertia[runtime_map.body_ids],
                base.body_inertia[self.base_map.body_ids],
                rtol=0.0,
                atol=0.0,
            )
            for base_joint_id, dual_joint_id in zip(
                self.base_map.joint_ids, runtime_map.joint_ids
            ):
                for field in ("jnt_pos", "jnt_axis", "jnt_range", "jnt_solref", "jnt_solimp"):
                    np.testing.assert_allclose(
                        getattr(dual, field)[dual_joint_id],
                        getattr(base, field)[base_joint_id],
                        rtol=0.0,
                        atol=0.0,
                    )
                self.assertEqual(dual.jnt_type[dual_joint_id], base.jnt_type[base_joint_id])
                self.assertEqual(dual.jnt_limited[dual_joint_id], base.jnt_limited[base_joint_id])
                base_dof = int(base.jnt_dofadr[base_joint_id])
                dual_dof = int(dual.jnt_dofadr[dual_joint_id])
                for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
                    self.assertEqual(
                        getattr(dual, field)[dual_dof],
                        getattr(base, field)[base_dof],
                    )
            for base_actuator_id, dual_actuator_id in zip(
                self.base_map.actuator_ids, runtime_map.actuator_ids
            ):
                for field in (
                    "actuator_gear",
                    "actuator_ctrlrange",
                    "actuator_forcerange",
                    "actuator_lengthrange",
                    "actuator_gainprm",
                    "actuator_biasprm",
                    "actuator_dynprm",
                ):
                    np.testing.assert_allclose(
                        getattr(dual, field)[dual_actuator_id],
                        getattr(base, field)[base_actuator_id],
                        rtol=0.0,
                        atol=0.0,
                    )
                for field in (
                    "actuator_ctrllimited",
                    "actuator_forcelimited",
                    "actuator_dyntype",
                    "actuator_gaintype",
                    "actuator_biastype",
                ):
                    self.assertEqual(
                        getattr(dual, field)[dual_actuator_id],
                        getattr(base, field)[base_actuator_id],
                    )
            for base_name, dual_name in zip(base_geom_names, fighter.geom_names):
                base_geom = int(base.geom(base_name).id)
                dual_geom = int(dual.geom(dual_name).id)
                for field in (
                    "geom_pos",
                    "geom_quat",
                    "geom_size",
                    "geom_friction",
                    "geom_solref",
                    "geom_solimp",
                ):
                    np.testing.assert_allclose(
                        getattr(dual, field)[dual_geom],
                        getattr(base, field)[base_geom],
                        rtol=0.0,
                        atol=0.0,
                    )
                for field in (
                    "geom_type",
                    "geom_condim",
                    "geom_group",
                    "geom_priority",
                ):
                    self.assertEqual(
                        getattr(dual, field)[dual_geom],
                        getattr(base, field)[base_geom],
                    )
                expected_contype, expected_conaffinity = duel._role_contact_mask(
                    role,
                    int(base.geom_contype[base_geom]),
                    int(base.geom_conaffinity[base_geom]),
                )
                self.assertEqual(
                    int(dual.geom_contype[dual_geom]), expected_contype
                )
                self.assertEqual(
                    int(dual.geom_conaffinity[dual_geom]), expected_conaffinity
                )

    @staticmethod
    def _collision_enabled(model, left, right):
        return bool(
            int(model.geom_contype[left]) & int(model.geom_conaffinity[right])
            or int(model.geom_contype[right]) & int(model.geom_conaffinity[left])
        )

    def test_contact_namespace_preserves_self_matrix_and_enables_every_cross_pair(self):
        base = self.base_model
        dual = self.model
        arena_contract = plant.load_arena_contract(PINNED_ARENA)
        source_names = [
            element.get("name")
            for element in ET.fromstring(PINNED_XML.read_bytes()).findall(".//geom")
        ]
        self.assertNotIn(None, source_names)
        for role in duel.ROLES:
            fighter = self.reference.contract.fighter(role)
            for left_index, left_name in enumerate(source_names):
                base_left = int(base.geom(left_name).id)
                dual_left = int(dual.geom(fighter.geom_names[left_index]).id)
                for right_index, right_name in enumerate(source_names):
                    base_right = int(base.geom(right_name).id)
                    dual_right = int(dual.geom(fighter.geom_names[right_index]).id)
                    self.assertEqual(
                        self._collision_enabled(base, base_left, base_right),
                        self._collision_enabled(dual, dual_left, dual_right),
                    )
                for arena_geom in arena_contract.geoms:
                    arena_id = int(dual.geom(str(arena_geom["name"])).id)
                    expected = bool(
                        int(base.geom_contype[base_left])
                        & int(dual.geom_conaffinity[arena_id])
                        or int(dual.geom_contype[arena_id])
                        & int(base.geom_conaffinity[base_left])
                    )
                    self.assertEqual(
                        expected,
                        self._collision_enabled(dual, dual_left, arena_id),
                        (role, left_name, arena_geom["name"]),
                    )

        player = self.reference.contract.fighter("player")
        opponent = self.reference.contract.fighter("opponent")
        for player_name in player.geom_names:
            player_geom = int(dual.geom(player_name).id)
            for opponent_name in opponent.geom_names:
                opponent_geom = int(dual.geom(opponent_name).id)
                self.assertTrue(
                    self._collision_enabled(dual, player_geom, opponent_geom),
                    (player_name, opponent_name),
                )

    def test_compiled_overlap_produces_real_cross_fighter_contacts(self):
        data = self.mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        player = self.reference.runtime_map("player")
        opponent = self.reference.runtime_map("opponent")
        player_root = player.root_qpos_address
        opponent_root = opponent.root_qpos_address
        data.qpos[opponent_root : opponent_root + 3] = data.qpos[
            player_root : player_root + 3
        ]
        data.qpos[opponent_root] += 0.3
        self.mujoco.mj_forward(self.model, data)
        names = [
            self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            or ""
            for geom_id in range(self.model.ngeom)
        ]
        cross_contacts = []
        for contact in data.contact[: data.ncon]:
            pair = (names[int(contact.geom[0])], names[int(contact.geom[1])])
            owners = {
                "player" if name.startswith("player__") else
                "opponent" if name.startswith("opponent__") else
                "arena"
                for name in pair
            }
            if owners == {"player", "opponent"}:
                cross_contacts.append(pair)
        self.assertTrue(cross_contacts)
        self.assertTrue(
            any(
                "mjgeom_authored" not in left or "mjgeom_authored" not in right
                for left, right in cross_contacts
            )
        )

    def test_every_scoring_striker_target_pair_generates_exact_contact(self):
        model = self.model
        mujoco = self.mujoco

        def body_geoms(role, suffixes):
            body_ids = {
                int(model.body(f"{role}__{suffix}").id)
                for suffix in suffixes
            }
            return [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in body_ids
            ]

        striker_bodies = {
            "hand": ("left_wrist_yaw_link_3467", "right_wrist_yaw_link_3293"),
            "foot": ("left_ankle_roll_link_3045", "right_ankle_roll_link_3090"),
            "shin": ("left_knee_link_3106", "right_knee_link_3429"),
        }
        target_bodies = {
            "pelvis": ("pelvis_3266",),
            "left_hip": (
                "left_hip_pitch_link_3457",
                "left_hip_roll_link_3425",
                "left_hip_yaw_link_2943",
            ),
            "right_hip": (
                "right_hip_pitch_link_3469",
                "right_hip_roll_link_3345",
                "right_hip_yaw_link_3191",
            ),
        }
        tested_pairs = 0
        for attacker_role, target_role in (
            ("player", "opponent"),
            ("opponent", "player"),
        ):
            striker_geoms = {
                part: body_geoms(attacker_role, suffixes)
                for part, suffixes in striker_bodies.items()
            }
            target_geoms = {
                "head": [int(model.geom(f"{target_role}__mjgeom_3064").id)],
                "torso": [int(model.geom(f"{target_role}__mjgeom_3285").id)],
                **{
                    zone: body_geoms(target_role, suffixes)
                    for zone, suffixes in target_bodies.items()
                },
            }
            self.assertEqual(
                {part: len(geoms) for part, geoms in striker_geoms.items()},
                {"hand": 2, "foot": 8, "shin": 2},
            )
            self.assertEqual(
                {zone: len(geoms) for zone, geoms in target_geoms.items()},
                {"head": 1, "torso": 1, "pelvis": 1, "left_hip": 3, "right_hip": 3},
            )
            target_root = self.reference.runtime_map(target_role).root_qpos_address
            for part, part_geoms in striker_geoms.items():
                for zone, zone_geoms in target_geoms.items():
                    for striker_geom in part_geoms:
                        for target_geom in zone_geoms:
                            self.assertTrue(
                                self._collision_enabled(
                                    model, striker_geom, target_geom
                                ),
                                (attacker_role, part, target_role, zone),
                            )
                            data = mujoco.MjData(model)
                            data.qpos[:] = model.qpos0
                            mujoco.mj_forward(model, data)
                            data.qpos[target_root : target_root + 3] += (
                                data.geom_xpos[striker_geom]
                                - data.geom_xpos[target_geom]
                            )
                            mujoco.mj_forward(model, data)
                            exact_pair = {striker_geom, target_geom}
                            self.assertTrue(
                                any(
                                    {
                                        int(contact.geom[0]),
                                        int(contact.geom[1]),
                                    }
                                    == exact_pair
                                    for contact in data.contact[: data.ncon]
                                ),
                                (attacker_role, part, target_role, zone),
                            )
                            tested_pairs += 1
        self.assertEqual(tested_pairs, 216)


if __name__ == "__main__":
    unittest.main()
