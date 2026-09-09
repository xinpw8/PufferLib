import copy
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from mujoco_arena_facts import (
    EXPECTED_ARENA_SIGNATURE,
    GAME_ASSEMBLY_SHA256,
    NATIVE_METHODS,
    SCHEMA,
    derive_arena_geoms,
    exact_fixed_timestep,
    load_json,
    verify_native_methods,
)


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence_out"
CONTRACT = EVIDENCE / "g1_arena_physics_contract.v1.json"
CONTRACT_SHA256 = "67128d45f8b5995d57b5ca925a2db7b2d613ede15bfa19f8df46c4e01c60e3e8"


class ArenaFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probes = [
            load_json(EVIDENCE / f"arena_level{index}_unity_colliders.json")
            for index in (1, 2, 3)
        ]
        cls.contract = load_json(CONTRACT)

    def test_exact_fixed_timestep_preserves_serialized_rational(self):
        timestep, serialized = exact_fixed_timestep(load_json(EVIDENCE / "static_survey.json"))
        self.assertEqual(timestep.numerator, 2_822_399)
        self.assertEqual(timestep.denominator, 141_120_000)
        self.assertEqual(float(timestep), 0.0199999929138322)
        self.assertEqual(serialized["rate_denominator"], 1)

    def test_all_shipped_levels_derive_the_same_complete_arena(self):
        geoms, metadata = derive_arena_geoms(self.probes)
        self.assertEqual(len(geoms), 17)
        self.assertEqual(Counter(geom["role"] for geom in geoms), {
            "floor": 1,
            "pillar": 8,
            "wall": 8,
        })
        self.assertEqual(
            {probe["geometry_signature_sha256"] for probe in self.probes},
            {EXPECTED_ARENA_SIGNATURE},
        )
        self.assertAlmostEqual(metadata["floor_top_z_m"], 0.009999997913837433)
        required_contact = {
            "priority", "contype", "conaffinity", "group", "condim", "solmix",
            "solref", "solimp", "margin_m", "gap_m", "friction", "fluidshape",
            "fluidcoef",
        }
        for geom in geoms:
            self.assertEqual(set(geom["contact"]), required_contact)
            self.assertEqual(geom["contact"], metadata["shared_contact"])
            self.assertEqual(len(geom["position_m"]), 3)
            self.assertEqual(len(geom["quaternion_wxyz"]), 4)
            self.assertEqual(len(geom["half_extents_m"]), 3)

    def test_cross_level_comparison_detects_tampered_geometry(self):
        probes = copy.deepcopy(self.probes)
        probes[1]["records"][0]["components"][1]["values"]["m_Size"]["x"] += 0.01
        with self.assertRaisesRegex(ValueError, "differs across shipped level"):
            derive_arena_geoms(probes)

    def test_checked_in_contract_is_complete_and_source_pinned(self):
        raw = CONTRACT.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), CONTRACT_SHA256)
        contract = self.contract
        self.assertEqual(contract["schema"], SCHEMA)
        self.assertEqual(contract["arena"]["geom_count"], 17)
        self.assertEqual(len(contract["arena"]["geoms"]), 17)
        self.assertEqual(
            contract["arena"]["probe_geometry_signature_sha256"],
            EXPECTED_ARENA_SIGNATURE,
        )
        self.assertEqual(contract["timestep"]["seconds_exact"]["fraction"], "2822399/141120000")
        self.assertEqual(contract["step_execution"]["mujoco_steps_per_fixed_update"], 1)
        self.assertEqual(contract["step_execution"]["application_level_substeps"], 1)
        self.assertTrue(contract["step_execution"]["mj_step1_and_mj_step2_are_phases_of_one_step"])
        self.assertTrue(contract["g1_plant_boundary"]["separate_scene_geometry_was_omitted"])
        self.assertEqual(contract["g1_plant_boundary"]["composed_ngeom"], 54)
        self.assertTrue(contract["spawn_points"]["native_application"]["proven_for_pinned_client_build"])
        self.assertFalse(contract["spawn_points"]["runtime_observation"])
        for role, slot in (("player", 0), ("opponent", 1)):
            spawn = contract["spawn_points"][role]
            self.assertEqual(spawn["slot"], slot)
            self.assertEqual(spawn["fallback_robot_id"], "g1")
            self.assertEqual(len(spawn["serialized_references"]), 3)
            self.assertEqual({item["container"] for item in spawn["serialized_references"]}, {
                "level1", "level2", "level3",
            })

    def test_checked_in_contract_hashes_its_tracked_inputs(self):
        expected = self.contract["sources"]["input_sha256"]
        paths = {
            "inventory": EVIDENCE / "inventory.json",
            "static_survey": EVIDENCE / "static_survey.json",
            "g1_base_report": EVIDENCE / "g1_29dof.recovered.report.json",
            "g1_base_mjcf": EVIDENCE / "g1_29dof.recovered.xml",
        }
        for key, path in paths.items():
            with self.subTest(key=key):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected[key])
        probe_hashes = [
            hashlib.sha256((EVIDENCE / f"arena_level{index}_unity_colliders.json").read_bytes()).hexdigest()
            for index in (1, 2, 3)
        ]
        self.assertEqual(probe_hashes, expected["arena_probes"])

    def test_native_method_slices_match_when_pinned_build_is_installed(self):
        inventory = load_json(EVIDENCE / "inventory.json")
        game_assembly = Path(inventory["install"]) / "GameAssembly.dll"
        if not game_assembly.is_file():
            self.skipTest("pinned installed GameAssembly.dll is unavailable")
        result = verify_native_methods(game_assembly)
        self.assertEqual(result["game_assembly_sha256"], GAME_ASSEMBLY_SHA256)
        self.assertEqual(set(result["methods"]), set(NATIVE_METHODS))
        self.assertTrue(all(method["verified"] for method in result["methods"].values()))


if __name__ == "__main__":
    unittest.main()
