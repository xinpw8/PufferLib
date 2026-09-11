#!/usr/bin/env python3
"""Recover the pinned G1 arena physics contract from static REK artifacts.

The output contains only derived numeric facts and source hashes. It does not
copy Unity objects, meshes, textures, managed assemblies, or native code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from mujoco_arena import EXPECTED_COLLIDER_COUNT, arena_geom, load_matching_probes
from mujoco_asset_probe import hierarchy_record, parse_with_string_vector_repair, safe_value
from mujoco_plant import mj_quaternion, mj_vector, sha256_file, world_transform


SCHEMA = "rek.g1_arena_physics_contract.v1"
GAME_ASSEMBLY_SHA256 = "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412"
EXPECTED_BUILD_FINGERPRINT = "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
EXPECTED_ARENA_SIGNATURE = "b95badfd29479bb9f4b6ba62370936ae83c5292183af33ba3d1f1cac7d82d7bc"
LEVEL_RELATIVE_PATHS = {
    "level1": "REK_Data/level1",
    "level2": "REK_Data/level2",
    "level3": "REK_Data/level3",
}

# These lengths and hashes come from the Cpp2IL method boundaries and the
# corresponding bytes in the pinned GameAssembly.dll. Checking both the whole
# file and each slice makes accidental use with another build fail closed.
NATIVE_METHODS = {
    "Mujoco.MjScene.FixedUpdate": {
        "rva": 0x20DDEB0,
        "length": 0x144,
        "sha256": "eda02c2c7df99f2c2ba0f918870b38164682c8f771251624b26a969e202a92ab",
    },
    "Mujoco.MjScene.StepScene": {
        "rva": 0x20E0040,
        "length": 0x25E,
        "sha256": "48c70e8043aa8527a325140efbd60202c221b3e49ccf070335e61a89c0ba2321",
    },
    "REKApp.FightCoordinator.SpawnSlot.MoveNext": {
        "rva": 0x23A2FB0,
        "length": 0x5B3,
        "sha256": "e53c79ed8308cab002c5b348c5f5423887130d189c73b38dd5f2074ec7279596",
    },
    "REKApp.RobotSpawner.SpawnLive.MoveNext": {
        "rva": 0x23A2CE0,
        "length": 0x28E,
        "sha256": "8028333e49b2045204706ca2be9d54f248c27a26b2d9cc293cd8b2d849fff5c7",
    },
    "REKApp.RobotSpawner.TryInstantiate": {
        "rva": 0x239DB40,
        "length": 0x2F5,
        "sha256": "1cbcc27d3d1a2ea44f2f694eb236b0fa55e58765fc64e3e0708726c11b78ef14",
    },
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def inventory_records(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["path"]).replace("\\", "/"): record
        for record in inventory.get("files", ())
    }


def exact_fixed_timestep(static_survey: dict[str, Any]) -> tuple[Fraction, dict[str, int]]:
    values = static_survey["settings"]["TimeManager"]["values"]
    serialized = {
        "count": int(values["Fixed Timestep.m_Count"]),
        "rate_denominator": int(values["Fixed Timestep.m_Rate.m_Denominator"]),
        "rate_numerator": int(values["Fixed Timestep.m_Rate.m_Numerator"]),
    }
    if serialized["rate_numerator"] <= 0 or serialized["rate_denominator"] <= 0:
        raise ValueError("invalid serialized fixed-timestep rational")
    timestep = Fraction(
        serialized["count"] * serialized["rate_denominator"],
        serialized["rate_numerator"],
    )
    return timestep, serialized


def _pe_rva_slice(image: bytes, rva: int, length: int) -> bytes:
    if len(image) < 0x40:
        raise ValueError("native image is too short for a DOS header")
    pe_offset = struct.unpack_from("<I", image, 0x3C)[0]
    if image[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise ValueError("native image has no PE signature")
    section_count = struct.unpack_from("<H", image, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", image, pe_offset + 20)[0]
    section_table = pe_offset + 24 + optional_size
    for index in range(section_count):
        offset = section_table + 40 * index
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", image, offset + 8
        )
        mapped_size = max(virtual_size, raw_size)
        if virtual_address <= rva and rva + length <= virtual_address + mapped_size:
            relative = rva - virtual_address
            if relative + length > raw_size:
                raise ValueError(f"RVA range 0x{rva:X}+0x{length:X} is not file-backed")
            return image[raw_offset + relative:raw_offset + relative + length]
    raise ValueError(f"RVA range 0x{rva:X}+0x{length:X} is outside PE sections")


def verify_native_methods(game_assembly: Path) -> dict[str, Any]:
    image = game_assembly.read_bytes()
    image_sha256 = hashlib.sha256(image).hexdigest()
    if image_sha256 != GAME_ASSEMBLY_SHA256:
        raise ValueError("GameAssembly.dll differs from the pinned native evidence")
    methods = {}
    for name, expected in NATIVE_METHODS.items():
        native_bytes = _pe_rva_slice(image, expected["rva"], expected["length"])
        digest = hashlib.sha256(native_bytes).hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"native method bytes differ: {name}")
        methods[name] = {
            "rva": f"0x{expected['rva']:X}",
            "length_bytes": expected["length"],
            "sha256": digest,
            "verified": True,
        }
    return {"game_assembly_sha256": image_sha256, "methods": methods}


def verify_live_files(inventory: dict[str, Any], game_root: Path) -> dict[str, dict[str, Any]]:
    records = inventory_records(inventory)
    required = [
        "GameAssembly.dll",
        "REK_Data/globalgamemanagers",
        "REK_Data/sharedassets0.assets",
        *LEVEL_RELATIVE_PATHS.values(),
    ]
    result = {}
    for relative in required:
        record = records.get(relative)
        if record is None:
            raise ValueError(f"inventory omits required file: {relative}")
        path = game_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(record["size"]) or digest != str(record["sha256"]).lower():
            raise ValueError(f"installed artifact differs from inventory: {relative}")
        result[relative] = {"size": size, "sha256": digest, "verified": True}
    return result


def _parse_target(obj: Any) -> dict[str, Any]:
    try:
        return obj.parse_as_dict()
    except Exception:
        value, _ = parse_with_string_vector_repair(obj)
        return value


def _transform_fact(environment: Any, reference: dict[str, Any]) -> dict[str, Any]:
    file_id = int(reference["m_FileID"])
    path_id = int(reference["m_PathID"])
    if file_id != 0:
        raise ValueError(f"spawn Transform is an unresolved external reference: {reference}")
    matches = [
        obj for obj in environment.objects
        if obj.path_id == path_id and obj.type.name in {"Transform", "RectTransform"}
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one spawn Transform {path_id}, got {len(matches)}")
    reader = matches[0]
    transform = reader.parse_as_object()
    hierarchy = hierarchy_record(transform.m_GameObject)
    if hierarchy is None:
        raise ValueError(f"could not recover hierarchy for Transform {path_id}")
    if not all(bool(node.get("active", True)) for node in hierarchy["transform_chain"]):
        raise ValueError(f"spawn Transform {path_id} has an inactive ancestor")
    world = world_transform(hierarchy["transform_chain"])
    owner = transform.m_GameObject.deref_parse_as_object()
    return {
        "serialized_reference": {"m_FileID": file_id, "m_PathID": path_id},
        "transform_path_id": path_id,
        "game_object_path_id": transform.m_GameObject.path_id,
        "game_object_path": hierarchy["game_object_path"],
        "game_object_active": bool(getattr(owner, "m_IsActive", True)),
        "unity_world_position_m": list(world.position),
        "unity_world_quaternion_wxyz": list(world.rotation),
        "mujoco_world_position_m": list(mj_vector(world.position)),
        "mujoco_world_quaternion_wxyz": list(mj_quaternion(world.rotation)),
        "max_transform_shear": world.shear,
    }


def extract_level_scene_fact(level_path: Path, dummy_dir: Path) -> dict[str, Any]:
    try:
        import UnityPy
        from UnityPy.helpers.TypeTreeGenerator import TypeTreeGenerator
    except ImportError as exc:
        raise RuntimeError("UnityPy with TypeTreeGeneratorAPI is required") from exc

    environment = UnityPy.load(str(level_path))
    first = next(iter(environment.objects), None)
    if first is None:
        raise ValueError(f"Unity scene contains no objects: {level_path}")
    generator = TypeTreeGenerator(first.assets_file.unity_version)
    generator.load_local_dll_folder(str(dummy_dir))
    environment.typetree_generator = generator

    found: dict[str, list[tuple[Any, Any, dict[str, Any]]]] = {
        "FightCoordinator": [],
        "MjGlobalSettings": [],
    }
    for obj in environment.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            header = obj.parse_monobehaviour_head()
            script = header.m_Script.deref_parse_as_object()
            class_name = str(getattr(script, "m_ClassName", "") or "")
        except Exception:
            continue
        if class_name in found:
            found[class_name].append((obj, header, _parse_target(obj)))

    for class_name, records in found.items():
        if len(records) != 1:
            raise ValueError(f"expected one {class_name} in {level_path.name}, got {len(records)}")
    coordinator_obj, coordinator_head, coordinator = found["FightCoordinator"][0]
    global_obj, global_head, global_settings = found["MjGlobalSettings"][0]
    if not bool(getattr(coordinator_head, "m_Enabled", False)):
        raise ValueError(f"FightCoordinator is disabled in {level_path.name}")
    if not bool(getattr(global_head, "m_Enabled", False)):
        raise ValueError(f"MjGlobalSettings is disabled in {level_path.name}")

    spawns = {}
    for role, field, slot in (
        ("player", "playerSpawnPoint", 0),
        ("opponent", "opponentSpawnPoint", 1),
    ):
        fact = _transform_fact(environment, coordinator[field])
        fact.update({
            "role": role,
            "slot": slot,
            "serialized_field": field,
            "fallback_robot_id": str(coordinator[f"{role}RobotId"]),
        })
        spawns[role] = fact

    return {
        "container": level_path.name,
        "unity_version": first.assets_file.unity_version,
        "fight_coordinator": {
            "path_id": coordinator_obj.path_id,
            "game_object_path_id": coordinator_head.m_GameObject.path_id,
            "serialized_bytes": len(coordinator_obj.get_raw_data()),
            "serialized_sha256": hashlib.sha256(coordinator_obj.get_raw_data()).hexdigest(),
            "spawns": spawns,
        },
        "mj_global_settings": {
            "path_id": global_obj.path_id,
            "game_object_path_id": global_head.m_GameObject.path_id,
            "serialized_bytes": len(global_obj.get_raw_data()),
            "serialized_sha256": hashlib.sha256(global_obj.get_raw_data()).hexdigest(),
            "global_options": safe_value(global_settings["GlobalOptions"]),
            "global_sizes": safe_value(global_settings["GlobalSizes"]),
        },
    }


def extract_scene_facts(game_root: Path, dummy_dir: Path) -> list[dict[str, Any]]:
    return [
        extract_level_scene_fact(game_root / relative, dummy_dir)
        for relative in LEVEL_RELATIVE_PATHS.values()
    ]


def contact_parameters(settings: dict[str, Any]) -> dict[str, Any]:
    filtering = settings["Filtering"]
    solver = settings["Solver"]
    friction = settings["Friction"]
    solimp = solver["SolImp"]
    fluid = settings["FluidCoefficients"]
    return {
        "priority": int(settings["Priority"]),
        "contype": int(filtering["Contype"]),
        "conaffinity": int(filtering["Conaffinity"]),
        "group": int(filtering["Group"]),
        "condim": int(solver["ConDim"]),
        "solmix": float(solver["SolMix"]),
        "solref": [float(solver["SolRef"]["TimeConst"]), float(solver["SolRef"]["DampRatio"])],
        "solimp": [
            float(solimp["DMin"]),
            float(solimp["DMax"]),
            float(solimp["Width"]),
            float(solimp["Midpoint"]),
            float(solimp["Power"]),
        ],
        "margin_m": float(solver["Margin"]),
        "gap_m": float(solver["Gap"]),
        "friction": [
            float(friction["Sliding"]),
            float(friction["Torsional"]),
            float(friction["Rolling"]),
        ],
        "fluidshape": {0: "none", 1: "ellipsoid"}[int(settings["FluidShapeType"])],
        "fluidcoef": [
            float(fluid["BluntDrag"]),
            float(fluid["SlenderDrag"]),
            float(fluid["AngularDrag"]),
            float(fluid["KuttaLift"]),
            float(fluid["MagnusLift"]),
        ],
    }


def derive_arena_geoms(probes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not probes:
        raise ValueError("arena probes are required")
    per_probe = []
    for probe in probes:
        geoms = []
        for record in probe["records"]:
            element, derived = arena_geom(record)
            geom = {
                "name": element.get("name"),
                "role": (
                    "floor" if "Floor" in str(record["name"])
                    else "pillar" if "Pillar" in str(record["name"])
                    else "wall"
                ),
                "type": "box",
                "position_m": list(derived["mujoco_position"]),
                "quaternion_wxyz": list(derived["mujoco_quaternion_wxyz"]),
                "half_extents_m": list(derived["mujoco_half_extents"]),
                "contact": contact_parameters(record["mj_static_settings"]),
                "source": {
                    "game_object_path": record["game_object_path"],
                    "game_object_path_id": record["game_object_path_id"],
                    "mj_static_collider_path_id": record["mj_static_collider_path_id"],
                    "box_collider_path_id": derived["source_box_collider_path_id"],
                    "mj_static_collider_serialized_sha256": record["mj_static_collider_serialized_sha256"],
                },
            }
            geoms.append(geom)
        per_probe.append(geoms)

    def physics_only(geoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value for key, value in geom.items() if key != "source"} for geom in geoms]

    physical = physics_only(per_probe[0])
    if any(physics_only(geoms) != physical for geoms in per_probe[1:]):
        raise ValueError("derived arena physics differs across shipped level containers")
    if len(physical) != EXPECTED_COLLIDER_COUNT:
        raise ValueError(f"expected {EXPECTED_COLLIDER_COUNT} arena geoms, got {len(physical)}")
    roles = Counter(geom["role"] for geom in physical)
    if roles != Counter({"floor": 1, "pillar": 8, "wall": 8}):
        raise ValueError(f"unexpected arena geom roles: {dict(roles)}")
    shared_contact = physical[0]["contact"]
    if any(geom["contact"] != shared_contact for geom in physical):
        raise ValueError("arena static colliders do not share contact parameters")
    floor = next(geom for geom in physical if geom["role"] == "floor")
    floor_top = floor["position_m"][2] + floor["half_extents_m"][2]
    metadata = {
        "geometry_sha256": hashlib.sha256(canonical_json(physical)).hexdigest(),
        "shared_contact": shared_contact,
        "floor_top_z_m": floor_top,
        "role_counts": dict(sorted(roles.items())),
    }
    return per_probe[0], metadata


def normalize_global_options(scene_facts: list[dict[str, Any]]) -> dict[str, Any]:
    values = [fact["mj_global_settings"]["global_options"] for fact in scene_facts]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError("MjGlobalSettings differs across shipped level containers")
    raw = values[0]
    flags = {str(key): int(value) for key, value in raw["Flag"].items()}
    return {
        "integrator": {"serialized": int(raw["Integrator"]), "mujoco": "implicitfast"},
        "cone": {"serialized": int(raw["Cone"]), "mujoco": "elliptic"},
        "jacobian": {"serialized": int(raw["Jacobian"]), "mujoco": "auto"},
        "solver": {"serialized": int(raw["Solver"]), "mujoco": "Newton"},
        "iterations": int(raw["Iterations"]),
        "tolerance": float(raw["Tolerance"]),
        "noslip_iterations": int(raw["NoSlipIterations"]),
        "noslip_tolerance": float(raw["NoSlipTolerance"]),
        "ccd_iterations": int(raw["CcdIterations"]),
        "ccd_tolerance": float(raw["CcdTolerance"]),
        "impratio": float(raw["ImpRatio"]),
        "density": float(raw["Density"]),
        "viscosity": float(raw["Viscosity"]),
        "wind": [float(raw["Wind"][axis]) for axis in ("x", "z", "y")],
        "magnetic": [float(raw["Magnetic"][axis]) for axis in ("x", "z", "y")],
        "override_margin": float(raw["OverrideMargin"]),
        "override_solref": [
            float(raw["OverrideSolRef"]["TimeConst"]),
            float(raw["OverrideSolRef"]["DampRatio"]),
        ],
        "override_solimp": [
            float(raw["OverrideSolImp"]["DMin"]),
            float(raw["OverrideSolImp"]["DMax"]),
            float(raw["OverrideSolImp"]["Width"]),
            float(raw["OverrideSolImp"]["Midpoint"]),
            float(raw["OverrideSolImp"]["Power"]),
        ],
        "flags": {
            "serialized_zero_means_enable": True,
            "raw": flags,
            "enabled": sorted(key.lower() for key, value in flags.items() if value == 0),
            "disabled": sorted(key.lower() for key, value in flags.items() if value != 0),
        },
    }


def normalize_spawn_points(scene_facts: list[dict[str, Any]]) -> dict[str, Any]:
    if not scene_facts:
        raise ValueError("level scene facts are required")
    normalized = {}
    for role in ("player", "opponent"):
        records = [fact["fight_coordinator"]["spawns"][role] for fact in scene_facts]
        pose_keys = (
            "slot",
            "serialized_field",
            "fallback_robot_id",
            "game_object_path",
            "game_object_active",
            "unity_world_position_m",
            "unity_world_quaternion_wxyz",
            "mujoco_world_position_m",
            "mujoco_world_quaternion_wxyz",
        )
        reference_pose = {key: records[0][key] for key in pose_keys}
        if any({key: record[key] for key in pose_keys} != reference_pose for record in records[1:]):
            raise ValueError(f"{role} spawn pose differs across shipped level containers")
        normalized[role] = {
            **reference_pose,
            "serialized_references": [
                {
                    "container": scene["container"],
                    "fight_coordinator_path_id": scene["fight_coordinator"]["path_id"],
                    "transform_path_id": record["transform_path_id"],
                    "game_object_path_id": record["game_object_path_id"],
                    "m_FileID": record["serialized_reference"]["m_FileID"],
                    "m_PathID": record["serialized_reference"]["m_PathID"],
                }
                for scene, record in zip(scene_facts, records)
            ],
        }
    return normalized


def inspect_base_plant(base_mjcf: Path, base_report: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(base_mjcf) != str(base_report["mjcf_sha256"]).lower():
        raise ValueError("G1 base MJCF differs from its recovery report")
    root = ET.parse(base_mjcf).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("G1 base MJCF has no worldbody")
    bodies = worldbody.findall("body")
    if len(bodies) != 1:
        raise ValueError(f"expected one G1 root body, got {len(bodies)}")
    arena_geoms = [geom for geom in root.findall(".//geom") if str(geom.get("name", "")).startswith("arena_")]
    direct_world_geoms = worldbody.findall("geom")
    robot_pairs = Counter(
        (int(geom.get("contype", "1")), int(geom.get("conaffinity", "1")))
        for geom in root.findall(".//geom")
    )
    top = bodies[0]
    return {
        "root_body_name": top.get("name"),
        "serialized_asset_pose_mujoco": {
            "position_m": [float(value) for value in str(top.get("pos", "0 0 0")).split()],
            "quaternion_wxyz": [float(value) for value in str(top.get("quat", "1 0 0 0")).split()],
        },
        "direct_world_geom_count": len(direct_world_geoms),
        "arena_named_geom_count": len(arena_geoms),
        "robot_geom_contact_pair_counts": [
            {"contype": pair[0], "conaffinity": pair[1], "count": count}
            for pair, count in sorted(robot_pairs.items())
        ],
    }


def build_contract(
    *,
    inventory: dict[str, Any],
    static_survey: dict[str, Any],
    base_report: dict[str, Any],
    base_plant: dict[str, Any],
    probes: list[dict[str, Any]],
    scene_facts: list[dict[str, Any]],
    native: dict[str, Any],
    input_hashes: dict[str, Any],
    verified_files: dict[str, Any],
) -> dict[str, Any]:
    build_values = {
        inventory.get("build_fingerprint"),
        static_survey.get("build_fingerprint"),
        base_report.get("build_fingerprint"),
        *(probe.get("build_fingerprint") for probe in probes),
    }
    if build_values != {EXPECTED_BUILD_FINGERPRINT}:
        raise ValueError(f"inputs do not identify the pinned build: {build_values}")
    signatures = {str(probe.get("geometry_signature_sha256")) for probe in probes}
    if signatures != {EXPECTED_ARENA_SIGNATURE}:
        raise ValueError(f"unexpected arena geometry signature: {signatures}")

    timestep, serialized_timestep = exact_fixed_timestep(static_survey)
    if float(base_report["global_configuration"]["timestep"]) != float(timestep):
        raise ValueError("G1 MJCF timestep differs from serialized Unity fixed timestep")
    geoms, arena_metadata = derive_arena_geoms(probes)
    global_options = normalize_global_options(scene_facts)
    spawns = normalize_spawn_points(scene_facts)
    arena_contact = arena_metadata["shared_contact"]
    collision_pairs = base_plant["robot_geom_contact_pair_counts"]
    if not all(
        (pair["contype"] & arena_contact["conaffinity"])
        or (arena_contact["contype"] & pair["conaffinity"])
        for pair in collision_pairs
    ):
        raise ValueError("at least one G1 geom contact mask is incompatible with the arena")

    return {
        "schema": SCHEMA,
        "build_fingerprint": EXPECTED_BUILD_FINGERPRINT,
        "scope": "static client-build G1 MuJoCo arena contract",
        "sources": {
            "input_sha256": input_hashes,
            "installed_files": verified_files,
            "arena_containers": [
                {
                    "name": probe["container"],
                    "sha256": probe["container_sha256"],
                    "probe_geometry_signature_sha256": probe["geometry_signature_sha256"],
                }
                for probe in probes
            ],
            "native": native,
        },
        "coordinate_mapping": {
            "unity_position_xyz_to_mujoco_xyz": ["x", "z", "y"],
            "unity_quaternion_wxyz_to_mujoco_wxyz": ["-w", "x", "z", "y"],
            "units": "metres",
            "provenance": base_report["unity_mapping_source"],
        },
        "timestep": {
            "seconds_exact": {
                "numerator": timestep.numerator,
                "denominator": timestep.denominator,
                "fraction": f"{timestep.numerator}/{timestep.denominator}",
            },
            "seconds_float": float(timestep),
            "frequency_hz_float": float(1 / timestep),
            "unity_serialized": serialized_timestep,
            "g1_mjcf_option_matches": True,
        },
        "step_execution": {
            "unity_callback": "Mujoco.MjScene.FixedUpdate",
            "step_scene_calls_per_fixed_update": 1,
            "mujoco_steps_per_fixed_update": 1,
            "application_level_substeps": 1,
            "without_control_callback": ["mj_step"],
            "with_control_callback": ["mj_step1", "control callback", "mj_step2"],
            "mj_step1_and_mj_step2_are_phases_of_one_step": True,
            "post_step": ["CheckForPhysicsException", "SyncUnityToMjState"],
        },
        "global_options": global_options,
        "spawn_points": {
            "evidence_kind": "serialized scene references with build-pinned native application path",
            "runtime_observation": False,
            "native_application": {
                "proven_for_pinned_client_build": True,
                "slot_0_field": "FightCoordinator.playerSpawnPoint at native field offset 0x1A0",
                "slot_1_field": "FightCoordinator.opponentSpawnPoint at native field offset 0x1B0",
                "flow": [
                    "FightCoordinator.SpawnSlot selects the Transform by slot",
                    "RobotSpawner.SpawnLive passes the selected Transform to TryInstantiate",
                    "TryInstantiate reads Transform.position and Transform.rotation",
                    "TryInstantiate passes both values to UnityEngine.Object.Instantiate",
                ],
            },
            "identical_pose_across_level_containers": True,
            **spawns,
        },
        "arena": {
            "source_hierarchy_root": "Enviro_StoreOctagon/Colliders",
            "source_component": "REKApp.MjStaticCollider",
            "shape_type_serialized": 4,
            "shape": "box",
            "geom_count": len(geoms),
            "role_counts": arena_metadata["role_counts"],
            "geometry_identical_across_level_containers": True,
            "probe_geometry_signature_sha256": EXPECTED_ARENA_SIGNATURE,
            "derived_geometry_sha256": arena_metadata["geometry_sha256"],
            "floor_top_z_m": arena_metadata["floor_top_z_m"],
            "shared_contact": arena_contact,
            "geoms": geoms,
        },
        "g1_plant_boundary": {
            **base_plant,
            "recovery_hierarchy_root": base_report["root_name"],
            "recovered_counts": base_report["counts"],
            "separate_scene_geometry_was_omitted": (
                base_plant["direct_world_geom_count"] == 0
                and base_plant["arena_named_geom_count"] == 0
            ),
            "reason": (
                "mujoco_plant selects the g1_29dof_Prefab_SONIC hierarchy; "
                "the arena colliders live under a separate level-scene hierarchy"
            ),
            "spawn_rebase_required": True,
            "serialized_asset_pose_is_not_an_arena_spawn_pose": True,
            "composed_ngeom": int(base_report["counts"]["MjGeom"]) + len(geoms),
        },
        "contact_mask_check": {
            "arena": {"contype": arena_contact["contype"], "conaffinity": arena_contact["conaffinity"]},
            "g1_geom_pairs": collision_pairs,
            "all_g1_pairs_can_contact_arena": True,
        },
        "unknowns": [
            "server-side build equality and server-authoritative physics were not measured",
            "a runtime observation of the instantiated Transform was not made",
            "the serialized spawn references establish arena anchors; this contract does not infer a replacement free-joint pelvis height",
            "game controller, observations, rewards, damage, score, round logic, and opponent policy remain outside this contract",
        ],
        "control_equivalent": False,
    }


def generate(
    *,
    inventory_path: Path,
    static_survey_path: Path,
    base_report_path: Path,
    base_mjcf_path: Path,
    probe_paths: list[Path],
    dummy_dir: Path,
    game_root: Path | None = None,
) -> dict[str, Any]:
    inventory = load_json(inventory_path)
    static_survey = load_json(static_survey_path)
    base_report = load_json(base_report_path)
    _, probes = load_matching_probes(probe_paths)
    if len(probes) != 3 or {probe["container"] for probe in probes} != set(LEVEL_RELATIVE_PATHS):
        raise ValueError("exactly the level1, level2, and level3 arena probes are required")
    root = game_root or Path(str(inventory["install"]))
    verified_files = verify_live_files(inventory, root)
    for probe in probes:
        relative = LEVEL_RELATIVE_PATHS[probe["container"]]
        expected = verified_files[relative]["sha256"]
        if probe["container_sha256"] != expected or probe["container_inventory_sha256"] != expected:
            raise ValueError(f"arena probe is not tied to installed {probe['container']}")
    native = verify_native_methods(root / "GameAssembly.dll")
    scene_facts = extract_scene_facts(root, dummy_dir)
    base_plant = inspect_base_plant(base_mjcf_path, base_report)
    input_hashes = {
        "inventory": sha256_file(inventory_path),
        "static_survey": sha256_file(static_survey_path),
        "g1_base_report": sha256_file(base_report_path),
        "g1_base_mjcf": sha256_file(base_mjcf_path),
        "arena_probes": [sha256_file(path) for path in probe_paths],
    }
    return build_contract(
        inventory=inventory,
        static_survey=static_survey,
        base_report=base_report,
        base_plant=base_plant,
        probes=probes,
        scene_facts=scene_facts,
        native=native,
        input_hashes=input_hashes,
        verified_files=verified_files,
    )


def main() -> int:
    evidence_dir = Path(__file__).resolve().parent / "evidence_out"
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=evidence_dir / "inventory.json")
    parser.add_argument("--static-survey", type=Path, default=evidence_dir / "static_survey.json")
    parser.add_argument("--base-report", type=Path, default=evidence_dir / "g1_29dof.recovered.report.json")
    parser.add_argument("--base-mjcf", type=Path, default=evidence_dir / "g1_29dof.recovered.xml")
    parser.add_argument("--arena-probe", type=Path, action="append")
    parser.add_argument("--dummy-dir", type=Path, required=True)
    parser.add_argument("--python-deps", type=Path, action="append", default=[])
    parser.add_argument("--game-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in reversed(args.python_deps):
        sys.path.insert(0, str(path.resolve()))
    probe_paths = args.arena_probe or [
        evidence_dir / f"arena_{level}_unity_colliders.json"
        for level in LEVEL_RELATIVE_PATHS
    ]
    contract = generate(
        inventory_path=args.inventory.resolve(),
        static_survey_path=args.static_survey.resolve(),
        base_report_path=args.base_report.resolve(),
        base_mjcf_path=args.base_mjcf.resolve(),
        probe_paths=[path.resolve() for path in probe_paths],
        dummy_dir=args.dummy_dir.resolve(),
        game_root=args.game_root.resolve() if args.game_root else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(contract, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({
        "schema": contract["schema"],
        "build_fingerprint": contract["build_fingerprint"],
        "geom_count": contract["arena"]["geom_count"],
        "derived_geometry_sha256": contract["arena"]["derived_geometry_sha256"],
        "timestep_fraction": contract["timestep"]["seconds_exact"]["fraction"],
        "out": str(args.out.resolve()),
        "sha256": sha256_file(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
