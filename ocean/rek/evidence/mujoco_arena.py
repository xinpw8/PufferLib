#!/usr/bin/env python3
"""Compose recovered REK static arena colliders into a recovered plant MJCF."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from mujoco_plant import (
    identity_matrix,
    matrix_multiply,
    mj_quaternion,
    mj_vector,
    number,
    quat,
    set_geom_settings,
    sha256_file,
    trs_matrix,
    vec,
    vector_text,
    world_transform,
)


EXPECTED_COLLIDER_COUNT = 17
EXPECTED_SHAPE_TYPE = 4
GAME_ASSEMBLY_SHA256 = "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412"


def transform_matrix(chain: list[dict[str, Any]]) -> tuple[tuple[float, float, float, float], ...]:
    matrix = identity_matrix()
    for node in chain:
        matrix = matrix_multiply(matrix, trs_matrix(
            vec(node["local_position"]),
            quat(node["local_rotation"]),
            vec(node["local_scale"]),
        ))
    return matrix


def transform_point(matrix: tuple[tuple[float, float, float, float], ...],
                    point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def safe_name(value: str) -> str:
    return "arena_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def arena_geom(record: dict[str, Any]) -> tuple[ET.Element, dict[str, Any]]:
    if not record["game_object_active"]:
        raise ValueError(f"inactive static collider: {record['game_object_path']}")
    if int(record["mj_static_shape_type"]) != EXPECTED_SHAPE_TYPE:
        raise ValueError(f"non-box MjStaticCollider: {record['game_object_path']}")
    if not all(bool(node["active"]) for node in record["transform_chain"]):
        raise ValueError(f"inactive transform ancestor: {record['game_object_path']}")
    colliders = [component for component in record["components"]
                 if component["type"] in {
                     "BoxCollider", "CapsuleCollider", "MeshCollider", "SphereCollider"
                 }]
    if len(colliders) != 1 or colliders[0]["type"] != "BoxCollider":
        raise ValueError(f"expected exactly one BoxCollider: {record['game_object_path']}")
    values = colliders[0]["values"]
    if not bool(values["m_Enabled"]) or bool(values["m_IsTrigger"]):
        raise ValueError(f"disabled or trigger BoxCollider: {record['game_object_path']}")

    chain = record["transform_chain"]
    transform = world_transform(chain)
    if transform.shear > 1e-6:
        raise ValueError(f"unsupported sheared collider transform: {record['game_object_path']}")
    world_center = transform_point(transform_matrix(chain), vec(values["m_Center"]))
    unity_size = vec(values["m_Size"])
    unity_half_extents = tuple(
        abs(scale) * size * 0.5
        for scale, size in zip(transform.lossy_scale, unity_size)
    )

    element = ET.Element("geom", name=safe_name(record["name"]), type="box")
    element.set("pos", vector_text(mj_vector(world_center)))
    element.set("quat", vector_text(mj_quaternion(transform.rotation)))
    element.set("size", vector_text(mj_vector(unity_half_extents)))
    set_geom_settings(element, record["mj_static_settings"])
    derived = {
        "name": element.get("name"),
        "source_game_object_path": record["game_object_path"],
        "source_game_object_path_id": record["game_object_path_id"],
        "source_box_collider_path_id": colliders[0]["path_id"],
        "unity_world_center": world_center,
        "unity_world_rotation_wxyz": transform.rotation,
        "unity_lossy_scale": transform.lossy_scale,
        "unity_box_size": unity_size,
        "mujoco_position": mj_vector(world_center),
        "mujoco_quaternion_wxyz": mj_quaternion(transform.rotation),
        "mujoco_half_extents": mj_vector(unity_half_extents),
        "max_transform_shear": transform.shear,
    }
    return element, derived


def load_matching_probes(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(documents) < 1:
        raise ValueError("at least one arena collider probe is required")
    for document in documents:
        if document.get("schema") != "rek.arena_unity_collider_probe.v1":
            raise ValueError("unsupported arena collider probe schema")
        if int(document["mj_static_collider_count"]) != EXPECTED_COLLIDER_COUNT:
            raise ValueError("unexpected arena collider count")
    build_fingerprints = {document["build_fingerprint"] for document in documents}
    signatures = {document["geometry_signature_sha256"] for document in documents}
    if len(build_fingerprints) != 1:
        raise ValueError("arena probes have different build fingerprints")
    if len(signatures) != 1:
        raise ValueError("arena probes have different physical geometry signatures")
    return documents[0], documents


def find_game_assembly_hash(inventory: dict[str, Any]) -> str:
    matches = [record for record in inventory["files"]
               if str(record["path"]).replace("\\", "/").endswith("/GameAssembly.dll")
               or str(record["path"]).replace("\\", "/") == "GameAssembly.dll"]
    if len(matches) != 1:
        raise ValueError(f"expected one GameAssembly.dll inventory record, got {len(matches)}")
    result = str(matches[0]["sha256"]).lower()
    if result != GAME_ASSEMBLY_SHA256:
        raise ValueError("GameAssembly.dll differs from native-method evidence")
    return result


def compose(base_mjcf: Path, base_report_path: Path, inventory_path: Path,
            probe_paths: list[Path]) -> tuple[ET.ElementTree, dict[str, Any]]:
    arena_probe, all_probes = load_matching_probes(probe_paths)
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if base_report["build_fingerprint"] != arena_probe["build_fingerprint"]:
        raise ValueError("base plant and arena build fingerprints differ")
    if inventory["build_fingerprint"] != arena_probe["build_fingerprint"]:
        raise ValueError("inventory and arena build fingerprints differ")
    if sha256_file(base_mjcf) != base_report["mjcf_sha256"]:
        raise ValueError("base MJCF differs from its generation report")

    tree = ET.parse(base_mjcf)
    root = tree.getroot()
    worldbodies = root.findall("worldbody")
    if len(worldbodies) != 1:
        raise ValueError(f"expected one worldbody, got {len(worldbodies)}")
    worldbody = worldbodies[0]
    derived_records = []
    for index, source_record in enumerate(arena_probe["records"]):
        geom, derived = arena_geom(source_record)
        worldbody.insert(index, geom)
        derived_records.append(derived)
    root.set("model", f"{root.get('model', 'rek')}_arena")

    base_counts = dict(base_report["counts"])
    report = {
        "schema": "rek.mujoco_arena_composition.v1",
        "build_fingerprint": arena_probe["build_fingerprint"],
        "base_mjcf_path": str(base_mjcf.resolve()),
        "base_mjcf_sha256": sha256_file(base_mjcf),
        "base_report_path": str(base_report_path.resolve()),
        "base_report_sha256": sha256_file(base_report_path),
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": sha256_file(inventory_path),
        "arena_probe_paths": [str(path.resolve()) for path in probe_paths],
        "arena_probe_sha256": [sha256_file(path) for path in probe_paths],
        "arena_containers": [document["container"] for document in all_probes],
        "arena_container_sha256": [document["container_sha256"] for document in all_probes],
        "arena_geometry_signature_sha256": arena_probe["geometry_signature_sha256"],
        "arena_geometry_identical_across_probes": True,
        "game_assembly_sha256": find_game_assembly_hash(inventory),
        "native_mapping": {
            "class": "REKApp.MjStaticCollider",
            "arena_defaults_rva": "0x23B3E60",
            "emit_box_rva": "0x23B3F10",
            "emit_world_pose_rva": "0x23B5290",
            "on_generate_mjcf_rva": "0x23B5940",
            "emit_box_inputs": [
                "BoxCollider.size",
                "BoxCollider.center",
                "abs(Transform.lossyScale)",
                "Transform.TransformPoint(center)",
                "Transform.rotation",
            ],
        },
        "arena_geom_count": len(derived_records),
        "composed_counts": {**base_counts, "MjStaticCollider": len(derived_records)},
        "predicted_model_dimensions": {
            **base_report["predicted_model_dimensions"],
            "ngeom": int(base_counts["MjGeom"]) + len(derived_records),
        },
        "derived_arena_geoms": derived_records,
        "control_equivalent": False,
        "limits": [
            "static arena collision geometry is recovered; runtime private-arena scene identity is not measured",
            "all three shipped level containers have the same normalized physical-collider signature",
            "robot spawn poses, second robot, controllers, observations, rewards, damage, score, and round logic are absent",
            "native code establishes collider-to-MJCF geometry inputs; exact server build equivalence remains unknown",
        ],
    }
    return tree, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mjcf", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--arena-probe", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tree, report = compose(
        args.base_mjcf.resolve(),
        args.base_report.resolve(),
        args.inventory.resolve(),
        [path.resolve() for path in args.arena_probe],
    )
    ET.indent(tree, space="  ")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    report["mjcf_path"] = str(args.out.resolve())
    report["mjcf_sha256"] = sha256_file(args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "schema", "build_fingerprint", "arena_containers",
        "arena_geometry_signature_sha256", "arena_geom_count",
        "predicted_model_dimensions", "control_equivalent", "mjcf_sha256",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
