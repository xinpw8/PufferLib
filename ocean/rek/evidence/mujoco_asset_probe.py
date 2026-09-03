#!/usr/bin/env python3
"""Inventory serialized MuJoCo/robot MonoBehaviours from one pinned REK build.

The probe never exports Unity objects or binary payloads. It resolves each
MonoBehaviour through its header, hashes the source object bytes, and records
only bounded JSON-compatible serialized values for an allowlisted class set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REK_TARGET_CLASSES = {
    "AIOpponentController",
    "EngineAIPolicyRunner",
    "KeyboardControlScheme",
    "MjImpactForce",
    "MjSimConfig",
    "MjStaticCollider",
    "MujocoPolicyRunner",
    "MocapClipConfig",
    "PhysXStepPolicy",
    "Robot",
    "RobotCatalog",
    "RobotConfig",
    "RobotImportData",
    "RobotJointDiagnostics",
    "RobotMjSync",
    "SonicMotionComposer",
    "SonicPolicyRunner",
}

SENSITIVE_KEYS = (
    "access_token",
    "authorization",
    "connect_ticket",
    "cookie",
    "password",
    "private_key",
    "recovery_code",
    "refresh_token",
    "secret",
    "session_token",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_target(namespace: str, class_name: str) -> bool:
    return namespace == "Mujoco" or class_name in REK_TARGET_CLASSES


def repair_string_vector_nodes(node: Any) -> int:
    """Repair a TypeTreeGeneratorAPI bug that labels string[] as string."""
    repaired = 0
    children = getattr(node, "m_Children", ())
    if (
        getattr(node, "m_Type", None) == "string"
        and children
        and getattr(children[0], "m_Type", None) == "Array"
        and len(getattr(children[0], "m_Children", ())) > 1
    ):
        element = children[0].m_Children[1]
        if getattr(element, "m_Type", None) == "string" and getattr(element, "m_Children", ()):
            node.m_Type = "vector"
            repaired += 1
    for child in children:
        repaired += repair_string_vector_nodes(child)
    return repaired


def parse_with_string_vector_repair(obj: Any) -> tuple[dict[str, Any], int]:
    from UnityPy.helpers import TypeTreeHelper

    node = obj._get_typetree_node()
    repaired = repair_string_vector_nodes(node)
    boost_reader = TypeTreeHelper.read_typetree_boost
    try:
        TypeTreeHelper.read_typetree_boost = None
        return obj.parse_as_dict(node=node, check_read=True), repaired
    finally:
        TypeTreeHelper.read_typetree_boost = boost_reader


def safe_value(value: Any, key: str = "", depth: int = 0) -> Any:
    if any(token in key.lower() for token in SENSITIVE_KEYS):
        return "<redacted>"
    if depth > 8:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 512 else value[:512] + "<truncated>"
    if isinstance(value, bytes):
        return {"byte_count": len(value), "sha256": sha256_bytes(value)}
    if isinstance(value, dict):
        return {
            str(child_key): safe_value(child_value, str(child_key), depth + 1)
            for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        limit = 512
        result = [safe_value(child, key, depth + 1) for child in value[:limit]]
        if len(value) > limit:
            result.append({"truncated_items": len(value) - limit})
        return result
    file_id = getattr(value, "m_FileID", None)
    path_id = getattr(value, "m_PathID", None)
    if isinstance(file_id, int) and isinstance(path_id, int):
        return {"m_FileID": file_id, "m_PathID": path_id}
    return {"unserialized_type": type(value).__name__}


def load_inventory(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != 1:
        raise ValueError("unsupported inventory schema")
    files = document.get("files")
    if not isinstance(files, list):
        raise ValueError("inventory files missing")
    by_path = {str(record["path"]).replace("\\", "/"): record for record in files}
    return document, by_path


def container_paths(game_root: Path, inventory_files: dict[str, dict[str, Any]]) -> list[Path]:
    result = []
    for relative, record in sorted(inventory_files.items()):
        if record.get("kind") not in {"asset_container", "unity_settings"}:
            continue
        path = game_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size"]):
            raise ValueError(f"inventory size mismatch: {relative}")
        result.append(path)
    if not result:
        raise ValueError("inventory names no Unity containers")
    return result


def transform_reader(game_object: Any) -> Any | None:
    for pair in getattr(game_object, "m_Component", ()):
        pointer = getattr(pair, "component", None)
        if pointer is None:
            continue
        try:
            reader = pointer.deref()
        except Exception:
            continue
        if reader.type.name in {"Transform", "RectTransform"}:
            return reader
    return None


def hierarchy_record(game_object_pointer: Any) -> dict[str, Any] | None:
    try:
        game_object = game_object_pointer.deref_parse_as_object()
        reader = transform_reader(game_object)
    except Exception:
        return None
    if reader is None:
        return None

    names = []
    transform_chain = []
    seen = set()
    current = reader
    local_transform = None
    while current is not None and len(names) < 256:
        identity = (current.assets_file.name, current.path_id)
        if identity in seen:
            names.append("<cycle>")
            break
        seen.add(identity)
        value = current.parse_as_object()
        value_dict = current.parse_as_dict()
        if local_transform is None:
            local_transform = {
                "transform_path_id": current.path_id,
                "local_position": safe_value(value_dict.get("m_LocalPosition")),
                "local_rotation": safe_value(value_dict.get("m_LocalRotation")),
                "local_scale": safe_value(value_dict.get("m_LocalScale")),
            }
        owner = value.m_GameObject.deref_parse_as_object()
        owner_name = str(getattr(owner, "m_Name", "") or "<unnamed>")
        father = getattr(value, "m_Father", None)
        sibling_index = None
        if father is not None and father:
            try:
                parent = father.deref_parse_as_object()
                for index, child in enumerate(getattr(parent, "m_Children", ())):
                    if child.path_id == current.path_id:
                        sibling_index = index
                        break
            except Exception:
                pass
        names.append(owner_name)
        transform_chain.append({
            "name": owner_name,
            "container": current.assets_file.name,
            "transform_path_id": current.path_id,
            "sibling_index": sibling_index,
            "active": bool(getattr(owner, "m_IsActive", True)),
            "local_position": safe_value(value_dict.get("m_LocalPosition")),
            "local_rotation": safe_value(value_dict.get("m_LocalRotation")),
            "local_scale": safe_value(value_dict.get("m_LocalScale")),
        })
        if father is None or not father:
            current = None
        else:
            current = father.deref()
    names.reverse()
    transform_chain.reverse()
    return {
        "game_object_path": "/".join(names),
        "transform_chain": transform_chain,
        **(local_transform or {}),
    }


def probe(inventory_path: Path, dummy_dir: Path) -> dict[str, Any]:
    try:
        import UnityPy
        from UnityPy.helpers.TypeTreeGenerator import TypeTreeGenerator
    except ImportError as exc:
        raise RuntimeError("UnityPy with TypeTreeGeneratorAPI is required") from exc

    inventory, inventory_files = load_inventory(inventory_path)
    game_root = Path(inventory["install"])
    paths = container_paths(game_root, inventory_files)
    environment = UnityPy.load(*map(str, paths))
    first_object = next(iter(environment.objects), None)
    if first_object is None:
        raise ValueError("Unity containers contain no objects")
    unity_version = first_object.assets_file.unity_version
    generator = TypeTreeGenerator(unity_version)
    generator.load_local_dll_folder(str(dummy_dir))
    environment.typetree_generator = generator

    candidates = []
    failures = []
    mono_behaviour_count = 0
    header_resolved_count = 0
    for obj in environment.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        mono_behaviour_count += 1
        location = {"container": obj.assets_file.name, "path_id": obj.path_id}
        try:
            head = obj.parse_monobehaviour_head()
            script = head.m_Script.deref_parse_as_object()
            namespace = str(getattr(script, "m_Namespace", "") or "")
            class_name = str(getattr(script, "m_ClassName", "") or "")
            assembly = str(getattr(script, "m_AssemblyName", "") or "")
            header_resolved_count += 1
        except Exception as exc:
            failures.append({**location, "stage": "header", "error_type": type(exc).__name__})
            continue
        if not is_target(namespace, class_name):
            continue

        owner = None
        hierarchy = None
        try:
            owner = str(getattr(head.m_GameObject.deref_parse_as_object(), "m_Name", "") or "")
            hierarchy = hierarchy_record(head.m_GameObject)
        except Exception:
            pass
        raw = obj.get_raw_data()
        record = {
            **location,
            "namespace": namespace,
            "class": class_name,
            "assembly": assembly,
            "owner": owner,
            "hierarchy": hierarchy,
            "enabled": bool(getattr(head, "m_Enabled", False)),
            "serialized_bytes": len(raw),
            "serialized_sha256": sha256_bytes(raw),
            "parsed": False,
            "parse_mode": None,
            "values": None,
        }
        try:
            record["values"] = safe_value(obj.parse_as_dict())
            record["parsed"] = True
            record["parse_mode"] = "generated_typetree"
        except Exception as initial_exc:
            try:
                parsed, repair_count = parse_with_string_vector_repair(obj)
                record["values"] = safe_value(parsed)
                record["parsed"] = True
                record["parse_mode"] = "generated_typetree_string_vector_repair"
                record["typetree_repair_count"] = repair_count
                record["initial_parse_error_type"] = type(initial_exc).__name__
            except Exception as repair_exc:
                record["parse_error_type"] = type(initial_exc).__name__
                record["parse_error"] = str(initial_exc)[:512]
                record["repair_error_type"] = type(repair_exc).__name__
        candidates.append(record)

    candidates.sort(key=lambda item: (item["namespace"], item["class"], item["container"], item["path_id"]))
    parsed_count = sum(bool(record["parsed"]) for record in candidates)
    return {
        "schema": "rek.mujoco_asset_probe.v1",
        "build_fingerprint": inventory["build_fingerprint"],
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "game_root": str(game_root),
        "unity_version": unity_version,
        "unitypy_version": UnityPy.__version__,
        "dummy_dir": str(dummy_dir),
        "container_count": len(paths),
        "mono_behaviour_count": mono_behaviour_count,
        "header_resolved_count": header_resolved_count,
        "target_count": len(candidates),
        "parsed_target_count": parsed_count,
        "unparsed_target_count": len(candidates) - parsed_count,
        "targets": candidates,
        "header_failures": failures,
        "limits": [
            "serialized component values are static candidate leads",
            "no Unity object or proprietary binary payload is exported",
            "server-side participation still requires runtime validation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dummy-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.inventory.resolve(), args.dummy_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "schema",
        "build_fingerprint",
        "unity_version",
        "mono_behaviour_count",
        "header_resolved_count",
        "target_count",
        "parsed_target_count",
        "unparsed_target_count",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
