#!/usr/bin/env python3
"""Export the complete build-pinned G1 motion set for the native duel runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

import gear_sonic_candidate as candidate
import sonic_candidate as plant


SCHEMA = "rek.g1_semantic_duel_assets.v1"
ROUTE_SCHEMA = "rek.g1_motion_route_contract.v1"
ROUTE_COUNT = 11


class SemanticDuelAssetError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return _sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _regular_file(path: Path, description: str) -> Path:
    if path.is_symlink():
        raise SemanticDuelAssetError(
            f"{description} must be a regular, non-symlink file"
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise SemanticDuelAssetError(
            f"{description} must be a regular, non-symlink file"
        )
    return resolved


def _load_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    source = _regular_file(path, description)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticDuelAssetError(f"{description} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SemanticDuelAssetError(f"{description} root must be an object")
    return payload, raw


def load_route_contract(path: Path) -> tuple[dict[str, Any], bytes]:
    contract, raw = _load_json(path, "G1 motion route contract")
    if contract.get("schema") != ROUTE_SCHEMA:
        raise SemanticDuelAssetError("G1 motion route contract schema mismatch")
    if contract.get("build_fingerprint") != plant.BUILD_FINGERPRINT:
        raise SemanticDuelAssetError("G1 motion route build fingerprint mismatch")
    if contract.get("parity_validated") is not False:
        raise SemanticDuelAssetError("route contract must not assert parity")
    routes = contract.get("routes")
    if not isinstance(routes, list) or len(routes) != ROUTE_COUNT:
        raise SemanticDuelAssetError("route contract must contain exactly 11 routes")
    route_ids = [route.get("route_id") for route in routes if isinstance(route, dict)]
    if route_ids != list(range(ROUTE_COUNT)):
        raise SemanticDuelAssetError("route contract IDs must be ordered 0 through 10")
    source = contract.get("source")
    runtime_manifest = source.get("runtime_assets_manifest") if isinstance(source, dict) else None
    if not isinstance(runtime_manifest, dict) or runtime_manifest.get(
        "canonical_sha256"
    ) != plant.EXPECTED_MANIFEST_CANONICAL_SHA256:
        raise SemanticDuelAssetError("route contract runtime manifest identity mismatch")
    return contract, raw


def _array_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value, dtype="<f4").tobytes(order="C")


def root_xyzw_to_wxyz(root_xyzw: np.ndarray) -> np.ndarray:
    value = np.asarray(root_xyzw, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 4 or not np.all(np.isfinite(value)):
        raise SemanticDuelAssetError("root quaternion array must be finite [frames,4]")
    return np.ascontiguousarray(value[:, (3, 0, 1, 2)], dtype=np.float32)


def _write_new(path: Path, payload: bytes) -> dict[str, Any]:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "file": path.name,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _copy_new(source: Path, destination: Path) -> dict[str, Any]:
    payload = _regular_file(source, source.name).read_bytes()
    return _write_new(destination, payload)


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticDuelAssetError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SemanticDuelAssetError(f"{field} must be finite")
    return result


def _route_config(route: dict[str, Any]) -> dict[str, Any]:
    config = route.get("mocap_clip_config")
    if not isinstance(config, dict):
        raise SemanticDuelAssetError("route is missing mocap_clip_config")
    mirror = config.get("mirror")
    loop = config.get("loop")
    if mirror not in (0, 1) or loop not in (0, 1):
        raise SemanticDuelAssetError("route mirror and loop fields must be binary")
    start_frame = config.get("startFrame")
    end_frame = config.get("endFrame")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise SemanticDuelAssetError("route startFrame must be an integer")
    if isinstance(end_frame, bool) or not isinstance(end_frame, int):
        raise SemanticDuelAssetError("route endFrame must be an integer")
    return {
        "mirror": mirror,
        "loop": loop,
        "playback_speed": _finite_number(config.get("playbackSpeed"), "playbackSpeed"),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "blend_in_seconds": _finite_number(config.get("blendInTime"), "blendInTime"),
        "blend_out_seconds": _finite_number(config.get("blendOutTime"), "blendOutTime"),
        "yaw_blend": _finite_number(config.get("yawBlend"), "yawBlend"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--route-contract", type=Path, required=True)
    parser.add_argument("--duel-model-xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    route_contract, route_raw = load_route_contract(args.route_contract)
    routes = route_contract["routes"]
    motions: dict[int, plant.MotionData] = {}
    for route in routes:
        npz = route.get("npz")
        if not isinstance(npz, dict):
            raise SemanticDuelAssetError("route is missing NPZ identity")
        path_id = npz.get("path_id")
        role = npz.get("role")
        if isinstance(path_id, bool) or not isinstance(path_id, int):
            raise SemanticDuelAssetError("NPZ path_id must be an integer")
        if not isinstance(role, str) or not role:
            raise SemanticDuelAssetError("NPZ role must be a nonempty string")
        if path_id not in motions:
            motions[path_id] = plant.load_motion(
                args.assets_dir, role, args.runtime_manifest
            )
        motion = motions[path_id]
        expected = {
            "output": motion.filename,
            "bytes": motion.size,
            "sha256": motion.sha256,
            "frames": int(motion.dof_pos.shape[0]),
            "dof": candidate.ACTION_DIM,
            "fps": motion.fps,
            "role": motion.role,
        }
        for field, value in expected.items():
            if npz.get(field) != value:
                raise SemanticDuelAssetError(
                    f"route {route['route_id']} NPZ {field} does not match extracted asset"
                )

    output = Path(os.path.abspath(args.out))
    output.mkdir(parents=True, exist_ok=False)
    try:
        files: dict[str, dict[str, Any]] = {}
        clips: list[dict[str, Any]] = []
        for path_id in sorted(motions):
            motion = motions[path_id]
            arrays = {
                f"motion_{path_id}_dof_position.f32le": (
                    motion.dof_pos,
                    [int(motion.dof_pos.shape[0]), candidate.ACTION_DIM],
                    "mujoco_joint_order",
                ),
                f"motion_{path_id}_root_position.f32le": (
                    motion.root_pos,
                    [int(motion.root_pos.shape[0]), 3],
                    "xyz_m",
                ),
                f"motion_{path_id}_root_rotation_xyzw.f32le": (
                    motion.root_rot_xyzw,
                    [int(motion.root_rot_xyzw.shape[0]), 4],
                    "xyzw",
                ),
                f"motion_{path_id}_root_rotation_wxyz.f32le": (
                    root_xyzw_to_wxyz(motion.root_rot_xyzw),
                    [int(motion.root_rot_xyzw.shape[0]), 4],
                    "wxyz",
                ),
            }
            clip_files: dict[str, str] = {}
            for filename, (values, shape, order) in arrays.items():
                record = _write_new(output / filename, _array_bytes(values))
                record.update({"dtype": "float32_le", "shape": shape, "order": order})
                files[filename] = record
                clip_files[order] = filename
            clips.append(
                {
                    "npz_path_id": path_id,
                    "role": motion.role,
                    "source_file": motion.filename,
                    "source_bytes": motion.size,
                    "source_sha256": motion.sha256,
                    "frames": int(motion.dof_pos.shape[0]),
                    "fps": motion.fps,
                    "files": clip_files,
                }
            )

        model_record = _copy_new(
            args.duel_model_xml,
            output / "model.two_fighter_arena.xml",
        )
        model_record["format"] = "MuJoCo XML"
        files["model.two_fighter_arena.xml"] = model_record

        route_records: list[dict[str, Any]] = []
        for route in routes:
            npz = route["npz"]
            route_records.append(
                {
                    "route_id": route["route_id"],
                    "name": route["name"],
                    "kind": route["kind"],
                    "runtime_move_index": route["runtime_move_index"],
                    "mocap_clip_config_path_id": route["mocap_clip_config"]["path_id"],
                    "npz_path_id": npz["path_id"],
                    "config": _route_config(route),
                }
            )

        report = {
            "schema": SCHEMA,
            "classification": "build_pinned_static_routes_with_public_family_controller_candidate",
            "rek_parity_claim": False,
            "build_fingerprint": plant.BUILD_FINGERPRINT,
            "physical_robot_order": "arena-major, player then opponent",
            "shared_contact_physics_per_arena": True,
            "source_contracts": {
                "route_contract_file_sha256": _sha256(route_raw),
                "route_contract_canonical_sha256": _canonical_sha256(route_contract),
                "runtime_manifest_canonical_sha256": plant.EXPECTED_MANIFEST_CANONICAL_SHA256,
                "runtime_inventory_sha256": next(iter(motions.values())).inventory_sha256,
            },
            "routes": route_records,
            "clips": clips,
            "files": files,
            "limits": [
                "Exact current REK Sonic model-weight identity is unknown.",
                "Static route identity does not establish trajectory parity.",
                "The bundle contains no reward, damage, hit-zone, or recovery model.",
                "The bundle must remain external and must not be committed.",
            ],
        }
        manifest_bytes = (
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        manifest_record = _write_new(
            output / "semantic_duel_assets_manifest.json", manifest_bytes
        )
    except BaseException:
        shutil.rmtree(output)
        raise

    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": manifest_record,
                "route_count": len(route_records),
                "unique_clip_count": len(clips),
                "rek_parity_claim": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
