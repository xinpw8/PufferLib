#!/usr/bin/env python3
"""Export validated external assets for the native two-G1 candidate runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

import g1_two_fighter_arena as duel
import gear_sonic_candidate as candidate
import sonic_candidate as plant


SCHEMA = "rek.g1_gear_sonic_native_duel_assets.v1"


def _write_new(path: Path, payload: bytes) -> dict[str, Any]:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "file": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _array_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value, dtype="<f4").tobytes(order="C")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--motion-role", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--spawn-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    motion = plant.load_motion(args.assets_dir, args.motion_role, args.manifest)
    reference = duel.create_two_fighter_arena_reference(
        xml=args.xml,
        arena=args.arena,
        spawn_contract=args.spawn_contract,
        runtime_manifest=args.manifest,
    )
    duel.assert_spawn_rebase_applied(reference)
    contract = reference.contract
    if tuple(int(getattr(reference.model, key)) for key in duel.EXPECTED_MODEL_DIMENSIONS) != tuple(
        duel.EXPECTED_MODEL_DIMENSIONS.values()
    ):
        raise duel.TwoFighterArenaError("compiled duel dimensions changed after validation")

    output = Path(os.path.abspath(args.out))
    output.mkdir(parents=True, exist_ok=False)
    try:
        arrays = {
            "motion_dof_position.f32le": (
                motion.dof_pos,
                [int(motion.dof_pos.shape[0]), candidate.ACTION_DIM],
            ),
            "motion_root_position.f32le": (
                motion.root_pos,
                [int(motion.root_pos.shape[0]), 3],
            ),
            "motion_root_rotation_xyzw.f32le": (
                motion.root_rot_xyzw,
                [int(motion.root_rot_xyzw.shape[0]), 4],
            ),
        }
        records: dict[str, Any] = {}
        for filename, (values, shape) in arrays.items():
            record = _write_new(output / filename, _array_bytes(values))
            record.update({"dtype": "float32_le", "shape": shape})
            records[filename] = record
        xml_bytes = contract.xml_text.encode("utf-8")
        records["model.two_fighter_arena.xml"] = {
            **_write_new(output / "model.two_fighter_arena.xml", xml_bytes),
            "format": "MuJoCo XML",
        }
        report = {
            "schema": SCHEMA,
            "classification": "public_family_candidate_with_build_pinned_initial_spawn",
            "rek_parity_claim": False,
            "motion": {
                "role": motion.role,
                "source_file": motion.filename,
                "source_bytes": motion.size,
                "source_sha256": motion.sha256,
                "frames": int(motion.dof_pos.shape[0]),
                "fps": motion.fps,
                "source_manifest_sha256": motion.manifest_sha256,
                "source_inventory_sha256": motion.inventory_sha256,
                "routing": "one fixed caller-supplied reference shared by all physical robots",
            },
            "duel": {
                "fighters_per_arena": 2,
                "physical_robot_order": "arena-major, player then opponent",
                "model_dimensions": duel.EXPECTED_MODEL_DIMENSIONS,
                "model_xml_sha256": contract.xml_sha256,
                "spawn_contract_sha256": contract.spawn_contract_sha256,
                "spawn_rebase_status": contract.spawn_rebase_status,
                "spawn_qpos0_prefixes": {
                    role: list(contract.spawn_rebase(role).free_joint_qpos_prefix)
                    for role in duel.ROLES
                },
                "shared_contact_physics_per_arena": True,
            },
            "source_contracts": {
                "recovered_xml_sha256": contract.recovered_xml_sha256,
                "arena_contract_sha256": contract.arena_contract_sha256,
                "arena_geometry_sha256": contract.arena_geometry_sha256,
                "runtime_manifest_file_sha256": contract.runtime_manifest_file_sha256,
                "runtime_manifest_canonical_sha256": contract.runtime_manifest_canonical_sha256,
            },
            "files": records,
            "limits": [
                "The controller and motion are a public-family candidate.",
                "Exact current REK Sonic configuration and model-weight identity are unknown.",
                "Semantic motion selection is not wired and must fail closed.",
                "Damage, hit zones, rewards, networking, and opponent policy are absent.",
                "No held-out REK trajectory is evaluated by this export.",
            ],
        }
        manifest_bytes = (
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        manifest_record = _write_new(
            output / "native_duel_assets_manifest.json", manifest_bytes
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": manifest_record,
                "frames": report["motion"]["frames"],
                "model_xml_sha256": contract.xml_sha256,
                "spawn_prefixes_verified": True,
                "rek_parity_claim": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
