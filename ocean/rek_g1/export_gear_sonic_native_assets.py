#!/usr/bin/env python3
"""Export validated external assets for the native Gear-Sonic candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

import gear_sonic_candidate as candidate
import sonic_candidate as plant


SCHEMA = "rek.g1_gear_sonic_native_assets.v1"


def _write_new(path: Path, payload: bytes) -> dict[str, object]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--motion-role", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    motion = plant.load_motion(args.assets_dir, args.motion_role, args.manifest)
    xml_contract = plant.inspect_xml_contract(args.xml)
    arena_contract = plant.load_arena_contract(args.arena)
    xml_text = plant.add_arena_geoms(xml_contract.source_bytes, arena_contract)
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
        records: dict[str, object] = {}
        for filename, (values, shape) in arrays.items():
            record = _write_new(output / filename, _array_bytes(values))
            record.update({"dtype": "float32_le", "shape": shape})
            records[filename] = record
        records["model.single_arena.xml"] = {
            **_write_new(output / "model.single_arena.xml", xml_text.encode("utf-8")),
            "format": "MuJoCo XML",
        }
        report = {
            "schema": SCHEMA,
            "classification": candidate.CLASSIFICATION,
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
            },
            "source_contracts": {
                "recovered_xml_sha256": xml_contract.source_sha256,
                "arena_contract_sha256": arena_contract.source_sha256,
                "arena_geometry_sha256": arena_contract.derived_geometry_sha256,
            },
            "files": records,
            "limits": [
                "The exported motion and composed plant are a public-family candidate.",
                "Exact current REK Sonic configuration and model-weight identity are unknown.",
                "No held-out REK trajectory is evaluated by this export.",
            ],
        }
        manifest_bytes = (
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        manifest_record = _write_new(
            output / "native_assets_manifest.json", manifest_bytes
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
                "classification": candidate.CLASSIFICATION,
                "rek_parity_claim": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
