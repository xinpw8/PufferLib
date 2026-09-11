#!/usr/bin/env python3
"""Compile and step a recovered REK MJCF with the official MuJoCo binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_summary(array: Any) -> dict[str, Any]:
    raw = array.tobytes()
    flat = array.reshape(-1)
    return {
        "count": int(flat.size),
        "finite": bool(all(math.isfinite(float(value)) for value in flat)),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate(path: Path, expected: dict[str, int], steps: int) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    actual = {
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items() if actual[key] != value
    }
    if mismatches:
        raise ValueError(f"model dimension mismatch: {mismatches}")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = {
        "qpos": array_summary(data.qpos.copy()),
        "qvel": array_summary(data.qvel.copy()),
        "qacc": array_summary(data.qacc.copy()),
        "contact_count": int(data.ncon),
    }
    max_contact_count = int(data.ncon)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        max_contact_count = max(max_contact_count, int(data.ncon))
    final = {
        "time": float(data.time),
        "qpos": array_summary(data.qpos.copy()),
        "qvel": array_summary(data.qvel.copy()),
        "qacc": array_summary(data.qacc.copy()),
        "ctrl": array_summary(data.ctrl.copy()),
        "contact_count": int(data.ncon),
    }
    if not all(summary["finite"] for name, summary in final.items()
               if name not in {"time", "contact_count"}):
        raise ValueError("non-finite state after stepping")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "mujoco_version": mujoco.__version__,
        "dimensions": actual,
        "expected_dimensions": expected,
        "steps": steps,
        "initial": initial,
        "final": final,
        "max_contact_count": max_contact_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expect-nbody", type=int, required=True)
    parser.add_argument("--expect-njnt", type=int, required=True)
    parser.add_argument("--expect-ngeom", type=int, required=True)
    parser.add_argument("--expect-nq", type=int, required=True)
    parser.add_argument("--expect-nv", type=int, required=True)
    parser.add_argument("--expect-nu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("steps must be non-negative")
    expected = {
        "nbody": args.expect_nbody,
        "njnt": args.expect_njnt,
        "ngeom": args.expect_ngeom,
        "nq": args.expect_nq,
        "nv": args.expect_nv,
        "nu": args.expect_nu,
    }
    result = validate(args.model.resolve(), expected, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": result["path"],
        "sha256": result["sha256"],
        "mujoco_version": result["mujoco_version"],
        "dimensions": result["dimensions"],
        "steps": result["steps"],
        "final_time": result["final"]["time"],
        "final_contact_count": result["final"]["contact_count"],
        "max_contact_count": result["max_contact_count"],
        "finite": all(summary["finite"] for name, summary in result["final"].items()
                      if name not in {"time", "contact_count"}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
