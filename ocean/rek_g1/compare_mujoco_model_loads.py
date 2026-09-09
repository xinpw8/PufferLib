"""Diagnose MuJoCo XML string versus path compilation differences."""

from __future__ import annotations

import argparse
import json
from typing import Any

import mujoco
import numpy as np

import gear_sonic_candidate as candidate


def _numeric_arrays(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = np.asarray(getattr(model, name))
        except Exception:
            continue
        if value.dtype.kind not in "biufc":
            continue
        arrays[name] = value
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--configured", action="store_true")
    args = parser.parse_args()
    with open(args.model, encoding="utf-8") as stream:
        source = stream.read()
    from_string = mujoco.MjModel.from_xml_string(source)
    from_path = mujoco.MjModel.from_xml_path(args.model)
    if args.configured:
        for index in range(candidate.ACTION_DIM):
            from_string.actuator_dyntype[index] = int(mujoco.mjtDyn.mjDYN_NONE)
            from_string.actuator_gaintype[index] = int(mujoco.mjtGain.mjGAIN_FIXED)
            from_string.actuator_biastype[index] = int(mujoco.mjtBias.mjBIAS_AFFINE)
            from_string.actuator_gainprm[index, 0] = candidate.KP_MUJOCO[index]
            from_string.actuator_biasprm[index, 0] = 0.0
            from_string.actuator_biasprm[index, 1] = -candidate.KP_MUJOCO[index]
            from_string.actuator_biasprm[index, 2] = -candidate.KD_MUJOCO[index]
            from_string.actuator_ctrllimited[index] = 0
            from_string.actuator_forcelimited[index] = 1
            from_string.actuator_forcerange[index] = (
                -candidate.PUBLIC_EFFORT_LIMIT_MUJOCO[index],
                candidate.PUBLIC_EFFORT_LIMIT_MUJOCO[index],
            )

            from_path.actuator_dyntype[index] = int(mujoco.mjtDyn.mjDYN_NONE)
            from_path.actuator_gaintype[index] = int(mujoco.mjtGain.mjGAIN_FIXED)
            from_path.actuator_biastype[index] = int(mujoco.mjtBias.mjBIAS_AFFINE)
            from_path.actuator_gainprm[index] = 0.0
            from_path.actuator_biasprm[index] = 0.0
            from_path.actuator_gainprm[index, 0] = candidate.KP_MUJOCO[index]
            from_path.actuator_biasprm[index, 1] = -candidate.KP_MUJOCO[index]
            from_path.actuator_biasprm[index, 2] = -candidate.KD_MUJOCO[index]
            from_path.actuator_ctrllimited[index] = 0
            from_path.actuator_forcelimited[index] = 1
            from_path.actuator_forcerange[index] = (
                -candidate.PUBLIC_EFFORT_LIMIT_MUJOCO[index],
                candidate.PUBLIC_EFFORT_LIMIT_MUJOCO[index],
            )
        from_string.opt.timestep = candidate.PHYSICS_DT
        from_path.opt.timestep = candidate.PHYSICS_DT
    string_arrays = _numeric_arrays(from_string)
    path_arrays = _numeric_arrays(from_path)
    if string_arrays.keys() != path_arrays.keys():
        raise RuntimeError("model numeric property sets differ")
    mismatches: list[dict[str, Any]] = []
    for name in string_arrays:
        left = string_arrays[name]
        right = path_arrays[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatches.append(
                {
                    "field": name,
                    "string_dtype": str(left.dtype),
                    "string_shape": list(left.shape),
                    "path_dtype": str(right.dtype),
                    "path_shape": list(right.shape),
                }
            )
            continue
        unequal = left != right
        if np.any(unequal):
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            mismatches.append(
                {
                    "field": name,
                    "max_absolute_error": float(difference.max(initial=0.0)),
                    "mismatch_count": int(np.count_nonzero(unequal)),
                    "shape": list(left.shape),
                }
            )
    print(
        json.dumps(
            {
                "checked_numeric_properties": len(string_arrays),
                "exact": not mismatches,
                "configured": args.configured,
                "mismatches": mismatches,
                "mujoco_version": mujoco.__version__,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
