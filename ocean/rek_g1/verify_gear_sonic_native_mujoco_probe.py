"""Compare the native GEAR-SONIC MuJoCo vector with the Python reference.

This verifier runs the same public-family candidate from the same source
artifacts and requires byte-identical final state and controller outputs.
It does not compare either candidate with a REK trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic_vector_env import GearSonicVectorEnv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--batch-models-dir", type=Path, required=True)
    parser.add_argument("--motion-role", required=True)
    parser.add_argument("--frame-mode", choices=("clamp", "loop"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--native-qpos", type=Path, required=True)
    parser.add_argument("--native-qvel", type=Path, required=True)
    parser.add_argument("--native-action", type=Path, required=True)
    parser.add_argument("--native-target", type=Path, required=True)
    return parser


def _read_exact(path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected exactly {expected_bytes}"
        )
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"{path} contains non-finite values")
    return value


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _comparison(reference: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    if reference.shape != native.shape or reference.dtype != native.dtype:
        raise ValueError(
            "comparison arrays differ in shape or dtype: "
            f"{reference.shape} {reference.dtype} versus {native.shape} {native.dtype}"
        )
    exact = bool(np.array_equal(reference, native))
    difference = np.abs(reference.astype(np.float64) - native.astype(np.float64))
    mismatch = np.argwhere(reference != native)
    first_mismatch = None
    if mismatch.size:
        location = tuple(int(index) for index in mismatch[0])
        first_mismatch = {
            "index": list(location),
            "python": float(reference[location]),
            "native": float(native[location]),
            "absolute_error": float(difference[location]),
        }
    return {
        "absolute_error_max": float(difference.max(initial=0.0)),
        "exact": exact,
        "first_mismatch": first_mismatch,
        "mismatch_count": int(mismatch.shape[0]),
        "native_sha256": _sha256(native),
        "python_sha256": _sha256(reference),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num-envs and steps must be positive")

    environment = GearSonicVectorEnv.from_artifacts(
        bundle=args.bundle,
        assets_dir=args.assets_dir,
        motion_role=args.motion_role,
        manifest=args.manifest,
        xml=args.xml,
        arena=args.arena,
        num_envs=args.num_envs,
        control_boundary="native-position-actuator",
        frame_mode=args.frame_mode,
        force_limit_source="public-model-config",
        physics_workers=1,
        batch_models_dir=args.batch_models_dir,
    )
    try:
        final_steps = None
        for _ in range(args.steps):
            final_steps = environment.step()
        assert final_steps is not None
        python_qpos = np.ascontiguousarray(
            np.stack([value.qpos.copy() for value in environment.data]),
            dtype=np.dtype("<f8"),
        )
        python_qvel = np.ascontiguousarray(
            np.stack([value.qvel.copy() for value in environment.data]),
            dtype=np.dtype("<f8"),
        )
        python_action = np.ascontiguousarray(
            np.stack([value.raw_action_policy for value in final_steps]),
            dtype=np.dtype("<f4"),
        )
        python_target = np.ascontiguousarray(
            np.stack([value.raw_targets_mujoco for value in final_steps]),
            dtype=np.dtype("<f4"),
        )
    finally:
        environment.close()

    shape_state_qpos = (args.num_envs, 36)
    shape_state_qvel = (args.num_envs, 35)
    shape_action = (args.num_envs, 29)
    native_qpos = _read_exact(args.native_qpos, np.dtype("<f8"), shape_state_qpos)
    native_qvel = _read_exact(args.native_qvel, np.dtype("<f8"), shape_state_qvel)
    native_action = _read_exact(args.native_action, np.dtype("<f4"), shape_action)
    native_target = _read_exact(args.native_target, np.dtype("<f4"), shape_action)

    comparisons = {
        "action": _comparison(python_action, native_action),
        "qpos": _comparison(python_qpos, native_qpos),
        "qvel": _comparison(python_qvel, native_qvel),
        "target": _comparison(python_target, native_target),
    }
    exact = all(value["exact"] for value in comparisons.values())
    report = {
        "classification": "public_family_candidate",
        "comparisons": comparisons,
        "exact": exact,
        "frame_mode": args.frame_mode,
        "motion_role": args.motion_role,
        "num_envs": args.num_envs,
        "policy_steps": args.steps,
        "rek_parity_claim": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
