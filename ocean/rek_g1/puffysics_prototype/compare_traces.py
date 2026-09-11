"""Compare a Puffysics prototype trace with the CUDA MuJoCo candidate.

The reference is another candidate simulator, not an authentic REK recording.
The sampled height/upright fall proxy is not a measured REK knockout rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(directory):
    directory = directory.resolve(strict=True)
    report_path, trace_path = directory / "report.json", directory / "trace.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as data:
        trace = {name: data[name] for name in data.files}
    qpos = trace["qpos"]
    if qpos.ndim != 3 or qpos.shape[2] != 72 or qpos.shape[0] < 2 or qpos.shape[1] < 1:
        raise ValueError(f"{directory}: expected qpos [ticks+1, arenas, 72]")
    if trace["qvel"].shape != (*qpos.shape[:2], 70):
        raise ValueError(f"{directory}: qvel shape differs from qpos")
    expected_actions = (qpos.shape[0]-1, qpos.shape[1]*2, 29)
    for name in ("raw_policy_actions", "position_targets"):
        if trace[name].shape != expected_actions:
            raise ValueError(f"{directory}: invalid {name} shape")
    return report, trace, {
        "run_directory": str(directory), "report_sha256": digest(report_path),
        "trace_sha256": digest(trace_path), "backend": report["backend"],
    }


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    valid = values[finite]
    return {
        "count": int(values.size), "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "max": float(valid.max()) if valid.size else None,
        "mean": float(valid.mean()) if valid.size else None,
        "rms": float(np.sqrt(np.mean(valid*valid))) if valid.size else None,
    }


def component_error(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("comparison operands have different shapes")
    with np.errstate(invalid="ignore", over="ignore"):
        error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    result = finite_summary(error)
    result["shape"] = list(error.shape)
    result["array_equal"] = bool(np.array_equal(reference, candidate))
    result["allclose_rtol_1e_minus_5_atol_1e_minus_6"] = bool(
        np.allclose(reference, candidate, rtol=1e-5, atol=1e-6, equal_nan=False)
    )
    return result, error


def unit_quaternions(wxyz):
    values = np.asarray(wxyz, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        norm = np.linalg.norm(values, axis=-1, keepdims=True)
        valid = np.isfinite(values).all(axis=-1, keepdims=True) & np.isfinite(norm) & (norm > 0)
        return np.where(valid, values / norm, np.nan)


def angular_error(reference, candidate):
    a, b = unit_quaternions(reference), unit_quaternions(candidate)
    vector = a[..., 0:1]*b[..., 1:] - b[..., 0:1]*a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    scalar = np.sum(a*b, axis=-1)
    return 2*np.arctan2(np.linalg.norm(vector, axis=-1), np.abs(scalar))


def fall_proxy(roots, period, height_threshold, upright_threshold):
    quaternions = unit_quaternions(roots[..., 3:7])
    upright = 1-2*(quaternions[..., 1]**2 + quaternions[..., 2]**2)
    valid = np.isfinite(roots).all(axis=-1) & np.isfinite(upright)
    fallen = valid & ((roots[..., 2] < height_threshold) | (upright < upright_threshold))
    first = []
    for row in range(roots.shape[1]):
        indices = np.flatnonzero(fallen[:, row])
        first.append(float(indices[0]*period) if indices.size else None)
    return {
        "definition": "sampled root_height < threshold OR normalized_root_up_z < threshold",
        "height_threshold_m": height_threshold, "upright_z_threshold": upright_threshold,
        "sample_period_seconds": period, "first_fall_seconds_per_robot": first,
        "fallen_robots": sum(value is not None for value in first),
        "invalid_samples": int((~valid).sum()),
        "authentic_rek_knockout_rule": False,
    }


def timestep_max(values):
    values = np.asarray(values).reshape(values.shape[0], -1)
    return [float(row[np.isfinite(row)].max()) if np.isfinite(row).any() else None for row in values]


def run_status(report, trace):
    finite = {name: bool(np.isfinite(value).all()) for name, value in trace.items()}
    stats = report.get("native_stats", {})
    return {
        "trace_finite": finite,
        "reported_finite_mapping_clock_checks_passed": report.get("finite_mapping_clock_checks_passed"),
        "initial_mapping": report.get("initial_mapping"),
        "clock_max_abs_error_seconds": report.get("clock_max_abs_error_seconds"),
        "native_nonfinite_arenas": stats.get("nonfinite_arenas"),
        "native_solver_failure_arenas": stats.get("articulated_solver_failure_arenas"),
        "native_contact_capacity_reached_arenas": stats.get("contact_capacity_reached_arenas"),
        "native_max_contact_count": stats.get("max_contact_count"),
        "native_predictive_joint_limit_impulses": stats.get("predictive_joint_limit_impulses"),
        "support_gaps": report.get("backend_metadata", {}).get("support_gaps", []),
    }


def compare(reference_run, candidate_run, *, height_threshold=0.45, upright_threshold=0.5):
    reference_report, reference, reference_identity = load_run(reference_run)
    candidate_report, candidate, candidate_identity = load_run(candidate_run)
    if reference_report["backend"] != "mujoco":
        raise ValueError("--reference-run must be the MuJoCo candidate, not an authentic REK label")
    if reference["qpos"].shape[1] != candidate["qpos"].shape[1]:
        raise ValueError("arena counts differ; matching paired worlds are required")
    for key in ("model_sha256", "assets_sha256", "encoder_sha256", "decoder_sha256",
                "fixed_reference_role", "reference_loop", "controller_period_seconds"):
        if reference_report[key] != candidate_report[key]:
            raise ValueError(f"runs have different {key}")
    period = float(reference_report["controller_period_seconds"])
    if not np.isfinite(period) or period <= 0:
        raise ValueError("controller period must be positive and finite")
    count = min(reference["qpos"].shape[0], candidate["qpos"].shape[0])
    arenas = reference["qpos"].shape[1]
    rows = arenas*2
    roots = [trace["qpos"][:count].reshape(count, rows, 36)[..., :7]
             for trace in (reference, candidate)]
    with np.errstate(invalid="ignore", over="ignore"):
        translation = np.linalg.norm(roots[1][..., :3].astype(np.float64)-roots[0][..., :3], axis=-1)
    rotation = angular_error(roots[0][..., 3:7], roots[1][..., 3:7])
    qindices = np.asarray(reference_report["initial_mapping"]["joint_qpos"], dtype=np.int64)
    dqindices = np.asarray(reference_report["initial_mapping"]["joint_qvel"], dtype=np.int64)
    for key, indices in (("joint_qpos", qindices), ("joint_qvel", dqindices)):
        if indices.shape != (2, 29) or not np.array_equal(indices, candidate_report["initial_mapping"][key]):
            raise ValueError(f"runs have inconsistent {key} mapping")
    joint_result, joint_errors = component_error(reference["qpos"][:count, :, qindices], candidate["qpos"][:count, :, qindices])
    velocity_result, _ = component_error(reference["qvel"][:count, :, dqindices], candidate["qvel"][:count, :, dqindices])
    initial_actions, _ = component_error(reference["raw_policy_actions"][0], candidate["raw_policy_actions"][0])
    all_actions, _ = component_error(reference["raw_policy_actions"][:count-1], candidate["raw_policy_actions"][:count-1])
    initial_targets, _ = component_error(reference["position_targets"][0], candidate["position_targets"][0])
    falls = [fall_proxy(root, period, height_threshold, upright_threshold) for root in roots]
    first_fall_delta = [None if a is None or b is None else b-a for a, b in zip(
        falls[0]["first_fall_seconds_per_robot"], falls[1]["first_fall_seconds_per_robot"], strict=True)]
    result = {
        "schema": "rek.puffysics_candidate_trace_comparison.v1",
        "reference_identity": reference_identity, "candidate_identity": candidate_identity,
        "reference_classification": "CUDA MuJoCo semantic candidate; not authentic REK",
        "authentic_rek_parity_established": False,
        "comparison_is_parity_acceptance_test": False,
        "role": reference_report["fixed_reference_role"], "reference_loop": reference_report["reference_loop"],
        "arenas": arenas, "robot_rows": rows, "compared_control_ticks": count-1,
        "compared_seconds": (count-1)*period,
        "reference_control_ticks": reference["qpos"].shape[0]-1,
        "candidate_control_ticks": candidate["qpos"].shape[0]-1,
        "aligned_initial_samples_included": True,
        "reference_status": run_status(reference_report, reference),
        "candidate_status": run_status(candidate_report, candidate),
        "initial_controller_actions": initial_actions, "initial_position_targets_rad": initial_targets,
        "controller_actions_common_prefix": all_actions,
        "root_translation_error_m": finite_summary(translation),
        "root_orientation_geodesic_error_rad": finite_summary(rotation),
        "joint_position_error_rad": joint_result, "joint_velocity_error_rad_per_second": velocity_result,
        "reference_fall_proxy": falls[0], "candidate_fall_proxy": falls[1],
        "first_fall_time_candidate_minus_reference_seconds": first_fall_delta,
        "time_series": {
            "seconds": (np.arange(count)*period).tolist(),
            "max_root_translation_error_m": timestep_max(translation),
            "max_root_orientation_error_rad": timestep_max(rotation),
            "max_joint_position_error_rad": timestep_max(joint_errors),
            "reference_min_root_height_m": [-value if value is not None else None for value in timestep_max(-roots[0][..., 2])],
            "candidate_min_root_height_m": [-value if value is not None else None for value in timestep_max(-roots[1][..., 2])],
        },
    }
    return result


def plot_comparison(result, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    series = result["time_series"]
    t = series["seconds"]
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes[:3],
            ("max_root_translation_error_m", "max_root_orientation_error_rad", "max_joint_position_error_rad"),
            ("Root error (m)", "Orientation error (rad)", "Joint error (rad)"), strict=True):
        ax.plot(t, series[key])
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[3].plot(t, series["reference_min_root_height_m"], label="MuJoCo candidate")
    axes[3].plot(t, series["candidate_min_root_height_m"], label=result["candidate_identity"]["backend"])
    axes[3].axhline(result["reference_fall_proxy"]["height_threshold_m"], color="gray", linestyle=":", label="height proxy threshold")
    axes[3].set_ylabel("Minimum root height (m)")
    axes[3].set_xlabel("Simulation time (s)")
    axes[3].legend()
    figure.suptitle("Puffysics versus MuJoCo candidate\nAuthentic REK parity is not established")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height-threshold-m", type=float, default=0.45)
    parser.add_argument("--upright-z-threshold", type=float, default=0.5)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.height_threshold_m) or not -1 <= args.upright_z_threshold <= 1:
        parser.error("finite height and an upright threshold in [-1,1] are required")
    result = compare(args.reference_run, args.candidate_run,
                     height_threshold=args.height_threshold_m, upright_threshold=args.upright_z_threshold)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.plot:
        try:
            plot_path = args.out / "comparison.png"
            plot_comparison(result, plot_path)
            result["plot"] = {"path": str(plot_path.resolve()), "sha256": digest(plot_path)}
        except ImportError as exc:
            result["plot_unavailable"] = str(exc)
    output = args.out / "comparison.json"
    with output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    summary = {key: value for key, value in result.items() if key != "time_series"}
    summary["comparison_path"] = str(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
