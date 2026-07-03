#!/usr/bin/env python3
"""Downsample PufferLib JSON metric logs into a separate directory."""

import argparse
import glob
import json
import math
import os
import shutil


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate(values):
    if not values:
        return None
    if all(is_number(v) and math.isfinite(v) for v in values):
        return sum(values) / len(values)
    return values[-1]


def downsample_metrics(metrics, points):
    agent_steps = metrics.get("agent_steps")
    if not isinstance(agent_steps, list):
        raise ValueError("metrics.agent_steps must be a list")

    length = len(agent_steps)
    if length <= points:
        return metrics, False

    mismatched = [
        key for key, value in metrics.items()
        if isinstance(value, list) and len(value) != length
    ]
    if mismatched:
        raise ValueError(
            "metric length mismatch: " + ", ".join(mismatched[:5])
        )

    metric_keys = [
        key for key, value in metrics.items()
        if isinstance(value, list) and len(value) == length
    ]
    if not metric_keys:
        raise ValueError("no list metrics match metrics.agent_steps length")

    final_steps = agent_steps[-1]
    next_bin = final_steps / (points - 1) if points > 1 else math.inf
    bin_width = next_bin
    bins = []
    current = {key: [] for key in metric_keys}

    for idx in range(length):
        for key in metric_keys:
            current[key].append(metrics[key][idx])

        if agent_steps[idx] < next_bin:
            continue

        bins.append({key: aggregate(current[key]) for key in metric_keys})
        current = {key: [] for key in metric_keys}
        next_bin += bin_width

    if current and any(current[key] for key in metric_keys):
        bins.append({key: aggregate(current[key]) for key in metric_keys})

    if not bins:
        bins.append({key: metrics[key][-1] for key in metric_keys})

    # Match pufferl.py: averaged bins come first, then the final raw log entry
    # is always copied exactly as the last point.
    final_bin = {key: metrics[key][-1] for key in metric_keys}
    if len(bins) >= points:
        bins = bins[:points - 1]
    bins.append(final_bin)

    out = {}
    for key, value in metrics.items():
        if key in metric_keys:
            out[key] = [row[key] for row in bins]
        else:
            out[key] = value

    return out, True


def convert_file(src, dst, points):
    with open(src, "r") as f:
        exp = json.load(f)

    metrics = exp.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("missing metrics object")

    exp["metrics"], changed = downsample_metrics(metrics, points)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(exp, f)

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing JSON logs")
    parser.add_argument("output_dir", help="Directory for downsampled JSON logs")
    parser.add_argument("--points", type=int, default=5,
        help="Maximum points per metric curve")
    parser.add_argument("--overwrite", action="store_true",
        help="Allow writing into an existing output directory")
    parser.add_argument("--dry-run", action="store_true",
        help="Print what would be converted without writing files")
    args = parser.parse_args()

    if args.points < 2:
        raise SystemExit("--points must be at least 2")

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if input_dir == output_dir:
        raise SystemExit("input and output directories must be different")
    if os.path.exists(output_dir) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"{output_dir} exists; pass --overwrite to replace files")

    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not paths:
        raise SystemExit(f"No JSON logs found in {input_dir}")

    converted = 0
    copied = 0
    skipped = 0
    for src in paths:
        rel = os.path.relpath(src, input_dir)
        dst = os.path.join(output_dir, rel)
        try:
            if args.dry_run:
                with open(src, "r") as f:
                    exp = json.load(f)
                _, changed = downsample_metrics(exp["metrics"], args.points)
            else:
                changed = convert_file(src, dst, args.points)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping {src}: {exc}")
            skipped += 1
            continue

        if changed:
            converted += 1
        else:
            copied += 1
            if not args.dry_run and src != dst:
                shutil.copyfile(src, dst)

    action = "Would write" if args.dry_run else "Wrote"
    print(
        f"{action} {converted + copied} logs to {output_dir} "
        f"({converted} downsampled, {copied} already <= {args.points} points, "
        f"{skipped} skipped)"
    )


if __name__ == "__main__":
    main()
