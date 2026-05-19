#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os


def scalar_at(values, idx):
    if isinstance(values, list):
        if idx < len(values):
            return values[idx]
        return None
    return values


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100000:
            return f"{value:.3e}"
        return f"{value:.6g}"
    return str(value)


def load_logs(path):
    if os.path.isdir(path):
        pattern = os.path.join(path, "*.json")
    else:
        pattern = path

    for fpath in sorted(glob.glob(pattern)):
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {fpath}: invalid json")
            continue
        except OSError as e:
            print(f"Skipping {fpath}: {e}")
            continue

        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            print(f"Skipping {fpath}: no metrics")
            continue
        yield fpath, data, metrics


def iter_points(path):
    for fpath, data, metrics in load_logs(path):
        steps = metrics.get("agent_steps")
        if not isinstance(steps, list):
            continue
        n = len(steps)
        for i in range(n):
            yield fpath, data, metrics, i


def best_point(path, key):
    best = None
    for fpath, data, metrics, idx in iter_points(path):
        value = scalar_at(metrics.get(key), idx)
        if not is_number(value):
            continue
        if best is None or value > best["value"]:
            best = {
                "value": value,
                "file": fpath,
                "data": data,
                "metrics": metrics,
                "idx": idx,
            }
    return best


def print_section(title, point):
    if point is None:
        print(f"{title}: no data")
        return

    metrics = point["metrics"]
    idx = point["idx"]
    data = point["data"]

    print(title)
    print(f"  file: {point['file']}")
    print(f"  row: {idx}")
    print(f"  env_name: {data.get('env_name', 'unknown')}")

    print("  cost:")
    for key in ("agent_steps", "uptime", "epoch", "SPS"):
        print(f"    {key}: {fmt(scalar_at(metrics.get(key), idx))}")

    print("  env:")
    env_keys = sorted(k for k in metrics if k.startswith("env/"))
    for key in env_keys:
        print(f"    {key[4:]}: {fmt(scalar_at(metrics.get(key), idx))}")

    print("  config:")
    for group in ("vec", "train", "policy", "env"):
        value = data.get(group)
        if isinstance(value, dict):
            compact = ", ".join(f"{k}={fmt(v)}" for k, v in sorted(value.items()))
            print(f"    {group}: {compact}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize PufferLib JSON logs written by pufferl.py.")
    parser.add_argument("path", help="Log directory or glob, e.g. logs/incremental_maze")
    args = parser.parse_args()

    score = best_point(args.path, "env/score")
    perf = best_point(args.path, "env/perf")

    print_section("best score", score)
    if perf is not None and (score is None or perf["file"] != score["file"] or perf["idx"] != score["idx"]):
        print()
        print_section("best perf", perf)


if __name__ == "__main__":
    main()
