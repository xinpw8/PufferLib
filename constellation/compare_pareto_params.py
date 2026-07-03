#!/usr/bin/env python3
"""Compare hyperparameters across Pareto-front points.

Inputs can be either:
    * the JSON written by constellation/cache_data.py
    * a raw log directory or glob of individual experiment JSON files

By default it compares CL vs vanilla points in one env using:
    train/cl_frac > 0  vs  train/cl_frac == 0

Examples:
    python constellation/compare_pareto_params.py --env boxoban
    python constellation/compare_pareto_params.py sweep.json --env boxoban --top-k 20
    python constellation/compare_pareto_params.py --left-source runs/cl --right-source runs/vanilla
    python constellation/compare_pareto_params.py --env boxoban \
        --left-filter 'train/cl_frac>0' --right-filter 'train/cl_frac==0'
    python constellation/compare_pareto_params.py --left-cache cl.json --right-cache vanilla.json \
        --left-env boxoban --right-env boxoban --left-filter '' --right-filter ''
"""

import argparse
import glob
import json
import math
import os
import re
from statistics import mean, median


DEFAULT_CACHE = "resources/constellation/experiments.json"
DEFAULT_LEFT_FILTER = "train/cl_frac>0"
DEFAULT_RIGHT_FILTER = "train/cl_frac==0"
DEFAULT_SCORE_KEY = "env/score"
DEFAULT_COST_KEY = "agent_steps"

PARAM_PREFIXES = ("train/", "policy/", "vec/")
METRIC_KEYS = {
    "SPS",
    "agent_steps",
    "uptime",
    "epoch",
    "env/perf",
    "env/score",
    "env/episode_return",
    "env/episode_length",
    "env/targets_hit",
    "env/final_puzzle_tick",
    "env/n",
    "perf/rollout",
    "perf/eval_gpu",
    "perf/eval_env",
    "perf/train_misc",
    "perf/train_forward",
    "perf/train",
}

CONDITION_RE = re.compile(r"^\s*([^<>=!]+?)\s*(<=|>=|==|!=|<|>)\s*(.*?)\s*$")


def parse_atom(text):
    text = text.strip()
    if text == "":
        return None
    low = text.lower()
    if low in ("none", "null", "nan"):
        return None
    if low == "true":
        return 1.0
    if low == "false":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")


def parse_series(value):
    if isinstance(value, str):
        if "," in value:
            return [parse_atom(part) for part in value.split(",")]
        return [parse_atom(value)]
    if isinstance(value, list):
        return [parse_atom(str(v)) if isinstance(v, str) else v for v in value]
    return [value]


def is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value):
    if value is None:
        return "n/a"
    if is_num(value):
        if value == 0:
            return "0"
        if abs(value) >= 100000 or abs(value) < 0.001:
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value)


def percentile(values, pct):
    values = sorted(v for v in values if is_num(v))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = pct * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def stat(values):
    values = [v for v in values if is_num(v)]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def stat_str(s):
    if s is None:
        return "n/a"
    return f"{fmt(s['median'])} [{fmt(s['q1'])}, {fmt(s['q3'])}]"


def unroll_nested_dict(d, prefix=""):
    if not isinstance(d, dict):
        return
    for key, value in d.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            yield from unroll_nested_dict(value, path)
        else:
            yield path, value


def load_cache(path, env):
    with open(path, "r") as f:
        root = json.load(f)

    if DEFAULT_SCORE_KEY in root or DEFAULT_COST_KEY in root:
        env_data = root
    else:
        if env is None:
            if len(root) == 1:
                env = next(iter(root))
            else:
                available = ", ".join(sorted(root)[:20])
                raise SystemExit(
                    f"--env is required for {path}. Available envs include: {available}"
                )
        if env not in root:
            available = ", ".join(sorted(root)[:30])
            raise SystemExit(f"Env '{env}' not found in {path}. Available: {available}")
        env_data = root[env]

    cols = {k: parse_series(v) for k, v in env_data.items()}
    n = max((len(v) for v in cols.values()), default=0)
    if n == 0:
        raise SystemExit(f"No rows found in {path}")

    for key, values in list(cols.items()):
        if len(values) == 1 and n > 1:
            cols[key] = values * n
        elif len(values) != n:
            del cols[key]
    return cols


def iter_log_paths(path):
    if os.path.isdir(path):
        pattern = os.path.join(path, "*.json")
        paths = sorted(glob.glob(pattern))
    elif glob.has_magic(path):
        paths = sorted(glob.glob(path))
    else:
        paths = [path]
    return paths


def flatten_config(exp):
    flat = {}
    for key, value in exp.items():
        if key in ("metrics", "sweep"):
            continue
        if isinstance(value, dict):
            for subkey, subvalue in unroll_nested_dict(value, key):
                if isinstance(subvalue, (dict, list)):
                    continue
                flat[subkey] = subvalue
        elif not isinstance(value, list):
            flat[key] = value
    return flat


def metric_length(metrics):
    steps = metrics.get(DEFAULT_COST_KEY)
    if isinstance(steps, list):
        return len(steps)
    lengths = [len(v) for v in metrics.values() if isinstance(v, list)]
    return max(lengths, default=0)


def row_value(values, idx):
    if isinstance(values, list):
        if idx >= len(values):
            return None
        return values[idx]
    return values


def load_raw_logs(path, env=None, pareto=True, score_key=DEFAULT_SCORE_KEY):
    rows = []
    paths = iter_log_paths(path)
    if not paths:
        raise SystemExit(f"No JSON logs matched {path}")

    skipped = 0
    for fpath in paths:
        try:
            with open(fpath, "r") as f:
                exp = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        if env is not None and exp.get("env_name") not in (None, env):
            continue

        metrics = exp.get("metrics")
        if not isinstance(metrics, dict):
            skipped += 1
            continue

        n = metric_length(metrics)
        if n <= 0:
            skipped += 1
            continue

        config = flatten_config(exp)
        for idx in range(n):
            row = dict(config)
            row["source/file"] = fpath
            row["source/row"] = idx
            for key, values in metrics.items():
                value = row_value(values, idx)
                if isinstance(value, list):
                    continue
                if is_num(value) or isinstance(value, str):
                    row[key] = parse_atom(value) if isinstance(value, str) else value
            rows.append(row)

    if not rows:
        raise SystemExit(f"No usable metric rows found in {path}")

    if pareto:
        rows = pareto_filter_rows(rows, score_key)
    return rows_to_cols(rows)


def rows_to_cols(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: [row.get(key) for row in rows] for key in keys}


def pareto_filter_rows(rows, score_key):
    scored = []
    for row in rows:
        score = row.get(score_key)
        steps = row.get("agent_steps")
        uptime = row.get("uptime")
        if is_num(score) and is_num(steps) and is_num(uptime):
            scored.append((row, score, steps, uptime))

    if not scored:
        return rows

    keep = []
    for i, (row_i, score_i, steps_i, uptime_i) in enumerate(scored):
        dominated = False
        for j, (_, score_j, steps_j, uptime_j) in enumerate(scored):
            if i == j:
                continue
            if score_j >= score_i and steps_j < steps_i and uptime_j < uptime_i:
                dominated = True
                break
        if not dominated:
            keep.append(row_i)
    return keep


def looks_like_cached_json(path):
    if os.path.isdir(path) or glob.has_magic(path):
        return False
    try:
        with open(path, "r") as f:
            root = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if isinstance(root, dict) and isinstance(root.get("metrics"), dict):
        return False
    if not isinstance(root, dict):
        return False
    return True


def load_source(path, env, pareto=True, score_key=DEFAULT_SCORE_KEY):
    if looks_like_cached_json(path):
        return load_cache(path, env)
    return load_raw_logs(path, env, pareto=pareto, score_key=score_key)


def get(cols, key, idx):
    values = cols.get(key)
    if values is None or idx >= len(values):
        return None
    return values[idx]


def split_query(query):
    if query is None:
        return []
    query = query.strip()
    if query == "":
        return []
    return [part for part in re.split(r"\s*(?:&&|,)\s*", query) if part]


def compile_filter(query):
    conditions = []
    for part in split_query(query):
        m = CONDITION_RE.match(part)
        if m is None:
            raise SystemExit(f"Could not parse filter condition: {part!r}")
        key, op, raw_target = m.groups()
        conditions.append((key.strip(), op, parse_atom(raw_target)))
    return conditions


def compare_values(left, op, right):
    if left is None:
        return False
    if is_num(left) and is_num(right):
        pass
    elif op in ("<", "<=", ">", ">="):
        return False

    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise AssertionError(op)


def select_rows(cols, query, score_key, min_score=None, top_k=0):
    n = max((len(v) for v in cols.values()), default=0)
    conditions = compile_filter(query)
    rows = []
    for idx in range(n):
        ok = True
        for key, op, target in conditions:
            if not compare_values(get(cols, key, idx), op, target):
                ok = False
                break
        if not ok:
            continue
        if min_score is not None:
            score = get(cols, score_key, idx)
            if not is_num(score) or score < min_score:
                continue
        rows.append(idx)

    if top_k and len(rows) > top_k:
        rows = sorted(rows, key=lambda i: get(cols, score_key, i) or -math.inf, reverse=True)
        rows = rows[:top_k]
    return rows


def numeric_values(cols, rows, key):
    return [get(cols, key, i) for i in rows if is_num(get(cols, key, i))]


def is_param_key(key):
    if key in METRIC_KEYS:
        return False
    if key.endswith("_norm") or key.startswith("sweep/"):
        return False
    if key.startswith(("perf/", "util/", "metrics/")):
        return False
    if key.startswith("env/level_") or key.startswith("env/cl_out_"):
        return False
    return key.startswith(PARAM_PREFIXES)


def candidate_param_keys(left_cols, right_cols, explicit):
    if explicit:
        return [k.strip() for k in explicit.split(",") if k.strip()]
    keys = sorted((set(left_cols) | set(right_cols)))
    return [k for k in keys if is_param_key(k)]


def effect_size(left_s, right_s, all_values):
    if left_s is None or right_s is None:
        return None
    denom = percentile(all_values, 0.75)
    q1 = percentile(all_values, 0.25)
    if denom is None or q1 is None:
        return None
    denom = denom - q1
    if denom == 0:
        denom = max(all_values) - min(all_values) if all_values else 0
    if denom == 0:
        return 0.0
    return (left_s["median"] - right_s["median"]) / denom


def log2_ratio(left_s, right_s):
    if left_s is None or right_s is None:
        return None
    lval = left_s["median"]
    rval = right_s["median"]
    if lval <= 0 or rval <= 0:
        return None
    return math.log(lval / rval, 2)


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def best_row(cols, rows, score_key, cost_key):
    if not rows:
        return None
    return max(
        rows,
        key=lambda i: (
            get(cols, score_key, i) if is_num(get(cols, score_key, i)) else -math.inf,
            -(get(cols, cost_key, i) if is_num(get(cols, cost_key, i)) else math.inf),
        ),
    )


def best_at_threshold(cols, rows, score_key, cost_key, threshold):
    eligible = [
        i for i in rows
        if is_num(get(cols, score_key, i))
        and get(cols, score_key, i) >= threshold
        and is_num(get(cols, cost_key, i))
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda i: get(cols, cost_key, i))


def auto_thresholds(left_cols, left_rows, right_cols, right_rows, score_key):
    scores = numeric_values(left_cols, left_rows, score_key) + numeric_values(right_cols, right_rows, score_key)
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo >= 3 and hi <= 100:
        start = max(1, int(math.ceil(lo)))
        end = int(math.floor(hi))
        return [float(x) for x in range(start, end + 1)]
    qs = [0.25, 0.5, 0.75, 0.9]
    vals = []
    for q in qs:
        value = percentile(scores, q)
        if value is not None and all(abs(value - old) > 1e-9 for old in vals):
            vals.append(value)
    return vals


def parse_thresholds(value, left_cols, left_rows, right_cols, right_rows, score_key):
    if value:
        return [float(x) for x in value.split(",") if x.strip()]
    return auto_thresholds(left_cols, left_rows, right_cols, right_rows, score_key)


def summarize_group(label, cols, rows, score_key, cost_key):
    score_s = stat(numeric_values(cols, rows, score_key))
    cost_s = stat(numeric_values(cols, rows, cost_key))
    best = best_row(cols, rows, score_key, cost_key)
    print(f"{label}: n={len(rows)}")
    print(f"  {score_key}: {stat_str(score_s)}")
    print(f"  {cost_key}: {stat_str(cost_s)}")
    if best is not None:
        print(
            f"  best_score: {fmt(get(cols, score_key, best))} "
            f"at {cost_key}={fmt(get(cols, cost_key, best))}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare hyperparameters across raw-log or cached Pareto-front points."
    )
    parser.add_argument("cache", nargs="?", default=DEFAULT_CACHE,
        help=f"cache JSON, raw log directory, or raw log glob (default: {DEFAULT_CACHE})")
    parser.add_argument("--env", help="Env key in the cached JSON, e.g. boxoban")
    parser.add_argument("--left-source",
        help="Optional separate source for left group: cache JSON, raw log directory, or glob")
    parser.add_argument("--right-source",
        help="Optional separate source for right group: cache JSON, raw log directory, or glob")
    parser.add_argument("--left-cache", help="Alias for --left-source")
    parser.add_argument("--right-cache", help="Alias for --right-source")
    parser.add_argument("--left-env", help="Env key/name for left source")
    parser.add_argument("--right-env", help="Env key/name for right source")
    parser.add_argument("--left-filter", default=None,
        help=f"Left filter, e.g. '{DEFAULT_LEFT_FILTER}'. Empty string means no filter.")
    parser.add_argument("--right-filter", default=None,
        help=f"Right filter, e.g. '{DEFAULT_RIGHT_FILTER}'. Empty string means no filter.")
    parser.add_argument("--left-label", default="cl")
    parser.add_argument("--right-label", default="vanilla")
    parser.add_argument("--score-key", default=DEFAULT_SCORE_KEY)
    parser.add_argument("--cost-key", default=DEFAULT_COST_KEY,
        help="Cost key for frontier threshold comparison, usually agent_steps or uptime")
    parser.add_argument("--min-score", type=float, default=None,
        help="Only compare rows with score >= this value")
    parser.add_argument("--top-k", type=int, default=0,
        help="Restrict each group to its top K rows by score before comparing params")
    parser.add_argument("--keys", default="",
        help="Comma-separated parameter keys to compare. Default: numeric train/policy/vec params")
    parser.add_argument("--max-diffs", type=int, default=25,
        help="Number of largest parameter differences to print")
    parser.add_argument("--thresholds", default="",
        help="Comma-separated score thresholds. Default: auto integer thresholds when score looks level-like")
    parser.add_argument("--threshold-keys", default="",
        help="Comma-separated keys to show in threshold table. Default: largest differing keys")
    parser.add_argument("--full", action="store_true",
        help="For raw log inputs, skip Pareto filtering and compare every logged point")
    args = parser.parse_args()

    separate = (
        args.left_source is not None or args.right_source is not None
        or args.left_cache is not None or args.right_cache is not None
    )
    left_filter = args.left_filter
    right_filter = args.right_filter
    if left_filter is None and right_filter is None and not separate:
        left_filter = DEFAULT_LEFT_FILTER
        right_filter = DEFAULT_RIGHT_FILTER
    left_filter = "" if left_filter is None else left_filter
    right_filter = "" if right_filter is None else right_filter

    left_path = args.left_source or args.left_cache or args.cache
    right_path = args.right_source or args.right_cache or args.cache
    left_env = args.left_env or args.env
    right_env = args.right_env or args.env

    left_cols = load_source(
        left_path, left_env, pareto=not args.full, score_key=args.score_key)
    right_cols = load_source(
        right_path, right_env, pareto=not args.full, score_key=args.score_key)
    left_rows = select_rows(left_cols, left_filter, args.score_key, args.min_score, args.top_k)
    right_rows = select_rows(right_cols, right_filter, args.score_key, args.min_score, args.top_k)

    print("Selection")
    print(f"  left:  {args.left_label} path={left_path} env={left_env or '<single>'} filter={left_filter!r}")
    print(f"  right: {args.right_label} path={right_path} env={right_env or '<single>'} filter={right_filter!r}")
    print()
    summarize_group(args.left_label, left_cols, left_rows, args.score_key, args.cost_key)
    summarize_group(args.right_label, right_cols, right_rows, args.score_key, args.cost_key)
    print()

    if not left_rows or not right_rows:
        raise SystemExit("One side has no selected rows; adjust --env/--*-filter.")

    diffs = []
    for key in candidate_param_keys(left_cols, right_cols, args.keys):
        left_values = numeric_values(left_cols, left_rows, key)
        right_values = numeric_values(right_cols, right_rows, key)
        left_s = stat(left_values)
        right_s = stat(right_values)
        if left_s is None or right_s is None:
            continue
        all_values = left_values + right_values
        eff = effect_size(left_s, right_s, all_values)
        ratio = log2_ratio(left_s, right_s)
        sort_value = abs(eff if eff is not None else 0.0)
        if sort_value == 0.0 and ratio is not None:
            sort_value = abs(ratio)
        diffs.append((sort_value, key, left_s, right_s, eff, ratio))

    diffs.sort(reverse=True, key=lambda x: x[0])
    print(f"Largest Parameter Differences ({args.left_label} vs {args.right_label})")
    rows = []
    for _, key, left_s, right_s, eff, ratio in diffs[:args.max_diffs]:
        rows.append([
            key,
            stat_str(left_s),
            stat_str(right_s),
            fmt(left_s["median"] - right_s["median"]),
            fmt(eff),
            fmt(ratio),
        ])
    print_table([
        "key",
        f"{args.left_label} median [q1,q3]",
        f"{args.right_label} median [q1,q3]",
        "delta",
        "iqr_effect",
        "log2_ratio",
    ], rows)

    thresholds = parse_thresholds(
        args.thresholds, left_cols, left_rows, right_cols, right_rows, args.score_key)
    threshold_keys = [k.strip() for k in args.threshold_keys.split(",") if k.strip()]
    if not threshold_keys:
        threshold_keys = [key for _, key, *_ in diffs[:8]]

    if thresholds:
        print()
        print(f"Best {args.cost_key} At Score Thresholds")
        headers = ["score>=", "group", args.cost_key, args.score_key] + threshold_keys
        rows = []
        for threshold in thresholds:
            for label, cols, selected in (
                    (args.left_label, left_cols, left_rows),
                    (args.right_label, right_cols, right_rows)):
                idx = best_at_threshold(cols, selected, args.score_key, args.cost_key, threshold)
                if idx is None:
                    rows.append([fmt(threshold), label, "n/a", "n/a"] + ["n/a"] * len(threshold_keys))
                    continue
                rows.append([
                    fmt(threshold),
                    label,
                    fmt(get(cols, args.cost_key, idx)),
                    fmt(get(cols, args.score_key, idx)),
                ] + [fmt(get(cols, key, idx)) for key in threshold_keys])
        print_table(headers, rows)


if __name__ == "__main__":
    main()
