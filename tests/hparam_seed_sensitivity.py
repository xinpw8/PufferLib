#!/usr/bin/env python3
"""
Correlate hyperparams with seed sensitivity among high-scoring breakout sweep runs.

1) Load logs/breakout/sweep_*_*.ini with final env/score >= SCORE_MIN
2) Sample (pilot) or take all configs
3) Retrain each config with several seeds (async as in original run)
4) seed_std / seed_range as sensitivity; correlate vs each hparam

Usage:
  python3 tests/hparam_seed_sensitivity.py --pilot
  python3 tests/hparam_seed_sensitivity.py --full
  python3 tests/hparam_seed_sensitivity.py --pilot --seeds 4 --sample 24
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUFFER = os.path.join(ROOT, "puffer")
LOG_DIR = os.path.join(ROOT, "logs", "breakout")
OUT_DIR = os.path.join(LOG_DIR, "seed_sensitivity")

# Hyperparams we re-apply when re-running a sweep trial
HP_KEYS = [
    "vec.total_agents",
    "vec.num_buffers",
    "vec.num_threads",
    "vec.gpu_env",
    "env.frameskip",
    "policy.hidden_size",
    "policy.num_layers",
    "train.total_timesteps",
    "train.learning_rate",
    "train.gamma",
    "train.gae_lambda",
    "train.replay_ratio",
    "train.clip_coef",
    "train.vf_coef",
    "train.vf_clip_coef",
    "train.max_grad_norm",
    "train.ent_coef",
    "train.momentum",
    "train.minibatch_size",
    "train.horizon",
    "train.prio_alpha",
    "train.prio_beta0",
    "train.vtrace_rho_clip",
    "train.vtrace_c_clip",
    "train.advantage_is",
    "train.pg_ratio_mode",
    "train.offpol_mom",
    "train.offpol_min_overlap",
    "train.offpol_min_trust",
]

# Continuous / ordinal params used for correlation (must be numeric)
CORR_KEYS = [
    "train.learning_rate",
    "train.ent_coef",
    "train.gamma",
    "train.gae_lambda",
    "train.replay_ratio",
    "train.clip_coef",
    "train.vf_coef",
    "train.vf_clip_coef",
    "train.max_grad_norm",
    "train.momentum",
    "train.prio_alpha",
    "train.prio_beta0",
    "train.total_timesteps",
    "train.horizon",
    "train.minibatch_size",
    "vec.total_agents",
    "env.frameskip",
    "train.offpol_min_overlap",
    "train.offpol_min_trust",
]

SEEDS_DEFAULT = [11, 22, 33, 44, 55]


def parse_ini(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = f"{section}.{k.strip()}" if section else k.strip()
            data[key] = v.strip()
    return data


def parse_metric_series(s: Optional[str]) -> List[float]:
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def load_high_score_runs(score_min: float) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob_sweep_logs()):
        d = parse_ini(path)
        scores = parse_metric_series(d.get("metrics.env/score"))
        if not scores:
            continue
        final = scores[-1]
        mx = max(scores)
        if final < score_min and mx < score_min:
            continue
        # Prefer final score for filtering; keep max too
        if final < score_min:
            continue
        ups = parse_metric_series(d.get("metrics.uptime"))
        hp = {}
        for k in HP_KEYS:
            if k in d:
                hp[k] = d[k]
        # defaults if missing from older logs
        hp.setdefault("vec.num_buffers", "1")
        hp.setdefault("vec.num_threads", "8")
        hp.setdefault("vec.gpu_env", "1")
        hp.setdefault("train.vtrace_rho_clip", "1.0")
        hp.setdefault("train.vtrace_c_clip", "1.0")
        base = os.path.basename(path)
        m = re.match(r"sweep_(\d+)_(\d+)\.ini", base)
        rows.append(
            {
                "path": path,
                "run_id": base[:-4],
                "run_idx": int(m.group(2)) if m else -1,
                "final_score": final,
                "max_score": mx,
                "uptime": ups[-1] if ups else None,
                "async": int(float(d.get("base.async", "1"))),
                "hp": hp,
            }
        )
    return rows


def glob_sweep_logs() -> List[str]:
    out = []
    for name in os.listdir(LOG_DIR):
        if re.match(r"sweep_\d+_\d+\.ini$", name):
            out.append(os.path.join(LOG_DIR, name))
    return out


def fnum(x: str) -> float:
    return float(x)


def sample_configs(
    runs: List[Dict[str, Any]], n: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """Stratified-ish sample: top scores + extremes of high-variance-suspect hparams + random."""
    if n >= len(runs):
        return list(runs)

    picked: Dict[str, Dict[str, Any]] = {}

    def add(r: Dict[str, Any]) -> None:
        picked[r["run_id"]] = r

    # Top by final score
    by_score = sorted(runs, key=lambda r: -r["final_score"])
    for r in by_score[: max(4, n // 6)]:
        add(r)

    # Extremes on each corr key
    for key in CORR_KEYS:
        ranked = sorted(
            runs,
            key=lambda r: fnum(r["hp"].get(key, "nan"))
            if r["hp"].get(key) not in (None, "")
            else float("nan"),
        )
        ranked = [r for r in ranked if r["hp"].get(key) not in (None, "")]
        if not ranked:
            continue
        add(ranked[0])
        add(ranked[-1])
        # median
        add(ranked[len(ranked) // 2])

    # Fill random
    rest = [r for r in runs if r["run_id"] not in picked]
    rng.shuffle(rest)
    for r in rest:
        if len(picked) >= n:
            break
        add(r)

    # Trim to n: prefer diversity — keep all extremes first then random
    out = list(picked.values())
    if len(out) > n:
        # keep score tops + extremes already added; shuffle and cut
        rng.shuffle(out)
        out = out[:n]
    return out


def run_train(cfg: Dict[str, Any], seed: int, tag: str) -> Optional[Dict[str, Any]]:
    run_id = f"sens_{tag}_{cfg['run_idx']:04d}_s{seed}"
    log_path = os.path.join(LOG_DIR, f"{run_id}.ini")
    overrides = [f"{k}={v}" for k, v in cfg["hp"].items()]
    cmd = [
        PUFFER,
        "train",
        "breakout",
        f"base.run_id={run_id}",
        f"base.seed={seed}",
        f"base.async={cfg['async']}",
        "base.checkpoint_interval=100000",
        *overrides,
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"  FAIL {run_id} rc={proc.returncode}: {proc.stderr[-300:]}", flush=True)
        return None
    if not os.path.isfile(log_path):
        print(f"  missing {log_path}", flush=True)
        return None
    d = parse_ini(log_path)
    scores = parse_metric_series(d.get("metrics.env/score"))
    ns = parse_metric_series(d.get("metrics.env/n"))
    ups = parse_metric_series(d.get("metrics.uptime"))
    return {
        "run_id": run_id,
        "src": cfg["run_id"],
        "seed": seed,
        "final_score": scores[-1] if scores else None,
        "max_score": max(scores) if scores else None,
        "n": ns[-1] if ns else None,
        "uptime": ups[-1] if ups else None,
        "wall": wall,
    }


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx < 1e-12 or deny < 1e-12:
        return float("nan")
    return num / (denx * deny)


def spearman(xs: List[float], ys: List[float]) -> float:
    def ranks(a: List[float]) -> List[float]:
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))


def residualize(y: List[float], x: List[float]) -> List[float]:
    """y - linear_fit(x)."""
    n = len(y)
    if n < 3:
        return y
    mx, my = mean(x), mean(y)
    varx = sum((xi - mx) ** 2 for xi in x)
    if varx < 1e-18:
        return [yi - my for yi in y]
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / varx
    a = my - b * mx
    return [yi - (a + b * xi) for xi, yi in zip(x, y)]


def analyze(sens: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("SEED SENSITIVITY vs HYPERPARAMS")
    print("=" * 72)
    print(f"configs measured: {len(sens)}")
    if len(sens) < 3:
        print("too few for correlation")
        return

    # Summary table
    print(f"\n{'src_run':<28} {'mean':>7} {'std':>7} {'range':>7} {'min':>7} {'max':>7} {'n':>3}")
    for s in sorted(sens, key=lambda r: -r["seed_std"]):
        print(
            f"{s['src']:<28} {s['seed_mean']:7.1f} {s['seed_std']:7.1f} "
            f"{s['seed_range']:7.1f} {s['seed_min']:7.1f} {s['seed_max']:7.1f} {s['n_ok']:3d}"
        )

    # Correlations with seed_std
    print("\n--- Correlation of hparam with seed_std (higher |r| => more tied to instability) ---")
    print(f"{'hparam':<28} {'pearson':>8} {'spearman':>8} {'partial*|score':>12}")
    seed_stds = [s["seed_std"] for s in sens]
    means = [s["seed_mean"] for s in sens]
    # partial: corr(hparam, residual of seed_std ~ mean_score)
    resid = residualize(seed_stds, means)

    rows = []
    for key in CORR_KEYS:
        xs = []
        ys = []
        rs = []
        for s, rstd in zip(sens, resid):
            if key not in s["hp"]:
                continue
            try:
                xv = fnum(s["hp"][key])
            except ValueError:
                continue
            xs.append(xv)
            ys.append(s["seed_std"])
            rs.append(rstd)
        if len(xs) < 5:
            continue
        pr = pearson(xs, ys)
        sp = spearman(xs, ys)
        # partial-ish: corr(hparam, residual seed_std after mean score)
        pp = pearson(xs, rs)
        rows.append((key, pr, sp, pp, abs(pp) if pp == pp else -1))

    rows.sort(key=lambda t: -t[4])
    for key, pr, sp, pp, _ in rows:
        print(f"{key:<28} {pr:8.3f} {sp:8.3f} {pp:12.3f}")

    # High vs low sensitivity cohorts
    print("\n--- High vs low seed_std cohort (top/bottom quartile) mean hparams ---")
    ordered = sorted(sens, key=lambda s: s["seed_std"])
    q = max(1, len(ordered) // 4)
    low, high = ordered[:q], ordered[-q:]
    print(f"low-std n={len(low)} mean_std={mean([s['seed_std'] for s in low]):.1f} "
          f"mean_score={mean([s['seed_mean'] for s in low]):.1f}")
    print(f"high-std n={len(high)} mean_std={mean([s['seed_std'] for s in high]):.1f} "
          f"mean_score={mean([s['seed_mean'] for s in high]):.1f}")
    print(f"{'hparam':<28} {'low_mean':>10} {'high_mean':>10} {'delta':>10}")
    for key in CORR_KEYS:
        def avg_key(group):
            vals = []
            for s in group:
                if key in s["hp"]:
                    try:
                        vals.append(fnum(s["hp"][key]))
                    except ValueError:
                        pass
            return mean(vals) if vals else float("nan")

        lo, hi = avg_key(low), avg_key(high)
        print(f"{key:<28} {lo:10.4g} {hi:10.4g} {hi-lo:10.4g}")

    # Suspect extremes: clip_coef near 1, high lr, etc.
    print("\n--- Slice: clip_coef >= 0.95 vs < 0.8 ---")
    hi_clip = [s for s in sens if fnum(s["hp"].get("train.clip_coef", "0")) >= 0.95]
    lo_clip = [s for s in sens if fnum(s["hp"].get("train.clip_coef", "0")) < 0.8]
    for name, g in [("clip>=0.95", hi_clip), ("clip<0.8", lo_clip)]:
        if g:
            print(
                f"  {name}: n={len(g)} seed_std mean={mean([s['seed_std'] for s in g]):.1f} "
                f"score mean={mean([s['seed_mean'] for s in g]):.1f}"
            )

    print("\n--- Slice: learning_rate >= 0.05 vs < 0.02 ---")
    hi_lr = [s for s in sens if fnum(s["hp"].get("train.learning_rate", "0")) >= 0.05]
    lo_lr = [s for s in sens if fnum(s["hp"].get("train.learning_rate", "0")) < 0.02]
    for name, g in [("lr>=0.05", hi_lr), ("lr<0.02", lo_lr)]:
        if g:
            print(
                f"  {name}: n={len(g)} seed_std mean={mean([s['seed_std'] for s in g]):.1f} "
                f"score mean={mean([s['seed_mean'] for s in g]):.1f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="small stratified sample")
    ap.add_argument("--full", action="store_true", help="all score>=threshold runs")
    ap.add_argument("--score-min", type=float, default=800.0)
    ap.add_argument("--sample", type=int, default=24, help="pilot sample size")
    ap.add_argument("--seeds", type=int, default=4, help="seeds per config")
    ap.add_argument("--seed-list", type=str, default="", help="comma seeds override")
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap configs after selection")
    ap.add_argument("--run-ids", type=str, default="",
                    help="comma-separated sweep run indices to evaluate")
    ap.add_argument("--after-ms", type=int, default=0,
                    help="only use sweep run IDs with timestamp >= this value")
    args = ap.parse_args()
    if not args.pilot and not args.full:
        args.pilot = True

    os.chdir(ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(PUFFER):
        print("missing ./puffer", file=sys.stderr)
        return 1

    runs = load_high_score_runs(args.score_min)
    if args.after_ms:
        runs = [r for r in runs
                if int(r["run_id"].split("_")[1]) >= args.after_ms]
    requested_ids = []
    if args.run_ids:
        requested_ids = [int(x) for x in args.run_ids.split(",") if x.strip()]
        by_idx = {r["run_idx"]: r for r in runs}
        missing = [idx for idx in requested_ids if idx not in by_idx]
        if missing:
            print(f"missing requested sweep runs: {missing}", file=sys.stderr)
            return 1
        runs = [by_idx[idx] for idx in requested_ids]
    print(f"Loaded {len(runs)} sweep runs with final score >= {args.score_min}")
    if not runs:
        return 1

    rng = random.Random(args.rng)
    if requested_ids:
        configs = list(runs)
        tag = "selected"
    elif args.full:
        configs = list(runs)
        tag = "full"
    else:
        configs = sample_configs(runs, args.sample, rng)
        tag = "pilot"
    if args.limit > 0:
        configs = configs[: args.limit]

    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
    else:
        seeds = SEEDS_DEFAULT[: args.seeds]

    print(f"Mode={tag} configs={len(configs)} seeds={seeds} "
          f"total_trains={len(configs)*len(seeds)}")

    # Save selection
    sel_path = os.path.join(OUT_DIR, f"{tag}_selected.csv")
    with open(sel_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "run_idx", "final_score", "async"] + CORR_KEYS)
        for c in configs:
            w.writerow(
                [c["run_id"], c["run_idx"], c["final_score"], c["async"]]
                + [c["hp"].get(k, "") for k in CORR_KEYS]
            )
    print(f"Wrote {sel_path}")

    sens: List[Dict[str, Any]] = []
    detail_rows = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg['run_id']} sweep_score={cfg['final_score']:.1f}", flush=True)
        scores = []
        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            r = run_train(cfg, seed, tag)
            if r and r["final_score"] is not None:
                scores.append(r["final_score"])
                detail_rows.append(
                    {
                        "src": cfg["run_id"],
                        "run_idx": cfg["run_idx"],
                        "seed": seed,
                        "final_score": r["final_score"],
                        "uptime": r["uptime"],
                        "wall": r["wall"],
                        "async": cfg["async"],
                        **{k: cfg["hp"].get(k, "") for k in CORR_KEYS},
                    }
                )
                print(f"    score={r['final_score']:.1f} wall={r['wall']:.1f}s", flush=True)
            else:
                print("    FAIL", flush=True)

        if len(scores) < 2:
            print("  skip (need >=2 successful seeds)", flush=True)
            continue
        sens.append(
            {
                "src": cfg["run_id"],
                "run_idx": cfg["run_idx"],
                "hp": cfg["hp"],
                "sweep_score": cfg["final_score"],
                "seed_mean": mean(scores),
                "seed_std": std(scores),
                "seed_min": min(scores),
                "seed_max": max(scores),
                "seed_range": max(scores) - min(scores),
                "n_ok": len(scores),
                "scores": scores,
            }
        )

    # Write results
    detail_path = os.path.join(OUT_DIR, f"{tag}_seed_details.csv")
    with open(detail_path, "w", newline="") as f:
        if detail_rows:
            fields = list(detail_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(detail_rows)
    print(f"\nWrote {detail_path}")

    sens_path = os.path.join(OUT_DIR, f"{tag}_seed_sensitivity.csv")
    with open(sens_path, "w", newline="") as f:
        fields = [
            "src", "run_idx", "sweep_score", "seed_mean", "seed_std",
            "seed_min", "seed_max", "seed_range", "n_ok", "scores",
        ] + CORR_KEYS
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sens:
            row = {
                "src": s["src"],
                "run_idx": s["run_idx"],
                "sweep_score": s["sweep_score"],
                "seed_mean": s["seed_mean"],
                "seed_std": s["seed_std"],
                "seed_min": s["seed_min"],
                "seed_max": s["seed_max"],
                "seed_range": s["seed_range"],
                "n_ok": s["n_ok"],
                "scores": " ".join(f"{x:.2f}" for x in s["scores"]),
            }
            for k in CORR_KEYS:
                row[k] = s["hp"].get(k, "")
            w.writerow(row)
    print(f"Wrote {sens_path}")

    analyze(sens)
    print(f"\nTotal wall {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
