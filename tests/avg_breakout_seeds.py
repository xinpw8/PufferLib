#!/usr/bin/env python3
"""Avg breakout env/score over N seeds. Extra CLI args forwarded as overrides."""
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUFFER = os.path.join(ROOT, "puffer")
LOG_DIR = os.path.join(ROOT, "logs", "breakout")
SEEDS = [11, 22, 33, 44, 55]


def parse_final_score(log_path):
    uptime = score = None
    in_m = False
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line == "[metrics]":
                in_m = True
                continue
            if in_m and line.startswith("["):
                break
            if not in_m or not line:
                continue
            if line.startswith("uptime"):
                uptime = [float(x) for x in line.split("=", 1)[1].split(",")]
            elif line.startswith("env/score"):
                score = [float(x) for x in line.split("=", 1)[1].split(",")]
    if not score:
        return None, None, None
    n = len(score)
    if uptime:
        n = min(n, len(uptime))
    return score[n - 1], max(score[:n]), (uptime[n - 1] if uptime else None)


def main():
    os.chdir(ROOT)
    if not os.path.isfile(PUFFER):
        print("missing ./puffer — build first", file=sys.stderr)
        return 1

    overrides = sys.argv[1:]
    tag = "is0"
    for a in overrides:
        if a.startswith("tag="):
            tag = a.split("=", 1)[1]
            overrides = [x for x in overrides if not x.startswith("tag=")]
            break

    print(f"breakout x {len(SEEDS)} seeds  tag={tag}", flush=True)
    if overrides:
        print(f"overrides: {overrides}", flush=True)

    rows = []
    t0 = time.time()
    for i, seed in enumerate(SEEDS, 1):
        run_id = f"avg_{tag}_s{seed}"
        log_path = os.path.join(LOG_DIR, f"{run_id}.ini")
        cmd = [
            PUFFER,
            "train",
            "breakout",
            f"base.run_id={run_id}",
            f"base.seed={seed}",
            "base.checkpoint_interval=100000",
            "base.wandb=0",
            *overrides,
        ]
        print(f"[{i}/{len(SEEDS)}] seed={seed} run_id={run_id}", flush=True)
        t_run = time.time()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        wall = time.time() - t_run
        if proc.returncode != 0:
            print(f"  FAIL rc={proc.returncode} ({wall:.1f}s)", flush=True)
            if proc.stderr:
                print(proc.stderr[-800:], flush=True)
            rows.append((seed, None, None, None, wall, False))
            continue
        if not os.path.isfile(log_path):
            print(f"  missing log {log_path}", flush=True)
            rows.append((seed, None, None, None, wall, False))
            continue
        final, mx, up = parse_final_score(log_path)
        print(
            f"  final={final:.1f}  max={mx:.1f}  uptime={up:.2f}s  wall={wall:.1f}s"
            if final is not None
            else f"  no score ({wall:.1f}s)",
            flush=True,
        )
        rows.append((seed, final, mx, up, wall, final is not None))

    finals = [r[1] for r in rows if r[5]]
    maxes = [r[2] for r in rows if r[5]]
    ups = [r[3] for r in rows if r[5] and r[3] is not None]
    print("-" * 56)
    print(f"tag={tag}  ok={len(finals)}/{len(SEEDS)}  total_wall={time.time()-t0:.1f}s")
    if not finals:
        print("no successful runs")
        return 1

    def fmt(name, xs):
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return f"{name}: mean={m:.1f}  std={s:.1f}  min={min(xs):.1f}  max={max(xs):.1f}"

    print(fmt("final", finals))
    print(fmt("max  ", maxes))
    if ups:
        print(fmt("uptime", ups))
    print("per seed:", ", ".join(
        f"{s}:{f:.0f}" if f is not None else f"{s}:FAIL" for s, f, *_ in rows
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
