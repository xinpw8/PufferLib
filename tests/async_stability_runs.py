#!/usr/bin/env python3
"""Rerun top-3 sweep configs x async on/off x 5 seeds. Report mean final score."""
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUFFER = os.path.join(ROOT, "puffer")
LOG_DIR = os.path.join(ROOT, "logs", "breakout")

# Top 3 fastest to score>850 from latest breakout sweep
CONFIGS = {
    "rank1_0974": {
        "source": "sweep_1784663645458_0974",
        "t_850": 0.930,
        "overrides": [
            "vec.total_agents=4096",
            "vec.num_buffers=1",
            "vec.num_threads=8",
            "vec.gpu_env=1",
            "env.frameskip=3",
            "policy.hidden_size=64",
            "policy.num_layers=2",
            "train.total_timesteps=58121468",
            "train.learning_rate=0.0778920427",
            "train.gamma=0.967346191",
            "train.gae_lambda=0.983749211",
            "train.replay_ratio=1.69979477",
            "train.clip_coef=1",
            "train.vf_coef=3.98431253",
            "train.vf_clip_coef=3.75568104",
            "train.max_grad_norm=2.9562645",
            "train.ent_coef=0.00864126533",
            "train.momentum=0.835777402",
            "train.minibatch_size=65536",
            "train.horizon=16",
            "train.vtrace_rho_clip=1.0",
            "train.vtrace_c_clip=1.0",
            "train.prio_alpha=0.0706752539",
            "train.prio_beta0=0.550367117",
        ],
    },
    "rank2_0376": {
        "source": "sweep_1784662088346_0376",
        "t_850": 1.031,
        "overrides": [
            "vec.total_agents=4096",
            "vec.num_buffers=1",
            "vec.num_threads=8",
            "vec.gpu_env=1",
            "env.frameskip=3",
            "policy.hidden_size=64",
            "policy.num_layers=2",
            "train.total_timesteps=54874228",
            "train.learning_rate=0.100000001",
            "train.gamma=0.957717001",
            "train.gae_lambda=0.995000005",
            "train.replay_ratio=1.54686964",
            "train.clip_coef=1",
            "train.vf_coef=3.34388185",
            "train.vf_clip_coef=2.58338332",
            "train.max_grad_norm=1.05714464",
            "train.ent_coef=0.0123929428",
            "train.momentum=0.658640265",
            "train.minibatch_size=65536",
            "train.horizon=32",
            "train.vtrace_rho_clip=1.0",
            "train.vtrace_c_clip=1.0",
            "train.prio_alpha=0.202560604",
            "train.prio_beta0=0.802189231",
        ],
    },
    "rank3_1018": {
        "source": "sweep_1784663765815_1018",
        "t_850": 1.086,
        "overrides": [
            "vec.total_agents=4096",
            "vec.num_buffers=1",
            "vec.num_threads=8",
            "vec.gpu_env=1",
            "env.frameskip=3",
            "policy.hidden_size=64",
            "policy.num_layers=2",
            "train.total_timesteps=57544972",
            "train.learning_rate=0.100000001",
            "train.gamma=0.969757736",
            "train.gae_lambda=0.995000005",
            "train.replay_ratio=1.56889141",
            "train.clip_coef=1",
            "train.vf_coef=3.51655364",
            "train.vf_clip_coef=3.15033555",
            "train.max_grad_norm=1.9778024",
            "train.ent_coef=0.0140517252",
            "train.momentum=0.822255731",
            "train.minibatch_size=65536",
            "train.horizon=32",
            "train.vtrace_rho_clip=1.0",
            "train.vtrace_c_clip=1.0",
            "train.prio_alpha=0.14672488",
            "train.prio_beta0=0.552500606",
        ],
    },
}

SEEDS = [11, 22, 33, 44, 55]
ASYNC_MODES = [1, 0]


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

    jobs = []
    for cfg_name, cfg in CONFIGS.items():
        for async_v in ASYNC_MODES:
            for seed in SEEDS:
                run_id = f"stab_{cfg_name}_async{async_v}_s{seed}"
                jobs.append((cfg_name, async_v, seed, run_id, cfg["overrides"]))

    print(f"Running {len(jobs)} trains...", flush=True)
    results = []
    t0 = time.time()

    for i, (cfg_name, async_v, seed, run_id, overrides) in enumerate(jobs, 1):
        log_path = os.path.join(LOG_DIR, f"{run_id}.ini")
        cmd = [
            PUFFER,
            "train",
            "breakout",
            f"base.run_id={run_id}",
            f"base.seed={seed}",
            f"base.async={async_v}",
            "base.checkpoint_interval=100000",  # avoid checkpoint spam
            "base.wandb=0",
            *overrides,
        ]
        print(f"[{i:02d}/{len(jobs)}] {run_id}", flush=True)
        t_run = time.time()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.time() - t_run
        if proc.returncode != 0:
            print(f"  FAIL rc={proc.returncode} ({elapsed:.1f}s)", flush=True)
            if proc.stderr:
                print(proc.stderr[-800:], flush=True)
            results.append(
                {
                    "cfg": cfg_name,
                    "async": async_v,
                    "seed": seed,
                    "run_id": run_id,
                    "final": None,
                    "max": None,
                    "uptime": None,
                    "wall": elapsed,
                    "ok": False,
                }
            )
            continue
        if not os.path.isfile(log_path):
            print(f"  missing log {log_path}", flush=True)
            results.append(
                {
                    "cfg": cfg_name,
                    "async": async_v,
                    "seed": seed,
                    "run_id": run_id,
                    "final": None,
                    "max": None,
                    "uptime": None,
                    "wall": elapsed,
                    "ok": False,
                }
            )
            continue
        final, mx, up = parse_final_score(log_path)
        print(
            f"  score={final:.1f} max={mx:.1f} uptime={up:.2f}s wall={elapsed:.1f}s",
            flush=True,
        )
        results.append(
            {
                "cfg": cfg_name,
                "async": async_v,
                "seed": seed,
                "run_id": run_id,
                "final": final,
                "max": mx,
                "uptime": up,
                "wall": elapsed,
                "ok": True,
            }
        )

    print(f"\nAll done in {time.time() - t0:.1f}s wall\n")

    # Per-run table
    print(f"{'cfg':<14} {'async':>5} {'seed':>4} {'final':>8} {'max':>8} {'uptime':>7}")
    for r in results:
        if r["ok"]:
            print(
                f"{r['cfg']:<14} {r['async']:>5} {r['seed']:>4} "
                f"{r['final']:8.1f} {r['max']:8.1f} {r['uptime']:7.2f}"
            )
        else:
            print(f"{r['cfg']:<14} {r['async']:>5} {r['seed']:>4}     FAIL")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    def std(xs):
        xs = [x for x in xs if x is not None]
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    print("\n=== Mean final score by config x async ===")
    print(f"{'cfg':<14} {'async':>5} {'n':>3} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for cfg_name in CONFIGS:
        for async_v in ASYNC_MODES:
            xs = [
                r["final"]
                for r in results
                if r["cfg"] == cfg_name and r["async"] == async_v and r["ok"]
            ]
            if not xs:
                print(f"{cfg_name:<14} {async_v:>5}   0      n/a")
                continue
            print(
                f"{cfg_name:<14} {async_v:>5} {len(xs):>3} "
                f"{mean(xs):8.1f} {std(xs):8.1f} {min(xs):8.1f} {max(xs):8.1f}"
            )

    print("\n=== Overall mean final score: async on vs off ===")
    for async_v, label in [(1, "async=1"), (0, "async=0")]:
        xs = [r["final"] for r in results if r["async"] == async_v and r["ok"]]
        print(
            f"  {label}: n={len(xs)}  mean={mean(xs):.1f}  std={std(xs):.1f}  "
            f"min={min(xs) if xs else float('nan'):.1f}  max={max(xs) if xs else float('nan'):.1f}"
        )

    print("\n=== Overall mean max score during run ===")
    for async_v, label in [(1, "async=1"), (0, "async=0")]:
        xs = [r["max"] for r in results if r["async"] == async_v and r["ok"]]
        print(
            f"  {label}: n={len(xs)}  mean={mean(xs):.1f}  std={std(xs):.1f}"
        )

    # hits >850
    print("\n=== Fraction final score > 850 ===")
    for async_v, label in [(1, "async=1"), (0, "async=0")]:
        xs = [r["final"] for r in results if r["async"] == async_v and r["ok"]]
        hits = sum(1 for x in xs if x > 850)
        print(f"  {label}: {hits}/{len(xs)}")

    out_csv = os.path.join(ROOT, "logs", "breakout", "async_stability_results.csv")
    with open(out_csv, "w") as f:
        f.write("cfg,async,seed,run_id,final,max,uptime,wall,ok\n")
        for r in results:
            f.write(
                f"{r['cfg']},{r['async']},{r['seed']},{r['run_id']},"
                f"{r['final']},{r['max']},{r['uptime']},{r['wall']},{int(r['ok'])}\n"
            )
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
