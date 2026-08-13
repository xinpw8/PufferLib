#!/usr/bin/env python3
"""
Decompose top-3 score variance into train seed / nondeterminism.
Uses in-process post-train eval. async=0 (more stable from prior study).
"""
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUFFER = os.path.join(ROOT, "puffer")
LOG_DIR = os.path.join(ROOT, "logs", "breakout")

CONFIGS = {
    "rank1_0974": [
        "vec.total_agents=4096", "vec.num_buffers=1", "vec.num_threads=8", "vec.gpu_env=1",
        "env.frameskip=3", "policy.hidden_size=64", "policy.num_layers=2",
        "train.total_timesteps=58121468", "train.learning_rate=0.0778920427",
        "train.gamma=0.967346191", "train.gae_lambda=0.983749211",
        "train.replay_ratio=1.69979477", "train.clip_coef=1", "train.vf_coef=3.98431253",
        "train.vf_clip_coef=3.75568104", "train.max_grad_norm=2.9562645",
        "train.ent_coef=0.00864126533", "train.momentum=0.835777402",
        "train.minibatch_size=65536", "train.horizon=16",
        "train.prio_alpha=0.0706752539", "train.prio_beta0=0.550367117",
    ],
    "rank2_0376": [
        "vec.total_agents=4096", "vec.num_buffers=1", "vec.num_threads=8", "vec.gpu_env=1",
        "env.frameskip=3", "policy.hidden_size=64", "policy.num_layers=2",
        "train.total_timesteps=54874228", "train.learning_rate=0.100000001",
        "train.gamma=0.957717001", "train.gae_lambda=0.995000005",
        "train.replay_ratio=1.54686964", "train.clip_coef=1", "train.vf_coef=3.34388185",
        "train.vf_clip_coef=2.58338332", "train.max_grad_norm=1.05714464",
        "train.ent_coef=0.0123929428", "train.momentum=0.658640265",
        "train.minibatch_size=65536", "train.horizon=32",
        "train.prio_alpha=0.202560604", "train.prio_beta0=0.802189231",
    ],
    "rank3_1018": [
        "vec.total_agents=4096", "vec.num_buffers=1", "vec.num_threads=8", "vec.gpu_env=1",
        "env.frameskip=3", "policy.hidden_size=64", "policy.num_layers=2",
        "train.total_timesteps=57544972", "train.learning_rate=0.100000001",
        "train.gamma=0.969757736", "train.gae_lambda=0.995000005",
        "train.replay_ratio=1.56889141", "train.clip_coef=1", "train.vf_coef=3.51655364",
        "train.vf_clip_coef=3.15033555", "train.max_grad_norm=1.9778024",
        "train.ent_coef=0.0140517252", "train.momentum=0.822255731",
        "train.minibatch_size=65536", "train.horizon=32",
        "train.prio_alpha=0.14672488", "train.prio_beta0=0.552500606",
    ],
}

SEEDS = [11, 22, 33, 44, 55]
# How much longer than default post-train eval (default ≈ train_epochs/2)
# Same-seed repeats for nondeterminism / train noise (one config)
REPEAT_SEED = 22
REPEAT_N = 5
REPEAT_CFG = "rank2_0376"


def parse_log(path):
    vals = {}
    in_m = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "[metrics]":
                in_m = True
                continue
            if in_m and line.startswith("["):
                break
            if not in_m or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = [float(x) for x in v.split(",")]
    score = vals.get("env/score", [])
    n = vals.get("env/n", [])
    up = vals.get("uptime", [])
    ep = vals.get("epoch", [])
    if not score:
        return None
    i = len(score) - 1
    return {
        "final_score": score[i],
        "final_n": n[i] if n else None,
        "uptime": up[i] if up else None,
        "epoch": ep[i] if ep else None,
        "score_series": score,
        "n_series": n,
        # last train-ish point: where agent_steps still increasing — use max score before plateau
        "max_score": max(score),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def run_train(run_id, seed, overrides, async_v=0):
    log_path = os.path.join(LOG_DIR, f"{run_id}.ini")
    cmd = [
        PUFFER, "train", "breakout",
        f"base.run_id={run_id}",
        f"base.seed={seed}",
        f"base.async={async_v}",
        "base.eval_episodes=1000000",  # don't stop early on episode count
        "base.checkpoint_interval=100000",
        *overrides,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"  FAIL {run_id} rc={proc.returncode} {proc.stderr[-400:]}", flush=True)
        return None
    if not os.path.isfile(log_path):
        print(f"  missing log {log_path}", flush=True)
        return None
    r = parse_log(log_path)
    r["wall"] = wall
    r["run_id"] = run_id
    return r


def main():
    os.chdir(ROOT)
    results = []  # dict records

    # --- Part A: fixed seed, each config ---
    print("=== Part A: fixed seed, each config ===", flush=True)
    for cfg, ov in CONFIGS.items():
        run_id = f"noiseA_{cfg}_s{REPEAT_SEED}"
        print(f"[A] {run_id}", flush=True)
        r = run_train(run_id, REPEAT_SEED, ov)
        if r:
            print(
                f"  score={r['final_score']:.1f} n={r['final_n']:.0f} "
                f"uptime={r['uptime']:.2f}s wall={r['wall']:.1f}s",
                flush=True,
            )
            results.append({"part": "A", "cfg": cfg, "seed": REPEAT_SEED, **r})

    # --- Part B: seed variance ---
    print("\n=== Part B: 5 seeds, all configs ===", flush=True)
    for cfg, ov in CONFIGS.items():
        for seed in SEEDS:
            run_id = f"noiseB_{cfg}_s{seed}"
            print(f"[B] {run_id}", flush=True)
            r = run_train(run_id, seed, ov)
            if r:
                print(
                    f"  score={r['final_score']:.1f} n={r['final_n']:.0f} "
                    f"uptime={r['uptime']:.2f}s",
                    flush=True,
                )
                results.append({"part": "B", "cfg": cfg, "seed": seed, **r})

    # --- Part C: same seed repeated (train nondeterminism) ---
    print("\n=== Part C: same seed × 5 repeats (rank2) ===", flush=True)
    for rep in range(REPEAT_N):
        run_id = f"noiseC_{REPEAT_CFG}_s{REPEAT_SEED}_r{rep}"
        print(f"[C] {run_id}", flush=True)
        r = run_train(run_id, REPEAT_SEED, CONFIGS[REPEAT_CFG])
        if r:
            print(f"  score={r['final_score']:.1f} n={r['final_n']:.0f}", flush=True)
            results.append({"part": "C", "cfg": REPEAT_CFG, "seed": REPEAT_SEED,
                            "rep": rep, **r})

    # ---------- Report ----------
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)

    print("\n--- A: fixed seed ---")
    print(f"{'cfg':<14} {'n_ep':>8} {'score':>8} {'uptime':>7}")
    for cfg in CONFIGS:
        rows = [r for r in results if r["part"] == "A" and r["cfg"] == cfg]
        if not rows:
            continue
        r = rows[0]
        print(f"{cfg:<14} {r['final_n']:8.0f} {r['final_score']:8.1f} {r['uptime']:7.2f}")

    print("\n--- B: across-seed spread ---")
    print(f"{'cfg':<14} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'range':>8}")
    for cfg in CONFIGS:
        xs = [r["final_score"] for r in results
              if r["part"] == "B" and r["cfg"] == cfg]
        if not xs:
            continue
        print(
            f"{cfg:<14} {mean(xs):8.1f} {std(xs):8.1f} "
            f"{min(xs):8.1f} {max(xs):8.1f} {max(xs)-min(xs):8.1f}"
        )

    print("\n--- C: same-seed train repeats (rank2) ---")
    xs = [r["final_score"] for r in results if r["part"] == "C"]
    if xs:
        print(f"  n={len(xs)} mean={mean(xs):.1f} std={std(xs):.1f} "
              f"min={min(xs):.1f} max={max(xs):.1f} range={max(xs)-min(xs):.1f}")
        print(f"  scores: {[round(x,1) for x in xs]}")

    print("\n--- Across-seed std ---")
    for cfg in CONFIGS:
        seed_scores = [r["final_score"] for r in results
                       if r["part"] == "B" and r["cfg"] == cfg]
        print(f"  {cfg}: across-seed std={std(seed_scores):.1f}  "
              f"scores={[round(x,1) for x in seed_scores]}")
    if xs:
        print(f"  same-seed train std (rank2)={std(xs):.1f}")

    out = os.path.join(LOG_DIR, "score_noise_study.csv")
    with open(out, "w") as f:
        f.write("part,cfg,seed,rep,final_score,final_n,uptime,wall,run_id\n")
        for r in results:
            f.write(
                f"{r.get('part')},{r.get('cfg')},{r.get('seed')},"
                f"{r.get('rep','')},{r.get('final_score')},{r.get('final_n')},"
                f"{r.get('uptime')},{r.get('wall')},{r.get('run_id')}\n"
            )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
