#!/usr/bin/env python3
"""Reseed-test top new-sweep solves (incl. <1s 0844 and baseline run 0000)."""
import os, re, subprocess, sys, time, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUFFER = os.path.join(ROOT, "puffer")
LOG_DIR = os.path.join(ROOT, "logs", "breakout")
OLD_MAX = 1784664224203
SEEDS = [11, 22, 33, 44, 55]

# Top-of-list + near-misses under 1s from new sweep analysis
TARGETS = [
    (844, "only <1s solve (0.86s)"),
    (0, "rank2 time — clipped old-unstable 0974"),
    (606, "rank3 time 1.12s"),
    (263, "rank4 time 1.15s"),
    (617, "rank5 time 1.24s"),
    (328, "near-miss 845.8 under 1s"),
    (655, "near-miss 842.8 under 1s"),
]

HP_KEYS = [
    "vec.total_agents", "vec.num_buffers", "vec.num_threads", "vec.gpu_env",
    "env.frameskip", "policy.hidden_size", "policy.num_layers",
    "train.total_timesteps", "train.learning_rate", "train.gamma", "train.gae_lambda",
    "train.replay_ratio", "train.clip_coef", "train.vf_coef", "train.vf_clip_coef",
    "train.max_grad_norm", "train.ent_coef", "train.momentum", "train.minibatch_size",
    "train.horizon", "train.prio_alpha", "train.prio_beta0",
    "train.vtrace_rho_clip", "train.vtrace_c_clip",
]


def parse_ini(path):
    data, section = {}, None
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
            data[f"{section}.{k.strip()}"] = v.strip()
    return data


def find_run(idx):
    for name in os.listdir(LOG_DIR):
        m = re.match(r"sweep_(\d+)_(\d+)\.ini$", name)
        if m and int(m.group(1)) > OLD_MAX and int(m.group(2)) == idx:
            return os.path.join(LOG_DIR, name)
    return None


def load_cfg(idx):
    path = find_run(idx)
    if not path:
        raise FileNotFoundError(idx)
    d = parse_ini(path)
    sc = [float(x) for x in d["metrics.env/score"].split(",")]
    up = [float(x) for x in d["metrics.uptime"].split(",")]
    t850 = next((u for u, s in zip(up, sc) if s > 850), None)
    hp = {}
    for k in HP_KEYS:
        if k not in d:
            continue
        v = d[k]
        # int-valued policy dims (sweep may log floats)
        if k in ("policy.num_layers", "policy.hidden_size", "vec.total_agents",
                 "train.horizon", "train.minibatch_size", "env.frameskip"):
            v = str(int(float(v.replace("_", ""))))
        if k == "train.total_timesteps":
            v = str(int(float(v.replace("_", ""))))
        hp[k] = v
    hp.setdefault("vec.num_buffers", "1")
    hp.setdefault("vec.num_threads", "8")
    hp.setdefault("vec.gpu_env", "1")
    hp.setdefault("train.vtrace_rho_clip", "1.0")
    hp.setdefault("train.vtrace_c_clip", "1.0")
    return {
        "idx": idx,
        "src": os.path.basename(path)[:-4],
        "async": int(float(d.get("base.async", "1"))),
        "sweep_score": max(sc),
        "sweep_t850": t850,
        "hp": hp,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_train(cfg, seed):
    run_id = f"newtop_{cfg['idx']:04d}_s{seed}"
    log_path = os.path.join(LOG_DIR, f"{run_id}.ini")
    cmd = [
        PUFFER, "train", "breakout",
        f"base.run_id={run_id}",
        f"base.seed={seed}",
        f"base.async={cfg['async']}",
        "base.checkpoint_interval=100000",
        *[f"{k}={v}" for k, v in cfg["hp"].items()],
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAIL rc={proc.returncode}: {proc.stderr[-300:]}", flush=True)
        return None
    d = parse_ini(log_path)
    sc = [float(x) for x in d["metrics.env/score"].split(",")]
    return {"score": sc[-1], "max": max(sc), "wall": wall}


def main():
    os.chdir(ROOT)
    results = []
    print("=== Config snapshot (note run0 vs old 0974) ===\n")
    # Old unstable #1 0974 (from prior sweep; log may be gone)
    old0974 = {
        "horizon": "16", "max_grad_norm": "2.9562645", "clip_coef": "1",
        "learning_rate": "0.0778920427", "ent_coef": "0.00864126533",
    }
    cfg0 = load_cfg(0)
    print("run 0000 is modified old-unstable 0974:")
    print(f"  horizon:        16 → {cfg0['hp']['train.horizon']}")
    print(f"  max_grad_norm:  2.96 → {cfg0['hp']['train.max_grad_norm']}")
    print(f"  clip_coef:      1.0 → {cfg0['hp']['train.clip_coef']}")
    print(f"  (lr/ent/gamma/gae/replay/vf/prio mostly unchanged from 0974)")
    print()

    for idx, note in TARGETS:
        cfg = load_cfg(idx)
        print(f"\n=== run {idx:04d} — {note} ===", flush=True)
        print(f"  sweep: score={cfg['sweep_score']:.1f} t850={cfg['sweep_t850']}", flush=True)
        print(f"  H={cfg['hp'].get('policy.hidden_size')} L={cfg['hp'].get('policy.num_layers')} "
              f"hzn={cfg['hp']['train.horizon']} clip={cfg['hp']['train.clip_coef']} "
              f"lr={cfg['hp']['train.learning_rate']} ent={cfg['hp']['train.ent_coef']}", flush=True)
        scores = []
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            r = run_train(cfg, seed)
            if r:
                scores.append(r["score"])
                print(f"    score={r['score']:.1f} wall={r['wall']:.1f}s", flush=True)
            else:
                print("    FAIL", flush=True)
        if len(scores) >= 2:
            results.append({
                "idx": idx, "note": note, "sweep_score": cfg["sweep_score"],
                "sweep_t850": cfg["sweep_t850"], "scores": scores,
                "mean": mean(scores), "std": std(scores),
                "min": min(scores), "max": max(scores),
                "hp": cfg["hp"],
            })
            print(f"  → mean={mean(scores):.1f} std={std(scores):.1f} "
                  f"range={min(scores):.1f}-{max(scores):.1f} "
                  f"all≥800={sum(s>=800 for s in scores)}/{len(scores)} "
                  f"all≥850={sum(s>=850 for s in scores)}/{len(scores)}", flush=True)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'run':>5} {'sweep_t':>8} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} "
          f"{'≥800':>5} {'≥850':>5}  note")
    for r in results:
        print(f"{r['idx']:5d} {r['sweep_t850']:8.3f} {r['mean']:7.1f} {r['std']:7.1f} "
              f"{r['min']:7.1f} {r['max']:7.1f} "
              f"{sum(s>=800 for s in r['scores']):2d}/{len(r['scores'])} "
              f"{sum(s>=850 for s in r['scores']):2d}/{len(r['scores'])}  {r['note']}")
        print(f"       scores={[round(s,1) for s in r['scores']]}")

    out = os.path.join(LOG_DIR, "seed_sensitivity", "new_sweep_tops_reseed.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("idx,note,sweep_score,sweep_t850,mean,std,min,max,scores,"
                "hidden,layers,horizon,clip,lr,ent,grad_norm,agents,frameskip\n")
        for r in results:
            h = r["hp"]
            f.write(
                f"{r['idx']},{r['note']!r},{r['sweep_score']},{r['sweep_t850']},"
                f"{r['mean']},{r['std']},{r['min']},{r['max']},"
                f"\"{' '.join(f'{s:.2f}' for s in r['scores'])}\","
                f"{h.get('policy.hidden_size')},{h.get('policy.num_layers')},"
                f"{h.get('train.horizon')},{h.get('train.clip_coef')},"
                f"{h.get('train.learning_rate')},{h.get('train.ent_coef')},"
                f"{h.get('train.max_grad_norm')},{h.get('vec.total_agents')},"
                f"{h.get('env.frameskip')}\n"
            )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
