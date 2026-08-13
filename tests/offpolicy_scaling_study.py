#!/usr/bin/env python3
"""Measure effective policy-update reuse across horizon, minibatch size, and RR.

Each run starts from the same policy checkpoint and performs two async rollout
epochs. Epoch 1 is fresh; epoch 2 was prefetched before epoch 1 trained and is
therefore one learner epoch stale. The trainer writes the raw per-minibatch
trace and this script writes a compact, one-row-per-epoch summary.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUFFER = os.path.join(ROOT, "puffer")
LOG_DIR = os.path.join(ROOT, "logs", "breakout")
DEFAULT_CHECKPOINT = os.path.join(
    ROOT,
    "checkpoints",
    "breakout",
    "gold_floorp6_low",
    "0000000032571392.bin",
)
MILESTONES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclass
class Case:
    axes: set[str]
    horizon: int
    agents: int
    minibatch: int
    replay_ratio: int

    @property
    def batch(self) -> int:
        return self.horizon * self.agents

    @property
    def name(self) -> str:
        return (
            f"h{self.horizon:03d}_a{self.agents:05d}_"
            f"mb{self.minibatch:05d}_rr{self.replay_ratio:03d}"
        )


def study_cases(selected_axes: set[str]) -> list[Case]:
    cases: dict[tuple[int, int, int, int], Case] = {}

    def add(axis: str, horizon: int, agents: int, minibatch: int, rr: int) -> None:
        if axis not in selected_axes:
            return
        key = (horizon, agents, minibatch, rr)
        if key in cases:
            cases[key].axes.add(axis)
        else:
            cases[key] = Case({axis}, horizon, agents, minibatch, rr)

    # Hold batch=65,536 and minibatch=32,768 fixed. This isolates sequence
    # length while keeping samples/epoch and optimizer updates/RR constant.
    for horizon in (8, 16, 32, 64, 128, 256):
        add("horizon", horizon, 65536 // horizon, 32768, 128)

    # Hold actor count fixed so batch/update interval scales with horizon.
    # Keeping minibatch at half-batch preserves two optimizer updates per RR.
    for horizon in (8, 16, 32, 64, 128, 256):
        batch = 2048 * horizon
        add("fixed_agents", horizon, 2048, batch // 2, 128)

    # Hold horizon, agents, batch, and nominal sample reuse fixed. Update count
    # changes inversely with minibatch size, as it does in the real algorithm.
    for minibatch in (8192, 16384, 32768, 65536):
        add("minibatch", 32, 2048, minibatch, 128)

    # Prefix/convergence check: these runs otherwise have identical configs.
    for rr in (1, 2, 4, 8, 32, 128, 512):
        add("replay", 32, 2048, 32768, rr)

    return list(cases.values())


def run_case(case: Case, checkpoint: str, seed: int, rerun: bool) -> str:
    run_id = f"offpol_scale_{case.name}_s{seed}"
    trace_path = os.path.join(LOG_DIR, f"{run_id}.reuse.csv")
    if os.path.isfile(trace_path) and not rerun:
        print(f"[cached] {case.name} seed={seed}", flush=True)
        return trace_path

    cmd = [
        PUFFER,
        "train",
        "breakout",
        f"base.run_id={run_id}",
        f"base.seed={seed}",
        f"base.load_model_path={checkpoint}",
        "base.eval_episodes=1000",
        "base.checkpoint_interval=100000",
        "base.wandb=0",
        f"vec.total_agents={case.agents}",
        f"train.total_timesteps={2 * case.batch}",
        "train.anneal_lr=0",
        f"train.horizon={case.horizon}",
        f"train.minibatch_size={case.minibatch}",
        f"train.replay_ratio={case.replay_ratio}",
        "train.muon_grad_scale=1",
        "train.advantage_is=1",
        "train.offpol_mom=1",
        "train.offpol_min_overlap=0.8",
        "train.offpol_min_trust=0.5",
        "train.reuse_logging=1",
        "train.ablation_logging=0",
        "sweep.downsample=1",
    ]
    print(
        f"[run] {case.name} seed={seed} axes={','.join(sorted(case.axes))}",
        flush=True,
    )
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"training failed for {case.name}: rc={proc.returncode}")
    if not os.path.isfile(trace_path):
        raise RuntimeError(f"missing trace for {case.name}: {trace_path}")
    print(f"      {time.time() - started:.2f}s", flush=True)
    return trace_path


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summarize(case: Case, seed: int, trace_path: str) -> list[dict[str, object]]:
    with open(trace_path, newline="") as fp:
        raw = list(csv.DictReader(fp))
    out = []
    for epoch in (1, 2):
        rows = [r for r in raw if int(r["epoch"]) == epoch]
        if not rows:
            raise RuntimeError(f"{trace_path} has no epoch {epoch}")
        last = rows[-1]
        zero = next((r for r in rows if f(r, "trust") == 0.0), None)
        nonzero = [r for r in rows if f(r, "step_scale") > 0.0]
        summary: dict[str, object] = {
            "case": case.name,
            "seed": seed,
            "axes": "+".join(sorted(case.axes)),
            "horizon": case.horizon,
            "agents": case.agents,
            "batch": case.batch,
            "minibatch": case.minibatch,
            "nominal_rr": case.replay_ratio,
            "epoch": epoch,
            "stale_epochs": epoch - 1,
            "updates": len(rows),
            "base_kl": f(rows[0], "kl"),
            "max_abs_kl_delta": max(abs(f(r, "kl_delta")) for r in rows),
            "final_abs_kl_delta": abs(f(last, "kl_delta")),
            "first_zero_rr": f(zero, "sample_rr") if zero else "",
            "last_nonzero_rr": f(nonzero[-1], "sample_rr") if nonzero else 0.0,
            "effective_trust_rr": f(last, "effective_trust_rr"),
            "effective_step_rr": f(last, "effective_step_rr"),
            "weight_path_l2": f(last, "cumulative_weight_path_l2"),
            "weight_displacement_l2": f(last, "weight_displacement_l2"),
            "effective_step_fraction": (
                f(last, "effective_step_rr") / case.replay_ratio
            ),
        }
        for milestone in MILESTONES:
            prefix = [r for r in rows if f(r, "sample_rr") <= milestone]
            summary[f"step_rr_at_{milestone}"] = (
                f(prefix[-1], "effective_step_rr") if prefix else ""
            )
            summary[f"abs_kl_delta_at_{milestone}"] = (
                abs(f(prefix[-1], "kl_delta")) if prefix else ""
            )
            summary[f"weight_path_l2_at_{milestone}"] = (
                f(prefix[-1], "cumulative_weight_path_l2") if prefix else ""
            )
            summary[f"weight_displacement_l2_at_{milestone}"] = (
                f(prefix[-1], "weight_displacement_l2") if prefix else ""
            )
        out.append(summary)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        action="append",
        choices=("horizon", "fixed_agents", "minibatch", "replay"),
        help="axis to run (repeatable; default: all)",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--seeds",
        default="11,22,33,44,55",
        help="comma-separated rollout seeds",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(LOG_DIR, "offpolicy_scaling_summary.csv"),
    )
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x]
    axes = set(args.axis or ("horizon", "fixed_agents", "minibatch", "replay"))
    cases = study_cases(axes)
    if not os.path.isfile(PUFFER):
        parser.error("missing ./puffer; build the GPU trainer first")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"missing checkpoint: {args.checkpoint}")

    summaries: list[dict[str, object]] = []
    for case in cases:
        for seed in seeds:
            trace_path = run_case(case, args.checkpoint, seed, args.rerun)
            summaries.extend(summarize(case, seed, trace_path))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print("\ncase                               seed ep stale cutoff  eff_step path_L2  net_L2 fraction")
    for row in summaries:
        cutoff = row["first_zero_rr"]
        cutoff_s = f"{float(cutoff):6.1f}" if cutoff != "" else "     -"
        print(
            f"{row['case']:<34} {row['seed']:>4} {row['epoch']:>2} {row['stale_epochs']:>5} "
            f"{cutoff_s} {row['effective_step_rr']:>9.3f} "
            f"{row['weight_path_l2']:>9.3f} "
            f"{row['weight_displacement_l2']:>7.3f} "
            f"{100 * float(row['effective_step_fraction']):>7.2f}%"
        )
    print(f"\nsummary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
