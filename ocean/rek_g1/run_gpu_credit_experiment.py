"""Matched native-Muon credit-span experiment using the production duel trainer.

The short/long arms differ only in gamma, lambda and rollout/RNN horizon.
Both hold LR constant, use identical rewards/masks/opponent, and pin input weights.
No claim about policy quality is inferred from the online training win rate.
"""
from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolved_config(default_config, native_config, credit_config):
    parser = configparser.ConfigParser()
    paths = [str(default_config), str(native_config), str(credit_config)]
    if parser.read(paths) != paths:
        raise FileNotFoundError("all three experiment configuration files are required")
    return parser


def numeric_plan(parser, *, physical_fighters, total_timesteps, minibatch_size):
    if physical_fighters <= 0 or physical_fighters % 2:
        raise ValueError("physical fighters must contain complete pairs")
    learners = physical_fighters // 2
    train = parser["train"]
    horizon = int(train["horizon"])
    gamma, lam = float(train["gamma"]), float(train["gae_lambda"])
    lr = float(train["learning_rate"])
    if not 0 < gamma < 1 or not 0 < lam < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("discounts and learning rate must be finite and positive")
    if ast.literal_eval(train["anneal_lr"]) or not ast.literal_eval(parser["base"]["reset_state"]):
        raise ValueError("matched experiment requires constant LR and horizon memory reset")
    if horizon not in (64, 256):
        raise ValueError("only the declared 64/256-tick credit arms are supported")
    rollout_steps = learners * horizon
    if total_timesteps <= 0 or total_timesteps % rollout_steps:
        raise ValueError("total timesteps must contain complete rollouts")
    if minibatch_size <= 0 or minibatch_size % horizon or rollout_steps % minibatch_size:
        raise ValueError("minibatch size must partition a rollout into whole sequences")
    replay_ratio = float(train["replay_ratio"])
    if replay_ratio not in (1.0, 4.0):
        raise ValueError("matched experiment supports validated replay ratios1 and4")
    return {
        "physical_fighters": physical_fighters, "learners": learners,
        "learner_steps": total_timesteps, "horizon_ticks": horizon,
        "horizon_seconds": horizon * .02,
        "reset_state_each_horizon": True, "gamma": gamma, "gae_lambda": lam,
        "discount_efold_seconds": -.02 / math.log(gamma),
        "gae_efold_seconds": -.02 / math.log(gamma * lam),
        "direct_gae_weight_at_3_seconds": (gamma * lam) ** 150,
        "direct_gae_has_3_second_span": horizon > 150,
        "learning_rate": lr, "anneal_lr": False,
        "rollouts": total_timesteps // rollout_steps,
        "optimizer_minibatches": int(replay_ratio) * (total_timesteps // minibatch_size),
        "replay_ratio": replay_ratio,
        "prioritized_sampling": "minibatch draws with replacement, not exact visits per sample",
        "minibatch_size": minibatch_size, "simulated_seconds_per_arena": total_timesteps / learners * .02,
        "base_seed": int(parser["base"]["seed"]),
        "native_network": "MinGRU", "native_optimizer": "Muon",
        "muon_momentum": float(train["beta1"]),
        "muon_weight_decay": 0.0,
        "vf_coef": float(train["vf_coef"]), "entropy_coef": float(train["ent_coef"]),
        "max_grad_norm": float(train["max_grad_norm"]),
        "reward": "unclipped native learner score delta minus opponent score delta",
        "opponent": "candidate-dummy", "opponent_changed": False,
        "checkpoint_resume_scope": "policy weights only; fresh native optimizer state",
    }


def effective_native_parameters(parser, plan):
    result = {}
    for section in parser.sections():
        target = result if section == "base" else result.setdefault(section, {})
        for key, text in parser[section].items():
            try:
                target[key] = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                target[key] = text
    result["vec"].update(total_agents=plan["learners"], num_buffers=1, num_threads=0)
    result["train"].update(total_timesteps=plan["learner_steps"], horizon=plan["horizon_ticks"],
                           minibatch_size=plan["minibatch_size"], gpus=1, reward_clip=0.0)
    # JSON representation of the production single-device nccl_id=b"" value.
    result.update(env_name="rek_g1", world_size=1, rank=0, nccl_id="", gpu_id=0)
    return result


def run(args):
    expected_sha = args.checkpoint_sha256.lower()
    if len(expected_sha) != 64 or digest(args.load_checkpoint) != expected_sha:
        raise ValueError("initial checkpoint does not match the pinned SHA256")
    if args.run_dir.exists() or args.output.exists():
        raise FileExistsError("experiment run directory or report already exists")
    parser = resolved_config(args.default_config, args.native_config, args.credit_config)
    plan = numeric_plan(parser, physical_fighters=args.total_agents,
                        total_timesteps=args.total_timesteps, minibatch_size=args.minibatch_size)
    inputs = args.run_dir.with_name(args.run_dir.name + ".inputs")
    inputs.mkdir(parents=True, exist_ok=False)
    materialized = inputs / "resolved-native.ini"
    with materialized.open("x", encoding="utf-8") as output:
        parser.write(output)
    plan.update({"schema": "rek.g1_matched_credit_span.v1",
                 "policy_observation_encoder_name": getattr(args, "policy_observation_encoder", "raw"),
                 "policy_observation_initialization": getattr(args, "policy_observation_warm_start", "matching-checkpoint"),
                 "initial_checkpoint": str(args.load_checkpoint.resolve()),
                 "initial_checkpoint_sha256": expected_sha,
                 "configuration_files": {
                     str(path.resolve()): digest(path) for path in
                     (args.default_config, args.native_config, args.credit_config, args.gpu_duel_config)
                 }, "resolved_config_sha256": digest(materialized),
                 "effective_native_parameters": effective_native_parameters(parser, plan)})
    plan_path = inputs / "experiment.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_plan": plan}), flush=True)
    if args.prepare_only:
        return plan
    # Importing the trainer is delayed so preparation and CPU tests need no CUDA.
    from train_gpu_duel import train
    training_args = SimpleNamespace(
        default_config=args.default_config, native_config=materialized,
        gpu_duel_config=args.gpu_duel_config, total_agents=args.total_agents,
        total_timesteps=args.total_timesteps, horizon=plan["horizon_ticks"],
        minibatch_size=args.minibatch_size, reward_clip=0.0, opponent="candidate-dummy",
        log_every=args.log_every, checkpoint_every=args.checkpoint_every,
        check_every_step=False, load_checkpoint=args.load_checkpoint,
        load_checkpoint_sha256=expected_sha,
        policy_observation_encoder=getattr(args, "policy_observation_encoder", "raw"),
        policy_observation_warm_start=getattr(args, "policy_observation_warm_start", "matching-checkpoint"),
        run_dir=args.run_dir, output=args.output,
    )
    result = train(training_args)
    if result["checkpoints"]["initial"]["sha256"] != expected_sha:
        raise RuntimeError("production trainer started from different weights")
    if result["training"]["agent_steps"] != plan["learner_steps"]:
        raise RuntimeError("actual learner-step count differs from experiment plan")
    result["credit_span_experiment"] = plan
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("default-config", "native-config", "credit-config", "gpu-duel-config",
                  "load-checkpoint", "run-dir", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--policy-observation-encoder", default="raw",
                        choices=("raw", "polar_xy_v1", "scaled_polar_xy_v1"))
    parser.add_argument("--policy-observation-warm-start", default="matching-checkpoint",
                        choices=("matching-checkpoint", "raw-initial-weights"))
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--total-timesteps", type=int, default=3276800)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
