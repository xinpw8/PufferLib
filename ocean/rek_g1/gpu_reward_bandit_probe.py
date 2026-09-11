"""Bounded native-PPO reward-learning diagnostic; no REK physics or training claim.

The native 223-observation/33-action MinGRU policy sees two legal actions.
Action zero gives +1 and action one gives -1. Every transition is terminal.
Frozen sampling before and after learning measures whether the rewarded action
became more probable. All simulator state and reward calculation stay on CUDA.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bandit_config(native, *, updates, learning_rate, agents=32, replay_ratio=1):
    if updates <= 0 or agents <= 0 or not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("updates/agents must be positive and LR finite nonnegative")
    if replay_ratio not in (1, 4):
        raise ValueError("diagnostic supports replay ratios1 and4")
    result = deepcopy(native)
    result.update(reset_state=True, world_size=1, rank=0, nccl_id=b"", gpu_id=0)
    result["vec"].update(total_agents=agents, num_buffers=1, num_threads=0)
    result["train"].update(total_timesteps=updates * agents * 64, horizon=64,
                           minibatch_size=agents * 64, replay_ratio=float(replay_ratio),
                           learning_rate=learning_rate, anneal_lr=False,
                           reward_clip=0.0, gpus=1)
    return result


class RewardBandit:
    def __init__(self, agents=32, *, device="cuda"):
        self.observations = torch.zeros((agents, 223), device=device)
        self.rewards = torch.zeros(agents, device=device)
        self.terminals = torch.zeros_like(self.rewards)
        self.action_mask = torch.zeros((agents, 33), device=device, dtype=torch.uint8)
        self.counts = torch.zeros(4, dtype=torch.float64, device=device)
        self.reset()

    def reset(self):
        self.observations.zero_()
        self.observations[:, 0] = 1
        self.rewards.zero_()
        self.terminals.zero_()
        self.action_mask.zero_()
        self.action_mask[:, :2] = 1
        self.counts.zero_()

    def step(self, actions):
        selected = actions[:, 0]
        correct = selected == 0
        self.rewards.copy_(torch.where(correct, 1.0, -1.0))
        self.terminals.fill_(1)
        self.counts[0].add_(selected.numel())
        self.counts[1].add_(correct.sum())
        self.counts[2].add_(self.rewards.sum())
        self.counts[3].add_(((selected < 0) | (selected > 1)).sum())

    def snapshot(self):
        steps, rewarded, rewards, invalid = self.counts.detach().cpu().tolist()
        return {"samples": int(steps), "rewarded_actions": int(rewarded),
                "rewarded_action_fraction": None if not steps else rewarded / steps,
                "mean_reward": None if not steps else rewards / steps,
                "invalid_actions": int(invalid)}


def run(args):
    if digest(args.checkpoint) != args.checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA256 differs from expected")
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if args.evaluation_rollouts < 1:
        raise ValueError("evaluation_rollouts must be positive")
    from train_gpu_duel import load_native_config
    from gpu_native_puffer import NativeExternalGpuPuffer
    from gpu_puffer_env import CudaTensorEnvAdapter
    native = bandit_config(load_native_config(args.default_config, args.native_config),
                           updates=args.updates, learning_rate=args.learning_rate,
                           agents=args.agents, replay_ratio=args.replay_ratio)
    args.run_dir.mkdir(parents=True)
    raw = RewardBandit(args.agents)
    trainer = NativeExternalGpuPuffer(native, CudaTensorEnvAdapter(raw, (33,)))
    curve = []
    try:
        trainer.load_weights(args.checkpoint)
        initial = args.run_dir / "initial.bin"
        trainer.save_weights(initial)
        if digest(initial) != args.checkpoint_sha256.lower():
            raise RuntimeError("native loaded weights differ from the pinned checkpoint")

        def frozen_sample(name):
            before = args.run_dir / f"{name}-before.bin"
            after = args.run_dir / f"{name}-after.bin"
            trainer.save_weights(before)
            raw.counts.zero_()
            for _ in range(args.evaluation_rollouts):
                trainer.rollouts()
            result = raw.snapshot()
            trainer.save_weights(after)
            result["weights_unchanged"] = digest(before) == digest(after)
            result["checkpoint_sha256"] = digest(before)
            if not result["weights_unchanged"] or result["invalid_actions"]:
                raise RuntimeError("frozen bandit evaluation changed weights or sampled illegal action")
            return result

        initial_eval = frozen_sample("initial-eval")
        # Clear init-time diagnostic losses before measuring actual learning.
        trainer.log(clear_metrics=False)
        raw.counts.zero_()
        start = time.perf_counter()
        for update in range(1, args.updates + 1):
            trainer.rollouts()
            trainer.train()
            if update % args.log_every == 0 or update == args.updates:
                sample = raw.snapshot()
                losses = trainer.log(clear_metrics=False).get("loss", {})
                if not losses or not all(math.isfinite(float(value)) for value in losses.values()):
                    raise RuntimeError("native training produced missing/nonfinite losses")
                record = {"optimizer_update": update * args.replay_ratio,
                          "training_rollout": update, "sample": sample, "loss": losses}
                curve.append(record)
                print(json.dumps(record), flush=True)
                raw.counts.zero_()
        training_wall = time.perf_counter() - start
        final = args.run_dir / "final.bin"
        trainer.save_weights(final)
        final_eval = frozen_sample("final-eval")
        improvement = final_eval["rewarded_action_fraction"] - initial_eval["rewarded_action_fraction"]
        objective_passed = (improvement >= args.minimum_probability_gain
                            or final_eval["rewarded_action_fraction"] >= .99)
        extension = Path(trainer.backend.__file__).resolve()
        report = {
            "schema": "rek.native_reward_bandit_probe.v1", "status": "completed",
            "diagnostic_only": True, "rek_policy_quality_measured": False,
            "reward_definition": "+1 for action0, -1 for action1, one-step terminal",
            "initial": initial_eval, "final": final_eval,
            "rewarded_action_probability_gain": improvement,
            "minimum_probability_gain": args.minimum_probability_gain,
            "learning_objective_passed": objective_passed,
            "learning_rate": args.learning_rate, "anneal_lr": False,
            "optimizer": "native Muon", "muon_momentum": native["train"]["beta1"],
            "vf_coef": native["train"]["vf_coef"], "entropy_coef": native["train"]["ent_coef"],
            "gamma": native["train"]["gamma"], "gae_lambda": native["train"]["gae_lambda"],
            "horizon": 64, "agents": args.agents,
            "optimizer_updates": args.updates * args.replay_ratio,
            "training_rollouts": args.updates, "replay_ratio": args.replay_ratio,
            "training_samples": args.updates * args.agents * 64,
            "training_wall_seconds": training_wall,
            "initial_checkpoint_sha256": digest(initial), "final_checkpoint_sha256": digest(final),
            "native_extension": str(extension), "native_extension_sha256": digest(extension),
            "curve": curve,
        }
        (args.run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
        return report
    finally:
        trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "default-config", "native-config", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--replay-ratio", type=int, choices=(1, 4), default=1)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--evaluation-rollouts", type=int, default=4)
    parser.add_argument("--minimum-probability-gain", type=float, default=.1)
    result = run(parser.parse_args())
    raise SystemExit(0 if result["learning_objective_passed"] else 1)
