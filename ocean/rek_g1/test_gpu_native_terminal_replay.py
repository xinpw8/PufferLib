"""Full native same-weight rollout/training replay across terminal boundaries.

Uses a synthetic diagnostic policy with a known recurrent signal, not a trained
REK checkpoint. Learning rate zero and before/after hashes prevent weight updates.
The unchanged native inference, sampling, select_copy, and captured PPO path run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

try:
    from .gpu_native_puffer import NativeExternalGpuPuffer
    from .gpu_puffer_env import CudaTensorEnvAdapter
    from .test_gpu_native_puffer import _args
except ImportError:
    from gpu_native_puffer import NativeExternalGpuPuffer
    from gpu_puffer_env import CudaTensorEnvAdapter
    from test_gpu_native_puffer import _args


def diagnostic_weights():
    # src/models.cu policy_weights_create registers encoder, decoder, MinGRU.
    # Zero encoder/recurrent matrices give a deterministic nonzero MinGRU state.
    hidden, obs, actions, layers = 256, 223, 33, 2
    weights = np.zeros(hidden * obs + (actions + 1) * hidden
                       + layers * 3 * hidden * hidden, dtype=np.float32)
    decoder = weights[hidden * obs:hidden * obs + (actions + 1) * hidden]
    decoder = decoder.reshape(actions + 1, hidden)
    decoder[:actions, 0] = np.linspace(-4, 4, actions, dtype=np.float32)
    decoder[actions, 0] = 1
    return weights


def replay_args():
    args = _args()
    args["reset_state"] = True
    args["train"].update(horizon=8, minibatch_size=64, total_timesteps=448,
                         learning_rate=0.0, anneal_lr=False, replay_ratio=1.0)
    return args


class ReplayFixture:
    def __init__(self, device="cuda"):
        self.observations = torch.zeros((8, 223), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(8, dtype=torch.float32, device=device)
        self.terminals = torch.zeros_like(self.rewards)
        self.action_mask = torch.ones((8, 33), dtype=torch.uint8, device=device)
        self.schedule = torch.zeros((8, 8), dtype=torch.float32, device=device)
        self.position = 0

    def reset(self):
        self.observations.zero_()
        self.rewards.zero_()
        self.terminals.zero_()
        self.position = 0

    def configure(self, reset_ticks):
        self.reset()
        self.schedule.zero_()
        for tick in reset_ticks:
            self.schedule[:, tick] = 1
        self.terminals.copy_(self.schedule[:, 0])

    def step(self, actions):
        self.position += 1
        self.rewards.fill_(5)
        if self.position < self.schedule.shape[1]:
            self.terminals.copy_(self.schedule[:, self.position])
        else:
            self.terminals.zero_()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_native_replay(*, expect_reset_mismatch=False):
    raw = ReplayFixture()
    trainer = NativeExternalGpuPuffer(replay_args(), CudaTensorEnvAdapter(raw, (33,)))
    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix="native-terminal-replay-") as directory:
            initial = Path(directory) / "diagnostic.bin"
            after = Path(directory) / "after.bin"
            weights = diagnostic_weights()
            if trainer.num_params() != weights.size:
                raise AssertionError("native parameter count no longer matches diagnostic ABI")
            weights.tofile(initial)
            trainer.load_weights(initial)
            frozen_sha = sha256(initial)
            for name, reset_ticks in (
                ("no_terminals_capture", []), ("interior_replay", [2, 4, 6]),
                ("checkpoint_replay", [4]), ("first_observation_replay", [0]),
                ("final_observation_replay", [7]), ("adjacent_replay", [3, 4]),
                ("no_terminals_again_replay", []),
            ):
                raw.configure(reset_ticks)
                trainer.rollouts()
                trainer.train()
                log = trainer.log(clear_metrics=False)
                trainer.save_weights(after)
                unchanged = sha256(after) == frozen_sha
                loss = dict(log["loss"])
                finite = bool(loss) and all(math.isfinite(float(value)) for value in loss.values())
                same_logprob = (finite and abs(float(loss["kl"])) <= 1e-6
                                and abs(float(loss["old_kl"])) <= 2e-6
                                and float(loss["clipfrac"]) == 0.0)
                cases.append({"case": name, "reset_before_observations": reset_ticks,
                              "loss": loss, "same_weight_logprob_matches": same_logprob,
                              "all_losses_finite": finite, "checkpoint_unchanged": unchanged,
                              "agent_steps": trainer.global_step})
            preserved = all(case["checkpoint_unchanged"] and case["all_losses_finite"] for case in cases)
            if expect_reset_mismatch:
                expected = (cases[0]["same_weight_logprob_matches"]
                            and cases[-1]["same_weight_logprob_matches"]
                            and not cases[1]["same_weight_logprob_matches"])
            else:
                expected = all(case["same_weight_logprob_matches"] for case in cases)
            extension = Path(trainer.backend.__file__).resolve()
            return {"stage": "passed" if preserved and expected else "failed",
                    "expect_reset_mismatch": expect_reset_mismatch,
                    "checkpoint_sha256": frozen_sha,
                    "checkpoint_description": "synthetic MinGRU diagnostic, no REK training quality claim",
                    "native_extension": str(extension), "native_extension_sha256": sha256(extension),
                    "learning_rate": 0, "parameters": int(weights.size),
                    "cuda_graphs_enabled": True, "horizon": 8, "cases": cases}
    finally:
        trainer.close()


class ReplayCpuTests(unittest.TestCase):
    def test_same_observation_terminal_schedule(self):
        fixture = ReplayFixture(device="cpu")
        fixture.configure([2, 4, 6])
        seen = []
        for tick in range(8):
            seen.append(float(fixture.terminals[0]))
            fixture.step(None)
        self.assertEqual(seen, [0, 0, 1, 0, 1, 0, 1, 0])
        fixture.configure([0, 7])
        self.assertEqual(float(fixture.terminals[0]), 1)

    def test_configuration_has_no_learning_update(self):
        args = replay_args()
        self.assertTrue(args["reset_state"])
        self.assertEqual(args["train"]["learning_rate"], 0)
        self.assertFalse(args["train"]["anneal_lr"])
        self.assertEqual(args["train"]["minibatch_size"], 8 * args["train"]["horizon"])

    def test_diagnostic_policy_abi(self):
        weights = diagnostic_weights()
        self.assertEqual(weights.size, 459008)
        self.assertEqual(weights.dtype, np.float32)
        self.assertFalse(np.any(weights[:256 * 223]))
        self.assertFalse(np.any(weights[256 * 223 + 34 * 256:]))
        self.assertGreater(np.ptp(weights[256 * 223:256 * 223 + 33 * 256:256]), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-cuda", action="store_true")
    parser.add_argument("--expect-reset-mismatch", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.run_cuda:
        unittest.main(argv=[__file__])
    else:
        report = run_native_replay(expect_reset_mismatch=args.expect_reset_mismatch)
        payload = json.dumps(report, indent=2) + "\n"
        if args.report:
            args.report.write_text(payload, encoding="utf-8")
        print(payload, end="")
        raise SystemExit(0 if report["stage"] == "passed" else 1)
