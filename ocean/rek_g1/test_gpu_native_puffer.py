from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

import gpu_native_puffer
import gpu_puffer_env


class _RekCudaFixture:
    """Pure CUDA fixture matching the REK G1 policy/environment ABI."""

    def __init__(self, total_agents: int = 8) -> None:
        self.observations = torch.zeros(
            total_agents, 223, dtype=torch.float32, device="cuda"
        )
        self.rewards = torch.zeros(total_agents, dtype=torch.float32, device="cuda")
        self.terminals = torch.zeros_like(self.rewards)
        self.action_mask = torch.ones(
            total_agents, 33, dtype=torch.uint8, device="cuda"
        )
        self.steps = torch.zeros((), dtype=torch.int32, device="cuda")
        self.invalid_actions = torch.zeros((), dtype=torch.int32, device="cuda")

    def reset(self) -> None:
        self.observations.zero_()
        self.rewards.zero_()
        self.terminals.zero_()
        self.action_mask.fill_(1)
        self.steps.zero_()
        self.invalid_actions.zero_()

    def step(self, actions: torch.Tensor) -> None:
        self.invalid_actions.add_(
            torch.count_nonzero(
                (actions[:, 0] < 0)
                | (actions[:, 0] >= 33)
            ).to(torch.int32)
        )
        self.steps.add_(1)
        self.rewards.fill_(5.0)
        self.terminals.copy_(self.steps.remainder(4).eq(0).to(torch.float32))
        self.observations.mul_(0.9)
        self.observations[:, 0].copy_(actions[:, 0].to(torch.float32) / 32.0)
        self.observations[:, 1].copy_(self.rewards / 5.0)


class _FixtureMetrics:
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.updates = torch.zeros((), dtype=torch.float32, device=self.device)
        self.completed = torch.zeros((), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.updates.zero_()
        self.completed.zero_()

    def update(
        self, observations: torch.Tensor, terminals: torch.Tensor
    ) -> None:
        self.updates.add_(1.0)
        self.completed.add_(terminals.sum())

    def snapshot(self, *, clear: bool = True) -> dict[str, float]:
        updates, completed = torch.stack((self.updates, self.completed)).cpu().tolist()
        values = {"n": completed, "fixture_updates": updates}
        if clear:
            self.updates.zero_()
            self.completed.zero_()
        return values


def _args() -> dict:
    return {
        "env_name": "rek_g1",
        "rank": 0,
        "world_size": 1,
        "gpu_id": 0,
        "nccl_id": b"",
        "profile": False,
        "cudagraphs": 1,
        "reset_state": False,
        "seed": 42,
        "vec": {"total_agents": 8, "num_buffers": 1, "num_threads": 0},
        "env": {},
        "policy": {"hidden_size": 256, "num_layers": 2},
        "train": {
            "total_timesteps": 64,
            "horizon": 4,
            "minibatch_size": 32,
            "learning_rate": 3.0e-4,
            "min_lr_ratio": 0.0,
            "anneal_lr": False,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "replay_ratio": 1.0,
            "max_grad_norm": 0.5,
            "clip_coef": 0.2,
            "vf_clip_coef": 1.0,
            "vf_coef": 0.5,
            "ent_coef": 1.0e-3,
            "min_ent_coef_ratio": 0.0,
            "anneal_ent_coef": False,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "prio_alpha": 0.5,
            "prio_beta0": 0.5,
        },
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NativeExternalGpuPufferTests(unittest.TestCase):
    def test_native_rollout_train_and_checkpoint(self) -> None:
        raw_env = _RekCudaFixture()
        metrics = _FixtureMetrics()
        env = gpu_puffer_env.CudaTensorEnvAdapter(
            raw_env, (33,), metric_plugins=(metrics,)
        )
        trainer = gpu_native_puffer.NativeExternalGpuPuffer(_args(), env)
        try:
            self.assertTrue(trainer.pufferl.external_gpu)
            self.assertEqual(trainer.pufferl.native_env_count, 0)
            self.assertFalse(trainer.pufferl.has_env_threads)
            self.assertEqual(trainer.pufferl.hypers.reward_clip, 0.0)
            self.assertEqual(trainer.pufferl.recurrent_state_nonzero_count, 0)
            self.assertEqual(trainer.num_params(), 459_008)

            with tempfile.TemporaryDirectory() as directory:
                initial = Path(directory) / "initial.bin"
                trained = Path(directory) / "trained.bin"
                restored = Path(directory) / "restored.bin"
                trainer.save_weights(initial)
                initial_digest = _digest(initial)

                for expected_step in (32, 64):
                    trainer.rollouts()
                    self.assertEqual(trainer.global_step, expected_step)
                    self.assertGreater(
                        trainer.pufferl.recurrent_state_nonzero_count, 0
                    )
                    self.assertEqual(int(raw_env.invalid_actions.cpu()), 0)
                    trainer.train()
                manifest = trainer.save_weights(trained)
                self.assertNotEqual(initial_digest, _digest(trained))
                self.assertFalse(manifest["reward"]["legacy_clipping_enabled"])
                self.assertEqual(
                    manifest["environment"]["native_cpu_environment_count"], 0
                )
                self.assertFalse(
                    manifest["environment"]["native_environment_threads"]
                )

                logs = trainer.log()
                self.assertEqual(logs["agent_steps"], 64)
                self.assertEqual(logs["env"]["fixture_updates"], 8.0)
                self.assertEqual(logs["env"]["n"], 16.0)

                trainer.load_weights(initial)
                trainer.save_weights(restored)
                self.assertEqual(initial_digest, _digest(restored))
        finally:
            trainer.close()


if __name__ == "__main__":
    unittest.main()
