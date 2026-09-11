"""Boundary GAE and value-only external bootstrap regression.

CPU tests run by default. --run-cuda exercises the actual native kernels and
rollout/PPO bridge with frozen diagnostic weights; it runs no REK physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from gpu_native_puffer import NativeExternalGpuPuffer
from gpu_puffer_env import CudaTensorEnvAdapter
from test_gpu_native_puffer import _args
from test_gpu_native_terminal_replay import diagnostic_weights, sha256


def oracle(values, rewards, dones, importance, final_value, final_reward, final_done,
           gamma=1.0, lam=.999, rho_clip=1.0, c_clip=1.0):
    """Incoming reward/done at row t describes action t-1, including final tail."""
    out = np.zeros_like(values, dtype=np.float32)
    carry = np.zeros(values.shape[0], dtype=np.float32)
    nv, nr, nd = final_value, final_reward, final_done
    for t in range(values.shape[1] - 1, -1, -1):
        rho = np.minimum(importance[:, t], np.float32(rho_clip))
        c = np.minimum(importance[:, t], np.float32(c_clip))
        nonterminal = np.float32(1) - nd
        delta = rho * (nr + np.float32(gamma) * nv * nonterminal - values[:, t])
        carry = delta + np.float32(gamma) * np.float32(lam) * c * carry * nonterminal
        out[:, t] = carry
        nv, nr, nd = values[:, t], rewards[:, t], dones[:, t]
    return out


class CpuTests(unittest.TestCase):
    def test_terminal_exactly_after_last_action(self):
        for horizon in (64, 256, 1024):
            zero = np.zeros((1, horizon), np.float32)
            actual = oracle(zero, zero, zero, np.ones_like(zero),
                            np.array([99], np.float32), np.ones(1, np.float32), np.ones(1, np.float32))
            self.assertEqual(float(actual[0, -1]), 1)
            self.assertGreater(float(actual[0, 0]), .35)

    def test_nonterminal_value_and_interior_done_alignment(self):
        values = np.full((1, 64), 2, np.float32)
        rewards, dones = np.zeros_like(values), np.zeros_like(values)
        actual = oracle(values, rewards, dones, np.ones_like(values),
                        np.array([3], np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32))
        self.assertEqual(float(actual[0, -1]), 1)
        values.fill(0)
        rewards[:, 32], dones[:, 32] = 1, 1
        actual = oracle(values, rewards, dones, np.ones_like(values),
                        np.zeros(1, np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32))
        self.assertEqual(float(actual[0, 31]), 1)
        self.assertTrue(np.all(actual[:, 32:] == 0))


class DeviceView:
    def __init__(self, pointer, shape, typestr="<f4"):
        self.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": typestr,
                                        "data": (int(pointer), False), "version": 3}


def native_view(tensor, shape):
    return torch.as_tensor(DeviceView(tensor.data_ptr(), shape), device="cuda")


def kernel_cases(backend):
    results = []
    for horizon in (64, 256, 1024):
        generator = np.random.default_rng(731 + horizon)
        values = generator.normal(0, .1, (4, horizon)).astype(np.float32)
        rewards = generator.normal(0, .02, values.shape).astype(np.float32)
        dones = np.zeros_like(values)
        dones[2, horizon // 2] = 1
        importance = generator.uniform(.5, 1.5, values.shape).astype(np.float32)
        final_values = np.array([.2, 5, .3, -.1], np.float32)
        final_rewards = np.array([0, 1, .5, -.2], np.float32)
        final_dones = np.array([0, 1, 0, 1], np.float32)
        expected = oracle(values, rewards, dones, importance, final_values, final_rewards, final_dones)
        tensors = [torch.as_tensor(array, device="cuda") for array in
                   (values, rewards, dones, importance, np.zeros_like(values), final_values, final_rewards, final_dones)]
        backend.puff_advantage_bootstrap(*(item.data_ptr() for item in tensors),
                                        4, horizon, 1.0, .999, 1.0, 1.0,
                                        torch.cuda.current_stream().cuda_stream)
        observed = tensors[4].cpu().numpy()
        np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-6)
        results.append({"horizon": horizon, "max_absolute_error": float(np.max(np.abs(observed - expected))),
                        "terminal_and_nonterminal_tails_tested": True})
    return results


class BoundaryFixture:
    def __init__(self, horizon):
        self.horizon = horizon
        self.observations = torch.zeros((8, 223), device="cuda")
        self.rewards = torch.zeros(8, device="cuda")
        self.terminals = torch.zeros(8, device="cuda")
        self.action_mask = torch.ones((8, 33), dtype=torch.uint8, device="cuda")
        self.action_mask[6:, 1:] = 0  # Forced no-op rows still have value targets.
        self.tick = 0

    def reset(self):
        self.tick = 0
        self.observations.zero_()
        self.rewards.zero_()
        self.terminals.zero_()

    def step(self, actions):
        self.tick += 1
        self.rewards.zero_()
        self.terminals.zero_()
        if self.tick % self.horizon == 0:
            self.rewards[0:4] = 1
            self.terminals[0:4] = 1
        elif self.tick % self.horizon == self.horizon // 2:
            self.rewards[4:6] = .5
            self.terminals[4:6] = 1


def boundary_replay(horizon, reward_clip=0.0):
    config = _args()
    config["reset_state"] = True
    config["train"].update(horizon=horizon, minibatch_size=8*horizon,
                          total_timesteps=16*horizon, learning_rate=0.0,
                          anneal_lr=False, replay_ratio=1.0, reward_clip=0.0,
                          gamma=1.0, gae_lambda=.999)
    raw = BoundaryFixture(horizon)
    trainer = NativeExternalGpuPuffer(config, CudaTensorEnvAdapter(raw, (33,)), reward_clip=reward_clip)
    try:
        with tempfile.TemporaryDirectory(prefix="native-bootstrap-") as directory:
            initial, final = Path(directory)/"initial.bin", Path(directory)/"final.bin"
            diagnostic_weights().tofile(initial)
            trainer.load_weights(initial)
            backend, native = trainer.backend, trainer.pufferl
            stream = trainer.stream.cuda_stream
            trainer.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(trainer.stream):
                backend.external_rollout_begin(native, stream)
                for tick in range(horizon):
                    backend.external_rollout_step(native, tick, stream)
                    backend.external_actions_to_int32(native, trainer.actions.data_ptr(), stream)
                    trainer.env.step(trainer.actions)
                state = native_view(native.external_primary_recurrent_state, (2, 8, 256))
                rng = torch.as_tensor(DeviceView(native.external_sampling_rng_ptr,
                                     (native.external_sampling_rng_bytes,), "|u1"), device="cuda")
                state_before, rng_before, actions_before = state.clone(), rng.clone(), trainer.actions.clone()
                backend.external_rollout_finish(native, stream)
                bootstrap = native_view(native.external_bootstrap_values, (8,)).clone()
                tail_reward = native_view(native.external_bootstrap_rewards, (8,)).clone()
                tail_done = native_view(native.external_bootstrap_terminals, (8,)).clone()
            torch.cuda.synchronize()
            assert native.external_bootstrap_ready
            assert torch.equal(state_before, state) and torch.equal(rng_before, rng)
            assert torch.equal(actions_before, trainer.actions)
            assert torch.equal(tail_reward, raw.rewards) and torch.equal(tail_done, raw.terminals)
            trainer.train()
            loss = trainer.log(clear_metrics=False)["loss"]
            assert abs(loss["kl"]) <= 1e-6 and abs(loss["old_kl"]) <= 2e-6 and loss["clipfrac"] == 0
            trainer.save_weights(final)
            assert sha256(initial) == sha256(final)
            expected_tail = tail_reward.clamp(-reward_clip, reward_clip) if reward_clip else tail_reward
            assert torch.equal(native_view(native.external_bootstrap_rewards, (8,)), expected_tail)
            with torch.cuda.stream(trainer.stream):
                backend.external_rollout_begin(native, stream)
                backend.external_rollout_step(native, 0, stream)
                first_value = native_view(native.rollouts.values, (horizon, 8))[0].clone()
                # Complete the begun rollout without taking an extra first action.
                for tick in range(horizon):
                    if tick:
                        backend.external_rollout_step(native, tick, stream)
                    backend.external_actions_to_int32(native, trainer.actions.data_ptr(), stream)
                    trainer.env.step(trainer.actions)
                backend.external_rollout_finish(native, stream)
            torch.cuda.synchronize()
            assert torch.equal(bootstrap, first_value)
            trainer.train()
            captured_loss = trainer.log(clear_metrics=False)["loss"]
            assert abs(captured_loss["kl"]) <= 1e-6 and abs(captured_loss["old_kl"]) <= 2e-6
            assert captured_loss["clipfrac"] == 0
            trainer.save_weights(final)
            assert sha256(initial) == sha256(final)
            return {"horizon": horizon, "loss": loss, "captured_replay_loss": captured_loss,
                    "weights_unchanged": True, "reward_clip": reward_clip, "tail_clip_matches_rollout_rule": True,
                    "bootstrap_equals_next_rollout_first_value": True,
                    "hidden_state_rng_and_actions_preserved": True,
                    "final_reward_and_done_captured": True}
    finally:
        trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-cuda", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.run_cuda:
        unittest.main(argv=[__file__])
    else:
        from pufferlib import _C
        assert getattr(_C, "supports_external_rollout_bootstrap", False)
        report = {"schema": "rek.native_external_bootstrap.v1", "stage": "passed",
                  "extension": str(_C.__file__), "extension_sha256": sha256(_C.__file__),
                  "advantage_kernels": kernel_cases(_C),
                  "boundary_replay": [boundary_replay(h) for h in (64, 256, 1024)],
                  "nonzero_reward_clip": boundary_replay(64, reward_clip=.25)}
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            with args.report.open("x") as stream:
                stream.write(payload)
        print(payload, end="")
