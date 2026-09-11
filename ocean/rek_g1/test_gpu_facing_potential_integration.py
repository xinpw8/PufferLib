"""CPU-only production-wrapper, training-provenance and opt-in fixtures."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gpu_candidate_dummy import GpuCandidateApproachDummy, GpuCandidateDummyDuel
from gpu_facing_potential import FacingPotentialConfig, GpuFacingPotential, resolve_training_reward
from gpu_policy_observation_encoder import GpuScaledPolarXYPolicyEncoder, load_policy_encoder_checkpoint
from run_gpu_credit_experiment import run as run_experiment
from train_gpu_duel import save_training_weights


def wrapper_fixture(scale=0.0, *, encoded=False):
    """Exercise production methods with test-only CPU buffers."""
    owner = object.__new__(GpuCandidateDummyDuel)
    owner.rows = 2
    observations = torch.zeros((4, 223))
    observations[:, 3] = 1
    observations[:, 86] = 1
    observations[:, 185] = 2
    observations[:, 188:190] = 120
    owner.duel = SimpleNamespace(observations=observations, rewards=torch.zeros(4),
        terminals=torch.zeros(4), action_mask=torch.ones((4, 33), dtype=torch.uint8),
        scheduler=SimpleNamespace(move_start_edge=torch.zeros(4)),
        combat=SimpleNamespace(tick_score_delta=torch.zeros(4)), check_status=lambda: None)
    owner.facing_potential = (GpuFacingPotential(2, "cpu", FacingPotentialConfig(.9, scale), allow_cpu_for_tests=True)
                              if scale else None)
    owner.reward_shaper = owner.facing_potential
    owner.policy_encoder = (GpuScaledPolarXYPolicyEncoder(2, "cpu", initialization="fresh-random", allow_cpu_for_tests=True)
                            if encoded else None)
    owner.observations, owner.rewards, owner.terminals = torch.empty((2, 223)), torch.empty(2), torch.empty(2)
    owner.action_mask = torch.empty((2, 33), dtype=torch.uint8)
    owner.learner_actions = torch.tensor([[17], [6]], dtype=torch.int32)
    owner.full_actions = torch.empty((4, 1), dtype=torch.int32)
    owner.dummy = GpuCandidateApproachDummy(2, "cpu")
    owner.recorded_raw_metrics = []
    owner.combat_metrics = SimpleNamespace(update=lambda obs, term: owner.recorded_raw_metrics.append((obs.clone(), term.clone())))
    owner.behavior_metrics = SimpleNamespace(update=lambda *args: None)
    owner._copy_learner_buffers()
    return owner


class SavingTrainer:
    def save_weights(self, path):
        payload = b"synthetic policy weights for CPU metadata fixture"
        Path(path).write_bytes(payload)
        manifest = {"checkpoint": {"sha256": hashlib.sha256(payload).hexdigest()},
                    "reward": {"transform": "none", "clip_magnitude": 0.0}}
        Path(str(path)+".manifest.json").write_text(json.dumps(manifest))
        return manifest


class FacingIntegrationTests(unittest.TestCase):
    def test_zero_setting_resolves_no_shaper_and_default_reward_path(self):
        config, descriptor = resolve_training_reward({"train": {"gamma": .99}}, scale=0, reward_clip=0)
        self.assertIsNone(config)
        self.assertEqual(descriptor["name"], "none")
        self.assertIsNone(descriptor["potential"])
        owner = wrapper_fixture()
        self.assertIsNone(owner.facing_potential)
        for tick in range(3):
            owner._select()
            owner.duel.rewards.copy_(torch.tensor([5., -5., -3., 3.]) * (tick+1))
            owner._publish()
            torch.testing.assert_close(owner.rewards, owner.duel.rewards[0::2], rtol=0, atol=0)
            torch.testing.assert_close(owner.observations, owner.duel.observations[0::2], rtol=0, atol=0)
        owner.log()

    def test_native_gamma_binary32_and_clip_validation(self):
        config, descriptor = resolve_training_reward({"train": {"gamma": .9995}}, scale=.25, reward_clip=0)
        expected = struct.unpack("f", struct.pack("f", .9995))[0]
        self.assertEqual(config.gamma, expected)
        self.assertNotEqual(config.gamma, .9995)
        config.validate_training_discount(expected, reward_clip=0)
        self.assertEqual(descriptor["gamma"], expected)
        self.assertEqual(descriptor["gamma_configured"], .9995)
        for scale, clip in ((.25, 1), (-1, 0), (float("nan"), 0), (float("inf"), 0)):
            with self.assertRaises(ValueError):
                resolve_training_reward({"train": {"gamma": .99}}, scale=scale, reward_clip=clip)
        self.assertEqual(resolve_training_reward({"train": {"gamma": .99}}, scale=0, reward_clip=1)[1]["name"], "symmetric_clamp")

    def test_wrapper_preserves_raw_state_dummy_scores_metrics_and_terminal_alignment(self):
        base, shaped = wrapper_fixture(), wrapper_fixture(.25)
        pointer = shaped.rewards.data_ptr()
        oracle = GpuFacingPotential(2, "cpu", FacingPotentialConfig(.9, .25), allow_cpu_for_tests=True)
        # Row0 ends on step1; step2 is the delayed reset. Row1 ends later.
        for tick, (opponent_x, terminals) in enumerate((([-1., 1.], [0., 0.]), ([1., -1.], [1., 0.]),
                                                       ([1., 1.], [0., 0.]), ([-1., -1.], [0., 1.]))):
            oracle.begin_transition(base.duel.observations[0::2], base.duel.terminals[0::2])
            base._select()
            shaped._select()
            torch.testing.assert_close(base.full_actions, shaped.full_actions, atol=0, rtol=0)
            for owner in (base, shaped):
                owner.duel.observations[0::2, 86] = torch.tensor(opponent_x)
                owner.duel.observations[:, 190] += tick+1
                owner.duel.rewards.copy_(torch.tensor([5., -5., -3., 3.]))
                owner.duel.terminals.copy_(torch.tensor(terminals).repeat_interleave(2))
                owner._publish()
            expected = oracle.finish_transition(base.duel.observations[0::2], base.duel.rewards[0::2], base.duel.terminals[0::2])
            torch.testing.assert_close(shaped.rewards, expected, atol=0, rtol=0)
            self.assertEqual(shaped.rewards.data_ptr(), pointer)
            for name in ("observations", "terminals", "action_mask"):
                torch.testing.assert_close(getattr(base, name), getattr(shaped, name), atol=0, rtol=0)
            for name in ("observations", "rewards", "terminals"):
                torch.testing.assert_close(getattr(base.duel, name), getattr(shaped.duel, name), atol=0, rtol=0)
            for original, candidate in zip(base.recorded_raw_metrics[-1], shaped.recorded_raw_metrics[-1]):
                torch.testing.assert_close(original, candidate, atol=0, rtol=0)
            if tick == 2:
                self.assertEqual(shaped.facing_potential.current_potential[0].item(), 0)
            base.log()
            shaped.log()

    def test_encoder_and_reward_transforms_remain_independent(self):
        shaped = wrapper_fixture(.25, encoded=True)
        raw_start = shaped.duel.observations.clone()
        shaped._select()
        shaped.duel.observations[0::2, 86] = -1
        shaped._publish()
        self.assertTrue(torch.equal(shaped.observations[:, 188], torch.ones(2)))
        torch.testing.assert_close(shaped.facing_potential.current_potential, torch.full((2,), .25, dtype=torch.float64))
        torch.testing.assert_close(shaped.rewards, torch.full((2,), -.475))
        self.assertTrue(torch.equal(shaped.duel.observations[1::2], raw_start[1::2]))
        shaped.log()

    def test_training_checkpoint_records_reward_separately_from_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            for scale in (0, .25):
                _, descriptor = resolve_training_reward({"train": {"gamma": .9995}}, scale=scale, reward_clip=0)
                path = Path(directory) / f"fixture-{scale}.bin"
                saved = save_training_weights(SavingTrainer(), path, "scaled_polar_xy_v1", "fresh-random", descriptor)
                actual = json.loads(Path(str(path)+".manifest.json").read_text())
                self.assertEqual(saved, actual)
                self.assertEqual(saved["training_reward_transform"], descriptor)
                self.assertEqual(saved["reward"]["transform"], descriptor["name"])
                self.assertEqual(saved["reward"]["native_transform"], "none")
                self.assertEqual(saved["policy_observation_encoder"]["name"], "scaled_polar_xy_v1")
                loaded, sha = load_policy_encoder_checkpoint(path, "scaled_polar_xy_v1")
                self.assertEqual(sha, saved["checkpoint"]["sha256"])
                self.assertEqual(loaded["training_reward_transform"], descriptor)

    def test_experiment_passes_explicit_setting_and_materializes_its_discount(self):
        config_dir = Path(__file__).resolve().parents[2] / "config"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            initial = directory / "initial.bin"
            initial.write_bytes(b"synthetic initial checkpoint")
            sha = hashlib.sha256(initial.read_bytes()).hexdigest()
            physics = directory / "physics.json"
            physics.write_text("{}")
            args = SimpleNamespace(checkpoint_sha256=sha, load_checkpoint=initial,
                default_config=config_dir/"default.ini", native_config=config_dir/"rek_g1.ini",
                credit_config=config_dir/"rek_g1_credit_long.ini", gpu_duel_config=physics,
                run_dir=directory/"trial", output=directory/"result.json", total_agents=1024,
                total_timesteps=3276800, minibatch_size=4096, prepare_only=False,
                log_every=1, checkpoint_every=8, facing_potential_scale=.25)
            def fake_train(training_args):
                self.assertEqual(training_args.facing_potential_scale, .25)
                self.assertEqual(training_args.reward_clip, 0)
                return {"checkpoints": {"initial": {"sha256": sha}}, "training": {"agent_steps": 3276800}}
            with patch("train_gpu_duel.train", side_effect=fake_train):
                result = run_experiment(args)
            descriptor = result["credit_span_experiment"]["training_reward_transform"]
            self.assertEqual(descriptor["facing_potential_scale"], .25)
            self.assertEqual(descriptor["gamma_configured"], result["credit_span_experiment"]["gamma"])


if __name__ == "__main__":
    unittest.main()
