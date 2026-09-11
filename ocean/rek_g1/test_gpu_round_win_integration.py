"""CPU production-method integration and checkpoint/experiment provenance."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gpu_round_win_reward import GpuRoundWinReward, RoundWinRewardConfig, resolve_reward_objective
from run_gpu_credit_experiment import numeric_plan, resolved_config, run
from test_gpu_facing_potential_integration import SavingTrainer, wrapper_fixture
from test_train_gpu_duel import environment_fixture
from train_gpu_duel import save_failure, save_training_weights


class RoundWinIntegrationTests(unittest.TestCase):
    def test_opt_in_default_and_mutual_exclusion(self):
        facing, win, metadata = resolve_reward_objective({"train": {"gamma": .99}})
        self.assertIsNone(facing)
        self.assertIsNone(win)
        self.assertEqual(metadata["name"], "none")
        for kwargs in ({"objective": "unknown"}, {"objective": "round-win", "facing_scale": .25},
                       {"objective": "round-win", "reward_clip": 1}):
            with self.assertRaises(ValueError): resolve_reward_objective({"train": {"gamma": 1}}, **kwargs)
        with self.assertRaises(ValueError): resolve_reward_objective({"train": {"gamma": .999}}, objective="round-win")

    def test_production_wrapper_preserves_raw_dummy_metrics_and_encoder_inputs(self):
        raw, shaped = wrapper_fixture(), wrapper_fixture(encoded=True)
        config = RoundWinRewardConfig(1, .5, 5)
        shaped.round_win_reward = GpuRoundWinReward(2, "cpu", config, allow_cpu_for_tests=True)
        shaped.reward_shaper = shaped.round_win_reward
        oracle = GpuRoundWinReward(2, "cpu", config, allow_cpu_for_tests=True)
        pointer = shaped.rewards.data_ptr()
        for tick, (points, results, winners, terminals) in enumerate((
            ([2, 0], [0, 0], [-1, -1], [0, 0]),
            ([10, 9], [1, 2], [0, 1], [1, 1]),
            ([0, 0], [0, 0], [-1, -1], [0, 0]),
            ([4, 4], [3, 4], [-1, -1], [1, 1]),
        )):
            oracle.begin_transition(raw.duel.observations[0::2], raw.duel.terminals[0::2])
            for owner in (raw, shaped):
                owner._select()
                owner.duel.observations[0::2, 190] = torch.tensor(points).float()
                owner.duel.observations[0::2, 210] = torch.tensor(results).float()
                owner.duel.observations[0::2, 211] = torch.tensor(winners).float()
                owner.duel.terminals.copy_(torch.tensor(terminals).repeat_interleave(2).float())
                owner.duel.rewards.copy_(torch.tensor([10., -10, -9, 9]))
                owner._publish()
            expected = oracle.finish_transition(raw.duel.observations[0::2], raw.duel.rewards[0::2], raw.duel.terminals[0::2])
            torch.testing.assert_close(shaped.rewards, expected, atol=0, rtol=0)
            torch.testing.assert_close(raw.full_actions, shaped.full_actions, atol=0, rtol=0)
            for name in ("observations", "rewards", "terminals", "action_mask"):
                torch.testing.assert_close(getattr(raw.duel, name), getattr(shaped.duel, name), atol=0, rtol=0)
            for original, candidate in zip(raw.recorded_raw_metrics[-1], shaped.recorded_raw_metrics[-1]):
                torch.testing.assert_close(original, candidate, atol=0, rtol=0)
            self.assertEqual(pointer, shaped.rewards.data_ptr())
            self.assertTrue(torch.equal(shaped.observations[:, 188], torch.ones(2)))
            if tick == 2: self.assertEqual(shaped.round_win_reward.current_potential.abs().max().item(), 0)
            raw.log()
            shaped.log()

    def test_initial_periodic_final_failed_checkpoints_store_objective_and_encoder(self):
        _, _, descriptor = resolve_reward_objective({"train": {"gamma": 1}}, objective="round-win")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trainer = SavingTrainer()
            trainer.global_step = 8192
            for name in ("initial.bin", "verified-8192.bin", "final.bin"):
                path = directory/name
                saved = save_training_weights(trainer, path, "scaled_polar_xy_v1", "fresh-random", descriptor)
                self.assertEqual(saved["training_reward_transform"], descriptor)
                self.assertEqual(saved["reward"]["margin_potential_scale"], .5)
                self.assertEqual(saved["reward"]["margin_points"], 5)
                self.assertNotIn("facing_potential_scale", saved["reward"])
                self.assertEqual(saved["policy_observation_encoder"]["name"], "scaled_polar_xy_v1")
                self.assertEqual(saved, json.loads(Path(str(path)+".manifest.json").read_text()))
            args = SimpleNamespace(run_dir=directory, policy_observation_encoder="scaled_polar_xy_v1",
                                   policy_observation_warm_start="fresh-random")
            failed = save_failure(args, environment_fixture(), trainer, RuntimeError("fixture"), 1, 0, descriptor)
            self.assertEqual(failed["capture_errors"], [])
            self.assertEqual(failed["training_reward_transform"], descriptor)
            manifest = json.loads((directory/"failed-policy.bin.manifest.json").read_text())
            self.assertEqual(manifest["training_reward_transform"], descriptor)

    def test_extended_win_config_and_runner_pass_explicit_mode_without_changing_game(self):
        config_dir = Path(__file__).resolve().parents[2]/"config"
        config = resolved_config(config_dir/"default.ini", config_dir/"rek_g1.ini", config_dir/"rek_g1_round_win.ini")
        plan = numeric_plan(config, physical_fighters=1024, total_timesteps=13107200, minibatch_size=4096)
        self.assertEqual((plan["gamma"], plan["gae_lambda"], plan["learning_rate"], plan["replay_ratio"]), (1, .995, .003, 4))
        self.assertEqual((plan["rollouts"], plan["optimizer_minibatches"], plan["simulated_seconds_per_arena"]), (100, 12800, 512))
        self.assertIsNone(plan["discount_efold_seconds"])
        self.assertTrue(plan["undiscounted"])
        original = resolved_config(config_dir/"default.ini", config_dir/"rek_g1.ini", config_dir/"rek_g1_credit_long.ini")
        differences = {(section, key) for section in original for key in original[section]
                       if original[section][key] != config[section][key]}
        self.assertEqual(differences, {("train", "gamma")})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial, physics = directory/"initial.bin", directory/"physics.json"
            initial.write_bytes(b"CPU-only initial checkpoint fixture")
            physics.write_text("{}")
            sha = hashlib.sha256(initial.read_bytes()).hexdigest()
            args = SimpleNamespace(checkpoint_sha256=sha, load_checkpoint=initial,
                default_config=config_dir/"default.ini", native_config=config_dir/"rek_g1.ini",
                credit_config=config_dir/"rek_g1_round_win.ini", gpu_duel_config=physics,
                run_dir=directory/"trial", output=directory/"report.json", total_agents=1024,
                total_timesteps=13107200, minibatch_size=4096, prepare_only=False,
                log_every=1, checkpoint_every=8, reward_objective="round-win",
                policy_observation_encoder="scaled_polar_xy_v1")
            def fake_train(actual):
                self.assertEqual(actual.reward_objective, "round-win")
                self.assertEqual((actual.margin_potential_scale, actual.margin_points, actual.reward_clip), (.5, 5, 0))
                self.assertEqual(actual.policy_observation_encoder, "scaled_polar_xy_v1")
                self.assertEqual(actual.policy_observation_warm_start, "matching-checkpoint")
                return {"checkpoints": {"initial": {"sha256": sha}}, "training": {"agent_steps": 13107200}}
            with patch("train_gpu_duel.train", side_effect=fake_train), redirect_stdout(io.StringIO()):
                actual = run(args)
            descriptor = actual["credit_span_experiment"]["training_reward_transform"]
            self.assertEqual(descriptor["gamma"], 1)
            self.assertFalse(descriptor["game_rules_or_scores_changed"])
            json.dumps(actual, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
