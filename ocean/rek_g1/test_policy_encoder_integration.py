"""CPU-only checkpoint/load and raw-versus-policy-view integration fixtures."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evaluate_gpu_dummy import save_frozen_weights
from gpu_candidate_dummy import GpuCandidateApproachDummy, GpuCandidateDummyDuel
from gpu_policy_observation_encoder import (
    GpuScaledPolarXYPolicyEncoder, SCALED_ENCODER_NAME,
    GpuStrikeAgeScaledPolarXYPolicyEncoder, STRIKE_AGE_ENCODER_NAME, SCALED_POLAR_INITIALIZATION,
    encoder_fingerprint,
    load_policy_encoder_checkpoint, policy_encoder_report, save_policy_weights,
)


class SavingTrainer:
    def __init__(self, payload=b"synthetic unit-test weights"):
        self.payload = payload

    def save_weights(self, path):
        path = Path(path)
        path.write_bytes(self.payload)
        result = {"schema": "fixture", "checkpoint": {"sha256": hashlib.sha256(self.payload).hexdigest(), "path": str(path)}}
        path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(result))
        return result


class CheckpointIntegrationTests(unittest.TestCase):
    def test_credit_cli_prepares_strike_age_without_early_torch_or_cuda_initialization(self):
        source_dir = Path(__file__).resolve().parent
        config_dir = source_dir.parents[1]/"config"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint, physics = directory/"source.bin", directory/"physics.json"
            saved = save_policy_weights(SavingTrainer(), checkpoint, SCALED_ENCODER_NAME)
            physics.write_text("{}")
            cli = [str(source_dir/"run_gpu_credit_experiment.py"),
                   "--default-config", str(config_dir/"default.ini"), "--native-config", str(config_dir/"rek_g1.ini"),
                   "--credit-config", str(config_dir/"rek_g1_round_win.ini"), "--gpu-duel-config", str(physics),
                   "--load-checkpoint", str(checkpoint), "--checkpoint-sha256", saved["checkpoint"]["sha256"],
                   "--run-dir", str(directory/"trial"), "--output", str(directory/"report.json"),
                   "--total-agents", "8", "--total-timesteps", "1024", "--minibatch-size", "1024",
                   "--reward-objective", "round-win", "--policy-observation-encoder", STRIKE_AGE_ENCODER_NAME,
                   "--policy-observation-warm-start", SCALED_POLAR_INITIALIZATION, "--prepare-only"]
            program = ("import pathlib,runpy,sys; sys.argv=sys.argv[1:]; "
                       "sys.path.insert(0,str(pathlib.Path(sys.argv[0]).parent)); "
                       "import run_gpu_credit_experiment; assert 'torch' not in sys.modules; "
                       "runpy.run_path(sys.argv[0],run_name='__main__'); "
                       "assert 'train_gpu_duel' not in sys.modules; import torch; assert not torch.cuda.is_initialized()")
            result = subprocess.run([sys.executable, "-B", "-c", program, *cli],
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            plan = json.loads((directory/"trial.inputs/experiment.json").read_text())
            self.assertEqual(plan["policy_observation_encoder_name"], STRIKE_AGE_ENCODER_NAME)
            self.assertEqual(plan["policy_observation_initialization"], SCALED_POLAR_INITIALIZATION)
            self.assertEqual(plan["initial_checkpoint_sha256"], saved["checkpoint"]["sha256"])
            self.assertFalse((directory/"trial").exists())

    def test_training_and_evaluation_cli_accept_explicit_new_encoder(self):
        import train_gpu_duel
        import evaluate_gpu_dummy

        argv = ["train_gpu_duel", "--gpu-duel-config", "unused.json", "--run-dir", "unused-run",
                "--output", "unused-output.json", "--total-timesteps", "4096",
                "--policy-observation-encoder", STRIKE_AGE_ENCODER_NAME,
                "--policy-observation-warm-start", SCALED_POLAR_INITIALIZATION,
                "--load-checkpoint", "unused.bin", "--load-checkpoint-sha256", "a"*64]
        with patch("sys.argv", argv), patch.object(train_gpu_duel, "train") as train:
            train_gpu_duel.main()
            parsed = train.call_args.args[0]
            self.assertEqual(parsed.policy_observation_encoder, STRIKE_AGE_ENCODER_NAME)
            self.assertEqual(parsed.policy_observation_warm_start, SCALED_POLAR_INITIALIZATION)
        argv = ["evaluate_gpu_dummy", "--checkpoint", "unused.bin", "--checkpoint-sha256", "a"*64,
                "--gpu-duel-config", "unused.json", "--total-agents", "8", "--ticks", "256",
                "--run-dir", "unused-run", "--policy-observation-encoder", STRIKE_AGE_ENCODER_NAME]
        with patch("sys.argv", argv), patch.object(evaluate_gpu_dummy, "evaluate") as evaluate:
            evaluate_gpu_dummy.main()
            self.assertEqual(evaluate.call_args.args[0].policy_observation_encoder, STRIKE_AGE_ENCODER_NAME)

    def test_strike_age_warm_start_is_pinned_and_does_not_relabel_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"source.bin", Path(directory)/"target.bin"
            trainer = SavingTrainer()
            saved = save_policy_weights(trainer, source, SCALED_ENCODER_NAME)
            sha = saved["checkpoint"]["sha256"]
            sidecar = source.with_suffix(".bin.manifest.json")
            original = sidecar.read_bytes()
            for initialization, pin in (("matching-checkpoint", sha), (SCALED_POLAR_INITIALIZATION, None),
                                         (SCALED_POLAR_INITIALIZATION, "0"*64)):
                with self.subTest(initialization=initialization, pin=pin), self.assertRaises(ValueError):
                    load_policy_encoder_checkpoint(source, STRIKE_AGE_ENCODER_NAME, initialization=initialization,
                                                   expected_sha256=pin)
            loaded, actual = load_policy_encoder_checkpoint(source, STRIKE_AGE_ENCODER_NAME,
                initialization=SCALED_POLAR_INITIALIZATION, expected_sha256=sha)
            self.assertEqual(loaded, saved)
            self.assertEqual(sidecar.read_bytes(), original)
            encoder = GpuStrikeAgeScaledPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=loaded,
                checkpoint_sha256=actual, initialization=SCALED_POLAR_INITIALIZATION, allow_cpu_for_tests=True)
            self.assertEqual(encoder.encoder_name, STRIKE_AGE_ENCODER_NAME)
            target_manifest = save_policy_weights(trainer, target, STRIKE_AGE_ENCODER_NAME, SCALED_POLAR_INITIALIZATION)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(sidecar.read_bytes(), original)
            self.assertEqual(load_policy_encoder_checkpoint(target, STRIKE_AGE_ENCODER_NAME)[1], sha)
            self.assertEqual(target_manifest["policy_observation_input_change"]["source_encoder_sha256"],
                             encoder_fingerprint(SCALED_ENCODER_NAME))
            self.assertTrue(target_manifest["policy_observation_input_change"]["semantically_changed_policy_inputs"])
            self.assertFalse(target_manifest["policy_observation_input_change"]["checkpoint_weights_transformed"])
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(target, SCALED_ENCODER_NAME)

    def test_strike_age_warm_start_rejects_other_source_target_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            for source_name in ("raw", "polar_xy_v1", STRIKE_AGE_ENCODER_NAME):
                source = Path(directory)/f"{source_name}.bin"
                saved = save_policy_weights(SavingTrainer(), source, source_name)
                with self.subTest(source=source_name), self.assertRaises(ValueError):
                    load_policy_encoder_checkpoint(source, STRIKE_AGE_ENCODER_NAME, initialization=SCALED_POLAR_INITIALIZATION,
                        expected_sha256=saved["checkpoint"]["sha256"])
            for target in ("raw", "polar_xy_v1", SCALED_ENCODER_NAME):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    policy_encoder_report(target, SCALED_POLAR_INITIALIZATION)

    def test_strike_age_warm_start_rejects_tampered_descriptor_fingerprint_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source.bin"
            saved = save_policy_weights(SavingTrainer(), source, SCALED_ENCODER_NAME)
            sha = saved["checkpoint"]["sha256"]
            for field in ("descriptor", "fingerprint", "checkpoint"):
                bad = json.loads(json.dumps(saved))
                if field == "descriptor":
                    bad["policy_observation_encoder"]["fixed_scale_divisors"]["189"] = 60.
                elif field == "fingerprint":
                    bad["policy_observation_encoder_sha256"] = "0"*64
                else:
                    bad["checkpoint"]["sha256"] = "0"*64
                source.with_suffix(".bin.manifest.json").write_text(json.dumps(bad))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_policy_encoder_checkpoint(source, STRIKE_AGE_ENCODER_NAME, initialization=SCALED_POLAR_INITIALIZATION,
                                                   expected_sha256=sha)
                with self.subTest(constructor=field), self.assertRaises(ValueError):
                    GpuStrikeAgeScaledPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=bad, checkpoint_sha256=sha,
                        initialization=SCALED_POLAR_INITIALIZATION, allow_cpu_for_tests=True)

    def test_raw_default_keeps_native_manifest_and_old_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.bin"
            result = save_policy_weights(SavingTrainer(), path)
            self.assertNotIn("policy_observation_encoder", result)
            manifest, sha = load_policy_encoder_checkpoint(path, "raw")
            self.assertEqual(sha, result["checkpoint"]["sha256"])
            self.assertEqual(manifest, result)
            self.assertEqual(load_policy_encoder_checkpoint(None, "raw"), (None, None))

    def test_transformed_save_load_and_evaluation_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoded.bin"
            trainer = SavingTrainer()
            saved = save_policy_weights(trainer, path, SCALED_ENCODER_NAME, "raw-initial-weights")
            loaded, sha = load_policy_encoder_checkpoint(path, SCALED_ENCODER_NAME)
            self.assertEqual(saved, loaded)
            self.assertEqual(saved["policy_observation_encoder"]["name"], SCALED_ENCODER_NAME)
            after = Path(directory) / "after.bin"
            frozen = save_frozen_weights(trainer, after, sha, SCALED_ENCODER_NAME)
            self.assertEqual(frozen["sha256"], sha)
            self.assertEqual(load_policy_encoder_checkpoint(after, SCALED_ENCODER_NAME)[1], sha)
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(path, "raw")
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(path, "polar_xy_v1")

    def test_transformed_missing_or_tampered_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.bin"
            trainer = SavingTrainer()
            save_policy_weights(trainer, path)
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(path, SCALED_ENCODER_NAME)
            saved = save_policy_weights(trainer, path, SCALED_ENCODER_NAME)
            path.write_bytes(b"changed weights")
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(path, SCALED_ENCODER_NAME)

    def test_pinned_raw_initialization_is_explicit_and_distinct_from_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw-initial.bin"
            saved = save_policy_weights(SavingTrainer(), path)
            sha = saved["checkpoint"]["sha256"]
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(path, SCALED_ENCODER_NAME)
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(path, SCALED_ENCODER_NAME, initialization="raw-initial-weights")
            loaded, actual = load_policy_encoder_checkpoint(path, SCALED_ENCODER_NAME,
                initialization="raw-initial-weights", expected_sha256=sha)
            encoder = GpuScaledPolarXYPolicyEncoder(1, "cpu", checkpoint_manifest=loaded,
                checkpoint_sha256=actual, initialization="raw-initial-weights", allow_cpu_for_tests=True)
            self.assertEqual(encoder.encoder_name, SCALED_ENCODER_NAME)
            with self.assertRaises(ValueError):
                load_policy_encoder_checkpoint(path, SCALED_ENCODER_NAME,
                    initialization="raw-initial-weights", expected_sha256="0" * 64)

    def test_fresh_random_requires_no_checkpoint_and_no_placeholder_hash(self):
        self.assertEqual(load_policy_encoder_checkpoint(None, SCALED_ENCODER_NAME, initialization="fresh-random"), (None, None))
        encoder = GpuScaledPolarXYPolicyEncoder(1, "cpu", initialization="fresh-random", allow_cpu_for_tests=True)
        self.assertEqual(encoder.observations.shape, (1, 223))
        with self.assertRaises(ValueError):
            load_policy_encoder_checkpoint(Path("unused.bin"), SCALED_ENCODER_NAME, initialization="fresh-random")
        with self.assertRaises(ValueError):
            load_policy_encoder_checkpoint(None, SCALED_ENCODER_NAME)

    def test_reporting_declares_changed_input_coordinate_system(self):
        result = policy_encoder_report(SCALED_ENCODER_NAME, "raw-initial-weights")
        self.assertEqual(result["policy_observation_initialization"], "raw-initial-weights")
        self.assertEqual(result["policy_observation_encoder"]["input_floats"], 223)
        self.assertIsNone(policy_encoder_report("raw")["policy_observation_encoder_sha256"])


class PublishIntegrationTests(unittest.TestCase):
    def test_strike_age_view_preserves_raw_dummy_actions_rewards_and_events(self):
        old, new = self.fixture(True), self.fixture(True)
        old.duel.observations[:, 198:200] = torch.tensor([42.76975, 118.02839])
        new.duel.observations.copy_(old.duel.observations)
        new.policy_encoder = GpuStrikeAgeScaledPolarXYPolicyEncoder(2, "cpu", initialization="fresh-random",
                                                                  allow_cpu_for_tests=True)
        original = new.duel.observations.clone()
        for _ in range(3):
            old._select()
            new._select()
            self.assertTrue(torch.equal(old.full_actions, new.full_actions))
            old._publish()
            new._publish()
            for name in ("rewards", "terminals", "action_mask"):
                self.assertTrue(torch.equal(getattr(old, name), getattr(new, name)))
            self.assertTrue(torch.equal(new.duel.observations, original))
            cols = [i for i in range(223) if i not in (198,199)]
            self.assertTrue(torch.equal(old.observations[:,cols], new.observations[:,cols]))
            self.assertTrue(torch.equal(new.observations[:,198:200], (old.observations[:,198:200].double()/120).float()))

    def fixture(self, encoded):
        # Bypass only the CUDA constructor for a CPU method-level fixture.
        # Production constructor still rejects CPU execution.
        owner = object.__new__(GpuCandidateDummyDuel)
        owner.rows = 2
        owner.facing_potential = None
        owner.reward_shaper = None
        raw = torch.zeros((4, 223), dtype=torch.float32)
        raw[:, 3] = 1
        raw[:, 86] = 1
        raw[:, 87] = 1
        raw[:, 188:190] = 120
        raw[:, 185] = 2
        owner.duel = SimpleNamespace(observations=raw, rewards=torch.arange(4, dtype=torch.float32),
            terminals=torch.zeros(4), action_mask=torch.ones((4, 33), dtype=torch.uint8),
            scheduler=SimpleNamespace(move_start_edge=torch.zeros(4)),
            combat=SimpleNamespace(tick_score_delta=torch.zeros(4)))
        owner.policy_encoder = (GpuScaledPolarXYPolicyEncoder(2, "cpu", initialization="fresh-random", allow_cpu_for_tests=True)
                                if encoded else None)
        owner.observations = torch.empty((2, 223))
        owner.rewards, owner.terminals = torch.empty(2), torch.empty(2)
        owner.action_mask = torch.empty((2, 33), dtype=torch.uint8)
        owner.learner_actions = torch.tensor([[17], [2]], dtype=torch.int32)
        owner.full_actions = torch.empty((4, 1), dtype=torch.int32)
        owner.dummy = GpuCandidateApproachDummy(2, "cpu")
        owner.combat_metrics = SimpleNamespace(update=lambda observations, terminals: self.assertIs(observations, raw))
        owner.behavior_metrics = SimpleNamespace(update=lambda observations, *args: self.assertIs(observations, raw))
        return owner

    def test_raw_publish_and_transformed_publish_preserve_physics_and_opponent(self):
        raw_owner, encoded_owner = self.fixture(False), self.fixture(True)
        original = encoded_owner.duel.observations.clone()
        pointers = [encoded_owner.observations.data_ptr(), encoded_owner.rewards.data_ptr(),
                    encoded_owner.terminals.data_ptr(), encoded_owner.action_mask.data_ptr()]
        for _ in range(3):
            raw_owner._select()
            encoded_owner._select()
            self.assertTrue(torch.equal(raw_owner.full_actions, encoded_owner.full_actions))
            raw_owner._publish()
            encoded_owner._publish()
            self.assertTrue(torch.equal(raw_owner.observations, original[0::2]))
            self.assertTrue(torch.equal(encoded_owner.duel.observations, original))
            self.assertTrue(torch.equal(encoded_owner.rewards, raw_owner.rewards))
            self.assertTrue(torch.equal(encoded_owner.terminals, raw_owner.terminals))
            self.assertTrue(torch.equal(encoded_owner.action_mask, raw_owner.action_mask))
            self.assertTrue(torch.equal(encoded_owner.observations[:, 188], torch.ones(2)))
            torch.testing.assert_close(encoded_owner.observations[:, 86], torch.full((2,), 2**0.5))
            self.assertEqual(pointers, [encoded_owner.observations.data_ptr(), encoded_owner.rewards.data_ptr(),
                                        encoded_owner.terminals.data_ptr(), encoded_owner.action_mask.data_ptr()])


if __name__ == "__main__":
    unittest.main()
