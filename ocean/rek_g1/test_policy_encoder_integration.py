"""CPU-only checkpoint/load and raw-versus-policy-view integration fixtures."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from evaluate_gpu_dummy import save_frozen_weights
from gpu_candidate_dummy import GpuCandidateApproachDummy, GpuCandidateDummyDuel
from gpu_policy_observation_encoder import (
    GpuScaledPolarXYPolicyEncoder, SCALED_ENCODER_NAME,
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
