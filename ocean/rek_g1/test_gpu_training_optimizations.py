"""CPU-only config and reporting regressions for opt-in GPU optimizations.

Mocked training below tests orchestration and metadata, not CUDA execution or
dynamics equivalence. Native policy execution remains a scheduled GPU test.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from gpu_puffer_policy import NativePufferPolicy
from gpu_semantic_duel import GpuDuelConfig
import train_gpu_duel
from verify_gpu_duel import load_config


ROOT = Path(__file__).resolve().parents[2]
PATH_FIELDS = ("model", "assets", "controller_manifest", "controller_source",
               "motion_features", "motion_library", "combat_library")


def raw_config(**updates):
    result = {key: f"fixture-{key}" for key in PATH_FIELDS}
    result.update(model_sha256="a" * 64, assets_sha256="b" * 64,
                  move_duration_ticks=list(range(1, 18)))
    result.update(updates)
    return result


def config_fixture(directory, **updates):
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(raw_config(**updates)), encoding="utf-8")
    return path, load_config(path)


class OptimizationConfigTests(unittest.TestCase):
    def test_old_config_preserves_original_defaults_and_held_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = config_fixture(directory)
        self.assertIsInstance(config, GpuDuelConfig)
        self.assertIs(config.conditional_reset_forward, False)
        self.assertIsNone(config.fused_combat_library)
        self.assertEqual(config.locomotion_segment_ticks, 1)
        self.assertEqual(config.move_duration_ticks, tuple(range(1, 18)))
        for name in PATH_FIELDS:
            self.assertIsInstance(getattr(config, name), Path)

    def test_explicit_optimizations_and_fused_library_path_survive_load(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = config_fixture(directory, conditional_reset_forward=True,
                                       fused_combat_library="fixture-fused.so")
        self.assertIs(config.conditional_reset_forward, True)
        self.assertEqual(config.fused_combat_library, Path("fixture-fused.so"))
        self.assertEqual(config.locomotion_segment_ticks, 1)

    def test_explicit_null_library_retains_reference_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = config_fixture(directory, fused_combat_library=None)
        self.assertIsNone(config.fused_combat_library)

    def test_unknown_configuration_option_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(TypeError):
            config_fixture(directory, invented_optimization=True)

    def test_native_policy_cpu_execution_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            NativePufferPolicy(223, (33,), device="cpu")

    def test_real_native_config_keeps_policy_shape_and_one_tick_held_control(self):
        config = train_gpu_duel.load_native_config(ROOT / "config/default.ini", ROOT / "config/rek_g1.ini")
        self.assertEqual(config["policy"]["hidden_size"], 256)
        self.assertEqual(config["policy"]["num_layers"], 2)
        self.assertEqual(config["env"]["locomotion_segment_ticks"], 1)
        self.assertEqual(config["env_name"], "rek_g1")


class ReferenceMeasurementFixture:
    pass


class FusedMeasurementFixture:
    pass


class TrainerFixture:
    def __init__(self, extension, batch):
        self.global_step = self.epoch = 0
        self.batch = batch
        self.closed = False
        self.backend = SimpleNamespace(__file__=str(extension), precision_bytes=4)
        self.pufferl = SimpleNamespace(native_env_count=0, has_env_threads=False)

    def rollouts(self):
        self.global_step += self.batch

    def train(self):
        self.epoch += 1

    def save_weights(self, path):
        path.write_bytes(f"fixture-policy-{self.epoch}".encode())
        return {"fixture": True}

    def log(self, **kwargs):
        return {"env": {"n": 0}}

    def close(self):
        self.closed = True


class TimerFixture:
    def call(self, name, function, *args, **kwargs):
        return function(*args, **kwargs)

    def snapshot(self):
        return {"fixture": True}


class TrainingMetadataTests(unittest.TestCase):
    def test_reference_and_optimized_metadata_without_gpu(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                library = directory / "fixture-library.so"
                library.write_bytes(b"fixture, not a native library")
                config_path, config = config_fixture(
                    directory, conditional_reset_forward=optimized,
                    fused_combat_library=str(library) if optimized else None,
                )
                physics = SimpleNamespace(model=SimpleNamespace(is_sparse=True))
                environment = SimpleNamespace(
                    config=config, rows=4, actions=torch.ones(4), physics=physics,
                    reset_forward_gate=object() if optimized else None,
                    measurement=FusedMeasurementFixture() if optimized else ReferenceMeasurementFixture(),
                    capture_step=Mock(), check_status=Mock(), close=Mock(),
                )
                trainer = TrainerFixture(library, batch=8)
                args = SimpleNamespace(
                    output=directory / "report.json", run_dir=directory / "run",
                    total_timesteps=8, total_agents=4, horizon=2, minibatch_size=8,
                    log_every=1, checkpoint_every=0, reward_clip=0.0, load_checkpoint=None,
                    check_every_step=False, opponent="self-play", gpu_duel_config=config_path,
                    default_config=ROOT / "config/default.ini", native_config=ROOT / "config/rek_g1.ini",
                )
                with patch.object(train_gpu_duel, "GpuSemanticDuel", return_value=environment), \
                     patch.object(train_gpu_duel, "create_rek_g1_native_puffer", return_value=trainer), \
                     patch.object(train_gpu_duel, "TrainingPhaseTimer", return_value=TimerFixture()), \
                     patch.object(torch.cuda, "synchronize"), \
                     patch.object(torch.cuda, "get_device_name", return_value="CPU-only fixture"), \
                     redirect_stdout(io.StringIO()):
                    report = train_gpu_duel.train(args)
                details = report["training"]
                self.assertTrue(details["physics_sparse_jacobian"])
                self.assertEqual(details["conditional_reset_forward"], optimized)
                self.assertEqual(details["combat_measurement_backend"], type(environment.measurement).__name__)
                self.assertEqual(details["agent_steps"], 8)
                self.assertEqual(details["native_cpu_environment_count"], 0)
                self.assertFalse(details["native_environment_threads"])
                if optimized:
                    self.assertEqual(report["inputs"]["fused_combat_library"]["sha256"],
                                     train_gpu_duel._sha256(library))
                else:
                    self.assertIsNone(report["inputs"]["fused_combat_library"])
                self.assertEqual(json.loads(args.output.read_text())["training"], details)
                self.assertTrue(trainer.closed)
                environment.capture_step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
