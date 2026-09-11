from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

from run_gpu_credit_experiment import digest, effective_native_parameters, numeric_plan, resolved_config, run


class CreditExperimentTests(unittest.TestCase):
    def config(self, arm):
        directory = Path(__file__).resolve().parents[2] / "config"
        return resolved_config(directory / "default.ini", directory / "rek_g1.ini",
                               directory / f"rek_g1_credit_{arm}.ini")

    def plan(self, arm):
        return numeric_plan(self.config(arm), physical_fighters=1024,
                            total_timesteps=3276800, minibatch_size=4096)

    def test_only_credit_span_differs_between_arms(self):
        short, long = self.config("short"), self.config("long")
        differences = {(section, key) for section in short for key in short[section]
                       if short[section][key] != long[section][key]}
        self.assertEqual(differences, {("train", "horizon"), ("train", "gamma"), ("train", "gae_lambda")})

    def test_same_native_updates_steps_and_constant_lr(self):
        short, long = self.plan("short"), self.plan("long")
        for field in ("optimizer_minibatches", "learner_steps", "learning_rate", "minibatch_size"):
            self.assertEqual(short[field], long[field])
        self.assertEqual(short["optimizer_minibatches"], 3200)
        self.assertEqual(short["replay_ratio"], 4)
        self.assertEqual(short["rollouts"], 100)
        self.assertEqual(long["rollouts"], 25)
        self.assertEqual(long["simulated_seconds_per_arena"], 128)
        self.assertEqual(long["learning_rate"], .003)
        self.assertEqual(long["muon_momentum"], .9)

    def test_three_second_credit_span(self):
        short, long = self.plan("short"), self.plan("long")
        self.assertFalse(short["direct_gae_has_3_second_span"])
        self.assertTrue(long["direct_gae_has_3_second_span"])
        self.assertLess(short["direct_gae_weight_at_3_seconds"], .00011)
        self.assertGreater(long["direct_gae_weight_at_3_seconds"], .4)

    def test_extended_budget_and_effective_native_parameters(self):
        config = self.config("long")
        plan = numeric_plan(config, physical_fighters=1024,
                            total_timesteps=13107200, minibatch_size=4096)
        effective = effective_native_parameters(config, plan)
        self.assertEqual(plan["optimizer_minibatches"], 12800)
        self.assertEqual(plan["simulated_seconds_per_arena"], 512)
        self.assertEqual(plan["rollouts"], 100)
        self.assertEqual(effective["seed"], 73)
        self.assertEqual(effective["vec"]["total_agents"], 512)
        self.assertEqual(effective["train"]["learning_rate"], .003)
        self.assertFalse(effective["train"]["anneal_lr"])
        self.assertEqual(effective["train"]["horizon"], 256)
        self.assertEqual(effective["train"]["minibatch_size"], 4096)
        self.assertEqual(effective["train"]["total_timesteps"], 13107200)

    def test_rejects_annealed_or_partial_runs(self):
        config = self.config("short")
        config["train"]["anneal_lr"] = "True"
        with self.assertRaises(ValueError):
            numeric_plan(config, physical_fighters=1024, total_timesteps=3276800, minibatch_size=4096)
        with self.assertRaises(ValueError):
            numeric_plan(self.config("long"), physical_fighters=1024, total_timesteps=1024, minibatch_size=4096)

    def test_declared_round_win_h1024_numeric_plan(self):
        directory = Path(__file__).resolve().parents[2] / "config"
        config = resolved_config(directory / "default.ini", directory / "rek_g1.ini",
                                 directory / "rek_g1_round_win_lr0003_h1024.ini")
        plan = numeric_plan(config, physical_fighters=1024,
                            total_timesteps=7340032, minibatch_size=4096)
        effective = effective_native_parameters(config, plan)
        self.assertEqual(plan["horizon_ticks"], 1024)
        self.assertEqual(plan["horizon_seconds"], 20.48)
        self.assertEqual(plan["rollouts"], 14)
        self.assertEqual(plan["optimizer_minibatches"], 7168)
        self.assertEqual(plan["simulated_seconds_per_arena"], 286.72)
        self.assertEqual(plan["learning_rate"], .0003)
        self.assertEqual(plan["gamma"], 1)
        self.assertEqual(plan["gae_lambda"], .999)
        self.assertFalse(plan["anneal_lr"])
        self.assertTrue(plan["reset_state_each_horizon"])
        self.assertEqual(effective["train"]["horizon"], 1024)
        self.assertEqual(effective["train"]["total_timesteps"], 7340032)
        self.assertEqual(effective["vec"]["num_threads"], 0)
        for total, batch in ((7340031, 4096), (7340032, 1536), (7340032, 3072)):
            with self.subTest(total=total, batch=batch), self.assertRaises(ValueError):
                numeric_plan(config, physical_fighters=1024,
                             total_timesteps=total, minibatch_size=batch)
        config["train"]["horizon"] = "512"
        with self.assertRaisesRegex(ValueError, "declared"):
            numeric_plan(config, physical_fighters=1024,
                         total_timesteps=7340032, minibatch_size=4096)

    def test_prepare_only_materializes_without_importing_gpu_trainer(self):
        config_dir = Path(__file__).resolve().parents[2] / "config"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "fixture.bin"
            checkpoint.write_bytes(b"CPU preparation fixture, not a native policy")
            physics_config = directory / "physics.json"
            physics_config.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                checkpoint_sha256=digest(checkpoint), load_checkpoint=checkpoint,
                default_config=config_dir / "default.ini", native_config=config_dir / "rek_g1.ini",
                credit_config=config_dir / "rek_g1_credit_long.ini", gpu_duel_config=physics_config,
                run_dir=directory / "long", output=directory / "report.json",
                total_agents=1024, total_timesteps=3276800, minibatch_size=4096, prepare_only=True,
            )
            result = run(args)
            saved = json.loads((directory / "long.inputs" / "experiment.json").read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved["effective_native_parameters"]["seed"], 73)
            self.assertEqual(saved["effective_native_parameters"]["train"]["learning_rate"], .003)
            self.assertFalse(args.run_dir.exists())
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
