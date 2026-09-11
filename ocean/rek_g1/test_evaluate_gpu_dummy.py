"""CPU-only evaluation orchestration and completed-round accounting fixtures."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from evaluate_gpu_dummy import (
    RoundJournal, evaluation_config, file_record, frozen_rollouts,
    ongoing_snapshot, pinned_checkpoint, save_frozen_weights,
)
from gpu_metrics import (
    ROUND_RESULT, ROUND_WINNER, SIDE0_POINTS, SIDE1_POINTS,
    TICK_ATTRIBUTED_CONTACTS, TICK_SCORED_HITS,
    RekG1GpuMetricCollector,
)
from gpu_behavior_metrics import GpuBehaviorMetricCollector


def fixture(arenas=2):
    return SimpleNamespace(rows=arenas * 2, observations=torch.zeros((arenas * 2, 223)),
                           terminals=torch.zeros(arenas * 2))


def terminal(duel, arena, result=1, winner=0, points=(9, 4)):
    duel.terminals[2 * arena:2 * arena + 2] = 1
    row = duel.observations[2 * arena]
    row[ROUND_RESULT], row[ROUND_WINNER] = result, winner
    row[SIDE0_POINTS], row[SIDE1_POINTS] = points


class EvaluationConfigTests(unittest.TestCase):
    def test_native_seed_is_base_seed_and_original_is_preserved(self):
        original = {"vec": {"total_agents": 8}, "train": {"seed": 42}, "seed": 73,
                    "reset_state": True, "policy": {"hidden_size": 256}}
        result = evaluation_config(original, fighters=8, horizon=64, ticks=128, seed=123)
        self.assertEqual(result["seed"], 123)
        self.assertEqual(result["train"]["seed"], 42)
        self.assertEqual(result["train"]["total_timesteps"], 512)
        self.assertEqual(result["train"]["minibatch_size"], 256)
        self.assertEqual(result["vec"], {"total_agents": 4, "num_buffers": 1, "num_threads": 0})
        self.assertTrue(result["reset_state"])
        self.assertEqual(original["vec"]["total_agents"], 8)

    def test_invalid_shapes_ticks_and_seed_fail(self):
        valid = dict(fighters=8, horizon=64, ticks=128, seed=73)
        for update in ({"fighters": 3}, {"fighters": 0}, {"ticks": 127}, {"ticks": 0},
                       {"horizon": 1}, {"seed": -1}, {"seed": 2**31},
                       {"minibatch_size": 100}, {"minibatch_size": 512}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                evaluation_config({"vec": {}, "train": {}}, **(valid | update))


class JournalTests(unittest.TestCase):
    def test_cross_horizon_rounds_and_multiple_outcomes(self):
        duel, journal = fixture(), RoundJournal(2, 2, "cpu")
        duel.observations[0::2, TICK_SCORED_HITS] = torch.tensor([1., 2.])
        duel.observations[0::2, TICK_ATTRIBUTED_CONTACTS] = 3
        journal.record(duel.observations, duel.terminals)
        terminal(duel, 0)
        journal.record(duel.observations, duel.terminals)
        first = journal.drain(0)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["end_control_tick"], 2)
        self.assertEqual(first[0]["control_ticks"], 2)
        self.assertEqual(first[0]["arena_scored_hits"], 2)
        self.assertEqual(first[0]["arena_attributed_contacts"], 6)
        self.assertEqual(first[0]["outcome"], "learner_win")
        duel.terminals.zero_()
        journal.record(duel.observations, duel.terminals)
        terminal(duel, 0, result=3, winner=-1, points=(4, 4))
        terminal(duel, 1, result=2, winner=1, points=(4, 9))
        journal.record(duel.observations, duel.terminals)
        second = journal.drain(2)
        self.assertEqual([event["outcome"] for event in second], ["tie", "opponent_win"])
        self.assertEqual([event["control_ticks"] for event in second], [2, 4])
        self.assertEqual([event["end_control_tick"] for event in second], [4, 4])
        self.assertEqual(second[1]["arena_scored_hits"], 8)
        self.assertEqual(journal.ongoing.tolist(), [[0., 0., 0.], [0., 0., 0.]])

    def test_redo_and_no_completed_rounds(self):
        duel, journal = fixture(1), RoundJournal(1, 2, "cpu")
        journal.record(duel.observations, duel.terminals)
        journal.record(duel.observations, duel.terminals)
        self.assertEqual(journal.drain(0), [])
        terminal(duel, 0, result=4, winner=-1)
        journal.record(duel.observations, duel.terminals)
        duel.terminals.zero_()
        journal.record(duel.observations, duel.terminals)
        self.assertEqual(journal.drain(2)[0]["outcome"], "redo")
        self.assertEqual(journal.ongoing[0, 0], 1)

    def test_invalid_terminal_metadata_fails(self):
        for result, winner, points in ((0, -1, (0, 0)), (1, -1, (1, 0)),
                                       (3, 0, (1, 1)), (1, 0, (float("nan"), 0)),
                                       (1, 0, (1.5, 0))):
            with self.subTest(result=result, winner=winner, points=points):
                duel, journal = fixture(1), RoundJournal(1, 2, "cpu")
                journal.record(duel.observations, duel.terminals)
                terminal(duel, 0, result=result, winner=winner, points=points)
                journal.record(duel.observations, duel.terminals)
                with self.assertRaises(RuntimeError):
                    journal.drain(0)

    def test_partial_or_excess_horizon_fails(self):
        duel, journal = fixture(1), RoundJournal(1, 2, "cpu")
        journal.record(duel.observations, duel.terminals)
        with self.assertRaises(RuntimeError):
            journal.drain(0)
        journal.record(duel.observations, duel.terminals)
        with self.assertRaises(RuntimeError):
            journal.record(duel.observations, duel.terminals)

    def test_ongoing_excludes_terminal_arenas_and_preserves_points(self):
        duel, journal = fixture(), RoundJournal(2, 2, "cpu")
        terminal(duel, 0)
        duel.observations[2, SIDE0_POINTS] = 3
        duel.observations[2, SIDE1_POINTS] = 6
        journal.record(duel.observations, duel.terminals)
        result = ongoing_snapshot(duel, journal)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["arena"], 1)
        self.assertEqual(result[0]["learner_points"], 3)
        self.assertEqual(result[0]["opponent_points"], 6)
        self.assertEqual(result[0]["control_ticks"], 1)

    def test_round_outcomes_match_existing_training_metric_path(self):
        duel, journal = fixture(), RoundJournal(2, 2, "cpu")
        behavior = GpuBehaviorMetricCollector(4, "cpu", learner_rows=(0, 2))
        combat = RekG1GpuMetricCollector(4, "cpu")
        actions = torch.ones((4, 1), dtype=torch.int32)
        duel.observations[:, 3] = 1
        events = []
        for tick in range(4):
            duel.terminals.zero_()
            if tick == 1:
                terminal(duel, 0, winner=0, points=(9, 4))
            if tick == 3:
                terminal(duel, 1, winner=1, points=(3, 9))
            combat.update(duel.observations, duel.terminals)
            behavior.update(duel.observations, duel.terminals, actions)
            journal.record(duel.observations, duel.terminals)
            if tick % 2:
                events.extend(journal.drain(tick - 1))
        combat_log, behavior_log = combat.snapshot(clear=False), behavior.snapshot(clear=False)
        self.assertEqual(len(events), combat_log["n"])
        self.assertEqual(len(events), behavior_log["completed_rounds"])
        self.assertEqual(sum(e["outcome"] == "learner_win" for e in events),
                         behavior_log["learner_round_wins"])
        self.assertEqual(sum(e["learner_points"] for e in events) / len(events),
                         combat_log["side0_points_per_round"])
        self.assertEqual(sum(e["opponent_points"] for e in events) / len(events),
                         behavior_log["opponent_points_per_completed_round"])


class FrozenPolicyTests(unittest.TestCase):
    def test_pinned_and_frozen_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fixture.bin"
            checkpoint.write_bytes(b"frozen-policy-fixture")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(pinned_checkpoint(checkpoint, digest)["sha256"], digest)
            trainer = SimpleNamespace(save_weights=lambda path: path.write_bytes(checkpoint.read_bytes()))
            final = save_frozen_weights(trainer, Path(directory) / "final.bin", digest)
            self.assertEqual(final["sha256"], digest)
            self.assertEqual(file_record(checkpoint)["bytes"], 21)
            with self.assertRaises(ValueError):
                pinned_checkpoint(checkpoint, "0" * 64)
            with self.assertRaises(ValueError):
                pinned_checkpoint(checkpoint, "not-a-digest")
            trainer.save_weights = lambda path: path.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                save_frozen_weights(trainer, Path(directory) / "changed.bin", digest)

    def test_native_rollouts_only_and_step_wrapper_restored(self):
        duel, journal = fixture(), RoundJournal(2, 2, "cpu")
        actions = torch.ones((2, 1), dtype=torch.int32)
        calls = []
        def step(actual):
            self.assertIs(actual, actions)
            calls.append("step")
        trainer = SimpleNamespace(env=SimpleNamespace(step=step), global_step=0)
        def rollouts():
            for _ in range(2):
                trainer.env.step(actions)
            trainer.global_step += 4
        trainer.rollouts = rollouts
        trainer.train = lambda: self.fail("evaluation must never perform a policy update")
        reports = []
        frozen_rollouts(trainer, duel, journal, ticks=6, horizon=2,
                        on_horizon=lambda ticks, events: reports.append((ticks, events)))
        self.assertEqual(len(calls), 6)
        self.assertEqual(reports, [(2, []), (4, []), (6, [])])
        self.assertIs(trainer.env.step, step)

    def test_failure_restores_step_and_does_not_claim_full_ticks(self):
        duel, journal = fixture(1), RoundJournal(1, 2, "cpu")
        original = lambda actions: None
        trainer = SimpleNamespace(env=SimpleNamespace(step=original), global_step=0,
                                  rollouts=lambda: None)
        reports = []
        with self.assertRaises(RuntimeError):
            frozen_rollouts(trainer, duel, journal, ticks=4, horizon=2,
                            on_horizon=lambda *event: reports.append(event))
        self.assertEqual(reports, [])
        self.assertIs(trainer.env.step, original)

    def test_fractional_arena_tick_count_is_rejected(self):
        duel, journal = fixture(), RoundJournal(2, 2, "cpu")
        trainer = SimpleNamespace(env=SimpleNamespace(step=lambda actions: None), global_step=0)
        def rollouts():
            for _ in range(2):
                trainer.env.step(None)
            trainer.global_step += 5
        trainer.rollouts = rollouts
        with self.assertRaisesRegex(RuntimeError, "learner-step count"):
            frozen_rollouts(trainer, duel, journal, ticks=2, horizon=2,
                            on_horizon=lambda *event: self.fail("invalid ticks reported"))


if __name__ == "__main__":
    unittest.main()
