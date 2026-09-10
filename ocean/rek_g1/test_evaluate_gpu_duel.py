import tempfile
from pathlib import Path
import unittest

import torch

try:
    from . import evaluate_gpu_duel as evaluation
    from .gpu_metrics import (
        ROUND_RESULT,
        ROUND_WINNER,
        ROUND_WON_BY_POINTS,
        SIDE0_POINTS,
        SIDE1_POINTS,
        TICK_ATTRIBUTED_CONTACTS,
        TICK_SCORED_HITS,
    )
    from .gpu_puffer_env import CudaTensorEnvAdapter
    from .gpu_puffer_policy import NativePufferPolicy
except ImportError:
    import evaluate_gpu_duel as evaluation
    from gpu_metrics import (
        ROUND_RESULT,
        ROUND_WINNER,
        ROUND_WON_BY_POINTS,
        SIDE0_POINTS,
        SIDE1_POINTS,
        TICK_ATTRIBUTED_CONTACTS,
        TICK_SCORED_HITS,
    )
    from gpu_puffer_env import CudaTensorEnvAdapter
    from gpu_puffer_policy import NativePufferPolicy


class _CudaMatchFixture:
    def __init__(self, total_agents: int = 8) -> None:
        self.total_agents = total_agents
        self.arenas = total_agents // 2
        self.device = torch.device("cuda")
        self.observations = torch.zeros(
            total_agents, 223, dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(
            total_agents, dtype=torch.float32, device=self.device
        )
        self.terminals = torch.zeros_like(self.rewards)
        self.action_mask = torch.zeros(
            total_agents, 33, dtype=torch.bool, device=self.device
        )
        rows = torch.arange(total_agents, device=self.device)
        self.action_mask[rows, rows.remainder(33)] = True
        self.step_count = torch.zeros((), dtype=torch.int64, device=self.device)
        self.invalid_actions = torch.zeros((), dtype=torch.int64, device=self.device)
        self.arena_ids = torch.arange(
            self.arenas, dtype=torch.float32, device=self.device
        )

    def reset(self) -> None:
        self.observations.zero_()
        self.rewards.zero_()
        self.terminals.zero_()
        self.step_count.zero_()
        self.invalid_actions.zero_()

    def step(self, actions: torch.Tensor) -> None:
        selected = self.action_mask.gather(1, actions.to(torch.int64))
        self.invalid_actions.add_(torch.count_nonzero(~selected))
        self.step_count.add_(1)
        terminal = self.step_count.remainder(4).eq(0)
        self.terminals.copy_(terminal.to(torch.float32).expand_as(self.terminals))
        self.observations.zero_()
        even = self.observations[0::2]
        even[:, SIDE0_POINTS].fill_(5.0)
        even[:, SIDE1_POINTS].fill_(2.0)
        even[:, TICK_SCORED_HITS].fill_(1.0)
        even[:, TICK_ATTRIBUTED_CONTACTS].fill_(2.0)
        even[:, ROUND_RESULT].copy_(
            terminal.to(torch.float32).expand(self.arenas) * ROUND_WON_BY_POINTS
        )
        winners = self.arena_ids.remainder(2.0)
        even[:, ROUND_WINNER].copy_(
            torch.where(terminal, winners, torch.full_like(winners, -1.0))
        )

    def close(self) -> None:
        pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SwappedCheckpointEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        self.fixture = _CudaMatchFixture()
        self.env = CudaTensorEnvAdapter(self.fixture, (33,))
        source = NativePufferPolicy(223, (33,), device="cuda")
        self.temporary = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temporary.name) / "same.bin"
        self.digest = source.save_native_checkpoint(self.checkpoint)
        self.policy_a = NativePufferPolicy(223, (33,), device="cuda")
        self.policy_b = NativePufferPolicy(223, (33,), device="cuda")
        self.assertEqual(self.policy_a.load_native_checkpoint(self.checkpoint), self.digest)
        self.assertEqual(self.policy_b.load_native_checkpoint(self.checkpoint), self.digest)
        self.evaluator = evaluation.SwappedCheckpointEvaluator(
            self.env, self.policy_a, self.policy_b
        )

    def tearDown(self) -> None:
        self.env.close()
        self.temporary.cleanup()

    def test_same_checkpoint_routes_complementary_interleaved_rows(self) -> None:
        self.assertEqual(
            self.evaluator.checkpoint_a_rows.cpu().tolist(), [0, 2, 5, 7]
        )
        self.assertEqual(
            self.evaluator.checkpoint_b_rows.cpu().tolist(), [1, 3, 4, 6]
        )
        result = self.evaluator.run(4, deterministic=True)
        metrics = result["metrics"]

        self.assertEqual(int(self.fixture.invalid_actions.cpu()), 0)
        self.assertEqual(metrics["action_mask_violations"], 0)
        self.assertEqual(metrics["terminal_state_reset_max_abs"], 0.0)
        self.assertEqual(metrics["completed_environment_episodes"], 4)
        self.assertEqual(metrics["truncated_environment_episodes_at_cutoff"], 0)
        self.assertEqual(metrics["attempted_environment_episodes"], 4)
        self.assertEqual(metrics["completed"]["checkpoint_a_wins"], 2)
        self.assertEqual(metrics["completed"]["checkpoint_b_wins"], 2)
        self.assertEqual(metrics["completed"]["checkpoint_a_points"], 14.0)
        self.assertEqual(metrics["completed"]["checkpoint_b_points"], 14.0)
        self.assertEqual(metrics["completed"]["scored_hits"], 16.0)
        self.assertEqual(metrics["completed"]["attributed_contacts"], 32.0)
        self.assertEqual(
            metrics["role_balance"]["checkpoint_a_as_side0"],
            {
                "completed": 2,
                "wins": 1,
                "win_rate": 0.5,
                "points": 10.0,
                "points_per_round": 5.0,
            },
        )
        self.assertEqual(
            metrics["role_balance"]["checkpoint_a_as_side1"],
            {
                "completed": 2,
                "wins": 1,
                "win_rate": 0.5,
                "points": 4.0,
                "points_per_round": 2.0,
            },
        )

    def test_cutoff_rounds_are_reported_but_excluded_from_outcomes(self) -> None:
        result = self.evaluator.run(3, deterministic=True)
        metrics = result["metrics"]
        self.assertEqual(metrics["completed_environment_episodes"], 0)
        self.assertEqual(metrics["truncated_environment_episodes_at_cutoff"], 4)
        self.assertIsNone(metrics["completed"]["checkpoint_a_round_win_rate"])
        truncated = metrics[
            "truncated_at_cutoff_excluded_from_completed_statistics"
        ]
        self.assertEqual(truncated["episodes"], 4)
        self.assertEqual(truncated["scored_hits"], 12.0)
        self.assertEqual(truncated["semantic_steps"], 12.0)

    def test_checkpoint_hash_pin_rejects_changed_payload(self) -> None:
        record = evaluation._pinned_checkpoint(self.checkpoint, self.digest)
        self.assertEqual(record["sha256"], self.digest)
        changed = Path(self.temporary.name) / "changed.bin"
        changed.write_bytes(self.checkpoint.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            evaluation._pinned_checkpoint(changed, self.digest)


if __name__ == "__main__":
    unittest.main()
