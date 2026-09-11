"""Bounded synthetic CPU/CUDA and two-graph reward transition comparison.

No physical simulator, policy training or learning-quality measurement. Inputs
change after capture, including terminal masks, sides, scores and winner flags.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from gpu_round_win_reward import (
    GpuRoundWinReward, RoundWinRewardConfig,
    SIDE, OWN_POINTS, OPPONENT_POINTS, ROUND_RESULT, ROUND_WINNER,
)


def inputs(tick, rows=16):
    gen = torch.Generator().manual_seed(7300+tick)
    current, following = [torch.randn((rows, 223), generator=gen) for _ in range(2)]
    index = torch.arange(rows)
    before, after = ((index+tick).remainder(7) == 0).float(), ((index+tick).remainder(3) == 0).float()
    for values, terminal, offset in ((current, before, 0), (following, after, 1)):
        values[:, SIDE] = (index+tick).remainder(2)
        values[:, OWN_POINTS] = torch.randint(0, 41, (rows,), generator=gen)
        values[:, OPPONENT_POINTS] = torch.randint(0, 41, (rows,), generator=gen)
        results = (index+tick+offset).remainder(4)+1
        winners = torch.where(results <= 2, (index+tick+offset).remainder(2), -1)
        values[:, ROUND_RESULT] = torch.where(terminal != 0, results, 0)
        values[:, ROUND_WINNER] = torch.where(terminal != 0, winners, -1)
    raw_reward = torch.randint(-5, 6, (rows,), generator=gen).float()
    return current, following, before, after, raw_reward


def verify():
    rows = 16
    config = RoundWinRewardConfig(1, .5, 5)
    gpu = GpuRoundWinReward(rows, "cuda", config)
    cpu = GpuRoundWinReward(rows, "cpu", config, allow_cpu_for_tests=True)
    buffers = [tensor.cuda() for tensor in inputs(0, rows)]
    current, following, before, after, raw_reward = buffers
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gpu.begin_transition(current, before)
            gpu.finish_transition(following, raw_reward, after)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    gpu.reset()
    begin, finish = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(begin): gpu.begin_transition(current, before)
    with torch.cuda.graph(finish): gpu.finish_transition(following, raw_reward, after)
    gpu.reset()
    pointer = gpu.rewards.data_ptr()
    errors = {name: 0.0 for name in ("rewards", "current_potential", "next_potential", "shaping_reward", "terminal_win")}
    for tick in range(24):
        host = inputs(tick, rows)
        for destination, source in zip(buffers, host): destination.copy_(source)
        cpu.begin_transition(host[0], host[2])
        begin.replay()
        # The wrapper mutates its live pre-step observation during physics.
        # The begin graph must already have retained a separate potential.
        current[:, OWN_POINTS].fill_(999)
        current[:, OPPONENT_POINTS].fill_(777)
        finish.replay()
        cpu.finish_transition(host[1], host[4], host[3])
        gpu.check_status()
        cpu.check_status()
        assert not gpu.pending.any().item()
        assert gpu.rewards.data_ptr() == pointer
        for name in errors:
            actual, expected = getattr(gpu, name).cpu(), getattr(cpu, name)
            errors[name] = max(errors[name], (actual-expected).abs().max().item())
            torch.testing.assert_close(actual, expected, atol=1e-7 if name == "rewards" else 1e-12, rtol=0)
        # All supplied buffers remain exact, except the intentional fixture
        # overwrite of two pre-step columns above.
        preserved = host[0].clone()
        preserved[:, OWN_POINTS], preserved[:, OPPONENT_POINTS] = 999, 777
        for actual, expected in zip(buffers, (preserved, *host[1:])):
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    gpu.reset()
    for destination, source in zip(buffers, inputs(0, rows)): destination.copy_(source)
    # Mutating a captured mask to an invalid value must still latch an error.
    after[0] = .5
    begin.replay()
    finish.replay()
    assert gpu.status[0].item() & 4
    return {"status": "passed", "gpu": torch.cuda.get_device_name(), "rows": rows,
            "transitions": 24, "separate_captured_graphs": 2, "config": config.metadata(),
            "max_cpu_cuda_abs_errors": errors, "raw_state_scores_preserved": True,
            "stable_owned_reward_pointer": True, "mutated_terminal_error_detected": True,
            "fixture_only": True, "production_integration_tested": False,
            "learning_improvement_measured": False}


def verify_production_wrapper():
    from gpu_candidate_dummy import GpuCandidateDummyDuel
    from verify_gpu_policy_encoder import TensorDuelFixture

    class RoundTensorDuelFixture(TensorDuelFixture):
        """Paired tensor-only outcomes and delayed score reset, no physics."""
        def __init__(self):
            super().__init__()
            self.arena = torch.arange(self.rows, device="cuda").floor_divide(2)
            self.side = torch.arange(self.rows, device="cuda").remainder(2)
            self.initial[:, SIDE] = self.side
            self.initial[:, ROUND_WINNER] = -1
            self.tick = 0
            self.reset()

        def reset(self):
            super().reset()
            self.tick = 0

        def step(self, actions):
            self.observations[:, 190:192].masked_fill_(self.terminals[:, None] != 0, 0)
            super().step(actions)
            self.tick += 1
            side0_delta = (self.arena+self.tick).remainder(2)
            side1_delta = 1-side0_delta
            own_delta = torch.where(self.side == 0, side0_delta, side1_delta)
            self.observations[:, 190].add_(own_delta)
            self.observations[:, 191].add_(1-own_delta)
            self.rewards.copy_(2*own_delta-1)
            self.combat.tick_score_delta.copy_(own_delta)
            self.terminals.copy_((self.arena+self.tick).remainder(3) == 0)
            result = (self.arena+self.tick).remainder(4)+1
            winner = torch.where(result <= 2, (self.arena+self.tick).remainder(2), -1)
            self.observations[:, ROUND_RESULT].copy_(torch.where(self.terminals != 0, result, 0))
            self.observations[:, ROUND_WINNER].copy_(torch.where(self.terminals != 0, winner, -1))

    config = RoundWinRewardConfig(1, .5, 5)
    raw = GpuCandidateDummyDuel(RoundTensorDuelFixture())
    shaped = GpuCandidateDummyDuel(RoundTensorDuelFixture(), round_win_reward_config=config)
    assert raw.round_win_reward is None and raw.reward_shaper is None
    assert shaped.facing_potential is None and shaped.reward_shaper is shaped.round_win_reward
    oracle = GpuRoundWinReward(raw.rows, "cpu", config, allow_cpu_for_tests=True)
    pointer, maximum = shaped.rewards.data_ptr(), 0.0
    for tick in range(18):
        if tick == 9:
            for owner in (raw, shaped): owner.reset()
            oracle.reset()
            assert not shaped.round_win_reward.pending.any().item()
            torch.testing.assert_close(shaped.rewards, torch.zeros_like(shaped.rewards), atol=0, rtol=0)
        oracle.begin_transition(raw.duel.observations[0::2].cpu(), raw.duel.terminals[0::2].cpu())
        actions = torch.full((raw.rows, 1), 17 if tick % 2 else 6, dtype=torch.int32, device="cuda")
        raw.step(actions)
        shaped.step(actions)
        expected = oracle.finish_transition(raw.duel.observations[0::2].cpu(), raw.duel.rewards[0::2].cpu(), raw.duel.terminals[0::2].cpu())
        maximum = max(maximum, (shaped.rewards.cpu()-expected).abs().max().item())
        torch.testing.assert_close(shaped.rewards.cpu(), expected, atol=1e-7, rtol=0)
        assert shaped.rewards.data_ptr() == pointer
        for field in ("observations", "terminals", "action_mask"):
            torch.testing.assert_close(getattr(raw, field), getattr(shaped, field), atol=0, rtol=0)
        for field in ("observations", "rewards", "terminals", "actions"):
            torch.testing.assert_close(getattr(raw.duel, field), getattr(shaped.duel, field), atol=0, rtol=0)
        torch.testing.assert_close(raw.full_actions, shaped.full_actions, atol=0, rtol=0)
        assert raw.metric_plugin.snapshot(clear=False) == shaped.metric_plugin.snapshot(clear=False)
        raw.log()
        shaped.log()
    return {"status": "passed", "steps": 18, "explicit_reset_after_steps": 9,
            "max_cpu_cuda_abs_error": maximum, "production_wrapper_captured": True,
            "raw_state_scores_opponent_actions_masks_metrics_unchanged": True,
            "physics_or_training_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    report = verify()
    report["production_wrapper"] = verify_production_wrapper()
    report["production_integration_tested"] = True
    report["sources"] = {}
    for name in ("gpu_round_win_reward.py", "verify_gpu_round_win_reward.py", "gpu_candidate_dummy.py",
                 "verify_gpu_policy_encoder.py"):
        path = Path(__file__).with_name(name)
        report["sources"][name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
