"""Bounded CUDA tensor/capture fixture for opt-in facing reward integration."""

import argparse
import json
from pathlib import Path

import torch

from evaluate_gpu_dummy import file_record
from gpu_candidate_dummy import GpuCandidateDummyDuel
from gpu_facing_potential import FacingPotentialConfig, GpuFacingPotential
from verify_gpu_policy_encoder import TensorDuelFixture


class TerminalTensorDuelFixture(TensorDuelFixture):
    """Synthetic delayed-reset boundaries; no real physics or training."""

    def __init__(self):
        super().__init__()
        self.ticks = torch.zeros(self.rows, dtype=torch.int32, device="cuda")
        self.offsets = torch.arange(self.rows, device="cuda").floor_divide(2)

    def reset(self):
        super().reset()
        self.ticks.zero_()

    def step(self, actions):
        super().step(actions)
        self.ticks.add_(1)
        self.observations[:, 86].copy_(torch.where((self.ticks+self.offsets).remainder(2) == 0, 1., -1.))
        self.terminals.copy_((self.ticks+self.offsets).remainder(3) == 0)


def verify():
    config = FacingPotentialConfig(.9, .25)
    raw = GpuCandidateDummyDuel(TerminalTensorDuelFixture())
    zero = GpuCandidateDummyDuel(TerminalTensorDuelFixture(), facing_potential_config=FacingPotentialConfig(.9, 0))
    shaped = GpuCandidateDummyDuel(TerminalTensorDuelFixture(), facing_potential_config=config)
    assert raw.facing_potential is None and zero.facing_potential is None
    assert not shaped.facing_potential.pending.any().item()
    oracle = GpuFacingPotential(raw.rows, "cpu", config, allow_cpu_for_tests=True)
    pointer = shaped.rewards.data_ptr()
    maximum = 0.0
    for step in range(12):
        if step == 6:
            raw.reset()
            zero.reset()
            shaped.reset()
            oracle.reset()
            assert not shaped.facing_potential.pending.any().item()
        oracle.begin_transition(raw.duel.observations[0::2].cpu(), raw.duel.terminals[0::2].cpu())
        actions = torch.full((raw.rows, 1), 17 if step % 2 else 6, dtype=torch.int32, device="cuda")
        raw.step(actions)
        zero.step(actions)
        shaped.step(actions)
        expected = oracle.finish_transition(raw.duel.observations[0::2].cpu(), raw.duel.rewards[0::2].cpu(), raw.duel.terminals[0::2].cpu())
        maximum = max(maximum, (expected-shaped.rewards.cpu()).abs().max().item())
        torch.testing.assert_close(shaped.rewards.cpu(), expected, atol=2e-6, rtol=0)
        torch.testing.assert_close(raw.rewards, zero.rewards, atol=0, rtol=0)
        for field in ("observations", "terminals", "action_mask"):
            torch.testing.assert_close(getattr(raw, field), getattr(shaped, field), atol=0, rtol=0)
            torch.testing.assert_close(getattr(raw, field), getattr(zero, field), atol=0, rtol=0)
        for field in ("observations", "rewards", "terminals", "actions"):
            torch.testing.assert_close(getattr(raw.duel, field), getattr(shaped.duel, field), atol=0, rtol=0)
        assert shaped.rewards.data_ptr() == pointer
        assert not shaped.facing_potential.pending.any().item()
        raw.log()
        zero.log()
        shaped.log()
    return {"status": "passed", "gpu": torch.cuda.get_device_name(), "captured_steps": 12,
            "explicit_reset_after_steps": 6, "mixed_terminal_every_ticks": 3,
            "max_cpu_cuda_abs_error": maximum, "zero_scale_allocates_no_shaper": True,
            "raw_state_dummy_rewards_masks_unchanged": True, "fixture_only": True,
            "learning_improvement_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = verify()
    report["sources"] = {name: file_record(Path(__file__).with_name(name)) for name in (
        "verify_gpu_facing_potential.py", "gpu_facing_potential.py", "gpu_candidate_dummy.py")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
