"""Explicit CUDA capture fixture for opt-in policy views, not game evidence."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_gpu_dummy import file_record
from gpu_candidate_dummy import GpuCandidateDummyDuel
from gpu_policy_observation_encoder import GpuPolarXYPolicyEncoder, GpuScaledPolarXYPolicyEncoder


def source_rows(rows):
    values = torch.zeros((rows, 223), dtype=torch.float32)
    values[:, 3] = 1
    values[:, 86] = 1
    values[:, 87] = torch.linspace(-1, 1, rows)
    values[:, 72] = torch.linspace(0, 90, rows)
    values[:, 158] = 45
    values[:, 188:190] = 120
    values[:, 185] = 2
    return values


class TensorDuelFixture:
    """Pure tensor transitions to exercise the production wrapper's graph."""

    def __init__(self, rows=16):
        self.rows = rows
        self.initial = source_rows(rows).cuda()
        self.observations = self.initial.clone()
        self.actions = torch.ones(rows, dtype=torch.int32, device="cuda")
        self.rewards, self.terminals = torch.zeros(rows, device="cuda"), torch.zeros(rows, device="cuda")
        self.action_mask = torch.ones((rows, 33), dtype=torch.uint8, device="cuda")
        self.scheduler = SimpleNamespace(move_start_edge=torch.zeros(rows, device="cuda"))
        self.combat = SimpleNamespace(tick_score_delta=torch.zeros(rows, device="cuda"))

    def reset(self):
        self.observations.copy_(self.initial)
        self.rewards.zero_()
        self.terminals.zero_()
        self.actions.fill_(1)

    def step(self, actions):
        self.actions.copy_(actions[:, 0])
        self.observations[:, 10].copy_(self.actions)
        self.observations[:, 189].sub_(0.02)
        self.rewards.copy_(self.actions.to(torch.float32))

    def check_status(self):
        pass

    def close(self):
        pass


def verify():
    results = []
    for constructor in (GpuPolarXYPolicyEncoder, GpuScaledPolarXYPolicyEncoder):
        cpu = constructor(32, "cpu", initialization="fresh-random", allow_cpu_for_tests=True)
        gpu = constructor(32, "cuda", initialization="fresh-random")
        raw_cpu = source_rows(32)
        raw = raw_cpu.cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gpu.encode(raw)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                gpu.encode(raw)
        stream.synchronize()
        gpu.reset_status()
        pointer = gpu.observations.data_ptr()
        maximum = 0.0
        for mutation in range(4):
            raw_cpu[:, 86].add_(0.125)
            raw_cpu[:, 87].sub_(0.0625)
            raw_cpu[:, 72].add_(0.5)
            raw_cpu[:, 189].sub_(0.5)
            raw.copy_(raw_cpu)
            graph.replay()
            actual, expected = gpu.observations.cpu(), cpu.encode(raw_cpu)
            maximum = max(maximum, (actual - expected).abs().max().item())
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
            torch.testing.assert_close(raw.cpu(), raw_cpu, rtol=0, atol=0)
            assert gpu.observations.data_ptr() == pointer
            gpu.check_status()
        raw[:, 3:7].zero_()
        graph.replay()
        try:
            gpu.check_status()
        except RuntimeError:
            pass
        else:
            raise AssertionError("captured encoder did not latch invalid quaternion")
        gpu.reset_status()
        raw.copy_(raw_cpu)
        graph.replay()
        gpu.check_status()
        results.append({"encoder": constructor.encoder_name, "source_mutations": 4,
                        "max_cpu_cuda_abs_error": maximum, "invalid_status_latched_and_cleared": True})

        raw_fixture, encoded_fixture = TensorDuelFixture(), TensorDuelFixture()
        raw_wrapper = GpuCandidateDummyDuel(raw_fixture)
        encoded_wrapper = GpuCandidateDummyDuel(encoded_fixture,
            policy_observation_encoder=constructor.encoder_name, policy_observation_initialization="fresh-random")
        actions = torch.full((8, 1), 17, dtype=torch.int32, device="cuda")
        expected_encoder = constructor(8, "cpu", initialization="fresh-random", allow_cpu_for_tests=True)
        for _ in range(3):
            raw_wrapper.step(actions)
            encoded_wrapper.step(actions)
            torch.testing.assert_close(raw_fixture.observations, encoded_fixture.observations, rtol=0, atol=0)
            torch.testing.assert_close(raw_fixture.actions, encoded_fixture.actions, rtol=0, atol=0)
            for name in ("rewards", "terminals", "action_mask"):
                torch.testing.assert_close(getattr(raw_wrapper, name), getattr(encoded_wrapper, name), rtol=0, atol=0)
            expected = expected_encoder.encode(raw_fixture.observations[0::2].cpu())
            torch.testing.assert_close(encoded_wrapper.observations.cpu(), expected, rtol=0, atol=2e-6)
        encoded_wrapper.log()
        results[-1]["captured_wrapper_steps_with_raw_state_opponent_rewards_masks_unchanged"] = 3
    return {"status": "passed", "gpu": torch.cuda.get_device_name(), "cases": results,
            "fixture_only": True, "physical_parity_or_learning_improvement_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = verify()
    result["sources"] = {name: file_record(Path(__file__).with_name(name)) for name in (
        "verify_gpu_policy_encoder.py", "gpu_policy_observation_encoder.py", "gpu_candidate_dummy.py")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
