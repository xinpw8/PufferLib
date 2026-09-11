"""Actual native greedy selection and immutable-policy regression fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from gpu_native_puffer import NativeExternalGpuPuffer
from gpu_puffer_env import CudaTensorEnvAdapter
from test_gpu_native_puffer import _args
from test_gpu_native_terminal_replay import diagnostic_weights


class SelectionFixture:
    def __init__(self):
        self.observations = torch.zeros((8, 223), device="cuda")
        self.rewards = torch.zeros(8, device="cuda")
        self.terminals = torch.zeros(8, device="cuda")
        self.action_mask = torch.ones((8, 33), dtype=torch.uint8, device="cuda")
        self.actions = torch.empty((4, 8), dtype=torch.int32, device="cuda")
        self.position = 0

    def reset(self):
        self.observations.zero_()
        self.rewards.zero_()
        self.terminals.zero_()
        self.action_mask.fill_(1)
        self.position = 0

    def step(self, actions):
        self.actions[self.position].copy_(actions[:, 0])
        self.position += 1


def native_selection(greedy, zero_weights=False):
    raw = SelectionFixture()
    args = _args()
    args.update(reset_state=True, greedy_evaluation=greedy)
    trainer = NativeExternalGpuPuffer(args, CudaTensorEnvAdapter(raw, (33,)))
    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix="rek-greedy-") as directory:
            initial, after = Path(directory) / "initial.bin", Path(directory) / "after.bin"
            weights = diagnostic_weights()
            if zero_weights:
                weights.fill(0)
            weights.tofile(initial)
            trainer.load_weights(initial)
            digest = hashlib.sha256(initial.read_bytes()).hexdigest()
            for name, legal in (("all", list(range(33))), ("limited", [1, 7, 17]),
                                ("busy", [0]), ("all_again", list(range(33)))):
                raw.position = 0
                raw.action_mask.zero_()
                raw.action_mask[:, legal] = 1
                trainer.rollouts()
                got = raw.actions.cpu()
                if not torch.isin(got, torch.tensor(legal)).all():
                    raise AssertionError("native selection emitted a masked action")
                expected = min(legal) if zero_weights else max(legal)
                if greedy and not (got == expected).all():
                    raise AssertionError(f"{name}: masked argmax mismatch")
                cases.append({"name": name, "legal": legal, "actions": got.tolist(),
                              "expected_greedy": expected})
            rejected = []
            if greedy:
                for call in (trainer.train, lambda: trainer.backend.train(trainer.pufferl)):
                    try:
                        call()
                    except RuntimeError as error:
                        if "greedy evaluation" not in str(error):
                            raise
                        rejected.append(True)
                    else:
                        raise AssertionError("greedy training was not rejected")
            trainer.save_weights(after)
            if hashlib.sha256(after.read_bytes()).hexdigest() != digest:
                raise AssertionError("evaluation changed policy weights")
            extension = Path(trainer.backend.__file__)
            return {"status": "passed", "greedy": greedy, "zero_weights": zero_weights,
                    "cases": cases, "weights_unchanged": True,
                    "training_rejected_python_and_native": rejected,
                    "extension": str(extension),
                    "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()}
    finally:
        trainer.close()


class GreedyGuardTests(unittest.TestCase):
    def test_python_rejects_training_before_backend_calls(self):
        trainer = NativeExternalGpuPuffer.__new__(NativeExternalGpuPuffer)
        trainer._closed = False
        trainer.greedy_evaluation = True
        trainer.backend = SimpleNamespace(train=lambda _: self.fail("backend reached"))
        with self.assertRaisesRegex(RuntimeError, "greedy evaluation"):
            trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-cuda", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--zero-weights", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.run_cuda:
        result = native_selection(args.greedy, args.zero_weights)
        text = json.dumps(result, indent=2) + "\n"
        if args.report:
            with args.report.open("x") as stream:
                stream.write(text)
        print(text, end="")
    else:
        unittest.main(argv=[__file__])
