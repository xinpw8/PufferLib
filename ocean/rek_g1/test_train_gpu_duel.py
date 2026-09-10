"""Failed training must retain diagnostic state without declaring success."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from train_gpu_duel import save_failure


class StructFixture:
    def __init__(self):
        self.tensor = torch.arange(16, dtype=torch.uint8).reshape(2, 8)

    def field(self, name):
        return torch.tensor([0, 3], dtype=torch.int32)


class TrainerFixture:
    global_step = 8192

    def save_weights(self, path):
        path.write_bytes(b"diagnostic-fixture")
        return {"checkpoint": {"path": str(path)}, "diagnostic_fixture": True}


class BrokenTensor:
    def detach(self):
        raise RuntimeError("fixture download failure")


def environment_fixture():
    return SimpleNamespace(
        scheduler=SimpleNamespace(rows=StructFixture(), statuses=torch.tensor([0, 311])),
        motion=SimpleNamespace(
            composers=StructFixture(), matchers=StructFixture(), slots=StructFixture(),
            route_commands=torch.zeros((2, 8), dtype=torch.uint8),
        ),
        actions=torch.ones(2), action_mask=torch.ones((2, 33), dtype=torch.uint8),
        observations=torch.full((2, 223), float("nan")),
        physics=SimpleNamespace(qpos=torch.zeros((1, 72)), qvel=torch.zeros((1, 70))),
    )


class FailureEvidenceTests(unittest.TestCase):
    def test_failure_preserves_state_and_marks_diagnostic_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(run_dir=Path(directory), check_every_step=True)
            environment = environment_fixture()
            report = save_failure(args, environment, TrainerFixture(), RuntimeError("311"), 2, 65)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["capture_errors"], [])
            self.assertEqual(report["diagnostic_checked_ticks"], 65)
            self.assertTrue(report["diagnostic_only"])
            self.assertEqual(len(report["snapshots"]), 12)
            np.testing.assert_array_equal(
                np.load(args.run_dir / "failure-scheduler_status.npy", allow_pickle=False), [0, 311],
            )
            self.assertTrue(np.isnan(np.load(args.run_dir / "failure-observations.npy")).all())
            self.assertEqual(json.loads((args.run_dir / "failure.json").read_text())["status"], "failed")
            self.assertTrue((args.run_dir / "failed-policy.bin").is_file())
            self.assertFalse((args.run_dir / "final.bin").exists())

    def test_one_failed_download_does_not_discard_other_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(run_dir=Path(directory), check_every_step=True)
            environment = environment_fixture()
            environment.actions = BrokenTensor()
            report = save_failure(args, environment, TrainerFixture(), RuntimeError("311"), 2, 65)
            self.assertEqual(len(report["capture_errors"]), 1)
            self.assertIn("actions: RuntimeError", report["capture_errors"][0])
            self.assertIn("qvel", report["snapshots"])
            self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
