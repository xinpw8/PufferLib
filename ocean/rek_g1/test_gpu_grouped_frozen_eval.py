"""CPU ordering tests; opt-in synthetic CUDA comparison needs no REK physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from gpu_grouped_frozen_eval import PolicyGroup, _PolicyRows, grouped_ticks, validate_groups


class FakeBackend:
    def __init__(self, log):
        self.log = log

    def external_rollout_begin(self, handle, stream):
        assert not handle.active
        handle.active, handle.step = True, 0
        self.log.append((handle.name, "begin", stream))

    def external_rollout_step(self, handle, step, stream):
        assert handle.active and handle.step == step
        handle.step += 1
        self.log.append((handle.name, "infer", step))

    def external_actions_to_int32(self, handle, pointer, stream):
        self.log.append((handle.name, "actions", pointer))

    def external_rollout_finish(self, handle, stream):
        assert handle.active and handle.step == handle.horizon
        handle.active = False
        handle.global_step += handle.arenas * handle.horizon
        self.log.append((handle.name, "finish", stream))


def fake_groups():
    log, groups = [], []
    for name, horizon, arenas in (("short", 4, 2), ("long", 8, 3)):
        spec = PolicyGroup(name, arenas, Path("weights.bin"), "0" * 64, horizon)
        handle = SimpleNamespace(name=name, horizon=horizon, arenas=arenas, active=False, global_step=0)
        trainer = SimpleNamespace(backend=FakeBackend(log), pufferl=handle)
        groups.append(SimpleNamespace(spec=spec, trainer=trainer, actions=SimpleNamespace(data_ptr=lambda: 42)))
    return groups, log


class GroupedOrderingTests(unittest.TestCase):
    def test_strike_age_policy_rows_refresh_retains_other_groups_raw_fields(self):
        import torch
        from gpu_policy_observation_encoder import GpuStrikeAgeScaledPolarXYPolicyEncoder

        raw = torch.zeros((4,223), dtype=torch.float32)
        raw[:,3] = 1
        raw[:,86] = 1
        raw[:,198:200] = torch.tensor([60.,120.])
        raw[:,190:192] = torch.tensor([23.,14.])
        original = raw.clone()
        candidate = SimpleNamespace(observations=raw, rewards=torch.arange(4.), terminals=torch.zeros(4),
                                    action_mask=torch.ones((4,33), dtype=torch.uint8))
        encoder = GpuStrikeAgeScaledPolarXYPolicyEncoder(2, "cpu", initialization="fresh-random", allow_cpu_for_tests=True)
        rows = _PolicyRows(candidate, 1, 2, encoder)
        self.assertEqual(rows.raw.data_ptr(), raw[1:].data_ptr())
        self.assertTrue(torch.equal(rows.observations[:,198:200], torch.tensor([[.5,1.],[.5,1.]])))
        self.assertTrue(torch.equal(rows.observations[:,190:192], raw[1:3,190:192]))
        self.assertTrue(torch.equal(raw, original))
        raw[1:3,198:200].fill_(240.)
        rows.refresh()
        self.assertTrue(torch.equal(rows.observations[:,198:200], torch.full((2,2),2.)))
        self.assertTrue(torch.equal(raw[[0,3]], original[[0,3]]))
        self.assertEqual(rows.rewards.data_ptr(), candidate.rewards[1:].data_ptr())
        self.assertEqual(rows.terminals.data_ptr(), candidate.terminals[1:].data_ptr())
        self.assertEqual(rows.action_mask.data_ptr(), candidate.action_mask[1:].data_ptr())

    def test_all_policies_select_before_one_physical_advance(self):
        groups, log = fake_groups()
        physical, observed = [], []

        def step():
            pending = log[len(observed):]
            self.assertEqual([event[0] for event in pending if event[1] == "actions"], ["short", "long"])
            physical.append(len(physical))

        def after(tick):
            self.assertEqual(len(physical), tick + 1)
            observed[:] = log

        grouped_ticks(groups, ticks=16, stream_pointer=99, physical_step=step, after_step=after)
        self.assertEqual(len(physical), 16)
        self.assertEqual([group.trainer.pufferl.global_step for group in groups], [32, 48])
        self.assertFalse(any(group.trainer.pufferl.active for group in groups))
        self.assertEqual(sum(event[:2] == ("short", "finish") for event in log), 4)
        self.assertEqual(sum(event[:2] == ("long", "finish") for event in log), 2)

    def test_each_group_local_step_restarts_at_its_own_horizon(self):
        groups, log = fake_groups()
        grouped_ticks(groups, ticks=16, stream_pointer=99, physical_step=lambda: None, after_step=lambda t: None)
        for name, horizon in (("short", 4), ("long", 8)):
            self.assertEqual([event[2] for event in log if event[:2] == (name, "infer")], list(range(horizon)) * (16 // horizon))

    def test_partition_and_end_boundary_validation(self):
        groups, _ = fake_groups()
        specs = [group.spec for group in groups]
        validate_groups(specs, arenas=5, ticks=16)
        for arenas, ticks in ((6, 16), (5, 15)):
            with self.assertRaises(ValueError):
                validate_groups(specs, arenas, ticks)
        with self.assertRaises(ValueError):
            validate_groups([specs[0], specs[0]], arenas=4, ticks=16)


def cuda_selection_regression(output, checkpoint):
    import torch
    from gpu_native_puffer import NativeExternalGpuPuffer
    from gpu_puffer_env import CudaTensorEnvAdapter
    from test_gpu_native_puffer import _args
    from evaluate_gpu_dummy import evaluation_config, save_frozen_weights

    output.mkdir(parents=True, exist_ok=False)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    class Fixture:
        def __init__(self, count):
            self.observations = torch.zeros((count, 223), device="cuda")
            self.rewards = torch.zeros(count, device="cuda")
            self.terminals = torch.zeros(count, device="cuda")
            self.action_mask = torch.ones((count, 33), dtype=torch.uint8, device="cuda")
            self.row = torch.arange(count, device="cuda") % 8
            self.actions = []
            self.reset()

        def reset(self):
            self.tick = 0
            self.observations.zero_()
            self.observations[:, 0].copy_(self.row.float() / 8)
            self.rewards.zero_()
            self.terminals.fill_(1)
            self.action_mask.fill_(1)
            self.actions.clear()

        def step(self, actions):
            self.actions.append(actions.clone())
            self.tick += 1
            self.observations[:, 0].copy_(actions[:, 0].float() / 32)
            self.observations[:, 1].fill_(self.tick / 32)
            self.rewards.copy_(actions[:, 0].float() / 16 - 1)
            self.terminals.copy_(((self.row + self.tick) % 5 == 0).float())
            self.action_mask.zero_()
            self.action_mask[:, 0] = 1
            self.action_mask.scatter_(1, ((self.row + self.tick) % 32 + 1)[:, None], 1)

        def close(self):
            pass

    specs = [PolicyGroup("stochastic", 8, checkpoint, expected, 8, False),
             PolicyGroup("greedy", 8, checkpoint, expected, 16, True)]

    def args_for(spec):
        args = _args()
        args["reset_state"] = True
        return evaluation_config(args, fighters=16, horizon=spec.horizon, ticks=32,
                                 seed=73, greedy=spec.greedy)

    reference = {}
    hashes = {}
    for spec in specs:
        fixture = Fixture(8)
        trainer = NativeExternalGpuPuffer(args_for(spec), CudaTensorEnvAdapter(fixture, (33,)), reset_environment=False)
        try:
            trainer.load_weights(checkpoint)
            for _ in range(32 // spec.horizon):
                trainer.rollouts()
            reference[spec.name] = torch.stack(fixture.actions).cpu()
            hashes[f"reference_{spec.name}"] = save_frozen_weights(trainer, output / f"reference-{spec.name}.bin", expected)["sha256"]
            assert trainer.global_step == 256
        finally:
            trainer.close()
    shared, groups = Fixture(16), []
    actions = torch.empty((16, 1), dtype=torch.int32, device="cuda")
    try:
        for index, spec in enumerate(specs):
            rows = _PolicyRows(shared, index * 8, 8, None)
            trainer = NativeExternalGpuPuffer(args_for(spec), CudaTensorEnvAdapter(rows, (33,)), reset_environment=False)
            trainer.load_weights(checkpoint)
            groups.append(SimpleNamespace(spec=spec, trainer=trainer, actions=actions[index * 8:index * 8 + 8]))
        grouped_ticks(groups, ticks=32, stream_pointer=int(torch.cuda.current_stream().cuda_stream),
                      physical_step=lambda: shared.step(actions), after_step=lambda tick: None)
        result = torch.stack(shared.actions).cpu()
        for index, group in enumerate(groups):
            assert torch.equal(reference[group.spec.name], result[:, index * 8:index * 8 + 8]), group.spec.name
            assert group.trainer.global_step == 256
            hashes[f"grouped_{group.spec.name}"] = save_frozen_weights(group.trainer, output / f"grouped-{group.spec.name}.bin", expected)["sha256"]
        report = {"schema": "rek.grouped_native_selection_regression.v1", "status": "passed",
                  "ticks": 32, "agents_per_policy": 8, "horizons": [8, 16],
                  "sampling": ["stochastic", "greedy"], "exact_action_comparisons": 512,
                  "interior_terminals": "(row+tick) modulo5==0", "mask_changes": "two legal actions after each step",
                  "checkpoint_sha256": expected, "after_hashes": hashes, "policy_updates": 0,
                  "limits": "Synthetic CUDA inputs; no physics run or combat-performance claim."}
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        for group in reversed(groups):
            group.trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args, remaining = parser.parse_known_args()
    if args.cuda_output is not None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required with --cuda-output")
        print(json.dumps(cuda_selection_regression(args.cuda_output, args.checkpoint), sort_keys=True))
    else:
        unittest.main(argv=[__file__, *remaining])
