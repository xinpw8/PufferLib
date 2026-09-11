"""Compare the combined CUDA reset boundary with the two-forward baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gpu_duel_physics import GpuDuelPhysics
from gpu_duel_reset import GpuDuelReset
from gpu_motion_assets import GpuMotionAssets


def verify(root, arenas=4):
    if arenas < 4 or arenas % 4:
        raise ValueError('arena count must be a positive multiple of four')
    manifest_path = root / 'semantic_duel_assets_manifest.json'
    assets = GpuMotionAssets(root, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    manifest = json.loads(manifest_path.read_text())
    model = 'model.two_fighter_arena.xml'
    systems = [GpuDuelPhysics(root / model, manifest['files'][model]['sha256'], arenas=arenas)
        for _ in range(3)]
    resets = [GpuDuelReset(physics, assets) for physics in systems]
    masks = [(torch.zeros(arenas, device='cuda', dtype=torch.bool),
        torch.zeros(arenas, device='cuda', dtype=torch.bool)) for _ in systems]
    all_worlds = torch.ones(arenas, device='cuda', dtype=torch.bool)
    reports = {}

    def state(physics):
        return dict(qpos=physics.qpos, qvel=physics.qvel, ctrl=physics.ctrl,
            time=physics.time, **physics.reset_reader_fields)

    def pair_states():
        resets[0].full(all_worlds, reset_clock=True)
        systems[0].qvel[:, 0] = torch.arange(arenas, device='cuda') * 0.03
        systems[0].ctrl.copy_(systems[0].qpos[:, resets[0].qindices].reshape(arenas, 58))
        for _ in range(3):
            systems[0].step()
        for destination in systems[1:]:
            for name, values in state(destination).items():
                values.copy_(state(systems[0])[name])
        for reset in resets:
            reset.pending.zero_()

    def snapshot():
        torch.cuda.synchronize()
        actual, repeated = {}, {}
        for name, values in state(systems[0]).items():
            actual[name] = float((values - state(systems[1])[name]).abs().max().item())
            repeated[name] = float((values - state(systems[2])[name]).abs().max().item())
        for name in ('qpos', 'qvel', 'ctrl', 'time'):
            if actual[name] != 0:
                raise AssertionError(f'reset mutation {name} differs: {actual[name]}')
        if actual['qacc_warmstart'] != 0:
            raise AssertionError('combined reset changed the solver warm start')
        # Parallel solver and body reductions can vary by rounding even
        # between unchanged baseline runs. Keep their measured differences in
        # the report, separately from the strict persistent-state assertions.
        for name, values in systems[0].reset_reader_fields.items():
            rounding = torch.finfo(values.dtype).eps * max(1.0,
                float(values.abs().max().item()))
            if actual[name] > repeated[name] + rounding:
                raise AssertionError(f'combined reset reader {name} exceeds rounding: '
                    f'combined={actual[name]}, repeat={repeated[name]}')
        if not torch.equal(resets[0].pending, resets[1].pending):
            raise AssertionError('pending reset flags differ')
        return {'combined_linf': actual, 'baseline_repeat_linf': repeated}

    pair_states()
    # Warm and capture each path with zero masks. Keep Warp scratch alive for
    # the full lifetime of each Torch graph.
    graphs, warp_captures, streams = [], [], []
    for index, (physics, reset, (begin, complete)) in enumerate(zip(systems, resets, masks)):
        operation = (lambda r=reset, b=begin, c=complete: r.after_substep(b, c)) if index == 1 else (
            lambda r=reset, b=begin, c=complete: (r.begin(b), r.complete(c)))
        operation()
        stream = torch.cuda.Stream()
        physics.stream = physics.wp.stream_from_torch(stream)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            with physics.wp.ScopedCapture(stream=physics.stream, external=True) as capture:
                operation()
        graphs.append(graph)
        warp_captures.append(capture)
        streams.append(stream)
    for name, begin_list, complete_list in (
        ('zero', [False] * 4, [False] * 4),
        ('begin', [True, False, True, False], [False] * 4),
        ('complete', [False] * 4, [False, True, False, True]),
        ('mixed', [True, False, False, False], [False, True, False, False]),
    ):
        # Setup uses the default stream, shared explicitly by Torch and Warp.
        for physics in systems:
            physics.stream = physics.wp.stream_from_torch(torch.cuda.current_stream())
        pair_states()
        before_velocity = systems[0].qvel.reshape(arenas, 2, 35)[..., :6].clone()
        for begin, complete in masks:
            begin.copy_(torch.tensor(begin_list * (arenas // 4), device='cuda'))
            complete.copy_(torch.tensor(complete_list * (arenas // 4), device='cuda'))
        for reset, (_, complete) in zip(resets, masks):
            reset.pending.copy_(complete)
        for graph in graphs:
            graph.replay()
        report = {'immediate': snapshot()}
        if not torch.equal(before_velocity, systems[1].qvel.reshape(arenas, 2, 35)[..., :6]):
            raise AssertionError('combined reset changed free-root velocity')
        for step in range(1, 11):
            for physics in systems:
                physics.step()
            if step in (1, 10):
                torch.cuda.synchronize()
                report[str(step)] = {
                    field: {
                        'combined_linf': float((getattr(systems[0], field)
                            - getattr(systems[1], field)).abs().max().item()),
                        'baseline_repeat_linf': float((getattr(systems[0], field)
                            - getattr(systems[2], field)).abs().max().item()),
                    } for field in ('qpos', 'qvel')
                }
        reports[name] = report
    # Time reset-only graph replays with zero masks, matching the common path.
    for begin, complete in masks:
        begin.zero_()
        complete.zero_()
    times = []
    for graph in graphs[:2]:
        for _ in range(5):
            graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 100)
    return {'status': 'ok', 'device': torch.cuda.get_device_name(), 'arenas': arenas,
        'cpu_physics_steps': 0, 'tests': reports,
        'exact_future_trajectory_claim': False,
        'zero_mask_reset_pair_ms': {'baseline': times[0], 'combined': times[1]},
        'forward_calls_per_boundary': {'baseline': 2, 'combined': 1}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assets', type=Path)
    parser.add_argument('--arenas', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(verify(args.assets, args.arenas), sort_keys=True))
