"""Paired CUDA-world regression for selected-arena physical resets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gpu_duel_physics import GpuDuelPhysics
from gpu_duel_reset import GpuDuelReset
from gpu_motion_assets import GpuMotionAssets


def verify(asset_root):
    manifest_path = asset_root / 'semantic_duel_assets_manifest.json'
    assets = GpuMotionAssets(asset_root, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    manifest = json.loads(manifest_path.read_text())
    model_file = 'model.two_fighter_arena.xml'
    model_hash = manifest['files'][model_file]['sha256']
    control = GpuDuelPhysics(asset_root / model_file, model_hash, arenas=4)
    subject = GpuDuelPhysics(asset_root / model_file, model_hash, arenas=4)
    repeat = GpuDuelPhysics(asset_root / model_file, model_hash, arenas=4)
    control_reset, subject_reset = GpuDuelReset(control, assets), GpuDuelReset(subject, assets)
    repeat_reset = GpuDuelReset(repeat, assets)
    all_worlds = torch.ones(4, device='cuda', dtype=torch.bool)
    mask = torch.tensor([False, True, False, True], device='cuda')
    unaffected = ~mask
    results = {}

    def identical(name, left, right):
        if not torch.equal(left, right):
            error = float((left - right).abs().max().item())
            raise AssertionError(f'{name}: paired CUDA state differed by {error}')

    for operation in ('full', 'begin', 'complete'):
        control_reset.full(all_worlds, reset_clock=True)
        subject_reset.full(all_worlds, reset_clock=True)
        repeat_reset.full(all_worlds, reset_clock=True)
        for physics in (control, subject, repeat):
            physics.qvel[:, 0] = torch.arange(4, device='cuda') * 0.03
            physics.ctrl.copy_(physics.qpos[:, torch.as_tensor(
                list(physics.joint_qpos[0]) + list(physics.joint_qpos[1]), device='cuda')])
        for _ in range(3):
            control.step()
            subject.step()
            repeat.step()
        # Begin the intervention from exactly paired future state and readers.
        for name in ('qpos', 'qvel', 'ctrl', 'time'):
            getattr(subject, name).copy_(getattr(control, name))
            getattr(repeat, name).copy_(getattr(control, name))
        for name, values in subject.reset_reader_fields.items():
            values.copy_(control.reset_reader_fields[name])
            repeat.reset_reader_fields[name].copy_(control.reset_reader_fields[name])
        identical(operation + ': initial qpos', control.qpos, subject.qpos)
        identical(operation + ': initial qvel', control.qvel, subject.qvel)
        before_clock = subject.time.clone()
        before_free_velocity = subject.qvel.reshape(4, 2, 35)[:, :, :6].clone()
        getattr(subject_reset, operation)(mask)
        identical(operation + ': clock', before_clock, subject.time)
        if operation != 'full':
            identical(operation + ': free root velocities', before_free_velocity,
                subject.qvel.reshape(4, 2, 35)[:, :, :6])
        for name, values in subject.reset_reader_fields.items():
            identical(operation + ': reader ' + name,
                control.reset_reader_fields[name][unaffected], values[unaffected])
        errors = {}
        for step in range(1, 11):
            control.step()
            subject.step()
            repeat.step()
            if step in (1, 10):
                errors[step] = {
                    name: {
                        'reset_linf': float((getattr(control, name)[unaffected]
                            - getattr(subject, name)[unaffected]).abs().max().item()),
                        'repeat_linf': float((getattr(control, name)[unaffected]
                            - getattr(repeat, name)[unaffected]).abs().max().item()),
                    } for name in ('qpos', 'qvel')
                }
        results[operation] = {'unaffected_worlds': 2, 'exact_reader_restoration': True,
            'future_state_measurements': errors}

    # Both masks work inside a CUDA graph. Native Warp forward allocates
    # scratch in graph nodes, so the current backend cannot conditionally skip
    # this call. Mask evaluation and state restoration remain device-side.
    dynamic_mask = torch.zeros(4, device='cuda', dtype=torch.bool)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    subject.stream = subject.wp.stream_from_torch(capture_stream)
    with torch.cuda.graph(graph, stream=capture_stream):
        with subject.wp.ScopedCapture(stream=subject.stream, external=True) as warp_capture:
            subject.forward_selected(dynamic_mask)
    before = {name: values.clone() for name, values in subject.reset_reader_fields.items()}
    for _ in range(10):
        graph.replay()
    for name, values in subject.reset_reader_fields.items():
        identical('zero-mask: ' + name, before[name], values)
    dynamic_mask[1] = True
    subject.qpos[1, 0] += 0.2
    graph.replay()
    assert not torch.equal(before['xpos'][1], subject.reset_reader_fields['xpos'][1])
    untouched = torch.tensor([0, 2, 3], device='cuda')
    for name, values in subject.reset_reader_fields.items():
        identical('dynamic-mask: ' + name, before[name][untouched], values[untouched])
    results['cuda_graph'] = {'zero_mask_replays': 10, 'dynamic_nonzero_mask': 'passed',
        'conditional_forward_skip': False}
    # Backend assumptions are enforced before any refresh can change state.
    subject.model.opt.run_collision_detection = False
    try:
        subject.forward_selected(dynamic_mask)
        raise AssertionError('disabled contact rebuilding was accepted')
    except ValueError:
        pass
    finally:
        subject.model.opt.run_collision_detection = True
    subject.model.callback.control = lambda *_: None
    try:
        subject.forward_selected(dynamic_mask)
        raise AssertionError('custom callback was accepted')
    except ValueError:
        pass
    finally:
        subject.model.callback.control = None
    return {'status': 'ok', 'device': torch.cuda.get_device_name(), 'arenas_per_instance': 4,
        'cpu_physics_steps': 0, 'exact_future_trajectory_claim': False, 'tests': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assets', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.assets), sort_keys=True))
