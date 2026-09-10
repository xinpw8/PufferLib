"""Compare native host and CUDA motion on the same authored clip arrays."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gpu_motion_assets import GpuMotionAssets
from gpu_native_motion import (AdvanceResult, Backends, Clip, Composer, ComposerCommand, Config,
    DeviceStructs, F, GpuNativeMotion, I, InputDecision, InputFrame, InputTiming,
    HeldState, LocomotionConfig, LocomotionInput, LocomotionResult, LocomotionState,
    Matcher, MatcherSlot, MirrorTable, P, ReferenceOutput, ReferenceTiming, upload,
    BaseVelocity, VelocityCommand)


def heading_wxyz(quaternion):
    w, x, y, z = quaternion
    return float(np.arctan2(
        np.float32(2.0) * (x * y + z * w),
        np.float32(1.0) - np.float32(2.0) * (y * y + z * z),
    ))


def bind(library, name, args):
    function = getattr(library, name)
    function.argtypes = args
    function.restype = I
    return function


def verify_inputs(gpu, native):
    count, ticks = gpu.count, 256
    held = DeviceStructs(HeldState, count, gpu.device, zero=True)
    frames = DeviceStructs(InputFrame, count, gpu.device)
    decisions = DeviceStructs(InputDecision, count, gpu.device)
    states = DeviceStructs(LocomotionState, count, gpu.device, zero=True)
    inputs = DeviceStructs(LocomotionInput, count, gpu.device)
    results = DeviceStructs(LocomotionResult, count, gpu.device, zero=True)
    held_host = (HeldState * count)()
    states_host = (LocomotionState * count)()
    timing = InputTiming(0.02, 0.5)
    config = LocomotionConfig(0.03, 0.03, 2.0, 1)
    timing_gpu, config_gpu = upload(timing, gpu.device), upload(config, gpu.device)
    held_function = bind(native, 'rek_g1_test_host_held_input', [P, P, P, I, I, P])
    locomotion_function = bind(native, 'rek_g1_native_locomotion_step', [P, P, P, P])
    comparisons = 0

    def exact(left, right):
        nonlocal comparisons
        for name, kind in left._fields_:
            a, b = getattr(left, name), getattr(right, name)
            if issubclass(kind, ct.Structure):
                exact(a, b)
            else:
                if kind is F:
                    assert bytes(F(a)) == bytes(F(b)), (name, a, b)
                else:
                    assert a == b, (name, a, b)
                comparisons += 1

    masks = [0, 1, 2, 4, 8, 16, 32, 17, 33, 18, 34, 20, 36, 24, 40,
        5, 9, 6, 10, 3, 12, 48, 64]
    for tick in range(ticks):
        frame_values = (InputFrame * count)(*[InputFrame(
            masks[(tick // 4 + row * 3) % len(masks)], tick % 5 == 0)
            for row in range(count)])
        settled = torch.tensor([(tick + row) % 7 != 0 for row in range(count)],
            dtype=torch.int32, device=gpu.device)
        busy = torch.tensor([(tick + row) % 9 == 0 for row in range(count)],
            dtype=torch.int32, device=gpu.device)
        frames.tensor.copy_(upload(frame_values, gpu.device).reshape_as(frames.tensor))
        gpu._call('rek_g1_cuda_held_input', held.tensor, frames.tensor, timing_gpu,
            settled, busy, decisions.tensor)
        decision_values = decisions.host()
        held_values = held.host()
        input_values = (LocomotionInput * count)()
        expected_results = (LocomotionResult * count)()
        statuses = []
        for row in range(count):
            expected = InputDecision()
            held_function(ct.byref(held_host[row]), ct.byref(frame_values[row]),
                ct.byref(timing), (tick + row) % 7 != 0, (tick + row) % 9 == 0,
                ct.byref(expected))
            exact(expected, decision_values[row])
            exact(held_host[row], held_values[row])
            sample_speed = (0.0, 0.029, 0.03, 0.031, 0.25)[(tick + row) % 5]
            input_values[row] = LocomotionInput(
                VelocityCommand(expected.forward, expected.strafe, expected.yaw),
                BaseVelocity(VelocityCommand(0, 0, sample_speed),
                    VelocityCommand(sample_speed, sample_speed, 0), (tick + row) % 31 != 0),
                0.02, 1, (tick + row) % 17 == 0, (tick + row) % 13 == 0,
                (tick + row) % 19 != 0)
            statuses.append(locomotion_function(ct.byref(states_host[row]),
                ct.byref(config), ct.byref(input_values[row]), ct.byref(expected_results[row])))
        inputs.tensor.copy_(upload(input_values, gpu.device).reshape_as(inputs.tensor))
        results.tensor.zero_()
        gpu._call('rek_g1_cuda_locomotion_step', states.tensor, config_gpu, inputs.tensor,
            results.tensor, gpu.statuses)
        assert statuses == gpu.statuses.cpu().tolist()
        actual_results = results.host()
        for row in range(count):
            exact(expected_results[row], actual_results[row])
            if statuses[row] == 0:
                states_host[row] = expected_results[row].next_state
        # Accepted native states become the next source states; validation-only
        # transfers here are deliberately outside the environment hot path.
        states.tensor.copy_(upload(states_host, gpu.device).reshape_as(states.tensor))
    return {'ticks': ticks, 'fighter_steps': ticks * count, 'exact_fields': comparisons}


def verify(asset_root: Path, feature_root: Path, gpu_library: Path, host_library: Path):
    digest = hashlib.sha256((asset_root / 'semantic_duel_assets_manifest.json').read_bytes()).hexdigest()
    assets = GpuMotionAssets(asset_root, digest)
    raw_nonzero_headings = 0
    max_frame_zero_heading = 0.0
    for clip in assets.clips.values():
        wxyz_name = clip['files']['wxyz']
        xyzw_name = clip['files']['xyzw']
        raw = np.fromfile(asset_root / wxyz_name, dtype='<f4').reshape(-1, 4)
        normalized = assets.host_arrays[wxyz_name]
        if abs(heading_wxyz(raw[0])) >= 1.0e-6:
            raw_nonzero_headings += 1
        max_frame_zero_heading = max(
            max_frame_zero_heading, abs(heading_wxyz(normalized[0])))
        assert np.array_equal(
            assets.host_arrays[xyzw_name].view(np.uint32),
            normalized[:, [1, 2, 3, 0]].copy().view(np.uint32),
        )
    assert raw_nonzero_headings > 0
    assert max_frame_zero_heading <= 2.0e-6, max_frame_zero_heading
    count = 4
    gpu = GpuNativeMotion(assets, feature_root, gpu_library, count)
    native = ct.CDLL(str(host_library))
    host_init = bind(native, 'sonic_motion_composer_native_init', [P, I, P])
    matcher_init = bind(native, 'sonic_motion_entry_matcher_native_init', [P, I, P, ct.c_size_t])
    register = bind(native, 'sonic_motion_entry_matcher_native_register', [P, P, P, ct.c_size_t])
    play = bind(native, 'sonic_motion_composer_native_play_action', [P, P, P])
    advance = bind(native, 'sonic_motion_composer_native_advance', [P, P])
    reference = bind(native, 'sonic_motion_composer_native_build_reference_rows', [P, P, P, P])
    heading = bind(native, 'sonic_motion_composer_native_consume_heading_delta', [P, P])
    ownership = bind(native, 'sonic_motion_composer_native_heading_clip_ownership', [P, P])
    cancel = bind(native, 'sonic_motion_composer_native_cancel_action', [P])
    speed = bind(native, 'sonic_motion_composer_native_set_locomotion_speed', [P, F])
    pointers = [ct.cast(getattr(native, name), P).value for name in (
        'sonic_motion_composer_libm_candidate_quaternion_slerp',
        'sonic_motion_composer_libm_candidate_atan2_f',
        'sonic_motion_composer_libm_candidate_sin_cos_f',
        'sonic_motion_entry_matcher_native_callback')]
    clips = {}
    features = {}
    feature_manifest = json.loads((feature_root / 'foot_features_manifest.json').read_text())
    for record in feature_manifest['clips']:
        clip_id = record['npz_path_id']
        clip = assets.clips[clip_id]
        positions = assets.host_arrays[clip['files']['mujoco_joint_order']]
        rotations = assets.host_arrays[clip['files']['wxyz']]
        clips[clip_id] = Clip(positions.ctypes.data, rotations.ctypes.data,
            positions.size, rotations.size, clip['frames'], clip['fps'])
        features[clip_id] = np.fromfile(feature_root / record['file'], dtype='<f4')
    host_composers = (Composer * count)()
    host_matchers = (Matcher * count)()
    host_slots = [(MatcherSlot * len(clips))() for _ in range(count)]
    for i in range(count):
        assert matcher_init(ct.byref(host_matchers[i]), 50, host_slots[i], len(clips)) == 0
        for clip_id, clip in clips.items():
            values = features[clip_id]
            assert register(ct.byref(host_matchers[i]), ct.byref(clip), values.ctypes.data, values.size) == 0
        backends = Backends(*pointers, ct.addressof(host_matchers[i]))
        assert host_init(ct.byref(host_composers[i]), 50, ct.byref(backends)) == 0
    indices = gpu.mirror_indices.cpu().numpy()
    negate = gpu.mirror_negate.cpu().numpy()
    mirror = MirrorTable(indices.ctypes.data, negate.ctypes.data, 29, 29)
    timing = ReferenceTiming((I * 10)(*range(0, 50, 5)), (I * 10)(*range(1, 51, 5)))
    host_positions = np.empty((count, 10, 29), dtype=np.float32)
    host_next_positions = np.empty_like(host_positions)
    host_rotations = np.empty((count, 10, 4), dtype=np.float32)
    outputs = [ReferenceOutput(host_positions[i].ctypes.data, host_next_positions[i].ctypes.data,
        host_rotations[i].ctypes.data, 290, 290, 40) for i in range(count)]
    max_error = {'root_quaternion': 0.0, 'heading': 0.0}
    exact_comparisons = 0

    def compare(a, b, prefix=''):
        nonlocal exact_comparisons
        for name, kind in a._fields_:
            x, y = getattr(a, name), getattr(b, name)
            path = f'{prefix}.{name}'
            if kind is P:
                continue
            if issubclass(kind, ct.Structure):
                compare(x, y, path)
            elif kind is F and name in ('prev_heading', 'last_heading_delta', 'pending_heading_delta'):
                error = abs(x - y)
                max_error['heading'] = max(max_error['heading'], error)
                assert error <= 1e-5, (path, x, y)
            else:
                assert x == y, (path, x, y)
                exact_comparisons += 1

    for tick in range(360):
        if tick % 15 == 0:
            route_ids = [(tick // 15 + i) % 24 for i in range(count)]
            gpu.command(torch.tensor(route_ids, device='cuda', dtype=torch.int64))
            assert torch.all(gpu.statuses == 0).item(), gpu.statuses.cpu().tolist()
            for i, route_id in enumerate(route_ids):
                route = assets.routes[route_id]
                config = Config(**route['config'])
                assert play(ct.byref(host_composers[i]), ct.byref(clips[route['npz_path_id']]), ct.byref(config)) == 0
            for a, b in zip(host_matchers, gpu.matchers.host()):
                compare(a.diagnostics, b.diagnostics)
        elif tick % 15 == 7:
            gpu.command(torch.zeros(count, dtype=torch.int64, device='cuda'),
                torch.full((count,), 4, dtype=torch.int32, device='cuda'),
                torch.full((count,), 0.75, device='cuda'))
            assert torch.all(gpu.statuses == 0).item()
            for state in host_composers:
                assert speed(ct.byref(state), F(0.75)) == 0
        elif tick % 15 == 13:
            gpu.command(torch.zeros(count, dtype=torch.int64, device='cuda'),
                torch.full((count,), 3, dtype=torch.int32, device='cuda'))
            assert torch.all(gpu.statuses == 0).item()
            for state in host_composers:
                assert cancel(ct.byref(state)) == 0
        gpu.advance()
        assert torch.all(gpu.statuses == 0).item()
        observed_advances = gpu.advances.host()
        for i in range(count):
            result = AdvanceResult()
            assert advance(ct.byref(host_composers[i]), ct.byref(result)) == 0
            compare(result, observed_advances[i])
        for a, b in zip(host_composers, gpu.composers.host()):
            compare(a, b)
        gpu.reference()
        assert torch.all(gpu.statuses == 0).item()
        for i in range(count):
            assert reference(ct.byref(host_composers[i]), ct.byref(timing), ct.byref(mirror), ct.byref(outputs[i])) == 0
        assert np.array_equal(host_positions.view(np.uint32), gpu.positions.cpu().numpy().view(np.uint32))
        assert np.array_equal(host_next_positions.view(np.uint32), gpu.next_positions.cpu().numpy().view(np.uint32))
        error = float(np.max(np.abs(host_rotations - gpu.rotations.cpu().numpy())))
        max_error['root_quaternion'] = max(max_error['root_quaternion'], error)
        assert error <= 1e-5, error
        exact_comparisons += host_positions.size + host_next_positions.size
        gpu.heading()
        assert torch.all(gpu.statuses == 0).item()
        observed_heading = gpu.heading_delta.cpu().numpy()
        observed_ownership = gpu.heading_ownership.cpu().numpy()
        for i in range(count):
            delta, own = F(), F()
            assert heading(ct.byref(host_composers[i]), ct.byref(delta)) == 0
            assert ownership(ct.byref(host_composers[i]), ct.byref(own)) == 0
            error = abs(delta.value - float(observed_heading[i]))
            max_error['heading'] = max(max_error['heading'], error)
            assert error <= 1e-5
            assert own.value == observed_ownership[i]

    input_comparisons = verify_inputs(gpu, native)
    return {'status': 'ok', 'device': torch.cuda.get_device_name(), 'fighters': count,
        'normalized_motion_assets': {'clips': len(assets.clips),
            'raw_nonzero_frame_zero_headings': raw_nonzero_headings,
            'max_frame_zero_heading_radians': max_frame_zero_heading},
        'held_input_and_locomotion': input_comparisons,
        'ticks': 360, 'authored_routes': 24, 'exact_comparisons': exact_comparisons,
        'max_absolute_cuda_vs_host_error': max_error,
        'composer_bytes': ct.sizeof(Composer), 'matcher_bytes': ct.sizeof(Matcher),
        'slot_bytes': ct.sizeof(MatcherSlot), 'command_bytes': ct.sizeof(ComposerCommand),
        'feature_manifest_sha256': hashlib.sha256((feature_root / 'foot_features_manifest.json').read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assets', type=Path)
    parser.add_argument('features', type=Path)
    parser.add_argument('gpu_library', type=Path)
    parser.add_argument('host_library', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.assets, args.features, args.gpu_library, args.host_library), sort_keys=True))
