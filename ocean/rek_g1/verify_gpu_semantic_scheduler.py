"""Validate GPU scheduling against original host pure functions, no physics."""
from __future__ import annotations

import argparse
import configparser
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gpu_motion_assets import GpuMotionAssets
from gpu_native_motion import (AdvanceResult, Backends, BaseVelocity, Clip, Composer,
    Config, F, I, LocomotionInput, LocomotionResult, LocomotionState, Matcher,
    MatcherSlot, MirrorTable, P, ReferenceOutput, ReferenceTiming, VelocityCommand)
from gpu_native_motion import GpuNativeMotion
from gpu_semantic_scheduler import (ActionTableStorage, Adapter, CommandConfig,
    GpuSemanticScheduler, PufferStep, SchedulerRow)


class Playback(ct.Structure):
    _fields_ = [('magnitude', F), ('scale', F), ('apply', ct.c_uint8)]


class HeadingUpdate(ct.Structure):
    _fields_ = [('clip', F), ('command', F), ('forgiveness', F), ('total', F)]


def verify(asset_root, feature_root, gpu_library, host_library, config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    durations = [config.getint('env', f'move_{i}_duration_ticks') for i in range(17)]
    manifest = hashlib.sha256((asset_root / 'semantic_duel_assets_manifest.json').read_bytes()).hexdigest()
    assets = GpuMotionAssets(asset_root, manifest)
    count = 4
    motion = GpuNativeMotion(assets, feature_root, gpu_library, count)
    gpu = GpuSemanticScheduler(motion, config.getint('env', 'locomotion_segment_ticks'), durations)
    native = ct.CDLL(str(host_library))

    def call(name, args, result=I):
        function = getattr(native, name)
        function.argtypes, function.restype = args, result
        return function

    init = call('sonic_motion_composer_native_init', [P, I, P])
    matcher_init = call('sonic_motion_entry_matcher_native_init', [P, I, P, ct.c_size_t])
    register = call('sonic_motion_entry_matcher_native_register', [P, P, P, ct.c_size_t])
    play = call('sonic_motion_composer_native_play_action', [P, P, P])
    immediate = call('sonic_motion_composer_native_play_action_immediate', [P, P, P])
    speed = call('sonic_motion_composer_native_set_locomotion_speed', [P, F])
    reference = call('sonic_motion_composer_native_build_reference_rows', [P, P, P, P])
    advance = call('sonic_motion_composer_native_advance', [P, P])
    consume = call('sonic_motion_composer_native_consume_heading_delta', [P, P])
    ownership = call('sonic_motion_composer_native_heading_clip_ownership', [P, P])
    sin_cos = call('sonic_motion_composer_libm_candidate_sin_cos_f', [P, F, P, P])
    locomotion = call('rek_g1_native_locomotion_step', [P, P, P, P])
    settled = call('rek_g1_native_transition_settled', [I, P, P, P])
    playback = call('rek_g1_native_playback_update', [VelocityCommand, P, P])
    heading_update = call('rek_g1_native_heading_update', [VelocityCommand, P, F, F, F, P])
    table_init = call('rek_g1_semantic_action_table_init', [P, ct.c_uint32, P])
    adapter_init = call('rek_g1_puffer_init', [P, P])
    mask_function = call('rek_g1_puffer_write_mask', [P, I, I, P, ct.c_size_t])
    adapter_step = call('rek_g1_puffer_step', [P, F, type(gpu.config_value.timing), I, I], PufferStep)
    table = ActionTableStorage()
    duration_array = (ct.c_uint32 * 17)(*durations)
    assert table_init(ct.byref(table), config.getint('env', 'locomotion_segment_ticks'), duration_array) == 0
    rows = (SchedulerRow * count)()
    composers = (Composer * count)()
    matchers = (Matcher * count)()
    slots = [(MatcherSlot * len(assets.clips))() for _ in range(count)]
    clips, features = {}, {}
    feature_manifest = json.loads((feature_root / 'foot_features_manifest.json').read_text())
    for record in feature_manifest['clips']:
        identity = record['npz_path_id']
        clip = assets.clips[identity]
        pos, rot = (assets.host_arrays[clip['files'][key]] for key in ('mujoco_joint_order', 'wxyz'))
        clips[identity] = Clip(pos.ctypes.data, rot.ctypes.data, pos.size, rot.size, clip['frames'], clip['fps'])
        features[identity] = np.fromfile(feature_root / record['file'], dtype='<f4')
    pointers = [ct.cast(getattr(native, name), P).value for name in (
        'sonic_motion_composer_libm_candidate_quaternion_slerp',
        'sonic_motion_composer_libm_candidate_atan2_f',
        'sonic_motion_composer_libm_candidate_sin_cos_f',
        'sonic_motion_entry_matcher_native_callback')]
    route_configs = [Config(**route['config']) for route in assets.routes]
    route_clips = [clips[route['npz_path_id']] for route in assets.routes]
    for row in range(count):
        assert matcher_init(ct.byref(matchers[row]), 50, slots[row], len(clips)) == 0
        for identity, clip in clips.items():
            feature = features[identity]
            assert register(ct.byref(matchers[row]), ct.byref(clip), feature.ctypes.data, feature.size) == 0
        backends = Backends(*pointers, ct.addressof(matchers[row]))
        assert init(ct.byref(composers[row]), 50, ct.byref(backends)) == 0
        assert immediate(ct.byref(composers[row]), ct.byref(route_clips[0]), ct.byref(route_configs[0])) == 0
        assert adapter_init(ct.byref(rows[row].adapter), ct.byref(table.table)) == 0
        rows[row].translation_settled = 1
    indices, negate = motion.mirror_indices.cpu().numpy(), motion.mirror_negate.cpu().numpy()
    mirror = MirrorTable(indices.ctypes.data, negate.ctypes.data, 29, 29)
    timing = ReferenceTiming((I * 10)(*range(0, 50, 5)), (I * 10)(*range(1, 51, 5)))
    positions = np.empty((count, 10, 29), dtype=np.float32)
    next_positions, rotations = np.empty_like(positions), np.empty((count, 10, 4), dtype=np.float32)
    refs = [ReferenceOutput(positions[i].ctypes.data, next_positions[i].ctypes.data,
        rotations[i].ctypes.data, 290, 290, 40) for i in range(count)]
    headings = np.zeros((count, 4), dtype=np.float32)
    headings[:, 0] = 1
    masks = np.empty((count, 33), dtype=np.uint8)
    phase = torch.zeros(count, dtype=torch.int32, device='cuda')
    command_config = CommandConfig(1, 1, 1, 50)
    kinds, move_routes = gpu.route_kinds.cpu().tolist(), gpu.move_routes.cpu().tolist()
    exact_fields = 0
    maximum_error = {'root_rotation': 0.0, 'heading': 0.0}
    moves_seen, desired = set(), [i * 4 for i in range(count)]

    def compare(a, b):
        nonlocal exact_fields
        for name, kind in a._fields_:
            left, right = getattr(a, name), getattr(b, name)
            if kind is P:
                continue
            if issubclass(kind, ct.Structure):
                compare(left, right)
            elif kind is F and name in ('prev_heading', 'last_heading_delta', 'pending_heading_delta'):
                assert abs(left - right) <= 1e-5, (name, left, right)
            else:
                assert (bytes(F(left)) == bytes(F(right))) if kind is F else left == right, (name, left, right)
                exact_fields += 1

    def update_mask(i):
        state = rows[i]
        assert mask_function(ct.byref(state.adapter), state.translation_settled,
            state.action_busy or state.recovery_active, masks[i].ctypes.data, 33) == 0

    def gate(state, sample):
        last = abs(state.last_driven_command.forward) >= np.float32(0.001) or abs(state.last_driven_command.strafe) >= np.float32(0.001)
        brake = abs(state.stop_brake_command.forward) >= np.float32(0.001) or abs(state.stop_brake_command.strafe) >= np.float32(0.001)
        if (state.locomotion_active and (kinds[state.current_route_id] == 1 or last)) \
                or (state.transition_settling and (kinds[state.transition_from_route_id] == 1 or last)) \
                or (state.stop_braking and (kinds[state.current_route_id] == 1 or brake)):
            return 0
        if not state.has_momentum or kinds[state.momentum_route_id] == 2:
            return 1
        value = ct.c_uint8()
        assert settled(state.momentum_route_id, ct.byref(gpu.config_value.locomotion), ct.byref(sample), ct.byref(value)) == 0
        return value.value

    for i in range(count):
        update_mask(i)
    for tick in range(1200):
        assert np.array_equal(masks, gpu.action_mask.cpu().numpy())
        actions = []
        for i in range(count):
            if tick < 25:
                action = 6 + i % 2
            elif tick == 25:
                action = 16 + i
            elif rows[i].adapter.scheduler.active:
                action = (0, 6, 7, 1)[(tick // 11 + i) % 4]
            elif tick % 180 < 24:
                action = (2, 3, 4, 5, 8, 9, 14, 15)[(tick // 3 + i) % 8]
            else:
                action = 16 + desired[i] % 17
                if not masks[i, action]:
                    action = 1
                else:
                    desired[i] += 1
            assert masks[i, action], (tick, i, action, masks[i])
            actions.append(action)
        velocities = np.zeros((count, 6), dtype=np.float32)
        if tick % 17 < 3:
            velocities[:, 3:5] = np.float32(0.031)
        velocity_gpu = torch.as_tensor(velocities, device='cuda')
        gpu.pre_step(torch.tensor(actions, device='cuda', dtype=torch.float32), velocity_gpu)
        gpu.check_status()
        for i, action in enumerate(actions):
            row, composer = rows[i], composers[i]
            step = adapter_step(ct.byref(row.adapter), action, gpu.config_value.timing,
                row.translation_settled, row.action_busy or row.recovery_active)
            assert step.status == 0
            row.semantic = step.semantic
            command = VelocityCommand(step.semantic.input.forward, step.semantic.input.strafe, step.semantic.input.yaw)
            sample = BaseVelocity(VelocityCommand(*velocities[i, :3]), VelocityCommand(*velocities[i, 3:]), 1)
            if step.semantic.move_start_edge:
                move = table.move_indices[step.semantic.move_registry_index]
                moves_seen.add(move)
                route = move_routes[move]
                assert play(ct.byref(composer), ct.byref(route_clips[route]), ct.byref(route_configs[route])) == 0
                row.locomotion.locomotion_active = 0
                row.active_route_id, row.effective_velocity = route, command
                assert step.semantic.input.yaw == 0
            else:
                locomotion_input = LocomotionInput(command, sample, 0.02, 1,
                    bool(composer.action_playing), bool(composer.current_layer.active and not composer.current_layer.config.loop), 1)
                result = LocomotionResult()
                assert locomotion(ct.byref(row.locomotion), ct.byref(gpu.config_value.locomotion), ct.byref(locomotion_input), ct.byref(result)) == 0
                if result.event:
                    route = result.event_route_id
                    assert play(ct.byref(composer), ct.byref(route_clips[route]), ct.byref(route_configs[route])) == 0
                    row.active_route_id = route
                row.locomotion, row.effective_velocity = result.next_state, result.effective_velocity
            update = Playback()
            assert playback(row.effective_velocity, ct.byref(command_config), ct.byref(update)) == 0
            if update.apply:
                assert speed(ct.byref(composer), update.scale) == 0
            assert reference(ct.byref(composer), ct.byref(timing), ct.byref(mirror), ct.byref(refs[i])) == 0
        for host, device in zip(rows, gpu.rows.host()):
            compare(host, device)
        assert np.array_equal(positions.view(np.uint32), motion.positions.cpu().numpy().view(np.uint32))
        assert np.array_equal(next_positions.view(np.uint32), motion.next_positions.cpu().numpy().view(np.uint32))
        error = float(np.max(np.abs(rotations - motion.rotations.cpu().numpy())))
        maximum_error['root_rotation'] = max(maximum_error['root_rotation'], error)
        assert error <= 1e-5
        gpu.post_step(velocity_gpu, phase)
        gpu.check_status()
        for i in range(count):
            row, composer = rows[i], composers[i]
            result = AdvanceResult()
            assert advance(ct.byref(composer), ct.byref(result)) == 0
            delta, own = F(), F()
            assert consume(ct.byref(composer), ct.byref(delta)) == 0
            assert ownership(ct.byref(composer), ct.byref(own)) == 0
            update = HeadingUpdate()
            assert heading_update(row.effective_velocity, ct.byref(command_config), own, delta, 0, ct.byref(update)) == 0
            sine, cosine = F(), F()
            assert sin_cos(None, F(np.float32(update.total) * np.float32(0.5)), ct.byref(sine), ct.byref(cosine))
            s, c = np.float32(sine.value), np.float32(cosine.value)
            w, x, y, z = headings[i]
            headings[i] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
            sample = BaseVelocity(VelocityCommand(*velocities[i, :3]), VelocityCommand(*velocities[i, 3:]), 1)
            row.translation_settled = gate(row.locomotion, sample)
            row.action_busy = bool(composer.current_layer.active and not composer.current_layer.config.loop)
            update_mask(i)
        for host, device in zip(rows, gpu.rows.host()):
            compare(host, device)
        for host, device in zip(composers, motion.composers.host()):
            compare(host, device)
        error = float(np.max(np.abs(headings - gpu.heading.cpu().numpy())))
        maximum_error['heading'] = max(maximum_error['heading'], error)
        assert error <= 1e-5, error
    assert moves_seen == set(range(17)), moves_seen
    semantic12 = gpu.observation12.cpu().numpy()
    assert np.max(np.abs(semantic12[:, :4] - headings)) <= 1e-5
    for i, row in enumerate(rows):
        expected = [row.effective_velocity.forward, row.effective_velocity.strafe,
            row.effective_velocity.yaw, row.active_route_id, row.locomotion.locomotion_active,
            row.locomotion.transition_settling, bool(composers[i].action_playing), row.action_busy]
        assert np.array_equal(semantic12[i, 4:], np.asarray(expected, dtype=np.float32))

    reset = torch.ones(count, dtype=torch.uint8, device='cuda')
    zero_flags = torch.zeros_like(reset)
    initial_heading = torch.zeros((count, 4), device='cuda')
    initial_heading[:, 0] = 1
    velocity_gpu.zero_()
    gpu.reset_rows(reset, initial_heading)
    gpu.check_status()
    # Suspended rows still dispatch categorical input, while their composer
    # and effective motion remain fixed, as in the original native runtime.
    suspended = torch.tensor([1, 0, 1, 0], device='cuda', dtype=torch.uint8)
    before = motion.composers.tensor.clone()
    actions = torch.full((count,), 6.0, device='cuda')
    gpu.pre_step(actions, velocity_gpu, suspended)
    gpu.post_step(velocity_gpu, phase, suspended)
    gpu.check_status()
    assert torch.equal(before[::2], motion.composers.tensor[::2])
    assert gpu.rows.field('adapter.scheduler.input_state.held').cpu().tolist() == [16] * count
    assert gpu.rows.field('recovery_active').cpu().tolist() == [1, 0, 1, 0]
    # Row-local completed spawn reset clears desired input and restores idle.
    gpu.reset_rows(suspended, initial_heading)
    gpu.post_step(velocity_gpu, phase, zero_flags, suspended, suspended)
    gpu.check_status()
    assert gpu.rows.field('adapter.scheduler.input_state.held').cpu().tolist() == [0, 16, 0, 16]
    # A full pre/post tick can be captured and replayed without host data reads.
    gpu.reset_rows(reset, initial_heading)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gpu.pre_step(actions, velocity_gpu)
        gpu.post_step(velocity_gpu, phase)
    for _ in range(64):
        graph.replay()
    gpu.check_status()
    assert gpu.rows.field('adapter.scheduler.input_state.yaw_ramp').cpu().tolist() == [1] * count
    assert gpu.active_route_ids.cpu().tolist() == [5] * count
    # Rejected categories latch the actual original adapter status and expose
    # no action mask. There is no substitute action or silent retry.
    gpu.reset_rows(reset, initial_heading)
    bad_actions = torch.tensor([-1, float('nan'), 1.5, 1000], device='cuda')
    gpu.pre_step(bad_actions, velocity_gpu)
    assert gpu.statuses.cpu().tolist() == [108, 106, 107, 108]
    assert not gpu.action_mask.any().item()
    gpu.reset_rows(reset, initial_heading)
    velocity_gpu[0, 0] = float('nan')
    gpu.pre_step(actions, velocity_gpu)
    assert gpu.statuses.cpu().tolist() == [202, 0, 0, 0]
    return {'status': 'ok', 'ticks': 1200, 'fighters': count, 'moves': sorted(moves_seen),
        'exact_named_fields': exact_fields, 'exact_reference_values': 1200 * count * 580,
        'reset_suspension_error_checks': 'passed', 'cuda_graph_replays': 64,
        'max_cuda_libm_error': maximum_error, 'physics_steps': 0,
        'device': torch.cuda.get_device_name()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('assets', 'features', 'gpu_library', 'host_library', 'config'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.assets, args.features, args.gpu_library, args.host_library, args.config), sort_keys=True))
