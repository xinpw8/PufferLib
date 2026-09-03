"""Convert the Windows private-AI recorder JSONL into the common trace format.

The raw recorder samples the server-authoritative visual state once per client
Unity FixedUpdate. That is a measured client tick domain. The recovered fight
and bone packet layouts expose no server tick, so this importer never labels the
outer trace ticks as server ticks and never synthesizes one.

Only numeric values present in every sample become channels. A field that is
missing or null in any sample is absent from the trace instead of being filled
with a default. The raw JSONL remains the primary evidence for omitted fields.
"""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import controlled_schedule
import raw_bone_validate
from trace import Trace, TraceWriter


SCHEMA_V1 = 'rek.private_ai.client_fixed.v1'
SCHEMA_V3 = 'rek.private_ai.client_fixed.v3'
SCHEMA_V5 = 'rek.private_ai.protocol.v5'
SCHEMA = SCHEMA_V3
SUPPORTED_SCHEMAS = {SCHEMA_V1, SCHEMA_V3}
COMMAND_SEQUENCE_SCHEMA = 'rek.client_fixed.command_schedule.v2'
TRANSPORT_METHODS = {
    'SendVelocityCommand',
    'SendMoveEvent',
    'SendSpecialEvent',
    'SendEStopToggle',
}
ROBOT_FLAG_FIELDS = (
    'visual_only',
    'player_controlled',
    'falling',
    'fallen',
    'dampened',
    'resetting',
    'motor_shutdown',
)
ROBOT_FLAG_PROPERTIES = {
    'visual_only': 'IsVisualOnly',
    'player_controlled': 'IsPlayerControlled',
    'falling': 'IsFalling',
    'fallen': 'IsFallen',
    'dampened': 'IsDampened',
    'resetting': 'IsResetting',
    'motor_shutdown': 'IsMotorShutdown',
}
TRANSPORT_INPUT_BOOL_FIELDS = (
    'network_initialized',
    'active',
    'punching',
    'recovering',
    'pending_move',
    'pending_special',
    'action_playing',
)
TRANSPORT_INPUT_INT_FIELDS = (
    'network_index',
    'pending_move_index',
    'pending_special_command',
)
TRANSPORT_INPUT_NUMBER_FIELDS = (
    'action_clip_frame',
    'action_clip_fps',
)
TRANSPORT_INPUT_OPAQUE_FIELDS = ('action_clip',)
EXPECTED_GAME_ASSEMBLY_SHA256 = (
    '6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412')
EXPECTED_METADATA_SHA256 = (
    'e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path):
    records = []
    with open(path, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'{path}:{line_number}: invalid JSON: {exc.msg}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{line_number}: record is not an object')
            records.append(record)
    if not records:
        raise ValueError(f'{path}: no records')
    return records


def _inventory_identity(path):
    inventory = json.loads(Path(path).read_text(encoding='utf-8'))
    fingerprint = inventory.get('build_fingerprint')
    if not fingerprint:
        raise ValueError('inventory has no build_fingerprint')
    if inventory.get('errors'):
        raise ValueError('inventory contains unreadable-file errors')
    files = {
        str(record.get('path', '')).replace('\\', '/'): record.get('sha256')
        for record in inventory.get('files', [])
    }
    expected = {
        'GameAssembly.dll': EXPECTED_GAME_ASSEMBLY_SHA256,
        'REK_Data/il2cpp_data/Metadata/global-metadata.dat':
            EXPECTED_METADATA_SHA256,
    }
    for name, digest in expected.items():
        if str(files.get(name, '')).lower() != digest:
            raise ValueError(f'inventory does not pin expected {name}')
    return inventory, fingerprint


def _number(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _put(frame, name, value):
    value = _number(value)
    if value is not None:
        frame[name] = value


def _real_number(value):
    if isinstance(value, bool):
        return None
    return _number(value)


def _put_real(frame, name, value):
    value = _real_number(value)
    if value is not None:
        frame[name] = value


def _put_bool(frame, name, value):
    if isinstance(value, bool):
        frame[name] = 1.0 if value else 0.0


def _put_int(frame, name, value):
    if isinstance(value, int) and not isinstance(value, bool):
        frame[name] = float(value)


def _component(values, index):
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _required_int(record, name, context):
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{context} has no measured integer {name}')
    return value


def _input_command(record, event_name):
    source = record.get('input')
    if not isinstance(source, dict):
        raise ValueError(f'{event_name} has no measured input object')
    return {
        'velocity_command': source.get('velocity_command'),
        'pending_move': source.get('pending_move'),
        'pending_move_index': source.get('pending_move_index'),
        'pending_special': source.get('pending_special'),
        'pending_special_command': source.get('pending_special_command'),
        'action_clip': source.get('action_clip'),
        'action_clip_frame': source.get('action_clip_frame'),
    }


def _measured_vector(values, length):
    if not isinstance(values, list) or len(values) != length:
        return None
    measured = [_real_number(value) for value in values]
    if any(value is None for value in measured):
        return None
    return measured


def _transport_event_input(record):
    source = record.get('input')
    if not isinstance(source, dict):
        return {}, []

    payload = {}
    excluded = []
    velocity = source.get('velocity_command')
    if 'velocity_command' in source and velocity is not None:
        measured_velocity = _measured_vector(velocity, 3)
        if measured_velocity is None:
            excluded.append('velocity_command')
        else:
            payload['velocity_command'] = measured_velocity

    for name in TRANSPORT_INPUT_BOOL_FIELDS:
        if name not in source or source[name] is None:
            continue
        if isinstance(source[name], bool):
            payload[name] = source[name]
        else:
            excluded.append(name)

    for name in TRANSPORT_INPUT_INT_FIELDS:
        if name not in source or source[name] is None:
            continue
        if isinstance(source[name], int) and not isinstance(source[name], bool):
            payload[name] = source[name]
        else:
            excluded.append(name)

    for name in TRANSPORT_INPUT_NUMBER_FIELDS:
        if name not in source or source[name] is None:
            continue
        value = _real_number(source[name])
        if value is None:
            excluded.append(name)
        else:
            payload[name] = value

    for name in TRANSPORT_INPUT_OPAQUE_FIELDS:
        if name in source and source[name] is not None:
            excluded.append(name)
    return payload, sorted(excluded)


def _transport_event_context(record, method):
    unity_frame = record.get('unity_frame')
    unity_time = _real_number(record.get('unity_unscaled_time'))
    expected_provenance = f'REKApp.RobotInputController.{method} prefix'
    provenance = record.get('provenance')
    if (not isinstance(unity_frame, int) or isinstance(unity_frame, bool)
            or unity_time is None
            or provenance != expected_provenance):
        return None
    return {
        'client_fixed_tick_at_observation': _required_int(
            record, 'client_fixed_tick_at_observation',
            'client_transport_method_invoked'),
        'unity_frame': unity_frame,
        'unity_unscaled_time': unity_time,
        'provenance': provenance,
    }


def _semantic_discrete_commands(records, recorder_schema, last_tick):
    if recorder_schema != SCHEMA_V3:
        return []
    commands = []
    fields = {
        'SendMoveEvent': 'pending_move_index',
        'SendSpecialEvent': 'pending_special_command',
        'SendEStopToggle': None,
    }
    for record in records:
        if record.get('event') != 'client_transport_method_invoked':
            continue
        method = record.get('method')
        if method not in fields:
            continue
        tick = _required_int(
            record, 'client_fixed_tick_at_observation',
            'client_transport_method_invoked')
        if tick > last_tick:
            continue
        if _transport_event_context(record, method) is None:
            raise ValueError(
                f'client_transport_method_invoked for {method} has no '
                'measured Unity frame/time and exact callback provenance')
        command = {'client_fixed_tick': tick, 'method': method}
        field = fields[method]
        if field is not None:
            source = record.get('input')
            value = source.get(field) if isinstance(source, dict) else None
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f'{method} has no measured integer {field}')
            command[field] = value
        commands.append(command)
    return commands


def _transport_observation_sha256(records, recorder_schema):
    sequence = []
    if recorder_schema == SCHEMA_V1:
        for record in records:
            if record.get('event') != 'client_send_frame':
                continue
            command = {
                'client_fixed_tick_at_observation': _required_int(
                    record, 'client_fixed_tick_at_observation', 'client_send_frame'),
                'client_send_frame_sequence': _required_int(
                    record, 'client_send_frame_sequence', 'client_send_frame'),
            }
            command.update(_input_command(record, 'client_send_frame'))
            sequence.append(command)
        if not sequence:
            raise ValueError('raw trace has no observed ClientSendFrame command sequence')
        observation = 'RobotInputController.ClientSendFrame prefix'
    elif recorder_schema == SCHEMA_V3:
        method_sequences = {}
        for record in records:
            if record.get('event') != 'client_transport_method_invoked':
                continue
            method = record.get('method')
            if method not in TRANSPORT_METHODS:
                raise ValueError(
                    f'client transport event has unsupported method {method!r}')
            command = {
                'client_fixed_tick_at_observation': _required_int(
                    record, 'client_fixed_tick_at_observation',
                    'client_transport_method_invoked'),
                'client_transport_invocation_sequence': _required_int(
                    record, 'client_transport_invocation_sequence',
                    'client_transport_method_invoked'),
                'method': method,
                'method_invocation_sequence': _required_int(
                    record, 'method_invocation_sequence',
                    'client_transport_method_invoked'),
            }
            command.update(_input_command(
                record, 'client_transport_method_invoked'))
            sequence.append(command)
            method_sequences.setdefault(method, []).append(
                command['method_invocation_sequence'])
        if not sequence:
            raise ValueError(
                'raw trace has no observed concrete client transport invocations')
        if not any(command['method'] == 'SendVelocityCommand'
                   for command in sequence):
            raise ValueError('raw trace has no observed SendVelocityCommand invocation')
        observed = [command['client_transport_invocation_sequence']
                    for command in sequence]
        if observed != list(range(1, len(sequence) + 1)):
            raise ValueError(
                'client transport invocation sequence is not contiguous from one')
        for method, method_observed in method_sequences.items():
            if method_observed != list(range(1, len(method_observed) + 1)):
                raise ValueError(
                    f'{method} invocation sequence is not contiguous from one')
        observation = 'RobotInputController concrete transport method prefixes'
    else:
        raise ValueError(f'unsupported recorder schema: {recorder_schema!r}')
    if not sequence:
        raise AssertionError('validated command sequence is unexpectedly empty')
    encoded = json.dumps(
        sequence, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), observation


def _command_sequence_sha256(samples, records, recorder_schema):
    sequence = []
    for sample in samples:
        tick = _required_int(sample, 'client_fixed_tick', 'sample')
        source = sample.get('input')
        if not isinstance(source, dict):
            raise ValueError(f'sample tick {tick} has no measured input object')
        velocity = _measured_vector(source.get('velocity_command'), 3)
        if velocity is None:
            raise ValueError(
                f'sample tick {tick} has no measured three-axis velocity command')
        local = _required_int(sample, 'local_fighter_index', 'sample')
        if local not in (0, 1):
            raise ValueError(
                f'sample tick {tick} has invalid local_fighter_index {local!r}')
        command = {
            'client_fixed_tick': tick,
            'local_fighter_index': local,
            'velocity_command': velocity,
        }
        if recorder_schema == SCHEMA_V1:
            command.update(_input_command(sample, 'sample'))
        sequence.append(command)
    if not sequence:
        raise ValueError('no sampled input command sequence')
    document = {
        'schema': COMMAND_SEQUENCE_SCHEMA,
        'fixed_tick_commands': sequence,
        'discrete_transport_commands': _semantic_discrete_commands(
            records, recorder_schema, sequence[-1]['client_fixed_tick']),
    }
    encoded = json.dumps(
        document, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return (
        hashlib.sha256(encoded).hexdigest(),
        'RecorderBehaviour.FixedUpdate measured velocity command plus discrete '
        'RobotInputController transport method prefixes')


def _safe_name(value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
    return value or None


def _flatten_robot(frame, slot, robot, bone_names):
    if not isinstance(robot, dict):
        return
    root_vectors = {
        'root_position': ('root.%d.pos', ('x', 'y', 'z')),
        'root_rotation': ('root.%d.quat', ('x', 'y', 'z', 'w')),
        'root_linear_velocity': ('root.%d.vel', ('x', 'y', 'z')),
        'root_angular_velocity': ('root.%d.angvel', ('x', 'y', 'z')),
    }
    for source, (prefix, axes) in root_vectors.items():
        values = robot.get(source)
        for index, axis in enumerate(axes):
            name = f'{prefix % slot}.{axis}'
            value = _component(values, index)
            if source in ('root_linear_velocity', 'root_angular_velocity'):
                _put_real(frame, name, value)
            else:
                _put(frame, name, value)

    for source in ROBOT_FLAG_FIELDS:
        _put_bool(frame, f'robot.{slot}.{source}', robot.get(source))
    _put_int(
        frame, f'contact.{slot}.floor_contact_count',
        robot.get('floor_contact_count'))

    bones = robot.get('bones')
    if not isinstance(bones, dict):
        return
    count = bones.get('count')
    if not isinstance(count, int) or count != len(bone_names):
        raise ValueError(
            f'fighter {slot} bone count changed or is unavailable: '
            f'{count!r} versus header {len(bone_names)}')

    layouts = {
        'world_positions_xyz': ('world.pos', ('x', 'y', 'z')),
        'world_rotations_xyzw': ('world.quat', ('x', 'y', 'z', 'w')),
        'local_positions_xyz': ('local.pos', ('x', 'y', 'z')),
        'local_rotations_xyzw': ('local.quat', ('x', 'y', 'z', 'w')),
    }
    for source, (kind, axes) in layouts.items():
        values = bones.get(source)
        expected = count * len(axes)
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(
                f'fighter {slot} {source} has length '
                f'{len(values) if isinstance(values, list) else None}; '
                f'expected {expected}')
        for bone_index, bone_name in enumerate(bone_names):
            measured_name = _safe_name(bone_name)
            identity = (f'{bone_index:02d}_{measured_name}' if measured_name
                        else f'{bone_index:02d}')
            for component, axis in enumerate(axes):
                offset = bone_index * len(axes) + component
                _put(frame,
                     f'joint.{slot}.{identity}.{kind}.{axis}',
                     values[offset])


def _flatten_sample(sample, bone_names, recorder_schema):
    frame = {}
    tick = sample.get('client_fixed_tick')
    _put(frame, 'tick.client', tick)

    local = sample.get('local_fighter_index')
    if local not in (0, 1):
        raise ValueError(f'sample has invalid local_fighter_index: {local!r}')
    command = sample.get('input')
    if isinstance(command, dict):
        velocity = command.get('velocity_command')
        for index, axis in enumerate(('x', 'y', 'z')):
            _put_real(frame, f'cmd.{local}.velocity.{axis}',
                      _component(velocity, index))
        for name in ('active', 'punching', 'recovering', 'pending_move',
                     'pending_special', 'network_initialized'):
            _put_bool(frame, f'cmd.{local}.{name}', command.get(name))

    transport = sample.get('transport_observation')
    if isinstance(transport, dict):
        if recorder_schema == SCHEMA_V1:
            _put(frame, 'seq.client_send',
                 transport.get('client_send_frame_sequence'))
        elif recorder_schema == SCHEMA_V3:
            _put(frame, 'seq.client_transport_invoke',
                 transport.get('client_transport_invocation_sequence'))
        elif recorder_schema == SCHEMA_V5:
            _put(frame, 'seq.client_transport_invoke',
                 transport.get('client_transport_invocation_sequence'))
            _put(frame, 'seq.raw_protocol_rx',
                 transport.get('raw_protocol_sequence'))
            _put(frame, 'seq.raw_fight_state_rx',
                 transport.get('raw_fight_state_sequence'))
            _put(frame, 'seq.raw_score_rx',
                 transport.get('raw_score_sequence'))
            _put(frame, 'seq.raw_hit_rx',
                 transport.get('raw_hit_sequence'))
        _put(frame, 'seq.server_snapshot_rx',
             transport.get('fight_state_snapshot_sequence'))

    _put(frame, 'round.state', sample.get('phase_value'))
    round_state = sample.get('round')
    if isinstance(round_state, dict):
        _put(frame, 'round.number', round_state.get('number'))
        _put(frame, 'round.timer', round_state.get('time_remaining'))
        _put(frame, 'round.active', round_state.get('active'))
        _put(frame, 'round.result', round_state.get('result_value'))
        _put(frame, 'round.winner', round_state.get('winner_index'))
        _put(frame, 'round.knockout', round_state.get('knockout'))
        for slot in (0, 1):
            _put(frame, f'score.{slot}', _component(round_state.get('clean_hits'), slot))
            _put(frame, f'downs.{slot}', _component(round_state.get('falls'), slot))

    fight = sample.get('fight')
    if isinstance(fight, dict):
        _put(frame, 'fight.format', fight.get('format_value'))
        _put(frame, 'fight.current_round', fight.get('current_round'))
        _put(frame, 'fight.result', fight.get('result_value'))
        _put(frame, 'fight.winner', fight.get('winner_index'))
        for slot in (0, 1):
            _put(frame, f'fight.rounds_won.{slot}',
                 _component(fight.get('rounds_won'), slot))

    for slot in (0, 1):
        _flatten_robot(frame, slot, sample.get(f'fighter_{slot}'), bone_names[slot])
    return frame


def _provenance(channel, recorder_schema):
    if channel == 'tick.client':
        if recorder_schema == SCHEMA_V5:
            return {
                'kind': 'controlled_experiment',
                'ref': 'RekEvidenceRecorder compact sample client_fixed_tick; '
                       'one unit is one client Unity FixedUpdate call',
            }
        return {'kind': 'controlled_experiment',
                'ref': 'RekEvidenceRecorder RecorderBehaviour.FixedUpdate sample index'}
    if channel.startswith('cmd.'):
        if recorder_schema == SCHEMA_V5:
            observation = (
                'exact REK_Input/REK_Move outbound request projections and '
                'separate controlled schedule log')
        else:
            observation = (
                'ClientSendFrame prefix' if recorder_schema == SCHEMA_V1 else
                'concrete SendVelocityCommand/SendMoveEvent/SendSpecialEvent/'
                'SendEStopToggle prefixes')
        return {'kind': 'class',
                'ref': 'REKApp.RobotInputController fields sampled in client '
                       f'FixedUpdate; transport invocations observed at {observation}'}
    if channel == 'seq.client_send':
        return {'kind': 'method',
                'ref': 'REKApp.RobotInputController.ClientSendFrame prefix observation counter'}
    if channel == 'seq.client_transport_invoke':
        return {'kind': 'method',
                'ref': 'REKApp.RobotInputController concrete transport method prefix observation counter'}
    if channel == 'seq.server_snapshot_rx':
        return {'kind': 'method',
                'ref': 'REKApp.FightCoordinator.ApplyFightStateSnapshot postfix observation counter'}
    if channel.startswith('seq.raw_'):
        return {
            'kind': 'message',
            'ref': 'RekEvidenceRecorder receive-boundary raw protocol packet counter',
        }
    if channel.startswith(('round.', 'score.', 'downs.', 'fight.')):
        return {'kind': 'class',
                'ref': 'REKApp.FightCoordinator, FightState and RoundState after ApplyFightStateSnapshot'}
    match = re.fullmatch(r'root\.([01])\.(vel|angvel)\.([xyz])', channel)
    if match:
        slot, kind, axis = match.groups()
        source = {
            'vel': 'root_linear_velocity',
            'angvel': 'root_angular_velocity',
        }[kind]
        source_property = {
            'vel': 'RootLinearVelocity',
            'angvel': 'RootAngularVelocity',
        }[kind]
        component = {'x': 0, 'y': 1, 'z': 2}[axis]
        raw_field = f'fighter_{slot}.{source}[{component}]'
        return {
            'kind': 'class',
            'ref': f'RekEvidenceRecorder sample {raw_field} read from '
                   f'REKApp.Robot.{source_property}',
            'raw_field': raw_field,
        }
    match = re.fullmatch(
        r'robot\.([01])\.(' + '|'.join(ROBOT_FLAG_FIELDS) + r')', channel)
    if match:
        slot, source = match.groups()
        raw_field = f'fighter_{slot}.{source}'
        return {
            'kind': 'class',
            'ref': f'RekEvidenceRecorder sample {raw_field} read from '
                   f'REKApp.Robot.{ROBOT_FLAG_PROPERTIES[source]}',
            'raw_field': raw_field,
        }
    match = re.fullmatch(r'contact\.([01])\.floor_contact_count', channel)
    if match:
        raw_field = f'fighter_{match.group(1)}.floor_contact_count'
        return {
            'kind': 'class',
            'ref': f'RekEvidenceRecorder sample {raw_field} read from '
                   'REKApp.Robot.FloorContactCount',
            'raw_field': raw_field,
        }
    if channel.startswith('root.'):
        return {'kind': 'class',
                'ref': 'REKApp.Robot RootTransform/RootLinearVelocity/RootAngularVelocity on visual-only client robot'}
    if channel.startswith('joint.'):
        return {'kind': 'class',
                'ref': 'REKApp.Robot.boneTransforms populated by OnBoneMessageReceived and ClientApplyBones'}
    raise ValueError(f'no provenance rule for channel {channel}')


def _events(records, samples, recorder_schema, complete_round):
    last_tick = _required_int(samples[-1], 'client_fixed_tick', 'sample')
    events = [{'tick': 0, 'kind': 'round_start',
               'basis': 'first client FixedUpdate with RoundState.IsActive=true'}]
    previous = None
    for sample in samples:
        tick = _required_int(sample, 'client_fixed_tick', 'sample')
        round_state = sample.get('round') or {}
        if previous is not None:
            for slot in (0, 1):
                for field, kind in (('clean_hits', 'score'), ('falls', 'fall')):
                    before = _component(previous.get(field), slot)
                    after = _component(round_state.get(field), slot)
                    if isinstance(before, int) and isinstance(after, int) and after > before:
                        events.append({'tick': tick, 'kind': kind, 'fighter': slot,
                                       'delta': after - before, 'value': after})
            before_ko = previous.get('knockout')
            after_ko = round_state.get('knockout')
            if (isinstance(before_ko, bool) and isinstance(after_ko, bool)
                    and not before_ko and after_ko):
                events.append({'tick': tick, 'kind': 'ko',
                               'winner': round_state.get('winner_index')})
        previous = round_state

    for record in records:
        kind = record.get('event')
        if kind == 'client_send_frame' and recorder_schema == SCHEMA_V1:
            tick = _required_int(
                record, 'client_fixed_tick_at_observation', kind)
            if tick <= last_tick:
                events.append({
                    'tick': tick,
                    'kind': 'command_send',
                    'sequence': _required_int(
                        record, 'client_send_frame_sequence', kind),
                })
        elif (kind == 'client_transport_method_invoked'
              and recorder_schema == SCHEMA_V3):
            method = record.get('method')
            if method not in TRANSPORT_METHODS:
                raise ValueError(
                    f'client transport event has unsupported method {method!r}')
            tick = _required_int(
                record, 'client_fixed_tick_at_observation', kind)
            if tick <= last_tick:
                context = _transport_event_context(record, method)
                if context is None:
                    raise ValueError(
                        f'{kind} for {method} has no measured Unity frame/time '
                        'and exact callback provenance')
                input_payload, excluded_input_fields = (
                    _transport_event_input(record))
                event = {
                    'tick': tick,
                    'kind': 'command_transport_invoked',
                    'sequence': _required_int(
                        record, 'client_transport_invocation_sequence', kind),
                    'method': method,
                    'method_sequence': _required_int(
                        record, 'method_invocation_sequence', kind),
                    **context,
                }
                if input_payload:
                    event['input'] = input_payload
                if excluded_input_fields:
                    event['excluded_input_fields'] = excluded_input_fields
                events.append(event)
        elif kind == 'fight_state_snapshot_applied':
            tick = _required_int(
                record, 'client_fixed_tick_at_observation', kind)
            if tick <= last_tick:
                events.append({
                    'tick': tick,
                    'kind': 'server_snapshot_rx',
                    'sequence': _required_int(
                        record, 'fight_state_snapshot_sequence', kind),
                })
        elif kind == 'capture_end' and 'round_not_active' in str(record.get('reason')):
            tick = _required_int(record, 'client_fixed_tick_at_end', kind)
            if complete_round:
                events.append({
                    'tick': tick,
                    'kind': 'round_end',
                    'basis': 'private-AI scope exited when RoundState.IsActive became false',
                })
    return events


def convert(raw_path, inventory_path, output_path, tick_limit=None):
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    if raw_path.name.endswith('.partial'):
        raise ValueError('refusing an unfinalized .partial recorder file')
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite {output_path}')

    records = _read_jsonl(raw_path)
    starts = [record for record in records if record.get('event') == 'capture_start']
    all_samples = [record for record in records if record.get('event') == 'sample']
    ends = [record for record in records if record.get('event') == 'capture_end']
    if len(starts) != 1:
        raise ValueError(f'expected one capture_start, found {len(starts)}')
    if len(all_samples) < 2:
        raise ValueError(
            f'need at least two client fixed-tick samples, found {len(all_samples)}')
    if len(ends) != 1:
        raise ValueError(f'expected one capture_end, found {len(ends)}')

    start = starts[0]
    end = ends[0]
    recorder_schema = start.get('schema')
    if recorder_schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f'unsupported recorder schema: {recorder_schema!r}')
    if start.get('tick_level_claim') is not True or start.get('tick_domain') != 'client_fixed_update':
        raise ValueError('recorder did not declare the measured client fixed-tick domain')
    if start.get('server_tick_available') is not False:
        raise ValueError('server tick availability was not recorded as false')
    if not str(start.get('server_tick_reason', '')).strip():
        raise ValueError('recorder did not state why the server tick is unavailable')
    fixed_delta_time = _number(start.get('fixed_delta_time'))
    if fixed_delta_time is None or fixed_delta_time <= 0:
        raise ValueError('recorder has no measured positive fixed_delta_time')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', str(start.get('plugin_sha256', ''))):
        raise ValueError('recorder has no plugin SHA-256 identity')
    if str(start.get('game_assembly_sha256', '')).lower() != EXPECTED_GAME_ASSEMBLY_SHA256:
        raise ValueError('raw trace GameAssembly hash does not match the pinned build')
    if str(start.get('global_metadata_sha256', '')).lower() != EXPECTED_METADATA_SHA256:
        raise ValueError('raw trace metadata hash does not match the pinned build')
    if recorder_schema == SCHEMA_V3:
        target_status = start.get('harmony_target_status') or {}
        required_targets = {
            'REKApp.RobotInputController.SendVelocityCommand',
            'REKApp.RobotInputController.SendMoveEvent',
            'REKApp.RobotInputController.SendSpecialEvent',
            'REKApp.RobotInputController.SendEStopToggle',
            'REKApp.FightCoordinator.ApplyFightStateSnapshot',
        }
        unverified_targets = sorted(
            target for target in required_targets
            if target_status.get(target) is not True)
        if unverified_targets:
            raise ValueError(
                f'recorder did not verify Harmony ownership for {unverified_targets}')

    inventory, fingerprint = _inventory_identity(inventory_path)
    server = start.get('server') or {}
    if not server.get('endpoint') or not server.get('session_id'):
        raise ValueError(
            'server-authoritative raw trace has no endpoint or ArenaID session identity')

    scope = start.get('scope') or {}
    required_scope = {
        'allowed': True,
        'network_connected': True,
        'network_is_client': True,
        'network_is_server': False,
        'opponent_is_ai': True,
        'opponent_slot_is_ai': True,
        'human_in_opponent_slot': False,
        'opponent_slot_has_client': False,
        'opponent_human_bit_set': False,
        'fighter_0_visual_only': True,
        'fighter_1_visual_only': True,
    }
    wrong_scope = {
        name: {'expected': expected, 'observed': scope.get(name)}
        for name, expected in required_scope.items()
        if scope.get(name) is not expected
    }
    if wrong_scope:
        raise ValueError(f'raw trace is outside the non-live AI scope: {wrong_scope}')
    if scope.get('sparring_bot_number') != 1:
        raise ValueError(
            'raw trace is not identified as the operator-selected sparring bot 1')

    bone_names = {
        0: start.get('fighter_0_bones'),
        1: start.get('fighter_1_bones'),
    }
    if not all(isinstance(bone_names[slot], list) and bone_names[slot]
               for slot in (0, 1)):
        raise ValueError('capture_start does not declare both fighter bone layouts')

    all_expected_ticks = list(range(len(all_samples)))
    actual_ticks = [sample.get('client_fixed_tick') for sample in all_samples]
    if actual_ticks != all_expected_ticks:
        raise ValueError(
            f'client fixed ticks are not contiguous from zero: {actual_ticks[:8]}')
    if _required_int(end, 'sample_count', 'capture_end') != len(all_samples):
        raise ValueError('capture_end sample_count disagrees with sample records')
    if _required_int(end, 'capture_error_count', 'capture_end') != 0:
        raise ValueError('raw trace contains capture errors')
    if _required_int(end, 'client_fixed_tick_at_end', 'capture_end') < len(all_samples):
        raise ValueError('capture_end client tick precedes the final sample')

    if recorder_schema == SCHEMA_V3:
        transport_records = [
            record for record in records
            if record.get('event') == 'client_transport_method_invoked']
        declared_transport_count = _required_int(
            end, 'client_transport_invocation_count', 'capture_end')
        if declared_transport_count != len(transport_records):
            raise ValueError(
                'capture_end transport invocation count disagrees with event records')
        declared_method_counts = end.get('client_transport_method_counts')
        if not isinstance(declared_method_counts, dict):
            raise ValueError('capture_end has no transport method counts')
        observed_method_counts = {
            method: sum(record.get('method') == method
                        for record in transport_records)
            for method in TRANSPORT_METHODS
            if any(record.get('method') == method
                   for record in transport_records)
        }
        if declared_method_counts != observed_method_counts:
            raise ValueError(
                'capture_end transport method counts disagree with event records')

    transport_observation_sha256, transport_observation = (
        _transport_observation_sha256(records, recorder_schema))

    if tick_limit is not None:
        if isinstance(tick_limit, bool) or tick_limit < 2:
            raise ValueError('tick_limit must be an integer of at least 2')
        if tick_limit > len(all_samples):
            raise ValueError(
                f'tick_limit {tick_limit} exceeds {len(all_samples)} measured samples')
        samples = all_samples[:tick_limit]
    else:
        samples = all_samples
    expected_ticks = list(range(len(samples)))
    command_sequence_sha256, command_observation = (
        _command_sequence_sha256(samples, records, recorder_schema))

    frames = [
        _flatten_sample(sample, bone_names, recorder_schema)
        for sample in samples]
    channel_union = set().union(*(set(frame) for frame in frames))
    channels = sorted(set.intersection(*(set(frame) for frame in frames)))
    if not channels:
        raise ValueError('no numeric channel is present in every sample')
    required = {'tick.client', 'round.timer', 'root.0.pos.x', 'root.1.pos.x'}
    missing_required = sorted(required - set(channels))
    if missing_required:
        raise ValueError(f'raw trace lacks required measured channels: {missing_required}')

    provenance = {
        channel: _provenance(channel, recorder_schema)
        for channel in channels}
    server_record = {
        'endpoint': server['endpoint'],
        'session_id': server['session_id'],
        'protocol': server.get('protocol'),
        'server_reported_version': server.get('server_reported_version'),
        'endpoint_provenance': server.get('endpoint_provenance'),
        'session_id_provenance': server.get('session_id_provenance'),
    }
    raw_hash = _sha256(raw_path)
    events = _events(
        records, samples, recorder_schema, complete_round=tick_limit is None)
    temporary = output_path.with_name(output_path.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(f'refusing existing temporary path {temporary}')
    try:
        with TraceWriter(
                temporary,
                channels,
                fingerprint,
                'rek',
                authority='server',
                server=server_record,
                provenance=provenance,
                tick_domain='client_fixed_update',
                server_tick_available=False,
                server_tick_reason=start.get('server_tick_reason'),
                fixed_delta_time=fixed_delta_time,
                client_buildid=(inventory.get('steam') or {}).get('buildid'),
                raw_recorder_schema=recorder_schema,
                raw_recorder_sha256=raw_hash,
                recorder_plugin_sha256=start.get('plugin_sha256'),
                command_sequence_sha256=command_sequence_sha256,
                command_sequence_schema=COMMAND_SEQUENCE_SCHEMA,
                command_observation=command_observation,
                transport_observation_sha256=transport_observation_sha256,
                transport_observation=transport_observation,
                observation_window_ticks=len(samples),
                raw_sample_count=len(all_samples),
                complete_round=tick_limit is None,
                omitted_nonuniform_channels=sorted(channel_union - set(channels)),
                event_provenance={
                    ('command_send' if recorder_schema == SCHEMA_V1
                     else 'command_transport_invoked'): command_observation,
                    'server_snapshot_rx': 'FightCoordinator.ApplyFightStateSnapshot postfix',
                    'score/fall/ko': 'changes in measured RoundState values',
                    'round_start/end': 'measured RoundState.IsActive scope boundary',
                }) as writer:
            for tick, frame in enumerate(frames):
                writer.append(tick, frame)
            for event in events:
                payload = dict(event)
                event_tick = payload.pop('tick')
                event_kind = payload.pop('kind')
                writer.event(event_tick, event_kind, **payload)

        loaded = Trace.load(temporary)
        if len(loaded) != len(samples) or loaded.ticks != expected_ticks:
            raise ValueError('written trace failed tick/count verification')
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        'schema': 1,
        'raw_path': str(raw_path.resolve()),
        'raw_sha256': raw_hash,
        'output_path': str(output_path.resolve()),
        'output_sha256': _sha256(output_path),
        'build_fingerprint': fingerprint,
        'ticks': len(samples),
        'channels': len(channels),
        'events': len(events),
        'tick_domain': 'client_fixed_update',
        'server_tick_available': False,
        'command_sequence_sha256': command_sequence_sha256,
        'command_sequence_schema': COMMAND_SEQUENCE_SCHEMA,
        'command_observation': command_observation,
        'transport_observation_sha256': transport_observation_sha256,
        'transport_observation': transport_observation,
        'observation_window_ticks': len(samples),
        'raw_sample_count': len(all_samples),
        'complete_round': tick_limit is None,
        'raw_recorder_schema': recorder_schema,
        'server_endpoint_present': True,
        'server_session_present': True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--ticks', type=int,
        help='emit exactly this measured client-fixed-tick prefix')
    args = parser.parse_args(argv)
    result = convert(args.raw, args.inventory, args.out, tick_limit=args.ticks)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
