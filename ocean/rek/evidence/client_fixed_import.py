"""Convert the Windows private-AI recorder JSONL into the common trace format.

Legacy v1/v3 captures sample once per client Unity FixedUpdate. Protocol v5
captures compact state every ten FixedUpdate calls and exact protocol-boundary
events. A v5 import additionally requires the completed semantic-control log and
its canonical schedule manifest, then labels measured compact observations on
the schedule's 50 Hz grid. The recovered packet layouts expose no server tick,
so this importer never labels trace ticks as server ticks and never synthesizes
one.

For v5 commands, the bridge log defines the accepted control-frame window and
the pinned schedule. RekEvidenceRecorder Send* prefix records are authoritative
for the local outbound stream inside that window. Those records prove method
invocation and projected REK_Input/REK_Move bodies. They do not prove method
completion, network delivery, server acceptance or execution, and they cannot
exclude an uninstrumented lower-level transmission path. Hook ownership is
attested at capture start, not continuously for each hook, so absence claims
mean zero recorder observations at the audited prefixes.

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
import struct
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
V5_OUTBOUND_LATEST_SUBSTEP_OFFSET = -1
V5_OUTBOUND_EVENTS = {
    'outbound_request_projection',
    'client_transport_method_invoked',
}
V5_FORBIDDEN_CONTROL_METHODS = {
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
            'kind': 'transport_message',
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


def _v5_start_client_tick(samples, start_unity_fixed_time, fixed_delta_time):
    candidates = []
    for index, sample in enumerate(samples):
        client_tick = _required_int(sample, 'client_fixed_tick', 'v5 sample')
        fixed_time = _real_number(sample.get('unity_fixed_time'))
        if fixed_time is None:
            raise ValueError(f'v5 sample {index} has no measured Unity fixed time')
        elapsed_fixed_ticks = (fixed_time - start_unity_fixed_time) / fixed_delta_time
        rounded = round(elapsed_fixed_ticks)
        if abs(elapsed_fixed_ticks - rounded) > 1e-3:
            raise ValueError(
                f'v5 sample {index} time is not on the controlled Unity fixed grid')
        candidates.append(client_tick - rounded)
    if len(set(candidates)) != 1:
        raise ValueError('v5 samples disagree on the controlled schedule start tick')
    start_tick = candidates[0]
    if start_tick < 0:
        raise ValueError('controlled schedule starts before the recorder capture')
    return start_tick


def _v5_normalized_samples(all_samples, start_tick):
    final_substep = (
        (controlled_schedule.DURATION_TICKS - 1) *
        controlled_schedule.FIXED_SUBSTEPS_PER_TICK)
    selected = []
    normalized_ticks = []
    phases = []
    for sample in all_samples:
        client_tick = _required_int(sample, 'client_fixed_tick', 'v5 sample')
        relative = client_tick - start_tick
        if relative < 0 or relative > final_substep:
            continue
        schedule_tick = (
            relative + controlled_schedule.FIXED_SUBSTEPS_PER_TICK // 2
        ) // controlled_schedule.FIXED_SUBSTEPS_PER_TICK
        phase = relative - (
            schedule_tick * controlled_schedule.FIXED_SUBSTEPS_PER_TICK)
        selected.append(sample)
        normalized_ticks.append(schedule_tick)
        phases.append(phase)

    if len(selected) not in {
            controlled_schedule.DURATION_TICKS - 1,
            controlled_schedule.DURATION_TICKS}:
        raise ValueError(
            'completed v5 schedule window does not contain the expected 50 Hz '
            f'compact samples: found {len(selected)}')
    if normalized_ticks != list(range(normalized_ticks[0], normalized_ticks[-1] + 1)):
        raise ValueError('v5 normalized schedule ticks are not contiguous')
    if len(set(phases)) != 1:
        raise ValueError('v5 compact sample phase changed inside the schedule window')
    if normalized_ticks[0] not in (0, 1) or normalized_ticks[-1] not in (
            controlled_schedule.DURATION_TICKS - 2,
            controlled_schedule.DURATION_TICKS - 1):
        raise ValueError('v5 compact samples do not span the completed schedule window')
    return selected, normalized_ticks, phases[0]


def _validate_v5_samples(samples, scope):
    local = scope.get('local_fighter_index')
    opponent = scope.get('opponent_slot')
    for index, sample in enumerate(samples):
        label = f'v5 schedule sample {index}'
        if sample.get('local_fighter_index') != local:
            raise ValueError(f'{label} local fighter changed')
        if sample.get('opponent_slot') != opponent:
            raise ValueError(f'{label} opponent slot changed')
        if sample.get('sparring_bot_number') != 1:
            raise ValueError(f'{label} is not Sparring Bot 1')
        round_state = sample.get('round')
        if not isinstance(round_state, dict) or round_state.get('active') is not True:
            raise ValueError(f'{label} is outside an active round')
        for slot in (0, 1):
            robot = sample.get(f'fighter_{slot}')
            if not isinstance(robot, dict) or robot.get('visual_only') is not True:
                raise ValueError(f'{label} fighter {slot} is not a measured visual-only robot')


def _schedule_tick_for_client_tick(client_tick, start_tick):
    relative = client_tick - start_tick
    return (
        relative + controlled_schedule.FIXED_SUBSTEPS_PER_TICK // 2
    ) // controlled_schedule.FIXED_SUBSTEPS_PER_TICK


def _v5_final_controlled_substep():
    return (
        controlled_schedule.DURATION_TICKS *
        controlled_schedule.FIXED_SUBSTEPS_PER_TICK - 1)


def _v5_control_window(control_log_path, control, manifest):
    control_records = _read_jsonl(control_log_path)
    starts = [
        record for record in control_records
        if record.get('event') == 'ack' and
        record.get('command') == 'StartMeasuredSchedule' and
        record.get('schedule_run_id') == control['run_id']
    ]
    if len(starts) != 1:
        raise ValueError(
            'control log does not contain exactly one accepted schedule-start '
            f'ack for run {control["run_id"]}')
    start = starts[0]
    required_start = {
        'protocol': 'rek.ui_bridge.v1',
        'status': 'accepted',
        'reason': 'measured_schedule_started',
        'applied': True,
        'client_request_issued': False,
        'server_acceptance_observed': False,
        'schedule_id': controlled_schedule.SCHEDULE_ID,
        'command_sequence_schema': controlled_schedule.SCHEMA,
        'command_sequence_sha256': manifest['sha256'],
        'unity_thread': 'main',
    }
    wrong_start = {
        name: {'expected': expected, 'observed': start.get(name)}
        for name, expected in required_start.items()
        if start.get(name) != expected
    }
    if wrong_start:
        raise ValueError(
            f'control schedule-start ack is malformed: {wrong_start}')
    start_frame = _required_int(start, 'unity_frame', 'control schedule-start ack')
    end_frame = _required_int(
        control['end'], 'unity_frame', 'control schedule_end')
    if end_frame <= start_frame:
        raise ValueError('control schedule Unity-frame window is empty or reversed')

    step_frames = [
        _required_int(step, 'unity_frame', 'control schedule step')
        for step in control['steps']
    ]
    if step_frames != sorted(step_frames):
        raise ValueError('control schedule Unity frames decreased')
    if step_frames[0] <= start_frame:
        raise ValueError(
            'control schedule step zero did not follow its accepted start frame')
    if step_frames[-1] >= end_frame:
        raise ValueError(
            'control schedule final step did not precede schedule_end')
    return {
        'start_unity_frame': start_frame,
        'end_unity_frame_exclusive': end_frame,
        'steps': [
            {
                'schedule_tick': step['schedule_tick'],
                'unity_frame': frame,
                'velocity_command_xyz': step['velocity_command_xyz'],
            }
            for step, frame in zip(control['steps'], step_frames)
        ],
    }


def _v5_expected_velocity(controlled_substep):
    if controlled_substep == V5_OUTBOUND_LATEST_SUBSTEP_OFFSET:
        return 0, controlled_schedule.EXPECTED_VELOCITY_SEGMENTS[0][2]
    schedule_tick = (
        controlled_substep // controlled_schedule.FIXED_SUBSTEPS_PER_TICK)
    for segment_index, (start, stop, velocity) in enumerate(
            controlled_schedule.EXPECTED_VELOCITY_SEGMENTS):
        if start <= schedule_tick < stop:
            return segment_index, velocity
    raise ValueError(
        f'controlled substep {controlled_substep} is outside the pinned schedule')


def _v5_expected_velocity_at_frame(unity_frame, control_window):
    expected = controlled_schedule.EXPECTED_VELOCITY_SEGMENTS[0][2]
    for step in control_window['steps']:
        if step['unity_frame'] > unity_frame:
            break
        expected = step['velocity_command_xyz']
    return expected


def _v5_float32_vector_bytes(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{label} is not a measured three-axis vector')
    components = [_real_number(component) for component in value]
    if any(component is None for component in components):
        raise ValueError(f'{label} is not a measured finite three-axis vector')
    return struct.pack('<3f', *components)


def _validate_v5_outbound_command_stream(records, schedule_sample_tick,
                                         local_slot, control_window):
    start_tick = schedule_sample_tick
    final_substep = _v5_final_controlled_substep()
    end_tick = start_tick + final_substep
    start_frame = control_window['start_unity_frame']
    end_frame = control_window['end_unity_frame_exclusive']
    outbound = [
        record for record in records
        if record.get('event') in V5_OUTBOUND_EVENTS
    ]
    outbound_frames = [
        _required_int(record, 'unity_frame', 'v5 outbound request')
        for record in outbound
    ]
    if outbound_frames != sorted(outbound_frames):
        raise ValueError('v5 raw outbound Unity frames decreased')
    outbound_client_ticks = [
        _required_int(
            record, 'client_fixed_tick_at_observation', 'v5 outbound request')
        for record in outbound
    ]
    if outbound_client_ticks != sorted(outbound_client_ticks):
        raise ValueError('v5 raw outbound client ticks decreased')
    stream = []
    for record, unity_frame in zip(outbound, outbound_frames):
        if start_frame <= unity_frame < end_frame:
            stream.append(record)
    if not stream:
        raise ValueError('v5 controlled window has no raw outbound requests')

    request_sequences = [
        _required_int(record, 'request_sequence', 'v5 outbound request')
        for record in stream
    ]
    if request_sequences != list(range(
            request_sequences[0], request_sequences[-1] + 1)):
        raise ValueError(
            'v5 controlled outbound window is not a contiguous recorder sequence')
    client_ticks = [
        _required_int(
            record, 'client_fixed_tick_at_observation', 'v5 outbound request')
        for record in stream
    ]
    if client_ticks != sorted(client_ticks):
        raise ValueError('v5 controlled outbound client ticks decreased')
    relative_client_ticks = [tick - start_tick for tick in client_ticks]
    if (relative_client_ticks[0] < 0 or
            relative_client_ticks[-1] > final_substep):
        raise ValueError(
            'v5 controlled outbound ticks do not fit the measured '
            'recorder/control phase')

    forbidden = [
        record for record in stream
        if record.get('event') == 'client_transport_method_invoked' and
        record.get('method') in V5_FORBIDDEN_CONTROL_METHODS
    ]
    if forbidden:
        methods = [record.get('method') for record in forbidden]
        raise ValueError(
            'v5 controlled window contains forbidden outbound invocation(s): '
            f'{methods}')

    inputs = [
        record for record in stream
        if record.get('event') == 'outbound_request_projection' and
        record.get('message') == 'REK_Input'
    ]
    moves = [
        record for record in stream
        if record.get('event') == 'outbound_request_projection' and
        record.get('message') == 'REK_Move'
    ]
    if not inputs:
        raise ValueError('v5 controlled window has no raw REK_Input projection')
    input_client_ticks = [
        _required_int(
            record, 'client_fixed_tick_at_observation', 'v5 REK_Input projection')
        for record in inputs
    ]
    input_delta_counts = {}
    for left, right in zip(input_client_ticks, input_client_ticks[1:]):
        delta = str(right - left)
        input_delta_counts[delta] = input_delta_counts.get(delta, 0) + 1

    segment_counts = [0] * len(controlled_schedule.EXPECTED_VELOCITY_SEGMENTS)
    for index, record in enumerate(inputs):
        if record.get('network_index_source_int32') != local_slot:
            raise ValueError(f'v5 REK_Input projection {index} came from another fighter')
        client_tick = _required_int(
            record, 'client_fixed_tick_at_observation', 'v5 REK_Input projection')
        controlled_substep = (
            client_tick - start_tick + V5_OUTBOUND_LATEST_SUBSTEP_OFFSET)
        segment_index, expected_velocity = _v5_expected_velocity(
            controlled_substep)
        unity_frame = _required_int(
            record, 'unity_frame', 'v5 REK_Input projection')
        frame_velocity = _v5_expected_velocity_at_frame(
            unity_frame, control_window)
        if (struct.pack('<3f', *expected_velocity) !=
                _v5_float32_vector_bytes(
                    frame_velocity, 'control schedule-step velocity')):
            raise ValueError(
                'v5 recorder client-tick phase disagrees with the control '
                f'transcript at REK_Input projection {index}')
        actual_bytes = _v5_float32_vector_bytes(
            record.get('velocity_command_xyz'),
            f'v5 REK_Input projection {index} velocity')
        expected_bytes = struct.pack('<3f', *expected_velocity)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f'v5 REK_Input projection {index} does not match the pinned '
                f'schedule at controlled substep {controlled_substep}')
        segment_counts[segment_index] += 1
    for index, (count, segment) in enumerate(zip(
            segment_counts, controlled_schedule.EXPECTED_VELOCITY_SEGMENTS)):
        if count == 0:
            start, stop, _velocity = segment
            raise ValueError(
                f'raw REK_Input stream has no projection for pinned velocity '
                f'segment {index} [{start}, {stop})')

    observed_moves = []
    for index, record in enumerate(moves):
        if record.get('network_index_source_int32') != local_slot:
            raise ValueError(f'v5 REK_Move projection {index} came from another fighter')
        observed_moves.append(_required_int(
            record, 'move_index_source_int32', 'v5 REK_Move projection'))
    expected_moves = [
        move_index for _schedule_tick, move_index in
        controlled_schedule.EXPECTED_MOVES
    ]
    if observed_moves != expected_moves:
        raise ValueError(
            'raw controlled window does not contain exactly eight REK_Move '
            'projections in the pinned order')

    for index, (record, (schedule_tick, _move_index)) in enumerate(
            zip(moves, controlled_schedule.EXPECTED_MOVES)):
        step_frame = next(
            step['unity_frame'] for step in control_window['steps']
            if step['schedule_tick'] == schedule_tick)
        first_input = next((
            candidate for candidate in inputs
            if _required_int(
                candidate,
                'unity_frame', 'v5 REK_Input projection') >= step_frame
        ), None)
        if first_input is None:
            raise ValueError(
                f'raw REK_Move projection {index} has no following REK_Input '
                'send boundary')
        move_tick = _required_int(
            record, 'client_fixed_tick_at_observation', 'v5 REK_Move projection')
        input_tick = _required_int(
            first_input,
            'client_fixed_tick_at_observation',
            'v5 REK_Input projection')
        move_sequence = _required_int(
            record, 'request_sequence', 'v5 REK_Move projection')
        input_sequence = _required_int(
            first_input, 'request_sequence', 'v5 REK_Input projection')
        if move_tick != input_tick or move_sequence != input_sequence + 1:
            raise ValueError(
                f'raw REK_Move projection {index} was not immediately paired '
                'with the first observed REK_Input send boundary at or after '
                f'schedule tick {schedule_tick}')

    encoded = json.dumps(
        stream, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()
    summary = {
        'schema': 'rek.raw_outbound_window.v1',
        'authority': 'RekEvidenceRecorder Send* prefix records',
        'start_client_fixed_tick': start_tick,
        'end_client_fixed_tick': end_tick,
        'start_unity_frame': start_frame,
        'end_unity_frame_exclusive': end_frame,
        'client_fixed_substeps': final_substep + 1,
        'latest_controlled_substep_offset': (
            V5_OUTBOUND_LATEST_SUBSTEP_OFFSET),
        'first_request_sequence': request_sequences[0],
        'last_request_sequence': request_sequences[-1],
        'records': len(stream),
        'rek_input_projections': len(inputs),
        'rek_move_projections': len(moves),
        'rek_special_invocations': 0,
        'rek_estop_invocations': 0,
        'rek_input_first_controlled_substep': (
            input_client_ticks[0] - start_tick +
            V5_OUTBOUND_LATEST_SUBSTEP_OFFSET),
        'rek_input_last_controlled_substep': (
            input_client_ticks[-1] - start_tick +
            V5_OUTBOUND_LATEST_SUBSTEP_OFFSET),
        'rek_input_segment_observation_counts': segment_counts,
        'rek_input_observed_delta_client_tick_counts': input_delta_counts,
        'cadence_claim': (
            'exact observed LateUpdate invocation ticks only; no periodic '
            'REK_Input cadence is asserted or synthesized'),
        'hook_coverage_claim': (
            'Harmony ownership is attested at capture start; continuous '
            'per-hook ownership is not independently re-attested'),
        'tick_phase': (
            'recorder samples before incrementing its FixedUpdate counter; a '
            'LateUpdate request at relative client tick r observes controlled '
            'substep r-1, with r=0 denoting the accepted activation frame'),
        'proof_limit': (
            'within audited RobotInputController Send* methods, prefix records '
            'prove invocation and projected request bodies; they do not prove '
            'method completion, network delivery, server acceptance, execution, '
            'continuous per-hook ownership, or absence of an uninstrumented '
            'lower-level transmission path'),
        'sha256': digest,
    }
    return {'records': stream, 'summary': summary, 'sha256': digest}


def _validate_v5_window_protocol(records, start_tick):
    final_substep = (
        (controlled_schedule.DURATION_TICKS - 1) *
        controlled_schedule.FIXED_SUBSTEPS_PER_TICK)
    in_window = []
    for record in records:
        if record.get('event') not in {
                'raw_bone_packet',
                'raw_fight_state_packet',
                'fight_state_snapshot_applied'}:
            continue
        client_tick = _required_int(
            record, 'client_fixed_tick_at_observation', record.get('event'))
        if 0 <= client_tick - start_tick <= final_substep:
            in_window.append(record)
    if not any(record.get('event') == 'raw_fight_state_packet'
               for record in in_window):
        raise ValueError('v5 controlled window has no raw REK_FightState packet')
    if not any(record.get('event') == 'fight_state_snapshot_applied'
               for record in in_window):
        raise ValueError('v5 controlled window has no applied fight-state snapshot')
    bone_slots = {
        record.get('fighter_slot') for record in in_window
        if record.get('event') == 'raw_bone_packet'
    }
    if bone_slots != {0, 1}:
        raise ValueError(
            'v5 controlled window has no raw REK_Bones packet for both fighters')


def _v5_events(records, samples, normalized_ticks, control, start_tick,
               outbound_stream):
    final_sample_substep = (
        (controlled_schedule.DURATION_TICKS - 1) *
        controlled_schedule.FIXED_SUBSTEPS_PER_TICK)
    outbound_record_ids = {
        id(record) for record in outbound_stream['records']
    }
    outbound_start_tick = outbound_stream['summary']['start_client_fixed_tick']
    events = [{
        'tick': 0,
        'kind': 'controlled_schedule_start',
        'schedule_run_id': control['run_id'],
    }]
    for step in control['steps']:
        payload = {
            'tick': step['schedule_tick'],
            'kind': 'command_schedule_step',
            'velocity_command': step['velocity_command_xyz'],
            'move_accepted_locally': step['move_accepted_locally'],
            'server_acceptance_observed': False,
        }
        if step.get('move_index') is not None:
            payload['move_index'] = step['move_index']
        events.append(payload)

    previous_round = None
    for sample, tick in zip(samples, normalized_ticks):
        round_state = sample.get('round') or {}
        if previous_round is not None:
            for slot in (0, 1):
                for field, kind in (('clean_hits', 'score'), ('falls', 'fall')):
                    before = _component(previous_round.get(field), slot)
                    after = _component(round_state.get(field), slot)
                    if (isinstance(before, int) and not isinstance(before, bool)
                            and isinstance(after, int) and not isinstance(after, bool)
                            and after > before):
                        events.append({
                            'tick': tick,
                            'kind': kind,
                            'fighter': slot,
                            'delta': after - before,
                            'value': after,
                        })
            if (previous_round.get('knockout') is False and
                    round_state.get('knockout') is True):
                events.append({
                    'tick': tick,
                    'kind': 'ko',
                    'winner': round_state.get('winner_index'),
                })
        previous_round = round_state

    event_names = {
        'outbound_request_projection': 'outbound_request_projection',
        'client_transport_method_invoked': 'outbound_request_invoked',
        'raw_bone_packet': 'raw_bone_packet_rx',
        'raw_fight_state_packet': 'raw_fight_state_packet_rx',
        'raw_score_packet': 'raw_score_packet_rx',
        'raw_hit_packet': 'raw_hit_packet_rx',
        'fight_state_snapshot_applied': 'server_snapshot_rx',
    }
    for record in records:
        output_kind = event_names.get(record.get('event'))
        if output_kind is None:
            continue
        client_tick = _required_int(
            record, 'client_fixed_tick_at_observation', record.get('event'))
        if record.get('event') in V5_OUTBOUND_EVENTS:
            if id(record) not in outbound_record_ids:
                continue
            relative_client_tick = client_tick - outbound_start_tick
            controlled_substep = (
                relative_client_tick + V5_OUTBOUND_LATEST_SUBSTEP_OFFSET)
            event = {
                name: value for name, value in record.items()
                if name != 'event'
            }
            event.update({
                'raw_event': record['event'],
                'tick': max(
                    0,
                    controlled_substep //
                    controlled_schedule.FIXED_SUBSTEPS_PER_TICK),
                'kind': output_kind,
                'controlled_window_client_tick': relative_client_tick,
                'controlled_fixed_substep': controlled_substep,
            })
            events.append(event)
            continue

        relative = client_tick - start_tick
        if relative < 0 or relative > final_sample_substep:
            continue
        event = {
            'tick': _schedule_tick_for_client_tick(client_tick, start_tick),
            'kind': output_kind,
        }
        for field in (
                'client_fixed_tick_at_observation', 'unity_frame',
                'unity_time', 'unity_unscaled_time',
                'message', 'method', 'request_sequence',
                'message_request_sequence', 'method_request_sequence',
                'raw_bone_packet_sequence', 'raw_protocol_sequence',
                'raw_fight_state_sequence', 'raw_score_sequence',
                'raw_hit_sequence', 'fight_state_snapshot_sequence',
                'fighter_slot', 'network_index_source_int32',
                'move_index_source_int32', 'velocity_command_xyz',
                'wire_body_sha256', 'wire_delivery', 'decoded',
                'request_only', 'server_acceptance', 'ack_observed'):
            if field in record:
                event[field] = record[field]
        events.append(event)

    events.append({
        'tick': controlled_schedule.DURATION_TICKS - 1,
        'kind': 'controlled_schedule_end',
        'schedule_run_id': control['run_id'],
        'complete': True,
    })
    return events


def _convert_v5(raw_path, inventory_path, output_path, control_log_path,
                schedule_manifest_path, tick_limit=None):
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    if tick_limit is not None:
        raise ValueError('v5 imports require the complete controlled schedule window')
    if control_log_path is None or schedule_manifest_path is None:
        raise ValueError(
            'v5 imports require --control-log and --schedule-manifest')
    if raw_path.name.endswith('.partial'):
        raise ValueError('refusing an unfinalized .partial recorder file')
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite {output_path}')

    manifest = controlled_schedule.validate_manifest(schedule_manifest_path)
    raw_validation = raw_bone_validate.validate(raw_path)
    records = _read_jsonl(raw_path)
    starts = [record for record in records if record.get('event') == 'capture_start']
    ends = [record for record in records if record.get('event') == 'capture_end']
    all_samples = [record for record in records if record.get('event') == 'sample']
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError('v5 raw capture does not have one start and one end record')
    start = starts[0]
    end = ends[0]
    sample_fixed_times = [
        _real_number(sample.get('unity_fixed_time')) for sample in all_samples]
    if any(value is None for value in sample_fixed_times):
        raise ValueError('v5 compact samples do not all have measured fixed times')
    capture_end_fixed_time = _real_number(end.get('unity_fixed_time_at_end'))
    if capture_end_fixed_time is None:
        capture_end_fixed_time = sample_fixed_times[-1]
    control = controlled_schedule.validate_control_log(
        control_log_path,
        manifest,
        unity_fixed_window=(sample_fixed_times[0], capture_end_fixed_time))

    inventory, fingerprint = _inventory_identity(inventory_path)
    if start.get('schema') != SCHEMA_V5:
        raise ValueError(f'unsupported v5 recorder schema: {start.get("schema")!r}')
    if str(start.get('game_assembly_sha256', '')).lower() != EXPECTED_GAME_ASSEMBLY_SHA256:
        raise ValueError('v5 raw trace GameAssembly hash does not match pinned build')
    if str(start.get('global_metadata_sha256', '')).lower() != EXPECTED_METADATA_SHA256:
        raise ValueError('v5 raw trace metadata hash does not match pinned build')
    if start.get('tick_level_claim') is not False:
        raise ValueError('v5 recorder must not claim compact samples are tick-complete')
    if start.get('tick_domain') != 'client_fixed_update':
        raise ValueError('v5 recorder did not declare its client fixed-update domain')
    if start.get('server_tick_available') is not False:
        raise ValueError('v5 recorder did not preserve absent server ticks')
    fixed_delta_time = _real_number(start.get('fixed_delta_time'))
    expected_fixed_delta = 1.0 / controlled_schedule.UNITY_FIXED_RATE_HZ
    if (fixed_delta_time is None or
            not math.isclose(fixed_delta_time, expected_fixed_delta,
                             rel_tol=0.0, abs_tol=1e-9)):
        raise ValueError('v5 recorder fixed_delta_time is not the controlled 500 Hz value')
    if start.get('client_sample_stride_ticks') != (
            controlled_schedule.FIXED_SUBSTEPS_PER_TICK):
        raise ValueError('v5 recorder compact sample stride is not ten fixed substeps')

    server = start.get('server') or {}
    session_hash = str(server.get('session_id_sha256', '')).lower()
    if (not server.get('endpoint') or
            not re.fullmatch(r'[0-9a-f]{64}', session_hash)):
        raise ValueError('v5 raw trace has no endpoint or hashed session identity')
    if server.get('session_identifier_recorded') is not False or server.get('session_id'):
        raise ValueError('v5 raw trace contains or claims a raw session identifier')

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
        'sparring_bot_number': 1,
    }
    wrong_scope = {
        name: {'expected': expected, 'observed': scope.get(name)}
        for name, expected in required_scope.items()
        if scope.get(name) is not expected
    }
    if wrong_scope:
        raise ValueError(f'v5 raw trace is outside exact private Bot 1 scope: {wrong_scope}')

    schedule_start_tick = _v5_start_client_tick(
        all_samples, control['start_unity_fixed_time'], fixed_delta_time)
    control_window = _v5_control_window(
        control_log_path, control, manifest)
    outbound_stream = _validate_v5_outbound_command_stream(
        records,
        schedule_start_tick,
        scope.get('local_fighter_index'),
        control_window)
    final_client_tick = outbound_stream['summary']['end_client_fixed_tick']
    if _required_int(end, 'client_fixed_tick_at_end', 'v5 capture_end') <= final_client_tick:
        raise ValueError('v5 capture ended before the completed controlled schedule')
    samples, trace_ticks, sample_phase = _v5_normalized_samples(
        all_samples, schedule_start_tick)
    _validate_v5_samples(samples, scope)
    transport_sha256 = outbound_stream['sha256']
    _validate_v5_window_protocol(records, schedule_start_tick)

    bone_names = {
        0: start.get('fighter_0_bones'),
        1: start.get('fighter_1_bones'),
    }
    if not all(isinstance(bone_names[slot], list) and bone_names[slot]
               for slot in (0, 1)):
        raise ValueError('v5 capture_start does not declare both fighter bone layouts')
    frames = [_flatten_sample(sample, bone_names, SCHEMA_V5) for sample in samples]
    channel_union = set().union(*(set(frame) for frame in frames))
    channels = sorted(set.intersection(*(set(frame) for frame in frames)))
    required = {'tick.client', 'round.timer', 'root.0.pos.x', 'root.1.pos.x'}
    missing_required = sorted(required - set(channels))
    if missing_required:
        raise ValueError(f'v5 raw trace lacks required measured channels: {missing_required}')
    provenance = {channel: _provenance(channel, SCHEMA_V5) for channel in channels}

    server_record = {
        'endpoint': server['endpoint'],
        'session_id': f'sha256:{session_hash}',
        'session_id_sha256': session_hash,
        'session_identifier_recorded': False,
        'protocol': server.get('protocol'),
        'arena_region': server.get('arena_region'),
        'arena_scene': server.get('arena_scene'),
        'endpoint_provenance': server.get('endpoint_provenance'),
        'session_id_sha256_provenance': server.get('session_id_sha256_provenance'),
    }
    raw_hash = _sha256(raw_path)
    if raw_validation.get('raw_sha256') != raw_hash:
        raise ValueError('v5 raw capture changed after protocol validation')
    events = _v5_events(
        records, samples, trace_ticks, control, schedule_start_tick,
        outbound_stream)
    validated_fighters = raw_validation.get('fighters')
    if not isinstance(validated_fighters, dict):
        raise ValueError('raw protocol validation has no fighter layouts')
    fighter_layouts = {}
    for slot in (0, 1):
        fighter = validated_fighters.get(str(slot))
        layout = fighter.get('bone_layout') if isinstance(fighter, dict) else None
        if not isinstance(layout, dict):
            raise ValueError(
                f'raw protocol validation has no fighter {slot} bone layout')
        required_layout = {
            'layout_id': layout.get('layout_id'),
            'bone_count': layout.get('bone_count'),
            'wire_body_bytes': layout.get('wire_body_bytes'),
            'identity_claimed': layout.get('identity_claimed'),
        }
        if (not isinstance(required_layout['layout_id'], str)
                or not required_layout['layout_id']
                or not isinstance(required_layout['bone_count'], int)
                or not isinstance(required_layout['wire_body_bytes'], int)
                or not isinstance(required_layout['identity_claimed'], bool)):
            raise ValueError(
                f'raw protocol validation fighter {slot} layout is incomplete')
        fighter_layouts[str(slot)] = required_layout
    fighter_pairing = {
        'local_fighter_index': scope.get('local_fighter_index'),
        'opponent_fighter_index': scope.get('opponent_slot'),
        'fighters': fighter_layouts,
    }
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
                tick_domain='controlled_schedule_50hz',
                tick_rate_hz=controlled_schedule.SCHEDULE_RATE_HZ,
                server_tick_available=False,
                server_tick_reason=start.get('server_tick_reason'),
                client_fixed_delta_time=fixed_delta_time,
                client_fixed_substeps_per_tick=(
                    controlled_schedule.FIXED_SUBSTEPS_PER_TICK),
                command_sample_phase_substeps=sample_phase,
                schedule_start_client_fixed_tick=schedule_start_tick,
                client_buildid=(inventory.get('steam') or {}).get('buildid'),
                raw_recorder_schema=SCHEMA_V5,
                raw_recorder_sha256=raw_hash,
                recorder_plugin_sha256=start.get('plugin_sha256'),
                command_sequence_sha256=manifest['sha256'],
                command_sequence_schema=controlled_schedule.SCHEMA,
                command_observation=(
                    'build-pinned semantic bridge schedule timing validated '
                    'against the complete RekEvidenceRecorder outbound prefix '
                    'stream'),
                schedule_id=controlled_schedule.SCHEDULE_ID,
                schedule_manifest_sha256=manifest['sha256'],
                schedule_run_id=control['run_id'],
                control_log_sha256=control['sha256'],
                transport_observation_sha256=transport_sha256,
                transport_observation_schema='rek.raw_outbound_window.v1',
                transport_observation=(
                    'complete zero-error recorder sequence of audited '
                    'RobotInputController Send* prefixes in the controlled window'),
                outbound_command_stream=outbound_stream['summary'],
                observation_window_ticks=len(samples),
                raw_sample_count=len(all_samples),
                compact_samples_tick_complete=False,
                complete_schedule=True,
                complete_round=False,
                fighter_pairing=fighter_pairing,
                omitted_nonuniform_channels=sorted(channel_union - set(channels)),
                event_provenance={
                    'command_schedule_step': 'validated semantic bridge control log',
                    'outbound_request_projection': (
                        'complete exact RekEvidenceRecorder REK_Input/REK_Move '
                        'prefix projections in the controlled window'),
                    'outbound_request_invoked': (
                        'RekEvidenceRecorder SendSpecialEvent/SendEStopToggle '
                        'prefix invocation observation; forbidden in this schedule'),
                    'raw_*_packet_rx': 'receive-boundary FastBufferReader copies',
                    'server_snapshot_rx': (
                        'FightCoordinator.ApplyFightStateSnapshot postfix'),
                    'score/fall/ko': 'changes in measured compact RoundState values',
                }) as writer:
            for tick, frame in zip(trace_ticks, frames):
                writer.append(tick, frame)
            for event in events:
                payload = dict(event)
                event_tick = payload.pop('tick')
                event_kind = payload.pop('kind')
                writer.event(event_tick, event_kind, **payload)

        loaded = Trace.load(temporary)
        if len(loaded) != len(samples) or loaded.ticks != trace_ticks:
            raise ValueError('written v5 trace failed tick/count verification')
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        'schema': 2,
        'raw_path': str(raw_path.resolve()),
        'raw_sha256': raw_hash,
        'output_path': str(output_path.resolve()),
        'output_sha256': _sha256(output_path),
        'build_fingerprint': fingerprint,
        'ticks': len(samples),
        'first_tick': trace_ticks[0],
        'last_tick': trace_ticks[-1],
        'channels': len(channels),
        'events': len(events),
        'tick_domain': 'controlled_schedule_50hz',
        'server_tick_available': False,
        'command_sequence_sha256': manifest['sha256'],
        'command_sequence_schema': controlled_schedule.SCHEMA,
        'command_sample_phase_substeps': sample_phase,
        'schedule_run_id': control['run_id'],
        'control_log_sha256': control['sha256'],
        'transport_observation_sha256': transport_sha256,
        'outbound_command_stream': outbound_stream['summary'],
        'raw_recorder_schema': SCHEMA_V5,
        'raw_protocol_validation_schema': raw_validation.get('schema'),
        'fighter_pairing': fighter_pairing,
        'server_endpoint_present': True,
        'server_session_hash_present': True,
        'complete_schedule': True,
    }


def _convert_legacy(raw_path, inventory_path, output_path, tick_limit=None):
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


def convert(raw_path, inventory_path, output_path, tick_limit=None,
            control_log_path=None, schedule_manifest_path=None):
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    if raw_path.name.endswith('.partial'):
        raise ValueError('refusing an unfinalized .partial recorder file')
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite {output_path}')
    records = _read_jsonl(raw_path)
    starts = [record for record in records if record.get('event') == 'capture_start']
    recorder_schema = starts[0].get('schema') if len(starts) == 1 else None
    if recorder_schema == SCHEMA_V5:
        return _convert_v5(
            raw_path,
            inventory_path,
            output_path,
            control_log_path,
            schedule_manifest_path,
            tick_limit=tick_limit)
    return _convert_legacy(
        raw_path, inventory_path, output_path, tick_limit=tick_limit)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--control-log',
        help='v5 semantic bridge JSONL containing one completed schedule run')
    parser.add_argument(
        '--schedule-manifest',
        help='v5 canonical command schedule manifest')
    parser.add_argument(
        '--ticks', type=int,
        help='legacy v1/v3 only: emit this measured client-fixed-tick prefix')
    args = parser.parse_args(argv)
    result = convert(
        args.raw,
        args.inventory,
        args.out,
        tick_limit=args.ticks,
        control_log_path=args.control_log,
        schedule_manifest_path=args.schedule_manifest)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
