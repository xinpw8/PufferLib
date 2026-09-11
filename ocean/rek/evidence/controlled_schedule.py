"""Validate the fixed 50 Hz command schedule and its bridge execution log."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


SCHEMA = 'rek.client_fixed.command_schedule.v2'
SCHEDULE_ID = 'rek.private_bot1.baseline.v1'
UNITY_FIXED_RATE_HZ = 500
SCHEDULE_RATE_HZ = 50
FIXED_SUBSTEPS_PER_TICK = 10
DURATION_TICKS = 2601
EXPECTED_SHA256 = (
    '39aaab9c3156e8f4d114daac4d4328257b81230ec8b8a372ad2739d38754ec0d')
EXPECTED_MOVES = (
    (900, 2),
    (1100, 3),
    (1300, 4),
    (1500, 5),
    (1700, 9),
    (1900, 10),
    (2100, 2),
    (2400, 3),
)
EXPECTED_VELOCITY_SEGMENTS = (
    (0, 50, (0.0, 0.0, 0.0)),
    (50, 150, (1.0, 0.0, 0.0)),
    (150, 200, (0.0, 0.0, 0.0)),
    (200, 300, (-1.0, 0.0, 0.0)),
    (300, 350, (0.0, 0.0, 0.0)),
    (350, 450, (0.0, -1.0, 0.0)),
    (450, 500, (0.0, 0.0, 0.0)),
    (500, 600, (0.0, 1.0, 0.0)),
    (600, 650, (0.0, 0.0, 0.0)),
    (650, 750, (0.0, 0.0, -1.0)),
    (750, 800, (0.0, 0.0, 0.0)),
    (800, 900, (0.0, 0.0, 1.0)),
    (900, 2100, (0.0, 0.0, 0.0)),
    (2100, 2300, (1.0, 0.0, 0.0)),
    (2300, 2400, (0.0, 0.0, 0.0)),
    (2400, 2600, (-1.0, 0.0, 0.0)),
    (2600, 2601, (0.0, 0.0, 0.0)),
)
_TOP_LEVEL_KEYS = {
    'duration_ticks',
    'fixed_substeps_per_tick',
    'move_commands',
    'schedule_id',
    'schedule_rate_hz',
    'schema',
    'unity_fixed_rate_hz',
    'velocity_component_order',
    'velocity_segments',
}
_HEX64 = re.compile(r'[0-9a-f]{64}')


def _read_json(path):
    try:
        document = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path}: invalid JSON: {exc.msg}') from exc
    if not isinstance(document, dict):
        raise ValueError(f'{path}: document is not an object')
    return document


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


def canonical_bytes(document):
    return json.dumps(
        document, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')


def canonical_sha256(document):
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{label} is not an integer')
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} is not a number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{label} is not finite')
    return value


def _vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f'{label} is not a three-component list')
    return tuple(_number(component, f'{label}[{index}]')
                 for index, component in enumerate(value))


def validate_manifest(path):
    document = _read_json(path)
    if set(document) != _TOP_LEVEL_KEYS:
        raise ValueError(
            'schedule manifest top-level keys differ from the controlled contract')
    exact_scalars = {
        'schema': SCHEMA,
        'schedule_id': SCHEDULE_ID,
        'unity_fixed_rate_hz': UNITY_FIXED_RATE_HZ,
        'schedule_rate_hz': SCHEDULE_RATE_HZ,
        'fixed_substeps_per_tick': FIXED_SUBSTEPS_PER_TICK,
        'duration_ticks': DURATION_TICKS,
        'velocity_component_order': ['forward', 'strafe', 'yaw'],
    }
    for name, expected in exact_scalars.items():
        if document.get(name) != expected:
            raise ValueError(
                f'schedule manifest {name} mismatch: '
                f'{document.get(name)!r} versus {expected!r}')

    moves = document.get('move_commands')
    if not isinstance(moves, list):
        raise ValueError('schedule manifest move_commands is not a list')
    observed_moves = []
    for index, move in enumerate(moves):
        if not isinstance(move, dict) or set(move) != {'tick', 'move_index'}:
            raise ValueError(f'schedule move {index} has invalid keys')
        observed_moves.append((
            _integer(move.get('tick'), f'schedule move {index} tick'),
            _integer(move.get('move_index'), f'schedule move {index} index'),
        ))
    if tuple(observed_moves) != EXPECTED_MOVES:
        raise ValueError('schedule manifest move command list mismatch')

    segments = document.get('velocity_segments')
    if not isinstance(segments, list):
        raise ValueError('schedule manifest velocity_segments is not a list')
    observed_segments = []
    for index, segment in enumerate(segments):
        if (not isinstance(segment, dict) or
                set(segment) != {'start', 'stop', 'velocity_command'}):
            raise ValueError(f'schedule velocity segment {index} has invalid keys')
        observed_segments.append((
            _integer(segment.get('start'), f'schedule segment {index} start'),
            _integer(segment.get('stop'), f'schedule segment {index} stop'),
            _vector(segment.get('velocity_command'),
                    f'schedule segment {index} velocity'),
        ))
    if tuple(observed_segments) != EXPECTED_VELOCITY_SEGMENTS:
        raise ValueError('schedule manifest velocity segment list mismatch')

    digest = canonical_sha256(document)
    if digest != EXPECTED_SHA256:
        raise ValueError('schedule manifest canonical SHA-256 is not the pinned contract')
    return {
        'document': document,
        'sha256': digest,
        'path': str(Path(path).resolve()),
    }


def _expected_steps():
    velocities = {
        start: velocity for start, _stop, velocity in EXPECTED_VELOCITY_SEGMENTS
    }
    moves = dict(EXPECTED_MOVES)
    ticks = sorted(set(velocities) | set(moves))
    current_velocity = None
    steps = []
    for tick in ticks:
        if tick in velocities:
            current_velocity = velocities[tick]
        steps.append((tick, current_velocity, moves.get(tick)))
    return tuple(steps)


def _schedule_identity(record, label, manifest):
    if record.get('protocol') != 'rek.ui_bridge.v1':
        raise ValueError(f'{label} bridge protocol mismatch')
    if record.get('schedule_id') != SCHEDULE_ID:
        raise ValueError(f'{label} schedule id mismatch')
    if record.get('command_sequence_schema') != SCHEMA:
        raise ValueError(f'{label} command sequence schema mismatch')
    digest = record.get('command_sequence_sha256')
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise ValueError(f'{label} has no lowercase command sequence SHA-256')
    if digest != manifest['sha256']:
        raise ValueError(f'{label} schedule SHA-256 does not match manifest')
    run_id = record.get('schedule_run_id')
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f'{label} has no schedule run id')
    return run_id


def validate_control_log(path, manifest, unity_fixed_window=None):
    records = _read_jsonl(path)
    schedule_records = [
        record for record in records
        if record.get('event') in {
            'schedule_step',
            'schedule_move_send_invoked',
            'schedule_move_send_completed',
            'schedule_end',
        }
    ]
    if not schedule_records:
        raise ValueError('control log has no schedule records')

    run_ids = {
        _schedule_identity(record, 'control schedule record', manifest)
        for record in schedule_records
    }
    if unity_fixed_window is None:
        if len(run_ids) != 1:
            raise ValueError('control log contains more than one schedule run')
        run_id = next(iter(run_ids))
    else:
        if (not isinstance(unity_fixed_window, tuple) or
                len(unity_fixed_window) != 2):
            raise ValueError('control-log Unity fixed-time window is malformed')
        window_start = _number(unity_fixed_window[0], 'control window start')
        window_end = _number(unity_fixed_window[1], 'control window end')
        if window_end < window_start:
            raise ValueError('control-log Unity fixed-time window is reversed')
        candidates = []
        for candidate_run_id in run_ids:
            run_records = [
                record for record in schedule_records
                if record.get('schedule_run_id') == candidate_run_id
            ]
            first_steps = [
                record for record in run_records
                if record.get('event') == 'schedule_step' and
                record.get('schedule_tick') == 0
            ]
            complete_ends = [
                record for record in run_records
                if record.get('event') == 'schedule_end' and
                record.get('complete') is True and
                record.get('reason') == 'complete'
            ]
            if len(first_steps) != 1 or len(complete_ends) != 1:
                continue
            candidate_start = _number(
                first_steps[0].get('unity_fixed_time'),
                f'control run {candidate_run_id} start fixed time')
            candidate_end = _number(
                complete_ends[0].get('unity_fixed_time'),
                f'control run {candidate_run_id} end fixed time')
            if (window_start - 1e-4 <= candidate_start and
                    candidate_end <= window_end + 1e-4):
                candidates.append(candidate_run_id)
        if len(candidates) != 1:
            raise ValueError(
                'control log does not identify exactly one completed schedule '
                f'run inside the raw capture window: found {len(candidates)}')
        run_id = candidates[0]

    schedule_records = [
        record for record in schedule_records
        if record.get('schedule_run_id') == run_id
    ]
    steps = [record for record in schedule_records
             if record.get('event') == 'schedule_step']
    ends = [record for record in schedule_records
            if record.get('event') == 'schedule_end']
    if len(ends) != 1:
        raise ValueError(f'expected one schedule_end, found {len(ends)}')
    if schedule_records[-1].get('event') != 'schedule_end':
        raise ValueError('control log has a schedule step after schedule_end')

    expected_steps = _expected_steps()
    if len(steps) != len(expected_steps):
        raise ValueError(
            f'controlled schedule step count mismatch: '
            f'{len(steps)} versus {len(expected_steps)}')
    start_time = None
    previous_time = None
    accepted_moves = []
    for index, (record, expected) in enumerate(zip(steps, expected_steps)):
        tick, velocity, move_index = expected
        label = f'control schedule step {index}'
        observed_tick = _integer(record.get('schedule_tick'), f'{label} tick')
        if observed_tick != tick:
            raise ValueError(
                f'{label} tick mismatch: {observed_tick} versus {tick}')
        substep = _integer(
            record.get('client_fixed_substep'), f'{label} fixed substep')
        if substep != tick * FIXED_SUBSTEPS_PER_TICK:
            raise ValueError(f'{label} is not at the required fixed substep')
        if record.get('fixed_substeps_per_schedule_tick') != FIXED_SUBSTEPS_PER_TICK:
            raise ValueError(f'{label} fixed-substep ratio mismatch')
        if _vector(record.get('velocity_command_xyz'), f'{label} velocity') != velocity:
            raise ValueError(f'{label} velocity does not match manifest')
        observed_move = record.get('move_index')
        if isinstance(observed_move, bool) or (
                observed_move is not None and not isinstance(observed_move, int)):
            raise ValueError(f'{label} move index is malformed')
        if observed_move != move_index:
            raise ValueError(f'{label} move index does not match manifest')
        if record.get('move_accepted_locally') is not True:
            raise ValueError(f'{label} was not accepted locally')
        if record.get('server_acceptance_observed') is not False:
            raise ValueError(f'{label} claims server acceptance')
        fixed_time = _number(record.get('unity_fixed_time'), f'{label} fixed time')
        if start_time is None:
            start_time = fixed_time
        expected_time = start_time + tick / SCHEDULE_RATE_HZ
        if not math.isclose(fixed_time, expected_time, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f'{label} fixed time does not match its schedule tick')
        if previous_time is not None and fixed_time < previous_time:
            raise ValueError('control schedule fixed time decreased')
        previous_time = fixed_time
        if observed_move is not None:
            accepted_moves.append((tick, observed_move))

    if tuple(accepted_moves) != EXPECTED_MOVES:
        raise ValueError('control log does not contain exactly eight accepted moves')

    for event_name in ('schedule_move_send_invoked',
                       'schedule_move_send_completed'):
        sends = [record for record in schedule_records
                 if record.get('event') == event_name]
        if len(sends) != len(EXPECTED_MOVES):
            raise ValueError(
                f'control log does not contain exactly eight {event_name} events')
        for index, (record, (tick, move_index)) in enumerate(
                zip(sends, EXPECTED_MOVES)):
            label = f'{event_name} {index}'
            if _integer(record.get('schedule_tick'), f'{label} tick') != tick:
                raise ValueError(f'{label} schedule tick mismatch')
            if _integer(record.get('move_index'), f'{label} move') != move_index:
                raise ValueError(f'{label} move index mismatch')
            if record.get('pending_move_readback') is not True:
                raise ValueError(f'{label} did not observe an armed pending move')
            if _integer(
                    record.get('pending_move_index_readback'),
                    f'{label} pending move readback') != move_index:
                raise ValueError(f'{label} pending move readback mismatch')
            if record.get('server_acceptance_observed') is not False:
                raise ValueError(f'{label} claims server acceptance')

    end = ends[0]
    if end.get('complete') is not True or end.get('reason') != 'complete':
        raise ValueError('control schedule does not have a successful complete marker')
    if _integer(end.get('schedule_tick'), 'schedule end tick') != DURATION_TICKS - 1:
        raise ValueError('control schedule ended at the wrong schedule tick')
    final_completed_substep = DURATION_TICKS * FIXED_SUBSTEPS_PER_TICK - 1
    if _integer(end.get('client_fixed_substep'), 'schedule end substep') != (
            final_completed_substep):
        raise ValueError('control schedule ended at the wrong fixed substep')
    if _integer(
            end.get('move_send_completed_count'),
            'schedule end completed-move count') != len(EXPECTED_MOVES):
        raise ValueError('control schedule did not complete exactly eight move sends')
    if end.get('final_neutral_send_observed') is not True:
        raise ValueError('control schedule did not observe its final neutral send')
    if end.get('server_acceptance_observed') is not False:
        raise ValueError('control schedule end claims server acceptance')
    end_time = _number(end.get('unity_fixed_time'), 'schedule end fixed time')
    expected_end_time = start_time + final_completed_substep / UNITY_FIXED_RATE_HZ
    if not math.isclose(end_time, expected_end_time, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError('control schedule end fixed time does not match duration')

    state_records = [record for record in records if record.get('event') == 'state']
    matching_states = []
    for record in state_records:
        control = record.get('control')
        if not isinstance(control, dict):
            continue
        if (control.get('schedule_run_id') == run_id and
                control.get('schedule_running') is True):
            matching_states.append(record)
    if not matching_states:
        raise ValueError(
            'control log has no state proving the active controlled schedule run')
    required_private_scope = {
        'proven': True,
        'active_gameplay_proven': True,
        'network_client_only': True,
        'context_is_solo': True,
        'opponent_is_ai': True,
        'opponent_slot_is_ai': True,
        'human_in_opponent_slot': False,
        'opponent_slot_client_known': True,
        'opponent_slot_has_client': False,
        'opponent_human_bit_set': False,
        'sparring_bot_number': 1,
        'exact_sparring_bot_1': True,
        'client_visual_only_fighter_pair': True,
        'round_active': True,
    }
    for record in matching_states:
        if record.get('protocol') != 'rek.ui_bridge.v1':
            raise ValueError('control state bridge protocol mismatch')
        build = record.get('build') or {}
        private_ai = record.get('private_ai') or {}
        state_control = record.get('control') or {}
        if (str(build.get('game_assembly_sha256', '')).lower() !=
                '6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412'):
            raise ValueError('control state GameAssembly hash mismatch')
        if (str(build.get('global_metadata_sha256', '')).lower() !=
                'e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd'):
            raise ValueError('control state metadata hash mismatch')
        wrong_private_scope = {
            name: {'expected': expected, 'observed': private_ai.get(name)}
            for name, expected in required_private_scope.items()
            if private_ai.get(name) is not expected
        }
        if wrong_private_scope:
            raise ValueError(
                'control state does not prove exact active private Bot 1 '
                f'gameplay: {wrong_private_scope}')
        if state_control.get('schedule_id') != SCHEDULE_ID:
            raise ValueError('control state schedule id mismatch')
        if state_control.get('command_sequence_schema') != SCHEMA:
            raise ValueError('control state command sequence schema mismatch')
        if state_control.get('command_sequence_sha256') != manifest['sha256']:
            raise ValueError('control state schedule SHA-256 mismatch')
        if (state_control.get('fixed_substeps_per_schedule_tick') !=
                FIXED_SUBSTEPS_PER_TICK):
            raise ValueError('control state fixed-substep ratio mismatch')

    return {
        'path': str(Path(path).resolve()),
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'run_id': run_id,
        'steps': steps,
        'end': end,
        'start_unity_fixed_time': start_time,
        'end_unity_fixed_time': end_time,
        'accepted_moves': accepted_moves,
    }
