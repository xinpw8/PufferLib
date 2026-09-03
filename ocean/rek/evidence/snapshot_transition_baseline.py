"""Sanitized snapshot-transition diagnostic for v2 REK command traces.

Only two-fighter root position, linear velocity, and angular velocity are
exported.  Rows are sampled at measured ``server_snapshot_rx`` events.  Inputs
summarize exact transport invocations in each open-left, closed-right snapshot
interval: velocity payload sums and counts, move/special integer sums and
counts, and EStop invocation counts.  Controller status flags are never model
inputs.

Same-tick snapshot callbacks are coalesced because the trace stores only one
sampled state per client tick.  Their exact per-tick multiplicities remain in
the numeric dataset, hash binding, and report counts.

The report compares state-only AR and ARX ridge fits under leave-one
endpoint/session-group-out evaluation.  Fixed lag candidates include a
negative-lag placebo.  This is diagnostic evidence only; it cannot establish
action alignment, simulator behavior, or parity.
"""

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np

from trace import PROVENANCE_KINDS, Trace


RAW_RECORDER_SCHEMA = 'rek.private_ai.client_fixed.v3'
COMMAND_SCHEMA = 'rek.client_fixed.command_schedule.v2'
TICK_DOMAIN = 'client_fixed_update'
LAG_CANDIDATES = (-1, 0, 1, 2)
PROTECTED_NAMES = {'envelope.json', 'parity_report.json'}
HASH_RE = re.compile(r'[0-9a-f]{64}')
TRANSPORT_METHODS = (
    'SendVelocityCommand', 'SendMoveEvent',
    'SendSpecialEvent', 'SendEStopToggle')
STATE_CHANNELS = tuple(
    f'root.{fighter}.{kind}.{axis}'
    for fighter in (0, 1)
    for kind in ('pos', 'vel', 'angvel')
    for axis in ('x', 'y', 'z'))
INPUT_FEATURES = (
    'transport.velocity.sum.x',
    'transport.velocity.sum.y',
    'transport.velocity.sum.z',
    'transport.velocity.count',
    'transport.move.index_sum',
    'transport.move.count',
    'transport.special.index_sum',
    'transport.special.count',
    'transport.estop.count',
)
DATASET_SCHEMA = 'rek.snapshot_transition.numeric.v2'


class SnapshotDiagnosticError(ValueError):
    """A source or artifact fails the measured-only diagnostic contract."""


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _hash_scalar(value):
    return np.asarray(value.encode('ascii'), dtype='S64')


def _hash_vector(values):
    return np.asarray([value.encode('ascii') for value in values], dtype='S64')


def _finite_number(value):
    return (not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _required_hash(record, key, context):
    value = record.get(key)
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise SnapshotDiagnosticError(f'{context} lacks exact {key}')
    return value


def _read_inventory(path):
    try:
        inventory = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SnapshotDiagnosticError('inventory cannot be read') from None
    if not isinstance(inventory, dict) or inventory.get('errors'):
        raise SnapshotDiagnosticError('inventory is incomplete')
    return _required_hash(inventory, 'build_fingerprint', 'inventory')


def _group_hash(endpoint, session):
    encoded = json.dumps(
        [endpoint, session], separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(b'rek.endpoint_session.v1\0' + encoded).hexdigest()


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(b'rek.numeric_array.v1\0')
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(np.asarray(array.shape, dtype='<u8').tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _trace_content_hash(
        group, ticks_hash, multiplicity_hash, state_hash, input_hash):
    digest = hashlib.sha256(b'rek.snapshot_trace_content.v2\0')
    for value in (
            group, ticks_hash, multiplicity_hash, state_hash, input_hash):
        digest.update(value.encode('ascii'))
    return digest.hexdigest()


def _dataset_schema_hash(build_hash, plugin_hash, fixed_dt):
    record = {
        'build_identity_sha256': _sha(build_hash),
        'command_schema_sha256': _sha(COMMAND_SCHEMA),
        'dataset_schema_sha256': _sha(DATASET_SCHEMA),
        'fixed_delta_time_hex': float(fixed_dt).hex(),
        'input_feature_name_sha256': [_sha(name) for name in INPUT_FEATURES],
        'plugin_identity_sha256': _sha(plugin_hash),
        'raw_recorder_schema_sha256': _sha(RAW_RECORDER_SCHEMA),
        'state_channel_name_sha256': [_sha(name) for name in STATE_CHANNELS],
        'tick_domain_sha256': _sha(TICK_DOMAIN),
    }
    blob = json.dumps(
        record, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return hashlib.sha256(blob).hexdigest(), record


def _command_event(event, trace_index, expected_sequence, method_sequences):
    method = event.get('method')
    if method not in TRANSPORT_METHODS:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} contains an unknown transport invocation')
    tick = event.get('tick')
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise SnapshotDiagnosticError(f'trace {trace_index} has an invalid transport tick')
    if event.get('client_fixed_tick_at_observation') != tick:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} has an unbound transport tick')
    if event.get('sequence') != expected_sequence:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} transport sequence is not contiguous')
    method_sequence = event.get('method_sequence')
    expected_method = method_sequences.get(method, 0) + 1
    if method_sequence != expected_method:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} method sequence is not contiguous')
    method_sequences[method] = expected_method
    if (isinstance(event.get('unity_frame'), bool)
            or not isinstance(event.get('unity_frame'), int)
            or not _finite_number(event.get('unity_unscaled_time'))
            or event.get('provenance') !=
            f'REKApp.RobotInputController.{method} prefix'):
        raise SnapshotDiagnosticError(
            f'trace {trace_index} transport invocation is not exactly observed')
    payload = event.get('input')
    if method != 'SendEStopToggle' and not isinstance(payload, dict):
        raise SnapshotDiagnosticError(
            f'trace {trace_index} transport invocation lacks measured input')
    if method == 'SendEStopToggle' and payload is not None and not isinstance(payload, dict):
        raise SnapshotDiagnosticError(
            f'trace {trace_index} EStop invocation has malformed input')
    if method == 'SendVelocityCommand':
        velocity = payload.get('velocity_command')
        if (not isinstance(velocity, list) or len(velocity) != 3
                or not all(_finite_number(value) for value in velocity)):
            raise SnapshotDiagnosticError(
                f'trace {trace_index} has an invalid velocity payload')
        semantic = tuple(float(value) for value in velocity)
    elif method == 'SendMoveEvent':
        value = payload.get('pending_move_index')
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value > (1 << 53)):
            raise SnapshotDiagnosticError(
                f'trace {trace_index} has an invalid move index')
        semantic = value
    elif method == 'SendSpecialEvent':
        value = payload.get('pending_special_command')
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value > (1 << 53)):
            raise SnapshotDiagnosticError(
                f'trace {trace_index} has an invalid special index')
        semantic = value
    else:
        semantic = None
    return {'tick': tick, 'method': method, 'value': semantic}


def _events(trace, trace_index):
    snapshots = []
    commands = []
    method_sequences = {}
    expected_transport = 1
    expected_snapshot = 1
    last_tick = trace.ticks[-1]
    for event in trace.events:
        if not isinstance(event, dict):
            raise SnapshotDiagnosticError(f'trace {trace_index} has an invalid event')
        kind = event.get('kind')
        if kind == 'command_transport_invoked':
            command = _command_event(
                event, trace_index, expected_transport, method_sequences)
            if command['tick'] < 0 or command['tick'] > last_tick:
                raise SnapshotDiagnosticError(
                    f'trace {trace_index} has a transport event outside its ticks')
            commands.append(command)
            expected_transport += 1
        elif kind == 'server_snapshot_rx':
            tick = event.get('tick')
            if (isinstance(tick, bool) or not isinstance(tick, int)
                    or tick < 0 or tick > last_tick
                    or event.get('sequence') != expected_snapshot):
                raise SnapshotDiagnosticError(
                    f'trace {trace_index} has an invalid snapshot sequence')
            snapshots.append(tick)
            expected_snapshot += 1
        elif ('method' in event or (isinstance(kind, str)
                                    and ('command' in kind or 'transport' in kind))):
            raise SnapshotDiagnosticError(
                f'trace {trace_index} contains an unknown command event')
    if any(right < left for left, right in zip(snapshots, snapshots[1:])):
        raise SnapshotDiagnosticError(
            f'trace {trace_index} snapshot ticks are decreasing')
    unique_ticks = []
    multiplicity = []
    for tick in snapshots:
        if unique_ticks and tick == unique_ticks[-1]:
            multiplicity[-1] += 1
        else:
            unique_ticks.append(tick)
            multiplicity.append(1)
    if len(unique_ticks) < max(abs(lag) for lag in LAG_CANDIDATES) + 2:
        raise SnapshotDiagnosticError(f'trace {trace_index} has too few snapshot ticks')
    if not commands or not any(
            command['method'] == 'SendVelocityCommand' for command in commands):
        raise SnapshotDiagnosticError(
            f'trace {trace_index} has no exact velocity invocation')
    return unique_ticks, multiplicity, commands


def _recompute_command_hash(trace, local_slot, commands):
    fixed = []
    names = [f'cmd.{local_slot}.velocity.{axis}' for axis in 'xyz']
    for row, tick in enumerate(trace.ticks):
        fixed.append({
            'client_fixed_tick': tick,
            'local_fighter_index': local_slot,
            'velocity_command': [float(trace.channels[name][row]) for name in names],
        })
    discrete = []
    for command in commands:
        method = command['method']
        if method == 'SendVelocityCommand':
            continue
        item = {'client_fixed_tick': command['tick'], 'method': method}
        if method == 'SendMoveEvent':
            item['pending_move_index'] = command['value']
        elif method == 'SendSpecialEvent':
            item['pending_special_command'] = command['value']
        discrete.append(item)
    document = {
        'schema': COMMAND_SCHEMA,
        'fixed_tick_commands': fixed,
        'discrete_transport_commands': discrete,
    }
    blob = json.dumps(
        document, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def _interval_inputs(snapshot_ticks, commands):
    result = np.zeros((len(snapshot_ticks) - 1, len(INPUT_FEATURES)), dtype='<f8')
    for command in commands:
        tick = command['tick']
        interval = int(np.searchsorted(snapshot_ticks, tick, side='left')) - 1
        if interval < 0 or interval >= result.shape[0]:
            continue
        method = command['method']
        if method == 'SendVelocityCommand':
            result[interval, :3] += command['value']
            result[interval, 3] += 1.0
        elif method == 'SendMoveEvent':
            result[interval, 4] += command['value']
            result[interval, 5] += 1.0
        elif method == 'SendSpecialEvent':
            result[interval, 6] += command['value']
            result[interval, 7] += 1.0
        else:
            result[interval, 8] += 1.0
    return result


def _load_source(path, trace_index, build_hash):
    try:
        trace = Trace.load(path)
    except (OSError, ValueError, KeyError, struct.error):
        raise SnapshotDiagnosticError(f'trace {trace_index} cannot be read') from None
    header = trace.header
    if trace.source != 'rek':
        raise SnapshotDiagnosticError(f'trace {trace_index} source is not rek')
    if trace.build_fingerprint != build_hash:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} does not match the inventory build')
    if trace.authority != 'server':
        raise SnapshotDiagnosticError(f'trace {trace_index} is not server-authoritative')
    server = trace.server
    endpoint = server.get('endpoint')
    session = server.get('session_id')
    if (not isinstance(endpoint, str) or not endpoint.strip()
            or not isinstance(session, str) or not session.strip()):
        raise SnapshotDiagnosticError(
            f'trace {trace_index} lacks endpoint/session grouping identity')
    if header.get('raw_recorder_schema') != RAW_RECORDER_SCHEMA:
        raise SnapshotDiagnosticError(f'trace {trace_index} is not a common v2 trace')
    if header.get('command_sequence_schema') != COMMAND_SCHEMA:
        raise SnapshotDiagnosticError(f'trace {trace_index} is not a v2 command trace')
    if header.get('tick_domain') != TICK_DOMAIN:
        raise SnapshotDiagnosticError(f'trace {trace_index} has a wrong tick domain')
    fixed_dt = header.get('fixed_delta_time')
    if not _finite_number(fixed_dt) or float(fixed_dt) <= 0.0:
        raise SnapshotDiagnosticError(f'trace {trace_index} lacks finite fixed_dt')
    raw_hash = _required_hash(header, 'raw_recorder_sha256', f'trace {trace_index}')
    plugin_hash = _required_hash(
        header, 'recorder_plugin_sha256', f'trace {trace_index}')
    command_hash = _required_hash(
        header, 'command_sequence_sha256', f'trace {trace_index}')
    del raw_hash
    if trace.ticks != list(range(len(trace.ticks))):
        raise SnapshotDiagnosticError(f'trace {trace_index} ticks are not contiguous')
    if not trace.ticks:
        raise SnapshotDiagnosticError(f'trace {trace_index} has no ticks')

    missing = set(STATE_CHANNELS) - set(trace.channels)
    if missing:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} lacks an allowlisted root-state channel')
    for name in STATE_CHANNELS:
        citation = trace.provenance.get(name)
        if (not isinstance(citation, dict)
                or citation.get('kind') not in PROVENANCE_KINDS
                or not isinstance(citation.get('ref'), str)
                or not citation['ref'].strip()):
            raise SnapshotDiagnosticError(
                f'trace {trace_index} lacks measured root-state provenance')

    slots = []
    for slot in (0, 1):
        names = [f'cmd.{slot}.velocity.{axis}' for axis in 'xyz']
        present = [name in trace.channels for name in names]
        if any(present) and not all(present):
            raise SnapshotDiagnosticError(
                f'trace {trace_index} has an incomplete sampled velocity command')
        if all(present):
            slots.append(slot)
    if len(slots) != 1:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} does not identify one local velocity command')
    local_slot = slots[0]
    sampled_velocity = np.column_stack([
        trace.channels[f'cmd.{local_slot}.velocity.{axis}'] for axis in 'xyz'])
    if not np.isfinite(sampled_velocity).all():
        raise SnapshotDiagnosticError(
            f'trace {trace_index} sampled velocity is nonfinite')

    snapshot_ticks, snapshot_multiplicity, commands = _events(trace, trace_index)
    if _recompute_command_hash(trace, local_slot, commands) != command_hash:
        raise SnapshotDiagnosticError(
            f'trace {trace_index} v2 command hash does not verify')
    state = np.asarray([
        [trace.channels[name][tick] for name in STATE_CHANNELS]
        for tick in snapshot_ticks
    ], dtype='<f8')
    if not np.isfinite(state).all():
        raise SnapshotDiagnosticError(f'trace {trace_index} root state is nonfinite')
    inputs = _interval_inputs(snapshot_ticks, commands)
    ticks = np.asarray(snapshot_ticks, dtype='<i8')
    snapshot_multiplicity = np.asarray(snapshot_multiplicity, dtype='<i8')
    group = _group_hash(endpoint, session)
    hashes = {
        'ticks': _array_hash(ticks),
        'snapshot_multiplicity': _array_hash(snapshot_multiplicity),
        'state': _array_hash(state),
        'input': _array_hash(inputs),
    }
    content = _trace_content_hash(
        group, hashes['ticks'], hashes['snapshot_multiplicity'],
        hashes['state'], hashes['input'])
    return {
        'group': group,
        'content': content,
        'ticks': ticks,
        'snapshot_multiplicity': snapshot_multiplicity,
        'state': state,
        'input': inputs,
        'hashes': hashes,
        'fixed_dt': float(fixed_dt),
        'plugin_hash': plugin_hash,
    }
def _output_path(path, suffix):
    path = Path(path)
    if path.name.casefold() in PROTECTED_NAMES:
        raise SnapshotDiagnosticError('refusing a protected canonical output name')
    if path.suffix.casefold() != suffix:
        raise SnapshotDiagnosticError(f'diagnostic output must use {suffix}')
    if path.exists():
        raise FileExistsError('refusing to overwrite diagnostic output')
    if not path.parent.is_dir():
        raise SnapshotDiagnosticError('diagnostic output parent does not exist')
    return path


def _write_npz(path, arrays):
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            np.savez_compressed(stream, **arrays)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def export_snapshot_dataset(trace_paths, inventory_path, output_path):
    """Export verified snapshot states and transport-only interval inputs."""
    paths = list(trace_paths)
    if len(paths) < 2:
        raise SnapshotDiagnosticError('at least two traces are required')
    output_path = _output_path(output_path, '.npz')
    build_hash = _read_inventory(inventory_path)
    records = [_load_source(path, index, build_hash)
               for index, path in enumerate(paths)]
    fixed = {record['fixed_dt'].hex() for record in records}
    plugins = {record['plugin_hash'] for record in records}
    if len(fixed) != 1:
        raise SnapshotDiagnosticError('traces have mixed fixed_dt')
    if len(plugins) != 1:
        raise SnapshotDiagnosticError('traces have mixed recorder plugins')
    groups = {record['group'] for record in records}
    if len(groups) < 2:
        raise SnapshotDiagnosticError(
            'at least two endpoint/session groups are required')
    records.sort(key=lambda record: record['content'])
    contents = [record['content'] for record in records]
    if len(contents) != len(set(contents)):
        raise SnapshotDiagnosticError('duplicate sanitized trace content')
    plugin_hash = records[0]['plugin_hash']
    fixed_dt = records[0]['fixed_dt']
    schema_hash, schema_record = _dataset_schema_hash(
        build_hash, plugin_hash, fixed_dt)
    arrays = {
        'diagnostic_only': np.asarray(True, dtype=np.bool_),
        'simulator_claim': np.asarray(False, dtype=np.bool_),
        'parity_claim': np.asarray(False, dtype=np.bool_),
        'action_alignment_verified': np.asarray(False, dtype=np.bool_),
        'schema_sha256': _hash_scalar(schema_hash),
        'build_identity_sha256': _hash_scalar(
            schema_record['build_identity_sha256']),
        'plugin_identity_sha256': _hash_scalar(
            schema_record['plugin_identity_sha256']),
        'fixed_delta_time': np.asarray(fixed_dt, dtype='<f8'),
        'state_channel_name_sha256': _hash_vector(
            schema_record['state_channel_name_sha256']),
        'input_feature_name_sha256': _hash_vector(
            schema_record['input_feature_name_sha256']),
        'trace_content_sha256': _hash_vector(contents),
        'group_identity_sha256': _hash_vector(
            [record['group'] for record in records]),
    }
    for index, record in enumerate(records):
        prefix = f'trace_{index:06d}'
        for name in ('ticks', 'snapshot_multiplicity', 'state', 'input'):
            arrays[f'{prefix}_{name}'] = record[name]
            arrays[f'{prefix}_{name}_sha256'] = _hash_scalar(
                record['hashes'][name])
    _write_npz(output_path, arrays)
    return {
        'diagnostic_only': True,
        'simulator_claim': False,
        'parity_claim': False,
        'action_alignment_verified': False,
        'schema_sha256': schema_hash,
        'trace_content_sha256': contents,
        'group_identity_sha256': [record['group'] for record in records],
        'metrics': {
            'trace_count': len(records),
            'group_count': len(groups),
            'state_dimensions': len(STATE_CHANNELS),
            'input_dimensions': len(INPUT_FEATURES),
            'snapshot_event_count': sum(
                sum(int(value) for value in record['snapshot_multiplicity'])
                for record in records),
            'unique_snapshot_tick_count': sum(
                len(record['ticks']) for record in records),
            'coalesced_duplicate_count': sum(
                sum(int(value) for value in record['snapshot_multiplicity'])
                - len(record['ticks']) for record in records),
            'transition_count': sum(len(record['input']) for record in records),
        },
    }


def _decode_hash(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != 'S' or array.dtype.itemsize != 64:
        raise SnapshotDiagnosticError(f'{name} is not a SHA-256 value')
    result = bytes(array).decode('ascii')
    if not HASH_RE.fullmatch(result):
        raise SnapshotDiagnosticError(f'{name} is not a SHA-256 value')
    return result


def _decode_hashes(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind != 'S' or array.dtype.itemsize != 64:
        raise SnapshotDiagnosticError(f'{name} is not a SHA-256 vector')
    result = [bytes(item).decode('ascii') for item in array]
    if any(not HASH_RE.fullmatch(item) for item in result):
        raise SnapshotDiagnosticError(f'{name} is not a SHA-256 vector')
    return result


def _flag(dataset, name, expected):
    value = np.asarray(dataset[name])
    if value.shape != () or value.dtype != np.dtype(np.bool_) or bool(value) is not expected:
        raise SnapshotDiagnosticError(f'dataset has an invalid {name} flag')


def _load_dataset(path):
    try:
        dataset = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        raise SnapshotDiagnosticError('snapshot dataset cannot be read') from None
    try:
        required = {
            'diagnostic_only', 'simulator_claim', 'parity_claim',
            'action_alignment_verified', 'schema_sha256',
            'build_identity_sha256', 'plugin_identity_sha256',
            'fixed_delta_time', 'state_channel_name_sha256',
            'input_feature_name_sha256', 'trace_content_sha256',
            'group_identity_sha256',
        }
        if not required.issubset(dataset.files):
            raise SnapshotDiagnosticError('dataset metadata is incomplete')
        _flag(dataset, 'diagnostic_only', True)
        _flag(dataset, 'simulator_claim', False)
        _flag(dataset, 'parity_claim', False)
        _flag(dataset, 'action_alignment_verified', False)
        schema_hash = _decode_hash(dataset['schema_sha256'], 'schema hash')
        build_identity = _decode_hash(
            dataset['build_identity_sha256'], 'build identity')
        plugin_identity = _decode_hash(
            dataset['plugin_identity_sha256'], 'plugin identity')
        state_names = _decode_hashes(
            dataset['state_channel_name_sha256'], 'state channel hashes')
        input_names = _decode_hashes(
            dataset['input_feature_name_sha256'], 'input feature hashes')
        if state_names != [_sha(name) for name in STATE_CHANNELS]:
            raise SnapshotDiagnosticError('dataset state allowlist does not verify')
        if input_names != [_sha(name) for name in INPUT_FEATURES]:
            raise SnapshotDiagnosticError('dataset input allowlist does not verify')
        contents = _decode_hashes(
            dataset['trace_content_sha256'], 'trace content hashes')
        groups = _decode_hashes(
            dataset['group_identity_sha256'], 'group identities')
        if len(contents) < 2 or len(contents) != len(groups):
            raise SnapshotDiagnosticError('dataset trace identities do not align')
        if len(set(contents)) != len(contents) or len(set(groups)) < 2:
            raise SnapshotDiagnosticError('dataset lacks independent group splits')
        fixed = np.asarray(dataset['fixed_delta_time'])
        if (fixed.shape != () or fixed.dtype != np.dtype('<f8')
                or not math.isfinite(float(fixed)) or float(fixed) <= 0.0):
            raise SnapshotDiagnosticError('dataset fixed_dt is invalid')
        schema_record = {
            'build_identity_sha256': build_identity,
            'command_schema_sha256': _sha(COMMAND_SCHEMA),
            'dataset_schema_sha256': _sha(DATASET_SCHEMA),
            'fixed_delta_time_hex': float(fixed).hex(),
            'input_feature_name_sha256': input_names,
            'plugin_identity_sha256': plugin_identity,
            'raw_recorder_schema_sha256': _sha(RAW_RECORDER_SCHEMA),
            'state_channel_name_sha256': state_names,
            'tick_domain_sha256': _sha(TICK_DOMAIN),
        }
        encoded = json.dumps(
            schema_record, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True).encode('ascii')
        if hashlib.sha256(encoded).hexdigest() != schema_hash:
            raise SnapshotDiagnosticError('dataset schema hash does not verify')
        expected = set(required)
        records = []
        for index, (content, group) in enumerate(zip(contents, groups)):
            prefix = f'trace_{index:06d}'
            names = {
                f'{prefix}_ticks', f'{prefix}_snapshot_multiplicity',
                f'{prefix}_state', f'{prefix}_input',
                f'{prefix}_ticks_sha256', f'{prefix}_state_sha256',
                f'{prefix}_input_sha256',
                f'{prefix}_snapshot_multiplicity_sha256',
            }
            expected.update(names)
            if not names.issubset(dataset.files):
                raise SnapshotDiagnosticError('dataset trace arrays are incomplete')
            ticks = np.asarray(dataset[f'{prefix}_ticks'])
            snapshot_multiplicity = np.asarray(
                dataset[f'{prefix}_snapshot_multiplicity'])
            state = np.asarray(dataset[f'{prefix}_state'])
            inputs = np.asarray(dataset[f'{prefix}_input'])
            if (ticks.dtype != np.dtype('<i8') or ticks.ndim != 1
                    or len(ticks) < 4
                    or snapshot_multiplicity.dtype != np.dtype('<i8')
                    or snapshot_multiplicity.shape != ticks.shape
                    or np.any(snapshot_multiplicity < 1)
                    or state.dtype != np.dtype('<f8')
                    or state.shape != (len(ticks), len(STATE_CHANNELS))
                    or inputs.dtype != np.dtype('<f8')
                    or inputs.shape != (len(ticks) - 1, len(INPUT_FEATURES))
                    or not np.isfinite(state).all()
                    or not np.isfinite(inputs).all()
                    or np.any(np.diff(ticks) <= 0)):
                raise SnapshotDiagnosticError('dataset numeric arrays are invalid')
            hashes = {}
            for name, array in (
                    ('ticks', ticks),
                    ('snapshot_multiplicity', snapshot_multiplicity),
                    ('state', state), ('input', inputs)):
                stored = _decode_hash(
                    dataset[f'{prefix}_{name}_sha256'], f'{name} content hash')
                if _array_hash(array) != stored:
                    raise SnapshotDiagnosticError(
                        f'dataset {name} content hash does not verify')
                hashes[name] = stored
            if _trace_content_hash(
                    group, hashes['ticks'], hashes['snapshot_multiplicity'],
                    hashes['state'], hashes['input']) != content:
                raise SnapshotDiagnosticError('dataset trace content hash does not verify')
            records.append({
                'content': content, 'group': group,
                'ticks': ticks,
                'snapshot_multiplicity': snapshot_multiplicity,
                'state': state, 'input': inputs,
            })
        if set(dataset.files) != expected:
            raise SnapshotDiagnosticError('dataset contains an unexpected field')
        return {
            'schema': schema_hash,
            'state_names': state_names,
            'input_names': input_names,
            'records': records,
        }
    finally:
        dataset.close()


def _lag_rows(record, lag):
    transition = np.arange(len(record['input']))
    input_row = transition - lag
    keep = (input_row >= 0) & (input_row < len(record['input']))
    transition = transition[keep]
    input_row = input_row[keep]
    return (
        record['state'][transition],
        record['input'][input_row],
        record['state'][transition + 1],
    )


def _fit(features, target, ridge):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(features)), standardized))
    gram = design.T @ design
    gram[1:, 1:] += ridge * len(features) * np.eye(features.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ target)
    if not np.isfinite(coefficients).all():
        raise SnapshotDiagnosticError('ridge fit produced nonfinite values')
    return mean, scale, coefficients


def _predict(model, features):
    mean, scale, coefficients = model
    design = np.column_stack((
        np.ones(len(features)), (features - mean) / scale))
    return design @ coefficients


def _error_metrics(prediction, target):
    absolute = np.abs(prediction - target)
    return {
        'mae': float(absolute.mean()),
        'rmse': float(np.sqrt(np.square(absolute).mean())),
        'max_abs_error': float(absolute.max()),
    }


def _support(training, heldout, input_hashes):
    records = []
    for column, identity in enumerate(input_hashes):
        train = training[:, column]
        test = heldout[:, column]
        lower = float(train.min())
        upper = float(train.max())
        records.append({
            'input_name_sha256': identity,
            'train_nonzero_count': int(np.count_nonzero(train)),
            'train_unique_count': int(len(np.unique(train))),
            'train_min': lower,
            'train_max': upper,
            'train_mean': float(train.mean()),
            'train_std': float(train.std()),
            'heldout_below_train_count': int(np.count_nonzero(test < lower)),
            'heldout_above_train_count': int(np.count_nonzero(test > upper)),
            'heldout_within_train_fraction': float(
                np.count_nonzero((test >= lower) & (test <= upper)) / len(test)),
        })
    return records


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0.0 else None


def _write_json(path, record):
    created = False
    try:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            created = True
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def report_snapshot_baseline(dataset_path, output_path, *, ridge=1e-6):
    """Write leave-one-group-out AR versus ARX snapshot metrics."""
    if not _finite_number(ridge) or float(ridge) <= 0.0:
        raise SnapshotDiagnosticError('ridge must be finite and positive')
    output_path = _output_path(output_path, '.json')
    dataset = _load_dataset(dataset_path)
    records = dataset['records']
    heldout_reports = []
    for heldout_group in sorted({record['group'] for record in records}):
        training_records = [
            record for record in records if record['group'] != heldout_group]
        heldout_records = [
            record for record in records if record['group'] == heldout_group]
        for lag in LAG_CANDIDATES:
            rows = [_lag_rows(record, lag) for record in training_records]
            train_state = np.concatenate([row[0] for row in rows])
            train_input = np.concatenate([row[1] for row in rows])
            train_target = np.concatenate([row[2] for row in rows])
            ar_model = _fit(train_state, train_target, float(ridge))
            arx_model = _fit(
                np.column_stack((train_state, train_input)),
                train_target, float(ridge))
            centered_input = train_input - train_input.mean(axis=0)
            input_rank = int(np.linalg.matrix_rank(centered_input))
            for record in heldout_records:
                state, inputs, target = _lag_rows(record, lag)
                ar = _error_metrics(_predict(ar_model, state), target)
                arx = _error_metrics(_predict(
                    arx_model, np.column_stack((state, inputs))), target)
                entry = next((item for item in heldout_reports
                              if item['trace_content_sha256'] == record['content']), None)
                if entry is None:
                    entry = {
                        'trace_content_sha256': record['content'],
                        'group_identity_sha256': record['group'],
                        'training_group_identity_sha256': sorted({
                            item['group'] for item in training_records}),
                        'training_trace_content_sha256': sorted(
                            item['content'] for item in training_records),
                        'snapshot_event_count': sum(
                            int(value)
                            for value in record['snapshot_multiplicity']),
                        'unique_snapshot_tick_count': len(record['ticks']),
                        'coalesced_duplicate_count': sum(
                            int(value)
                            for value in record['snapshot_multiplicity'])
                            - len(record['ticks']),
                        'lag': [],
                    }
                    heldout_reports.append(entry)
                entry['lag'].append({
                    'lag_intervals': lag,
                    'negative_lag_placebo': lag < 0,
                    'train_row_count': len(train_state),
                    'heldout_row_count': len(state),
                    'input_dimensions': len(INPUT_FEATURES),
                    'input_rank': input_rank,
                    'input_support': _support(
                        train_input, inputs, dataset['input_names']),
                    'state_only_ar': ar,
                    'arx': arx,
                    'arx_minus_ar_rmse': arx['rmse'] - ar['rmse'],
                    'arx_to_ar_rmse_ratio': _safe_ratio(
                        arx['rmse'], ar['rmse']),
                })
    heldout_reports.sort(key=lambda item: item['trace_content_sha256'])
    report = {
        'diagnostic_only': True,
        'simulator_claim': False,
        'parity_claim': False,
        'action_alignment_verified': False,
        'schema_sha256': dataset['schema'],
        'state_channel_name_sha256': dataset['state_names'],
        'input_feature_name_sha256': dataset['input_names'],
        'trace_content_sha256': [record['content'] for record in records],
        'group_identity_sha256': [record['group'] for record in records],
        'metrics': {
            'trace_count': len(records),
            'group_count': len({record['group'] for record in records}),
            'snapshot_event_count': sum(
                sum(int(value) for value in record['snapshot_multiplicity'])
                for record in records),
            'unique_snapshot_tick_count': sum(
                len(record['ticks']) for record in records),
            'coalesced_duplicate_count': sum(
                sum(int(value) for value in record['snapshot_multiplicity'])
                - len(record['ticks']) for record in records),
            'lag_candidates': list(LAG_CANDIDATES),
            'ridge': float(ridge),
            'heldout_trace': heldout_reports,
        },
    }
    _write_json(output_path, report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('traces', nargs='+')
    export.add_argument('--inventory', required=True)
    export.add_argument('--out', required=True)
    report = commands.add_parser('report')
    report.add_argument('dataset')
    report.add_argument('--out', required=True)
    report.add_argument('--ridge', type=float, default=1e-6)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'export':
            result = export_snapshot_dataset(
                args.traces, args.inventory, args.out)
        else:
            result = report_snapshot_baseline(
                args.dataset, args.out, ridge=args.ridge)
    except (SnapshotDiagnosticError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
