"""Focused tests for the v2 snapshot-transition diagnostic."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import snapshot_transition_baseline as snapshot
from trace import TraceWriter


BUILD = 'a' * 64
OTHER_BUILD = 'b' * 64
RAW_HASH = '1' * 64
PLUGIN_HASH = '2' * 64
SECRET_ENDPOINT_A = 'secret-a.internal.invalid:7001'
SECRET_ENDPOINT_B = 'secret-b.internal.invalid:7002'
SECRET_SESSION_A = 'Arena-super-secret-alpha'
SECRET_SESSION_B = 'Arena-super-secret-beta'
SECRET_PATH = r'C:\private\captures\operator-session.jsonl'
SECRET_PROVENANCE = 'private provenance marker 5918273'
SNAPSHOT_TICKS = (0, 2, 4, 6, 8)


def v2_command_hash(velocities, discrete, local_slot=0):
    document = {
        'schema': snapshot.COMMAND_SCHEMA,
        'fixed_tick_commands': [
            {
                'client_fixed_tick': tick,
                'local_fighter_index': local_slot,
                'velocity_command': [float(value) for value in velocity],
            }
            for tick, velocity in enumerate(velocities)
        ],
        'discrete_transport_commands': discrete,
    }
    blob = json.dumps(
        document, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def write_inventory(path, build=BUILD):
    path.write_text(json.dumps({
        'schema': 4,
        'build_fingerprint': build,
        'errors': [],
        'install': SECRET_PATH,
        'files': [{'path': SECRET_PATH, 'sha256': '3' * 64}],
    }), encoding='utf-8')
    return path


def write_trace(path, *, endpoint=SECRET_ENDPOINT_A,
                session=SECRET_SESSION_A, offset=0.0, status=0.0,
                build=BUILD, source='rek', authority='server',
                fixed_dt=0.002, omit_root=None, omit_raw_hash=False,
                omit_plugin_hash=False, omit_command_hash=False,
                bad_command_hash=False, extra_method=None,
                malformed_estop=False, snapshot_ticks=SNAPSHOT_TICKS,
                snapshot_sequences=None):
    velocities = [
        [((tick % 3) - 1) * 0.25, 0.0, (tick % 2) * 0.5]
        for tick in range(9)
    ]
    transport = [
        ('SendVelocityCommand', 1, [0.25, 0.0, 0.5]),
        ('SendVelocityCommand', 3, [-0.25, 0.0, 0.0]),
        ('SendMoveEvent', 3, 0),
        ('SendVelocityCommand', 5, [0.0, 0.0, 0.5]),
        ('SendSpecialEvent', 5, 2),
        ('SendEStopToggle', 6, None),
        ('SendVelocityCommand', 7, [0.25, 0.0, 0.0]),
    ]
    if extra_method:
        transport.append((extra_method, 7, None))
    discrete = []
    for method, tick, value in transport:
        if method == 'SendMoveEvent':
            discrete.append({
                'client_fixed_tick': tick,
                'method': method,
                'pending_move_index': value,
            })
        elif method == 'SendSpecialEvent':
            discrete.append({
                'client_fixed_tick': tick,
                'method': method,
                'pending_special_command': value,
            })
        elif method == 'SendEStopToggle':
            discrete.append({'client_fixed_tick': tick, 'method': method})
    command_hash = v2_command_hash(velocities, discrete)
    if bad_command_hash:
        command_hash = 'f' * 64

    state_channels = [
        name for name in snapshot.STATE_CHANNELS if name != omit_root]
    status_channels = [
        'cmd.0.active', 'cmd.0.network_initialized', 'cmd.0.pending_move',
        'cmd.0.pending_special', 'cmd.0.punching', 'cmd.0.recovering',
    ]
    channels = [
        *state_channels,
        'cmd.0.velocity.x', 'cmd.0.velocity.y', 'cmd.0.velocity.z',
        *status_channels,
        'round.state',
    ]
    provenance = {
        name: {'kind': 'class', 'ref': SECRET_PROVENANCE}
        for name in channels
    }
    metadata = {
        'tick_domain': snapshot.TICK_DOMAIN,
        'fixed_delta_time': fixed_dt,
        'raw_recorder_schema': snapshot.RAW_RECORDER_SCHEMA,
        'command_sequence_schema': snapshot.COMMAND_SCHEMA,
        'server_tick_available': False,
    }
    if not omit_raw_hash:
        metadata['raw_recorder_sha256'] = RAW_HASH
    if not omit_plugin_hash:
        metadata['recorder_plugin_sha256'] = PLUGIN_HASH
    if not omit_command_hash:
        metadata['command_sequence_sha256'] = command_hash
    with TraceWriter(
            path, channels, build, source, authority=authority,
            server={'endpoint': endpoint, 'session_id': session},
            provenance=provenance, **metadata) as writer:
        for tick, velocity in enumerate(velocities):
            values = {
                name: offset + tick * (channel + 1) * 0.01
                for channel, name in enumerate(state_channels)
            }
            values.update({
                'cmd.0.velocity.x': velocity[0],
                'cmd.0.velocity.y': velocity[1],
                'cmd.0.velocity.z': velocity[2],
                'round.state': 2.0,
            })
            values.update({name: status + tick for name in status_channels})
            writer.append(tick, values)
        if snapshot_sequences is None:
            snapshot_sequences = range(1, len(snapshot_ticks) + 1)
        for sequence, tick in zip(snapshot_sequences, snapshot_ticks):
            writer.event(tick, 'server_snapshot_rx', sequence=sequence)
        method_counts = {}
        for sequence, (method, tick, value) in enumerate(transport, 1):
            method_counts[method] = method_counts.get(method, 0) + 1
            if method == 'SendVelocityCommand':
                payload = {'velocity_command': value, 'active': bool(status)}
            elif method == 'SendMoveEvent':
                payload = {'pending_move_index': value, 'pending_move': bool(status)}
            elif method == 'SendSpecialEvent':
                payload = {
                    'pending_special_command': value,
                    'pending_special': bool(status),
                }
            elif method == 'SendEStopToggle':
                payload = 'malformed' if malformed_estop else {'active': bool(status)}
            else:
                payload = {'active': bool(status)}
            writer.event(
                tick, 'command_transport_invoked', sequence=sequence,
                method=method, method_sequence=method_counts[method],
                client_fixed_tick_at_observation=tick,
                unity_frame=100 + sequence,
                unity_unscaled_time=tick * fixed_dt,
                provenance=f'REKApp.RobotInputController.{method} prefix',
                input=payload)
    return path


class SnapshotTransitionBaselineTests(unittest.TestCase):
    def test_sanitized_grouped_export_and_ar_arx_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = write_inventory(root / 'inventory-secret.json')
            traces = [
                write_trace(
                    root / 'private-alpha-one.trace', offset=0.0, status=0.0,
                    snapshot_ticks=(0, 2, 2, 4, 6, 8)),
                write_trace(
                    root / 'private-alpha-two.trace', offset=1.0, status=100.0),
                write_trace(
                    root / 'private-beta.trace', endpoint=SECRET_ENDPOINT_B,
                    session=SECRET_SESSION_B, offset=2.0, status=-100.0),
            ]
            dataset_path = root / 'snapshot-dataset.npz'
            summary = snapshot.export_snapshot_dataset(
                traces, inventory, dataset_path)

            self.assertTrue(summary['diagnostic_only'])
            self.assertFalse(summary['simulator_claim'])
            self.assertFalse(summary['parity_claim'])
            self.assertFalse(summary['action_alignment_verified'])
            self.assertEqual(summary['metrics']['trace_count'], 3)
            self.assertEqual(summary['metrics']['group_count'], 2)
            self.assertEqual(summary['metrics']['state_dimensions'], 18)
            self.assertEqual(summary['metrics']['input_dimensions'], 9)
            self.assertEqual(summary['metrics']['snapshot_event_count'], 16)
            self.assertEqual(summary['metrics']['unique_snapshot_tick_count'], 15)
            self.assertEqual(summary['metrics']['coalesced_duplicate_count'], 1)
            self.assertEqual(summary['metrics']['transition_count'], 12)

            status_names = (
                'cmd.0.active', 'cmd.0.network_initialized',
                'cmd.0.pending_move', 'cmd.0.pending_special',
                'cmd.0.punching', 'cmd.0.recovering')
            with np.load(dataset_path, allow_pickle=False) as dataset:
                state_hashes = {
                    bytes(item).decode('ascii')
                    for item in dataset['state_channel_name_sha256']
                }
                input_hashes = {
                    bytes(item).decode('ascii')
                    for item in dataset['input_feature_name_sha256']
                }
                self.assertEqual(
                    state_hashes, {snapshot._sha(name) for name in snapshot.STATE_CHANNELS})
                self.assertTrue(all(
                    snapshot._sha(name) not in input_hashes for name in status_names))
                self.assertEqual(
                    input_hashes,
                    {snapshot._sha(name) for name in snapshot.INPUT_FEATURES})
                inputs = [dataset[key] for key in dataset.files
                          if key.endswith('_input')]
                multiplicities = [
                    dataset[key] for key in dataset.files
                    if key.endswith('_snapshot_multiplicity')]
                self.assertEqual(
                    sorted(tuple(int(value) for value in item)
                           for item in multiplicities),
                    [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1),
                     (1, 2, 1, 1, 1)])
                self.assertTrue(all(float(values[:, 8].sum()) == 1.0
                                    for values in inputs))
                self.assertTrue(all(float(values[:, 4].sum()) == 0.0
                                    and float(values[:, 5].sum()) == 1.0
                                    and float(values[:, 6].sum()) == 2.0
                                    and float(values[:, 7].sum()) == 1.0
                                    for values in inputs))
                np.testing.assert_array_equal(inputs[0], inputs[1])
                np.testing.assert_array_equal(inputs[1], inputs[2])

            raw_dataset = dataset_path.read_bytes()
            for secret in (
                    SECRET_ENDPOINT_A, SECRET_ENDPOINT_B,
                    SECRET_SESSION_A, SECRET_SESSION_B, SECRET_PATH,
                    SECRET_PROVENANCE, 'private-alpha-one.trace',
                    RAW_HASH, PLUGIN_HASH, 'round.state'):
                self.assertNotIn(secret.encode('utf-8'), raw_dataset)

            report_path = root / 'snapshot-report.json'
            report = snapshot.report_snapshot_baseline(dataset_path, report_path)
            self.assertTrue(report['diagnostic_only'])
            self.assertFalse(report['simulator_claim'])
            self.assertFalse(report['parity_claim'])
            self.assertFalse(report['action_alignment_verified'])
            self.assertEqual(report['metrics']['lag_candidates'], [-1, 0, 1, 2])
            self.assertEqual(report['metrics']['snapshot_event_count'], 16)
            self.assertEqual(report['metrics']['unique_snapshot_tick_count'], 15)
            self.assertEqual(report['metrics']['coalesced_duplicate_count'], 1)
            self.assertEqual(len(report['metrics']['heldout_trace']), 3)
            self.assertEqual(
                sorted((item['snapshot_event_count'],
                        item['unique_snapshot_tick_count'],
                        item['coalesced_duplicate_count'])
                       for item in report['metrics']['heldout_trace']),
                [(5, 5, 0), (5, 5, 0), (6, 5, 1)])
            for heldout in report['metrics']['heldout_trace']:
                self.assertEqual(len(heldout['lag']), 4)
                self.assertNotIn(
                    heldout['group_identity_sha256'],
                    heldout['training_group_identity_sha256'])
                self.assertTrue(heldout['lag'][0]['negative_lag_placebo'])
                for lag in heldout['lag']:
                    self.assertIn('state_only_ar', lag)
                    self.assertIn('arx', lag)
                    self.assertIn('input_rank', lag)
                    self.assertEqual(len(lag['input_support']), 9)
            raw_report = report_path.read_bytes()
            for secret in (
                    SECRET_ENDPOINT_A, SECRET_ENDPOINT_B,
                    SECRET_SESSION_A, SECRET_SESSION_B, SECRET_PATH,
                    SECRET_PROVENANCE, 'private-alpha-one.trace',
                    RAW_HASH, PLUGIN_HASH, 'round.state'):
                self.assertNotIn(secret.encode('utf-8'), raw_report)

    def test_content_hashes_reject_numeric_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = write_inventory(root / 'inventory.json')
            traces = [
                write_trace(root / 'first.trace'),
                write_trace(
                    root / 'second.trace', endpoint=SECRET_ENDPOINT_B,
                    session=SECRET_SESSION_B, offset=1.0),
            ]
            dataset_path = root / 'dataset.npz'
            snapshot.export_snapshot_dataset(traces, inventory, dataset_path)
            with np.load(dataset_path, allow_pickle=False) as dataset:
                copied = {key: np.array(dataset[key], copy=True)
                          for key in dataset.files}
            for name, array_name, message in (
                    ('state', 'trace_000000_state',
                     'state content hash does not verify'),
                    ('multiplicity', 'trace_000000_snapshot_multiplicity',
                     'snapshot_multiplicity content hash does not verify')):
                with self.subTest(name=name):
                    changed = {key: np.array(value, copy=True)
                               for key, value in copied.items()}
                    if name == 'state':
                        changed[array_name][0, 0] += 1.0
                    else:
                        changed[array_name][0] += 1
                    tampered = root / f'tampered-{name}.npz'
                    np.savez_compressed(tampered, **changed)
                    report = root / f'tampered-{name}-report.json'
                    with self.assertRaisesRegex(
                            snapshot.SnapshotDiagnosticError, message):
                        snapshot.report_snapshot_baseline(tampered, report)
                    self.assertFalse(report.exists())

    def test_snapshot_ticks_coalesce_but_decreases_and_gaps_fail_closed(self):
        cases = (
            ('decreasing',
             {'snapshot_ticks': (0, 4, 2, 6, 8)},
             'snapshot ticks are decreasing'),
            ('sequence-gap',
             {'snapshot_sequences': (1, 2, 4, 5, 6)},
             'invalid snapshot sequence'),
        )
        for name, options, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = write_inventory(root / 'inventory.json')
                first = write_trace(root / 'first.trace', **options)
                second = write_trace(
                    root / 'second.trace', endpoint=SECRET_ENDPOINT_B,
                    session=SECRET_SESSION_B, offset=1.0)
                with self.assertRaisesRegex(snapshot.SnapshotDiagnosticError, message):
                    snapshot.export_snapshot_dataset(
                        [first, second], inventory, root / 'dataset.npz')
                self.assertFalse((root / 'dataset.npz').exists())

    def test_source_identity_command_hash_and_groups_fail_closed(self):
        cases = (
            ('inventory-build', {}, {}, OTHER_BUILD, 'inventory build'),
            ('source', {'source': 'clone:test'}, {}, BUILD, 'source is not rek'),
            ('authority', {'authority': 'local'}, {}, BUILD, 'server-authoritative'),
            ('endpoint', {'endpoint': ' '}, {}, BUILD, 'grouping identity'),
            ('raw-hash', {'omit_raw_hash': True}, {}, BUILD, 'raw_recorder_sha256'),
            ('plugin-hash', {'omit_plugin_hash': True}, {}, BUILD,
             'recorder_plugin_sha256'),
            ('command-field', {'omit_command_hash': True}, {}, BUILD,
             'command_sequence_sha256'),
            ('command-hash', {'bad_command_hash': True}, {}, BUILD, 'does not verify'),
            ('one-group', {}, {}, BUILD, 'two endpoint/session groups'),
        )
        for name, first_options, second_options, inventory_build, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = write_inventory(root / 'inventory.json', inventory_build)
                first = write_trace(root / 'first.trace', **first_options)
                if name == 'one-group':
                    second = write_trace(root / 'second.trace', offset=1.0)
                else:
                    second = write_trace(
                        root / 'second.trace', endpoint=SECRET_ENDPOINT_B,
                        session=SECRET_SESSION_B, offset=1.0,
                        **second_options)
                with self.assertRaisesRegex(snapshot.SnapshotDiagnosticError, message):
                    snapshot.export_snapshot_dataset(
                        [first, second], inventory, root / 'dataset.npz')
                self.assertFalse((root / 'dataset.npz').exists())

    def test_estop_and_unknown_transport_fail_closed(self):
        for name, options, message in (
                ('malformed-estop', {'malformed_estop': True}, 'EStop.*malformed'),
                ('unknown', {'extra_method': 'SendMysteryCommand'}, 'unknown transport')):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = write_inventory(root / 'inventory.json')
                first = write_trace(root / 'first.trace', **options)
                second = write_trace(
                    root / 'second.trace', endpoint=SECRET_ENDPOINT_B,
                    session=SECRET_SESSION_B, offset=1.0)
                with self.assertRaisesRegex(snapshot.SnapshotDiagnosticError, message):
                    snapshot.export_snapshot_dataset(
                        [first, second], inventory, root / 'dataset.npz')

    def test_create_only_protected_names_and_exact_root_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = write_inventory(root / 'inventory.json')
            traces = [
                write_trace(root / 'first.trace'),
                write_trace(
                    root / 'second.trace', endpoint=SECRET_ENDPOINT_B,
                    session=SECRET_SESSION_B, offset=1.0),
            ]
            for name in ('envelope.json', 'parity_report.json'):
                with self.assertRaisesRegex(
                        snapshot.SnapshotDiagnosticError, 'protected canonical'):
                    snapshot.export_snapshot_dataset(traces, inventory, root / name)
                self.assertFalse((root / name).exists())
            dataset = root / 'dataset.npz'
            snapshot.export_snapshot_dataset(traces, inventory, dataset)
            original = dataset.read_bytes()
            with self.assertRaises(FileExistsError):
                snapshot.export_snapshot_dataset(traces, inventory, dataset)
            self.assertEqual(dataset.read_bytes(), original)
            with self.assertRaisesRegex(
                    snapshot.SnapshotDiagnosticError, 'root-state channel'):
                snapshot.export_snapshot_dataset([
                    write_trace(root / 'missing.trace', omit_root='root.1.angvel.z'),
                    traces[1],
                ], inventory, root / 'missing.npz')
            for name in ('envelope.json', 'parity_report.json'):
                with self.assertRaisesRegex(
                        snapshot.SnapshotDiagnosticError, 'protected canonical'):
                    snapshot.report_snapshot_baseline(dataset, root / name)
                self.assertFalse((root / name).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
