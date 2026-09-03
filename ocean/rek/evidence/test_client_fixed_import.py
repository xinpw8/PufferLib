"""Focused tests for importing private-AI client-fixed recorder output."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import client_fixed_import
import check_artifacts
import differ
from trace import Trace, TraceWriter


def inventory_record():
    return {
        'schema': 4,
        'build_fingerprint': 'f' * 64,
        'steam': {'buildid': '24969755'},
        'errors': [],
        'files': [
            {
                'path': 'GameAssembly.dll',
                'sha256': client_fixed_import.EXPECTED_GAME_ASSEMBLY_SHA256,
            },
            {
                'path': 'REK_Data/il2cpp_data/Metadata/global-metadata.dat',
                'sha256': client_fixed_import.EXPECTED_METADATA_SHA256,
            },
        ],
    }


def robot(slot, tick):
    return {
        'visual_only': True,
        'player_controlled': slot == 0,
        'falling': False,
        'fallen': False,
        'dampened': False,
        'resetting': False,
        'motor_shutdown': False,
        'tilt_angle': 0.1 * tick,
        'floor_contact_count': 2,
        'root_position': [float(slot + tick), 1.0, 2.0],
        'root_rotation': [0.0, 0.0, 0.0, 1.0],
        'root_linear_velocity': [0.1, 0.0, 0.0],
        'root_angular_velocity': [None if tick == 1 and slot == 0 else 0.0, 0.0, 0.0],
        'bones': {
            'count': 1,
            'world_positions_xyz': [
                None if tick == 1 and slot == 0 else float(slot + tick),
                1.0,
                2.0,
            ],
            'world_rotations_xyzw': [0.0, 0.0, 0.0, 1.0],
            'local_positions_xyz': [0.0, 1.0, 0.0],
            'local_rotations_xyzw': [0.0, 0.0, 0.0, 1.0],
        },
    }


def sample(tick):
    return {
        'event': 'sample',
        'sample_index': tick,
        'client_fixed_tick': tick,
        'unity_frame': 100 + tick,
        'unity_time': tick * 0.02,
        'unity_fixed_time': tick * 0.02,
        'unity_unscaled_time': tick * 0.02,
        'fight_epoch': 7,
        'phase': 'Active',
        'phase_value': 2,
        'local_fighter_index': 0,
        'opponent_slot': 1,
        'sparring_bot_number': 1,
        'client_ai_difficulty': 1,
        'transport_observation': {
            'client_send_frame_sequence': 0,
            'client_transport_invocation_sequence': tick,
            'fight_state_snapshot_sequence': tick // 2,
            'server_tick': None,
        },
        'input': {
            'network_index': 0,
            'network_initialized': True,
            'active': True,
            'punching': False,
            'recovering': False,
            'velocity_command': [0.25, 0.0, -0.5],
            'pending_move': False,
            'pending_move_index': None,
            'pending_special': False,
            'pending_special_command': None,
            'action_playing': False,
            'action_clip': 'fixture_clip',
            'action_clip_frame': 0.0,
            'action_clip_fps': 60.0,
        },
        'round': {
            'number': 1,
            'duration': 60.0,
            'time_remaining': 60.0 - tick * 0.02,
            'active': True,
            'redo': False,
            'clean_hits': [tick, 0],
            'falls': [0, 0],
            'result': 'None',
            'result_value': 0,
            'winner_index': -1,
            'knockout': False,
        },
        'fight': {
            'format': 'BestOfThree',
            'format_value': 1,
            'current_round': 1,
            'rounds_won': [0, 0],
            'result': 'None',
            'result_value': 0,
            'winner_index': -1,
        },
        'fighter_0': robot(0, tick),
        'fighter_1': robot(1, tick),
    }


def records(session_id='arena-test-session'):
    return [
        {
            'event': 'capture_start',
            'schema': client_fixed_import.SCHEMA,
            'tick_level_claim': True,
            'tick_domain': 'client_fixed_update',
            'fixed_delta_time': 0.02,
            'server_tick_available': False,
            'server_tick_reason': 'test fixture has no server tick',
            'game_assembly_sha256': client_fixed_import.EXPECTED_GAME_ASSEMBLY_SHA256,
            'global_metadata_sha256': client_fixed_import.EXPECTED_METADATA_SHA256,
            'plugin_sha256': 'a' * 64,
            'harmony_target_status': {
                'REKApp.RobotInputController.SendVelocityCommand': True,
                'REKApp.RobotInputController.SendMoveEvent': True,
                'REKApp.RobotInputController.SendSpecialEvent': True,
                'REKApp.RobotInputController.SendEStopToggle': True,
                'REKApp.FightCoordinator.ApplyFightStateSnapshot': True,
            },
            'server': {
                'endpoint': 'test.invalid:7000',
                'session_id': session_id,
                'protocol': 'Unity.Netcode.Transports.UTP.UnityTransport',
                'endpoint_provenance': 'REKApp.NetworkSession.serverAddress+port',
                'session_id_provenance': 'REKApp.GameContext.ArenaID',
            },
            'scope': {
                'allowed': True,
                'network_connected': True,
                'network_is_client': True,
                'network_is_server': False,
                'local_fighter_index': 0,
                'opponent_slot': 1,
                'opponent_is_ai': True,
                'opponent_slot_is_ai': True,
                'human_in_opponent_slot': False,
                'opponent_slot_has_client': False,
                'opponent_human_bit_set': False,
                'fighter_0_visual_only': True,
                'fighter_1_visual_only': True,
                'sparring_bot_number': 1,
            },
            'fighter_0_bones': ['pelvis'],
            'fighter_1_bones': ['pelvis'],
        },
        sample(0),
        {
            'event': 'client_transport_method_invoked',
            'client_transport_invocation_sequence': 1,
            'method_invocation_sequence': 1,
            'client_fixed_tick_at_observation': 1,
            'unity_frame': 101,
            'unity_unscaled_time': 0.02,
            'method': 'SendVelocityCommand',
            'input': sample(1)['input'],
            'provenance': (
                'REKApp.RobotInputController.SendVelocityCommand prefix'),
        },
        sample(1),
        {
            'event': 'capture_end',
            'reason': 'scope_exit:round_not_active',
            'client_fixed_tick_at_end': 2,
            'unity_fixed_time_at_end': 0.04,
            'sample_count': 2,
            'capture_error_count': 0,
            'client_send_frame_count': 0,
            'client_transport_invocation_count': 1,
            'client_transport_method_counts': {
                'SendVelocityCommand': 1,
            },
        },
    ]


class ClientFixedImportTests(unittest.TestCase):
    def write_fixture(self, root, fixture_records):
        raw = root / 'capture.jsonl'
        raw.write_text(
            ''.join(json.dumps(record) + '\n' for record in fixture_records),
            encoding='utf-8')
        inventory = root / 'inventory.json'
        inventory.write_text(json.dumps(inventory_record()), encoding='utf-8')
        return raw, inventory

    def test_converts_measured_client_ticks_with_server_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, inventory = self.write_fixture(root, records())
            output = root / 'rek-001.trace'
            result = client_fixed_import.convert(raw, inventory, output)
            trace = Trace.load(output)

            self.assertEqual(trace.ticks, [0, 1])
            self.assertEqual(trace.authority, 'server')
            self.assertEqual(trace.server['session_id'], 'arena-test-session')
            self.assertEqual(trace.header['tick_domain'], 'client_fixed_update')
            self.assertFalse(trace.header['server_tick_available'])
            self.assertIn('root.0.pos.x', trace.channels)
            self.assertIn('root.0.vel.x', trace.channels)
            self.assertIn('root.0.angvel.y', trace.channels)
            self.assertIn('robot.0.visual_only', trace.channels)
            self.assertIn('robot.0.fallen', trace.channels)
            self.assertIn('contact.0.floor_contact_count', trace.channels)
            self.assertIn('seq.client_transport_invoke', trace.channels)
            self.assertNotIn('seq.client_send', trace.channels)
            self.assertIn('joint.1.00_pelvis.world.quat.w', trace.channels)
            self.assertNotIn('root.0.angvel.x', trace.channels)
            self.assertNotIn('joint.0.00_pelvis.world.pos.x', trace.channels)
            self.assertNotIn('cmd.0.pending_move_index', trace.channels)
            self.assertNotIn('cmd.0.pending_special_command', trace.channels)
            self.assertEqual(trace.channels['root.0.vel.x'], [0.1, 0.1])
            self.assertEqual(trace.channels['robot.0.fallen'], [0.0, 0.0])
            self.assertEqual(
                trace.channels['contact.0.floor_contact_count'], [2.0, 2.0])
            self.assertEqual(
                trace.provenance['root.0.vel.x']['raw_field'],
                'fighter_0.root_linear_velocity[0]')
            self.assertEqual(
                trace.provenance['robot.0.fallen']['raw_field'],
                'fighter_0.fallen')
            self.assertEqual(
                trace.provenance['contact.0.floor_contact_count']['raw_field'],
                'fighter_0.floor_contact_count')
            self.assertEqual(
                set(trace.header['channels']), set(trace.header['provenance']))
            self.assertIn('round_start', {event['kind'] for event in trace.events})
            self.assertIn('round_end', {event['kind'] for event in trace.events})
            self.assertIn(
                'command_transport_invoked',
                {event['kind'] for event in trace.events})
            transport_event = next(
                event for event in trace.events
                if event['kind'] == 'command_transport_invoked')
            self.assertEqual(
                transport_event['client_fixed_tick_at_observation'], 1)
            self.assertEqual(transport_event['unity_frame'], 101)
            self.assertEqual(transport_event['unity_unscaled_time'], 0.02)
            self.assertEqual(
                transport_event['provenance'],
                'REKApp.RobotInputController.SendVelocityCommand prefix')
            self.assertEqual(transport_event['input']['network_index'], 0)
            self.assertFalse(transport_event['input']['pending_move'])
            self.assertFalse(transport_event['input']['pending_special'])
            self.assertNotIn('pending_move_index', transport_event['input'])
            self.assertNotIn('pending_special_command', transport_event['input'])
            self.assertEqual(
                transport_event['input']['velocity_command'], [0.25, 0.0, -0.5])
            self.assertNotIn('action_clip', transport_event['input'])
            self.assertEqual(
                transport_event['excluded_input_fields'], ['action_clip'])
            self.assertNotIn(b'fixture_clip', output.read_bytes())
            self.assertEqual(result['ticks'], 2)
            self.assertTrue(result['server_session_present'])
            self.assertEqual(
                result['raw_recorder_schema'], client_fixed_import.SCHEMA_V3)
            self.assertEqual(
                trace.header['command_sequence_schema'],
                client_fixed_import.COMMAND_SEQUENCE_SCHEMA)
            self.assertEqual(
                result['command_observation'],
                'RecorderBehaviour.FixedUpdate measured velocity command plus '
                'discrete RobotInputController transport method prefixes')
            self.assertEqual(
                result['transport_observation'],
                'RobotInputController concrete transport method prefixes')

    def test_fixed_tick_window_hashes_the_command_schedule_not_transport_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = []
            for name, transport_tick, trailing_velocity in (
                    ('first', 0, [1.0, 0.0, 0.0]),
                    ('second', 1, [-1.0, 0.0, 0.0])):
                fixture = records(session_id=f'arena-{name}')
                fixture[2]['client_fixed_tick_at_observation'] = transport_tick
                trailing = sample(2)
                trailing['input']['velocity_command'] = trailing_velocity
                fixture.insert(-1, trailing)
                fixture[-1]['sample_count'] = 3
                fixture[-1]['client_fixed_tick_at_end'] = 3
                run = root / name
                run.mkdir()
                raw, inventory = self.write_fixture(run, fixture)
                output = run / 'window.trace'
                result = client_fixed_import.convert(
                    raw, inventory, output, tick_limit=2)
                trace = Trace.load(output)
                hashes.append(result['command_sequence_sha256'])
                self.assertEqual(trace.ticks, [0, 1])
                self.assertEqual(trace.header['observation_window_ticks'], 2)
                self.assertEqual(trace.header['raw_sample_count'], 3)
                self.assertFalse(trace.header['complete_round'])
                self.assertTrue(all(event['tick'] <= 1 for event in trace.events))
            self.assertEqual(hashes[0], hashes[1])

    def test_command_hash_includes_discrete_move_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = []
            for name, include_move in (('idle', False), ('move', True)):
                fixture = records(session_id=f'arena-{name}')
                if include_move:
                    move_input = sample(1)['input']
                    move_input['pending_move'] = True
                    move_input['pending_move_index'] = 0
                    fixture.insert(-1, {
                        'event': 'client_transport_method_invoked',
                        'client_transport_invocation_sequence': 2,
                        'method_invocation_sequence': 1,
                        'client_fixed_tick_at_observation': 1,
                        'unity_frame': 102,
                        'unity_unscaled_time': 0.021,
                        'method': 'SendMoveEvent',
                        'input': move_input,
                        'provenance': (
                            'REKApp.RobotInputController.SendMoveEvent prefix'),
                    })
                    fixture[-1]['client_transport_invocation_count'] = 2
                    fixture[-1]['client_transport_method_counts'] = {
                        'SendVelocityCommand': 1,
                        'SendMoveEvent': 1,
                    }
                run = root / name
                run.mkdir()
                raw, inventory = self.write_fixture(run, fixture)
                output = run / 'window.trace'
                hashes.append(client_fixed_import.convert(
                    raw, inventory, output, tick_limit=2)[
                        'command_sequence_sha256'])
            self.assertNotEqual(hashes[0], hashes[1])

    def test_maps_move_and_special_zero_indices_on_transport_events_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = records()

            move_input = sample(1)['input']
            move_input['pending_move'] = True
            move_input['pending_move_index'] = 0
            fixture.insert(-1, {
                'event': 'client_transport_method_invoked',
                'client_transport_invocation_sequence': 2,
                'method_invocation_sequence': 1,
                'client_fixed_tick_at_observation': 1,
                'unity_frame': 102,
                'unity_unscaled_time': 0.021,
                'method': 'SendMoveEvent',
                'input': move_input,
                'provenance': (
                    'REKApp.RobotInputController.SendMoveEvent prefix'),
            })

            special_input = sample(1)['input']
            special_input['pending_special'] = True
            special_input['pending_special_command'] = 0
            fixture.insert(-1, {
                'event': 'client_transport_method_invoked',
                'client_transport_invocation_sequence': 3,
                'method_invocation_sequence': 1,
                'client_fixed_tick_at_observation': 1,
                'unity_frame': 103,
                'unity_unscaled_time': 0.022,
                'method': 'SendSpecialEvent',
                'input': special_input,
                'provenance': (
                    'REKApp.RobotInputController.SendSpecialEvent prefix'),
            })
            fixture[-1]['client_transport_invocation_count'] = 3
            fixture[-1]['client_transport_method_counts'] = {
                'SendVelocityCommand': 1,
                'SendMoveEvent': 1,
                'SendSpecialEvent': 1,
            }

            raw, inventory = self.write_fixture(root, fixture)
            output = root / 'rek-transport-payloads.trace'
            client_fixed_import.convert(raw, inventory, output)
            trace = Trace.load(output)
            transport_events = {
                event['method']: event
                for event in trace.events
                if event['kind'] == 'command_transport_invoked'
            }

            self.assertEqual(
                transport_events['SendMoveEvent']['input']['pending_move_index'],
                0)
            self.assertTrue(
                transport_events['SendMoveEvent']['input']['pending_move'])
            self.assertEqual(
                transport_events['SendSpecialEvent']['input'][
                    'pending_special_command'],
                0)
            self.assertTrue(
                transport_events['SendSpecialEvent']['input']['pending_special'])
            self.assertNotIn('cmd.0.pending_move_index', trace.channels)
            self.assertNotIn('cmd.0.pending_special_command', trace.channels)

    def test_new_channels_require_present_type_valid_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = records()
            samples = [record for record in fixture
                       if record.get('event') == 'sample']
            samples[1]['fighter_0']['falling'] = None
            samples[1]['fighter_0']['fallen'] = 'false'
            del samples[1]['fighter_0']['floor_contact_count']
            samples[1]['fighter_0']['root_linear_velocity'][0] = False
            samples[1]['input']['pending_move'] = 0
            for measured_sample in samples:
                measured_sample['fighter_1']['floor_contact_count'] = 0

            raw, inventory = self.write_fixture(root, fixture)
            output = root / 'rek-missing-fields.trace'
            client_fixed_import.convert(raw, inventory, output)
            trace = Trace.load(output)

            self.assertNotIn('robot.0.falling', trace.channels)
            self.assertNotIn('robot.0.fallen', trace.channels)
            self.assertNotIn('contact.0.floor_contact_count', trace.channels)
            self.assertNotIn('root.0.vel.x', trace.channels)
            self.assertNotIn('cmd.0.pending_move', trace.channels)
            self.assertEqual(trace.channels['cmd.0.velocity.x'], [0.25, 0.25])
            self.assertEqual(trace.channels['robot.0.resetting'], [0.0, 0.0])
            self.assertEqual(
                trace.channels['contact.1.floor_contact_count'], [0.0, 0.0])
            self.assertEqual(trace.channels['root.0.vel.y'], [0.0, 0.0])

    def test_command_identity_rejects_malformed_velocity_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = records()
            samples = [record for record in fixture
                       if record.get('event') == 'sample']
            samples[1]['input']['velocity_command'][0] = False
            raw, inventory = self.write_fixture(root, fixture)
            with self.assertRaisesRegex(
                    ValueError, 'no measured three-axis velocity command'):
                client_fixed_import.convert(
                    raw, inventory, root / 'malformed-command.trace')

    def test_transport_input_sanitization_preserves_zero_and_false(self):
        payload, excluded = client_fixed_import._transport_event_input({
            'input': {
                'pending_move': False,
                'pending_move_index': 0,
                'pending_special': None,
                'pending_special_command': '0',
                'velocity_command': [0.0, False, 0.0],
                'action_clip': 'fixture_clip',
                'action_clip_frame': float('nan'),
            },
        })

        self.assertFalse(payload['pending_move'])
        self.assertEqual(payload['pending_move_index'], 0)
        self.assertNotIn('pending_special', payload)
        self.assertNotIn('pending_special_command', payload)
        self.assertNotIn('velocity_command', payload)
        self.assertNotIn('action_clip', payload)
        self.assertNotIn('action_clip_frame', payload)
        self.assertEqual(excluded, [
            'action_clip',
            'action_clip_frame',
            'pending_special_command',
            'velocity_command',
        ])

    def test_transport_event_rejects_missing_time_or_callback_provenance(self):
        for name, mutate in (
                ('absent', lambda event: event.pop('unity_frame')),
                ('null', lambda event: event.update(
                    unity_unscaled_time=None)),
                ('malformed', lambda event: event.update(
                    provenance='unverified callback'))):
            with self.subTest(name=name):
                fixture = records()
                transport = next(
                    record for record in fixture
                    if record.get('event') ==
                    'client_transport_method_invoked')
                mutate(transport)
                samples = [record for record in fixture
                           if record.get('event') == 'sample']
                with self.assertRaisesRegex(
                        ValueError, 'no measured Unity frame/time'):
                    client_fixed_import._events(
                        fixture, samples, client_fixed_import.SCHEMA_V3,
                        complete_round=True)

    def test_refuses_server_authority_without_arena_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, inventory = self.write_fixture(root, records(session_id=None))
            output = root / 'rek-001.trace'
            with self.assertRaisesRegex(ValueError, 'no endpoint or ArenaID'):
                client_fixed_import.convert(raw, inventory, output)
            self.assertFalse(output.exists())

    def test_refuses_unfinalized_partial_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, inventory = self.write_fixture(root, records())
            partial = raw.with_name(raw.name + '.partial')
            raw.rename(partial)
            with self.assertRaisesRegex(ValueError, 'unfinalized'):
                client_fixed_import.convert(
                    partial, inventory, root / 'rek-001.trace')

    def test_refuses_v3_without_concrete_transport_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = records()
            fixture = [record for record in fixture
                       if record.get('event') !=
                       'client_transport_method_invoked']
            fixture[-1]['client_transport_invocation_count'] = 0
            fixture[-1]['client_transport_method_counts'] = {}
            raw, inventory = self.write_fixture(root, fixture)
            with self.assertRaisesRegex(
                    ValueError, 'no observed concrete client transport'):
                client_fixed_import.convert(
                    raw, inventory, root / 'rek-001.trace')

    def write_small_trace(self, path, command_sequence):
        with TraceWriter(
                path, ['root.0.pos.x'], 'f' * 64, 'rek', authority='local',
                provenance={'root.0.pos.x': {
                    'kind': 'class', 'ref': 'test measured field'}},
                command_sequence_sha256=command_sequence,
                command_sequence_schema=(
                    client_fixed_import.COMMAND_SEQUENCE_SCHEMA)) as writer:
            writer.append(0, {'root.0.pos.x': 0.0})
            writer.append(1, {'root.0.pos.x': 1.0})

    def test_baseline_refuses_different_command_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'first.trace'
            second = root / 'second.trace'
            self.write_small_trace(first, 'a' * 64)
            self.write_small_trace(second, 'b' * 64)
            with self.assertRaisesRegex(SystemExit, 'different command sequences'):
                differ.baseline([first, second], root / 'envelope.json', 'p99')

    def test_completeness_gate_requires_three_repeats_of_one_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_small_trace(root / 'first.trace', 'a' * 64)
            self.write_small_trace(root / 'second.trace', 'a' * 64)
            self.write_small_trace(root / 'third.trace', 'b' * 64)
            results = check_artifacts.check(root)
            rek = [result for result in results
                   if result['artifact'] == 'REK traces'][0]
            self.assertEqual(rek['state'], 'INCOMPLETE')
            self.assertIn('different command sequences', rek['detail'])


if __name__ == '__main__':
    unittest.main()
