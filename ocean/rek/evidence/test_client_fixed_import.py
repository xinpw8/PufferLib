"""Focused tests for importing private-AI client-fixed recorder output."""

import copy
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import client_fixed_import
import check_artifacts
import controlled_schedule
import differ
import test_raw_bone_validate as raw_fixture
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


def schedule_manifest():
    return {
        'duration_ticks': controlled_schedule.DURATION_TICKS,
        'fixed_substeps_per_tick': (
            controlled_schedule.FIXED_SUBSTEPS_PER_TICK),
        'move_commands': [
            {'move_index': move, 'tick': tick}
            for tick, move in controlled_schedule.EXPECTED_MOVES
        ],
        'schedule_id': controlled_schedule.SCHEDULE_ID,
        'schedule_rate_hz': controlled_schedule.SCHEDULE_RATE_HZ,
        'schema': controlled_schedule.SCHEMA,
        'unity_fixed_rate_hz': controlled_schedule.UNITY_FIXED_RATE_HZ,
        'velocity_component_order': ['forward', 'strafe', 'yaw'],
        'velocity_segments': [
            {
                'start': start,
                'stop': stop,
                'velocity_command': list(velocity),
            }
            for start, stop, velocity in
            controlled_schedule.EXPECTED_VELOCITY_SEGMENTS
        ],
    }


def control_records(manifest, run_id='synthetic-controlled-run', start_time=5.2):
    digest = controlled_schedule.canonical_sha256(manifest)
    start_frame = 1000
    records = []
    for tick, velocity, move_index in controlled_schedule._expected_steps():
        records.append({
            'event': 'schedule_step',
            'protocol': 'rek.ui_bridge.v1',
            'schedule_id': controlled_schedule.SCHEDULE_ID,
            'command_sequence_schema': controlled_schedule.SCHEMA,
            'command_sequence_sha256': digest,
            'schedule_run_id': run_id,
            'schedule_tick': tick,
            'client_fixed_substep': (
                tick * controlled_schedule.FIXED_SUBSTEPS_PER_TICK),
            'fixed_substeps_per_schedule_tick': (
                controlled_schedule.FIXED_SUBSTEPS_PER_TICK),
            'velocity_command_xyz': list(velocity),
            'move_index': move_index,
            'move_accepted_locally': True,
            'server_acceptance_observed': False,
            'unity_fixed_time': (
                start_time + tick / controlled_schedule.SCHEDULE_RATE_HZ),
            'unity_frame': start_frame + 1 + tick,
        })
        if move_index is not None:
            for event_name in (
                    'schedule_move_send_invoked',
                    'schedule_move_send_completed'):
                records.append({
                    'event': event_name,
                    'protocol': 'rek.ui_bridge.v1',
                    'schedule_id': controlled_schedule.SCHEDULE_ID,
                    'command_sequence_schema': controlled_schedule.SCHEMA,
                    'command_sequence_sha256': digest,
                    'schedule_run_id': run_id,
                    'schedule_tick': tick,
                    'move_index': move_index,
                    'pending_move_readback': True,
                    'pending_move_index_readback': move_index,
                    'server_acceptance_observed': False,
                    'unity_frame': start_frame + 1 + tick,
                })
    records.insert(0, {
        'event': 'ack',
        'protocol': 'rek.ui_bridge.v1',
        'request_id': 'synthetic-start-request',
        'command': 'StartMeasuredSchedule',
        'status': 'accepted',
        'reason': 'measured_schedule_started',
        'applied': True,
        'client_request_issued': False,
        'server_acceptance_observed': False,
        'lease_connection_id': 1,
        'schedule_run_id': run_id,
        'schedule_id': controlled_schedule.SCHEDULE_ID,
        'command_sequence_schema': controlled_schedule.SCHEMA,
        'command_sequence_sha256': digest,
        'unity_frame': start_frame,
        'unity_time': start_time,
        'unity_thread': 'main',
    })
    records.insert(1, {
        'event': 'state',
        'protocol': 'rek.ui_bridge.v1',
        'build': {
            'game_assembly_sha256': (
                client_fixed_import.EXPECTED_GAME_ASSEMBLY_SHA256),
            'global_metadata_sha256': (
                client_fixed_import.EXPECTED_METADATA_SHA256),
        },
        'private_ai': {
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
        },
        'control': {
            'schedule_id': controlled_schedule.SCHEDULE_ID,
            'command_sequence_schema': controlled_schedule.SCHEMA,
            'command_sequence_sha256': digest,
            'schedule_run_id': run_id,
            'schedule_running': True,
            'fixed_substeps_per_schedule_tick': (
                controlled_schedule.FIXED_SUBSTEPS_PER_TICK),
        },
    })
    records.append({
        'event': 'schedule_end',
        'protocol': 'rek.ui_bridge.v1',
        'schedule_id': controlled_schedule.SCHEDULE_ID,
        'command_sequence_schema': controlled_schedule.SCHEMA,
        'command_sequence_sha256': digest,
        'schedule_run_id': run_id,
        'schedule_tick': controlled_schedule.DURATION_TICKS - 1,
        'client_fixed_substep': (
            controlled_schedule.DURATION_TICKS *
            controlled_schedule.FIXED_SUBSTEPS_PER_TICK - 1),
        'move_send_completed_count': len(controlled_schedule.EXPECTED_MOVES),
        'final_neutral_send_observed': True,
        'reason': 'complete',
        'complete': True,
        'server_acceptance_observed': False,
        'unity_fixed_time': (
            start_time +
            (controlled_schedule.DURATION_TICKS *
             controlled_schedule.FIXED_SUBSTEPS_PER_TICK - 1) /
            controlled_schedule.UNITY_FIXED_RATE_HZ),
        'unity_frame': (
            start_frame + controlled_schedule.DURATION_TICKS + 1),
    })
    return records


def _v5_sample(sample_index):
    client_tick = sample_index * controlled_schedule.FIXED_SUBSTEPS_PER_TICK
    record = sample(sample_index)
    record['sample_index'] = sample_index
    record['client_fixed_tick'] = client_tick
    record['unity_fixed_time'] = 5.0 + client_tick / controlled_schedule.UNITY_FIXED_RATE_HZ
    record['unity_time'] = record['unity_fixed_time']
    record['unity_unscaled_time'] = record['unity_fixed_time']
    record['fighter_0']['bones'] = None
    record['fighter_1']['bones'] = None
    record['transport_observation'].update({
        'raw_protocol_sequence': 0,
        'raw_fight_state_sequence': 0,
        'raw_score_sequence': 0,
        'raw_hit_sequence': 0,
    })
    return record


def _rewrite_input_request(record, network_index, velocity):
    body = bytes((network_index & 0xff,)) + struct.pack('<3f', *velocity)
    record.update(raw_fixture.encoded_body(body))
    record['network_index_source_int32'] = network_index
    record['network_index_wire_uint8'] = network_index & 0xff
    record['velocity_command_xyz'] = list(velocity)
    record['velocity_float32_bit_patterns'] = [
        f'0x{value:08x}' for value in struct.unpack('<3I', body[1:])
    ]


def _rewrite_move_request(record, network_index, move_index):
    body = bytes((network_index & 0xff, move_index & 0xff))
    record.update(raw_fixture.encoded_body(body))
    record['network_index_source_int32'] = network_index
    record['network_index_wire_uint8'] = network_index & 0xff
    record['move_index_source_int32'] = move_index
    record['move_index_wire_uint8'] = move_index & 0xff


def v5_records():
    base = raw_fixture.fixture()
    start = copy.deepcopy(base[0])
    end = copy.deepcopy(base[-1])
    start['fixed_delta_time'] = 1.0 / controlled_schedule.UNITY_FIXED_RATE_HZ

    request_records = []
    request_sequence = 1
    input_sequence = 0
    move_sequence = 0
    schedule_sample_tick = 100
    control_start_frame = 1000

    pre_window_input = raw_fixture.input_request(
        sequence=request_sequence, timestamp=5.198)
    input_sequence += 1
    pre_window_input['message_request_sequence'] = input_sequence
    pre_window_input['client_fixed_tick_at_observation'] = schedule_sample_tick - 1
    pre_window_input['unity_frame'] = control_start_frame - 1
    _rewrite_input_request(pre_window_input, 0, (0.0, 0.0, 0.0))
    request_records.append(pre_window_input)
    request_sequence += 1

    activation_input = raw_fixture.input_request(
        sequence=request_sequence, timestamp=5.199)
    input_sequence += 1
    activation_input['message_request_sequence'] = input_sequence
    activation_input['client_fixed_tick_at_observation'] = schedule_sample_tick
    activation_input['unity_frame'] = control_start_frame
    _rewrite_input_request(activation_input, 0, (0.0, 0.0, 0.0))
    request_records.append(activation_input)
    request_sequence += 1

    segment_velocities = {
        tick: velocity
        for start, stop, velocity in controlled_schedule.EXPECTED_VELOCITY_SEGMENTS
        for tick in range(start, stop)
    }
    moves = dict(controlled_schedule.EXPECTED_MOVES)
    observed_boundaries = sorted({
        *(start for start, _stop, _velocity in
          controlled_schedule.EXPECTED_VELOCITY_SEGMENTS),
        *(tick for tick, _move_index in controlled_schedule.EXPECTED_MOVES),
    })
    for schedule_tick in observed_boundaries:
        relative_substep = (
            schedule_tick * controlled_schedule.FIXED_SUBSTEPS_PER_TICK)
        timestamp = 5.201 + relative_substep / controlled_schedule.UNITY_FIXED_RATE_HZ
        input_record = raw_fixture.input_request(
            sequence=request_sequence, timestamp=timestamp)
        input_sequence += 1
        input_record['message_request_sequence'] = input_sequence
        input_record['client_fixed_tick_at_observation'] = (
            schedule_sample_tick + relative_substep + 1)
        input_record['unity_frame'] = (
            control_start_frame + schedule_tick + 1)
        _rewrite_input_request(
            input_record, 0, segment_velocities[schedule_tick])
        request_records.append(input_record)
        request_sequence += 1

        if schedule_tick in moves:
            move_record = raw_fixture.move_request(
                sequence=request_sequence, timestamp=timestamp)
            move_sequence += 1
            move_record['message_request_sequence'] = move_sequence
            move_record['client_fixed_tick_at_observation'] = (
                schedule_sample_tick + relative_substep + 1)
            move_record['unity_frame'] = (
                control_start_frame + schedule_tick + 1)
            _rewrite_move_request(move_record, 0, moves[schedule_tick])
            request_records.append(move_record)
            request_sequence += 1

    protocol_records = [
        copy.deepcopy(record) for record in base[1:-1]
        if record.get('event') in {
            'raw_bone_packet',
            'decoded_bone_snapshot',
            'raw_fight_state_packet',
            'fight_state_snapshot_applied',
            'raw_score_packet',
            'raw_hit_packet',
        }
    ]
    for offset, record in enumerate(protocol_records, 1):
        record['client_fixed_tick_at_observation'] = 120 + offset

    sample_count = 2701
    samples = [_v5_sample(index) for index in range(sample_count)]
    end.update({
        'sample_count': sample_count,
        'client_transport_invocation_count': len(request_records),
        'client_transport_method_counts': {
            'SendVelocityCommand': input_sequence,
            'SendMoveEvent': move_sequence,
        },
        'client_fixed_tick_at_end': sample_count * 10,
    })
    return [start, *samples, *request_records, *protocol_records, end]


def _refresh_v5_request_counters(records):
    request_records = [
        record for record in records
        if record.get('event') in {
            'outbound_request_projection',
            'client_transport_method_invoked',
        }
    ]
    method_by_message = {
        'REK_Input': 'SendVelocityCommand',
        'REK_Move': 'SendMoveEvent',
    }
    method_counts = {}
    for sequence, record in enumerate(request_records, 1):
        record['request_sequence'] = sequence
        if record.get('event') == 'outbound_request_projection':
            method = method_by_message[record['message']]
            method_counts[method] = method_counts.get(method, 0) + 1
            record['message_request_sequence'] = method_counts[method]
        else:
            method = record['method']
            method_counts[method] = method_counts.get(method, 0) + 1
            record['method_request_sequence'] = method_counts[method]
    end = next(record for record in records if record.get('event') == 'capture_end')
    end['client_transport_invocation_count'] = len(request_records)
    end['client_transport_method_counts'] = method_counts


def _insert_v5_request(records, request):
    request_tick = request['client_fixed_tick_at_observation']
    indices = [
        index for index, record in enumerate(records)
        if record.get('event') in {
            'outbound_request_projection',
            'client_transport_method_invoked',
        }
    ]
    insert_at = indices[-1] + 1
    for index in indices:
        record_tick = records[index]['client_fixed_tick_at_observation']
        if record_tick > request_tick:
            insert_at = index
            break
    records.insert(insert_at, request)
    _refresh_v5_request_counters(records)


class ClientFixedImportTests(unittest.TestCase):
    def write_fixture(self, root, fixture_records):
        raw = root / 'capture.jsonl'
        raw.write_text(
            ''.join(json.dumps(record) + '\n' for record in fixture_records),
            encoding='utf-8')
        inventory = root / 'inventory.json'
        inventory.write_text(json.dumps(inventory_record()), encoding='utf-8')
        return raw, inventory

    def write_v5_fixture(self, root, fixture_records=None, controls=None,
                         manifest=None):
        fixture_records = fixture_records or v5_records()
        manifest = manifest or schedule_manifest()
        controls = controls or control_records(manifest)
        raw, inventory = self.write_fixture(root, fixture_records)
        manifest_path = root / 'schedule.json'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        control_path = root / 'control.jsonl'
        control_path.write_text(
            ''.join(json.dumps(record) + '\n' for record in controls),
            encoding='utf-8')
        return raw, inventory, control_path, manifest_path

    def test_v5_converts_only_completed_controlled_schedule_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, inventory, control, manifest = self.write_v5_fixture(root)
            output = root / 'controlled.trace'
            result = client_fixed_import.convert(
                raw,
                inventory,
                output,
                control_log_path=control,
                schedule_manifest_path=manifest)
            trace = Trace.load(output)

            self.assertEqual(trace.ticks, list(range(2601)))
            self.assertEqual(trace.header['tick_domain'], 'controlled_schedule_50hz')
            self.assertEqual(trace.header['tick_rate_hz'], 50)
            self.assertFalse(trace.header['compact_samples_tick_complete'])
            self.assertTrue(trace.header['complete_schedule'])
            self.assertFalse(trace.header['complete_round'])
            self.assertEqual(trace.header['command_sample_phase_substeps'], 0)
            self.assertEqual(
                trace.header['command_sequence_schema'], controlled_schedule.SCHEMA)
            self.assertEqual(
                trace.header['command_sequence_sha256'],
                controlled_schedule.canonical_sha256(schedule_manifest()))
            self.assertEqual(
                trace.header['fighter_pairing']['local_fighter_index'], 0)
            self.assertEqual(
                trace.header['fighter_pairing']['opponent_fighter_index'], 1)
            self.assertEqual(
                trace.header['fighter_pairing']['fighters']['0']['bone_count'],
                len(v5_records()[0]['fighter_0_bones']))
            self.assertEqual(
                result['fighter_pairing'], trace.header['fighter_pairing'])
            outbound = trace.header['outbound_command_stream']
            self.assertEqual(outbound['start_client_fixed_tick'], 100)
            self.assertEqual(outbound['end_client_fixed_tick'], 26109)
            self.assertEqual(outbound['start_unity_frame'], 1000)
            self.assertEqual(outbound['end_unity_frame_exclusive'], 3602)
            self.assertEqual(outbound['client_fixed_substeps'], 26010)
            self.assertEqual(outbound['first_request_sequence'], 2)
            self.assertEqual(outbound['last_request_sequence'], 32)
            self.assertEqual(outbound['records'], 31)
            self.assertEqual(outbound['rek_input_projections'], 23)
            self.assertEqual(outbound['rek_move_projections'], 8)
            self.assertEqual(outbound['rek_special_invocations'], 0)
            self.assertEqual(outbound['rek_estop_invocations'], 0)
            self.assertIn('no periodic', outbound['cadence_claim'])
            self.assertIn('continuous per-hook ownership', outbound['proof_limit'])
            self.assertEqual(result['outbound_command_stream'], outbound)
            self.assertTrue(trace.server['session_id'].startswith('sha256:'))
            self.assertFalse(trace.server['session_identifier_recorded'])
            self.assertNotIn('arena-test-session', output.read_bytes().decode(
                'latin-1', errors='ignore'))
            scheduled_moves = [
                event['move_index'] for event in trace.events
                if event['kind'] == 'command_schedule_step'
                and 'move_index' in event
            ]
            projected_moves = [
                event['move_index_source_int32'] for event in trace.events
                if event['kind'] == 'outbound_request_projection'
                and event.get('message') == 'REK_Move'
            ]
            expected_moves = [move for _tick, move in controlled_schedule.EXPECTED_MOVES]
            self.assertEqual(scheduled_moves, expected_moves)
            self.assertEqual(projected_moves, expected_moves)
            outbound_events = [
                event for event in trace.events
                if event['kind'] in {
                    'outbound_request_projection',
                    'outbound_request_invoked',
                }
            ]
            self.assertNotIn(
                1, {event['request_sequence'] for event in outbound_events})
            activation = next(
                event for event in outbound_events
                if event['request_sequence'] == 2)
            self.assertEqual(activation['controlled_window_client_tick'], 0)
            self.assertEqual(activation['controlled_fixed_substep'], -1)
            self.assertEqual(activation['tick'], 0)
            final_neutral = next(
                event for event in outbound_events
                if event.get('message') == 'REK_Input' and
                event['controlled_fixed_substep'] == 26000)
            self.assertEqual(final_neutral['tick'], 2600)
            self.assertEqual(final_neutral['velocity_command_xyz'], [0.0, 0.0, 0.0])
            self.assertEqual(final_neutral['client_fixed_tick_at_observation'], 26101)
            self.assertIn(
                'raw_fight_state_packet_rx',
                {event['kind'] for event in trace.events})
            self.assertEqual(result['raw_recorder_schema'], client_fixed_import.SCHEMA_V5)
            self.assertTrue(result['server_session_hash_present'])
            self.assertTrue(result['complete_schedule'])

    def test_v5_requires_explicit_control_log_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, inventory = self.write_fixture(root, v5_records())
            with self.assertRaisesRegex(
                    ValueError, '--control-log and --schedule-manifest'):
                client_fixed_import.convert(
                    raw, inventory, root / 'missing-control.trace')

    def test_v5_normalizes_constant_compact_sample_phase_without_invention(self):
        samples = [
            {'client_fixed_tick': tick}
            for tick in range(0, 27001, 10)
        ]
        selected, ticks, phase = client_fixed_import._v5_normalized_samples(
            samples, start_tick=3)
        self.assertEqual(len(selected), 2600)
        self.assertEqual(ticks, list(range(1, 2601)))
        self.assertEqual(phase, -3)

    def test_v5_selects_the_one_control_run_inside_raw_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_document = schedule_manifest()
            controls = [
                *control_records(
                    manifest_document,
                    run_id='matching-run',
                    start_time=5.2),
                *control_records(
                    manifest_document,
                    run_id='different-round',
                    start_time=100.0),
            ]
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, controls=controls, manifest=manifest_document)
            output = root / 'selected-run.trace'
            result = client_fixed_import.convert(
                raw,
                inventory,
                output,
                control_log_path=control,
                schedule_manifest_path=manifest)
            self.assertEqual(result['schedule_run_id'], 'matching-run')
            self.assertEqual(Trace.load(output).header['schedule_run_id'], 'matching-run')

    def test_v5_rejects_control_move_not_locally_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_document = schedule_manifest()
            controls = control_records(manifest_document)
            move = next(record for record in controls
                        if record.get('move_index') is not None)
            move['move_accepted_locally'] = False
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, controls=controls, manifest=manifest_document)
            with self.assertRaisesRegex(ValueError, 'not accepted locally'):
                client_fixed_import.convert(
                    raw,
                    inventory,
                    root / 'rejected-move.trace',
                    control_log_path=control,
                    schedule_manifest_path=manifest)

    def test_v5_rejects_raw_move_sequence_that_differs_from_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = v5_records()
            moves = [record for record in fixture
                     if record.get('event') == 'outbound_request_projection'
                     and record.get('message') == 'REK_Move']
            _rewrite_move_request(moves[3], 0, 10)
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, fixture_records=fixture)
            with self.assertRaisesRegex(ValueError, 'pinned order'):
                client_fixed_import.convert(
                    raw,
                    inventory,
                    root / 'wrong-move.trace',
                    control_log_path=control,
                    schedule_manifest_path=manifest)

    def test_v5_rejects_extra_missing_or_reordered_raw_moves(self):
        for case in ('extra', 'missing', 'reordered'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = v5_records()
                moves = [
                    record for record in fixture
                    if record.get('event') == 'outbound_request_projection' and
                    record.get('message') == 'REK_Move'
                ]
                if case == 'extra':
                    _insert_v5_request(fixture, copy.deepcopy(moves[0]))
                elif case == 'missing':
                    fixture.remove(moves[3])
                    _refresh_v5_request_counters(fixture)
                else:
                    first_index = moves[0]['move_index_source_int32']
                    second_index = moves[1]['move_index_source_int32']
                    _rewrite_move_request(moves[0], 0, second_index)
                    _rewrite_move_request(moves[1], 0, first_index)
                raw, inventory, control, manifest = self.write_v5_fixture(
                    root, fixture_records=fixture)
                with self.assertRaisesRegex(ValueError, 'pinned order'):
                    client_fixed_import.convert(
                        raw,
                        inventory,
                        root / f'{case}-move.trace',
                        control_log_path=control,
                        schedule_manifest_path=manifest)

    def test_v5_rejects_special_or_estop_invocation_in_controlled_window(self):
        for method in ('SendSpecialEvent', 'SendEStopToggle'):
            with self.subTest(method=method), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = v5_records()
                relative_substep = 4000
                request = raw_fixture.invocation_request(
                    method,
                    sequence=999,
                    timestamp=(
                        5.201 + relative_substep /
                        controlled_schedule.UNITY_FIXED_RATE_HZ))
                request['client_fixed_tick_at_observation'] = 101 + relative_substep
                request['unity_frame'] = (
                    1001 + relative_substep //
                    controlled_schedule.FIXED_SUBSTEPS_PER_TICK)
                _insert_v5_request(fixture, request)
                raw, inventory, control, manifest = self.write_v5_fixture(
                    root, fixture_records=fixture)
                with self.assertRaisesRegex(
                        ValueError, 'forbidden outbound invocation'):
                    client_fixed_import.convert(
                        raw,
                        inventory,
                        root / f'{method}.trace',
                        control_log_path=control,
                        schedule_manifest_path=manifest)

    def test_v5_rejects_input_value_or_timing_outside_pinned_velocity(self):
        for case in ('value', 'timing'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = v5_records()
                target = next(
                    record for record in fixture
                    if record.get('message') == 'REK_Input' and
                    record.get('velocity_command_xyz') == [1.0, 0.0, 0.0])
                if case == 'value':
                    _rewrite_input_request(target, 0, (0.0, 0.0, 0.0))
                else:
                    target['client_fixed_tick_at_observation'] = 600
                raw, inventory, control, manifest = self.write_v5_fixture(
                    root, fixture_records=fixture)
                expected_error = (
                    'does not match the pinned schedule'
                    if case == 'value' else
                    'client-tick phase disagrees with the control transcript')
                with self.assertRaisesRegex(ValueError, expected_error):
                    client_fixed_import.convert(
                        raw,
                        inventory,
                        root / f'wrong-input-{case}.trace',
                        control_log_path=control,
                        schedule_manifest_path=manifest)

    def test_v5_accepts_pre_step_frame_input_at_client_tick_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = v5_records()
            request = raw_fixture.input_request(sequence=999, timestamp=6.2)
            request['client_fixed_tick_at_observation'] = 600
            request['unity_frame'] = 1050
            _rewrite_input_request(request, 0, (0.0, 0.0, 0.0))
            _insert_v5_request(fixture, request)
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, fixture_records=fixture)
            output = root / 'pre-step-boundary.trace'
            client_fixed_import.convert(
                raw,
                inventory,
                output,
                control_log_path=control,
                schedule_manifest_path=manifest)
            event = next(
                event for event in Trace.load(output).events
                if event.get('client_fixed_tick_at_observation') == 600 and
                event.get('message') == 'REK_Input')
            self.assertEqual(event['controlled_window_client_tick'], 500)
            self.assertEqual(event['controlled_fixed_substep'], 499)
            self.assertEqual(event['velocity_command_xyz'], [0.0, 0.0, 0.0])

    def test_v5_requires_accepted_start_ack_for_exact_frame_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controls = control_records(schedule_manifest())
            start = next(
                record for record in controls
                if record.get('event') == 'ack' and
                record.get('command') == 'StartMeasuredSchedule')
            start['status'] = 'rejected'
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, controls=controls)
            with self.assertRaisesRegex(
                    ValueError, 'schedule-start ack is malformed'):
                client_fixed_import.convert(
                    raw,
                    inventory,
                    root / 'rejected-start.trace',
                    control_log_path=control,
                    schedule_manifest_path=manifest)

    def test_v5_requires_raw_input_projection_in_every_velocity_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = v5_records()
            target = next(
                record for record in fixture
                if record.get('message') == 'REK_Input' and
                record.get('velocity_command_xyz') == [0.0, -1.0, 0.0])
            fixture.remove(target)
            _refresh_v5_request_counters(fixture)
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, fixture_records=fixture)
            with self.assertRaisesRegex(
                    ValueError, 'no projection for pinned velocity segment'):
                client_fixed_import.convert(
                    raw,
                    inventory,
                    root / 'missing-input-segment.trace',
                    control_log_path=control,
                    schedule_manifest_path=manifest)

    def test_v5_move_must_follow_first_observed_input_send_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = v5_records()
            move = next(
                record for record in fixture
                if record.get('message') == 'REK_Move')
            move['client_fixed_tick_at_observation'] += 1
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, fixture_records=fixture)
            with self.assertRaisesRegex(ValueError, 'not immediately paired'):
                client_fixed_import.convert(
                    raw,
                    inventory,
                    root / 'late-move.trace',
                    control_log_path=control,
                    schedule_manifest_path=manifest)

    def test_v5_rejects_control_schedule_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controls = control_records(schedule_manifest())
            next(record for record in controls
                 if record.get('event') == 'schedule_step')[
                     'command_sequence_sha256'] = '0' * 64
            raw, inventory, control, manifest = self.write_v5_fixture(
                root, controls=controls)
            with self.assertRaisesRegex(ValueError, 'does not match manifest'):
                client_fixed_import.convert(
                    raw,
                    inventory,
                    root / 'wrong-hash.trace',
                    control_log_path=control,
                    schedule_manifest_path=manifest)

    def test_v5_control_contract_rejects_incomplete_or_misaligned_run(self):
        mutations = {
            'fixed substep': lambda records: next(
                record for record in records
                if record.get('event') == 'schedule_step').update(
                    client_fixed_substep=1),
            'complete marker': lambda records: next(
                record for record in records
                if record.get('event') == 'schedule_end').update(
                    complete=False),
            'private Bot 1': lambda records: next(
                record for record in records
                if record.get('event') == 'state')['private_ai'].update(
                    context_is_solo=False),
        }
        for expected_error, mutate in mutations.items():
            with self.subTest(expected_error=expected_error), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_document = schedule_manifest()
                manifest_path = root / 'schedule.json'
                manifest_path.write_text(json.dumps(manifest_document))
                controls = control_records(manifest_document)
                mutate(controls)
                control_path = root / 'control.jsonl'
                control_path.write_text(''.join(
                    json.dumps(record) + '\n' for record in controls))
                manifest = controlled_schedule.validate_manifest(manifest_path)
                with self.assertRaisesRegex(ValueError, expected_error):
                    controlled_schedule.validate_control_log(control_path, manifest)

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

    def write_small_trace(self, path, command_sequence, pairing=None):
        with TraceWriter(
                path, ['root.0.pos.x'], 'f' * 64, 'rek', authority='local',
                provenance={'root.0.pos.x': {
                    'kind': 'class', 'ref': 'test measured field'}},
                command_sequence_sha256=command_sequence,
                command_sequence_schema=(
                    client_fixed_import.COMMAND_SEQUENCE_SCHEMA),
                fighter_pairing=pairing) as writer:
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

    def test_baseline_envelope_propagates_command_sequence_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'first.trace'
            second = root / 'second.trace'
            envelope = root / 'envelope.json'
            self.write_small_trace(first, 'a' * 64)
            self.write_small_trace(second, 'a' * 64)
            differ.baseline([first, second], envelope, 'p99')
            document = json.loads(envelope.read_text())
            self.assertEqual(
                document['command_sequence_schema'],
                client_fixed_import.COMMAND_SEQUENCE_SCHEMA)

    def test_baseline_refuses_different_fighter_pairings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'first.trace'
            second = root / 'second.trace'
            self.write_small_trace(
                first, 'a' * 64, {'fighters': {'0': 't800', '1': 't800'}})
            self.write_small_trace(
                second, 'a' * 64, {'fighters': {'0': 't800', '1': 'g1'}})
            with self.assertRaisesRegex(SystemExit, 'different fighter pairings'):
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

    def test_completeness_gate_rejects_mixed_fighter_pairings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_small_trace(
                root / 'first.trace', 'a' * 64,
                {'fighters': {'0': 't800', '1': 't800'}})
            self.write_small_trace(
                root / 'second.trace', 'a' * 64,
                {'fighters': {'0': 't800', '1': 't800'}})
            self.write_small_trace(
                root / 'third.trace', 'a' * 64,
                {'fighters': {'0': 't800', '1': 'g1'}})
            results = check_artifacts.check(root)
            rek = [result for result in results
                   if result['artifact'] == 'REK traces'][0]
            self.assertEqual(rek['state'], 'INCOMPLETE')
            self.assertIn('different fighter pairings', rek['detail'])


if __name__ == '__main__':
    unittest.main()
