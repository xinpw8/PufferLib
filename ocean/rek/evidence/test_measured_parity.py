"""Focused tests for training-only REK self-comparison metrics."""

import math
import unittest

from trace import Trace

import measured_parity as metrics


STATE = 'state.x'
QUATERNION_CHANNELS = (
    'root.0.quat.x',
    'root.0.quat.y',
    'root.0.quat.z',
    'root.0.quat.w',
)
SERVER_ALIGNMENT = {
    'mode': 'event_identity',
    'event_kind': 'server_snapshot_rx',
    'identity_fields': ['sequence'],
}
TICK_ALIGNMENT = {'mode': 'tick_identity'}
GAMEPLAY_EVENTS = [
    {'kind': 'round_start', 'identity_fields': []},
]
RECORDER_EVENTS = [
    {'kind': 'server_snapshot_rx', 'identity_fields': ['sequence']},
    {'kind': 'command_transport_invoked', 'identity_fields': ['sequence']},
]


def make_trace(*, phase=0, quaternion_sign=1.0, gameplay_tick=0,
               transport=((1, 3), (2, 8)), server=None, source='rek',
               zero_quaternion=False, include_hit=False):
    ticks = list(range(14))
    server = server or tuple(
        (sequence, tick + phase)
        for sequence, tick in ((1, 2), (2, 6), (3, 10))
    )
    quaternion_w = 0.0 if zero_quaternion else quaternion_sign
    channels = {
        STATE: [float(tick - phase) for tick in ticks],
        QUATERNION_CHANNELS[0]: [0.0] * len(ticks),
        QUATERNION_CHANNELS[1]: [0.0] * len(ticks),
        QUATERNION_CHANNELS[2]: [0.0] * len(ticks),
        QUATERNION_CHANNELS[3]: [quaternion_w] * len(ticks),
    }
    events = [
        {'tick': gameplay_tick, 'kind': 'round_start'},
        *[
            {'tick': tick, 'kind': 'server_snapshot_rx', 'sequence': sequence}
            for sequence, tick in server
        ],
        *[
            {'tick': tick + phase,
             'kind': 'command_transport_invoked', 'sequence': sequence}
            for sequence, tick in transport
        ],
    ]
    if include_hit:
        events.append({'tick': 5, 'kind': 'hit'})
    header = {
        'source': source,
        'build_fingerprint': 'build-1',
        'command_sequence_sha256': 'command-1',
        'tick_domain': 'client_fixed_update',
        'fixed_delta_time': 0.002,
    }
    return Trace(header, ticks, channels, events)


def state_groups():
    return {'root.0.quat': QUATERNION_CHANNELS}


class MeasuredParityTests(unittest.TestCase):
    def test_three_scopes_accept_representation_phase_and_measured_recorder_jitter(self):
        traces = [
            make_trace(phase=0, quaternion_sign=1.0,
                       transport=((1, 3), (2, 8))),
            make_trace(phase=1, quaternion_sign=-1.0,
                       transport=((1, 4), (2, 9), (3, 12))),
            make_trace(phase=0, quaternion_sign=-1.0,
                       transport=((1, 3), (2, 8))),
            make_trace(phase=1, quaternion_sign=1.0,
                       transport=((1, 4), (2, 9), (3, 12))),
        ]

        report = metrics.leave_one_out_rek(
            traces,
            scalar_channels=[STATE],
            quaternion_groups=state_groups(),
            state_alignment=SERVER_ALIGNMENT,
            gameplay_event_specs=GAMEPLAY_EVENTS,
            gameplay_alignment=TICK_ALIGNMENT,
            recorder_event_specs=RECORDER_EVENTS,
            recorder_alignment=SERVER_ALIGNMENT,
            accept_at='p99')

        self.assertEqual(report['pair_count'], 12)
        self.assertEqual(report['all_scopes_pairs_passed'], 12)
        self.assertTrue(report['all_real_rek_pairs_accepted'])
        self.assertTrue(report['diagnostic_only'])
        self.assertFalse(report['clone_acceptance_changed'])

        fold = report['folds'][0]
        state = fold['training_envelopes']['transition_state']
        quaternion = state['quaternions']['root.0.quat']
        self.assertEqual(quaternion['metric'], 'sign_invariant_angular_error')
        self.assertEqual(quaternion['unit'], 'rad')
        self.assertEqual(quaternion['threshold'], 0.0)
        self.assertGreater(quaternion['sample_count'], 0)

        recorder = fold['training_envelopes'][
            'recorder_transport_diagnostics']['events'][
                'command_transport_invoked']
        self.assertEqual(
            recorder['unmatched_count']['training_pair_count'], 3)
        self.assertEqual(recorder['unmatched_count']['sample_count'], 3)
        self.assertEqual(recorder['unmatched_count']['accept_at'], 'p99')
        self.assertGreater(
            recorder['matched_offset_ticks']['sample_count'], 0)

        shifted = next(
            evaluation for evaluation in fold['evaluations']
            if evaluation['scopes']['transition_state'][
                'alignment']['shift_ticks'] != 0)
        self.assertTrue(shifted['passed'])
        self.assertEqual(
            shifted['scopes']['transition_state']['scalars'][STATE][
                'observed_at_quantile'],
            0.0)

    def test_q_and_negative_q_are_one_rotation_but_zero_quaternions_fail_closed(self):
        self.assertEqual(
            metrics.quaternion_angular_error(
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, -1.0)),
            0.0)
        self.assertIsNone(metrics.quaternion_angular_error(
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0)))

        training = [make_trace(), make_trace(zero_quaternion=True)]
        envelope = metrics.calibrate_state_scope(
            training, [], state_groups(), TICK_ALIGNMENT, 'p99')
        record = envelope['quaternions']['root.0.quat']
        self.assertFalse(envelope['available'])
        self.assertFalse(record['available'])
        self.assertGreater(record['invalid_sample_count'], 0)

    def test_phase_is_measured_by_event_identity_not_event_order(self):
        reference = make_trace(
            phase=0,
            server=((1, 2), (2, 6), (3, 10)))
        candidate = make_trace(
            phase=1,
            server=((1, 3), (3, 11)))
        alignment = metrics.measure_alignment(
            reference, candidate, SERVER_ALIGNMENT)

        self.assertTrue(alignment['available'])
        self.assertEqual(alignment['matched_identity_count'], 2)
        self.assertEqual(alignment['shift_ticks'], 1)
        self.assertEqual(alignment['residual_ticks']['max'], 0.0)

    def test_event_baseline_without_matched_timing_samples_fails_closed(self):
        training = [make_trace(), make_trace(), make_trace()]
        envelope = metrics.calibrate_event_scope(
            training,
            [{'kind': 'hit', 'identity_fields': []}],
            TICK_ALIGNMENT,
            'p99',
            'gameplay_events')

        offset = envelope['events']['hit']['matched_offset_ticks']
        self.assertFalse(envelope['available'])
        self.assertFalse(offset['available'])
        self.assertEqual(offset['sample_count'], 0)
        self.assertEqual(offset['accept_at'], 'p99')
        self.assertIsNone(offset['threshold'])

        result = metrics.evaluate_event_pair(
            training[0], make_trace(include_hit=True), envelope)
        self.assertFalse(result['passed'])

    def test_heldout_event_differences_do_not_inflate_training_tolerance(self):
        training = [make_trace(), make_trace(), make_trace()]
        heldout = make_trace(
            transport=((1, 3), (2, 8), (3, 11), (4, 13)))
        envelope = metrics.calibrate_event_scope(
            training,
            [{'kind': 'command_transport_invoked',
              'identity_fields': ['sequence']}],
            SERVER_ALIGNMENT,
            'p99',
            'recorder_transport_diagnostics')
        threshold = envelope['events'][
            'command_transport_invoked']['unmatched_count']

        self.assertEqual(threshold['sample_count'], 3)
        self.assertEqual(threshold['threshold'], 0.0)
        result = metrics.evaluate_event_pair(training[0], heldout, envelope)
        event = result['events']['command_transport_invoked']
        self.assertEqual(event['total_unmatched_count'], 2)
        self.assertFalse(event['unmatched_count_passed'])
        self.assertFalse(result['passed'])

    def test_gameplay_and_recorder_tolerances_do_not_cross_scopes(self):
        training = [
            make_trace(transport=((1, 1),)),
            make_trace(transport=((1, 6),)),
            make_trace(transport=((1, 11),)),
        ]
        heldout = make_trace(gameplay_tick=2, transport=((1, 9),))
        gameplay_envelope = metrics.calibrate_event_scope(
            training, GAMEPLAY_EVENTS, TICK_ALIGNMENT, 'max',
            'gameplay_events')
        recorder_envelope = metrics.calibrate_event_scope(
            training,
            [{'kind': 'command_transport_invoked',
              'identity_fields': ['sequence']}],
            SERVER_ALIGNMENT,
            'max',
            'recorder_transport_diagnostics')

        gameplay = metrics.evaluate_event_pair(
            training[0], heldout, gameplay_envelope)
        recorder = metrics.evaluate_event_pair(
            training[0], heldout, recorder_envelope)
        self.assertFalse(gameplay['passed'])
        self.assertTrue(recorder['passed'])
        self.assertEqual(
            gameplay_envelope['events']['round_start'][
                'matched_offset_ticks']['threshold'],
            0.0)
        self.assertGreater(
            recorder_envelope['events']['command_transport_invoked'][
                'matched_offset_ticks']['threshold'],
            0.0)

    def test_diagnostic_rejects_clone_sources_and_nonfinite_state(self):
        rek = make_trace()
        clone = make_trace(source='clone:test')
        with self.assertRaisesRegex(
                metrics.MetricError, 'REK traces only'):
            metrics.calibrate_state_scope(
                [rek, clone], [STATE], {}, TICK_ALIGNMENT, 'p99')

        bad = make_trace()
        bad.channels[STATE][2] = math.nan
        envelope = metrics.calibrate_state_scope(
            [rek, bad], [STATE], {}, TICK_ALIGNMENT, 'p99')
        self.assertFalse(envelope['available'])
        self.assertGreater(
            envelope['scalars'][STATE]['invalid_sample_count'], 0)

    def test_envelopes_are_bound_to_the_training_trace_identity(self):
        training = [make_trace(), make_trace(), make_trace()]
        state_envelope = metrics.calibrate_state_scope(
            training, [STATE], {}, TICK_ALIGNMENT, 'p99')
        event_envelope = metrics.calibrate_event_scope(
            training, GAMEPLAY_EVENTS, TICK_ALIGNMENT, 'p99',
            'gameplay_events')

        for field, changed in (
                ('build_fingerprint', 'build-2'),
                ('command_sequence_sha256', 'command-2'),
                ('tick_domain', 'another_tick_domain'),
                ('fixed_delta_time', 0.004)):
            with self.subTest(field=field):
                reference = make_trace()
                candidate = make_trace()
                reference.header[field] = changed
                candidate.header[field] = changed
                with self.assertRaisesRegex(
                        metrics.MetricError, 'do not match the envelope'):
                    metrics.evaluate_state_pair(
                        reference, candidate, state_envelope)
                with self.assertRaisesRegex(
                        metrics.MetricError, 'do not match the envelope'):
                    metrics.evaluate_event_pair(
                        reference, candidate, event_envelope)


if __name__ == '__main__':
    unittest.main(verbosity=2)
