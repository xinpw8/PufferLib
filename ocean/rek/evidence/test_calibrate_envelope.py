"""Focused tests for the read-only REK envelope calibration diagnostic."""

import tempfile
import unittest
from pathlib import Path

from trace import Trace

import calibrate_envelope as calibration


def make_trace(channels, values, *, command='same-command'):
    ticks = list(range(len(next(iter(values.values())))))
    header = {
        'source': 'rek',
        'build_fingerprint': 'build-1',
        'command_sequence_sha256': command,
        'fixed_delta_time': 0.002,
        'server': {
            'endpoint': 'test.invalid:1',
            'session_id': 'test-session',
            'protocol': 'test',
            'server_reported_version': None,
        },
    }
    return Trace(header, ticks, {name: list(values[name]) for name in channels}, [])


class CalibrationTests(unittest.TestCase):
    def test_q_and_negative_q_have_zero_angular_error(self):
        names = ('root.0.quat.x', 'root.0.quat.y',
                 'root.0.quat.z', 'root.0.quat.w')
        positive = make_trace(names, {
            names[0]: [0.0, 0.0], names[1]: [0.0, 0.0],
            names[2]: [0.0, 0.0], names[3]: [1.0, 1.0],
        })
        negative = make_trace(names, {
            names[0]: [0.0, 0.0], names[1]: [0.0, 0.0],
            names[2]: [0.0, 0.0], names[3]: [-1.0, -1.0],
        })
        record = calibration.quaternion_error_series(
            positive, negative, names)
        self.assertEqual(record['angular_error_rad'], [0.0, 0.0])
        self.assertEqual(record['sign_invariant_chord'], [0.0, 0.0])
        self.assertEqual(record['direct_chord'], [2.0, 2.0])
        self.assertEqual(record['negative_dot_ticks'], 2)
        self.assertEqual(record['opposite_sign_same_rotation_ticks'], 2)

    def test_edge_alignment_recovers_tick_zero_phase_offset(self):
        record = calibration.edge_alignment([2, 7, 12], [5, 10, 15])
        self.assertEqual(record['median_phase_shift_ticks'], 3)
        self.assertEqual(
            record['residual_after_median_shift_ticks']['max'], 0.0)

    def test_scalar_phase_shift_can_remove_a_pure_sampling_offset(self):
        channels = ('state.x',)
        reference = make_trace(channels, {'state.x': [0, 1, 2, 3, 4]})
        candidate = make_trace(channels, {'state.x': [9, 0, 1, 2, 3]})
        thresholds = {'state.x': 0.0}
        raw = calibration.scalar_pair_failures(
            reference, candidate, thresholds, 'p99')
        shifted = calibration.scalar_pair_failures(
            reference, candidate, thresholds, 'p99', shift=1)
        self.assertFalse(raw['passed'])
        self.assertTrue(shifted['passed'])

    def test_validation_requires_one_build_command_and_channel_set(self):
        traces = [make_trace(('x',), {'x': [0, 0]}) for _ in range(4)]
        calibration.validate_traces(['a', 'b', 'c', 'd'], traces)
        traces[-1].header['command_sequence_sha256'] = 'other-command'
        with self.assertRaisesRegex(ValueError, 'shared command sequence'):
            calibration.validate_traces(['a', 'b', 'c', 'd'], traces)

    def test_canonical_artifact_names_are_never_written(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('envelope.json', 'parity_report.json'):
                path = Path(directory) / name
                with self.assertRaisesRegex(ValueError, 'protected canonical'):
                    calibration.write_diagnostic(path, {'diagnostic_only': True})
                self.assertFalse(path.exists())

    def test_diagnostic_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            calibration.write_diagnostic(path, {'diagnostic_only': True})
            with self.assertRaises(FileExistsError):
                calibration.write_diagnostic(path, {'diagnostic_only': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
