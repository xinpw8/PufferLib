"""Reduce the closed exact-comparison output; never turns its failure into parity."""
import argparse
import hashlib
import json
import math
from pathlib import Path

LABELS = ('encoder_tokens', 'decoder_identical_raw_input', 'connected_own_tokens',
          'decoder_identical_batch8_tokens')

def analyze(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == 130, 'Expected header, 128 comparisons, and summary'
    header, footer = rows[0], rows[-1]
    assert header['schema'] == 'rek.native_crossbatch.v1'
    assert footer == {'kind': 'summary', 'fixtures': 32, 'tested_common_rows': 64,
                      'bit_exact_all': False, 'tolerance_relaxed': False,
                      'unity_or_physics_parity_claimed': False}
    groups = {label: [] for label in LABELS}
    for fixture in range(32):
        for offset, label in enumerate(LABELS):
            row = rows[1 + fixture * 4 + offset]
            assert (row['kind'], row['fixture'], row['label']) == ('comparison', fixture, label)
            assert row['values'] == (128 if offset == 0 else 58)
            assert 0 <= row['unequal_values'] <= row['unequal_bits'] <= row['values']
            assert math.isfinite(row['max_abs']) and row['max_abs'] >= 0
            assert (row['first'] is None) == (row['unequal_bits'] == 0)
            groups[label].append(row)
    result = {}
    for label, group in groups.items():
        worst = max(group, key=lambda row: row['max_abs'])
        result[label] = {
            'fixture_pairs': len(group), 'values': sum(row['values'] for row in group),
            'unequal_bits': sum(row['unequal_bits'] for row in group),
            'unequal_values': sum(row['unequal_values'] for row in group),
            'max_abs': worst['max_abs'], 'worst_fixture': worst['fixture'],
            'fixture_pairs_over_existing_2e_5_criterion': sum(row['max_abs'] > 2e-5 for row in group),
        }
    assert result['encoder_tokens']['unequal_bits'] == 0
    return {'schema': 'rek.native_crossbatch_closed_assessment.v1',
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'model_sha256': {
                key: header[key] for key in ('encoder2', 'decoder2', 'encoder8', 'decoder8')},
            'synthetic_fixture_pairs': 32, 'common_rows': 64, 'measurements': result,
            'exact_test_passed': False, 'expected_exact_test_exit_code': 1,
            'sampled_numerical_compatibility': all(row['max_abs'] <= 2e-5 for row in result.values()),
            'criterion_origin': 'Existing decoder native-versus-ORT absolute error criterion; applied here as a bounded compatibility judgment, not a new exact-test pass.',
            'limits': ['Synthetic fixed inputs, not a closed-loop physical rollout.',
                       'No universal guarantee near untested encoder quantization boundaries.',
                       'Contact-sensitive future trajectories may diverge.',
                       'No Unity, official-server physics, fight-performance, or real-time app claim.',
                       'Paired-fixture output stores aggregate errors and the first mismatch, not every raw decoder value.']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stdout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.stdout)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
