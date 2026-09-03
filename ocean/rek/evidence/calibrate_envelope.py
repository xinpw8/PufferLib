"""Diagnose whether a REK self-variance envelope is calibrated.

This is deliberately separate from ``differ.py``.  It never emits a clone
trace, never declares parity, and refuses to write either of the canonical
``envelope.json`` or ``parity_report.json`` artifacts.

The diagnostic answers three questions from existing REK traces only:

* Does an envelope trained on N-1 REK repeats accept the held-out REK repeat?
* How much error is an artifact of the two equivalent quaternion signs q/-q?
* Are client tick zero and observed update edges phase-aligned between repeats?

No process discovery, window handling, input API, network operation, or live
game attachment is implemented here.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

from trace import Trace


QUANTILES = {
    'median': 0.5,
    'p95': 0.95,
    'p99': 0.99,
    'max': 1.0,
}
PROTECTED_OUTPUTS = {'envelope.json', 'parity_report.json'}


def quantile(values, q):
    """Linear-interpolated quantile, matching differ.py's definition."""
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def distribution(values, absolute=False):
    values = [abs(float(value)) if absolute else float(value) for value in values]
    if not values:
        return {
            'n': 0, 'min': None, 'median': None, 'p95': None,
            'p99': None, 'max': None, 'mean': None,
        }
    return {
        'n': len(values),
        'min': min(values),
        'median': quantile(values, 0.5),
        'p95': quantile(values, 0.95),
        'p99': quantile(values, 0.99),
        'max': max(values),
        'mean': sum(values) / len(values),
    }


def transition_ticks(trace, channel):
    values = trace.channels.get(channel)
    if not values:
        return []
    return [
        trace.ticks[index]
        for index in range(1, len(values))
        if values[index] != values[index - 1]
    ]


def pose_transition_ticks(trace):
    names = [
        name for name in trace.channels
        if name.startswith(('root.', 'joint.'))
    ]
    if not names:
        return []
    edges = []
    for index in range(1, len(trace.ticks)):
        if any(trace.channels[name][index] != trace.channels[name][index - 1]
               for name in names):
            edges.append(trace.ticks[index])
    return edges


def edge_alignment(reference_edges, candidate_edges):
    paired = min(len(reference_edges), len(candidate_edges))
    offsets = [
        candidate_edges[index] - reference_edges[index]
        for index in range(paired)
    ]
    shift = int(round(statistics.median(offsets))) if offsets else None
    residuals = ([offset - shift for offset in offsets]
                 if shift is not None else [])
    return {
        'reference_edges': len(reference_edges),
        'candidate_edges': len(candidate_edges),
        'paired_edges': paired,
        'candidate_minus_reference_ticks': distribution(offsets),
        'median_phase_shift_ticks': shift,
        'residual_after_median_shift_ticks': distribution(
            residuals, absolute=True),
    }


def quaternion_groups(channels):
    """Return complete x/y/z/w channel groups keyed by their common prefix."""
    groups = {}
    for name in channels:
        for axis in ('x', 'y', 'z', 'w'):
            suffix = f'.quat.{axis}'
            if name.endswith(suffix):
                prefix = name[:-len(axis)]
                groups.setdefault(prefix, {})[axis] = name
                break
    return {
        prefix: tuple(parts[axis] for axis in ('x', 'y', 'z', 'w'))
        for prefix, parts in groups.items()
        if set(parts) == {'x', 'y', 'z', 'w'}
    }


def _unit_quaternion(trace, index, names):
    values = tuple(float(trace.channels[name][index]) for name in names)
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0.0:
        return None, norm
    return tuple(value / norm for value in values), norm


def quaternion_error_series(reference, candidate, names, shift=0):
    """Return sign-invariant angular errors and representation diagnostics.

    ``shift`` is the candidate tick offset: reference tick ``t`` is compared
    with candidate tick ``t + shift``.
    """
    reference_indexes, candidate_indexes = _aligned_indexes(
        len(reference.ticks), shift)
    angles = []
    direct_chords = []
    invariant_chords = []
    negative_dots = 0
    opposite_sign_same_rotation = 0
    invalid = 0
    norm_errors = []
    for reference_index, candidate_index in zip(
            reference_indexes, candidate_indexes):
        q_ref, norm_ref = _unit_quaternion(
            reference, reference_index, names)
        q_candidate, norm_candidate = _unit_quaternion(
            candidate, candidate_index, names)
        if q_ref is None or q_candidate is None:
            invalid += 1
            continue
        norm_errors.extend((abs(norm_ref - 1.0), abs(norm_candidate - 1.0)))
        dot = sum(a * b for a, b in zip(q_ref, q_candidate))
        dot = max(-1.0, min(1.0, dot))
        if dot < 0.0:
            negative_dots += 1
        angle = 2.0 * math.acos(min(1.0, abs(dot)))
        direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(q_ref, q_candidate)))
        inverted = math.sqrt(sum((a + b) ** 2 for a, b in zip(q_ref, q_candidate)))
        invariant = min(direct, inverted)
        if dot < 0.0 and angle <= 1e-6:
            opposite_sign_same_rotation += 1
        angles.append(angle)
        direct_chords.append(direct)
        invariant_chords.append(invariant)
    return {
        'shared_ticks': len(reference_indexes),
        'valid_ticks': len(angles),
        'invalid_zero_or_nonfinite': invalid,
        'negative_dot_ticks': negative_dots,
        'opposite_sign_same_rotation_ticks': opposite_sign_same_rotation,
        'angular_error_rad': angles,
        'direct_chord': direct_chords,
        'sign_invariant_chord': invariant_chords,
        'norm_error': norm_errors,
    }


def _overlap_ticks(traces):
    if not traces:
        return []
    common = set(traces[0].ticks)
    for trace in traces[1:]:
        common &= set(trace.ticks)
    return sorted(common)


def _aligned_indexes(length, shift):
    if abs(shift) >= length:
        return range(0), range(0)
    if shift >= 0:
        return range(0, length - shift), range(shift, length)
    return range(-shift, length), range(0, length + shift)


def _scalar_thresholds(training, accept_at):
    q = QUANTILES[accept_at]
    channels = sorted(set.intersection(
        *(set(trace.channels) for trace in training)))
    thresholds = {}
    for channel in channels:
        errors = []
        for left, right in combinations(range(len(training)), 2):
            left_values = training[left].channels[channel]
            right_values = training[right].channels[channel]
            errors.extend(abs(a - b) for a, b in zip(left_values, right_values))
        thresholds[channel] = quantile(errors, q)
    return thresholds


def _quaternion_thresholds(training, groups, accept_at):
    q = QUANTILES[accept_at]
    thresholds = {}
    for prefix, names in groups.items():
        errors = []
        for left, right in combinations(training, 2):
            errors.extend(quaternion_error_series(
                left, right, names)['angular_error_rad'])
        thresholds[prefix] = quantile(errors, q)
    return thresholds


def channel_category(name):
    if '.quat.' in name:
        return 'quaternion_component'
    if name.startswith('joint.') and '.pos.' in name:
        return 'joint_position_component'
    if name.startswith('root.'):
        return 'root_component'
    return name.split('.', 1)[0]


def scalar_pair_failures(reference, candidate, thresholds, accept_at, shift=0):
    q = QUANTILES[accept_at]
    reference_indexes, candidate_indexes = _aligned_indexes(
        len(reference.ticks), shift)
    failed = []
    categories = Counter()
    first_pointwise_exceedance = None
    for channel, allowed in thresholds.items():
        if channel not in reference.channels or channel not in candidate.channels:
            failed.append(channel)
            categories['missing'] += 1
            continue
        reference_values = reference.channels[channel]
        candidate_values = candidate.channels[channel]
        errors = []
        for reference_index, candidate_index in zip(
                reference_indexes, candidate_indexes):
            error = abs(reference_values[reference_index] -
                        candidate_values[candidate_index])
            errors.append(error)
            if error > allowed and (
                    first_pointwise_exceedance is None or
                    reference.ticks[reference_index] < first_pointwise_exceedance):
                first_pointwise_exceedance = reference.ticks[reference_index]
        if quantile(errors, q) > allowed:
            failed.append(channel)
            categories[channel_category(channel)] += 1
    return {
        'shared_ticks': len(reference_indexes),
        'failed_channels': len(failed),
        'passed_channels': len(thresholds) - len(failed),
        'total_channels': len(thresholds),
        'passed': not failed,
        'first_pointwise_exceedance_tick': first_pointwise_exceedance,
        'failure_categories': dict(sorted(categories.items())),
        'failure_names': failed,
    }


def quaternion_pair_failures(
        reference, candidate, thresholds, groups, accept_at, shift=0):
    q = QUANTILES[accept_at]
    failed = []
    for prefix, allowed in thresholds.items():
        errors = quaternion_error_series(
            reference, candidate, groups[prefix], shift=shift)['angular_error_rad']
        if quantile(errors, q) > allowed:
            failed.append(prefix.rstrip('.'))
    return {
        'failed_groups': len(failed),
        'passed_groups': len(thresholds) - len(failed),
        'total_groups': len(thresholds),
        'passed': not failed,
        'failure_names': failed,
    }


def _trace_label(path):
    return Path(path).name


def validate_traces(paths, traces):
    if len(traces) < 4:
        raise ValueError('leave-one-out calibration needs at least four REK traces')
    for path, trace in zip(paths, traces):
        if trace.source != 'rek':
            raise ValueError(f'{path} has source={trace.source!r}; expected REK')
    fingerprints = {trace.build_fingerprint for trace in traces}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError('traces do not identify one shared client build')
    command_hashes = {trace.header.get('command_sequence_sha256') for trace in traces}
    if len(command_hashes) != 1 or None in command_hashes:
        raise ValueError('traces do not identify one shared command sequence')
    channel_sets = {tuple(sorted(trace.channels)) for trace in traces}
    if len(channel_sets) != 1:
        raise ValueError('traces do not expose identical channel sets')
    tick_vectors = {tuple(trace.ticks) for trace in traces}
    if len(tick_vectors) != 1 or not traces[0].ticks:
        raise ValueError('calibration requires identical non-empty client tick vectors')
    ticks = traces[0].ticks
    if ticks != list(range(ticks[0], ticks[0] + len(ticks))):
        raise ValueError('calibration requires contiguous unit-spaced client ticks')


def cadence_record(trace, channel):
    edges = transition_ticks(trace, channel)
    intervals = [right - left for left, right in zip(edges, edges[1:])]
    fixed_delta = trace.header.get('fixed_delta_time')
    fixed_delta = (float(fixed_delta)
                   if isinstance(fixed_delta, (int, float)) else None)
    return {
        'edge_count': len(edges),
        'first_edge_tick': edges[0] if edges else None,
        'last_edge_tick': edges[-1] if edges else None,
        'interval_ticks': distribution(intervals),
        'interval_ms': distribution(
            [value * fixed_delta * 1000.0 for value in intervals]
            if fixed_delta is not None else []),
    }


def diagnose(paths, traces, accept_at='p99'):
    validate_traces(paths, traces)
    groups = quaternion_groups(traces[0].channels)
    failure_frequency = Counter()
    quaternion_failure_frequency = Counter()
    loo_pairs = []

    for heldout_index, heldout in enumerate(traces):
        training = [
            trace for index, trace in enumerate(traces)
            if index != heldout_index
        ]
        training_paths = [
            path for index, path in enumerate(paths)
            if index != heldout_index
        ]
        scalar_thresholds = _scalar_thresholds(training, accept_at)
        quat_thresholds = _quaternion_thresholds(training, groups, accept_at)
        for reference, reference_path in zip(training, training_paths):
            reference_edges = transition_ticks(reference, 'seq.server_snapshot_rx')
            heldout_edges = transition_ticks(heldout, 'seq.server_snapshot_rx')
            alignment = edge_alignment(reference_edges, heldout_edges)
            shift = alignment['median_phase_shift_ticks'] or 0
            raw = scalar_pair_failures(
                reference, heldout, scalar_thresholds, accept_at)
            phase_shifted = scalar_pair_failures(
                reference, heldout, scalar_thresholds, accept_at, shift=shift)
            quat = quaternion_pair_failures(
                reference, heldout, quat_thresholds, groups, accept_at)
            quat_shifted = quaternion_pair_failures(
                reference, heldout, quat_thresholds, groups, accept_at,
                shift=shift)
            failure_frequency.update(raw.pop('failure_names'))
            quaternion_failure_frequency.update(quat.pop('failure_names'))
            phase_shifted.pop('failure_names')
            quat_shifted.pop('failure_names')
            loo_pairs.append({
                'heldout': _trace_label(paths[heldout_index]),
                'reference': _trace_label(reference_path),
                'training': [_trace_label(path) for path in training_paths],
                'server_snapshot_alignment': alignment,
                'raw_scalar': raw,
                'raw_scalar_after_server_phase_shift': phase_shifted,
                'sign_invariant_quaternion': quat,
                'sign_invariant_quaternion_after_server_phase_shift': quat_shifted,
            })

    pair_quaternion_records = []
    all_angles = []
    all_direct = []
    all_invariant = []
    all_norm_errors = []
    negative_dots = 0
    opposite_sign_same_rotation = 0
    valid_quaternion_samples = 0
    invalid_quaternion_samples = 0
    for left_index, right_index in combinations(range(len(traces)), 2):
        pair_angles = []
        pair_direct = []
        pair_invariant = []
        pair_negative = 0
        pair_opposite_same = 0
        pair_invalid = 0
        for names in groups.values():
            record = quaternion_error_series(
                traces[left_index], traces[right_index], names)
            pair_angles.extend(record['angular_error_rad'])
            pair_direct.extend(record['direct_chord'])
            pair_invariant.extend(record['sign_invariant_chord'])
            all_norm_errors.extend(record['norm_error'])
            pair_negative += record['negative_dot_ticks']
            pair_opposite_same += record['opposite_sign_same_rotation_ticks']
            pair_invalid += record['invalid_zero_or_nonfinite']
        all_angles.extend(pair_angles)
        all_direct.extend(pair_direct)
        all_invariant.extend(pair_invariant)
        negative_dots += pair_negative
        opposite_sign_same_rotation += pair_opposite_same
        valid_quaternion_samples += len(pair_angles)
        invalid_quaternion_samples += pair_invalid
        pair_quaternion_records.append({
            'left': _trace_label(paths[left_index]),
            'right': _trace_label(paths[right_index]),
            'valid_samples': len(pair_angles),
            'negative_dot_samples': pair_negative,
            'opposite_sign_same_rotation_samples': pair_opposite_same,
            'angular_error_rad': distribution(pair_angles),
            'direct_chord': distribution(pair_direct),
            'sign_invariant_chord': distribution(pair_invariant),
        })

    signal_names = (
        'seq.server_snapshot_rx',
        'seq.client_transport_invoke',
        'round.timer',
    )
    per_trace_alignment = []
    edge_cache = {}
    for path, trace in zip(paths, traces):
        signals = {name: cadence_record(trace, name) for name in signal_names}
        pose_edges = pose_transition_ticks(trace)
        pose_intervals = [right - left
                          for left, right in zip(pose_edges, pose_edges[1:])]
        fixed_delta = trace.header.get('fixed_delta_time')
        signals['pose_any_change'] = {
            'edge_count': len(pose_edges),
            'first_edge_tick': pose_edges[0] if pose_edges else None,
            'last_edge_tick': pose_edges[-1] if pose_edges else None,
            'interval_ticks': distribution(pose_intervals),
            'interval_ms': distribution(
                [value * float(fixed_delta) * 1000.0 for value in pose_intervals]
                if isinstance(fixed_delta, (int, float)) else []),
        }
        per_trace_alignment.append({
            'trace': _trace_label(path),
            'fixed_delta_time_s': fixed_delta,
            'ticks': len(trace.ticks),
            'signals': signals,
        })
        edge_cache[_trace_label(path)] = {
            name: (pose_edges if name == 'pose_any_change'
                   else transition_ticks(trace, name))
            for name in (*signal_names, 'pose_any_change')
        }

    pair_alignment = []
    for left_index, right_index in combinations(range(len(paths)), 2):
        left = _trace_label(paths[left_index])
        right = _trace_label(paths[right_index])
        pair_alignment.append({
            'left': left,
            'right': right,
            'signals': {
                name: edge_alignment(
                    edge_cache[left][name], edge_cache[right][name])
                for name in (*signal_names, 'pose_any_change')
            },
        })

    server_contexts = []
    for path, trace in zip(paths, traces):
        session = str(trace.server.get('session_id', ''))
        server_contexts.append({
            'trace': _trace_label(path),
            'endpoint': trace.server.get('endpoint'),
            'session_sha256_prefix': (
                hashlib.sha256(session.encode()).hexdigest()[:12]
                if session else None),
            'protocol': trace.server.get('protocol'),
            'server_reported_version': trace.server.get('server_reported_version'),
        })

    raw_passes = sum(pair['raw_scalar']['passed'] for pair in loo_pairs)
    shifted_passes = sum(
        pair['raw_scalar_after_server_phase_shift']['passed']
        for pair in loo_pairs)
    quat_passes = sum(
        pair['sign_invariant_quaternion']['passed'] for pair in loo_pairs)
    quat_shifted_passes = sum(
        pair['sign_invariant_quaternion_after_server_phase_shift']['passed']
        for pair in loo_pairs)
    first_server_edges = [
        item['signals']['seq.server_snapshot_rx']['first_edge_tick']
        for item in per_trace_alignment
        if item['signals']['seq.server_snapshot_rx']['first_edge_tick'] is not None
    ]

    return {
        'schema': 1,
        'diagnostic_only': True,
        'parity_claim': False,
        'canonical_outputs_written': False,
        'build_fingerprint': traces[0].build_fingerprint,
        'command_sequence_sha256': traces[0].header.get(
            'command_sequence_sha256'),
        'accept_at': accept_at,
        'traces': [str(Path(path).resolve()) for path in paths],
        'server_contexts': server_contexts,
        'leave_one_out': {
            'trace_count': len(traces),
            'pair_count': len(loo_pairs),
            'raw_scalar_pairs_passed': raw_passes,
            'server_phase_shifted_scalar_pairs_passed': shifted_passes,
            'sign_invariant_quaternion_pairs_passed': quat_passes,
            'server_phase_shifted_sign_invariant_quaternion_pairs_passed':
                quat_shifted_passes,
            'all_real_rek_pairs_accepted_raw': raw_passes == len(loo_pairs),
            'frequently_failed_scalar_channels': [
                {'channel': name, 'failed_pairs': count}
                for name, count in failure_frequency.most_common(30)
            ],
            'frequently_failed_quaternion_groups': [
                {'group': name, 'failed_pairs': count}
                for name, count in quaternion_failure_frequency.most_common(30)
            ],
            'pairs': loo_pairs,
        },
        'quaternion_representation': {
            'groups': len(groups),
            'valid_pair_tick_samples': valid_quaternion_samples,
            'invalid_pair_tick_samples': invalid_quaternion_samples,
            'negative_dot_samples': negative_dots,
            'negative_dot_fraction': (
                negative_dots / valid_quaternion_samples
                if valid_quaternion_samples else None),
            'opposite_sign_same_rotation_samples': opposite_sign_same_rotation,
            'angular_error_rad': distribution(all_angles),
            'direct_chord': distribution(all_direct),
            'sign_invariant_chord': distribution(all_invariant),
            'direct_minus_sign_invariant_chord': distribution([
                direct - invariant
                for direct, invariant in zip(all_direct, all_invariant)
            ]),
            'quaternion_norm_error': distribution(all_norm_errors),
            'pairs': pair_quaternion_records,
        },
        'tick_alignment': {
            'imported_absolute_time_channel_available': any(
                name.startswith(('time.', 'utc.'))
                for name in traces[0].channels),
            'first_server_snapshot_edge_ticks': first_server_edges,
            'first_server_snapshot_edge_spread_ticks': (
                max(first_server_edges) - min(first_server_edges)
                if first_server_edges else None),
            'tick_zero_phase_aligned_to_first_server_snapshot': (
                len(set(first_server_edges)) == 1 if first_server_edges else None),
            'per_trace': per_trace_alignment,
            'pairs': pair_alignment,
        },
        'limits': [
            'This diagnoses the imported client-observation traces only.',
            'It does not establish server physics, hidden controller state, or simulator parity.',
            'A phase shift measured from fight-state snapshots is diagnostic and is not silently applied to the canonical verifier.',
            'The imported traces have no wall-clock timestamp channel, so absolute-time alignment cannot be tested from these trace files.',
        ],
    }


def write_diagnostic(path, document):
    path = Path(path)
    if path.name.lower() in PROTECTED_OUTPUTS:
        raise ValueError(f'refusing protected canonical output name {path.name}')
    encoded = json.dumps(document, indent=1) + '\n'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)


def print_summary(document):
    loo = document['leave_one_out']
    quat = document['quaternion_representation']
    timing = document['tick_alignment']
    print('diagnostic only: no parity claim')
    print(f"REK traces: {loo['trace_count']}; leave-one-out pairs: {loo['pair_count']}")
    print(f"raw scalar real-REK pairs accepted: {loo['raw_scalar_pairs_passed']}/{loo['pair_count']}")
    print('server-phase-shifted scalar pairs accepted: '
          f"{loo['server_phase_shifted_scalar_pairs_passed']}/{loo['pair_count']}")
    print('sign-invariant quaternion pairs accepted: '
          f"{loo['sign_invariant_quaternion_pairs_passed']}/{loo['pair_count']}")
    print(f"quaternion negative-dot samples: {quat['negative_dot_samples']}/"
          f"{quat['valid_pair_tick_samples']}")
    print('first server-snapshot edge ticks: '
          f"{timing['first_server_snapshot_edge_ticks']}")
    print('tick-zero server phase aligned: '
          f"{timing['tick_zero_phase_aligned_to_first_server_snapshot']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+')
    parser.add_argument(
        '--accept-at', choices=tuple(QUANTILES), default='p99')
    parser.add_argument('--out', help='new diagnostic JSON path; existing files are refused')
    parser.add_argument('--json', action='store_true',
                        help='print the full diagnostic JSON to stdout')
    args = parser.parse_args(argv)

    traces = [Trace.load(path) for path in args.traces]
    document = diagnose(args.traces, traces, accept_at=args.accept_at)
    if args.out:
        write_diagnostic(args.out, document)
    if args.json:
        print(json.dumps(document, indent=1))
    else:
        print_summary(document)
        if args.out:
            print(f'wrote {args.out}')
    return 0 if document['leave_one_out']['all_real_rek_pairs_accepted_raw'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
