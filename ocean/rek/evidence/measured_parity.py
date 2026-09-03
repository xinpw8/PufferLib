"""Measured metrics for held-out REK self-comparison.

This module is diagnostic.  It calibrates on REK repeats and evaluates another
REK repeat; it rejects clone traces and has no path that changes ``differ.py``'s
canonical clone verdict.

Every alignment choice is supplied by the caller.  ``event_identity`` alignment
matches an explicitly named event by explicitly named payload fields, then uses
an observed integer offset that minimises absolute residual timing error.  No
channel name, event kind, phase window, or tolerance is guessed here.

The three scopes stay separate:

* transition/state scalars use absolute error;
* complete quaternion groups use sign-invariant angular error;
* gameplay events and recorder/transport events receive independent,
  leave-one-out count and timing envelopes.

Event tolerances record their sample counts and quantiles.  Missing alignment
evidence, ambiguous event identities, missing matched events, non-finite state,
and invalid quaternions all fail closed.
"""

import math
from itertools import combinations


QUANTILES = {
    'median': 0.5,
    'p95': 0.95,
    'p99': 0.99,
    'max': 1.0,
}


class MetricError(ValueError):
    """The requested metric is ambiguous or unsupported by the trace data."""


def quantile(values, probability):
    """Linear-interpolated quantile matching ``differ.py``."""
    values = sorted(float(value) for value in values)
    if not values:
        raise MetricError('a quantile needs at least one measured value')
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def distribution(values):
    values = [float(value) for value in values]
    if not values:
        return {
            'sample_count': 0,
            'min': None,
            'median': None,
            'p95': None,
            'p99': None,
            'max': None,
        }
    return {
        'sample_count': len(values),
        'min': min(values),
        'median': quantile(values, 0.5),
        'p95': quantile(values, 0.95),
        'p99': quantile(values, 0.99),
        'max': max(values),
    }


def _probability(accept_at):
    try:
        return QUANTILES[accept_at]
    except KeyError as exc:
        raise MetricError(
            f'accept_at must be one of {tuple(QUANTILES)}, got {accept_at!r}') from exc


def _exact_keys(document, expected, label):
    if not isinstance(document, dict):
        raise MetricError(f'{label} must be a mapping')
    actual = set(document)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise MetricError(
            f'{label} keys do not match: missing={missing}, unknown={unknown}')


def _validate_alignment(specification):
    if not isinstance(specification, dict):
        raise MetricError('alignment must be an explicit mapping')
    mode = specification.get('mode')
    if mode == 'tick_identity':
        _exact_keys(specification, {'mode'}, 'tick alignment')
        return dict(specification)
    if mode == 'event_identity':
        _exact_keys(
            specification,
            {'mode', 'event_kind', 'identity_fields'},
            'event alignment')
        kind = specification['event_kind']
        fields = specification['identity_fields']
        if not isinstance(kind, str) or not kind:
            raise MetricError('alignment event_kind must be a non-empty string')
        if not isinstance(fields, (list, tuple)) or not fields:
            raise MetricError(
                'event_identity alignment needs at least one identity field')
        if any(not isinstance(field, str) or not field for field in fields):
            raise MetricError('alignment identity fields must be non-empty strings')
        if len(set(fields)) != len(fields):
            raise MetricError('alignment identity_fields contains a duplicate')
        return {
            'mode': mode,
            'event_kind': kind,
            'identity_fields': list(fields),
        }
    raise MetricError(
        "alignment mode must be 'tick_identity' or 'event_identity'")


def _validate_event_spec(specification):
    _exact_keys(specification, {'kind', 'identity_fields'}, 'event metric')
    kind = specification['kind']
    fields = specification['identity_fields']
    if not isinstance(kind, str) or not kind:
        raise MetricError('event kind must be a non-empty string')
    if not isinstance(fields, (list, tuple)):
        raise MetricError('event identity_fields must be a list')
    if any(not isinstance(field, str) or not field for field in fields):
        raise MetricError('event identity fields must be non-empty strings')
    if len(set(fields)) != len(fields):
        raise MetricError(f'event {kind!r} identity_fields contains a duplicate')
    return {'kind': kind, 'identity_fields': list(fields)}


def _validate_rek_traces(traces, minimum):
    traces = list(traces)
    if len(traces) < minimum:
        raise MetricError(
            f'this diagnostic needs at least {minimum} REK traces, got {len(traces)}')
    for index, trace in enumerate(traces):
        if trace.source != 'rek':
            raise MetricError(
                f'trace {index} has source={trace.source!r}; '
                'measured self-comparison accepts REK traces only')
        if not trace.ticks:
            raise MetricError(f'trace {index} has no ticks')
        if any(type(tick) is not int for tick in trace.ticks):
            raise MetricError(f'trace {index} has a non-integer tick')
        if any(right != left + 1
               for left, right in zip(trace.ticks, trace.ticks[1:])):
            raise MetricError(
                f'trace {index} ticks are not contiguous and unit-spaced')

    fingerprints = {trace.build_fingerprint for trace in traces}
    if len(fingerprints) != 1 or None in fingerprints or '' in fingerprints:
        raise MetricError('REK traces do not identify one shared client build')
    commands = {
        trace.header.get('command_sequence_sha256') for trace in traces
    }
    if len(commands) != 1 or None in commands or '' in commands:
        raise MetricError('REK traces do not identify one shared command sequence')
    tick_domains = {trace.header.get('tick_domain') for trace in traces}
    if len(tick_domains) != 1 or None in tick_domains or '' in tick_domains:
        raise MetricError('REK traces do not identify one shared tick domain')

    fixed_steps = []
    for trace in traces:
        value = trace.header.get('fixed_delta_time')
        if type(value) not in (int, float):
            raise MetricError('REK traces do not measure fixed_delta_time')
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise MetricError('fixed_delta_time must be finite and positive')
        fixed_steps.append(value)
    if len(set(fixed_steps)) != 1:
        raise MetricError('REK traces do not share one measured fixed_delta_time')
    return traces


def _trace_identity(trace):
    return {
        'build_fingerprint': trace.build_fingerprint,
        'command_sequence_sha256': trace.header.get(
            'command_sequence_sha256'),
        'tick_domain': trace.header.get('tick_domain'),
        'fixed_delta_time': float(trace.header.get('fixed_delta_time')),
    }


def _require_envelope_identity(trace, envelope):
    expected = envelope.get('trace_identity')
    observed = _trace_identity(trace)
    if expected != observed:
        raise MetricError(
            'evaluation traces do not match the envelope build, command '
            'sequence, tick domain, and fixed step')


def complete_quaternion_groups(channels):
    """Return complete ``*.quat.{x,y,z,w}`` groups without filling gaps."""
    parts = {}
    for name in channels:
        head, separator, axis = name.rpartition('.')
        if separator and head.endswith('.quat') and axis in ('x', 'y', 'z', 'w'):
            parts.setdefault(head, {})[axis] = name
    return {
        name: tuple(group[axis] for axis in ('x', 'y', 'z', 'w'))
        for name, group in sorted(parts.items())
        if set(group) == {'x', 'y', 'z', 'w'}
    }


def _identity_value(value, label):
    if type(value) not in (int, str):
        raise MetricError(
            f'{label} must be an integer or string identity, got {value!r}')
    return value


def _event_map(trace, specification):
    specification = _validate_event_spec(specification)
    kind = specification['kind']
    fields = specification['identity_fields']
    selected = [event for event in trace.events if event.get('kind') == kind]
    if not fields:
        if len(selected) > 1:
            raise MetricError(
                f'event {kind!r} occurs {len(selected)} times but no identity '
                'fields were supplied')
        identities = [('singleton',)] if selected else []
    else:
        identities = []
        for index, event in enumerate(selected):
            missing = [field for field in fields if field not in event]
            if missing:
                raise MetricError(
                    f'event {kind!r} occurrence {index} lacks identity fields '
                    f'{missing}')
            identities.append(tuple(
                _identity_value(event[field], f'event {kind!r}.{field}')
                for field in fields))

    mapped = {}
    for event, identity in zip(selected, identities):
        tick = event.get('tick')
        if type(tick) is not int:
            raise MetricError(f'event {kind!r} has a non-integer tick')
        if identity in mapped:
            raise MetricError(
                f'event {kind!r} repeats identity {identity!r}')
        mapped[identity] = tick
    return mapped


def _observed_median(offsets):
    """Choose an observed integer offset minimising L1 residual error."""
    candidates = sorted(set(offsets))
    if not candidates:
        raise MetricError('cannot measure phase without matched events')
    return min(
        candidates,
        key=lambda candidate: (
            sum(abs(offset - candidate) for offset in offsets),
            abs(candidate),
            candidate,
        ))


def measure_alignment(reference, candidate, specification):
    """Measure candidate tick offset from an explicit alignment specification."""
    specification = _validate_alignment(specification)
    if specification['mode'] == 'tick_identity':
        return {
            'available': True,
            'mode': 'tick_identity',
            'shift_ticks': 0,
            'definition': 'candidate_tick = reference_tick + shift_ticks',
            'matched_identity_count': None,
            'raw_offset_ticks': None,
            'residual_ticks': None,
        }

    event_spec = {
        'kind': specification['event_kind'],
        'identity_fields': specification['identity_fields'],
    }
    reference_events = _event_map(reference, event_spec)
    candidate_events = _event_map(candidate, event_spec)
    shared = sorted(set(reference_events) & set(candidate_events), key=repr)
    if not shared:
        return {
            'available': False,
            'mode': 'event_identity',
            'event_kind': specification['event_kind'],
            'identity_fields': specification['identity_fields'],
            'shift_ticks': None,
            'definition': 'candidate_tick = reference_tick + shift_ticks',
            'matched_identity_count': 0,
            'reason': 'no shared alignment-event identities',
            'raw_offset_ticks': distribution([]),
            'residual_ticks': distribution([]),
        }
    offsets = [
        candidate_events[identity] - reference_events[identity]
        for identity in shared
    ]
    shift = _observed_median(offsets)
    residuals = [abs(offset - shift) for offset in offsets]
    return {
        'available': True,
        'mode': 'event_identity',
        'event_kind': specification['event_kind'],
        'identity_fields': specification['identity_fields'],
        'shift_ticks': shift,
        'definition': 'candidate_tick = reference_tick + shift_ticks',
        'matched_identity_count': len(shared),
        'raw_offset_ticks': distribution(offsets),
        'residual_ticks': distribution(residuals),
    }


def _aligned_indexes(reference, candidate, shift):
    candidate_index = {tick: index for index, tick in enumerate(candidate.ticks)}
    pairs = []
    for reference_index, tick in enumerate(reference.ticks):
        candidate_tick = tick + shift
        if candidate_tick in candidate_index:
            pairs.append((tick, reference_index, candidate_index[candidate_tick]))
    return pairs


def _finite_error(left, right):
    left = float(left)
    right = float(right)
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return abs(left - right)


def quaternion_angular_error(left, right):
    """Sign-invariant angular distance in radians, or ``None`` if invalid."""
    left = tuple(float(value) for value in left)
    right = tuple(float(value) for value in right)
    if not all(math.isfinite(value) for value in left + right):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return 2.0 * math.acos(min(1.0, abs(max(-1.0, dot))))


def _threshold_record(values, invalid_count, accept_at, training_pair_count,
                      unavailable_reason=None):
    probability = _probability(accept_at)
    available = bool(values) and invalid_count == 0 and unavailable_reason is None
    reason = unavailable_reason
    if reason is None and invalid_count:
        reason = f'{invalid_count} invalid training sample(s)'
    if reason is None and not values:
        reason = 'no measured training samples'
    return {
        'available': available,
        'accept_at': accept_at,
        'quantile_probability': probability,
        'sample_count': len(values),
        'training_pair_count': training_pair_count,
        'invalid_sample_count': invalid_count,
        'threshold': quantile(values, probability) if available else None,
        'distribution': distribution(values),
        'reason': reason,
    }


def _validate_state_configuration(traces, scalar_channels, quaternion_groups):
    scalar_channels = list(scalar_channels)
    if len(set(scalar_channels)) != len(scalar_channels):
        raise MetricError('transition/state scalar channels contain a duplicate')
    groups = {name: tuple(channels)
              for name, channels in quaternion_groups.items()}
    if not scalar_channels and not groups:
        raise MetricError('transition/state scope contains no metrics')
    grouped_channels = set()
    for name, channels in groups.items():
        if len(channels) != 4 or len(set(channels)) != 4:
            raise MetricError(
                f'quaternion group {name!r} must name four distinct channels')
        expected = tuple(f'{name}.{axis}' for axis in ('x', 'y', 'z', 'w'))
        if not name.endswith('.quat') or channels != expected:
            raise MetricError(
                f'quaternion group {name!r} must be the complete ordered '
                f'*.quat.{{x,y,z,w}} channels, got {channels!r}')
        grouped_channels.update(channels)
    overlap = sorted(set(scalar_channels) & grouped_channels)
    if overlap:
        raise MetricError(
            f'quaternion components also configured as scalars: {overlap[:8]}')
    required = set(scalar_channels) | grouped_channels
    for index, trace in enumerate(traces):
        missing = sorted(required - set(trace.channels))
        if missing:
            raise MetricError(
                f'trace {index} lacks configured transition/state channels '
                f'{missing[:8]}')
    return scalar_channels, groups


def calibrate_state_scope(training, scalar_channels, quaternion_groups,
                          alignment, accept_at):
    """Build a training-only transition/state envelope from REK repeats."""
    training = _validate_rek_traces(training, minimum=2)
    alignment = _validate_alignment(alignment)
    scalar_channels, quaternion_groups = _validate_state_configuration(
        training, scalar_channels, quaternion_groups)
    pair_count = len(training) * (len(training) - 1) // 2
    scalar_errors = {name: [] for name in scalar_channels}
    scalar_invalid = {name: 0 for name in scalar_channels}
    quaternion_errors = {name: [] for name in quaternion_groups}
    quaternion_invalid = {name: 0 for name in quaternion_groups}
    pair_alignments = []
    unavailable = None

    for left_index, right_index in combinations(range(len(training)), 2):
        measured = measure_alignment(
            training[left_index], training[right_index], alignment)
        pair_alignments.append({
            'left_training_index': left_index,
            'right_training_index': right_index,
            **measured,
        })
        if not measured['available']:
            unavailable = (
                f'training pair {left_index}/{right_index} has no measured alignment')
            continue
        pairs = _aligned_indexes(
            training[left_index], training[right_index], measured['shift_ticks'])
        if not pairs:
            unavailable = (
                f'training pair {left_index}/{right_index} has no aligned ticks')
            continue

        for name in scalar_channels:
            left_values = training[left_index].channels[name]
            right_values = training[right_index].channels[name]
            for _, left_sample, right_sample in pairs:
                error = _finite_error(
                    left_values[left_sample], right_values[right_sample])
                if error is None:
                    scalar_invalid[name] += 1
                else:
                    scalar_errors[name].append(error)

        for name, channels in quaternion_groups.items():
            for _, left_sample, right_sample in pairs:
                left = tuple(
                    training[left_index].channels[channel][left_sample]
                    for channel in channels)
                right = tuple(
                    training[right_index].channels[channel][right_sample]
                    for channel in channels)
                error = quaternion_angular_error(left, right)
                if error is None:
                    quaternion_invalid[name] += 1
                else:
                    quaternion_errors[name].append(error)

    scalars = {
        name: {
            'metric': 'absolute_error',
            'unit': 'channel_native',
            **_threshold_record(
                scalar_errors[name], scalar_invalid[name], accept_at, pair_count,
                unavailable),
        }
        for name in scalar_channels
    }
    quaternions = {
        name: {
            'metric': 'sign_invariant_angular_error',
            'unit': 'rad',
            'channels': list(quaternion_groups[name]),
            **_threshold_record(
                quaternion_errors[name], quaternion_invalid[name], accept_at,
                pair_count, unavailable),
        }
        for name in quaternion_groups
    }
    available = (
        unavailable is None and
        all(record['available'] for record in scalars.values()) and
        all(record['available'] for record in quaternions.values())
    )
    return {
        'scope': 'transition_state',
        'rek_self_diagnostic_only': True,
        'trace_identity': _trace_identity(training[0]),
        'available': available,
        'reason': unavailable,
        'alignment': alignment,
        'training_trace_count': len(training),
        'training_pair_count': pair_count,
        'training_pair_alignments': pair_alignments,
        'scalars': scalars,
        'quaternions': quaternions,
    }


def evaluate_state_pair(reference, candidate, envelope):
    """Evaluate one held-out REK pair against a training-only state envelope."""
    _validate_rek_traces([reference, candidate], minimum=2)
    if envelope.get('scope') != 'transition_state':
        raise MetricError('state evaluation requires a transition_state envelope')
    _require_envelope_identity(reference, envelope)
    alignment = measure_alignment(reference, candidate, envelope['alignment'])
    if not envelope.get('available') or not alignment['available']:
        return {
            'scope': 'transition_state',
            'available': False,
            'passed': False,
            'alignment': alignment,
            'reason': envelope.get('reason') or alignment.get('reason'),
            'scalars': {},
            'quaternions': {},
            'failures': ['transition_state:baseline_unavailable'],
        }
    pairs = _aligned_indexes(
        reference, candidate, alignment['shift_ticks'])
    if not pairs:
        return {
            'scope': 'transition_state',
            'available': False,
            'passed': False,
            'alignment': alignment,
            'reason': 'held-out pair has no aligned ticks',
            'scalars': {},
            'quaternions': {},
            'failures': ['transition_state:no_aligned_ticks'],
        }
    probability = _probability(next(iter(
        list(envelope['scalars'].values()) +
        list(envelope['quaternions'].values())))['accept_at'])
    scalars = {}
    quaternions = {}
    failures = []

    for name, threshold in envelope['scalars'].items():
        errors = []
        invalid = 0
        for _, left_sample, right_sample in pairs:
            error = _finite_error(
                reference.channels[name][left_sample],
                candidate.channels[name][right_sample])
            if error is None:
                invalid += 1
            else:
                errors.append(error)
        observed = quantile(errors, probability) if errors else None
        passed = (
            threshold['available'] and invalid == 0 and observed is not None and
            observed <= threshold['threshold'])
        scalars[name] = {
            'metric': threshold['metric'],
            'unit': threshold['unit'],
            'sample_count': len(errors),
            'invalid_sample_count': invalid,
            'observed_at_quantile': observed,
            'allowed': threshold['threshold'],
            'passed': passed,
        }
        if not passed:
            failures.append(f'scalar:{name}')

    for name, threshold in envelope['quaternions'].items():
        channels = threshold['channels']
        errors = []
        invalid = 0
        for _, left_sample, right_sample in pairs:
            left = tuple(reference.channels[channel][left_sample]
                         for channel in channels)
            right = tuple(candidate.channels[channel][right_sample]
                          for channel in channels)
            error = quaternion_angular_error(left, right)
            if error is None:
                invalid += 1
            else:
                errors.append(error)
        observed = quantile(errors, probability) if errors else None
        passed = (
            threshold['available'] and invalid == 0 and observed is not None and
            observed <= threshold['threshold'])
        quaternions[name] = {
            'metric': threshold['metric'],
            'unit': threshold['unit'],
            'channels': channels,
            'sample_count': len(errors),
            'invalid_sample_count': invalid,
            'observed_at_quantile': observed,
            'allowed': threshold['threshold'],
            'passed': passed,
        }
        if not passed:
            failures.append(f'quaternion:{name}')

    return {
        'scope': 'transition_state',
        'available': True,
        'passed': not failures,
        'alignment': alignment,
        'aligned_tick_count': len(pairs),
        'scalars': scalars,
        'quaternions': quaternions,
        'failures': failures,
    }


def _event_pair_measurement(reference, candidate, specification, shift):
    reference_events = _event_map(reference, specification)
    candidate_events = _event_map(candidate, specification)
    shared = sorted(set(reference_events) & set(candidate_events), key=repr)
    unmatched_reference = len(set(reference_events) - set(candidate_events))
    unmatched_candidate = len(set(candidate_events) - set(reference_events))
    raw_offsets = [
        candidate_events[identity] - reference_events[identity]
        for identity in shared
    ]
    residuals = [abs(offset - shift) for offset in raw_offsets]
    return {
        'reference_count': len(reference_events),
        'candidate_count': len(candidate_events),
        'matched_identity_count': len(shared),
        'unmatched_reference_count': unmatched_reference,
        'unmatched_candidate_count': unmatched_candidate,
        'total_unmatched_count': unmatched_reference + unmatched_candidate,
        'raw_offset_ticks': distribution(raw_offsets),
        'absolute_residual_offset_ticks': distribution(residuals),
        '_residual_values': residuals,
    }


def calibrate_event_scope(training, event_specs, alignment, accept_at, scope):
    """Calibrate one explicit event scope from pairwise REK measurements."""
    if scope not in ('gameplay_events', 'recorder_transport_diagnostics'):
        raise MetricError('event scope must be gameplay or recorder/transport')
    training = _validate_rek_traces(training, minimum=2)
    alignment = _validate_alignment(alignment)
    event_specs = [_validate_event_spec(spec) for spec in event_specs]
    kinds = [spec['kind'] for spec in event_specs]
    if not event_specs:
        raise MetricError(f'{scope} contains no event metrics')
    if len(set(kinds)) != len(kinds):
        raise MetricError(f'{scope} contains a duplicate event kind')

    pair_count = len(training) * (len(training) - 1) // 2
    unmatched = {kind: [] for kind in kinds}
    residuals = {kind: [] for kind in kinds}
    pairs_without_matches = {kind: 0 for kind in kinds}
    pair_records = []
    unavailable = None
    for left_index, right_index in combinations(range(len(training)), 2):
        measured = measure_alignment(
            training[left_index], training[right_index], alignment)
        record = {
            'left_training_index': left_index,
            'right_training_index': right_index,
            'alignment': measured,
            'events': {},
        }
        if not measured['available']:
            unavailable = (
                f'training pair {left_index}/{right_index} has no measured alignment')
            pair_records.append(record)
            continue
        for specification in event_specs:
            kind = specification['kind']
            event_record = _event_pair_measurement(
                training[left_index], training[right_index], specification,
                measured['shift_ticks'])
            unmatched[kind].append(event_record['total_unmatched_count'])
            values = event_record.pop('_residual_values')
            residuals[kind].extend(values)
            if not values:
                pairs_without_matches[kind] += 1
            record['events'][kind] = event_record
        pair_records.append(record)

    events = {}
    for specification in event_specs:
        kind = specification['kind']
        no_match_reason = None
        if pairs_without_matches[kind]:
            no_match_reason = (
                f'{pairs_without_matches[kind]} training pair(s) have no matched '
                f'{kind!r} identities')
        count_threshold = _threshold_record(
            unmatched[kind], 0, accept_at, pair_count, unavailable)
        offset_threshold = _threshold_record(
            residuals[kind], 0, accept_at, pair_count,
            unavailable or no_match_reason)
        events[kind] = {
            'identity_fields': specification['identity_fields'],
            'unmatched_count': count_threshold,
            'matched_offset_ticks': offset_threshold,
            'available': (
                count_threshold['available'] and offset_threshold['available']),
        }
    available = unavailable is None and all(
        record['available'] for record in events.values())
    return {
        'scope': scope,
        'rek_self_diagnostic_only': True,
        'trace_identity': _trace_identity(training[0]),
        'available': available,
        'reason': unavailable,
        'alignment': alignment,
        'training_trace_count': len(training),
        'training_pair_count': pair_count,
        'event_specs': event_specs,
        'events': events,
        'training_pairs': pair_records,
    }


def evaluate_event_pair(reference, candidate, envelope):
    """Evaluate a held-out REK pair against one training-only event envelope."""
    _validate_rek_traces([reference, candidate], minimum=2)
    scope = envelope.get('scope')
    if scope not in ('gameplay_events', 'recorder_transport_diagnostics'):
        raise MetricError('event evaluation received an unknown scope')
    _require_envelope_identity(reference, envelope)
    alignment = measure_alignment(reference, candidate, envelope['alignment'])
    if not envelope.get('available') or not alignment['available']:
        return {
            'scope': scope,
            'available': False,
            'passed': False,
            'alignment': alignment,
            'reason': envelope.get('reason') or alignment.get('reason'),
            'events': {},
            'failures': [f'{scope}:baseline_unavailable'],
        }

    results = {}
    failures = []
    for specification in envelope['event_specs']:
        kind = specification['kind']
        thresholds = envelope['events'][kind]
        measured = _event_pair_measurement(
            reference, candidate, specification, alignment['shift_ticks'])
        residual_values = measured.pop('_residual_values')
        count_threshold = thresholds['unmatched_count']
        offset_threshold = thresholds['matched_offset_ticks']
        probability = offset_threshold['quantile_probability']
        observed_offset = (
            quantile(residual_values, probability) if residual_values else None)
        count_passed = (
            count_threshold['available'] and
            measured['total_unmatched_count'] <= count_threshold['threshold'])
        offset_passed = (
            offset_threshold['available'] and observed_offset is not None and
            observed_offset <= offset_threshold['threshold'])
        passed = count_passed and offset_passed
        results[kind] = {
            **measured,
            'unmatched_count_allowed': count_threshold['threshold'],
            'unmatched_count_passed': count_passed,
            'offset_at_quantile': observed_offset,
            'offset_allowed_ticks': offset_threshold['threshold'],
            'offset_passed': offset_passed,
            'passed': passed,
        }
        if not passed:
            failures.append(f'{scope}:{kind}')
    return {
        'scope': scope,
        'available': True,
        'passed': not failures,
        'alignment': alignment,
        'events': results,
        'failures': failures,
    }


def leave_one_out_rek(traces, *, scalar_channels, quaternion_groups,
                      state_alignment, gameplay_event_specs,
                      gameplay_alignment, recorder_event_specs,
                      recorder_alignment, accept_at):
    """Run training-only, three-scope leave-one-out REK self-comparison."""
    traces = _validate_rek_traces(traces, minimum=4)
    gameplay_kinds = {
        _validate_event_spec(specification)['kind']
        for specification in gameplay_event_specs
    }
    recorder_kinds = {
        _validate_event_spec(specification)['kind']
        for specification in recorder_event_specs
    }
    overlap = sorted(gameplay_kinds & recorder_kinds)
    if overlap:
        raise MetricError(
            f'event kinds cannot cross gameplay and recorder scopes: {overlap}')
    folds = []
    scope_passes = {
        'transition_state': 0,
        'gameplay_events': 0,
        'recorder_transport_diagnostics': 0,
    }
    pair_count = 0
    all_passes = 0

    for heldout_index, heldout in enumerate(traces):
        training_indexes = [
            index for index in range(len(traces)) if index != heldout_index
        ]
        training = [traces[index] for index in training_indexes]
        state_envelope = calibrate_state_scope(
            training, scalar_channels, quaternion_groups,
            state_alignment, accept_at)
        gameplay_envelope = calibrate_event_scope(
            training, gameplay_event_specs, gameplay_alignment, accept_at,
            'gameplay_events')
        recorder_envelope = calibrate_event_scope(
            training, recorder_event_specs, recorder_alignment, accept_at,
            'recorder_transport_diagnostics')
        evaluations = []
        for reference_index in training_indexes:
            transition_state = evaluate_state_pair(
                traces[reference_index], heldout, state_envelope)
            gameplay = evaluate_event_pair(
                traces[reference_index], heldout, gameplay_envelope)
            recorder = evaluate_event_pair(
                traces[reference_index], heldout, recorder_envelope)
            scopes = {
                'transition_state': transition_state,
                'gameplay_events': gameplay,
                'recorder_transport_diagnostics': recorder,
            }
            for name, result in scopes.items():
                scope_passes[name] += int(result['passed'])
            passed = all(result['passed'] for result in scopes.values())
            all_passes += int(passed)
            pair_count += 1
            evaluations.append({
                'reference_index': reference_index,
                'heldout_index': heldout_index,
                'scopes': scopes,
                'passed': passed,
            })
        folds.append({
            'heldout_index': heldout_index,
            'training_indexes': training_indexes,
            'training_envelopes': {
                'transition_state': state_envelope,
                'gameplay_events': gameplay_envelope,
                'recorder_transport_diagnostics': recorder_envelope,
            },
            'evaluations': evaluations,
        })

    return {
        'schema': 1,
        'diagnostic_only': True,
        'parity_claim': False,
        'clone_acceptance_changed': False,
        'trace_count': len(traces),
        'accept_at': accept_at,
        'pair_count': pair_count,
        'scope_pairs_passed': scope_passes,
        'all_scopes_pairs_passed': all_passes,
        'all_real_rek_pairs_accepted': all_passes == pair_count,
        'folds': folds,
    }
