"""Deterministic two-actor command schedules with a fail-closed mode gate.

This module assigns no meaning to a command channel. Actors are the numeric
identifiers 0 and 1, channels are non-negative integer identifiers, and values
are finite numbers. The adapter that records REK and the candidate clone must
establish the meaning and provenance of each channel independently.

A schedule is a JSON-shaped mapping with this schema::

    {
      "schema": 1,
      "ticks": 1,
      "actors": [0, 1],
      "channels": [0],
      "segments": [
        {"actor": 0, "channel": 0, "start": 0, "stop": 1, "value": 0.0},
        {"actor": 1, "channel": 0, "start": 0, "stop": 1, "value": 0.0}
      ]
    }

Segments use half-open tick ranges. Every actor/channel track must cover the
entire schedule exactly once. Gaps and overlaps are errors, which prevents an
executor from filling an unspecified interval with an inferred command.

The safety gate consumes observed runtime facts. It does not supply defaults or
infer a safe state. Only the shipped ArenaRekSingleBot scene, with local
non-server authority, two fighters, and an active AI controller bound to the
non-human opponent slot, is accepted.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType


SCHEMA = 1
ACTORS = (0, 1)
SAFE_MODE = MappingProxyType({
    'scene': 'ArenaRekSingleBot',
    'is_solo_arena': True,
    'opponent_is_ai': True,
    'opponent_human': False,
    'is_ranked_arena': False,
    'championship': False,
    'network_is_server': False,
    'networked_client': False,
    'fighter_count': 2,
    'opponent_ai_controller_active': True,
    'live_opponent': False,
    'matchmaking': False,
})

_SCHEDULE_KEYS = frozenset(('schema', 'ticks', 'actors', 'channels', 'segments'))
_SEGMENT_KEYS = frozenset(('actor', 'channel', 'start', 'stop', 'value'))


class ScenarioError(ValueError):
    """The schedule is ambiguous, incomplete, or outside the numeric schema."""


class UnsafeModeError(RuntimeError):
    """Observed runtime facts do not prove the allowed non-live mode."""


def _exact_keys(mapping, expected, label):
    actual = set(mapping)
    missing = sorted(expected - actual, key=repr)
    unknown = sorted(actual - expected, key=repr)
    if missing or unknown:
        raise ScenarioError(
            f'{label} keys do not match the schema: missing={missing}, '
            f'unknown={unknown}')


def _integer(value, label, minimum=None):
    if type(value) is not int:
        raise ScenarioError(f'{label} must be an integer, got {value!r}')
    if minimum is not None and value < minimum:
        raise ScenarioError(f'{label} must be >= {minimum}, got {value}')
    return value


def _number(value, label):
    if type(value) not in (int, float):
        raise ScenarioError(f'{label} must be a numeric command, got {value!r}')
    value = float(value)
    if not math.isfinite(value):
        raise ScenarioError(f'{label} must be finite, got {value!r}')
    # JSON distinguishes -0.0 from 0.0 even though the trace's float64 command
    # does not. Normalising zero keeps the semantic schedule hash canonical.
    return 0.0 if value == 0.0 else value


def validate_schedule(document):
    """Validate and return the schedule in its canonical, JSON-safe form."""
    if not isinstance(document, Mapping):
        raise ScenarioError('schedule must be a mapping')
    _exact_keys(document, _SCHEDULE_KEYS, 'schedule')

    schema = _integer(document['schema'], 'schema')
    if schema != SCHEMA:
        raise ScenarioError(f'unsupported scenario schema {schema}; expected {SCHEMA}')
    ticks = _integer(document['ticks'], 'ticks', minimum=1)

    actors = document['actors']
    if not isinstance(actors, (list, tuple)):
        raise ScenarioError('actors must be a list containing 0 and 1')
    for index, actor in enumerate(actors):
        _integer(actor, f'actors[{index}]', minimum=0)
    if len(actors) != len(set(actors)):
        raise ScenarioError('actors contains a duplicate identifier')
    if set(actors) != set(ACTORS):
        raise ScenarioError(f'actors must be exactly {list(ACTORS)}, got {list(actors)!r}')

    channels = document['channels']
    if not isinstance(channels, (list, tuple)) or not channels:
        raise ScenarioError('channels must be a non-empty list of integers')
    for index, channel in enumerate(channels):
        _integer(channel, f'channels[{index}]', minimum=0)
    if len(channels) != len(set(channels)):
        raise ScenarioError('channels contains a duplicate identifier')
    channels = sorted(channels)

    segments = document['segments']
    if not isinstance(segments, (list, tuple)):
        raise ScenarioError('segments must be a list')

    normal = []
    tracks = {(actor, channel): [] for actor in ACTORS for channel in channels}
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ScenarioError(f'segments[{index}] must be a mapping')
        _exact_keys(segment, _SEGMENT_KEYS, f'segments[{index}]')

        actor = _integer(segment['actor'], f'segments[{index}].actor', minimum=0)
        channel = _integer(
            segment['channel'], f'segments[{index}].channel', minimum=0)
        if actor not in ACTORS:
            raise ScenarioError(f'segments[{index}] references unknown actor {actor}')
        if channel not in channels:
            raise ScenarioError(
                f'segments[{index}] references unknown channel {channel}')

        start = _integer(segment['start'], f'segments[{index}].start', minimum=0)
        stop = _integer(segment['stop'], f'segments[{index}].stop', minimum=0)
        if start >= stop:
            raise ScenarioError(
                f'segments[{index}] must have start < stop, got [{start}, {stop})')
        if stop > ticks:
            raise ScenarioError(
                f'segments[{index}] stops at {stop}, after schedule tick {ticks}')

        item = {
            'actor': actor,
            'channel': channel,
            'start': start,
            'stop': stop,
            'value': _number(segment['value'], f'segments[{index}].value'),
        }
        normal.append(item)
        tracks[(actor, channel)].append(item)

    for (actor, channel), track in sorted(tracks.items()):
        cursor = 0
        for segment in sorted(track, key=lambda item: (item['start'], item['stop'])):
            start = segment['start']
            if start > cursor:
                raise ScenarioError(
                    f'gap on actor {actor} channel {channel}: [{cursor}, {start})')
            if start < cursor:
                raise ScenarioError(
                    f'overlap on actor {actor} channel {channel} at tick {start}')
            cursor = segment['stop']
        if cursor < ticks:
            raise ScenarioError(
                f'gap on actor {actor} channel {channel}: [{cursor}, {ticks})')

    normal.sort(key=lambda item: (
        item['actor'], item['channel'], item['start'], item['stop'], item['value']))
    return {
        'schema': SCHEMA,
        'ticks': ticks,
        'actors': list(ACTORS),
        'channels': channels,
        'segments': normal,
    }


def canonical_bytes(document):
    """Canonical UTF-8 JSON bytes used as the schedule identity."""
    normal = validate_schedule(document)
    return json.dumps(
        normal, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode('utf-8')


def scenario_sha256(document):
    """Return the lowercase SHA-256 digest of the canonical schedule."""
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def commands_at(document, tick):
    """Return ``{actor: {channel: value}}`` for one validated schedule tick."""
    normal = validate_schedule(document)
    tick = _integer(tick, 'tick', minimum=0)
    if tick >= normal['ticks']:
        raise ScenarioError(
            f'tick {tick} is outside schedule [0, {normal["ticks"]})')

    commands = {actor: {} for actor in ACTORS}
    for segment in normal['segments']:
        if segment['start'] <= tick < segment['stop']:
            commands[segment['actor']][segment['channel']] = segment['value']
    return commands


def trace_channels(document):
    """Channel names for the existing REK trace format, in canonical order."""
    normal = validate_schedule(document)
    return [
        f'cmd.{actor}.{channel}'
        for actor in normal['actors']
        for channel in normal['channels']
    ]


def trace_frame(document, tick):
    """Flatten one tick into values accepted by ``trace.TraceWriter.append``."""
    commands = commands_at(document, tick)
    return {
        f'cmd.{actor}.{channel}': value
        for actor, channels in commands.items()
        for channel, value in channels.items()
    }


def require_safe_mode(observed):
    """Raise unless all observed facts explicitly identify the allowed mode."""
    if not isinstance(observed, Mapping):
        raise UnsafeModeError('mode facts must be an explicit mapping')

    actual_keys = set(observed)
    expected_keys = set(SAFE_MODE)
    missing = sorted(expected_keys - actual_keys, key=repr)
    unknown = sorted(actual_keys - expected_keys, key=repr)
    mismatched = {}
    for key, expected in SAFE_MODE.items():
        if key not in observed:
            continue
        actual = observed[key]
        if type(actual) is not type(expected) or actual != expected:
            mismatched[key] = {'expected': expected, 'observed': actual}
    if missing or unknown or mismatched:
        raise UnsafeModeError(
            'runtime mode is not proven safe: '
            f'missing={missing}, unknown={unknown}, mismatched={mismatched}')
    return dict(SAFE_MODE)
