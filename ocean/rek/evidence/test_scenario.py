"""Focused checks for deterministic scenarios and the non-live safety gate.

Run directly with::

    python ocean/rek/evidence/test_scenario.py
"""

import copy
import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scenario


def valid_schedule():
    return {
        'schema': 1,
        'ticks': 4,
        'actors': [0, 1],
        'channels': [0, 2],
        'segments': [
            {'actor': 0, 'channel': 0, 'start': 0, 'stop': 2, 'value': 0},
            {'actor': 0, 'channel': 0, 'start': 2, 'stop': 4, 'value': 0.5},
            {'actor': 0, 'channel': 2, 'start': 0, 'stop': 4, 'value': -1},
            {'actor': 1, 'channel': 0, 'start': 0, 'stop': 4, 'value': 0},
            {'actor': 1, 'channel': 2, 'start': 0, 'stop': 1, 'value': 1},
            {'actor': 1, 'channel': 2, 'start': 1, 'stop': 4, 'value': 0},
        ],
    }


class ScenarioTests(unittest.TestCase):
    def test_canonical_hash_is_stable_and_pinned(self):
        schedule = valid_schedule()
        reordered = {
            'segments': list(reversed(schedule['segments'])),
            'channels': [2, 0],
            'actors': [1, 0],
            'ticks': 4,
            'schema': 1,
        }
        reordered['segments'][0] = dict(
            reversed(list(reordered['segments'][0].items())))

        digest = scenario.scenario_sha256(schedule)
        self.assertEqual(digest, scenario.scenario_sha256(reordered))
        self.assertEqual(
            digest,
            '4af3e93dd03a1569fada7782060deb707844da47a5fbb09a11df6215c491b7f3')

    def test_schedule_expands_into_existing_trace_channel_shape(self):
        schedule = valid_schedule()
        self.assertEqual(
            scenario.trace_channels(schedule),
            ['cmd.0.0', 'cmd.0.2', 'cmd.1.0', 'cmd.1.2'])
        self.assertEqual(
            scenario.trace_frame(schedule, 2),
            {'cmd.0.0': 0.5, 'cmd.0.2': -1.0,
             'cmd.1.0': 0.0, 'cmd.1.2': 0.0})

    def test_gaps_are_rejected(self):
        schedule = valid_schedule()
        schedule['segments'][0]['stop'] = 1
        with self.assertRaisesRegex(scenario.ScenarioError, 'gap on actor 0 channel 0'):
            scenario.validate_schedule(schedule)

    def test_overlaps_are_rejected(self):
        schedule = valid_schedule()
        schedule['segments'][1]['start'] = 1
        with self.assertRaisesRegex(
                scenario.ScenarioError, 'overlap on actor 0 channel 0'):
            scenario.validate_schedule(schedule)

    def test_unknown_actors_and_channels_are_rejected(self):
        for field, value, message in (
                ('actor', 2, 'unknown actor 2'),
                ('channel', 1, 'unknown channel 1')):
            with self.subTest(field=field):
                schedule = valid_schedule()
                schedule['segments'][0][field] = value
                with self.assertRaisesRegex(scenario.ScenarioError, message):
                    scenario.validate_schedule(schedule)

    def test_only_numeric_command_channels_and_values_are_accepted(self):
        schedule = valid_schedule()
        schedule['channels'][0] = 'move'
        with self.assertRaisesRegex(scenario.ScenarioError, 'must be an integer'):
            scenario.validate_schedule(schedule)

        for value in (True, math.nan, math.inf):
            with self.subTest(value=value):
                schedule = valid_schedule()
                schedule['segments'][0]['value'] = value
                with self.assertRaises(scenario.ScenarioError):
                    scenario.validate_schedule(schedule)

    def test_safety_gate_accepts_only_explicit_single_bot_non_live_mode(self):
        accepted = {
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
        }
        self.assertEqual(scenario.require_safe_mode(accepted), accepted)

        unsafe = (
            ('wrong scene', {'scene': 'Arena'}),
            ('not solo', {'is_solo_arena': False}),
            ('unknown opponent', {'opponent_is_ai': None}),
            ('human opponent', {'opponent_human': True}),
            ('ranked', {'is_ranked_arena': True}),
            ('championship', {'championship': True}),
            ('server authority', {'network_is_server': True}),
            ('networked client', {'networked_client': True}),
            ('one fighter', {'fighter_count': 1}),
            ('extra fighter', {'fighter_count': 3}),
            ('AI controller absent', {'opponent_ai_controller_active': False}),
            ('live opponent', {'live_opponent': True}),
            ('matchmaking', {'matchmaking': True}),
            ('non-boolean unknown', {'live_opponent': 0}),
        )
        for name, change in unsafe:
            with self.subTest(name=name):
                observed = dict(accepted)
                observed.update(change)
                with self.assertRaises(scenario.UnsafeModeError):
                    scenario.require_safe_mode(observed)

        for key in accepted:
            with self.subTest(missing=key):
                observed = dict(accepted)
                del observed[key]
                with self.assertRaises(scenario.UnsafeModeError):
                    scenario.require_safe_mode(observed)

        observed = dict(accepted)
        observed['mode_verified'] = True
        with self.assertRaises(scenario.UnsafeModeError):
            scenario.require_safe_mode(observed)

        unknown = copy.deepcopy(accepted)
        unknown['matchmaking'] = None
        with self.assertRaises(scenario.UnsafeModeError):
            scenario.require_safe_mode(unknown)


if __name__ == '__main__':
    unittest.main(verbosity=2)
