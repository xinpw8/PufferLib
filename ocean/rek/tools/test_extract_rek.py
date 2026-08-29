"""Checks for extract_rek.py that need no game install and no UnityPy.

The extractor's job is to turn REK's shipped animation clips into the move
table, and the parts that decide whether it gets that right are pure functions
over typetree shapes: where a clip's duration hides, which events bracket the
hit window, what counts as a combat clip, which rig it is skinned to. All of
that is testable here against realistic Unity data, so a mistake shows up
before someone runs the tool once on a machine this repo cannot reach.

    python ocean/rek/tools/test_extract_rek.py
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('extract_rek', HERE / 'extract_rek.py')
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

checks = []


def check(fn):
    checks.append(fn)
    return fn


# Unity 2022 humanoid clip: m_Length is absent and the duration is buried on
# m_MuscleClip, which is exactly the shape a fixed path would miss.
HUMANOID_CLIP = {
    'm_Name': 'Cross Punch',
    'm_SampleRate': 30.0,
    'm_MuscleClip': {
        'm_StartTime': 0.0,
        'm_StopTime': 0.8333,
        'm_AverageSpeed': {'x': 0.0, 'y': 0.0, 'z': 0.31, 'w': 0.0},
        'm_AverageAngularSpeed': 0.02,
    },
    'm_Events': [
        {'time': 0.25, 'functionName': 'HitboxOn', 'data': 'RightHand'},
        {'time': 0.36, 'functionName': 'HitboxOff', 'data': 'RightHand'},
    ],
}


@check
def duration_found_at_either_nesting_level():
    d, src = ex.clip_duration(HUMANOID_CLIP)
    assert abs(d - 0.8333) < 1e-4 and src == 'm_MuscleClip.m_StopTime', (d, src)
    # Generic/legacy clips populate m_Length, which wins when it is real.
    d, src = ex.clip_duration({'m_Length': 1.5, 'm_MuscleClip': {'m_StopTime': 9.0}})
    assert (d, src) == (1.5, 'm_Length'), (d, src)
    # A zero m_Length is Unity saying "not stored here", not a zero-length clip.
    d, src = ex.clip_duration({'m_Length': 0.0, 'm_MuscleClip': {'m_StopTime': 2.0}})
    assert (d, src) == (2.0, 'm_MuscleClip.m_StopTime'), (d, src)
    assert ex.clip_duration({'m_Name': 'x'}) == (None, None)


@check
def root_motion_comes_off_the_clip():
    speed = ex.as_vec(ex.deep_get(HUMANOID_CLIP, 'm_AverageSpeed'))
    assert speed == (0.0, 0.0, 0.31), speed
    assert ex.as_vec([1, 2, 3]) == (1.0, 2.0, 3.0)
    assert ex.as_vec(None) is None


@check
def hit_window_comes_from_events_when_they_exist():
    evs = ex.clip_events(HUMANOID_CLIP)
    assert [e['function'] for e in evs] == ['HitboxOn', 'HitboxOff'], evs
    clip = {'duration_s': 0.8333, 'events': evs}
    startup, active, recovery, why = ex.envelope(clip, 72.0)
    assert why == 'events' and (startup, active) == (18, 8), (startup, active, why)
    assert startup + active + recovery == 60, (startup, active, recovery)


@check
def tick_rate_is_the_unit_and_it_matters():
    # The same clip, read at two rates. This is why the fixed timestep has to
    # come out of the game before the table means anything.
    clip = {'duration_s': 0.8333, 'events': ex.clip_events(HUMANOID_CLIP)}
    assert ex.envelope(clip, 72.0)[:2] == (18, 8)
    assert ex.envelope(clip, 30.0)[:2] == (8, 3)


@check
def lone_contact_event_centres_the_window():
    clip = {'duration_s': 1.2,
            'events': [{'time': 0.62, 'function': 'OnImpact', 'string': ''}]}
    startup, active, _, why = ex.envelope(clip, 30.0)
    assert why == 'contact-event', why
    # Window straddles the contact frame rather than starting on it.
    assert startup < round(0.62 * 30) <= startup + active, (startup, active)


@check
def no_events_is_inferred_and_says_so():
    startup, active, recovery, why = ex.envelope({'duration_s': 0.9, 'events': []}, 30.0)
    assert why == 'inferred', why
    assert startup >= 0 and active >= 1 and recovery >= 1
    assert startup + active + recovery == 27, (startup, active, recovery)


@check
def combat_clips_are_told_from_the_rest_of_the_library():
    # A mocap library ships far more locomotion and reaction than combat, and
    # every one swept in would add a fake entry to the action head.
    for name in ('Jab', 'Cross Punch', 'Roundhouse Kick', 'Mma Kick', 'Punching',
                 'Uppercut', 'Front Kick'):
        assert ex.is_combat_clip(name), name
    for name in ('Walking', 'Idle', 'Standing React Large Gut', 'Getting Up',
                 'Death From Front', 'Left Turn', 'Victory Idle', 'Guard Stance'):
        assert not ex.is_combat_clip(name), name


@check
def kicks_scale_off_the_leg_not_the_arm():
    assert ex.limb_of('Roundhouse Kick') == 'LIMB_LEG'
    assert ex.limb_of('Knee Strike') == 'LIMB_LEG'
    assert ex.limb_of('Jab') == 'LIMB_ARM'


@check
def rig_names_the_source_library():
    lib, hits = ex.identify_library(
        ['mixamorig:Hips', 'mixamorig:LeftHand', 'Armature/mixamorig:Spine'])
    # One vote per bone: overlapping signatures must not inflate the count.
    assert (lib, hits) == ('Adobe Mixamo', {'Adobe Mixamo': 3}), (lib, hits)
    assert ex.identify_library(['Bip01 Pelvis', 'Bip01 L Hand'])[0].startswith('3ds Max Biped')
    assert ex.identify_library(['Hips', 'Spine'])[0] is None


@check
def hit_radius_comes_off_the_limb_that_throws_it():
    cols = [{'type': 'CapsuleCollider', 'owner': 'mixamorig:RightHand', 'radius': 0.08},
            {'type': 'CapsuleCollider', 'owner': 'mixamorig:LeftFoot', 'radius': 0.12},
            {'type': 'BoxCollider', 'owner': 'Torso', 'size': (0.4, 0.6, 0.25)}]
    assert ex.limb_radius(cols, 'LIMB_ARM') == 0.08
    assert ex.limb_radius(cols, 'LIMB_LEG') == 0.12
    assert ex.limb_radius([], 'LIMB_ARM') is None


@check
def walk_reaches_into_lists_of_dicts():
    # Unity buries almost everything one list deep; a dict-only walker sees none
    # of it.
    assert [w for k, _, w in ex.walk({'a': [{'b': {'c': 1}}]}) if k == 'c'] == ['a[0]/b/c']
    assert ex.deep_get({'a': [{'m_StopTime': 4.0}]}, 'm_StopTime') == 4.0


def _survey(**over):
    clips = [
        {'name': 'Jab', 'combat': True, 'limb_guess': 'LIMB_ARM', 'chassis': None,
         'duration_s': 0.7333, 'duration_source': 'm_Length', 'sample_rate': 30.0,
         'average_speed': [0.0, 0.0, 0.22], 'average_angular_speed': 0.0,
         'events': [{'time': 0.20, 'function': 'EnableHitbox', 'string': ''},
                    {'time': 0.30, 'function': 'DisableHitbox', 'string': ''}],
         'file': 'sharedassets0.assets'},
        {'name': 'Roundhouse Kick', 'combat': True, 'limb_guess': 'LIMB_LEG',
         'chassis': None, 'duration_s': 1.2, 'duration_source': 'm_Length',
         'sample_rate': 30.0, 'average_speed': [0.05, 0.0, 0.30],
         'average_angular_speed': 0.1,
         'events': [{'time': 0.62, 'function': 'OnImpact', 'string': ''}],
         'file': 'sharedassets0.assets'},
        # Duplicate name in a second bundle, and a locomotion clip. Neither
        # belongs in the emitted table.
        {'name': 'Jab', 'combat': True, 'limb_guess': 'LIMB_ARM', 'chassis': None,
         'duration_s': 0.7333, 'duration_source': 'm_Length', 'sample_rate': 30.0,
         'average_speed': None, 'average_angular_speed': None, 'events': [],
         'file': 'sharedassets1.assets'},
        {'name': 'Walking', 'combat': False, 'limb_guess': 'LIMB_LEG',
         'chassis': None, 'duration_s': 1.0, 'duration_source': 'm_Length',
         'sample_rate': 30.0, 'average_speed': [0, 0, 1.4],
         'average_angular_speed': 0, 'events': [], 'file': 'x.assets'},
    ]
    report = {
        'install': '/fake', 'unity_version': '2022.3.40f1', 'files_scanned': 3,
        'time': {'Fixed Timestep': 0.013888889, 'fixed_hz': 72.0},
        'physics': {}, 'rig': {'library': 'Adobe Mixamo'},
        'animation_clips': clips, 'animator_transitions': [],
        'colliders': [{'type': 'CapsuleCollider', 'owner': 'mixamorig:RightHand',
                       'radius': 0.085}],
        'mono_behaviours': [],
    }
    report.update(over)
    return report


def _emit(report, extra=()):
    with tempfile.TemporaryDirectory() as d:
        rp, out = Path(d) / 'survey.json', Path(d) / 'moves_generated.h'
        rp.write_text(json.dumps(report))
        proc = subprocess.run(
            [sys.executable, str(HERE / 'extract_rek.py'), '--emit',
             '--report', str(rp), '--out', str(out), *extra],
            capture_output=True, text=True)
        return proc, out.read_text() if out.exists() else ''


@check
def emit_uses_the_games_own_tick_rate():
    proc, header = _emit(_survey())
    assert proc.returncode == 0, proc.stderr
    assert '72 Hz' in proc.stdout, proc.stdout
    # 0.20 s of startup is 14 frames at 72 Hz. At the placeholder 30 it would be
    # 6, and every envelope in the table would be wrong by that ratio.
    assert '"Jab", LIMB_ARM, 14, 7,' in header, header


@check
def emit_dedupes_clips_and_drops_locomotion():
    _, header = _emit(_survey())
    assert header.count('"Jab"') == 1, header
    assert 'Walking' not in header


@check
def emit_records_where_every_value_came_from():
    _, header = _emit(_survey())
    assert 'timing=events' in header and 'timing=contact-event' in header
    assert 'root=clip' in header and 'radius=collider' in header
    # Reach at the contact frame is not recoverable from muscle curves. It has
    # to be marked, not quietly defaulted.
    assert 'reach=UNMEASURED' in header


@check
def emit_refuses_a_table_that_is_entirely_guesswork():
    report = _survey()
    for c in report['animation_clips']:
        c['events'] = []
    proc, _ = _emit(report)
    assert proc.returncode != 0, proc.stdout
    assert 'inferred' in proc.stdout + proc.stderr
    # ...unless asked for it explicitly, and then every row is tagged.
    proc, header = _emit(report, extra=['--allow-inferred'])
    assert proc.returncode == 0, proc.stderr
    rows = [l for l in header.split('\n') if l.startswith('    {"')]
    tagged = [l for l in rows if 'timing=inferred' in l]
    assert len(rows) == 3 and len(tagged) == 2, header  # neutral + 2 moves


@check
def emit_needs_a_frame_unit_from_somewhere():
    report = _survey()
    report['time'] = {}
    proc, _ = _emit(report)
    assert proc.returncode != 0 and 'no unit' in proc.stdout + proc.stderr


@check
def emitted_header_matches_the_schema_moves_h_expects():
    moves_h = (HERE.parent / 'moves.h').read_text()
    expected = int(moves_h.split('REK_MOVES_SCHEMA_EXPECTED')[1].split('\n')[0].strip())
    assert ex.MOVES_SCHEMA == expected, (ex.MOVES_SCHEMA, expected)
    _, header = _emit(_survey())
    assert f'#define REK_MOVES_SCHEMA {expected}' in header
    # Field count must match MoveDef, or C initialises the struct one position
    # shifted and the table is silently wrong rather than loudly broken.
    row = [l for l in header.split('\n') if '"Jab"' in l][0].split('//')[0]
    assert row.strip().rstrip(',').strip('{}').count(',') == 11, row


def main():
    failed = 0
    for fn in checks:
        try:
            fn()
            print(f'  ok    {fn.__name__.replace("_", " ")}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {fn.__name__.replace("_", " ")}: {e}')
    print(f'\n{len(checks) - failed}/{len(checks)} checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
