"""Pull REK's real move roster out of the shipped Unity assets.

Run this on the machine with REK installed (Windows), not on the training box.

    pip install UnityPy
    python extract_rek.py --survey                  # inventory what's in there
    python extract_rek.py --emit --out moves_generated.h

Two phases on purpose. REK's asset schema is not documented and the field names
its move definitions use are unknown until someone looks, so `--survey` dumps an
inventory (script names, candidate move fields, animation clip lengths) to JSON
for inspection. `--emit` then turns that inventory into ocean/rek/moves_generated.h,
which moves.h picks up automatically in place of the placeholder table.

Nothing here writes into the game install; it is read-only against the assets.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Field-name fragments worth capturing off a MonoBehaviour. Unity projects name
# these inconsistently, so match on fragments rather than exact keys.
FIELD_HINTS = {
    'startup': ('startup', 'windup', 'anticipation', 'charge'),
    'active': ('active', 'hitframe', 'hitwindow', 'strike'),
    'recovery': ('recovery', 'cooldown', 'endlag', 'retract'),
    'damage': ('damage', 'power', 'points', 'score'),
    'reach': ('reach', 'range', 'distance', 'length'),
    'radius': ('radius', 'size', 'extent', 'width'),
    'balance': ('balance', 'stagger', 'stability', 'poise', 'knockdown'),
    'root_motion': ('rootmotion', 'lunge', 'step', 'advance', 'displacement'),
    'guard': ('guard', 'block', 'unblockable', 'breaks'),
}

MOVE_NAME_HINTS = ('move', 'attack', 'strike', 'punch', 'kick', 'jab', 'hook',
                   'cross', 'uppercut', 'combo', 'ability', 'action')

DEFAULT_INSTALL_GLOBS = [
    r'C:\Program Files (x86)\Steam\steamapps\common\REK',
    r'C:\Program Files\Steam\steamapps\common\REK',
    r'D:\SteamLibrary\steamapps\common\REK',
    r'E:\SteamLibrary\steamapps\common\REK',
    os.path.expanduser('~/.steam/steam/steamapps/common/REK'),
    os.path.expanduser('~/Library/Application Support/Steam/steamapps/common/REK'),
]


def find_install(explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f'No such path: {p}')
        return p
    for cand in DEFAULT_INSTALL_GLOBS:
        p = Path(cand)
        if p.exists():
            return p
    sys.exit(
        'Could not find the REK install. Pass it explicitly:\n'
        '  python extract_rek.py --survey --path "C:\\...\\steamapps\\common\\REK"'
    )


def asset_files(root):
    """Every Unity container worth opening, in a stable order."""
    patterns = ('*.assets', '*.bundle', '*.unity3d', 'resources.assets',
                'level*', 'sharedassets*.assets', '*.dat')
    seen = []
    for pat in patterns:
        for path in sorted(root.rglob(pat)):
            if path.is_file() and path not in seen:
                seen.append(path)
    return seen


def norm(key):
    return re.sub(r'[^a-z0-9]', '', key.lower())


def classify_field(key):
    n = norm(key)
    for slot, hints in FIELD_HINTS.items():
        if any(h in n for h in hints):
            return slot
    return None


def looks_like_move(name):
    n = norm(name)
    return any(h in n for h in MOVE_NAME_HINTS)


def survey(root, out_path):
    try:
        import UnityPy
    except ImportError:
        sys.exit('UnityPy is required: pip install UnityPy')

    report = {
        'install': str(root),
        'unity_version': None,
        'mono_behaviours': [],
        'animation_clips': [],
        'colliders': [],
        'files_scanned': 0,
    }

    for path in asset_files(root):
        try:
            bundle = UnityPy.load(str(path))
        except Exception:
            continue
        report['files_scanned'] += 1

        for obj in bundle.objects:
            if report['unity_version'] is None:
                report['unity_version'] = getattr(obj.assets_file, 'unity_version', None)

            try:
                if obj.type.name == 'AnimationClip':
                    data = obj.read()
                    name = getattr(data, 'm_Name', '') or getattr(data, 'name', '')
                    length = getattr(data, 'm_MuscleClipSize', None)
                    # Unity exposes clip length in different places by version.
                    for attr in ('m_MuscleClip', 'm_ClipBindingConstant'):
                        if length is None:
                            length = getattr(data, attr, None)
                    report['animation_clips'].append({
                        'name': str(name),
                        'length_s': getattr(data, 'm_Length', None),
                        'sample_rate': getattr(data, 'm_SampleRate', None),
                        'file': path.name,
                    })

                elif obj.type.name == 'MonoBehaviour':
                    data = obj.read()
                    tree = None
                    if getattr(obj, 'serialized_type', None) and obj.serialized_type.nodes:
                        tree = obj.read_typetree()
                    if not isinstance(tree, dict):
                        continue
                    name = tree.get('m_Name') or getattr(data, 'm_Name', '')
                    fields = {}
                    for key, value in tree.items():
                        if not isinstance(value, (int, float, bool)):
                            continue
                        slot = classify_field(key)
                        if slot:
                            fields.setdefault(slot, []).append({'key': key, 'value': value})
                    # Keep anything that either reads like a move by name or
                    # carries enough timing fields to be one.
                    if fields and (looks_like_move(str(name)) or len(fields) >= 3):
                        report['mono_behaviours'].append({
                            'name': str(name),
                            'fields': fields,
                            'raw_keys': sorted(k for k in tree.keys()),
                            'file': path.name,
                        })

                elif obj.type.name in ('CapsuleCollider', 'BoxCollider', 'SphereCollider'):
                    tree = obj.read_typetree() if getattr(obj, 'serialized_type', None) else None
                    if isinstance(tree, dict):
                        report['colliders'].append({
                            'type': obj.type.name,
                            'radius': tree.get('m_Radius'),
                            'height': tree.get('m_Height'),
                            'center': tree.get('m_Center'),
                            'file': path.name,
                        })
            except Exception:
                continue

    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f'Scanned {report["files_scanned"]} asset files under {root}')
    print(f'Unity version : {report["unity_version"]}')
    print(f'Move candidates: {len(report["mono_behaviours"])}')
    print(f'Animation clips: {len(report["animation_clips"])}')
    print(f'Colliders      : {len(report["colliders"])}')
    print(f'\nWrote {out_path}')
    if not report['mono_behaviours']:
        print(
            '\nNo move-like MonoBehaviours matched. REK may keep its move table in\n'
            'IL2CPP metadata rather than serialized assets. Next step is\n'
            'AssetRipper or Il2CppDumper over GameAssembly.dll; send the JSON and\n'
            'the dumped class list and the field mapping can be pinned from there.'
        )


def pick(fields, slot, default):
    """First value classified into `slot`, else the default."""
    entries = fields.get(slot)
    if not entries:
        return default
    return entries[0]['value']


def emit(report_path, out_path, tick_hz):
    report = json.loads(Path(report_path).read_text())
    moves = report.get('mono_behaviours', [])
    if not moves:
        sys.exit(f'{report_path} has no move candidates — run --survey first.')

    clip_len = {c['name']: c.get('length_s') for c in report.get('animation_clips', [])}

    rows = []
    for mv in moves:
        f = mv['fields']
        name = re.sub(r'[^A-Za-z0-9_]', '_', mv['name'])[:23] or 'move'

        startup = pick(f, 'startup', 4)
        active = pick(f, 'active', 3)
        recovery = pick(f, 'recovery', 8)

        # Values under ~2 are almost certainly seconds, not frames.
        def to_frames(v, fallback):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return fallback
            return max(1, int(round(v * tick_hz))) if v < 2.0 else int(round(v))

        startup = to_frames(startup, 4)
        active = to_frames(active, 3)
        recovery = to_frames(recovery, 8)

        # If the move has an animation clip, prefer its length as the total and
        # rescale the phases to fit rather than trusting three unrelated fields.
        length_s = clip_len.get(mv['name'])
        if length_s:
            total = max(3, int(round(float(length_s) * tick_hz)))
            phase_sum = startup + active + recovery
            if phase_sum > 0:
                k = total / phase_sum
                startup = max(1, int(round(startup * k)))
                active = max(1, int(round(active * k)))
                recovery = max(1, total - startup - active)

        rows.append({
            'name': name,
            'startup': startup,
            'active': active,
            'recovery': recovery,
            'reach': float(pick(f, 'reach', 0.7)),
            'radius': float(pick(f, 'radius', 0.24)),
            'damage': float(pick(f, 'damage', 1.0)),
            'balance_cost': 0.02 * startup / 4.0,
            'balance_impact': float(pick(f, 'balance', 0.2)),
            'root_motion': float(pick(f, 'root_motion', 0.1)),
            'guard_breaks': bool(pick(f, 'guard', 0)),
        })

    lines = [
        '// Generated by ocean/rek/tools/extract_rek.py from the shipped REK',
        '// Unity assets. Do not edit by hand — re-run the extractor instead.',
        f'// Source survey: {Path(report_path).name}',
        f'// Frame counts are at {tick_hz:g} Hz, matching REK_TICK_HZ.',
        '',
        '#pragma once',
        '',
        'static const MoveDef REK_MOVES_GENERATED[] = {',
        '    {"neutral", 0, 0, 0, 0.00f, 0.00f, 0.0f, 0.00f, 0.00f, 0.00f, false},',
    ]
    for r in rows:
        lines.append(
            '    {{"{name}", {startup}, {active}, {recovery}, {reach:.2f}f, {radius:.2f}f, '
            '{damage:.1f}f, {balance_cost:.2f}f, {balance_impact:.2f}f, {root_motion:.2f}f, '
            '{guard}}},'.format(guard='true' if r['guard_breaks'] else 'false', **r)
        )
    lines += [
        '};',
        '',
        '#define REK_NUM_MOVES_GENERATED '
        '((int)(sizeof(REK_MOVES_GENERATED) / sizeof(REK_MOVES_GENERATED[0])))',
        '',
    ]

    Path(out_path).write_text('\n'.join(lines))
    print(f'Wrote {out_path} with {len(rows)} moves (plus neutral).')
    print('Rebuild so moves.h picks it up:  ./build.sh rek')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', help='REK install directory (auto-detected if omitted)')
    ap.add_argument('--survey', action='store_true', help='inventory the assets to JSON')
    ap.add_argument('--emit', action='store_true', help='write moves_generated.h from the survey')
    ap.add_argument('--report', default='rek_survey.json', help='survey JSON path')
    ap.add_argument('--out', default='moves_generated.h', help='header to write with --emit')
    ap.add_argument('--tick-hz', type=float, default=30.0,
        help='must match REK_TICK_HZ in moves.h (default 30)')
    args = ap.parse_args()

    if not args.survey and not args.emit:
        ap.error('pick one of --survey or --emit')

    if args.survey:
        survey(find_install(args.path), Path(args.report))
    if args.emit:
        emit(args.report, args.out, args.tick_hz)


if __name__ == '__main__':
    main()
