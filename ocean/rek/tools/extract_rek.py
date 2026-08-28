"""Pull REK's real move roster out of the shipped Unity assets.

Run this on the machine with REK installed (Windows), not on the training box.

    pip install UnityPy
    python extract_rek.py --list                    # where is it installed?
    python extract_rek.py --survey                  # inventory what's in there
    python extract_rek.py --emit --out moves_generated.h

Driving this from WSL works, but reading hundreds of MB of assets across the
/mnt/c boundary is slow. UnityPy is pure Python and nothing here needs POSIX, so
pointing Windows Python at the same file is markedly faster if the wait grates.

Two phases on purpose. REK's asset schema is not documented and the field names
its move definitions use are unknown until someone looks, so `--survey` dumps an
inventory (script names, candidate move fields, animation clip lengths) to JSON
for inspection. `--emit` then turns that inventory into ocean/rek/moves_generated.h,
which moves.h picks up automatically in place of the placeholder table.

Nothing here writes into the game install; it is read-only against the assets.
"""

import argparse
import glob
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

# REK v0.0.119 fields two chassis. Their in-game names changed from the
# manufacturers' — L100 was the Unitree G1, H100 was the EngineAI T-800 — so a
# survey should match on both the new and old names to catch assets that still
# carry the original naming internally.
CHASSIS_HINTS = {
    'L100': ('l100', 'g1', 'unitree'),
    'H100': ('h100', 't800', 't-800', 'engineai'),
}


# Mirror of ocean/rek/chassis.h. Only needed to convert REK's absolute reach
# into the limb fraction moves.h stores; keep in sync if chassis.h changes.
CHASSIS_GEOMETRY = {
    'L100': {'arm_len': 0.410, 'leg_len': 0.698, 'body_radius': 0.280},
    'H100': {'arm_len': 0.567, 'leg_len': 1.028, 'body_radius': 0.343},
}

# Move names that mean a leg threw it, so reach scales off leg length rather
# than arm length. "roundhouse" and "axe"/"heel" are kicks whose names never say
# kick, which is exactly the kind of miss that silently shortens a move's reach.
LEG_MOVE_HINTS = ('kick', 'knee', 'stomp', 'sweep', 'leg', 'roundhouse',
                  'shin', 'heel', 'axe', 'push_off', 'teep', 'thrust')

# Bump when MoveDef's field layout changes. moves.h checks it.
MOVES_SCHEMA = 2


def chassis_of(name):
    """Which robot an asset belongs to, or None if it looks shared."""
    n = norm(name)
    for chassis, hints in CHASSIS_HINTS.items():
        if any(h.replace('-', '') in n for h in hints):
            return chassis
    return None

# REK's public Steam listing. A playtest or dev build will have its own id, so
# this is only a default — the Unity-marker tier below finds the install either
# way.
DEFAULT_APPID = '4582660'

# Places a Steam root can live. The /mnt/* entries matter because this is
# normally driven from WSL, where the game sits on a Windows drive: native
# Windows paths like C:\Program Files (x86)\Steam resolve to nothing there.
STEAM_ROOT_GLOBS = [
    '/mnt/*/Program Files (x86)/Steam',
    '/mnt/*/Program Files/Steam',
    '/mnt/*/Steam',
    '/mnt/*/SteamLibrary',
    '/mnt/*/Games/Steam',
    # Native Windows, if run under Windows Python rather than WSL.
    r'C:\Program Files (x86)\Steam',
    r'C:\Program Files\Steam',
    os.path.expanduser('~/.steam/steam'),
    os.path.expanduser('~/.local/share/Steam'),
    os.path.expanduser('~/Library/Application Support/Steam'),
]

UNITY_MARKERS = ('globalgamemanagers', 'resources.assets')


def steam_libraries():
    """Every Steam library root on this machine.

    Steam records extra libraries in steamapps/libraryfolders.vdf, and games are
    usually not on the same drive as Steam itself. Parsed with a regex over the
    quoted "path" values rather than a vdf dependency — the format is stable and
    this only needs the paths.
    """
    roots, seen = [], set()

    def add(p):
        try:
            rp = Path(p).resolve()
        except OSError:
            return
        if rp in seen or not rp.is_dir():
            return
        seen.add(rp)
        roots.append(rp)

    for pattern in STEAM_ROOT_GLOBS:
        for base in (glob.glob(pattern) if '*' in pattern else [pattern]):
            base = Path(base)
            if not base.is_dir():
                continue
            add(base)
            vdf = base / 'steamapps' / 'libraryfolders.vdf'
            if vdf.is_file():
                try:
                    text = vdf.read_text(encoding='utf-8', errors='replace')
                except OSError:
                    continue
                for raw in re.findall(r'"path"\s*"([^"]+)"', text):
                    add(windows_to_wsl(raw.replace('\\\\', '\\')))
    return roots


def windows_to_wsl(path):
    """C:\\Foo\\Bar -> /mnt/c/Foo/Bar when running under WSL, else unchanged."""
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', str(path))
    if m and Path('/mnt').is_dir():
        drive, rest = m.group(1).lower(), m.group(2).replace('\\', '/')
        candidate = Path(f'/mnt/{drive}') / rest
        if Path(f'/mnt/{drive}').is_dir():
            return str(candidate)
    return str(path)


def is_unity_app(path):
    """A Unity game ships <Name>_Data/globalgamemanagers next to the exe."""
    try:
        for data_dir in path.glob('*_Data'):
            if any((data_dir / m).exists() for m in UNITY_MARKERS):
                return True
    except OSError:
        pass
    return False


def candidate_installs(appid=DEFAULT_APPID):
    """Every plausible REK install, best guess first.

    Three tiers, each falling through to the next:
      1. appmanifest_<appid>.acf -> "installdir"  (the only reliable folder name)
      2. a common/ directory whose name looks like REK
      3. any Unity app in a Steam library, so an unknown appid still resolves
    """
    found, seen = [], set()

    def offer(path, why):
        p = Path(path)
        if not p.is_dir() or p in seen:
            return
        seen.add(p)
        found.append((p, why))

    for lib in steam_libraries():
        steamapps = lib / 'steamapps'
        if not steamapps.is_dir():
            steamapps = lib
        common = steamapps / 'common'

        manifest = steamapps / f'appmanifest_{appid}.acf'
        if manifest.is_file():
            try:
                text = manifest.read_text(encoding='utf-8', errors='replace')
                m = re.search(r'"installdir"\s*"([^"]+)"', text)
                if m:
                    offer(common / m.group(1), f'appmanifest_{appid}.acf')
            except OSError:
                pass

        if common.is_dir():
            try:
                entries = sorted(common.iterdir())
            except OSError:
                entries = []
            for entry in entries:
                if not entry.is_dir():
                    continue
                if 'rek' in re.sub(r'[^a-z]', '', entry.name.lower()):
                    offer(entry, 'name match')
            for entry in entries:
                if entry.is_dir() and is_unity_app(entry):
                    offer(entry, 'unity app')

    return found


def find_install(explicit=None, appid=DEFAULT_APPID):
    if explicit:
        p = Path(windows_to_wsl(explicit))
        if not p.exists():
            sys.exit(f'No such path: {p}')
        return p

    candidates = candidate_installs(appid)
    for path, why in candidates:
        if is_unity_app(path):
            print(f'Using REK install: {path}  ({why})')
            return path

    if candidates:
        print('Found these, but none look like a Unity app:', file=sys.stderr)
        for path, why in candidates:
            print(f'  {path}  ({why})', file=sys.stderr)
    sys.exit(
        'Could not find the REK install.\n'
        'List what is visible:   python extract_rek.py --list\n'
        'Or point at it:         python extract_rek.py --survey --path '
        '"/mnt/d/SteamLibrary/steamapps/common/REK"\n'
        'If it is a playtest build, pass its id too: --appid 1234567'
    )


def list_candidates(appid):
    libs = steam_libraries()
    print(f'Steam libraries ({len(libs)}):')
    for lib in libs:
        print(f'  {lib}')
    if not libs:
        print('  none found — pass --path explicitly')

    cands = candidate_installs(appid)
    print(f'\nInstall candidates ({len(cands)}):')
    for path, why in cands:
        print(f'  [{"unity" if is_unity_app(path) else "  -  "}] {path}  ({why})')
    if not cands:
        print('  none found')


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
                            'chassis': chassis_of(str(name)) or chassis_of(path.name),
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
    by_chassis = {}
    for mv in report['mono_behaviours']:
        by_chassis[mv.get('chassis')] = by_chassis.get(mv.get('chassis'), 0) + 1
    for k, v in sorted(by_chassis.items(), key=lambda kv: str(kv[0])):
        print(f'  {k or "(shared/unknown)"}: {v}')
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

        # Which limb throws it decides how reach scales onto each chassis.
        limb = 'LIMB_LEG' if any(k in norm(name) for k in LEG_MOVE_HINTS) else 'LIMB_ARM'

        # REK stores an absolute reach; moves.h stores a fraction of the limb, so
        # the move transfers to both chassis. Invert against the chassis the
        # asset belongs to (default L100) using the same geometry chassis.h uses.
        ref = mv.get('chassis') or 'L100'
        geom = CHASSIS_GEOMETRY.get(ref, CHASSIS_GEOMETRY['L100'])
        limb_len = geom['leg_len'] if limb == 'LIMB_LEG' else geom['arm_len']
        raw_reach = float(pick(f, 'reach', 0.7))
        extension = max(0.05, (raw_reach - geom['body_radius']) / limb_len)

        rows.append({
            'name': name,
            'limb': limb,
            'startup': startup,
            'active': active,
            'recovery': recovery,
            'extension': extension,
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
        '// Schema guard: moves.h refuses a header written for an older MoveDef',
        '// layout. A stale generated table would otherwise still compile, with',
        '// every field shifted one position — silently wrong rather than broken.',
        f'#define REK_MOVES_SCHEMA {MOVES_SCHEMA}',
        '',
        'static const MoveDef REK_MOVES_GENERATED[] = {',
        '    {"neutral", LIMB_ARM, 0, 0, 0, 0.00f, 0.00f, 0.0f, 0.00f, 0.00f, 0.00f, false},',
    ]
    for r in rows:
        lines.append(
            '    {{"{name}", {limb}, {startup}, {active}, {recovery}, {extension:.3f}f, '
            '{radius:.2f}f, {damage:.1f}f, {balance_cost:.2f}f, {balance_impact:.2f}f, '
            '{root_motion:.2f}f, {guard}}},'.format(
                guard='true' if r['guard_breaks'] else 'false', **r)
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
    ap.add_argument('--list', action='store_true',
        help='show every Steam library and install candidate, then exit')
    ap.add_argument('--survey', action='store_true', help='inventory the assets to JSON')
    ap.add_argument('--emit', action='store_true', help='write moves_generated.h from the survey')
    ap.add_argument('--appid', default=DEFAULT_APPID,
        help=f'Steam appid to look up (default {DEFAULT_APPID}; playtest builds differ)')
    ap.add_argument('--report', default='rek_survey.json', help='survey JSON path')
    ap.add_argument('--out', default='moves_generated.h', help='header to write with --emit')
    ap.add_argument('--tick-hz', type=float, default=30.0,
        help='must match REK_TICK_HZ in moves.h (default 30)')
    args = ap.parse_args()

    if args.list:
        list_candidates(args.appid)
        return

    if not args.survey and not args.emit:
        ap.error('pick one of --list, --survey or --emit')

    if args.survey:
        survey(find_install(args.path, args.appid), Path(args.report))
    if args.emit:
        emit(args.report, args.out, args.tick_hz)


if __name__ == '__main__':
    main()
