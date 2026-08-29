"""Pull REK's real move roster out of the shipped Unity assets.

Run this on the machine with REK installed (Windows), not on the training box.

    pip install UnityPy
    python extract_rek.py --list                    # where is it installed?
    python extract_rek.py --survey                  # inventory what's in there
    python extract_rek.py --emit --out moves_generated.h

REK's moves are canned animations taken from an off-the-shelf mocap library, so
the clips are the authority here. A move's timing is not a designer's number in
a script field — it is the length of the clip and the hit events fired along it.
That is what --survey goes after first, along with the handful of other values
the env would otherwise have to invent: the fixed timestep (which decides what a
"frame" even means), the limb colliders (real hit volumes), the rig fingerprint
(which names the source library, so the same clips can be checked at full
fidelity without going back to the install), and per-move script fields for the
things animation cannot express — damage, balance, guard-break.

Two phases on purpose. REK's asset schema is not documented, so --survey dumps
an inventory to JSON for inspection and --emit turns it into
ocean/rek/moves_generated.h, which moves.h picks up in place of the placeholder
table. Every emitted row carries its own provenance, and --emit refuses to write
a table whose timing is entirely guesswork.

Driving this from WSL works, but reading hundreds of MB of assets across the
/mnt/c boundary is slow. UnityPy is pure Python and nothing here needs POSIX, so
pointing Windows Python at the same file is markedly faster if the wait grates.

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

# --- Prebuilt animation libraries -------------------------------------------
#
# REK's moves are canned clips taken from an off-the-shelf mocap library, not
# hand-keyed and not generated. That matters twice over. It means the clip is
# the authority on a move's timing, and it means that once the library is
# identified the same clips can be inspected at the source, at full fidelity,
# without going back to the install.
#
# Rigs are the reliable fingerprint: every library skins to its own skeleton and
# the bone names survive into the shipped Avatar's transform table.
RIG_SIGNATURES = {
    'mixamorig': 'Adobe Mixamo',
    'ccbase': 'Reallusion Character Creator / ActorCore',
    'bip01': '3ds Max Biped (Kubold Animset Pro and similar)',
    'rokoko': 'Rokoko',
    'mocaponline': 'MocapOnline',
    'ue4mannequin': 'Epic mannequin rig',
    'thighl': 'Epic mannequin rig',
}

# Animation events that bracket the frames a strike can actually connect on.
# Unity projects name these freely, so match fragments. The `on` list opens the
# hit window and the `off` list closes it.
EVENT_ON_HINTS = ('enablehit', 'hiton', 'hitboxon', 'openhit', 'starthit',
                  'attackstart', 'damageon', 'colliderOn'.lower(), 'activatehit',
                  'weaponon', 'begincontact', 'hitstart')
EVENT_OFF_HINTS = ('disablehit', 'hitoff', 'hitboxoff', 'closehit', 'endhit',
                   'attackend', 'damageoff', 'colliderOff'.lower(), 'deactivatehit',
                   'weaponoff', 'endcontact', 'hitend')
# A single point event marking the contact frame rather than a window.
EVENT_POINT_HINTS = ('hit', 'impact', 'contact', 'strike', 'damage', 'connect')

# Fallback envelope when a clip carries no hit events, as fractions of clip
# length. Taken from how a struck mocap clip is shaped rather than from REK:
# the limb accelerates out, contacts near the two-thirds mark, and the rest is
# retraction. Anything emitted from this is tagged `inferred` and counted, so a
# table built mostly on guesses cannot pass itself off as measured.
INFERRED_CONTACT_FRAC = 0.62
INFERRED_ACTIVE_FRAC = 0.10

# Where in a Unity settings object each value hides. Searched recursively by key
# so the reader survives the schema drift between Unity versions.
TIME_KEYS = ('Fixed Timestep', 'm_FixedTimestep', 'Maximum Allowed Timestep',
             'm_MaximumTimestep', 'm_TimeScale')
PHYSICS_KEYS = ('m_Gravity', 'm_DefaultSolverIterations', 'm_DefaultContactOffset',
                'm_BounceThreshold', 'm_SleepThreshold', 'm_DefaultMaxAngularSpeed')


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


# Clips that are not moves. A mocap library ships far more locomotion and
# reaction than combat, and every one of those swept into the table would add a
# fake entry to the action head.
EXCLUDE_NAME_HINTS = ('idle', 'walk', 'run', 'jog', 'turn', 'strafe', 'death',
                      'die', 'getup', 'get_up', 'stand', 'fall', 'knock',
                      'stagger', 'flinch', 'react', 'block', 'guard', 'dodge',
                      'victory', 'defeat', 'taunt', 'intro', 'outro', 'pose',
                      'locomotion', 'jump', 'land', 'crouch', 'tpose', 'apose')

# Collider owners that carry a strike. Used to read a move's hit radius off the
# real limb volume instead of guessing it.
ARM_COLLIDER_HINTS = ('hand', 'fist', 'forearm', 'lowerarm', 'wrist', 'glove')
LEG_COLLIDER_HINTS = ('foot', 'toe', 'shin', 'calf', 'lowerleg', 'ankle')


def walk(node, path=''):
    """Every (key, value, path) in a nested typetree, depth first."""
    if isinstance(node, dict):
        for k, v in node.items():
            here = f'{path}/{k}' if path else str(k)
            yield str(k), v, here
            yield from walk(v, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            here = f'{path}[{i}]'
            yield from walk(v, here)


def deep_get(tree, key):
    """First value stored under `key` at any depth, else None.

    Unity moves fields between nesting levels across versions — a humanoid
    clip's duration lives on m_MuscleClip in one release and at the top level in
    another — so searching by name beats hardcoding a path that will rot.
    """
    for k, v, _ in walk(tree):
        if k == key:
            return v
    return None


def as_vec(v):
    """Unity vector -> (x, y, z), whatever shape UnityPy handed back."""
    if isinstance(v, dict):
        return (float(v.get('x', 0.0)), float(v.get('y', 0.0)), float(v.get('z', 0.0)))
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return tuple(float(c) for c in v[:3])
    return None


def clip_duration(tree):
    """Clip length in seconds, and where the number came from."""
    length = deep_get(tree, 'm_Length')
    try:
        if length is not None and float(length) > 0.0:
            return float(length), 'm_Length'
    except (TypeError, ValueError):
        pass
    stop, start = deep_get(tree, 'm_StopTime'), deep_get(tree, 'm_StartTime')
    try:
        if stop is not None:
            d = float(stop) - float(start or 0.0)
            if d > 0.0:
                return d, 'm_MuscleClip.m_StopTime'
    except (TypeError, ValueError):
        pass
    return None, None


def clip_events(tree):
    """Animation events on a clip, sorted by time."""
    raw = deep_get(tree, 'm_Events')
    if not isinstance(raw, list):
        return []
    out = []
    for ev in raw:
        if not isinstance(ev, dict):
            continue
        out.append({
            'time': float(ev.get('time', 0.0) or 0.0),
            'function': str(ev.get('functionName', '') or ''),
            'string': str(ev.get('data', '') or ''),
            'float': ev.get('floatParameter'),
            'int': ev.get('intParameter'),
        })
    return sorted(out, key=lambda e: e['time'])


def limb_of(name):
    return 'LIMB_LEG' if any(h in norm(name) for h in LEG_MOVE_HINTS) else 'LIMB_ARM'


def is_combat_clip(name):
    n = norm(name)
    if any(h in n for h in EXCLUDE_NAME_HINTS):
        return False
    return any(h in n for h in MOVE_NAME_HINTS) or any(h in n for h in LEG_MOVE_HINTS)


def identify_library(bone_paths):
    """Name the prebuilt animation library from the rig it is skinned to."""
    hits = {}
    for p in bone_paths:
        n = norm(p)
        # One vote per bone. Several signatures can be substrings of the same
        # name, and double-counting them would make a rig look better attested
        # than it is.
        for sig, lib in RIG_SIGNATURES.items():
            if sig in n:
                hits[lib] = hits.get(lib, 0) + 1
                break
    if not hits:
        return None, hits
    return max(hits.items(), key=lambda kv: kv[1])[0], hits


def survey(root, out_path):
    """Inventory everything the move table is derived from.

    The clips are the point. REK's moves are canned animations out of a mocap
    library, so a move's timing is not a designer's number sitting in a
    MonoBehaviour — it is the length of the clip and the events fired along it.
    Physics settings, the rig fingerprint and the limb colliders come along
    because each pins down a constant that is otherwise invented.
    """
    try:
        import UnityPy
    except ImportError:
        sys.exit('UnityPy is required: pip install UnityPy')

    report = {
        'install': str(root),
        'unity_version': None,
        'files_scanned': 0,
        'time': {},
        'physics': {},
        'rig': {'library': None, 'signature_hits': {}, 'sample_bones': []},
        'animation_clips': [],
        'animator_transitions': [],
        'colliders': [],
        'mono_behaviours': [],
    }
    bone_paths = set()

    for path in asset_files(root):
        try:
            bundle = UnityPy.load(str(path))
        except Exception:
            continue
        report['files_scanned'] += 1

        for obj in bundle.objects:
            if report['unity_version'] is None:
                report['unity_version'] = getattr(obj.assets_file, 'unity_version', None)

            kind = obj.type.name
            try:
                tree = None
                if kind in ('AnimationClip', 'Avatar', 'AnimatorController',
                            'TimeManager', 'PhysicsManager', 'MonoBehaviour',
                            'CapsuleCollider', 'BoxCollider', 'SphereCollider'):
                    tree = obj.read_typetree()
                if not isinstance(tree, dict):
                    continue

                if kind == 'TimeManager':
                    for k, v, _ in walk(tree):
                        if k in TIME_KEYS:
                            report['time'][k] = v

                elif kind == 'PhysicsManager':
                    for k, v, _ in walk(tree):
                        if k in PHYSICS_KEYS:
                            report['physics'][k] = as_vec(v) or v

                elif kind == 'Avatar':
                    # m_TOS maps a transform-path hash to the path itself, which
                    # is the only place the rig's bone names survive into a
                    # build. This is what identifies the source library.
                    tos = deep_get(tree, 'm_TOS')
                    if isinstance(tos, list):
                        for entry in tos:
                            if isinstance(entry, dict):
                                v = entry.get('second')
                                if isinstance(v, str):
                                    bone_paths.add(v)
                            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                                if isinstance(entry[1], str):
                                    bone_paths.add(entry[1])

                elif kind == 'AnimationClip':
                    name = str(tree.get('m_Name', '') or '')
                    duration, duration_src = clip_duration(tree)
                    speed = as_vec(deep_get(tree, 'm_AverageSpeed'))
                    report['animation_clips'].append({
                        'name': name,
                        'combat': is_combat_clip(name),
                        'limb_guess': limb_of(name),
                        'chassis': chassis_of(name) or chassis_of(path.name),
                        'duration_s': duration,
                        'duration_source': duration_src,
                        'sample_rate': deep_get(tree, 'm_SampleRate'),
                        'average_speed': speed,
                        'average_angular_speed': deep_get(tree, 'm_AverageAngularSpeed'),
                        'events': clip_events(tree),
                        'file': path.name,
                    })

                elif kind == 'AnimatorController':
                    # State speed multipliers and transition exit times decide
                    # how much of a clip a pilot is actually committed to, which
                    # is the difference between clip length and recovery.
                    for k, v, where in walk(tree):
                        if k == 'm_TransitionDuration':
                            report['animator_transitions'].append({
                                'controller': str(tree.get('m_Name', '') or ''),
                                'path': where,
                                'duration': v,
                            })
                        elif k == 'm_ExitTime' and report['animator_transitions']:
                            report['animator_transitions'][-1]['exit_time'] = v

                elif kind in ('CapsuleCollider', 'BoxCollider', 'SphereCollider'):
                    owner = ''
                    try:
                        go = obj.read().m_GameObject.read()
                        owner = str(getattr(go, 'm_Name', '') or '')
                    except Exception:
                        pass
                    report['colliders'].append({
                        'type': kind,
                        'owner': owner,
                        'radius': tree.get('m_Radius'),
                        'height': tree.get('m_Height'),
                        'center': as_vec(tree.get('m_Center')),
                        'size': as_vec(tree.get('m_Size')),
                        'file': path.name,
                    })

                elif kind == 'MonoBehaviour':
                    # Secondary now: with canned clips the timing lives in the
                    # animation, but per-move damage, balance and guard flags
                    # still sit in script fields.
                    name = tree.get('m_Name') or ''
                    fields = {}
                    for key, value in tree.items():
                        if isinstance(value, (int, float, bool)):
                            slot = classify_field(key)
                            if slot:
                                fields.setdefault(slot, []).append(
                                    {'key': key, 'value': value})
                    if fields and (looks_like_move(str(name)) or len(fields) >= 3):
                        report['mono_behaviours'].append({
                            'name': str(name),
                            'chassis': chassis_of(str(name)) or chassis_of(path.name),
                            'fields': fields,
                            'raw_keys': sorted(tree.keys()),
                            'file': path.name,
                        })
            except Exception:
                continue

    library, hits = identify_library(bone_paths)
    report['rig']['library'] = library
    report['rig']['signature_hits'] = hits
    report['rig']['sample_bones'] = sorted(bone_paths)[:40]

    # Fixed timestep settles REK_TICK_HZ. Every frame count in moves.h is in
    # this unit, so getting it wrong scales the whole table.
    for key in ('Fixed Timestep', 'm_FixedTimestep'):
        try:
            step = float(report['time'][key])
            if step > 0:
                report['time']['fixed_hz'] = round(1.0 / step, 4)
                break
        except (KeyError, TypeError, ValueError):
            continue

    out_path.write_text(json.dumps(report, indent=2, default=str))

    clips = report['animation_clips']
    combat = [c for c in clips if c['combat']]
    with_events = [c for c in combat if c['events']]
    print(f'Scanned {report["files_scanned"]} asset files under {root}')
    print(f'Unity version  : {report["unity_version"]}')
    print(f'Fixed timestep : {report["time"].get("Fixed Timestep") or report["time"].get("m_FixedTimestep")} '
          f'({report["time"].get("fixed_hz", "?")} Hz)  <- REK_TICK_HZ')
    print(f'Rig / library  : {library or "unidentified"}  {hits or ""}')
    print(f'Animation clips: {len(clips)} total, {len(combat)} look like moves, '
          f'{len(with_events)} carry hit events')
    print(f'Colliders      : {len(report["colliders"])}')
    print(f'Script fields  : {len(report["mono_behaviours"])} move-like MonoBehaviours')
    print(f'\nWrote {out_path}')

    if not combat:
        print('\nNo combat clips matched by name. Send the JSON: the clip list is '
              'in there and the name filter can be widened to REK\'s own naming.')
    elif not with_events:
        print('\nNo hit events on any combat clip. Active windows will be inferred '
              'from clip shape and tagged as such; the parity harness has to pin '
              'them down by observation.')


def pick(fields, slot, default):
    """First value classified into `slot`, else the default."""
    entries = fields.get(slot)
    if not entries:
        return default
    return entries[0]['value']


def envelope(clip, tick_hz):
    """Split a clip into startup / active / recovery frames.

    The hit window comes from the clip's own animation events when they exist —
    an enable/disable pair brackets it exactly, a lone contact event marks its
    centre. With neither, it falls back to the shape of a struck mocap clip and
    says so, because a guess that reports itself is recoverable and one that
    does not is what put invented numbers in this table the first time.
    """
    total = max(3, int(round(clip['duration_s'] * tick_hz)))
    on = off = point = None
    for ev in clip['events']:
        n = norm(ev['function']) + norm(ev['string'])
        if on is None and any(h in n for h in EVENT_ON_HINTS):
            on = ev['time']
        elif off is None and any(h in n for h in EVENT_OFF_HINTS):
            off = ev['time']
        elif point is None and any(h in n for h in EVENT_POINT_HINTS):
            point = ev['time']

    if on is not None and off is not None and off > on:
        startup = int(round(on * tick_hz))
        active = max(1, int(round((off - on) * tick_hz)))
        source = 'events'
    elif point is not None:
        active = max(1, int(round(INFERRED_ACTIVE_FRAC * clip['duration_s'] * tick_hz)))
        startup = max(0, int(round(point * tick_hz)) - active // 2)
        source = 'contact-event'
    else:
        contact = INFERRED_CONTACT_FRAC * clip['duration_s']
        active = max(1, int(round(INFERRED_ACTIVE_FRAC * clip['duration_s'] * tick_hz)))
        startup = max(0, int(round(contact * tick_hz)) - active // 2)
        source = 'inferred'

    recovery = max(1, total - startup - active)
    return startup, active, recovery, source


def limb_radius(colliders, limb):
    """Hit radius off the real limb collider, or None if none is attributable."""
    hints = LEG_COLLIDER_HINTS if limb == 'LIMB_LEG' else ARM_COLLIDER_HINTS
    radii = []
    for c in colliders:
        owner = norm(c.get('owner') or '')
        if not any(h in owner for h in hints):
            continue
        if c['type'] == 'CapsuleCollider' or c['type'] == 'SphereCollider':
            r = c.get('radius')
        else:
            size = c.get('size')
            r = max(size[0], size[2]) / 2.0 if size else None
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        if r > 0:
            radii.append(r)
    if not radii:
        return None
    radii.sort()
    return radii[len(radii) // 2]


def emit(report_path, out_path, tick_hz, allow_inferred):
    report = json.loads(Path(report_path).read_text())
    clips = [c for c in report.get('animation_clips', [])
             if c.get('combat') and c.get('duration_s')]
    if not clips:
        sys.exit(f'{report_path} has no timed combat clips — run --survey first.')

    if tick_hz is None:
        tick_hz = report.get('time', {}).get('fixed_hz')
        if not tick_hz:
            sys.exit('No fixed timestep in the survey and no --tick-hz given. The '
                     'frame counts have no unit without it; re-run --survey on the '
                     'install or pass --tick-hz explicitly.')
        print(f'Using REK\'s own fixed timestep: {tick_hz:g} Hz')

    # Same clip can ship in several bundles; keep the first of each name.
    seen, unique = set(), []
    for c in sorted(clips, key=lambda c: c['name']):
        key = norm(c['name'])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    script_fields = {norm(m['name']): m['fields'] for m in report.get('mono_behaviours', [])}
    colliders = report.get('colliders', [])

    rows, inferred_timing = [], 0
    for c in unique:
        name = re.sub(r'[^A-Za-z0-9_]', '_', c['name'])[:23] or 'move'
        limb = c.get('limb_guess') or limb_of(c['name'])
        startup, active, recovery, timing_src = envelope(c, tick_hz)
        if timing_src == 'inferred':
            inferred_timing += 1

        # Root motion straight off the clip: mean root speed over its length is
        # exactly how far the move carries the robot forward.
        speed = c.get('average_speed')
        if speed:
            planar = (speed[0] ** 2 + speed[2] ** 2) ** 0.5
            root_motion = planar * c['duration_s']
            root_src = 'clip'
        else:
            root_motion, root_src = 0.10, 'default'

        radius = limb_radius(colliders, limb)
        radius_src = 'collider' if radius else 'default'
        if not radius:
            radius = 0.24

        f = script_fields.get(norm(c['name']), {})
        geom = CHASSIS_GEOMETRY.get(c.get('chassis') or 'L100', CHASSIS_GEOMETRY['L100'])
        limb_len = geom['leg_len'] if limb == 'LIMB_LEG' else geom['arm_len']
        raw_reach = f.get('reach')
        if raw_reach:
            extension = max(0.05, (float(pick(f, 'reach', 0.7)) - geom['body_radius']) / limb_len)
            ext_src = 'script'
        else:
            # Reach at the contact frame needs the pose, and a humanoid clip's
            # muscle curves do not give it up cheaply. Left for the parity
            # harness to measure against the game.
            extension = 1.0 if limb == 'LIMB_ARM' else 1.1
            ext_src = 'UNMEASURED'

        rows.append({
            'name': name,
            'limb': limb,
            'startup': startup,
            'active': active,
            'recovery': recovery,
            'extension': extension,
            'radius': float(radius),
            'damage': float(pick(f, 'damage', 1.0)),
            'balance_cost': round(0.02 * startup / 4.0, 4),
            'balance_impact': float(pick(f, 'balance', 0.2)),
            'root_motion': round(float(root_motion), 4),
            'guard_breaks': bool(pick(f, 'guard', 0)),
            'why': f'timing={timing_src} root={root_src} radius={radius_src} reach={ext_src}',
        })

    unmeasured = sum(1 for r in rows if 'UNMEASURED' in r['why'])
    if inferred_timing == len(rows) and not allow_inferred:
        sys.exit(
            f'Every one of the {len(rows)} clips fell back to an inferred hit window '
            '— no animation events matched. That is a guessed table wearing a\n'
            'generated filename, which is the failure this extractor exists to stop.\n'
            'Send the survey JSON so the event-name hints can be fixed, or pass\n'
            '--allow-inferred to emit it anyway with every row tagged.')

    lines = [
        '// Generated by ocean/rek/tools/extract_rek.py from the shipped REK',
        '// Unity assets. Do not edit by hand — re-run the extractor instead.',
        f'// Source survey : {Path(report_path).name}',
        f'// Unity version : {report.get("unity_version")}',
        f'// Animation rig : {report.get("rig", {}).get("library") or "unidentified"}',
        f'// Frame counts are at {tick_hz:g} Hz, which must equal REK_TICK_HZ.',
        '//',
        '// Provenance is on every row. `timing=events` came out of the clip\'s own',
        '// hit events; `timing=inferred` did not and is a shape guess over a real',
        '// clip length. `reach=UNMEASURED` means the value below is a placeholder',
        f'// the parity harness still has to pin down ({unmeasured}/{len(rows)} rows).',
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
            '{root_motion:.2f}f, {guard}}},  // {why}'.format(
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
    measured = len(rows) - inferred_timing
    print(f'Wrote {out_path} with {len(rows)} moves (plus neutral).')
    print(f'  timing from clip events : {measured}/{len(rows)}')
    print(f'  reach still unmeasured  : {unmeasured}/{len(rows)}')
    if tick_hz != 30.0:
        print(f'\nREK ticks at {tick_hz:g} Hz, not 30. Set REK_TICK_HZ in moves.h to '
              f'match before rebuilding, or every frame count here is in the wrong unit.')
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
    ap.add_argument('--tick-hz', type=float, default=None,
        help='frame unit for the emitted table; defaults to the fixed timestep '
             'found by --survey, which is the value REK actually runs at')
    ap.add_argument('--allow-inferred', action='store_true',
        help='emit even when no clip carried hit events and every window is a guess')
    args = ap.parse_args()

    if args.list:
        list_candidates(args.appid)
        return

    if not args.survey and not args.emit:
        ap.error('pick one of --list, --survey or --emit')

    if args.survey:
        survey(find_install(args.path, args.appid), Path(args.report))
    if args.emit:
        emit(args.report, args.out, args.tick_hz, args.allow_inferred)


if __name__ == '__main__':
    main()
