"""Step 4: inventory what the build actually contains.

Reports physics and time configuration, articulation and joint parameters,
collision geometry, shipped inference models, native code, and whether the
build is IL2CPP. Every finding carries the object it was read from and the
container file it lives in, and the whole survey carries the build fingerprint
from inventory.json.

    python inventory.py --out inventory.json
    python static_survey.py --inventory inventory.json --out static_survey.json

Contract, and the reason this replaces the earlier extractor: it emits only
values it read. There are no defaults, no fallbacks and no name-based guesses.
A quantity that is not in the build is reported absent, not filled in.

Every record carries a `role`, because "this component exists in the build" and
"this component participates in the authoritative transition function" are
different claims and conflating them is how the last model went wrong:

    authoritative       shown to drive the transition function. Nothing static
                        earns this: it requires a runtime trace or a controlled
                        experiment, so this tool never assigns it.
    candidate_lead      plausibly relevant. Name matches and animation clips
                        land here.
    client_render_only  established as presentation-side.
    unknown_role        present and serialized; its part in the simulation is
                        not established.
    absent              looked for, not found.

Animation clips are catalogued rather than ignored. A physics-based controller
can still be driven by reference motions, phase signals or skill latents taken
from a motion library, so the clips are evidence about the controller's inputs.
What must not happen again is a clip duration becoming a startup, active or
recovery window: that inference is what produced the discarded model, and how
clips are consumed can only be established by tracing the code that reads them.

Requires UnityPy for the asset containers. IL2CPP type recovery is out of scope
here — it needs Il2CppDumper against GameAssembly.dll and global-metadata.dat,
and this reports what that step would need rather than pretending to do it.
"""

import argparse
import json
import sys
from pathlib import Path

# Settings objects worth reading in full. These are single global objects, so
# capturing every scalar on them costs nothing and avoids deciding in advance
# which knob matters.
SETTINGS_TYPES = ('TimeManager', 'PhysicsManager', 'Physics2DSettings',
                  'QualitySettings', 'PlayerSettings')

# Per-body and per-joint components. These carry the parameters an articulated
# reimplementation needs, and there is no way to derive them from anything else.
BODY_TYPES = ('ArticulationBody', 'Rigidbody', 'ConfigurableJoint',
              'CharacterJoint', 'HingeJoint', 'FixedJoint', 'SpringJoint')

COLLIDER_TYPES = ('CapsuleCollider', 'BoxCollider', 'SphereCollider',
                  'MeshCollider', 'TerrainCollider')

ROLES = ('authoritative', 'candidate_lead', 'client_render_only',
         'unknown_role', 'absent')

# Rig fingerprints. Which motion library a build is skinned to is a real lead
# about where its reference motions came from; it is not a claim about how they
# are used.
RIG_SIGNATURES = {
    'mixamorig': 'Adobe Mixamo',
    'ccbase': 'Reallusion Character Creator / ActorCore',
    'bip01': '3ds Max Biped (Kubold Animset Pro and similar)',
    'rokoko': 'Rokoko',
    'mocaponline': 'MocapOnline',
    'ue4mannequin': 'Epic mannequin rig',
}

# What no amount of static asset reading can establish. Listed in the output so
# the gap is visible rather than assumed closed.
NOT_RECOVERABLE_STATICALLY = (
    'which physics scene practice mode actually runs',
    'the controller observation vector and its construction',
    'how controller outputs are interpreted as joint targets or torques',
    'recurrent controller state and skill phase handling',
    'how animation clips are consumed, if at all',
    'contact-to-score logic',
    'input buffering, cancellation and action duration',
    'network command and state schemas',
    'execution order within a tick',
    'any server-side parameter',
)

# Words that suggest a MonoBehaviour is worth a human look. Recorded as leads,
# never as findings.
RECON_HINTS = ('balance', 'stagger', 'damage', 'score', 'round', 'match',
               'knock', 'fall', 'getup', 'guard', 'block', 'hit', 'strike',
               'policy', 'model', 'inference', 'controller', 'skill', 'ability',
               'tick', 'timestep', 'latency', 'buffer', 'cancel', 'command',
               'netcode', 'replicate', 'authoritative', 'server')


def scalars(tree, prefix='', out=None, depth=0):
    """Every scalar in a typetree, flattened to dotted keys."""
    if out is None:
        out = {}
    if depth > 8:
        return out
    if isinstance(tree, dict):
        for k, v in tree.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v, (int, float, bool, str)):
                out[key] = v
            else:
                scalars(v, key, out, depth + 1)
    elif isinstance(tree, list):
        for i, v in enumerate(tree[:64]):
            scalars(v, f'{prefix}[{i}]', out, depth + 1)
    return out


def owner_name(obj):
    try:
        return str(getattr(obj.read().m_GameObject.read(), 'm_Name', '') or '')
    except Exception:
        return ''


def survey(root: Path, inventory: dict, out_path: Path) -> dict:
    try:
        import UnityPy
    except ImportError:
        sys.exit('UnityPy is required: pip install UnityPy')

    report = {
        'schema': 1,
        'build_fingerprint': inventory.get('build_fingerprint'),
        'steam_buildid': (inventory.get('steam') or {}).get('buildid'),
        'install': str(root),
        'unity_version': None,
        'settings': {},
        'bodies': [],
        'colliders': [],
        'model_assets': [],
        'native_code': [],
        'il2cpp': {},
        'animation_clips': [],
        'rig': {'library': None, 'signature_hits': {}, 'sample_bones': []},
        'name_hits': [],
        'containers_scanned': 0,
        'absent': [],
        'not_recoverable_statically': list(NOT_RECOVERABLE_STATICALLY),
        'role_note': ('No record here is marked authoritative. Static presence '
                      'is not participation in the transition function; that '
                      'needs a runtime trace or a controlled experiment.'),
    }
    bone_paths = set()
    errors = []

    # Native code and shipped models come straight off the inventory: they are
    # files, not Unity objects, and they were already hashed.
    for f in inventory.get('files', []):
        if f['kind'] in ('native_plugin', 'burst_library'):
            report['native_code'].append(f)
        elif f['kind'] == 'model_asset':
            report['model_assets'].append(
                dict(f, source='file', role='candidate_lead'))
        elif f['kind'] == 'il2cpp_code':
            report['il2cpp']['code'] = f
        elif f['kind'] == 'il2cpp_metadata':
            report['il2cpp']['metadata'] = f

    containers = [root / f['path'] for f in inventory.get('files', [])
                  if f['kind'] in ('asset_container', 'unity_settings')]

    for path in containers:
        try:
            bundle = UnityPy.load(str(path))
        except Exception as e:
            report['absent'].append({'container': path.name, 'error': str(e)})
            continue
        report['containers_scanned'] += 1

        for obj in bundle.objects:
            if report['unity_version'] is None:
                report['unity_version'] = getattr(obj.assets_file, 'unity_version', None)
            kind = obj.type.name
            try:
                if kind in SETTINGS_TYPES:
                    report['settings'][kind] = {
                        'role': 'unknown_role',
                        'container': path.name,
                        'values': scalars(obj.read_typetree()),
                    }

                elif kind in BODY_TYPES:
                    tree = obj.read_typetree()
                    report['bodies'].append({
                        'role': 'unknown_role',
                        'type': kind,
                        'owner': owner_name(obj),
                        'container': path.name,
                        'values': scalars(tree),
                    })

                elif kind in COLLIDER_TYPES:
                    tree = obj.read_typetree()
                    report['colliders'].append({
                        'role': 'unknown_role',
                        'type': kind,
                        'owner': owner_name(obj),
                        'container': path.name,
                        'values': scalars(tree),
                    })

                elif kind == 'Avatar':
                    # m_TOS is the only place rig bone names survive into a
                    # build, and the rig names the motion library.
                    tree = obj.read_typetree()
                    tos = tree.get('m_TOS')
                    if isinstance(tos, list):
                        for entry in tos:
                            v = (entry.get('second') if isinstance(entry, dict)
                                 else (entry[1] if isinstance(entry, (list, tuple))
                                       and len(entry) == 2 else None))
                            if isinstance(v, str):
                                bone_paths.add(v)

                elif kind == 'AnimationClip':
                    tree = obj.read_typetree()
                    flat = scalars(tree)
                    duration = flat.get('m_Length') or flat.get(
                        'm_MuscleClip.m_StopTime')
                    report['animation_clips'].append({
                        'role': 'candidate_lead',
                        'name': str(tree.get('m_Name', '') or ''),
                        'container': path.name,
                        'duration_s': duration,
                        'sample_rate': flat.get('m_SampleRate'),
                        'event_count': len(tree.get('m_Events') or []),
                        'events': [{'time': e.get('time'),
                                    'function': e.get('functionName'),
                                    'data': e.get('data')}
                                   for e in (tree.get('m_Events') or [])[:32]
                                   if isinstance(e, dict)],
                        'caution': 'duration and events are evidence about this '
                                   'clip, not about any attack envelope',
                    })

                elif kind in ('NNModel', 'ModelAsset'):
                    tree = obj.read_typetree()
                    report['model_assets'].append({
                        'role': 'candidate_lead',
                        'source': 'unity_object',
                        'type': kind,
                        'name': str(tree.get('m_Name', '')),
                        'container': path.name,
                        'keys': sorted(tree.keys()),
                    })

                elif kind == 'MonoBehaviour':
                    tree = obj.read_typetree()
                    if not isinstance(tree, dict):
                        continue
                    name = str(tree.get('m_Name', '') or '')
                    blob = (name + ' ' + ' '.join(map(str, tree.keys()))).lower()
                    hits = sorted({h for h in RECON_HINTS if h in blob})
                    if hits:
                        report['name_hits'].append({
                            'role': 'candidate_lead',
                            'name': name,
                            'container': path.name,
                            'hints': hits,
                            'keys': sorted(tree.keys())[:60],
                        })
            except Exception as e:
                # Never swallow this silently. A schema mismatch on the real
                # build would otherwise produce a plausible-looking empty
                # survey, and absence would read as "not present" when it
                # actually meant "could not be read".
                errors.append({'container': path.name, 'type': kind,
                               'error': f'{type(e).__name__}: {e}'})

    hits = {}
    for bone in bone_paths:
        n = ''.join(c for c in bone.lower() if c.isalnum())
        for sig, lib in RIG_SIGNATURES.items():
            if sig in n:
                hits[lib] = hits.get(lib, 0) + 1
                break
    report['rig'] = {
        'role': 'candidate_lead',
        'library': max(hits, key=hits.get) if hits else None,
        'signature_hits': hits,
        'sample_bones': sorted(bone_paths)[:40],
    }

    # Read failures are surfaced, and grouped, so a systematic breakage is
    # obvious rather than looking like an empty build.
    if errors:
        by_type = {}
        for e in errors:
            by_type.setdefault(e['type'], []).append(e['error'])
        report['read_errors'] = [
            {'type': t, 'count': len(v), 'example': v[0]}
            for t, v in sorted(by_type.items())]

    for want, where in (('TimeManager', 'fixed timestep / control rate'),
                        ('PhysicsManager', 'gravity, solver iterations, contact offset')):
        if want not in report['settings']:
            report['absent'].append({'missing': want, 'needed_for': where})
    if not report['bodies']:
        report['absent'].append({
            'missing': 'ArticulationBody/Rigidbody/joint components',
            'needed_for': 'articulated dynamics parameters',
            'note': 'may live in AssetBundles loaded at runtime, or be built in '
                    'code under IL2CPP rather than serialized'})

    out_path.write_text(json.dumps(report, indent=1, default=str))
    return report


def summarise(r: dict) -> None:
    print(f'build fingerprint : {r["build_fingerprint"]}')
    print(f'steam buildid     : {r["steam_buildid"]}')
    print(f'unity version     : {r["unity_version"]}')
    print(f'containers scanned: {r["containers_scanned"]}')

    tm = r['settings'].get('TimeManager', {}).get('values', {})
    if tm:
        print('\nTimeManager (the control/physics rate, measured not assumed):')
        for k, v in sorted(tm.items()):
            print(f'  {k} = {v}')
    else:
        print('\nTimeManager: NOT FOUND — tick rate remains unknown')

    pm = r['settings'].get('PhysicsManager', {}).get('values', {})
    if pm:
        print('\nPhysicsManager:')
        for k, v in sorted(pm.items()):
            print(f'  {k} = {v}')
    else:
        print('\nPhysicsManager: NOT FOUND — physics config remains unknown')

    print(f'\narticulation/joint components: {len(r["bodies"])}')
    kinds = {}
    for b in r['bodies']:
        kinds[b['type']] = kinds.get(b['type'], 0) + 1
    for k, n in sorted(kinds.items()):
        print(f'  {k}: {n}')
    print(f'colliders          : {len(r["colliders"])}')

    print(f'\nshipped inference models: {len(r["model_assets"])}')
    for m in r['model_assets'][:20]:
        print(f'  {m.get("path") or m.get("name")}  ({m.get("source")})')

    print(f'\nnative code (plugins, Burst): {len(r["native_code"])}')
    for f in r['native_code'][:20]:
        print(f'  {f["path"]}')

    il2 = r['il2cpp']
    if il2.get('code') and il2.get('metadata'):
        print('\nIL2CPP build. Type and method recovery needs Il2CppDumper against:')
        print(f'  {il2["code"]["path"]}  sha256 {il2["code"]["sha256"][:16]}...')
        print(f'  {il2["metadata"]["path"]}  sha256 {il2["metadata"]["sha256"][:16]}...')
        print('  That is step 4b and is not attempted here.')
    else:
        print('\nNot an IL2CPP build (or the pair was not found).')

    clips = r.get('animation_clips', [])
    withev = [c for c in clips if c.get('event_count')]
    print(f'\nanimation clips: {len(clips)} ({len(withev)} carry events)')
    rig = r.get('rig') or {}
    print(f'  rig / motion library: {rig.get("library") or "unidentified"} '
          f'{rig.get("signature_hits") or ""}')
    for c in clips[:12]:
        print(f'  {c["name"]:<34} {c["duration_s"]}s  {c["event_count"]} event(s)')
    if clips:
        print('  These are candidate leads about what feeds the controller. A '
              'clip duration is not an attack envelope, and how these are '
              'consumed cannot be read off the assets.')

    print(f'\nreconnaissance leads (names only, NOT findings): {len(r["name_hits"])}')
    for h in r['name_hits'][:15]:
        print(f'  {h["name"] or "(unnamed)"}  {h["hints"]}')

    print('\nnot recoverable from static assets — these need runtime evidence:')
    for item in r.get('not_recoverable_statically', []):
        print(f'  - {item}')

    if r.get('read_errors'):
        print('\nOBJECTS THAT COULD NOT BE READ — absence below may mean '
              'unreadable, not missing:')
        for e in r['read_errors']:
            print(f'  {e["type"]:<24} {e["count"]:>5}x  {e["example"]}')

    if r['absent']:
        print('\nabsent / unreadable — these stay unknown until measured:')
        for a in r['absent'][:20]:
            print(f'  {a}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inventory', default='inventory.json',
        help='inventory.json from inventory.py; supplies the build fingerprint')
    ap.add_argument('--out', default='static_survey.json')
    ap.add_argument('--path', help='install directory (defaults to the one in the inventory)')
    args = ap.parse_args()

    inv = json.loads(Path(args.inventory).read_text())
    root = Path(args.path or inv['install'])
    if not root.is_dir():
        sys.exit(f'{root} is not a directory')
    report = survey(root, inv, Path(args.out))
    summarise(report)
    print(f'\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
