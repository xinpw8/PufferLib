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
A quantity that is not in the build is reported absent, not filled in. Field
names alone ("balance", "damage", "startup") are reconnaissance and are recorded
as such under `name_hits`, never promoted to a mechanism.

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
        'name_hits': [],
        'containers_scanned': 0,
        'absent': [],
    }

    # Native code and shipped models come straight off the inventory: they are
    # files, not Unity objects, and they were already hashed.
    for f in inventory.get('files', []):
        if f['kind'] in ('native_plugin', 'burst_library'):
            report['native_code'].append(f)
        elif f['kind'] == 'model_asset':
            report['model_assets'].append(dict(f, source='file'))
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
                        'container': path.name,
                        'values': scalars(obj.read_typetree()),
                    }

                elif kind in BODY_TYPES:
                    tree = obj.read_typetree()
                    report['bodies'].append({
                        'type': kind,
                        'owner': owner_name(obj),
                        'container': path.name,
                        'values': scalars(tree),
                    })

                elif kind in COLLIDER_TYPES:
                    tree = obj.read_typetree()
                    report['colliders'].append({
                        'type': kind,
                        'owner': owner_name(obj),
                        'container': path.name,
                        'values': scalars(tree),
                    })

                elif kind in ('NNModel', 'ModelAsset'):
                    tree = obj.read_typetree()
                    report['model_assets'].append({
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
                            'name': name,
                            'container': path.name,
                            'hints': hits,
                            'keys': sorted(tree.keys())[:60],
                        })
            except Exception:
                continue

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

    print(f'\nreconnaissance leads (names only, NOT findings): {len(r["name_hits"])}')
    for h in r['name_hits'][:15]:
        print(f'  {h["name"] or "(unnamed)"}  {h["hints"]}')

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
