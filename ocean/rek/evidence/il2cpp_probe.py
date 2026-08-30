"""Step 4b, the part that does not need a decompiler: what is named in the binary.

An IL2CPP build has no managed assemblies to read. Full type and method recovery
needs Il2CppDumper against GameAssembly.dll and global-metadata.dat, and this
does not attempt that. What it does is bounded and honest: verify the metadata
file really is IL2CPP metadata, report its version so the right dumper can be
chosen, and extract the identifier strings that survive in the metadata and
native binaries, classified against the questions the survey could not answer.

    python il2cpp_probe.py --inventory inventory.json --out il2cpp_probe.json

This is `strings` with domain classification. A name in a binary proves a name is
in the binary: it is a `candidate_lead` and a target for instrumentation, never a
finding about behaviour. It is useful precisely because the next step — tracing
the input-to-controller path — needs somewhere to attach, and this is what names
the candidates.

The one near-decisive thing it can show is which inference runtime, if any, is
linked. Unity.Sentis, Barracuda or onnxruntime symbols in the shipped binaries
mean a neural controller runs in the client; their absence across every native
binary is evidence, though not proof, that it does not.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# IL2CPP metadata header: uint32 sanity, int32 version. Everything after that
# moves between versions, so nothing after that is read here.
IL2CPP_SANITY = 0xFAB11BAF

# What each name would be a lead about. Deliberately broad: this is for finding
# somewhere to attach a hook, not for concluding anything.
BUCKETS = {
    'inference_runtime': ('sentis', 'barracuda', 'onnxruntime', 'onnx', 'nnmodel',
                          'tensorrt', 'libtorch', 'torchscript', 'tflite',
                          'inferenceengine', 'worker', 'tensorshape'),
    'controller': ('policy', 'controller', 'actuator', 'balance', 'stabil',
                   'locomotion', 'gait', 'skill', 'motionmatch', 'reference',
                   'observation', 'action', 'latent', 'recurrent', 'phase'),
    'physics': ('articulation', 'rigidbody', 'physicsscene', 'physx', 'solver',
                'contact', 'collision', 'joint', 'drive', 'fixedupdate',
                'timestep', 'simulate', 'inertia', 'friction'),
    'netcode': ('netcode', 'mirror', 'fishnet', 'photon', 'transport', 'rpc',
                'snapshot', 'reconcil', 'predict', 'serverauth', 'tickrate',
                'handshake', 'protocolversion', 'session'),
    'match_rules': ('roundstart', 'roundend', 'knockdown', 'scoreboard',
                    'matchstate', 'downcount', 'winner', 'ko'),
    'input': ('inputaction', 'buffer', 'cancel', 'command', 'keybind',
              'playerinput', 'edge', 'held'),
    'animation': ('animationclip', 'animator', 'statemachine', 'blendtree',
                  'rootmotion', 'avatar', 'mixamo'),
}

# Identifier-shaped strings only. A build is full of paths, shader source and
# UI text, and none of that is a hook target.
IDENTIFIER = re.compile(rb'[A-Za-z_][A-Za-z0-9_.<>`|]{3,127}')


def read_metadata_header(path: Path):
    """Verify the file is IL2CPP metadata and read its version."""
    with path.open('rb') as f:
        head = f.read(8)
    if len(head) < 8:
        return {'valid': False, 'why': 'file is shorter than the header'}
    sanity = int.from_bytes(head[:4], 'little')
    version = int.from_bytes(head[4:8], 'little', signed=True)
    if sanity != IL2CPP_SANITY:
        return {'valid': False, 'sanity': hex(sanity),
                'why': f'sanity is {hex(sanity)}, expected {hex(IL2CPP_SANITY)} — '
                       'not IL2CPP metadata, or it is encrypted/packed'}
    return {'valid': True, 'sanity': hex(sanity), 'metadata_version': version,
            'note': 'pass this version to Il2CppDumper; it selects the layout. '
                    'Nothing past the header is parsed here, because everything '
                    'past it moves between versions.'}


def extract_strings(path: Path, min_len=4, cap=4_000_000):
    """Identifier-shaped byte runs. Format independent, so version drift is
    irrelevant — the cost is that this finds names, not structure."""
    data = path.read_bytes()
    truncated = len(data) > cap
    if truncated:
        data = data[:cap]
    out = set()
    for m in IDENTIFIER.finditer(data):
        try:
            out.add(m.group().decode('ascii'))
        except UnicodeDecodeError:
            continue
    return out, truncated


def classify(names):
    hits = {b: set() for b in BUCKETS}
    for name in names:
        low = name.lower()
        for bucket, needles in BUCKETS.items():
            if any(n in low for n in needles):
                hits[bucket].add(name)
    return {b: sorted(v) for b, v in hits.items()}


def probe(root: Path, inventory: dict, out_path: Path) -> dict:
    report = {
        'schema': 1,
        'build_fingerprint': inventory.get('build_fingerprint'),
        'role_note': ('Every name here is a candidate_lead and a target for '
                      'instrumentation. A name in a binary proves a name is in '
                      'a binary.'),
        'metadata': {},
        'scanned': [],
        'buckets': {},
        'inference_runtime_present': None,
        'absent': [],
    }

    kinds = ('il2cpp_metadata', 'il2cpp_code', 'native_plugin', 'burst_library',
             'unity_runtime')
    targets = [f for f in inventory.get('files', []) if f['kind'] in kinds]
    if not targets:
        report['absent'].append({'missing': 'IL2CPP and native binaries',
                                 'note': 'inventory lists none; is this a Mono build?'})

    all_names = set()
    for f in targets:
        path = root / f['path']
        if not path.exists():
            report['absent'].append({'missing': f['path'],
                                     'note': 'listed in the inventory, not on disk'})
            continue
        if f['kind'] == 'il2cpp_metadata':
            report['metadata'] = dict(read_metadata_header(path), path=f['path'])
        try:
            names, truncated = extract_strings(path)
        except OSError as e:
            report['absent'].append({'missing': f['path'], 'note': str(e)})
            continue
        all_names |= names
        report['scanned'].append({'path': f['path'], 'kind': f['kind'],
                                  'size': f['size'], 'names': len(names),
                                  'truncated': truncated})

    buckets = classify(all_names)
    report['buckets'] = {b: {'role': 'candidate_lead', 'count': len(v),
                             'names': v[:400]}
                         for b, v in buckets.items()}
    report['total_names'] = len(all_names)
    report['inference_runtime_present'] = bool(buckets['inference_runtime'])
    return report


def summarise(r: dict) -> None:
    print(f'build fingerprint : {r["build_fingerprint"]}')
    md = r.get('metadata') or {}
    if not md:
        print('metadata          : no global-metadata.dat in the inventory')
    elif md.get('valid'):
        print(f'metadata          : valid IL2CPP, version '
              f'{md["metadata_version"]} ({md["path"]})')
        print(f'                    {md["note"]}')
    else:
        print(f'metadata          : NOT VALID — {md.get("why")}')

    print(f'binaries scanned  : {len(r["scanned"])}, '
          f'{r.get("total_names", 0)} distinct identifiers')
    for sc in r['scanned']:
        flag = '  (TRUNCATED)' if sc['truncated'] else ''
        print(f'  {sc["path"]:<52} {sc["names"]:>7} names{flag}')

    print('\\ncandidate leads by bucket:')
    for bucket, v in r['buckets'].items():
        print(f'  {bucket:<20} {v["count"]:>5}')
        for name in v['names'][:6]:
            print(f'      {name}')

    present = r['inference_runtime_present']
    print()
    if present:
        names = r['buckets']['inference_runtime']['names'][:8]
        print('An inference runtime is linked into this build: ' + ', '.join(names))
        print('A neural controller therefore runs in the client, and its weights '
              'and tensor shapes are recoverable. This is the strongest single '
              'result available without a decompiler.')
    elif r['scanned']:
        print('No inference-runtime symbols in any scanned binary. Evidence that '
              'no neural controller runs client-side — not proof: it could be '
              'statically inlined, name-mangled, packed, or server-side only.')

    print('\\nNext: Il2CppDumper against the metadata and GameAssembly for actual '
          'types and method signatures. These names are where to point it.')
    if r['absent']:
        print('\\nnot scanned:')
        for a in r['absent']:
            print(f'  {a}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inventory', default='inventory.json')
    ap.add_argument('--out', default='il2cpp_probe.json')
    ap.add_argument('--path', help='install directory (defaults to the inventory)')
    args = ap.parse_args()

    inv = json.loads(Path(args.inventory).read_text())
    root = Path(args.path or inv['install'])
    if not root.is_dir():
        sys.exit(f'{root} is not a directory')
    report = probe(root, inv, Path(args.out))
    Path(args.out).write_text(json.dumps(report, indent=1))
    summarise(report)
    print(f'\\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
