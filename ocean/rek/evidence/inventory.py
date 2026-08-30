"""Step 1: pin one exact REK build and inventory it.

Every trace, every recovered constant and every parity claim has to name the
build it came from. Without that an update silently invalidates results and
nothing downstream can tell. This walks the install, hashes every file, and
derives Merkle roots over the result.

    python inventory.py --out inventory.json
    python inventory.py --verify inventory.json      # has the build moved?

Three roots, because they answer different questions:

    manifest     every file seen, volatile ones included. The complete record.
    immutable    every shipped file. This is the build identity that traces
                 cite. No hand-picked category list, so a behaviour change
                 cannot hide in a bucket nobody thought to include — an
                 Addressables bundle, a controller weight file, a Burst
                 library, a physics plugin.
    behavioural  the subset most likely to matter, for triage only.

Volatile files — logs, crash dumps, local save state — are recorded in the
manifest but excluded from the immutable root, since they change on their own
and would otherwise make the identity useless.

This pins the *client*. It does not pin a server. If the authoritative
simulation runs remotely, the server can be updated independently and a trace
must additionally record protocol version, endpoint, session and any
server-reported version — see trace.py.

Read-only against the install.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

CHUNK = 1 << 20

# What a file is, decided by name and extension. Order matters: the first match
# wins, so the specific IL2CPP and Unity runtime names are tested before the
# generic extension buckets.
def classify(rel: Path) -> str:
    name = rel.name.lower()
    suffix = rel.suffix.lower()
    parts = {p.lower() for p in rel.parts}

    if name == 'global-metadata.dat':
        return 'il2cpp_metadata'
    if name == 'gameassembly.dll' or name.startswith('libil2cpp.'):
        return 'il2cpp_code'
    if name.startswith('unityplayer.'):
        return 'unity_runtime'
    if name in ('globalgamemanagers', 'globalgamemanagers.assets'):
        return 'unity_settings'
    if suffix in ('.exe', '.x86_64') or (suffix == '' and 'linux' in name):
        return 'executable'
    if suffix in ('.dll', '.so', '.dylib', '.pdb'):
        # Burst emits its own native libraries; they carry compiled game logic
        # and are worth separating from third-party plugins.
        if 'burst' in name or 'burst' in parts:
            return 'burst_library'
        return 'native_plugin' if 'plugins' in parts else 'managed_or_native'
    if suffix in ('.onnx', '.sentis', '.nn', '.tflite', '.pt', '.pth', '.plan', '.engine'):
        return 'model_asset'
    # Addressables ship behaviour in bundles and decide which ones load through
    # a catalog. Missing either would mean fingerprinting a build while the part
    # that actually changed sat in 'other'.
    if name.startswith('catalog') and suffix in ('.json', '.bin', '.hash'):
        return 'addressables_catalog'
    if 'addressables' in parts or 'aa' in parts:
        return 'addressables_content'
    if suffix in ('.bundle', '.unity3d') or name.startswith('sharedassets') \
            or name.startswith('level') or name == 'resources.assets' \
            or suffix in ('.assets', '.resource', '.ress', '.sharedassets'):
        return 'asset_container'
    if suffix in ('.json', '.ini', '.cfg', '.xml', '.yaml', '.yml', '.toml', '.config'):
        return 'config'
    if suffix in ('.log', '.txt') or 'crashes' in parts or 'logs' in parts:
        return 'volatile'
    return 'other'


# Categories whose contents define the build's behaviour. A change in any of
# these is a different build for our purposes.
BEHAVIOURAL = ('executable', 'il2cpp_code', 'il2cpp_metadata', 'unity_runtime',
               'unity_settings', 'asset_container', 'addressables_catalog',
               'addressables_content', 'native_plugin', 'burst_library',
               'model_asset', 'managed_or_native')

# Files that change on their own — logs, crash dumps, local save state. They are
# not part of the shipped install, so they are excluded from the immutable root
# and marked in the manifest rather than dropped.
VOLATILE = ('volatile',)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def merkle_root(entries) -> str:
    """Merkle root over (path, content hash) pairs, sorted by path.

    A tree rather than a hash of a concatenation so that any single file can
    later be proven in or out of a given build without redistributing the whole
    manifest, and so two manifests can be diffed structurally. Leaves and
    internal nodes are domain-separated; an odd node is promoted unchanged.
    """
    level = [hashlib.sha256(b'leaf\0' + path.encode() + b'\0' + digest.encode()).digest()
             for path, digest in sorted(entries)]
    if not level:
        return hashlib.sha256(b'empty').hexdigest()
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashlib.sha256(b'node\0' + level[i] + level[i + 1]).digest())
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def steam_manifest(root: Path) -> dict:
    """The appmanifest for this install, if the install sits under a library.

    buildid is the value that actually pins the version. Version strings in a
    UI can lag or be cosmetic; buildid changes on every shipped update.
    """
    steamapps = None
    for parent in root.parents:
        if parent.name.lower() == 'common' and parent.parent.name.lower() == 'steamapps':
            steamapps = parent.parent
            break
    if steamapps is None:
        return {}
    for acf in sorted(steamapps.glob('appmanifest_*.acf')):
        try:
            text = acf.read_text(errors='replace')
        except OSError:
            continue
        installdir = re.search(r'"installdir"\s+"([^"]+)"', text)
        if installdir and installdir.group(1).lower() == root.name.lower():
            out = {'manifest_file': acf.name, 'manifest_sha256': sha256(acf)}
            for key in ('appid', 'name', 'buildid', 'LastUpdated', 'SizeOnDisk',
                        'installdir', 'StateFlags', 'betakey'):
                m = re.search(r'"%s"\s+"([^"]*)"' % re.escape(key), text)
                if m:
                    out[key] = m.group(1)
            return out
    return {}


def scan(root: Path) -> dict:
    files, errors = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(root)
            kind = classify(rel)
            try:
                st = p.stat()
                files.append({
                    'path': rel.as_posix(),
                    'kind': kind,
                    'size': st.st_size,
                    'sha256': sha256(p),
                })
            except OSError as e:
                errors.append({'path': rel.as_posix(), 'error': str(e)})

    # Three roots, because they answer different questions and conflating them
    # is how a build gets mis-identified.
    #
    #   manifest  every file seen, including volatile ones. Complete record.
    #   immutable every shipped file. THIS is the build's identity: no heuristic
    #             selection, so a behaviour change cannot hide in a category
    #             someone forgot to list.
    #   behavioural  the subset most likely to matter, for triage only. Never
    #             the identity — that was the earlier mistake.
    immutable = [f for f in files if f['kind'] not in VOLATILE]
    behavioural = [f for f in files if f['kind'] in BEHAVIOURAL]

    roots = {
        'manifest': merkle_root((f['path'], f['sha256']) for f in files),
        'immutable': merkle_root((f['path'], f['sha256']) for f in immutable),
        'behavioural': merkle_root((f['path'], f['sha256']) for f in behavioural),
    }

    by_kind = {}
    for f in files:
        e = by_kind.setdefault(f['kind'], {'count': 0, 'bytes': 0})
        e['count'] += 1
        e['bytes'] += f['size']

    return {
        'schema': 1,
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'install': str(root),
        'steam': steam_manifest(root),
        # build_fingerprint is the immutable root. Traces cite this.
        'build_fingerprint': roots['immutable'],
        'merkle_roots': roots,
        'immutable_file_count': len(immutable),
        'behavioural_file_count': len(behavioural),
        'file_count': len(files),
        'by_kind': dict(sorted(by_kind.items())),
        'files': files,
        'errors': errors,
    }


def summarise(inv: dict) -> None:
    steam = inv.get('steam') or {}
    print(f'install          : {inv["install"]}')
    print(f'build fingerprint: {inv["build_fingerprint"]}  (immutable Merkle root)')
    for name, root in inv.get('merkle_roots', {}).items():
        print(f'  {name:<12} {root}')
    if steam:
        print(f'steam buildid    : {steam.get("buildid", "?")}  '
              f'appid {steam.get("appid", "?")}  '
              f'"{steam.get("name", "?")}"')
    else:
        print('steam buildid    : no appmanifest found beside this install')
    print(f'files            : {inv["file_count"]} total, '
          f'{inv["immutable_file_count"]} shipped, '
          f'{inv["behavioural_file_count"]} behaviour-bearing')
    for kind, e in inv['by_kind'].items():
        print(f'  {kind:<20} {e["count"]:>5}  {e["bytes"] / 1e6:>10.1f} MB')

    # The two questions the next steps depend on, answered from the inventory
    # rather than from assumption.
    models = [f for f in inv['files'] if f['kind'] == 'model_asset']
    print(f'\nshipped model assets: {len(models)}')
    for f in models[:20]:
        print(f'  {f["path"]}  ({f["size"] / 1e6:.2f} MB)')
    if len(models) > 20:
        print(f'  ... and {len(models) - 20} more')

    il2cpp = [f for f in inv['files'] if f['kind'].startswith('il2cpp')]
    print(f'il2cpp            : {"yes" if il2cpp else "no (Mono build?)"}')
    for f in il2cpp:
        print(f'  {f["path"]}  {f["sha256"][:16]}...')
    if inv['errors']:
        print(f'\n{len(inv["errors"])} files could not be read:')
        for e in inv['errors'][:10]:
            print(f'  {e["path"]}: {e["error"]}')


def verify(old_path: Path, root: Path | None) -> int:
    old = json.loads(old_path.read_text())
    root = root or Path(old['install'])
    new = scan(root)

    if new['build_fingerprint'] == old['build_fingerprint']:
        print(f'build unchanged: {new["build_fingerprint"]}')
        return 0

    print('BUILD CHANGED — every trace and constant recorded against the old '
          'fingerprint is now unverified.')
    print(f'  was: {old["build_fingerprint"]}')
    print(f'  now: {new["build_fingerprint"]}')
    for name in ('manifest', 'immutable', 'behavioural'):
        a = (old.get('merkle_roots') or {}).get(name)
        b = (new.get('merkle_roots') or {}).get(name)
        if a and b:
            print(f'  {name:<12} {"same" if a == b else "DIFFERENT"}')
    old_files = {f['path']: f for f in old['files']}
    new_files = {f['path']: f for f in new['files']}
    for path in sorted(set(old_files) | set(new_files)):
        a, b = old_files.get(path), new_files.get(path)
        if a is None:
            print(f'  added    {path}')
        elif b is None:
            print(f'  removed  {path}')
        elif a['sha256'] != b['sha256']:
            print(f'  changed  {path}  ({a["size"]} -> {b["size"]} bytes)')
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', help='install directory (auto-detected if omitted)')
    ap.add_argument('--appid', default=None, help='Steam appid, for auto-detection')
    ap.add_argument('--out', default='inventory.json', help='where to write the inventory')
    ap.add_argument('--verify', metavar='INVENTORY',
        help='re-hash the install and report drift against this inventory')
    args = ap.parse_args()

    root = None
    if args.path:
        root = Path(args.path)
    elif not args.verify:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from install_discovery import find_install, DEFAULT_APPID
        root = find_install(None, args.appid or DEFAULT_APPID)

    if args.verify:
        return verify(Path(args.verify), root)

    if not root.is_dir():
        sys.exit(f'{root} is not a directory')
    inv = scan(root)
    Path(args.out).write_text(json.dumps(inv, indent=1))
    summarise(inv)
    print(f'\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
