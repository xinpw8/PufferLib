"""Step 1: pin one exact REK build and inventory it.

Every trace, every recovered constant and every parity claim has to name the
build it came from. Without that an update silently invalidates results and
nothing downstream can tell. This walks the install, hashes every file, and
derives a single build fingerprint from the files that decide behaviour.

    python inventory.py --out inventory.json
    python inventory.py --verify inventory.json      # has the build moved?

The fingerprint deliberately covers only the decisive set — executables, the
IL2CPP pair, UnityPlayer, asset containers, native plugins. Log files, save
data, crash dumps and shader caches change on their own and would make the
fingerprint useless as an identity if they counted toward it.

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
    if suffix in ('.bundle', '.unity3d') or name.startswith('sharedassets') \
            or name.startswith('level') or name == 'resources.assets' \
            or suffix == '.assets' or suffix == '.resource' or suffix == '.resS'.lower():
        return 'asset_container'
    if suffix in ('.json', '.ini', '.cfg', '.xml', '.yaml', '.yml', '.toml', '.config'):
        return 'config'
    if suffix in ('.log', '.txt') or 'crashes' in parts or 'logs' in parts:
        return 'volatile'
    return 'other'


# Categories whose contents define the build's behaviour. A change in any of
# these is a different build for our purposes.
DECISIVE = ('executable', 'il2cpp_code', 'il2cpp_metadata', 'unity_runtime',
            'unity_settings', 'asset_container', 'native_plugin',
            'burst_library', 'model_asset', 'managed_or_native')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


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


def scan(root: Path, include_volatile: bool) -> dict:
    files, errors = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = p.relative_to(root)
            kind = classify(rel)
            if kind == 'volatile' and not include_volatile:
                continue
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

    decisive = [f for f in files if f['kind'] in DECISIVE]
    fingerprint = hashlib.sha256()
    for f in sorted(decisive, key=lambda f: f['path']):
        fingerprint.update(f['path'].encode())
        fingerprint.update(b'\0')
        fingerprint.update(f['sha256'].encode())
        fingerprint.update(b'\n')

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
        'build_fingerprint': fingerprint.hexdigest(),
        'decisive_file_count': len(decisive),
        'file_count': len(files),
        'by_kind': dict(sorted(by_kind.items())),
        'files': files,
        'errors': errors,
    }


def summarise(inv: dict) -> None:
    steam = inv.get('steam') or {}
    print(f'install          : {inv["install"]}')
    print(f'build fingerprint: {inv["build_fingerprint"]}')
    if steam:
        print(f'steam buildid    : {steam.get("buildid", "?")}  '
              f'appid {steam.get("appid", "?")}  '
              f'"{steam.get("name", "?")}"')
    else:
        print('steam buildid    : no appmanifest found beside this install')
    print(f'files            : {inv["file_count"]} '
          f'({inv["decisive_file_count"]} decisive)')
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


def verify(old_path: Path, root: Path | None, include_volatile: bool) -> int:
    old = json.loads(old_path.read_text())
    root = root or Path(old['install'])
    new = scan(root, include_volatile)

    if new['build_fingerprint'] == old['build_fingerprint']:
        print(f'build unchanged: {new["build_fingerprint"]}')
        return 0

    print('BUILD CHANGED — every trace and constant recorded against the old '
          'fingerprint is now unverified.')
    print(f'  was: {old["build_fingerprint"]}')
    print(f'  now: {new["build_fingerprint"]}')
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
    ap.add_argument('--include-volatile', action='store_true',
        help='also hash logs and crash dumps (excluded by default: they change '
             'on their own and would break the fingerprint as an identity)')
    args = ap.parse_args()

    root = None
    if args.path:
        root = Path(args.path)
    elif not args.verify:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from install_discovery import find_install, DEFAULT_APPID
        root = find_install(None, args.appid or DEFAULT_APPID)

    if args.verify:
        return verify(Path(args.verify), root, args.include_volatile)

    if not root.is_dir():
        sys.exit(f'{root} is not a directory')
    inv = scan(root, args.include_volatile)
    Path(args.out).write_text(json.dumps(inv, indent=1))
    summarise(inv)
    print(f'\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
