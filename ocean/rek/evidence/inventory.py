"""Step 1: pin one exact REK build and inventory it.

Every trace, every recovered constant and every parity claim has to name the
build it came from. Without that an update silently invalidates results and
nothing downstream can tell. This walks the install, hashes every file, and
derives Merkle roots over the result.

    python inventory.py --out inventory.json
    python inventory.py --verify inventory.json      # has the build moved?
    python inventory.py --verify-instrumented inventory.json --overlay-out instrumentation_overlay.json

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
import getpass
import hashlib
import json
import os
import platform
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


def scan(root: Path, progress=None) -> dict:
    """Hash every file under `root`.

    `progress` is called with (files, bytes) every couple of seconds. A full
    install is several gigabytes and takes minutes to hash; without some sign of
    life that is indistinguishable from a hang, and someone will kill it.
    """
    files, errors = [], []
    hashed_bytes = 0
    last_report = time.time()
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
                hashed_bytes += st.st_size
                if progress is not None and time.time() - last_report > 2.0:
                    progress(len(files), hashed_bytes)
                    last_report = time.time()
            except OSError as e:
                # Recorded as a leaf, not dropped. A file that could not be read
                # is still part of the install, and omitting it would silently
                # compute the identity over a subset — so a scan taken with the
                # game running would disagree with one taken with it closed and
                # nothing would say why.
                errors.append({'path': rel.as_posix(), 'error': str(e)})
                files.append({
                    'path': rel.as_posix(),
                    'kind': kind,
                    'size': None,
                    'sha256': None,
                    'unreadable': True,
                })

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

    def leaves(group):
        # UNREADABLE is a distinct leaf value, so an unread file changes the
        # root rather than vanishing from it.
        return ((f['path'], f['sha256'] or 'UNREADABLE') for f in group)

    roots = {
        'manifest': merkle_root(leaves(files)),
        'immutable': merkle_root(leaves(immutable)),
        'behavioural': merkle_root(leaves(behavioural)),
    }

    by_kind = {}
    for f in files:
        e = by_kind.setdefault(f['kind'], {'count': 0, 'bytes': 0})
        e['count'] += 1
        e['bytes'] += f['size'] or 0

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
        print(f'\nWARNING: {len(inv["errors"])} file(s) could not be read. This '
              f'fingerprint is NOT a reliable identity — close the game and any '
              f'tool holding the install, then re-run.')
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


def host_identity() -> dict:
    """Identity of the host which performed an inventory verification."""
    return {
        'hostname': platform.node(),
        'user': getpass.getuser(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
    }


def _inventory_index(files: list[dict], label: str) -> tuple[dict, list[dict]]:
    """Index manifest records without silently accepting duplicate paths."""
    indexed = {}
    violations = []
    for offset, record in enumerate(files):
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            violations.append({
                'type': 'invalid_inventory_record',
                'source': label,
                'index': offset,
            })
            continue
        path = record['path']
        if path in indexed:
            violations.append({
                'type': 'duplicate_inventory_path',
                'source': label,
                'path': path,
            })
            continue
        indexed[path] = record
    return indexed, violations


def _file_summary(files: list[dict]) -> tuple[dict, int]:
    by_kind = {}
    total_bytes = 0
    for record in files:
        kind = record.get('kind', 'other')
        entry = by_kind.setdefault(kind, {'count': 0, 'bytes': 0})
        entry['count'] += 1
        size = record.get('size')
        if isinstance(size, int):
            entry['bytes'] += size
            total_bytes += size
    return dict(sorted(by_kind.items())), total_bytes


def verify_instrumented(old_path: Path, root: Path | None,
                        overlay_out: Path) -> int:
    """Verify a shipped inventory while recording every subsequently added file.

    The old inventory alone defines the shipped file set and build fingerprint.
    Every original nonvolatile file must still be readable and byte-identical.
    Files absent from the old manifest are an instrumentation overlay: they are
    all hashed, recorded and rooted separately, including volatile files. They
    never participate in the shipped build fingerprint.
    """
    old_path = old_path.resolve()
    old = json.loads(old_path.read_text(encoding='utf-8'))
    root = (root or Path(old['install'])).resolve()
    overlay_out = overlay_out.resolve()
    try:
        overlay_out.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            '--overlay-out must be outside the install: writing it inside the '
            'tree would change the overlay after it was hashed')
    new = scan(root)

    old_files, violations = _inventory_index(old.get('files', []), 'base')
    new_files, new_index_violations = _inventory_index(new.get('files', []), 'observed')
    violations.extend(new_index_violations)

    old_nonvolatile = {
        path: record for path, record in old_files.items()
        if record.get('kind') not in VOLATILE
    }
    old_volatile = {
        path: record for path, record in old_files.items()
        if record.get('kind') in VOLATILE
    }

    if old.get('errors'):
        violations.append({
            'type': 'base_inventory_has_errors',
            'count': len(old['errors']),
        })

    baseline_leaves = []
    for path, record in sorted(old_nonvolatile.items()):
        digest = record.get('sha256')
        if not isinstance(digest, str) or not digest:
            violations.append({
                'type': 'base_inventory_unreadable',
                'path': path,
            })
            digest = 'UNREADABLE'
        baseline_leaves.append((path, digest))
    recorded_base_root = merkle_root(baseline_leaves)
    if recorded_base_root != old.get('build_fingerprint'):
        violations.append({
            'type': 'base_inventory_fingerprint_mismatch',
            'recorded': old.get('build_fingerprint'),
            'recomputed': recorded_base_root,
        })

    verified_base_files = []
    observed_base_leaves = []
    for path, expected in sorted(old_nonvolatile.items()):
        observed = new_files.get(path)
        if observed is None:
            violations.append({'type': 'base_file_removed', 'path': path})
            observed_base_leaves.append((path, 'MISSING'))
            verified_base_files.append({
                'path': path,
                'kind': expected.get('kind'),
                'expected_size': expected.get('size'),
                'expected_sha256': expected.get('sha256'),
                'observed_size': None,
                'observed_sha256': None,
                'state': 'removed',
            })
            continue

        observed_digest = observed.get('sha256')
        observed_base_leaves.append((path, observed_digest or 'UNREADABLE'))
        state = 'unchanged'
        if observed_digest != expected.get('sha256'):
            state = 'changed'
            violations.append({
                'type': 'base_file_changed',
                'path': path,
                'expected_size': expected.get('size'),
                'observed_size': observed.get('size'),
                'expected_sha256': expected.get('sha256'),
                'observed_sha256': observed_digest,
            })
        verified_base_files.append({
            'path': path,
            'kind': expected.get('kind'),
            'expected_size': expected.get('size'),
            'expected_sha256': expected.get('sha256'),
            'observed_size': observed.get('size'),
            'observed_sha256': observed_digest,
            'state': state,
        })

    added = [record for path, record in sorted(new_files.items())
             if path not in old_files]
    added_paths = {record['path'] for record in added}
    overlay_errors = [error for error in new.get('errors', [])
                      if error.get('path') in added_paths]
    for error in overlay_errors:
        violations.append({
            'type': 'overlay_file_unreadable',
            'path': error.get('path'),
            'error': error.get('error'),
        })

    def leaves(group):
        return ((record['path'], record.get('sha256') or 'UNREADABLE')
                for record in group)

    overlay_immutable = [record for record in added
                         if record.get('kind') not in VOLATILE]
    overlay_behavioural = [record for record in added
                           if record.get('kind') in BEHAVIOURAL]
    overlay_roots = {
        'manifest': merkle_root(leaves(added)),
        'immutable': merkle_root(leaves(overlay_immutable)),
        'behavioural': merkle_root(leaves(overlay_behavioural)),
    }
    overlay_by_kind, overlay_bytes = _file_summary(added)

    volatile_observations = []
    for path, expected in sorted(old_volatile.items()):
        observed = new_files.get(path)
        volatile_observations.append({
            'path': path,
            'kind': expected.get('kind'),
            'expected_size': expected.get('size'),
            'expected_sha256': expected.get('sha256'),
            'observed_size': observed.get('size') if observed else None,
            'observed_sha256': observed.get('sha256') if observed else None,
            'state': ('removed' if observed is None else
                      'unchanged' if observed.get('sha256') == expected.get('sha256')
                      else 'changed'),
        })

    artifact = {
        'schema': 'rek.instrumentation_overlay.v1',
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'verified' if not violations else 'failed',
        'host': host_identity(),
        'install': {
            'base_recorded_path': old.get('install'),
            'observed_path': str(root),
            'base_steam': old.get('steam') or {},
            'observed_steam': new.get('steam') or {},
        },
        'base_inventory': {
            'path': str(old_path),
            'sha256': sha256(old_path),
            'schema': old.get('schema'),
        },
        # This remains the immutable fingerprint from the pre-instrumentation
        # inventory. Overlay files are deliberately absent from this root.
        'build_fingerprint': old.get('build_fingerprint'),
        'base': {
            'required_file_count': len(old_nonvolatile),
            'verified_file_count': sum(
                record['state'] == 'unchanged' for record in verified_base_files),
            'recorded_merkle_root': recorded_base_root,
            'observed_merkle_root': merkle_root(observed_base_leaves),
            'files': verified_base_files,
        },
        'overlay': {
            'file_count': len(added),
            'bytes': overlay_bytes,
            # The overlay fingerprint covers every addition. The other roots
            # are diagnostic partitions and never replace the manifest root.
            'fingerprint': overlay_roots['manifest'],
            'merkle_roots': overlay_roots,
            'by_kind': overlay_by_kind,
            'files': added,
        },
        'original_volatile': {
            'file_count': len(volatile_observations),
            'files': volatile_observations,
        },
        'errors': new.get('errors', []),
        'violations': violations,
    }

    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    overlay_out.write_text(json.dumps(artifact, indent=1), encoding='utf-8')

    print(f'base build fingerprint: {artifact["build_fingerprint"]}')
    print(f'base files            : {artifact["base"]["verified_file_count"]}/'
          f'{artifact["base"]["required_file_count"]} unchanged')
    print(f'overlay files         : {len(added)}')
    print(f'overlay fingerprint   : {artifact["overlay"]["fingerprint"]}')
    print(f'overlay artifact      : {overlay_out}')
    if violations:
        print(f'instrumented verification FAILED: {len(violations)} violation(s)')
        for violation in violations:
            path = violation.get('path')
            print(f'  {violation["type"]}{f": {path}" if path else ""}')
        return 1
    print('instrumented verification passed')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', help='install directory (auto-detected if omitted)')
    ap.add_argument('--appid', default=None, help='Steam appid, for auto-detection')
    ap.add_argument('--out', default='inventory.json', help='where to write the inventory')
    verification = ap.add_mutually_exclusive_group()
    verification.add_argument('--verify', metavar='INVENTORY',
        help='re-hash the install and report drift against this inventory')
    verification.add_argument('--verify-instrumented', metavar='INVENTORY',
        help='verify original shipped files and inventory every added file as an overlay')
    ap.add_argument('--overlay-out',
        help='JSON artifact written by --verify-instrumented (required in that mode)')
    args = ap.parse_args()

    if args.verify_instrumented and not args.overlay_out:
        ap.error('--verify-instrumented requires --overlay-out')
    if args.overlay_out and not args.verify_instrumented:
        ap.error('--overlay-out requires --verify-instrumented')

    root = None
    if args.path:
        root = Path(args.path)
    elif not args.verify and not args.verify_instrumented:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from install_discovery import find_install, DEFAULT_APPID
        root = find_install(None, args.appid or DEFAULT_APPID)

    if args.verify:
        return verify(Path(args.verify), root)
    if args.verify_instrumented:
        return verify_instrumented(
            Path(args.verify_instrumented), root, Path(args.overlay_out))

    if not root.is_dir():
        sys.exit(f'{root} is not a directory')
    def progress(n, nbytes):
        print(f'  ... {n} files, {nbytes / 1e9:.2f} GB hashed',
              file=sys.stderr, flush=True)

    inv = scan(root, progress)
    Path(args.out).write_text(json.dumps(inv, indent=1))
    summarise(inv)
    print(f'\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
