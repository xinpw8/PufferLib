"""Find the installed REK build and enumerate its Unity containers.

Carried over from the earlier extractor unchanged in behaviour: locating a Steam
install across Windows drives from WSL is generic, and none of it depends on any
claim about how REK works. Everything that inferred combat semantics from asset
names is gone.

    python install_discovery.py --list

Nothing here writes into the game install; it is read-only.
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

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

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true',
        help='show every Steam library and install candidate, then exit')
    ap.add_argument('--appid', default=DEFAULT_APPID,
        help=f'Steam appid to look up (default {DEFAULT_APPID}; playtest builds differ)')
    ap.add_argument('--path', help='install directory, if auto-detection misses it')
    ap.add_argument('--json', action='store_true', help='print the resolved root as JSON')
    args = ap.parse_args()

    if args.list:
        list_candidates(args.appid)
        return
    root = find_install(args.path, args.appid)
    if args.json:
        print(json.dumps({'install': str(root)}))
    else:
        print(root)


if __name__ == '__main__':
    main()
