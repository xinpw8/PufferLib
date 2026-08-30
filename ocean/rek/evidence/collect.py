"""Run every non-interactive collection step, in order, into one directory.

    python collect.py --out evidence_out

Does steps 1, 4 and 4b — pin the build, probe the IL2CPP binaries, survey the
assets — and then reports what the package still needs. Each step records
whether it succeeded, so a partial run is legible rather than looking like a
build with nothing in it.

Step 2, the authority test, is not here and cannot be: it needs someone playing
the game and marking what they see while the network is cut. Run it separately:

    python authority_test.py --name REK --out evidence_out/authority_practice.json

Everything written here is JSON, and every file carries the same build
fingerprint. check_artifacts.py verifies that at the end — artifacts describing
different builds cannot be reasoned about together, and a collection run that
straddled a Steam update is exactly how that happens.

Read-only against the install.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_artifacts
import il2cpp_probe
import inventory as inventory_mod
import static_survey as static_survey_mod
from install_discovery import DEFAULT_APPID, find_install


def _step(log, name, fn):
    started = time.time()
    try:
        result = fn()
        log.append({'step': name, 'ok': True,
                    'seconds': round(time.time() - started, 1)})
        return result
    except SystemExit as e:
        log.append({'step': name, 'ok': False, 'error': f'exited: {e}'})
    except Exception as e:
        log.append({'step': name, 'ok': False,
                    'error': f'{type(e).__name__}: {e}',
                    'traceback': traceback.format_exc().splitlines()[-6:]})
    return None


def collect(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def do_inventory():
        inv = inventory_mod.scan(root)
        (out / 'inventory.json').write_text(json.dumps(inv, indent=1))
        return inv

    print(f'[1/3] hashing {root} ...')
    inv = _step(log, 'inventory', do_inventory)
    if inv is None:
        print('inventory failed; nothing downstream can run without it')
        return log, None
    print(f'      fingerprint {inv["build_fingerprint"][:16]}...  '
          f'{inv["file_count"]} files')

    print('[2/3] probing IL2CPP and native binaries ...')
    probe = _step(log, 'il2cpp_probe',
                  lambda: il2cpp_probe.probe(root, inv, out / 'il2cpp_probe.json'))
    if probe is not None:
        (out / 'il2cpp_probe.json').write_text(json.dumps(probe, indent=1))
        rt = probe.get('inference_runtime_present')
        print(f'      inference runtime linked: '
              f'{"yes" if rt else "no" if rt is False else "unknown"}')

    print('[3/3] surveying Unity assets ...')
    survey = _step(log, 'static_survey',
                   lambda: static_survey_mod.survey(root, inv,
                                                    out / 'static_survey.json'))
    if survey is not None:
        tm = (survey.get('settings') or {}).get('TimeManager')
        print(f'      tick rate: {"found" if tm else "NOT FOUND"}, '
              f'{len(survey.get("bodies", []))} body/joint components')

    (out / 'collect_log.json').write_text(json.dumps(
        {'install': str(root), 'steps': log,
         'collected_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
        indent=1))
    return log, inv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', help='install directory (auto-detected if omitted)')
    ap.add_argument('--appid', default=DEFAULT_APPID)
    ap.add_argument('--out', default='evidence_out')
    args = ap.parse_args()

    root = Path(args.path) if args.path else find_install(None, args.appid)
    if not root.is_dir():
        sys.exit(f'{root} is not a directory')

    out = Path(args.out)
    log, inv = collect(root, out)

    failed = [s for s in log if not s['ok']]
    if failed:
        print('\nsteps that did not complete:')
        for s in failed:
            print(f'  {s["step"]}: {s["error"]}')
            if s['step'] == 'static_survey' and 'UnityPy' in str(s['error']):
                print('    -> pip install UnityPy, then re-run')

    print('\n--- package state ---')
    results = check_artifacts.check(out)
    for r in results:
        print(f'  {r["state"]:<12} {r["artifact"]:<22} {r["detail"]}')
    stage = check_artifacts.stage_of(results)
    print(f'\nstage: {stage}')

    print('\nstill needed, in order:')
    print(f'  authority test   python authority_test.py --name REK '
          f'--out {out}/authority_practice.json')
    print('                   (and again during a live match, for contrast)')
    print('  recorder         written only once the above name the control and')
    print('                   state paths — every channel citing its source')
    print('  experiments      repeated controlled runs, then differ.py baseline')

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
