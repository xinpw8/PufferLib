from pathlib import Path
import collections, datetime, hashlib, json, shutil, subprocess

root = Path(__file__).resolve().parent
repo = Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
package = repo / 'ocean/rek_g1/native_clone'
out = package / 'validation/root_integration'
out.mkdir(parents=True, exist_ok=True)
for name in ('app.txt', 'app.exit.txt'):
    source = root / 'integration-tests' / name
    target = out / name
    if target.exists():
        assert target.read_bytes() == source.read_bytes()
    else:
        shutil.copyfile(source, target)
assert (out / 'app.exit.txt').read_text().strip() == '0'
receipt = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Actual copied repository package; CPU-only checks. Live viewer was not changed.',
    'checks': [
        {'command': 'node --test league/input.test.cjs league/controls.test.cjs league/paced_loop.test.cjs league/human_session.test.cjs league/server.test.cjs league/protocol.test.cjs league/native_phase.test.cjs',
         'cwd': 'app', 'exit_code': 0, 'passed': 28, 'failed': 0, 'transcript': 'app.txt'},
        {'command': r'C:\Python312\python.exe -m unittest -v test_passive.py',
         'cwd': 'passive_support', 'exit_code': 0, 'passed': 11, 'failed': 0,
         'evidence': 'Observed tool result during this publication turn; no separate complete transcript was saved.'},
        {'command': 'bash -n ocean/rek_g1/native_clone/build.sh', 'cwd': 'repository root', 'exit_code': 0}
    ]
}
(out / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
files = []
for path in sorted(package.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
        continue
    data = path.read_bytes()
    assert len(data) < 100 * 1024 * 1024, f'GitHub file limit: {path}'
    files.append({'path': path.relative_to(repo).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
(root / 'STAGE-MANIFEST.json').write_text(json.dumps({'files': files}, indent=2) + '\n', encoding='utf-8')
(root / 'stage-paths.bin').write_bytes(b'\0'.join(f['path'].encode('utf-8') for f in files) + b'\0')
print(json.dumps({'files': len(files), 'bytes': sum(f['bytes'] for f in files),
                  'largest': sorted(files, key=lambda f: -f['bytes'])[:6]}, indent=2))
