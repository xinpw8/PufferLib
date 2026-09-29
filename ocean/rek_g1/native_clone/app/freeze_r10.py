"""Pin the follow-camera presentation change; physics, controls and pacing are unchanged."""
from pathlib import Path
import hashlib, json, sys, tarfile

root = Path(__file__).resolve().parent
archive = Path(sys.argv[1]).resolve()
manifest = root / 'SOURCE-MANIFEST.json'
parent = root / 'SOURCE-MANIFEST.parent-r9.json'
old = manifest.read_bytes()
assert hashlib.sha256(old).hexdigest() == '9138485e02c0c6ec69a8f84aad87f3469b046170e9db720433b2508a674481a2'
assert not archive.exists() and not parent.exists()
tests = json.loads((root / 'validation/tests-r10.json').read_text())
assert tests['exit_code'] == 0
for item in tests['files']:
    source = (root / item['path']).resolve()
    assert source.is_relative_to(root) and source.stat().st_size == item['bytes']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256'], item['path']
assert hashlib.sha256((root / 'validation/tests-r10.txt').read_bytes()).hexdigest() == tests['stdout_sha256']
parent.write_bytes(old)
paths = [p for p in sorted(root.rglob('*')) if p.is_file() and p != manifest and '__pycache__' not in p.parts]
records = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
manifest.write_text(json.dumps({'schema': 'rek.native_clone.app_source.v1', 'revision': 10,
    'full_official_parity_proven': False, 'runtime_throughput_measured': False,
    'scope': 'Third-person follow camera behind the human-controlled fighter (render-only). No native physics, control or pacing change.',
    'parent_manifest': {'path': parent.name, 'sha256': hashlib.sha256(old).hexdigest()},
    'files': records}, indent=2) + '\n', encoding='utf-8')
with tarfile.open(archive, 'x') as output:
    for p in paths + [manifest]:
        output.add(p, arcname='app/' + p.relative_to(root).as_posix(), recursive=False)
print(json.dumps({'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'files': len(records), 'archive_bytes': archive.stat().st_size,
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
