"""Pin the tested viewer transport and performance diagnostics; physics is unchanged."""
from pathlib import Path
import hashlib, json, sys, tarfile

root = Path(__file__).resolve().parent
archive = Path(sys.argv[1]).resolve()
manifest = root / 'SOURCE-MANIFEST.json'
parent = root / 'SOURCE-MANIFEST.parent-r8.json'
old = manifest.read_bytes()
assert hashlib.sha256(old).hexdigest() == '456d96225d79a01821d5959742b9120bbcd14f8b58bf7094666abc2518e7fd10'
assert not archive.exists() and not parent.exists()
tests = json.loads((root / 'validation/tests-r9.json').read_text())
assert tests['exit_code'] == 0
for item in tests['files']:
    source = (root / item['path']).resolve()
    assert source.is_relative_to(root) and source.stat().st_size == item['bytes']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256'], item['path']
assert hashlib.sha256((root / 'validation/tests-r9.txt').read_bytes()).hexdigest() == tests['stdout_sha256']
parent.write_bytes(old)
paths = [p for p in sorted(root.rglob('*')) if p.is_file() and p != manifest and '__pycache__' not in p.parts]
records = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
manifest.write_text(json.dumps({'schema': 'rek.native_clone.app_source.v1', 'revision': 9,
    'full_official_parity_proven': False, 'runtime_throughput_measured': False,
    'scope': 'Conditional images, hidden-tab image suppression, recent pace and bounded performance diagnostics. No native physics change.',
    'parent_manifest': {'path': parent.name, 'sha256': hashlib.sha256(old).hexdigest()},
    'files': records}, indent=2) + '\n', encoding='utf-8')
with tarfile.open(archive, 'x') as output:
    for p in paths + [manifest]:
        output.add(p, arcname='app/' + p.relative_to(root).as_posix(), recursive=False)
print(json.dumps({'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'files': len(records), 'archive_bytes': archive.stat().st_size,
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
