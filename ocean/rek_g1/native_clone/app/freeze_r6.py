"""Freeze one-arena preparation after genuine-model CPU validation."""
from pathlib import Path
import hashlib, json, sys, tarfile
root = Path(__file__).resolve().parent
archive = Path(sys.argv[1]).resolve()
manifest = root/'SOURCE-MANIFEST.json'
parent = root/'SOURCE-MANIFEST.parent-r5.json'
assert not archive.exists() and not parent.exists(), 'Fresh revision 6 archive and parent required'
old = manifest.read_bytes()
assert hashlib.sha256(old).hexdigest() == '8f34c4cdfaafa2955061a6b546d23964af4457b2af8d96a91553cdbb68e37b9e'
assert json.loads(old)['revision'] == 5
for name in ['tests-r6-local.json', 'tests-r6-real-models.json']:
    assert json.loads((root/'validation'/name).read_text())['exit_code'] == 0, name
assert not list(root.rglob('__pycache__'))
parent.write_bytes(old)
files = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in sorted(root.rglob('*')) if p.is_file() and p != manifest]
result = {'schema': 'rek.native_clone.app_source.v1', 'revision': 6,
          'full_official_parity_proven': False, 'runtime_throughput_measured': False,
          'parent_manifest': {'path': parent.name, 'sha256': hashlib.sha256(old).hexdigest()}, 'files': files}
manifest.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
with tarfile.open(archive, 'x') as output:
    output.add(root, arcname='app')
print(json.dumps({'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  'files': len(files), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
                  'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
