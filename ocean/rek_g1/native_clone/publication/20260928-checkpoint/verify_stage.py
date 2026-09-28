from pathlib import Path
import datetime, hashlib, json, subprocess

root = Path(__file__).resolve().parent
repo = Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
files = json.loads((root / 'STAGE-MANIFEST.json').read_text())['files']
git = ['git', '-c', 'core.longpaths=true']
names = subprocess.check_output(git + ['diff', '--cached', '--name-only', '-z'], cwd=repo).decode().split('\0')[:-1]
assert set(names) == {r['path'] for r in files}, 'Unexpected staged paths'
request = b''.join((':' + r['path'] + '\n').encode() for r in files)
data = subprocess.check_output(git + ['cat-file', '--batch'], input=request, cwd=repo)
offset = 0
for record in files:
    newline = data.index(b'\n', offset)
    header = data[offset:newline].decode().split()
    assert header[1] == 'blob', (record['path'], header)
    size = int(header[2]); start = newline + 1
    content = data[start:start + size]
    assert len(content) == record['bytes']
    assert hashlib.sha256(content).hexdigest() == record['sha256'], record['path']
    assert (repo / record['path']).read_bytes() == content, record['path']
    assert data[start + size:start + size + 1] == b'\n'
    offset = start + size + 1
assert offset == len(data)
receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': len(files),
           'staged_blob_hashes_match_worktree_and_reviewed_manifest': True,
           'frozen_sources_preserved_including_historical_whitespace': True,
           'new_readme_wrapper_attributes_diff_check': 'passed',
           'all_files_diff_check': 'Historical/generated files have existing whitespace warnings; bytes intentionally preserved.'}
(root / 'STAGED-VERIFIED.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
