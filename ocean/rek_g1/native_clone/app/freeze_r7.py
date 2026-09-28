"""Freeze the tested absolute-deadline app without changing prior revisions."""
from pathlib import Path
import hashlib,json,sys,tarfile
root=Path(__file__).resolve().parent;archive=Path(sys.argv[1]).resolve()
manifest=root/'SOURCE-MANIFEST.json';parent=root/'SOURCE-MANIFEST.parent-r6.json'
assert not archive.exists() and not parent.exists()
old=manifest.read_bytes()
assert hashlib.sha256(old).hexdigest()=='c2b3b2b16802801ba2372c967a44be394d3846f08479b0500927aa085df77f70'
assert hashlib.sha256((root/'prepare.cjs').read_bytes()).hexdigest()=='199a198ad2d319590444acd8bbb40bd1346c9ef513c016ba495057b2f149dd0b'
assert json.loads((root/'validation/tests-r7.json').read_text())['exit_code']==0
assert not list(root.rglob('__pycache__'));parent.write_bytes(old)
files=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p!=manifest]
manifest.write_text(json.dumps({'schema':'rek.native_clone.app_source.v1','revision':7,'full_official_parity_proven':False,'runtime_throughput_measured':False,'parent_manifest':{'path':parent.name,'sha256':hashlib.sha256(old).hexdigest()},'files':files},indent=2)+'\n',encoding='utf-8')
with tarfile.open(archive,'x') as t:t.add(root,arcname='app')
print(json.dumps({'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'files':len(files),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
