"""Freeze the tested manual control translation while retaining app-r7 identity."""
from pathlib import Path
import hashlib,json,sys,tarfile
root=Path(__file__).resolve().parent;archive=Path(sys.argv[1]).resolve()
manifest=root/'SOURCE-MANIFEST.json';parent=root/'SOURCE-MANIFEST.parent-r7.json'
assert not archive.exists() and not parent.exists()
old=manifest.read_bytes()
assert hashlib.sha256(old).hexdigest()=='6be9fa74673eca108da8c07cd414d27d55b6cc9f625db1ab70e2f5c3e6f02cca'
assert json.loads((root/'validation/tests-r8.json').read_text())['exit_code']==0
assert json.loads((root/'validation/controls-r8-evidence.json').read_text())['negative_original_boundary']['tests_failed']==3
assert not list(root.rglob('__pycache__'));parent.write_bytes(old)
files=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p!=manifest]
manifest.write_text(json.dumps({'schema':'rek.native_clone.app_source.v1','revision':8,'full_official_parity_proven':False,'runtime_throughput_measured':False,'parent_manifest':{'path':parent.name,'sha256':hashlib.sha256(old).hexdigest()},'files':files},indent=2)+'\n',encoding='utf-8')
with tarfile.open(archive,'x') as t:t.add(root,arcname='app')
print(json.dumps({'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'files':len(files),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
