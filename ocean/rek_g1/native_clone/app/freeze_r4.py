"""Freeze revision4 once, preserving the exact revision3 source manifest."""
from pathlib import Path
import hashlib,json,sys,tarfile
root=Path(__file__).resolve().parent
archive=Path(sys.argv[1]).resolve()
manifest=root/'SOURCE-MANIFEST.json'
parent=root/'SOURCE-MANIFEST.parent-r3.json'
if archive.exists() or parent.exists():
    raise SystemExit('Fresh revision4 archive and parent receipt required')
old=manifest.read_bytes()
metadata=json.loads(old)
if metadata.get('schema')!='rek.native_clone.app_source.v1' or metadata.get('revision') is not None:
    raise SystemExit('Expected original revision3 manifest')
tests=root/'validation/tests-r4-final.json'
if json.loads(tests.read_text(encoding='utf-8'))['exit_code']!=0:
    raise SystemExit('Final CPU tests must pass')
parent.write_bytes(old)
files=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and p!=manifest:
        files.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,
                      'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
result={'schema':'rek.native_clone.app_source.v1','revision':4,'full_official_parity_proven':False,
        'parent_manifest':{'path':parent.name,'sha256':hashlib.sha256(old).hexdigest()},'files':files}
manifest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
with tarfile.open(archive,'x') as output:
    output.add(root,arcname='app')
print(json.dumps({'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  'files':len(files),'archive':str(archive),'archive_bytes':archive.stat().st_size,
                  'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
