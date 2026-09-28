"""Create a source manifest and archive once; never include live run outputs."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
manifest=ROOT/'SOURCE-MANIFEST.json'
archive=ROOT.parent/'app-source-r3.tar'
if manifest.exists() or archive.exists():
    raise SystemExit('Fresh manifest/archive required')
files=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file():
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest.write_text(json.dumps({'schema':'rek.native_clone.app_source.v1','full_official_parity_proven':False,'files':files},indent=2)+'\n',encoding='utf-8')
with tarfile.open(archive,'x') as t:
    t.add(ROOT,arcname=ROOT.name)
print(json.dumps({'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'files':len(files),'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
