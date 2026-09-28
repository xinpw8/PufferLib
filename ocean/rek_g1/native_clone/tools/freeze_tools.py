from pathlib import Path
import ast,hashlib,json,tarfile
root=Path(__file__).resolve().parent
out=root/'MANIFEST.json';archive=root.parent/'tools-r5.tar'
assert not out.exists() and not archive.exists()
files=[]
for p in sorted(root.iterdir()):
    if not p.is_file():continue
    if p.suffix=='.py':ast.parse(p.read_text())
    files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
out.write_text(json.dumps({'schema':'rek.native_clone.verification_tools.v1','executed_gpu':False,'files':files},indent=2)+'\n')
with tarfile.open(archive,'x') as t:
    for p in sorted(root.iterdir()):
        if p.is_file():t.add(p,arcname='tools-r5/'+p.name)
print(json.dumps({'manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':files}))
