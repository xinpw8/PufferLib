from pathlib import Path
import hashlib,json,shutil,datetime
src=Path(__file__).resolve().parent
dst=Path(r'R:\pufferlib\rek-evidence\2026-09-27\rek-native-clone-r1\match-lifecycle-r1')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
dst.mkdir(parents=True,exist_ok=True)
files=[]
for p in sorted(src.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='PUBLICATION.json': continue
    rel=p.relative_to(src);target=dst/rel;sha=digest(p)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=sha:raise RuntimeError(f'differing existing destination {target}')
    else:shutil.copyfile(p,target)
    if digest(target)!=sha:raise RuntimeError(f'readback differs {target}')
    files.append(dict(path=str(rel).replace('\\','/'),bytes=p.stat().st_size,sha256=sha))
receipt=dict(schema='rek.native_match_lifecycle.publication.v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),destination=str(dst),files=files,file_count=len(files),bytes=sum(x['bytes'] for x in files),all_readbacks_equal=True)
out=dst/'PUBLICATION.json'
if out.exists():raise RuntimeError('preserve existing publication receipt')
out.write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(out,src/'PUBLICATION.json')
print(json.dumps({'path':str(out),'sha256':digest(out),'files':len(files),'bytes':receipt['bytes']}))
