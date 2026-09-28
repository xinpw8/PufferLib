from pathlib import Path
import hashlib,json,shutil

root=Path(__file__).resolve().parent
source=root/'payload/native_clone'
destination=Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile\ocean\rek_g1\native_clone')
records=[]
for path in sorted(source.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts:continue
    relative=path.relative_to(source);out=destination/relative
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        assert out.read_bytes()==data,'Preserving differing existing file: '+str(out)
    else:
        with out.open('xb') as f:f.write(data)
    assert hashlib.sha256(out.read_bytes()).hexdigest()==digest
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    records.append({'path':relative.as_posix(),'bytes':len(data),'sha256':digest})
(root/'COPIED.json').write_text(json.dumps({'destination':str(destination),'files':records},indent=2)+'\n')
print(json.dumps({'destination':str(destination),'files':len(records),'bytes':sum(f['bytes'] for f in records)}))
