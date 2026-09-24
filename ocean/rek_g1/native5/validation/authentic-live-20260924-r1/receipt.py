import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
base=root.parent
files=[]
for attempt in ['authentic-s801-retry9','authentic-s802-retry2']:
    for name in ['summary.json','relay.stdout.jsonl','encoder.stdout.jsonl','worker.stdout.jsonl']:
        file=base/attempt/'trial'/name
        before=file.stat()
        digest=hashlib.sha256()
        with file.open('rb') as h:
            for chunk in iter(lambda:h.read(1024*1024),b''):digest.update(chunk)
        after=file.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        files.append({'path':str(file),'bytes':after.st_size,'sha256':digest.hexdigest()})
artifacts=[]
for file in sorted(root.iterdir()):
    if file.is_file() and file.suffix in ['.py','.md','.json','.jsonl'] and file.name!='receipts.json':
        body=file.read_bytes()
        artifacts.append({'name':file.name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
with (root/'receipts.json').open('x') as h:
    json.dump({'source_files':files,'artifacts':artifacts,'no_game_connection':True,'no_runtime_changes':True},h,indent=2);h.write('\n')
print(json.dumps({'source_files':len(files),'artifact_files':len(artifacts),'receipt':str(root/'receipts.json')}))
