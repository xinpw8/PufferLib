import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'runtime-r1/closed-r1.tar'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='e684f9cdfccfb5bbd326cbff23517a0858233ca0f028da05dbe3d92407881af5'
out=ROOT/'runtime-r1/closed';out.mkdir()
with tarfile.open(p) as t:
    for member in t:
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        target=out/member.name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(t.extractfile(member).read())
manifest=json.loads((out/'CLOSED-SOURCE-MANIFEST.json').read_bytes())
for entry in manifest['files']:
    b=(out/entry['path']).read_bytes()
    assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
trace=out/'unity-harness-r1/out/oracle/oracle.jsonl'
assert trace.read_bytes()==(ROOT/'runtime-r1/oracle.jsonl').read_bytes()
receipt={'files':len(manifest['files']),'bytes':sum(x['bytes'] for x in manifest['files']),
    'all_source_local_hashes_equal':True,'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'manifest_sha256':hashlib.sha256((out/'CLOSED-SOURCE-MANIFEST.json').read_bytes()).hexdigest()}
with (ROOT/'runtime-r1/DOWNLOAD-VERIFIED.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
