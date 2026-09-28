from pathlib import Path
import hashlib,json,shutil,datetime
root=Path(__file__).parent
dst=Path(r'\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-27\rek-native-clone-r1\controller-r1')
assert dst.parent.is_dir() and not dst.exists()
evidence=root/'evidence';evidence.mkdir(exist_ok=True)
oracle=root.parent/'original-history/runtime-r1/oracle.jsonl'
target=evidence/'original-history-oracle.jsonl'
assert not target.exists()
shutil.copyfile(oracle,target)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(oracle)==sha(target)=='8495728eb588d9ff51da93d1f6783fbb3c8757cc80f56a16f6892a85bd9f308c'
dst.mkdir()
records=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(root);q=dst/rel;q.parent.mkdir(parents=True,exist_ok=True)
    before=sha(p)
    with p.open('rb') as source,q.open('xb') as dest:shutil.copyfileobj(source,dest)
    after=sha(p);remote=sha(q)
    assert before==after==remote
    records.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':before})
receipt={'schema':'rek.native_clone.controller_publication.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(root),'destination':str(dst),'files':records,'count':len(records),'bytes':sum(x['bytes'] for x in records),'source_and_nas_readback_hashes_match':True,'production_worktree_unchanged':True}
payload=json.dumps(receipt,indent=2)+'\n'
(root/'PUBLICATION.json').write_text(payload)
(dst/'PUBLICATION.json').write_text(payload)
assert sha(root/'PUBLICATION.json')==sha(dst/'PUBLICATION.json')
print(json.dumps({'destination':str(dst),'count':len(records),'bytes':receipt['bytes'],'publication_sha256':sha(root/'PUBLICATION.json')},indent=2))
