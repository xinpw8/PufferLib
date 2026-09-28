from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
root=Path(__file__).parent
out=Path(r'\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-27\rek-native-clone-r1\perf-graph-r1')
assert out.parent.is_dir() and not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tests=[]
for argv in [['python','test_source_contract.py'],['python','-m','unittest','test_compare_workers.py']]:
    r=subprocess.run(argv,cwd=root,text=True,capture_output=True)
    tests.append({'argv':argv,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0
(root/'TESTS-FINAL.json').write_text(json.dumps({'gpu_runs_by_this_agent':0,'cpu_lifecycle_result':json.loads((root/'tests-cpu.json').read_text()),'python_checks':tests},indent=2)+'\n')
out.mkdir();records=[]
sources=[(root,'source-and-analysis'),(root.parent/'remote-result/perf-graph-comparison-r1','executed-comparison')]
for base,prefix in sources:
    for p in sorted(base.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        rel=Path(prefix)/p.relative_to(base);q=out/rel;q.parent.mkdir(parents=True,exist_ok=True)
        before=sha(p)
        with p.open('rb') as a,q.open('xb') as b:shutil.copyfileobj(a,b)
        assert before==sha(p)==sha(q)
        records.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':before})
receipt={'schema':'rek.native_clone.optional_graph_publication.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'destination':str(out),'decision':'disabled_strict_equivalence_failed','files':records,'count':len(records),'bytes':sum(x['bytes']for x in records),'source_nas_hashes_match':True,'no_new_gpu_runs_by_publisher':True}
payload=json.dumps(receipt,indent=2)+'\n';(root/'PUBLICATION.json').write_text(payload);(out/'PUBLICATION.json').write_text(payload)
assert sha(root/'PUBLICATION.json')==sha(out/'PUBLICATION.json')
print(json.dumps({'path':str(out),'count':len(records),'bytes':receipt['bytes'],'sha256':sha(out/'PUBLICATION.json')}))
