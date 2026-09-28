"""Snapshot the actual working files, including local edits; never checkout/reset."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path(r'C:\Users\Daniel\codex-rek-puffysics-training-profile')
G1=REPO/'ocean/rek_g1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    before=subprocess.run(['git','-C',str(REPO),'status','--short'],capture_output=True,text=True,check=True).stdout
    paths=[p for p in G1.iterdir() if p.is_file() and p.suffix in ('.c','.cu','.h')]
    paths += [G1/'native5'/name for name in ('robot_state.cu','robot_state.cuh','motion_assets.cu','motion_assets.cuh','runtime.cu','runtime_api.h','test_robot_state.cu')]
    paths += list((G1/'test_semantic_direct_support').glob('*'))
    rows=[]
    for src in sorted(paths):
        rel=src.relative_to(REPO);raw=src.read_bytes();h=hashlib.sha256(raw).hexdigest()
        for folder in ('original','source-r1'):
            dest=ROOT/folder/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as out:out.write(raw)
            if sha(dest)!=h:raise RuntimeError('snapshot mismatch')
        if sha(src)!=h:raise RuntimeError('source changed during snapshot')
        rows.append({'path':rel.as_posix(),'bytes':len(raw),'sha256':h})
    receipt={'schema':'rek.native_clone.controller_snapshot.v1','utc':datetime.now(timezone.utc).isoformat(),'source_root':str(REPO),'source_files':rows,'git_status':before,'working_tree_modified_by_snapshot':False}
    with (ROOT/'SNAPSHOT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')
    with (ROOT/'git-status-before.txt').open('x',encoding='utf-8') as f:f.write(before)
    print(json.dumps({'files':len(rows),'snapshot_sha256':sha(ROOT/'SNAPSHOT.json')}))

if __name__=='__main__':main()
