"""Freeze source/evidence into fresh local and physical-NAS directories."""
from pathlib import Path
import datetime,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'public-ready-r1'
NAS=Path(r'\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-28\rek-parity-continuation-r1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not DEST.exists() and not NAS.exists(),'fresh destinations required'
    assert NAS.parent.is_dir(),'physical server project directory unavailable'
    selected={p.relative_to(ROOT):p for p in ROOT.iterdir() if p.is_file()}
    for name in ['native','comparison','revision-r2','revision-r3','capture-readback-r1','oracle-readback-r1','oracle-readback-r2','oracle-readback-r3','replay-readback-r2','replay-readback-r3','execution','isolation','boundary-test-r1']:
        for p in (ROOT/name).rglob('*'):
            rel=p.relative_to(ROOT)
            if not p.is_file() or p.is_symlink() or '__pycache__' in rel.parts or 'obj' in rel.parts:continue
            if name=='native' and any(x.startswith('build-') for x in rel.parts):continue
            if name=='revision-r2' and 'build-plugin-r2' in rel.parts and p.suffix!='.dll':continue
            if name=='boundary-test-r1' and p.name=='test_boundary':continue
            selected[rel]=p
    selected[Path('build-plugin-r1/RekSlerpOracle.dll')]=ROOT/'build-plugin-r1/RekSlerpOracle.dll'
    prior=Path(r'C:\rekagent\work\rek-native-clone-20260927-r1\remote_task.py')
    selected[Path('dependencies/previous_remote_task.py')]=prior
    reviews=Path(r'C:\rekagent\work\rek-github-checkpoint-20260928-r1\slerp-review')
    for p in reviews.iterdir():
        if p.is_file() and p.suffix in ('.json','.txt','.log'):selected[Path('review')/p.name]=p
    DEST.mkdir();records=[]
    for rel,p in sorted(selected.items(),key=lambda x:str(x[0])):
        assert not any(x in ('wineprefix','home','game','obj','__pycache__') for x in rel.parts)
        before=sha(p);out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
        assert sha(out)==before and sha(p)==before
        records.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':before})
    receipt={'schema':'rek.slerp_boundary.publication.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'local_package':str(DEST),'nas':str(NAS),'files':records,'bytes':sum(x['bytes'] for x in records),
             'raw_game_profile_auth_included':False,'general_slerp_replacement':False,'no_more_experiment_runs_planned':True}
    raw=(json.dumps(receipt,indent=2)+'\n').encode()
    (DEST/'PUBLICATION.json').write_bytes(raw)
    NAS.mkdir()
    for p in sorted(DEST.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(DEST);out=NAS/rel;out.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,out.open('xb') as dest:shutil.copyfileobj(src,dest,1024*1024)
        assert sha(out)==sha(p),'NAS readback mismatch'
    summary={'local':str(DEST),'nas':str(NAS),'files':len(records)+1,'bytes':receipt['bytes']+len(raw),'publication_sha256':sha(DEST/'PUBLICATION.json')}
    (ROOT/'PUBLICATION-RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
