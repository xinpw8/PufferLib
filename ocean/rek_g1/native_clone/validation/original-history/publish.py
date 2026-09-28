import datetime,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEST=Path(r'R:\pufferlib\rek-evidence\2026-09-27\rek-native-clone-r1\original-history')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert DEST.parent.parent.is_dir(),'physical evidence root unavailable'
    assert not (DEST/'PUBLICATION.json').exists(),'publication is already frozen'
    selected=sorted(p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&{'obj','__pycache__'} and p.suffix not in {'.tar','.pdb'} and p.name!='PUBLICATION.json')
    receipts=[]
    for p in selected:
        rel=p.relative_to(ROOT);dest=DEST/rel;h=sha(p);size=p.stat().st_size
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert dest.is_file() and sha(dest)==h,'refuse differing destination: '+str(dest)
        else:
            with p.open('rb') as src,dest.open('xb') as out:shutil.copyfileobj(src,out)
        assert sha(p)==h and sha(dest)==h and dest.stat().st_size==size
        receipts.append({'path':str(rel),'bytes':size,'sha256':h})
    report={'schema':'rek.original_history.publication.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(ROOT),'destination':str(DEST),'files':receipts,'file_count':len(receipts),'total_bytes':sum(x['bytes'] for x in receipts),'all_destination_hashes_readback_verified':True}
    b=(json.dumps(report,indent=2)+'\n').encode()
    for base in [ROOT,DEST]:
        with (base/'PUBLICATION.json').open('xb') as f:f.write(b)
    assert (DEST/'PUBLICATION.json').read_bytes()==b
    print(json.dumps({'path':str(DEST),'files':len(receipts),'bytes':report['total_bytes'],'publication_sha256':sha(DEST/'PUBLICATION.json')}))
if __name__=='__main__':main()
