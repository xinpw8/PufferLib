"""Preserve only closed task-owned diagnostic output and pinned stage receipts."""
import datetime, hashlib, json, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SESSION=ROOT/'unity-harness-r1'
NAME='rek-native-clone-history-20260927-r1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    state=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
    assert not state['State']['Running'] and state['State']['Status']=='exited'
    assert state['State']['ExitCode']==0 and not state['State']['OOMKilled']
    files=list((SESSION/'out').rglob('*'))
    files += [p for p in SESSION.iterdir() if p.is_file()]
    files += list((SESSION/'input').rglob('*'))
    files += [SESSION/'game/BepInEx/LogOutput.log',SESSION/'game/REK_COMPOSER_ORACLE_ISOLATED.json']
    files += list((ROOT/'isolation').glob('*receipt.json'))
    files=sorted(set(p for p in files if p.is_file()))
    assert all(not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()) for p in files)
    entries=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    receipt={'schema':'rek.original_history.closed_capture.v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'container_id':state['Id'],'image_id':state['Image'],'state':state['State'],'files':entries,
        'full_private_runtime_not_duplicated':'575 static source copies retained remotely; byte inventory in STAGED.json'}
    with (ROOT/'CLOSED-SOURCE-MANIFEST.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    with tarfile.open(ROOT/'closed-r1.tar','x') as t:
        for p in files:t.add(p,arcname=str(p.relative_to(ROOT)))
        t.add(ROOT/'CLOSED-SOURCE-MANIFEST.json',arcname='CLOSED-SOURCE-MANIFEST.json')
    for p,e in zip(files,entries):assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes']
    print(json.dumps({'files':len(entries),'bytes':sum(x['bytes'] for x in entries),'archive_sha256':sha(ROOT/'closed-r1.tar'),'manifest_sha256':sha(ROOT/'CLOSED-SOURCE-MANIFEST.json')}))
if __name__=='__main__':main()
