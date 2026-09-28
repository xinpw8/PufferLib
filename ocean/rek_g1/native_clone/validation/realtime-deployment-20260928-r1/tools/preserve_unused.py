"""Hash-verify every closed unused-viewer file on the physical server."""
from pathlib import Path,PurePosixPath
import hashlib,importlib.util,json,stat

HERE=Path(__file__).resolve().parent
NAS=Path(r'R:\pufferlib\rek-evidence\2026-09-28\rek-realtime-r1\closed-unused-run-r5')
SOURCE='/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r5'
spec=importlib.util.spec_from_file_location('remote_task',r'C:\rekagent\work\rek-native-clone-20260927-r1\remote_task.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)

def main():
    local=HERE/'closed-unused-run-r5'
    assert not local.exists() and not NAS.exists()
    with remote.connect() as client,client.open_sftp() as sftp:
        with sftp.open(SOURCE+'/CLOSED-UNPLAYED.json','rb') as f:data=f.read()
        closed=json.loads(data);assert closed['steps']==0 and closed['human_demonstration'] is False
        records=closed['files']+[{'path':'CLOSED-UNPLAYED.json','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]
        for record in records:
            name=record['path'];relative=PurePosixPath(name)
            assert not relative.is_absolute() and all(p not in ('..','.') and '\\' not in p and ':' not in p for p in relative.parts)
            source=SOURCE+'/'+name
            assert stat.S_ISREG(sftp.lstat(source).st_mode)
            with sftp.open(source,'rb') as f:payload=f.read()
            assert len(payload)==record['bytes'] and hashlib.sha256(payload).hexdigest()==record['sha256'],name
            for parent in [local,NAS]:
                path=parent.joinpath(*relative.parts);path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:f.write(payload)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    receipt={'source':SOURCE,'local':str(local),'nas':str(NAS),'source_local_nas_hashes_equal':True,'files':records}
    content=(json.dumps(receipt,indent=2)+'\n').encode()
    for parent in [local,NAS]:
        with (parent/'FULL-BACKUP-VERIFIED.json').open('xb') as f:f.write(content)
    print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records),'nas':str(NAS),'verified':True}))

if __name__=='__main__':main()
