"""Close only the identified, never-played revision-5 viewer before replacement."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, select, signal, time, urllib.request

ROOT=Path('/home/spark-advantage/rek-training/rek-native-clone-20260927-r1')
OWNED={2949931:131757839,2949941:131757847,2949942:131757847}

def identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    return int(fields[19])

def verify():
    for pid,start in OWNED.items():assert identity(pid)==start,'Viewer identity changed'
    health=json.loads((ROOT/'run-r5/recorder-health.json').read_text())
    assert health['pid']==2949931 and health['steps']==0,'Recorder has executed steps'
    with (ROOT/'run-r5/session.jsonl').open() as trace:
        for line in trace:
            event=json.loads(line)
            assert not (event.get('kind')=='worker_request' and event.get('op')=='step'),'A step was requested in this viewer'
    for port in [18771,18772]:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/snapshot',timeout=2) as response:
            state=json.load(response)
        assert state['ok'] and state['paused'],'Human viewer is active or unhealthy'
        if port==18772:
            assert state['tick']==0 and state['pace']['steps']==0,'Replacement candidate has been played'
            assert not state.get('held'),'Human input is held'
    process=json.loads((ROOT/'run-r5/server-process.json').read_text())
    assert process['pid']==2949931 and process['start_ticks']==131757839
    argv=Path('/proc/2949931/cmdline').read_bytes().split(b'\0')
    assert b'/home/spark-advantage/rek-training/rek-playback-speed-20260928-r1/app-stage-r1/app/launch_logged.cjs' in argv
    assert str(ROOT/'run-r5').encode() in argv

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    verify()
    receipt=ROOT/'run-r5/CLOSED-UNPLAYED.json'
    assert not receipt.exists(),'Closure already recorded'
    if not args.execute:
        print(json.dumps({'plan_only':True,'node':2949931,'port':18772,'tick':0}));return
    descriptors={pid:os.pidfd_open(pid) for pid in OWNED}
    try:
        verify()
        signal.pidfd_send_signal(descriptors[2949931],signal.SIGTERM)
        # Node's existing shutdown handler closes both owned workers and fsyncs.
        deadline=time.monotonic()+20
        pending=set(descriptors.values())
        while pending and time.monotonic()<deadline:
            ready,_,_=select.select(list(pending),[],[],min(.2,max(0,deadline-time.monotonic())))
            pending.difference_update(ready)
        assert not pending,'Owned viewer did not close; no escalation performed'
    finally:
        for descriptor in descriptors.values():os.close(descriptor)
    files=[]
    for path in sorted((ROOT/'run-r5').rglob('*')):
        assert not path.is_symlink(),'Unexpected symlink in closed run'
        if not path.is_file():continue
        name=path.relative_to(ROOT/'run-r5').as_posix()
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        files.append({'path':name,'bytes':path.stat().st_size,'sha256':digest})
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'closed_processes':OWNED,
            'steps':0,'human_demonstration':False,'manifest_scope':'all closed run files except this receipt','files':files}
    with receipt.open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
