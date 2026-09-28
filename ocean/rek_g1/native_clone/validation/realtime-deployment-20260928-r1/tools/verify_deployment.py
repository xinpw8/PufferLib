"""Read-only deployed API, image, process and backup verification."""
from pathlib import Path
import datetime,hashlib,importlib.util,io,json,urllib.request
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT='/home/spark-advantage/rek-training/rek-native-clone-20260927-r1'
spec=importlib.util.spec_from_file_location('remote_task',r'C:\rekagent\work\rek-native-clone-20260927-r1\remote_task.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)

def main():
    destination=HERE/'deployment-r6-verified';destination.mkdir()
    with urllib.request.urlopen('http://127.0.0.1:18772/api/snapshot',timeout=5) as r:state=json.load(r)
    assert state['ok'] and state['paused'] and state['tick']==0 and not state['renderFailure']
    with urllib.request.urlopen('http://127.0.0.1:18772/',timeout=5) as r:
        assert r.status==200 and b'REK' in r.read()
    with urllib.request.urlopen('http://127.0.0.1:18772/frame.png',timeout=5) as r:
        image=r.read();digest=hashlib.sha256(image).hexdigest()
        assert digest==r.headers['X-Rek-Frame-Sha256']==state['frame']['sha256']
    with Image.open(io.BytesIO(image)) as png:png.load();size=png.size;assert size==(1280,720)
    (destination/'preview.png').write_bytes(image)
    with remote.connect() as client,client.open_sftp() as sftp:
        for name in ['STARTED.json','server-process.json','identity.json','server.json','worker.json','startup-preview.png']:
            with sftp.open(ROOT+'/run-r6/'+name,'rb') as f:payload=f.read()
            (destination/name).write_bytes(payload)
        script='''from pathlib import Path
import json,os,urllib.request
node=3297624
def identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    return {'pid':pid,'start_ticks':int(fields[19]),'argv':[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0') if v],'cpus':sorted(os.sched_getaffinity(pid))}
records=[identity(node)]+[identity(int(pid)) for pid in Path(f'/proc/{node}/task/{node}/children').read_text().split()]
assert records[0]['start_ticks']==132749534 and len(records)==3
assert all(r['cpus']==[5,6,7,8,9,15,16,17,18,19] for r in records)
for pid,start in [(1092025,125031755),(1092037,125031762)]:assert identity(pid)['start_ticks']==start
old=json.load(urllib.request.urlopen('http://127.0.0.1:18771/api/snapshot',timeout=2))
health=json.loads(Path(%r).read_text())
assert health['ok'] and not any(health.get(k) for k in ['recorderFailure','workerFailure','rendererFailure'])
print(json.dumps({'new_processes':records,'old_viewer':{'ok':old['ok'],'paused':old['paused'],'tick':old['tick']},'recorder':health}))
'''%(ROOT+'/run-r6/recorder-health.json')
        import shlex
        _,out,err=client.exec_command(shlex.join(['python3','-c',script]));payload=out.read();error=err.read()
        assert out.channel.recv_exit_status()==0,error.decode();processes=json.loads(payload)
    mirror=json.loads((HERE/'viewer-r6-active/mirror-status.json').read_text())
    tunnel=json.loads((HERE/'viewer-r6-active/tunnel-status.json').read_text())
    assert mirror['phase'] in ['watching','copying'] and not mirror['errors']
    nas_status=json.loads(Path(r'R:\pufferlib\rek-evidence\2026-09-28\rek-realtime-r1\viewer-r6\mirror-status.json').read_text())
    assert nas_status['phase']=='watching' and not nas_status['errors'] and nas_status['source_recorder_health']['ok']
    assert tunnel['phase']=='listening' and not tunnel['errors']
    for item in [mirror,tunnel,nas_status]:assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(item['utc'])).total_seconds()<30
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':'http://127.0.0.1:18772/','paused':True,'tick':0,
             'frame_sha256':digest,'frame_dimensions':size,'processes':processes,'mirror':mirror,'nas_last_success':nas_status,'tunnel':tunnel,
             'performance_trial':'app60-result-r7-affinity','deployed_viewer_driven_by_agent':False}
    (destination/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'verified':True,'url':receipt['url'],'png':digest,'old_viewer':processes['old_viewer'],'children':processes['new_processes']}))

if __name__=='__main__':main()
