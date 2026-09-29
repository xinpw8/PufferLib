"""Read-only local transport and deployed control-module verification."""
from pathlib import Path
import datetime,hashlib,importlib.util,io,json,shlex,urllib.request
from PIL import Image
HERE=Path(__file__).resolve().parent
REMOTE='/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r7'
APP='/home/spark-advantage/rek-training/rek-controls-fix-20260929-r1/app-stage-r8/app'
spec=importlib.util.spec_from_file_location('remote_task',r'C:\rekagent\work\rek-native-clone-20260927-r1\remote_task.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)

def main():
    out=HERE/'deployment-verified';out.mkdir()
    with urllib.request.urlopen('http://127.0.0.1:18773/api/snapshot',timeout=5) as r:state=json.load(r)
    assert state['ok'] and not state['renderFailure']
    with urllib.request.urlopen('http://127.0.0.1:18773/controls.js',timeout=5) as r:controls=r.read()
    assert controls==(HERE/'app/league/public/controls.js').read_bytes()
    with urllib.request.urlopen('http://127.0.0.1:18773/frame.png',timeout=5) as r:
        image=r.read();frame_sha=hashlib.sha256(image).hexdigest();assert frame_sha==r.headers['X-Rek-Frame-Sha256']
    with Image.open(io.BytesIO(image)) as png:png.load();assert png.size==(1280,720)
    (out/'preview.png').write_bytes(image)
    with remote.connect() as client,client.open_sftp() as sftp:
        for name in ['STARTED.json','server-process.json','identity.json','server.json','worker.json']:
            with sftp.open(REMOTE+'/'+name,'rb') as f:payload=f.read()
            (out/name).write_bytes(payload)
        script='''from pathlib import Path
import hashlib,json,os,subprocess,urllib.request
node=3543413
def identity(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    return {'pid':pid,'start_ticks':int(fields[19]),'cpus':sorted(os.sched_getaffinity(pid))}
records=[identity(node)]+[identity(int(p)) for p in Path(f'/proc/{node}/task/{node}/children').read_text().split()]
assert records[0]['start_ticks']==133443582 and len(records)==3
assert all(r['cpus']==[5,6,7,8,9,15,16,17,18,19] for r in records)
module=Path(%r)/'league/input.cjs'
assert hashlib.sha256(module.read_bytes()).hexdigest()=='a0e6b8e7b60adf035c22db20cc61990c4700bd3d91d1b39375e2222f9022414a'
code="const {HumanInput}=require(process.argv[1]);console.log(JSON.stringify(Array.from({length:17},(_,i)=>{const h=new HumanInput();h.update({seq:1,held:[],move:16+i});return h.next().moveIndex;})));"
moves=json.loads(subprocess.check_output(['node','-e',code,str(module)]))
assert moves==[6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16]
old={}
for port in [18771,18772]:
    s=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/snapshot',timeout=2));old[port]={'ok':s['ok'],'paused':s['paused'],'tick':s['tick']}
health=json.loads(Path(%r).read_text());assert health['ok'] and not any(health.get(k) for k in ['recorderFailure','workerFailure','rendererFailure'])
print(json.dumps({'processes':records,'deployed_category_to_move':moves,'input_module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),'old_viewers_observed_only':old,'recorder':health}))
'''%(APP,REMOTE+'/recorder-health.json')
        _,stdout,stderr=client.exec_command(shlex.join(['python3','-c',script]));payload=stdout.read();error=stderr.read()
        assert stdout.channel.recv_exit_status()==0,error.decode();observed=json.loads(payload)
    nas_status=json.loads(Path(r'R:\pufferlib\rek-evidence\2026-09-29\rek-controls-fix-r1\viewer-r7\mirror-status.json').read_text())
    tunnel=json.loads((HERE/'viewer-r7-active/tunnel-status.json').read_text())
    assert nas_status['phase']=='watching' and not nas_status['errors'] and nas_status['source_recorder_health']['ok']
    assert tunnel['phase']=='listening' and not tunnel['errors']
    for item in [nas_status,tunnel]:assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(item['utc'])).total_seconds()<30
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':'http://127.0.0.1:18773/','paused_at_observation':state['paused'],'tick_at_observation':state['tick'],
             'frame_sha256':frame_sha,'frame_dimensions':[1280,720],'served_controls_sha256':hashlib.sha256(controls).hexdigest(),
             **observed,'nas_mirror':nas_status,'tunnel':tunnel,'native_input_sent_by_verification':False}
    (out/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['url','paused_at_observation','tick_at_observation','processes','deployed_category_to_move','old_viewers_observed_only']}))

if __name__=='__main__':main()
