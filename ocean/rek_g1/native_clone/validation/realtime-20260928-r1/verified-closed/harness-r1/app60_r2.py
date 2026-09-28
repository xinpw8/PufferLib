"""One private60s app test: fixed20ms native steps,50Hz input and20Hz image GETs.
Both existing human viewers are guarded read-only. Only our fresh port18774 app
receives input and is closed. All requests/states/image hashes are preserved.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys,time,urllib.request,urllib.error
from dual_guard import make_guard
from cpu_reference import HARNESS,HARNESS_SHA,BINARY,BINARY_SHA,ROOT,sha


def validate_state(value):
    if value.get('ok') is not True or value.get('failureBits')!=0:raise RuntimeError('Private evaluator failure')
    for field,count in [('qpos',72),('qvel',70),('raw',446),('mask',66)]:
        a=value.get(field)
        if not isinstance(a,list) or len(a)!=count or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in a):raise RuntimeError('Invalid native state '+field)
    if len(value.get('commandResults',[]))!=2 or len(value.get('actions',[]))!=2:raise RuntimeError('Missing native command/action state')


def packet(index):
    phase=(index//50)%8
    return {'seq':index+1,'held':[['W'],['S'],['A'],['D'],['Q'],['E'],[],[]][phase],
        'move':16+(index//75)%17 if index%75==0 else None,'cancelAction':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--launcher',type=Path,required=True);p.add_argument('--launcher-sha',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(json.dumps({'execute':False,'seconds':60,'input_target_hz':50,'image_get_target_hz':20,'state_get_target_hz':10,'cold_prefix_seconds':5}));return
    if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):raise RuntimeError('Assertions must remain enabled')
    if a.out.resolve().parent!=ROOT or a.run.resolve().parent!=ROOT or a.out.exists() or (a.run/'session.jsonl').exists():raise RuntimeError('Fresh private task run/output required')
    if sha(HARNESS)!=HARNESS_SHA or sha(BINARY)!=BINARY_SHA or sha(a.launcher)!=a.launcher_sha:raise RuntimeError('Source/binary pin changed')
    spec=importlib.util.spec_from_file_location('bench',HARNESS);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    identity,server,backend=b.prepared(a.run)
    assert server['port']==18774 and backend['executable']==str(BINARY) and identity['worker']['arenas']==1
    assert backend['env']['REK_PHYSICS_BACKEND']=='mujoco_cuda' and backend['env']['REK_ALLOW_CPU_EVALUATION']=='0'
    env={k:v for k,v in os.environ.items() if not k.startswith('REK_')};env.update(backend['env'])
    a.out.mkdir();directory=a.out/'app-process';directory.mkdir();guard=make_guard(b,120);child=None;summary={'success':False};samples=[];images=[];requests=[];base='http://127.0.0.1:18774';warm=None
    try:
        guard.start();child=b.Owned(['node',str(a.launcher),str(a.run)],env,directory,guard)
        assert child.receive(timeout=30).get('ready') is True
        deadline=time.monotonic()+30
        while True:
            guard.poll();state=b.http(base+'/api/snapshot')
            if state.get('ok') and not state.get('switching'):break
            if time.monotonic()>=deadline:raise RuntimeError('Private initialization timeout')
            time.sleep(.05)
        validate_state(state);assert state['paused'] and state['tick']==0
        child.capture_children()
        # An asynchronous renderer can become ready after the native worker.
        frame_deadline=time.monotonic()+15;frame_wait_started=time.monotonic();startup503=0
        while True:
            guard.poll()
            try:
                with urllib.request.urlopen(base+'/frame.png',timeout=2) as response:initial_png=response.read()
                if not initial_png.startswith(b'\x89PNG\r\n\x1a\n'):raise RuntimeError('Invalid initial renderer PNG')
                (a.out/'initial-frame.png').write_bytes(initial_png);break
            except urllib.error.HTTPError as error:
                if error.code!=503:raise
                startup503+=1
                if time.monotonic()>=frame_deadline:raise RuntimeError('Initial renderer did not become ready')
                time.sleep(.05)
        summary.update(initial_frame_wait_seconds=time.monotonic()-frame_wait_started,initial_frame_pending_503=startup503)
        b.http(base+'/api/play',{'paused':False})
        started=time.monotonic();end=started+60;next_input=next_frame=next_state=started;index=0;missed=0
        while time.monotonic()<end:
            guard.poll();now=time.monotonic()
            if now>=next_input:
                payload=packet(index);sent=time.monotonic_ns();reply=b.http(base+'/api/input',payload)
                if reply.get('ok') is not True or reply.get('accepted') is not True:raise RuntimeError('Private input rejected')
                requests.append({'sent_monotonic_ns':sent,'reply_monotonic_ns':time.monotonic_ns(),'payload':payload,'reply':reply});index+=1;next_input+=.02
                if time.monotonic()-next_input>.02:missed+=1;next_input=time.monotonic()
            if time.monotonic()>=next_state:
                state=b.http(base+'/api/state');validate_state(state)
                if state.get('paused') is not False:raise RuntimeError('Private app paused or match ended before60s')
                sample={'monotonic_ns':time.monotonic_ns(),'state':state};samples.append(sample);next_state+=.1
                if warm is None and time.monotonic()-started>=5:warm=sample
            if time.monotonic()>=next_frame:
                with urllib.request.urlopen(base+'/frame.png',timeout=2) as response:
                    raw=response.read();headers=dict(response.headers)
                if not raw.startswith(b'\x89PNG\r\n\x1a\n'):raise RuntimeError('Invalid rendered PNG')
                pin=hashlib.sha256(raw).hexdigest();expected=headers.get('X-Rek-Frame-Sha256') or headers.get('x-rek-frame-sha256')
                if expected!=pin:raise RuntimeError('Frame transport hash mismatch')
                images.append({'monotonic_ns':time.monotonic_ns(),'bytes':len(raw),'sha256':pin,'headers':headers});next_frame+=.05
            delay=min(next_input,next_state,next_frame,end)-time.monotonic()
            if delay>0:time.sleep(min(delay,.01))
        b.http(base+'/api/input',{'seq':index+1,'held':[],'move':None,'cancelAction':False});b.http(base+'/api/play',{'paused':True});final=b.http(base+'/api/snapshot');validate_state(final);assert final['paused'] is True
        elapsed=time.monotonic()-started
        pace=final['pace'];warm_pace=warm['state']['pace'];intervals=pace['activeIntervals']-warm_pace['activeIntervals'];wallms=pace['activeWallMs']-warm_pace['activeWallMs']
        summary.update(success=True,elapsed_seconds=elapsed,final_tick=final['tick'],pace=pace,steady_after5s={'active_intervals':intervals,'active_wall_ms':wallms,'real_time_ratio':intervals*20/wallms},input_packets=len(requests),input_packets_per_second=len(requests)/elapsed,input_deadline_misses=missed,image_gets=len(images),final=final,owned_server=child.identity)
    except BaseException as error:summary['error']=type(error).__name__+': '+str(error);raise
    finally:
        if child is not None:child.close(server=True)
        guard.close();summary['guard_failure']=guard.failure
        if guard.failure is not None:summary['success']=False
        health=a.run/'recorder-health.json'
        if health.exists():summary['closed_recorder_health']=json.loads(health.read_text())
        for name,data in [('summary.json',summary),('samples.json',samples),('image-receipts.json',images),('input-requests.json',requests),('guard.json',guard.dual_observations)]:
            (a.out/name).write_text(json.dumps(data,indent=2 if name=='summary.json' else None)+'\n')
        files=[{'path':f.relative_to(a.out).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(a.out.rglob('*')) if f.is_file()];(a.out/'MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='final'}),flush=True)


if __name__=='__main__':main()
