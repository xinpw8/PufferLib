"""Fresh sequential eager/graph workers, fixed commands, strict protocol equality.

This script runs GPU workers only when explicitly invoked. Rendering is excluded.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,selectors,struct,subprocess,time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def requests():
    result=[{'op':'snapshot'},{'op':'reset'},{'op':'snapshot'}]
    def step(command=None,**extra):result.append(dict(op='step',command=command or {},steps=1,**extra))
    # The original runtime countdown runs before any attack edge.
    for _ in range(200):step()
    for command in [{'forward':1},{'forward':-1},{'strafe':1},{'strafe':-1},
                    {'yaw':1},{'yaw':-1},{'forward':.25,'strafe':-.5,'yaw':.125},{}]:
        for _ in range(3):step(command)
    step({'moveIndex':0});step({'moveIndex':1});step({'cancelAction':True})
    for _ in range(123):step()
    result.extend([{'op':'reset'},{'op':'snapshot'}])
    for _ in range(3):step({'forward':.3,'strafe':.6,'yaw':-.2})
    result.extend([{'op':'reset'},{'op':'step','action':1,'steps':2},
                   {'op':'reset'},{'op':'step','command':{},'humanSide':1,'steps':2},
                   {'op':'reset'},{'op':'snapshot'},
                   {'op':'step','command':{},'steps':512,'benchmark':True}])
    return result

def differences(a,b,path='$',limit=32):
    found=[]
    def walk(x,y,p):
        if len(found)>=limit:return
        if type(x) is dict and type(y) is dict:
            if set(x)!=set(y):found.append({'path':p,'eager_keys':sorted(x),'graph_keys':sorted(y)});return
            for k in x:walk(x[k],y[k],p+'.'+k)
        elif type(x) is list and type(y) is list:
            if len(x)!=len(y):found.append({'path':p,'eager_length':len(x),'graph_length':len(y)});return
            for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,f'{p}[{i}]')
        elif isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
            if not math.isfinite(x) or not math.isfinite(y) or struct.pack('>d',float(x))!=struct.pack('>d',float(y)):
                found.append({'path':p,'eager':x,'graph':y,'absolute_error':abs(x-y)})
        elif type(x)!=type(y) or x!=y:found.append({'path':p,'eager':x,'graph':y})
    walk(a,b,path);return found

def run_worker(binary,config,environment,out,schedule,timeout):
    out.mkdir();config_path=out/'worker.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('REK_')};env.update(environment)
    replies=[];timings=[];terminal_count=0
    with (out/'stderr.log').open('w') as stderr,(out/'protocol.jsonl').open('w') as log:
        p=subprocess.Popen([str(binary),'--config',str(config_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,env=env,bufsize=1)
        selector=selectors.DefaultSelector();selector.register(p.stdout,selectors.EVENT_READ)
        def receive():
            if not selector.select(timeout):raise TimeoutError('Native worker response timeout')
            line=p.stdout.readline()
            if not line:raise RuntimeError(f'Native worker EOF, exit={p.poll()}')
            value=json.loads(line);log.write(json.dumps({'receive_monotonic_ns':time.monotonic_ns(),'reply':value})+'\n');log.flush();return value
        try:
            ready=receive();assert ready.get('event')=='ready',ready
            first_state=None
            for serial,request in enumerate(schedule,1):
                wire={k:v for k,v in request.items() if k!='benchmark'};wire['id']=serial
                start=time.monotonic_ns();log.write(json.dumps({'send_monotonic_ns':start,'request':wire})+'\n');log.flush()
                p.stdin.write(json.dumps(wire)+'\n');p.stdin.flush();reply=receive();elapsed=(time.monotonic_ns()-start)/1e9
                assert reply.get('id')==serial and reply.get('ok') is True,reply
                state=reply['state'];assert state['ok'] is True and state['failureBits']==0,state
                assert len(state['qpos'])==72 and len(state['qvel'])==70 and len(state['raw'])==446 and len(state['mask'])==66
                assert all(math.isfinite(v) for key in ['qpos','qvel','raw'] for v in state[key])
                if first_state is None:first_state=state;assert state['tick']==0,'Capture must not advance host tick'
                if request['op']=='reset':
                    assert state['tick']==0,'Cold reset tick'
                    assert not differences(state['qpos'],first_state['qpos']),'Cold reset qpos differs from initial state'
                    assert not differences(state['qvel'],first_state['qvel']),'Cold reset qvel differs from initial state'
                if request.get('benchmark'):
                    timings.append({'ticks':512,'arenas':config['arenas'],'wall_seconds':elapsed,'one_arena_control_sps':512/elapsed,'aggregate_arena_control_sps':512*config['arenas']/elapsed,'rendering':False,'includes_upload_status_snapshot_protocol':True})
                replies.append(reply);terminal_count+=len(reply.get('rounds',[]))
            assert terminal_count>0,'Fixed short-round schedule did not expose a terminal'
        finally:
            p.stdin.close()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
            selector.close();(out/'exit.json').write_text(json.dumps({'exit_code':p.returncode})+'\n')
        assert p.returncode==0,p.returncode
    (out/'timings.json').write_text(json.dumps(timings,indent=2)+'\n')
    return replies,timings,terminal_count

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--run-directory',default='run-r2')
    parser.add_argument('--binary',type=Path);parser.add_argument('--output',default='perf-graph-comparison-r1');parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args();root=args.root.resolve();binary=(args.binary or root/'build-r4/rek-native-clone').resolve()
    assert args.output and '/' not in args.output and '\\' not in args.output and args.output not in ['.','..']
    out=root/args.output;out.mkdir()
    run=root/args.run_directory;original_config=json.loads((run/'worker.json').read_text());server=json.loads((run/'server.json').read_text())
    assert original_config['backend']=='mujoco' and original_config['arenas']==4
    config=dict(original_config);config['round_seconds']=2
    schedule=requests();(out/'requests.json').write_text(json.dumps(schedule,indent=2)+'\n')
    pins={'binary':{'path':str(binary),'sha256':sha(binary)},'driver':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
          'original_worker':{'path':str(run/'worker.json'),'sha256':sha(run/'worker.json')},'server':{'path':str(run/'server.json'),'sha256':sha(run/'server.json')},
          'schedule_sha256':sha(out/'requests.json'),'authorized_overrides':{'round_seconds':2,'cuda_graph_step':[False,True]},'sequential_fresh_processes':True}
    (out/'INPUTS.json').write_text(json.dumps(pins,indent=2)+'\n')
    summary={'ok':False,'full_official_parity_claimed':False,'tolerance_relaxed':False}
    try:
        eager,et,er=run_worker(binary,dict(config,cuda_graph_step=False),server['backends'][0]['env'],out/'eager',schedule,args.timeout)
        graph,gt,gr=run_worker(binary,dict(config,cuda_graph_step=True),server['backends'][0]['env'],out/'graph',schedule,args.timeout)
        mismatches=[]
        for i,(a,b) in enumerate(zip(eager,graph)):
            ds=differences(a,b)
            if ds:mismatches.append({'request_index':i,'request':schedule[i],'differences':ds})
        assert len(eager)==len(graph)==len(schedule)
        summary.update(ok=not mismatches,replies_compared=len(eager),all_state_fields_rounds_command_events_exact=not mismatches,
            mismatch_replies=len(mismatches),mismatches=mismatches[:32],eager_terminals=er,graph_terminals=gr,
            eager_benchmark=et,graph_benchmark=gt,benchmark_speed_ratio=et[0]['wall_seconds']/gt[0]['wall_seconds'],
            scope='All functional single-control-step replies compare exactly, including tick/qpos/qvel/raw/masks/results and automatic round-reset progression. Benchmark512-step final state and all emitted terminals/command events compare exactly; intermediate benchmark poses are not returned. No frame rendering or official-game claim.')
        (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
        assert not mismatches,'Graph/eager protocol mismatch; do not enable graph by default'
    except Exception as e:
        summary['error']=str(e);(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');raise
    finally:
        summary['binary_unchanged']=sha(binary)==pins['binary']['sha256']
        (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
        records=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
        (out/'MANIFEST.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
