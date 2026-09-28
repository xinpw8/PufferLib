"""Run only the new native clone, preserving every request and reply."""
from pathlib import Path
import argparse,base64,hashlib,json,math,os,selectors,subprocess,time

def assert_no_deferred(reply):
    human=[e for e in reply.get('commandEvents',[]) if e['side']==0]
    assert not any(e.get('attempted') or e.get('accepted') for e in human),human
    feedback=reply['state']['commandResults'][0]
    assert not feedback['attempted'] and not feedback['accepted'],feedback

def wait_phase(call,phase,no_deferred=False,max_ticks=2000):
    state=call('snapshot')['state']
    for _ in range(max_ticks):
        if state['phase']==phase:return state
        assert not state['fightResult'],'Match completed before requested phase'
        result=call('step',steps=1,command={})
        if no_deferred:assert_no_deferred(result)
        state=result['state']
    assert state['phase']==phase,'Native phase wait exceeded bounded2000ticks'
    return state

def assert_inactive_edge(reply):
    attempts=[e for e in reply.get('commandEvents',[]) if e['side']==0 and e['attempted']]
    assert len(attempts)==1,attempts
    assert attempts[0]['accepted']==0 and attempts[0]['rejected']==1 and attempts[0]['reason']==4,attempts[0]
    assert not any(e['accepted'] for e in reply.get('commandEvents',[]) if e['side']==0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--label',default='verification-r5');parser.add_argument('--run-directory',default='run-r4');args=parser.parse_args()
    assert args.label and '/' not in args.label and '\\' not in args.label and args.label not in ['.','..']
    root=args.root.resolve();out=root/args.label;out.mkdir()
    run=root/args.run_directory
    config=json.loads((run/'worker.json').read_text());config['round_seconds']=2
    (out/'worker.short-round.json').write_text(json.dumps(config,indent=2)+'\n')
    server=json.loads((run/'server.json').read_text());env={k:v for k,v in os.environ.items() if not k.startswith('REK_')}
    env.update(server['backends'][0]['env']);env['OMP_NUM_THREADS']='2'
    log=(out/'protocol.jsonl').open('w');stderr=(out/'worker.stderr.log').open('w')
    p=subprocess.Popen([server['backends'][0]['executable'],'--config',str(out/'worker.short-round.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,env=env,bufsize=1)
    selector=selectors.DefaultSelector();selector.register(p.stdout,selectors.EVENT_READ)
    serial=0;checks=[];rounds=[];matches=[];events=[]
    def receive(timeout=120):
        if not selector.select(timeout):raise TimeoutError('New clone response timed out')
        line=p.stdout.readline()
        if not line:raise RuntimeError('New clone exited: '+str(p.poll()))
        try:value=json.loads(line)
        except json.JSONDecodeError:
            log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),'invalid_stdout':line})+'\n');log.flush();raise
        log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),'reply':value})+'\n');log.flush();return value
    def call(op,expect=True,**payload):
        nonlocal serial
        serial+=1;request=dict(id=serial,op=op,**payload);log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),'request':request})+'\n');log.flush()
        p.stdin.write(json.dumps(request)+'\n');p.stdin.flush();reply=receive()
        assert reply['id']==serial and reply['ok']==expect,reply
        if expect and 'state' in reply:
            s=reply['state'];assert s['failureBits']==0 and s['ok'],s
            assert len(s['qpos'])==72 and len(s['qvel'])==70 and len(s['raw'])==446
            assert all(math.isfinite(v) for key in ['qpos','qvel','raw'] for v in s[key])
            assert len(s['commandResults'])==2
        rounds.extend(reply.get('rounds',[]));events.extend(reply.get('commandEvents',[]))
        for s in reply.get('rounds',[]):
            if s['fightResult'] and s['fightWinner'] in [0,1]:matches.append({k:s[k] for k in ['tick','fightResult','fightWinner','score']})
        return reply
    try:
        ready=receive();assert ready['event']=='ready',ready
        first=call('snapshot')['state'];assert first['tick']==0
        for invalid in [{'command':{'forward':1.1}},{'command':{'moveIndex':17}},{'command':{'cancelAction':1}},{'command':{},'action':1}]:
            call('step',expect=False,**invalid);assert call('snapshot')['state']['tick']==0
        checks.append('Invalid commands rejected before advancing physics')
        frame=call('frame');(out/'initial.png').write_bytes(base64.b64decode(frame['png']));checks.append('Original mesh renderer produced PNG')
        result=call('step',steps=5,command={'forward':.25,'strafe':-.5,'yaw':.125})
        assert result['state']['tick']==5 and result['state']['actions'][0]==-1
        before=result['state']['tick'];call('step',expect=False,action=1)
        assert call('snapshot')['state']['tick']==before
        checks.append('Continuous combined command runs; categorical handoff rejected until explicit reset')
        call('reset');assert call('snapshot')['state']['qpos']==first['qpos']
        checks.append('Cold reset restores exact initial positions')
        # The short round can already be over after warmup. Read the native
        # phase rather than assuming that a particular tick accepts input.
        call('step',steps=200,command={})
        inactive=wait_phase(call,3)
        assert inactive['phase']==3 and not inactive['fightResult'],inactive
        moved=call('step',steps=10,command={'moveIndex':0})
        assert_inactive_edge(moved)
        checks.append('Inactive ten-step batch reports one rejected edge with clone INPUT_INACTIVE reason4 and zero accepted edges')
        active=wait_phase(call,2,no_deferred=True)
        assert active['phase']==2
        assert_no_deferred(call('step',steps=1,command={}))
        active=call('snapshot')['state'];assert active['phase']==2,active
        fresh=call('step',steps=1,command={'moveIndex':0})
        attempts=[e for e in fresh['commandEvents'] if e['side']==0 and e['attempted']]
        assert len(attempts)==1,attempts
        edge=attempts[0]
        assert (edge['accepted']==1 and edge['reason']==0) or (edge['accepted']==0 and edge['rejected']==1 and edge['reason'] in [2,3]),edge
        checks.append('Rejected inactive edge is not deferred across phase transition; fresh active edge receives native acceptance or recovery/punching rejection')
        call('step',command={'moveIndex':1})
        call('step',command={'cancelAction':True})
        call('step',steps=30,command={})
        # Vary all axes including simultaneous commands and release, preserving outcomes.
        for cmd in [{'forward':1},{'forward':-1},{'strafe':1},{'strafe':-1},{'yaw':1},{'yaw':-1},{'forward':.3,'strafe':.6,'yaw':-.2},{}]:
            call('step',steps=10,command=cmd)
        checks.append('Forward, backward, both strafe/yaw directions, combined analogue values and release stay finite')
        start=time.monotonic();bench=call('step',steps=512,command={});elapsed=time.monotonic()-start
        frame=call('frame');(out/'after-control.png').write_bytes(base64.b64decode(frame['png']))
        checks.append('512 native ticks without sticky runtime, scheduler, combat or physics errors')
        if not rounds:
            call('step',steps=512,command={})
        assert rounds,'No native round terminal captured'
        checks.append('Native short-round terminal and scores captured')
        call('reset');call('step',steps=2,action=1);call('reset')
        call('step',steps=2,humanSide=1,command={})
        checks.append('Categorical mode and side1 direct mode work after explicit reset')
        summary=dict(ok=True,scope='short-round integration verification; no official-game parity or policy evaluation claim',checks=checks,
            benchmark=dict(control_ticks=512,arenas=config['arenas'],wall_seconds=elapsed,one_arena_control_sps=512/elapsed,aggregate_arena_control_sps=512*config['arenas']/elapsed,physics_substeps_per_control=10,rendering_during_benchmark=False),
            command_events=events,rounds=[{k:s[k] for k in ['tick','roundResult','winner','score','fightResult','fightWinner']} for s in rounds],matches=matches)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    finally:
        p.stdin.close()
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=20)
        log.close();stderr.close();selector.close()
        (out/'exit.json').write_text(json.dumps({'worker_exit':p.returncode})+'\n')
        records=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.rglob('*')) if f.is_file()]
        (out/'MANIFEST.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
