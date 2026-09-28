from pathlib import Path
import base64,json,os,selectors,subprocess,time
root=Path('/home/spark-advantage/rek-training/rek-native-clone-20260927-r1')
out=root/'match-verification-r2';out.mkdir()
server=json.loads((root/'run-r4/server.json').read_text());config=json.loads((root/'run-r4/worker.json').read_text());config['round_seconds']=20
(out/'worker.json').write_text(json.dumps(config,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith('REK_')};env.update(server['backends'][0]['env'])
stderr=(out/'stderr.log').open('w');log=(out/'protocol.jsonl').open('w')
p=subprocess.Popen([server['backends'][0]['executable'],'--config',str(out/'worker.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,env=env,bufsize=1)
sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ);serial=0;rounds=[];events=[];started=time.monotonic()
def receive():
    assert sel.select(60),'Native worker timeout'
    line=p.stdout.readline();assert line,'Native worker EOF'
    v=json.loads(line);log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),'reply':v})+'\n');log.flush();return v
def call(op,**payload):
    global serial
    serial+=1;c=dict(id=serial,op=op,**payload);log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),'request':c})+'\n');log.flush()
    p.stdin.write(json.dumps(c)+'\n');p.stdin.flush();r=receive();assert r['ok'],r
    return r
try:
    assert receive()['event']=='ready'
    # Baseline participant stands idle; only the native Bot1 chooses attacks.
    for _ in range(40):
        result=call('step',command={},steps=512,stopAtRound=True);state=result['state'];assert not state['failureBits']
        rounds.extend(result.get('rounds',[]));events.extend(result.get('commandEvents',[]))
        if result.get('rounds'):print(json.dumps({'tick':state['tick'],'score':state['score'],'roundResult':state['roundResult'],'fightResult':state['fightResult']}),flush=True)
        if state['fightResult'] or state['tick']>=18000:break
    assert state['fightResult'] in [1,2], 'No completed native match within bounded test'
    assert max(s['roundNumber'] for s in rounds)>1, 'Rounds did not advance'
    completed=state.copy()
    result=call('step',command={},steps=300);state=result['state']
    assert state['fightResult']==completed['fightResult'] and state['fightWinner']==completed['fightWinner'], 'Completed result lost'
    assert not result.get('rounds'), 'Spurious extra terminal after completed match'
    frame=call('frame');(out/'final.png').write_bytes(base64.b64decode(frame['png']))
    summary=dict(ok=True,scope='20-second native Bot1 versus idle integration test; not human/policy/official evaluation',wall_seconds=time.monotonic()-started,
        final_tick=state['tick'],command_events=len(events),accepted_bot_attacks=sum(e['side']==1 and bool(e['accepted']) for e in events),
        rounds=[{k:s[k] for k in ['tick','roundNumber','score','falls','roundResult','winner','fightResult','fightWinner']} for s in rounds],
        final_fight_result=state['fightResult'],final_fight_winner=state['fightWinner'],completed_matches=int(state['fightResult'] in [1,2]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
finally:
    p.stdin.close()
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=20)
    log.close();stderr.close();sel.close();(out/'exit.json').write_text(json.dumps({'exit_code':p.returncode})+'\n')
