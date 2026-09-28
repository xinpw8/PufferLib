'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),net=require('node:net');
const {once}=require('node:events'),{setTimeout:delay}=require('node:timers/promises');
const {League}=require('./league.cjs'),{serve}=require('./server.cjs');
const {TestRenderer}=require('./test_renderer.cjs');
const {phaseLabel,roundStatus}=require('./public/state_labels.js');
test('native phase labels do not invent a timer or treat round terminal as match complete',()=>{
  assert.deepEqual([0,1,2,3,4].map(phaseLabel),['Idle','Countdown','Fighting','Between rounds','Match complete']);
  assert.equal(phaseLabel(99),'Phase unknown');
  assert.equal(roundStatus({phase:3,roundNumber:2,terminal:1,paused:false,intermissionSeconds:999}),'Round 2 · Between rounds');
  assert.equal(roundStatus({phase:1,roundNumber:1,paused:true}),'Round 1 · Countdown · paused');
  assert.equal(roundStatus({phase:4,roundNumber:3,paused:true}),'Round 3 · Match complete');
});
test('launched zero-intermission server continues stepping native between-round phase without an extra browser timer',async()=>{
  const launch=fs.readFileSync(path.join(__dirname,'../launch_logged.cjs'),'utf8');
  assert(launch.includes('{Worker:RecordedWorker,Renderer:RecordedWorker,intermissionMs:0}'));
  const listener=net.createServer();listener.listen(0,'127.0.0.1');await once(listener,'listening');
  const port=listener.address().port;await new Promise(r=>listener.close(r));
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-native-phase-'));
  const league=new League({file:path.join(dir,'league.json')});
  league.registerPolicy({id:'bot1',label:'Bot1',backend:'mujoco',kind:'scripted',configHash:'c'.repeat(64),scriptedVersion:'test'});
  const cfg=path.join(dir,'server.json'),wc=path.join(dir,'worker.json');fs.writeFileSync(wc,'{"round_seconds":120}');
  fs.writeFileSync(cfg,JSON.stringify({port,leagueFile:league.file,backends:[{id:'mujoco',workerConfig:wc,configHash:'c'.repeat(64)}]}));
  let steps=0;
  class Worker{
    constructor(){this.ready=Promise.resolve();this.state={ok:true,tick:0,phase:2,roundNumber:1,terminal:0,fightResult:0,fightWinner:-1,score:[0,0],roundResult:0,winner:-1,qpos:Array(72).fill(0)};}
    async request(op){
      if(op==='frame')return {png:''};
      if(op==='step'){
        steps++;this.state={...this.state,tick:steps,phase:3,terminal:steps===1?1:0,roundResult:1,winner:0,score:[8,4]};
        return {state:{...this.state},rounds:steps===1?[{...this.state}]:[]};
      }
      return {state:{...this.state}};
    }
    async close(){}
  }
  let service;
  try{
    service=await serve(cfg,{Worker,Renderer:TestRenderer,intermissionMs:0});if(!service.server.listening)await once(service.server,'listening');
    const base=`http://127.0.0.1:${port}`;
    const post=async(route,value)=>{const r=await fetch(base+route,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});assert.equal(r.status,200);return r.json();};
    await post('/api/select',{backend:'mujoco',opponent:'bot1',humanSide:0,roundSeconds:120});
    await post('/api/play',{paused:false});
    const deadline=Date.now()+700;while(steps<5&&Date.now()<deadline)await delay(10);
    await post('/api/play',{paused:true});
    assert(steps>=5,'native intermission must keep advancing at control cadence');
    const state=await (await fetch(base+'/api/snapshot')).json();
    assert.equal(state.phase,3);assert.equal(state.intermissionSeconds,0);assert.equal(state.session.completedRounds,1);
    assert.equal(state.session.completedMatches,0,'no inferred match result');
  }finally{
    if(service){const closed=once(service.server,'close');await service.close();await closed;}
    for(const name of fs.readdirSync(dir))fs.unlinkSync(path.join(dir,name));fs.rmdirSync(dir);
  }
});
