'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),net=require('node:net'),http=require('node:http');
const {once}=require('node:events'),{setTimeout:delay}=require('node:timers/promises');
const {League}=require('./league.cjs'),{serve}=require('./server.cjs');
const {TestRenderer}=require('./test_renderer.cjs');
test('paused protocol preserves continuous inputs, native events and exclusive requests across delayed HTTP bodies',async()=>{
  const reservation=net.createServer();reservation.listen(0,'127.0.0.1');await once(reservation,'listening');
  const port=reservation.address().port;await new Promise(r=>reservation.close(r));
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-clone-protocol-'));
  const league=new League({file:path.join(dir,'league.json')});
  league.registerPolicy({id:'bot1',label:'Bot1',backend:'mujoco',kind:'scripted',configHash:'b'.repeat(64),scriptedVersion:'test'});
  const cfg=path.join(dir,'server.json'),wc=path.join(dir,'worker.json');fs.writeFileSync(wc,'{"round_seconds":120}');
  fs.writeFileSync(cfg,JSON.stringify({port,leagueFile:league.file,backends:[{id:'mujoco',workerConfig:wc,configHash:'b'.repeat(64)}]}));
  let fake,active=0,maxActive=0,releaseStep,enteredStep,mode=null;
  class Worker{
    constructor(){fake=this;this.ready=Promise.resolve();this.calls=[];this.reset();}
    reset(){mode=null;this.state={ok:true,tick:0,score:[0,0],wins:[0,0],terminal:0,roundResult:0,roundNumber:1,
      fightResult:0,fightWinner:-1,phase:2,qpos:Array(72).fill(.25),qvel:Array(70).fill(.125),raw:Array(446).fill(0),mask:Array(66).fill(1)};}
    async request(op,args={}){
      active++;maxActive=Math.max(maxActive,active);this.calls.push({op,args});
      try{
        if(op==='frame')return {png:''};
        if(op==='reset')this.reset();
        if(op==='step'){
          const next=args.command?'direct':'categorical';if(mode&&mode!==next)throw Error('Reset required to switch ownership');mode=next;
          if(enteredStep){enteredStep();enteredStep=null;await new Promise(r=>releaseStep=r);}
          this.state.tick+=args.steps;
          return {state:{...this.state},rounds:[],commandEvents:[{tick:this.state.tick,side:0,attempted:1,accepted:1,reason:0}]};
        }
        return {state:{...this.state}};
      }finally{active--;}
    }
    async close(){}
  }
  let service;
  try{
    service=await serve(cfg,{Worker,Renderer:TestRenderer});if(!service.server.listening)await once(service.server,'listening');
    const base=`http://127.0.0.1:${port}`;
    async function api(route,value){const r=await fetch(base+route,value===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});return {status:r.status,data:await r.json()};}
    function slow(route){
      let done;const result=new Promise(resolve=>done=resolve);
      const req=http.request(base+route,{method:'POST',headers:{'Content-Type':'application/json'}},res=>{let s='';res.on('data',x=>s+=x);res.on('end',()=>done({status:res.statusCode,data:JSON.parse(s)}));});
      req.flushHeaders();req.write(' ');return {finish:v=>{req.end(JSON.stringify(v));return result;}};
    }
    assert.equal((await api('/api/select',{backend:'mujoco',opponent:'bot1',humanSide:0,roundSeconds:120})).status,200);
    const command={forward:.3,strafe:-.7,yaw:.2,moveIndex:4,cancelAction:false};
    const waitingPlay=slow('/api/play');await delay(30);
    const entered=new Promise(resolve=>enteredStep=resolve);
    const stepping=api('/api/step',{command,steps:3});await entered;
    assert.equal((await waitingPlay.finish({paused:false})).status,409,'play body completion cannot bypass exclusive paused step');
    assert.equal((await api('/api/step',{command,steps:1})).status,409);
    releaseStep();const result=await stepping;
    assert.equal(result.status,200);assert.equal(result.data.state.paused,true);assert.equal(result.data.state.switching,false);
    assert.deepEqual(fake.calls.find(c=>c.op==='step').args,{command,steps:3,humanSide:0,stopAtRound:true});
    assert.equal(result.data.commandEvents[0].accepted,1);assert.equal(result.data.state.qpos.length,72);assert.equal(result.data.state.qvel.length,70);
    const tick=fake.state.tick;assert.equal((await api('/api/step',{action:0})).status,400);assert.equal(fake.state.tick,tick,'ownership error does not mutate native state');
    const waitingInput=slow('/api/input');await delay(30);
    assert.equal((await api('/api/reset',{})).status,200);
    assert.equal((await waitingInput.finish({held:['W'],seq:1})).status,409,'old delayed packet cannot cross a completed reset');
    assert.equal((await api('/api/step',{action:0})).status,200);
    fake.state={...fake.state,terminal:1,roundResult:1,winner:1,score:[20,2],fightResult:2,fightWinner:1};
    assert.equal((await api('/api/step',{action:0})).status,200);
    const state=(await api('/api/snapshot')).data;
    assert.equal(state.paused,true);assert.equal(state.session.orangeMatchWins,1);assert.equal(state.session.blueMatchWins,0);
    assert.equal((await api('/api/play',{paused:false})).status,409);
    assert.equal((await api('/api/step',{action:0})).status,409);
    assert.equal(maxActive,1,'no concurrent native request');
  }finally{
    if(releaseStep)releaseStep();if(service){const closed=once(service.server,'close');await service.close();await closed;}
    for(const name of fs.readdirSync(dir))fs.unlinkSync(path.join(dir,name));fs.rmdirSync(dir);
  }
});
