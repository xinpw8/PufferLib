'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const net=require('node:net');
const {once}=require('node:events');
const {setTimeout:delay}=require('node:timers/promises');
const {League}=require('./league.cjs');
const {serve}=require('./server.cjs');

test('human server pauses, extends rounds, retains results and leaves league fixtures unchanged',async()=>{
  const reservation=net.createServer();reservation.listen(0,'127.0.0.1');
  await once(reservation,'listening');const port=reservation.address().port;
  await new Promise(resolve=>reservation.close(resolve));
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-human-server-test-'));
  const workerPath=path.join(directory,'worker.json'),configPath=path.join(directory,'server.json');
  const league=new League({file:path.join(directory,'league.json')});
  league.registerPolicy({id:'scripted',backend:'semantic_cuda',kind:'scripted',label:'Scripted',
    configHash:'a'.repeat(64),scriptedVersion:'test'});
  fs.writeFileSync(workerPath,JSON.stringify({round_seconds:20,arenas:1,seed:73}));
  const training={status:'completed',steps:33554432,trainingSps:2508920.2201066464};
  fs.writeFileSync(path.join(directory,'training.json'),JSON.stringify(training));
  fs.writeFileSync(configPath,JSON.stringify({port,leagueFile:league.file,backends:[{
    id:'semantic_cuda',configHash:'a'.repeat(64),workerConfig:workerPath,trainingFile:'training.json'}]}));
  const leagueBefore=fs.readFileSync(league.file,'utf8');
  const workers=[];
  class Worker {
    constructor(spec){this.spec=spec;this.ready=Promise.resolve();this.steps=0;this.closed=false;
      this.duration=JSON.parse(fs.readFileSync(spec.config,'utf8')).round_seconds;
      this.reset();workers.push(this);}
    reset(){this.state={ok:true,tick:0,score:[0,0],terminal:0,winner:-1,roundResult:0,roundNumber:1,
      timeRemaining:this.duration,mask:Array(66).fill(1),raw:Array(446).fill(0)};}
    async request(op){
      if(op==='reset')this.reset();
      if(op==='frame')return {png:''};
      let rounds=[];
      if(op==='step'){
        this.steps++;
        if(this.state.terminal)this.state={...this.state,terminal:0,roundNumber:this.state.roundNumber+1,
          score:[0,0],winner:-1,roundResult:0};
        else {this.state={...this.state,terminal:1,score:[7,3],winner:0,roundResult:1};}
        this.state.tick++;this.state.timeRemaining=this.duration-this.state.tick*.02;
        if(this.state.terminal)rounds=[{...this.state}];
      }
      return {state:{...this.state},rounds};
    }
    async close(){this.closed=true;}
  }
  let instance;
  try{
    instance=await serve(configPath,{Worker,intermissionMs:100});
    if(!instance.server.listening)await once(instance.server,'listening');
    const origin=`http://127.0.0.1:${port}`;
    async function api(url,body){
      const response=await fetch(origin+url,body===undefined?{}:{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      return {status:response.status,value:await response.json()};
    }
    let response=await api('/api/select',{backend:'semantic_cuda',opponent:'scripted',humanSide:1});
    assert.equal(response.status,200);assert.equal(response.value.active.roundSeconds,300);
    assert.equal(response.value.active.trainingRoundSeconds,20);
    assert.deepEqual(response.value.backends[0].training,training);
    assert.equal(workers[0].duration,300);
    await delay(80);assert.equal(workers[0].steps,0,'loading does not consume the human clock');
    assert.equal((await api('/api/state')).value.paused,true);
    assert.equal((await api('/api/play',{paused:false})).status,200);
    let state;
    for(let n=0;n<30;n++){
      state=(await api('/api/state')).value;
      if(state.terminal)break;await delay(10);
    }
    assert.equal(state.terminal,1);assert.equal(state.session.completedRounds,1);
    assert.equal(state.session.bluePoints,7);assert.equal(state.session.orangePoints,3);
    assert.equal(state.session.blueWins,1);assert.equal(state.session.orangeWins,0);
    assert.ok(state.intermissionSeconds>0);
    await api('/api/play',{paused:true});const pausedStep=workers[0].steps;
    await delay(120);assert.equal(workers[0].steps,pausedStep);
    await api('/api/reset',{});state=(await api('/api/state')).value;
    assert.equal(state.paused,true);assert.equal(state.tick,0);assert.equal(state.session.bluePoints,7);
    await api('/api/play',{paused:false});
    for(let n=0;n<30;n++){
      state=(await api('/api/state')).value;
      if(state.session.completedRounds===2)break;await delay(10);
    }
    assert.equal(state.session.completedRounds,2,'same native tick after reset is a new round stream');
    await api('/api/play',{paused:true});
    response=await api('/api/select',{backend:'semantic_cuda',opponent:'scripted',roundSeconds:19});
    assert.equal(response.status,400);assert.equal(workers.length,1);
    const oldConfig=workers[0].spec.config;
    response=await api('/api/select',{backend:'semantic_cuda',opponent:'scripted',roundSeconds:120});
    assert.equal(response.status,200);assert.equal(workers[1].duration,120);
    assert.equal(workers[0].closed,true);assert.equal(fs.existsSync(oldConfig),false);
    state=(await api('/api/state')).value;assert.equal(state.session.completedRounds,0);
    assert.equal(JSON.parse(fs.readFileSync(workerPath,'utf8')).round_seconds,20);
    assert.equal(fs.readFileSync(league.file,'utf8'),leagueBefore,'human play never changes rankings');
  }finally{
    if(instance){const closed=once(instance.server,'close');await instance.close();await closed;}
    for(const name of fs.readdirSync(directory))fs.unlinkSync(path.join(directory,name));
    fs.rmdirSync(directory);
  }
  assert.ok(workers.every(worker=>worker.closed&&!fs.existsSync(worker.spec.config)));
});
