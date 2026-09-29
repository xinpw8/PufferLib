'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),net=require('node:net'),crypto=require('node:crypto');
const {once}=require('node:events'),{setTimeout:delay}=require('node:timers/promises');
const {League}=require('./league.cjs'),{serve}=require('./server.cjs');
async function until(condition,label){const deadline=Date.now()+2000;while(!condition()&&Date.now()<deadline)await delay(5);assert(condition(),label);}
async function fixture(extra={}){
  const listener=net.createServer();listener.listen(0,'127.0.0.1');await once(listener,'listening');
  const port=listener.address().port;await new Promise(r=>listener.close(r));
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-decoupled-render-'));
  const league=new League({file:path.join(dir,'league.json')});
  league.registerPolicy({id:'bot1',label:'Bot1',backend:'mujoco',kind:'scripted',configHash:'d'.repeat(64),scriptedVersion:'test'});
  const cfg=path.join(dir,'server.json'),wc=path.join(dir,'worker.json');fs.writeFileSync(wc,'{"round_seconds":120}');
  fs.writeFileSync(cfg,JSON.stringify({port,leagueFile:league.file,backends:[{id:'mujoco',workerConfig:wc,configHash:'d'.repeat(64)}]}));
  const context={physics:null,renderer:null,stepGate:null};
  class Worker{
    constructor(){context.physics=this;this.ready=Promise.resolve();this.steps=0;this.reset();}
    reset(){this.state={ok:true,tick:0,phase:2,roundNumber:1,terminal:0,fightResult:0,fightWinner:-1,
      score:[0,0],wins:[0,0],roundResult:0,winner:-1,qpos:Array(72).fill(0)};}
    async request(op,args={}){
      assert.notEqual(op,'frame','physics worker must never render');
      if(op==='reset')this.reset();
      if(op==='step'){
        this.steps++;if(context.stepGate)await context.stepGate;
        this.state={...this.state,tick:this.state.tick+args.steps,qpos:Array(72).fill(this.state.tick+args.steps)};
      }
      return {state:{...this.state},rounds:[]};
    }
    async close(){}
  }
  class Renderer{
    constructor(options){assert.equal(options.renderOnly,true);context.renderer=this;this.ready=Promise.resolve({rendererOnly:true});this.jobs=[];this.active=0;this.maxActive=0;}
    async request(op,args){assert.equal(op,'frame');this.active++;this.maxActive=Math.max(this.maxActive,this.active);
      try{return await new Promise((resolve,reject)=>this.jobs.push({args,done:false,
        finish:()=>{this.jobs.find(j=>j.args===args).done=true;resolve({png:Buffer.from(`frame ${args.generation}/${args.snapshotTick}`).toString('base64'),generation:args.generation,snapshotTick:args.snapshotTick});},
        fail:()=>{this.jobs.find(j=>j.args===args).done=true;reject(Error('synthetic render failure'));}}));}
      finally{this.active--;}
    }
    async close(){for(const job of this.jobs)if(!job.done)job.fail();}
  }
  const service=await serve(cfg,{Worker,Renderer,intermissionMs:0,...extra});if(!service.server.listening)await once(service.server,'listening');
  const base=`http://127.0.0.1:${port}`;
  async function api(route,value){const r=await fetch(base+route,value===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});return {status:r.status,data:await r.json()};}
  assert.equal((await api('/api/select',{backend:'mujoco',opponent:'bot1',humanSide:0,roundSeconds:120})).status,200);
  await until(()=>context.renderer.jobs.length===1,'initial render requested');
  return {...context,context,api,base,close:async()=>{
    const closed=once(service.server,'close');await service.close();await closed;
    for(const name of fs.readdirSync(dir))fs.unlinkSync(path.join(dir,name));fs.rmdirSync(dir);
  }};
}
test('blocked renderer permits physics progress, pause and reset, and stale generation never publishes',async()=>{
  const f=await fixture();try{
    const old=f.renderer.jobs[0];await f.api('/api/play',{paused:false});
    await until(()=>f.physics.steps>=5,'steps continue while first render remains unresolved');
    assert.equal(f.renderer.jobs.length,1,'only one rendering job in flight');
    assert.equal((await f.api('/api/play',{paused:true})).status,200);
    const stopped=f.physics.steps,atPause=(await f.api('/api/snapshot')).data.pace.recent;
    assert.equal(atPause.activeIntervals,stopped-1,'only completed control intervals counted');
    await delay(50);assert.equal(f.physics.steps,stopped);
    assert.deepEqual((await f.api('/api/snapshot')).data.pace.recent,atPause,'paused wall time does not change recent pace');
    assert.equal((await f.api('/api/reset',{})).status,200,'reset does not wait for renderer');
    assert.equal((await f.api('/api/snapshot')).data.tick,0);
    assert.equal((await f.api('/api/snapshot')).data.pace.recent.realTimeRatio,null,'reset starts fresh recent measurement');
    old.finish();await until(()=>f.renderer.jobs.length===2,'reset frame follows old in-flight job');
    assert.equal((await fetch(f.base+'/frame.png')).status,503,'old generation discarded');
    const reset=f.renderer.jobs[1];assert.equal(reset.args.snapshotTick,0);assert(reset.args.generation>old.args.generation);
    reset.finish();await until(()=>f.renderer.active===0,'reset renderer completed');await delay(5);
    const response=await fetch(f.base+'/frame.png'),bytes=Buffer.from(await response.arrayBuffer());
    assert.equal(response.status,200);assert.equal(response.headers.get('x-rek-snapshot-tick'),'0');
    assert.equal(response.headers.get('x-rek-frame-generation'),String(reset.args.generation));
    assert.equal(response.headers.get('x-rek-frame-sha256'),crypto.createHash('sha256').update(bytes).digest('hex'));
    const state=(await f.api('/api/snapshot')).data;assert.equal(state.frame.tick,0);assert(state.frame.ageMs>=0);assert.equal(f.renderer.maxActive,1);
  }finally{await f.close();}
});
test('chase camera follows the human-controlled side across selections',async()=>{
  const f=await fixture();try{
    assert.equal(f.renderer.jobs[0].args.followSide,0);f.renderer.jobs[0].finish();
    assert.equal((await f.api('/api/select',{backend:'mujoco',opponent:'bot1',humanSide:1,roundSeconds:120})).status,200);
    await until(()=>f.context.renderer.jobs.length===1,'renderer recreated for the new side');
    const renderer=f.context.renderer;assert.equal(renderer.jobs[0].args.followSide,1);renderer.jobs[0].finish();
    await f.api('/api/play',{paused:false});await until(()=>renderer.jobs.length>=2,'play frames requested');
    await f.api('/api/play',{paused:true});assert(renderer.jobs.every(job=>job.args.followSide===1));
    for(const job of renderer.jobs)if(!job.done)job.finish();
  }finally{await f.close();}
});
test('reset waits for delayed physics step before changing render generation and tick-zero publication',async()=>{
  const f=await fixture();let release;
  try{
    f.context.stepGate=new Promise(r=>release=r);await f.api('/api/play',{paused:false});await until(()=>f.physics.steps===1,'step entered');
    let done=false;const reset=f.api('/api/reset',{}).then(x=>{done=true;return x;});await delay(30);assert.equal(done,false);
    release();f.context.stepGate=null;assert.equal((await reset).status,200);
    f.renderer.jobs[0].finish();await until(()=>f.renderer.jobs.length===2,'reset snapshot queued');
    assert.equal(f.renderer.jobs[1].args.snapshotTick,0);assert(f.renderer.jobs[1].args.qpos.every(x=>x===0));
    f.renderer.jobs[1].finish();await delay(5);const state=(await f.api('/api/snapshot')).data;
    assert.equal(state.frame.tick,0);assert.equal(state.tick,0);assert.equal(state.paused,true);
  }finally{if(release)release();await f.close();}
});
test('paused frame:true waits for exact frame, renderer failures remain isolated and next capture recovers',async()=>{
  const f=await fixture();try{
    let done=false;const step=f.api('/api/step',{action:0,steps:3,frame:true}).then(x=>{done=true;return x;});
    await until(()=>f.physics.steps===1,'paused step executes');await delay(30);assert.equal(done,false);
    f.renderer.jobs[0].fail();await until(()=>f.renderer.jobs.length===2,'exact frame retained behind failed render');
    assert.equal(f.renderer.jobs[1].args.snapshotTick,3);
    assert.equal((await f.api('/api/snapshot')).data.ok,true,'render failure is not physics failure');
    f.renderer.jobs[1].finish();const reply=await step;assert.equal(reply.status,200);assert.equal(reply.data.state.frame.tick,3);
    assert.equal(reply.data.state.renderFailure,null);assert.equal(reply.data.state.paused,true);
    assert.equal((await f.api('/api/play',{paused:false})).status,200);
    await until(()=>f.physics.steps>=4,'physics still runs after renderer error');await f.api('/api/play',{paused:true});
    assert.equal(f.renderer.maxActive,1);
  }finally{await f.close();}
});
test('closed renderer is latched without repeated requests while physics keeps advancing',async()=>{
  const f=await fixture();try{
    f.renderer.closed=true;f.renderer.jobs[0].fail();
    await f.api('/api/play',{paused:false});await until(()=>f.physics.steps>=5,'physics progresses after renderer closes');
    await f.api('/api/play',{paused:true});
    const state=(await f.api('/api/snapshot')).data;
    assert.equal(state.ok,true);assert.match(state.renderFailure,/render/i);assert.equal(f.renderer.jobs.length,1);
  }finally{await f.close();}
});

test('conditional frame GET transfers only new snapshot identities, including after reset',async t=>{
  const diagnostic={recorderWriteMs:{count:4,max:1.25}};
  const f=await fixture({diagnostics:()=>diagnostic});
  try{
    f.renderer.jobs[0].finish();await until(()=>f.renderer.active===0,'initial render done');await delay(5);
    const first=await fetch(f.base+'/frame.png'),initial=Buffer.from(await first.arrayBuffer()),etag=first.headers.get('etag');
    assert.equal(first.status,200);assert.match(etag,/^"rek-\d+-0-[0-9a-f]{64}"$/);
    const before=f.renderer.jobs.length;
    for(const validator of [etag,`W/${etag}`,`"unrelated", ${etag}`,'*']){
      const duplicate=await fetch(f.base+'/frame.png',{headers:{'If-None-Match':validator}});
      assert.equal(duplicate.status,304);assert.equal((await duplicate.arrayBuffer()).byteLength,0);
      assert.equal(duplicate.headers.get('etag'),etag);assert.equal(duplicate.headers.get('x-rek-snapshot-tick'),'0');
      assert.equal(duplicate.headers.get('x-rek-frame-sha256'),crypto.createHash('sha256').update(initial).digest('hex'));
    }
    t.diagnostic(JSON.stringify({proof:'HTTP image-body bytes only; headers excluded',firstImageBytes:initial.length,
      duplicateRequests:4,parentDuplicateImageBytes:initial.length*4,conditionalDuplicateImageBytes:0}));
    assert.equal(f.renderer.jobs.length,before,'HTTP image reads never request rendering');
    assert.deepEqual((await f.api('/api/snapshot')).data.diagnostics,diagnostic);
    const stepping=f.api('/api/step',{action:0,steps:1,frame:true});
    await until(()=>f.renderer.jobs.length===2,'new snapshot requested');f.renderer.jobs[1].finish();await stepping;
    const next=await fetch(f.base+'/frame.png',{headers:{'If-None-Match':etag}});
    assert.equal(next.status,200);assert.notEqual(next.headers.get('etag'),etag);assert.equal(next.headers.get('x-rek-snapshot-tick'),'1');
    await next.arrayBuffer();
    await f.api('/api/reset',{});await until(()=>f.renderer.jobs.length===3,'reset snapshot requested');f.renderer.jobs[2].finish();await delay(5);
    const reset=await fetch(f.base+'/frame.png',{headers:{'If-None-Match':etag}});
    assert.equal(reset.status,200);assert.notEqual(reset.headers.get('etag'),etag,'new generation cannot reuse prior tick-zero validator');
    assert.equal(reset.headers.get('x-rek-snapshot-tick'),'0');await reset.arrayBuffer();
  }finally{await f.close();}
});
