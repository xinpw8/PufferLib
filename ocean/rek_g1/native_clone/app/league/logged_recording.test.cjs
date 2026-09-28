'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),net=require('node:net');
const {spawn}=require('node:child_process'),{once}=require('node:events');
const {setTimeout:delay}=require('node:timers/promises'),crypto=require('node:crypto');
const {League}=require('./league.cjs');

test('actual logger preserves every input/request/state and reports reset/error with coalesced health',async()=>{
  const reservation=net.createServer();reservation.listen(0,'127.0.0.1');await once(reservation,'listening');
  const port=reservation.address().port;await new Promise(resolve=>reservation.close(resolve));
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-logged-recording-'));
  const wc=path.join(directory,'worker.json'),league=new League({file:path.join(directory,'league.json')});
  league.registerPolicy({id:'bot1',label:'Bot1',backend:'mujoco',kind:'scripted',configHash:'b'.repeat(64),scriptedVersion:'test'});
  fs.writeFileSync(wc,'{"round_seconds":120}');fs.writeFileSync(path.join(directory,'identity.json'),'{}');
  fs.writeFileSync(path.join(directory,'server.json'),JSON.stringify({port,leagueFile:league.file,
    backends:[{id:'mujoco',workerConfig:wc,configHash:'b'.repeat(64),env:{}}],
    initial:{backend:'mujoco',opponent:'bot1',humanSide:0,roundSeconds:120}}));
  const child=spawn(process.execPath,['--require',path.join(__dirname,'test_logged_worker.cjs'),
    path.join(__dirname,'..','launch_logged.cjs'),directory],{stdio:['ignore','pipe','pipe'],windowsHide:true});
  let stderr='';child.stderr.on('data',x=>stderr+=x);child.stdout.resume();
  const base=`http://127.0.0.1:${port}`;
  async function api(route,value){const r=await fetch(base+route,value===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});return {status:r.status,data:await r.json()};}
  try{
    let ready=false;
    for(let n=0;n<100;n++){
      assert.equal(child.exitCode,null,stderr);
      try{ready=(await api('/api/snapshot')).data.ok===true;}catch{}
      if(ready)break;await delay(20);
    }
    assert.equal(ready,true,stderr);
    const command={forward:.25,strafe:-.5,yaw:.125,moveIndex:-1,cancelAction:false};
    for(let n=1;n<=50;n++){
      assert.equal((await api('/api/input',{seq:n,held:['W']})).status,200);
      const reply=await api('/api/step',{steps:1,command,frame:n===50});
      assert.equal(reply.status,200);assert.equal(reply.data.state.tick,n);
    }
    const rows=fs.readFileSync(path.join(directory,'session.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
    assert.deepEqual(rows.map(x=>x.sequence),rows.map((_,i)=>i+1));
    assert.deepEqual(rows.filter(x=>x.kind==='human_input').map(x=>x.value.seq),Array.from({length:50},(_,i)=>i+1));
    const requests=rows.filter(x=>x.kind==='worker_request'&&x.op==='step');
    const replies=rows.filter(x=>x.kind==='worker_reply'&&x.op==='step');
    assert.equal(requests.length,50);assert.equal(replies.length,50);
    assert.ok(requests.every(x=>JSON.stringify(x.args.command)===JSON.stringify(command)));
    assert.deepEqual(replies.map(x=>x.result.state.tick),Array.from({length:50},(_,i)=>i+1));
    assert.deepEqual(replies.map(x=>x.result.state.qpos[0]),Array.from({length:50},(_,i)=>i+1));
    const frame=rows.filter(x=>x.kind==='rendered_frame'&&x.tick===50).at(-1);assert.ok(frame);
    const png=fs.readFileSync(path.join(directory,frame.path));
    assert.equal(crypto.createHash('sha256').update(png).digest('hex'),frame.sha256);
    const response=await fetch(base+'/frame.png');assert.equal(response.headers.get('x-rek-frame-sha256'),frame.sha256);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()),png);
    assert.equal((await api('/api/step',{steps:1,command:{...command,moveIndex:16}})).status,400);
    let health=JSON.parse(fs.readFileSync(path.join(directory,'recorder-health.json'),'utf8'));
    assert.equal(health.lastRequestError.message,'Injected request failure');
    assert.equal((await api('/api/reset',{})).status,200);
    health=JSON.parse(fs.readFileSync(path.join(directory,'recorder-health.json'),'utf8'));
    assert.equal(health.lastTick,0);assert.equal(health.steps,50);
    const finalRows=fs.readFileSync(path.join(directory,'session.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(finalRows.filter(x=>x.kind==='worker_error').length,1);
    assert.equal(finalRows.filter(x=>x.kind==='worker_request'&&x.op==='step').length,51);
    assert.equal(finalRows.filter(x=>x.kind==='worker_reply'&&x.op==='step').length,50);
  }finally{
    if(child.exitCode===null){const exited=once(child,'exit');child.kill('SIGTERM');await exited;}
    // Only the exact newly-created test directory is removed, with native Node APIs.
    assert.ok(directory.startsWith(path.join(os.tmpdir(),'rek-logged-recording-')));
    fs.rmSync(directory,{recursive:true,force:true});
  }
});
