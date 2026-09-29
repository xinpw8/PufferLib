'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const WebSocket=require('../vendor/ws');
const auth=require('../auth.cjs');
const {Lobby}=require('../lobby.cjs');
const {encodeTick,decodeTick,SIZE}=require('../protocol.cjs');
const {serve,hostAllowed}=require('../server.cjs');

const qpos=Array.from({length:72},(_,i)=>i/100);
const nativeState=(tick=0,extra={})=>({ok:true,tick,phase:2,timeRemaining:120-tick*.02,roundNumber:1,terminal:0,winner:-1,
  fightResult:0,fightWinner:-1,score:[3,1],falls:[0,1],wins:[0,0],qpos,...extra});

test('passwords hash with scrypt and verify only the right password',()=>{
  const stored=auth.hashPassword('correct horse');
  assert.ok(auth.verifyPassword('correct horse',stored));
  assert.ok(!auth.verifyPassword('wrong horse',stored));
  assert.throws(()=>auth.hashPassword('short'));
});
test('tokens are signed, expire and carry the cleaned name',()=>{
  const secret=Buffer.alloc(32,7),token=auth.issueToken(secret,'Dan',1000);
  assert.equal(auth.verifyToken(secret,token,2000),'Dan');
  assert.equal(auth.verifyToken(Buffer.alloc(32,8),token,2000),null);
  assert.equal(auth.verifyToken(secret,token,1000+8*86400000),null);
  assert.equal(auth.verifyToken(secret,token.replace(/.$/,c=>c==='A'?'B':'A'),2000),null);
  assert.equal(auth.cleanName('<script>x</script>'),'scriptxscript');assert.equal(auth.cleanName(''),'Player');
});
test('login limiter blocks after repeated failures',()=>{
  const l=new auth.LoginLimiter({limit:3,windowMs:1000});
  for(let i=0;i<3;i++)l.fail('ip',i);
  assert.ok(l.blocked('ip',10));assert.ok(!l.blocked('ip',2000));
});
test('lobby: solo idle, Bot 1 on request, second player replaces Bot 1, seats compact',()=>{
  const lobby=new Lobby(),a={id:1},b={id:2},c={id:3};
  assert.equal(lobby.join(a),0);assert.equal(lobby.mode(),'idle');
  assert.ok(lobby.requestBot(a));assert.equal(lobby.mode(),'bot');
  assert.equal(lobby.join(b),1);assert.equal(lobby.mode(),'pvp');
  assert.equal(lobby.join(c),-1);
  lobby.leave(a);assert.deepEqual(lobby.seats,[b,c]);assert.equal(lobby.mode(),'pvp');
  lobby.leave(c);assert.deepEqual(lobby.seats,[b,null]);assert.equal(lobby.mode(),'idle');
  assert.ok(!lobby.requestBot(c));
});
test('tick messages round-trip in 328 bytes',()=>{
  const buffer=encodeTick({mode:'pvp',state:nativeState(7),events:[{side:1,attempted:1,accepted:1,reason:0}],serverTime:1234.5,holdSeconds:0});
  assert.equal(buffer.length,SIZE);assert.equal(SIZE,328);
  const t=decodeTick(buffer);
  assert.equal(t.mode,'pvp');assert.equal(t.tick,7);assert.equal(t.serverTime,1234.5);
  assert.deepEqual(t.points,[3,1]);assert.equal(t.commands[1],3);assert.ok(Math.abs(t.qpos[71]-.71)<1e-6);
});
test('host allowlist accepts exact hosts and dotted suffixes only',()=>{
  assert.ok(hostAllowed('rek.clipfrac.com',['rek.clipfrac.com']));
  assert.ok(hostAllowed('a-b.trycloudflare.com',['.trycloudflare.com']));
  assert.ok(!hostAllowed('trycloudflare.com',['.trycloudflare.com']));
  assert.ok(!hostAllowed('evil.com',['rek.clipfrac.com']));
});

class FakeWorker{
  static instances=[];
  constructor(){this.requests=[];this.tick=0;this.closed=false;this.ready=Promise.resolve();FakeWorker.instances.push(this);}
  async request(op,args={}){
    this.requests.push({op,args});
    if(op==='reset')this.tick=0;
    if(op==='step')this.tick++;
    return {ok:true,state:nativeState(this.tick),commandEvents:[],rounds:[]};
  }
  async close(){this.closed=true;}
}

async function withServer(fn){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-online-test-'));
  const scene=path.join(dir,'scene');fs.mkdirSync(scene);
  fs.writeFileSync(path.join(scene,'scene.json'),JSON.stringify({textures:[]}));fs.writeFileSync(path.join(scene,'scene.bin'),Buffer.alloc(8));
  fs.writeFileSync(path.join(dir,'worker.json'),JSON.stringify({backend:'mujoco',round_seconds:120}));
  const port=20000+Math.floor(Math.random()*20000);
  const config={port,secret:'ab'.repeat(32),passwordHash:auth.hashPassword('letmein123'),allowedHosts:['rek.example'],
    binary:'/bin/false',workerConfig:path.join(dir,'worker.json'),env:{},roundSeconds:120,sceneDir:scene,logDir:path.join(dir,'logs')};
  fs.writeFileSync(path.join(dir,'config.json'),JSON.stringify(config));
  FakeWorker.instances=[];
  const app=serve(path.join(dir,'config.json'),{Worker:FakeWorker});
  await new Promise(r=>app.server.once('listening',r));
  try{await fn({port,app,base:`http://127.0.0.1:${port}`});}
  finally{await app.close();process.removeAllListeners('SIGTERM');process.removeAllListeners('SIGINT');fs.rmSync(dir,{recursive:true});}
}
async function login(base,password,name){
  const r=await fetch(base+'/login',{method:'POST',redirect:'manual',headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:new URLSearchParams({password,name})});
  return {status:r.status,location:r.headers.get('location'),cookie:(r.headers.get('set-cookie')||'').split(';')[0]};
}
function player(port,cookie){
  const ws=new WebSocket(`ws://127.0.0.1:${port}/ws`,{headers:{Cookie:cookie,Origin:`http://127.0.0.1:${port}`}});
  const messages=[],ticks=[];
  ws.on('message',(data,binary)=>{if(binary)ticks.push(decodeTick(data));else messages.push(JSON.parse(data));});
  const until=async(predicate,ms=3000)=>{const end=Date.now()+ms;while(Date.now()<end){const hit=predicate();if(hit)return hit;await new Promise(r=>setTimeout(r,10));}throw Error('Timed out');};
  return {ws,messages,ticks,until,open:new Promise((resolve,reject)=>{ws.once('open',resolve);ws.once('error',reject);}),
    lobby:()=>messages.filter(m=>m.t==='lobby').at(-1)};
}

test('pages and sockets require the password; wrong passwords are refused',async()=>{
  await withServer(async({port,base})=>{
    assert.equal((await fetch(base+'/',{redirect:'manual'})).status,303);
    assert.equal((await fetch(base+'/scene/scene.json')).status,401);
    const foreignHost=await new Promise((resolve,reject)=>require('node:http').get({port,path:'/',headers:{Host:'evil.com'}},r=>{r.resume();resolve(r.statusCode);}).on('error',reject));
    assert.equal(foreignHost,403);
    const bad=await login(base,'nope','X');assert.equal(bad.location,'/login?failed=1');assert.equal(bad.cookie,'');
    const unauth=new WebSocket(`ws://127.0.0.1:${port}/ws`,{headers:{Origin:`http://127.0.0.1:${port}`}});
    await assert.rejects(new Promise((resolve,reject)=>{unauth.once('open',resolve);unauth.once('error',reject);}));
    const good=await login(base,'letmein123','Dan');assert.equal(good.location,'/');assert.match(good.cookie,/^rek_auth=/);
    assert.equal((await fetch(base+'/scene/scene.json',{headers:{Cookie:good.cookie}})).status,200);
    const foreign=new WebSocket(`ws://127.0.0.1:${port}/ws`,{headers:{Cookie:good.cookie,Origin:'https://evil.com'}});
    await assert.rejects(new Promise((resolve,reject)=>{foreign.once('open',resolve);foreign.once('error',reject);}));
  });
});

test('one player idles, fights Bot 1 on request; a second player starts PvP with both commands',async()=>{
  await withServer(async({port,base,app})=>{
    const a=player(port,(await login(base,'letmein123','Ann')).cookie);await a.open;
    await a.until(()=>a.lobby()?.seat===0&&a.lobby().canFightBot);
    assert.equal(app.mode,'idle');
    a.ws.send(JSON.stringify({t:'bot'}));
    await a.until(()=>a.lobby()?.mode==='bot');
    await a.until(()=>a.ticks.some(t=>t.mode==='bot'&&t.tick>2),6000);
    const worker=FakeWorker.instances[0];
    a.ws.send(JSON.stringify({t:'input',seq:1,held:['W'],move:21,cancelAction:false}));
    await a.until(()=>worker.requests.some(r=>r.op==='step'&&r.args.command?.forward===1));
    const aiStep=worker.requests.find(r=>r.op==='step'&&r.args.command?.moveIndex===1);
    assert.ok(aiStep,'left jab (category 21) reaches native move 1');assert.equal(aiStep.args.humanSide,0);

    const b=player(port,(await login(base,'letmein123','Bo')).cookie);await b.open;
    await b.until(()=>b.lobby()?.mode==='pvp'&&b.lobby().seat===1);
    await a.until(()=>a.lobby()?.mode==='pvp'&&a.lobby().players[1]?.name==='Bo');
    const resetsBefore=worker.requests.filter(r=>r.op==='reset').length;
    await a.until(()=>worker.requests.some(r=>r.op==='step'&&Array.isArray(r.args.commands)),6000);
    assert.ok(resetsBefore>=2,'new match resets the simulation');
    b.ws.send(JSON.stringify({t:'input',seq:5,held:['D'],move:null,cancelAction:false}));
    a.ws.send(JSON.stringify({t:'input',seq:6,held:['S'],move:null,cancelAction:false}));
    const pair=await a.until(()=>worker.requests.find(r=>r.op==='step'&&r.args.commands?.[1].strafe===-1&&r.args.commands?.[0].forward===-1));
    assert.equal(pair.args.command,undefined);

    b.ws.close();
    await a.until(()=>a.lobby()?.mode==='idle'&&a.lobby().canFightBot);
    a.ws.close();
  });
});

test('a spectator takes the seat that a leaving player frees',async()=>{
  await withServer(async({port,base})=>{
    const cookie=(await login(base,'letmein123','P')).cookie;
    const [a,b,c]=[player(port,cookie),player(port,cookie),player(port,cookie)];
    await Promise.all([a.open,b.open,c.open]);
    await c.until(()=>c.lobby()?.seat===-1&&c.lobby().mode==='pvp');
    a.ws.close();
    await c.until(()=>c.lobby()?.seat===1&&c.lobby().mode==='pvp');
    await b.until(()=>b.lobby()?.seat===0);
    b.ws.close();c.ws.close();
  });
});
