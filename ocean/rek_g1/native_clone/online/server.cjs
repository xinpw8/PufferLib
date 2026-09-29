'use strict';
// Online REK: one authoritative native simulation, streamed to browsers that
// render it locally. Clients receive a 328-byte state per 20 ms tick and send
// key changes over the same WebSocket; nothing is rendered on the server.
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const zlib=require('node:zlib');
const crypto=require('node:crypto');
const {performance}=require('node:perf_hooks');
const {WebSocketServer}=require('./vendor/ws');
const {NativeWorker}=require('../app/league/worker.cjs');
const {HumanInput}=require('../app/league/input.cjs');
const {startPacedLoop}=require('../app/league/paced_loop.cjs');
const {createHumanConfig}=require('../app/league/human_session.cjs');
const auth=require('./auth.cjs');
const {Lobby}=require('./lobby.cjs');
const {encodeTick}=require('./protocol.cjs');

const COUNTDOWN_MS=3000,ROUND_BREAK_MS=3000,MATCH_END_MS=8000;
const MAX_BUFFERED=256*1024,LOG_CAP_BYTES=20*1024*1024;

function cappedLog(file){
  let written=fs.existsSync(file)?fs.statSync(file).size:0;
  return (event,fields={})=>{
    if(written>=LOG_CAP_BYTES)return;
    const line=JSON.stringify({utc:new Date().toISOString(),event,...fields})+'\n';
    written+=Buffer.byteLength(line);
    try{fs.appendFileSync(file,line,{mode:0o600});}catch(error){process.stderr.write(`log write failed: ${error.message}\n`);}
  };
}

function loadStatic(config){
  const here=path.join(__dirname,'public'),files=new Map();
  const add=(route,file,type,cache)=>{
    const body=fs.readFileSync(file),etag='"'+crypto.createHash('sha256').update(body).digest('base64url').slice(0,22)+'"';
    const gz=/png$/.test(type)?null:zlib.gzipSync(body,{level:9});
    files.set(route,{body,gz,type,etag,cache});
  };
  add('/',path.join(here,'index.html'),'text/html; charset=utf-8','no-cache');
  add('/login',path.join(here,'login.html'),'text/html; charset=utf-8','no-cache');
  add('/style.css',path.join(here,'style.css'),'text/css; charset=utf-8','no-cache');
  add('/client.js',path.join(here,'client.js'),'text/javascript; charset=utf-8','no-cache');
  add('/scene.js',path.join(here,'scene.js'),'text/javascript; charset=utf-8','no-cache');
  add('/controls.js',path.join(__dirname,'../app/league/public/controls.js'),'text/javascript; charset=utf-8','no-cache');
  add('/vendor/three.module.min.js',path.join(here,'vendor/three.module.min.js'),'text/javascript; charset=utf-8','private, max-age=86400');
  const scene=JSON.parse(fs.readFileSync(path.join(config.sceneDir,'scene.json'),'utf8'));
  add('/scene/scene.json',path.join(config.sceneDir,'scene.json'),'application/json','private, max-age=86400');
  add('/scene/scene.bin',path.join(config.sceneDir,'scene.bin'),'application/octet-stream','private, max-age=86400');
  for(const t of scene.textures)add('/scene/'+t.file,path.join(config.sceneDir,t.file),'image/png','private, max-age=86400');
  return files;
}

function hostAllowed(host,allowed){
  if(typeof host!=='string')return false;
  return allowed.some(entry=>entry.startsWith('.')?host.endsWith(entry)&&host.length>entry.length:host===entry);
}

function serve(configPath,{Worker=NativeWorker,now=()=>Date.now()}={}){
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  const port=config.port,host=config.host||'127.0.0.1';
  const allowedHosts=[...(config.allowedHosts||[]),`127.0.0.1:${port}`,`localhost:${port}`];
  const secret=Buffer.from(config.secret,'hex');
  if(secret.length<32)throw Error('Config secret must be at least 32 bytes of hex');
  fs.mkdirSync(config.logDir,{recursive:true,mode:0o700});
  const log=cappedLog(path.join(config.logDir,'online.log'));
  const files=loadStatic(config),limiter=new auth.LoginLimiter();
  const lobby=new Lobby(),inputs=[new HumanInput(),new HumanInput()],clients=new Set();
  let worker=null,humanConfig=null,state=null,mode='idle',hold={until:0,reason:''};
  let busy=false,generation=0,seatKey='',transition=Promise.resolve(),lastEvents=[],failure=null,nextId=1,closing=false;
  const stats={steps:0,stepMs:[],lateHolds:0,matches:0,startedAt:now()};

  const holdSeconds=()=>Math.max(0,(hold.until-now())/1000);
  function tickPacket(events=[]){
    return state?.qpos?encodeTick({mode,state,events,serverTime:performance.now(),holdSeconds:holdSeconds()}):null;
  }
  function sendTick(events){
    const packet=tickPacket(events);if(!packet)return;
    for(const c of clients)if(c.ws.readyState===1&&c.ws.bufferedAmount<MAX_BUFFERED)c.ws.send(packet);
  }
  function lobbyMessage(client){
    const seat=lobby.seatOf(client);
    return JSON.stringify({t:'lobby',mode,seat,players:lobby.seats.map(c=>c?{name:c.name,you:c===client}:null),
      spectators:lobby.spectators.size,hold:{reason:hold.reason,seconds:holdSeconds()},
      canFightBot:mode==='idle'&&lobby.players===1&&seat===0,failure});
  }
  function sendLobby(){for(const c of clients)if(c.ws.readyState===1)c.ws.send(lobbyMessage(c));}

  async function ensureWorker(){
    if(worker&&!worker.closed)return;
    if(humanConfig)humanConfig.close();
    humanConfig=createHumanConfig(config.workerConfig,config.roundSeconds||120);
    worker=new Worker({executable:config.binary,config:humanConfig.path,env:config.env,
      logFile:path.join(config.logDir,'worker.stderr.log')});
    await worker.ready;
  }
  async function stopStepping(){generation++;timer.pause();while(busy)await new Promise(r=>setTimeout(r,2));}
  // Serialized: applies the lobby's mode, resetting the simulation for every new match.
  async function applyMode(force=false){
    const next=lobby.mode(),key=lobby.seats.map(c=>c?.id??'-').join(',');
    if(!force&&next===mode&&key===seatKey)return;
    await stopStepping();
    seatKey=key;inputs.forEach(i=>i.reset());
    try{
      await ensureWorker();
      state=(await worker.request('reset')).state;failure=null;lastEvents=[];
    }catch(error){
      failure='Simulator unavailable: '+error.message;log('worker_failure',{error:error.message});
      if(worker){await worker.close().catch(()=>{});worker=null;}
      mode='idle';hold={until:0,reason:''};sendLobby();return;
    }
    mode=next;
    hold=mode==='idle'?{until:0,reason:''}:{until:now()+COUNTDOWN_MS,reason:'countdown'};
    log('mode',{mode,players:lobby.seats.map(c=>c?.name??null)});
    sendTick([]);sendLobby();
    if(mode!=='idle')timer.resume();
  }
  function queue(force=false){
    if(closing)return transition;
    transition=transition.then(()=>applyMode(force)).catch(error=>log('transition_error',{error:error.message}));
    return transition;
  }

  let lastHoldBroadcast=0;
  async function tick(){
    if(mode==='idle'||busy||!state)return;
    const t=now();
    if(t<hold.until){if(t-lastHoldBroadcast>=100){lastHoldBroadcast=t;sendTick([]);}return;}
    if(hold.reason==='matchEnd'){
      hold={until:0,reason:''};timer.pause();
      if(mode==='bot')lobby.stopBot();
      queue(true);return;
    }
    if(hold.reason){hold={until:0,reason:''};sendLobby();}
    busy=true;const g=generation;
    try{
      const args=mode==='pvp'?{commands:[inputs[0].next(state,0),inputs[1].next(state,1)],steps:1}
        :{command:inputs[0].next(state,0),humanSide:0,steps:1};
      const started=performance.now();
      const result=await worker.request('step',args);
      if(g!==generation)return;
      const ms=performance.now()-started;stats.steps++;stats.stepMs.push(ms);if(stats.stepMs.length>3000)stats.stepMs.shift();
      state=result.state;lastEvents=result.commandEvents||[];
      if(state.fightResult>0){
        hold={until:now()+MATCH_END_MS,reason:'matchEnd'};inputs.forEach(i=>i.release());stats.matches++;
        log('match_end',{mode,winner:state.fightWinner,result:state.fightResult,points:state.score,players:lobby.seats.map(c=>c?.name??null)});
        sendLobby();
      }else if(state.terminal){hold={until:now()+ROUND_BREAK_MS,reason:'roundBreak'};inputs.forEach(i=>i.release());sendLobby();}
      sendTick(lastEvents);
    }catch(error){
      if(g!==generation)return;
      failure='Simulator stopped: '+error.message;log('step_failure',{error:error.message});
      timer.pause();generation++;
      if(worker){worker.close().catch(()=>{});worker=null;}
      mode='idle';lobby.stopBot();seatKey='';sendLobby();
    }finally{busy=false;}
  }
  const timer=startPacedLoop(tick,{periodMs:20,startPaused:true});

  function authed(req){return auth.verifyToken(secret,auth.readCookie(req.headers.cookie));}
  function clientAddress(req){return String(req.headers['cf-connecting-ip']||req.socket.remoteAddress||'');}
  function secure(req){return req.headers['x-forwarded-proto']==='https'||/"scheme":"https"/.test(String(req.headers['cf-visitor']||''));}
  function securityHeaders(req){
    const h=req.headers.host;
    return {'X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin','X-Frame-Options':'DENY',
      'Content-Security-Policy':`default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self' wss://${h} ws://${h}; frame-ancestors 'none'; base-uri 'none'; form-action 'self'`};
  }
  async function readForm(req){
    let data='';for await(const chunk of req){data+=chunk;if(data.length>2048)throw Error('Request too large');}
    return new URLSearchParams(data);
  }
  const server=http.createServer(async(req,res)=>{
    const headers=securityHeaders(req);
    const send=(code,extra,body)=>{res.writeHead(code,{...headers,...extra});res.end(body);};
    try{
      if(!hostAllowed(req.headers.host,allowedHosts))return send(403,{'Content-Type':'text/plain'},'Forbidden');
      const url=new URL(req.url,'http://local');
      if(req.method==='GET'&&url.pathname==='/health')return send(200,{'Content-Type':'text/plain','Cache-Control':'no-store'},'ok');
      if(req.method==='POST'&&url.pathname==='/login'){
        const origin=req.headers.origin;
        if(origin&&origin!==`https://${req.headers.host}`&&origin!==`http://${req.headers.host}`)return send(403,{'Content-Type':'text/plain'},'Forbidden');
        const address=clientAddress(req);
        if(limiter.blocked(address))return send(429,{'Content-Type':'text/plain','Retry-After':'900'},'Too many attempts. Try again later.');
        const form=await readForm(req);
        if(!auth.verifyPassword(form.get('password')||'',config.passwordHash)){
          limiter.fail(address);log('login_failed',{address});
          return send(303,{Location:'/login?failed=1','Cache-Control':'no-store'},'');
        }
        limiter.clear(address);const name=auth.cleanName(form.get('name'));
        const cookie=`${auth.COOKIE}=${encodeURIComponent(auth.issueToken(secret,name))}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${auth.TOKEN_DAYS*86400}${secure(req)?'; Secure':''}`;
        log('login',{address,name});
        return send(303,{Location:'/','Set-Cookie':cookie,'Cache-Control':'no-store'},'');
      }
      if(req.method==='GET'&&url.pathname==='/logout')
        return send(303,{Location:'/login','Set-Cookie':`${auth.COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`,'Cache-Control':'no-store'},'');
      if(req.method!=='GET'&&req.method!=='HEAD')return send(405,{'Content-Type':'text/plain'},'Method not allowed');
      const publicRoute=url.pathname==='/login'||url.pathname==='/style.css';
      if(!publicRoute&&!authed(req)){
        if(url.pathname==='/')return send(303,{Location:'/login','Cache-Control':'no-store'},'');
        return send(401,{'Content-Type':'text/plain'},'Login required');
      }
      if(url.pathname==='/api/stats'){
        const sorted=[...stats.stepMs].sort((a,b)=>a-b),q=p=>sorted.length?sorted[Math.min(sorted.length-1,Math.floor(p*sorted.length))]:null;
        return send(200,{'Content-Type':'application/json','Cache-Control':'no-store'},JSON.stringify({mode,players:lobby.players,
          spectators:lobby.spectators.size,steps:stats.steps,matches:stats.matches,scheduler:timer.snapshot(),
          stepMs:{p50:q(.5),p95:q(.95),p99:q(.99),max:sorted.at(-1)??null,window:sorted.length},failure}));
      }
      const file=files.get(url.pathname);
      if(!file)return send(404,{'Content-Type':'text/plain'},'Not found');
      if(url.pathname==='/login'&&url.searchParams.has('failed'))
        return send(200,{'Content-Type':file.type,'Cache-Control':'no-store'},
          file.body.toString().replace('<!--MESSAGE-->','<p class="error">Wrong password.</p>'));
      if(String(req.headers['if-none-match']||'')===file.etag)return send(304,{ETag:file.etag,'Cache-Control':file.cache},'');
      const gzip=file.gz&&/\bgzip\b/.test(String(req.headers['accept-encoding']||''));
      const body=gzip?file.gz:file.body;
      return send(200,{'Content-Type':file.type,'Cache-Control':file.cache,ETag:file.etag,Vary:'Accept-Encoding',
        'Content-Length':body.length,...(gzip?{'Content-Encoding':'gzip'}:{})},req.method==='HEAD'?undefined:body);
    }catch(error){return send(400,{'Content-Type':'text/plain'},'Bad request');}
  });

  const wss=new WebSocketServer({noServer:true,maxPayload:4096,perMessageDeflate:false});
  server.on('upgrade',(req,socket,head)=>{
    const reject=code=>{socket.write(`HTTP/1.1 ${code} Rejected\r\nConnection: close\r\n\r\n`);socket.destroy();};
    if(!hostAllowed(req.headers.host,allowedHosts))return reject(403);
    const origin=req.headers.origin;
    if(origin!==`https://${req.headers.host}`&&origin!==`http://${req.headers.host}`)return reject(403);
    const name=authed(req);if(!name)return reject(401);
    if(new URL(req.url,'http://local').pathname!=='/ws')return reject(404);
    wss.handleUpgrade(req,socket,head,ws=>connected(ws,name,clientAddress(req)));
  });
  function connected(ws,name,address){
    const client={ws,name,id:nextId++,alive:true};clients.add(client);
    ws.on('pong',()=>{client.alive=true;});
    const seat=lobby.join(client);log('join',{id:client.id,name,address,seat});
    ws.send(JSON.stringify({t:'hello',id:client.id,name,roundSeconds:config.roundSeconds||120}));
    const packet=tickPacket();if(packet)ws.send(packet);
    queue();sendLobby();
    ws.on('message',(data,isBinary)=>{
      if(isBinary)return;
      let m;try{m=JSON.parse(data);}catch{return;}
      const seat=lobby.seatOf(client);
      if(m.t==='ping')return ws.send(JSON.stringify({t:'pong',c:m.c,s:performance.now()}));
      if(m.t==='input'){
        if(seat<0||mode==='idle'||(mode==='bot'&&seat!==0))return;
        try{inputs[seat].update(m);}catch(error){ws.send(JSON.stringify({t:'error',error:error.message}));}
        return;
      }
      if(m.t==='release'){if(seat>=0)inputs[seat].release();return;}
      if(m.t==='bot'){if(lobby.requestBot(client))queue();return;}
    });
    ws.on('close',()=>{
      clients.delete(client);if(closing)return;const seat=lobby.seatOf(client);if(seat>=0)inputs[seat].release();
      lobby.leave(client);log('leave',{id:client.id,name});queue();sendLobby();
    });
  }
  const heartbeat=setInterval(()=>{for(const c of clients){if(!c.alive){c.ws.terminate();continue;}c.alive=false;c.ws.ping();}},15000);

  server.listen(port,host,()=>{
    process.stdout.write(JSON.stringify({ready:true,url:`http://${host}:${port}`})+'\n');
    // Start the native worker early so the first player does not wait for CUDA setup.
    queue(true);
  });
  async function close(){
    closing=true;clearInterval(heartbeat);timer.stop();for(const c of clients)c.ws.terminate();wss.close();server.close();
    while(busy)await new Promise(r=>setTimeout(r,5));
    if(worker)await worker.close();if(humanConfig)humanConfig.close();
  }
  function shutdown(){close().finally(()=>process.exit(0));}
  process.once('SIGTERM',shutdown);process.once('SIGINT',shutdown);
  return {server,close,lobby,stats,get mode(){return mode;},get state(){return state;}};
}
if(require.main===module)serve(process.argv[2]);
module.exports={serve,hostAllowed};
