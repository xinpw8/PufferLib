'use strict';
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const {League}=require('./league.cjs');
const {NativeWorker}=require('./worker.cjs');
const {HumanInput}=require('./input.cjs');
const {publicStanding}=require('./public_result.cjs');
const {HumanSession,createHumanConfig,HUMAN_ROUND_SECONDS,DEFAULT_HUMAN_ROUND_SECONDS}=require('./human_session.cjs');

async function serve(configPath,{Worker=NativeWorker,intermissionMs=3000}={}){
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  const port=config.port||18768,host='127.0.0.1';
  const league=new League({file:config.leagueFile});
  const backends=config.backends;
  for(const backend of backends)if(backend.trainingFile)
    backend.training=JSON.parse(fs.readFileSync(path.resolve(path.dirname(configPath),backend.trainingFile),'utf8'));
  let active=null,worker=null,state={ok:false,failure:'Select a backend'},png=null;
  let busy=false,switching=false,lastClient=0,frameDue=0;
  let humanConfig=null,paused=true,roundRestartAt=0;
  const session=new HumanSession();
  const input=new HumanInput();
  function options(backend){
    const definition=backends.find(b=>b.id===backend);
    if(!definition)return [];
    const saved=league.snapshot().policies;
    return league.opponentOptions({backend,configHash:definition.configHash}).map(p=>({
      ...p,label:saved[`${backend}/${p.id}`].label,
      ...(p.pairedStats.games?p.pairedStats:p.stats),ranked:p.rank!==null,
    }));
  }
  function catalog(){return {backends:backends.map(({id,label,available=true,error,runtimeNote,warning,training})=>
    ({id,label,available,error,runtimeNote,warning,training})),
    policies:backends.flatMap(b=>options(b.id)),active,
    humanRoundSeconds:HUMAN_ROUND_SECONDS,defaultHumanRoundSeconds:DEFAULT_HUMAN_ROUND_SECONDS};}
  async function select(value){
    if(switching)throw new Error('Backend switch already in progress');
    const backend=backends.find(b=>b.id===value.backend&&b.available!==false);
    if(!backend)throw new Error('Backend unavailable');
    const policy=league.snapshot().policies[`${backend.id}/${value.opponent}`];
    if(!policy||policy.configHash!==backend.configHash)throw new Error('Opponent/configuration mismatch');
    const humanSide=value.humanSide??(policy.kind==='trained'?1:0);
    if(humanSide!==0&&humanSide!==1)throw new Error('Human side must be 0 or 1');
    const roundSeconds=value.roundSeconds??DEFAULT_HUMAN_ROUND_SECONDS;
    if(!HUMAN_ROUND_SECONDS.includes(roundSeconds))throw new Error('Human round duration must be 20, 120 or 300 seconds');
    switching=true;input.reset();
    try{
      while(busy)await new Promise(resolve=>setTimeout(resolve,5));
      // This private viewer override never changes the training/league config.
      const nextConfig=createHumanConfig(backend.workerConfig,roundSeconds);
      if(worker)await worker.close();
      if(humanConfig)humanConfig.close();
      humanConfig=nextConfig;
      state={ok:false,failure:'Loading native evaluator'};png=null;
      worker=new Worker({executable:backend.executable,config:humanConfig.path,
        env:backend.env,logFile:backend.logFile});
      await worker.ready;
      if(policy.kind==='trained')await worker.request('policy',{
        side:1-humanSide,checkpoint:policy.checkpoint.path,sha256:policy.checkpoint.sha256,
        hiddenSize:policy.checkpoint.model.hiddenSize||256,layers:policy.checkpoint.model.layers||2,
        precision:policy.checkpoint.model.precision||'bf16',deterministic:policy.checkpoint.model.actionSelection!=='sampled',
        observationEncoding:policy.checkpoint.model.observationEncoding,
        recurrentResetTicks:policy.checkpoint.model.recurrentResetTicks||0,
        legacyFastHidden:policy.checkpoint.model.legacyFastHidden||0});
      else if(humanSide===1)await worker.request('policy',{side:0,scripted:true,checkpoint:''});
      state=(await worker.request('snapshot')).state;
      png=Buffer.from((await worker.request('frame')).png,'base64');
      active={backend:backend.id,opponent:policy.id,humanSide,roundSeconds,
        trainingRoundSeconds:humanConfig.trainingRoundSeconds};lastClient=Date.now();
      session.reset();paused=true;roundRestartAt=0;
      return catalog();
    }catch(error){state={ok:false,failure:error.message};throw error;}
    finally{switching=false;}
  }
  async function tick(){
    if(Date.now()-lastClient>2000){input.release();paused=true;return;}
    if(paused||Date.now()<roundRestartAt||busy||switching||!worker||!state.ok)return;
    busy=true;
    try{
      const result=await worker.request('step',{action:input.next(state,active.humanSide),humanSide:active.humanSide,steps:1});
      state=result.state;
      // Capture the terminal on the server, before the native automatic reset.
      // A 100 ms browser poll can miss a one-tick terminal completely.
      for(const completed of result.rounds||[])session.record(completed);
      if(state.terminal){
        session.record(state);input.release();roundRestartAt=Date.now()+intermissionMs;
      }
      state.moveDisposition=input.disposition;state.held=input.held;
      if(Date.now()>=frameDue){png=Buffer.from((await worker.request('frame')).png,'base64');frameDue=Date.now()+50;}
    }catch(error){state={...state,ok:false,failure:error.message};input.release();}
    finally{busy=false;}
  }
  const timer=setInterval(tick,20);
  async function body(req){
    let data='';for await(const chunk of req){data+=chunk;if(data.length>8192)throw new Error('Request too large');}
    return JSON.parse(data||'{}');
  }
  const server=http.createServer(async(req,res)=>{
    const headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'};
    const send=(code,type,value)=>{res.writeHead(code,{...headers,'Content-Type':type});res.end(value);};
    const json=(code,value)=>send(code,'application/json',JSON.stringify(value));
    try{
      if(req.headers.host!==`${host}:${port}`&&req.headers.host!==`localhost:${port}`)
        return json(403,{error:'Loopback Host required'});
      if(req.method==='POST'){
        if(req.headers.origin&&![`http://${host}:${port}`,`http://localhost:${port}`].includes(req.headers.origin))
          return json(403,{error:'Same-origin request required'});
        if(!String(req.headers['content-type']).startsWith('application/json'))
          return json(415,{error:'JSON required'});
      }
      const url=new URL(req.url,`http://${host}:${port}`);
      if(req.method==='GET'&&url.pathname==='/api/catalog')return json(200,catalog());
      if(req.method==='GET'&&url.pathname==='/api/standings')return json(200,
        backends.flatMap(b=>league.standings({backend:b.id,configHash:b.configHash})).map(publicStanding));
      if(req.method==='GET'&&url.pathname==='/api/state'){lastClient=Date.now();return json(200,{...state,active,switching,
        paused,intermissionSeconds:Math.max(0,(roundRestartAt-Date.now())/1000),session:session.snapshot()});}
      if(req.method==='GET'&&url.pathname==='/health')return json(state.ok?200:503,{ok:state.ok,failure:state.failure});
      if(req.method==='GET'&&url.pathname==='/frame.png')return png?send(200,'image/png',png):json(503,{error:'Frame unavailable'});
      if(req.method==='POST'&&url.pathname==='/api/input'){
        if(switching)return json(409,{error:'Switch in progress'});
        lastClient=Date.now();return json(200,{ok:true,accepted:input.update(await body(req))});
      }
      if(req.method==='POST'&&url.pathname==='/api/select')return json(200,await select(await body(req)));
      if(req.method==='POST'&&url.pathname==='/api/play'){
        if(switching||!worker)return json(409,{error:'Evaluator not ready'});
        const value=await body(req);
        if(typeof value.paused!=='boolean')throw new Error('Paused must be a boolean');
        paused=value.paused;if(paused)input.release();lastClient=Date.now();
        // A step already in flight may finish. Once this reply arrives the
        // paused snapshot is stable and no subsequent step can start.
        while(paused&&busy)await new Promise(resolve=>setTimeout(resolve,5));
        return json(200,{ok:true,paused});
      }
      if(req.method==='POST'&&url.pathname==='/api/reset'){
        if(switching||!worker)return json(409,{error:'Evaluator not ready'});
        switching=true;try{
          while(busy)await new Promise(resolve=>setTimeout(resolve,5));
          input.reset();paused=true;roundRestartAt=0;session.newRoundStream();
          state=(await worker.request('reset')).state;
          png=Buffer.from((await worker.request('frame')).png,'base64');return json(200,{ok:true});
        }finally{switching=false;}
      }
      if(req.method==='GET'){
        const routes={'/':'index.html','/app.js':'app.js','/style.css':'style.css'};
        const file=routes[url.pathname];if(file){const ext=path.extname(file);
          return send(200,{'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css'}[ext],
            fs.readFileSync(path.join(__dirname,'public',file)));}
      }
      return json(404,{error:'Not found'});
    }catch(error){return json(400,{error:error.message});}
  });
  server.listen(port,host,()=>process.stdout.write(JSON.stringify({ready:true,url:`http://${host}:${port}`})+'\n'));
  if(config.initial)select(config.initial).catch(error=>process.stderr.write(error.message+'\n'));
  async function close(){
    clearInterval(timer);switching=true;server.close();
    process.removeListener('SIGTERM',shutdown);process.removeListener('SIGINT',shutdown);
    while(busy)await new Promise(resolve=>setTimeout(resolve,5));
    if(worker)await worker.close();if(humanConfig)humanConfig.close();
  }
  function shutdown(){close().finally(()=>process.exit(0));}
  process.once('SIGTERM',shutdown);process.once('SIGINT',shutdown);
  return {server,close};
}
if(require.main===module)serve(process.argv[2]).catch(error=>{console.error(error);process.exitCode=1;});
module.exports={serve};
