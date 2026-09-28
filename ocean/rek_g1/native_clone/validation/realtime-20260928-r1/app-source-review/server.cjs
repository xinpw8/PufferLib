'use strict';
const http=require('node:http');
const {performance}=require('node:perf_hooks');
const {startPacedLoop}=require('./paced_loop.cjs');
const {LatestFrameQueue}=require('./render_queue.cjs');
const fs=require('node:fs');
const path=require('node:path');
const {framePayload}=require('./frame_payload.cjs');
const {League}=require('./league.cjs');
const {NativeWorker}=require('./worker.cjs');
const {HumanInput,validateCommand}=require('./input.cjs');
const {publicStanding}=require('./public_result.cjs');
const {HumanSession,createHumanConfig,HUMAN_ROUND_SECONDS,DEFAULT_HUMAN_ROUND_SECONDS}=require('./human_session.cjs');

async function serve(configPath,{Worker=NativeWorker,Renderer=NativeWorker,intermissionMs=3000}={}){
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  const port=config.port||18768,host='127.0.0.1';
  const league=new League({file:config.leagueFile});
  const backends=config.backends;
  for(const backend of backends)if(backend.trainingFile)
    backend.training=JSON.parse(fs.readFileSync(path.resolve(path.dirname(configPath),backend.trainingFile),'utf8'));
  let active=null,worker=null,renderer=null,state={ok:false,failure:'Select a backend'},png=null;
  let busy=false,switching=false,lastClient=0,workerGeneration=0,frameGeneration=0;
  let frameState=null,renderFailure=null;
  let humanConfig=null,paused=true,roundRestartAt=0;
  const session=new HumanSession();
  const input=new HumanInput();
  let pace={steps:0,activeIntervals:0,activeWallMs:0,lastStepMs:null,lastFrameMs:null},lastStepStart=null;
  const runtimeState=()=>({...state,active,switching,paused,
    frame:frameState?{...frameState,ageMs:Math.max(0,Date.now()-frameState.publishedAtMs)}:null,renderFailure,
    intermissionSeconds:Math.max(0,(roundRestartAt-Date.now())/1000),session:session.snapshot(),
    pace:{...pace,scheduler:timer.snapshot(),simulationStepSeconds:.02,realTimeRatio:pace.activeWallMs>0?pace.activeIntervals*20/pace.activeWallMs:null}});
  const renderQueue=new LatestFrameQueue({
    render:snapshot=>{
      if(renderer.closed)throw Error('Native renderer unavailable; select a backend to recreate it');
      return renderer.request('frame',snapshot);
    },
    onFrame:(reply,snapshot)=>{
      const payload=framePayload(reply);
      png=payload.png;renderFailure=null;pace.lastFrameMs=snapshot.durationMs;
      frameState={tick:snapshot.snapshotTick,generation:snapshot.generation,publishedAtMs:Date.now(),
        sha256:payload.sha256};
    },
    onError:error=>{renderFailure=error.message;},
  });
  function invalidateFrames(){
    renderQueue.invalidate(++frameGeneration);png=null;frameState=null;renderFailure=null;
  }
  function frameSnapshot(){return {qpos:state.qpos,snapshotTick:state.tick,generation:frameGeneration};}
  function offerFrame(){
    if(renderer?.closed){renderFailure=renderFailure||'Native renderer unavailable; select a backend to recreate it';return;}
    try{renderQueue.offer(frameSnapshot());}catch(error){renderFailure=error.message;}
  }
  function recordResult(result){
    state=result.state;
    for(const completed of result.rounds||[]){session.record(completed);session.recordFight(completed);}
    if(state.terminal){session.record(state);input.release();roundRestartAt=Date.now()+intermissionMs;}
    if(state.fightResult>0){session.recordFight(state);paused=true;timer.pause();input.release();}
    state.moveDisposition=state.commandResults?.[active.humanSide]??null;state.held=[...input.held];
  }
  function options(backend){
    const definition=backends.find(b=>b.id===backend);
    if(!definition)return [];
    const saved=league.snapshot().policies;
    return league.opponentOptions({backend,configHash:definition.configHash}).map(p=>({
      ...p,label:saved[`${backend}/${p.id}`].label,
      ...(p.pairedStats.games?p.pairedStats:p.stats),ranked:p.rank!==null,
    }));
  }
  function catalog(){return {backends:backends.map(({id,label,available=true,error,runtimeNote,warning,roundEndNote,training})=>
    ({id,label,available,error,runtimeNote,warning,roundEndNote,training})),
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
    switching=true;workerGeneration++;timer.pause();input.reset();
    try{
      while(busy)await new Promise(resolve=>setTimeout(resolve,5));
      invalidateFrames();
      // This private viewer override never changes the training/league config.
      const nextConfig=createHumanConfig(backend.workerConfig,roundSeconds);
      if(worker)await worker.close();
      if(renderer)await renderer.close();
      if(humanConfig)humanConfig.close();
      humanConfig=nextConfig;
      state={ok:false,failure:'Loading native evaluator'};png=null;
      worker=new Worker({executable:backend.executable,config:humanConfig.path,
        env:backend.env,logFile:backend.logFile});
      renderer=new Renderer({executable:backend.executable,config:humanConfig.path,
        env:backend.env,logFile:backend.logFile,renderOnly:true});
      await Promise.all([worker.ready,renderer.ready]);
      if(policy.kind==='trained')await worker.request('policy',{
        side:1-humanSide,checkpoint:policy.checkpoint.path,sha256:policy.checkpoint.sha256,
        hiddenSize:policy.checkpoint.model.hiddenSize||256,layers:policy.checkpoint.model.layers||2,
        precision:policy.checkpoint.model.precision||'bf16',deterministic:policy.checkpoint.model.actionSelection!=='sampled',
        observationEncoding:policy.checkpoint.model.observationEncoding,
        recurrentResetTicks:policy.checkpoint.model.recurrentResetTicks||0,
        legacyFastHidden:policy.checkpoint.model.legacyFastHidden||0});
      else if(humanSide===1)await worker.request('policy',{side:0,scripted:true,checkpoint:''});
      state=(await worker.request('snapshot')).state;
      offerFrame();
      active={backend:backend.id,opponent:policy.id,humanSide,roundSeconds,
        trainingRoundSeconds:humanConfig.trainingRoundSeconds};lastClient=Date.now();
      session.reset();paused=true;roundRestartAt=0;
      pace={steps:0,activeIntervals:0,activeWallMs:0,lastStepMs:null,lastFrameMs:null};lastStepStart=null;
      return catalog();
    }catch(error){state={ok:false,failure:error.message};throw error;}
    finally{switching=false;}
  }
  async function tick(){
    if(Date.now()-lastClient>2000){input.release();paused=true;timer.pause();lastStepStart=null;return;}
    if(paused||Date.now()<roundRestartAt||busy||switching||!worker||!state.ok){
      if(paused||!state.ok)timer.pause();lastStepStart=null;return;
    }
    busy=true;
    try{
      const started=performance.now();
      if(lastStepStart!==null){pace.activeIntervals++;pace.activeWallMs+=started-lastStepStart;}
      lastStepStart=started;
      const result=await worker.request('step',{command:input.next(state,active.humanSide),humanSide:active.humanSide,steps:1});
      pace.lastStepMs=performance.now()-started;pace.steps++;recordResult(result);
      offerFrame();
    }catch(error){state={...state,ok:false,failure:error.message};timer.pause();input.release();}
    finally{busy=false;}
  }
  const timer=startPacedLoop(tick,{periodMs:20,startPaused:true});
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
      if(req.method==='GET'&&url.pathname==='/api/state'){lastClient=Date.now();return json(200,runtimeState());}
      if(req.method==='GET'&&url.pathname==='/api/snapshot')return json(200,runtimeState());
      if(req.method==='GET'&&url.pathname==='/api/protocol')return json(200,{schema:'rek.native_clone.step.v1',
        controlStepSeconds:.02,physicsStepSeconds:.002,request:'POST /api/step while paused',
        command:{forward:'[-1,1]',strafe:'[-1,1]',yaw:'[-1,1]',moveIndex:'-1..16, first step only',cancelAction:'boolean'},
        actions:33,observationFloats:446,maskBytes:66,qpos:72,qvel:70,
        reset:'POST /api/reset; remains paused',terminal:'rounds contains captured native terminals; no official parity claim'});
      if(req.method==='GET'&&url.pathname==='/health')return json(state.ok?200:503,{ok:state.ok,failure:state.failure});
      if(req.method==='GET'&&url.pathname==='/frame.png'){
        if(!png)return json(503,{error:renderFailure||'Frame unavailable'});
        res.setHeader('X-Rek-Snapshot-Tick',String(frameState.tick));
        res.setHeader('X-Rek-Frame-Generation',String(frameState.generation));
        res.setHeader('X-Rek-Frame-Sha256',frameState.sha256);
        return send(200,'image/png',png);
      }
      if(req.method==='POST'&&url.pathname==='/api/input'){
        if(switching)return json(409,{error:'Switch in progress'});
        const generation=workerGeneration;
        const value=await body(req);
        if(switching||generation!==workerGeneration)return json(409,{error:'Switch in progress or input generation expired'});
        lastClient=Date.now();return json(200,{ok:true,accepted:input.update(value)});
      }
      if(req.method==='POST'&&url.pathname==='/api/select')return json(200,await select(await body(req)));
      if(req.method==='POST'&&url.pathname==='/api/play'){
        if(switching||!worker)return json(409,{error:'Evaluator not ready'});
        const generation=workerGeneration;
        const value=await body(req);
        if(switching||!worker||generation!==workerGeneration)return json(409,{error:'Evaluator not ready'});
        if(typeof value.paused!=='boolean')throw new Error('Paused must be a boolean');
        if(!value.paused&&state.fightResult>0)return json(409,{error:'Match finished. Reset to begin another match.'});
        paused=value.paused;if(paused){timer.pause();input.release();lastStepStart=null;}else timer.resume();lastClient=Date.now();
        // A step already in flight may finish. Once this reply arrives the
        // paused snapshot is stable and no subsequent step can start.
        while(paused&&busy)await new Promise(resolve=>setTimeout(resolve,5));
        return json(200,{ok:true,paused});
      }
      if(req.method==='POST'&&url.pathname==='/api/step'){
        if(switching||!worker||!state.ok)return json(409,{error:'Evaluator not ready'});
        if(!paused)return json(409,{error:'Pause manual play before stepping'});
        const generation=workerGeneration;
        const value=await body(req),steps=value.steps??1;
        if(!Number.isInteger(steps)||steps<1||steps>512)throw Error('steps must be1..512');
        if((value.command!==undefined)===(value.action!==undefined))throw Error('Provide exactly one command or action');
        const args={steps,humanSide:active.humanSide,stopAtRound:true};
        if(value.command!==undefined)args.command=validateCommand(value.command);
        else{if(!Number.isInteger(value.action)||value.action<0||value.action>32)throw Error('action must be0..32');args.action=value.action;}
        // body() yielded, so another request may have acquired the worker.
        if(switching||!paused||state.fightResult>0||generation!==workerGeneration)return json(409,{error:'Worker in use or match finished; pause/reset first'});
        switching=true;try{
          while(busy)await new Promise(resolve=>setTimeout(resolve,5));
          input.release();lastStepStart=null;
          const result=await worker.request('step',args);recordResult(result);
          if(value.frame===true)await renderQueue.exact(frameSnapshot());
          else offerFrame();
          return json(200,{...result,state:{...runtimeState(),switching:false}});
        }finally{switching=false;}
      }
      if(req.method==='POST'&&url.pathname==='/api/reset'){
        if(switching||!worker)return json(409,{error:'Evaluator not ready'});
        switching=true;workerGeneration++;timer.pause();try{
          while(busy)await new Promise(resolve=>setTimeout(resolve,5));
          invalidateFrames();
          session.abandonFight(state);input.reset();paused=true;roundRestartAt=0;session.newRoundStream();lastStepStart=null;
          state=(await worker.request('reset')).state;
          offerFrame();return json(200,{ok:true});
        }finally{switching=false;}
      }
      if(req.method==='GET'){
        const routes={'/':'index.html','/app.js':'app.js','/controls.js':'controls.js','/state_labels.js':'state_labels.js','/style.css':'style.css'};
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
    timer.stop();renderQueue.close();switching=true;server.close();
    process.removeListener('SIGTERM',shutdown);process.removeListener('SIGINT',shutdown);
    while(busy)await new Promise(resolve=>setTimeout(resolve,5));
    if(worker)await worker.close();if(renderer)await renderer.close();if(humanConfig)humanConfig.close();
  }
  function shutdown(){close().finally(()=>process.exit(0));}
  process.once('SIGTERM',shutdown);process.once('SIGINT',shutdown);
  return {server,close};
}
if(require.main===module)serve(process.argv[2]).catch(error=>{console.error(error);process.exitCode=1;});
module.exports={serve};
