#!/usr/bin/env node
'use strict';
// Continue the authentic frozen-policy A/B campaign (planned-ab-rounds.json).
// Uses the same sha-pinned wrapper + driver as baseline-r1. Every attempt gets a
// new directory; existing artifacts are preserved. Completion rule is the plan's:
// observed terminal round with result and points; duration expiry is incomplete.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),os=require('node:os');
const {spawn}=require('node:child_process');
const stage=process.env.AB_STAGE||'/home/spark-advantage/rek-training/f7-action-id-live-20260924-r1';
const here=path.join(stage,'root-campaign');
const source='/home/spark-advantage/rek-training/primitive-live-eval-20260918-r1/source';
const driver=process.env.AB_DRIVER||path.join(stage,'live_transfer_run_masked.cjs');
const recorder=path.join(stage,'record_passive_defender.cjs');
const probe=path.join(here,'probe_state.cjs');
const DRIVER_SHA=process.env.AB_DRIVER_SHA||"f0009e3b19e0ce21501645dcd25e96956135cb2d05eeec6cb01dbc865c4074e6";
const MAX_ATTEMPTS=Number(process.env.AB_MAX_ATTEMPTS||5);
const LOBBY_WAIT_MS=Number(process.env.AB_LOBBY_WAIT_MS||480000);
const sha=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const utc=()=>new Date().toISOString();
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const {trialExitCode,provenG1T800Pairing}=require(driver);
if(typeof provenG1T800Pairing!=='function')throw Error('pairing-aware driver required');
const logStream=fs.createWriteStream(path.join(here,'campaign.log'),{flags:'a'});
function log(event,detail={}){const v={utc:utc(),event,...detail};const s=JSON.stringify(v);logStream.write(s+'\n');console.log(s);}
const ledgerPath=path.join(here,'ledger.json');
const ledger=fs.existsSync(ledgerPath)?JSON.parse(fs.readFileSync(ledgerPath,'utf8')):{};
const saveLedger=()=>fs.writeFileSync(ledgerPath,JSON.stringify(ledger,null,2)+'\n');

function run(cmd,args,{stdout,stderr,timeoutMs}){
  return new Promise(resolve=>{
    const child=spawn(cmd,args,{stdio:['ignore',stdout?'pipe':'ignore',stderr?'pipe':'ignore']});
    let out='';
    if(stdout){const s=typeof stdout==='string'?fs.createWriteStream(stdout,{flags:'wx',mode:0o600}):null;child.stdout.on('data',d=>{out+=d;s?.write(d);});child.on('close',()=>s?.end());}
    if(stderr){const s=fs.createWriteStream(stderr,{flags:'wx',mode:0o600});child.stderr.pipe(s);}
    const t=setTimeout(()=>{try{child.kill('SIGTERM');}catch{}},timeoutMs);
    child.on('close',(code,signal)=>{clearTimeout(t);resolve({code,signal,out});});
  });
}
async function probeState(config){
  const r=await run(process.execPath,[probe,config],{stdout:true,timeoutMs:25000});
  const line=r.out.trim().split('\n').filter(Boolean).pop();
  let s=null;try{s=JSON.parse(line);}catch{}
  fs.appendFileSync(path.join(here,'probe.jsonl'),(line||JSON.stringify({utc:utc(),event:'probe_failed',code:r.code}))+'\n');
  return s;
}
function gameAlive(){
  try{const pid=fs.readFileSync('/dev/null','utf8').trim();
    // host-visible: any REK.exe process with that wine command line
    const procs=fs.readdirSync('/proc').filter(x=>/^\d+$/.test(x));
    return procs.some(p=>{try{return fs.readFileSync(`/proc/${p}/cmdline`).toString().split('\0').includes('/opt/codexrook/rek-core-referee-20260924-r1/REK.exe');}catch{return false;}});
  }catch{return false;}
}
function launchWindow(s){
  // Fresh lobby entry, or a between-round / idle window so the driver streams from a round's first tick.
  if(!s||s.isolated_session_verified!==true)return null;
  if(s.scene==='Lobby'&&s.lease_held===false&&s.g1_policy_stream_running===false)return 'lobby';
  const p=s.private_ai||{};
  if(s.scene!=='Arena'||p.proven!==true||s.lease_held!==false||s.g1_policy_stream_running!==false)return null;
  if(p.client_ai_difficulty!==0||p.sparring_bot_number!==1)return null;
  if(['BetweenRounds','FightOver'].includes(p.phase)&&p.round_active===false)return 'between_rounds:'+p.phase;
  if(p.phase==='Idle'&&p.round_inactive===true)return 'idle';
  return null;
}
// One persistent bridge connection polls get_state; it is fully closed (relay exited)
// before this resolves, because the bridge accepts one client at a time.
function watchOnce(cfg,timeoutMs){
  return new Promise((resolve,reject)=>{
    const child=spawn(cfg.relay[0],cfg.relay.slice(1),{stdio:['pipe','pipe','pipe']});
    let n=0,last=null,outcome=null,poll=null,killer=null,sawState=false;
    const deadline=setTimeout(()=>settle(Error('no launch window within '+timeoutMs+' ms; last='+JSON.stringify(last))),timeoutMs);
    function settle(err,val){
      if(outcome)return;outcome={err,val};clearInterval(poll);clearTimeout(deadline);
      try{child.stdin.end();}catch{}
      killer=setTimeout(()=>{if(child.exitCode===null&&child.signalCode===null)child.kill('SIGTERM');},300);
    }
    child.on('exit',()=>{clearTimeout(killer);
      if(!outcome)outcome={err:Object.assign(Error('watch relay exited; last='+JSON.stringify(last)),{dead_on_arrival:!sawState})};
      clearInterval(poll);clearTimeout(deadline);
      outcome.err?reject(outcome.err):resolve(outcome.val);});
    child.stderr.on('data',()=>{});
    const rl=require('node:readline').createInterface({input:child.stdout});
    rl.on('line',raw=>{
      if(outcome)return;
      let m;try{m=JSON.parse(raw);}catch{return;}
      if(m.event==='hello'){poll=setInterval(()=>{try{child.stdin.write(JSON.stringify({type:'get_state',request_id:'watch-'+(++n)})+'\n');}catch{}},1000);return;}
      if(m.event!=='state')return;
      sawState=true;
      const p=m.private_ai||{},c=m.control||{},f=m.foreground||{};
      const unsupportedPairing=provenG1T800Pairing(m);
      const st={utc:utc(),scene:m.scene,lobby_screen:m.lobby_screen,isolated_session_verified:f.isolated_session_verified,
        lease_held:c.lease_held,g1_policy_stream_running:c.g1_policy_stream_running,
        ...(unsupportedPairing?{unsupported_pairing:unsupportedPairing}:{}),
        private_ai:{proven:p.proven,phase:p.phase,round_active:p.round_active,round_number:p.round_number,round_inactive:p.round_inactive,
          post_fight_prompt:p.post_fight_prompt,post_fight_is_winner:p.post_fight_is_winner,fight_epoch:p.fight_epoch,
          client_ai_difficulty:p.client_ai_difficulty,sparring_bot_number:p.sparring_bot_number}};
      last=st;fs.appendFileSync(path.join(here,'probe.jsonl'),JSON.stringify(st)+'\n');
      if(unsupportedPairing&&st.lease_held===false&&st.g1_policy_stream_running===false){
        settle(null,st);return;
      }
      const w=launchWindow(st);
      if(w){log('launch_window',{window:w,phase:st.private_ai.phase,round:st.private_ai.round_number,fight_epoch:st.private_ai.fight_epoch});settle(null,st);return;}
      if(n%15===1)log('waiting_for_launch_window',{scene:st.scene,lobby_screen:st.lobby_screen,phase:st.private_ai.phase,round:st.private_ai.round_number,lease_held:st.lease_held});
    });
  });
}
async function watchForWindow(config,timeoutMs){
  const cfg=JSON.parse(fs.readFileSync(config,'utf8'));
  const deadline=Date.now()+timeoutMs;let tries=0;
  while(true){
    try{return await watchOnce(cfg,Math.max(5000,deadline-Date.now()));}
    catch(e){
      if(!gameAlive())throw Error('game_disappeared_while_watching');
      if(!e.dead_on_arrival||Date.now()>=deadline||++tries>20)throw e;
      log('watch_relay_retry',{tries});await sleep(3000);
    }
  }
}
let plannedNextRelaunch=false;
let relaunches=fs.existsSync(path.join(here,'campaign.log'))?fs.readFileSync(path.join(here,'campaign.log'),'utf8').trim().split('\n').filter(Boolean).map(x=>JSON.parse(x)).filter(x=>x.event==='relaunch_start'&&!x.planned_restart).length:0;
const MAX_RELAUNCHES=Number(process.env.AB_MAX_RELAUNCHES||3);
async function relaunchGame(){
  const planned=plannedNextRelaunch;plannedNextRelaunch=false;
  if(!planned&&relaunches>=MAX_RELAUNCHES)throw Error('relaunch budget exhausted');
  if(gameAlive())throw Error('Refusing prefix cleanup with a live game');
  const cleanup=await run('/bin/bash',[path.join(here,'clear_dead_prefix.sh')],{stdout:true,timeoutMs:30000});
  if(cleanup.code!==0)throw Error('Dedicated dead-prefix cleanup failed');
  if(!planned)relaunches++;let k=2;while(fs.existsSync(`/home/spark-advantage/codexrook-runtime/live-transfer-20260915/live-attack-gate-20260921-r${k}`)||fs.existsSync(path.join(here,`relaunch-r${k}.stdout.txt`)))k++;
  const suffix='r'+k;
  log('relaunch_start',{suffix,planned_restart:planned,cumulative_unplanned_relaunches:relaunches});
  const r=await run('/bin/bash',[path.join(here,'relaunch.sh'),suffix],{stdout:path.join(here,`relaunch-${suffix}.stdout.txt`),stderr:path.join(here,`relaunch-${suffix}.stderr.txt`),timeoutMs:120000});
  log('relaunch_submitted',{suffix,exit:r.code});
  if(r.code!==0)throw Error('relaunch failed with exit '+r.code);
  // Wait for the client to reach the Lobby (fresh process: login screen).
  const deadline=Date.now()+420000;
  while(Date.now()<deadline){
    await sleep(15000);
    if(!gameAlive())throw Error('relaunched REK.exe exited');
    let st=null;try{st=await watchForWindow(anyConfigPath,20000);}catch(e){log('relaunch_wait',{message:e.message.slice(0,120)});}
    if(st&&st.scene==='Lobby'){log('relaunch_ready',{lobby_screen:st.lobby_screen});return;}
  }
  throw Error('relaunched client did not reach Lobby');
}
let anyConfigPath=null;
async function recycleUnsupportedPairing(st,entry){
  if(!st||st.isolated_session_verified!==true||st.scene!=='Arena'||
     st.lease_held!==false||st.g1_policy_stream_running!==false||
     st.private_ai?.proven!==true||st.private_ai.client_ai_difficulty!==0||
     st.private_ai.sparring_bot_number!==1||
     st.unsupported_pairing?.code!=='unsupported_pairing:local_g1_opponent_t800')
    throw Error('Unsupported-pair recycle scope not verified');
  if(relaunches>=MAX_RELAUNCHES)throw Error('relaunch budget exhausted before unsupported-pair recycle');
  log('unsupported_pairing_excluded',{label:entry?.label,policy_rng_seed:entry?.policy_rng_seed,
    retry_same_seed:true,counted:false,proof:st.unsupported_pairing});
  const recycled=await run('/bin/bash',[path.join(here,'recycle_owned_client.sh')],{stdout:true,timeoutMs:30000});
  if(recycled.code!==0)throw Error('Scoped unsupported-pair client recycle failed');
  plannedNextRelaunch=false;
  log('unsupported_pairing_recycle_end',{label:entry?.label,exit:recycled.code});
}
async function waitForLaunchWindow(config,entry){
  anyConfigPath=config;
  for(;;){
    if(!gameAlive()){log('game_not_running',{});await relaunchGame();}
    let st;
    try{st=await watchForWindow(config,LOBBY_WAIT_MS);}
    catch(error){if(gameAlive())throw error;log('game_disappeared_while_watching');continue;}
    if(st?.unsupported_pairing){await recycleUnsupportedPairing(st,entry);continue;}
    return st;
  }
}
function attemptPaths(label,attempt){
  const name=attempt===1&&fs.readdirSync(path.join(stage,label)).length===0?label:`${label}-retry${attempt}`;
  const dir=path.join(stage,name);
  if(name!==label&&fs.existsSync(dir))throw Error('attempt dir exists: '+dir);
  if(name!==label)fs.mkdirSync(dir,{mode:0o700});
  let config=path.join(stage,'configs',`${label}.json`);
  if(name!==label){
    const c=JSON.parse(fs.readFileSync(config,'utf8'));c.out=path.join(dir,'trial');
    config=path.join(stage,'configs',`${name}.json`);
    fs.writeFileSync(config,JSON.stringify(c,null,2)+'\n',{flag:'wx',mode:0o600});
  }
  return {name,dir,config,media:path.join(dir,'media')};
}
function verdict(dir,expectedCheckpoint){
  const sp=path.join(dir,'trial','summary.json'),mp=path.join(dir,'media','capture-manifest.json');
  const summary=fs.existsSync(sp)?JSON.parse(fs.readFileSync(sp,'utf8')):null;
  const manifest=fs.existsSync(mp)?JSON.parse(fs.readFileSync(mp,'utf8')):null;
  const fr=summary?.final_round||null;
  const scores=fr?.clean_hits;
  const scorePair=Array.isArray(scores)&&scores.length===2&&scores.every(x=>Number.isInteger(x)&&x>=0);
  const expectedWinner=scorePair?(scores[0]===scores[1]?-1:scores[0]>scores[1]?0:1):null;
  const resultValid=scorePair&&fr.winner_index===expectedWinner&&
    ((expectedWinner===-1&&fr.result==='Tie'&&fr.result_value===3)||
     (expectedWinner!==-1&&fr.result==='WonByPoints'&&fr.result_value===1));
  const terminal=!!fr&&fr.active===false&&fr.time_remaining===0&&resultValid&&summary.predictions>0&&summary.applied>0;
  const sameBot=summary?.opponent?.client_ai_difficulty===0&&summary?.opponent?.sparring_bot_number===1;
  const ir=summary?.initial_round||null;
  // Policy must own the round from its first tick: no pre-existing hits, no elapsed time.
  const fairStart=!!ir&&ir.active===true&&ir.time_remaining>=117&&ir.time_remaining<=120&&ir.duration===120&&fr?.duration===120&&ir.redo===false&&fr.redo===false&&Array.isArray(ir.clean_hits)&&ir.clean_hits[0]===0&&ir.clean_hits[1]===0&&ir.number===fr?.number;
  const complete=terminal&&sameBot&&fairStart&&summary.checkpoint_sha256===expectedCheckpoint;
  return {complete,terminal,same_bot:sameBot,fair_start:fairStart,initial_round:ir,stop_reason:summary?.stop_reason??null,
    ...(summary?.unsupported_pairing?{excluded_cause:summary.unsupported_pairing.code,unsupported_pairing:summary.unsupported_pairing}:{}),
    predictions:summary?.predictions??null,applied:summary?.applied??null,rejected:summary?.rejected??null,
    final_round:fr,video:manifest?.delivery_status??null,video_sha256:manifest?.video_sha256??null};
}
async function attempt(entry,n){
  const p=attemptPaths(entry.label,n);
  const prov={utc_start:utc(),host:os.hostname(),label:entry.label,attempt:n,plan_order:entry.order,
    config:p.config,config_sha256:sha(p.config),driver,driver_sha256:sha(driver),recorder,recorder_sha256:sha(recorder),
    checkpoint_sha256:entry.checkpoint_sha256,policy_rng_seed:entry.policy_rng_seed,
    command:[process.execPath,recorder,p.config,driver,p.media]};
  fs.writeFileSync(path.join(p.dir,'attempt-provenance.json'),JSON.stringify(prov,null,2)+'\n',{flag:'wx',mode:0o600});
  log('attempt_start',{label:entry.label,attempt:n,dir:p.dir});
  const r=await run(process.execPath,[recorder,p.config,driver,p.media],{
    stdout:path.join(p.dir,'wrapper.stdout.jsonl'),stderr:path.join(p.dir,'wrapper.stderr.txt'),timeoutMs:420000});
  fs.writeFileSync(path.join(p.dir,'wrapper.exit-code.txt'),`${r.code}\n`,{flag:'wx'});
  const v=verdict(p.dir,entry.checkpoint_sha256);
  const rec={attempt:n,dir:p.dir,config:p.config,utc_start:prov.utc_start,utc_end:utc(),wrapper_exit:r.code,wrapper_signal:r.signal,...v};
  (ledger[entry.label]??=[]).push(rec);saveLedger();
  log('attempt_end',{label:entry.label,attempt:n,complete:v.complete,stop_reason:v.stop_reason,excluded_cause:v.excluded_cause,final_round:v.final_round,wrapper_exit:r.code});
  return rec;
}
async function main(){
  if(os.hostname()!=='spark-4ae3')throw Error('Spark required');
  if(sha(driver)!==DRIVER_SHA)throw Error('driver sha changed');
  const lock=path.join(here,'campaign.lock');
  try{fs.writeFileSync(lock,String(process.pid),{flag:'wx'});}catch{throw Error('campaign already running (lock present)');}
  process.on('exit',()=>{try{fs.unlinkSync(lock);}catch{}});
  const plan=JSON.parse(fs.readFileSync(path.join(stage,process.env.AB_PLAN||'planned-rounds.json'),'utf8')).plan;
  const anyConfig=plan[0].config_path;
  log('campaign_start',{pid:process.pid,plan_rounds:plan.length,max_attempts:MAX_ATTEMPTS});
  for(const entry of plan){
    const counted=Object.values(ledger).flatMap(v=>v.filter(a=>a.complete));
    const nonwins=counted.filter(a=>a.final_round.winner_index!==0).length;
    if(nonwins>=3){log('criterion_failed',{counted:counted.length,nonwins,required:'18 wins in 20 fixed labels'});process.exitCode=1;break;}
    if((ledger[entry.label]||[]).some(a=>a.complete)){log('skip_complete',{label:entry.label});continue;}
    const prior=(ledger[entry.label]||[]).length;
    let done=false;
    for(let n=prior+1;n<=MAX_ATTEMPTS&&!done;n++){
      await waitForLaunchWindow(anyConfig,entry);
      const rec=await attempt(entry,n);
      done=rec.complete;
      if(done&&rec.final_round.number>=1&&gameAlive()){
        const st=await probeState(rec.config);
        const p=st?.private_ai||{};
        if(!st||st.isolated_session_verified!==true||st.scene!=='Arena'||p.proven!==true||
           p.client_ai_difficulty!==0||p.sparring_bot_number!==1||st.lease_held!==false||st.g1_policy_stream_running!==false)
          throw Error('Completed-round recycle scope not verified');
        log('proactive_recycle_start',{completed_label:entry.label,round:rec.final_round.number,reason:'fresh_client_after_each_counted_round'});
        const recycled=await run('/bin/bash',[path.join(here,'recycle_owned_client.sh')],{stdout:true,timeoutMs:30000});
        if(recycled.code!==0)throw Error('Scoped client recycle failed');
        plannedNextRelaunch=true;
        log('proactive_recycle_end',{completed_label:entry.label,round:rec.final_round.number,exit:recycled.code});
      }
      if(!done){if(!gameAlive())log('game_died_during_attempt',{label:entry.label,attempt:n});await sleep(5000);}
    }
    if(!done){log('label_incomplete',{label:entry.label});throw Error('attempt budget exhausted for '+entry.label);}
  }
  log('campaign_end',{summary:Object.fromEntries(Object.entries(ledger).map(([k,v])=>[k,v.filter(a=>a.complete).map(a=>a.final_round?.clean_hits)]))});
}
if(process.argv[2]==='--verdict'){console.log(JSON.stringify(verdict(process.argv[3])));logStream.end();}
else main().catch(e=>{log('campaign_error',{message:e.message});process.exitCode=2;});
