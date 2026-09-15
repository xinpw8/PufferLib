#!/usr/bin/env node
'use strict';
// Transport orchestration only. Observation encoding is C++; inference is CUDA.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawn} = require('node:child_process');
const {EventEmitter} = require('node:events');
const readline = require('node:readline');

function requireValue(ok, message) { if (!ok) throw new Error(message); }
function privateArena(s) {
  const p = s?.private_ai;
  return p?.proven === true && p.solo_route_proven === true &&
    p.context_is_solo === true && p.exact_sparring_bot_1 === true &&
    p.opponent_is_ai === true && p.human_in_opponent_slot === false;
}
function canExitLostPrivateSession(s) {
  return privateArena(s) && s.private_ai.round_active===false &&
    s.private_ai.round_inactive===true && s.private_ai.post_fight_prompt===true &&
    s.private_ai.post_fight_is_winner===false;
}
function validateWorkerReady(ready, sha) {
  requireValue(ready?.type==='ready' && ready.checkpoint_sha256===sha &&
    ready.native_cuda===true && ready.environment_stepping===false &&
    ready.observation_schema==='rek.native5.scaled_polar_xy.v1' &&
    ready.precision==='bf16' && ready.selection==='sampled' &&
    ready.observations===223 && ready.actions===33 &&
    ready.hidden_size===256 && ready.num_layers===2, 'worker identity mismatch');
}
function validateWorkerAction(prediction, source, sha) {
  requireValue(source && prediction?.type==='action' &&
    prediction.seq===source.sequence && prediction.round_id===source.round &&
    prediction.checkpoint_sha256===sha, 'prediction_source_identity_mismatch');
  requireValue(Number.isInteger(prediction.action) && prediction.action>=0 &&
    prediction.action<33, 'prediction_action_out_of_range');
}
function guardedCallback(handler, onFailure) {
  return value => {try {handler(value);} catch(error) {onFailure(error instanceof Error ? error : new Error(String(error)));}};
}
function measuredQpc(clock) {
  const ticks=clock?.qpc_ticks;
  requireValue((Number.isSafeInteger(ticks)&&ticks>0) ||
    (typeof ticks==='string'&&/^[1-9][0-9]*$/.test(ticks)), 'source_qpc_unavailable');
  return BigInt(ticks);
}
class LiveActionPacer {
  constructor() {this.pending=null;this.lastAckQpc=null;this.lastOfferedQpc=null;}
  offer(source, received) {
    // No queue is retained while encoding, inferring, or awaiting the game ack.
    if(this.pending)return 'action_inflight';
    const qpc=measuredQpc(source.clock);
    if(this.lastAckQpc!==null && qpc<=this.lastAckQpc)return 'source_before_last_ack';
    if(this.lastOfferedQpc!==null && qpc<=this.lastOfferedQpc)return 'source_clock_not_new';
    this.pending={sequence:source.observation_sequence,round:source.round_identity_sha256,
      received,qpc,requestId:null,action:null,sent:null};
    this.lastOfferedQpc=qpc;
    return null;
  }
  sent(requestId, action, now) {
    requireValue(this.pending && this.pending.requestId===null, 'action_already_inflight');
    Object.assign(this.pending,{requestId,action,sent:now});
  }
  acknowledge(ack) {
    const p=this.pending;
    if(!p || p.requestId===null || ack.request_id!==p.requestId)return false;
    requireValue(ack.observation_sequence===p.sequence && ack.round_identity_sha256===p.round &&
      ack.action===p.action, 'action_ack_identity_mismatch');
    const qpc=measuredQpc(ack.clock);
    requireValue(qpc>=p.qpc && (this.lastAckQpc===null || qpc>=this.lastAckQpc), 'action_ack_clock_regressed');
    // Rejected actions also consume a measured game decision boundary.
    this.lastAckQpc=qpc;
    this.pending=null;
    return true;
  }
  abandon() {
    requireValue(!this.pending || this.pending.requestId===null, 'cannot_abandon_unacknowledged_action');
    this.pending=null;
  }
}
function childEndpoint(name, spec, out) {
  requireValue(Array.isArray(spec) && spec.length > 0 && spec.every(x => typeof x === 'string'), `invalid ${name} command`);
  const bus = new EventEmitter();
  const stdout = fs.createWriteStream(path.join(out, `${name}.stdout.jsonl`), {flags:'wx'});
  const stderr = fs.createWriteStream(path.join(out, `${name}.stderr.txt`), {flags:'wx'});
  const child = spawn(spec[0], spec.slice(1), {stdio:['pipe','pipe','pipe']});
  child.stderr.pipe(stderr);
  child.stdin.on('error', error => bus.emit('failure', error));
  const lines = readline.createInterface({input:child.stdout});
  lines.on('line', raw => {
    stdout.write(raw + '\n');
    try { bus.emit('message', JSON.parse(raw)); } catch(e) { bus.emit('invalid', raw); }
  });
  child.on('error', error => bus.emit('failure', error));
  child.on('exit', (code,signal) => { bus.emit('exit', {code,signal}); stdout.end(); });
  const requests = fs.createWriteStream(path.join(out, `${name}.stdin.jsonl`), {flags:'wx'});
  function send(object) {
    requireValue(!child.killed && child.exitCode === null && child.stdin.writable, `${name} is not writable`);
    const line = JSON.stringify(object) + '\n'; requests.write(line); child.stdin.write(line);
  }
  function wait(predicate, timeout=10000) {
    return new Promise((resolve,reject) => {
      const timer=setTimeout(() => finish(new Error(`${name} response timeout`)), timeout);
      function finish(error, value) {clearTimeout(timer); bus.off('message',onMessage);bus.off('exit',onExit);bus.off('failure',onFailure);bus.off('invalid',onInvalid); error ? reject(error) : resolve(value);}
      function onMessage(value) {
        try {
          if(value?.type==='fatal' || (name==='worker' && value?.type==='error')) {
            finish(new Error(`${name} startup failure:${value.code||'unknown'}`)); return;
          }
          if(predicate(value))finish(null,value);
        } catch(error) {finish(error);}
      }
      function onExit(value) {finish(new Error(`${name} exited ${JSON.stringify(value)}`));}
      function onFailure(error) {finish(error);}
      function onInvalid() {finish(new Error(`${name} invalid JSON response`));}
      bus.on('message',onMessage);bus.once('exit',onExit);bus.once('failure',onFailure);bus.once('invalid',onInvalid);
    });
  }
  return {child,bus,send,wait,close(){child.stdin.end(); requests.end();}};
}
async function run(configPath) {
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  requireValue(config.projection === 'client_pose_projection_v1', 'explicit projection required');
  requireValue(Number.isFinite(config.max_seconds) && config.max_seconds > 0 && config.max_seconds <= 600, 'max_seconds must be 0..600');
  requireValue(/^[a-f0-9]{64}$/.test(config.checkpoint_sha256), 'checkpoint SHA required');
  fs.mkdirSync(config.out, {recursive:false});
  fs.writeFileSync(path.join(config.out,'run-config.json'), JSON.stringify(config,null,2)+'\n', {flag:'wx'});
  const events=fs.createWriteStream(path.join(config.out,'orchestrator.jsonl'), {flags:'wx'});
  const log=(event, detail={}) => {const value={utc:new Date().toISOString(),event,...detail};events.write(JSON.stringify(value)+'\n');console.log(JSON.stringify(value));};
  const endpoints=[]; let relay,encoder,worker, leased=false, streaming=false, stopping=false;
  let nextId=0, sourceCount=0, skipped=0, predictions=0, applied=0, rejected=0, unmatchedAcks=0;
  let firstRound=null,lastRound=null,lastSourceAt=0,stopReason='not_started';
  const actions=Array(33).fill(0), reasons={}, droppedSources={}, pacer=new LiveActionPacer();
  let doneResolve; const done=new Promise(resolve => doneResolve=resolve);
  let timer, watchdog;
  async function request(type, fields={}, event='ack') {
    const request_id=`live-${++nextId}`;
    const response=relay.wait(x => x.event===event && x.request_id===request_id);
    relay.send({type,request_id,...fields}); return response;
  }
  async function command(command) {
    const ack=await request('command',{command});
    log('command_result',{command,status:ack.status,reason:ack.reason});
    requireValue(ack.status==='accepted', `${command}: ${ack.reason}`); return ack;
  }
  function finish(reason) {if(!stopping){stopping=true;stopReason=reason;doneResolve();}}
  try {
    encoder=childEndpoint('encoder',config.encoder,config.out); endpoints.push(encoder);
    worker=childEndpoint('worker',config.worker,config.out); endpoints.push(worker);
    relay=childEndpoint('relay',config.relay,config.out); endpoints.push(relay);
    for(const endpoint of endpoints) {
      endpoint.bus.on('failure',error=>finish(error.message));
      endpoint.bus.on('exit',x=>finish(`child_exit:${JSON.stringify(x)}`));
      endpoint.bus.on('invalid',()=>finish('child_invalid_json'));
    }
    const [ready]=await Promise.all([
      worker.wait(x=>x.type==='ready',60000), relay.wait(x=>x.event==='hello',30000)
    ]);
    validateWorkerReady(ready, config.checkpoint_sha256);
    log('inference_ready',{checkpoint_sha256:ready.checkpoint_sha256,device:ready.device,precision:ready.precision,selection:ready.selection,projection:config.projection});
    let state=await request('get_state',{},'state');
    await command('AcquireExclusiveControl'); leased=true;
    if(canExitLostPrivateSession(state)) {
      requireValue(config.enter_private===true, 'lost private session requires explicit private-entry recovery');
      await command('ExitLostPrivateSession');
      const deadline=Date.now()+15000;
      do {
        await new Promise(r=>setTimeout(r,100));
        state=await request('get_state',{},'state');
        requireValue(Date.now()<deadline,'lost private session exit timeout');
      } while(privateArena(state));
    }
    if(!privateArena(state)) {
      requireValue(config.enter_private===true, 'current client is not verified solo Bot1 arena');
      const entered=new Set();const deadline=Date.now()+45000;
      while(!privateArena(state)) {
        const menu={Login:'ConfirmLoggedIn',Home:'NavigateFreePlay',FreePlay:'EnterSolo'}[state.lobby_screen];
        if(menu&&!entered.has(menu)){await command(menu);entered.add(menu);}
        else requireValue(state.scene==='Arena'||menu||state.lobby_screen==='Intro', 'no supported private-practice menu route');
        await new Promise(r=>setTimeout(r,150));
        state=await request('get_state',{},'state');
        requireValue(Date.now()<deadline,'private-practice entry timeout');
      }
      log('private_practice_observed',{opponent:'Sparring Bot 1',human_opponent:false});
    }
    if(!state.private_ai.active_gameplay_proven || state.private_ai.round_active!==true) {
      await command('StartRound');
      const deadline=Date.now()+30000;
      while(Date.now()<deadline) {
        const s=await request('get_state',{},'state');
        requireValue(privateArena(s),'private arena proof lost');
        if(s.private_ai.active_gameplay_proven && s.private_ai.round_active)break;
        await new Promise(r=>setTimeout(r,100));
        requireValue(Date.now()<deadline,'active round timeout');
      }
    }
    relay.bus.on('message',guardedCallback(source => {
      if(source.event==='g1_policy_end') {log('stream_end',source); finish(`stream_end:${source.reason}`);return;}
      if(source.event==='g1_policy_action') {
        if(!pacer.acknowledge(source)){unmatchedAcks++;log('unmatched_action_ack',{request_id:source.request_id,seq:source.observation_sequence});return;}
        source.applied ? applied++ : rejected++;
        reasons[source.reason]=(reasons[source.reason]||0)+1;
        return;
      }
      if(source.event!=='g1_policy_state'||stopping)return;
      sourceCount++;lastSourceAt=Date.now();
      if(source.round){firstRound??=source.round;lastRound=source.round;}
      if(!streaming)return;
      const dropped=pacer.offer(source,Date.now());
      if(dropped){skipped++;droppedSources[dropped]=(droppedSources[dropped]||0)+1;return;}
      encoder.send(source);
    },error=>finish(`relay_callback:${error.message}`)));
    encoder.bus.on('message',guardedCallback(encoded=> {
      if(stopping||encoded.event!=='policy_observation')return;
      if(encoded.ready!==true) {log('observation_unavailable',{reason:encoded.reason,unavailable:encoded.unavailable});pacer.abandon();return;}
      const r=encoded.worker_request;
      const pending=pacer.pending;
      if(!pending || r?.seq!==pending.sequence || r?.round_id!==pending.round) {finish('encoder_source_identity_mismatch');return;}
      worker.send(r);
    },error=>finish(`encoder_callback:${error.message}`)));
    worker.bus.on('message',guardedCallback(prediction=> {
      if(stopping)return;
      if(prediction.type==='terminal'){pacer.abandon();finish('source_round_terminal');return;}
      if(prediction.type==='error'||prediction.type==='fatal'){finish(`worker_error:${prediction.code}`);return;}
      if(prediction.type!=='action')return;
      const source=pacer.pending;
      validateWorkerAction(prediction, source, config.checkpoint_sha256);
      predictions++;actions[prediction.action]++;
      if(Date.now()-source.received>=200){log('stale_prediction_discarded',{seq:prediction.seq,local_latency_ms:Date.now()-source.received});pacer.abandon();return;}
      const request_id=`action-${++nextId}`;
      pacer.sent(request_id,prediction.action,Date.now());
      relay.send({type:'policy_action',request_id,round_identity_sha256:source.round,observation_sequence:source.sequence,action:prediction.action});
    },error=>finish(`worker_callback:${error.message}`)));
    streaming=true;
    await command('StartG1PolicyStream');
    log('live_policy_started');
    timer=setTimeout(()=>finish('requested_duration_complete'),config.max_seconds*1000);
    watchdog=setInterval(()=>{
      if(lastSourceAt && Date.now()-lastSourceAt>2000)finish('source_stream_missing');
      if(pacer.pending?.sent!==null && pacer.pending?.sent!==undefined && Date.now()-pacer.pending.sent>2000)finish('action_ack_missing');
    },250);
    process.once('SIGINT',()=>finish('operator_interrupt'));
    process.once('SIGTERM',()=>finish('operator_termination'));
    await done;
  } catch(error) {finish(error.message);log('error',{message:error.message});}
  finally {
    clearTimeout(timer);clearInterval(watchdog);streaming=false;
    if(leased){try{await command('StopG1PolicyStream');}catch(e){log('stop_error',{message:e.message});}try{await command('ReleaseExclusiveControl');}catch(e){log('release_error',{message:e.message});}}
    const summary={stop_reason:stopReason,source_count:sourceCount,skipped_sources:skipped,dropped_sources:droppedSources,
      unmatched_action_acks:unmatchedAcks,action_inflight_at_stop:pacer.pending?.requestId!==null&&pacer.pending?.requestId!==undefined,
      last_ack_qpc_ticks:pacer.lastAckQpc?.toString()??null,predictions,applied,rejected,actions,reasons,initial_round:firstRound,final_round:lastRound,projection:config.projection,checkpoint_sha256:config.checkpoint_sha256,authentic_client:true,global_input_emitted:false};
    fs.writeFileSync(path.join(config.out,'summary.json'),JSON.stringify(summary,null,2)+'\n',{flag:'wx'});log('summary',summary);
    for(const e of endpoints)e.close();
    setTimeout(()=>{for(const e of endpoints)if(e.child.exitCode===null)e.child.kill('SIGTERM');},2000).unref();
    events.end();
    if(predictions===0 || applied===0)process.exitCode=2;
  }
}
if(require.main===module) {requireValue(process.argv.length===3,'usage: node live_transfer_run.cjs config.json');run(process.argv[2]).catch(e=>{console.error(e.message);process.exitCode=2;});}
module.exports={privateArena,canExitLostPrivateSession,validateWorkerReady,validateWorkerAction,guardedCallback,childEndpoint,LiveActionPacer};
