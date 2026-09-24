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
  return privateAiRouteNoHuman(s) && p.policy_proven === true && knownBot(p);
}
function knownBot(p) {
  return Number.isInteger(p?.client_ai_difficulty) && p.client_ai_difficulty>=0 &&
    p.client_ai_difficulty<=255 && p.sparring_bot_number===p.client_ai_difficulty+1;
}
function botIdentity(p) {
  requireValue(knownBot(p),'unknown_private_ai_identity');
  return {client_ai_difficulty:p.client_ai_difficulty,sparring_bot_number:p.sparring_bot_number};
}
function validateBotIdentity(opponent, expected) {
  requireValue(knownBot(opponent) && opponent.opponent_is_ai===true &&
    opponent.human_in_opponent_slot===false &&
    opponent.client_ai_difficulty===expected.client_ai_difficulty &&
    opponent.sparring_bot_number===expected.sparring_bot_number,
    'private_ai_identity_changed_or_unproven');
}
function trialExitCode(summary) {
  if(summary.predictions<=0 || summary.applied<=0)return 2;
  if(summary.stop_reason==='requested_duration_complete')return 0;
  const terminal=summary.final_round?.active===false &&
    Number.isInteger(summary.final_round.result_value) && summary.final_round.result_value>0;
  return terminal && ['source_round_terminal','stream_end:active_round_not_observed'].includes(summary.stop_reason)?0:2;
}
function canExitLostPrivateSession(s) {
  return privateArena(s) && s.private_ai.round_active===false &&
    s.private_ai.round_inactive===true && s.private_ai.post_fight_prompt===true &&
    s.private_ai.post_fight_is_winner===false;
}
function privateAiRouteNoHuman(s) {
  const p=s?.private_ai;
  return s?.scene==='Arena' && s.foreground?.isolated_session_verified===true &&
    p!==undefined && p!==null &&
    p.network_client_only===true && p.context_is_solo===true && p.solo_route_proven===true &&
    p.opponent_is_ai===true && p.opponent_slot_is_ai===true && p.human_in_opponent_slot===false &&
    p.opponent_slot_client_known===true && p.opponent_slot_has_client===false && p.opponent_human_bit_set===false;
}
function provenG1T800Pairing(s) {
  const p=s?.private_ai,m=s?.measured_pairing,l=m?.local_fighter,o=m?.opponent_fighter;
  if(!privateAiRouteNoHuman(s)||!knownBot(p)||p.proven!==true||p.policy_proven!==true||
     p.client_ai_difficulty!==0||p.sparring_bot_number!==1||
     p.round_active!==true||p.client_visual_only_fighter_pair!==true||
     ![0,1].includes(p.local_slot)||p.opponent_slot!==(p.local_slot^1)||
     m?.local_slot!==p.local_slot||m?.opponent_slot!==p.opponent_slot||
     m.reason!=='mixed_supported_runtime_models_rejected'||m.exact_g1_vs_g1!==false||
     l?.semantic_robot_id!=='g1'||l.exact_g1_bone_signature!==true||l.bone_count!==30||
     l.runtime_bone_signature_sha256!=='9d18e697233d9578b398fbe849cd59d65cb27a5c2223b2602db66a82a410e987'||
     o?.semantic_robot_id!=='t800'||o.exact_t800_bone_signature!==true||o.bone_count!==26||
     o.runtime_bone_signature_sha256!=='ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab')return null;
  return {code:'unsupported_pairing:local_g1_opponent_t800',reason:m.reason,
    observed_utc:s.observed_utc??null,round_number:p.round_number,
    local_slot:p.local_slot,opponent_slot:p.opponent_slot,
    local_robot_id:'g1',opponent_robot_id:'t800',
    local_runtime_object_name:l.runtime_object_name,opponent_runtime_object_name:o.runtime_object_name,
    local_bone_signature_sha256:l.runtime_bone_signature_sha256,
    opponent_bone_signature_sha256:o.runtime_bone_signature_sha256};
}
function rejectUnsupportedActivePairing(s) {
  const proof=provenG1T800Pairing(s);
  if(proof){const error=new Error(proof.code);error.unsupported_pairing=proof;throw error;}
}
function canReadyPrivateAiSession(s) {
  const p=s?.private_ai;
  return privateAiRouteNoHuman(s) && knownBot(p) && p.phase==='Idle' &&
    p.round_active===false && p.round_inactive===true && p.client_visual_only_fighter_pair===false;
}
function canSkipObservedIntro(s) {
  return s?.scene==='Lobby' && s.lobby_screen==='Intro' &&
    s.foreground?.isolated_session_verified===true && s.foreground.execution_surface==='spark_wine_x98' &&
    s.intro_skip?.skip_allowed===true && s.intro_skip.active===true && s.intro_skip.finished===false &&
    s.intro_skip.skip_shown===true && s.intro_skip.skip_enabled===true;
}
async function exitObservedUnsupportedPairing(state,{command,getState,
    wait=ms=>new Promise(resolve=>setTimeout(resolve,ms)),now=Date.now,log=()=>{},timeoutMs=15000}) {
  const proof=provenG1T800Pairing(state);
  if(!proof)return state;
  requireValue(Number.isInteger(timeoutMs)&&timeoutMs>0&&timeoutMs<=30000,'unsupported exit timeout invalid');
  log('unsupported_private_pair_native_exit_requested',{unsupported_pairing:proof,process_restart:false});
  await command('ExitUnsupportedPrivateAiPairing');
  const deadline=now()+timeoutMs;let confirmed=false;
  while(now()<deadline) {
    await wait(100);state=await getState();
    requireValue(state.foreground?.isolated_session_verified===true &&
      state.foreground.execution_surface==='spark_wine_x98','isolated Spark proof lost during unsupported exit');
    if(state.scene==='Lobby' && state.lobby_screen==='Home') {
      log('unsupported_private_pair_home_observed',{process_restart:false});return state;
    }
    if(!confirmed && provenG1T800Pairing(state) && state.unsupported_pairing_exit?.confirmation_required===true) {
      await command('ExitUnsupportedPrivateAiPairing');confirmed=true;
      log('unsupported_private_pair_native_confirmation_requested',{process_restart:false});
    }
  }
  throw Error('unsupported private pair native Home exit timeout; client left running');
}
async function ensurePrivateArena(state,{enterPrivate,command,getState,
    wait=ms=>new Promise(resolve=>setTimeout(resolve,ms)),now=Date.now,log=()=>{},entryTimeoutMs=45000}) {
  requireValue(Number.isInteger(entryTimeoutMs)&&entryTimeoutMs>0&&entryTimeoutMs<=120000,
    'entryTimeoutMs must be a positive integer <=120000');
  rejectUnsupportedActivePairing(state);
  if(privateArena(state) && !canReadyPrivateAiSession(state))return state;
  requireValue(enterPrivate===true,'current client is not verified ready private AI arena');
  const entered=new Set();const deadline=now()+entryTimeoutMs;
  while(!privateArena(state) || canReadyPrivateAiSession(state)) {
    if(canReadyPrivateAiSession(state)) {
      // Idle AI difficulty can precede the server's new-pilot ownership reset.
      // One native ready request reveals the active opponent; policy actions
      // require active private AI scope after the fighters spawn.
      await command('ReadyPrivateAiSession');
      log('private_ai_ready_probe',{opponent_identity:'unverified_until_active_spawn'});
      const readyDeadline=now()+30000;
      while(true) {
        await wait(100);state=await getState();
        requireValue(privateAiRouteNoHuman(state),'private no-human route proof lost during ready bootstrap');
        rejectUnsupportedActivePairing(state);
        const p=state.private_ai;
        if(p.phase==='RoundActive' && p.round_active===true && p.client_visual_only_fighter_pair===true) {
          requireValue(knownBot(p),'ready bootstrap active opponent identity unavailable; policy input withheld');
          if(privateArena(state) && p.policy_active_gameplay_proven===true) {
            log('private_practice_observed',{opponent:botIdentity(p),human_opponent:false,after_native_ready:true});
            return state;
          }
        }
        requireValue(now()<readyDeadline,'private AI ready bootstrap timeout; policy input withheld');
      }
    }
    const menu=canSkipObservedIntro(state)?'SkipIntro':{Login:'ConfirmLoggedIn',Home:'NavigateFreePlay',FreePlay:'EnterSolo'}[state.lobby_screen];
    if(menu&&!entered.has(menu)){await command(menu);entered.add(menu);}
    else requireValue(state.scene==='Arena'||menu||state.lobby_screen==='Intro','no supported private-practice menu route');
    await wait(150);state=await getState();
    rejectUnsupportedActivePairing(state);
    requireValue(now()<deadline,'private-practice entry timeout');
  }
  log('private_practice_observed',{opponent:botIdentity(state.private_ai),human_opponent:false});
  return state;
}
function betweenPrivateRounds(s) {
  return privateArena(s) && ['BetweenRounds','FightOver'].includes(s.private_ai.phase) &&
    s.private_ai.round_active===false;
}
function canRequestPrivateRound(s) {
  return privateArena(s) && s.private_ai.phase==='Idle' &&
    s.private_ai.round_active===false &&
    (s.private_ai.post_fight_prompt!==true || s.private_ai.post_fight_is_winner===true);
}
function validateWorkerReady(ready, sha, featureMaskSha = null, schema = 'rek.native5.scaled_polar_xy.v1') {
  requireValue(ready?.type==='ready' && ready.checkpoint_sha256===sha &&
    ready.native_cuda===true && ready.environment_stepping===false &&
    ready.observation_schema===schema &&
    ready.precision==='bf16' && ready.selection==='sampled' &&
    ready.observations===223 && ready.actions===33 &&
    ready.hidden_size===256 && ready.num_layers===2, 'worker identity mismatch');
  if(featureMaskSha!==null) requireValue(ready.feature_mask_sha256===featureMaskSha, 'worker feature mask mismatch');
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
async function sendAndWait(endpoint,message,predicate) {
  const response=endpoint.wait(predicate);
  try {endpoint.send(message);} catch(error) {
    // A disconnected pipe can throw before the caller receives the pending
    // response. Its eventual timeout/exit must not become unhandled.
    response.catch(()=>{});throw error;
  }
  return response;
}
function measuredQpc(clock) {
  const ticks=clock?.qpc_ticks;
  requireValue((Number.isSafeInteger(ticks)&&ticks>0) ||
    (typeof ticks==='string'&&/^[1-9][0-9]*$/.test(ticks)), 'source_qpc_unavailable');
  return BigInt(ticks);
}
function validateStartupRound(source, minimumRemaining) {
  const r=source?.round;
  requireValue(r?.active===true && r.duration===120 && r.redo===false && r.result_value===0 &&
    Number.isFinite(r.time_remaining) && r.time_remaining>=minimumRemaining && r.time_remaining<=120 &&
    Array.isArray(r.clean_hits) && r.clean_hits.length===2 && r.clean_hits.every(x=>x===0),
    'startup_requires_untouched_fair_120s_round');
}
async function prewarmPolicySource({pollSource,opponent,now=Date.now,
    wait=ms=>new Promise(resolve=>setTimeout(resolve,ms)),log=()=>{}}) {
  const started=now();let previous=null,consecutive=0,samples=0;
  log('source_warmup_started',{controlled:false,worker_requests:0,encoder_requests:0,budget_ms:2000});
  while(now()-started<2000) {
    const sent=now(),source=await pollSource(Math.min(750,Math.max(1,2000-(sent-started)))),received=now();samples++;
    requireValue(received>=sent&&sent>=started,'prewarm_local_clock_regressed');
    requireValue(source?.event==='g1_policy_state' && source.schema==='rek.g1_policy_source.v1' &&
      source.stream_active===false && source.global_input_emitted===false && source.phase===1 &&
      /^[a-f0-9]{64}$/.test(source.round_identity_sha256||'') && [0,1].includes(source.local_slot),
      'prewarm_requires_uncontrolled_native_source');
    validateBotIdentity(source.opponent,opponent);validateStartupRound(source,117.5);
    const input=source.input;
    requireValue(input?.active===true && input.pending_move===false && input.pending_special===false &&
      input.pending_estop===false && input.punching===false && input.recovering===false &&
      Array.isArray(input.velocity_command_xyz) && input.velocity_command_xyz.length===3 &&
      input.velocity_command_xyz.every(x=>x===0),'prewarm_requires_neutral_pending_free_controller');
    const qpc=measuredQpc(source.clock),frequency=source.clock.qpc_frequency_hz,frame=source.clock.unity_frame;
    requireValue(Number.isSafeInteger(frequency)&&frequency>0&&Number.isSafeInteger(frame)&&frame>=0&&
      Number.isSafeInteger(source.observation_sequence)&&source.observation_sequence>0,'invalid_prewarm_clock_or_sequence');
    let intervalMs=null,fresh=false;
    if(previous) {
      requireValue(source.round_identity_sha256===previous.round && source.local_slot===previous.slot &&
        frequency===previous.frequency && qpc>previous.qpc && source.observation_sequence>previous.sequence &&
        frame>=previous.frame,'prewarm_source_binding_changed_or_clock_regressed');
      intervalMs=Number(qpc-previous.qpc)/frequency*1000;
      fresh=frame>previous.frame && intervalMs<=200 && received-sent<=200;
    }
    consecutive=fresh?consecutive+1:0;
    previous={round:source.round_identity_sha256,slot:source.local_slot,frequency,qpc,frame,
      sequence:source.observation_sequence};
    log('source_warmup_sample',{controlled:false,samples,sequence:previous.sequence,unity_frame:frame,
      request_ms:received-sent,source_interval_ms:intervalMs,consecutive_fresh_intervals:consecutive,
      time_remaining:source.round.time_remaining});
    requireValue(received-started<2000,'source_warmup_budget_exceeded');
    if(consecutive>=2) {
      const receipt={round_identity_sha256:previous.round,local_slot:previous.slot,
        qpc_ticks:previous.qpc.toString(),qpc_frequency_hz:frequency,unity_frame:frame,
        observation_sequence:previous.sequence,samples,elapsed_ms:received-started,
        time_remaining:source.round.time_remaining};
      log('source_warmup_ready',{controlled:false,worker_requests:0,encoder_requests:0,...receipt});return receipt;
    }
    await wait(20);
  }
  throw Error('source_warmup_budget_exceeded');
}
function validateControlledStartup(source,readiness) {
  requireValue(source?.stream_active===true && source.round_identity_sha256===readiness.round_identity_sha256 &&
    source.local_slot===readiness.local_slot && source.clock?.qpc_frequency_hz===readiness.qpc_frequency_hz &&
    measuredQpc(source.clock)>BigInt(readiness.qpc_ticks) && source.clock.unity_frame>readiness.unity_frame &&
    source.observation_sequence>readiness.observation_sequence,'controlled_start_not_bound_to_warmed_round');
  validateStartupRound(source,117);
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
  const signalsSent=[];
  const closed = new Promise(resolve => child.once('close', (code,signal) =>
    resolve({name,code,signal,close_observed:true,signals_sent:signalsSent.slice()})));
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
  return {child,bus,send,wait,closed,close(){child.stdin.end(); requests.end();},
    terminate(signal){signalsSent.push(signal);child.kill(signal);}};
}
async function run(configPath) {
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  requireValue(config.projection === 'client_pose_projection_v1', 'explicit projection required');
  requireValue(config.observation_schema === 'rek.native5.scaled_polar_xy.balance8_v1', 'explicit balance8 schema required');
  requireValue(Number.isFinite(config.max_seconds) && config.max_seconds > 0 && config.max_seconds <= 600, 'max_seconds must be 0..600');
  requireValue(/^[a-f0-9]{64}$/.test(config.checkpoint_sha256), 'checkpoint SHA required');
  requireValue(/^[a-f0-9]{64}$/.test(config.feature_mask_sha256), 'feature mask SHA required');
  // Live attack gate: attack categories 16..32 not listed in
  // allowed_attacks are masked out of every worker request; optional root-distance gate.
  const gate=config.live_attack_gate||null;
  requireValue(gate&&Array.isArray(gate.allowed_attacks)&&gate.allowed_attacks.every(a=>Number.isInteger(a)&&a>=16&&a<33),'live_attack_gate.allowed_attacks required');
  const allowedAttacks=new Set(gate.allowed_attacks);
  const attackRange=gate.range_m===undefined||gate.range_m===null?null:Number(gate.range_m);
  requireValue(attackRange===null||(Number.isFinite(attackRange)&&attackRange>0),'live_attack_gate.range_m must be positive');
  const forceRange=gate.force_attack_range_m===undefined||gate.force_attack_range_m===null?null:Number(gate.force_attack_range_m);
  requireValue(forceRange===null||(Number.isFinite(forceRange)&&forceRange>0),'live_attack_gate.force_attack_range_m must be positive');
  // Kick cooldown: mask categories in cooldown_actions for cooldown_s after ANY attack start (balance recovery).
  const cooldownS=gate.cooldown_s===undefined||gate.cooldown_s===null?null:Number(gate.cooldown_s);
  requireValue(cooldownS===null||(Number.isFinite(cooldownS)&&cooldownS>0),'live_attack_gate.cooldown_s must be positive');
  const cooldownActions=new Set(Array.isArray(gate.cooldown_actions)?gate.cooldown_actions:[16]);
  let lastAttackStartMs=-Infinity,cooldownMasked=0;
  let gatedRequests=0,gatedByRange=0,forcedAttacks=0,lastSource=null;
  fs.mkdirSync(config.out, {recursive:false});
  fs.writeFileSync(path.join(config.out,'run-config.json'), JSON.stringify(config,null,2)+'\n', {flag:'wx'});
  const events=fs.createWriteStream(path.join(config.out,'orchestrator.jsonl'), {flags:'wx'});
  const log=(event, detail={}) => {const value={utc:new Date().toISOString(),event,...detail};events.write(JSON.stringify(value)+'\n');console.log(JSON.stringify(value));};
  const endpoints=[]; let relay,encoder,worker, leased=false, streaming=false, stopping=false;
  let nextId=0, sourceCount=0, skipped=0, predictions=0, applied=0, rejected=0, unmatchedAcks=0;
  let firstRound=null,lastRound=null,lastSourceAt=0,stopReason='not_started',opponent=null,unsupportedPairing=null;
  let startupReadiness=null,controlledStartupValidated=false;
  const actions=Array(33).fill(0), reasons={}, droppedSources={}, pacer=new LiveActionPacer();
  let doneResolve; const done=new Promise(resolve => doneResolve=resolve);
  let timer, watchdog;
  async function request(type, fields={}, event='ack') {
    const request_id=`live-${++nextId}`;
    return sendAndWait(relay,{type,request_id,...fields},x => x.event===event && x.request_id===request_id);
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
    const [ready,,manifest]=await Promise.all([
      worker.wait(x=>x.type==='ready',60000), relay.wait(x=>x.event==='hello',30000),
      encoder.wait(x=>x.event==='projection_manifest',30000)
    ]);
    validateWorkerReady(ready, config.checkpoint_sha256, config.feature_mask_sha256, config.observation_schema);
    requireValue(manifest.observation_schema===config.observation_schema && manifest.projection===config.projection &&
      manifest.candidate_physics_stepped===false,'encoder_manifest_identity_mismatch');
    log('inference_ready',{checkpoint_sha256:ready.checkpoint_sha256,feature_mask_sha256:ready.feature_mask_sha256,device:ready.device,precision:ready.precision,selection:ready.selection,projection:config.projection});
    let state=await request('get_state',{},'state');
    await command('AcquireExclusiveControl'); leased=true;
    state=await exitObservedUnsupportedPairing(state,{command,getState:()=>request('get_state',{},'state'),log});
    if(betweenPrivateRounds(state)) {
      log('waiting_for_automatic_round_transition',{phase:state.private_ai.phase});
      const deadline=Date.now()+30000;
      do {
        await new Promise(r=>setTimeout(r,100));
        state=await request('get_state',{},'state');
        requireValue(privateArena(state),'private arena proof lost between rounds');
        requireValue(Date.now()<deadline,'automatic round transition timeout');
      } while(betweenPrivateRounds(state));
    }
    if(canExitLostPrivateSession(state)) {
      requireValue(config.enter_private===true, 'lost private session requires explicit private-entry recovery');
      await command('ExitLostG1PolicySession');
      const deadline=Date.now()+15000;
      do {
        await new Promise(r=>setTimeout(r,100));
        state=await request('get_state',{},'state');
        requireValue(Date.now()<deadline,'lost private session exit timeout');
      } while(privateArena(state));
    }
    state=await ensurePrivateArena(state,{enterPrivate:config.enter_private,command,
      getState:()=>request('get_state',{},'state'),log});
    if(!state.private_ai.policy_active_gameplay_proven || state.private_ai.round_active!==true) {
      const deadline=Date.now()+30000;
      let startRequested=false;
      while(Date.now()<deadline) {
        if(canRequestPrivateRound(state)&&!startRequested){await command('StartG1PolicyRound');startRequested=true;}
        state=await request('get_state',{},'state');
        requireValue(privateArena(state),'private arena proof lost');
        rejectUnsupportedActivePairing(state);
        if(state.private_ai.policy_active_gameplay_proven && state.private_ai.round_active)break;
        await new Promise(r=>setTimeout(r,100));
        requireValue(Date.now()<deadline,'active round timeout');
      }
    }
    requireValue(privateArena(state) && state.private_ai.policy_active_gameplay_proven===true &&
      state.private_ai.round_active===true,'active private AI scope required before policy stream');
    opponent=botIdentity(state.private_ai);
    log('active_private_ai_opponent',opponent);
    // This event also starts the existing passive recorder. Warm the measured
    // source path before the bridge action watchdog starts. No warmup source is
    // sent to the encoder or worker, so replay/RNN/RNG histories remain untouched.
    startupReadiness=await prewarmPolicySource({opponent,log,pollSource:async(timeoutMs)=>{
      const request_id=`warm-${++nextId}`;
      const response=relay.wait(x=>x.request_id===request_id &&
        (x.event==='g1_policy_state'||x.event==='error'),timeoutMs);
      try{relay.send({type:'get_policy_state',request_id});}catch(error){response.catch(()=>{});throw error;}
      const source=await response;
      requireValue(source.event==='g1_policy_state',`source_warmup_unavailable:${source.reason||'unknown'}`);
      return source;
    }});
    requireValue(!stopping,'startup_aborted_before_control');
    relay.bus.on('message',guardedCallback(source => {
      if(source.event==='g1_policy_end') {log('stream_end',source); finish(`stream_end:${source.reason}`);return;}
      if(source.event==='g1_policy_action') {
        if(!pacer.acknowledge(source)){unmatchedAcks++;log('unmatched_action_ack',{request_id:source.request_id,seq:source.observation_sequence});return;}
        source.applied ? applied++ : rejected++;
        if(source.applied&&source.reason==='accepted_locally_and_armed'&&Number.isInteger(source.action)&&source.action>=16)lastAttackStartMs=Date.now();
        reasons[source.reason]=(reasons[source.reason]||0)+1;
        return;
      }
      if(source.event!=='g1_policy_state'||stopping)return;
      validateBotIdentity(source.opponent,opponent);
      if(streaming&&!controlledStartupValidated){
        validateControlledStartup(source,startupReadiness);controlledStartupValidated=true;
        log('controlled_source_started',{controlled:true,sequence:source.observation_sequence,
          round_identity_sha256:source.round_identity_sha256,time_remaining:source.round?.time_remaining});
      }
      sourceCount++;lastSourceAt=Date.now();
      if(source.round){firstRound??=source.round;lastRound=source.round;}
      if(!streaming)return;
      const dropped=pacer.offer(source,Date.now());
      if(dropped){skipped++;droppedSources[dropped]=(droppedSources[dropped]||0)+1;return;}
      lastSource=source;encoder.send(source);
    },error=>finish(`relay_callback:${error.message}`)));
    encoder.bus.on('message',guardedCallback(encoded=> {
      if(stopping||encoded.event!=='policy_observation')return;
      if(encoded.ready!==true) {log('observation_unavailable',{reason:encoded.reason,unavailable:encoded.unavailable});pacer.abandon();return;}
      const r=encoded.worker_request;
      requireValue(r?.observation_schema===config.observation_schema, 'encoder_observation_schema_mismatch');
      const pending=pacer.pending;
      if(!pending || r?.seq!==pending.sequence || r?.round_id!==pending.round) {finish('encoder_source_identity_mismatch');return;}
      if(Array.isArray(r.mask)&&r.mask.length===33){
        let dist=null;
        if(attackRange!==null&&lastSource&&Array.isArray(lastSource.fighters)&&lastSource.fighters.length>=2){
          const a=lastSource.fighters[0].root_position_xyz,b=lastSource.fighters[1].root_position_xyz;
          if(a&&b)dist=Math.hypot(a[0]-b[0],a[2]-b[2]);
        }
        if(dist===null&&forceRange!==null&&lastSource&&Array.isArray(lastSource.fighters)&&lastSource.fighters.length>=2){
          const a=lastSource.fighters[0].root_position_xyz,b=lastSource.fighters[1].root_position_xyz;
          if(a&&b)dist=Math.hypot(a[0]-b[0],a[2]-b[2]);
        }
        const outOfRange=attackRange!==null&&(dist===null||dist>attackRange);
        let changed=false;
        for(let k=16;k<33;k++){if(r.mask[k]&&(!allowedAttacks.has(k)||outOfRange)){r.mask[k]=0;changed=true;}}
        if(changed){gatedRequests++;if(outOfRange)gatedByRange++;}
        if(cooldownS!==null&&Date.now()-lastAttackStartMs<cooldownS*1000){
          let cm=false;for(const k of cooldownActions){if(r.mask[k]){r.mask[k]=0;cm=true;}}
          if(cm)cooldownMasked++;
        }
        // Force an allowed attack when eligible and inside striking range.
        if(forceRange!==null&&dist!==null&&dist<=forceRange){
          const eligible=[];for(let k=16;k<33;k++)if(r.mask[k]&&allowedAttacks.has(k))eligible.push(k);
          if(eligible.length){for(let k=0;k<16;k++)r.mask[k]=0;forcedAttacks++;}
        }
      }
      worker.send(r);
    },error=>finish(`encoder_callback:${error.message}`)));
    worker.bus.on('message',guardedCallback(prediction=> {
      if(stopping)return;
      if(prediction.type==='terminal'){pacer.abandon();finish('source_round_terminal');return;}
      if(prediction.type==='error'||prediction.type==='fatal'){finish(`worker_error:${prediction.code}`);return;}
      if(prediction.type!=='action')return;
      const source=pacer.pending;
      validateWorkerAction(prediction, source, config.checkpoint_sha256);
      requireValue(prediction.feature_mask_sha256===config.feature_mask_sha256, 'prediction_feature_mask_mismatch');
      requireValue(prediction.observation_schema===config.observation_schema, 'prediction_observation_schema_mismatch');
      predictions++;actions[prediction.action]++;
      if(Date.now()-source.received>=200){log('stale_prediction_discarded',{seq:prediction.seq,local_latency_ms:Date.now()-source.received});pacer.abandon();return;}
      const request_id=`action-${++nextId}`;
      pacer.sent(request_id,prediction.action,Date.now());
      relay.send({type:'policy_action',request_id,round_identity_sha256:source.round,observation_sequence:source.sequence,action:prediction.action});
    },error=>finish(`worker_callback:${error.message}`)));
    streaming=true;
    await command('StartG1PolicyStreamAnyAi');
    log('live_policy_started');
    timer=setTimeout(()=>finish('requested_duration_complete'),config.max_seconds*1000);
    watchdog=setInterval(()=>{
      if(lastSourceAt && Date.now()-lastSourceAt>2000)finish('source_stream_missing');
      if(pacer.pending?.sent!==null && pacer.pending?.sent!==undefined && Date.now()-pacer.pending.sent>2000)finish('action_ack_missing');
    },250);
    process.once('SIGINT',()=>finish('operator_interrupt'));
    process.once('SIGTERM',()=>finish('operator_termination'));
    await done;
  } catch(error) {unsupportedPairing=error.unsupported_pairing??null;finish(error.message);log('error',{message:error.message,...(unsupportedPairing?{unsupported_pairing:unsupportedPairing}:{})});}
  finally {
    clearTimeout(timer);clearInterval(watchdog);streaming=false;
    let streamStopped=false,leaseReleased=false;
    if(leased){try{await command('StopG1PolicyStream');streamStopped=true;}catch(e){log('stop_error',{message:e.message});}try{await command('ReleaseExclusiveControl');leaseReleased=true;}catch(e){log('release_error',{message:e.message});}}
    const summary={stop_reason:stopReason,source_count:sourceCount,skipped_sources:skipped,dropped_sources:droppedSources,
      unmatched_action_acks:unmatchedAcks,action_inflight_at_stop:pacer.pending?.requestId!==null&&pacer.pending?.requestId!==undefined,
      last_ack_qpc_ticks:pacer.lastAckQpc?.toString()??null,predictions,applied,rejected,actions,reasons,opponent,initial_round:firstRound,final_round:lastRound,projection:config.projection,checkpoint_sha256:config.checkpoint_sha256,authentic_client:true,global_input_emitted:false,
      feature_mask_sha256:config.feature_mask_sha256,
      ...(unsupportedPairing?{unsupported_pairing:unsupportedPairing}:{}),
      observation_schema:config.observation_schema,
      startup_readiness:startupReadiness,controlled_startup_validated:controlledStartupValidated,
      live_attack_gate:{allowed_attacks:[...allowedAttacks].sort((a,b)=>a-b),range_m:attackRange,force_attack_range_m:forceRange,cooldown_s:cooldownS,cooldown_actions:[...cooldownActions],cooldown_masked:cooldownMasked,gated_requests:gatedRequests,gated_by_range:gatedByRange,forced_attacks:forcedAttacks}};
    fs.writeFileSync(path.join(config.out,'summary.json'),JSON.stringify(summary,null,2)+'\n',{flag:'wx'});log('summary',summary);
    for(const e of endpoints)e.close();
    const cleanupTimer=setTimeout(()=>{for(const e of endpoints)if(e.child.exitCode===null)e.terminate('SIGTERM');},2000);
    cleanupTimer.unref();
    const closed=await Promise.all(endpoints.map(e=>e.closed));
    clearTimeout(cleanupTimer);
    log('policy_endpoints_closed',{schema:'rek.policy_transport_closed.v1',
      checkpoint_sha256:config.checkpoint_sha256,trial_output:config.out,
      summary_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(config.out,'summary.json'))).digest('hex'),
      stream_stopped:streamStopped,lease_released:leaseReleased,endpoints:closed});
    events.end();
    process.exitCode=trialExitCode(summary);
  }
}
if(require.main===module) {requireValue(process.argv.length===3,'usage: node live_transfer_run_masked.cjs config.json');run(process.argv[2]).catch(e=>{console.error(e.message);process.exitCode=2;});}
module.exports={privateArena,botIdentity,validateBotIdentity,trialExitCode,canExitLostPrivateSession,canReadyPrivateAiSession,canSkipObservedIntro,exitObservedUnsupportedPairing,ensurePrivateArena,betweenPrivateRounds,canRequestPrivateRound,validateWorkerReady,validateWorkerAction,guardedCallback,sendAndWait,childEndpoint,LiveActionPacer,provenG1T800Pairing,rejectUnsupportedActivePairing,prewarmPolicySource,validateControlledStartup};
