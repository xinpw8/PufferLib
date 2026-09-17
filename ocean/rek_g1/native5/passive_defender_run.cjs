#!/usr/bin/env node
'use strict';
// One authentic private-AI round. Action 1 releases held controls; physics remains live.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const {privateArena,botIdentity,validateBotIdentity,canExitLostPrivateSession,ensurePrivateArena,
  canRequestPrivateRound,guardedCallback,sendAndWait,childEndpoint,LiveActionPacer}=require('./live_transfer_run.cjs');
const ISOLATION='wine_get_version=11.13;display=:98;prefix=/opt/codexrook/wineprefix;marker=spark-x98';
const requireValue=(ok,message)=>{if(!ok)throw Error(message);};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

function validateConfig(config,platform=process.platform,hostname=os.hostname()) {
  requireValue(platform==='linux' && hostname==='spark-4ae3','isolated Spark host required; Windows execution forbidden');
  requireValue(['safe_start','active_attach'].includes(config.mode),'explicit mode safe_start or active_attach required');
  requireValue(Number.isFinite(config.max_seconds)&&config.max_seconds>0&&config.max_seconds<=180,'max_seconds must be >0 and <=180');
  requireValue(typeof config.out==='string'&&path.isAbsolute(config.out),'absolute fresh output directory required');
  requireValue(Array.isArray(config.relay)&&config.relay.length>0&&config.relay.every(x=>typeof x==='string'),'existing relay command required');
}
function assertIsolation(state) {
  requireValue(state?.foreground?.isolated_session_verified===true &&
    state.foreground.isolated_session_proof===ISOLATION,'exact isolated Spark state required');
}
function assertUnowned(state) {
  assertIsolation(state);
  requireValue(state.control?.lease_held===false,'preexisting or unknown exclusive lease');
  requireValue(state.control?.g1_policy_stream_running===false,'preexisting or unknown policy stream');
  for(const [key,value] of Object.entries(state.control))
    requireValue(!key.endsWith('_running')||value!==true,`preexisting control mode:${key}`);
}
function assertActiveScope(state) {
  assertIsolation(state);
  requireValue(privateArena(state)&&state.private_ai.policy_active_gameplay_proven===true&&
    state.private_ai.round_active===true&&state.private_ai.client_visual_only_fighter_pair===true&&
    state.measured_pairing?.exact_g1_vs_g1===true,'active isolated private AI G1 pair required');
}
function observedNeutral(source) {
  const input=source?.input,v=input?.velocity_command_xyz;
  return input?.desired_action===1&&Array.isArray(v)&&v.length===3&&v.every(x=>Number.isFinite(x)&&x===0)&&
    input.pending_move===false&&input.pending_special===false&&input.pending_estop===false&&
    input.requested_move_index===null&&input.move_request_pending_transport===false;
}
function validateSource(source,opponent,pinnedRound,pinnedSlot) {
  requireValue(source.protocol==='rek.ui_bridge.v1'&&source.schema==='rek.g1_policy_source.v1'&&
    source.global_input_emitted===false,'unexpected native source contract');
  validateBotIdentity(source.opponent,opponent);
  requireValue(/^[a-f0-9]{64}$/.test(source.round_identity_sha256)&&
    (pinnedRound===null||source.round_identity_sha256===pinnedRound),'source_round_identity_changed');
  requireValue(source.local_slot===pinnedSlot&&Array.isArray(source.fighters)&&source.fighters.length===2,
    'source_fighter_scope_changed');
  // Exact G1 bone signatures are checked by the native publisher on every frame.
  requireValue(source.fighters.every(f=>Array.isArray(f.bone_names)&&f.bone_names.length===30),
    'source_g1_bones_unavailable');
  requireValue(Number.isSafeInteger(source.observation_sequence)&&source.observation_sequence>0,'invalid source sequence');
  requireValue(source.round&&typeof source.round.active==='boolean','source_round_unavailable');
}
function passiveExitCode(summary) {
  return summary.round_completed&&summary.neutral_command_verified&&summary.cleanup.verified ? 0 : 2;
}

async function run(configPath,dependencies={}) {
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  // Dependency overrides are for offline fixtures only; the CLI has no host bypass.
  validateConfig(config,dependencies.platform??process.platform,dependencies.hostname??os.hostname());
  fs.mkdirSync(config.out,{recursive:false,mode:0o700});
  const write=(name,value)=>fs.writeFileSync(path.join(config.out,name),JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});
  write('run-config.json',config);
  write('provenance.json',{utc:new Date().toISOString(),host:os.hostname(),platform:process.platform,
    command:[process.execPath,__filename,path.resolve(configPath)],relay_command:config.relay,
    config_sha256:hash(configPath),runner_sha256:hash(__filename),helpers_sha256:hash(require.resolve('./live_transfer_run.cjs')),
    control:'only action 1, neutral held controls',policy_used:false,encoder_used:false,checkpoint_used:false,
    authority:'native client replicated observations; server acceptance unknown',global_input_emitted:false});
  const events=fs.createWriteStream(path.join(config.out,'orchestrator.jsonl'),{flags:'wx'});
  const sources=fs.createWriteStream(path.join(config.out,'g1_policy_state.jsonl'),{flags:'wx'});
  const log=(event,detail={})=>{const value={utc:new Date().toISOString(),event,...detail};events.write(JSON.stringify(value)+'\n');
    if(!dependencies.quiet)console.log(JSON.stringify(value));};
  let relay,leased=false,acquireAttempted=false,streaming=false,stopping=false,nextId=0,timer,watchdog;
  let stopReason='not_started',opponent=null,pinnedRound=null,pinnedSlot=null,firstRound=null,lastRound=null;
  let sourceCount=0,validatedSources=0,neutralSources=0,requested=0,applied=0,rejected=0,unmatched=0;
  let lastSourceAt=0,streamStartedAt=0,firstAppliedQpc=null,postAckSources=0,postAckNeutral=0,streamEnd=null;
  let initialState=null,finalState=null,relayExit=null,stoppedAt=0;
  const dropped={},reasons={},pacer=new LiveActionPacer(),cleanup={stop_accepted:false,release_accepted:false,verified:false};
  let resolveDone;const done=new Promise(resolve=>resolveDone=resolve);
  function finish(reason){if(!stopping){stopping=true;stopReason=reason;stoppedAt=Date.now();resolveDone();}}
  async function request(type,fields={},event='ack') {
    const request_id=`passive-${++nextId}`;
    return sendAndWait(relay,{type,request_id,...fields},x=>x.event===event&&x.request_id===request_id);
  }
  async function command(name) {
    requireValue(!stopping||['StopG1PolicyStream','ReleaseExclusiveControl'].includes(name),'run already stopping');
    const ack=await request('command',{command:name});
    log('command_result',{command:name,status:ack.status,reason:ack.reason});
    requireValue(ack.status==='accepted',`${name}:${ack.reason}`);return ack;
  }
  async function getState(){const state=await request('get_state',{},'state');assertIsolation(state);return state;}
  const signalInt=()=>finish('operator_interrupt'),signalTerm=()=>finish('operator_termination');
  process.once('SIGINT',signalInt);process.once('SIGTERM',signalTerm);
  try {
    relay=(dependencies.endpointFactory??childEndpoint)('relay',config.relay,config.out);
    relay.bus.on('failure',e=>finish(`relay_failure:${e.message}`));
    relay.bus.on('exit',value=>{relayExit=value;log('relay_exit',value);finish('relay_exit');});
    relay.bus.on('invalid',()=>finish('relay_invalid_json'));
    relay.bus.on('message',guardedCallback(source=>{
      // Preserve every source, including inflight, terminal, and rejected-scope frames.
      if(source.event==='g1_policy_state'){sources.write(JSON.stringify(source)+'\n');sourceCount++;}
      if(source.event==='g1_policy_end'){streamEnd=source;log('stream_end',source);finish(`stream_end:${source.reason}`);return;}
      if(source.event==='g1_policy_action') {
        if(!pacer.acknowledge(source)){unmatched++;return;}
        requireValue(source.action===1&&source.global_input_emitted===false,'nonneutral action acknowledgment');
        if(source.applied===true){applied++;firstAppliedQpc??=pacer.lastAckQpc;}else rejected++;
        reasons[source.reason]=(reasons[source.reason]||0)+1;return;
      }
      if(source.event==='error'||source.type==='fatal'){finish(`relay_error:${source.reason??source.code}`);return;}
      if(source.event!=='g1_policy_state'||stopping||!streaming)return;
      validateSource(source,opponent,pinnedRound,pinnedSlot);
      pinnedRound??=source.round_identity_sha256;validatedSources++;lastSourceAt=Date.now();
      firstRound??=source.round;lastRound=source.round;
      const neutral=observedNeutral(source);if(neutral)neutralSources++;
      if(firstAppliedQpc!==null&&BigInt(source.clock.qpc_ticks)>firstAppliedQpc){postAckSources++;if(neutral)postAckNeutral++;}
      if(source.round.active===false){finish('source_round_terminal');return;}
      requireValue(source.stream_active===true,'source_stream_inactive');
      requireValue(source.input?.desired_action===1,'owned held state is not neutral');
      requireValue(source.action_mask?.[1]===true,'native neutral mask unavailable');
      const drop=pacer.offer(source,Date.now());
      if(drop){dropped[drop]=(dropped[drop]||0)+1;return;}
      const request_id=`passive-action-${++nextId}`;
      pacer.sent(request_id,1,Date.now());requested++;
      relay.send({type:'policy_action',request_id,round_identity_sha256:pinnedRound,
        observation_sequence:source.observation_sequence,action:1});
    },e=>finish(`relay_callback:${e.message}`)));
    await relay.wait(x=>x.event==='hello',30000);
    initialState=await getState();assertUnowned(initialState);write('initial-state.json',initialState);
    if(config.mode==='active_attach')assertActiveScope(initialState);
    else requireValue(initialState.private_ai?.round_active!==true,'safe_start refuses a preexisting active round; choose active_attach explicitly');
    acquireAttempted=true;await command('AcquireExclusiveControl');leased=true;
    let state=initialState;
    if(config.mode==='safe_start') {
      if(canExitLostPrivateSession(state)) {
        requireValue(config.enter_private===true,'explicit private-entry recovery required');
        await command('ExitLostG1PolicySession');const deadline=Date.now()+15000;
        do{await delay(100);state=await getState();requireValue(Date.now()<deadline,'lost private exit timeout');}while(privateArena(state));
      }
      state=await ensurePrivateArena(state,{enterPrivate:config.enter_private,command,getState,log,entryTimeoutMs:120000});
      if(state.private_ai.round_active!==true) {
        requireValue(canRequestPrivateRound(state),'safe_start requires native Idle; no automatic round repeat');
        await command('StartG1PolicyRound');const deadline=Date.now()+30000;
        do {
          state=await getState();requireValue(privateArena(state),'private arena proof lost while starting');
          if(state.private_ai.policy_active_gameplay_proven===true&&state.private_ai.round_active===true)break;
          requireValue(Date.now()<deadline,'active round timeout');await delay(100);
        }while(!stopping);
      }
    }
    assertActiveScope(state);write('active-state.json',state);
    opponent=botIdentity(state.private_ai);pinnedSlot=state.private_ai.local_slot;
    requireValue(pinnedSlot===0||pinnedSlot===1,'local slot unavailable');
    log('active_private_ai_opponent',opponent);
    log('passive_defender_scope',{mode:config.mode,opponent,round_number:state.private_ai.round_number,
      whole_round_from_first_tick:false,neutral_control_does_not_freeze_physics:true});
    streaming=true;streamStartedAt=lastSourceAt=Date.now();
    // Timer starts before the command acknowledgment so stalled startup cannot extend collection.
    timer=setTimeout(()=>finish('duration_cap_incomplete'),config.max_seconds*1000);
    watchdog=setInterval(()=>{
      if(Date.now()-lastSourceAt>2000)finish('source_stream_missing');
      if(pacer.pending?.sent!=null&&Date.now()-pacer.pending.sent>2000)finish('action_ack_missing');
    },250);
    await command('StartG1PolicyStreamAnyAi');log('passive_defender_started',{action:1});
    await done;
  }catch(error){finish(error.message);log('error',{message:error.message});}
  finally {
    clearTimeout(timer);clearInterval(watchdog);streaming=false;stopping=true;
    // Attempt cleanup even if acquisition was sent but its acknowledgment was lost.
    if(acquireAttempted) {
      try{await command('StopG1PolicyStream');cleanup.stop_accepted=true;}catch(e){log('stop_error',{message:e.message});}
      try{await command('ReleaseExclusiveControl');cleanup.release_accepted=true;}catch(e){log('release_error',{message:e.message});}
      try{finalState=await getState();write('final-state.json',finalState);
        cleanup.verified=cleanup.stop_accepted&&cleanup.release_accepted&&
          finalState.control?.lease_held===false&&finalState.control?.g1_policy_stream_running===false;
      }catch(e){log('cleanup_readback_error',{message:e.message});}
    }
    if(relay) {
      try{relay.close();}catch(e){log('relay_close_error',{message:e.message});}
      if(relay.child.exitCode===null)await Promise.race([new Promise(resolve=>relay.child.once('close',resolve)),delay(2000)]);
      if(relay.child.exitCode===null){relay.child.kill('SIGTERM');log('relay_termination_requested');
        await Promise.race([new Promise(resolve=>relay.child.once('close',resolve)),delay(1000)]);}
    }
    const terminal=lastRound?.active===false&&Number.isInteger(lastRound.result_value)&&lastRound.result_value>0;
    const roundCompleted=terminal&&['source_round_terminal','stream_end:active_round_not_observed'].includes(stopReason);
    const summary={mode:config.mode,stop_reason:stopReason,round_completed:roundCompleted,outcome:roundCompleted?'completed_round':'incomplete',
      collection_seconds:streamStartedAt?(stoppedAt-streamStartedAt)/1000:0,max_seconds:config.max_seconds,
      source_count:sourceCount,validated_source_count:validatedSources,dropped_sources:dropped,requested,applied,rejected,
      unmatched_action_acks:unmatched,action_inflight_at_stop:pacer.pending!==null,only_requested_action:1,
      neutral_source_count:neutralSources,post_ack_source_count:postAckSources,post_ack_neutral_source_count:postAckNeutral,
      neutral_command_verified:applied>0&&postAckSources>0&&postAckNeutral===postAckSources,
      last_ack_qpc_ticks:pacer.lastAckQpc?.toString()??null,reasons,opponent,local_slot:pinnedSlot,
      round_identity_sha256:pinnedRound,initial_round:firstRound,final_round:lastRound,stream_end:streamEnd,cleanup,relay_exit:relayExit,
      attach_status:config.mode==='active_attach'?'attached_to_preexisting_active_round':'native_start_requested_or_ready_bootstrap',
      whole_round_from_first_tick:false,policy_used:false,encoder_used:false,checkpoint_used:false,
      neutral_control_does_not_freeze_physics:true,server_acceptance:'unknown',global_input_emitted:false};
    summary.exit_code=passiveExitCode(summary);write('summary.json',summary);log('summary',summary);
    write('exit.json',{code:summary.exit_code,stop_reason:stopReason,relay_exit:relayExit,cleanup_verified:cleanup.verified});
    process.off('SIGINT',signalInt);process.off('SIGTERM',signalTerm);
    await Promise.all([new Promise(resolve=>events.end(resolve)),new Promise(resolve=>sources.end(resolve))]);
    return summary;
  }
}
if(require.main===module) {
  if(process.argv.length!==3){console.error('usage: node passive_defender_run.cjs config.json');process.exitCode=2;}
  else run(process.argv[2]).then(summary=>{process.exitCode=summary.exit_code;}).catch(error=>{console.error(error.message);process.exitCode=2;});
}
module.exports={run,validateConfig,assertUnowned,assertActiveScope,observedNeutral,validateSource,passiveExitCode,ISOLATION};
