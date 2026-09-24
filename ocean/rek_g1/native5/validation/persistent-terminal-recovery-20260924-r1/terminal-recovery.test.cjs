'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const driver=require('./live_transfer_run_masked.cjs'),old=require('./baseline/live_transfer_run_masked.cjs');
// Entire fixture is synthetic. No captured packet, private source or native-execution receipt is embedded.
const body=Buffer.alloc(33);body[0]=4;body[1]=3;body.writeUInt16LE(19,8);body.writeUInt16LE(9,10);
body[13]=1;body[15]=2;body[16]=1;body[17]=1;body[20]=1;body[31]=1;
const fixture={synthetic:true,passive_policy_readback_was_issued:false,
 initial_native:{round:{number:3,duration:120,time_remaining:119.9,active:true,redo:false,clean_hits:[0,0]}},
 last_policy_source:{round_identity_sha256:'b'.repeat(64),observation_sequence:100,local_slot:0,
  opponent:{client_ai_difficulty:0,sparring_bot_number:1},
  clock:{qpc_ticks:'10000000',qpc_frequency_hz:10000000,unity_frame:100},
  round:{number:3,duration:120,time_remaining:0.07,active:true,redo:false},
  fight:{current_round:3,rounds_won:[1,1],result:'InProgress',result_value:0,winner_index:-1}},
 policy_stream_end:{reason:'policy_state_unavailable:InvalidDataException',
  clock:{qpc_ticks:'11000000',qpc_frequency_hz:10000000,unity_frame:101}},
 recorded_error:{reason:'policy_state_unavailable:policy_opponent_identity_changed'},
 native_capture_end:{stopwatch_timestamp_ticks:'10900000'},
 native_terminal_packet:{wire_body_base64:body.toString('base64'),wire_body_sha256:crypto.createHash('sha256').update(body).digest('hex')},
 native_terminal_postfix:{
  round:{number:3,duration:120,time_remaining:0,active:false,redo:false,clean_hits:[19,9],falls:[0,0],
    result:'WonByPoints',result_value:1,winner_index:0,knockout:false},
  fight:{current_round:3,rounds_won:[2,1],result:'WonByRounds',result_value:1,winner_index:0}},
 recorded_post_stop_state:{private_ai:{policy_proven:true,local_slot:0,opponent_slot:1,network_client_only:true,
  context_is_solo:true,solo_route_proven:true,opponent_is_ai:true,opponent_slot_is_ai:true,human_in_opponent_slot:false,
  opponent_slot_client_known:true,opponent_slot_has_client:false,opponent_human_bit_set:false,
  client_ai_difficulty:1,sparring_bot_number:2,fight_epoch:3,phase:'FightOver',round_active:false,round_number:3,round_inactive:true}}};
function example(){
 // All fields below are synthetic expectations, not native execution evidence.
 const last=structuredClone(fixture.last_policy_source),end=structuredClone(fixture.policy_stream_end);
 end.round_identity_sha256=last.round_identity_sha256;
 const state={scene:'Arena',foreground:{isolated_session_verified:true,execution_surface:'spark_wine_x98'},
 control:{lease_held:true,g1_policy_stream_running:false},private_ai:structuredClone(fixture.recorded_post_stop_state.private_ai)};
 const qpc=BigInt(end.clock.qpc_ticks)+100000n,receipt=BigInt(fixture.native_capture_end.stopwatch_timestamp_ticks);
 const candidate={event:'g1_policy_state',schema:'rek.g1_policy_source.v1',stream_active:false,global_input_emitted:false,
 phase:4,round_identity_sha256:last.round_identity_sha256,observation_sequence:last.observation_sequence+1,local_slot:last.local_slot,
 opponent:{client_ai_difficulty:1,sparring_bot_number:2,opponent_is_ai:true,human_in_opponent_slot:false},
 clock:{...end.clock,qpc_ticks:qpc.toString()},round:structuredClone(fixture.native_terminal_postfix.round),
 fight:structuredClone(fixture.native_terminal_postfix.fight),
 referee:{schema:'rek.g1_received_referee.v1',available:true,reason:'received_snapshot_applied_and_bound',observation_hooks_verified:true,
 receipt_age_seconds:Number(qpc-receipt)/10000000,receipt_qpc_ticks:receipt.toString(),receipt_qpc_frequency_hz:10000000,
 packet_phase:4,packet_round_number:3,packet_round_active:false,packet_round_redo:false,packet_round_result:1,
 wire_body_base64:fixture.native_terminal_packet.wire_body_base64,wire_body_sha256:fixture.native_terminal_packet.wire_body_sha256}};
 const context={state,last,end,firstRound:structuredClone(fixture.initial_native.round),opponent:last.opponent,fightEpoch:3,elapsedMs:100};
 return {candidate,context};
}
test('synthetic terminal packet validates the exact result contract without claiming native execution',()=>{
 const {candidate,context}=example(),v=driver.validateRecoveredTerminal(candidate,context);
 assert.deepEqual(v.round.clean_hits,[19,9]);assert.equal(v.validated,true);assert.equal(v.policy_actions_after_stop,0);
 assert.equal(v.fight_epoch,3);assert.equal(v.original_opponent.sparring_bot_number,1);assert.equal(v.post_match_opponent.sparring_bot_number,2);
});
test('unrelated or incomplete diagnostic/end/source identity never triggers readback',async()=>{
 const {context:c}=example();let reads=0;
 for(const [diagnostic,end]of [['other_error',c.end],[fixture.recorded_error.reason,{...c.end,reason:'policy_action_watchdog_expired'}],
 [fixture.recorded_error.reason,{...c.end,round_identity_sha256:'a'.repeat(64)}]]){
 const x=await driver.recoverOpponentTransitionTerminal({diagnostic,end,last:c.last,getState:async()=>{reads++;},getSource:async()=>{reads++;}});assert.equal(x,null);}
 assert.equal(reads,0);
});
test('same-round recovery performs exactly two passive reads, never emits action or command',async()=>{
 const {candidate,context:c}=example();const calls=[];let clock=0;
 const result=await driver.recoverOpponentTransitionTerminal({...c,diagnostic:fixture.recorded_error.reason,
 now:()=>clock,getState:async()=>{calls.push('get_state');clock+=40;return c.state;},
 getSource:async()=>{calls.push('get_policy_state');clock+=40;return candidate;}});
 assert.equal(result.validated,true);assert.deepEqual(calls,['get_state','get_policy_state']);
});
test('wrong epoch, round, slot, hash, active state, human scope or stale source rejects',()=>{
 const mutations=[(s,c)=>c.state.private_ai.fight_epoch++,(s,c)=>c.state.private_ai.round_number++,
 (s,c)=>c.state.private_ai.local_slot=1,(s,c)=>c.state.private_ai.human_in_opponent_slot=true,
 (s,c)=>c.state.control.g1_policy_stream_running=true,(s,c)=>c.state.control.lease_held=false,
 (s,c)=>c.state.foreground.execution_surface='native_windows_isolated_desktop',
 s=>s.round_identity_sha256='0'.repeat(64),s=>s.local_slot=1,s=>s.stream_active=true,s=>s.round.active=true,
 s=>s.round.time_remaining=0.0741855,s=>s.round.number=4,s=>s.round.duration=30,s=>s.round.redo=true,
 s=>s.observation_sequence=1,s=>s.clock.qpc_frequency_hz=1000,(s,c)=>c.elapsedMs=501,
 s=>s.clock.qpc_ticks=String(BigInt(s.clock.qpc_ticks)+10000000n)];
 for(const mutate of mutations){const {candidate:s,context:c}=example();mutate(s,c);assert.throws(()=>driver.validateRecoveredTerminal(s,c));}
});
test('unknown/same/skipped opponent progression and changed fight result rejects',()=>{
 for(const mutate of [(s,c)=>c.state.private_ai.client_ai_difficulty=0,(s,c)=>c.state.private_ai.sparring_bot_number=1,
 (s,c)=>{c.state.private_ai.client_ai_difficulty=2;c.state.private_ai.sparring_bot_number=3;},
 s=>s.opponent.client_ai_difficulty=0,s=>s.opponent.human_in_opponent_slot=true,
 s=>s.fight.current_round=4,s=>s.fight.rounds_won=[3,1],s=>s.fight.winner_index=1,s=>s.fight.result='InProgress']){
 const {candidate:s,context:c}=example();mutate(s,c);assert.throws(()=>driver.validateRecoveredTerminal(s,c));}
});
test('missing/stale/unbound referee or mismatched genuine wire bytes rejects',()=>{
 for(const mutate of [s=>s.referee=null,s=>s.referee.available=false,s=>s.referee.receipt_age_seconds=.51,
 s=>s.referee.packet_round_number=4,s=>s.referee.packet_round_active=true,s=>s.referee.receipt_qpc_ticks='1',
 s=>s.referee.wire_body_sha256='a'.repeat(64),s=>s.round.clean_hits=[20,9],s=>s.round.result_value=0,
 s=>{const b=Buffer.from(s.referee.wire_body_base64,'base64');b[8]=20;s.referee.wire_body_base64=b.toString('base64');
 s.referee.wire_body_sha256=crypto.createHash('sha256').update(b).digest('hex');}]){
 const {candidate:s,context:c}=example();mutate(s,c);assert.throws(()=>driver.validateRecoveredTerminal(s,c));}
});
test('old zero-recovery exit verdicts remain identical; validated terminal exit keeps original error reason',()=>{
 for(const stop_reason of ['requested_duration_complete','source_round_terminal','stream_end:active_round_not_observed',
 'stream_end:policy_state_unavailable:InvalidDataException','operator_termination'])for(const active of [true,false]){
 const s={stop_reason,predictions:5,applied:5,final_round:{active,result_value:1}};assert.equal(driver.trialExitCode(s),old.trialExitCode(s));}
 const s={stop_reason:'stream_end:policy_state_unavailable:InvalidDataException',predictions:5,applied:5,
 final_round:{active:false,result_value:1},terminal_recovery:{validated:true}};
 assert.equal(driver.trialExitCode(s),0);s.final_round.active=true;assert.equal(driver.trialExitCode(s),2);
});
test('recovery is after acknowledged Stop and before release; encoder/worker action handlers stay identical',()=>{
 const s=fs.readFileSync(__dirname+'/live_transfer_run_masked.cjs','utf8'),b=fs.readFileSync(__dirname+'/baseline/live_transfer_run_masked.cjs','utf8');
 const start=s.indexOf('if(streamStopped && terminalRecoveryTrigger'),stop=s.lastIndexOf("await command('StopG1PolicyStream')"),release=s.lastIndexOf("await command('ReleaseExclusiveControl')");
 assert(stop<start&&start<release);assert(s.includes('streaming=false;'));
 const callbacks=x=>x.slice(x.indexOf("    encoder.bus.on('message',guardedCallback(encoded"),x.indexOf("    streaming=true;"));
 assert.equal(callbacks(s),callbacks(b));assert(s.includes("if(source.event!=='g1_policy_state'||stopping)return;"));
 assert(s.includes('if(stopping)return;'));
});
