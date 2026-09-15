'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {test}=require('node:test');
const {once}=require('node:events');
const {validateWorkerReady,validateWorkerAction,guardedCallback,childEndpoint,LiveActionPacer,canExitLostPrivateSession}=require('./live_transfer_run.cjs');
const sha='a'.repeat(64),round='b'.repeat(64);
const ready=()=>({type:'ready',checkpoint_sha256:sha,native_cuda:true,environment_stepping:false,
  observation_schema:'rek.native5.scaled_polar_xy.v1',precision:'bf16',selection:'sampled',
  observations:223,actions:33,hidden_size:256,num_layers:2});
test('lost-session exit requires proven solo AI scope, explicit inactivity, and a loss prompt',()=>{
  const private_ai={proven:true,solo_route_proven:true,context_is_solo:true,exact_sparring_bot_1:true,
    opponent_is_ai:true,human_in_opponent_slot:false,round_active:false,round_inactive:true,
    post_fight_prompt:true,post_fight_is_winner:false};
  assert.equal(canExitLostPrivateSession({private_ai}),true);
  for(const key of Object.keys(private_ai)) {
    const changed={...private_ai};delete changed[key];
    assert.equal(canExitLostPrivateSession({private_ai:changed}),false,key);
    assert.equal(canExitLostPrivateSession({private_ai:{...private_ai,[key]:!private_ai[key]}}),false,key);
  }
  assert.equal(canExitLostPrivateSession({}),false);
});
test('worker readiness pins the full selected inference contract',()=>{
  assert.doesNotThrow(()=>validateWorkerReady(ready(),sha));
  for(const key of Object.keys(ready())) {
    const value=ready(); delete value[key];
    assert.throws(()=>validateWorkerReady(value,sha),/identity mismatch/,key);
  }
  for(const [key,value] of Object.entries({precision:'fp32',selection:'greedy',observations:222,
    actions:34,hidden_size:128,num_layers:1,environment_stepping:true,native_cuda:false,
    observation_schema:'unknown',checkpoint_sha256:'c'.repeat(64)})) {
    assert.throws(()=>validateWorkerReady({...ready(),[key]:value},sha),/identity mismatch/,key);
  }
});
test('action identity and categorical range are checked before relay',()=>{
  const source={sequence:12,round};const prediction={type:'action',seq:12,round_id:round,checkpoint_sha256:sha,action:0};
  for(const action of [0,1,17,32])assert.doesNotThrow(()=>validateWorkerAction({...prediction,action},source,sha));
  for(const action of [-1,33,1.5,NaN,Infinity,'1',null,undefined])
    assert.throws(()=>validateWorkerAction({...prediction,action},source,sha),/out_of_range/);
  for(const patch of [{seq:11},{round_id:'c'.repeat(64)},{checkpoint_sha256:'d'.repeat(64)},{type:'terminal'}])
    assert.throws(()=>validateWorkerAction({...prediction,...patch},source,sha),/identity_mismatch/);
  assert.throws(()=>validateWorkerAction(prediction,null,sha),/identity_mismatch/);
});
test('callback write failures are routed through shutdown instead of escaping',()=>{
  let observed=null,handled=0;
  const callback=guardedCallback(()=>{throw new Error('simulated closed stdin');},error=>{observed=error;handled++;});
  assert.doesNotThrow(()=>callback({}));assert.equal(observed.message,'simulated closed stdin');assert.equal(handled,1);
  let value=null;guardedCallback(x=>{value=x;},()=>assert.fail())(42);assert.equal(value,42);
});
test('one decision stays inflight until the matching Unity action acknowledgment',()=>{
  const p=new LiveActionPacer();
  const source=(seq,qpc)=>({observation_sequence:seq,round_identity_sha256:round,clock:{qpc_ticks:qpc}});
  assert.equal(p.offer(source(2,100),1000),null);
  assert.equal(p.offer(source(3,110),1001),'action_inflight');
  p.sent('action-92',4,1002);
  assert.equal(p.pending.sequence,2);
  for(let seq=4;seq<=15;seq++)assert.equal(p.offer(source(seq,110+seq),1003),'action_inflight');
  assert.throws(()=>p.sent('action-93',5,1004),/already_inflight/);
  assert.throws(()=>p.abandon(),/unacknowledged/);
  const ack={request_id:'action-92',observation_sequence:2,round_identity_sha256:round,action:4,applied:true,clock:{qpc_ticks:200}};
  assert.equal(p.acknowledge({...ack,request_id:'old-action'}),false);
  assert.equal(p.pending.sequence,2,'unmatched ack cannot unlock the next inference');
  assert.throws(()=>p.acknowledge({...ack,observation_sequence:3}),/identity_mismatch/);
  assert.equal(p.pending.sequence,2);
  assert.equal(p.acknowledge(ack),true);assert.equal(p.pending,null);
  assert.equal(p.offer(source(16,199),1005),'source_before_last_ack');
  assert.equal(p.offer(source(17,200),1006),'source_before_last_ack');
  assert.equal(p.pending,null,'queued pre-ack states are discarded, not retained');
  assert.equal(p.offer(source(18,201),1007),null);assert.equal(p.pending.sequence,18);
});
test('rejected game actions advance the measured boundary without action replay',()=>{
  const p=new LiveActionPacer();
  p.offer({observation_sequence:1,round_identity_sha256:round,clock:{qpc_ticks:'9007199254740992'}},1);
  p.sent('action-1',17,2);
  const ack={request_id:'action-1',observation_sequence:1,round_identity_sha256:round,action:17,
    applied:false,reason:'not_ready',clock:{qpc_ticks:'9007199254741010'}};
  assert.equal(p.acknowledge(ack),true);assert.equal(p.pending,null);
  assert.equal(p.acknowledge(ack),false,'duplicate ack is ignored');
  assert.equal(p.offer({observation_sequence:2,round_identity_sha256:round,clock:{qpc_ticks:'9007199254741000'}},3),'source_before_last_ack');
  assert.equal(p.offer({observation_sequence:3,round_identity_sha256:round,clock:{qpc_ticks:'9007199254741011'}},4),null);
});
test('missing or regressing measured clocks cannot be replaced with host time',()=>{
  const p=new LiveActionPacer();
  for(const clock of [undefined,{}, {qpc_ticks:null},{qpc_ticks:NaN},{qpc_ticks:2**53},{qpc_ticks:'guess'}])
    assert.throws(()=>p.offer({observation_sequence:1,round_identity_sha256:round,clock},1),/qpc_unavailable/);
  const source={observation_sequence:1,round_identity_sha256:round,clock:{qpc_ticks:100}};
  p.offer(source,1);p.abandon();
  assert.equal(p.offer(source,2),'source_clock_not_new','failed encoding does not queue the same source again');
  p.offer({...source,observation_sequence:2,clock:{qpc_ticks:101}},3);p.sent('a',1,4);
  const ack={request_id:'a',observation_sequence:2,round_identity_sha256:round,action:1};
  assert.throws(()=>p.acknowledge(ack),/qpc_unavailable/);
  assert.throws(()=>p.acknowledge({...ack,clock:{qpc_ticks:100}}),/clock_regressed/);
  assert.equal(p.pending.sequence,2);
});
test('worker startup fatal rejects promptly, without waiting for ready timeout',async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-live-endpoint-test-'));
  const spec=[process.execPath,'-e','console.log(JSON.stringify({type:"fatal",code:"fixture_failure"}));setTimeout(()=>{},10000);'];
  const endpoint=childEndpoint('worker',spec,dir);
  endpoint.bus.on('failure',()=>{});
  const started=Date.now();
  try {
    await assert.rejects(endpoint.wait(x=>x.type==='ready',5000),/startup failure:fixture_failure/);
    assert(Date.now()-started<3000,'fatal must reject before response timeout');
  } finally {
    endpoint.close();const stopped=once(endpoint.child,'exit');endpoint.child.kill('SIGTERM');await stopped;
    // Only the three files created by this endpoint inside its owned temporary directory.
    for(const name of ['worker.stdout.jsonl','worker.stderr.txt','worker.stdin.jsonl'])fs.unlinkSync(path.join(dir,name));
    fs.rmdirSync(dir);
  }
});
