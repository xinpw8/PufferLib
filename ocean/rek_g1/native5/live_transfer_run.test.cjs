'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {test}=require('node:test');
const {once}=require('node:events');
const {validateWorkerReady,validateWorkerAction,guardedCallback,sendAndWait,childEndpoint,LiveActionPacer,canExitLostPrivateSession,betweenPrivateRounds,canRequestPrivateRound,canExitUnexpectedPrivateAiSession,canReadyPrivateAiSession,ensurePrivateArena}=require('./live_transfer_run.cjs');
const sha='a'.repeat(64),round='b'.repeat(64);
const ready=()=>({type:'ready',checkpoint_sha256:sha,native_cuda:true,environment_stepping:false,
  observation_schema:'rek.native5.scaled_polar_xy.v1',precision:'bf16',selection:'sampled',
  observations:223,actions:33,hidden_size:256,num_layers:2});

function unexpectedPrivateAi() {
  return {scene:'Arena',lobby_screen:null,foreground:{isolated_session_verified:true},private_ai:{
    proven:false,reason:'unexpected_sparring_bot_difficulty',network_client_only:true,
    context_is_solo:true,solo_route_proven:true,opponent_is_ai:true,opponent_slot_is_ai:true,
    human_in_opponent_slot:false,opponent_slot_client_known:true,opponent_slot_has_client:false,
    opponent_human_bit_set:false,exact_sparring_bot_1:false,client_ai_difficulty:2,sparring_bot_number:3,
    phase:'Idle',round_active:false,round_inactive:true}};
}
function exactPrivateAi() {
  const s=unexpectedPrivateAi();Object.assign(s.private_ai,{proven:true,reason:'private_ai_session_proven',
    exact_sparring_bot_1:true,client_ai_difficulty:0,sparring_bot_number:1});return s;
}
function bootstrapPrivateAi() {
  const s=unexpectedPrivateAi();s.private_ai.client_visual_only_fighter_pair=false;return s;
}
function activePrivateAi(exact=true) {
  const s=exact?exactPrivateAi():unexpectedPrivateAi();Object.assign(s.private_ai,{phase:'RoundActive',
    round_active:true,round_inactive:false,client_visual_only_fighter_pair:true,active_gameplay_proven:exact});return s;
}
const homeState=()=>({scene:'Lobby',lobby_screen:'Home'});
const freePlayState=()=>({scene:'Lobby',lobby_screen:'FreePlay'});
function recoveryFixture(states,extra={}) {
  let clock=0,index=0;const commands=[],events=[];
  return {commands,events,options:{enterPrivate:true,
    command:async command=>commands.push(command),getState:async()=>states[Math.min(index++,states.length-1)],
    wait:async ms=>{clock+=ms;},now:()=>clock,log:(event,detail)=>events.push({event,...detail}),...extra}};
}
test('unexpected-AI exit requires isolated solo, known no-human occupancy, inactive Idle, and a measured nonzero difficulty',()=>{
  const s=unexpectedPrivateAi();assert.equal(canExitUnexpectedPrivateAiSession(s),true);
  for(const key of Object.keys(s.private_ai)) {
    const p={...s.private_ai};delete p[key];
    assert.equal(canExitUnexpectedPrivateAiSession({...s,private_ai:p}),false,`missing ${key}`);
  }
  for(const patch of [{proven:true},{reason:'solo_route_not_proven'},{network_client_only:false},
    {context_is_solo:false},{solo_route_proven:false},{opponent_is_ai:false},{opponent_slot_is_ai:false},
    {human_in_opponent_slot:true},{opponent_slot_client_known:false},{opponent_slot_has_client:true},
    {opponent_human_bit_set:true},{exact_sparring_bot_1:true},{client_ai_difficulty:0},{client_ai_difficulty:1.5},
    {client_ai_difficulty:256,sparring_bot_number:257},{sparring_bot_number:2},{phase:'RoundActive'},
    {round_active:true},{round_inactive:false}])
    assert.equal(canExitUnexpectedPrivateAiSession({...s,private_ai:{...s.private_ai,...patch}}),false,JSON.stringify(patch));
  assert.equal(canExitUnexpectedPrivateAiSession({...s,scene:'Lobby'}),false);
  assert.equal(canExitUnexpectedPrivateAiSession({...s,foreground:{isolated_session_verified:false}}),false);
  assert.equal(canExitUnexpectedPrivateAiSession(exactPrivateAi()),false);
  assert.equal(canExitUnexpectedPrivateAiSession({}),false);
});
test('existing wrong-bot Idle arena exits once, observes Lobby Home, then reenters through native menus',async()=>{
  const good=exactPrivateAi(),f=recoveryFixture([unexpectedPrivateAi(),homeState(),freePlayState(),good]);
  assert.equal(await ensurePrivateArena(unexpectedPrivateAi(),f.options),good);
  assert.deepEqual(f.commands,['ExitUnexpectedPrivateAiSession','NavigateFreePlay','EnterSolo']);
  assert.equal(f.events.filter(x=>x.event==='unexpected_private_ai_recovery').length,1);
});
test('wrong-bot result during entry permits one fresh menu-route attempt and still demands exact Bot1',async()=>{
  const good=exactPrivateAi(),f=recoveryFixture([freePlayState(),unexpectedPrivateAi(),homeState(),freePlayState(),good]);
  assert.equal(await ensurePrivateArena(homeState(),f.options),good);
  assert.deepEqual(f.commands,['NavigateFreePlay','EnterSolo','ExitUnexpectedPrivateAiSession','NavigateFreePlay','EnterSolo']);
});
test('a second wrong-bot reservation stops without a second exit or any StartRound request',async()=>{
  const f=recoveryFixture([homeState(),freePlayState(),unexpectedPrivateAi()]);
  await assert.rejects(ensurePrivateArena(unexpectedPrivateAi(),f.options),/unexpected private AI remained after one exit\/reentry/);
  assert.deepEqual(f.commands,['ExitUnexpectedPrivateAiSession','NavigateFreePlay','EnterSolo']);
});
test('recovery requires explicit private entry and never changes difficulty',async()=>{
  const f=recoveryFixture([exactPrivateAi()],{enterPrivate:false});
  await assert.rejects(ensurePrivateArena(unexpectedPrivateAi(),f.options),/current client is not verified solo Bot1 arena/);
  assert.deepEqual(f.commands,[]);
  const good=exactPrivateAi();assert.equal(await ensurePrivateArena(good,f.options),good);
});
test('unexpected active AI or human occupancy cannot invoke the exit command',async()=>{
  for(const patch of [{phase:'RoundActive',round_active:true,round_inactive:false},{human_in_opponent_slot:true}]) {
    const s=unexpectedPrivateAi();Object.assign(s.private_ai,patch);const f=recoveryFixture([s]);
    await assert.rejects(ensurePrivateArena(s,f.options),/private-practice entry timeout/);
    assert.deepEqual(f.commands,[]);
  }
});
test('native exit rejection or missing Lobby Home postcondition prevents reentry',async()=>{
  const rejected=recoveryFixture([homeState()],{command:async()=>{throw Error('native exit rejected');}});
  await assert.rejects(ensurePrivateArena(unexpectedPrivateAi(),rejected.options),/native exit rejected/);
  const stuck=recoveryFixture([{scene:'Arena',lobby_screen:null}]);
  await assert.rejects(ensurePrivateArena(unexpectedPrivateAi(),stuck.options),/unexpected private AI exit timeout/);
  assert.deepEqual(stuck.commands,['ExitUnexpectedPrivateAiSession']);
});
test('ready bootstrap requires explicitly absent visual pair and preserves the existing exact-Bot1 start path',()=>{
  assert.equal(canReadyPrivateAiSession(bootstrapPrivateAi()),true);
  assert.equal(canReadyPrivateAiSession(unexpectedPrivateAi()),false,'missing pair measurement is not absence');
  const pair=bootstrapPrivateAi();pair.private_ai.client_visual_only_fighter_pair=true;
  assert.equal(canReadyPrivateAiSession(pair),false);
  assert.equal(canReadyPrivateAiSession(exactPrivateAi()),false);
  assert.equal(canReadyPrivateAiSession(activePrivateAi(false)),false);
});
test('Idle inherited difficulty may bootstrap once into active exact Bot1 without exit or policy input',async()=>{
  const good=activePrivateAi(),f=recoveryFixture([bootstrapPrivateAi(),good]);
  assert.equal(await ensurePrivateArena(bootstrapPrivateAi(),f.options),good);
  assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
  assert.equal(f.events[0].opponent_identity,'unverified_until_active_spawn');
});
test('ready bootstrap withholds policy until active spawned exact-Bot1 proof and input readiness',async()=>{
  const waiting=activePrivateAi();waiting.private_ai.active_gameplay_proven=false;
  const unspawned=activePrivateAi(false);unspawned.private_ai.client_visual_only_fighter_pair=false;
  const good=activePrivateAi(),f=recoveryFixture([unspawned,waiting,good]);
  assert.equal(await ensurePrivateArena(bootstrapPrivateAi(),f.options),good);
  assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
});
test('active spawned Bot3 after readiness stops with no policy stream, attack, retry, or difficulty mutation',async()=>{
  const f=recoveryFixture([activePrivateAi(false)]);
  await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),f.options),/active opponent is not exact Bot1; policy input withheld/);
  assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
});
test('ready bootstrap stops on human/route changes, native rejection, and timeout without another request',async()=>{
  for(const change of [{human_in_opponent_slot:true},{solo_route_proven:false},{network_client_only:false}]) {
    const s=bootstrapPrivateAi();Object.assign(s.private_ai,change);const f=recoveryFixture([s]);
    await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),f.options),/route proof lost/);
    assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
  }
  const f=recoveryFixture([bootstrapPrivateAi()]);
  await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),f.options),/ready bootstrap timeout/);
  assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
  const rejected=recoveryFixture([activePrivateAi()],{command:async()=>{throw Error('ready rejected');}});
  await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),rejected.options),/ready rejected/);
});

test('automatic between-round transition waits without sending StartRound',()=>{
  const private_ai={proven:true,solo_route_proven:true,context_is_solo:true,exact_sparring_bot_1:true,
    opponent_is_ai:true,human_in_opponent_slot:false,round_active:false,phase:'BetweenRounds'};
  assert.equal(betweenPrivateRounds({private_ai}),true);
  assert.equal(betweenPrivateRounds({private_ai:{...private_ai,phase:'FightOver'}}),true);
  for(const changed of [{phase:'Idle'},{phase:'RoundActive',round_active:true},{human_in_opponent_slot:true},{proven:false}])
    assert.equal(betweenPrivateRounds({private_ai:{...private_ai,...changed}}),false);
  assert.equal(betweenPrivateRounds({}),false);
});

test('round start uses the native Idle path; post-win space flag is not an idle prerequisite',()=>{
  const private_ai={proven:true,solo_route_proven:true,context_is_solo:true,exact_sparring_bot_1:true,
    opponent_is_ai:true,human_in_opponent_slot:false,round_active:false,phase:'Idle',space_gate_would_allow:true};
  assert.equal(canRequestPrivateRound({private_ai}),true);
  for(const phase of ['RoundActive','BetweenRounds','FightOver'])
    assert.equal(canRequestPrivateRound({private_ai:{...private_ai,phase,active_gameplay_proven:false}}),false);
  assert.equal(canRequestPrivateRound({private_ai:{...private_ai,space_gate_would_allow:false}}),true);
  assert.equal(canRequestPrivateRound({private_ai:{...private_ai,post_fight_prompt:true,post_fight_is_winner:false}}),false);
  assert.equal(canRequestPrivateRound({private_ai:{...private_ai,post_fight_prompt:true,post_fight_is_winner:true}}),true);
});

test('send failure consumes the pending response rejection during disconnect cleanup',async()=>{
  const endpoint={wait:()=>new Promise((resolve,reject)=>setImmediate(()=>reject(Error('late response timeout')))),
    send:()=>{throw Error('relay is not writable');}};
  await assert.rejects(sendAndWait(endpoint,{},()=>true),/relay is not writable/);
  await new Promise(resolve=>setImmediate(resolve));
});
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
