'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {test}=require('node:test');
const {once}=require('node:events');
const {validateWorkerReady,validateWorkerAction,guardedCallback,sendAndWait,childEndpoint,LiveActionPacer,canExitLostPrivateSession,betweenPrivateRounds,canRequestPrivateRound,privateArena,botIdentity,validateBotIdentity,trialExitCode,canReadyPrivateAiSession,ensurePrivateArena}=require('./live_transfer_run.cjs');
const sha='a'.repeat(64),round='b'.repeat(64);
const ready=()=>({type:'ready',checkpoint_sha256:sha,native_cuda:true,environment_stepping:false,
  observation_schema:'rek.native5.scaled_polar_xy.v1',precision:'bf16',selection:'sampled',
  observations:223,actions:33,hidden_size:256,num_layers:2});

function privateAi(difficulty=0) {
  return {scene:'Arena',lobby_screen:null,foreground:{isolated_session_verified:true},private_ai:{
    proven:difficulty===0,policy_proven:true,network_client_only:true,
    context_is_solo:true,solo_route_proven:true,opponent_is_ai:true,opponent_slot_is_ai:true,
    human_in_opponent_slot:false,opponent_slot_client_known:true,opponent_slot_has_client:false,
    opponent_human_bit_set:false,exact_sparring_bot_1:difficulty===0,
    client_ai_difficulty:difficulty,sparring_bot_number:difficulty+1,
    phase:'Idle',round_active:false,round_inactive:true,client_visual_only_fighter_pair:true}};
}
function bootstrapPrivateAi(difficulty=2) {
  const s=privateAi(difficulty);s.private_ai.client_visual_only_fighter_pair=false;
  s.private_ai.policy_proven=false;return s;
}
function activePrivateAi(difficulty=0) {
  const s=privateAi(difficulty);Object.assign(s.private_ai,{phase:'RoundActive',
    round_active:true,round_inactive:false,policy_active_gameplay_proven:true});return s;
}
const homeState=()=>({scene:'Lobby',lobby_screen:'Home'});
const freePlayState=()=>({scene:'Lobby',lobby_screen:'FreePlay'});
function recoveryFixture(states,extra={}) {
  let clock=0,index=0;const commands=[],events=[];
  return {commands,events,options:{enterPrivate:true,
    command:async command=>commands.push(command),getState:async()=>states[Math.min(index++,states.length-1)],
    wait:async ms=>{clock+=ms;},now:()=>clock,log:(event,detail)=>events.push({event,...detail}),...extra}};
}

test('any known private bot is accepted, with measured identity and independent legacy Bot1 proof',()=>{
  for(const difficulty of [0,1,2,7,254,255]) {
    const s=privateAi(difficulty);assert.equal(privateArena(s),true);
    assert.deepEqual(botIdentity(s.private_ai),{client_ai_difficulty:difficulty,sparring_bot_number:difficulty+1});
    assert.equal(s.private_ai.proven,difficulty===0);
  }
});

test('unknown bot identity, public route, humans, occupancy uncertainty and missing isolation reject scope',()=>{
  const s=privateAi(2);
  for(const patch of [{policy_proven:false},{network_client_only:false},{context_is_solo:false},
    {solo_route_proven:false},{opponent_is_ai:false},{opponent_slot_is_ai:false},
    {human_in_opponent_slot:true},{opponent_slot_client_known:false},{opponent_slot_has_client:true},
    {opponent_human_bit_set:true},{client_ai_difficulty:null},{client_ai_difficulty:-1},
    {client_ai_difficulty:1.5},{client_ai_difficulty:256,sparring_bot_number:257},{sparring_bot_number:2}])
    assert.equal(privateArena({...s,private_ai:{...s.private_ai,...patch}}),false,JSON.stringify(patch));
  for(const key of ['policy_proven','network_client_only','context_is_solo','solo_route_proven',
    'opponent_is_ai','opponent_slot_is_ai','human_in_opponent_slot','opponent_slot_client_known',
    'opponent_slot_has_client','opponent_human_bit_set','client_ai_difficulty','sparring_bot_number']) {
    const p={...s.private_ai};delete p[key];
    assert.equal(privateArena({...s,private_ai:p}),false,key);
  }
  assert.equal(privateArena({...s,scene:'Lobby'}),false);
  assert.equal(privateArena({...s,foreground:{isolated_session_verified:false}}),false);
  assert.equal(privateArena({}),false);
});

test('existing higher-bot arena is retained without exit, reentry, difficulty mutation or input',async()=>{
  for(const good of [privateAi(2),activePrivateAi(7)]) {
    const f=recoveryFixture([good],{enterPrivate:false});
    assert.equal(await ensurePrivateArena(good,f.options),good);assert.deepEqual(f.commands,[]);
  }
});

test('private entry accepts the assigned higher bot through the native private menu route',async()=>{
  const good=privateAi(2),f=recoveryFixture([freePlayState(),good]);
  assert.equal(await ensurePrivateArena(homeState(),f.options),good);
  assert.deepEqual(f.commands,['NavigateFreePlay','EnterSolo']);
  assert.deepEqual(f.events[0].opponent,{client_ai_difficulty:2,sparring_bot_number:3});
});

test('unproven entry requires explicit authorization and does not control a human arena',async()=>{
  const f=recoveryFixture([privateAi()],{enterPrivate:false});
  await assert.rejects(ensurePrivateArena(homeState(),f.options),/not verified ready private AI arena/);
  assert.deepEqual(f.commands,[]);
  const s=privateAi(2);s.private_ai.human_in_opponent_slot=true;const human=recoveryFixture([s]);
  await assert.rejects(ensurePrivateArena(s,human.options),/private-practice entry timeout/);
  assert.deepEqual(human.commands,[]);
});

test('ready bootstrap requires measured absent visual pair and inactive private Idle',()=>{
  for(const difficulty of [0,2,255])assert.equal(canReadyPrivateAiSession(bootstrapPrivateAi(difficulty)),true);
  const missing=bootstrapPrivateAi();delete missing.private_ai.client_visual_only_fighter_pair;
  assert.equal(canReadyPrivateAiSession(missing),false);
  assert.equal(canReadyPrivateAiSession(privateAi()),false);
  assert.equal(canReadyPrivateAiSession(activePrivateAi(2)),false);
});

test('native ready accepts any spawned private AI without an exit or difficulty mutation',async()=>{
  for(const difficulty of [0,2,255]) {
    const good=activePrivateAi(difficulty),f=recoveryFixture([bootstrapPrivateAi(),good]);
    assert.equal(await ensurePrivateArena(bootstrapPrivateAi(),f.options),good);
    assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
    assert.equal(f.events[0].opponent_identity,'unverified_until_active_spawn');
    assert.deepEqual(f.events[1].opponent,botIdentity(good.private_ai));
  }
});

test('ready bootstrap waits for spawned active policy proof even if legacy Bot1 scope is false',async()=>{
  const waiting=activePrivateAi(2);waiting.private_ai.policy_active_gameplay_proven=false;
  const unspawned=activePrivateAi(2);unspawned.private_ai.client_visual_only_fighter_pair=false;
  const good=activePrivateAi(2),f=recoveryFixture([unspawned,waiting,good]);
  assert.equal(await ensurePrivateArena(bootstrapPrivateAi(),f.options),good);
  assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
});

test('ready bootstrap stops on human/route changes, unknown identity, rejection and timeout',async()=>{
  for(const change of [{human_in_opponent_slot:true},{solo_route_proven:false},{network_client_only:false}]) {
    const s=bootstrapPrivateAi();Object.assign(s.private_ai,change);const f=recoveryFixture([s]);
    await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),f.options),/route proof lost/);
    assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
  }
  const unknown=activePrivateAi(2);unknown.private_ai.client_ai_difficulty=null;
  await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),recoveryFixture([unknown]).options),/identity unavailable/);
  const f=recoveryFixture([bootstrapPrivateAi()]);
  await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),f.options),/ready bootstrap timeout/);
  assert.deepEqual(f.commands,['ReadyPrivateAiSession']);
  const rejected=recoveryFixture([activePrivateAi()],{command:async()=>{throw Error('ready rejected');}});
  await assert.rejects(ensurePrivateArena(bootstrapPrivateAi(),rejected.options),/ready rejected/);
});

test('a stream pins bot identity and rejects mid-round changes or human occupancy',()=>{
  const expected=botIdentity(privateAi(2).private_ai);
  const measured={...expected,opponent_is_ai:true,human_in_opponent_slot:false};
  assert.doesNotThrow(()=>validateBotIdentity(measured,expected));
  for(const patch of [{client_ai_difficulty:0,sparring_bot_number:1},{client_ai_difficulty:null},
    {sparring_bot_number:4},{opponent_is_ai:false},{human_in_opponent_slot:true}])
    assert.throws(()=>validateBotIdentity({...measured,...patch},expected),/identity_changed_or_unproven/);
  assert.throws(()=>validateBotIdentity(undefined,expected),/identity_changed_or_unproven/);
});

test('earlier successful actions cannot turn a later scope failure into a successful exit',()=>{
  const s={predictions:100,applied:99,final_round:{active:false,result_value:1}};
  for(const stop_reason of ['source_round_terminal','stream_end:active_round_not_observed','requested_duration_complete'])
    assert.equal(trialExitCode({...s,stop_reason}),0);
  for(const stop_reason of ['relay_callback:private_ai_identity_changed_or_unproven',
    'stream_end:policy_opponent_identity_changed','private arena proof lost','source_stream_missing'])
    assert.equal(trialExitCode({...s,stop_reason}),2);
  assert.equal(trialExitCode({...s,stop_reason:'source_round_terminal',final_round:{active:true,result_value:0}}),2);
  assert.equal(trialExitCode({...s,stop_reason:'requested_duration_complete',applied:0}),2);
});

test('automatic between-round transition waits without sending a round request',()=>{
  const s=privateAi(2);
  for(const phase of ['BetweenRounds','FightOver'])
    assert.equal(betweenPrivateRounds({...s,private_ai:{...s.private_ai,phase}}),true);
  assert.equal(betweenPrivateRounds(s),false);
  assert.equal(betweenPrivateRounds(activePrivateAi(2)),false);
  assert.equal(betweenPrivateRounds({}),false);
});

test('any private AI round starts through native Idle; a loss prompt requires exit',()=>{
  const s=privateAi(2);
  assert.equal(canRequestPrivateRound(s),true);
  for(const phase of ['RoundActive','BetweenRounds','FightOver'])
    assert.equal(canRequestPrivateRound({...s,private_ai:{...s.private_ai,phase}}),false);
  assert.equal(canRequestPrivateRound({...s,private_ai:{...s.private_ai,post_fight_prompt:true,post_fight_is_winner:false}}),false);
  assert.equal(canRequestPrivateRound({...s,private_ai:{...s.private_ai,post_fight_prompt:true,post_fight_is_winner:true}}),true);
});

test('lost-session exit requires proven private AI, explicit inactivity, and a loss prompt',()=>{
  const s=privateAi(2);Object.assign(s.private_ai,{post_fight_prompt:true,post_fight_is_winner:false});
  assert.equal(canExitLostPrivateSession(s),true);
  for(const patch of [{policy_proven:false},{human_in_opponent_slot:true},{round_active:true},
    {round_inactive:false},{post_fight_prompt:false},{post_fight_is_winner:true}])
    assert.equal(canExitLostPrivateSession({...s,private_ai:{...s.private_ai,...patch}}),false);
  for(const key of ['round_active','round_inactive','post_fight_prompt','post_fight_is_winner']) {
    const p={...s.private_ai};delete p[key];assert.equal(canExitLostPrivateSession({...s,private_ai:p}),false,key);
  }
  assert.equal(canExitLostPrivateSession({}),false);
});

test('send failure consumes the pending response rejection during disconnect cleanup',async()=>{
  const endpoint={wait:()=>new Promise((resolve,reject)=>setImmediate(()=>reject(Error('late response timeout')))),
    send:()=>{throw Error('relay is not writable');}};
  await assert.rejects(sendAndWait(endpoint,{},()=>true),/relay is not writable/);
  await new Promise(resolve=>setImmediate(resolve));
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
