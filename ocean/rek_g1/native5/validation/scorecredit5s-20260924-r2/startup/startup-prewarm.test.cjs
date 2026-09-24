'use strict';
const assert=require('node:assert/strict'),test=require('node:test'),fs=require('node:fs'),path=require('node:path');
const file=path.join(__dirname,'live_transfer_run_prewarm_r1.cjs');
const {prewarmPolicySource,validateControlledStartup}=require(file);
const opponent={client_ai_difficulty:0,sparring_bot_number:1};
function fixture(sequence,ticks,frame=sequence){return {event:'g1_policy_state',schema:'rek.g1_policy_source.v1',
  stream_active:false,global_input_emitted:false,phase:1,round_identity_sha256:'a'.repeat(64),local_slot:0,
  observation_sequence:sequence,clock:{qpc_ticks:ticks,qpc_frequency_hz:1000,unity_frame:frame},
  opponent:{...opponent,opponent_is_ai:true,human_in_opponent_slot:false},
  round:{active:true,duration:120,redo:false,result_value:0,time_remaining:119,clean_hits:[0,0]},
  input:{active:true,pending_move:false,pending_special:false,pending_estop:false,punching:false,recovering:false,
    velocity_command_xyz:[0,0,0]}};}
async function run(rows,{requestMs=40}={}){
  let ms=0,index=0;const logs=[];
  const receipt=await prewarmPolicySource({opponent,now:()=>ms,wait:async delay=>{ms+=delay;},
    log:(event,details)=>logs.push({event,...details}),pollSource:async()=>{ms+=requestMs;
      assert.ok(index<rows.length,'unexpected source request');return structuredClone(rows[index++]);}});
  return {receipt,logs,index};
}
test('cold >250 ms gaps cannot satisfy readiness; two later fresh intervals can',async()=>{
  const rows=[fixture(1,1000),fixture(2,1425),fixture(3,1760),fixture(4,1910),fixture(5,2050)];
  const {receipt,logs,index}=await run(rows);assert.equal(index,5);assert.equal(receipt.observation_sequence,5);
  assert.deepEqual(logs.filter(x=>x.event==='source_warmup_sample').map(x=>x.consecutive_fresh_intervals),[0,0,0,1,2]);
  assert.equal(logs.at(-1).worker_requests,0);assert.equal(logs.at(-1).encoder_requests,0);assert.equal(logs.at(-1).controlled,false);
});
test('same Unity frame and slow round-trip each reset readiness',async()=>{
  const rows=[fixture(1,1000,1),fixture(2,1050,1),fixture(3,1100,2),fixture(4,1150,3)];
  assert.equal((await run(rows)).index,4);
  let ms=0,index=0;const receipt=await prewarmPolicySource({opponent,now:()=>ms,wait:async d=>{ms+=d;},
    pollSource:async()=>{const i=index++;ms+=i===1?210:40;return fixture(i+1,1000+i*100);}});
  assert.equal(receipt.samples,4);
});
test('sustained slow source aborts within bounded warmup without any control request',async()=>{
  let ms=0,calls=0;await assert.rejects(prewarmPolicySource({opponent,now:()=>ms,wait:async d=>{ms+=d;},
    pollSource:async()=>{ms+=300;return fixture(++calls,1000+calls*300);}}),/source_warmup_budget_exceeded/);
  assert.ok(calls<=7);assert.ok(ms<=2240);
});
test('rejects unsafe, non-neutral, late, redo, scored, rebound and clock-regressed sources',async()=>{
  const mutations=[s=>s.stream_active=true,s=>s.global_input_emitted=true,s=>s.phase=2,
    s=>s.opponent.human_in_opponent_slot=true,s=>s.opponent.sparring_bot_number=2,
    s=>s.input.velocity_command_xyz=[0,0,1],s=>s.input.pending_move=true,s=>s.input.punching=true,
    s=>s.round.time_remaining=117.49,s=>s.round.redo=true,s=>s.round.clean_hits=[0,1],
    s=>s.round_identity_sha256='b'.repeat(64),s=>s.local_slot=1,s=>s.clock.qpc_frequency_hz=2000,
    s=>s.clock.qpc_ticks=999,s=>s.observation_sequence=1,s=>s.clock.unity_frame=0];
  for(const mutate of mutations){const second=fixture(2,1100);mutate(second);
    await assert.rejects(run([fixture(1,1000),second,fixture(3,1200)]));}
});
test('first controlled sample binds to warm round/slot/QPC/sequence and original fair-start limits',async()=>{
  const {receipt}=await run([fixture(1,1000),fixture(2,1100),fixture(3,1200)]);
  const source=fixture(4,1300);source.stream_active=true;source.round.time_remaining=117;
  validateControlledStartup(source,receipt);
  for(const mutate of [s=>s.stream_active=false,s=>s.round_identity_sha256='b'.repeat(64),s=>s.local_slot=1,
    s=>s.clock.qpc_ticks=1200,s=>s.clock.unity_frame=3,s=>s.observation_sequence=3,
    s=>s.round.time_remaining=116.99,s=>s.round.clean_hits=[1,0],s=>s.round.redo=true]){
    const bad=structuredClone(source);mutate(bad);assert.throws(()=>validateControlledStartup(bad,receipt));}
});
test('warmup uses read-only source polling; main encoder/worker receive no warmup requests',()=>{
  const code=fs.readFileSync(file,'utf8');
  const begin=code.indexOf('startupReadiness=await prewarmPolicySource');
  const end=code.indexOf("relay.bus.on('message'",begin);const warmup=code.slice(begin,end);
  assert.match(warmup,/type:'get_policy_state'/);assert.doesNotMatch(warmup,/encoder\.send|worker\.send|policy_action|command\(/);
  assert.ok(code.indexOf("encoder.wait(x=>x.event==='projection_manifest'")<code.indexOf("await command('AcquireExclusiveControl')"));
  assert.ok(code.indexOf("log('active_private_ai_opponent'")<begin);
  assert.ok(begin<code.indexOf("await command('StartG1PolicyStreamAnyAi')"));
  assert.match(code,/Date\.now\(\)-source\.received>=200/);
  assert.match(code,/worker\.send\(r\)/);
});
