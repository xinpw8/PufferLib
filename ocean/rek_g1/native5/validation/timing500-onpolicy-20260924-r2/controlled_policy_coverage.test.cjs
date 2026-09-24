'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {deriveCoverage,receipt,validateReceipt}=require('./controlled_policy_coverage.cjs');
function fixture(){
  const id='a'.repeat(64),cp='b'.repeat(64);
  const source=(seq,t,remaining,stream,active=true)=>({observation_sequence:seq,round_identity_sha256:id,local_slot:0,
    stream_active:stream,global_input_emitted:false,clock:{qpc_ticks:1000+Math.round(t*1000),qpc_frequency_hz:1000,unity_frame:seq,unity_time:t},
    round:{number:1,duration:120,time_remaining:remaining,active,redo:false,result_value:active?0:1,clean_hits:[0,0]}});
  const sources=[source(1,0,119.9,false),source(2,.6,118.85,false),source(3,.7,118.75,true),
    source(4,.8,118.65,true),source(5,2,117.45,true),source(6,2.6,0,false,false)];
  const summary={checkpoint_sha256:cp,controlled_startup_validated:true,initial_round:sources[2].round,final_round:sources[5].round,
    startup_readiness:{round_identity_sha256:id,local_slot:0,observation_sequence:2,qpc_ticks:'1600',qpc_frequency_hz:1000,unity_frame:2,time_remaining:118.85}};
  const acks=[1.1,1.8,2.5].map(t=>({applied:true,observation_sequence:4,round_identity_sha256:id,clock:{unity_time:t}}));
  const contact={round_identity_sha256:id,local_slot:0,checkpoint_sha256:cp,completed_policy_round:false,
    terminal_evidence_consistent:true,terminal_point_counters_consistent:true,native_capture_complete:true,
    initial_observation_within_first_second:true,outcome:'win',policy_control_coverage:{applied_action_returns:3,
      first_applied_seconds_after_first_observation:1.1,terminal_seconds_after_last_applied:.1,maximum_applied_action_gap_seconds:.7,tolerance_seconds:1}};
  return {summary,contact,sources,acks,firstDecisionSeq:4};
}
const derive=f=>deriveCoverage(f.summary,f.contact,f.sources,f.acks,f.firstDecisionSeq);
test('explicit derivative preserves legacy false, raw timer and original tolerance',()=>{
  const f=fixture(),original=JSON.stringify(f),c=derive(f);
  assert.equal(c.legacy_completed_policy_round,false);assert.equal(c.controlled_policy_interval_complete,true);
  assert(Math.abs(c.controlled_to_first_applied_seconds-.4)<1e-12);assert.equal(c.tolerance_seconds,1);
  assert.equal(c.actual_control_start_time_remaining_seconds,118.75);assert.equal(c.native_round_elapsed_before_control_seconds,1.25);
  assert.equal(JSON.stringify(f),original);
});
test('unchanged first-ACK, inter-ACK and terminal one-second bounds reject adverse coverage',()=>{
  const changes=[f=>{f.acks[0].clock.unity_time=1.75;f.contact.policy_control_coverage.first_applied_seconds_after_first_observation=1.75;},
    f=>{f.acks[0].clock.unity_time=.65;f.contact.policy_control_coverage.first_applied_seconds_after_first_observation=.65;},
    f=>{f.acks[1].clock.unity_time=1.2;f.contact.policy_control_coverage.maximum_applied_action_gap_seconds=1.3;},
    f=>{f.sources.at(-1).clock.unity_time=4;f.contact.policy_control_coverage.terminal_seconds_after_last_applied=1.5;},
    f=>{f.sources.at(-1).clock.unity_time=2.4;f.contact.policy_control_coverage.terminal_seconds_after_last_applied=-.1;}];
  for(const change of changes){const f=fixture();change(f);assert.throws(()=>derive(f));}
});
test('identity, readiness QPC/slot/round, score, driver bounds and native guards stay strict',()=>{
  const changes=[f=>f.sources[2].round_identity_sha256='c'.repeat(64),f=>f.sources[2].local_slot=1,
    f=>f.sources[2].clock.qpc_ticks=1600,f=>f.sources[2].round.number=2,
    f=>f.sources[3].round.clean_hits=[1,0],f=>{f.sources[2].round.time_remaining=116.9;},
    f=>f.summary.controlled_startup_validated=false,f=>f.contact.native_capture_complete=false,
    f=>f.contact.terminal_evidence_consistent=false,f=>f.contact.terminal_point_counters_consistent=false,
    f=>f.contact.initial_observation_within_first_second=false,f=>f.firstDecisionSeq=2];
  for(const change of changes){const f=fixture();change(f);assert.throws(()=>derive(f));}
});
test('receipt binds unchanged legacy analysis, actual sources and worker sequence; tampering fails',()=>{
  const c=derive(fixture()),inputs=['trial/summary.json','contact-analysis/summary.json','trial/relay.stdout.jsonl','trial/worker.stdin.jsonl']
    .map(file=>({file,bytes:42,sha256:'d'.repeat(64)}));
  const r=receipt(c,inputs);assert.deepEqual(validateReceipt(r,c,inputs),c);
  for(const mutate of [x=>x.inputs[1].sha256='e'.repeat(64),x=>x.coverage.legacy_completed_policy_round=true,
    x=>x.coverage.actual_control_start_time_remaining_seconds=120,x=>x.schema='other']){
    const changed=structuredClone(r);mutate(changed);assert.throws(()=>validateReceipt(changed,c,inputs));
  }
});
