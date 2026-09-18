'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {analyzeRecords,packetMatch,MOVE_ORDER}=require('./analyze_live_contacts.cjs');
const SHA='d'.repeat(64),ROUND='a'.repeat(64);
function state(seq,time,frame,terminal=false,slot=0){
  return {event:'g1_policy_state',schema:'rek.g1_policy_source.v1',observation_sequence:seq,round_identity_sha256:ROUND,local_slot:slot,
    clock:{unity_time:time,unity_frame:frame,qpc_ticks:Math.round(time*1000),qpc_frequency_hz:1000,utc:new Date(time*1000).toISOString()},
    round:{number:1,duration:120,time_remaining:terminal?0:120,active:!terminal,result_value:terminal?1:0,winner_index:terminal?0:-1,clean_hits:terminal?[2,0]:[0,0]},
    input:{},fighters:[{root_position_xyz:[0,.8,0],root_rotation_xyzw:[0,0,0,1]},
      {root_position_xyz:[.3,.8,.4],root_rotation_xyzw:[0,1,0,0]}]};
}
function fixture(){
  const request={type:'policy_action',request_id:'action-1',round_identity_sha256:ROUND,observation_sequence:10,action:16};
  const ack={...request,event:'g1_policy_action',applied:true,execute_move_returned:true,reason:'local_return',clock:{unity_time:10.01,unity_frame:100}};
  const dispatch={...request,event:'g1_policy_dispatch',move_index:6,send_method_returned:true,clock:{unity_time:10.02,unity_frame:100}};
  const score={event:'raw_score_packet',unity_time:10.1,unity_frame:101,monotonic_receipt_time:30.1,client_fixed_tick_at_observation:6,
    decoded:{fighter_index:0,new_hit_count:2,points_awarded:2}};
  const hit={event:'raw_hit_packet',unity_time:10.1,unity_frame:101,monotonic_receipt_time:30.1005,client_fixed_tick_at_observation:6,
    decoded:{position_xyz:[.15,.9,.2],surface_normal_xyz:[1,0,0],relative_speed:3,is_kick:1}};
  return {config:{checkpoint_sha256:SHA},summary:{checkpoint_sha256:SHA,stop_reason:'source_round_terminal',opponent:{sparring_bot_number:1}},
    worker:[{type:'ready',native_cuda:true,checkpoint_sha256:SHA}],requests:[request],
    relay:[state(10,10,100),ack,dispatch,state(11,10.1,101),state(12,10.2,102,true)],
    native:[{event:'capture_start',pid:1,utc:new Date(10000).toISOString(),stopwatch_timestamp_ticks:10000,stopwatch_frequency_hz:1000,scope:{local_fighter_index:0}},
      ...[state(10,10,100),state(11,10.1,101),state(12,10.2,102,true)].map(p=>({event:'root_pose_sample',round_number:1,unity_frame:p.clock.unity_frame,unity_time:p.clock.unity_time,stopwatch_timestamp_ticks:p.clock.qpc_ticks,
        fighter_0_root:{world_position_xyz:p.fighters[0].root_position_xyz},fighter_1_root:{world_position_xyz:p.fighters[1].root_position_xyz}})),
      {event:'outbound_request_projection',message:'REK_Move',request_sequence:1,message_request_sequence:1,unity_frame:100,unity_realtime_since_startup:30.02,move_index_wire_uint8:6},score,hit,
      {event:'raw_fight_state_packet',unity_time:10.2,unity_frame:102,decoded:{round_number:1,round_active:0,round_result:1,round_winner:0,hits_0:2,hits_1:0}},
      {event:'capture_end',stopwatch_timestamp_ticks:10200,raw_score_packet_count:1,raw_hit_packet_count:1,capture_error_count:0}]};
}
test('authentic win, cumulative points and request geometry with noncausal association',()=>{
  const r=analyzeRecords(fixture());assert.equal(r.summary.outcome,'win');assert.deepEqual(r.summary.terminal_awarded_points_by_slot,[2,0]);
  assert.deepEqual(r.summary.reconciled_full_round_points_by_slot,[2,0]);assert.equal(r.attacks[0].requested_move_index,6);
  assert.equal(r.attacks[0].request_geometry.root_distance_ground_xz_m,.5);
  assert.ok(Math.abs(r.attacks[0].request_geometry.local_bearing_to_opponent_rad-Math.atan2(.4,.3))<1e-12);
  assert.deepEqual(r.hits[0].position_xyz,[.15,.9,.2]);assert.equal(r.awards[0].unique_same_frame_hit_association,true);
  assert.ok(Math.abs(r.awards[0].candidate_requests[0].seconds-.08)<1e-10);
  assert.equal(r.awards[0].candidate_requests[0].basis,'native_client_monotonic_receipt_minus_request');
  assert.equal(r.awards[0].causal_request_id,null);assert.equal(r.attacks[0].contact_outcome,'unknown');
});
test('no recorder still measures replicated outcome and weighted points, never fabricates zero score events',()=>{
  const f=fixture();f.native=null;const r=analyzeRecords(f);
  assert.equal(r.summary.outcome,'win');assert.deepEqual(r.summary.terminal_awarded_points_by_slot,[2,0]);
  assert.equal(r.summary.score_events,null);assert.equal(r.summary.observed_point_awards_by_slot,null);
  assert.equal(r.summary.reconciled_full_round_points_by_slot,null);assert.equal(r.attacks[0].contact_outcome,'unknown');
});
test('missing unreliable hit is unknown, not a missed attack',()=>{
  const f=fixture();f.native=f.native.filter(x=>x.event!=='raw_hit_packet');f.native.at(-1).raw_hit_packet_count=0;
  const r=analyzeRecords(f);assert.equal(r.hits.length,0);assert.equal(r.awards[0].unique_same_frame_hit_association,false);
  assert.equal(r.attacks[0].contact_outcome,'unknown');assert.deepEqual(r.summary.reconciled_full_round_points_by_slot,[2,0]);
});
test('ambiguous score/hit receipt pairing retains all candidates and does not choose latest request',()=>{
  const f=fixture();f.native.push({...f.native.find(x=>x.event==='raw_hit_packet')});f.native.find(x=>x.event==='capture_end').raw_hit_packet_count=2;
  const extra={...f.requests[0],request_id:'action-2'};f.requests.push(extra);
  f.relay.push({...f.relay.find(x=>x.event==='g1_policy_dispatch'),request_id:'action-2'});
  const r=analyzeRecords(f);assert.equal(r.awards[0].matching_hit_indices.length,2);assert.equal(r.awards[0].unique_same_frame_hit_association,false);
  assert.deepEqual(r.awards[0].window_sensitivity[0].request_ids,['action-1','action-2']);assert.equal(r.awards[0].causal_request_id,null);
});
test('local false return can coexist with later native dispatch, neither is server acceptance',()=>{
  const f=fixture();f.relay.find(x=>x.event==='g1_policy_action').applied=false;
  const r=analyzeRecords(f);assert.equal(r.summary.locally_applied_attack_returns,0);assert.equal(r.summary.native_dispatched_attack_requests,1);
  assert.equal(r.attacks[0].server_acceptance,'unknown');assert.equal(r.attacks[0].executed_move_index,null);
});
test('five-point awards remain separate with unresolved cause',()=>{
  const f=fixture();const score=f.native.find(x=>x.event==='raw_score_packet');score.decoded.points_awarded=score.decoded.new_hit_count=5;
  const r=analyzeRecords(f);assert.equal(r.awards[0].award_class,'five_point_award_cause_unresolved');assert.deepEqual(r.awards[0].candidate_requests,[]);
  assert.equal(r.summary.five_point_awards,1);assert.equal(r.summary.reconciled_full_round_points_by_slot,null);
});
test('missing terminal or contradictory winners cannot produce a measured win',()=>{
  const f=fixture();f.relay=f.relay.filter(x=>!(x.event==='g1_policy_state'&&x.round.active===false));
  f.native=f.native.filter(x=>x.event!=='raw_fight_state_packet');assert.equal(analyzeRecords(f).summary.outcome,'unknown');
  const g=fixture();g.native.find(x=>x.event==='raw_fight_state_packet').decoded.round_winner=1;
  assert.equal(analyzeRecords(g).summary.outcome,'unknown');
});
test('local slot1 interprets the same terminal winner as a loss',()=>{
  const f=fixture();for(const x of f.relay)if(x.event==='g1_policy_state')x.local_slot=1;
  f.native[0].scope.local_fighter_index=1;const r=analyzeRecords(f);assert.equal(r.summary.outcome,'loss');
  assert.deepEqual(r.awards[0].candidate_requests,[]);
});
test('native cumulative-counter gap and incomplete capture prevent full packet reconciliation',()=>{
  let f=fixture();f.native.find(x=>x.event==='raw_score_packet').decoded.new_hit_count=4;
  assert.equal(analyzeRecords(f).summary.reconciled_full_round_points_by_slot,null);
  f=fixture();f.native.pop();assert.equal(analyzeRecords(f).summary.native_capture_complete,false);
});
test('counter-name ambiguity does not overwrite conflicting terminal point counters',()=>{
  const f=fixture();f.native.find(x=>x.event==='raw_fight_state_packet').decoded.hits_0=3;
  const r=analyzeRecords(f);assert.equal(r.summary.outcome,'win');assert.equal(r.summary.terminal_awarded_points_by_slot,null);
});
test('different fixed tick or excessive receipt lag does not create unique hit association',()=>{
  const f=fixture(),s=f.native.find(x=>x.event==='raw_score_packet'),h=f.native.find(x=>x.event==='raw_hit_packet');
  assert.equal(packetMatch(s,{...h,client_fixed_tick_at_observation:7}),false);
  assert.equal(packetMatch(s,{...h,monotonic_receipt_time:31}),false);
});
test('identity mismatches fail closed; move mapping covers all17 native IDs',()=>{
  let f=fixture();f.worker[0].checkpoint_sha256='e'.repeat(64);assert.throws(()=>analyzeRecords(f),/worker_checkpoint/);
  f=fixture();f.relay.find(x=>x.event==='g1_policy_dispatch').move_index=7;assert.throws(()=>analyzeRecords(f),/dispatch_identity/);
  f=fixture();f.requests[0].observation_sequence=99;assert.throws(()=>analyzeRecords(f),/source_missing/);
  assert.deepEqual([...MOVE_ORDER].sort((a,b)=>a-b),Array.from({length:17},(_,i)=>i));
});
test('tie enum overrides a default winner index and terminal alone is not full policy coverage',()=>{
  const f=fixture();f.relay.at(-1).round.result_value=3;f.native.find(x=>x.event==='raw_fight_state_packet').decoded.round_result=3;
  const r=analyzeRecords(f);assert.equal(r.summary.outcome,'draw');assert.equal(r.summary.terminal_observed,true);assert.equal(r.summary.completed_policy_round,false);
});
test('integer counter truncates each float award while preserving raw awarded total',()=>{
  const f=fixture();f.native.find(x=>x.event==='raw_score_packet').decoded.points_awarded=2.75;
  const r=analyzeRecords(f);assert.deepEqual(r.summary.observed_point_awards_by_slot,[2.75,0]);assert.deepEqual(r.summary.reconciled_full_round_points_by_slot,[2,0]);
});
test('different capture time, round or geometry fails closed',()=>{
  let f=fixture();f.native[0].stopwatch_timestamp_ticks+=3000;assert.throws(()=>analyzeRecords(f),/native_/);
  f=fixture();f.native.find(x=>x.event==='root_pose_sample').round_number=2;assert.throws(()=>analyzeRecords(f),/native_root_round/);
  f=fixture();for(const x of f.native)if(x.event==='root_pose_sample')x.fighter_0_root.world_position_xyz=[9,9,9];assert.throws(()=>analyzeRecords(f),/native_pose_clock/);
});
test('an early terminal knockout with beginning and local control coverage completes a policy round',()=>{
  const f=fixture();f.relay.at(-1).round.result_value=2;f.relay.at(-1).round.time_remaining=119.8;
  f.native.find(x=>x.event==='raw_fight_state_packet').decoded.round_result=2;
  const request={...f.requests[0],request_id:'action-2',observation_sequence:11,action:1};f.requests.push(request);
  f.relay.push({...request,event:'g1_policy_action',applied:true,clock:{unity_time:10.15,unity_frame:101}});
  assert.equal(analyzeRecords(f).summary.completed_policy_round,true);
});
