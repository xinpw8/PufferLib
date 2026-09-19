'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {validateRefereePacket,extractCalls,joinFivePointScores,NAMES}=require('./native_referee_data.cjs');
function packet(t,seq=0,type=0,faller=-1,mask=0,seconds=0,points=0,round=1) {
  return {source_line:Math.round(t*100)+1,unity_frame:Math.round(t*60),unity_time:t,unity_unscaled_time:t,
    monotonic_receipt_time:t,client_fixed_tick_at_observation:Math.round(t*500),fight_epoch:1,
    decoded:{round_number:round,is_redo:0,referee_call_sequence:seq,referee_call_type:type,referee_call_name:NAMES[type],
      referee_call_faller:faller,referee_call_points:points,referee_count_mask:mask,referee_count_seconds:seconds,hits_0:0,hits_1:0}};
}
const extract=p=>extractCalls('policy/test','session-test',p);
function score(t,slot=0,counter=5){return {source_line:1000,score_id:'S1',monotonic_receipt_time:t,
  unity_time:t,unity_unscaled_time:t,unity_frame:60,client_fixed_tick_at_observation:500,
  decoded:{points_awarded:5,fighter_index:slot,new_hit_count:counter}};}
test('empty sequence zero and repeated latched snapshots do not invent Slip incidents',()=>{
  const r=extract([packet(0),packet(.1),packet(1,1,0,1,2),packet(1.1,1,0,1,2),packet(2,1,0,1,2,1)]);
  assert.equal(r.calls.length,1);assert.equal(r.calls[0].repeated_snapshots,2);assert.equal(r.calls[0].observed_new_call,true);
  assert.equal(r.counts.length,1);assert.equal(r.counts[0].right_censored,true);assert.deepEqual(r.counts[0].received_count_seconds,[0,1]);
});
test('explicit Slip count 0,1,2 then Knockout resolves one countout and joins exact counter',()=>{
  const p=packet(4,2,4,1,0,0,5);p.decoded.hits_0=5;
  const r=extract([packet(0),packet(1,1,0,1,2),packet(2,1,0,1,2,1),packet(3,1,0,1,2,2),p]);
  assert.equal(r.counts[0].explicit_countout,true);assert.equal(r.counts[0].duration_receipt_seconds,3);
  assert.equal(r.counts[0].resolution_call_name,'Knockout');assert.deepEqual(r.counts[0].received_count_seconds,[0,1,2]);
  const awards=joinFivePointScores(r.calls,[score(3.99)]);assert.equal(awards[0].association,'unique_receipt_counter_and_referee_agreement');
  assert.equal(awards[0].faller_slot,1);assert.equal(r.calls[1].scorer_slot,0);assert.equal(awards[0].causal_hit_id,null);
});
test('zero reset and round change create distinct call namespaces',()=>{
  const r=extract([packet(0),packet(1,1,0,1,2),packet(2),packet(3,1,0,1,2),packet(4,1,0,1,2,0,0,2)]);
  assert.equal(new Set(r.calls.map(c=>c.call_id)).size,3);assert.equal(r.calls[2].left_censored,true);
  assert.match(r.calls[1].call_id,/epoch1/);assert.equal(r.counts[0].end_reason,'zero_sequence_reset');
});
test('255 to 1 wraps without reusing call ID or falsely resetting count',()=>{
  const r=extract([packet(0),packet(1,255,0,1,2),packet(4,1,4,1,0,0,5)]);
  assert.match(r.calls[1].call_id,/wrap1:seq1/);assert.equal(r.calls[1].observed_new_call,true);
  assert.equal(r.counts[0].explicit_countout,true);
});
test('first nonzero snapshot and uncertain decreasing sequence remain censored',()=>{
  const r=extract([packet(0,4,0,1,2),packet(1,2,0,1,2)]);
  assert.ok(r.calls.every(c=>c.left_censored));assert.equal(r.calls[1].sequence_reset_basis,'sequence_decrease_uncertain_reset');
  assert.ok(r.counts.every(c=>c.left_censored));
});
test('reused sequence with conflicting call payload fails closed',()=>{
  assert.throws(()=>extract([packet(0),packet(1,1,0,1,2),packet(2,1,4,1,0,0,5)]),/same_sequence_conflicting/);
});
test('ambiguous score duplicates, wrong recipient/counter, and distant receipts do not get assigned',()=>{
  const p=packet(2,1,4,1,0,0,5);p.decoded.hits_0=5;const calls=extract([packet(0),p]).calls;
  assert.equal(joinFivePointScores(calls,[score(2),score(2.01)])[0].association,'ambiguous');
  for(const s of [score(2,1),score(2,0,6),score(3)])assert.equal(joinFivePointScores(calls,[s])[0].associated_referee_call_id,null);
});
test('DoubleKnockout can match one score per slot but never invents single-faller identity',()=>{
  const p=packet(4,2,6,-1,0,0,5);p.decoded.hits_0=5;p.decoded.hits_1=5;
  const r=extract([packet(0),packet(1,1,5,-1,3),p]);
  assert.equal(r.counts.filter(c=>c.explicit_countout).length,2);
  const awards=joinFivePointScores(r.calls,[score(4,0),score(4,1)]);
  assert.ok(awards.every(a=>a.association==='unique_receipt_counter_and_referee_agreement'&&a.faller_slot===null));
  assert.equal(r.calls[1].scorer_slot,null);
});
test('BeatCount clears a count without misclassifying it as a countout',()=>{
  const r=extract([packet(0),packet(1,1,2,1,2),packet(2,2,3,1,0)]);
  assert.equal(r.counts[0].explicit_countout,false);assert.equal(r.counts[0].resolution_call_name,'BeatCount');
});
test('exact packed bytes and SHA-256 must agree with every decoded referee field',()=>{
  const r=packet(0),b=Buffer.alloc(33);b[1]=1;b[14]=255;b[18]=255;b[21]=255;b[29]=255;
  Object.assign(r.decoded,{phase:0,round_active:0,time_remaining:0,knockout_occurred:0,round_result:0,round_winner:-1,
    rounds_won_0:0,rounds_won_1:0,fight_result:0,fight_winner:-1,format:0,human_slot_mask:0,champion_slot:-1,
    fault_mask:0,fault_stress_0:0,fault_stress_1:0,ai_level:0,decided_winner_bits:0});
  r.wire_body_bytes=33;r.wire_body_base64=b.toString('base64');r.wire_body_sha256=crypto.createHash('sha256').update(b).digest('hex');
  validateRefereePacket(r);r.decoded.referee_count_mask=2;assert.throws(()=>validateRefereePacket(r),/byte_mismatch/);
  r.decoded.referee_count_mask=0;r.wire_body_sha256='0'.repeat(64);assert.throws(()=>validateRefereePacket(r),/invalid_referee_wire/);
});
