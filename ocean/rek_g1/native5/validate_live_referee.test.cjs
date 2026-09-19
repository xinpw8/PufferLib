'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),crypto=require('node:crypto'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {validatePayload,ObservationAudit,receiptKey,NULL_WHEN_UNAVAILABLE,loadNative,run}=require('./validate_live_referee.cjs');
const names=['Slip','SlipEStop','Knockdown','BeatCount','Knockout','DoubleKnockdown','DoubleKnockout'];
function fixture(options={}){
  const b=Buffer.alloc(33);b[0]=1;b[1]=1;b[2]=1;b[25]=options.mask??0;b[26]=options.seconds??0;b[27]=options.callSequence??0;b[28]=options.type??0;b.writeInt8(options.faller??-1,29);b[30]=options.points??0;
  const hash=crypto.createHash('sha256').update(b).digest('hex'),base64=b.toString('base64'),seq=options.receiptSequence??1,frame=options.frame??100;
  const qpc=options.qpc??1000000,frequency=10000000,age=.02,time=options.time??10;
  const source={event:'g1_policy_state',schema:'rek.g1_policy_source.v1',round_identity_sha256:'a'.repeat(64),round:{number:1,redo:false},clock:{qpc_ticks:qpc+age*frequency,qpc_frequency_hz:frequency},referee:{
    schema:'rek.g1_received_referee.v1',available:true,reason:'received_snapshot_applied_and_bound',source:'received_REK_FightState_33_byte_body',provenance:'ApplyFightStateSnapshot_prefix_copy_postfix_client_mirror_verification',
    authority_scope:'server_authored_packet_observed_on_client_not_server_current_state',observation_hooks_verified:true,maximum_receipt_age_seconds:.5,receipt_age_seconds:age,
    receipt_sequence:seq,lifecycle:options.lifecycle??1,receipt_qpc_ticks:qpc,receipt_qpc_frequency_hz:frequency,receipt_unity_frame:frame,receipt_unity_time:time,receipt_unity_unscaled_time:time+.5,
    wire_body_sha256:hash,wire_body_base64:base64,count_mask:b[25],count_seconds:b[26],slot0_count_active:Boolean(b[25]&1),slot1_count_active:Boolean(b[25]&2),
    call_available:b[27]>0,call_sequence:b[27],call_type:b[27]?b[28]:null,call_name:b[27]?(names[b[28]]??null):null,call_faller:b[27]?b.readInt8(29):null,call_points:b[27]?b[30]:null,
    call_observation_sequence:b[27]?(options.callId??1):null,call_sequence_transition:options.transition??(b[27]?'initial_latched_call':'empty_sequence'),call_history_censored:b[27]?(options.censored??true):null,
    packet_phase:1,packet_round_number:1,packet_round_active:true,packet_round_redo:false,packet_round_knockout_occurred:false,packet_round_result:0,server_tick:null,server_time:null,server_fight_epoch:null}};
  const key=receiptKey(hash,frame,time,time+.5),native={frequency,start_qpc:1,end_qpc:100000000,receipts:new Map([[key,[{wire_body_base64:base64,source_line:10}] ]])};return {source,native};
}
function unavailable(){const f=fixture();Object.assign(f.source.referee,{available:false,reason:'referee_snapshot_not_observed',receipt_age_seconds:null,call_available:false});for(const key of NULL_WHEN_UNAVAILABLE)f.source.referee[key]=null;return f;}
function addNative(target,other){for(const [key,value] of other.receipts)target.receipts.set(key,value);}
test('wire offsets, signed faller and exact recorder receipt are validated',()=>{
  const f=fixture({mask:3,seconds:2,callSequence:8,type:6,faller:-1,points:5});assert.equal(validatePayload(f.source,f.native).available,true);
  f.source.referee.call_faller=255;assert.throws(()=>validatePayload(f.source,f.native),/call_payload/);
});
test('zero sequence never exposes the default Slip byte as a call',()=>{
  const f=fixture();assert.equal(validatePayload(f.source,f.native).available,true);f.source.referee.call_available=true;assert.throws(()=>validatePayload(f.source,f.native),/zero_sequence/);
});
test('unavailable snapshots require explicit null data and cannot verify the hook',()=>{
  const f=unavailable(),audit=new ObservationAudit(f.native);audit.observe(f.source,1);assert.equal(audit.report().verification_passed,false);
  assert.deepEqual(audit.report().unavailable_reasons,{referee_snapshot_not_observed:1});f.source.referee.count_mask=0;assert.throws(()=>validatePayload(f.source,f.native),/not_null:count_mask/);
});
test('missing payload, unknown clocks and changed freshness budget fail closed',()=>{
  let f=fixture();f.source.referee=null;assert.throws(()=>validatePayload(f.source,f.native),/payload_missing/);
  f=fixture();f.source.referee.receipt_qpc_frequency_hz++;assert.throws(()=>validatePayload(f.source,f.native),/frequency/);
  f=fixture();f.source.referee.maximum_receipt_age_seconds=1;assert.throws(()=>validatePayload(f.source,f.native),/budget_changed/);
});
test('available age must equal QPC age and stay within 0.5 seconds',()=>{
  const f=fixture();f.source.referee.receipt_age_seconds=.01;assert.throws(()=>validatePayload(f.source,f.native),/age_qpc/);
  f.source.clock.qpc_ticks=6000001;f.source.referee.receipt_age_seconds=.5000001;assert.throws(()=>validatePayload(f.source,f.native),/freshness_budget/);
  f.source.clock.qpc_ticks=6000000;f.source.referee.receipt_age_seconds=.5;assert.equal(validatePayload(f.source,f.native).age_seconds,.5);
});
test('hash tampering, changed decoded fields and missing recorder prefixes reject',()=>{
  let f=fixture();f.source.referee.wire_body_sha256='b'.repeat(64);assert.throws(()=>validatePayload(f.source,f.native),/hash_mismatch/);
  f=fixture();f.source.referee.count_seconds=1;assert.throws(()=>validatePayload(f.source,f.native),/decoded_byte/);
  f=fixture();f.source.referee.receipt_unity_time+=.000001;assert.throws(()=>validatePayload(f.source,f.native),/no_exact_native/);
  f=fixture();f.source.referee.receipt_unity_frame++;assert.throws(()=>validatePayload(f.source,f.native),/no_exact_native/);
});
test('round, redo and capture-clock mismatches cannot bind',()=>{
  let f=fixture();f.source.round.number=2;assert.throws(()=>validatePayload(f.source,f.native),/round_or_redo/);
  f=fixture();f.source.round.redo=true;assert.throws(()=>validatePayload(f.source,f.native),/round_or_redo/);
  f=fixture();f.native.end_qpc=999999;assert.throws(()=>validatePayload(f.source,f.native),/outside_native/);
});
test('repeated policy snapshots and periodic latched packets remain one call',()=>{
  const a=fixture({callSequence:1}),b=fixture({receiptSequence:2,qpc:2000000,frame:101,time:10.1,callSequence:1,transition:'repeated_latched_call'});addNative(a.native,b.native);
  const audit=new ObservationAudit(a.native);audit.observe(a.source,1);audit.observe(a.source,2);audit.observe(b.source,3);
  const r=audit.report();assert.equal(r.available_source_count,3);assert.equal(r.unique_bridge_receipts,2);assert.equal(r.unique_latched_call_ids,1);assert.equal(r.calls[0].policy_observations,3);
  assert.equal(r.repeated_policy_observations_of_cached_receipt,1);
});
test('a repeated latched call cannot acquire a new call identity',()=>{
  const a=fixture({callSequence:1}),b=fixture({receiptSequence:2,qpc:2000000,frame:101,time:10.1,callSequence:1,callId:2,transition:'repeated_latched_call'});addNative(a.native,b.native);
  const audit=new ObservationAudit(a.native);audit.observe(a.source,1);assert.throws(()=>audit.observe(b.source,2),/repeated_call_assigned_new/);
});
test('255-to-1 wrap is separate while gaps remain censored',()=>{
  const a=fixture({callSequence:255}),b=fixture({receiptSequence:2,qpc:2000000,frame:101,time:10.1,callSequence:1,callId:2,transition:'observed_255_to_1_wrap',censored:false});addNative(a.native,b.native);
  const audit=new ObservationAudit(a.native);audit.observe(a.source,1);audit.observe(b.source,2);assert.equal(audit.report().unique_latched_call_ids,2);assert.equal(audit.report().cache_censored_call_ids,1);
  const gap=fixture({callSequence:4,transition:'sequence_gap_censored',censored:false});assert.throws(()=>validatePayload(gap.source,gap.native),/uncensored_discontinuity/);
});
test('stale unavailable age is diagnostic while all receipt fields stay null',()=>{
  const f=unavailable();f.source.referee.reason='referee_receipt_stale';f.source.referee.receipt_age_seconds=.6;assert.equal(validatePayload(f.source,f.native).available,false);
});
test('streamed files bind exact receipts, preserve source hashes and refuse incomplete captures',async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-referee-validator-')),trial=path.join(dir,'trial'),output=path.join(dir,'output'),nativeFile=path.join(dir,'native.jsonl');
  fs.mkdirSync(trial);const f=fixture(),r=f.source.referee,b=Buffer.from(r.wire_body_base64,'base64'),d={};
  for(const [key,offset] of Object.entries({phase:0,round_number:1,round_active:2,is_redo:3,knockout_occurred:12,round_result:13,rounds_won_0:15,rounds_won_1:16,fight_result:17,format:19,human_slot_mask:20,fault_mask:22,fault_stress_0:23,fault_stress_1:24,referee_count_mask:25,referee_count_seconds:26,referee_call_sequence:27,referee_call_type:28,referee_call_points:30,ai_level:31,decided_winner_bits:32}))d[key]=b[offset];
  for(const [key,offset] of Object.entries({round_winner:14,fight_winner:18,champion_slot:21,referee_call_faller:29}))d[key]=b.readInt8(offset);
  Object.assign(d,{hits_0:b.readInt16LE(8),hits_1:b.readInt16LE(10),time_remaining:b.readFloatLE(4),referee_call_name:names[b[28]]});
  const packet={event:'raw_fight_state_packet',wire_body_base64:r.wire_body_base64,wire_body_sha256:r.wire_body_sha256,wire_body_bytes:33,decoded:d,
    unity_frame:r.receipt_unity_frame,unity_time:r.receipt_unity_time,unity_unscaled_time:r.receipt_unity_unscaled_time,monotonic_receipt_time:10.6,client_fixed_tick_at_observation:1};
  const rows=[{event:'capture_start',pid:123,scene:'Arena',stopwatch_frequency_hz:10000000,stopwatch_timestamp_ticks:1,initial_state:{round:{number:1}}},packet,
    {event:'capture_end',stopwatch_timestamp_ticks:100000000,capture_error_count:0,raw_fight_state_packet_count:1}];
  const relay=path.join(trial,'relay.stdout.jsonl');
  try{
    fs.writeFileSync(relay,JSON.stringify(f.source)+'\n');fs.writeFileSync(nativeFile,rows.map(JSON.stringify).join('\n')+'\n');
    const report=await run(trial,nativeFile,output);assert.equal(report.verification_passed,true);assert.equal(report.unique_bridge_receipts,1);assert.equal(report.native.calls.length,0);
    assert.equal(report.inputs[0].sha256,crypto.createHash('sha256').update(fs.readFileSync(relay)).digest('hex'));
    await assert.rejects(run(trial,nativeFile,output),/output_exists/);
    rows[2].raw_fight_state_packet_count=2;fs.writeFileSync(nativeFile,rows.map(JSON.stringify).join('\n')+'\n');await assert.rejects(loadNative(nativeFile),/packet_count_mismatch/);
  }finally{
    const report=path.join(output,'live-referee-validation.json');if(fs.existsSync(report))fs.unlinkSync(report);if(fs.existsSync(output))fs.rmdirSync(output);
    if(fs.existsSync(relay))fs.unlinkSync(relay);fs.rmdirSync(trial);if(fs.existsSync(nativeFile))fs.unlinkSync(nativeFile);fs.rmdirSync(dir);
  }
});
