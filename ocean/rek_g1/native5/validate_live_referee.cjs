#!/usr/bin/env node
'use strict';

// Offline wire/receipt validation only. No bridge connections or game commands.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const {validateRefereePacket,NAMES,extractCalls}=require('./native_referee_data.cjs');
const check=(ok,why)=>{if(!ok)throw Error(why);},sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const finite=Number.isFinite;
const NULL_WHEN_UNAVAILABLE=['receipt_sequence','lifecycle','receipt_qpc_ticks','receipt_qpc_frequency_hz','receipt_unity_frame','receipt_unity_time','receipt_unity_unscaled_time',
  'wire_body_sha256','wire_body_base64','count_mask','count_seconds','slot0_count_active','slot1_count_active','call_sequence','call_type','call_name','call_faller','call_points',
  'call_observation_sequence','call_sequence_transition','call_history_censored','packet_phase','packet_round_number','packet_round_active','packet_round_redo','packet_round_knockout_occurred','packet_round_result'];
const CALL_FIELDS=['call_type','call_name','call_faller','call_points','call_observation_sequence','call_history_censored'];
const TRANSITIONS=new Set(['empty_sequence','initial_latched_call','repeated_latched_call','same_sequence_payload_changed_censored','observed_255_to_1_wrap','sequence_gap_censored','sequence_decrease_censored','changed_received_sequence']);
const CENSORED_TRANSITIONS=new Set(['initial_latched_call','same_sequence_payload_changed_censored','sequence_gap_censored','sequence_decrease_censored']);
function tick(value){check(Number.isSafeInteger(value)&&value>0,'invalid_qpc_integer');return BigInt(value);}
const receiptKey=(hash,frame,time,unscaled)=>JSON.stringify([hash,frame,time,unscaled]);
function decode(body64,hash){
  check(typeof body64==='string'&&typeof hash==='string'&&/^[a-f0-9]{64}$/.test(hash),'invalid_wire_encoding');
  const b=Buffer.from(body64,'base64');check(b.length===33&&b.toString('base64')===body64&&sha(b)===hash,'wire_bytes_or_hash_mismatch');
  check(b[0]<=6&&b[2]<=1&&b[3]<=1&&b[12]<=1&&b[13]<=4&&b[25]<=3,'wire_schema_value_invalid');return b;
}
function validatePayload(source,native){
  check(source.event==='g1_policy_state'&&source.schema==='rek.g1_policy_source.v1','policy_source_schema');
  check(typeof source.round_identity_sha256==='string'&&/^[a-f0-9]{64}$/.test(source.round_identity_sha256),'round_identity_missing');
  const r=source.referee;check(r&&r.schema==='rek.g1_received_referee.v1'&&typeof r.available==='boolean','referee_payload_missing_or_wrong_schema');
  check(typeof r.reason==='string'&&r.reason.length>0&&typeof r.observation_hooks_verified==='boolean','referee_status_invalid');
  check(r.source==='received_REK_FightState_33_byte_body'&&r.provenance==='ApplyFightStateSnapshot_prefix_copy_postfix_client_mirror_verification'&&
    r.authority_scope==='server_authored_packet_observed_on_client_not_server_current_state','referee_provenance_mismatch');
  check(r.maximum_receipt_age_seconds===.5,'freshness_budget_changed');
  for(const key of ['server_tick','server_time','server_fight_epoch'])check(r[key]===null,'invented_server_clock_or_epoch');
  if(!r.available){
    for(const key of NULL_WHEN_UNAVAILABLE)check(r[key]===null,'unavailable_field_not_null:'+key);
    check(r.call_available===false,'unavailable_call_available');
    check(r.receipt_age_seconds===null||(finite(r.receipt_age_seconds)&&r.receipt_age_seconds>=0),'unavailable_age_invalid');
    return {available:false,reason:r.reason};
  }
  check(r.observation_hooks_verified&&r.reason==='received_snapshot_applied_and_bound','available_without_verified_hook');
  const b=decode(r.wire_body_base64,r.wire_body_sha256);
  for(const [key,offset] of Object.entries({packet_phase:0,packet_round_number:1,packet_round_result:13,count_mask:25,count_seconds:26,call_sequence:27}))
    check(r[key]===b[offset],'decoded_byte_mismatch:'+key);
  for(const [key,offset] of Object.entries({packet_round_active:2,packet_round_redo:3,packet_round_knockout_occurred:12}))
    check(r[key]===(b[offset]!==0),'decoded_boolean_mismatch:'+key);
  check(r.packet_round_number===source.round?.number&&r.packet_round_redo===source.round?.redo,'source_round_or_redo_mismatch');
  check(r.slot0_count_active===Boolean(b[25]&1)&&r.slot1_count_active===Boolean(b[25]&2),'count_slot_bits_mismatch');
  check(Number.isSafeInteger(r.receipt_sequence)&&r.receipt_sequence>0&&Number.isSafeInteger(r.lifecycle)&&r.lifecycle>0,'receipt_identity_invalid');
  check(Number.isSafeInteger(r.receipt_qpc_frequency_hz)&&r.receipt_qpc_frequency_hz>0&&r.receipt_qpc_frequency_hz===source.clock?.qpc_frequency_hz&&
    r.receipt_qpc_frequency_hz===native.frequency,'qpc_frequency_mismatch');
  const age=Number(tick(source.clock.qpc_ticks)-tick(r.receipt_qpc_ticks))/r.receipt_qpc_frequency_hz;
  if(native.start_qpc!==undefined)check(r.receipt_qpc_ticks>=native.start_qpc&&r.receipt_qpc_ticks<=native.end_qpc,'receipt_outside_native_process_capture');
  check(age>=0&&age<=.5&&finite(r.receipt_age_seconds)&&r.receipt_age_seconds>=0&&r.receipt_age_seconds<=.5,'available_receipt_outside_freshness_budget');
  check(Math.abs(age-r.receipt_age_seconds)<=1e-9,'receipt_age_qpc_mismatch');
  check(Number.isSafeInteger(r.receipt_unity_frame)&&r.receipt_unity_frame>=0&&finite(r.receipt_unity_time)&&r.receipt_unity_time>=0&&
    finite(r.receipt_unity_unscaled_time)&&r.receipt_unity_unscaled_time>=0,'receipt_unity_clock_invalid');
  check(TRANSITIONS.has(r.call_sequence_transition),'call_transition_invalid');
  if(b[27]===0){check(r.call_available===false&&r.call_sequence_transition==='empty_sequence','zero_sequence_is_not_empty');for(const key of CALL_FIELDS)check(r[key]===null,'empty_call_field_not_null:'+key);}
  else{
    check(r.call_available===true&&r.call_type===b[28]&&r.call_faller===b.readInt8(29)&&r.call_points===b[30]&&r.call_name===(NAMES[b[28]]??null),'call_payload_mismatch');
    check(Number.isSafeInteger(r.call_observation_sequence)&&r.call_observation_sequence>0&&typeof r.call_history_censored==='boolean','call_identity_or_censoring_invalid');
    check(r.call_sequence_transition!=='empty_sequence','nonzero_sequence_empty_transition');
    if(CENSORED_TRANSITIONS.has(r.call_sequence_transition))check(r.call_history_censored===true,'uncensored_discontinuity');
    if(['changed_received_sequence','observed_255_to_1_wrap'].includes(r.call_sequence_transition))check(r.call_history_censored===false,'censored_contiguous_transition');
  }
  const key=receiptKey(r.wire_body_sha256,r.receipt_unity_frame,r.receipt_unity_time,r.receipt_unity_unscaled_time),matches=native.receipts.get(key);
  check(matches?.length>0,'no_exact_native_receipt_match');
  check(matches.every(p=>p.wire_body_base64===r.wire_body_base64),'native_matching_hash_bytes_disagree');
  return {available:true,payload:r,wire:b,receipt_key:key,native_matches:matches.map(p=>p.source_line),age_seconds:age};
}
const inc=(map,key)=>{map[key]=(map[key]||0)+1;};
class ObservationAudit{
  constructor(native){this.native=native;this.sources=0;this.available=0;this.unavailable={};this.masks={};this.receiptMasks={};this.transitions={};this.receipts=new Map();this.calls=new Map();this.rounds=new Set();this.ageMin=Infinity;this.ageMax=0;this.last=null;this.repeatedSources=0;this.ambiguousMatches=0;}
  observe(source,line){
    const value=validatePayload(source,this.native);this.sources++;this.rounds.add(source.round_identity_sha256);
    if(!value.available){inc(this.unavailable,value.reason);return;}
    this.available++;const r=value.payload;inc(this.masks,r.count_mask);this.ageMin=Math.min(this.ageMin,value.age_seconds);this.ageMax=Math.max(this.ageMax,value.age_seconds);
    const key=source.round_identity_sha256+':'+r.lifecycle+':'+r.receipt_sequence;
    const stable={round:source.round_identity_sha256,lifecycle:r.lifecycle,sequence:r.receipt_sequence,qpc:r.receipt_qpc_ticks,key:value.receipt_key,call_id:r.call_observation_sequence,
      call_payload:JSON.stringify([r.call_sequence,r.call_type,r.call_faller,r.call_points]),
      transition:r.call_sequence_transition,censored:r.call_history_censored};
    if(this.last){check(r.receipt_sequence>=this.last.sequence&&r.lifecycle>=this.last.lifecycle&&r.receipt_qpc_ticks>=this.last.qpc,'receipt_order_regressed');
      if(r.receipt_sequence===this.last.sequence)check(r.lifecycle===this.last.lifecycle&&stable.round===this.last.round&&stable.key===this.last.key,'receipt_sequence_reused');
      if(r.call_available&&r.lifecycle===this.last.lifecycle&&r.receipt_sequence===this.last.sequence+1&&stable.call_payload===this.last.call_payload)
        check(r.call_observation_sequence===this.last.call_id&&r.call_sequence_transition==='repeated_latched_call'&&r.call_history_censored===this.last.censored,'repeated_call_assigned_new_identity');
    }
    const previous=this.receipts.get(key);
    if(previous){check(JSON.stringify(previous)===JSON.stringify(stable),'cached_receipt_changed');this.repeatedSources++;}
    else{this.receipts.set(key,stable);inc(this.receiptMasks,r.count_mask);inc(this.transitions,r.call_sequence_transition);this.ambiguousMatches+=value.native_matches.length>1;}
    if(r.call_available){const callKey=source.round_identity_sha256+':'+r.lifecycle+':'+r.call_observation_sequence;
      const signature=JSON.stringify([r.call_sequence,r.call_type,r.call_faller,r.call_points,r.call_history_censored]);const prior=this.calls.get(callKey);
      if(prior){check(prior.signature===signature,'latched_call_identity_changed');prior.policy_observations++;}
      else this.calls.set(callKey,{call_id:callKey,signature,sequence:r.call_sequence,type:r.call_type,name:r.call_name,faller:r.call_faller,points:r.call_points,
        cache_history_censored:r.call_history_censored,first_seen_transition:r.call_sequence_transition,first_source_line:line,first_receipt_sequence:r.receipt_sequence,
        first_receipt_unity_frame:r.receipt_unity_frame,first_receipt_unity_time:r.receipt_unity_time,policy_observations:1,
        validation_observation_left_censored:this.last===null||this.last.round!==source.round_identity_sha256||this.last.lifecycle!==r.lifecycle||r.call_sequence_transition==='repeated_latched_call'});
    }
    this.last=stable;
  }
  report(){const calls=[...this.calls.values()].map(({signature,...c})=>c);return {
    verification_passed:this.available>0,source_count:this.sources,available_source_count:this.available,unavailable_source_count:this.sources-this.available,unavailable_reasons:this.unavailable,
    round_identities:[...this.rounds],unique_bridge_receipts:this.receipts.size,unique_native_receipt_keys:new Set([...this.receipts.values()].map(r=>r.key)).size,
    repeated_policy_observations_of_cached_receipt:this.repeatedSources,receipts_with_multiple_exact_native_matches:this.ambiguousMatches,
    receipt_age_seconds:{minimum:this.available?this.ageMin:null,maximum:this.available?this.ageMax:null,maximum_allowed:.5},
    count_mask_policy_observations:this.masks,count_mask_unique_bridge_receipts:this.receiptMasks,call_transitions_unique_bridge_receipts:this.transitions,
    unique_latched_call_ids:calls.length,cache_censored_call_ids:calls.filter(c=>c.cache_history_censored).length,
    validation_left_censored_call_ids:calls.filter(c=>c.validation_observation_left_censored).length,calls,
    semantic_limit:'Call IDs deduplicate latched observations. Initial/gapped calls are censored; repeated payloads are never counted as new calls. No server-current-state, attacker, executed-action or terminal-KO inference.'};}
}
async function scan(file,visit){
  const before=fs.statSync(file),digest=crypto.createHash('sha256'),input=fs.createReadStream(file);input.on('data',b=>digest.update(b));let line=0;
  for await(const raw of readline.createInterface({input,crlfDelay:Infinity})){line++;if(!raw.trim())continue;let r;try{r=JSON.parse(raw);}catch{throw Error('invalid_json_line_'+line);}try{visit(r,line);}catch(e){throw Error(path.basename(file)+':'+line+':'+e.message);}}
  const after=fs.statSync(file);check(before.size===after.size&&before.mtimeMs===after.mtimeMs,'input_changed_during_validation');
  return {file:path.resolve(file),bytes:after.size,lines:line,sha256:digest.digest('hex')};
}
async function loadNative(file){
  const receipts=new Map(),packets=[];let header=null,end=null,errors=0;
  const provenance=await scan(file,(r,line)=>{
    if(r.event==='capture_start'){check(header===null,'multiple_native_capture_starts');header=r;check(Number.isSafeInteger(r.pid)&&r.pid>0&&r.scene==='Arena'&&Number.isSafeInteger(r.stopwatch_frequency_hz)&&r.stopwatch_frequency_hz>0,'native_capture_header_invalid');}
    if(r.event==='capture_end'){check(end===null,'multiple_native_capture_ends');end=r;}
    if(r.event==='capture_error')errors++;
    if(r.event!=='raw_fight_state_packet')return;
    validateRefereePacket(r);decode(r.wire_body_base64,r.wire_body_sha256);check(header!==null&&end===null,'native_packet_outside_capture');
    const p={...r,source_line:line};packets.push(p);const key=receiptKey(r.wire_body_sha256,r.unity_frame,r.unity_time,r.unity_unscaled_time);if(!receipts.has(key))receipts.set(key,[]);receipts.get(key).push(p);
  });
  check(header&&end&&errors===0&&end.capture_error_count===0,'native_capture_not_complete');
  check(end.raw_fight_state_packet_count===packets.length,'native_packet_count_mismatch');
  const extracted=extractCalls(path.basename(file),'pid'+header.pid,packets);
  tick(header.stopwatch_timestamp_ticks);tick(end.stopwatch_timestamp_ticks);check(end.stopwatch_timestamp_ticks>=header.stopwatch_timestamp_ticks,'native_capture_clock_regressed');
  return {frequency:header.stopwatch_frequency_hz,start_qpc:header.stopwatch_timestamp_ticks,end_qpc:end.stopwatch_timestamp_ticks,receipts,provenance,pid:header.pid,packet_count:packets.length,summary:{
    packet_count:packets.length,unique_receipt_keys:receipts.size,initial_round_number:header.initial_state?.round?.number??null,
    calls:extracted.calls.map(c=>({sequence:c.call_sequence,name:c.call_name,faller_slot:c.faller_slot,points:c.points,observed_new_call:c.observed_new_call,left_censored:c.left_censored,repeated_snapshots:c.repeated_snapshots})),
    counts:extracted.counts.map(c=>({slot:c.faller_slot,left_censored:c.left_censored,right_censored:c.right_censored,explicit_countout:c.explicit_countout,duration_receipt_seconds:c.duration_receipt_seconds,resolution_call_name:c.resolution_call_name}))}};
}
async function run(trial,nativeFile,output){
  check(!fs.existsSync(output),'output_exists');const native=await loadNative(nativeFile),audit=new ObservationAudit(native);
  const relay=await scan(path.join(trial,'relay.stdout.jsonl'),(r,line)=>{if(r.event==='g1_policy_state')audit.observe(r,line);});
  check(audit.sources>0,'no_policy_source_observations');const result={schema:'rek.live_referee_validation.v1',created_utc:new Date().toISOString(),validator_sha256:sha(fs.readFileSync(__filename)),
    ...audit.report(),native_process_id:native.pid,native:native.summary,inputs:[relay,native.provenance],
    matching_contract:'Identical validated 33-byte body SHA/base64 and exact Unity frame, time and unscaled time at the two read-only ApplyFightStateSnapshot prefixes. QPC age is independently recomputed from the policy clock; receipt frequencies must equal native capture frequency.',
    memory_contract:'Streams entire input files; retains referee packets, receipt identities and deduplicated calls only, never pose or full policy logs.',no_game_connection:true,no_runtime_changes:true};
  fs.mkdirSync(output);fs.writeFileSync(path.join(output,'live-referee-validation.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});return result;
}
module.exports={decode,validatePayload,ObservationAudit,receiptKey,NULL_WHEN_UNAVAILABLE,loadNative,run};
if(require.main===module)Promise.resolve().then(()=>{check(process.argv.length===5,'usage_validate_live_referee_TRIAL_RECORDER_NEW_OUTPUT');return run(...process.argv.slice(2));}).then(r=>{console.log(JSON.stringify({verification_passed:r.verification_passed,source_count:r.source_count,available:r.available_source_count,unavailable:r.unavailable_reasons,unique_receipts:r.unique_bridge_receipts,call_ids:r.unique_latched_call_ids}));if(!r.verification_passed)process.exitCode=2;}).catch(e=>{console.error(e.message);process.exitCode=2;});
