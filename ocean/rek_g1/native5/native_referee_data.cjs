#!/usr/bin/env node
'use strict';

// Offline decoding of received referee snapshots. No game input or inferred moves.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {scan,nativePose,precedingPose,validateWirePacket}=require('./native_hit_receipt_data.cjs');
const check=(ok,why)=>{if(!ok)throw Error(why);},hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const NAMES=['Slip','SlipEStop','Knockdown','BeatCount','Knockout','DoubleKnockdown','DoubleKnockout'];
const CLOCK=['unity_frame','unity_time','unity_unscaled_time','monotonic_receipt_time','client_fixed_tick_at_observation'];
const pick=(r,ks)=>Object.fromEntries(ks.filter(k=>r[k]!==undefined).map(k=>[k,r[k]]));
const describe=a=>{a=a.filter(Number.isFinite).sort((a,b)=>a-b);return {count:a.length,minimum:a[0]??null,median:a.length?a[Math.floor((a.length-1)/2)]:null,maximum:a.at(-1)??null};};
const tally=(a,key)=>{const out={};for(const r of a){const k=key(r);out[k]=(out[k]||0)+1;}return out;};
function validateRefereePacket(r) {
  check(CLOCK.every(k=>Number.isFinite(r[k])),'invalid_referee_clock');
  const b=Buffer.from(r.wire_body_base64||'','base64');
  check(b.length===33&&r.wire_body_bytes===33&&hash(b)===r.wire_body_sha256,'invalid_referee_wire');
  const fields={phase:0,round_number:1,round_active:2,is_redo:3,knockout_occurred:12,round_result:13,
    rounds_won_0:15,rounds_won_1:16,fight_result:17,format:19,human_slot_mask:20,fault_mask:22,
    fault_stress_0:23,fault_stress_1:24,referee_count_mask:25,referee_count_seconds:26,
    referee_call_sequence:27,referee_call_type:28,referee_call_points:30,ai_level:31,decided_winner_bits:32};
  for(const [k,offset] of Object.entries(fields))check(r.decoded?.[k]===b.readUInt8(offset),'referee_decoded_byte_mismatch');
  for(const [k,offset] of Object.entries({round_winner:14,fight_winner:18,champion_slot:21,referee_call_faller:29}))
    check(r.decoded[k]===b.readInt8(offset),'referee_decoded_signed_mismatch');
  check(r.decoded.hits_0===b.readInt16LE(8)&&r.decoded.hits_1===b.readInt16LE(10)&&
    Math.fround(r.decoded.time_remaining)===b.readFloatLE(4),'referee_decoded_numeric_mismatch');
  check(NAMES[r.decoded.referee_call_type]===r.decoded.referee_call_name&&r.decoded.referee_count_mask<=3,'invalid_referee_enum');
}
function extractCalls(capture,session,packets,poses=[]) {
  const calls=[],counts=[],snapshots=[],active=new Map();let previous=null,current=null,epoch=0,wrap=0;
  const close=(episode,p,reason,resolution)=>{
    episode.end_receipt_clock=pick(p,CLOCK);episode.duration_receipt_seconds=p.monotonic_receipt_time-episode.start_receipt_clock.monotonic_receipt_time;
    episode.end_reason=reason;episode.resolution_call_id=resolution?.call_id??null;
    episode.resolution_call_name=resolution?.call_name??null;episode.right_censored=reason!=='count_mask_cleared';
    episode.explicit_countout=reason==='count_mask_cleared'&&(resolution?.call_name==='DoubleKnockout'||
      resolution?.call_name==='Knockout'&&resolution.faller_slot===episode.faller_slot);
    counts.push(episode);
  };
  for(const p of packets) {
    const d=p.decoded,seq=d.referee_call_sequence,prev=previous?.decoded;
    check(!previous||p.monotonic_receipt_time>=previous.monotonic_receipt_time,'nonmonotonic_referee_receipts');
    let reset=null;
    if(prev&&(d.round_number!==prev.round_number||d.is_redo!==prev.is_redo||p.fight_epoch!==previous.fight_epoch))reset='fight_or_round_changed';
    else if(prev&&seq===0&&prev.referee_call_sequence!==0)reset='zero_sequence_reset';
    else if(prev&&seq>0&&seq<prev.referee_call_sequence){if(prev.referee_call_sequence===255&&seq===1)wrap++;else reset='sequence_decrease_uncertain_reset';}
    if(reset){for(const episode of active.values())close(episode,p,reset,null);active.clear();epoch++;wrap=0;current=null;}
    let fresh=null;
    if(seq===0)current=null;
    else if(!current||current.call_sequence!==seq) {
      const historical=!previous||reset==='fight_or_round_changed'||reset==='sequence_decrease_uncertain_reset';
      fresh={schema:'rek.native_referee_call.v1',capture,session,
        call_id:`${session}:${capture}:round${d.round_number}:epoch${epoch}:wrap${wrap}:seq${seq}`,
        fight_epoch:p.fight_epoch??null,round_number:d.round_number,is_redo:d.is_redo,call_sequence:seq,
        call_type:d.referee_call_type,call_name:d.referee_call_name,faller_slot:d.referee_call_faller,
        points:d.referee_call_points,count_mask_at_first_receipt:d.referee_count_mask,count_seconds_at_first_receipt:d.referee_count_seconds,
        observed_new_call:!historical,left_censored:historical,sequence_reset_basis:reset,
        first_receipt_clock:pick(p,CLOCK),last_receipt_clock:pick(p,CLOCK),first_source_line:p.source_line,
        wire_body_sha256:p.wire_body_sha256,repeated_snapshots:0,
        counters_at_first_receipt:[d.hits_0,d.hits_1],counters_at_previous_snapshot:prev?[prev.hits_0,prev.hits_1]:null,
        prior_native_paired_root_pose:precedingPose(poses,p,'native'),
        scorer_slot:null,attacker:null,active_clip:null,executed_move_index:null,server_clock:null};
      calls.push(fresh);current=fresh;
    } else {
      check(current.call_type===d.referee_call_type&&current.faller_slot===d.referee_call_faller&&current.points===d.referee_call_points,'same_sequence_conflicting_call_payload');
      current.repeated_snapshots++;current.last_receipt_clock=pick(p,CLOCK);
    }
    for(const slot of [0,1]) {
      const on=Boolean(d.referee_count_mask&(1<<slot)),episode=active.get(slot);
      if(on&&!episode)active.set(slot,{schema:'rek.native_referee_count.v1',capture,session,
        count_id:`${session}:${capture}:epoch${epoch}:count${counts.length+active.size+1}:slot${slot}`,
        faller_slot:slot,round_number:d.round_number,start_receipt_clock:pick(p,CLOCK),start_source_line:p.source_line,
        onset_call_id:fresh?.call_id??null,onset_call_name:fresh?.call_name??null,
        left_censored:!previous||Boolean(reset),received_count_seconds:[d.referee_count_seconds]});
      else if(on&&!episode.received_count_seconds.includes(d.referee_count_seconds))episode.received_count_seconds.push(d.referee_count_seconds);
      else if(!on&&episode){close(episode,p,'count_mask_cleared',fresh);active.delete(slot);}
    }
    snapshots.push({schema:'rek.native_referee_snapshot.v1',capture,session,source_line:p.source_line,
      receipt_clock:pick(p,CLOCK),fight_epoch:p.fight_epoch??null,wire_body_sha256:p.wire_body_sha256,decoded:d,
      deduplicated_call_id:current?.call_id??null,server_clock:null});
    previous=p;
  }
  if(previous)for(const episode of active.values())close(episode,previous,'capture_end',null);
  return {calls,counts,snapshots};
}
function joinFivePointScores(calls,scores,maxOffset=.25) {
  for(const c of calls)c.scorer_slot=null;
  const awards=scores.filter(s=>s.decoded.points_awarded===5);
  const candidates=new Map();
  for(const s of awards) {
    const slot=s.decoded.fighter_index;
    candidates.set(s,calls.filter(c=>c.observed_new_call&&c.points===5&&['Knockout','DoubleKnockout'].includes(c.call_name)&&
      Math.abs(c.first_receipt_clock.monotonic_receipt_time-s.monotonic_receipt_time)<=maxOffset&&
      c.counters_at_first_receipt[slot]===s.decoded.new_hit_count&&
      (c.call_name==='DoubleKnockout'||[0,1].includes(c.faller_slot)&&c.faller_slot!==slot)));
  }
  return awards.map(s=>{
    const near=candidates.get(s),c=near.length===1?near[0]:null;
    const unique=c&&awards.filter(other=>other.decoded.fighter_index===s.decoded.fighter_index&&candidates.get(other).includes(c)).length===1;
    if(unique&&c.call_name==='Knockout')c.scorer_slot=s.decoded.fighter_index;
    return {score_id:s.score_id,source_line:s.source_line,scorer_slot:s.decoded.fighter_index,
      points_awarded:5,new_received_counter:s.decoded.new_hit_count,receipt_clock:pick(s,CLOCK),
      candidate_referee_call_ids:near.map(x=>x.call_id),associated_referee_call_id:unique?c.call_id:null,
      association:unique?'unique_receipt_counter_and_referee_agreement':near.length?'ambiguous':'unmatched',
      referee_minus_score_receipt_seconds:unique?c.first_receipt_clock.monotonic_receipt_time-s.monotonic_receipt_time:null,
      received_counter_delta_since_prior_snapshot:unique&&c.counters_at_previous_snapshot?
        s.decoded.new_hit_count-c.counters_at_previous_snapshot[s.decoded.fighter_index]:null,
      faller_slot:unique&&c.call_name==='Knockout'?c.faller_slot:null,referee_call_name:unique?c.call_name:null,
      causal_packet_id:null,causal_hit_id:null,attacker:null,executed_move_index:null,server_clock:null};
  });
}
const SCHEMA={schema:'rek.native_referee_teacher.schema.v1',authority:'33-byte REK_FightState body copied at client ApplyFightStateSnapshot prefix',
  fields:'received call sequence/type/faller/points, count mask/seconds, round state and received counters',
  count_mask:'bit 0 fighter slot 0; bit 1 fighter slot 1',
  call_names:NAMES,zero_sequence:'empty snapshot, never a Slip event',
  deduplication:'capture/session/round/reset/wrap namespaced call IDs; repeated sequence is one latched call; first nonzero snapshot is left-censored',
  reset:'round/redo/fight epoch change, observed sequence zero, or decreasing sequence other than 255 to 1; uncertain decreases are censored',
  wrap:'server skips zero, so 255 to 1 is a wrap, with a new wrap namespace',
  score_join:'same exact capture, unique +/-0.25 s client monotonic receipt window, explicit five-point Knockout/DoubleKnockout and exact received scorer counter; single-fighter scorer differs from explicit faller',
  association_limit:'there is no shared causal packet/event ID; timing and counter agreement corroborate an award but do not prove a physical hit or an executed action',
  pose_join:'closest prior native line with nonlater shared Unity clocks and frame, maximum 0.1 s; receipt time is not authoritative physical event time',
  countout:'received count mask on then cleared with a new Knockout/DoubleKnockout call; source ResolveCountExpiry identifies this call as deadline expiry',
  terminal_limit:'Knockout referee call need not mean terminal round KO; preserve knockout_occurred, round_result and round_active separately',
  units:'seconds; pose distances in uncalibrated Unity numeric units',
  split:'unchanged six process-clock session groups from balance/receipt audit; all ten prespecified policy trials retained',
  unknowns:{attacker:null,active_clip:null,executed_move_index:null,causal_hit_id:null,server_clock:null},
  visual_flag_limit:'visual-only robot.IsFalling/IsFallen are not the received referee state',no_training:true,raw_teacher_data_in_git:false};
async function exportAudit(nativeDirectory,receiptAuditFile,output) {
  check(!fs.existsSync(output),'output_exists');
  const receiptBytes=fs.readFileSync(receiptAuditFile),receipt=JSON.parse(receiptBytes);
  check(receipt.schema==='rek.native_hit_receipt.audit.v1'&&receipt.captures.length===10&&receipt.session_groups.length===6,'invalid_receipt_audit');
  const expected=['baseline-r1-retry1','shaped-r1-retry2','baseline-r2','shaped-r2','baseline-r3-retry2','shaped-r3','baseline-r4','shaped-r4-retry3','baseline-r5','shaped-r5-retry4'];
  check(expected.every((n,i)=>receipt.captures[i].capture==='policy/'+n),'prespecified_capture_selection_changed');
  const reports=[],allCalls=[],allCounts=[],allSnapshots=[],allAwards=[];
  for(const ref of receipt.captures) {
    const packets=[],poses=[],scores=[],flags={samples:0,actor_observations:0,falling_true:0,fallen_true:0,missing_flags:0};
    let fightEpoch=null,start=0,end=0,errors=0;
    check(path.basename(ref.native_source.file)===ref.native_source.file,'unsafe_capture_filename');
    const source=await scan(path.join(nativeDirectory,ref.native_source.file),(r,line)=>{
      if(r.event==='capture_start')start++;if(r.event==='capture_end')end++;if(r.event==='capture_error')errors++;
      if(r.event==='root_pose_sample'){fightEpoch=r.fight_epoch;poses.push({...nativePose(r,line),fight_epoch:r.fight_epoch});}
      if(r.event==='raw_fight_state_packet'){validateRefereePacket(r);packets.push({...pick(r,['decoded','wire_body_sha256',...CLOCK]),source_line:line,fight_epoch:fightEpoch});}
      if(r.event==='raw_score_packet'){validateWirePacket(r);scores.push({...pick(r,['decoded',...CLOCK]),source_line:line,score_id:`${ref.capture}:S${String(scores.length+1).padStart(4,'0')}`});}
      if(r.event==='sample'){flags.samples++;for(const slot of [0,1]){const f=r['fighter_'+slot];flags.actor_observations++;flags.falling_true+=f?.falling===true;flags.fallen_true+=f?.fallen===true;
        flags.missing_flags+=typeof f?.falling!=='boolean'||typeof f?.fallen!=='boolean';}}
    });
    check(source.sha256===ref.native_source.sha256,'native_source_hash_mismatch');
    check(start===1&&end===1&&!errors&&scores.length===ref.received_score_packets,'capture_integrity_mismatch');
    const extracted=extractCalls(ref.capture,ref.session,packets,poses),awards=joinFivePointScores(extracted.calls,scores);
    for(const a of awards){a.capture=ref.capture;a.session=ref.session;}
    const observed=extracted.calls.filter(c=>c.observed_new_call),completed=extracted.counts.filter(c=>!c.left_censored&&!c.right_censored);
    reports.push({capture:ref.capture,session:ref.session,native_source:source,wire_packets_validated:packets.length,
      call_counts:tally(observed,c=>c.call_name),calls_by_faller:tally(observed,c=>c.call_name+':slot'+c.faller_slot),
      observed_calls:observed.length,left_censored_calls:extracted.calls.length-observed.length,
      repeated_call_snapshots:extracted.calls.reduce((n,c)=>n+c.repeated_snapshots,0),zero_sequence_packets:packets.filter(p=>p.decoded.referee_call_sequence===0).length,
      count_episodes:extracted.counts.length,complete_count_episodes:completed.length,explicit_countouts:completed.filter(c=>c.explicit_countout).length,
      count_duration_receipt_seconds:describe(completed.map(c=>c.duration_receipt_seconds)),
      count_tick_patterns:tally(completed,c=>c.received_count_seconds.join(',')),
      five_point_awards:awards.length,five_point_associations:tally(awards,a=>a.association),
      award_referee_minus_score_receipt_seconds:describe(awards.map(a=>a.referee_minus_score_receipt_seconds)),
      award_counter_delta_patterns:tally(awards,a=>String(a.received_counter_delta_since_prior_snapshot)),
      awards_by_scorer_and_faller:tally(awards,a=>'scorer'+a.scorer_slot+':faller'+a.faller_slot+':'+a.referee_call_name),
      referee_calls_with_prior_root_pose:extracted.calls.filter(c=>c.prior_native_paired_root_pose).length,
      visual_robot_flags:flags,received_terminal_ko_packets:packets.filter(p=>p.decoded.knockout_occurred!==0).length,
      final_received_round:pick(packets.at(-1).decoded,['round_number','round_active','hits_0','hits_1','knockout_occurred','round_result','round_result_name'])});
    allCalls.push(...extracted.calls);allCounts.push(...extracted.counts);allSnapshots.push(...extracted.snapshots);allAwards.push(...awards);
  }
  const observed=allCalls.filter(c=>c.observed_new_call),complete=allCounts.filter(c=>!c.left_censored&&!c.right_censored);
  const report={schema:'rek.native_referee.audit.v1',created_utc:new Date().toISOString(),
    exporter_sha256:hash(fs.readFileSync(__filename)),receipt_exporter_sha256:hash(fs.readFileSync(require.resolve('./native_hit_receipt_data.cjs'))),
    receipt_audit_sha256:hash(receiptBytes),supervision:SCHEMA,session_groups:receipt.session_groups,captures:reports,
    totals:{captures:reports.length,received_referee_snapshots:allSnapshots.length,observed_calls:observed.length,
      calls_by_type:tally(observed,c=>c.call_name),calls_by_type_and_faller:tally(observed,c=>c.call_name+':slot'+c.faller_slot),
      left_censored_calls:allCalls.length-observed.length,count_episodes:allCounts.length,complete_count_episodes:complete.length,
      explicit_countouts:complete.filter(c=>c.explicit_countout).length,count_duration_receipt_seconds:describe(complete.map(c=>c.duration_receipt_seconds)),
      five_point_awards:allAwards.length,five_point_associations:tally(allAwards,a=>a.association),
      award_referee_minus_score_receipt_seconds:describe(allAwards.map(a=>a.referee_minus_score_receipt_seconds)),
      visual_actor_observations:reports.reduce((n,r)=>n+r.visual_robot_flags.actor_observations,0),
      visual_falling_true:reports.reduce((n,r)=>n+r.visual_robot_flags.falling_true,0),visual_fallen_true:reports.reduce((n,r)=>n+r.visual_robot_flags.fallen_true,0)},
    no_training:true,no_runtime_changes:true,no_executed_move_labels:true};
  fs.mkdirSync(output,{recursive:true,mode:0o700});
  for(const [name,value,jsonl] of [['native-referee-snapshots.private.jsonl',allSnapshots,true],['native-referee-calls.private.jsonl',allCalls,true],
    ['native-referee-counts.private.jsonl',allCounts,true],['native-referee-awards.private.jsonl',allAwards,true],
    ['native-referee-audit.json',report,false],['native-referee-schema.json',SCHEMA,false]])
    fs.writeFileSync(path.join(output,name),jsonl?value.map(JSON.stringify).join('\n')+'\n':JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});
  return {...report.totals,report_sha256:hash(fs.readFileSync(path.join(output,'native-referee-audit.json')))};
}
module.exports={validateRefereePacket,extractCalls,joinFivePointScores,exportAudit,NAMES,SCHEMA};
if(require.main===module)Promise.resolve().then(()=>{check(process.argv.length===5,'usage_native_referee_data_NATIVE_DIRECTORY_RECEIPT_AUDIT_NEW_PRIVATE_OUTPUT');return exportAudit(...process.argv.slice(2));})
  .then(r=>console.log(JSON.stringify(r))).catch(e=>{console.error(/^[a-zA-Z0-9_]+$/.test(e.message)?e.message:e.code||'referee_export_failed');process.exitCode=1;});
