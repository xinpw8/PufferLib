#!/usr/bin/env node
'use strict';

// Offline native-packet/pose receipt export. No game connection or execution labels.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const check=(ok,reason)=>{if(!ok)throw Error(reason);},finite=Number.isFinite;
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finite);
const SELECTED=['baseline-r1-retry1','shaped-r1-retry2','baseline-r2','shaped-r2','baseline-r3-retry2','shaped-r3','baseline-r4','shaped-r4-retry3','baseline-r5','shaped-r5-retry4'];
const clockFields=['unity_frame','unity_time','unity_unscaled_time','monotonic_receipt_time','client_fixed_tick_at_observation'];
const select=(row,keys)=>Object.fromEntries(keys.filter(k=>row[k]!==undefined).map(k=>[k,row[k]]));
function describe(values) {
  const a=values.filter(finite).sort((a,b)=>a-b);
  return {count:a.length,minimum:a[0]??null,median:a.length?a[Math.floor((a.length-1)*.5)]:null,
    p95:a.length?a[Math.floor((a.length-1)*.95)]:null,maximum:a.at(-1)??null};
}
async function scan(file,visit) {
  const before=fs.statSync(file),digest=crypto.createHash('sha256'),input=fs.createReadStream(file);let line=0;
  input.on('data',bytes=>digest.update(bytes));
  for await(const text of readline.createInterface({input,crlfDelay:Infinity})) {
    line++;if(!text.trim())continue;let row;try{row=JSON.parse(text);}catch{throw Error('invalid_json_line_'+line);}
    visit(row,line);
  }
  const after=fs.statSync(file);check(before.size===after.size&&before.mtimeMs===after.mtimeMs&&before.ino===after.ino,'source_changed_during_read');
  return {file:path.basename(file),bytes:after.size,sha256:digest.digest('hex'),lines:line};
}
function validateRoot(position,rotation) {
  check(vector(position,3)&&vector(rotation,4)&&Math.hypot(...rotation)>0,'invalid_root_pose');
  return {position_xyz:position,rotation_xyzw:rotation};
}
function nativePose(raw,line) {
  const fighters=[0,1].map(slot=>{
    const f=raw['fighter_'+slot+'_root'];check(f,'missing_paired_root');
    return {slot,...validateRoot(f.world_position_xyz,f.world_rotation_xyzw)};
  });
  check(finite(raw.unity_time)&&Number.isSafeInteger(raw.unity_frame)&&Number.isSafeInteger(raw.stopwatch_timestamp_ticks),'invalid_native_pose_clock');
  return {source_line:line,source:'native_root_pose_sample',unity_time:raw.unity_time,unity_unscaled_time:raw.unity_unscaled_time,
    unity_frame:raw.unity_frame,qpc_ticks:raw.stopwatch_timestamp_ticks,client_fixed_tick:raw.client_fixed_tick,
    local_slot:raw.local_fighter_index,round_number:raw.round_number,fighters};
}
function relayPose(raw,line) {
  check(raw.schema==='rek.g1_policy_source.v1'&&raw.fighters?.length===2,'invalid_policy_source');
  const fighters=raw.fighters.map((f,slot)=>{
    check(Array.isArray(f.bone_names)&&f.bone_names.length===30&&f.bone_world_positions_xyz?.length===30&&
      f.bone_world_positions_xyz.every(v=>vector(v,3))&&f.bone_local_rotations_xyzw?.length===30&&
      f.bone_local_rotations_xyzw.every(v=>vector(v,4)),'invalid_paired_bones');
    return {slot,...validateRoot(f.root_position_xyz,f.root_rotation_xyzw),bone_names:f.bone_names,
      bone_world_positions_xyz:f.bone_world_positions_xyz,bone_local_rotations_xyzw:f.bone_local_rotations_xyzw,tilt_degrees:f.tilt_angle};
  });
  check(finite(raw.clock.unity_time)&&finite(Date.parse(raw.clock.utc))&&Number.isSafeInteger(raw.clock.qpc_ticks)&&raw.clock.qpc_frequency_hz>0,'invalid_relay_clock');
  return {source_line:line,source:'relay_g1_policy_state',utc:raw.clock.utc,unity_time:raw.clock.unity_time,unity_frame:raw.clock.unity_frame,
    qpc_ticks:raw.clock.qpc_ticks,qpc_frequency_hz:raw.clock.qpc_frequency_hz,round_identity_sha256:raw.round_identity_sha256,
    local_slot:raw.local_slot,round_number:raw.round.number,fighters,
    request_context:select(raw.input,['desired_action','velocity_command_xyz','requested_move_index','requested_move_qpc_ticks']),
    executed_move_index:null};
}
function nativeFilenameTime(name) {
  const m=name.match(/^rek-private-ai-root-motion-(\d{8})T(\d{6})\.(\d+)Z-pid\d+-[a-f0-9]+\.jsonl$/);
  if(!m)return null;const d=m[1],t=m[2];
  return Date.parse(`${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}T${t.slice(0,2)}:${t.slice(2,4)}:${t.slice(4,6)}.${m[3].slice(0,3)}Z`);
}
function chooseNativeFile(names,firstUtc) {
  const at=Date.parse(firstUtc),candidates=names.filter(name=>{const time=nativeFilenameTime(name);return time!==null&&Math.abs(time-at)<1000;});
  check(candidates.length===1,'native_capture_not_unique_by_start_window');return candidates[0];
}
function verifyCaptureBinding(native,relay,start,end) {
  check(start&&end&&Number.isSafeInteger(start.pid)&&start.stopwatch_frequency_hz===relay[0].qpc_frequency_hz,'capture_header_or_clock_missing');
  const hz=start.stopwatch_frequency_hz,first=relay[0],last=relay.at(-1);
  const qpcStart=(first.qpc_ticks-start.stopwatch_timestamp_ticks)/hz,utcStart=(Date.parse(first.utc)-Date.parse(start.utc))/1000;
  check(Math.abs(qpcStart-utcStart)<.25&&qpcStart>=-1&&qpcStart<=2,'capture_start_clock_mismatch');
  check(Math.abs(last.qpc_ticks-end.stopwatch_timestamp_ticks)/hz<=2,'capture_end_clock_mismatch');
  const byFrame=new Map();for(const pose of relay){const rows=byFrame.get(pose.unity_frame)||[];rows.push(pose);byFrame.set(pose.unity_frame,rows);}
  const anchors=[];
  for(let i=0;i<native.length;i+=50)for(const p of byFrame.get(native[i].unity_frame)||[]) {
    const n=native[i];
    if(Math.abs(n.qpc_ticks-p.qpc_ticks)/hz>.15||Math.abs(n.unity_time-p.unity_time)>.05)continue;
    if(n.fighters.every((f,s)=>Math.hypot(...f.position_xyz.map((v,j)=>v-p.fighters[s].position_xyz[j]))<=.05)) {anchors.push(p.unity_time);break;}
  }
  const span=anchors.length?Math.max(...anchors)-Math.min(...anchors):0;
  check(anchors.length>=2&&span>=1,'paired_pose_clock_binding_missing');
  check(native.every(n=>n.round_number===first.round_number&&n.local_slot===first.local_slot)&&relay.every(p=>p.round_number===first.round_number&&p.round_identity_sha256===first.round_identity_sha256&&p.local_slot===first.local_slot),'capture_round_or_actor_mismatch');
  return {basis:'same Unity frame, QPC, Unity time and both root positions',matching_anchors:anchors.length,anchor_span_seconds:span,
    maximum_qpc_offset_seconds:.15,maximum_unity_offset_seconds:.05,maximum_root_difference_unity_units:.05,
    pid:start.pid,pid_is_unique_process_session:false};
}
function precedingPose(poses,event,kind,maxAge=.1) {
  let lo=0,hi=poses.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(poses[mid].unity_time<=event.unity_time)lo=mid+1;else hi=mid;}
  for(let i=lo-1;i>=0;i--) {
    const pose=poses[i],age=event.unity_time-pose.unity_time;if(age>maxAge)return null;
    if(pose.unity_frame>event.unity_frame)continue;
    if(kind==='native'&&(pose.source_line>=event.source_line||
      finite(pose.unity_unscaled_time)&&finite(event.unity_unscaled_time)&&pose.unity_unscaled_time>event.unity_unscaled_time))continue;
    // The relay is a different log. Same-frame LateUpdate observations may have
    // happened after receipt despite equal Unity time; only earlier frames qualify.
    if(kind==='relay'&&pose.unity_frame>=event.unity_frame)continue;
    return {...pose,receipt_minus_pose_unity_seconds:age,
      ordering_basis:kind==='native'?'earlier native file line and nonlater shared Unity clocks':'strictly earlier Unity frame and nonlater Unity time'};
  }
  return null;
}
function packetAssociation(a,b) {
  return a.unity_frame===b.unity_frame&&a.client_fixed_tick_at_observation===b.client_fixed_tick_at_observation&&
    finite(a.monotonic_receipt_time)&&finite(b.monotonic_receipt_time)&&Math.abs(a.monotonic_receipt_time-b.monotonic_receipt_time)<=.01;
}
function validatePacket(row) {
  check(finite(row.unity_time)&&finite(row.monotonic_receipt_time)&&Number.isSafeInteger(row.unity_frame)&&Number.isSafeInteger(row.client_fixed_tick_at_observation),'invalid_packet_receipt_clock');
  check(typeof row.wire_body_sha256==='string'&&/^[a-f0-9]{64}$/.test(row.wire_body_sha256),'invalid_packet_hash');
  if(row.event==='raw_hit_packet')check(vector(row.decoded?.position_xyz,3)&&vector(row.decoded?.surface_normal_xyz,3)&&finite(row.decoded?.relative_speed)&&[0,1].includes(row.decoded?.is_kick),'invalid_hit_packet');
  else check([0,1].includes(row.decoded?.fighter_index)&&finite(row.decoded?.points_awarded)&&row.decoded.points_awarded>=0&&Number.isSafeInteger(row.decoded?.new_hit_count),'invalid_score_packet');
}
function validateWirePacket(row) {
  validatePacket(row);check(typeof row.wire_body_base64==='string','missing_wire_body');
  const body=Buffer.from(row.wire_body_base64,'base64'),isHit=row.event==='raw_hit_packet';
  check(body.length===(isHit?29:7)&&body.length===row.wire_body_bytes&&hash(body)===row.wire_body_sha256,'wire_body_hash_or_size_mismatch');
  const sameFloat=(offset,value)=>body.readFloatLE(offset)===Math.fround(value);
  if(isHit)check(row.decoded.position_xyz.every((v,i)=>sameFloat(i*4,v))&&
    row.decoded.surface_normal_xyz.every((v,i)=>sameFloat(12+i*4,v))&&sameFloat(24,row.decoded.relative_speed)&&
    body.readUInt8(28)===row.decoded.is_kick,'decoded_hit_disagrees_with_wire');
  else check(body.readUInt8(0)===row.decoded.fighter_index&&body.readInt16LE(1)===row.decoded.new_hit_count&&sameFloat(3,row.decoded.points_awarded),'decoded_score_disagrees_with_wire');
}
function receiptGeometry(hit,pose) {
  if(!pose)return null;
  return pose.fighters.map(f=>{
    const [x,y,z,w]=f.rotation_xyzw.map(v=>v/Math.hypot(...f.rotation_xyzw));
    const heading=Math.atan2(2*(x*z-w*y),1-2*(y*y+z*z)),c=Math.cos(heading),s=Math.sin(heading);
    const offset=hit.position_xyz.map((v,j)=>v-f.position_xyz[j]);
    return {slot:f.slot,contact_minus_root_forward_lateral_vertical:[c*offset[0]+s*offset[2],-s*offset[0]+c*offset[2],offset[1]],
      contact_distance_to_root:Math.hypot(...offset),role:'unknown'};
  });
}
function associate(capture,session,packets,nativePoses,relayPoses) {
  const hits=packets.filter(p=>p.event==='raw_hit_packet'),scores=packets.filter(p=>p.event==='raw_score_packet');
  const hitIds=new Map(hits.map((p,i)=>[p,`${capture}:H${String(i+1).padStart(4,'0')}`]));
  const scoreIds=new Map(scores.map((p,i)=>[p,`${capture}:S${String(i+1).padStart(4,'0')}`]));
  const common=(p,id)=>({schema:'rek.native_receipt_teacher.v1',capture,session,event_id:id,source_line:p.source_line,
    receipt_clock:select(p,clockFields),wire_body_sha256:p.wire_body_sha256,
    prior_native_paired_root_pose:precedingPose(nativePoses,p,'native'),
    prior_relay_paired_pose:precedingPose(relayPoses,p,'relay'),
    attacker:null,defender:null,active_clip:null,executed_move_index:null,causal_request_id:null,server_clock:null});
  return {
    hits:hits.map(p=>{
      const near=scores.filter(s=>packetAssociation(p,s)),unique=near.length===1&&hits.filter(h=>packetAssociation(h,near[0])).length===1;
      const row={...common(p,hitIds.get(p)),kind:'received_hit_effect',decoded:p.decoded,
        associated_score_receipt_ids:near.map(s=>scoreIds.get(s)),receipt_association:unique?'unique_same_frame_noncausal':near.length?'ambiguous_noncausal':'no_matching_receipt',
        causal_score_event_id:null,contact_occurrence_authority:'received effects packet',physical_contact_time:null};
      row.geometry_at_native_prior_pose=receiptGeometry(p.decoded,row.prior_native_paired_root_pose);return row;
    }),
    scores:scores.map(p=>({...common(p,scoreIds.get(p)),kind:'received_score',decoded:p.decoded,
      associated_hit_receipt_ids:hits.filter(h=>packetAssociation(h,p)).map(h=>hitIds.get(h)),
      award_class:p.decoded.points_awarded===5?'five_point_award_cause_unassigned':'point_award',causal_hit_event_id:null}))};
}
const SCHEMA={schema:'rek.native_receipt_teacher.schema.v1',selection:'all ten prespecified completed baseline/shaped A/B trials; no additional selection',
  native_pose_join:'closest earlier native-file paired root pose with nonlater Unity time/frame and unscaled time; maximum age 0.1 s',
  relay_pose_join:'closest paired 30-bone pose in a strictly earlier Unity frame and nonlater Unity time; maximum age 0.1 s',
  packet_association:'same Unity frame and client fixed tick; monotonic receipt difference <= 0.01 s; associations remain noncausal',
  clock_limit:'packet monotonic receipt time has no identical pose field; joins use shared Unity time and record/frame order, never an invented QPC/server conversion',
  geometry_limit:'received effect coordinates with preceding rendered poses; not synchronized authoritative physical-contact geometry',
  units:'Unity numeric distance and speed units; metre calibration unverified; seconds',
  labels:{attacker:null,defender:null,active_clip:null,executed_move_index:null,causal_request_id:null,server_clock:null},
  split:'preserve all six process-clock groups from the baseline transition audit; these ten trials cover its four policy groups',
  known_values:'hit position/normal/relative_speed/is_kick, score recipient/points, client receipt clocks, paired root and bone geometry',
  wire_validation:'verify each recorded wire-body SHA-256, packed body length and decoded fields against little-endian bytes',
  missing_hit_is_miss:false,latest_request_is_executed_move:false,raw_teacher_data_in_git:false,
  why_previous_audit_lacked_hit_fields:'previous audit ingested relay.stdout.jsonl visual policy observations; raw hit/score packets are emitted separately by RekEvidenceRecorder under the Wine runtime evidence directory'};
async function exportAudit(base,nativeDirectory,baselineAuditFile,output) {
  check(!fs.existsSync(output),'output_exists');
  const baselineBytes=fs.readFileSync(baselineAuditFile),baseline=JSON.parse(baselineBytes);
  check(baseline.schema==='rek.balance_transition.audit.v1'&&baseline.session_groups.length===6,'invalid_baseline_audit');
  const names=fs.readdirSync(nativeDirectory),records=[],allHits=[],allScores=[];
  for(const name of SELECTED) {
    const capture='policy/'+name,reference=baseline.captures.find(c=>c.capture===capture);check(reference,'selected_capture_missing_from_baseline');
    const relay=[];
    const relaySource=await scan(path.join(base,name,'trial','relay.stdout.jsonl'),(r,line)=>{if(r.event==='g1_policy_state')relay.push(relayPose(r,line));});
    check(relaySource.sha256===reference.source.sha256&&relay.length>2,'relay_hash_mismatch');
    const filename=chooseNativeFile(names,relay[0].utc),native=[],packets=[],counts={},composer={samples:0,playing_true:0,clip_nonnull:0,frame_nonnegative:0,fps_positive:0};
    let start=null,end=null;
    const nativeSource=await scan(path.join(nativeDirectory,filename),(r,line)=>{
      counts[r.event]=(counts[r.event]||0)+1;
      if(r.event==='capture_start')start=r;if(r.event==='capture_end')end=r;
      if(r.event==='root_pose_sample')native.push(nativePose(r,line));
      if(r.event==='raw_hit_packet'||r.event==='raw_score_packet'){validateWirePacket(r);packets.push({...select(r,['event',...clockFields,'wire_body_sha256','decoded']),source_line:line});}
      if(r.event==='sample'){composer.samples++;composer.playing_true+=r.input?.action_playing===true;composer.clip_nonnull+=r.input?.action_clip!=null;
        composer.frame_nonnegative+=finite(r.input?.action_clip_frame)&&r.input.action_clip_frame>=0;composer.fps_positive+=r.input?.action_clip_fps>0;}
    });
    check(counts.capture_start===1&&counts.capture_end===1&&!counts.capture_error&&end.capture_error_count===0,'incomplete_or_errored_native_capture');
    check(end.raw_hit_packet_count===(counts.raw_hit_packet||0)&&end.raw_score_packet_count===(counts.raw_score_packet||0),'packet_count_mismatch');
    for(const list of [native,relay])check(list.every((p,i)=>i===0||p.unity_time>=list[i-1].unity_time),'nonmonotonic_pose_clock');
    const binding=verifyCaptureBinding(native,relay,start,end),linked=associate(capture,reference.session,packets,native,relay);
    allHits.push(...linked.hits);allScores.push(...linked.scores);
    const both=[...linked.hits,...linked.scores],pointAwards={};for(const row of linked.scores)pointAwards[row.decoded.points_awarded]=(pointAwards[row.decoded.points_awarded]||0)+1;
    records.push({capture,session:reference.session,native_source:nativeSource,relay_source:relaySource,binding,
      build:select(start,['schema','plugin_version','plugin_sha256','game_assembly_sha256','global_metadata_sha256','application_version','unity_version']),wire_bodies_validated:packets.length,
      received_hit_packets:linked.hits.length,received_score_packets:linked.scores.length,awards_by_value:pointAwards,composer_availability:composer,
      hit_receipt_associations:{unique_noncausal:linked.hits.filter(h=>h.receipt_association==='unique_same_frame_noncausal').length,
        ambiguous_noncausal:linked.hits.filter(h=>h.receipt_association==='ambiguous_noncausal').length,
        no_matching_receipt:linked.hits.filter(h=>h.receipt_association==='no_matching_receipt').length},
      native_pose_joined:both.filter(r=>r.prior_native_paired_root_pose).length,relay_pose_joined:both.filter(r=>r.prior_relay_paired_pose).length,
      native_pose_age_seconds:describe(both.map(r=>r.prior_native_paired_root_pose?.receipt_minus_pose_unity_seconds)),
      relay_pose_age_seconds:describe(both.map(r=>r.prior_relay_paired_pose?.receipt_minus_pose_unity_seconds)),
      hit_relative_speed:describe(linked.hits.map(h=>h.decoded.relative_speed)),hit_normal_magnitude:describe(linked.hits.map(h=>Math.hypot(...h.decoded.surface_normal_xyz))),
      hit_kick_flag_count:linked.hits.filter(h=>h.decoded.is_kick===1).length,native_capture_complete:true,causal_labels_created:0});
  }
  check(records.length===10&&allHits.length===151&&allScores.length===169,'prespecified_capture_or_packet_count_changed');
  const report={schema:'rek.native_hit_receipt.audit.v1',created_utc:new Date().toISOString(),exporter_sha256:hash(fs.readFileSync(__filename)),
    baseline_audit_sha256:hash(baselineBytes),supervision:SCHEMA,session_groups:baseline.session_groups,
    sessions_with_native_hit_data:[...new Set(records.map(r=>r.session))].sort(),captures:records,
    totals:{captures:records.length,hits:allHits.length,scores:allScores.length,five_point_awards:allScores.filter(s=>s.decoded.points_awarded===5).length,
      native_pose_joined:[...allHits,...allScores].filter(r=>r.prior_native_paired_root_pose).length,
      relay_pose_joined:[...allHits,...allScores].filter(r=>r.prior_relay_paired_pose).length,
      unique_noncausal_hit_score_associations:allHits.filter(h=>h.receipt_association==='unique_same_frame_noncausal').length,
      ambiguous_noncausal_hit_score_associations:allHits.filter(h=>h.receipt_association==='ambiguous_noncausal').length,
      relative_speed:describe(allHits.map(h=>h.decoded.relative_speed)),kick_flag_count:allHits.filter(h=>h.decoded.is_kick===1).length},
    no_training:true,no_simulator_changes:true,no_execution_labels:true};
  fs.mkdirSync(output,{recursive:true,mode:0o700});
  for(const [name,value,jsonl] of [['native-hit-receipts.private.jsonl',allHits,true],['native-score-receipts.private.jsonl',allScores,true],
    ['native-hit-receipt-audit.json',report,false],['native-hit-receipt-schema.json',SCHEMA,false]])
    fs.writeFileSync(path.join(output,name),jsonl?value.map(JSON.stringify).join('\n')+'\n':JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});
  return {captures:records.length,hits:allHits.length,scores:allScores.length,...report.totals,report_sha256:hash(fs.readFileSync(path.join(output,'native-hit-receipt-audit.json')))};
}
module.exports={nativePose,relayPose,nativeFilenameTime,chooseNativeFile,verifyCaptureBinding,precedingPose,packetAssociation,validatePacket,validateWirePacket,receiptGeometry,associate,scan,SCHEMA,exportAudit};
if(require.main===module)Promise.resolve().then(()=>{check(process.argv.length===6,'usage_native_hit_receipt_data_TRIAL_BASE_NATIVE_DIRECTORY_BASELINE_AUDIT_NEW_PRIVATE_OUTPUT');return exportAudit(...process.argv.slice(2));})
  .then(r=>console.log(JSON.stringify(r))).catch(e=>{console.error(/^[a-zA-Z0-9_]+$/.test(e.message)?e.message:e.code||'native_receipt_export_failed');process.exitCode=1;});
