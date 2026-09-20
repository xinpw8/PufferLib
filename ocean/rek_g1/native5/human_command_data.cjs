#!/usr/bin/env node
'use strict';

// Offline human command-interface observations. Never connects to a game.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const check=(ok,why)=>{if(!ok)throw Error(why);};
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const finite=Number.isFinite,vec=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finite);
const BONES=['pelvis','left_hip_pitch_link','left_hip_roll_link','left_hip_yaw_link','left_knee_link','left_ankle_pitch_link','left_ankle_roll_link','right_hip_pitch_link','right_hip_roll_link','right_hip_yaw_link','right_knee_link','right_ankle_pitch_link','right_ankle_roll_link','waist_yaw_link','waist_roll_link','torso_link','left_shoulder_pitch_link','left_shoulder_roll_link','left_shoulder_yaw_link','left_elbow_link','left_wrist_roll_link','left_wrist_pitch_link','left_wrist_yaw_link','right_shoulder_pitch_link','right_shoulder_roll_link','right_shoulder_yaw_link','right_elbow_link','right_wrist_roll_link','right_wrist_pitch_link','right_wrist_yaw_link'];
const MOVE_ORDER=[6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16];
const COMMANDS=[[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],[0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1]];
// These depend on unavailable held-intent, native settle/busy or move-route gates.
// Zero padding is not a measured zero. Every consumer must apply this same mask.
const MASKED_FIELDS=[176,177,178,179,180,181,182,183];
const FEATURE_MASK=Array.from({length:223},(_,i)=>+!MASKED_FIELDS.includes(i));
const MAX_POSE_AGE_FIXED_SECONDS=.075,MAX_SAMPLE_GAP_SECONDS=.025;
const HUMAN_HASHES=['547cec42f700e97b9594f8d2c6df88f966052c0e0e5ce7395c4862b619fb6d7f','ec55c32a6e2272a8e7656d73260ca35876be2cd6c8d8d6fe8c9dfe77cd25a309'];
const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
function norm(q){check(vec(q,4),'quaternion_shape');const n=Math.hypot(...q);check(n*n>.8&&n*n<1.2,'quaternion_norm');return q.map(x=>x/n);}
const mul=(a,b)=>[a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
const conj=q=>[q[0],-q[1],-q[2],-q[3]],unityQuat=q=>norm([-q[3],q[0],q[2],q[1]]),unityVec=v=>[v[0],v[2],v[1]];
const heading=q=>Math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));
function hinge(j,local){const q=mul(conj(j.rest),local),along=q.slice(1).reduce((s,x,i)=>s+x*j.axis[i],0);check(Math.hypot(q[0],along)>1e-8,'singular_hinge');return wrap(2*Math.atan2(along,q[0]));}

// This is deliberately a restricted reader for the fully expanded, explicit
// recovered XML. Includes/default inheritance/non-quaternion orientation fail.
function calibration(xml){
  check(!/<!|<include\b|<default\b|\b(?:euler|axisangle|xyaxes|zaxis)=/.test(xml),'unsupported_nonexpanded_model_xml');
  const stack=[],bodies=new Map(),joints=new Map(),motors=[];
  for(const match of xml.matchAll(/<\/?[^>]+>/g)){
    const tag=match[0];if(tag.startsWith('</body')){check(stack.length>0,'body_stack_underflow');stack.pop();continue;}
    const attrs=Object.fromEntries([...tag.matchAll(/([A-Za-z_][\w:]*)="([^"<>]*)"/g)].map(x=>[x[1],x[2]]));
    if(/^<body\b/.test(tag)){check(attrs.name&&!bodies.has(attrs.name),'body_identity');const b={...attrs,parent:stack.at(-1)??null};bodies.set(attrs.name,b);if(!tag.endsWith('/>'))stack.push(attrs.name);}
    if(/^<joint\b/.test(tag)){check(stack.length&&attrs.name&&!joints.has(attrs.name),'joint_identity');joints.set(attrs.name,{...attrs,body:stack.at(-1)});}
    if(/^<motor\b/.test(tag))motors.push(attrs.joint);
  }
  check(stack.length===0&&motors.length===58,'model_structure');
  const boneIndex=name=>BONES.findIndex(b=>new RegExp('^player__'+b+'_[0-9]+$').test(name));
  const numberArray=(s,n)=>{check(typeof s==='string','explicit_calibration_value_missing');const v=s.trim().split(/\s+/).map(Number);check(vec(v,n),'calibration_number_array');return v;};
  const mapped=motors.slice(0,29).map(name=>{
    const j=joints.get(name);check(j?.type==='hinge','actuator_not_hinge');const b=bodies.get(j.body),bone=boneIndex(j.body),parent=boneIndex(b.parent??'');
    check(bone>0&&parent>=0,'unmapped_joint_or_parent');const axis=numberArray(j.axis,3);check(Math.abs(Math.hypot(...axis)-1)<1e-6,'nonunit_hinge_axis');
    return {bone,parent,name:BONES[bone],rest:norm(numberArray(b.quat,4)),axis};
  });
  check(new Set(mapped.map(j=>j.bone)).size===29,'duplicate_joint_bone');
  return {schema:'rek.human_command.calibration.v1',model_sha256:sha(Buffer.from(xml)),bone_names:BONES,joints:mapped,
    source:'same expanded model actuator order, body rest quaternion and local hinge axes as native live Calibration; no physics stepped'};
}
function decodeBody(r,size){const b=Buffer.from(r.wire_body_base64,'base64');check(b.length===size&&b.toString('base64')===r.wire_body_base64&&sha(b)===r.wire_body_sha256,'wire_hash_or_length');return b;}
function pose(r,c){
  check(r.bone_count===30&&JSON.stringify(r.bone_names)===JSON.stringify(BONES)&&vec(r.world_positions_xyz,90)&&vec(r.world_rotations_xyzw,120),'bone_signature_or_shape');
  const wire=decodeBody(r,842);check(wire[0]===r.network_index&&wire[1]===30,'bone_wire_header');
  for(let i=0;i<30;i++)for(let k=0;k<7;k++)check(wire.readFloatLE(2+28*i+4*k)===Math.fround(k<3?r.world_positions_xyz[3*i+k]:r.world_rotations_xyzw[4*i+k-3]),'bone_wire_projection');
  const rotations=BONES.map((_,i)=>unityQuat(r.world_rotations_xyzw.slice(4*i,4*i+4)));
  const positions=BONES.map((_,i)=>unityVec(r.world_positions_xyz.slice(3*i,3*i+3)));
  return {q:c.joints.map(j=>hinge(j,mul(conj(rotations[j.parent]),rotations[j.bone]))),positions,
    fixed_tick:r.client_fixed_tick_at_observation,receipt_time:r.monotonic_receipt_time,unity_unscaled_time:r.unity_unscaled_time};
}
function mapCommand(r){
  if(r.message==='REK_Move'){check(Number.isInteger(r.move_index_wire_uint8)&&r.move_index_wire_uint8>=0&&r.move_index_wire_uint8<17,'move_id');return {action:16+MOVE_ORDER.indexOf(r.move_index_wire_uint8),reason:'observed_one_shot_move_request'};}
  check(r.message==='REK_Input'&&vec(r.velocity_command_xyz,3),'unknown_command');
  const v=r.velocity_command_xyz;
  if(v.some(x=>![-1,0,1].includes(x)))return {action:-1,reason:'continuous_or_ramped_output_not_quantized'};
  const index=COMMANDS.findIndex(a=>a.every((x,i)=>x===v[i]));
  return index<0?{action:-1,reason:'combined_translation_not_in_33_action_vocabulary'}:
    {action:index+1,reason:index===0?'observed_neutral_output_not_human_release_intent':'exact_discrete_outgoing_command'};
}
function validateCommand(r){
  const move=r.message==='REK_Move';check(move||r.message==='REK_Input','unsupported_request_type');
  const b=decodeBody(r,move?2:13);check(r.request_only===true&&r.server_acceptance===null&&r.ack_observed===false&&r.network_index_wire_uint8===0&&b[0]===0,'request_scope_or_semantics');
  if(move)check(b[1]===r.move_index_wire_uint8,'move_wire_projection');
  else{check(vec(r.velocity_command_xyz,3),'command_shape');for(let i=0;i<3;i++)check(b.readFloatLE(1+4*i)===Math.fround(r.velocity_command_xyz[i]),'command_wire_projection');}
  check(Number.isSafeInteger(r.request_sequence)&&r.request_sequence>0&&finite(r.unity_realtime_since_startup)&&Number.isSafeInteger(r.client_fixed_tick_at_observation),'command_clock');
}
function samplePose(s,packets){
  return [0,1].map(slot=>{
    const f=s['fighter_'+slot],p=packets[slot];check(f&&vec(f.root_position,3)&&vec(f.root_rotation,4)&&p,'sample_pose_missing');
    const rotation=unityQuat(f.root_rotation);return {...p,root:unityVec(f.root_position),rotation,heading:heading(rotation)};
  });
}
class Projection{
  constructor(){this.reset();}
  reset(){this.previous=null;this.hitAge=[0,0];this.hitSpeed=[0,0];this.hitValid=[false,false];}
  process(s,packets,line){
    const before=this.previous,t=s.unity_unscaled_time;
    const good=s.phase_value===1&&s.round?.active===true&&s.round?.redo===false&&s.round?.result_value===0&&s.local_fighter_index===0;
    if(!good){this.reset();return {ready:false,reason:'inactive_or_redo_or_wrong_slot'};}
    check(finite(t)&&finite(s.round.duration)&&s.round.duration>0&&finite(s.round.time_remaining)&&s.round.time_remaining>=0&&s.round.time_remaining<=s.round.duration,'sample_clock');
    check(vec(s.round.clean_hits,2)&&s.round.clean_hits.every(x=>Number.isInteger(x)&&x>=0),'score_counter');
    if(!packets.every(Boolean)){this.reset();return {ready:false,reason:'pose_warmup'};}
    const ages=packets.map(p=>(s.client_fixed_tick-p.fixed_tick)*.002);
    if(ages.some(x=>x<0||x>MAX_POSE_AGE_FIXED_SECONDS)){this.reset();return {ready:false,reason:'pose_age_outside_fixed_clock_guard',ages};}
    const current={s,t,poses:samplePose(s,packets),line};this.previous=current;
    if(!before)return {ready:false,reason:'derivative_warmup'};
    const dt=t-before.t;
    if(dt<=0||dt>MAX_SAMPLE_GAP_SECONDS||s.sample_index!==before.s.sample_index+1||s.client_fixed_tick-before.s.client_fixed_tick!==10||s.round.number!==before.s.round.number||s.fight_epoch!==before.s.fight_epoch){
      this.reset();this.previous=current;return {ready:false,reason:'gap_or_identity_change'};
    }
    check(s.round.time_remaining<=before.s.round.time_remaining+.001,'round_timer_increased');
    const obs=Array(223).fill(0),delta=s.round.clean_hits.map((n,i)=>n-before.s.round.clean_hits[i]);check(delta.every(n=>n>=0),'score_decrease');
    for(let slot=0;slot<2;slot++){
      const b=86*slot,p=current.poses[slot],old=before.poses[slot];
      obs.splice(b,3,...p.root);obs.splice(b+3,4,...p.rotation);
      const vx=(p.root[0]-old.root[0])/dt,vy=(p.root[1]-old.root[1])/dt;
      obs[b+7]=Math.cos(p.heading)*vx+Math.sin(p.heading)*vy;obs[b+8]=-Math.sin(p.heading)*vx+Math.cos(p.heading)*vy;obs[b+12]=wrap(p.heading-old.heading)/dt;
      obs.splice(b+13,29,...p.q);obs.splice(b+42,29,...p.q.map((q,j)=>wrap(q-old.q[j])/dt));obs[b+73]=p.root[2];obs[b+77]=2;
      if(this.hitValid[slot])this.hitAge[slot]=Math.min(120,this.hitAge[slot]+dt);
      if(delta[slot^1]>0){this.hitValid[slot]=true;this.hitAge[slot]=0;this.hitSpeed[slot]=Math.max(...[6,12,22,29,4,10].map(e=>Math.hypot(...current.poses[slot^1].positions[e].map((x,k)=>x-before.poses[slot^1].positions[e][k]))/dt));}
      obs[190+slot]=s.round.clean_hits[slot];obs[196+slot]=+this.hitValid[slot];obs[198+slot]=this.hitAge[slot];obs[200+slot]=this.hitSpeed[slot];obs[217+slot]=delta[slot];
    }
    const p=current.poses[0],e=current.poses[1],dx=e.root[0]-p.root[0],dy=e.root[1]-p.root[1];
    obs[86]=Math.hypot(dx,dy);obs[87]=wrap(Math.atan2(dy,dx)-p.heading)/Math.PI;obs[172]=Math.cos(.5*p.heading);obs[175]=Math.sin(.5*p.heading);
    obs[184]=0;obs[185]=2;obs[186]=1;obs[188]=s.round.duration/120;obs[189]=s.round.time_remaining/120;obs[209]=.5;
    obs[210]=s.round.result_value;obs[211]=s.round.winner_index;obs[212]=+(s.round.result_value===2);obs[213]=s.fight.result_value;obs[214]=s.fight.winner_index;obs[221]=obs[222]=delta[0]+delta[1];
    check(vec(obs,223)&&obs.every(x=>finite(Math.fround(x))),'nonfinite_observation');
    return {ready:true,observation:obs,pose_ages:ages,dt,source_lines:[before.line,line,...packets.map(p=>p.line)]};
  }
}

async function scan(file,visit){const initial=fs.statSync(file),stream=fs.createReadStream(file),digest=crypto.createHash('sha256');stream.on('data',b=>digest.update(b));let n=0;
  for await(const text of readline.createInterface({input:stream,crlfDelay:Infinity})){n++;if(!text.trim())continue;try{visit(JSON.parse(text),n);}catch(e){throw Error(path.basename(file)+':'+n+':'+e.message);}}
  const final=fs.statSync(file);check(initial.size===final.size&&initial.mtimeMs===final.mtimeMs,'source_changed');return {file:path.resolve(file),sha256:digest.digest('hex'),bytes:final.size,lines:n};}
const inc=(map,key)=>{map[key]=(map[key]??0)+1;};
async function readCapture(file,c,split){
  const rows=[],ledger=[],packets=[null,null],projection=new Projection(),reasons={},mapping={},events={};
  let header=null,footer=null,pending=null,sequence=split*1000000,needReset=true,firstTime=null,lastMove=null,lastMovement=null,lastRequest=null,roundNumber=null;
  const finish=(complete,incompleteReason='right_censored_decision_interval')=>{
    if(!pending)return;const commands=pending.commands,moves=commands.filter(r=>r.message==='REK_Move');let selected=null,reason;
    if(!complete)reason=incompleteReason;
    else if(moves.length>1)reason='multiple_one_shot_requests_in_50hz_interval';
    else if(moves.length===1){selected=moves[0];reason=selected.mapping.reason;}
    else if(commands.length===0)reason='no_observed_outgoing_command';
    else{selected=commands.at(-1);reason=selected.mapping.reason;
      // Preserve any within-interval changes rather than silently choosing the
      // last command. Pure repeated wire bodies have one unambiguous target.
      if(new Set(commands.map(x=>x.wire_body_sha256)).size>1){selected=null;reason='multiple_distinct_movement_outputs_in_50hz_interval';}
    }
    const action=selected?.mapping.action??-1;
    rows.push({...pending,commands:undefined,split,sequence_id:sequence,reset_before:needReset?1:0,action,weight:action>=0?1:0,target_reason:reason,target_id:selected?.id??null,
      target_ids:commands.map(x=>x.id),decision_interval_complete:complete});
    inc(reasons,reason);needReset=false;pending=null;
  };
  const provenance=await scan(file,(r,line)=>{
    inc(events,r.event);
    if(r.event==='capture_start'){check(!header&&!footer&&r.schema==='rek.private_ai.protocol.v5'&&r.scope?.allowed===true&&r.scope.local_fighter_index===0&&r.scope.opponent_is_ai===true&&r.scope.human_in_opponent_slot===false,'capture_scope');header=r;return;}
    check(header&&!footer,'record_outside_capture');
    if(r.event==='capture_error')throw Error('capture_error');
    if(r.event==='raw_bone_packet'){check([0,1].includes(r.fighter_slot),'pose_slot');const value=pose(r,c);packets[r.fighter_slot]={...value,line};return;}
    if(r.event==='sample'){
      const adjacent=!pending||(r.sample_index===pending.sample_index+1&&r.client_fixed_tick-pending.client_fixed_tick===10&&
        r.unity_unscaled_time-pending.unscaled_time>0&&r.unity_unscaled_time-pending.unscaled_time<=MAX_SAMPLE_GAP_SECONDS&&r.round.number===pending.round_number&&r.fight_epoch===pending.fight_epoch&&r.phase_value===1&&r.round.active===true&&!r.round.redo);
      finish(adjacent,'gap_or_phase_boundary_after_observation');roundNumber??=r.round.number;check(r.round.number===roundNumber,'multiple_rounds_in_capture');
      const projected=projection.process(r,packets,line);
      if(!projected.ready){inc(reasons,'observation_'+projected.reason);needReset=true;sequence++;return;}
      firstTime??=r.unity_unscaled_time;
      pending={observation:projected.observation,time:r.unity_unscaled_time-firstTime,unscaled_time:r.unity_unscaled_time,fight_epoch:r.fight_epoch,sample_index:r.sample_index,source_line:line,source_lines:projected.source_lines,
        client_fixed_tick:r.client_fixed_tick,pose_ages:projected.pose_ages,round_number:r.round.number,commands:[],
        preceding_history:{last_move:lastMove,last_movement:lastMovement,last_request:lastRequest}};return;
    }
    if(r.event==='outbound_request_projection'){
      validateCommand(r);check(lastRequest?(r.request_sequence===lastRequest.request_sequence+1&&r.unity_realtime_since_startup>=lastRequest.time):r.request_sequence===1,'request_order_or_gap');
      const mapped=mapCommand(r),id=`R${split+1}-C${r.request_sequence}`;
      const entry={id,message:r.message,source_line:line,request_sequence:r.request_sequence,client_fixed_tick:r.client_fixed_tick_at_observation,time:r.unity_realtime_since_startup,
        native_move_id:r.move_index_wire_uint8??null,velocity_command_xyz:r.velocity_command_xyz??null,wire_body_sha256:r.wire_body_sha256,wire_body_base64:r.wire_body_base64,mapping:mapped,
        preceding_move:lastMove,preceding_movement:lastMovement,decision_source_line:pending?.source_line??null,
        human_physical_held_intent:null,server_acceptance:null,execution:null};
      check(!pending||line>pending.source_line&&r.client_fixed_tick_at_observation>=pending.client_fixed_tick,'target_not_after_observation');
      ledger.push(entry);inc(mapping,mapped.reason);if(pending)pending.commands.push(entry);
      const summary={id,source_line:line,request_sequence:r.request_sequence,time:r.unity_realtime_since_startup,move:r.move_index_wire_uint8??null,command:r.velocity_command_xyz??null};
      if(r.message==='REK_Move')lastMove=summary;else lastMovement=summary;lastRequest=summary;return;
    }
    if(r.event==='capture_end'){footer=r;finish(false);}
  });
  check(header&&footer&&footer.capture_error_count===0&&events.sample>1,'incomplete_capture');
  check(footer.sample_count===events.sample&&footer.raw_bone_packet_count===events.raw_bone_packet&&footer.client_transport_invocation_count===ledger.length&&
    footer.client_transport_method_counts?.SendMoveEvent===ledger.filter(r=>r.message==='REK_Move').length&&
    footer.client_transport_method_counts?.SendVelocityCommand===ledger.filter(r=>r.message==='REK_Input').length,'footer_count_mismatch');
  check(rows.length>0&&ledger.length>0,'empty_capture');
  for(const row of rows)check(row.source_lines.every(n=>n<=row.source_line)&&row.target_ids.every(id=>ledger.find(x=>x.id===id).source_line>row.source_line),'future_information');
  return {provenance,rows,ledger,summary:{split,round_number:roundNumber,pid:header.pid,events,rows:rows.length,labeled_rows:rows.filter(r=>r.action>=0).length,
    labeled_moves:rows.filter(r=>r.action>=16).length,unlabeled_rows:rows.filter(r=>r.action<0).length,sequences:new Set(rows.map(r=>r.sequence_id)).size,
    reasons,mapping,move_requests:ledger.filter(r=>r.message==='REK_Move').length,movement_requests:ledger.filter(r=>r.message==='REK_Input').length,
    maximum_pose_age_fixed_seconds:Math.max(...rows.flatMap(r=>r.pose_ages)),action_counts:rows.reduce((m,r)=>(inc(m,r.action),m),{})}};
}
function binary(rows){
  const b=Buffer.alloc(256+rows.length*1056);b.write('REKBC001',0,'ascii');[1,223,33,rows.length,1056,0].forEach((x,i)=>b.writeUInt32LE(x,8+4*i));FEATURE_MASK.forEach((x,i)=>b[32+i]=x);
  rows.forEach((r,i)=>{const o=256+i*1056;check([0,1].includes(r.split)&&Number.isInteger(r.sequence_id)&&[0,1].includes(r.reset_before)&&Number.isInteger(r.action)&&r.action>=-1&&r.action<33&&vec(r.observation,223)&&finite(r.time)&&r.time>=0&&finite(r.weight)&&r.weight===(r.action<0?0:1),'binary_row');
    b.writeUInt32LE(r.split,o);b.writeUInt32LE(r.sequence_id,o+4);b.writeUInt32LE(r.reset_before,o+8);b.writeInt32LE(r.action,o+12);b.writeFloatLE(r.weight,o+16);b.writeDoubleLE(r.time,o+24);
    r.observation.forEach((v,k)=>b.writeFloatLE(FEATURE_MASK[k]?v:0,o+32+4*k));for(let k=0;k<33;k++)b.writeFloatLE(1,o+924+4*k);
  });return b;
}
async function verifyNativeCalibration(c,sourceFile,encodedFile){
  const expected=new Map();let comparisons=0,maximum=0,manifestSeen=false;
  const source=await scan(sourceFile,r=>{if(r.event!=='g1_policy_state')return;
    expected.set(r.observation_sequence,r.fighters.map(f=>c.joints.map(j=>hinge(j,unityQuat(f.bone_local_rotations_xyzw[j.bone])))));
  });
  const encoded=await scan(encodedFile,r=>{
    if(r.event==='projection_manifest'){check(r.model_sha256===c.model_sha256,'native_model_calibration_mismatch');manifestSeen=true;}
    if(r.event!=='policy_observation'||!r.ready)return;
    const predicted=expected.get(r.worker_request.seq);check(predicted,'native_observation_sequence_unmatched');
    for(let slot=0;slot<2;slot++)for(let j=0;j<29;j++){
      const error=Math.abs(wrap(predicted[slot][j]-r.worker_request.observation[86*slot+13+j]));maximum=Math.max(maximum,error);comparisons++;
      check(error<=1e-10,'native_calibration_projection_mismatch');
    }
  });
  check(manifestSeen&&comparisons>0,'native_calibration_verification_empty');
  return {schema:'rek.human_command.native_calibration_check.v1',comparisons,maximum_angle_error_radians:maximum,model_sha256:c.model_sha256,
    inputs:[source,encoded],semantics:'Exact same recovered XML calibration and hinge projection against recorded native live encoder outputs. Human world-to-local conversion has separate synthetic tests; human network-pose/rendered-pose timing equivalence is not claimed.',passed:true};
}
async function run(model,train,heldout,out){
  check(!fs.existsSync(out),'output_exists');const xml=fs.readFileSync(model),c=calibration(xml.toString('utf8'));
  const captures=[];for(const [split,file] of [train,heldout].entries())captures.push(await readCapture(file,c,split));
  check(captures.every((x,i)=>x.provenance.sha256===HUMAN_HASHES[i]),'not_the_pinned_human_demonstrations');
  check(captures[0].summary.pid===captures[1].summary.pid,'human_session_identity_changed');
  const rows=captures.flatMap(x=>x.rows),data=binary(rows),ledger=captures.flatMap(x=>x.ledger);
  const manifest={schema:'rek.human_command_imitation.v1',tool_sha256:sha(fs.readFileSync(__filename)),created_utc:new Date().toISOString(),sources:captures.map(x=>x.provenance),
    model:{file:path.resolve(model),sha256:sha(xml)},projection:'client_pose_projection_v1_masked_human_packet_joints_v1',observation_schema:'rek.native5.scaled_polar_xy.v1',feature_mask:FEATURE_MASK,
    feature_mask_sha256:sha(Buffer.from(FEATURE_MASK)),masked_feature_indices:MASKED_FIELDS,masked_reason:'native held intent, transition-settled, busy and route are unavailable; zero padding is not measured zero',
    fixed_mask_required_in:['behavioral_cloning','live_policy_inference','subsequent_PPO_rollout_and_training'],
    projection_semantics:'Rendered sample roots and scores; root derivatives from preceding50Hz sample; hinge angles from strictly preceding received world-bone quaternions via recovered parent/rest/axis calibration. Joint packet age is explicit. Legacy structural zeros remain structural, not evidence of balanced physics.',
    clock_contract:'Strict original source-line order plus shared recorder500Hz client_fixed_tick;50Hz sample.unity_unscaled_time for root derivatives. Raw callback realtime and FixedUpdate clocks are not substituted for each other.',
    history_contract:'All prior command IDs/times are preserved in row and command ledgers. Current/future targets never enter observations. Masked legacy slots are not repurposed for history.',
    target_semantics:'Observed human native outgoing command-interface requests; no physical-key intent, send completion, server acceptance or executed-action assertion. Neutral output can be automatic; never labeled as a human release.',
    action_mapping:{move_order:MOVE_ORDER,movement_action_1_through_15:COMMANDS,hold_action_0:'never inferred from missing messages',continuous_yaw:'unlabeled; no sign quantization',unsupported_combination:'unlabeled'},
    class_support:'All33 known interface categories, for unconditioned command classification only. Not a measured native game-legality mask.',
    alignment:'Observation at sampled state; targets are later file-ordered commands before the next sample. One move request has priority while all outgoing movement records remain in ledger. Multiple moves or changing movement outputs in one interval are unlabeled. Final interval is right-censored.',
    split:'Whole first round training; whole second round held out. Both belong to one human session, not independent held-out sessions.',normalization:'none; native scaled_polar_xy field units unchanged',
    guards:{maximum_pose_age_client_fixed_seconds:MAX_POSE_AGE_FIXED_SECONDS,maximum_sample_gap_seconds:MAX_SAMPLE_GAP_SECONDS,consecutive_sample_indices:true,client_fixed_tick_stride:10,active_phase:1,redo_allowed:false},
    binary:{magic:'REKBC001',header_bytes:256,row_bytes:1056,byte_order:'little_endian',layout:'header:version@8,obs@12,actions@16,rows@20,rowBytes@24,reserved@28,featureMask223uint8@32,pad@255; row:splitu32@0,sequenceu32@4,resetu32@8,actioni32@12,weightf32@16,reserved@20,timef64@24,obs223f32@32,classSupport33f32@924',sha256:sha(data),bytes:data.length},
    captures:captures.map(x=>x.summary),total_rows:rows.length,total_command_records:ledger.length,no_game_connection:true,no_training:true};
  fs.mkdirSync(out);const write=(name,value)=>fs.writeFileSync(path.join(out,name),typeof value==='string'||Buffer.isBuffer(value)?value:JSON.stringify(value,null,2)+'\n',{flag:'wx'});
  write('human-commands.bin',data);write('feature-mask.bin',Buffer.from(FEATURE_MASK));write('manifest.json',manifest);write('calibration.json',c);write('command-ledger.jsonl',ledger.map(x=>JSON.stringify(x)).join('\n')+'\n');
  write('row-ledger.jsonl',rows.map(({observation,...r})=>JSON.stringify(r)).join('\n')+'\n');return manifest;
}
module.exports={BONES,MOVE_ORDER,COMMANDS,FEATURE_MASK,MASKED_FIELDS,HUMAN_HASHES,calibration,pose,hinge,mul,conj,norm,unityQuat,unityVec,Projection,mapCommand,validateCommand,readCapture,binary,verifyNativeCalibration,run};
if(require.main===module)Promise.resolve().then(()=>{check(process.argv.length===6,'usage_human_command_data_MODEL_XML_TRAIN_RAW_HELDOUT_RAW_NEW_OUTPUT');return run(...process.argv.slice(2));}).then(r=>console.log(JSON.stringify({rows:r.total_rows,commands:r.total_command_records,captures:r.captures}))).catch(e=>{console.error(e.message);process.exitCode=1;});
