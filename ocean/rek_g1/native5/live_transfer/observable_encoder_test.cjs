'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto'),cp=require('node:child_process');
const [binary,model]=process.argv.slice(2);
assert(binary&&model,'observable_encoder_test.cjs BINARY PRIVATE_MODEL_XML');
const schema='rek.native5.observable_balance.v1';
let assertions=0,cases=0;
function check(value,message){assert(value,message);assertions++;}
const near=(actual,expected,tolerance=1e-5)=>check(Math.abs(actual-expected)<=tolerance,`${actual} != ${expected}`);
function fixture(seq=1,ticks=1000,options={}){
 const frequency=options.frequency??1000,mask=options.countMask??0;
 const b=Buffer.alloc(33);b[0]=1;b[1]=options.roundNumber??1;b[2]=1;b[3]=options.redo?1:0;b[25]=mask;b[29]=255;
 const qpc=options.receiptQpc??ticks-20,frame=100+seq;
 const r={schema:'rek.g1_received_referee.v1',available:true,reason:'received_snapshot_applied_and_bound',
  source:'received_REK_FightState_33_byte_body',provenance:'ApplyFightStateSnapshot_prefix_copy_postfix_client_mirror_verification',authority_scope:'server_authored_packet_observed_on_client_not_server_current_state',
  observation_hooks_verified:true,maximum_receipt_age_seconds:.5,receipt_age_seconds:(ticks-qpc)/frequency,receipt_sequence:options.receiptSequence??seq,lifecycle:options.lifecycle??1,
  receipt_qpc_ticks:qpc,receipt_qpc_frequency_hz:frequency,receipt_unity_frame:frame,receipt_unity_time:frame/50,receipt_unity_unscaled_time:frame/50+.5,
  wire_body_sha256:crypto.createHash('sha256').update(b).digest('hex'),wire_body_base64:b.toString('base64'),count_mask:mask,count_seconds:0,slot0_count_active:Boolean(mask&1),slot1_count_active:Boolean(mask&2),
  call_available:false,call_sequence:0,call_type:null,call_name:null,call_faller:null,call_points:null,call_observation_sequence:null,call_sequence_transition:'empty_sequence',call_history_censored:null,
  packet_phase:1,packet_round_number:b[1],packet_round_active:true,packet_round_redo:Boolean(b[3]),packet_round_knockout_occurred:false,packet_round_result:0,server_tick:null,server_time:null,server_fight_epoch:null};
 return {event:'g1_policy_state',schema:'rek.g1_policy_source.v1',global_input_emitted:false,observation_sequence:seq,round_identity_sha256:options.round??'a'.repeat(64),local_slot:options.side??0,phase:1,stream_active:true,
  clock:{qpc_ticks:ticks,qpc_frequency_hz:frequency},fighters:[0,1].map(side=>({root_position_xyz:[side*2,1,0],root_rotation_xyzw:[0,0,0,1],fallen:false,visual_only:false,runner:{current_move_index:null}})),
  input:{punching:false,action_busy:false,velocity_command_xyz:[0,0,0],requested_move_index:null,requested_move_qpc_ticks:null,move_send_method_returned:false,desired_action:1},
  round:{number:b[1],redo:Boolean(b[3]),duration:120,time_remaining:119,active:true,result_value:0,clean_hits:options.points??[0,0],falls:[0,0]},
  referee:r,action_mask:Array(33).fill(1)};
}
function run(samples,extra=[]){
 const result=cp.spawnSync(binary,['--model',model,'--projection','client_pose_projection_v1','--observation-schema',schema,...extra],{input:samples.map(s=>typeof s==='string'?s:JSON.stringify(s)).join('\n')+'\n',encoding:'utf8',maxBuffer:32*1024*1024});
 check(result.status===0,'encoder process failed: '+result.stderr);const lines=result.stdout.trim().split('\n').map(JSON.parse);const manifest=lines.shift();
 check(manifest.observation_schema===schema&&manifest.legacy_checkpoint_compatible===false&&manifest.joint_pose_available===0,'manifest identity');
 check(manifest.structural_feature_mask.length===223&&manifest.structural_feature_mask.every(x=>x===0||x===1),'numeric structural mask');
 check(manifest.fields.length===223&&manifest.fields.every((field,index)=>field.index===index&&field.source.length>0&&field.kind.length>0&&field.structurally_available===Boolean(manifest.structural_feature_mask[index])),'indexed manifest inventory');
 check(manifest.structural_feature_mask.reduce((a,b)=>a+b,0)===166,'structural topology');
 for(const line of lines){
  if(!line.ready)continue;const o=line.worker_request.observation;
  check(o.length===223&&o.every(Number.isFinite),'finite feature dimensions');check(line.worker_request.mask.length===33,'mask dimensions');
  for(let i=0;i<223;i++)if(!manifest.structural_feature_mask[i])check(o[i]===0,'excluded structural feature '+i);
  for(const base of [0,86]){for(let i=13;i<=70;i++)check(o[base+i]===0,'unknown joint padded '+(base+i));check(o[base+74]===0&&o[base+75]===0,'joint availability false');}
  check(line.provenance.authoritative_server_state===false&&line.provenance.candidate_physics_stepped===false&&line.provenance.independent_native_receipt_match_verified===false,'authority boundaries');
 }
 cases++;return lines;
}
function ready(line){check(line.ready===true,'expected ready');return line.worker_request.observation;}
function rejected(sample,why){const row=run([sample])[0];check(row.ready===false&&!row.worker_request,why);}
function call(sample,{sequence=1,type=2,faller=0,points=1,id=1,transition='changed_received_sequence',censored=false}={}){
 const r=sample.referee,b=Buffer.from(r.wire_body_base64,'base64');b[27]=sequence;b[28]=type;b.writeInt8(faller,29);b[30]=points;
 Object.assign(r,{wire_body_base64:b.toString('base64'),wire_body_sha256:crypto.createHash('sha256').update(b).digest('hex'),call_available:true,call_sequence:sequence,call_type:type,call_name:['Slip','SlipEStop','Knockdown','BeatCount','Knockout','DoubleKnockdown','DoubleKnockout'][type]??null,call_faller:faller,call_points:points,call_observation_sequence:id,call_sequence_transition:transition,call_history_censored:censored});return sample;
}
let rows=run([fixture()]);let o=ready(rows[0]);check(o[203]===0&&o[202]===1,'first sample explicitly masks history');near(o[73],1);near(o[86],2);near(o[72],0);
for(let side=0;side<2;side++)for(let countMask=0;countMask<4;countMask++){
 const sample=fixture(1,1000,{side,countMask,points:[3,7]});o=ready(run([sample])[0]);check(o[204]===((countMask>>side)&1)&&o[205]===((countMask>>(side^1))&1),'absolute to relative count bits');check(o[190]===[3,7][side]&&o[191]===[3,7][side^1],'point perspective');
}
const a=fixture(1,1000),b=fixture(2,1020,{points:[5,1]});b.fighters[0].root_position_xyz=[.02,1.2,0];b.fighters[1].root_position_xyz=[2,1.1,.04];
rows=run([a,b]);o=ready(rows[1]);check(o[203]===1&&o[217]===5&&o[218]===1,'history and awarded point deltas');near(o[7],1);near(o[9],10,1e-4);near(o[95],5,1e-4);
const tilted=fixture();tilted.fighters[0].root_rotation_xyzw=[Math.sin(Math.PI/8),0,0,Math.cos(Math.PI/8)];o=ready(run([tilted])[0]);near(o[72],.25);check(o[71]===1,'tilted heading available');
const opposite=structuredClone(tilted);opposite.observation_sequence=2;opposite.clock.qpc_ticks=1020;opposite.referee=fixture(2,1020).referee;opposite.fighters[0].root_rotation_xyzw=tilted.fighters[0].root_rotation_xyzw.map(x=>-x);
rows=run([tilted,opposite]);for(let i=3;i<=6;i++)near(ready(rows[1])[i],ready(rows[0])[i]);near(ready(rows[1])[12],0);
const vertical=fixture();vertical.fighters[0].root_rotation_xyzw=[0,0,Math.sin(Math.PI/4),Math.cos(Math.PI/4)];o=ready(run([vertical])[0]);check(o[71]===0&&o[76]===0,'vertical forward axis masks heading');
const teleported=fixture(2,1020);teleported.fighters[0].root_position_xyz=[3,.7,0];teleported.fighters[0].fallen=true;teleported.round.falls=[1,0];o=ready(run([fixture(),teleported])[1]);check(o[203]===1,'body reset does not clear observation history');near(o[7],150);near(o[9],-15,1e-4);
for(const [dt,wanted] of [[0,0],[-1,0],[250,1],[251,0]]){o=ready(run([fixture(),fixture(2,1000+dt,dt<0?{receiptQpc:980}:{})])[1]);check(o[203]===wanted,'history interval boundary');}
rows=run([fixture(),fixture(2,1400,{points:[5,0]}),fixture(3,1420,{points:[6,0]})]);check(ready(rows[1])[203]===0&&ready(rows[1])[217]===0&&ready(rows[2])[217]===1,'gap retains current for next history');
rows=run([fixture(),fixture(2,1020,{round:'b'.repeat(64),roundNumber:2,lifecycle:2})]);check(ready(rows[1])[203]===0,'genuine round boundary masks history');
rows=run([fixture(),fixture(2,1020,{side:1})]);check(ready(rows[1])[203]===0,'perspective boundary masks history');
rows=run([fixture(),fixture(2,1020,{lifecycle:2})]);check(ready(rows[1])[203]===1,'receipt lifecycle change alone preserves pose history');
rows=run([fixture(),{type:'reset'},fixture(2,1020)]);check(ready(rows[2])[203]===0,'explicit reset clears history');
rows=run([fixture(1,1000,{points:[5,0]}),fixture(2,1020,{points:[4,0]}),fixture(3,1040,{points:[4,0]})]);check(!rows[1].ready&&ready(rows[2])[203]===0,'counter regression fails closed and resets history');
const absent=fixture();for(const key of ['receipt_sequence','lifecycle','receipt_qpc_ticks','receipt_qpc_frequency_hz','receipt_unity_frame','receipt_unity_time','receipt_unity_unscaled_time','wire_body_sha256','wire_body_base64','count_mask','count_seconds','slot0_count_active','slot1_count_active','call_sequence','call_type','call_name','call_faller','call_points','call_observation_sequence','call_sequence_transition','call_history_censored','packet_phase','packet_round_number','packet_round_active','packet_round_redo','packet_round_knockout_occurred','packet_round_result'])absent.referee[key]=null;
Object.assign(absent.referee,{available:false,reason:'referee_snapshot_not_observed',observation_hooks_verified:false,call_available:false,receipt_age_seconds:null});o=ready(run([absent])[0]);check(o[202]===0&&o[204]===0&&o[205]===0,'unavailable count zero means masked padding');
for(const [name,change] of [
 ['wire hash',s=>s.referee.wire_body_sha256='0'.repeat(64)],['wire base64',s=>s.referee.wire_body_base64=' '+s.referee.wire_body_base64],
 ['count byte',s=>s.referee.count_mask=1],['slot bits',s=>s.referee.slot0_count_active=true],['round number',s=>s.round.number=2],['round redo',s=>s.round.redo=true],
 ['clock frequency',s=>s.referee.receipt_qpc_frequency_hz=2000],['stale receipt',s=>{s.referee.receipt_qpc_ticks=499;s.referee.receipt_age_seconds=.501;}],
 ['future receipt',s=>s.referee.receipt_qpc_ticks=1001],['age mismatch',s=>s.referee.receipt_age_seconds=.03],['hooks missing',s=>s.referee.observation_hooks_verified=false],
 ['invented server tick',s=>s.referee.server_tick=10],['zero lifecycle',s=>s.referee.lifecycle=0],['absent referee',s=>delete s.referee],
 ['empty mask',s=>s.action_mask.fill(0)],['invalid root',s=>s.fighters[0].root_rotation_xyzw.fill(0)],['global input',s=>s.global_input_emitted=true],
 ]){const sample=fixture();change(sample);rejected(sample,name);}
const repeated=fixture(2,1020);repeated.referee=structuredClone(fixture().referee);repeated.referee.receipt_age_seconds=.04;rows=run([fixture(),repeated]);check(rows[1].ready,'unchanged receipt reused while fresh');
const changed=structuredClone(repeated);changed.referee.lifecycle=2;rows=run([fixture(),changed]);check(!rows[1].ready,'same sequence cannot change lifecycle');
rows=run([fixture(1,1000,{receiptSequence:3}),fixture(2,1020,{receiptSequence:2})]);check(!rows[1].ready,'receipt regression rejected');
rows=run([fixture(1,1000,{lifecycle:3}),fixture(2,1020,{lifecycle:2})]);check(!rows[1].ready,'lifecycle regression rejected');
const boundary=fixture(1,1000,{receiptQpc:500});check(run([boundary])[0].ready,'exact 500ms freshness allowed');
for(let type=0;type<8;type++)check(run([call(fixture(),{type,faller:-1})])[0].ready,'wire call type and signed faller are preserved');
for(const transition of ['initial_latched_call','same_sequence_payload_changed_censored','sequence_gap_censored','sequence_decrease_censored']){
 check(run([call(fixture(),{transition,censored:true})])[0].ready,'explicit call discontinuity is censored');
 rejected(call(fixture(),{transition,censored:false}),'uncensored discontinuity rejected');
}
for(const transition of ['changed_received_sequence','observed_255_to_1_wrap'])rejected(call(fixture(),{transition,censored:true}),'contiguous call cannot be censored');
rows=run([call(fixture()),call(fixture(2,1020),{transition:'repeated_latched_call'})]);check(rows[1].ready,'repeated latched call has stable identity');
rows=run([call(fixture()),call(fixture(2,1020),{transition:'repeated_latched_call',id:2})]);check(!rows[1].ready,'repeated call cannot acquire a new identity');
rows=run([call(fixture()),call(fixture(2,1020,{receiptSequence:4}),{type:4,sequence:2,id:1})]);check(!rows[1].ready,'call identity cannot change payload across missing receipts');
rows=run([call(fixture()),call(fixture(2,1020,{lifecycle:2}),{type:4,sequence:2,id:1})]);check(rows[1].ready,'call identity is scoped to lifecycle');
for(const [name,change] of [['call decoded points',s=>s.referee.call_points++],['call decoded name',s=>s.referee.call_name='Knockout'],['call unavailable',s=>s.referee.call_available=false]]){const sample=call(fixture());change(sample);rejected(sample,name);}
const lock=fixture();lock.fighters[0].visual_only=true;lock.input.action_busy=null;lock.input.requested_move_index=7;lock.input.requested_move_qpc_ticks=990;lock.input.move_send_method_returned=true;
rejected(lock,'visual native busy unavailable without opt-in');rows=run([lock],['--busy-projection','dispatched_request_v4_duration']);check(rows[0].provenance.projected_busy,'declared duration is separately identified');for(let k=0;k<33;k++)check(rows[0].worker_request.mask[k]===Number([0,1,6,7].includes(k)),'duration mask intersection');
for(let desired=0;desired<16;desired++){
 const sample=fixture();sample.input.desired_action=desired;sample.action_mask[17]=0;const row=run([sample])[0];const translating=(desired>=2&&desired<=5)||desired>=8;
 for(let k=0;k<33;k++)check(row.worker_request.mask[k]===Number(k!==17&&(k<16||!translating)),'held category mask preserves all source restrictions');
}
const cadence=Array.from({length:8},(_,i)=>fixture(i+1,1000+20*i));rows=run(cadence,['--action-stride','5']);for(let i=0;i<rows.length;i++)for(let k=0;k<33;k++)check(rows[i].worker_request.mask[k]===Number(i%5===0||k===0),'first-ready cadence clock');
const end=fixture();end.phase=3;end.round.active=false;end.round.result_value=1;o=ready(run([end])[0]);check(o[185]===4,'terminal semantic phase');
for(const args of [[],['--observation-schema','rek.native5.scaled_polar_xy.v1']]){const result=cp.spawnSync(binary,['--model',model,'--projection','client_pose_projection_v1',...args],{encoding:'utf8'});check(result.status===2,'schema cannot fall back to legacy');}
check(fs.statSync(binary).size>0,'binary exists');
process.stdout.write(JSON.stringify({event:'observable_encoder_tests',passed:true,cases,assertions,physics_steps:0,policy_calls:0,game_processes_launched:0})+'\n');
