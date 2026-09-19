#!/usr/bin/env node
'use strict';

// Offline validation of frozen diagnostic root predictors. No fitting or game connection.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const balance=require('./balance_transition_data.cjs');
const referee=require('./native_referee_data.cjs');
const check=(ok,why)=>{if(!ok)throw Error(why);};
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const finite=Number.isFinite,wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
const BONES=['left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link'];
const HORIZONS=[.1,.25,.5,1],STRATA=['all','local_count_overlap','opponent_count_overlap','no_count_overlap'];
const localPair=(x,z,yaw)=>[Math.cos(yaw)*x+Math.sin(yaw)*z,-Math.sin(yaw)*x+Math.cos(yaw)*z];
async function scan(file,visit){
  const before=fs.statSync(file),digest=crypto.createHash('sha256'),input=fs.createReadStream(file);
  input.on('data',b=>digest.update(b));let line=0;
  for await(const raw of readline.createInterface({input,crlfDelay:Infinity})){
    line++;if(!raw.trim())continue;let r;try{r=JSON.parse(raw);}catch{throw Error('invalid_json_line_'+line);}visit(r,line);
  }
  const after=fs.statSync(file);check(before.size===after.size&&before.mtimeMs===after.mtimeMs,'input_changed');
  return {file:path.resolve(file),bytes:after.size,lines:line,sha256:digest.digest('hex')};
}
function readJson(file){const bytes=fs.readFileSync(file);return {value:JSON.parse(bytes),source:{file:path.resolve(file),bytes:bytes.length,sha256:hash(bytes)}};}
function validateModel(model){
  const p=balance.FEATURE_NAMES.length;
  check(model.lambda===.01&&Number.isSafeInteger(model.fit_rows)&&model.fit_rows>p,'unsupported_frozen_model');
  check(model.mean?.length===p&&model.scale?.length===p&&model.weights?.length===p+1,'model_shape');
  check(model.mean.every(finite)&&model.scale.every(x=>finite(x)&&x>0)&&model.weights.every(w=>w.length===5&&w.every(finite)),'model_values');
}
function validateSessionSeparation(captures,prior){
  const groups=balance.groupSessions(captures);
  for(const capture of captures)for(const old of prior.session_groups)
    check(Math.abs(capture.origin-Date.parse(old.origin_utc))>1000,'historical_session_overlap');
  for(const group of groups){const members=captures.filter(c=>c.session===group.id);check(new Set(members.map(c=>c.pid)).size===1,'group_multiple_process_ids');}
  return groups.map(g=>({id:'external-'+g.id,origin_utc:new Date(g.origin).toISOString(),pid:captures.find(c=>c.session===g.id).pid,captures:g.captures}));
}
async function loadCapture(spec){
  const rows=[],origins=[],frames=new Map(),roundIds=new Set(),pointTimes=[-Infinity,-Infinity];
  let points=null,first=null;const availability={actor_observations:0,visual_only:0,falling_true:0,fallen_true:0,floor_contact_nonzero:0,raw_root_velocity_nonzero:0,nonzero_round_falls:0};
  const relay=await scan(path.join(spec.trial,'relay.stdout.jsonl'),r=>{
    if(r.event==='state'&&finite(r.unity_unscaled_time)&&finite(Date.parse(r.observed_utc)))origins.push(Date.parse(r.observed_utc)-1000*r.unity_unscaled_time);
    if(r.event!=='g1_policy_state')return;
    check(r.schema==='rek.g1_policy_source.v1'&&[0,1].includes(r.local_slot)&&r.fighters.length===2,'invalid_policy_source');
    first??=r;check(r.local_slot===first.local_slot&&r.clock.qpc_frequency_hz===first.clock.qpc_frequency_hz,'policy_identity_change');roundIds.add(r.round_identity_sha256);
    const t=r.clock.qpc_ticks/r.clock.qpc_frequency_hz;check(!rows.length||t>rows.at(-1).t,'policy_clock_not_increasing');
    if(points)for(let slot=0;slot<2;slot++){check(r.round.clean_hits[slot]>=points[slot],'point_counter_decreased');if(r.round.clean_hits[slot]!==points[slot])pointTimes[slot]=t;}points=r.round.clean_hits;
    const states=r.fighters.map(balance.rootState),actors=r.fighters.map((f,slot)=>{
      availability.actor_observations++;availability.visual_only+=f.visual_only===true;availability.falling_true+=f.falling===true;availability.fallen_true+=f.fallen===true;
      availability.floor_contact_nonzero+=f.floor_contact_count>0;availability.raw_root_velocity_nonzero+=Array.isArray(f.root_linear_velocity_xyz)&&f.root_linear_velocity_xyz.some(n=>n!==0);
      const pose=BONES.flatMap(name=>{const i=f.bone_names.indexOf(name);check(i>=0,'missing_bone');const p=f.bone_world_positions_xyz[i];return [...localPair(p[0]-states[slot][0],p[2]-states[slot][1],states[slot][3]),p[1]-states[slot][2]];});
      return {slot,state:states[slot],opponent:states[slot^1],pose};
    });
    availability.nonzero_round_falls+=r.round.falls.some(n=>n!==0);
    const request=r.input.requested_move_qpc_ticks;
    rows.push({t,unity_time:r.clock.unity_time,sequence:r.observation_sequence,active:r.round.active===true,actors,local_slot:r.local_slot,
      command:r.input.velocity_command_xyz,desired_action:r.input.desired_action,requested_attack:r.input.requested_move_index!=null,
      requested_move_index:r.input.requested_move_index??null,requested_move_qpc_ticks:request??null,
      request_age:request==null?3:Math.max(0,Math.min(3,t-request/r.clock.qpc_frequency_hz)),point_age:pointTimes.map(x=>t-x),points:r.round.clean_hits,falls:r.round.falls});
    if(!frames.has(r.clock.unity_frame))frames.set(r.clock.unity_frame,[]);
    frames.get(r.clock.unity_frame).push({clock:r.clock,roots:r.fighters.map(f=>f.root_position_xyz)});
  });
  check(rows.length>2&&origins.length>0&&roundIds.size===1,'capture_identity_unavailable');origins.sort((a,b)=>a-b);
  check(origins.at(-1)-origins[0]<1000,'process_origin_drift');
  let header=null,end=null,epoch=null,errorCount=0;const packets=[],anchors=[],recovery={observations:0,complete:0,can_get_up_true:0};
  const native=await scan(spec.native_file,(r,line)=>{
    if(r.event==='capture_start'){check(header===null,'multiple_capture_headers');header=r;epoch=r.initial_state?.fight_epoch;check(r.pid===spec.expected_pid&&r.scene==='Arena'&&r.scope.local_fighter_index===first.local_slot,'native_identity_mismatch');check(r.initial_state.round.number===first.round.number,'native_round_mismatch');check(r.stopwatch_frequency_hz===first.clock.qpc_frequency_hz,'native_clock_frequency');}
    if(r.event==='capture_end'){check(end===null,'multiple_capture_ends');end=r;}
    if(r.event==='capture_error')errorCount++;
    if(r.event==='sample')for(const slot of [0,1]){const a=r['fighter_'+slot]?.recovery_authority;if(a){recovery.observations++;recovery.complete+=a.complete===true;recovery.can_get_up_true+=a.can_get_up===true;}}
    if(r.event==='raw_fight_state_packet'){referee.validateRefereePacket(r);check(r.decoded.round_number===first.round.number,'native_packet_round_mismatch');packets.push({...r,source_line:line,fight_epoch:epoch});}
    if(r.event!=='root_pose_sample')return;epoch=r.fight_epoch;
    if(r.root_pose_sample_index%50!==0)return;
    for(const p of frames.get(r.unity_frame)||[]){
      if(Math.abs(r.stopwatch_timestamp_ticks-p.clock.qpc_ticks)/first.clock.qpc_frequency_hz>.15||Math.abs(r.unity_time-p.clock.unity_time)>.05)continue;
      if([0,1].every(s=>Math.hypot(...r['fighter_'+s+'_root'].world_position_xyz.map((x,i)=>x-p.roots[s][i]))<=.05)){anchors.push(p.clock.unity_time);break;}
    }
  });
  check(header&&end&&!errorCount&&end.capture_error_count===0,'native_capture_incomplete');
  check(packets.length===end.raw_fight_state_packet_count,'native_packet_count_mismatch');
  check(anchors.length>=2&&Math.max(...anchors)-Math.min(...anchors)>=1,'native_pose_clock_binding_missing');
  const extracted=referee.extractCalls(spec.id,'pid'+header.pid,packets);
  const counts=extracted.counts.filter(c=>!c.left_censored&&!c.right_censored).map(c=>({slot:c.faller_slot,start:c.start_receipt_clock.unity_time,end:c.end_receipt_clock.unity_time,explicit_countout:c.explicit_countout,resolution:c.resolution_call_name}));
  return {id:spec.id,rows,origin:origins[Math.floor((origins.length-1)/2)],pid:header.pid,round_identity_sha256:[...roundIds][0],round_number:first.round.number,
    local_slot:first.local_slot,provenance:{relay,native},counts,referee_calls:extracted.calls.filter(c=>c.observed_new_call).map(c=>({name:c.call_name,slot:c.faller_slot,points:c.points,unity_time:c.first_receipt_clock.unity_time})),
    availability,recovery:{...recovery,sample_semantics:'continuous native sample records only',initial_authority:[0,1].map(slot=>header.initial_state?.['fighter_'+slot]?.recovery_authority??null)},
    binding:{matching_pose_anchors:anchors.length,anchor_span_seconds:Math.max(...anchors)-Math.min(...anchors)},initial_time_remaining:first.round.time_remaining};
}
function strata(capture,start,end){
  const overlap=slot=>capture.counts.some(c=>c.slot===slot&&c.start<=end&&c.end>=start);
  const local=overlap(capture.local_slot),opponent=overlap(capture.local_slot^1);
  return ['all',...(local?['local_count_overlap']:[]),...(opponent?['opponent_count_overlap']:[]),...(!local&&!opponent?['no_count_overlap']:[])];
}
function emptyMetric(){return {count:0,nonfinite:0,sum:[0,0,0,0]};}
function addMetric(m,predicted,actual){
  m.count++;if(!predicted.every(finite)){m.nonfinite++;return;}
  m.sum[0]+=(predicted[0]-actual[0])**2+(predicted[1]-actual[1])**2;m.sum[1]+=(predicted[2]-actual[2])**2;
  m.sum[2]+=wrap(predicted[3]-actual[3])**2;m.sum[3]+=(predicted[4]-actual[4])**2;
}
function finishMetric(m){const n=m.count-m.nonfinite;return {count:m.count,nonfinite:m.nonfinite,planar_rmse_unity_units:n?Math.sqrt(m.sum[0]/n):null,height_rmse_unity_units:n?Math.sqrt(m.sum[1]/n):null,yaw_rmse_degrees:n?Math.sqrt(m.sum[2]/n)*180/Math.PI:null,tilt_rmse_degrees:n?Math.sqrt(m.sum[3]/n)*180/Math.PI:null};}
function evaluate(capture,model){
  const rows=balance.transitions(capture),map=new Map(rows.map(r=>[r.index,r])),metrics={};
  for(const horizon of ['one_step',...HORIZONS])metrics[horizon]=Object.fromEntries(STRATA.map(s=>[s,Object.fromEntries(['frozen_root','constant_velocity','ridge'].map(m=>[m,emptyMetric()]))]));
  const put=(h,start,end,predicted,actual)=>{for(const s of strata(capture,start,end))for(const m of Object.keys(predicted))addMetric(metrics[h][s][m],predicted[m],actual);};
  for(const r of rows)put('one_step',capture.rows[r.index].unity_time,capture.rows[r.index+1].unity_time,{frozen_root:r.state,constant_velocity:balance.advance(r.state,r.velocity,r.dt),ridge:balance.advance(r.state,balance.predict(model,r.x),r.dt)},r.next_state);
  for(const horizon of HORIZONS){let nextStart=-Infinity;
    for(let i=1;i<capture.rows.length-1;i++){
      const start=map.get(i);if(!start||start.t<nextStart)continue;let j=i,valid=true;
      while(j<capture.rows.length-1&&capture.rows[j].t-start.t<horizon){if(!map.has(j)){valid=false;break;}j++;}
      if(!valid||j>=capture.rows.length-1)continue;
      let state=[...start.state],velocity=[...start.velocity];
      for(let k=i;k<j;k++){const recorded=map.get(k),elapsed=recorded.t-start.t;
        const context={...start,command:recorded.command,requested_attack:recorded.requested_attack,request_age:recorded.request_age,point_age:start.point_age.map(age=>age+elapsed)};
        velocity=balance.predict(model,balance.features(start,state,velocity,context));state=balance.advance(state,velocity,recorded.dt);
      }
      const dt=capture.rows[j].t-start.t,prev=capture.rows[i-1].actors[start.slot].state;
      const inertial=[start.state[0]+(start.state[0]-prev[0])*dt/start.previous_dt,start.state[1]+(start.state[1]-prev[1])*dt/start.previous_dt,start.state[2]+start.velocity[2]*dt,wrap(start.state[3]+start.velocity[3]*dt),start.state[4]+start.velocity[4]*dt];
      put(horizon,capture.rows[i].unity_time,capture.rows[j].unity_time,{frozen_root:start.state,constant_velocity:inertial,ridge:state},capture.rows[j].actors[start.slot].state);nextStart=capture.rows[j].t;
    }
  }
  return {transition_rows:rows.length,metrics};
}
function pool(results){
  const first=results[0].metrics,out={};for(const h of Object.keys(first)){out[h]={};for(const s of STRATA){out[h][s]={};for(const m of Object.keys(first[h][s])){
    const sum=emptyMetric();for(const r of results){const x=r.metrics[h][s][m];sum.count+=x.count;sum.nonfinite+=x.nonfinite;x.sum.forEach((n,i)=>sum.sum[i]+=n);}out[h][s][m]=finishMetric(sum);
  }}}return out;
}
const SCHEMA={schema:'rek.balance_transfer_validation.v1',model:'All six unchanged historical fixed-lambda ridge models; no fitting, selection or tuning on external data',
  targets:balance.SCHEMA.target_order,features:balance.FEATURE_NAMES,units:balance.SCHEMA.units,
  split:'Entire Windows process-clock-origin group held out from all historical training sessions; adjacent rounds are not independent test sessions',
  strata:'Observed count-mask episode overlap on shared Unity clock; labels only stratify errors, never enter predictor inputs. Zero visual falls/fallen are not labels.',
  open_loop:balance.SCHEMA.open_loop,horizons_seconds:HORIZONS,
  count_authority:'Validated received 33-byte FightState packets, explicit uncensored count-mask episodes and resolution calls; receipt time is not server event time',
  inference_limit:'Rendered-root conditional prediction under recorded controls is not action-conditioned physical causality or a coupled dynamics model',
  recovered_input_limit:'Root tilt, pose and measured receipt deltas do not supply floor contact truth, tracking, complete strike contact streams, physical substep clocks or calibrated standing height',
  no_fitting:true,no_rl_training:true,no_runtime_changes:true};
async function run(manifestFile,output){
  check(!fs.existsSync(output),'output_exists');const manifest=readJson(manifestFile),m=manifest.value,prior=readJson(m.prior_audit);
  check(prior.value.schema==='rek.balance_transition.audit.v1'&&prior.value.session_groups.length===6,'prior_audit_schema');
  check(JSON.stringify(prior.value.supervision.feature_order)===JSON.stringify(balance.FEATURE_NAMES),'feature_order_changed');
  check(m.models.length===6&&new Set(m.models.map(x=>x.held_out_session)).size===6,'frozen_models_incomplete');
  const models=m.models.map(spec=>{const r=readJson(spec.file);validateModel(r.value);const fold=prior.value.folds.find(f=>f.held_out_session===spec.held_out_session);check(fold&&fold.training_rows===r.value.fit_rows,'historical_model_fold_mismatch');return {...spec,...r};});
  const captures=[];for(const spec of m.captures)captures.push(await loadCapture(spec));
  const sessions=validateSessionSeparation(captures,prior.value);for(const c of captures)c.session='external-'+c.session;
  const results=[];for(const model of models){const evaluated=captures.map(c=>({capture:c.id,session:c.session,...evaluate(c,model.value)}));
    results.push({historical_held_out_session:model.held_out_session,training_sessions:prior.value.session_groups.filter(s=>s.id!==model.held_out_session).map(s=>s.id),model_source:model.source,
      captures:evaluated.map(r=>({capture:r.capture,session:r.session,transition_rows:r.transition_rows,metrics:pool([r])})),
      sessions:sessions.map(s=>({session:s.id,metrics:pool(evaluated.filter(r=>r.session===s.id))}))});
  }
  const report={schema:SCHEMA.schema,created_utc:new Date().toISOString(),tool_sha256:hash(fs.readFileSync(__filename)),historical_evaluator_sha256:hash(fs.readFileSync(require.resolve('./balance_transition_data.cjs'))),
    supervision:SCHEMA,manifest_source:manifest.source,prior_audit_source:prior.source,session_groups:sessions,
    captures:captures.map(({rows,origin,...c})=>({...c,observations:rows.length,process_clock_origin_utc:new Date(origin).toISOString()})),models:results,
    integration_status:'Unqualified as physical balance replacement: required contact/dynamics inputs remain unavailable even if some rendered-root metrics improve',
    no_fitting:true,no_runtime_changes:true};
  fs.mkdirSync(output,{recursive:false});fs.writeFileSync(path.join(output,'balance-transfer-validation.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  fs.writeFileSync(path.join(output,'schema.json'),JSON.stringify(SCHEMA,null,2)+'\n',{flag:'wx'});
  return {sessions:sessions.length,captures:captures.length,models:models.length,report:path.resolve(output,'balance-transfer-validation.json')};
}
module.exports={validateModel,validateSessionSeparation,strata,evaluate,pool,SCHEMA,run};
if(require.main===module)Promise.resolve().then(()=>{check(process.argv.length===4,'usage_validate_balance_transfer_MANIFEST_NEW_OUTPUT');return run(...process.argv.slice(2));}).then(r=>console.log(JSON.stringify(r))).catch(e=>{console.error(e.message);process.exitCode=1;});
