#!/usr/bin/env node
'use strict';

// Offline observational sequence audit. This never starts a game or trains an RL policy.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const readline = require('node:readline');
const check = (ok, reason) => {if (!ok) throw Error(reason);};
const finite = Number.isFinite;
const wrap = x => Math.atan2(Math.sin(x), Math.cos(x));
const sha = x => crypto.createHash('sha256').update(x).digest('hex');
const quantile = (values, q) => {const a = values.filter(finite).sort((a,b) => a-b); return a.length ? a[Math.floor((a.length-1)*q)] : null;};
const describe = values => ({count: values.length, median: quantile(values,.5), p95: quantile(values,.95), maximum: values.length ? Math.max(...values) : null});
const BONES = ['left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link'];
const FEATURE_NAMES = ['height','tilt_rad','velocity_forward','velocity_lateral','velocity_vertical','yaw_rate','tilt_rate',
  'opponent_forward','opponent_lateral','opponent_height_difference',
  ...BONES.flatMap(name => ['forward','lateral','vertical'].map(axis => name + '_' + axis)),
  'held_command_0','held_command_1','held_command_2','requested_attack_present','request_age_seconds_capped_3',
  'received_points_change_within_250ms','opponent_received_points_change_within_250ms'];
const TARGET_NAMES = ['velocity_forward','velocity_lateral','velocity_vertical','yaw_rate','tilt_rate'];
const HORIZONS = [.1,.25,.5,1];
const POLICY_RUNS = ['baseline-r1-retry1','shaped-r1-retry2','baseline-r2','shaped-r2','baseline-r3-retry2','shaped-r3','baseline-r4','shaped-r4-retry3','baseline-r5','shaped-r5-retry4'];

function rootState(fighter) {
  const p = fighter.root_position_xyz, q = fighter.root_rotation_xyzw;
  check(Array.isArray(p) && p.length === 3 && p.every(finite) && Array.isArray(q) && q.length === 4 && q.every(finite), 'invalid_root_pose');
  const norm = Math.hypot(...q); check(norm > 0, 'invalid_root_quaternion');
  const [x,y,z,w] = q.map(x => x/norm);
  return [p[0], p[2], p[1], Math.atan2(2*(x*z-w*y), 1-2*(y*y+z*z)), Math.acos(Math.max(-1,Math.min(1,1-2*(x*x+z*z))))];
}
function localPair(x,z,yaw) {const c=Math.cos(yaw),s=Math.sin(yaw); return [c*x+s*z,-s*x+c*z];}
function rate(before, now, dt) {
  const v=localPair((now[0]-before[0])/dt,(now[1]-before[1])/dt,now[3]);
  return [...v,(now[2]-before[2])/dt,wrap(now[3]-before[3])/dt,(now[4]-before[4])/dt];
}
function advance(state, velocity, dt) {
  const c=Math.cos(state[3]),s=Math.sin(state[3]);
  return [state[0]+dt*(c*velocity[0]-s*velocity[1]),state[1]+dt*(s*velocity[0]+c*velocity[1]),state[2]+dt*velocity[2],
    wrap(state[3]+dt*velocity[3]),state[4]+dt*velocity[4]];
}
function features(row, state=row.state, velocity=row.velocity, context=row) {
  const relative=localPair(context.opponent[0]-state[0],context.opponent[1]-state[1],state[3]);
  return [state[2],state[4],...velocity,...relative,context.opponent[2]-state[2],...context.pose,
    ...context.command,context.requested_attack ? 1 : 0,context.request_age,
    context.point_age[0]<=.25 ? 1 : 0,context.point_age[1]<=.25 ? 1 : 0];
}
async function scan(file, visit) {
  const before=fs.statSync(file), hash=crypto.createHash('sha256'), input=fs.createReadStream(file);
  input.on('data', bytes => hash.update(bytes)); let count=0;
  for await (const line of readline.createInterface({input,crlfDelay:Infinity})) {
    count++; if(!line.trim()) continue; let row;
    try {row=JSON.parse(line);} catch {throw Error('invalid_json_line_'+count);}
    visit(row,count);
  }
  const after=fs.statSync(file); check(before.size===after.size && before.mtimeMs===after.mtimeMs,'input_changed');
  return {bytes:after.size,sha256:hash.digest('hex'),lines:count};
}
function groupSessions(captures, toleranceMs=1000) {
  const sessions=[];
  for (const capture of [...captures].sort((a,b)=>a.origin-b.origin)) {
    check(finite(capture.origin),'missing_process_clock_origin');
    let session=sessions.find(s=>Math.abs(s.origin-capture.origin)<=toleranceMs);
    if (!session) {session={id:'session-'+String(sessions.length+1).padStart(2,'0'),origin:capture.origin,captures:[]}; sessions.push(session);}
    session.captures.push(capture.id); capture.session=session.id;
  }
  return sessions;
}
async function loadCapture(base, family, name) {
  const directory=path.join(base,family,name,'trial'), id=family.startsWith('authentic')?'policy/'+name:'passive/'+name;
  const rows=[],origins=[], availability={samples:0,visual_only:0,falling_true:0,fallen_true:0,root_velocity_nonzero:0,floor_contact_nonzero:0,
    last_hit_nonnull:0,runner_move_nonnull:0,runner_name_nonnull:0,joint_positions_nonnull:0};
  const rawTiltError=[], pointTimes=[-Infinity,-Infinity], increments=[]; let priorPoints=null, lastRound=null;
  const provenance=await scan(path.join(directory,'relay.stdout.jsonl'), raw => {
    if(raw.event==='state' && finite(raw.unity_unscaled_time) && finite(Date.parse(raw.observed_utc)))
      origins.push(Date.parse(raw.observed_utc)-1000*raw.unity_unscaled_time);
    if(raw.event!=='g1_policy_state') return;
    check(raw.schema==='rek.g1_policy_source.v1' && [0,1].includes(raw.local_slot),'invalid_source');
    check(raw.fighters.length===2 && raw.clock.qpc_frequency_hz>0,'invalid_fighters_or_clock');
    const t=raw.clock.qpc_ticks/raw.clock.qpc_frequency_hz;
    if(lastRound!==null) check(lastRound===raw.round_identity_sha256,'multiple_rounds_in_capture');
    lastRound=raw.round_identity_sha256;
    if(priorPoints) for(let slot=0;slot<2;slot++) if(raw.round.clean_hits[slot]!==priorPoints[slot]) {
      pointTimes[slot]=t; increments.push({t,slot,delta:raw.round.clean_hits[slot]-priorPoints[slot]});
    }
    priorPoints=raw.round.clean_hits;
    const states=raw.fighters.map(rootState);
    const actors=raw.fighters.map((f,slot)=>{
      availability.samples++;
      availability.visual_only+=f.visual_only===true;
      availability.falling_true+=f.falling===true; availability.fallen_true+=f.fallen===true;
      availability.root_velocity_nonzero+=Array.isArray(f.root_linear_velocity_xyz)&&f.root_linear_velocity_xyz.some(x=>x!==0);
      availability.floor_contact_nonzero+=f.floor_contact_count>0;
      availability.last_hit_nonnull+=f.last_hit!=null;
      availability.runner_move_nonnull+=f.runner?.current_move_index!=null;
      availability.runner_name_nonnull+=f.runner?.current_motion_name!=null;
      availability.joint_positions_nonnull+=f.joint_positions!=null;
      if(finite(f.tilt_angle)) rawTiltError.push(Math.abs(f.tilt_angle-states[slot][4]*180/Math.PI));
      const pose=BONES.flatMap(name=>{
        const index=f.bone_names.indexOf(name); check(index>=0,'required_bone_missing');
        const p=f.bone_world_positions_xyz[index]; check(Array.isArray(p)&&p.every(finite),'invalid_bone_position');
        return [...localPair(p[0]-states[slot][0],p[2]-states[slot][1],states[slot][3]),p[1]-states[slot][2]];
      });
      return {slot,state:states[slot],opponent:states[slot^1],pose};
    });
    const command=raw.input.velocity_command_xyz;
    check(Array.isArray(command)&&command.length===3&&command.every(finite),'invalid_command');
    const requestTime=raw.input.requested_move_qpc_ticks;
    rows.push({t,unity_time:raw.clock.unity_time,sequence:raw.observation_sequence,active:raw.round.active===true,
      actors,local_slot:raw.local_slot,command,desired_action:raw.input.desired_action,
      requested_attack:raw.input.requested_move_index!=null,
      requested_move_index:raw.input.requested_move_index??null,
      requested_move_qpc_ticks:requestTime??null,
      request_age:requestTime==null?3:Math.max(0,Math.min(3,t-requestTime/raw.clock.qpc_frequency_hz)),
      point_age:pointTimes.map(time=>t-time),points:raw.round.clean_hits,falls:raw.round.falls});
  });
  check(rows.length>2 && origins.length>0,'insufficient_capture');
  const summaryPath=path.join(directory,'summary.json'); const summary=fs.existsSync(summaryPath)?JSON.parse(fs.readFileSync(summaryPath)):null;
  const origin=quantile(origins,.5); check(Math.max(...origins)-Math.min(...origins)<1000,'process_origin_drift_or_relaunch');
  return {id,name,family,rows,origin,provenance,availability,increments,summary,
    origin_span_ms:Math.max(...origins)-Math.min(...origins),tilt_derived_error:describe(rawTiltError)};
}
function transitions(capture) {
  const result=[];
  for(let i=1;i<capture.rows.length-1;i++) {
    const prev=capture.rows[i-1],now=capture.rows[i],next=capture.rows[i+1];
    const dt=next.t-now.t,previousDt=now.t-prev.t;
    // Positive, bounded receipt deltas only. Do not interpolate or bridge lost samples.
    if(!prev.active||!now.active||!next.active||dt<.005||dt>.06||previousDt<.005||previousDt>.06) continue;
    const actor=now.actors[now.local_slot],previous=prev.actors[prev.local_slot],following=next.actors[next.local_slot];
    const velocity=rate(previous.state,actor.state,previousDt);
    const target=rate(actor.state,following.state,dt);
    // Express outgoing velocity in the current heading, not next heading.
    const local=localPair((following.state[0]-actor.state[0])/dt,(following.state[1]-actor.state[1])/dt,actor.state[3]);
    target[0]=local[0];target[1]=local[1];
    const row={...actor,capture:capture.id,session:capture.session,index:i,t:now.t,dt,previous_dt:previousDt,
      velocity,target,next_state:following.state,command:now.command,requested_attack:now.requested_attack,
      requested_move_index:now.requested_move_index,desired_action:now.desired_action,request_age:now.request_age,
      point_age:[now.point_age[now.local_slot],now.point_age[now.local_slot^1]].map(age=>Math.min(3,age)),executed_move_index:null};
    row.x=features(row); check(row.x.length===FEATURE_NAMES.length&&row.x.every(finite)&&row.target.every(finite),'nonfinite_transition');
    result.push(row);
  }
  return result;
}
function fitRidge(rows, lambda=.01) {
  check(rows.length>FEATURE_NAMES.length,'insufficient_training_rows');
  const n=rows.length,p=FEATURE_NAMES.length,mean=Array(p).fill(0),scale=Array(p).fill(0);
  for(const row of rows) for(let j=0;j<p;j++) mean[j]+=row.x[j]/n;
  for(const row of rows) for(let j=0;j<p;j++) scale[j]+=(row.x[j]-mean[j])**2/n;
  for(let j=0;j<p;j++) scale[j]=Math.max(Math.sqrt(scale[j]),1e-6);
  const d=p+1,a=Array.from({length:d},()=>Array(d+5).fill(0));
  for(const row of rows) {
    const x=[1,...row.x.map((v,j)=>(v-mean[j])/scale[j])];
    for(let j=0;j<d;j++) {for(let k=0;k<=j;k++) a[j][k]+=x[j]*x[k]/n;
      for(let k=0;k<5;k++) a[j][d+k]+=x[j]*row.target[k]/n;}
  }
  for(let j=0;j<d;j++) {for(let k=j+1;k<d;k++) a[j][k]=a[k][j];if(j>0)a[j][j]+=lambda;}
  for(let j=0;j<d;j++) {
    let best=j;for(let k=j+1;k<d;k++) if(Math.abs(a[k][j])>Math.abs(a[best][j]))best=k;
    [a[j],a[best]]=[a[best],a[j]];check(Math.abs(a[j][j])>1e-12,'singular_training_matrix');
    const divisor=a[j][j];for(let k=j;k<d+5;k++)a[j][k]/=divisor;
    for(let k=0;k<d;k++)if(k!==j){const coefficient=a[k][j];for(let l=j;l<d+5;l++)a[k][l]-=coefficient*a[j][l];}
  }
  return {mean,scale,weights:a.map(row=>row.slice(d)),lambda,fit_rows:n};
}
function predict(model,x) {
  const z=[1,...x.map((v,j)=>(v-model.mean[j])/model.scale[j])];
  return [0,1,2,3,4].map(k=>z.reduce((sum,v,j)=>sum+v*model.weights[j][k],0));
}
function metric() {return {count:0,position_sum:0,height_sum:0,yaw_sum:0,tilt_sum:0,nonfinite:0};}
function addMetric(m,predicted,actual) {
  m.count++;if(!predicted.every(finite)){m.nonfinite++;return;}
  m.position_sum+=(predicted[0]-actual[0])**2+(predicted[1]-actual[1])**2;
  m.height_sum+=(predicted[2]-actual[2])**2;m.yaw_sum+=wrap(predicted[3]-actual[3])**2;
  m.tilt_sum+=(predicted[4]-actual[4])**2;
}
function finishMetric(m) {const n=m.count-m.nonfinite;return {count:m.count,nonfinite:m.nonfinite,
  planar_rmse_unity_units:n?Math.sqrt(m.position_sum/n):null,height_rmse_unity_units:n?Math.sqrt(m.height_sum/n):null,
  yaw_rmse_degrees:n?Math.sqrt(m.yaw_sum/n)*180/Math.PI:null,tilt_rmse_degrees:n?Math.sqrt(m.tilt_sum/n)*180/Math.PI:null};}
function evaluateOneStep(rows,model) {
  const metrics={frozen_root:metric(),constant_velocity:metric(),ridge:metric()};
  for(const row of rows) {
    addMetric(metrics.frozen_root,row.state,row.next_state);
    addMetric(metrics.constant_velocity,advance(row.state,row.velocity,row.dt),row.next_state);
    addMetric(metrics.ridge,advance(row.state,predict(model,row.x),row.dt),row.next_state);
  }
  return Object.fromEntries(Object.entries(metrics).map(([k,m])=>[k,finishMetric(m)]));
}
function evaluateOpenLoop(captures,rows,model) {
  const rowMap=new Map(rows.map(r=>[r.capture+':'+r.index,r])), result={};
  for(const horizon of HORIZONS) {
    const metrics={frozen_root:metric(),constant_velocity:metric(),ridge:metric()};
    for(const capture of captures) {
      let nextStart=-Infinity;
      for(let i=1;i<capture.rows.length-1;i++) {
        const start=rowMap.get(capture.id+':'+i);if(!start||start.t<nextStart)continue;
        let j=i,valid=true;while(j<capture.rows.length-1&&capture.rows[j].t-start.t<horizon) {
          if(!rowMap.has(capture.id+':'+j)){valid=false;break;}j++;
        }
        if(!valid||j>=capture.rows.length-1)continue;
        const actual=capture.rows[j].actors[capture.rows[j].local_slot].state,dt=capture.rows[j].t-start.t;
        let state=[...start.state],velocity=[...start.velocity];
        // Only the recorded command/request schedule is replayed. Pose and opponent
        // context stay at the window start; no future observed root/pose is injected.
        for(let k=i;k<j;k++) {
          const recorded=rowMap.get(capture.id+':'+k);
          const elapsed=recorded.t-start.t;
          const context={...start,command:recorded.command,requested_attack:recorded.requested_attack,request_age:recorded.request_age,
            point_age:start.point_age.map(age=>age+elapsed)};
          velocity=predict(model,features(start,state,velocity,context));state=advance(state,velocity,recorded.dt);
        }
        addMetric(metrics.frozen_root,start.state,actual);
        // Constant world velocity is the fair inertial baseline for a full window.
        const inertial=[start.state[0]+(start.state[0]-capture.rows[i-1].actors[start.slot].state[0])*dt/start.previous_dt,
          start.state[1]+(start.state[1]-capture.rows[i-1].actors[start.slot].state[1])*dt/start.previous_dt,
          start.state[2]+start.velocity[2]*dt,wrap(start.state[3]+start.velocity[3]*dt),start.state[4]+start.velocity[4]*dt];
        addMetric(metrics.constant_velocity,inertial,actual);addMetric(metrics.ridge,state,actual);nextStart=capture.rows[j].t;
      }
    }
    result[horizon]=Object.fromEntries(Object.entries(metrics).map(([k,m])=>[k,finishMetric(m)]));
  }
  return result;
}
function highTiltWindows(capture) {
  const events=[], threshold=Math.PI/3,minimumDuration=.2;
  for(let slot=0;slot<2;slot++) {
    let start=null;
    for(let i=0;i<=capture.rows.length;i++) {
      const row=capture.rows[i],high=row&&row.active&&row.actors[slot].state[4]>=threshold;
      if(high&&start===null)start=i;
      if(!high&&start!==null) {
        const end=i-1,duration=capture.rows[end].t-capture.rows[start].t;
        if(duration>=minimumDuration)events.push({slot,start,end,duration});start=null;
      }
    }
  }
  const comparisons=[];
  for(const event of events) {
    const onset=capture.rows[event.start];let pre=event.start;
    while(pre>0&&onset.t-capture.rows[pre].t<.5)pre--;
    if(onset.t-capture.rows[pre].t>.56||onset.t-capture.rows[pre].t<.45)continue;
    const base=capture.rows[pre],actor=base.actors[event.slot],gap=Math.hypot(actor.opponent[0]-actor.state[0],actor.opponent[1]-actor.state[1]);
    let nearest=null,best=Infinity;
    // Match within capture and actor, separated from every high-tilt episode.
    for(let i=1;i<capture.rows.length-1;i+=5) {
      const r=capture.rows[i],a=r.actors[event.slot];
      if(!r.active||Math.abs(r.t-base.t)<2||events.some(e=>e.slot===event.slot&&r.t>=capture.rows[e.start].t-1&&r.t<=capture.rows[e.end].t+1))continue;
      if(a.state[4]>=threshold)continue;
      const dg=Math.hypot(a.opponent[0]-a.state[0],a.opponent[1]-a.state[1]);
      const d=((dg-gap)/.5)**2+((a.state[2]-actor.state[2])/.2)**2+((a.state[4]-actor.state[4])/.3)**2+
        (event.slot===base.local_slot&&r.desired_action!==base.desired_action?1:0);
      if(d<best){best=d;nearest=i;}
    }
    if(nearest===null)continue;
    const control=capture.rows[nearest],ca=control.actors[event.slot];let end=nearest;
    while(end<capture.rows.length-1&&capture.rows[end].t-control.t<.5)end++;
    if(!capture.rows[end].active||capture.rows[end].t-control.t>.56)continue;
    comparisons.push({slot:event.slot,role:event.slot===base.local_slot?'local':'opponent',matching_distance:Math.sqrt(best),
      precursor_tilt_change_degrees:(onset.actors[event.slot].state[4]-actor.state[4])*180/Math.PI,
      control_tilt_change_degrees:(capture.rows[end].actors[event.slot].state[4]-ca.state[4])*180/Math.PI,
      precursor_height_change_unity_units:onset.actors[event.slot].state[2]-actor.state[2],
      control_height_change_unity_units:capture.rows[end].actors[event.slot].state[2]-ca.state[2],
      precursor_points_received_last_250ms:base.point_age[event.slot]<=.25,
      control_points_received_last_250ms:control.point_age[event.slot]<=.25});
  }
  return {episodes:events.length,local_episodes:events.filter(e=>e.slot===capture.rows[0].local_slot).length,
    durations_seconds:describe(events.map(e=>e.duration)),matched_windows:comparisons,
    definition:'pelvis-derived tilt >= 60 degrees for >= 0.2 seconds in active round; geometric proxy, not referee fall labels'};
}
function requestMotion(capture,horizon=1) {
  const samples=[],seen=new Set();
  for(let i=0;i<capture.rows.length;i++) {
    const start=capture.rows[i],key=start.requested_move_qpc_ticks;
    if(key===null||key===undefined||seen.has(key)||!start.requested_attack)continue;
    seen.add(key);if(!start.active||start.request_age>.1)continue;
    let j=i;while(j<capture.rows.length-1&&capture.rows[j].t-start.t<horizon)j++;
    const end=capture.rows[j];if(!end.active||end.t-start.t<horizon||end.t-start.t>horizon+.06)continue;
    const window=capture.rows.slice(i,j+1);if(window.some((r,k)=>k>0&&r.t-window[k-1].t>.06))continue;
    const a=start.actors[start.local_slot].state,b=end.actors[start.local_slot].state;
    const [forward,lateral]=localPair(b[0]-a[0],b[1]-a[1],a[3]);
    samples.push({move:start.requested_move_index,forward,lateral,planar:Math.hypot(forward,lateral),
      height:b[2]-a[2],tilt_change:(b[4]-a[4])*180/Math.PI,
      maximum_tilt:Math.max(...window.map(r=>r.actors[start.local_slot].state[4]*180/Math.PI)),
      newer_request_within_window:window.some(r=>r.requested_move_qpc_ticks!==key)});
  }
  const summarize=values=>({windows:values.length,forward_displacement_unity_units:describe(values.map(s=>s.forward)),
    planar_displacement_unity_units:describe(values.map(s=>s.planar)),height_change_unity_units:describe(values.map(s=>s.height)),
    tilt_change_degrees:describe(values.map(s=>s.tilt_change)),maximum_tilt_degrees:describe(values.map(s=>s.maximum_tilt)),
    root_moved_over_001_unity_units:values.filter(s=>s.planar>.01).length,
    newer_request_within_window:values.filter(s=>s.newer_request_within_window).length});
  return {horizon_seconds:horizon,all:summarize(samples),by_requested_move:[...new Set(samples.map(s=>s.move))].sort((a,b)=>a-b)
    .map(move=>({requested_move_index:move,...summarize(samples.filter(s=>s.move===move))})),
    interpretation:'root response following observed client request; not proof of playback or causal attack displacement; later commands and reset/interpolation effects can contribute'};
}
function aggregateFolds(folds) {
  const combine=metrics=>{
    const result={};
    for(const model of ['frozen_root','constant_velocity','ridge']) {
      const m=metrics.map(row=>row[model]),count=m.reduce((s,v)=>s+v.count,0);
      result[model]={count,nonfinite:m.reduce((s,v)=>s+v.nonfinite,0)};
      for(const field of ['planar_rmse_unity_units','height_rmse_unity_units','yaw_rmse_degrees','tilt_rmse_degrees'])
        result[model][field]=Math.sqrt(m.reduce((s,v)=>s+v.count*v[field]**2,0)/count);
    }
    result.ridge_planar_better_than_frozen_sessions=metrics.filter(m=>m.ridge.planar_rmse_unity_units<m.frozen_root.planar_rmse_unity_units).length;
    return result;
  };
  return {weighting:'pooled squared error across held-out rows/windows; each session appears once in test; not independent-frame confidence intervals',
    one_step:combine(folds.map(f=>f.one_step)),open_loop:Object.fromEntries(HORIZONS.map(h=>[h,combine(folds.map(f=>f.open_loop[h]))]))};
}
const SCHEMA={schema:'rek.balance_transition.supervision.v1',feature_order:FEATURE_NAMES,target_order:TARGET_NAMES,
  target_semantics:'next received client-rendered root finite difference; local +X projected into Unity XZ',
  actor:'local controlled actor only for regression; both actors for geometry audit',
  time:'QPC seconds; bounded positive receipt deltas, no interpolation',units:'Unity numeric distance units; metre calibration unverified; radians and seconds',
  command_semantics:'client held-command/request context as observed at start; server acceptance and active move unknown',
  executed_move_index:null,contact_label:null,fall_label:null,
  contact_context:'recent received point-counter change is receipt context, never physical-contact truth',
  point_age:'seconds since observed counter change, capped at 3; 3 also represents no change observed since capture start; pre-capture receipt history unknown',
  pose_context:'root-relative ankle and wrist positions; support geometry does not prove floor contact',
  split:'leave one process-clock-origin session group out; captures within 1000 ms origin grouped together',
  output_privacy:'raw transitions and fitted diagnostic weights remain only in private output directory; Git report contains aggregates',
  model:'fixed-lambda 0.01 standardized linear ridge predicts five root rates; no hyperparameter search',
  open_loop:'replay recorded local command/request schedule; initialize measured root and finite-difference velocity once; hold start relative pose/opponent fixed; advance point receipt age without future receipts',
  limits:['No server execution, contact normals, forces, motor state, support-contact truth, or active clip identity.',
    'Root persistence is an explicit diagnostic baseline, not a replay of the full native5 simulator.',
    'Process groups use stable UTC minus Unity unscaled time; direct game PID is absent from the policy stream.',
    'High-tilt proxy windows are observational matches and do not identify causal attack/fall outcomes.',
    'Future command replay is conditional prediction, not a new-policy counterfactual rollout.',
    'Open-loop compact pose and opponent context are frozen; this is not a complete coupled-body simulator.']};

async function exportAudit(base,output) {
  check(!fs.existsSync(output),'output_exists');const captures=[];
  for(const name of POLICY_RUNS) captures.push(await loadCapture(base,'authentic-policy-ab-20260919-r1',name));
  for(const name of ['round-r4','round-r5','round-r11','round-r12'])captures.push(await loadCapture(base,'passive-defender-20260917-r1',name));
  const sessions=groupSessions(captures);check(sessions.length>=2,'no_independent_session_split');
  const all=captures.flatMap(transitions),folds=[];
  fs.mkdirSync(output,{recursive:true,mode:0o700});
  const dataset=fs.createWriteStream(path.join(output,'private-transitions.jsonl'),{flags:'wx',mode:0o600});
  for(const row of all) if(!dataset.write(JSON.stringify(row)+'\n'))await new Promise(resolve=>dataset.once('drain',resolve));
  await new Promise((resolve,reject)=>{dataset.once('error',reject);dataset.end(resolve);});
  for(const session of sessions) {
    const train=all.filter(r=>r.session!==session.id),test=all.filter(r=>r.session===session.id),model=fitRidge(train);
    check(new Set(train.map(r=>r.session)).has(session.id)===false,'session_leak');
    const held=captures.filter(c=>c.session===session.id);
    folds.push({held_out_session:session.id,training_sessions:sessions.length-1,training_rows:train.length,test_rows:test.length,
      test_captures:session.captures,one_step:evaluateOneStep(test,model),open_loop:evaluateOpenLoop(held,test,model)});
    fs.writeFileSync(path.join(output,'private-'+session.id+'-ridge.json'),JSON.stringify(model)+'\n',{flag:'wx',mode:0o600});
  }
  const windows=captures.map(c=>({capture:c.id,session:c.session,...highTiltWindows(c)}));
  const report={schema:'rek.balance_transition.audit.v1',created_utc:new Date().toISOString(),exporter_sha256:sha(fs.readFileSync(__filename)),supervision:SCHEMA,
    captures:captures.map(c=>({capture:c.id,session:c.session,source:c.provenance,process_clock_origin_utc:new Date(c.origin).toISOString(),
      origin_estimate_span_ms:c.origin_span_ms,observations:c.rows.length,active_seconds:c.rows.at(-1).t-c.rows[0].t,
      receipt_interval_seconds:describe(c.rows.slice(1).map((r,i)=>r.t-c.rows[i].t)),
      initial_points:c.rows[0].points,final_points:c.rows.at(-1).points,initial_falls:c.rows[0].falls,final_falls:c.rows.at(-1).falls,
      terminal_observed:c.rows.at(-1).active===false,availability:c.availability,tilt_quaternion_difference_degrees:c.tilt_derived_error,
      root_height_unity_units:describe(c.rows.flatMap(r=>r.actors.map(a=>a.state[2]))),
      tilt_degrees:describe(c.rows.flatMap(r=>r.actors.map(a=>a.state[4]*180/Math.PI))),
      attack_request_root_response:requestMotion(c),
      received_point_counter_increments:c.increments.length,five_point_counter_increments:c.increments.filter(x=>x.delta===5).length})),
    session_groups:sessions.map(s=>({id:s.id,captures:s.captures,origin_utc:new Date(s.origin).toISOString()})),
    transition_rows:all.length,aggregate_held_out:aggregateFolds(folds),folds,high_tilt_windows:windows,
    no_rl_training:true,no_simulator_mutation:true,no_causal_action_outcome_labels:true};
  fs.writeFileSync(path.join(output,'balance-transition-audit.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});
  fs.writeFileSync(path.join(output,'supervision-schema.json'),JSON.stringify(SCHEMA,null,2)+'\n',{flag:'wx',mode:0o600});
  return {captures:captures.length,sessions:sessions.length,transitions:all.length,high_tilt_episodes:windows.reduce((n,w)=>n+w.episodes,0),
    matched_windows:windows.reduce((n,w)=>n+w.matched_windows.length,0),report_sha256:sha(fs.readFileSync(path.join(output,'balance-transition-audit.json')))};
}
module.exports={rootState,rate,advance,features,groupSessions,transitions,fitRidge,predict,evaluateOneStep,evaluateOpenLoop,highTiltWindows,requestMotion,aggregateFolds,FEATURE_NAMES,SCHEMA,exportAudit};
if(require.main===module)Promise.resolve().then(()=>{check(process.argv.length===4,'usage_balance_transition_data_BASE_NEW_PRIVATE_OUTPUT');return exportAudit(...process.argv.slice(2));})
  .then(result=>console.log(JSON.stringify(result))).catch(error=>{console.error(/^[a-zA-Z0-9_]+$/.test(error.message)?error.message:error.code||'audit_failed');process.exitCode=1;});
