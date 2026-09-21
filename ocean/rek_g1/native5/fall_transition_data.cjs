'use strict';

// Offline received-count-onset forecast. This never connects to or controls REK.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const readline = require('node:readline');
const {rootState, rate} = require('./balance_transition_data.cjs');
const {decode} = require('./validate_live_referee.cjs');
const check = (ok, why) => { if (!ok) throw Error(why); };
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const finite = Number.isFinite;
const BONES = ['left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link'];
const STATE_FEATURES = ['local_height', 'local_tilt', 'local_forward_rate', 'local_lateral_rate', 'local_vertical_rate', 'local_yaw_rate', 'local_tilt_rate',
  'opponent_height', 'opponent_tilt', 'opponent_forward_rate', 'opponent_lateral_rate', 'opponent_vertical_rate', 'opponent_yaw_rate', 'opponent_tilt_rate',
  'opponent_relative_forward', 'opponent_relative_lateral', 'relative_heading_cos', 'relative_heading_sin',
  ...['local', 'opponent'].flatMap(s => BONES.flatMap(b => ['forward', 'lateral', 'vertical'].map(a => s + '_' + b + '_' + a)))];
const ACTION_FEATURES = ['command_forward', 'command_lateral', 'command_yaw', 'prior_command_forward', 'prior_command_lateral', 'prior_command_yaw',
  'request_age_capped3', 'requests_in_preceding_1s', ...Array.from({length:17}, (_, i) => 'requested_move_' + i)];
const FEATURE_ORDER = [...STATE_FEATURES, ...ACTION_FEATURES];
const HORIZON = .5, HISTORY = .2, STRIDE = .5;
const localPair = (x, z, yaw) => [Math.cos(yaw)*x + Math.sin(yaw)*z, -Math.sin(yaw)*x + Math.cos(yaw)*z];
const inc = (o, k) => { o[k] = (o[k] || 0) + 1; };
const describe = values => {
  const a=values.filter(finite).sort((a,b)=>a-b);
  return {count:a.length,minimum:a[0]??null,median:a.length?a[Math.floor((a.length-1)/2)]:null,
    p95:a.length?a[Math.floor((a.length-1)*.95)]:null,maximum:a.at(-1)??null};
};

async function scan(file, visit) {
  check(!/onedrive/i.test(file), 'onedrive_path_forbidden');
  const before = fs.statSync(file), digest = crypto.createHash('sha256'), input = fs.createReadStream(file);
  input.on('data', b => digest.update(b)); let lines = 0;
  for await (const raw of readline.createInterface({input, crlfDelay:Infinity})) {
    ++lines; if (raw.trim()) visit(JSON.parse(raw), lines);
  }
  const after = fs.statSync(file);
  check(before.size === after.size && before.mtimeMs === after.mtimeMs, 'source_changed');
  return {bytes:after.size, lines, sha256:digest.digest('hex')};
}
async function fileHash(file) {
  const digest = crypto.createHash('sha256'), before = fs.statSync(file);
  for await (const b of fs.createReadStream(file)) digest.update(b);
  const after = fs.statSync(file);
  check(before.size === after.size && before.mtimeMs === after.mtimeMs, 'source_changed');
  return digest.digest('hex');
}
function compact(source) {
  check(source.schema === 'rek.g1_policy_source.v1' && [0,1].includes(source.local_slot), 'invalid_source');
  const t = source.clock.qpc_ticks / source.clock.qpc_frequency_hz, referee = source.referee;
  check(finite(t), 'invalid_source_time');
  const out = {t, utc:source.clock.utc, round:source.round_identity_sha256, active:source.round.active === true,
    slot:source.local_slot, valid:false};
  if (!referee?.available) return out;
  const b = decode(referee.wire_body_base64, referee.wire_body_sha256);
  check(referee.count_mask === b[25] && referee.call_sequence === b[27] &&
    referee.receipt_qpc_frequency_hz === source.clock.qpc_frequency_hz, 'invalid_referee_fields');
  const receiptTime = referee.receipt_qpc_ticks / referee.receipt_qpc_frequency_hz;
  check(receiptTime <= t && t - receiptTime <= .5 + 1e-9, 'invalid_receipt_age');
  const states = source.fighters.map(rootState), pose = source.fighters.map((f,s) => BONES.flatMap(name => {
    const p = f.bone_world_positions_xyz[f.bone_names.indexOf(name)];
    check(Array.isArray(p) && p.length === 3 && p.every(finite), 'missing_bone');
    return [...localPair(p[0] - states[s][0], p[2] - states[s][1], states[s][3]), p[1] - states[s][2]];
  }));
  const command = source.input.velocity_command_xyz, requestedTime = source.input.requested_move_qpc_ticks;
  check(Array.isArray(command) && command.length === 3 && command.every(finite), 'invalid_command');
  check(source.input.requested_move_index==null || Number.isInteger(source.input.requested_move_index) &&
    source.input.requested_move_index>=0 && source.input.requested_move_index<17, 'invalid_requested_move');
  return {...out, valid:true, states, pose, command, requested_move:source.input.requested_move_index,
    requested_time:requestedTime == null ? null : requestedTime / source.clock.qpc_frequency_hz,
    receipt_time:receiptTime, receipt_sequence:referee.receipt_sequence, lifecycle:referee.lifecycle,
    count_mask:b[25], call_type:b[28], call_faller:b.readInt8(29), call_sequence:b[27],
    call_censored:referee.call_history_censored === true, call_name:referee.call_name};
}
function featureRow(now, before, requestTimes) {
  const order = [now.slot, now.slot ^ 1], x = [];
  for (const s of order) x.push(now.states[s][2], now.states[s][4], ...rate(before.states[s], now.states[s], now.t-before.t));
  const a = now.states[order[0]], b = now.states[order[1]];
  x.push(...localPair(b[0]-a[0], b[1]-a[1], a[3]), Math.cos(b[3]-a[3]), Math.sin(b[3]-a[3]));
  for (const s of order) x.push(...now.pose[s]);
  x.push(...now.command, ...before.command, now.requested_time == null ? 3 : Math.min(3, Math.max(0,now.t-now.requested_time)),
    requestTimes.filter(t => t > now.t-1 && t <= now.t).length,
    ...Array.from({length:17}, (_, i) => +(i === now.requested_move)));
  check(x.length === FEATURE_ORDER.length && x.every(finite), 'nonfinite_features');
  return x;
}
function buildRows(observations, capture) {
  const onsets = [], episodes = [], activeCounts = new Map(), censored = [], requests = [], issues = {}, rows = [];
  let previousReceipt = null, lastRequest = null;
  for (let i=0; i<observations.length; i++) {
    const now = observations[i];
    if (i) check(now.t >= observations[i-1].t, 'policy_clock_regressed');
    if (!now.valid) { censored.push(now.t); continue; }
    if (now.requested_time != null && now.requested_time !== lastRequest) {
      if (now.requested_time <= now.t) requests.push(now.requested_time);
      lastRequest = now.requested_time;
    }
    if (now.receipt_sequence === previousReceipt?.receipt_sequence) continue;
    const prior = previousReceipt;
    if (prior && now.round === prior.round && now.lifecycle === prior.lifecycle) {
      if (now.receipt_time - prior.receipt_time > .3 || now.call_censored) censored.push(now.receipt_time);
      const rise = now.count_mask & ~prior.count_mask;
      for (const slot of [0,1]) if (rise & (1 << slot)) {
        const explicit = !now.call_censored && ['Slip','SlipEStop','Knockdown','DoubleKnockdown'].includes(now.call_name) &&
          (now.call_name === 'DoubleKnockdown' || now.call_faller === slot);
        if (explicit) {
          const event={id:capture+':'+now.lifecycle+':'+now.call_sequence+':'+slot,
            t:now.receipt_time, slot, name:now.call_name};
          onsets.push(event);activeCounts.set(slot,event);
        }
        else { censored.push(now.receipt_time); inc(issues,'unconfirmed_count_onset'); }
      }
      for(const [slot,event] of activeCounts) if(!(now.count_mask&(1<<slot))) {
        const countout=!now.call_censored && (now.call_name==='DoubleKnockout' ||
          now.call_name==='Knockout' && now.call_faller===slot);
        episodes.push({...event,end:now.receipt_time,duration:now.receipt_time-event.t,
          resolution:now.call_name,explicit_countout:countout});activeCounts.delete(slot);
      }
    } else { censored.push(now.receipt_time);activeCounts.clear(); }
    previousReceipt = now;
  }
  let lastSelected = -Infinity, historyIndex = 0, futureIndex = 0;
  for (let i=1;i<observations.length;i++) {
    const now = observations[i];
    if (now.t-lastSelected < STRIDE) continue;
    lastSelected = now.t;
    while (historyIndex+1 < i && observations[historyIndex+1].t <= now.t-HISTORY) historyIndex++;
    while (futureIndex < observations.length && observations[futureIndex].t < now.t+HORIZON) futureIndex++;
    const before=observations[historyIndex], future=observations[futureIndex];
    if (!future || now.t-before.t < HISTORY-1e-9 || now.t-before.t > HISTORY+.06) { inc(issues,'incomplete_window'); continue; }
    const window=observations.slice(historyIndex,futureIndex+1);
    if (window.some((r,j) => !r.valid || !r.active || r.round!==now.round || r.lifecycle!==now.lifecycle ||
      (j>0 && r.t-window[j-1].t>.1))) { inc(issues,'censored_observation_window'); continue; }
    if (now.count_mask!==0 || before.count_mask!==0) { inc(issues,'count_already_active'); continue; }
    if (censored.some(t=>t>=before.t && t<=future.t)) { inc(issues,'censored_referee_window'); continue; }
    const events=onsets.filter(e=>e.t>now.t && e.t<=now.t+HORIZON);
    rows.push({capture,round:now.round,t:now.t,x:featureRow(now,before,requests),
      y:[now.slot,now.slot^1].map(s=>+events.some(e=>e.slot===s)),events:events.map(e=>e.id)});
  }
  return {rows,onsets,episodes,issues};
}

async function exportFiles(base, output) {
  check(!fs.existsSync(output), 'output_exists');
  const names = fs.readdirSync(base).filter(n=>/^live-round_outcome_v1-r[0-9]+$/.test(n)).sort((a,b)=>+a.split('-r').at(-1)-+b.split('-r').at(-1));
  const rows=[], captures=[], excluded=[];
  for (const name of names) {
    const reportFile=path.join(base,name,'referee-validation/live-referee-validation.json');
    if (!fs.existsSync(reportFile)) { excluded.push({capture:name,reason:'no_completed_referee_validation'}); continue; }
    const reportBytes=fs.readFileSync(reportFile), report=JSON.parse(reportBytes);
    check(report.schema==='rek.live_referee_validation.v1' && report.verification_passed && report.inputs.length===2, 'invalid_prior_validation');
    const relay=path.join(base,name,'trial/relay.stdout.jsonl'), native=report.inputs[1].file;
    const observations=[];
    const [provenance,nativeHash]=await Promise.all([
      scan(relay,r=>{if(r.event==='g1_policy_state') observations.push(compact(r));}),fileHash(native)]);
    check(provenance.sha256===report.inputs[0].sha256 && nativeHash===report.inputs[1].sha256, 'validated_source_hash_mismatch');
    check(observations.length===report.source_count, 'source_count_mismatch');
    const result=buildRows(observations,name), nativeBirth=path.basename(native).match(/(\d{8}T\d{6})/)[1];
    const group='process-'+report.native_process_id+'-'+nativeBirth.substring(0,8);
    for(const row of result.rows){row.group=group;rows.push(row);}
    captures.push({capture:name,group,utc:observations[0]?.utc,relay_sha256:provenance.sha256,native_sha256:nativeHash,
      prior_validation_sha256:sha(reportBytes),source_observations:observations.length,rows:result.rows.length,
      onsets:result.onsets.length,observed_onset_types:result.onsets.reduce((m,e)=>(inc(m,e.name),m),{}),
      onsets_by_slot:[0,1].map(s=>result.onsets.filter(e=>e.slot===s).length),excluded_windows:result.issues});
    captures.at(-1).received_count_resolution={completed_episodes:result.episodes.length,
      explicit_countouts:result.episodes.filter(e=>e.explicit_countout).length,
      duration_seconds:describe(result.episodes.map(e=>e.duration)),
      by_onset_type:Object.fromEntries([...new Set(result.episodes.map(e=>e.name))].map(name=>[name,{
        duration_seconds:describe(result.episodes.filter(e=>e.name===name).map(e=>e.duration)),
        resolution_types:result.episodes.filter(e=>e.name===name).reduce((m,e)=>(inc(m,e.resolution),m),{})}]))};
    console.error(JSON.stringify({capture:name,rows:result.rows.length,onsets:result.onsets.length}));
  }
  const groups=[...new Set(captures.sort((a,b)=>a.utc.localeCompare(b.utc)).map(c=>c.group))];
  check(groups.length>=10,'insufficient_process_groups');
  const trainEnd=Math.floor(groups.length*.6),calEnd=Math.floor(groups.length*.8);
  const splitOf=g=>groups.indexOf(g)<trainEnd?0:groups.indexOf(g)<calEnd?1:2;
  rows.forEach(r=>r.split=splitOf(r.group));
  const out=Buffer.alloc(24+rows.length*(4+FEATURE_ORDER.length+2)*4);out.write('REKFAL1\0');
  [1,FEATURE_ORDER.length,STATE_FEATURES.length,rows.length].forEach((n,i)=>out.writeUInt32LE(n,8+i*4));
  let offset=24;
  for(const row of rows) for(const n of [row.split,groups.indexOf(row.group),+row.capture.split('-r').at(-1),row.t,
    ...row.x,...row.y]) {check(finite(Math.fround(n)),'float32_overflow');out.writeFloatLE(n,offset);offset+=4;}
  const splits=['train','calibration','test'].map((name,s)=>{
    const rr=rows.filter(r=>r.split===s),positive=[0,1].map(j=>rr.reduce((n,r)=>n+r.y[j],0));
    return {name,rows:rr.length,groups:groups.filter(g=>splitOf(g)===s),positive_rows:positive,
      positive_fraction:positive.map(n=>n/rr.length),unique_onsets_represented:new Set(rr.flatMap(r=>r.events)).size,
      requested_move_support:Array.from({length:17},(_,m)=>rr.filter(r=>r.x[STATE_FEATURES.length+8+m]===1).length)};
  });
  const manifest={schema:'rek.received_fall_onset_dataset.v1',created_utc:new Date().toISOString(),
    exporter_sha256:sha(fs.readFileSync(__filename)),feature_order:FEATURE_ORDER,state_feature_count:STATE_FEATURES.length,
    target_order:['future_local_received_count_onset','future_opponent_received_count_onset'],
    rows:rows.length,captures,excluded_captures:excluded,splits,binary_sha256:sha(out),
    horizon_seconds:HORIZON,history_seconds:HISTORY,minimum_prediction_spacing_seconds:STRIDE,
    target_authority:'received 33-byte REK_FightState count-mask rise corroborated by uncensored explicit Slip/SlipEStop/Knockdown/DoubleKnockdown call',
    target_time:'client packet receipt; physical server fall time is unavailable',
    action_semantics:'prior/current locally requested action and held commands; execution acceptance and opponent actions unavailable',
    feature_semantics:'all features observed at or before forecast time; current and preceding 0.2-second rendered geometry; no future action replay',
    negative_semantics:'no newly observed count onset within complete 0.5-second received observation window; not proof of absence of a physical fall',
    split_semantics:'chronological 60/20/20 complete process groups; no row, round or process appears across splits; calibration separate from final test',
    person_split_available:false,person_split_reason:'these sources are agent versus AI rounds; no identified multi-person human cohort',
    metric_units:'seconds and radians; geometry remains Unity numeric units with metric calibration unverified',
    ground_truth_contact_available:false,executed_move_available:false,causal_effect_identified:false,
    physical_reset_label_available:false,physical_reset_reason:'count resolution is observed referee state; body reset execution and server timestamp are unavailable',
    runtime_enabled:false,binary_format:'24-byte LE header: magic REKFAL1 null, version=1, features, state_features, rows; float32 rows: split, group, capture_number, receipt_seconds, features, two labels'};
  fs.mkdirSync(output,{recursive:true});
  fs.writeFileSync(path.join(output,'fall-transition.bin'),out,{flag:'wx'});
  fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
  return {rows:rows.length,captures:captures.length,onsets:captures.reduce((n,c)=>n+c.onsets,0),splits};
}
module.exports={compact,featureRow,buildRows,FEATURE_ORDER,STATE_FEATURES,exportFiles};
if(require.main===module) exportFiles(...process.argv.slice(2)).then(r=>console.log(JSON.stringify(r))).catch(e=>{console.error(e.message);process.exitCode=1;});
