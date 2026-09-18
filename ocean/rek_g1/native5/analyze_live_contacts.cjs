#!/usr/bin/env node
'use strict';

// Offline analysis only. Native dispatch, client return values, score receipts
// and geometric context remain separate; absent effects packets are not misses.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const readline=require('node:readline');
const {poseFrom,brackets}=require('./join_passive_hit_events.cjs');
const MOVE_ORDER=[6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16]; // G1PolicyStreamContract.MoveOrder
const WINDOWS=[.25,.5,1,2,3];
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const check=(ok,why)=>{if(!ok)throw Error(why);};
const select=(value,keys)=>Object.fromEntries(keys.filter(k=>value?.[k]!==undefined).map(k=>[k,value[k]]));
const roundFields=['number','duration','time_remaining','active','redo','clean_hits','falls','result','result_value','winner_index','knockout'];
const identity=x=>`${x.round_identity_sha256}:${x.observation_sequence}`;
function clock(raw){return raw.clock??select(raw,['unity_frame','unity_time','monotonic_receipt_time','client_fixed_tick_at_observation']);}
function geometry(p){return {...select(p,['local_slot','root_distance_ground_xz_m','root_distance_3d_m','heading_axis','local_heading_rad','opponent_heading_rad','local_bearing_to_opponent_rad','opponent_bearing_to_local_rad']),
  local_root_position_xyz:p.local_root.root_position_xyz,opponent_root_position_xyz:p.opponent_root.root_position_xyz};}
function packetMatch(a,b){
  if(a.unity_frame!==b.unity_frame)return false;
  if(a.client_fixed_tick_at_observation!==undefined&&b.client_fixed_tick_at_observation!==undefined&&a.client_fixed_tick_at_observation!==b.client_fixed_tick_at_observation)return false;
  return !finite(a.monotonic_receipt_time)||!finite(b.monotonic_receipt_time)||Math.abs(a.monotonic_receipt_time-b.monotonic_receipt_time)<=.01;
}
function bindCapture(native,poses){
  const start=native.find(r=>r.event==='capture_start'),end=native.find(r=>r.event==='capture_end');
  check(start&&Number.isSafeInteger(start.pid)&&start.pid>0,'native_capture_header');
  const first=poses[0],last=poses.at(-1),hz=start.stopwatch_frequency_hz;
  check(hz===first.clock.qpc_frequency_hz,'native_clock_frequency');
  const firstUtc=Date.parse(first.clock.utc),startUtc=Date.parse(start.utc);
  check(finite(firstUtc)&&finite(startUtc)&&Math.abs((first.clock.qpc_ticks-start.stopwatch_timestamp_ticks)/hz-(firstUtc-startUtc)/1000)<.25,'native_utc_qpc_mismatch');
  check((first.clock.qpc_ticks-start.stopwatch_timestamp_ticks)/hz>=-1&&(first.clock.qpc_ticks-start.stopwatch_timestamp_ticks)/hz<=2,'native_start_window');
  if(end)check(Math.abs(last.clock.qpc_ticks-end.stopwatch_timestamp_ticks)/hz<=2,'native_end_window');
  const byFrame=new Map();for(const p of poses){if(!byFrame.has(p.clock.unity_frame))byFrame.set(p.clock.unity_frame,[]);byFrame.get(p.clock.unity_frame).push(p);}
  const anchors=[];
  for(const r of native){
    if(r.event==='raw_fight_state_packet')check(r.decoded?.round_number===first.round.number,'native_round_number_mismatch');
    if(['raw_score_packet','raw_hit_packet','raw_fight_state_packet'].includes(r.event))check(finite(r.unity_time)&&r.unity_time>=first.clock.unity_time-1&&r.unity_time<=last.clock.unity_time+1,'native_event_window');
    if(r.event!=='root_pose_sample')continue;
    check(r.round_number===first.round.number,'native_root_round_mismatch');
    for(const p of byFrame.get(r.unity_frame)??[]){
      const roots=[r.fighter_0_root?.world_position_xyz,r.fighter_1_root?.world_position_xyz],policy=[p.local_root,p.opponent_root];
      if(Math.abs(r.stopwatch_timestamp_ticks-p.clock.qpc_ticks)/hz>.15||Math.abs(r.unity_time-p.clock.unity_time)>.05)continue;
      if(policy.every(f=>Array.isArray(roots[f.slot])&&roots[f.slot].length===3&&roots[f.slot].every(finite)&&Math.hypot(...roots[f.slot].map((n,i)=>n-f.root_position_xyz[i]))<=.05)){anchors.push(p.clock.unity_time);break;}
    }
  }
  const span=anchors.length?Math.max(...anchors)-Math.min(...anchors):0;
  check(anchors.length>=2&&span>=Math.min(1,(last.clock.unity_time-first.clock.unity_time)*.5),'native_pose_clock_binding_missing');
  return {method:'concurrent_Unity_frame_QPC_and_both_root_positions',matching_pose_anchors:anchors.length,anchor_span_seconds:span,
    direct_process_id_equality:'unavailable_in_policy_stream',maximum_anchor_qpc_offset_seconds:.15,maximum_anchor_unity_offset_seconds:.05,maximum_anchor_root_difference_m:.05};
}
function analyzeRecords({relay,requests,native=null,summary,config,worker}){
  check(/^[a-f0-9]{64}$/.test(config.checkpoint_sha256),'checkpoint_hash');
  check(summary.checkpoint_sha256===config.checkpoint_sha256,'summary_checkpoint_mismatch');
  const ready=worker.filter(x=>x.type==='ready');
  check(ready.length===1&&ready[0].checkpoint_sha256===config.checkpoint_sha256&&ready[0].native_cuda===true,'worker_checkpoint_mismatch');
  const sources=new Map(),poses=[],segments=[],acks=new Map(),dispatches=new Map();
  let prior=null;
  relay.forEach((raw,index)=>{
    if(raw.event==='g1_policy_state'){
      check(!sources.has(identity(raw)),'duplicate_observation_identity');
      const p=poseFrom(raw,index+1);p.round=select(raw.round,roundFields);
      if(!prior||p.round_identity_sha256!==prior.round_identity_sha256||p.local_slot!==prior.local_slot||p.clock.qpc_frequency_hz!==prior.clock.qpc_frequency_hz||p.clock.qpc_ticks<prior.clock.qpc_ticks||p.clock.unity_time<prior.clock.unity_time||p.clock.unity_frame<prior.clock.unity_frame)segments.push([]);
      segments.at(-1).push(p);poses.push(p);prior=p;sources.set(identity(raw),p);
    }
    if(raw.event==='g1_policy_action'){
      check(!acks.has(raw.request_id),'duplicate_action_ack');acks.set(raw.request_id,raw);
    }
    if(raw.event==='g1_policy_dispatch'){
      if(!dispatches.has(raw.request_id))dispatches.set(raw.request_id,[]);
      dispatches.get(raw.request_id).push(raw);
    }
  });
  check(poses.length>0,'no_client_poses');
  const rounds=[...new Set(poses.map(p=>p.round_identity_sha256))];
  check(rounds.length===1&&poses.every(p=>p.local_slot===poses[0].local_slot),'one_round_and_local_slot_required');
  const local=poses[0].local_slot,nativeRows=native??[];
  const starts=nativeRows.filter(x=>x.event==='capture_start'),ends=nativeRows.filter(x=>x.event==='capture_end');
  check(starts.length<=1&&ends.length<=1,'multiple_native_captures');
  if(starts[0]?.scope?.local_fighter_index!==undefined)check(starts[0].scope.local_fighter_index===local,'native_local_slot_mismatch');
  const captureBinding=native===null?null:bindCapture(nativeRows,poses);
  const hitRaw=nativeRows.filter(x=>x.event==='raw_hit_packet'),scoreRaw=nativeRows.filter(x=>x.event==='raw_score_packet');
  const projections=nativeRows.filter(x=>x.event==='outbound_request_projection'&&x.message==='REK_Move');
  const attacks=requests.filter(x=>x.type==='policy_action'&&x.action>=16).map((request,index)=>{
    check(Number.isInteger(request.action)&&request.action<33,'attack_category');
    const source=sources.get(identity(request));check(source,'request_source_missing');
    const ack=acks.get(request.request_id),sent=dispatches.get(request.request_id)??[];
    if(ack)check(identity(ack)===identity(request)&&ack.action===request.action,'ack_identity_mismatch');
    const move=MOVE_ORDER[request.action-16];
    for(const d of sent)check(identity(d)===identity(request)&&d.move_index===move&&d.send_method_returned===true,'dispatch_identity_mismatch');
    const nativeCandidates=projections.filter(p=>p.move_index_wire_uint8===move&&sent.some(d=>d.clock?.unity_frame===p.unity_frame));
    return {index,request_id:request.request_id,observation_sequence:request.observation_sequence,round_identity_sha256:request.round_identity_sha256,
      policy_action:request.action,requested_move_index:move,source_clock:source.clock,request_geometry:geometry(source),
      local_ack:ack?select(ack,['applied','reason','execute_move_returned','pending_move_index','clock']):null,
      native_dispatches:sent.map(d=>({clock:d.clock,move_index:d.move_index,send_method_returned:true})),
      native_request_projection_candidates:nativeCandidates.map(p=>select(p,['request_sequence','message_request_sequence','unity_frame','unity_realtime_since_startup','move_index_wire_uint8'])),
      server_acceptance:'unknown',executed_move_index:null,contact_outcome:'unknown',causal_score_event_ids:[]};
  });
  check(new Set(attacks.map(a=>a.request_id)).size===attacks.length,'duplicate_attack_request');
  const poseContext=raw=>{
    const b=brackets(segments,raw);
    return {...b,candidates:b.candidates.map(c=>({...c,...Object.fromEntries(['prior','same_time','next','same_frame'].map(k=>[k,c[k].map(p=>({observation_sequence:p.observation_sequence,clock:p.clock,
      pose_minus_event_unity_seconds:p.pose_minus_event_unity_seconds,pose_minus_event_unity_frames:p.pose_minus_event_unity_frames,...geometry(p)}))]))}))};
  };
  const ages=(attack,event)=>{
    const ns=attack.native_request_projection_candidates;
    if(ns.length===1&&finite(ns[0].unity_realtime_since_startup)&&finite(event.monotonic_receipt_time))return [{seconds:event.monotonic_receipt_time-ns[0].unity_realtime_since_startup,basis:'native_client_monotonic_receipt_minus_request'}];
    return attack.native_dispatches.filter(d=>finite(d.clock?.unity_time)&&finite(event.unity_time)&&event.unity_frame>=d.clock.unity_frame)
      .map(d=>({seconds:event.unity_time-d.clock.unity_time,basis:'client_Unity_scaled_time_receipt_minus_dispatch'}));
  };
  const awards=scoreRaw.map((raw,index)=>{
    const d=raw.decoded;check([0,1].includes(d?.fighter_index)&&finite(d.points_awarded)&&d.points_awarded>=0&&Number.isInteger(d.new_hit_count),'invalid_score_packet');
    const matchingHits=hitRaw.map((h,i)=>packetMatch(raw,h)?i:-1).filter(i=>i>=0);
    const candidates=d.fighter_index===local&&[1,2].includes(d.points_awarded)?attacks.flatMap(a=>ages(a,raw).filter(x=>x.seconds>=0&&x.seconds<=3).map(age=>({request_id:a.request_id,requested_move_index:a.requested_move_index,...age}))):[];
    return {index,clock:clock(raw),fighter_index:d.fighter_index,new_hit_count:d.new_hit_count,points_awarded:d.points_awarded,
      award_class:d.points_awarded===5?'five_point_award_cause_unresolved':'other_award',matching_hit_indices:matchingHits,
      candidate_requests:candidates,window_sensitivity:WINDOWS.map(seconds=>({window_seconds:seconds,request_ids:[...new Set(candidates.filter(c=>c.seconds<=seconds).map(c=>c.request_id))]})),
      unique_same_frame_hit_association:matchingHits.length===1&&scoreRaw.filter(s=>packetMatch(s,hitRaw[matchingHits[0]])).length===1,
      poses:poseContext(raw),causal_request_id:null,executed_move_index:null};
  });
  const hits=hitRaw.map((raw,index)=>({index,clock:clock(raw),...select(raw.decoded,['position_xyz','surface_normal_xyz','relative_speed','is_kick']),
    same_frame_score_indices:awards.filter(s=>s.matching_hit_indices.includes(index)).map(s=>s.index),poses:poseContext(raw),
    confirmed_attacker:null,body_zone:null,executed_move_index:null,causal_request_id:null}));
  const terminal=[];
  for(const p of poses)if(p.round.active===false&&p.round.result_value>0)terminal.push({basis:'client_replicated_round',...p.round});
  for(const raw of nativeRows)if(raw.event==='raw_fight_state_packet'&&raw.decoded?.round_active===0&&raw.decoded.round_result>0){
    const d=raw.decoded;
    if(poses[0].round.number!==undefined)check(d.round_number===poses[0].round.number,'native_round_number_mismatch');
    terminal.push({basis:'received_fight_state_packet',number:d.round_number,active:false,result_value:d.round_result,winner_index:d.round_winner,
      redo:d.is_redo===1,clean_hits:[d.hits_0,d.hits_1],knockout:d.knockout_occurred===1});
  }
  const terminalKeys=new Set(terminal.map(t=>JSON.stringify([t.result_value,t.winner_index,t.redo===true])));
  const consistent=terminal.length>0&&terminalKeys.size===1,last=consistent?terminal.at(-1):null;
  const validPoints=value=>Array.isArray(value)&&value.length===2&&value.every(n=>Number.isSafeInteger(n)&&n>=0);
  const terminalPoints=terminal.filter(t=>validPoints(t.clean_hits)).map(t=>t.clean_hits);
  const pointsConsistent=terminalPoints.length>0&&new Set(terminalPoints.map(p=>JSON.stringify(p))).size===1;
  const outcome=!last||last.redo||last.result_value===4?'unknown':last.result_value===3?'draw':![1,2].includes(last.result_value)?'unknown':last.winner_index===local?'win':last.winner_index===(local^1)?'loss':'unknown';
  const sum=[0,0],integerSum=[0,0];for(const a of awards){sum[a.fighter_index]+=a.points_awarded;integerSum[a.fighter_index]+=Math.trunc(a.points_awarded);}
  const finalized=starts.length===1&&ends.length===1;
  const countMatch=finalized&&ends[0].raw_score_packet_count===awards.length&&ends[0].raw_hit_packet_count===hits.length;
  const errors=nativeRows.filter(x=>x.event==='capture_error').length;
  const complete=finalized&&countMatch&&errors===0&&ends[0].capture_error_count===0;
  const nativeCounterReconciliation=[0,1].map(slot=>{
    const ss=awards.filter(a=>a.fighter_index===slot);let running=0;
    const everyCumulativeMatches=ss.every(a=>{running+=Math.trunc(a.points_awarded);return a.new_hit_count===running;});
    return {slot,observed_point_awards:sum[slot],observed_integer_counter_increments:integerSum[slot],last_received_new_hit_count:ss.at(-1)?.new_hit_count??null,
      every_received_counter_equals_awards_from_zero:everyCumulativeMatches,terminal_clean_hits:last?.clean_hits?.[slot]??null,
      terminal_counter_matches_award_sum:last?.clean_hits?.[slot]===integerSum[slot]};
  });
  const fullPoints=complete&&consistent&&pointsConsistent&&nativeCounterReconciliation.every(r=>r.every_received_counter_equals_awards_from_zero&&r.terminal_counter_matches_award_sum);
  const first=poses[0],final=poses.at(-1),beginning=finite(first.round.duration)&&first.round.time_remaining>=first.round.duration-1;
  const appliedClocks=[];
  for(const request of requests.filter(r=>r.type==='policy_action')){
    const ack=acks.get(request.request_id);if(!ack)continue;
    check(identity(ack)===identity(request)&&ack.action===request.action&&sources.has(identity(request)),'control_ack_identity_mismatch');
    if(ack.applied===true&&finite(ack.clock?.unity_time))appliedClocks.push(ack.clock.unity_time);
  }
  appliedClocks.sort((a,b)=>a-b);
  const gaps=appliedClocks.slice(1).map((t,i)=>t-appliedClocks[i]);
  const coverage={applied_action_returns:appliedClocks.length,first_applied_seconds_after_first_observation:appliedClocks.length?appliedClocks[0]-first.clock.unity_time:null,
    terminal_seconds_after_last_applied:appliedClocks.length?final.clock.unity_time-appliedClocks.at(-1):null,
    maximum_applied_action_gap_seconds:gaps.length?Math.max(...gaps):null,tolerance_seconds:1,
    basis:'observed local policy action returns; server acceptance remains unknown'};
  const covered=beginning&&consistent&&appliedClocks.length>=2&&coverage.first_applied_seconds_after_first_observation>=0&&coverage.first_applied_seconds_after_first_observation<=1&&
    coverage.terminal_seconds_after_last_applied>=0&&coverage.terminal_seconds_after_last_applied<=1&&coverage.maximum_applied_action_gap_seconds<=1&&['win','loss','draw'].includes(outcome);
  const result={schema:'rek.authentic_live_contact_analysis.v1',authentic_client:true,checkpoint_sha256:config.checkpoint_sha256,
    local_slot:local,opponent:summary.opponent??null,round_identity_sha256:rounds[0],stop_reason:summary.stop_reason,
    outcome,terminal_evidence_consistent:consistent,terminal_records:terminal,first_observed_round:poses[0].round,last_observed_round:poses.at(-1).round,
    terminal_awarded_points_by_slot:consistent&&pointsConsistent?terminalPoints.at(-1):null,
    terminal_point_counters_consistent:pointsConsistent,
    native_point_counter_semantics:'Round.CleanHits accumulates integer awarded points, including referee awards; this field is not a strike-event count',
    point_semantics_source:['PointTracker.RecordHit: integer-truncate award and add to RoundState.CleanHits',
      'PointTracker.RecordRefereeAward: same integer accumulation',
      'FightCoordinator.OnPointScoredNetwork: sends current counter as newHitCount',
      'FightCoordinator.OnScoreReceived: assigns received newHitCount directly to currentRound.CleanHits'],
    initial_observation_within_first_second:beginning,terminal_observed:terminal.length>0,completed_policy_round:covered,
    observed_round_clock_seconds:first.round.time_remaining-final.round.time_remaining,policy_control_coverage:coverage,
    policy_attack_requests:attacks.length,locally_applied_attack_returns:attacks.filter(a=>a.local_ack?.applied===true).length,
    native_dispatched_attack_requests:attacks.filter(a=>a.native_dispatches.length).length,native_dispatch_returns:attacks.reduce((n,a)=>n+a.native_dispatches.length,0),
    native_capture_supplied:native!==null,native_capture_complete:complete,native_capture_binding:captureBinding,native_capture_error_records:errors,
    score_events:native===null?null:awards.length,received_hit_effects:native===null?null:hits.length,
    observed_point_awards_by_slot:native===null?null:sum,reconciled_full_round_points_by_slot:fullPoints?integerSum:null,
    point_counter_reconciliation:native===null?null:nativeCounterReconciliation,five_point_awards:awards.filter(a=>a.points_awarded===5).length,
    causal_attack_outcomes:0,definitive_contact_regions:false,
    limits:['Local execute returns and SendMoveEvent returns do not prove server acceptance or playback.',
      'Score recipients and awarded points are packet observations. A same-frame score/hit pairing remains noncausal.',
      'Every attack retains unknown contact outcome. No missing packet is converted to a miss.',
      'Candidate windows are sensitivity analyses, not execution-duration or unique-active-action proof.',
      'Packet contact coordinates are received effects data; adjacent root poses describe receipt context, without interpolation.',
      'Facing uses projected pelvis-local +X in Unity XZ. Runtime forward offset is not newly measured.',
      'Five-point awards are separate; no knockout cause is inferred from their value.',
      'Terminal scoreboard points are separate from local policy control coverage and complete packet reconciliation.',
      'Completed-policy-round coverage requires the first observation within one second of round start, an applied return within one second of that observation, and at most one second between returns or before terminal; server control remains unknown.',
      'Native correlation requires concurrent frame, QPC and root geometry anchors. Direct process ID equality is unavailable in the policy stream.']};
  return {summary:result,attacks,awards,hits};
}
async function loadLines(file,keep){
  const before=fs.statSync(file),hash=crypto.createHash('sha256'),rows=[];
  const input=fs.createReadStream(file);input.on('data',b=>hash.update(b));
  let line=0;for await(const text of readline.createInterface({input,crlfDelay:Infinity})){
    line++;if(!text.trim())continue;let value;try{value=JSON.parse(text);}catch{throw Error(`invalid_json_line_${line}`);}
    const selected=keep(value);if(selected)rows.push(selected===true?value:selected);
  }
  const after=fs.statSync(file);check(before.size===after.size&&before.mtimeMs===after.mtimeMs,'input_changed_during_analysis');
  return {rows,provenance:{file:path.basename(file),sha256:hash.digest('hex'),bytes:after.size,lines:line}};
}
async function analyzeFiles(trial,nativeFile,output){
  check(!fs.existsSync(output),'output_exists');
  const relay=await loadLines(path.join(trial,'relay.stdout.jsonl'),x=>['g1_policy_state','g1_policy_action','g1_policy_dispatch'].includes(x.event));
  const requests=await loadLines(path.join(trial,'relay.stdin.jsonl'),x=>x.type==='policy_action');
  const worker=await loadLines(path.join(trial,'worker.stdout.jsonl'),x=>x.type==='ready');
  const native=nativeFile==='-'?null:await loadLines(nativeFile,x=>x.event==='root_pose_sample'&&x.root_pose_sample_index%50===0?
    select(x,['event','round_number','unity_frame','unity_time','stopwatch_timestamp_ticks','fighter_0_root','fighter_1_root']):
    ['capture_start','capture_end','capture_error','raw_hit_packet','raw_score_packet','raw_fight_state_packet','outbound_request_projection'].includes(x.event));
  const summary=JSON.parse(fs.readFileSync(path.join(trial,'summary.json'))),config=JSON.parse(fs.readFileSync(path.join(trial,'run-config.json')));
  const result=analyzeRecords({relay:relay.rows,requests:requests.rows,worker:worker.rows,native:native?.rows??null,summary,config});
  result.summary.inputs=[relay,requests,worker,...(native?[native]:[])].map(x=>x.provenance);
  for(const name of ['run-config.json','summary.json']){const b=fs.readFileSync(path.join(trial,name));result.summary.inputs.push({file:name,bytes:b.length,sha256:crypto.createHash('sha256').update(b).digest('hex')});}
  fs.mkdirSync(output);
  fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify(result.summary,null,2)+'\n',{flag:'wx'});
  for(const [name,rows] of [['attacks',result.attacks],['score-events',result.awards],['hit-events',result.hits]])fs.writeFileSync(path.join(output,`${name}.jsonl`),rows.map(JSON.stringify).join('\n')+(rows.length?'\n':''),{flag:'wx'});
  return result.summary;
}
module.exports={analyzeRecords,analyzeFiles,packetMatch,MOVE_ORDER};
if(require.main===module){
  Promise.resolve().then(()=>{check(process.argv.length===5,'usage: analyze_live_contacts.cjs TRIAL_DIRECTORY RECORDER_JSONL_OR_- NEW_OUTPUT_DIRECTORY');return analyzeFiles(...process.argv.slice(2));})
    .then(result=>console.log(JSON.stringify(result))).catch(error=>{const code=/^[a-z0-9_]+$/.test(error.message)?error.message:/^[A-Z0-9_]+$/.test(error.code??'')?error.code:'invalid_input';console.error(`Live contact analysis rejected: ${code}`);process.exitCode=2;});
}
