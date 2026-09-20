#!/usr/bin/env node
'use strict';

// Reproducible public-field aggregation of existing offline validators only.
// No game connections; no new completion thresholds or outcome exclusions.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const check=(ok,why)=>{if(!ok)throw Error(why);},hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const finite=Number.isFinite,OUTCOMES=new Set(['win','loss','draw']);
function json(file,inputs){if(!fs.existsSync(file))return null;const b=fs.readFileSync(file);inputs.push({file:path.basename(path.dirname(file))+'/'+path.basename(file),sha256:hash(b),bytes:b.length});return JSON.parse(b);}
async function firstReady(file){if(!fs.existsSync(file))return null;
  for await(const line of readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity})){if(!line.trim())continue;let r;try{r=JSON.parse(line);}catch{return null;}if(r.type==='ready')return r;}
  return null;
}
async function scoreAwards(file){if(!fs.existsSync(file))return null;const counts=[{count:0,points:0,five_count:0,five_points:0,nonfive_count:0,nonfive_points:0,histogram:{}},{count:0,points:0,five_count:0,five_points:0,nonfive_count:0,nonfive_points:0,histogram:{}}],ids=new Set();
  const digest=crypto.createHash('sha256'),stream=fs.createReadStream(file);stream.on('data',b=>digest.update(b));
  for await(const line of readline.createInterface({input:stream,crlfDelay:Infinity})){
    if(!line.trim())continue;const r=JSON.parse(line);check([0,1].includes(r.fighter_index)&&Number.isInteger(r.points_awarded)&&r.points_awarded>0&&Number.isInteger(r.index)&&!ids.has(r.index),'score_event_schema_or_duplicate');ids.add(r.index);
    const c=counts[r.fighter_index];c.count++;c.points+=r.points_awarded;const type=r.points_awarded===5?'five':'nonfive';c[type+'_count']++;c[type+'_points']+=r.points_awarded;c.histogram[r.points_awarded]=(c.histogram[r.points_awarded]??0)+1;
  }
  return {by_slot:counts,source_sha256:digest.digest('hex'),semantics:'Observed awards grouped by amount. Five points alone does not prove countout cause; non-five awards are not asserted to be causally attributed strikes.'};
}
function failureCategory(failure){if(!failure)return null;const s=String(failure);
  if(/publication|json|sharing|another process/i.test(s))return 'json_publication_or_file_access';
  if(/readiness|prewarm|worker|encoder/i.test(s))return 'inference_preparation';
  if(/continuation|authenticated|account/i.test(s))return 'authenticated_continuation_or_account_scope';
  if(/ownership|owned|competing|client|cleanup|isolation/i.test(s))return 'client_ownership_or_cleanup_scope';
  if(/deadline|timeout/i.test(s))return 'deadline_or_timeout';return 'other_orchestration_failure';
}
function status(result,contact){
  if(contact?.completed_policy_round===true)return 'complete';
  if(contact?.completed_policy_round===false)return 'incomplete';
  if(result?.driver_complete===true)return 'awaiting_strict_analysis';
  return result?'orchestration_incomplete':'pending';
}
function existingReasons(result,c){const reasons=[];
  if(!result)reasons.push('result_not_yet_available');
  if(result?.failure)reasons.push(failureCategory(result.failure));
  if(result&&result.driver_complete!==true)reasons.push('driver_not_complete');
  if(!c)reasons.push('strict_contact_analysis_not_yet_available');
  if(c?.completed_policy_round===false){
    reasons.push('existing_analyzer_classifies_incomplete');
    for(const name of ['initial_observation_within_first_second','terminal_observed','terminal_evidence_consistent','native_capture_complete','terminal_point_counters_consistent'])if(c[name]===false)reasons.push(name+':false');
    const coverage=c.policy_control_coverage,t=coverage?.tolerance_seconds;
    if(finite(t))for(const name of ['first_applied_seconds_after_first_observation','maximum_applied_action_gap_seconds','terminal_seconds_after_last_applied'])
      if(finite(coverage[name])&&coverage[name]>t)reasons.push(name+':exceeds_existing_analyzer_tolerance');
  }
  return [...new Set(reasons)];
}
async function trial(directory,id,cohort){
  const inputs=[],result=json(path.join(directory,'result.json'),inputs),config=json(path.join(directory,'trial','run-config.json'),inputs)??json(path.join(directory,'config.json'),inputs);
  const contact=json(path.join(directory,'contact-analysis','summary.json'),inputs),referee=json(path.join(directory,'referee-validation','live-referee-validation.json'),inputs);
  const summary=result?.summary??json(path.join(directory,'trial','summary.json'),inputs),ready=await firstReady(path.join(directory,'trial','worker.stdout.jsonl'));
  if(contact)check(contact.schema==='rek.authentic_live_contact_analysis.v1','unsupported_contact_summary');
  if(referee)check(referee.schema==='rek.live_referee_validation.v1','unsupported_referee_validation');
  const local=contact?.local_slot??summary?.local_slot??null,opponent=local===0?1:local===1?0:null;
  const awards=await scoreAwards(path.join(directory,'contact-analysis','score-events.jsonl'));
  const rawPoints=contact?.terminal_awarded_points_by_slot??summary?.final_round?.clean_hits??null;
  const outcome=contact?.outcome??summary?.round_outcome??null;
  const selection=ready?.selection??null,mask=ready?.feature_mask_sha256??config?.feature_mask_sha256??null;
  const identity={checkpoint_sha256:summary?.checkpoint_sha256??contact?.checkpoint_sha256??ready?.checkpoint_sha256??config?.checkpoint_sha256??null,
    selection,selection_requested:config?.selection??null,feature_mask_sha256:mask===''?null:mask,
    feature_mask_status:mask==null?'not_reported':mask===''?'explicitly_disabled':'reported_enabled',
    projection:summary?.projection??config?.projection??null,observation_schema:ready?.observation_schema??null,
    inference_precision:ready?.precision??null,inference_seed:ready?.seed??null,
    opponent_difficulty:contact?.opponent?.client_ai_difficulty??summary?.opponent?.client_ai_difficulty??null,
    opponent_bot:contact?.opponent?.sparring_bot_number??summary?.opponent?.sparring_bot_number??null};
  const consistency={checkpoint:ready?.checkpoint_sha256&&identity.checkpoint_sha256?ready.checkpoint_sha256===identity.checkpoint_sha256:null,
    selection:config?.selection&&selection?config.selection===selection:null,
    feature_mask:config?.feature_mask_sha256!=null&&ready?.feature_mask_sha256!=null?config.feature_mask_sha256===ready.feature_mask_sha256:null};
  const coverage=contact?.policy_control_coverage;
  const totalsConsistent=awards&&contact?.observed_point_awards_by_slot?awards.by_slot.every((c,s)=>c.points===contact.observed_point_awards_by_slot[s]):null;
  const countouts=referee?.native?.counts?.filter(c=>c.explicit_countout===true)??null;
  return {trial_id:id,cohort,status:status(result,contact),recorded_outcome:OUTCOMES.has(outcome)?outcome:null,
    incomplete_or_pending_reasons:existingReasons(result,contact),orchestration_ok:result?.orchestration_ok??null,driver_complete:result?.driver_complete??null,
    failure_category:failureCategory(result?.failure),failure_sha256:result?.failure?hash(String(result.failure)):null,
    identity,configuration_group_sha256:hash(JSON.stringify(identity)),configuration_consistency:consistency,
    local_slot:local,terminal_points_by_slot:rawPoints,terminal_points:rawPoints&&local!=null?{local:rawPoints[local],opponent:rawPoints[opponent],margin:rawPoints[local]-rawPoints[opponent]}:null,
    observed_awards:awards&&local!=null?{local:awards.by_slot[local],opponent:awards.by_slot[opponent],matches_analyzer_award_totals:totalsConsistent,source_sha256:awards.source_sha256,semantics:awards.semantics}:null,
    strict_analysis:{completed_policy_round:contact?.completed_policy_round??null,terminal_evidence_consistent:contact?.terminal_evidence_consistent??null,
      initial_observation_within_first_second:contact?.initial_observation_within_first_second??null,native_capture_complete:contact?.native_capture_complete??null,
      terminal_point_counters_consistent:contact?.terminal_point_counters_consistent??null},
    control:{maximum_applied_action_gap_seconds:coverage?.maximum_applied_action_gap_seconds??null,
      first_applied_delay_seconds:coverage?.first_applied_seconds_after_first_observation??null,terminal_after_last_applied_seconds:coverage?.terminal_seconds_after_last_applied??null,
      existing_analyzer_tolerance_seconds:coverage?.tolerance_seconds??null,policy_attack_requests:contact?.policy_attack_requests??null,
      native_dispatched_attack_requests:contact?.native_dispatched_attack_requests??null,locally_applied_attack_returns:contact?.locally_applied_attack_returns??null,
      predictions:summary?.predictions??null,applied:summary?.applied??null,rejected:summary?.rejected??null,
      observed_round_clock_seconds:contact?.observed_round_clock_seconds??null},
    referee:{verification_passed:referee?.verification_passed??null,available_source_count:referee?.available_source_count??null,
      unavailable_source_count:referee?.unavailable_source_count??null,unavailable_reasons:referee?.unavailable_reasons??null,unique_receipts:referee?.unique_bridge_receipts??null,
      unique_latched_call_ids:referee?.unique_latched_call_ids??null,maximum_receipt_age_seconds:referee?.receipt_age_seconds?.maximum??null,
      uncensored_countouts_against_local:countouts&&local!=null?countouts.filter(c=>c.slot===local&&!c.left_censored&&!c.right_censored).length:null,
      uncensored_countouts_against_opponent:countouts&&local!=null?countouts.filter(c=>c.slot===opponent&&!c.left_censored&&!c.right_censored).length:null},
    provenance:{inputs,worker_ready_record_sha256:ready?hash(JSON.stringify(ready)):null,worker_command_sha256:config?.worker?hash(JSON.stringify(config.worker)):null},
    note:'Status reproduces existing strict analyzer; referee verification and orchestration are reported separately. No unfavorable result is recategorized by this aggregation.'};
}
function aggregate(rows){const complete=rows.filter(x=>x.status==='complete'),wins=complete.filter(x=>x.recorded_outcome==='win').length,losses=complete.filter(x=>x.recorded_outcome==='loss').length,draws=complete.filter(x=>x.recorded_outcome==='draw').length;
  const scored=complete.filter(x=>x.terminal_points),awards=complete.filter(x=>x.observed_awards),gaps=complete.map(x=>x.control.maximum_applied_action_gap_seconds).filter(finite);
  const sum=(xs,fn)=>xs.reduce((n,x)=>n+fn(x),0);
  return {attempts:rows.length,strict_completed_rounds:complete.length,wins,losses,draws,recorded_outcomes_all_attempts:{wins:rows.filter(x=>x.recorded_outcome==='win').length,losses:rows.filter(x=>x.recorded_outcome==='loss').length,draws:rows.filter(x=>x.recorded_outcome==='draw').length},
    incomplete_or_pending:rows.length-complete.length,points_rounds:scored.length,points:scored.length?{local:sum(scored,x=>x.terminal_points.local),opponent:sum(scored,x=>x.terminal_points.opponent),margin:sum(scored,x=>x.terminal_points.margin)}:null,
    award_breakdown_rounds:awards.length,award_points:awards.length?Object.fromEntries(['local','opponent'].map(s=>[s,{five_point:sum(awards,x=>x.observed_awards[s].five_points),non_five_point:sum(awards,x=>x.observed_awards[s].nonfive_points)}])):null,
    maximum_applied_action_gap_seconds:gaps.length?Math.max(...gaps):null,
    policy_attack_requests:sum(complete,x=>x.control.policy_attack_requests??0),attack_count_rounds:complete.filter(x=>x.control.policy_attack_requests!=null).length,
    referee_verified_completed_rounds:complete.filter(x=>x.referee.verification_passed===true).length};
}
function markdown(report){const lines=['# Authentic private-AI trial summary','',`Snapshot: ${report.created_utc}`,'','Existing strict analysis determines completion. Development cohorts are never added to future holdout results.','',
  '| Cohort | Attempts | Complete | W / L / D | Points | Five-point awards, local : opponent |',
  '| --- | ---: | ---: | --- | --- | --- |'];
  for(const c of report.cohorts){const s=c.aggregate;lines.push(`| ${c.cohort} | ${s.attempts} | ${s.strict_completed_rounds} | ${s.wins} / ${s.losses} / ${s.draws} | ${s.points?`${s.points.local}:${s.points.opponent}`:'unknown'} | ${s.award_points?`${s.award_points.local.five_point}:${s.award_points.opponent.five_point}`:'unknown'} |`);}
  lines.push('','## Unchanged checkpoint/configuration results','',
    'Cohort totals above combine different policies. Use these separate groups when comparing fighting performance. Incomplete attempts remain in the attempt count.','',
    '| Cohort | Checkpoint | Selection | Input mask | Seed | Bot / difficulty | Attempts / complete | W / L / D | Points | Non-five / five points, local : opponent |',
    '| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |');
  for(const g of report.configuration_groups??[]){const i=g.identity,s=g.aggregate,a=s.award_points;
    const mask=i.feature_mask_sha256?i.feature_mask_sha256.slice(0,12):i.feature_mask_status;
    lines.push(`| ${g.cohort} | ${i.checkpoint_sha256?.slice(0,12)??'unknown'} | ${i.selection??'unknown'} | ${mask??'unknown'} | ${i.inference_seed??'unknown'} | ${i.opponent_bot??'unknown'} / ${i.opponent_difficulty??'unknown'} | ${s.attempts} / ${s.strict_completed_rounds} | ${s.wins} / ${s.losses} / ${s.draws} | ${s.points?`${s.points.local}:${s.points.opponent}`:'unknown'} | ${a?`${a.local.non_five_point}/${a.local.five_point} : ${a.opponent.non_five_point}/${a.opponent.five_point}`:'unknown'} |`);
  }
  lines.push('','| Trial | Cohort | Selection | Status / recorded outcome | Points | Non-five / five points, local : opponent | Max control gap (s) | Attack requests |','| --- | --- | --- | --- | --- | --- | ---: | ---: |');
  for(const r of report.trials){const a=r.observed_awards;lines.push(`| ${r.trial_id} | ${r.cohort} | ${r.identity.selection??'unknown'} | ${r.status} / ${r.recorded_outcome??'unknown'} | ${r.terminal_points?`${r.terminal_points.local}:${r.terminal_points.opponent}`:'unknown'} | ${a?`${a.local.nonfive_points}/${a.local.five_points} : ${a.opponent.nonfive_points}/${a.opponent.five_points}`:'unknown'} | ${r.control.maximum_applied_action_gap_seconds?.toFixed(3)??'unknown'} | ${r.control.policy_attack_requests??'unknown'} |`);}
  lines.push('','Missing analyses and failed attempts remain visible. Non-five-point awards are grouped by amount; move causality is not inferred. Five-point values alone do not establish countout cause.','');return lines.join('\n');
}
async function run(root,previousRoot,out,cohortFile){
  check(!fs.existsSync(out),'output_exists');const cohortInputs=[],spec=cohortFile?json(cohortFile,cohortInputs):null;
  if(spec)check(spec.schema==='rek.authentic_trial_cohorts.v1'&&Array.isArray(spec.holdout_trial_ids)&&spec.holdout_trial_ids.every(x=>typeof x==='string'&&/^live-[A-Za-z0-9_.-]+$/.test(x))&&new Set(spec.holdout_trial_ids).size===spec.holdout_trial_ids.length,'cohort_manifest_invalid');
  const holdout=new Set(spec?.holdout_trial_ids??[]),trials=[];
  const directories=fs.readdirSync(root,{withFileTypes:true}).filter(x=>x.isDirectory()&&/^live-[A-Za-z0-9_.-]+$/.test(x.name)).map(x=>x.name).sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
  for(const id of directories)trials.push(await trial(path.join(root,id),id,holdout.has(id)?'future_frozen_holdout_20':'current_development'));
  for(const id of ['live-round_outcome_v1-r2','live-round_outcome_v1-r4'])if(fs.existsSync(path.join(previousRoot,id)))trials.push(await trial(path.join(previousRoot,id),'prior/'+id,'prior_development'));
  const names=['prior_development','current_development','future_frozen_holdout_20'];
  const report={schema:'rek.authentic_trial_aggregate.v1',created_utc:new Date().toISOString(),tool_sha256:hash(fs.readFileSync(__filename)),
    completion_contract:'Use existing contact-analysis completed_policy_round verbatim; report other checks independently without new gates. Preserve all recorded outcomes and incomplete attempts.',
    cohort_contract:'Default every newly discovered trial to current development. Only explicit holdout_trial_ids assigns future frozen holdout. Prior two development wins remain separate.',
    holdout_target_rounds:20,holdout_assignment_manifest:cohortInputs,unseen_declared_holdout_ids:[...holdout].filter(id=>!directories.includes(id)),
    cohorts:names.map(cohort=>({cohort,aggregate:aggregate(trials.filter(x=>x.cohort===cohort))})),
    configuration_groups:[...new Set(trials.map(x=>x.cohort+':'+x.configuration_group_sha256))].map(key=>{const matching=trials.filter(x=>x.cohort+':'+x.configuration_group_sha256===key);return {cohort:matching[0].cohort,configuration_group_sha256:matching[0].configuration_group_sha256,identity:matching[0].identity,aggregate:aggregate(matching)};}),
    trials,no_game_connection:true,no_raw_account_records:true,no_new_validation_gates:true};
  fs.mkdirSync(out);fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});fs.writeFileSync(path.join(out,'README.md'),markdown(report),{flag:'wx'});return report;
}
module.exports={scoreAwards,status,existingReasons,trial,aggregate,markdown,run};
if(require.main===module)Promise.resolve().then(()=>{check(process.argv.length===5||process.argv.length===6,'usage_aggregate_authentic_trials_ROOT_PREVIOUS_ROOT_NEW_OUTPUT_OPTIONAL_COHORT_JSON');return run(...process.argv.slice(2));}).then(r=>console.log(JSON.stringify(r.cohorts))).catch(e=>{console.error(e.message);process.exitCode=1;});
