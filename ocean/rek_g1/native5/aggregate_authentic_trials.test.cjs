'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const v=require('./aggregate_authentic_trials.cjs');
async function fixture(fn){const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-public-aggregate-'));try{return await fn(dir);}finally{
  assert.equal(path.dirname(path.resolve(dir)),path.resolve(os.tmpdir()));assert.ok(path.basename(dir).startsWith('rek-public-aggregate-'));fs.rmSync(dir,{recursive:true});}}
function write(dir,relative,value){const file=path.join(dir,relative);fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,typeof value==='string'?value:JSON.stringify(value));}
function createTrial(dir,id,local=0,outcome='win',complete=true){
  const location=path.join(dir,id),points=local===0?[9,4]:[4,9];
  write(location,'result.json',{driver_complete:true,orchestration_ok:false,failure:'unknown account SecretAccount',summary:{checkpoint_sha256:'a'.repeat(64),round_outcome:outcome,local_slot:local,final_round:{clean_hits:points}}});
  write(location,'trial/run-config.json',{checkpoint_sha256:'a'.repeat(64),selection:'argmax',feature_mask_sha256:'',expected_account:'SecretAccount',worker:['SecretWorkerCommand'],projection:'projection'});
  write(location,'trial/worker.stdout.jsonl',JSON.stringify({type:'ready',checkpoint_sha256:'a'.repeat(64),selection:'sampled',feature_mask_sha256:'',observation_schema:'223',precision:'bf16',seed:73})+'\n');
  write(location,'contact-analysis/summary.json',{schema:'rek.authentic_live_contact_analysis.v1',completed_policy_round:complete,outcome,local_slot:local,terminal_awarded_points_by_slot:points,
    observed_point_awards_by_slot:points,policy_attack_requests:3,terminal_evidence_consistent:true,initial_observation_within_first_second:true,native_capture_complete:true,
    policy_control_coverage:{maximum_applied_action_gap_seconds:.2,tolerance_seconds:1}});
  write(location,'contact-analysis/score-events.jsonl',[
    {index:0,fighter_index:local,points_awarded:5},{index:1,fighter_index:local,points_awarded:4},{index:2,fighter_index:local^1,points_awarded:4}].map(JSON.stringify).join('\n')+'\n');
  return location;
}
test('status follows existing analyzer, with referee/orchestration separate',()=>{
  assert.equal(v.status({driver_complete:true,orchestration_ok:false},{completed_policy_round:true}),'complete');
  assert.equal(v.status({driver_complete:true},{completed_policy_round:false}),'incomplete');
  assert.equal(v.status({driver_complete:true},null),'awaiting_strict_analysis');assert.equal(v.status(null,null),'pending');
});
test('only existing analyzer tolerance diagnoses an incomplete control gap',()=>{
  const c={completed_policy_round:false,policy_control_coverage:{tolerance_seconds:1,maximum_applied_action_gap_seconds:.8}};
  assert.ok(!v.existingReasons({},c).some(x=>x.includes('exceeds')));c.policy_control_coverage.maximum_applied_action_gap_seconds=1.01;
  assert.ok(v.existingReasons({},c).some(x=>x.includes('exceeds_existing')));
});
test('slot1 local score accounting and amount-only award breakdown are correct',async()=>fixture(async dir=>{
  const file=createTrial(dir,'live-test-r1',1);const r=await v.trial(file,'live-test-r1','development');
  assert.deepEqual(r.terminal_points,{local:9,opponent:4,margin:5});assert.equal(r.observed_awards.local.five_points,5);assert.equal(r.observed_awards.local.nonfive_points,4);
  assert.equal(r.observed_awards.opponent.five_points,0);assert.equal(r.observed_awards.matches_analyzer_award_totals,true);
}));
test('actual ready selection takes precedence and public report excludes raw account/commands',async()=>fixture(async dir=>{
  const file=createTrial(dir,'live-test-r1'),r=await v.trial(file,'live-test-r1','development');
  assert.equal(r.identity.selection,'sampled');assert.equal(r.identity.selection_requested,'argmax');assert.equal(r.configuration_consistency.selection,false);
  assert.equal(r.identity.feature_mask_status,'explicitly_disabled');assert.equal(r.failure_category,'authenticated_continuation_or_account_scope');
  assert.ok(!JSON.stringify(r).includes('SecretAccount'));assert.ok(!JSON.stringify(r).includes('SecretWorkerCommand'));assert.equal(r.status,'complete');
}));
test('unfavorable incomplete outcomes remain visible outside strict totals',async()=>fixture(async dir=>{
  const a=await v.trial(createTrial(dir,'live-a-r1',0,'win',true),'live-a-r1','development');
  const b=await v.trial(createTrial(dir,'live-b-r1',0,'loss',false),'live-b-r1','development');const s=v.aggregate([a,b]);
  assert.equal(s.attempts,2);assert.equal(s.strict_completed_rounds,1);assert.equal(s.recorded_outcomes_all_attempts.losses,1);assert.equal(s.incomplete_or_pending,1);
}));
test('prior development and explicit future holdout never merge; fresh outputs only',async()=>fixture(async dir=>{
  const current=path.join(dir,'current'),previous=path.join(dir,'previous');fs.mkdirSync(current);fs.mkdirSync(previous);
  createTrial(current,'live-current-r1');createTrial(current,'live-current-r2');createTrial(previous,'live-round_outcome_v1-r2');createTrial(previous,'live-round_outcome_v1-r4');
  write(dir,'cohorts.json',{schema:'rek.authentic_trial_cohorts.v1',holdout_trial_ids:['live-current-r2','live-current-r3']});
  const out=path.join(dir,'output'),r=await v.run(current,previous,out,path.join(dir,'cohorts.json'));
  assert.deepEqual(r.cohorts.map(c=>c.aggregate.strict_completed_rounds),[2,1,1]);assert.deepEqual(r.unseen_declared_holdout_ids,['live-current-r3']);
  assert.ok(!fs.readFileSync(path.join(out,'summary.json'),'utf8').includes('SecretAccount'));await assert.rejects(v.run(current,previous,out),/output_exists/);
}));
test('duplicate score-event records fail rather than double-count awards',async()=>fixture(async dir=>{
  write(dir,'scores.jsonl',[{index:0,fighter_index:0,points_awarded:1},{index:0,fighter_index:0,points_awarded:1}].map(JSON.stringify).join('\n'));
  await assert.rejects(v.scoreAwards(path.join(dir,'scores.jsonl')),/duplicate/);
}));
