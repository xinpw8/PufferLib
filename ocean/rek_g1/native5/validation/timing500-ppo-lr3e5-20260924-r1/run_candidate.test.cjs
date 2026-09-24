'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {ORIGINAL,PARENT_SETTINGS,SETTINGS,sha,validate,command}=require('./run_candidate.cjs');
const config=JSON.parse(fs.readFileSync(path.join(__dirname,'experiment.json'))),plan=JSON.parse(fs.readFileSync(path.join(__dirname,'parent-run-plan.json')));
function replay(){return {schema:'rek.authentic_behavior_replay.v5',verification_passed:true,teacher_was_recorded_worker:true,
  matching_sampled_actions:24010,rows:24010,rounds:8,mismatches:0,checkpoint_sha256:ORIGINAL,dataset_sha256:plan.dataset.sha256,
  identity_sha256:plan.identity.sha256,worker_sha256:plan.worker.sha256,native_object_sha256:plan.native_object.sha256,
  feature_mask_sha256:plan.feature_mask.sha256,observation_schema:plan.observation_schema,replay_sha256:config.replay.sha256,
  seed_per_new_worker:'recorded_per_sequence',precision:'bf16',terminal_race_requests_retained:true,logprob_precision:'float32_native_sampler_reduction'};}
test('frozen parent plan is exact; the sole hyperparameter change is learning rate',()=>{
  assert.equal(sha(fs.readFileSync(path.join(__dirname,'parent-run-plan.json'))),config.parent_run_plan.sha256);
  assert.equal(validate(config,plan,replay()),true);
  assert.deepEqual(Object.keys(SETTINGS).filter(k=>SETTINGS[k]!==PARENT_SETTINGS[k]),['learning_rate']);
  assert.equal(SETTINGS.learning_rate/PARENT_SETTINGS.learning_rate,3);
});
test('native train argv retains actual checkpoint, replay, row contract and one epoch',()=>{
  const a=command(config,plan,'train');assert.equal(a[0],plan.binaries.trainer.path);assert.equal(a[1],plan.dataset.path);
  assert.equal(a[2],config.replay.path);assert.equal(a[3],plan.behavior.path);assert.equal(a[4],ORIGINAL);
  assert.equal(a[5],path.join(config.stage,'train-one-epoch','policy.bin'));
  assert.deepEqual(a.slice(6),['1','.00003','128','.2','.2','0','.001','--allow-distributional-bf16-batch','--targets=complete-mc-zero-baseline','--observation-schema=rek.native5.scaled_polar_xy.balance8_v1']);
});
test('mismatched behavior/sequence/replay, masks or optimizer settings are rejected',()=>{
  for(const alter of [r=>r.mismatches=1,r=>r.matching_sampled_actions--,r=>r.teacher_was_recorded_worker=false,
    r=>r.checkpoint_sha256='c'.repeat(64),r=>r.dataset_sha256='c'.repeat(64),r=>r.feature_mask_sha256='c'.repeat(64),
    r=>r.replay_sha256='c'.repeat(64),r=>r.terminal_race_requests_retained=false]){
    const r=replay();alter(r);assert.throws(()=>validate(config,plan,r));
  }
  for(const field of ['epochs','horizon','clip','vf_coef','entropy']){const c=structuredClone(config);c.settings[field]++;assert.throws(()=>validate(c,plan,replay()));}
});
test('optional diagnosis performs zero optimizer epochs and writes a distinct output',()=>{
  const a=command(config,plan,'train'),b=command(config,plan,'diagnose');
  assert.deepEqual(a.map((v,i)=>v===b[i]?null:i).filter(x=>x!==null),[5,6]);assert.equal(b[6],'0');
  assert.equal(b[5],path.join(config.stage,'diagnose-zero-epoch','policy.bin'));assert.throws(()=>command(config,plan,'sweep'));
});
