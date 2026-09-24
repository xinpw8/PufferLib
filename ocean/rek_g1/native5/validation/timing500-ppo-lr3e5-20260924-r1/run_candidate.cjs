'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict'),cp=require('node:child_process');
const ORIGINAL='5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4';
const PARENT_SETTINGS={epochs:1,learning_rate:.00001,horizon:128,clip:.2,vf_clip:.2,vf_coef:0,entropy:.001,targets:'complete_mc_zero_baseline',reward:'received-score-delta-div5-half-life-5s.v1'};
const SETTINGS={...PARENT_SETTINGS,learning_rate:.00003};
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const pin=record=>{assert(path.isAbsolute(record.path));const bytes=fs.readFileSync(record.path);assert.equal(sha(bytes),record.sha256,record.path+' changed');return bytes;};
function validate(config,plan,replay){
  assert.equal(config.schema,'rek.timing500_ppo_lr3e5.v1');assert(path.isAbsolute(config.stage));
  assert.equal(plan.schema,'rek.scorecredit5s_onpolicy_run_plan.v1');
  assert.equal(plan.rounds,8);assert.equal(plan.minimum_completed_rounds,8);assert.equal(plan.rows,24010);
  assert.equal(plan.behavior.sha256,ORIGINAL);assert.equal(plan.observation_schema,'rek.native5.scaled_polar_xy.balance8_v1');
  assert.equal(plan.weight_migration,false);assert.equal(plan.resampled,false);
  assert.deepEqual(plan.settings,PARENT_SETTINGS);assert.deepEqual(config.settings,SETTINGS);assert.equal(config.expected_updates,192);
  assert.equal(config.checkpoint_selection,'fixed final epoch1; no development-selected checkpoint');
  assert.equal(replay.schema,'rek.authentic_behavior_replay.v5');assert.equal(replay.verification_passed,true);
  assert.equal(replay.teacher_was_recorded_worker,true);assert.equal(replay.matching_sampled_actions,plan.rows);
  assert.equal(replay.rows,plan.rows);assert.equal(replay.rounds,plan.rounds);assert.equal(replay.mismatches,0);
  assert.equal(replay.checkpoint_sha256,ORIGINAL);assert.equal(replay.dataset_sha256,plan.dataset.sha256);
  assert.equal(replay.identity_sha256,plan.identity.sha256);assert.equal(replay.worker_sha256,plan.worker.sha256);
  assert.equal(replay.native_object_sha256,plan.native_object.sha256);assert.equal(replay.feature_mask_sha256,plan.feature_mask.sha256);
  assert.equal(replay.observation_schema,plan.observation_schema);assert.equal(replay.replay_sha256,config.replay.sha256);
  assert.equal(replay.seed_per_new_worker,'recorded_per_sequence');assert.equal(replay.precision,'bf16');
  assert.equal(replay.terminal_race_requests_retained,true);assert.equal(replay.logprob_precision,'float32_native_sampler_reduction');
  return true;
}
function command(config,plan,mode){
  assert(['train','diagnose'].includes(mode));assert.deepEqual(config.settings,SETTINGS);
  return [plan.binaries.trainer.path,plan.dataset.path,config.replay.path,plan.behavior.path,plan.behavior.sha256,
    path.join(config.stage,mode==='train'?'train-one-epoch':'diagnose-zero-epoch','policy.bin'),
    mode==='train'?'1':'0','.00003','128','.2','.2','0','.001','--allow-distributional-bf16-batch',
    '--targets=complete-mc-zero-baseline','--observation-schema='+plan.observation_schema];
}
function load(file){
  const config=JSON.parse(fs.readFileSync(file)),plan=JSON.parse(pin(config.parent_run_plan)),replay=JSON.parse(pin(config.replay_receipt));
  assert.equal(sha(fs.readFileSync(path.join(__dirname,'parent-run-plan.json'))),config.parent_run_plan.sha256,'local parent plan changed');
  validate(config,plan,replay);
  for(const p of [config.replay,plan.selection,plan.dataset,plan.identity,plan.behavior,plan.worker,plan.native_object,plan.feature_mask,...Object.values(plan.binaries)])pin(p);
  return {config,plan,replay};
}
async function main(){
  const [mode,execution]=process.argv.slice(2);assert.equal(process.argv.length,4,'usage: node run_candidate.cjs train|diagnose --check|--run');
  assert(['--check','--run'].includes(execution));
  const configFile=path.join(__dirname,'experiment.json'),{config,plan}=load(configFile),argv=command(config,plan,mode);
  console.log(JSON.stringify({argv,executing:execution==='--run',parent_run_plan:config.parent_run_plan,settings:config.settings,prospective_checkpoint:'final epoch1',new_behavior_logprobs:false}));
  if(execution==='--check')return;
  const out=path.dirname(argv[5]);assert(!fs.existsSync(out),'output already exists');fs.mkdirSync(out);
  const write=(name,value)=>fs.writeFileSync(path.join(out,name),value,{flag:'wx',mode:0o600});
  write('command.json',JSON.stringify(argv,null,2)+'\n');write('experiment.json',fs.readFileSync(configFile));
  write('started.utc',new Date().toISOString()+'\n');
  const stdout=fs.openSync(path.join(out,'stdout.jsonl'),'wx'),stderr=fs.openSync(path.join(out,'stderr.txt'),'wx');
  const result=await new Promise((resolve,reject)=>{
    const child=cp.spawn('/usr/bin/time',['-f','full_native_process_wall_seconds=%e\nexit_code=%x','-o',path.join(out,'time.txt'),...argv],{stdio:['ignore',stdout,stderr]});
    child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));
  });
  fs.closeSync(stdout);fs.closeSync(stderr);write('exit-code.json',JSON.stringify(result)+'\n');write('finished.utc',new Date().toISOString()+'\n');
  if(result.code===0){
    const events=fs.readFileSync(path.join(out,'stdout.jsonl'),'utf8').trim().split(/\r?\n/).map(JSON.parse);
    const saved=events.findLast(x=>x.phase==='saved'),epoch=events.findLast(x=>x.phase==='epoch');
    assert(saved&&fs.existsSync(argv[5]));assert.equal(sha(fs.readFileSync(argv[5])),saved.sha256);
    if(mode==='train'){assert.equal(epoch.epoch,1);assert.equal(epoch.updates,config.expected_updates);}
    write('checkpoint-receipt.json',JSON.stringify({path:argv[5],sha256:saved.sha256,mode,epochs:mode==='train'?1:0,
      updates:epoch?.updates??0,parent_checkpoint_sha256:ORIGINAL,selection:config.checkpoint_selection,environment_stepping:false},null,2)+'\n');
  }
  console.log(JSON.stringify(result));process.exitCode=result.code??2;
}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
module.exports={ORIGINAL,PARENT_SETTINGS,SETTINGS,sha,validate,command,load};
