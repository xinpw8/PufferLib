'use strict';
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),assert=require('node:assert/strict');
const {sha}=require('./selection.cjs');
function command(plan,mode){
 assert.equal(plan.schema,'rek.scorecredit5s_onpolicy_run_plan.v1');assert(plan.rounds>=plan.minimum_completed_rounds);
 const replay=path.join(plan.stage,'score-delta-5s/behavior-replay-v5.bin');
 if(mode==='replay')return[plan.binaries.replay.path,plan.dataset.path,plan.behavior.path,plan.behavior.sha256,replay,plan.identity.path,plan.worker.path,'--observation-schema='+plan.observation_schema];
 assert(['train','diagnose'].includes(mode));assert.deepEqual(plan.settings,{epochs:1,learning_rate:.00001,horizon:128,clip:.2,vf_clip:.2,vf_coef:0,entropy:.001,targets:'complete_mc_zero_baseline',reward:'received-score-delta-div5-half-life-5s.v1'});
 return[plan.binaries.trainer.path,plan.dataset.path,replay,plan.behavior.path,plan.behavior.sha256,path.join(plan.stage,mode+'-score-delta/policy.bin'),mode==='train'?'1':'0','.00001','128','.2','.2','0','.001','--allow-distributional-bf16-batch','--targets=complete-mc-zero-baseline','--observation-schema='+plan.observation_schema];
}
async function main(){
 assert.equal(process.argv.length,5,'Usage: node run_native.cjs RUN_PLAN replay|diagnose|train --check|--run');assert(['--check','--run'].includes(process.argv[4]));
 const plan=JSON.parse(fs.readFileSync(process.argv[2])),mode=process.argv[3],argv=command(plan,mode);
 for(const record of [plan.selection,plan.dataset,plan.identity,plan.behavior,plan.worker,plan.native_object,plan.feature_mask,...Object.values(plan.binaries)])assert.equal(sha(fs.readFileSync(record.path)),record.sha256,record.path+' changed');
 console.log(JSON.stringify({argv,executing:process.argv[4]==='--run'}));if(process.argv[4]==='--check')return;
 if(mode!=='replay'){const file=path.join(plan.stage,'score-delta-5s/behavior-replay-v5.bin'),r=JSON.parse(fs.readFileSync(file+'.json'));assert.equal(r.verification_passed,true);assert.equal(r.matching_sampled_actions,plan.rows);assert.equal(r.mismatches,0);assert.equal(r.checkpoint_sha256,plan.behavior.sha256);assert.equal(r.dataset_sha256,plan.dataset.sha256);assert.equal(r.identity_sha256,plan.identity.sha256);assert.equal(r.teacher_was_recorded_worker,true);assert.equal(sha(fs.readFileSync(file)),r.replay_sha256);}
 const logs=path.join(plan.stage,mode+'-score-delta');assert(!fs.existsSync(logs));fs.mkdirSync(logs);
 fs.writeFileSync(path.join(logs,'command.json'),JSON.stringify(argv)+'\n',{flag:'wx'});fs.writeFileSync(path.join(logs,'started.utc'),new Date().toISOString()+'\n',{flag:'wx'});
 const stdout=fs.openSync(path.join(logs,'stdout.jsonl'),'wx'),stderr=fs.openSync(path.join(logs,'stderr.txt'),'wx');
 const result=await new Promise((resolve,reject)=>{const child=cp.spawn('/usr/bin/time',['-f','full_native_process_wall_seconds=%e\nexit_code=%x','-o',path.join(logs,'time.txt'),...argv],{stdio:['ignore',stdout,stderr]});child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});
 fs.closeSync(stdout);fs.closeSync(stderr);fs.writeFileSync(path.join(logs,'exit-code.json'),JSON.stringify(result)+'\n',{flag:'wx'});fs.writeFileSync(path.join(logs,'finished.utc'),new Date().toISOString()+'\n',{flag:'wx'});console.log(JSON.stringify(result));process.exitCode=result.code??2;
}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
module.exports={command};
