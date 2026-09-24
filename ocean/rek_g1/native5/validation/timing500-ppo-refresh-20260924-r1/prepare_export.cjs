'use strict';
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),assert=require('node:assert/strict');
const {loadSelection,pinArtifacts,completedSummary,sha}=require('./selection.cjs');
const {exportV3}=require('./source/authentic_trajectory_v3.cjs');
const {derive}=require('./derive_score_delta.cjs');
const publish=(p,x)=>fs.writeFileSync(p,JSON.stringify(x,null,2)+'\n',{flag:'wx'});
async function main(){
 assert.equal(process.argv.length,4,'Usage: node prepare_export.cjs SELECTION_JSON --check|--run');assert(['--check','--run'].includes(process.argv[3]));
 const file=path.resolve(process.argv[2]),c=loadSelection(file);pinArtifacts(c);require('./selection.cjs').loadNativeSelection(c);
 for(const r of c.rounds){assert.equal(fs.readFileSync(path.join(r.directory,'wrapper.exit-code.txt'),'utf8').trim(),'0');completedSummary(JSON.parse(fs.readFileSync(path.join(r.directory,'trial/summary.json'))),c);}
 if(process.argv[3]==='--check'){console.log(JSON.stringify({selected_completed_rounds:c.rounds.length,minimum_completed_rounds:c.minimum_completed_rounds,checkpoint_sha256:c.checkpoint.sha256,no_writes:true,no_gpu:true}));return;}
 for(const name of ['evidence','export','score-delta-5s','selection.frozen.json','export-config.json','run-plan.json'])assert(!fs.existsSync(path.join(c.stage,name)),name+' already exists');
 const frozen=path.join(c.stage,'selection.frozen.json');publish(frozen,c);
 const sidecars=cp.spawnSync(process.execPath,[path.join(__dirname,'prepare_strict_sidecars.cjs'),frozen],{encoding:'utf8',maxBuffer:16*1024*1024,env:{...process.env,CUDA_VISIBLE_DEVICES:''}});
 fs.writeFileSync(path.join(c.stage,'sidecars.stdout.jsonl'),sidecars.stdout,{flag:'wx'});fs.writeFileSync(path.join(c.stage,'sidecars.stderr.txt'),sidecars.stderr,{flag:'wx'});assert.equal(sidecars.status,0,sidecars.stderr);
 const ec={schema:'rek.authentic_export_config.v3',observation_schema:c.observation_schema,gamma_per_20ms:0.9998844821426083,lambda_per_20ms:0.9978673240629938,
  worker:c.worker,checkpoint:c.checkpoint,native_object:c.native_object,feature_mask:c.feature_mask,
  rounds:c.rounds.map(r=>({id:r.id,seed:r.seed,evidence:path.join(c.stage,'evidence',r.id),config:r.config}))};
 const exportConfig=path.join(c.stage,'export-config.json');publish(exportConfig,ec);const baseline=await exportV3(exportConfig,path.join(c.stage,'export'));const scored=derive(c.stage,c);
 const nativeRoot='/home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2/build-score-delta';
 const binaries={replay:{path:nativeRoot+'/replay-authentic-behavior',sha256:'fa622e8a238d5b44a1fbb839c41b4750eb80d328795eeb8d992da5771b6c9927'},trainer:{path:nativeRoot+'/authentic-ppo',sha256:'741035dad9303be7636c77be97ec44584011c95987170e86008d5515020b30e8'}};
 for(const b of Object.values(binaries))assert.equal(sha(fs.readFileSync(b.path)),b.sha256);
 const plan={schema:'rek.scorecredit5s_onpolicy_run_plan.v1',stage:c.stage,selection:{path:frozen,sha256:sha(fs.readFileSync(frozen))},minimum_completed_rounds:c.minimum_completed_rounds,rounds:c.rounds.length,rows:baseline.rows,behavior:c.checkpoint,worker:c.worker,native_object:c.native_object,feature_mask:c.feature_mask,observation_schema:c.observation_schema,binaries,
  dataset:{path:path.join(c.stage,'score-delta-5s/authentic-score-delta-v5.bin'),sha256:scored.dataset_sha256},identity:{path:path.join(c.stage,'score-delta-5s/behavior-identity.json'),sha256:scored.identity_sha256},
  settings:{epochs:1,learning_rate:.00003,horizon:128,clip:.2,vf_clip:.2,vf_coef:0,entropy:.001,targets:'complete_mc_zero_baseline',reward:'received-score-delta-div5-half-life-5s.v1'},checkpoint_selection:'fixed final epoch1; no development-selected checkpoint',original_actor_logits_recorded:false,replay_required:true,weight_migration:false,resampled:false,gpu_used:false};
 publish(path.join(c.stage,'run-plan.json'),plan);console.log(JSON.stringify(plan));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
