'use strict';
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto'),os=require('node:os'),assert=require('node:assert/strict');
const p=require('./plan.cjs');
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const put=(file,obj)=>fs.writeFileSync(file,JSON.stringify(obj,null,2)+'\n',{flag:'wx',mode:0o600});
const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!k.startsWith('REK_')));Object.assign(env,p.environment);
const event=(type,data={})=>{const row={utc:new Date().toISOString(),event:type,...data};fs.appendFileSync(p.stage+'/events.jsonl',JSON.stringify(row)+'\n',{mode:0o600});console.log(JSON.stringify(row));};
function memory(){const text=fs.readFileSync('/proc/meminfo','utf8');const availableKiB=Number(text.match(/^MemAvailable:\s+(\d+)/m)?.[1]);assert(availableKiB>=64*1024*1024,'Less than 64 GiB available; do not start another GPU run');return{availableGiB:availableKiB/1024**2};}
function checkedNative(dir,exe,args,cwd,seconds=180){
 fs.mkdirSync(dir,{recursive:true,mode:0o700});assert(!fs.existsSync(dir+'/command.json'));
 put(dir+'/command.json',{executable:exe,args,cwd,environment:p.environment,timeoutSeconds:seconds,host:os.hostname(),memory:memory(),startedUtc:new Date().toISOString()});
 const out=fs.openSync(dir+'/stdout.txt','wx',0o600),err=fs.openSync(dir+'/stderr.txt','wx',0o600);const start=performance.now();let result;
 try{result=cp.spawnSync('/usr/bin/time',['-v','-o',dir+'/process-timing.txt','timeout','--signal=TERM','--kill-after=10s',String(seconds),exe,...args],{cwd,env,stdio:['ignore',out,err],timeout:(seconds+20)*1000});}finally{fs.closeSync(out);fs.closeSync(err);}
 const receipt={exitCode:result.status,signal:result.signal,error:result.error?.message??null,wallSeconds:(performance.now()-start)/1000,finishedUtc:new Date().toISOString()};put(dir+'/exit.json',receipt);fs.writeFileSync(dir+'/exit-code.txt',String(result.status)+'\n',{flag:'wx'});return receipt;
}
function checkpoint(dir,id,steps){return dir+'/checkpoints/rek_native5/'+id+'/'+String(steps).padStart(16,'0')+'.bin';}
function train(arm,budget,seed,label){
 const dir=p.stage+'/'+label+'/'+arm.id;const id=arm.id+'-s'+seed;const args=['train','--headless',
 '--vec.total_agents=512','--train.minibatch_size=8192','--train.replay_ratio=1','--train.total_timesteps='+budget,'--train.horizon='+arm.horizon,
 '--train.learning_rate='+arm.lr,'--train.ent_coef='+arm.entropy,'--train.gamma='+arm.gamma,'--train.gae_lambda='+arm.lambda,'--train.clip_coef='+arm.clip,'--train.vf_coef='+arm.vf,
 '--train.vf_clip_coef=.2','--train.max_grad_norm=.5','--train.momentum=.95','--train.anneal_lr=1','--train.min_lr_ratio=0','--train.anneal_ent_coef=0',
 '--policy.hidden_size=256','--policy.num_layers=2','--base.seed='+seed,'--base.run_id='+id,'--base.checkpoint_interval=64','--base.reset_every_horizon=0',
 '--base.load_model_path='+p.warm,'--base.log_dir='+dir+'/logs','--base.checkpoint_dir='+dir+'/checkpoints','--sweep.metric=score','--sweep.downsample=1',
 '--env.seed=419','--env.round_seconds=120','--env.opponent_checkpoint=None','--env.model_path='+p.assets+'/model.two_fighter_arena.xml','--env.physics_export_path='+p.physics,'--env.assets_path='+p.assets,'--env.motion_features_path='+p.features];
 event('training_start',{id:arm.id,label,budget,seed,hparams:arm});const receipt=checkedNative(dir,p.trainer,args,p.previous+'/build');
 let metrics=null,model=null,modelSha=null,error=null;
 try{assert.equal(receipt.exitCode,0);model=checkpoint(dir,id,budget);assert.equal(sha(checkpoint(dir,id,0)),p.warmSha,'Warm-start weights changed before first update');const bytes=fs.readFileSync(model);assert.equal(bytes.length,fs.statSync(p.warm).size);for(let i=0;i<bytes.length;i+=4)assert(Number.isFinite(bytes.readFloatLE(i)),'Nonfinite checkpoint');modelSha=sha(model);metrics=require('./metrics.cjs').summarizeTraining(dir,budget);}catch(e){error=e.message;}
 const report={arm,seed,budget,receipt,model,modelSha,metrics,error};put(dir+'/training-result.json',report);event('training_end',{id:arm.id,label,budget,exit:receipt.exitCode,modelSha,metrics,error});return report;
}
function evaluate(label,model,modelSha,seeds,arenas=64){
 const dir=p.stage+'/evaluations/'+label;assert(!fs.existsSync(dir));fs.mkdirSync(dir,{recursive:true,mode:0o700});const files=[];
 for(const seed of seeds){const target=dir+'/seed-'+seed;const records=target+'/rounds.private.jsonl';event('evaluation_start',{label,seed,arenas,modelSha});
 const result=checkedNative(target,p.stage+'/eval-build/fast-policy-eval',[p.stage+'/runtime.json',model,modelSha,String(arenas),'1',String(seed),'sampled','bf16',records],p.previous+'/build');assert.equal(result.exitCode,0,'Native evaluation failed');
 const rows=fs.readFileSync(target+'/stdout.txt','utf8').trim().split('\n').flatMap(x=>{try{return[JSON.parse(x)]}catch{return[]}});assert(rows.some(x=>x.event==='frozen_policy_evaluation'&&x.failure_bits===0&&x.checkpoint_sha256===modelSha));files.push(records);}
 const combined=dir+'/rounds.private.jsonl';fs.writeFileSync(combined,files.map(f=>fs.readFileSync(f,'utf8')).join(''),{flag:'wx',mode:0o600});
 const metrics=require('./metrics.cjs').summarizeEval(combined,modelSha);assert.equal(metrics.n,arenas*seeds.length);put(dir+'/result.json',{label,model,modelSha,seeds,arenas,metrics});event('evaluation_end',{label,modelSha,metrics});return metrics;
}
function prepare(){
 assert.equal(sha(p.trainer),p.trainerSha);assert.equal(sha(p.warm),p.warmSha);assert.equal(sha(p.previous+'/build/fast_runtime.o'),'4197999792ea8951b4c9234153cabad39a32d7b71d25e63ccf6e6039a53f8e50');
 assert(!fs.existsSync(p.stage+'/plan.json'));fs.copyFileSync('/home/spark-advantage/rek-training/normalized-sweep-20260921-r1/scripts/sweep-runtime.json',p.stage+'/runtime.json',fs.constants.COPYFILE_EXCL);
 put(p.stage+'/plan.json',{createdUtc:new Date().toISOString(),...p,initialModelSha:p.warmSha,trainerSha:p.trainerSha,evalSelection:'Side zero only. Side-one results are saved but excluded due to asymmetric calibration.'});
 put(p.stage+'/pins.json',[p.trainer,p.warm,p.previous+'/build/fast_runtime.o',p.stage+'/runtime.json',__filename,p.stage+'/plan.cjs'].map(file=>({file,sha256:sha(file)})));event('prepared',{arms:p.arms.length,...memory()});
}
function screen(){
 const baseline=evaluate('unchanged-baseline-screen',p.warm,p.warmSha,p.screenSeeds);const results=[];
 for(const arm of p.arms){const row=train(arm,p.screenSteps,73,'screen');if(!row.error){row.evaluation=evaluate('screen-'+arm.id,row.model,row.modelSha,p.screenSeeds);row.comparison=require('./metrics.cjs').compareObjectives(row.evaluation,baseline);}results.push(row);}
 const successful=results.filter(x=>!x.error);const guarded=successful.filter(x=>x.comparison.scoreImprovedWithGuards);
 const rank=(a,b)=>b.evaluation.meanOwnPoints-a.evaluation.meanOwnPoints||b.evaluation.winRate-a.evaluation.winRate||b.evaluation.meanMargin-a.evaluation.meanMargin;
 guarded.sort(rank);const explored=successful.slice().sort(rank);
 const selected=(guarded.length?guarded:explored).slice(0,4).map(x=>x.arm.id);
 put(p.stage+'/screen-results.json',{baseline,results,guardedRank:guarded.map(x=>x.arm.id),selected,selectionPassedGuards:guarded.length>0,selectedAreProvisional:true,matchWins:null});event('screen_complete',{selected,selectionPassedGuards:guarded.length>0});
}
function confirm(){
 const screen=JSON.parse(fs.readFileSync(p.stage+'/screen-results.json'));const baseline=evaluate('unchanged-baseline-confirm',p.warm,p.warmSha,p.confirmationSeeds,128);const results=[];
 for(const id of ['control',...screen.selected.filter(x=>x!=='control')])for(const seed of [73,947]){const arm=p.arms.find(x=>x.id===id),row=train(arm,p.promotionSteps,seed,'confirm-s'+seed);if(!row.error){row.evaluation=evaluate('confirm-'+id+'-s'+seed,row.model,row.modelSha,p.confirmationSeeds,128);row.comparison=require('./metrics.cjs').compareObjectives(row.evaluation,baseline);}results.push(row);}
 put(p.stage+'/confirmation-results.json',{baseline,results,authenticMatchWins:null,livePromotion:false,limitation:'Frozen simulator round evaluation only; new seeds, same candidate environment. No claim of authentic REK transfer.'});event('confirmation_complete',{runs:results.length});
}
function main(){assert.equal(os.hostname(),'spark-4ae3');const mode=process.argv[2];assert(['prepare','smoke','screen','confirm'].includes(mode));assert.equal(sha(p.trainer),p.trainerSha);assert.equal(sha(p.warm),p.warmSha);
 if(mode==='prepare')return prepare();if(mode==='smoke'){const row=train(p.control,1048576,73,'smoke');assert(!row.error,row.error);evaluate('smoke-baseline',p.warm,p.warmSha,[10001],16);return;}
 if(mode==='screen')return screen();return confirm();}
if(require.main===module){try{main()}catch(e){console.error(e.stack);process.exitCode=1;}}
module.exports={checkpoint};
