'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const source='/home/spark-advantage/rek-training/human-attackbc-live-20260924-r1';
const out='/home/spark-advantage/rek-training/humanbc-timing500-live-20260924-r1';
const oldBridge='ea8511a87b9e456547f13ffdb2f9af5e941b8b04f97bf40419d38e1c9b31ff3e';
const newBridge='11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d';
const checkpoint='/home/spark-advantage/rek-training/scorecredit-human-attackbc-20260924-r1/train-five-epochs/policy.bin';
const checkpointSha='5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4';
const driverSha='b57654d54a76c80b38c9e44f59f53efb2502959a70dc568835ec5a8f6d238214';
const controllerSha='160299c466f6aeb8b0f9cebfd63132316f98625fe89af5f851a505472e44b151';
const allonesSha='59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b';
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function configuration(template,index){
 assert(Number.isInteger(index)&&index>=0&&index<20);const seed=1201+index,label='timing500-s'+seed,cfg=structuredClone(template);
 cfg.checkpoint_sha256=checkpointSha;cfg.worker[1]=checkpoint;cfg.worker[2]=checkpointSha;cfg.worker[3]=String(seed);cfg.out=out+'/'+label+'/trial';
 assert.equal(template.relay.filter(x=>x===oldBridge).length,1);cfg.relay=template.relay.map(x=>x===oldBridge?newBridge:x);
 for(const key of Object.keys(template).filter(k=>!['checkpoint_sha256','worker','out','relay'].includes(k)))assert.deepEqual(cfg[key],template[key]);
 assert.deepEqual(cfg.relay.map(x=>x===newBridge?oldBridge:x),template.relay);
 assert.deepEqual(cfg.worker.slice(4),template.worker.slice(4));assert.equal(cfg.worker[0],template.worker[0]);
 return {seed,label,cfg};
}
function main(){
 assert.equal(os.hostname(),'spark-4ae3');assert.equal(process.argv.length,2);assert(!fs.existsSync(out+'/planned-rounds.json'),'Fresh cohort required');
 assert.equal(sha(checkpoint),checkpointSha);assert.equal(sha(source+'/live_transfer_run_masked.cjs'),driverSha);assert.equal(sha(source+'/root-campaign/campaign.cjs'),controllerSha);
 const template=JSON.parse(fs.readFileSync(source+'/configs/humanbc-s1101.json'));
 assert.equal(template.observation_schema,'rek.native5.scaled_polar_xy.balance8_v1');assert.equal(template.feature_mask_sha256,allonesSha);assert.equal(sha(template.worker[5]),allonesSha);
 assert.equal(sha(template.worker[0]),'52741aed67037073bff5ecb21af63864550cba93a316dfe59a41b8b50126d7b2');assert.equal(sha(template.encoder[0]),'4c820e2e78260fd80918baf6d61979196c1694e5882f785655915ff0ae0869af');
 assert.deepEqual(template.live_attack_gate.allowed_attacks,Array.from({length:17},(_,i)=>16+i));for(const k of ['range_m','force_attack_range_m','cooldown_s'])assert(template.live_attack_gate[k]==null);
 assert(template.relay.includes('BOX64_DYNAREC_STRONGMEM=2')&&template.relay.includes('BOX64_DYNAREC_WEAKBARRIER=0'));
 fs.mkdirSync(out,{mode:0o700,recursive:true});for(const p of [out+'/configs',out+'/root-campaign']){assert(!fs.existsSync(p));fs.mkdirSync(p,{mode:0o700});}
 const copies=['record_passive_defender.cjs','progress.cjs','live_transfer_run_masked.cjs','root-campaign/campaign.cjs','root-campaign/probe_state.cjs','root-campaign/relaunch.sh','root-campaign/clear_dead_prefix.sh','root-campaign/recycle_owned_client.sh'];
 const artifacts=copies.map(name=>{const before=sha(source+'/'+name);
  if(name==='root-campaign/relaunch.sh'){
   const text=fs.readFileSync(source+'/'+name,'utf8');assert.equal(text.split(oldBridge).length,2);
   fs.writeFileSync(out+'/'+name,text.replace(oldBridge,newBridge),{flag:'wx',mode:0o600});
  }else{fs.copyFileSync(source+'/'+name,out+'/'+name,fs.constants.COPYFILE_EXCL);assert.equal(sha(out+'/'+name),before);}
  return {name,sha256:sha(out+'/'+name),source_sha256:before,byte_identical:sha(out+'/'+name)===before};});
 const plan=[];
 for(let i=0;i<20;i++){
  const {seed,label,cfg}=configuration(template,i);fs.mkdirSync(out+'/'+label,{mode:0o700});const config_path=out+'/configs/'+label+'.json';
  fs.writeFileSync(config_path,JSON.stringify(cfg,null,2)+'\n',{flag:'wx',mode:0o600});plan.push({label,order:i,policy_rng_seed:seed,checkpoint_sha256:checkpointSha,config_path,config_sha256:sha(config_path)});
 }
 const receipt={created_utc:new Date().toISOString(),target_wins:18,rounds:20,stop_nonwins:3,observation_schema:template.observation_schema,checkpoint_sha256:checkpointSha,driver_sha256:driverSha,controller_sha256:controllerSha,
  worker_sha256:sha(template.worker[0]),encoder_sha256:sha(template.encoder[0]),feature_mask_sha256:allonesSha,source_stage:source,source_config_sha256:sha(source+'/configs/humanbc-s1101.json'),
  hypothesis:'Keep attack-BC checkpoint5b19 unchanged and test separately versioned500ms action freshness/750ms watchdog to tolerate measured next-Unity-frame dispatch jitter. No fighting improvement inferred from runtime completion.',
  runtime_change:'Only rebuilt bridge timing contract, hello advertisement, relay expected plugin hash and launch plugin hash pin. Driver, controller, recorder, encoder, native worker, feature mask, control-scope guards, startup grace1s and host inference deadline200ms unchanged.',
  bridge_sha256:newBridge,prior_bridge_sha256:oldBridge,action_freshness_ms:500,action_watchdog_ms:750,startup_grace_ms:1000,
  timing_risk:'Held commands can persist longer; watchdog is checked on Unity Update and cannot guarantee wall-clock neutralization during a stalled frame. Spark isolated privateAI only; no latency-parity claim.',
  inherited_duration_projection:'dispatched_request_v4_duration, unchanged and estimated; not authoritative action busy state',artifacts,plan,controller_started:false};
 fs.writeFileSync(out+'/planned-rounds.json',JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o600});
 console.log(JSON.stringify({stage:out,checkpoint_sha256:checkpointSha,bridge_sha256:newBridge,driver_sha256:driverSha,controller_sha256:controllerSha,seeds:[1201,1220],target_wins:18,stop_nonwins:3,planned_rounds_sha256:sha(out+'/planned-rounds.json'),controller_started:false}));
}
if(require.main===module)main();
module.exports={configuration,checkpointSha,driverSha,controllerSha,allonesSha,oldBridge,newBridge};
