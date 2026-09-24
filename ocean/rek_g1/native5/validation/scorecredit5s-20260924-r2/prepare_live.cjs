'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const [checkpoint,checkpointSha,driver]=process.argv.slice(2);
const source='/home/spark-advantage/rek-training/balance8-live-20260924-r1';
const out='/home/spark-advantage/rek-training/scorecredit5s-live-20260924-r2';
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
assert.equal(os.hostname(),'spark-4ae3');
assert([checkpoint,driver].every(p=>typeof p==='string'&&path.isAbsolute(p)));
assert.equal(sha(checkpoint),checkpointSha);assert(!fs.existsSync(out),'Fresh cohort required');
const template=JSON.parse(fs.readFileSync(source+'/configs/balance8-s901.json'));
assert.equal(template.observation_schema,'rek.native5.scaled_polar_xy.balance8_v1');
assert.equal(sha(template.worker[0]),'52741aed67037073bff5ecb21af63864550cba93a316dfe59a41b8b50126d7b2');
assert.equal(sha(template.encoder[0]),'4c820e2e78260fd80918baf6d61979196c1694e5882f785655915ff0ae0869af');
assert.deepEqual(template.live_attack_gate.allowed_attacks,Array.from({length:17},(_,i)=>16+i));
for(const k of ['range_m','force_attack_range_m','cooldown_s'])assert(template.live_attack_gate[k]==null);
assert(template.relay.includes('BOX64_DYNAREC_STRONGMEM=2')&&template.relay.includes('BOX64_DYNAREC_WEAKBARRIER=0'));
for(const p of [out,out+'/configs',out+'/root-campaign'])fs.mkdirSync(p,{mode:0o700});
for(const name of ['record_passive_defender.cjs','progress.cjs','root-campaign/probe_state.cjs','root-campaign/relaunch.sh','root-campaign/clear_dead_prefix.sh','root-campaign/recycle_owned_client.sh'])fs.copyFileSync(source+'/'+name,out+'/'+name,fs.constants.COPYFILE_EXCL);
fs.copyFileSync(driver,out+'/live_transfer_run_masked.cjs',fs.constants.COPYFILE_EXCL);
const driverSha=sha(driver),controllerSource=source+'/root-campaign/campaign_terminal_retry_r1.cjs';
assert.equal(sha(controllerSource),'7629f4cd199d76afe77c82fc0b86f716d6065fbce986ba4b305ef801368b26b6');
let controller=fs.readFileSync(controllerSource,'utf8');
const oldDriver='0666e0567173fb389ff5cce6a50cc93bab12560fd5325418969691b54d5196a1';
assert(controller.includes(oldDriver));controller=controller.replaceAll(oldDriver,driverSha);
fs.writeFileSync(out+'/root-campaign/campaign.cjs',controller,{flag:'wx'});
const plan=[];
for(let i=0;i<20;i++){
 const seed=1001+i,label='credit5s-s'+seed,cfg=structuredClone(template);
 cfg.checkpoint_sha256=checkpointSha;cfg.worker[1]=checkpoint;cfg.worker[2]=checkpointSha;cfg.worker[3]=String(seed);
 cfg.out=out+'/'+label+'/trial';fs.mkdirSync(out+'/'+label);
 const config_path=out+'/configs/'+label+'.json';fs.writeFileSync(config_path,JSON.stringify(cfg,null,2)+'\n',{flag:'wx'});
 for(const key of Object.keys(template).filter(k=>!['checkpoint_sha256','worker','out'].includes(k)))assert.deepEqual(cfg[key],template[key]);
 assert.deepEqual(cfg.worker.slice(4),template.worker.slice(4));assert.equal(cfg.worker[0],template.worker[0]);
 plan.push({label,order:i,policy_rng_seed:seed,checkpoint_sha256:checkpointSha,config_path});
}
const receipt={created_utc:new Date().toISOString(),target_wins:18,rounds:20,stop_nonwins:3,
 observation_schema:template.observation_schema,checkpoint_sha256:checkpointSha,driver_sha256:driverSha,
 controller_sha256:sha(out+'/root-campaign/campaign.cjs'),worker_sha256:sha(template.worker[0]),encoder_sha256:sha(template.encoder[0]),
 hypothesis:'Behavior-matched25Hz authentic score-delta/5 reward with5second discount half-life; no terminal outcome/potential bonus.',
 runtime_change:'Read-only source warmup before control if selected driver provides it; shorter retry delay only after observed terminal. Existing freshness, watchdog, private-AI checks unchanged.',
 inherited_duration_projection:'dispatched_request_v4_duration, unchanged and estimated; not measured authoritative action busy state',
 plan,controller_started:false};
fs.writeFileSync(out+'/planned-rounds.json',JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({stage:out,checkpoint_sha256:checkpointSha,driver_sha256:driverSha,seeds:[1001,1020],controller_started:false}));
