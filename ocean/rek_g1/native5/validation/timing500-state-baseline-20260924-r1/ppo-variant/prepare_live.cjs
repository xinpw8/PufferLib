'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const source='/home/spark-advantage/rek-training/humanbc-timing500-live-20260924-r1';
const out='/home/spark-advantage/rek-training/timing500-state-baseline-live-20260924-r1';
const oldSha='5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4';
const parentSha='c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533';
const driverSha='b57654d54a76c80b38c9e44f59f53efb2502959a70dc568835ec5a8f6d238214';
const controllerSha='160299c466f6aeb8b0f9cebfd63132316f98625fe89af5f851a505472e44b151';
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function configuration(template,checkpoint,checkpointSha,index){
 assert(Number.isInteger(index)&&index>=0&&index<20);assert(/^[a-f0-9]{64}$/.test(checkpointSha)&&checkpointSha!==oldSha);
 const cfg=structuredClone(template),seed=1601+index,label='baseline-s'+seed;
 assert.equal(template.checkpoint_sha256,oldSha);
 cfg.checkpoint_sha256=checkpointSha;cfg.worker[1]=checkpoint;cfg.worker[2]=checkpointSha;cfg.worker[3]=String(seed);cfg.out=out+'/'+label+'/trial';
 for(const k of Object.keys(template).filter(k=>!['checkpoint_sha256','worker','out'].includes(k)))assert.deepEqual(cfg[k],template[k]);
 assert.equal(cfg.worker[0],template.worker[0]);assert.deepEqual(cfg.worker.slice(4),template.worker.slice(4));
 return {cfg,seed,label};
}
function main(){
 assert.equal(os.hostname(),'spark-4ae3');assert.equal(process.argv.length,4,'usage checkpoint checkpoint_sha256');
 const [checkpoint,checkpointSha]=process.argv.slice(2);assert(path.isAbsolute(checkpoint));assert.equal(sha(checkpoint),checkpointSha);assert.equal(fs.statSync(checkpoint).size,1836032);
 assert(!fs.existsSync(out),'fresh output required');assert.equal(sha(source+'/live_transfer_run_masked.cjs'),driverSha);assert.equal(sha(source+'/root-campaign/campaign.cjs'),controllerSha);
 const template=JSON.parse(fs.readFileSync(source+'/configs/timing500-s1201.json'));
 assert.equal(template.observation_schema,'rek.native5.scaled_polar_xy.balance8_v1');
 assert.equal(sha(template.worker[0]),'52741aed67037073bff5ecb21af63864550cba93a316dfe59a41b8b50126d7b2');
 assert.equal(sha(template.encoder[0]),'4c820e2e78260fd80918baf6d61979196c1694e5882f785655915ff0ae0869af');
 assert.equal(sha(template.worker[5]),template.feature_mask_sha256);
 assert.equal(template.feature_mask_sha256,'59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b');
 assert.equal(template.relay.at(-1),'11fcfa2bea541f2829553c3833c301717c0182a59eebb0020f5f7b8cb066ca3d');
 assert.deepEqual(template.live_attack_gate.allowed_attacks,Array.from({length:17},(_,i)=>16+i));
 for(const k of ['range_m','force_attack_range_m','cooldown_s'])assert(template.live_attack_gate[k]==null);
 fs.mkdirSync(out,{mode:0o700});for(const name of ['configs','root-campaign'])fs.mkdirSync(out+'/'+name,{mode:0o700});
 const copies=['record_passive_defender.cjs','progress.cjs','live_transfer_run_masked.cjs','root-campaign/campaign.cjs','root-campaign/probe_state.cjs','root-campaign/relaunch.sh','root-campaign/clear_dead_prefix.sh','root-campaign/recycle_owned_client.sh'];
 const artifacts=copies.map(name=>{fs.copyFileSync(source+'/'+name,out+'/'+name,fs.constants.COPYFILE_EXCL);assert.equal(sha(out+'/'+name),sha(source+'/'+name));return {name,sha256:sha(out+'/'+name)};});
 for(const name of ['capture_resources.cjs','snapshot_controls.cjs']){fs.copyFileSync(__dirname+'/'+name,out+'/'+name,fs.constants.COPYFILE_EXCL);assert.equal(sha(out+'/'+name),sha(__dirname+'/'+name));artifacts.push({name,sha256:sha(out+'/'+name),passive_only:true});}
 const plan=Array.from({length:20},(_,i)=>{const {cfg,seed,label}=configuration(template,checkpoint,checkpointSha,i),file=out+'/configs/'+label+'.json';fs.mkdirSync(out+'/'+label,{mode:0o700});fs.writeFileSync(file,JSON.stringify(cfg,null,2)+'\n',{flag:'wx',mode:0o600});return {label,order:i,policy_rng_seed:seed,checkpoint_sha256:checkpointSha,config_path:file,config_sha256:sha(file)};});
 const receipt={created_utc:new Date().toISOString(),target_wins:18,rounds:20,stop_nonwins:3,checkpoint_sha256:checkpointSha,parent_checkpoint_sha256:parentSha,driver_sha256:driverSha,controller_sha256:controllerSha,observation_schema:template.observation_schema,feature_mask_sha256:template.feature_mask_sha256,bridge_sha256:template.relay.at(-1),source_stage:source,
  hypothesis:'Baseline-only sibling of the f147 zero-baseline update: one-epoch native PPO from C2 on the same five closed C2 rounds and immutable behavior replay, LR3e-5, entropy.001, VF0. Advantages are unchanged complete-MC returns minus the pinned leave-one-episode-out pre-action state prediction. Actual score-delta/5, QPC five-second half-life, runtime, encoder, masks and unforced sampled actions are unchanged. Final epoch1 checkpoint is fixed prospectively. New RNG seeds do not control server randomness. Lower heldout return error is not proof of policy gain.',artifacts,plan,controller_started:false};
 fs.writeFileSync(out+'/planned-rounds.json',JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o600});console.log(JSON.stringify({stage:out,checkpoint_sha256:checkpointSha,planned_rounds_sha256:sha(out+'/planned-rounds.json'),controller_started:false}));
}
if(require.main===module)main();module.exports={configuration,oldSha,parentSha};
