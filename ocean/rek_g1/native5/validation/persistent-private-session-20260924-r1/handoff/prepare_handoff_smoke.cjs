'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto'),assert=require('node:assert/strict');
assert.equal(os.hostname(),'spark-4ae3');
const stage='/home/spark-advantage/rek-training/persistent-round-handoff-20260924-r1';
const out=stage+'/live',old='/home/spark-advantage/rek-training/persistent-private-session-20260924-r1/live';
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const names=['campaign.cjs','live_transfer_run_masked.cjs','record_passive_defender.cjs','policy_handoff.cjs'];
const pins=process.argv.slice(2);assert.equal(pins.length,names.length);
for(let i=0;i<names.length;i++){assert.match(pins[i],/^[a-f0-9]{64}$/);assert.equal(sha(stage+'/'+names[i]),pins[i]);}
assert(!fs.existsSync(out),'Preserve existing evidence');
const template=JSON.parse(fs.readFileSync(old+'/configs/persistent-s1701.json'));
assert.equal(template.checkpoint_sha256,'f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84');
assert.equal(sha(template.worker[1]),template.checkpoint_sha256);assert.equal(sha(template.worker[5]),template.feature_mask_sha256);
assert.equal(template.relay.at(-1),'5a2edac6c586f1ea401d92e0086ebfc468dbc591e2bb11115aef056682280e7a');
assert(template.relay.at(-3).endsWith('/RekUiPipeClient-startup-20260924-r1.exe'));
assert.equal(sha('/home/spark-advantage/codexrook-runtime/rek-core-referee-20260924-r1/BepInEx/plugins/RekUiBridgeAgent.dll'),template.relay.at(-1));
const oldLedger=JSON.parse(fs.readFileSync(old+'/root-campaign/ledger.json'));
assert.equal(Object.values(oldLedger).flat().filter(r=>r.complete).length,3,'Wait for original smoke to finish naturally');
assert(!fs.existsSync(old+'/root-campaign/campaign.lock'),'Old controller still owns stage');
fs.mkdirSync(out,{mode:0o700});for(const name of ['configs','root-campaign'])fs.mkdirSync(out+'/'+name,{mode:0o700});
const copies=[
 ['root-campaign/campaign.cjs',stage+'/campaign.cjs'],
 ...names.slice(1).map(n=>[n,stage+'/'+n]),
 ...['progress.cjs','root-campaign/probe_state.cjs','root-campaign/clear_dead_prefix.sh','root-campaign/relaunch.sh'].map(n=>[n,old+'/'+n])
];
const artifacts=copies.map(([name,src])=>{fs.copyFileSync(src,out+'/'+name,fs.constants.COPYFILE_EXCL);return {name,source:src,sha256:sha(out+'/'+name)};});
const plan=[];
for(let i=0;i<4;i++){
 const seed=1801+i,label='handoff-s'+seed,cfg=structuredClone(template);
 cfg.worker[3]=String(seed);cfg.out=out+'/'+label+'/trial';
 for(const k of Object.keys(template).filter(k=>!['worker','out'].includes(k)))assert.deepEqual(cfg[k],template[k]);
 assert.deepEqual(cfg.worker.filter((_,j)=>j!==3),template.worker.filter((_,j)=>j!==3));
 fs.mkdirSync(out+'/'+label,{mode:0o700});const file=out+'/configs/'+label+'.json';
 fs.writeFileSync(file,JSON.stringify(cfg,null,2)+'\n',{flag:'wx',mode:0o600});
 plan.push({label,order:i,policy_rng_seed:seed,checkpoint_sha256:cfg.checkpoint_sha256,config_path:file,config_sha256:sha(file)});
}
const receipt={created_utc:new Date().toISOString(),scope:'Four-round runtime handoff smoke; no policy acceptance claim',runtime_only:true,rounds:4,
 intended_runtime_evidence:'Same process; native round1 to round2 and subsequent match start; each controlled round starts at zero score with at least117 seconds remaining; no media-triggered control delay',
 policy_unchanged:true,checkpoint_sha256:template.checkpoint_sha256,driver_sha256:pins[1],controller_sha256:pins[0],
 recorder_sha256:pins[2],handoff_sha256:pins[3],bridge_sha256:template.relay.at(-1),
 normal_round_process_restart:false,launcher_invoked_by_preparation:false,artifacts,plan,controller_started:false};
fs.writeFileSync(out+'/planned-rounds.json',JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o600});
console.log(JSON.stringify({stage:out,plan_sha256:sha(out+'/planned-rounds.json'),...Object.fromEntries(['controller','driver','recorder','handoff'].map((n,i)=>[n+'_sha256',pins[i]])),controller_started:false}));
