'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const test=require('node:test'),assert=require('node:assert/strict');
const current=require('./prepare_live.cjs');
const priorDir=path.resolve(__dirname,'../fixtures/fresh-preparer');
const prior=require(path.join(priorDir,'prepare_live.cjs'));
const sha=text=>crypto.createHash('sha256').update(text).digest('hex');
const template={checkpoint_sha256:current.oldSha,worker:['worker','old',current.oldSha,'1201','sampled','allones','schema'],out:'old',encoder:['encoder','schema'],relay:['relay','samebridge'],live_attack_gate:{allowed_attacks:Array.from({length:17},(_,i)=>i+16),range_m:null},feature_mask_sha256:'unchanged',round_seconds:120};
test('all20 configuration changes are exactly checkpoint, RNG seed and output',()=>{
 const before=structuredClone(template);
 for(let i=0;i<20;i++){
  const fresh=prior.configuration(template,'/explicit/new/policy.bin','a'.repeat(64),i);
  const baseline=current.configuration(template,'/explicit/new/policy.bin','a'.repeat(64),i);
  assert.equal(baseline.seed,1601+i);assert.equal(baseline.label,'baseline-s'+(1601+i));
  assert.equal(baseline.cfg.worker[1],'/explicit/new/policy.bin');assert.equal(baseline.cfg.worker[2],'a'.repeat(64));
  assert.equal(baseline.cfg.out,'/home/spark-advantage/rek-training/timing500-state-baseline-live-20260924-r1/'+baseline.label+'/trial');
  const normalized=structuredClone(baseline.cfg);normalized.worker[3]=fresh.cfg.worker[3];normalized.out=fresh.cfg.out;
  assert.deepEqual(normalized,fresh.cfg);assert.deepEqual(template,before);
 }
});
test('fixed C2 parent and invalid arguments retain original rejection behavior',()=>{
 assert.equal(current.parentSha,'c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533');
 assert.equal(current.parentSha,prior.parentSha);assert.equal(current.oldSha,prior.oldSha);
 for(const index of [-1,20,.5,NaN])assert.throws(()=>current.configuration(template,'/new','a'.repeat(64),index));
 for(const checkpointSha of [current.oldSha,'invalid','A'.repeat(64)])assert.throws(()=>current.configuration(template,'/new',checkpointSha,0));
 assert.throws(()=>current.configuration({...template,checkpoint_sha256:'b'.repeat(64)},'/new','a'.repeat(64),0));
});
test('live preparer changes only cohort scope, hypothesis and passive helper copying',()=>{
 const old=fs.readFileSync(path.join(priorDir,'prepare_live.cjs'),'utf8');
 assert.equal(sha(old),'3fe3dfe56af59593737fb4ee57966d9a3d15293bcd843901fca9bba5648839b2');
 let edited=fs.readFileSync(path.join(__dirname,'prepare_live.cjs'),'utf8');
 edited=edited.replaceAll('timing500-state-baseline-live-20260924-r1','timing500-ppo-refresh-live-20260924-r1').replace("seed=1601+index,label='baseline-s'","seed=1501+index,label='refresh-s'");
 edited=edited.split('\n').filter(line=>!line.startsWith(" for(const name of ['capture_resources.cjs','snapshot_controls.cjs'])")).join('\n');
 const originalHypothesis=old.split('\n').find(line=>line.startsWith('  hypothesis:'));
 edited=edited.split('\n').map(line=>line.startsWith('  hypothesis:')?originalHypothesis:line).join('\n');
 assert.equal(edited,old);
});
test('passive helpers are exact existing helpers apart from stage and label scope',()=>{
 for(const name of ['capture_resources.cjs','snapshot_controls.cjs']){
  const priorText=fs.readFileSync(path.join(priorDir,name),'utf8');
  const freshText=fs.readFileSync(path.join(__dirname,name),'utf8').replaceAll('timing500-state-baseline-live-20260924-r1','timing500-ppo-refresh-live-20260924-r1').replaceAll('/baseline-s','/refresh-s');
  assert.equal(freshText,priorText);
 }
});
