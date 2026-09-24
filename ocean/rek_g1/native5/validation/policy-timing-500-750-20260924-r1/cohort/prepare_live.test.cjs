'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {configuration,checkpointSha,oldBridge,newBridge}=require('./prepare_live.cjs');
const template={checkpoint_sha256:checkpointSha,worker:['worker','cp',checkpointSha,'1101','sampled','allones','--schema'],out:'old',relay:['docker','exec','relay','policy-relay',oldBridge],encoder:['encoder'],live_attack_gate:{allowed_attacks:Array.from({length:17},(_,i)=>i+16),range_m:null},enter_private:true};
test('same checkpoint/settings; only prospective seed/output and bridge expected identity change',()=>{
 const before=JSON.stringify(template);
 for(let i=0;i<20;i++){
  const {seed,label,cfg}=configuration(template,i);assert.equal(seed,1201+i);assert.equal(label,'timing500-s'+seed);assert.equal(cfg.checkpoint_sha256,checkpointSha);assert.equal(cfg.relay.at(-1),newBridge);
  assert.deepEqual(cfg.worker.slice(4),template.worker.slice(4));assert.deepEqual(cfg.encoder,template.encoder);assert.deepEqual(cfg.live_attack_gate,template.live_attack_gate);assert.equal(cfg.enter_private,true);
 }
 assert.equal(JSON.stringify(template),before);
});
test('missing/duplicate old bridge identity and invalid cohort indices rejected',()=>{
 for(const relay of [[],['relay',oldBridge,oldBridge],['relay',newBridge]])assert.throws(()=>configuration({...template,relay},0));
 for(const index of [-1,20,.5,NaN])assert.throws(()=>configuration(template,index));
});
