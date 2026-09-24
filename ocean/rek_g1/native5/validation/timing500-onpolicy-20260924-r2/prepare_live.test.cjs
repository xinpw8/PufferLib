'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {configuration,oldSha}=require('./prepare_live.cjs');
test('only checkpoint, RNG seed and output change; runtime, masks and controls remain exact',()=>{
 const t={checkpoint_sha256:oldSha,worker:['worker','old',oldSha,'1201','sampled','allones','schema'],out:'old',encoder:['encoder','schema'],relay:['relay','samebridge'],live_attack_gate:{allowed_attacks:[16,17],range_m:null},feature_mask_sha256:'unchanged'};
 const before=structuredClone(t),r=configuration(t,'/new/policy.bin','a'.repeat(64),0);
 assert.deepEqual(t,before);assert.equal(r.seed,1301);assert.equal(r.label,'timingppo-s1301');assert.equal(r.cfg.worker[1],'/new/policy.bin');
 for(const k of ['encoder','relay','live_attack_gate','feature_mask_sha256'])assert.deepEqual(r.cfg[k],t[k]);
 assert.deepEqual(r.cfg.worker.slice(4),t.worker.slice(4));assert.equal(configuration(t,'/new','b'.repeat(64),19).seed,1320);
 assert.throws(()=>configuration(t,'/new',oldSha,0));assert.throws(()=>configuration(t,'/new','a'.repeat(64),20));
});
