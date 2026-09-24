'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {configuration,oldSha,parentSha}=require('./prepare_live.cjs');
test('only checkpoint, RNG seed and output change; runtime, masks and controls remain exact',()=>{
 const t={checkpoint_sha256:oldSha,worker:['worker','old',oldSha,'1201','sampled','allones','schema'],out:'old',encoder:['encoder','schema'],relay:['relay','samebridge'],live_attack_gate:{allowed_attacks:[16,17],range_m:null},feature_mask_sha256:'unchanged'};
 const before=structuredClone(t),r=configuration(t,'/new/policy.bin','a'.repeat(64),0);
 assert.deepEqual(t,before);assert.equal(r.seed,1501);assert.equal(r.label,'refresh-s1501');assert.equal(r.cfg.worker[1],'/new/policy.bin');
 for(const k of ['encoder','relay','live_attack_gate','feature_mask_sha256'])assert.deepEqual(r.cfg[k],t[k]);
 assert.deepEqual(r.cfg.worker.slice(4),t.worker.slice(4));assert.equal(configuration(t,'/new','b'.repeat(64),19).seed,1520);
 assert.throws(()=>configuration(t,'/new',oldSha,0));assert.throws(()=>configuration(t,'/new','a'.repeat(64),20));
});

test('training parent is actual C2 and distinct from the frozen runtime template',()=>{
 assert.equal(parentSha,'c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533');
 assert.notEqual(parentSha,oldSha);
});
