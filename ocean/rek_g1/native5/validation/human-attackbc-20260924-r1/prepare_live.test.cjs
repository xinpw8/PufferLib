'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {configuration,checkpointSha,allonesSha}=require('./prepare_live.cjs');
const template={checkpoint_sha256:'old',worker:['worker','old.bin','old','1001','sampled','allones','--observation-schema=rek.native5.scaled_polar_xy.balance8_v1'],out:'old',encoder:['encoder','--busy-projection','dispatched_request_v4_duration'],feature_mask_sha256:allonesSha,observation_schema:'rek.native5.scaled_polar_xy.balance8_v1',relay:['relay','BOX64_DYNAREC_STRONGMEM=2','BOX64_DYNAREC_WEAKBARRIER=0'],live_attack_gate:{allowed_attacks:Array.from({length:17},(_,i)=>16+i),range_m:null},capture_controller:'frozen_policy',enter_private:true,max_seconds:140};
test('twenty unique prospective seeds1101..1120 with frozen checkpoint',()=>{
 const rows=Array.from({length:20},(_,i)=>configuration(template,i));assert.deepEqual(rows.map(x=>x.seed),Array.from({length:20},(_,i)=>1101+i));assert.equal(new Set(rows.map(x=>x.label)).size,20);
 for(const r of rows){assert.equal(r.cfg.checkpoint_sha256,checkpointSha);assert.equal(r.cfg.worker[2],checkpointSha);assert.equal(r.cfg.worker[3],String(r.seed));assert(r.cfg.out.endsWith('/'+r.label+'/trial'));}
});
test('all runtime settings, full action set and allones remain byte-value equivalent',()=>{
 const before=JSON.stringify(template),{cfg}=configuration(template,0);assert.equal(JSON.stringify(template),before);
 for(const k of Object.keys(template).filter(k=>!['checkpoint_sha256','worker','out'].includes(k)))assert.deepEqual(cfg[k],template[k]);assert.deepEqual(cfg.worker.slice(4),template.worker.slice(4));assert.equal(cfg.worker[0],template.worker[0]);
 assert.deepEqual(cfg.live_attack_gate.allowed_attacks,Array.from({length:17},(_,i)=>16+i));assert.equal(cfg.feature_mask_sha256,allonesSha);
});
test('out of cohort indices are rejected',()=>{for(const i of [-1,20,.5,NaN])assert.throws(()=>configuration(template,i));});
