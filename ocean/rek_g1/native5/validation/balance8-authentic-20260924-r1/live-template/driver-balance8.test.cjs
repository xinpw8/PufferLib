'use strict';
const assert=require('node:assert/strict'),test=require('node:test');
const {validateWorkerReady,validateWorkerAction}=require('./live_transfer_run_masked.cjs');
const schema='rek.native5.scaled_polar_xy.balance8_v1',sha='a'.repeat(64),mask='b'.repeat(64);
const ready={type:'ready',checkpoint_sha256:sha,feature_mask_sha256:mask,native_cuda:true,environment_stepping:false,observation_schema:schema,precision:'bf16',selection:'sampled',observations:223,actions:33,hidden_size:256,num_layers:2};
test('explicit balance8 worker manifest accepted',()=>validateWorkerReady(ready,sha,mask,schema));
test('legacy manifest cannot pass balance8 binding',()=>assert.throws(()=>validateWorkerReady({...ready,observation_schema:'rek.native5.scaled_polar_xy.v1'},sha,mask,schema)));
test('mask and checkpoint binding retained',()=>{assert.throws(()=>validateWorkerReady(ready,'c'.repeat(64),mask,schema));assert.throws(()=>validateWorkerReady(ready,sha,'d'.repeat(64),schema));});
test('all 33 action categories still accepted',()=>{for(let action=0;action<33;action++)validateWorkerAction({type:'action',seq:2,round_id:'e'.repeat(64),checkpoint_sha256:sha,action},{sequence:2,round:'e'.repeat(64)},sha);});
