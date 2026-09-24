'use strict';
const fs=require('node:fs'),cp=require('node:child_process'),assert=require('node:assert/strict');
const stage='/home/spark-advantage/rek-training/balance8-authentic-20260924-r1';
const raw='/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/evidence/noprior-s601/trial/encoder.stdin.jsonl';
const all=fs.readFileSync(raw,'utf8').trim().split('\n').map(JSON.parse).filter(x=>x.observation_sequence);
const start=all.findIndex((x,i)=>i+2<all.length&&[1,2].every(k=>{const a=all[i+k-1],b=all[i+k],dt=(b.clock.qpc_ticks-a.clock.qpc_ticks)/b.clock.qpc_frequency_hz;return dt>0&&dt<=.25;}));
assert(start>=0);const samples=all.slice(start,start+3);
const binary=stage+'/build/encode-balance8',schema='rek.native5.scaled_polar_xy.balance8_v1';
const model='/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact/model.two_fighter_arena.xml';
let cases=0;
function run(items){
 const r=cp.spawnSync(binary,['--model',model,'--projection','client_pose_projection_v1','--observation-schema',schema,'--busy-projection','dispatched_request_v4_duration'],{input:items.map(JSON.stringify).join('\n')+'\n',encoding:'utf8'});
 assert.equal(r.status,0,r.stderr);const rows=r.stdout.trim().split('\n').map(JSON.parse);assert.equal(rows.shift().observation_schema,schema);cases++;return rows;
}
function copy(){return structuredClone(samples);}
let s=copy(),r=run(s);assert.equal(r[0].ready,false);assert.equal(r[1].ready,true);assert.equal(r[1].worker_request.observation[203],1);
s=copy();s[1].clock.qpc_ticks=s[0].clock.qpc_ticks+Math.round(.251*s[0].clock.qpc_frequency_hz);s[1].referee.receipt_qpc_ticks=s[1].clock.qpc_ticks-100;s[1].referee.receipt_age_seconds=100/s[1].clock.qpc_frequency_hz;r=run(s.slice(0,2));assert.equal(r[1].ready,false);
s=copy();s[1].referee.receipt_qpc_ticks=s[1].clock.qpc_ticks-Math.round(.501*s[1].clock.qpc_frequency_hz);s[1].referee.receipt_age_seconds=.501;r=run(s.slice(0,2));assert.equal(r[1].ready,false);
s=copy();s[1].round_identity_sha256='b'.repeat(64);s[1].referee.lifecycle++;r=run(s.slice(0,2));assert.equal(r[1].ready,false);
s=copy();s[1].round.active=false;s[1].phase=0;r=run(s.slice(0,2));assert.equal(r[1].ready,false);
s=copy();for(const key of ['receipt_sequence','lifecycle','receipt_qpc_ticks','receipt_qpc_frequency_hz','receipt_unity_frame','receipt_unity_time','receipt_unity_unscaled_time','wire_body_sha256','wire_body_base64','count_mask','count_seconds','slot0_count_active','slot1_count_active','call_sequence','call_type','call_name','call_faller','call_points','call_observation_sequence','call_sequence_transition','call_history_censored','packet_phase','packet_round_number','packet_round_active','packet_round_redo','packet_round_knockout_occurred','packet_round_result'])s[1].referee[key]=null;
Object.assign(s[1].referee,{available:false,reason:'referee_snapshot_not_observed',observation_hooks_verified:false,call_available:false,receipt_age_seconds:null});r=run(s.slice(0,2));assert.equal(r[1].ready,true);for(const c of [202,204,205])assert.equal(r[1].worker_request.observation[c],0);
s=copy();s[1].fighters[0].root_rotation_xyzw=[0,0,0,0];r=run(s.slice(0,2));assert.equal(r[1].ready,false);
s=copy();r=run([s[0],{type:'reset'},s[1]]);assert.equal(r[2].ready,false);
console.log(JSON.stringify({balance8_actual_encoder_edge_tests:'passed',cases,first_gap_stale_round_inactive_unavailable_invalid_reset:true,gpu_calls:0}));
