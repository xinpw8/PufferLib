'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const SCHEMA='rek.native5.scaled_polar_xy.balance8_v1';
const PINS={checkpoint:'c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533',worker:'52741aed67037073bff5ecb21af63864550cba93a316dfe59a41b8b50126d7b2',native_object:'4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c',feature_mask:'59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b'};
const NATIVE_CAPTURE_SHA='7f3189c1d619f7494eaa6f611eedcc850152d387f1a02bb0f74e8bcf9d516cf1';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function validateSelection(c){
 assert.equal(c.schema,'rek.scorecredit5s_onpolicy_selection.v1');assert.equal(c.observation_schema,SCHEMA);assert(path.isAbsolute(c.stage));
 const minimum=c.minimum_completed_rounds??20;assert(Number.isInteger(minimum)&&minimum>0);
 assert(Array.isArray(c.rounds)&&c.rounds.length>=minimum,`need at least ${minimum} explicitly selected completed rounds`);
 for(const key of Object.keys(PINS)){assert(path.isAbsolute(c[key]?.path));assert.equal(c[key].sha256,PINS[key],key+' identity mismatch');}
 assert.equal(new Set(c.rounds.map(x=>x.id)).size,c.rounds.length,'duplicate episode id');
 assert.equal(new Set(c.rounds.map(x=>path.resolve(x.directory))).size,c.rounds.length,'duplicate episode path');
 for(const r of c.rounds){assert(/^[A-Za-z0-9_-]+$/.test(r.id));assert(path.isAbsolute(r.directory)&&path.isAbsolute(r.config));assert(Number.isSafeInteger(r.seed)&&r.seed>=0);}
 return {...c,minimum_completed_rounds:minimum};
}
function loadSelection(file){return validateSelection(JSON.parse(fs.readFileSync(file)));}
function pinArtifacts(c){for(const key of Object.keys(PINS))assert.equal(sha(fs.readFileSync(c[key].path)),c[key].sha256,key+' file changed');}
function validateNativeSelection(capture,ids){
 assert.equal(capture.schema,'rek.closed_c2_native_capture_selection.v1');assert.equal(capture.campaign,'timing500-ppo-live-20260924-r1');
 assert.equal(capture.campaign_end_utc,'2026-09-24T11:00:34.706Z');assert.equal(capture.external_analysis_frozen,true);
 assert.equal(capture.rounds.length,5);assert.deepEqual(capture.rounds.map(r=>r.label),ids);assert.equal(new Set(ids).size,5);
 assert.equal(new Set(capture.rounds.map(r=>r.native_source.file)).size,5);
 for(const r of capture.rounds){const p=r.native_source;assert.equal(r.strict_validation_passed,true);
  assert.equal(path.posix.dirname(p.file),'/home/spark-advantage/codexrook-runtime/wineprefix/drive_c/rekagent/evidence/runtime/rek-private-ai-protocol-v7');
  assert(Number.isSafeInteger(p.bytes)&&p.bytes>0&&/^[a-f0-9]{64}$/.test(p.sha256));}
 return capture;
}
function loadNativeSelection(c){
 assert.equal(c.native_capture_selection?.sha256,NATIVE_CAPTURE_SHA);const bytes=fs.readFileSync(c.native_capture_selection.path);
 assert.equal(sha(bytes),NATIVE_CAPTURE_SHA,'native capture selection changed');
 return validateNativeSelection(JSON.parse(bytes),c.rounds.map(r=>r.id));
}
function startupRound(source,minimum){
 const r=source?.round;assert(r?.active===true&&r.duration===120&&r.redo===false&&r.result_value===0&&Number.isFinite(r.time_remaining)&&r.time_remaining>=minimum&&r.time_remaining<=120,'startup outside recorded driver contract');assert.deepEqual(r.clean_hits,[0,0]);
}
function initialSource(summary,first,controlled,readinessSource){
 if(!summary.controlled_startup_validated){assert.deepEqual(first.round,summary.initial_round);return first;}
 const r=summary.startup_readiness;assert(r&&controlled&&readinessSource,'controlled/readiness source missing');
 assert.equal(readinessSource.stream_active,false);assert.equal(readinessSource.global_input_emitted,false);assert.equal(readinessSource.observation_sequence,r.observation_sequence);
 assert(Number.isSafeInteger(readinessSource.clock.qpc_ticks));assert.equal(String(readinessSource.clock.qpc_ticks),String(r.qpc_ticks));assert.equal(readinessSource.clock.qpc_frequency_hz,r.qpc_frequency_hz);assert.equal(readinessSource.clock.unity_frame,r.unity_frame);
 assert.equal(readinessSource.round_identity_sha256,r.round_identity_sha256);assert.equal(readinessSource.local_slot,r.local_slot);assert.equal(readinessSource.round.time_remaining,r.time_remaining);startupRound(readinessSource,117.5);
 assert.equal(controlled.stream_active,true);assert.equal(controlled.round_identity_sha256,r.round_identity_sha256);assert.equal(controlled.local_slot,r.local_slot);assert.equal(controlled.clock.qpc_frequency_hz,r.qpc_frequency_hz);
 assert(Number.isSafeInteger(controlled.clock.qpc_ticks)&&BigInt(controlled.clock.qpc_ticks)>BigInt(r.qpc_ticks));assert(controlled.clock.unity_frame>r.unity_frame&&controlled.observation_sequence>r.observation_sequence);startupRound(controlled,117);
 assert.equal(first.round_identity_sha256,r.round_identity_sha256);assert.equal(first.local_slot,r.local_slot);assert.equal(first.round.number,controlled.round.number);assert.equal(readinessSource.round.number,controlled.round.number);
 assert.deepEqual(controlled.round,summary.initial_round);return controlled;
}
function completedSummary(summary,c){
 assert.equal(summary.checkpoint_sha256,c.checkpoint.sha256);assert.equal(summary.observation_schema,SCHEMA);assert.equal(summary.feature_mask_sha256,c.feature_mask.sha256);assert.equal(summary.authentic_client,true);
 assert.equal(summary.opponent?.sparring_bot_number,1);assert.equal(summary.opponent?.client_ai_difficulty,0);
 const a=summary.initial_round,b=summary.final_round;assert(a&&b);assert.equal(a.duration,120);assert.equal(b.duration,120);assert.equal(a.redo,false);assert.equal(b.redo,false);assert.equal(a.active,true);assert.equal(b.active,false);assert.equal(b.time_remaining,0);assert.equal(b.result,'WonByPoints');assert(a.time_remaining>=(summary.controlled_startup_validated?117:119));assert.deepEqual(a.clean_hits,[0,0]);
 assert(summary.predictions>0&&summary.applied>0);return true;
}
module.exports={SCHEMA,PINS,NATIVE_CAPTURE_SHA,sha,validateSelection,loadSelection,pinArtifacts,initialSource,completedSummary,validateNativeSelection,loadNativeSelection};
