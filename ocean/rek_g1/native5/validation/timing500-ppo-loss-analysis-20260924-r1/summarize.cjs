'use strict';
// Whitelist-only aggregation of private, validated offline analysis reports.
const fs=require('node:fs'),crypto=require('node:crypto');
function aggregate(rounds){
 const sum=f=>rounds.reduce((n,r)=>n+f(r),0);
 const paired=f=>[0,1].map(i=>sum(r=>f(r)[i]));
 return {rounds:rounds.length,points:paired(r=>r.scored_native_totals),ordinary_points:paired(r=>r.ordinary_points),
  plus5_awards:[0,1].map(i=>sum(r=>r.point_histogram[i]['5']||0)),
  predictions:sum(r=>r.predictions),local_applied:sum(r=>r.local_applied),armed_requests:sum(r=>r.attack_requests),
  native_dispatches:sum(r=>r.native_move_dispatch.count),unique_dispatch_matches:sum(r=>r.native_move_dispatch.ack_with_unique_dispatch_within_100ms),
  category17_legal_decisions:sum(r=>r.readiness.legal17),translation_blocked_decisions:sum(r=>r.readiness.translation_blocked),projected_busy_decisions:sum(r=>r.readiness.projected_busy),
  request_geometry:{facing45:sum(r=>r.attack_geometry.facing45),behind90:sum(r=>r.attack_geometry.behind90),distance_over1:sum(r=>r.attack_geometry.over1),facing45_distance_point8:sum(r=>r.attack_geometry.facing45_within_point8)},
  pooled_source_hz:sum(r=>r.cadence.observations-1)/sum(r=>r.cadence.seconds),
  attack_category_counts:Object.fromEntries(Array.from({length:17},(_,i)=>[i+16,sum(r=>r.action_counts[i+16])])),
  descriptive_associations:Object.fromEntries([17,21,23,26].map(cat=>{const requests=rounds.flatMap(r=>r.requests.filter(q=>q.policy_action===cat));const bins={all:requests,close_facing:requests.filter(q=>q.geometry_at_policy_observation.distance_unity<=.8&&Math.abs(q.geometry_at_policy_observation.bearing_rad[0])<=Math.PI/4),behind90:requests.filter(q=>Math.abs(q.geometry_at_policy_observation.bearing_rad[0])>Math.PI/2)};return [cat,Object.fromEntries(Object.entries(bins).map(([k,v])=>[k,{requests:v.length,ordinary_score_followed_within2s:v.filter(q=>q.ordinary_score_receipts_after[2].length>0).length}]))];}))};
}
function sanitize(rounds){
 const ids=new Set();for(const r of rounds){if(ids.has(r.label))throw Error('Duplicate round');ids.add(r.label);if(r.referee_validation.verification_passed!==true||r.final_round.active||r.final_round.redo||r.final_round.time_remaining!==0)throw Error('Unvalidated or incomplete round');}
 const c2=rounds.filter(r=>r.label.startsWith('timingppo-')),parent=rounds.filter(r=>r.label.startsWith('timing500-'));
 if(c2.length!==5||parent.length!==8)throw Error('Expected fixed five C2 and eight parent rounds');
 return {schema:'rek.c2_closed_sanitized_analysis.v1',selection:'Five completed C2 versus eight completed parent rounds; unmatched seeds',
  semantics:{score:'independently decoded native score with exact relay scorer/counter join',requests:'local armed ACK plus native outbound projection, not server execution',associations:'descriptive temporal windows, not hit/miss labels; requests can share subsequent score receipts',units:'Unity root distances, radians, seconds; no metre calibration'},
  c2:aggregate(c2),parent:aggregate(parent),rounds:rounds.map(r=>({label:r.label,checkpoint_sha256:r.checkpoint,points:r.scored_native_totals,ordinary_points:r.ordinary_points,plus5_awards:r.point_histogram.map(h=>h['5']||0),predictions:r.predictions,local_applied:r.local_applied,armed_requests:r.attack_requests,source_hz:r.cadence.hz,native_sha256:r.provenance.native.sha256,native_bytes:r.provenance.native.bytes,referee_validation_passed:true}))};
}
module.exports={sanitize,aggregate};
if(require.main===module){const [output,...inputs]=process.argv.slice(2);if(!output||inputs.length!==3)throw Error('Usage: node summarize.cjs NEW_OUTPUT_JSON INITIAL_REPORT WIN_REPORT FINAL_REPORT');const reports=inputs.map(file=>{const bytes=fs.readFileSync(file);return {value:JSON.parse(bytes),sha256:crypto.createHash('sha256').update(bytes).digest('hex')};});const result=sanitize(reports.flatMap(r=>r.value.rounds));result.private_report_sha256=reports.map(r=>r.sha256);fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({output,rounds:result.rounds.length,c2:result.c2.points,parent:result.parent.points}));}
