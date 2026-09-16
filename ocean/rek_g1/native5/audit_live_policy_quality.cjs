'use strict';
// Passive analysis of saved telemetry. No game input or policy inference.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const readline=require('node:readline');
const translating=category=>(category>=2&&category<=5)||(category>=8&&category<=15);
async function records(file,visit){
  for await(const line of readline.createInterface({input:fs.createReadStream(file)})){
    if(line.trim())visit(JSON.parse(line));
  }
}
async function digest(file){const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(file))h.update(b);return h.digest('hex');}
async function audit(dir,correctedFile){
  const sources=new Map(),encoded=new Map(),corrected=new Map(),acks=[];
  await records(path.join(dir,'relay.stdout.jsonl'),x=>{
    if(x.event==='g1_policy_state')sources.set(x.observation_sequence,x);
    if(x.event==='g1_policy_action')acks.push(x);
  });
  await records(path.join(dir,'encoder.stdout.jsonl'),x=>{if(x.ready===true)encoded.set(x.worker_request.seq,x);});
  if(correctedFile)await records(correctedFile,x=>{if(x.ready===true)corrected.set(x.worker_request.seq,x);});
  let advertisedWhileHeld=0,rejectedWhileHeld=0,blockedRejected=0,blockedApplied=0;
  let attacks=0,accepted=0,facing=0,busy=0,subMinimum=0,changedObs=0,changedMasks=0,compared=0;
  const actionCounts=Array.from({length:33},()=>({requests:0,applied:0,rejected:0,held_translation:0}));
  for(const ack of acks){
    const src=sources.get(ack.observation_sequence);if(!src)throw Error('action_source_missing');
    const translated=translating(src.input.desired_action);
    const count=actionCounts[ack.action];count.requests++;count.applied+=Number(ack.applied);count.rejected+=Number(!ack.applied);
    if(ack.action<16)continue;
    attacks++;accepted+=Number(ack.applied);count.held_translation+=Number(translated);
    if(translated&&src.action_mask[ack.action])advertisedWhileHeld++;
    if(translated&&!ack.applied)rejectedWhileHeld++;
    const c=corrected.get(ack.observation_sequence);
    if(c&&!c.worker_request.mask[ack.action]){
      if(ack.applied)blockedApplied++;else blockedRejected++;
    }
  }
  for(const [seq,x] of encoded){
    const o=x.worker_request.observation;
    facing+=Number(Math.abs(o[87]*Math.PI)<.16);busy+=Number(o[182]>.5);subMinimum+=Number(o[86]<.44-1e-5);
    const c=corrected.get(seq);if(c){
      compared++;
      changedObs+=Number(JSON.stringify(o)!==JSON.stringify(c.worker_request.observation));
      changedMasks+=Number(JSON.stringify(x.worker_request.mask)!==JSON.stringify(c.worker_request.mask));
    }
  }
  const summary=JSON.parse(fs.readFileSync(path.join(dir,'summary.json'),'utf8'));
  const hashes={};for(const f of ['relay.stdout.jsonl','encoder.stdout.jsonl','summary.json'])hashes[f]=await digest(path.join(dir,f));
  if(correctedFile)hashes.corrected_encoder=await digest(correctedFile);
  return {schema:'rek.live_policy_quality_audit.v1',trial:path.basename(dir),checkpoint_sha256:summary.checkpoint_sha256,
    observations:encoded.size,attacks_requested:attacks,attacks_applied:accepted,attacks_rejected:attacks-accepted,
    attack_source_advertised_while_translation_held:advertisedWhileHeld,rejected_attacks_while_translation_held:rejectedWhileHeld,
    facing_within_0_16_rad_fraction:facing/encoded.size,projected_busy_fraction:busy/encoded.size,
    root_gap_below_compact_minimum_fraction:subMinimum/encoded.size,
    final_round:summary.final_round,action_counts:actionCounts,
    corrected_replay:correctedFile?{compared_observations:compared,changed_observations:changedObs,changed_masks:changedMasks,
      recorded_rejected_actions_now_masked:blockedRejected,recorded_applied_actions_now_masked:blockedApplied}:null,
    limits:['Offline replay measures mask changes, not counterfactual wins or new policy actions.',
      'Facing uses measured root yaw; compact training uses logical heading.',
      'Applied is a local acknowledgment, not authoritative server acceptance.'],
    artifact_sha256:hashes};
}
if(require.main===module){
  const [dir,out,corrected]=process.argv.slice(2);
  if(!dir||!out)throw Error('Usage: node audit_live_policy_quality.cjs TRIAL NEW_REPORT [CORRECTED_ENCODER_JSONL]');
  audit(dir,corrected).then(result=>{fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(result));}).catch(e=>{console.error(e.stack);process.exitCode=1;});
}
module.exports={translating,audit};
