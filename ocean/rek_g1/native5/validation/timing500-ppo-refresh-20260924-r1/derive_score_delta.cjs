'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const HEADER=384,ROW=1128,CONTRACT='received-score-delta-div5-half-life-5s.v1';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function transition(dt,own,opponent,nextOwn,nextOpponent){
 assert(Number.isFinite(dt)&&dt>0);
 assert([own,opponent,nextOwn,nextOpponent].every(Number.isInteger));
 assert(nextOwn>=own&&nextOpponent>=opponent);
 return {gamma:Math.fround(Math.pow(2,-dt/5)),reward:Math.fround(((nextOwn-own)-(nextOpponent-opponent))/5)};
}
function derive(stage,selection){
 selection=require('./selection.cjs').validateSelection(selection);assert.equal(stage,selection.stage);
 const original=path.join(stage,'export'),out=path.join(stage,'score-delta-5s');assert(!fs.existsSync(out));
 const source=fs.readFileSync(path.join(original,'authentic-trajectories-v3.bin'));
 const identityBytes=fs.readFileSync(path.join(original,'behavior-identity.json')),identity=JSON.parse(identityBytes);
 assert.equal(source.subarray(0,8).toString(),'REKRL003');assert.equal(source.readUInt32LE(8),3);
 assert.equal(source.subarray(256,288).toString('hex'),sha(identityBytes));
 assert.equal(source.subarray(320,352).toString('hex'),selection.checkpoint.sha256);
 for(const key of ['worker','checkpoint','native_object','feature_mask'])assert.equal(identity.artifacts[key].sha256,selection[key].sha256);
 assert.equal(identity.rounds.length,selection.rounds.length);assert(identity.rounds.length>=selection.minimum_completed_rounds);
 assert.equal(new Set(identity.rounds.map(r=>r.round_identity_sha256)).size,identity.rounds.length,'duplicate actual rounds');
 identity.rounds.forEach((r,i)=>{assert.equal(r.trial_id,selection.rounds[i].id);assert.equal(r.seed,selection.rounds[i].seed);});
 const rows=source.readUInt32LE(20);assert.equal(source.length,HEADER+rows*ROW);assert.equal(rows,identity.rounds.reduce((n,r)=>n+r.rows,0));
 const binary=Buffer.from(source);binary.write('REKRL005');binary.writeUInt32LE(5,8);
 const stats=[],discounts=[];let unchanged=0;
 for(const round of identity.rounds){
  let sum=0,seconds=0,minDt=Infinity,maxDt=0,changedRewards=0,changedGammas=0,zeroPolicy=0;
  for(let i=round.row_begin;i<round.row_end;i++){
   const at=HEADER+i*ROW,dt=source.readDoubleLE(at+1064);
   const own=source.readInt32LE(at+1096),opponent=source.readInt32LE(at+1100),nextOwn=source.readInt32LE(at+1104),nextOpponent=source.readInt32LE(at+1108);
   const t=transition(dt,own,opponent,nextOwn,nextOpponent);assert(t.gamma>0&&t.gamma<=1&&Number.isFinite(t.reward));
   binary.writeFloatLE(t.gamma,at+1072);binary.writeFloatLE(t.reward,at+1080);
   changedRewards+=Number(t.reward!==source.readFloatLE(at+1080));changedGammas+=Number(t.gamma!==source.readFloatLE(at+1072));
   sum+=t.reward;seconds+=dt;minDt=Math.min(minDt,dt);maxDt=Math.max(maxDt,dt);zeroPolicy+=Number(source.readFloatLE(at+16)===0);discounts.push(dt);
   for(let j=0;j<ROW;j++)if(!(j>=1072&&j<1076)&&!(j>=1080&&j<1084)){assert.equal(binary[at+j],source[at+j]);unchanged++;}
  }
  const score=round.awarded_points_by_slot,slot=round.local_slot,margin=(score[slot]-score[1-slot])/5;
  assert(Math.abs(sum-margin)<1e-6);assert.equal(zeroPolicy,round.terminal_race_rejected);
  stats.push({trial_id:round.trial_id,seed:round.seed,rows:round.rows,points:score,undiscounted_reward_sum:sum,expected_net_point_margin_div5:margin,seconds,decisions_per_second:round.rows/seconds,minimum_dt_seconds:minDt,maximum_dt_seconds:maxDt,changed_rewards:changedRewards,changed_gammas:changedGammas,actor_weight_zero_rows:zeroPolicy});
 }
 const sourceIdentity={path:path.join(original,'behavior-identity.json'),sha256:sha(identityBytes)},sourceDataset={path:path.join(original,'authentic-trajectories-v3.bin'),sha256:sha(source)};
 const nextIdentity={...identity,schema:'rek.authentic_behavior_identity.v5',reward_contract:CONTRACT,recorded_behavior_checkpoint:true,weight_migration:false,
  original_identity:sourceIdentity,original_dataset:sourceDataset,minimum_completed_rounds:selection.minimum_completed_rounds,
  reward:'((next_own_received_score-own_received_score)-(next_opponent_received_score-opponent_received_score))/5; no terminal win bonus; no score potential; no reward clipping',
  gamma_per_20ms:Math.pow(2,-.02/5),discounts:'float32(pow(2,-actual_source_to_next_source_QPC_seconds/5)); half-life=5 seconds',
  lambda_semantics:'Original lambda cells retained; complete_mc_zero_baseline uses lambda=1. Only complete-MC mode is selected for this experiment.',
  provenance:'Reward and gamma derived only from recorded score counters and QPC intervals. Observations, actions, masks, recurrence, actual worker seed, outcome metadata, and loss weights remain bitwise unchanged.',
  reward_exporter:{path:__filename,sha256:sha(fs.readFileSync(__filename))}};
 const nextBytes=Buffer.from(JSON.stringify(nextIdentity,null,2)+'\n');Buffer.from(sha(nextBytes),'hex').copy(binary,256);
 const receipt={schema:'rek.balance8_onpolicy_score_delta_export.v1',reward_contract:CONTRACT,source_dataset:sourceDataset,source_identity:sourceIdentity,
  dataset_sha256:sha(binary),identity_sha256:sha(nextBytes),checkpoint_sha256:identity.artifacts.checkpoint.sha256,rows,rounds:stats,
  preserved_row_bytes:unchanged,all223_observations_actions_masks_seeds_and_loss_weights_bitwise_unchanged:true,
  terminal_rejected_actor_weight_zero:true,terminal_outcome_metadata_retained_but_not_added_to_reward:true,
  original_actor_logits_recorded:false,native_exact_action_replay_required:true,no_gpu_execution:true,training_performed:false};
 fs.mkdirSync(out);for(const [name,bytes]of [['authentic-score-delta-v5.bin',binary],['behavior-identity.json',nextBytes],['export-receipt.json',JSON.stringify(receipt,null,2)+'\n']])fs.writeFileSync(path.join(out,name),bytes,{flag:'wx'});
 return receipt;
}
if(require.main===module){assert.equal(process.argv.length,3);const selection=require('./selection.cjs').loadSelection(process.argv[2]);console.log(JSON.stringify(derive(selection.stage,selection)));}
module.exports={transition,derive,CONTRACT};
