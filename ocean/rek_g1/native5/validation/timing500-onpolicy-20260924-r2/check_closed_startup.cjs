'use strict';
const fs=require('node:fs'),path=require('node:path'),readline=require('node:readline'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {SCHEMA,sha,pinArtifacts,validateSelection,completedSummary,initialSource}=require('./selection.cjs');
async function scan(file,visit){
 const before=fs.statSync(file),hash=crypto.createHash('sha256'),input=fs.createReadStream(file);input.on('data',b=>hash.update(b));let lines=0;
 for await(const line of readline.createInterface({input,crlfDelay:Infinity})){lines++;visit(JSON.parse(line));}
 const after=fs.statSync(file);assert.equal(after.size,before.size);assert.equal(after.mtimeMs,before.mtimeMs);return{path:file,sha256:hash.digest('hex'),bytes:before.size,lines};
}
async function main(){
 assert(process.argv.length===2||(process.argv.length===3&&process.argv[2]==='--save'));
 const c=JSON.parse(fs.readFileSync(path.join(__dirname,'selection.example.json')));assert.equal(c.rounds.length,6);assert.throws(()=>validateSelection(c),/at least 20/);pinArtifacts(c);
 const driver='/home/spark-advantage/rek-training/humanbc-timing500-live-20260924-r1/live_transfer_run_masked.cjs',driverSha='b57654d54a76c80b38c9e44f59f53efb2502959a70dc568835ec5a8f6d238214';assert.equal(sha(fs.readFileSync(driver)),driverSha);
 const rounds=[];
 for(const r of c.rounds){
  const trial=path.join(r.directory,'trial'),summaryBytes=fs.readFileSync(path.join(trial,'summary.json')),s=JSON.parse(summaryBytes),configBytes=fs.readFileSync(r.config),config=JSON.parse(configBytes);
  assert.equal(fs.readFileSync(path.join(r.directory,'wrapper.exit-code.txt'),'utf8').trim(),'0');completedSummary(s,c);assert.equal(s.controlled_startup_validated,true);
  assert.deepEqual(config.worker,[c.worker.path,c.checkpoint.path,c.checkpoint.sha256,String(r.seed),'sampled',c.feature_mask.path,'--observation-schema='+SCHEMA]);assert.equal(config.checkpoint_sha256,c.checkpoint.sha256);assert.equal(config.observation_schema,SCHEMA);assert.equal(config.feature_mask_sha256,c.feature_mask.sha256);
  let first=null,readiness=null,controlled=null,last=null,ackCount=0,rejected=0,lastAck=null;
  const relay=await scan(path.join(trial,'relay.stdout.jsonl'),x=>{
   if(x.event==='g1_policy_action'){ackCount++;rejected+=Number(!x.applied);lastAck=x;return;}
   if(x.event!=='g1_policy_state')return;if(!first)first=x;
   assert.equal(x.round_identity_sha256,first.round_identity_sha256);assert.equal(x.local_slot,first.local_slot);assert.equal(x.round.number,first.round.number);
   if(x.observation_sequence===s.startup_readiness.observation_sequence)readiness=x;if(!controlled&&x.stream_active===true)controlled=x;last=x;
  });
  const initial=initialSource(s,first,controlled,readiness);assert.deepEqual(first.round.clean_hits,[0,0]);assert.deepEqual(last.round,s.final_round);
  let ready=null,actions=0,lastAction=null,firstAction=null,requests=0;
  const stdout=await scan(path.join(trial,'worker.stdout.jsonl'),x=>{if(x.type==='ready'){assert.equal(ready,null);ready=x;}if(x.type==='action'){actions++;firstAction??=x;lastAction=x;assert.equal(x.checkpoint_sha256,c.checkpoint.sha256);assert.equal(x.observation_schema,SCHEMA);assert.equal(x.feature_mask_sha256,c.feature_mask.sha256);assert.equal(x.decision_index,actions);assert.equal(x.recurrent_reset,actions===1);}});
  assert.equal(ready.checkpoint_sha256,c.checkpoint.sha256);assert.equal(ready.seed,r.seed);assert.equal(ready.feature_mask_sha256,c.feature_mask.sha256);assert.equal(ready.observation_schema,SCHEMA);assert.equal(ready.selection,'sampled');
  const stdin=await scan(path.join(trial,'worker.stdin.jsonl'),x=>{assert.equal(x.type,'step');assert.equal(x.round_id,first.round_identity_sha256);assert.equal(x.observation_schema,SCHEMA);requests+=Number(!x.terminal);});
  assert.equal(actions,s.predictions);assert.equal(requests,actions);assert.equal(ackCount,actions);assert.equal(rejected,s.rejected);assert.equal(lastAck.observation_sequence,lastAction.seq);assert.equal(lastAck.action,lastAction.action);assert.equal(lastAck.applied,false);assert.equal(lastAck.reason,'policy_stream_not_owned');
  rounds.push({id:r.id,seed:r.seed,points:s.final_round.clean_hits,winner_slot:s.final_round.winner_index,local_slot:first.local_slot,round_identity_sha256:first.round_identity_sha256,
   actual_initial_time_remaining_seconds:initial.round.time_remaining,native_round_elapsed_before_control_seconds:120-initial.round.time_remaining,
   earliest_raw_sequence:first.observation_sequence,readiness_sequence:readiness.observation_sequence,first_controlled_sequence:initial.observation_sequence,first_worker_sequence:firstAction.seq,last_worker_sequence:lastAction.seq,
   recorded_passive_qpc_seconds:(initial.clock.qpc_ticks-first.clock.qpc_ticks)/initial.clock.qpc_frequency_hz,readiness_to_control_qpc_seconds:(initial.clock.qpc_ticks-readiness.clock.qpc_ticks)/initial.clock.qpc_frequency_hz,
   predictions:actions,rejected_sampled_rows_preserved:rejected,summary_sha256:sha(summaryBytes),config_sha256:sha(configBytes),sources:[relay,stdout,stdin],startup_binding_passed:true});
 }
 assert.equal(new Set(rounds.map(r=>r.round_identity_sha256)).size,6);
 const receipt={schema:'rek.timing500_closed_startup_cpu_check.v1',driver:{path:driver,sha256:driverSha},behavior:c.checkpoint,actual_artifact_pins_verified:true,closed_rounds:rounds,
  predictions:rounds.reduce((n,r)=>n+r.predictions,0),sample_replay_performed:false,actor_logits_recorded:false,default_minimum20_rejects_six_example:true,
  actual_selection_frozen:false,training_dataset_exported:false,gpu_used:false,active_cohort_changed:false};
 if(process.argv[2]==='--save')fs.writeFileSync(path.join(__dirname,'closed-startup-cpu-test.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify(receipt));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
