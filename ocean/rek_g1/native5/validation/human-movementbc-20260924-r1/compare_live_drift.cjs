'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process'),assert=require('node:assert/strict');
const {compare,probabilities}=require('../human-attackbc-20260924-r1/compare_live_drift.cjs');
const stage=__dirname,base='/home/spark-advantage/rek-training',prior=base+'/scorecredit-human-attackbc-20260924-r1';
const episode=base+'/human-attackbc-live-20260924-r1/humanbc-s1101/trial',exe=prior+'/build/diagnose-live-drift';
const baseline=prior+'/train-five-epochs/policy.bin',baselineSha='5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4';
const mask=base+'/joint-mask-transfer-20260924-r1/all-ones.bin',maskSha='59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex'),read=p=>fs.readFileSync(p),lines=b=>b.toString().trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
function movementMetrics(requests,records){
 const rows=records.filter(r=>r.type==='drift_row');assert.equal(rows.length,requests.length);
 const out={rows:rows.length,mean_legal_movement_mass:0,mean_unmasked_movement_mass:0,mean_legal_action6_mass:0,mean_legal_action7_mass:0,sampled_movement:0,sampled_action6:0,sampled_action7:0,conditional_unmasked_movement_argmax_counts:Array(14).fill(0)};
 for(let i=0;i<rows.length;i++){
  const r=rows[i],p=probabilities(r.logits,requests[i].mask),u=probabilities(r.logits,Array(33).fill(1));
  out.mean_legal_movement_mass+=p.slice(2,16).reduce((a,b)=>a+b,0);out.mean_unmasked_movement_mass+=u.slice(2,16).reduce((a,b)=>a+b,0);out.mean_legal_action6_mass+=p[6];out.mean_legal_action7_mass+=p[7];
  out.sampled_movement+=r.action>=2&&r.action<=15;out.sampled_action6+=r.action===6;out.sampled_action7+=r.action===7;
  let best=2;for(let a=3;a<=15;a++)if(r.logits[a]>r.logits[best])best=a;out.conditional_unmasked_movement_argmax_counts[best-2]++;
 }
 for(const k of Object.keys(out))if(k.startsWith('mean_'))out[k]/=rows.length;
 return out;
}
function main(){
 assert.equal(process.argv.length,5,'usage candidate_checkpoint candidate_sha --check|--run');const [candidate,candidateSha,mode]=process.argv.slice(2);assert(['--check','--run'].includes(mode));
 assert.equal(sha(read(baseline)),baselineSha);assert.equal(sha(read(candidate)),candidateSha);assert.equal(sha(read(mask)),maskSha);assert.equal(sha(read(exe)),'61e8dc64fd2bafec4084d470f0c048ee4e0a7bbebf17b66b0628fed7f260d8b2');
 const input=read(episode+'/worker.stdin.jsonl'),actualBytes=read(episode+'/worker.stdout.jsonl');
 assert.equal(sha(input),'5403b918106442ed5c2addaabdd0e1d7bfa0f0b1a5fc61a8dcb4255cf88d0a00');assert.equal(sha(actualBytes),'ededa4388ea31a7b632825f52e4a52a9c4fa7705fad4eea864d25311a8dc8a0c');
 const requests=lines(input).filter(q=>q.type==='step'&&!q.terminal),recorded=lines(actualBytes),ready=recorded.find(r=>r.type==='ready'),actions=recorded.filter(r=>r.type==='action'),summary=JSON.parse(read(episode+'/summary.json'));
 assert.equal(ready.checkpoint_sha256,baselineSha);assert.equal(ready.seed,1101);assert.equal(ready.feature_mask_sha256,maskSha);assert.equal(requests.length,actions.length);assert.equal(actions.length,787);assert.equal(summary.predictions,787);assert.equal(summary.applied,786);assert.equal(summary.rejected,1);assert.equal(summary.final_round.active,true);
 assert(requests.every(q=>q.observation_schema==='rek.native5.scaled_polar_xy.balance8_v1'&&q.round_id===requests[0].round_id));
 const actual=new Map(actions.map(r=>[r.seq,r]));assert.equal(actual.size,787);
 const plan={baseline_argv:[exe,baseline,baselineSha,'1101',mask],candidate_argv:[exe,candidate,candidateSha,'1101',mask],source_stdin_sha256:sha(input),source_stdout_sha256:sha(actualBytes),source_summary_sha256:sha(read(episode+'/summary.json')),
  worker_seed:1101,rows:requests.length,first_seq:requests[0].seq,last_seq:requests.at(-1).seq,includes_last_rejected_sample:true,source_completed_round:false,source_stop_reason:summary.stop_reason,source_score_snapshot:summary.final_round.clean_hits,
  full223_allones:true,observations_and_legal_masks_unchanged:true,not_a_counterfactual_game_rollout:true};
 console.log(JSON.stringify(plan));if(mode==='--check')return;
 const out=path.join(stage,'live-drift');assert(!fs.existsSync(out));fs.mkdirSync(out);const results=[];
 for(const [label,argv]of [['baseline',plan.baseline_argv],['candidate',plan.candidate_argv]]){
  const started=process.hrtime.bigint(),r=cp.spawnSync(argv[0],argv.slice(1),{input,maxBuffer:32*1024*1024,timeout:90000});
  fs.writeFileSync(path.join(out,label+'.stdout.jsonl'),r.stdout??'',{flag:'wx'});fs.writeFileSync(path.join(out,label+'.stderr.txt'),r.stderr??'',{flag:'wx'});
  fs.writeFileSync(path.join(out,label+'.execution.json'),JSON.stringify({exit_code:r.status,signal:r.signal,error:r.error?.message??null,wall_seconds:Number(process.hrtime.bigint()-started)/1e9})+'\n',{flag:'wx'});assert.equal(r.status,0,String(r.stderr));results.push(lines(r.stdout));
 }
 const report={...plan,...compare(requests,...results,actual),baseline_movement:movementMetrics(requests,results[0]),candidate_movement:movementMetrics(requests,results[1]),schema:'rek.conditional_movement_bc_full_live_drift.v1',binary_sha256:sha(read(exe)),checkpoint_sha256:candidateSha,environment_steps:0,game_connection:false,optimizer_updates:0,
  limitation:'Closed watchdog-truncated observation stream; not a complete round or counterfactual trajectory. Direction argmax is descriptive, not human-label CE. Heldout human per-class CE/recall is reported by native BC.'};
 fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(report));
}
if(require.main===module){try{main();}catch(e){console.error(e.stack);process.exitCode=1;}}
module.exports={movementMetrics};
