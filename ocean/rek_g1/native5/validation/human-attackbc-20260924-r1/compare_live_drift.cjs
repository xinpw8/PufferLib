'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process'),assert=require('node:assert/strict');
const stage=__dirname,base='/home/spark-advantage/rek-training';
const episode=base+'/scorecredit5s-live-20260924-r2/credit5s-s1001-retry3/trial';
const mask=base+'/joint-mask-transfer-20260924-r1/all-ones.bin',exe=path.join(stage,'build/diagnose-live-drift');
const baseCheckpoint=base+'/balance8-onpolicy-20260924-r2/train-score-delta/policy.bin',baseSha='0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex'),read=p=>fs.readFileSync(p),lines=b=>b.toString().trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
function probabilities(z,mask){const max=Math.max(...z.slice(0,33).filter((_,a)=>mask[a]));const ex=z.slice(0,33).map((x,a)=>mask[a]?Math.exp(x-max):0),sum=ex.reduce((a,b)=>a+b,0);return ex.map(v=>v/sum);}
function compare(requests,before,after,actual){
  const b=before.filter(x=>x.type==='drift_row'),a=after.filter(x=>x.type==='drift_row');assert.equal(b.length,requests.length);assert.equal(a.length,b.length);
  const total={rows:b.length,baseline_matching_recorded_actions:0,sampled_action_changes:0,baseline_sampled_attacks:0,candidate_sampled_attacks:0,baseline_sampled_category17:0,candidate_sampled_category17:0,mean_legal_kl:0,max_legal_kl:0,baseline_mean_legal_attack_mass:0,candidate_mean_legal_attack_mass:0,baseline_mean_unmasked_attack_mass:0,candidate_mean_unmasked_attack_mass:0,baseline_mean_legal_category17_mass:0,candidate_mean_legal_category17_mass:0};
  for(let i=0;i<b.length;i++){
    const q=requests[i];assert.equal(b[i].seq,q.seq);assert.equal(a[i].seq,q.seq);assert.equal(b[i].round_id,q.round_id);assert.equal(a[i].round_id,q.round_id);assert.equal(q.mask[b[i].action],1);assert.equal(q.mask[a[i].action],1);
    assert.equal(b[i].recurrent_reset,i===0);assert.equal(a[i].recurrent_reset,i===0);assert.equal(actual.get(q.seq)?.action,b[i].action,'baseline does not reproduce actual sampled action');
    total.baseline_matching_recorded_actions++;total.sampled_action_changes+=b[i].action!==a[i].action;
    total.baseline_sampled_attacks+=b[i].action>=16;total.candidate_sampled_attacks+=a[i].action>=16;total.baseline_sampled_category17+=b[i].action===17;total.candidate_sampled_category17+=a[i].action===17;
    const p=probabilities(b[i].logits,q.mask),r=probabilities(a[i].logits,q.mask),u=probabilities(b[i].logits,Array(33).fill(1)),v=probabilities(a[i].logits,Array(33).fill(1));
    const kl=p.reduce((s,x,j)=>s+(x>0?x*Math.log(x/r[j]):0),0);assert(Number.isFinite(kl));total.mean_legal_kl+=kl;total.max_legal_kl=Math.max(total.max_legal_kl,kl);
    total.baseline_mean_legal_attack_mass+=p.slice(16).reduce((s,x)=>s+x,0);total.candidate_mean_legal_attack_mass+=r.slice(16).reduce((s,x)=>s+x,0);
    total.baseline_mean_unmasked_attack_mass+=u.slice(16).reduce((s,x)=>s+x,0);total.candidate_mean_unmasked_attack_mass+=v.slice(16).reduce((s,x)=>s+x,0);
    total.baseline_mean_legal_category17_mass+=p[17];total.candidate_mean_legal_category17_mass+=r[17];
  }
  for(const k of Object.keys(total))if(k.includes('mean_'))total[k]/=b.length;
  return total;
}
function main(){
  assert.equal(process.argv.length,5,'usage candidate_checkpoint candidate_sha --check|--run');const [candidate,candidateSha,mode]=process.argv.slice(2);assert(['--check','--run'].includes(mode));
  assert.equal(sha(read(baseCheckpoint)),baseSha);assert.equal(sha(read(candidate)),candidateSha);assert.equal(sha(read(mask)),'59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b');
  assert.equal(sha(read(exe)),'61e8dc64fd2bafec4084d470f0c048ee4e0a7bbebf17b66b0628fed7f260d8b2');
  const input=read(episode+'/worker.stdin.jsonl'),actualBytes=read(episode+'/worker.stdout.jsonl'),req=lines(input),requests=req.filter(q=>q.type==='step'&&!q.terminal),actualRows=lines(actualBytes);
  assert.equal(requests.length,1702);assert(requests.every(q=>q.observation_schema==='rek.native5.scaled_polar_xy.balance8_v1'&&q.round_id===requests[0].round_id));
  assert.equal(actualRows.find(r=>r.type==='ready').checkpoint_sha256,baseSha);const actual=new Map(actualRows.filter(r=>r.type==='action').map(r=>[r.seq,r]));
  const plan={baseline_argv:[exe,baseCheckpoint,baseSha,'1001',mask],candidate_argv:[exe,candidate,candidateSha,'1001',mask],source_stdin_sha256:sha(input),source_stdout_sha256:sha(actualBytes),worker_seed:1001,rows:requests.length,full223_allones:true,observations_and_legal_masks_unchanged:true,not_a_counterfactual_game_rollout:true};
  console.log(JSON.stringify(plan));if(mode==='--check')return;
  const out=path.join(stage,'live-drift');assert(!fs.existsSync(out));fs.mkdirSync(out);const results=[];
  for(const [label,argv] of [['baseline',plan.baseline_argv],['candidate',plan.candidate_argv]]){
    const started=process.hrtime.bigint(),r=cp.spawnSync(argv[0],argv.slice(1),{input,maxBuffer:32*1024*1024,timeout:90000});
    fs.writeFileSync(path.join(out,label+'.stdout.jsonl'),r.stdout??'',{flag:'wx'});fs.writeFileSync(path.join(out,label+'.stderr.txt'),r.stderr??'',{flag:'wx'});
    fs.writeFileSync(path.join(out,label+'.execution.json'),JSON.stringify({exit_code:r.status,signal:r.signal,error:r.error?.message??null,wall_seconds:Number(process.hrtime.bigint()-started)/1e9})+'\n',{flag:'wx'});
    assert.equal(r.status,0,String(r.stderr));results.push(lines(r.stdout));
  }
  const report={...plan,...compare(requests,...results,actual),schema:'rek.conditional_bc_full_live_drift.v1',binary_sha256:sha(read(exe)),checkpoint_sha256:candidateSha,environment_steps:0,game_connection:false,optimizer_updates:0,limitation:'Fixed recorded observation stream. Changed sampled attacks do not imply a closed-loop game count or timing guarantee.'};
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(report));
}
if(require.main===module){try{main();}catch(e){console.error(e.stack);process.exitCode=1;}}
module.exports={probabilities,compare};
