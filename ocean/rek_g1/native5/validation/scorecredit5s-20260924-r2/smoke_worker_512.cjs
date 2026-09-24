'use strict';
const fs=require('node:fs'),cp=require('node:child_process'),crypto=require('node:crypto'),readline=require('node:readline'),assert=require('node:assert/strict');
const stage='/home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2';
const schema='rek.native5.scaled_polar_xy.balance8_v1',expected='0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12';
const checkpoint=stage+'/train-score-delta/policy.bin',worker='/home/spark-advantage/rek-training/balance8-authentic-20260924-r1/build/live-policy-worker-balance8';
const source='/home/spark-advantage/rek-training/balance8-live-20260924-r1/balance8-s901-retry4/trial/worker.stdin.jsonl';
const mask='/home/spark-advantage/rek-training/joint-mask-transfer-20260924-r1/all-ones.bin';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex'),pin=(p,h)=>assert.equal(hash(fs.readFileSync(p)),h);
const publish=(p,b)=>fs.writeFileSync(p,b,{flag:'wx'});
const describe=a=>{const s=[...a].sort((x,y)=>x-y);return {count:s.length,minimum:s[0],median:s[Math.floor((s.length-1)*.5)],p95:s[Math.floor((s.length-1)*.95)],maximum:s.at(-1)};};
async function main(){
 pin(checkpoint,expected);pin(worker,'52741aed67037073bff5ecb21af63864550cba93a316dfe59a41b8b50126d7b2');pin(mask,'59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b');
 const sourceBytes=fs.readFileSync(source),requests=sourceBytes.toString().trim().split('\n').map(JSON.parse).filter(x=>x.type==='step'&&!x.terminal).slice(0,512);
 assert.equal(requests.length,512);assert(requests.every(x=>x.observation_schema===schema&&x.round_id===requests[0].round_id));
 const out=stage+'/worker-smoke-512';fs.mkdirSync(out);
 const args=[checkpoint,expected,'901','sampled',mask,'--observation-schema='+schema],started=new Date().toISOString(),begin=process.hrtime.bigint();
 const child=cp.spawn(worker,args,{stdio:['pipe','pipe','pipe']});let stderr='';child.stderr.on('data',b=>{stderr+=b});
 const ended=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});
 const timeout=setTimeout(()=>child.kill('SIGTERM'),90000),lines=readline.createInterface({input:child.stdout,crlfDelay:Infinity})[Symbol.asyncIterator]();
 const output=[],latencies=[],gpu=[],chosen=[];
 async function next(){const x=await lines.next();assert(!x.done,'worker stdout ended early');const r=JSON.parse(x.value);output.push(r);return r;}
 try{
  const ready=await next();assert.equal(ready.type,'ready');assert.equal(ready.seed,901);assert.equal(ready.native_cuda,true);assert.equal(ready.environment_stepping,false);
  const identity=r=>{assert.equal(r.checkpoint_sha256,expected);assert.equal(r.observation_schema,schema);assert.equal(r.selection,'sampled');assert.equal(r.precision,'bf16');assert.equal(r.feature_mask_sha256,hash(fs.readFileSync(mask)));};identity(ready);
  const startupMs=Number(process.hrtime.bigint()-begin)/1e6;
  for(let i=0;i<requests.length;i++){
   const q=requests[i],t=process.hrtime.bigint();child.stdin.write(JSON.stringify(q)+'\n');const a=await next();latencies.push(Number(process.hrtime.bigint()-t)/1e6);
   assert.equal(a.type,'action');identity(a);assert.equal(a.seq,q.seq);assert.equal(a.round_id,q.round_id);assert.equal(a.decision_index,i+1);assert.equal(a.recurrent_reset,i===0);assert.equal(a.round_changed,i===0);assert.equal(q.mask[a.action],1);assert.equal(a.legal_actions,q.mask.reduce((x,y)=>x+y,0));
   assert(Number.isFinite(a.gpu_ms)&&a.gpu_ms>=0);gpu.push(a.gpu_ms);chosen.push(a.action);
  }
  child.stdin.end(JSON.stringify({type:'close',seq:requests.at(-1).seq+1})+'\n');assert.equal((await next()).type,'closed');const status=await ended;assert.equal(status.code,0,stderr);assert.equal(status.signal,null);
  const report={schema:'rek.native_worker_latency_smoke.v1',passed:true,started_utc:started,finished_utc:new Date().toISOString(),full_process_wall_seconds:Number(process.hrtime.bigint()-begin)/1e9,startup_to_ready_ms:startupMs,requests:requests.length,
   request_mode:'one request at a time; await its action before submitting the next; recorded chronology and first-round reset preserved',checkpoint_sha256:expected,observation_schema:schema,worker_argv:[worker,...args],worker_sha256:hash(fs.readFileSync(worker)),source:{path:source,sha256:hash(sourceBytes),first_seq:requests[0].seq,last_seq:requests.at(-1).seq},
   request_to_action_wall_ms:describe(latencies),native_gpu_ms:describe(gpu),all_actions_legal:true,all_schema_checkpoint_mask_seed_and_recurrent_checks_passed:true,first_action:chosen[0],last_action:chosen.at(-1),exit_code:status.code,
   contention:'Root reported a concurrent unrelated wan-i2v GPU job; no process, job, GPU workload, or runtime settings were changed by this smoke.',environment_steps:0,game_connection:false,encoder_invoked:false,optimizer_updates:0};
  publish(out+'/worker.stdin.jsonl',requests.map(JSON.stringify).join('\n')+'\n');publish(out+'/worker.stdout.jsonl',output.map(JSON.stringify).join('\n')+'\n');publish(out+'/worker.stderr.txt',stderr);
  publish(out+'/latencies.json',JSON.stringify({request_to_action_wall_ms:latencies,native_gpu_ms:gpu})+'\n');publish(out+'/report.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{clearTimeout(timeout);if(child.exitCode===null)child.kill('SIGTERM');}
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
