'use strict';
// Diagnostic only: private isolated GPU worker; no live service requests.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const base='/home/spark-advantage/rek-training/semantic-fast-20260914-v1';
const {NativeWorker}=require(base+'/human-viewer-v3-20260915/league/worker.cjs');
const out=process.argv[2];assert(out&&path.isAbsolute(out));fs.mkdirSync(out,{mode:0o700});
const cases=[];
async function run(version,seconds){
  const privateConfig=JSON.parse(fs.readFileSync(base+`/eval-run-${version}/server.json`));
  const backend=privateConfig.backends[0];
  const league=JSON.parse(fs.readFileSync(privateConfig.leagueFile));
  const policy=league.policies[`semantic_cuda/compact-${version}-33m-bf16-sampled`];
  assert(policy);
  const config={...JSON.parse(fs.readFileSync(backend.workerConfig)),arenas:1,round_seconds:seconds};
  const name=version+'-'+seconds,configPath=path.join(out,name+'.private.json');
  fs.writeFileSync(configPath,JSON.stringify(config),{flag:'wx',mode:0o600});
  const worker=new NativeWorker({executable:backend.executable,config:configPath,env:backend.env,logFile:path.join(out,name+'.stderr.txt')});
  const trace=fs.openSync(path.join(out,name+'.trace.jsonl'),'wx',0o600);
  const counts=Array(33).fill(0),routes={},entryMoves={},scoreEvents=[];
  let s,start,last,busyTicks=0,facingTicks=0,totalTravel=0,minGap=Infinity,maxGap=0,attackEntries=0;
  try{
    await worker.ready;
    const loaded=await worker.request('policy',{side:0,checkpoint:policy.checkpoint.path,sha256:policy.checkpoint.sha256,
      hiddenSize:256,layers:2,precision:'bf16',deterministic:false,seed:73,observationEncoding:'scaled_polar_xy'});
    assert.equal(loaded.sha256,policy.checkpoint.sha256);
    start=loaded.state;last=start;
    for(let i=0;i<seconds*50+500;i++){
      s=(await worker.request('step',{humanSide:1,action:1,steps:1})).state;
      assert.equal(s.failureBits,0);assert.equal(s.ok,true);assert.equal(s.actions[1],1);
      const o=s.raw,dx=o[223]-o[0],dy=o[224]-o[1],gap=Math.hypot(dx,dy);
      const yaw=2*Math.atan2(o[175],o[172]),bearing=Math.atan2(Math.sin(Math.atan2(dy,dx)-yaw),Math.cos(Math.atan2(dy,dx)-yaw));
      minGap=Math.min(minGap,gap);maxGap=Math.max(maxGap,gap);if(Math.abs(bearing)<.16)facingTicks++;
      if(o[183])busyTicks++;
      counts[s.actions[0]]++;routes[o[179]]=(routes[o[179]]||0)+1;
      totalTravel+=Math.hypot(o[0]-last.raw[0],o[1]-last.raw[1]);
      if(o[183]&&!last.raw[183]){attackEntries++;entryMoves[s.actions[0]]=(entryMoves[s.actions[0]]||0)+1;}
      const compact={tick:s.tick,score:s.score,falls:s.falls,actions:s.actions,root:[o.slice(0,2),o.slice(223,225)],gap_m:gap,
        facing_error_rad:bearing,busy:!!o[183],route:o[179],terminal:s.terminal};
      if(s.score.some((n,j)=>n!==last.score[j]))scoreEvents.push(compact);
      if(i%10===0||s.terminal||s.score.some((n,j)=>n!==last.score[j]))fs.writeSync(trace,JSON.stringify(compact)+'\n');
      last=s;if(s.terminal)break;
    }
    assert.equal(s.terminal,1);
    const result={version,round_seconds:seconds,seed:73,policy_side:0,opponent:'external neutral category1',checkpoint_sha256:policy.checkpoint.sha256,
      terminal_tick:s.tick,score:s.score,falls:s.falls,winner:s.winner,round_result:s.roundResult,
      first_score_tick:scoreEvents[0]?.tick??null,score_events:scoreEvents,action_counts:counts,route_ticks:routes,
      attack_entries:attackEntries,attack_entry_categories:entryMoves,busy_ticks:busyTicks,facing_within_0_16_rad_ticks:facingTicks,
      min_gap_m:minGap,max_gap_m:maxGap,policy_root_path_length_m:totalTravel,
      initial_root:[start.raw.slice(0,2),start.raw.slice(223,225)],final_root:[s.raw.slice(0,2),s.raw.slice(223,225)],
      cpu_physics:false,python_runtime:false,live_user_interaction:false};
    cases.push(result);console.log(JSON.stringify(result));
  }finally{await worker.close();fs.closeSync(trace);}
}
(async()=>{
  for(const version of ['v2','v3'])for(const seconds of [20,300])await run(version,seconds);
  fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify({cases},null,2)+'\n',{flag:'wx'});
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
