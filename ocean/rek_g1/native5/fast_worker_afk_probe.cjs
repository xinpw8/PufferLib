'use strict';
// Exercise the actual native human-evaluator protocol in isolated processes.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {NativeWorker}=require('../league/worker.cjs');
const [executable,baseConfig,checkpoint,sha256,output]=process.argv.slice(2);
assert(executable&&baseConfig&&checkpoint&&/^[a-f0-9]{64}$/.test(sha256||'')&&output,
  'Usage: fast_worker_afk_probe.cjs EXECUTABLE BASE_CONFIG CHECKPOINT SHA256 NEW_OUTPUT');
for(const filename of [executable,baseConfig,checkpoint,output])assert(path.isAbsolute(filename),'Use absolute paths');
const hash=filename=>crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
assert.equal(hash(checkpoint),sha256,'Checkpoint SHA-256 mismatch');
fs.mkdirSync(output,{mode:0o700});
const base=JSON.parse(fs.readFileSync(baseConfig,'utf8'));
const env={REK_PHYSICS_BACKEND:'semantic_cuda',REK_FAST_OPPONENT_MODE:'neutral',REK_FAST_RANDOM_RESETS:'0',
  REK_FAST_SHAPING_WEIGHT:'0',REK_FAST_SHAPING_GAMMA:'.999',REK_FAST_SHAPING_TARGET:'.65',
  REK_FAST_SHAPING_BEARING_WEIGHT:'0',REK_FAST_RESET_GAP_MIN:'.55',REK_FAST_RESET_GAP_MAX:'2.5',
  REK_FAST_RESET_HEADING_SPREAD_RAD:'3.14159265',REK_FAST_MOVE_SPEED:'1',REK_FAST_YAW_SPEED:'1.8',
  REK_FAST_BODY_RADIUS:'.22',REK_FAST_HIT_SPEED:'.35'};
const cases=[];
function save(filename,data){fs.writeFileSync(path.join(output,filename),JSON.stringify(data,null,2)+'\n',{flag:'wx',mode:0o600});}
async function run(side,seconds){
  const name=`side${side}-${seconds}s`,configPath=path.join(output,name+'.private.json');
  fs.writeFileSync(configPath,JSON.stringify({...base,arenas:1,seed:73,round_seconds:seconds}),{flag:'wx',mode:0o600});
  const logFile=path.join(output,name+'.stderr.txt');
  const worker=new NativeWorker({executable,config:configPath,env,logFile});
  const trace=fs.openSync(path.join(output,name+'.trace.jsonl'),'wx',0o600);
  const actionCounts=Array(33).fill(0),routeTicks={},entryMoves={},scoreEvents=[];
  const offset=223*side,other=1-side;
  let start,last,state,busyTicks=0,facingTicks=0,pathLength=0,minGap=Infinity,maxGap=0,attackEntries=0,policyPoints=0,firstHit=null;
  const begin=process.hrtime.bigint();
  try{
    await worker.ready;
    assert.match(fs.readFileSync(logFile,'utf8'),/semantic_cuda_v4:/,'Worker did not identify the V4 runtime');
    const loaded=await worker.request('policy',{side,checkpoint,sha256,hiddenSize:256,layers:2,precision:'bf16',
      deterministic:false,seed:73,observationEncoding:'scaled_polar_xy',recurrentResetTicks:0});
    assert.equal(loaded.sha256,sha256);start=loaded.state;last=start;
    for(let i=0;i<seconds*50+2;i++){
      state=(await worker.request('step',{humanSide:other,action:1,steps:1})).state;
      assert.equal(state.failureBits,0);assert.equal(state.ok,true);assert.equal(state.actions[other],1);
      assert(state.raw.every(Number.isFinite));
      const o=state.raw.slice(offset,offset+223),previous=last.raw.slice(offset,offset+223);
      const dx=o[86]-o[0],dy=o[87]-o[1],gap=Math.hypot(dx,dy),yaw=2*Math.atan2(o[175],o[172]);
      const bearing=Math.atan2(Math.sin(Math.atan2(dy,dx)-yaw),Math.cos(Math.atan2(dy,dx)-yaw));
      minGap=Math.min(minGap,gap);maxGap=Math.max(maxGap,gap);if(Math.abs(bearing)<.16)facingTicks++;
      if(o[183])busyTicks++;actionCounts[state.actions[side]]++;routeTicks[o[179]]=(routeTicks[o[179]]||0)+1;
      pathLength+=Math.hypot(o[0]-previous[0],o[1]-previous[1]);
      if(o[183]&&!previous[183]){attackEntries++;entryMoves[state.actions[side]]=(entryMoves[state.actions[side]]||0)+1;}
      const gained=state.score[side]-last.score[side];policyPoints+=gained;
      assert.equal(gained,o[217],'Reported policy contact points disagree with score delta');
      if(gained>0&&firstHit===null)firstHit=state.tick;
      const changed=state.score.some((n,j)=>n!==last.score[j]);
      const compact={tick:state.tick,score:state.score,falls:state.falls,actions:state.actions,
        roots:[state.raw.slice(0,2),state.raw.slice(223,225)],gap_m:gap,facing_error_rad:bearing,
        policy_contact_points:gained,busy:!!o[183],route:o[179],terminal:state.terminal};
      if(changed)scoreEvents.push(compact);
      if(i%10===0||state.terminal||changed)fs.writeSync(trace,JSON.stringify(compact)+'\n');
      last=state;if(state.terminal)break;
    }
    assert.equal(state.terminal,1);assert.equal(state.completedRounds,1);assert.equal(state.roundNumber,1);
    assert.equal(state.score[other],0,'Neutral fighter scored points');assert.deepEqual(state.falls,[0,0]);
    const result={runtime:'semantic_cuda_v4',worker_sha256:hash(executable),checkpoint_sha256:sha256,
      policy_side:side,opponent:'external neutral category1',round_seconds:seconds,seed:73,arenas:1,
      precision:'bf16',action_selection:'sampled',observation_encoding:'scaled_polar_xy',reset_randomization:false,
      terminal_tick:state.tick,score:state.score,falls:state.falls,winner:state.winner,round_result:state.roundResult,
      policy_contact_points:policyPoints,first_policy_hit_tick:firstHit,first_policy_hit_seconds:firstHit===null?null:firstHit*.02,
      score_events:scoreEvents,action_counts:actionCounts,route_ticks:routeTicks,attack_entries:attackEntries,
      attack_entry_categories:entryMoves,busy_ticks:busyTicks,facing_within_0_16_rad_ticks:facingTicks,
      facing_fraction:facingTicks/state.tick,min_gap_m:minGap,max_gap_m:maxGap,policy_root_path_length_m:pathLength,
      initial_roots:[start.raw.slice(0,2),start.raw.slice(223,225)],final_roots:[state.raw.slice(0,2),state.raw.slice(223,225)],
      protocol_wall_seconds:Number(process.hrtime.bigint()-begin)/1e9,failure_bits:0,shaping_weight:0,
      cpu_physics:false,python_runtime:false,live_user_interaction:false,training_sps:null};
    cases.push(result);console.log(JSON.stringify(result));
  }finally{await worker.close();fs.closeSync(trace);}
}
(async()=>{
  save('provenance.json',{command:[process.execPath,...process.argv.slice(1)],host:require('node:os').hostname(),
    executable_sha256:hash(executable),source_sha256:hash(__filename),base_config_sha256:hash(baseConfig),
    checkpoint_sha256:sha256,environment:env});
  for(const seconds of [20,300])for(const side of [0,1])await run(side,seconds);
  save('summary.json',{status:'completed',cases});
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
