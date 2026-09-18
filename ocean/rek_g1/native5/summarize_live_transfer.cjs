#!/usr/bin/env node
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const readline=require('node:readline');
async function records(file,visit){for await(const line of readline.createInterface({input:fs.createReadStream(file)})){try{visit(JSON.parse(line));}catch(e){if(!(e instanceof SyntaxError))throw e;}}}
const percentile=(xs,p)=>xs.length?[...xs].sort((a,b)=>a-b)[Math.min(xs.length-1,Math.floor(xs.length*p))]:null;
// Native CleanHits is an awarded-point total. Preserve the source field names.
const awardedPoints=round=>Array.isArray(round?.clean_hits)&&round.clean_hits.length===2&&round.clean_hits.every(Number.isSafeInteger)?round.clean_hits.slice():null;
async function main(dir){
  const summary=JSON.parse(fs.readFileSync(path.join(dir,'summary.json')));
  let first,last,snapshots=0,dispatches=0,frames=new Set(),attacks=0,appliedAttacks=0;
  const latency=[],gpu=[],requestAges=[],displacements=[0,0];
  await records(path.join(dir,'relay.stdout.jsonl'),x=>{
    if(x.event==='g1_policy_state'){
      first??=x;last=x;snapshots++;frames.add(x.clock.unity_frame);
      for(let i=0;i<2;i++)displacements[i]=Math.max(displacements[i],Math.hypot(...x.fighters[i].root_position_xyz.map((v,j)=>v-first.fighters[i].root_position_xyz[j])));
    }
    if(x.event==='g1_policy_action'&&x.action>=16){attacks++;if(x.applied)appliedAttacks++;}
    if(x.event==='g1_policy_dispatch'&&x.send_method_returned===true)dispatches++;
  });
  await records(path.join(dir,'worker.stdout.jsonl'),x=>{if(x.type==='action'){latency.push(x.latency_ms);gpu.push(x.gpu_ms);}});
  const elapsed=first&&last?(last.clock.qpc_ticks-first.clock.qpc_ticks)/first.clock.qpc_frequency_hz:null;
  const report={...summary,source_elapsed_wall_seconds:elapsed,observed_round_elapsed_seconds:first&&last?first.round.time_remaining-last.round.time_remaining:null,
    live_decisions_per_second:elapsed>0?summary.predictions/elapsed:null,distinct_render_frames:frames.size,
    local_slot:first?.local_slot??null,local_attack_requests:attacks,local_attack_applied_returns:appliedAttacks,native_move_send_returns:dispatches,
    maximum_root_displacement_metres:displacements,worker_latency_ms:{median:percentile(latency,.5),p95:percentile(latency,.95),max:latency.length?Math.max(...latency):null},
    gpu_event_ms:{median:percentile(gpu,.5),p95:percentile(gpu,.95)},
    awarded_points:{initial_by_slot:awardedPoints(summary.initial_round??first?.round),final_by_slot:awardedPoints(summary.final_round??last?.round)},
    score_semantics:{native_counter_unit:'cumulative_integer_awarded_points',includes_clean_strike_and_referee_awards:true,
      native_increment:'truncate_toward_zero(pointsAwarded)',clean_strike_event_count:null,clean_strike_event_count_inferred:false,
      award_cause_recoverable_from_counters:false,provenance:'live_transfer/SCORE_COUNTER_PROVENANCE.md'},
    evidence_limits:['Received client poses and replicated round counters; no authoritative server action-acceptance field.',
      'Client-pose feature projection and opt-in V4 requested-duration timing estimate are explicitly used.',
      'Native CleanHits/newHitCount are cumulative integer awarded points, including referee awards; their deltas are not strike-event counts.',
      'Aggregate score counters do not identify award causes or establish the number of clean strikes.',
      'Live decision rate is real-time client control rate, not headless training SPS.']};
  const hashes={};
  for(const name of ['run-config.json','relay.stdin.jsonl','relay.stdout.jsonl','encoder.stdin.jsonl','encoder.stdout.jsonl','worker.stdin.jsonl','worker.stdout.jsonl','summary.json']){
    const hash=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(path.join(dir,name)))hash.update(chunk);hashes[name]=hash.digest('hex');
  }
  report.artifact_sha256=hashes;
  fs.writeFileSync(path.join(dir,'measured-report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(report));
}
main(process.argv[2]).catch(e=>{console.error(e.message);process.exitCode=1;});
