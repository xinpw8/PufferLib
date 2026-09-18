'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const {test}=require('node:test');
const script=path.join(__dirname,'summarize_live_transfer.cjs');
const files=['run-config.json','relay.stdin.jsonl','relay.stdout.jsonl','encoder.stdin.jsonl','encoder.stdout.jsonl','worker.stdin.jsonl','worker.stdout.jsonl','summary.json'];
function source(index,round){return {event:'g1_policy_state',local_slot:0,clock:{unity_frame:index,qpc_ticks:index*1000,qpc_frequency_hz:1000},round,
  fighters:[{root_position_xyz:[index/10,0,0]},{root_position_xyz:[1,0,0]}]};}
function fixture(t,summary,states){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-score-summary-'));
  t.after(()=>fs.rmSync(dir,{recursive:true,force:true}));
  for(const name of files)fs.writeFileSync(path.join(dir,name),'');
  fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify(summary));
  fs.writeFileSync(path.join(dir,'relay.stdout.jsonl'),states.map(x=>JSON.stringify(x)).join('\n')+'\n');
  fs.writeFileSync(path.join(dir,'worker.stdout.jsonl'),JSON.stringify({type:'action',latency_ms:1,gpu_ms:.5})+'\n');
  const before=fs.readFileSync(path.join(dir,'summary.json'));
  const run=spawnSync(process.execPath,[script,dir],{encoding:'utf8'});
  assert.equal(run.status,0,run.stderr);
  assert.deepEqual(fs.readFileSync(path.join(dir,'summary.json')),before);
  const report=JSON.parse(fs.readFileSync(path.join(dir,'measured-report.json'),'utf8'));
  assert.deepEqual(JSON.parse(run.stdout),report);
  return {dir,report};
}
test('weighted kick and referee awards remain points, while raw summary fields stay unchanged',t=>{
  const initial={number:1,active:true,clean_hits:[0,0],time_remaining:120};
  const final={number:1,active:false,clean_hits:[11,3],time_remaining:116};
  const summary={predictions:4,initial_round:initial,final_round:final,checkpoint_sha256:'a'.repeat(64)};
  const states=[2,4,6,11].map((points,i)=>source(i,{number:1,clean_hits:[points,i===3?3:0],time_remaining:120-i}));
  const {report}=fixture(t,summary,states);
  for(const [key,value] of Object.entries(summary))assert.deepEqual(report[key],value);
  assert.deepEqual(report.awarded_points,{initial_by_slot:[0,0],final_by_slot:[11,3]});
  assert.equal(report.score_semantics.native_counter_unit,'cumulative_integer_awarded_points');
  assert.equal(report.score_semantics.native_increment,'truncate_toward_zero(pointsAwarded)');
  assert.equal(report.score_semantics.includes_clean_strike_and_referee_awards,true);
  assert.equal(report.score_semantics.clean_strike_event_count,null);
  assert.equal(report.score_semantics.clean_strike_event_count_inferred,false);
  assert.equal(report.score_semantics.award_cause_recoverable_from_counters,false);
  assert(report.evidence_limits.some(x=>x.includes('deltas are not strike-event counts')));
  assert(!report.evidence_limits.some(x=>x.includes('reported as clean-hit counters')));
  assert.equal(Object.keys(report.artifact_sha256).length,files.length);
});
test('missing summary rounds use observed stream round totals without summing samples',t=>{
  const {report}=fixture(t,{predictions:2},[source(0,{clean_hits:[2,5],time_remaining:120}),source(1,{clean_hits:[7,5],time_remaining:119})]);
  assert.deepEqual(report.awarded_points,{initial_by_slot:[2,5],final_by_slot:[7,5]});
  assert.equal(report.initial_round,undefined);assert.equal(report.final_round,undefined);
});
test('unknown or invalid point counters remain null rather than invented zero totals',t=>{
  const {report}=fixture(t,{predictions:0,initial_round:{clean_hits:[1.5,0]},final_round:{clean_hits:['2',0]}},[]);
  assert.deepEqual(report.awarded_points,{initial_by_slot:null,final_by_slot:null});
  assert.deepEqual(report.initial_round.clean_hits,[1.5,0]);
  assert.deepEqual(report.final_round.clean_hits,['2',0]);
});
test('round reset totals remain separate and an existing measured report is preserved',t=>{
  const summary={predictions:2,initial_round:{number:1,clean_hits:[11,7]},final_round:{number:2,clean_hits:[0,0]}};
  const {dir,report}=fixture(t,summary,[]);
  assert.deepEqual(report.awarded_points,{initial_by_slot:[11,7],final_by_slot:[0,0]});
  const before=fs.readFileSync(path.join(dir,'measured-report.json'));
  const rerun=spawnSync(process.execPath,[script,dir],{encoding:'utf8'});
  assert.equal(rerun.status,1);assert.deepEqual(fs.readFileSync(path.join(dir,'measured-report.json')),before);
});
