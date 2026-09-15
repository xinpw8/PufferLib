'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');

// This smoke intentionally targets only the isolated candidate on port18770.
// It never selects a production backend or sends Windows desktop input.
const stage=path.resolve(process.argv[2]);
const config=JSON.parse(fs.readFileSync(path.join(stage,'test-run','server.json'),'utf8'));
assert.equal(config.port,18770,'Refusing to access any production evaluation port');
const origin='http://127.0.0.1:18770';
const {NativeWorker}=require(path.join(stage,'league','worker.cjs'));
const {HumanSession}=require(path.join(stage,'league','human_session.cjs'));
const backend=config.backends.find(value=>value.id==='semantic_cuda');
assert(backend,'Expected isolated semantic CUDA candidate');
const hash=filename=>crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const delay=milliseconds=>new Promise(resolve=>setTimeout(resolve,milliseconds));
let seq=Date.now()*1024,privateWorker=null;
const result={schema:'rek.human_viewer.native_smoke.v1',host:os.hostname(),architecture:process.arch,
  tested_port:18770,production_ports_touched:[],python_runtime:false,cpu_physics_steps:false,training_benchmark:false,
  script_sha256:hash(__filename),native_worker_sha256:hash(backend.executable)};
async function api(endpoint,body){
  const response=await fetch(origin+endpoint,{signal:AbortSignal.timeout(15000),
    ...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json',Origin:origin},body:JSON.stringify(body)})});
  const value=await response.json();assert.equal(response.status,200,`${endpoint}: ${JSON.stringify(value)}`);
  assert.notEqual(value.ok,false,`${endpoint}: ${JSON.stringify(value)}`);return value;
}
const input=(held,move=null)=>api('/api/input',{seq:++seq,held,move});
function summary(state){return {tick:state.tick,time_remaining_seconds:state.timeRemaining,paused:state.paused,
  actions:state.actions,score:state.score,falls:state.falls,failure_bits:state.failureBits,
  move_disposition:state.moveDisposition??null,route:state.raw?.[179],move_busy:state.raw?.[182]};}
async function until(predicate){
  const deadline=Date.now()+10000;
  while(Date.now()<deadline){
    const state=await api('/api/state');assert.equal(state.failureBits,0);assert.equal(state.ok,true);
    if(predicate(state))return state;await delay(15);
  }
  throw new Error('Native state condition did not become observable within 10 seconds');
}
async function run(){
  const health=await api('/health'),initial=await api('/api/state');
  assert.equal(health.ok,true);assert.equal(initial.paused,true);assert.equal(initial.tick,0);
  assert.equal(initial.timeRemaining,300);assert.equal(initial.active.humanSide,0);
  result.initial=summary(initial);
  const keys=['W','S','A','D','Q','E'];let accepted=0;
  for(let bits=0;bits<64;bits++){
    const response=await input(keys.filter((_,index)=>bits&(1<<index)));
    assert.equal(response.accepted,true);accepted++;
  }
  await input([]);await delay(120);const paused=await api('/api/state');
  assert.equal(paused.tick,0);assert.equal(paused.timeRemaining,300);assert.equal(paused.paused,true);
  result.paused_input={accepted_chords:accepted,total_chords:64,rejections:0,gpu_state_steps:0};

  await api('/api/play',{paused:false});await input(['W','A','Q']);
  const moved=await until(state=>state.actions[0]===8&&state.tick>=10);
  assert(Math.hypot(moved.raw[0]-initial.raw[0],moved.raw[1]-initial.raw[1])>0,'Held chord did not move the robot');
  result.held_diagonal_yaw=summary(moved);
  await input(['Q']);const settled=await until(state=>state.actions[0]===6&&state.mask[17]===1);
  await input(['Q'],17);const attack=await until(state=>state.moveDisposition==='accepted'&&state.raw[182]===1);
  result.attack={...summary(attack),translation_released_before_edge:true,held_yaw_before_attack:summary(settled),
    attack_edges_sent:1,attack_category:17};
  await api('/api/play',{paused:true});await input([]);const stopped=await api('/api/state');
  await delay(120);const stable=await api('/api/state');assert.equal(stable.tick,stopped.tick);
  assert.equal(stable.paused,true);assert.equal(stable.failureBits,0);result.final_paused=summary(stable);

  const originalHash=hash(backend.workerConfig);
  const originalConfig=JSON.parse(fs.readFileSync(backend.workerConfig,'utf8'));
  assert.equal(originalConfig.round_seconds,20,'Private terminal probe must use original20s config');
  privateWorker=new NativeWorker({executable:backend.executable,config:backend.workerConfig,env:backend.env,
    logFile:path.join(stage,'private-terminal-worker.stderr.txt')});
  await privateWorker.ready;let native=(await privateWorker.request('snapshot')).state;
  assert.equal(native.tick,0);assert.equal(native.timeRemaining,20);
  let requests=0,completed=null;
  while(!native.terminal&&requests<4){
    const response=await privateWorker.request('step',{action:1,humanSide:0,steps:512,stopAtRound:true});
    native=response.state;requests++;assert.equal(native.failureBits,0);
    if(response.rounds?.length){assert.equal(response.rounds.length,1);completed=response.rounds[0];}
  }
  assert.equal(native.terminal,1);assert(completed,'Native worker omitted completed terminal record');
  assert.equal(completed.tick,native.tick);assert.deepEqual(completed.score,native.score);
  const session=new HumanSession();assert.equal(session.record(completed),true);
  assert.equal(session.record(native),false,'Same terminal was double-counted across native outputs');
  const totals=session.snapshot();assert.equal(totals.completedRounds,1);assert.equal(totals.invalidRounds,0);
  assert.equal(totals.bluePoints,native.score[0]);assert.equal(totals.orangePoints,native.score[1]);
  assert.equal(totals.lastRound.winner,native.winner<0?null:native.winner);
  result.native_terminal={requests,chunk_limit:512,stop_at_round:true,tick:native.tick,
    round_number:native.roundNumber,round_result:native.roundResult,winner:native.winner,score:native.score,
    falls:native.falls,failure_bits:native.failureBits,scoreboard:totals,duplicate_record_counted:false,
    original_worker_config_sha256:originalHash};
  await privateWorker.close();privateWorker=null;
  assert.equal(hash(backend.workerConfig),originalHash,'Training worker config changed');
  result.ok=true;
}
run().catch(error=>{result.ok=false;result.error=error.message;process.exitCode=1;}).finally(async()=>{
  if(privateWorker)await privateWorker.close();
  try{await api('/api/play',{paused:true});await input([]);}catch(error){result.final_pause_error=error.message;process.exitCode=1;}
  fs.writeFileSync(path.join(stage,'native-smoke.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(result));
});
