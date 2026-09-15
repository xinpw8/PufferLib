'use strict';
// Isolated native GPU diagnosis. No policy, CPU physics, or live HTTP controls.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {NativeWorker}=require('../league/worker.cjs');
const [executable,baseConfig,output,mode='all']=process.argv.slice(2);
assert(executable&&baseConfig&&output,'Usage: fast_position_probe.cjs EXECUTABLE BASE_CONFIG NEW_OUTPUT');
assert(['all','paired_i'].includes(mode));
const expectedNoHitResets=process.env.REK_EXPECT_NO_HIT_RESETS;
assert(expectedNoHitResets===undefined||['0','1'].includes(expectedNoHitResets));
fs.mkdirSync(output,{mode:0o700});
const config=JSON.parse(fs.readFileSync(baseConfig,'utf8'));
config.arenas=1;config.round_seconds=300;
const configPath=path.join(output,'worker.private.json');
fs.writeFileSync(configPath,JSON.stringify(config,null,2)+'\n',{flag:'wx',mode:0o600});
const env={REK_PHYSICS_BACKEND:'semantic_cuda',REK_FAST_MOVE_SPEED:'1',REK_FAST_YAW_SPEED:'1.8',
  REK_FAST_BODY_RADIUS:'.22',REK_FAST_HIT_SPEED:'.35',REK_FAST_DOWN_DAMAGE:'4'};
const worker=new NativeWorker({executable,config:configPath,env,logFile:path.join(output,'worker.stderr.txt')});
const protocol=fs.openSync(path.join(output,'protocol.private.jsonl'),'wx',0o600);
const ticks=fs.openSync(path.join(output,'ticks.jsonl'),'wx',0o600);
const summaries=[];
let trial='',spawn,normalTimedReset=null;
const xy=(s,side)=>[s.raw[side*223],s.raw[side*223+1]];
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const gap=s=>distance(xy(s,0),xy(s,1));
async function request(op,args={}){
  fs.writeSync(protocol,JSON.stringify({trial,direction:'request',op,...args})+'\n');
  const reply=await worker.request(op,args);
  fs.writeSync(protocol,JSON.stringify({trial,direction:'response',...reply})+'\n');
  if(reply.state){assert.equal(reply.state.ok,true);assert.equal(reply.state.failureBits,0);assert(reply.state.raw.every(Number.isFinite));}
  return reply.state;
}
async function step(action,phase){
  const state=await request('step',{humanSide:1,action,steps:1});
  fs.writeSync(ticks,JSON.stringify({trial,phase,tick:state.tick,action,roots:[xy(state,0),xy(state,1)],
    gap_m:gap(state),score:state.score,falls:state.falls,busy:state.raw[223+183],route:state.raw[223+179],
    hits_this_tick:state.raw[223+221],down:state.raw[223+79],reset_elapsed_seconds:state.raw[223+208],
    roundNumber:state.roundNumber,terminal:state.terminal})+'\n');
  return state;
}
async function reset(name){trial=name;const state=await request('reset');spawn=[xy(state,0),xy(state,1)];return state;}
async function settle(){let state;for(let i=0;i<40;i++)state=await step(1,'settle');return state;}
async function attack(action,initial,limit=300){
  let previous=initial,totalHits=0,peakJump=0,firstBusy=-1,lastBusy=-1,hitEvents=[],fallEvents=[],warps=[];
  for(let i=0;i<limit;i++){
    const state=await step(i===0?action:0,'single_attack');
    if(state.raw[223+183]){if(firstBusy<0)firstBusy=state.tick;lastBusy=state.tick;}
    const hits=state.raw[223+221];totalHits+=hits;
    if(hits)hitEvents.push({tick:state.tick,offset:i+1,hits,score:state.score});
    for(let side=0;side<2;side++){
      const jump=distance(xy(state,side),xy(previous,side));peakJump=Math.max(peakJump,jump);
      if(state.falls[side]>previous.falls[side])fallEvents.push({tick:state.tick,offset:i+1,side,score:state.score});
      if(jump>.1)warps.push({tick:state.tick,offset:i+1,side,jump_m:jump,to_spawn_m:distance(xy(state,side),spawn[side]),
        from:xy(previous,side),to:xy(state,side),falls:state.falls,score:state.score});
    }
    previous=state;
  }
  const result={trial,action,single_input_edge:true,initial_gap_m:gap(initial),initial_spawn_distance_m:distance(xy(initial,1),spawn[1]),
    ticks_observed:limit,first_busy_tick:firstBusy,last_busy_tick:lastBusy,total_hits:totalHits,hit_events:hitEvents,fall_events:fallEvents,
    maximum_planar_step_m:peakJump,warps,final_score:previous.score,final_falls:previous.falls,final_spawn_distance_m:distance(xy(previous,1),spawn[1])};
  summaries.push(result);console.log(JSON.stringify(result));return result;
}
(async()=>{
  await worker.ready;
  if(mode==='all')for(let action=16;action<=32;action++){
    await reset(`far_action_${action}`);
    for(let t=0;t<70;t++)await step(3,'move_off_spawn');
    const initial=await settle();assert(initial.mask[33+action]===1);
    const result=await attack(action,initial,230);
    assert(result.initial_spawn_distance_m>.1,'Far fixture did not displace orange');
    assert(result.initial_gap_m>2,'Opponent was not far away');
    assert.equal(result.total_hits,0,'Unexpected far-range hit');
    assert.equal(result.warps.length,0,'Far attack caused a position warp');
    assert(result.maximum_planar_step_m<1e-5,'Far attack translated a settled root');
    assert(result.first_busy_tick>=0&&result.last_busy_tick<initial.tick+230,'Canned move did not start and finish');
  }
  if(mode==='all')for(const action of [17,18])for(const desired of [.45,.55,.65,.75,.85,.95,1.1,1.3]){
    let state=await reset(`close_action_${action}_gap_${desired}`);
    for(let t=0;t<500&&gap(state)>desired+.25;t++)state=await step(2,'approach');
    const initial=await settle();assert(initial.mask[33+action]===1);
    await attack(action,initial);
  }
  if(mode==='paired_i'){
    let state=await reset('paired_i_first_attack');
    for(let t=0;t<500&&gap(state)>.90;t++)state=await step(2,'approach');
    const first=await attack(18,await settle());
    for(let t=0;t<100;t++)await step(0,'no_input_between_attacks');
    trial='paired_i_second_attack';
    const second=await attack(18,await request('snapshot'));
    if(expectedNoHitResets!==undefined){
      assert.equal(first.total_hits,1);assert.equal(second.total_hits,1);
      assert.deepEqual(first.final_score,[0,1]);assert.deepEqual(first.final_falls,[0,0]);
      assert.equal(first.maximum_planar_step_m,0);assert.equal(first.warps.length,0);
      if(expectedNoHitResets==='1'){
        assert.deepEqual(second.final_score,[0,2],'Two single kicks must award two contact points');
        assert.deepEqual(second.final_falls,[0,0],'Contacts must not fabricate a knockdown');
        assert.equal(second.maximum_planar_step_m,0,'Contact sequence moved a settled root');
        assert.equal(second.warps.length,0,'Contact sequence reset fighter positions');
      }else{
        assert.deepEqual(second.final_score,[0,7]);assert.deepEqual(second.final_falls,[1,0]);
        assert.equal(second.warps.length,1);assert.equal(second.warps[0].side,1);
        assert.equal(second.warps[0].tick-second.hit_events[0].tick,25);
        assert(second.warps[0].jump_m>1);assert.equal(second.warps[0].to_spawn_m,0);
      }
      trial='normal_timed_round_reset';
      let terminal=await request('snapshot');
      const beforeRoots=[xy(terminal,0),xy(terminal,1)];
      for(let batch=0;batch<31&&!terminal.terminal;batch++)
        terminal=await request('step',{humanSide:1,action:1,steps:512,stopAtRound:true});
      assert.equal(terminal.terminal,1);assert.equal(terminal.timeRemaining,0);
      assert.equal(terminal.completedRounds,1);assert.equal(terminal.roundNumber,1);
      assert.deepEqual(terminal.score,second.final_score);
      assert.deepEqual([xy(terminal,0),xy(terminal,1)],beforeRoots,'Idle changed roots before timed completion');
      const resetState=await step(1,'legitimate_timed_reset');
      assert.equal(resetState.terminal,0);assert.equal(resetState.roundNumber,2);
      assert.equal(resetState.completedRounds,1);assert(resetState.timeRemaining>299);
      assert.deepEqual(resetState.score,[0,0]);assert.deepEqual(resetState.falls,[0,0]);
      for(let side=0;side<2;side++)assert(distance(xy(resetState,side),spawn[side])<1e-6);
      normalTimedReset={terminal_tick:terminal.tick,terminal_time_remaining:terminal.timeRemaining,
        terminal_score:terminal.score,terminal_falls:terminal.falls,completed_rounds:terminal.completedRounds,
        terminal_roots:[xy(terminal,0),xy(terminal,1)],reset_tick:resetState.tick,
        reset_round_number:resetState.roundNumber,reset_score:resetState.score,
        reset_roots:[xy(resetState,0),xy(resetState,1)],passed:true};
    }
  }
  const result={status:'completed',runtime:'semantic_cuda',round_seconds:300,arenas:1,
    expected_no_hit_resets:expectedNoHitResets===undefined?null:expectedNoHitResets==='1',
    mode,controlled_side:1,opponent:'neutral external category1',far_attacks_checked:summaries.filter(s=>s.trial.startsWith('far_')).length,
    far_attacks_with_warps:summaries.filter(s=>s.trial.startsWith('far_')&&s.warps.length).length,
    close_single_attack_trials:summaries.filter(s=>s.trial.startsWith('close_')).length,close_single_attacks_with_multiple_hits:summaries.filter(s=>s.trial.startsWith('close_')&&s.total_hits>1).length,
    close_single_attacks_with_knockdowns:summaries.filter(s=>s.trial.startsWith('close_')&&s.fall_events.length).length,
    close_single_attacks_with_spawn_warps:summaries.filter(s=>s.trial.startsWith('close_')&&s.warps.length).length,
    cpu_physics:false,python_runtime:false,production_interaction:false};
  fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify({summary:result,trials:summaries,normal_timed_reset:normalTimedReset},null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(result));
})().catch(error=>{console.error(error.stack);process.exitCode=1;}).finally(async()=>{
  await worker.close();fs.closeSync(protocol);fs.closeSync(ticks);
});
