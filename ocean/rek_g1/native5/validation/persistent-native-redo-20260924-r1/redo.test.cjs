'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const driver=require('./live_transfer_run_masked.cjs'),oldDriver=require('./baseline/live_transfer_run_masked.cjs');
const code=fs.readFileSync(__dirname+'/campaign.cjs','utf8'),oldCode=fs.readFileSync(__dirname+'/baseline/campaign.cjs','utf8');
const cp='a'.repeat(64),opponent={client_ai_difficulty:0,sparring_bot_number:1};
function source(i=1,redo=false){return {event:'g1_policy_state',schema:'rek.g1_policy_source.v1',
 stream_active:false,global_input_emitted:false,phase:1,round_identity_sha256:'b'.repeat(64),local_slot:0,
 observation_sequence:i,clock:{qpc_ticks:10000000+i*500000,qpc_frequency_hz:10000000,unity_frame:100+i},
 opponent:{...opponent,opponent_is_ai:true,human_in_opponent_slot:false},
 round:{number:2,duration:redo?30:120,time_remaining:(redo?30:120)-i*.05,active:true,redo,result_value:0,clean_hits:[0,0]},
 input:{active:true,pending_move:false,pending_special:false,pending_estop:false,punching:false,recovering:false,velocity_command_xyz:[0,0,0]}};}
function warmOptions(redo=false,playNativeRedo=false,mutate=()=>{}){let now=0,n=0;const logs=[];
 return {logs,options:{opponent,playNativeRedo,now:()=>now,wait:async ms=>{now+=ms;},log:(event,data)=>logs.push({event,...data}),
 pollSource:async()=>{now+=40;const s=source(++n,redo);mutate(s,n);return s;}}};}
function extract(text,from,to,context={}){return vm.runInNewContext(text.slice(text.indexOf(from),text.indexOf(to,text.indexOf(from)))+'\n'+from.match(/function (\w+)/)[1],context);}
test('default continues to reject native30s redo, explicit opt-in accepts measured sample',()=>{
 const s=source(1,true);s.round.time_remaining=29.749779;
 assert.throws(()=>driver.validateStartupRound(s,2.5),/120s/);
 assert.equal(driver.validateStartupRound(s,2.5,true).kind,'redo_30s');
 assert.deepEqual(s.round.clean_hits,[0,0]);assert.equal(s.round.duration,30);assert.equal(s.round.redo,true);
});
test('only exact120/nonredo or30/redo contracts accepted with same zero-score start budget',()=>{
 for(const mutate of [s=>s.round.duration=60,s=>s.round.duration=120,s=>s.round.redo=false,
   s=>s.round.time_remaining=30.001,s=>s.round.time_remaining=27.499,s=>s.round.active=false,
   s=>s.round.clean_hits=[0,1],s=>s.round.result_value=1]){
   const s=source(1,true);mutate(s);assert.throws(()=>driver.validateStartupRound(s,2.5,true));
 }
 const exact=source(1,true);exact.round.time_remaining=27.5;assert.equal(driver.validateStartupRound(exact,2.5,true).kind,'redo_30s');
});
test('prewarm latches redo kind without any encoder or worker requests',async()=>{
 const f=warmOptions(true,true),ready=await driver.prewarmPolicySource(f.options);
 assert.equal(ready.round_kind,'redo_30s');assert.equal(ready.round_duration,30);assert.equal(ready.round_redo,true);
 assert.equal(ready.play_native_redo,true);assert.equal(ready.samples,3);
 const log=f.logs.find(x=>x.event==='source_warmup_ready');assert.equal(log.worker_requests,0);assert.equal(log.encoder_requests,0);
 const s=source(4,true);s.stream_active=true;driver.validateControlledStartup(s,ready);
 s.round.time_remaining=26.999;assert.throws(()=>driver.validateControlledStartup(s,ready));
});
test('changed native round kind during prewarm or at control start rejects',async()=>{
 const f=warmOptions(true,true,(s,n)=>{if(n===2){s.round.duration=120;s.round.redo=false;s.round.time_remaining=119.9;}});
 await assert.rejects(driver.prewarmPolicySource(f.options),/contract_changed/);
 const ready=await driver.prewarmPolicySource(warmOptions(true,true).options),s=source(4,false);s.stream_active=true;
 assert.throws(()=>driver.validateControlledStartup(s,ready),/contract_changed/);
});
test('private opponent, neutral input and source identity checks stay enforced on redos',async()=>{
 for(const mutate of [s=>s.opponent.human_in_opponent_slot=true,s=>s.input.pending_move=true,
   s=>s.input.velocity_command_xyz=[1,0,0],s=>s.phase=2]){
   await assert.rejects(driver.prewarmPolicySource(warmOptions(true,true,mutate).options));
 }
 const ready=await driver.prewarmPolicySource(warmOptions(true,true).options),s=source(4,true);s.stream_active=true;
 s.round_identity_sha256='c'.repeat(64);assert.throws(()=>driver.validateControlledStartup(s,ready),/bound/);
});
test('regular120 default prewarm matches old measured readiness fields',async()=>{
 const before=await oldDriver.prewarmPolicySource(warmOptions(false,false).options);
 const after=await driver.prewarmPolicySource(warmOptions(false,false).options);
 for(const [k,v]of Object.entries(before))assert.deepEqual(after[k],v,k);
 assert.equal(after.round_kind,'regular_120s');assert.equal(after.play_native_redo,false);
 const s=source(4,false);s.stream_active=true;oldDriver.validateControlledStartup(s,before);driver.validateControlledStartup(s,after);
});
const verdict=extract(code,'function verdict(dir,expectedCheckpoint){','async function attempt(',{fs,path});
const oldVerdict=extract(oldCode,'function verdict(dir,expectedCheckpoint){','async function attempt(',{fs,path});
function summary(redo=false){return {predictions:20,applied:20,opponent,checkpoint_sha256:cp,
 play_native_redo:redo,controlled_startup_validated:true,startup_readiness:{round_kind:redo?'redo_30s':'regular_120s',
   round_duration:redo?30:120,round_redo:redo,play_native_redo:redo},
 initial_round:{number:2,duration:redo?30:120,time_remaining:redo?29.7:119.7,active:true,redo,clean_hits:[0,0]},
 final_round:{number:2,duration:redo?30:120,time_remaining:0,active:false,redo,clean_hits:[2,1],winner_index:0,result:'WonByPoints',result_value:1}};}
function fixture(s){const dir=fs.mkdtempSync(__dirname+'/test-output-');fs.mkdirSync(dir+'/trial');
 fs.writeFileSync(dir+'/trial/summary.json',JSON.stringify(s));return dir;}
test('fair terminal redo has separate auxiliary completion and never120 completion',()=>{
 const dir=fixture(summary(true)),v=verdict(dir,cp);assert.equal(v.auxiliary_complete,true);
 assert.equal(v.complete,false);assert.equal(v.fair_start,false);assert.equal(v.auxiliary_fair_start,true);assert.equal(v.round_kind,'redo_30s');
});
test('redo score/result/identity/start/opt-in mismatches never become auxiliary completion',()=>{
 for(const mutate of [s=>s.play_native_redo=false,s=>s.controlled_startup_validated=false,
   s=>s.startup_readiness.round_kind='regular_120s',s=>s.startup_readiness.play_native_redo=false,
   s=>s.initial_round.time_remaining=26.99,s=>s.initial_round.clean_hits=[1,0],s=>s.initial_round.number=1,
   s=>s.final_round.duration=120,s=>s.final_round.redo=false,s=>s.final_round.active=true,
   s=>s.final_round.time_remaining=.1,s=>s.final_round.winner_index=1,s=>s.opponent={client_ai_difficulty:1,sparring_bot_number:2},
   s=>s.checkpoint_sha256='f'.repeat(64),s=>s.applied=0]){
   const s=summary(true);mutate(s);const v=verdict(fixture(s),cp);assert.equal(v.auxiliary_complete,false);assert.equal(v.complete,false);
 }
});
test('old regular120 verdict fields and complete expression remain unchanged',()=>{
 const expression='const complete=terminal&&sameBot&&fairStart&&summary.checkpoint_sha256===expectedCheckpoint;';
 assert(oldCode.includes(expression)&&code.includes(expression));
 for(const mutate of [s=>{},s=>s.initial_round.time_remaining=116.9,s=>s.final_round.clean_hits=[1,1],s=>s.final_round.redo=true]){
   const s=summary();mutate(s);const dir=fixture(s),before=oldVerdict(dir,cp),after=verdict(dir,cp);
   for(const [key,value]of Object.entries(before))assert.equal(JSON.stringify(after[key]),JSON.stringify(value),key);
 }
});
test('auxiliary clean handoff requires same checkpoint and transport proof',()=>{
 const successor=extract(code,'function successorHandoff(rec,entry){','async function main(){');
 const rec={complete:false,auxiliary_complete:true,checkpoint_sha256:cp,policy_handoff:{event:'policy_handoff_ready'}};
 assert.equal(successor(rec,{checkpoint_sha256:cp}),rec);assert.equal(successor(rec,{checkpoint_sha256:'f'.repeat(64)}),null);
 rec.policy_handoff=null;assert.equal(successor(rec,{checkpoint_sha256:cp}),null);
});
test('controller plays auxiliary then regular using same fixed label/seed, no wait or metric mixing',async()=>{
 const dir=fs.mkdtempSync(__dirname+'/test-campaign-');fs.mkdirSync(dir+'/root-campaign');
 const entry={label:'trace-s1801',policy_rng_seed:1801,checkpoint_sha256:cp,config_path:dir+'/config.json'};
 fs.writeFileSync(dir+'/planned-rounds.json',JSON.stringify({runtime_only:true,plan:[entry]}));
 const ledger={},logs=[],seen=[];let probes=0,pauses=0;
 const successor=extract(code,'function successorHandoff(rec,entry){','async function main(){');
 const context={fs,path,os:{hostname:()=> 'spark-4ae3'},process:{pid:321,env:{},on:()=>{}},
  stage:dir,here:dir+'/root-campaign',sha:()=> 'pin',driver:'driver',DRIVER_SHA:'pin',MAX_ATTEMPTS:2,
  ledger,pendingMedia:new Set(),mediaFailed:false,log:(event,data)=>logs.push({event,...data}),
  successorHandoff:successor,gameAlive:()=>true,waitForLaunchWindow:async()=>{probes++;},sleep:async()=>{pauses++;},
  attempt:async(e,n)=>{seen.push({label:e.label,seed:e.policy_rng_seed,cp:e.checkpoint_sha256,n});
    const s=summary(n===1),v=verdict(fixture(s),cp);const rec={...v,attempt:n,dir:'attempt-'+n,checkpoint_sha256:cp,
      policy_handoff:{event:'policy_handoff_ready',utc:'fixed'}};(ledger[e.label]??=[]).push(rec);return rec;}};
 const main=extract(code,'async function main(){','module.exports=',context);await main();
 assert.deepEqual(seen,[{label:entry.label,seed:1801,cp,n:1},{label:entry.label,seed:1801,cp,n:2}]);
 assert.equal(probes,1);assert.equal(pauses,0);assert.equal(ledger[entry.label].filter(r=>r.complete).length,1);
 assert.equal(ledger[entry.label].filter(r=>r.auxiliary_complete).length,1);
 const auxiliary=logs.find(e=>e.event==='auxiliary_redo_completed');assert.equal(auxiliary.counted_in_120s_cohort,false);
 assert.equal(logs.filter(e=>e.event==='native_round_handoff_start').length,1);
});
test('worker callbacks, action path, endpoint closure and capture runtime are unchanged',()=>{
 const current=fs.readFileSync(__dirname+'/live_transfer_run_masked.cjs','utf8'),old=fs.readFileSync(__dirname+'/baseline/live_transfer_run_masked.cjs','utf8');
 const actions=s=>s.slice(s.indexOf("    relay.bus.on('message',guardedCallback(source"),s.indexOf('  } catch(error) {unsupportedPairing'));
 assert.equal(actions(current),actions(old));
 const scope=s=>s.slice(s.indexOf('function privateArena('),s.indexOf('function validateStartupRound('));assert.equal(scope(current),scope(old));
 for(const name of ['record_passive_defender.cjs','policy_handoff.cjs'])assert.equal(fs.readFileSync(__dirname+'/'+name,'utf8'),fs.readFileSync(__dirname+'/baseline/'+name,'utf8'));
 assert(code.includes('n<=MAX_ATTEMPTS&&!done'));assert(code.includes("v.filter(a=>a.complete)"));
});
