'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(__dirname+'/persistent_campaign.cjs','utf8');
function extract(start,end,context={}){return vm.runInNewContext(source.slice(source.indexOf(start),source.indexOf(end,source.indexOf(start)))+'\n'+start.match(/function (\w+)/)[1],context);}
const launch=extract('function launchWindow(s){','// One persistent bridge connection');
const state=()=>({isolated_session_verified:true,scene:'Arena',lease_held:false,g1_policy_stream_running:false,private_ai:{proven:true,client_ai_difficulty:0,sparring_bot_number:1,phase:'BetweenRounds',round_active:false}});
test('native round transition stays eligible without restarting client',()=>{
 assert.equal(launch(state()),'between_rounds:BetweenRounds');
 const s=state();s.private_ai.phase='FightOver';assert.equal(launch(s),'between_rounds:FightOver');
 s.private_ai.phase='Idle';s.private_ai.round_inactive=true;assert.equal(launch(s),'idle');
});
test('active, public, leased or unidentified matches are not launch windows',()=>{
 for(const mutate of [s=>s.isolated_session_verified=false,s=>s.lease_held=true,s=>s.g1_policy_stream_running=true,s=>s.private_ai.proven=false,s=>s.private_ai.round_active=true,s=>s.private_ai.client_ai_difficulty=1]){const s=state();mutate(s);assert.equal(launch(s),null);}
});
test('controller cannot invoke a live-client kill or round-based restart',()=>{
 assert(!source.includes('recycle_owned_client.sh'));assert(!source.includes('proactive_recycle_start'));
 assert(source.includes("if(gameAlive())throw Error('Refusing prefix cleanup with a live game')"));
 assert(source.includes("if(done)log('completed_round_client_preserved'"));
});
test('unsupported pair returned for guarded native Home recovery without teardown',()=>{
 const logs=[];const preserve=extract('function preserveUnsupportedPairing(st,entry){','async function waitForLaunchWindow',{log:(...x)=>logs.push(x)});
 const s=state();s.unsupported_pairing={code:'unsupported_pairing:local_g1_opponent_t800'};
 assert.equal(preserve(s,{label:'test'}),s);assert.equal(logs[0][0],'unsupported_pairing_client_preserved');
 s.private_ai.proven=false;assert.throws(()=>preserve(s,{label:'test'}));
});
