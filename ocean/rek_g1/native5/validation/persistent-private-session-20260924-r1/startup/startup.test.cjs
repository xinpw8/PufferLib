'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {canSkipObservedIntro,exitObservedUnsupportedPairing,ensurePrivateArena,provenG1T800Pairing}=require('./live_transfer_run_masked.cjs');
const foreground={isolated_session_verified:true,execution_surface:'spark_wine_x98'};
const intro={scene:'Lobby',lobby_screen:'Intro',foreground,intro_skip:{skip_allowed:true,active:true,finished:false,skip_shown:true,skip_enabled:true}};
function arena(){return {scene:'Arena',foreground,private_ai:{network_client_only:true,context_is_solo:true,solo_route_proven:true,
 opponent_is_ai:true,opponent_slot_is_ai:true,human_in_opponent_slot:false,opponent_slot_client_known:true,
 opponent_slot_has_client:false,opponent_human_bit_set:false,client_ai_difficulty:0,sparring_bot_number:1,
 proven:true,policy_proven:true,round_active:true,round_inactive:false,client_visual_only_fighter_pair:true,
 local_slot:0,opponent_slot:1,phase:'RoundActive'}};}
function mixed(){return {...arena(),measured_pairing:{local_slot:0,opponent_slot:1,reason:'mixed_supported_runtime_models_rejected',exact_g1_vs_g1:false,
 local_fighter:{semantic_robot_id:'g1',exact_g1_bone_signature:true,bone_count:30,runtime_bone_signature_sha256:'9d18e697233d9578b398fbe849cd59d65cb27a5c2223b2602db66a82a410e987'},
 opponent_fighter:{semantic_robot_id:'t800',exact_t800_bone_signature:true,bone_count:26,runtime_bone_signature_sha256:'ec0f8d0ae5bd170464f5393f9860959e47a54b8e73e4dc259a6fb955f46d3dab'}}};}
test('Skip requires all observed native intro fields and isolated Spark',()=>{
 assert(canSkipObservedIntro(intro));
 for(const [key,value] of [['skip_allowed',false],['active',false],['finished',true],['skip_shown',false],['skip_enabled',false]])assert(!canSkipObservedIntro({...intro,intro_skip:{...intro.intro_skip,[key]:value}}));
 assert(!canSkipObservedIntro({...intro,foreground:{...foreground,execution_surface:'native_windows_isolated_desktop'}}));
 assert(!canSkipObservedIntro({...intro,lobby_screen:'Login'}));assert(!canSkipObservedIntro({...intro,intro_skip:undefined}));
});
test('native observed Skip runs promptly once, then existing Home/FreePlay route',async()=>{
 let time=0;const commands=[],states=[{scene:'Lobby',lobby_screen:'Home',foreground},{scene:'Lobby',lobby_screen:'FreePlay',foreground},arena()];
 const result=await ensurePrivateArena(intro,{enterPrivate:true,command:async c=>commands.push({c,time}),getState:async()=>states.shift(),now:()=>time,wait:async ms=>{time+=ms;}});
 assert.deepEqual(commands.map(x=>x.c),['SkipIntro','NavigateFreePlay','EnterSolo']);assert.equal(commands[0].time,0);assert.equal(result.scene,'Arena');
});
test('unobserved or disabled Skip never gets inferred from screen name',async()=>{
 let time=0;const commands=[];const hidden={...intro,intro_skip:{...intro.intro_skip,skip_shown:false}};
 await assert.rejects(ensurePrivateArena(hidden,{enterPrivate:true,command:async c=>commands.push(c),getState:async()=>hidden,now:()=>time,wait:async ms=>{time+=ms;},entryTimeoutMs:500}),/entry timeout/);
 assert.deepEqual(commands,[]);
});
test('exact unsupported private pair uses native exit then observed confirmation and Home',async()=>{
 let time=0;const commands=[],state=mixed(),home={scene:'Lobby',lobby_screen:'Home',foreground};
 assert(provenG1T800Pairing(state));
 const states=[{...state,unsupported_pairing_exit:{confirmation_required:true}},home];
 const result=await exitObservedUnsupportedPairing(state,{command:async c=>commands.push(c),getState:async()=>states.shift(),now:()=>time,wait:async ms=>{time+=ms;}});
 assert.equal(result,home);assert.deepEqual(commands,['ExitUnsupportedPrivateAiPairing','ExitUnsupportedPrivateAiPairing']);
});
test('ordinary pair or any human occupancy cannot request unsupported exit',async()=>{
 for(const state of [arena(),{...mixed(),private_ai:{...mixed().private_ai,human_in_opponent_slot:true}}]){
  const result=await exitObservedUnsupportedPairing(state,{command:async()=>assert.fail('unexpected command')});assert.equal(result,state);
 }
});
test('missing visible confirmation stays bounded and leaves client running',async()=>{
 let time=0;const commands=[],state=mixed();
 await assert.rejects(exitObservedUnsupportedPairing(state,{command:async c=>commands.push(c),getState:async()=>state,now:()=>time,wait:async ms=>{time+=ms;},timeoutMs:300}),/client left running/);
 assert.deepEqual(commands,['ExitUnsupportedPrivateAiPairing']);
});
test('scope loss or a native command refusal cannot trigger fallback input',async()=>{
 await assert.rejects(exitObservedUnsupportedPairing(mixed(),{command:async()=>{throw Error('native proof rejected');}}),/native proof rejected/);
 let calls=0;await assert.rejects(exitObservedUnsupportedPairing(mixed(),{command:async()=>calls++,getState:async()=>({...mixed(),foreground:{isolated_session_verified:false}}),wait:async()=>{}}),/proof lost/);assert.equal(calls,1);
});
test('both additive commands are accepted by relay/bridge but not Windows surface',()=>{
 const root=path.join(__dirname,'variant/windows');const protocol=fs.readFileSync(path.join(root,'RekUiBridgeAgent/BridgeProtocol.cs'),'utf8');
 const relay=fs.readFileSync(path.join(root,'RekUiPipeClient/PolicyRelay.cs'),'utf8');const surface=fs.readFileSync(path.join(root,'PolicyExecutionIsolationContract.cs'),'utf8');
 for(const name of ['SkipIntro','ExitUnsupportedPrivateAiPairing']){assert(protocol.includes('    '+name+','));assert(relay.includes('"'+name+'"'));assert(!surface.includes('"'+name+'"'));}
});
