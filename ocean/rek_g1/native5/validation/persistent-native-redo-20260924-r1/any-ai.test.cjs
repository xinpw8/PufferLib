'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),os=require('node:os');
const {EventEmitter}=require('node:events');
const driver=require('../persistent-native-redo-20260924-r1/live_transfer_run_masked.cjs');
const code=fs.readFileSync(__dirname+'/campaign.cjs','utf8'),before=fs.readFileSync(__dirname+'/baseline/campaign.cjs','utf8');
const cp='a'.repeat(64);
function extract(text,from,to,context={}){return vm.runInNewContext(text.slice(text.indexOf(from),text.indexOf(to,text.indexOf(from)))+'\n'+from.match(/function (\w+)/)[1],context);}
const mode=extract(code,'function runtimeOpponentMode(plan,env) {','function knownOpponent(');
const known=extract(code,'function knownOpponent(p) {','function opponentMetrics(');
const metrics=extract(code,'function opponentMetrics(records) {','const logStream=',{knownOpponent:known});
const verdict=extract(code,'function verdict(dir,expectedCheckpoint,','async function attempt(',{fs,path,knownOpponent:known,runtimeAnyAi:false});
const oldVerdict=extract(before,'function verdict(dir,expectedCheckpoint){','async function attempt(',{fs,path});
const launch=extract(code,'function launchWindow(s,','// One persistent bridge connection',{knownOpponent:known,runtimeAnyAi:false});
function rawState(){return {event:'state',scene:'Arena',foreground:{isolated_session_verified:true},
 control:{lease_held:false,g1_policy_stream_running:false},private_ai:{proven:false,policy_proven:true,
 network_client_only:true,context_is_solo:true,solo_route_proven:true,opponent_is_ai:true,opponent_slot_is_ai:true,
 human_in_opponent_slot:false,opponent_slot_client_known:true,opponent_slot_has_client:false,opponent_human_bit_set:false,
 client_ai_difficulty:1,sparring_bot_number:2,phase:'BetweenRounds',round_active:false,round_inactive:true}};}
function compact(r){return {scene:r.scene,isolated_session_verified:r.foreground.isolated_session_verified,
 lease_held:r.control.lease_held,g1_policy_stream_running:r.control.g1_policy_stream_running,
 private_ai:{...r.private_ai,known_private_ai_proven:driver.privateArena(r),known_private_ai_ready:driver.canReadyPrivateAiSession(r)}};}
function summary(bot=2,redo=false){return {predictions:20,applied:20,
 opponent:{client_ai_difficulty:bot-1,sparring_bot_number:bot},authentic_client:true,global_input_emitted:false,
 checkpoint_sha256:cp,controlled_startup_validated:true,play_native_redo:redo,
 startup_readiness:{round_kind:redo?'redo_30s':'regular_120s',round_duration:redo?30:120,round_redo:redo,play_native_redo:redo},
 initial_round:{number:2,duration:redo?30:120,time_remaining:redo?29.7:119.7,active:true,redo,clean_hits:[0,0]},
 final_round:{number:2,duration:redo?30:120,time_remaining:0,active:false,redo,clean_hits:[11,21],winner_index:1,result:'WonByPoints',result_value:1}};}
function fixture(s){const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-any-ai-'));fs.mkdirSync(dir+'/trial');
 fs.writeFileSync(dir+'/trial/summary.json',JSON.stringify(s));return dir;}
test('any-AI mode requires all three explicit opt-ins and valid booleans',()=>{
 assert.equal(mode({},{}),false);assert.equal(mode({runtime_only:true},{}),false);
 assert.equal(mode({runtime_only:true,runtime_any_ai:true},{AB_RUNTIME_ANY_AI:'1'}),true);
 for(const [p,e]of [[{runtime_any_ai:true},{AB_RUNTIME_ANY_AI:'1'}],[{runtime_only:true,runtime_any_ai:true},{}],
 [{runtime_only:true},{AB_RUNTIME_ANY_AI:'1'}],[{runtime_only:true,runtime_any_ai:'true'},{AB_RUNTIME_ANY_AI:'1'}],
 [{runtime_only:true,runtime_any_ai:true},{AB_RUNTIME_ANY_AI:'yes'}]])assert.throws(()=>mode(p,e));
});
test('raw known private no-human proof permits Bot2 only in opted-in launch mode',()=>{
 const s=compact(rawState());assert.equal(s.private_ai.known_private_ai_proven,true);
 assert.equal(launch(s),null);assert.equal(launch(s,true),'between_rounds:BetweenRounds');
 s.private_ai.phase='Idle';assert.equal(launch(s,true),'idle');
 s.private_ai.phase='RoundActive';s.private_ai.round_active=true;assert.equal(launch(s,true),null);
});
test('every original no-human route predicate is still required for any-AI watcher',()=>{
 for(const key of ['policy_proven','network_client_only','context_is_solo','solo_route_proven','opponent_is_ai',
 'opponent_slot_is_ai','opponent_slot_client_known']){const r=rawState();r.private_ai[key]=false;assert.equal(launch(compact(r),true),null,key);}
 for(const key of ['human_in_opponent_slot','opponent_slot_has_client','opponent_human_bit_set']){
 const r=rawState();r.private_ai[key]=true;assert.equal(launch(compact(r),true),null,key);}
 for(const mutate of [r=>r.foreground.isolated_session_verified=false,r=>r.control.lease_held=true,
 r=>r.control.g1_policy_stream_running=true,r=>r.private_ai.client_ai_difficulty=-1,
 r=>r.private_ai.client_ai_difficulty=256,r=>r.private_ai.sparring_bot_number=1]){
 const r=rawState();mutate(r);assert.equal(launch(compact(r),true),null);}
});
test('pre-ready native helper is accepted only for known private inactive Idle with no fighter binding',()=>{
 const r=rawState();Object.assign(r.private_ai,{phase:'Idle',policy_proven:false,client_visual_only_fighter_pair:false});
 const s=compact(r);assert.equal(s.private_ai.known_private_ai_proven,false);assert.equal(s.private_ai.known_private_ai_ready,true);
 assert.equal(launch(s,true),'idle');assert.equal(launch(s),null);
 for(const mutate of [r=>r.private_ai.phase='BetweenRounds',r=>r.private_ai.phase='RoundActive',
 r=>r.private_ai.round_active=true,r=>r.private_ai.round_inactive=false,r=>r.private_ai.opponent_slot_has_client=true,
 r=>r.private_ai.human_in_opponent_slot=true,r=>r.private_ai.solo_route_proven=false,r=>r.private_ai.client_visual_only_fighter_pair=true]){
 const x=JSON.parse(JSON.stringify(r));mutate(x);assert.equal(launch(compact(x),true),null);}
 const forged={...s,private_ai:{...s.private_ai,phase:'BetweenRounds'}};assert.equal(launch(forged,true),null);
});
test('passive watcher calls actual privateArena on raw state, observes owned relay exit before return',async()=>{
 const child=new EventEmitter();child.stdout=new EventEmitter();child.stderr=new EventEmitter();
 child.exitCode=null;child.signalCode=null;let lines,closed=false,writes=[];
 child.stdin={write:x=>writes.push(JSON.parse(x)),end:()=>setImmediate(()=>{closed=true;child.exitCode=0;child.emit('exit',0,null);})};
 const context={spawn:()=>child,runtimeAnyAi:true,privateArena:driver.privateArena,canReadyPrivateAiSession:driver.canReadyPrivateAiSession,provenG1T800Pairing:()=>null,
 launchWindow:s=>launch(s,true),utc:()=> 'fixed',fs:{appendFileSync:()=>{}},path,here:'stage',log:()=>{},
 require:n=>{assert.equal(n,'node:readline');return {createInterface:()=>{lines=new EventEmitter();return lines;}};},
 setTimeout,clearTimeout,setInterval,clearInterval};
 const watch=extract(code,'function watchOnce(cfg,timeoutMs){','async function watchForWindow(',context);
 const p=watch({relay:['fake']},2000);lines.emit('line',JSON.stringify({event:'hello'}));
 lines.emit('line',JSON.stringify(rawState()));const r=await p;
 assert(closed);assert.equal(r.private_ai.known_private_ai_proven,true);assert.equal(writes.length,1);assert.equal(writes[0].type,'get_state');
});
test('Bot2 regular result becomes runtime complete without being relabeled Bot1',()=>{
 const dir=fixture(summary()),a=verdict(dir,cp),b=verdict(dir,cp,true);
 assert.equal(a.complete,false);assert.equal(b.complete,true);assert.equal(b.same_bot,false);
 assert.equal(b.eligible_opponent,true);assert.equal(b.opponent.sparring_bot_number,2);assert.equal(b.runtime_any_ai,true);
});
test('Bot2 redo remains separate auxiliary result with unchanged fair-start/terminal checks',()=>{
 const s=summary(2,true),v=verdict(fixture(s),cp,true);
 assert.equal(v.complete,false);assert.equal(v.auxiliary_complete,true);assert.equal(v.same_bot,false);
 for(const mutate of [s=>s.initial_round.clean_hits=[1,0],s=>s.initial_round.time_remaining=26.9,
 s=>s.final_round.active=true,s=>s.final_round.duration=120,s=>s.play_native_redo=false,
 s=>s.checkpoint_sha256='b'.repeat(64),s=>s.opponent.sparring_bot_number=3,
 s=>s.authentic_client=false,s=>s.global_input_emitted=true,s=>s.controlled_startup_validated=false]){
 const x=summary(2,true);mutate(x);const q=verdict(fixture(x),cp,true);assert.equal(q.complete,false);assert.equal(q.auxiliary_complete,false);}
});
test('all original default verdict fields remain identical for Bot1 and non-Bot1 samples',()=>{
 for(const bot of [1,2])for(const redo of [false,true])for(const mutate of [s=>{},s=>s.initial_round.time_remaining=10,s=>s.applied=0]){
 const s=summary(bot,redo);mutate(s);const dir=fixture(s),old=oldVerdict(dir,cp),now=verdict(dir,cp);
 for(const [k,v]of Object.entries(old))assert.equal(JSON.stringify(now[k]),JSON.stringify(v),k);}
});
test('opponent metrics separate Bot1/Bot2 and regular/auxiliary records',()=>{
 const rows=[verdict(fixture(summary(1)),cp,true),verdict(fixture(summary(2)),cp,true),verdict(fixture(summary(2,true)),cp,true)];
 const m=metrics(rows);assert.equal(m.Bot1.regular_120s.completed,1);assert.equal(m.Bot2.regular_120s.completed,1);
 assert.equal(m.Bot2.redo_30s.completed,1);assert.equal(m.Bot1.redo_30s.completed,0);assert.equal(m.Bot2.regular_120s.losses,1);
});
test('original pairing recovery, attempt identity, fairness and handoff implementations are preserved',()=>{
 for(const [start,end]of [['function preserveUnsupportedPairing(','async function waitForLaunchWindow('],
 ['function attemptPaths(','function verdict('],['function successorHandoff(','async function main(){']]){
 const piece=s=>s.slice(s.indexOf(start),s.indexOf(end,s.indexOf(start)));assert.equal(piece(code),piece(before));}
 assert(code.includes('const nonwins=counted.filter(a=>a.final_round.winner_index!==0).length;'));
 assert(code.includes('if(nonwins>=3&&!runtimeOnly)'));assert(code.includes('n<=MAX_ATTEMPTS&&!done'));
 assert(code.includes('runtimeAnyAi=runtimeOpponentMode(planDocument,process.env);'));
});
