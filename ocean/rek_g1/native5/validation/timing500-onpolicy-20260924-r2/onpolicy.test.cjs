'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {SCHEMA,PINS,sha,validateSelection,completedSummary,initialSource}=require('./selection.cjs');
const {derive}=require('./derive_score_delta.cjs'),{command}=require('./run_native.cjs');
const base=require('./source/authentic_trajectory_data.cjs'),v3=require('./source/authentic_trajectory_v3.cjs');
const template=JSON.parse(fs.readFileSync(path.join(__dirname,'selection.example.json')));
test('pinned mask is the independently hashed actual 223-byte all-ones mask',()=>{assert.equal(PINS.feature_mask,sha(Buffer.alloc(223,1)));for(const h of Object.values(PINS))assert(/^[a-f0-9]{64}$/.test(h));});
function selection(n=20){const c=structuredClone(template);delete c.minimum_completed_rounds;c.rounds=Array.from({length:n},(_,i)=>({id:'fixture'+i,seed:1001+i,directory:path.resolve(os.tmpdir(),'closed-fixture-'+i),config:path.resolve(os.tmpdir(),'closed-fixture-'+i+'.json')}));return c;}
function summary(){return {checkpoint_sha256:PINS.checkpoint,observation_schema:SCHEMA,feature_mask_sha256:PINS.feature_mask,authentic_client:true,opponent:{sparring_bot_number:1,client_ai_difficulty:0},predictions:2,applied:1,initial_round:{duration:120,time_remaining:119.7,redo:false,active:true,clean_hits:[0,0]},final_round:{duration:120,time_remaining:0,redo:false,active:false,result:'WonByPoints',clean_hits:[1,0]}};}
test('default needs twenty explicit episodes; no inferred selection or duplicate ids/paths',()=>{
 assert.equal(validateSelection(selection()).minimum_completed_rounds,20);assert.throws(()=>validateSelection(selection(19)),/at least 20/);
 for(const field of ['id','directory']){const c=selection();c.rounds[1][field]=c.rounds[0][field];assert.throws(()=>validateSelection(c),/duplicate/);}
});
test('explicit smaller batch override is recorded rather than silently applied',()=>{const c=selection(1);c.minimum_completed_rounds=1;assert.equal(validateSelection(c).minimum_completed_rounds,1);});
test('fixed actual checkpoint, native object, worker and mask identities required',()=>{for(const key of Object.keys(PINS)){const c=selection();c[key].sha256='1'.repeat(64);assert.throws(()=>validateSelection(c),/identity/);}});
test('only authentic complete non-redo 120-second Bot1 episodes enter preparation',()=>{
 const c=selection();assert(completedSummary(summary(),c));
 for(const alter of [s=>s.final_round.active=true,s=>s.final_round.time_remaining=1,s=>s.initial_round.redo=true,s=>s.initial_round.duration=30,s=>s.opponent.sparring_bot_number=2,s=>s.checkpoint_sha256='1'.repeat(64),s=>s.initial_round.time_remaining=118]){const s=summary();alter(s);assert.throws(()=>completedSummary(s,c));}
});
test('controlled startup binds the exact source while retaining earlier passive observations',()=>{
 const s=summary();s.initial_round={...s.initial_round,number:1,result_value:0,time_remaining:118.75};
 const common={round_identity_sha256:'a'.repeat(64),local_slot:0};
 const first={...common,round:{...s.initial_round,time_remaining:119.9}},readiness={...common,stream_active:false,global_input_emitted:false,round:{...s.initial_round,time_remaining:118.85},observation_sequence:4,clock:{qpc_ticks:100,qpc_frequency_hz:1000,unity_frame:9}},controlled={...common,stream_active:true,round:s.initial_round,observation_sequence:5,clock:{qpc_ticks:200,qpc_frequency_hz:1000,unity_frame:10}};
 s.controlled_startup_validated=true;s.startup_readiness={...common,observation_sequence:4,qpc_ticks:'100',qpc_frequency_hz:1000,unity_frame:9,time_remaining:118.85};
 assert.equal(initialSource(s,first,controlled,readiness),controlled);assert(completedSummary(s,selection()));
 const attempts=[c=>c.round_identity_sha256='b'.repeat(64),c=>c.local_slot=1,c=>c.clock.qpc_ticks=100,c=>c.clock.qpc_frequency_hz=999,c=>c.clock.unity_frame=9,c=>c.observation_sequence=4,c=>c.round.clean_hits=[1,0],c=>c.round.number=2];
 for(const mutate of attempts){const c=structuredClone(controlled);mutate(c);assert.throws(()=>initialSource(s,first,c,readiness));}
 assert.throws(()=>initialSource(s,first,null,readiness));
 const lateReady=structuredClone(readiness),lateReadySummary=structuredClone(s);lateReady.round.time_remaining=lateReadySummary.startup_readiness.time_remaining=117.49;
 assert.throws(()=>initialSource(lateReadySummary,first,controlled,lateReady),/startup outside recorded driver contract/);
 const lateControl=structuredClone(controlled),lateControlSummary=structuredClone(s);lateControl.round.time_remaining=116.99;lateControlSummary.initial_round=lateControl.round;
 assert.throws(()=>initialSource(lateControlSummary,first,lateControl,readiness),/startup outside recorded driver contract/);
});
test('dynamic twenty-episode reward derivation preserves every non-reward/gamma byte and true behavior identity',()=>{
 const stage=fs.mkdtempSync(path.join(os.tmpdir(),'rek-scorecredit5s-cpu-fixture-')),c=selection();c.stage=stage;fs.mkdirSync(path.join(stage,'export'));
 const rows=[],rounds=[];
 for(let i=0;i<20;i++){const own=i%2?0:1,opponent=i%2?5:0,begin=rows.length;
  for(let j=0;j<2;j++){const terminal=j===1,priorOwn=j?own:0,priorOpp=j?opponent:0,gamma=Math.fround(.999**2);rows.push({sequence:i,reset:Number(j===0),action:0,policyWeight:terminal?0:1,valueWeight:1,obs:Array(223).fill(i),mask:Array(33).fill(1),time:j*.04,nextTime:(j+1)*.04,dt:.04,gamma,lambda:Math.fround(.995**2),own:priorOwn,opponent:priorOpp,nextOwn:own,nextOpponent:opponent,sourceSeq:j+1,nextSourceSeq:j+2,terminal,applied:!terminal,outcome:terminal?(own?1:-1):0,reward:base.reward(gamma,priorOwn,priorOpp,own,opponent,terminal,terminal?(own?1:-1):0),behaviorSeed:1001+i});}
  rounds.push({trial_id:c.rounds[i].id,seed:1001+i,row_begin:begin,row_end:rows.length,rows:2,terminal_race_rejected:1,awarded_points_by_slot:[own,opponent],local_slot:0,round_identity_sha256:sha(String(i))});
 }
 const artifacts=Object.fromEntries(Object.keys(PINS).map(k=>[k,c[k]])),identity=Buffer.from(JSON.stringify({rounds,artifacts})),binary=v3.pack(rows,20,identity,Buffer.alloc(223,1),artifacts);
 fs.writeFileSync(path.join(stage,'export/authentic-trajectories-v3.bin'),binary);fs.writeFileSync(path.join(stage,'export/behavior-identity.json'),identity);
 const r=derive(stage,c);assert.equal(r.rows,40);assert.equal(r.rounds.length,20);assert.equal(r.checkpoint_sha256,PINS.checkpoint);assert(r.all223_observations_actions_masks_seeds_and_loss_weights_bitwise_unchanged);
 const out=fs.readFileSync(path.join(stage,'score-delta-5s/authentic-score-delta-v5.bin'));assert.equal(out.subarray(0,8).toString(),'REKRL005');assert.equal(out.subarray(320,352).toString('hex'),PINS.checkpoint);
 r.rounds.forEach((x,i)=>assert(Math.abs(x.undiscounted_reward_sum-(i%2?-1:.2))<1e-6));
});
test('native command keeps one epoch, LR1e-5, H128, complete MC, existing PPO options',()=>{
 const p={schema:'rek.scorecredit5s_onpolicy_run_plan.v1',stage:'/private',rounds:20,minimum_completed_rounds:20,dataset:{path:'data'},identity:{path:'identity'},behavior:{path:'actual',sha256:PINS.checkpoint},worker:{path:'worker'},binaries:{replay:{path:'replayer'},trainer:{path:'trainer'}},observation_schema:SCHEMA,settings:{epochs:1,learning_rate:.00001,horizon:128,clip:.2,vf_clip:.2,vf_coef:0,entropy:.001,targets:'complete_mc_zero_baseline',reward:'received-score-delta-div5-half-life-5s.v1'}};
 const argv=command(p,'train');assert.deepEqual(argv.slice(6,13),['1','.00001','128','.2','.2','0','.001']);assert(argv.includes('--targets=complete-mc-zero-baseline'));assert.equal(command(p,'replay')[3],PINS.checkpoint);assert.equal(command(p,'diagnose')[6],'0');
});
