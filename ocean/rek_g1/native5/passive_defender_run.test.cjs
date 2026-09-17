'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const assert=require('node:assert/strict');
const {run,validateConfig,assertUnowned,assertActiveScope,observedNeutral,validateSource,passiveExitCode,ISOLATION}=require('./passive_defender_run.cjs');
const {ensurePrivateArena}=require('./live_transfer_run.cjs');
const ROUND='a'.repeat(64),OTHER='b'.repeat(64);
function state(active=true) {
  return {event:'state',protocol:'rek.ui_bridge.v1',scene:'Arena',foreground:{isolated_session_verified:true,isolated_session_proof:ISOLATION},
    control:{lease_held:false,g1_policy_stream_running:false,schedule_running:false},measured_pairing:{exact_g1_vs_g1:true},
    private_ai:{policy_proven:true,policy_active_gameplay_proven:active,network_client_only:true,context_is_solo:true,
      solo_route_proven:true,opponent_is_ai:true,opponent_slot_is_ai:true,human_in_opponent_slot:false,
      opponent_slot_client_known:true,opponent_slot_has_client:false,opponent_human_bit_set:false,
      client_ai_difficulty:3,sparring_bot_number:4,phase:active?'RoundActive':'Idle',round_active:active,
      round_inactive:!active,client_visual_only_fighter_pair:true,local_slot:0,opponent_slot:1,round_number:1}};
}
function source(sequence=1) {
  return {event:'g1_policy_state',protocol:'rek.ui_bridge.v1',schema:'rek.g1_policy_source.v1',
    observation_sequence:sequence,round_identity_sha256:ROUND,local_slot:0,stream_active:true,global_input_emitted:false,
    opponent:{client_ai_difficulty:3,sparring_bot_number:4,opponent_is_ai:true,human_in_opponent_slot:false},
    fighters:[0,1].map(()=>({bone_names:Array(30).fill('fixture_bone')})),clock:{qpc_ticks:1000+sequence*100},
    input:{desired_action:1,velocity_command_xyz:[0,0,0],pending_move:false,pending_special:false,pending_estop:false,
      requested_move_index:null,move_request_pending_transport:false},action_mask:Array(33).fill(true),
    round:{number:1,active:true,time_remaining:120-sequence,result_value:0,clean_hits:[0,sequence]}};
}

if(process.argv[2]==='--fixture') {
  const scenario=process.argv[3];let live=state(!['safe','ready','human_ready'].includes(scenario)),timer,seq=0;
  if(scenario==='owned')live.control.lease_held=true;
  if(scenario==='ready'||scenario==='human_ready'){live.private_ai.client_visual_only_fighter_pair=false;live.private_ai.policy_proven=false;}
  const send=x=>process.stdout.write(JSON.stringify(x)+'\n');
  console.error('offline fake relay only');send({event:'hello',protocol:'rek.ui_bridge.v1'});
  require('node:readline').createInterface({input:process.stdin}).on('line',line=>{
    const r=JSON.parse(line);
    if(r.type==='get_state'){send({...live,request_id:r.request_id});return;}
    if(r.type==='command') {
      if(r.command==='AcquireExclusiveControl')live.control.lease_held=true;
      if(['StartG1PolicyRound','ReadyPrivateAiSession'].includes(r.command)) {
        live={...state(true),control:live.control};if(scenario==='human_ready')live.private_ai.human_in_opponent_slot=true;
      }
      if(r.command==='StartG1PolicyStreamAnyAi') {
        live.control.g1_policy_stream_running=true;
        timer=setInterval(()=>{
          const s=source(++seq);
          if(scenario==='changed'&&seq===2)s.round_identity_sha256=OTHER;
          if(scenario==='botchanged'&&seq===2)s.opponent={...s.opponent,client_ai_difficulty:4,sparring_bot_number:5};
          if(scenario==='nonneutral'&&seq>=2)s.input.velocity_command_xyz=[1,0,0];
          if(seq===4&&!['cap','noack'].includes(scenario)) {
            s.round.active=false;s.round.result_value=1;s.round.time_remaining=0;
            live.private_ai.round_active=false;clearInterval(timer);
          }
          send(s);
        },10);
      }
      if(r.command==='StopG1PolicyStream') {
        clearInterval(timer);live.control.g1_policy_stream_running=false;
        send({event:'g1_policy_end',reason:'requested_stop',owned_velocity_neutralized:true,
          neutral_send_method_returned:true,global_input_emitted:false});
      }
      if(r.command==='ReleaseExclusiveControl'&&scenario!=='cleanup_bad')live.control.lease_held=false;
      send({event:'ack',request_id:r.request_id,status:'accepted',reason:'fixture'});return;
    }
    if(r.type==='policy_action') {
      assert.equal(r.action,1,'the fake relay rejects all nonneutral actions');
      if(scenario==='noack')return;
      send({event:'g1_policy_action',...r,type:undefined,applied:true,reason:'held_state_applied_locally',
        global_input_emitted:false,clock:{qpc_ticks:1000+r.observation_sequence*100+50}});
    }
  }).on('close',()=>{clearInterval(timer);process.exitCode=0;});
}else {
  const {test}=require('node:test');
  function delayedEntry(entryTimeoutMs,arrivalMs=60000) {
    let clock=0;const commands=[],lobby={scene:'Lobby',lobby_screen:'FreePlay'},arena=state(false);
    return {commands,arena,now:()=>clock,start:()=>ensurePrivateArena(lobby,{enterPrivate:true,entryTimeoutMs,
      command:async command=>commands.push(command),getState:async()=>clock>=arrivalMs?arena:lobby,
      wait:async ms=>{clock+=ms;},now:()=>clock})};
  }
  test('private entry retains the learned driver 45 second default',async()=>{
    const f=delayedEntry();await assert.rejects(f.start(),/private-practice entry timeout/);
    assert.equal(f.now(),45000);assert.deepEqual(f.commands,['EnterSolo']);
  });
  test('explicit 120 second entry wait accepts a delayed native reservation without reentering',async()=>{
    const f=delayedEntry(120000);assert.equal(await f.start(),f.arena);
    assert.equal(f.now(),60000);assert.deepEqual(f.commands,['EnterSolo']);
    const never=delayedEntry(120000,Infinity);await assert.rejects(never.start(),/private-practice entry timeout/);
    assert.equal(never.now(),120000);assert.deepEqual(never.commands,['EnterSolo']);
  });
  test('entry wait rejects invalid or unbounded durations before any command',async()=>{
    for(const value of [0,-1,120001,1.5,Infinity,NaN,'120000',null]) {
      const f=delayedEntry(value);await assert.rejects(f.start(),/entryTimeoutMs/);
      assert.deepEqual(f.commands,[]);assert.equal(f.now(),0);
    }
  });
  const config=()=>({mode:'active_attach',max_seconds:1,out:path.join(os.tmpdir(),'unused-passive-output'),relay:['fixture']});
  test('host, explicit mode, bounded time and fresh absolute output contract',()=>{
    assert.doesNotThrow(()=>validateConfig(config(),'linux','spark-4ae3'));
    for(const [platform,host] of [['win32','spark-4ae3'],['linux','workstation'],['darwin','spark-4ae3']])
      assert.throws(()=>validateConfig(config(),platform,host),/Spark host/);
    for(const max_seconds of [0,-1,181,Infinity,NaN,'120'])
      assert.throws(()=>validateConfig({...config(),max_seconds},'linux','spark-4ae3'),/max_seconds/);
    assert.throws(()=>validateConfig({...config(),mode:'automatic'},'linux','spark-4ae3'),/explicit mode/);
    assert.throws(()=>validateConfig({...config(),out:'relative'},'linux','spark-4ae3'),/absolute/);
  });
  test('preexisting and unknown lease, stream, controller and isolation reject',()=>{
    assert.doesNotThrow(()=>assertUnowned(state()));
    for(const control of [{lease_held:true,g1_policy_stream_running:false},{lease_held:false,g1_policy_stream_running:true},
      {lease_held:false},{lease_held:false,g1_policy_stream_running:false,single_motion_trial_running:true}])
      assert.throws(()=>assertUnowned({...state(),control}),/preexisting|unknown/);
    assert.throws(()=>assertUnowned({...state(),foreground:{isolated_session_verified:true}}),/isolated/);
  });
  test('active input scope requires exact G1, no humans, private route and active proof',()=>{
    assert.doesNotThrow(()=>assertActiveScope(state()));
    assert.throws(()=>assertActiveScope({...state(),measured_pairing:{exact_g1_vs_g1:false}}),/G1 pair/);
    for(const patch of [{human_in_opponent_slot:true},{policy_proven:false},{policy_active_gameplay_proven:false},
      {round_active:false},{opponent_slot_has_client:true},{client_visual_only_fighter_pair:false}])
      assert.throws(()=>assertActiveScope({...state(),private_ai:{...state().private_ai,...patch}}),/G1 pair/);
  });
  test('native source pins round, bot, local slot, schema and G1-shaped pair',()=>{
    const s=source();assert.doesNotThrow(()=>validateSource(s,s.opponent,ROUND,0));
    for(const patch of [{round_identity_sha256:OTHER},{local_slot:1},{schema:'other'},
      {fighters:[]},{global_input_emitted:true},{observation_sequence:0}])
      assert.throws(()=>validateSource({...s,...patch},s.opponent,ROUND,0));
    assert.throws(()=>validateSource({...s,opponent:{...s.opponent,human_in_opponent_slot:true}},s.opponent,ROUND,0),/identity_changed/);
  });
  test('neutral readback is exact held action and zero command, never a frozen-pose assertion',()=>{
    const s=source();assert.equal(observedNeutral(s),true);
    for(const input of [{desired_action:0},{velocity_command_xyz:[0,0,0.001]},{pending_move:true},
      {pending_special:true},{pending_estop:true},{requested_move_index:0},{move_request_pending_transport:true}])
      assert.equal(observedNeutral({...s,input:{...s.input,...input}}),false);
    s.fighters[0].root_position_xyz=[5,2,3];assert.equal(observedNeutral(s),true);
  });
  test('completion needs terminal round, neutral verification and cleanup proof',()=>{
    const s={round_completed:true,neutral_command_verified:true,cleanup:{verified:true}};
    assert.equal(passiveExitCode(s),0);
    assert.equal(passiveExitCode({...s,round_completed:false}),2);
    assert.equal(passiveExitCode({...s,neutral_command_verified:false}),2);
    assert.equal(passiveExitCode({...s,cleanup:{verified:false}}),2);
  });
  async function integration(scenario,mode='active_attach') {
    const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-passive-fixture-'));
    const cfg={...config(),mode,enter_private:true,max_seconds:scenario==='cap'||scenario==='noack'?0.09:1,
      out:path.join(dir,'trial'),relay:[process.execPath,__filename,'--fixture',scenario]};
    const file=path.join(dir,'fixture.config.json');fs.writeFileSync(file,JSON.stringify(cfg));
    const summary=await run(file,{platform:'linux',hostname:'spark-4ae3',quiet:true});
    const records=name=>fs.readFileSync(path.join(cfg.out,name),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
    const commands=records('relay.stdin.jsonl');
    assert(commands.filter(x=>x.type==='policy_action').every(x=>x.action===1));
    assert(!commands.some(x=>/EStop|Freeze|Move|Attack/.test(x.command??'')));
    assert(fs.readFileSync(path.join(cfg.out,'relay.stderr.txt'),'utf8').includes('offline fake relay'));
    assert.deepEqual(records('g1_policy_state.jsonl'),records('relay.stdout.jsonl').filter(x=>x.event==='g1_policy_state'));
    assert.equal(JSON.parse(fs.readFileSync(path.join(cfg.out,'exit.json'),'utf8')).code,summary.exit_code);
    return {summary,commands,dir};
  }
  test('active attach records all frames and only neutral commands, completes one round, cleans up',async()=>{
    const {summary:s,commands}=await integration('normal');assert.equal(s.exit_code,0);assert.equal(s.round_completed,true);
    assert.equal(s.opponent.sparring_bot_number,4);assert.equal(s.requested,3);assert.equal(s.applied,3);
    assert.equal(s.source_count,4);assert.equal(s.neutral_command_verified,true);assert.equal(s.cleanup.verified,true);
    assert.equal(s.attach_status,'attached_to_preexisting_active_round');
    assert.deepEqual(commands.filter(x=>x.type==='command').map(x=>x.command),
      ['AcquireExclusiveControl','StartG1PolicyStreamAnyAi','StopG1PolicyStream','ReleaseExclusiveControl']);
  });
  test('safe start and native ready each request one native round, never repeat',async()=>{
    for(const [scenario,start] of [['safe','StartG1PolicyRound'],['ready','ReadyPrivateAiSession']]) {
      const {summary:s,commands}=await integration(scenario,'safe_start');assert.equal(s.exit_code,0);
      assert.equal(commands.filter(x=>x.command===start).length,1);
      assert.equal(commands.filter(x=>x.command==='StartG1PolicyStreamAnyAi').length,1);
    }
  });
  test('visible preexisting lease refuses ownership and emits no command or action',async()=>{
    const {summary:s,commands}=await integration('owned');assert.equal(s.exit_code,2);
    assert.match(s.stop_reason,/preexisting/);assert(commands.every(x=>x.type==='get_state'));
  });
  test('safe_start refuses active attachment; ready rejects newly observed human',async()=>{
    const active=await integration('normal','safe_start');assert.equal(active.summary.exit_code,2);
    assert(active.commands.every(x=>x.type==='get_state'));
    const human=await integration('human_ready','safe_start');assert.equal(human.summary.exit_code,2);
    assert.equal(human.commands.filter(x=>x.type==='policy_action').length,0);assert.equal(human.summary.cleanup.verified,true);
  });
  test('round or bot identity change stops before another action and retains offending source',async()=>{
    for(const scenario of ['changed','botchanged']) {
      const {summary:s}=await integration(scenario);assert.equal(s.exit_code,2);assert.equal(s.requested,1);
      assert.equal(s.source_count,2);assert.equal(s.cleanup.verified,true);assert.match(s.stop_reason,/identity_changed/);
    }
  });
  test('duration cap remains incomplete; no ack never permits a second inflight action',async()=>{
    const cap=await integration('cap');assert.equal(cap.summary.exit_code,2);assert.equal(cap.summary.stop_reason,'duration_cap_incomplete');
    const noack=await integration('noack');assert.equal(noack.summary.requested,1);assert.equal(noack.summary.applied,0);
    assert(noack.summary.source_count>1);assert(noack.summary.dropped_sources.action_inflight>0);
  });
  test('nonzero measured command and unsuccessful cleanup cannot be reported as verified success',async()=>{
    const nonneutral=await integration('nonneutral');assert.equal(nonneutral.summary.neutral_command_verified,false);
    assert.equal(nonneutral.summary.exit_code,2);
    const bad=await integration('cleanup_bad');assert.equal(bad.summary.cleanup.verified,false);assert.equal(bad.summary.exit_code,2);
  });
}
