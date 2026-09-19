'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {EventEmitter}=require('node:events');
const {runContinuation,validateInvocation,PROOF}=require('./windows_authenticated_continue.cjs');
const SHA='a'.repeat(64),PROTOCOL='rek.ui_bridge.v1';

function state(screen='Login') {
  return {event:'state',protocol:PROTOCOL,scene:'Lobby',lobby_screen:screen,
    build:{plugin_sha256:SHA},foreground:{isolated_session_verified:true,
      isolated_session_proof:PROOF,execution_surface:'native_windows_isolated_desktop',
      windows_policy_account:{context_fighter_name:'moogleod',home_display_matches:true,allowed:true}},
    control:{lease_held:false,g1_policy_stream_running:false},login:{
      native_authenticated_continuation_allowed:true,already_logged_in:true,
      fighter_name_label:'moogleod',context_fighter_name:'moogleod',
      continuation_attempted:false,prelease_idle:true,reason:'ready'}};
}
function fixture(options={}) {
  const socket=new EventEmitter(),requests=[],logs=[],scheduled=new Map();
  let nextTimer=0;
  socket.write=data=>{requests.push(JSON.parse(data));return true;};
  socket.destroy=()=>{socket.destroyed=true;socket.emit('close');};
  const timers={setTimeout:(fn,ms)=>{scheduled.set(++nextTimer,{fn,ms});return nextTimer;},
    clearTimeout:id=>scheduled.delete(id)};
  const result=runContinuation({socket,bridgeSha256:SHA,log:value=>logs.push(value),timers,...options});
  const raw=text=>socket.emit('data',Buffer.from(text));
  const send=message=>raw(JSON.stringify(message)+'\n');
  const reply=(message=state())=>send({...message,request_id:requests.at(-1).request_id});
  const ack=(extra={})=>send({event:'ack',protocol:PROTOCOL,
    request_id:'native-authenticated-continuation',command:'ConfirmLoggedIn',applied:true,...extra});
  const tick=ms=>{
    const found=[...scheduled].find(([,timer])=>timer.ms===ms);
    assert.ok(found,`No scheduled ${ms} ms timer`);
    scheduled.delete(found[0]);found[1].fn();
  };
  socket.emit('connect');
  return {socket,requests,logs,scheduled,result,raw,send,reply,ack,tick};
}
function assertRequests(f,commands) {
  assert.equal(f.requests.filter(r=>r.type==='command').length,commands);
  assert.ok(f.requests.every(r=>r.type==='get_state'||
    (r.type==='command'&&r.command==='ConfirmLoggedIn')));
  assert.equal(new Set(f.requests.map(r=>r.request_id)).size,f.requests.length);
  assert.equal(f.socket.destroyed,true);
  assert.equal(f.scheduled.size,0);
}

test('Intro polling uses unique request IDs, then continues moogleod once and verifies Home',async()=>{
  const f=fixture();
  f.reply(state('Intro'));f.tick(1000);
  f.reply(state('Intro'));f.tick(1000);
  f.reply();f.ack();f.reply(state('Home'));
  assert.deepEqual(await f.result,{passed:true,reason:'moogleod_home_observed',command_sent:true,ack_received:true});
  assertRequests(f,1);
  assert.equal(f.requests.filter(r=>r.type==='get_state').length,4);
});

test('verified moogleod Home succeeds without a continuation',async()=>{
  const f=fixture();f.reply(state('Home'));
  assert.equal((await f.result).passed,true);assertRequests(f,0);
});

for(const [name,mutate,reason] of [
  ['unauthenticated',s=>{s.login.already_logged_in=false;},'human_or_unverified_login:ready'],
  ['native readiness denied',s=>{s.login.native_authenticated_continuation_allowed=false;},'human_or_unverified_login:ready'],
  ['missing readiness',s=>{delete s.login;},'human_or_unverified_login:undefined'],
  ['wrong label account',s=>{s.login.fighter_name_label='other';},'human_or_unverified_login:ready'],
  ['wrong context account',s=>{s.login.context_fighter_name='other';},'human_or_unverified_login:ready'],
  ['previous attempt',s=>{s.login.continuation_attempted=true;},'human_or_unverified_login:ready'],
  ['non-idle continuation',s=>{s.login.prelease_idle=false;},'human_or_unverified_login:ready'],
  ['Default process desktop',s=>{s.foreground.isolated_session_proof=PROOF.replace('desktop=RekPolicyEval','desktop=Default');},'runtime_scope_mismatch'],
  ['wrong input desktop',s=>{s.foreground.isolated_session_proof=PROOF.replace('input_desktop=Default','input_desktop=RekPolicyEval');},'runtime_scope_mismatch'],
  ['unverified isolation',s=>{s.foreground.isolated_session_verified=false;},'runtime_scope_mismatch'],
  ['wrong execution surface',s=>{s.foreground.execution_surface='spark_wine_x98';},'runtime_scope_mismatch'],
  ['wrong bridge hash',s=>{s.build.plugin_sha256='b'.repeat(64);},'runtime_scope_mismatch'],
  ['wrong protocol',s=>{s.protocol='other';},'runtime_scope_mismatch'],
  ['held lease',s=>{s.control.lease_held=true;},'runtime_scope_mismatch'],
  ['active policy stream',s=>{s.control.g1_policy_stream_running=true;},'runtime_scope_mismatch'],
  ['missing idle evidence',s=>{delete s.control.lease_held;},'runtime_scope_mismatch'],
  ['unexpected scene',s=>{s.scene='Arena';},'unexpected_scene'],
  ['unexpected screen',s=>{s.lobby_screen='FreePlay';},'unexpected_lobby_screen'],
])test(`refuses ${name} before any command`,async()=>{
  const f=fixture(),s=state();mutate(s);f.reply(s);
  assert.equal((await f.result).reason,reason);assertRequests(f,0);
});

for(const [name,mutate] of [
  ['wrong account',s=>{s.foreground.windows_policy_account.context_fighter_name='other';}],
  ['unverified display',s=>{s.foreground.windows_policy_account.home_display_matches=false;}],
  ['account denied',s=>{s.foreground.windows_policy_account.allowed=false;}],
])test(`requires Home account proof: ${name}`,async()=>{
  const f=fixture();f.reply();f.ack();const home=state('Home');mutate(home);f.reply(home);
  assert.equal((await f.result).reason,'home_account_mismatch');assertRequests(f,1);
});

test('revalidates runtime scope after continuation',async()=>{
  const f=fixture();f.reply();f.ack();const home=state('Home');home.build.plugin_sha256='b'.repeat(64);f.reply(home);
  assert.equal((await f.result).reason,'runtime_scope_mismatch');assertRequests(f,1);
});

test('accepted command without Home stops without retrying',async()=>{
  const f=fixture();f.reply();f.ack();f.reply(state('Login'));
  assert.equal((await f.result).reason,'home_postcondition_not_observed');assertRequests(f,1);
});

for(const [name,trigger,reason] of [
  ['timeout',f=>f.tick(90000),'state_or_continuation_timeout'],
  ['disconnect',f=>f.socket.emit('end'),'pipe_closed'],
  ['close without end',f=>f.socket.emit('close'),'pipe_closed'],
  ['transport error',f=>f.socket.emit('error',{code:'ECONNRESET'}),'ECONNRESET'],
  ['bridge error',f=>f.send({event:'error',reason:'request_queue_full_or_duplicate'}),'bridge_error:request_queue_full_or_duplicate'],
])test(`unknown continuation outcome after ${name} never retries`,async()=>{
  const f=fixture();f.reply();trigger(f);
  const result=await f.result;
  assert.equal(result.reason,reason);assert.equal(result.command_sent,true);assert.equal(result.ack_received,false);
  f.socket.emit('connect');f.ack();f.send({...state(),request_id:f.requests[0].request_id});
  assertRequests(f,1);
});

test('write failure after a possible send cannot cause a second continuation',async()=>{
  const f=fixture();
  f.socket.write=data=>{f.requests.push(JSON.parse(data));throw Object.assign(Error('write failed'),{code:'EPIPE'});};
  f.reply();assert.equal((await f.result).reason,'EPIPE');f.socket.emit('connect');assertRequests(f,1);
});

test('continuation rejection stops immediately with its reason',async()=>{
  const f=fixture();f.reply();f.ack({applied:false,reason:'authenticated_continuation_callback_or_postcondition_failed'});
  assert.equal((await f.result).reason,'continuation_rejected:authenticated_continuation_callback_or_postcondition_failed');
  assertRequests(f,1);
});

test('wrong-command acknowledgement fails closed',async()=>{
  const f=fixture();f.reply();f.ack({command:'AcquireExclusiveControl'});
  assert.equal((await f.result).reason,'invalid_continuation_ack');assertRequests(f,1);
});

test('stale states and duplicate acknowledgements cannot schedule extra requests',async()=>{
  const f=fixture(),old=f.requests[0].request_id;
  f.reply(state('Intro'));f.send({...state(),request_id:old});
  assert.equal(f.requests.length,1);f.tick(1000);
  f.send({...state(),request_id:old});assert.equal(f.requests.length,2);
  f.reply();f.ack();f.ack();assert.equal(f.requests.length,4);
  f.reply(state('Home'));assert.equal((await f.result).passed,true);assertRequests(f,1);
});

test('handles split JSON packets without duplicating the command',async()=>{
  const f=fixture(),payload=JSON.stringify({...state(),request_id:f.requests[0].request_id})+'\n';
  f.raw(payload.slice(0,17));f.raw(payload.slice(17));f.ack();f.reply(state('Home'));
  assert.equal((await f.result).passed,true);assertRequests(f,1);
});

for(const [name,payload,reason] of [
  ['invalid JSON','{broken}\n','invalid_json'],
  ['null message','null\n','invalid_message'],
  ['array message','[]\n','invalid_message'],
  ['oversized message','x'.repeat(1048577),'message_bound_exceeded'],
])test(`rejects ${name}`,async()=>{
  const f=fixture();f.raw(payload);assert.equal((await f.result).reason,reason);assertRequests(f,0);
});

test('duplicate/queue errors stop polling immediately',async()=>{
  const f=fixture();f.send({event:'error',request_id:f.requests[0].request_id,reason:'request_queue_full_or_duplicate'});
  assert.equal((await f.result).reason,'bridge_error:request_queue_full_or_duplicate');assertRequests(f,0);
});

test('evidence write failure stops before a command is sent',async()=>{
  const f=fixture({log:()=>{throw Error('disk full');}});
  assert.equal((await f.result).reason,'evidence_write_failed');assertRequests(f,0);
});

test('invocation requires explicit approval, exact local host and hash, and an absolute non-OneDrive output',()=>{
  const args=[SHA,'C:\\rekagent\\tmp\\continuation.jsonl','--approved-authenticated-continuation'];
  assert.deepEqual(validateInvocation(args,'win32','D21'),{bridgeSha256:SHA,out:args[1]});
  for(const bad of [args.slice(0,2),[...args,'extra'],['bad',...args.slice(1)],
      [SHA,'relative.jsonl',args[2]],[SHA,'C:\\Users\\Daniel\\OneDrive\\log.jsonl',args[2]],
      [SHA,args[1],'--unapproved']])assert.throws(()=>validateInvocation(bad,'win32','D21'));
  assert.throws(()=>validateInvocation(args,'linux','D21'));
  assert.throws(()=>validateInvocation(args,'win32','other'));
});
