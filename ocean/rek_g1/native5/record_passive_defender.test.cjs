'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),path=require('node:path');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const {captureIdentity,captureArgs,superviseChild,recordingLimitEvidence,cleanChildExit}=require('./record_passive_defender.cjs');

test('capture labels distinguish neutral control from an explicitly pinned frozen policy',()=>{
  assert.deepEqual(captureIdentity({}),{schema:'rek.passive_defender.capture.v1',controller:'neutral_action_1',
    checkpoint_sha256:null,filename:'authentic-rek-passive-defender.mp4'});
  const checkpoint='a'.repeat(64);
  const config={capture_controller:'frozen_policy',checkpoint_sha256:checkpoint,worker:['native-worker','weights.bin',checkpoint,'73']};
  assert.deepEqual(captureIdentity(config),{schema:'rek.frozen_policy.capture.v1',controller:'frozen_policy',
    checkpoint_sha256:checkpoint,filename:'authentic-rek-policy-fight.mp4'});
  assert.throws(()=>captureIdentity({...config,capture_controller:undefined}));
  assert.throws(()=>captureIdentity({...config,checkpoint_sha256:'b'.repeat(64)}));
  assert.throws(()=>captureIdentity({...config,worker:undefined}));
  assert.throws(()=>captureIdentity({capture_controller:'unknown'}));
});
test('capture is bounded, cursor-free and restricted to isolated display',()=>{
  const args=captureArgs(path.resolve('test-capture.mp4'),140);
  assert(args.includes(':98.0+0,0'));assert.equal(args[args.indexOf('-draw_mouse')+1],'0');
  assert.equal(args[args.indexOf('-t')+1],'140');assert(args.includes('700k'));
  assert.equal(args[args.indexOf('-framerate')+1],'20');assert(args.includes('libx264'));
  assert.equal(args[args.indexOf('-fs')+1],'18000000');assert(args.includes('-n'));
});
test('invalid duration and relative paths rejected',()=>{
  for(const n of [0,-1,191,Infinity,1.5])assert.throws(()=>captureArgs(path.resolve('a.webm'),n));
  assert.throws(()=>captureArgs('relative.webm',140));
});

async function fixture(body='') {
  const child=spawn(process.execPath,['-e',`
    process.on('SIGTERM',()=>{});
    process.stdin.on('data',()=>{});
    setInterval(()=>{},1000);
    ${body}
    console.log('ready');
  `],{stdio:['pipe','pipe','pipe']});
  const kill=child.kill.bind(child);
  // Windows forcibly kills on SIGTERM. Emulate an ignored TERM there so the
  // escalation contract is tested with the same real-child fixture on both OSes.
  if(process.platform==='win32')child.kill=signal=>signal==='SIGTERM'?true:kill(signal);
  await once(child.stdout,'data');
  return {child,kill,async cleanup(){if(child.exitCode===null&&child.signalCode===null){
    child.ref();const closed=once(child,'close');kill('SIGKILL');await closed;}}};
}

test('driver ignoring SIGTERM is killed after its cleanup grace, only the owned child is signaled',{timeout:5000},async()=>{
  const f=await fixture(),other=await fixture();
  try {
    const started=Date.now();
    const result=await superviseChild(f.child,{timeoutMs:40,termGraceMs:100,killGraceMs:250}).done;
    assert.equal(result.timed_out,true);assert.equal(result.close_observed,true);
    assert.deepEqual(result.signals_sent,['SIGTERM','SIGKILL']);
    assert(Date.now()-started>=120);assert(Date.now()-started<2000);
    assert.equal(other.child.exitCode,null);assert.equal(other.child.signalCode,null);
  }finally{await f.cleanup();await other.cleanup();}
});

test('driver gets cleanup time after operator SIGTERM without premature SIGKILL',{timeout:5000},async()=>{
  const f=await fixture(`const cleanup=()=>setTimeout(()=>process.exit(0),60);
    process.on('SIGTERM',cleanup);process.stdin.on('data',cleanup);`);
  if(process.platform==='win32')f.child.kill=signal=>{
    if(signal==='SIGTERM'){f.child.stdin.write('cleanup\n');return true;}return f.kill(signal);
  };
  try {
    const control=superviseChild(f.child,{timeoutMs:1000,termGraceMs:250,killGraceMs:100});
    control.stop('operator_signal');const result=await control.done;
    assert.equal(result.code,0);assert.equal(result.timed_out,false);
    assert.deepEqual(result.signals_sent,['SIGTERM']);assert.equal(result.stop_reason,'operator_signal');
  }finally{await f.cleanup();}
});

test('capture receives q and can finish cleanly without signals',{timeout:5000},async()=>{
  const f=await fixture(`process.stdin.on('data',data=>{if(String(data).includes('q'))process.exit(0);});`);
  try {
    const control=superviseChild(f.child,{timeoutMs:1000,graceful:()=>f.child.stdin.write('q\n'),
      graceMs:200,termGraceMs:100,killGraceMs:100});
    control.stop();control.stop();const result=await control.done;
    assert.equal(result.code,0);assert.deepEqual(result.signals_sent,[]);assert.equal(result.stop_reason,'requested_stop');
  }finally{await f.cleanup();}
});

test('capture ignoring q and SIGTERM has bounded escalation',{timeout:5000},async()=>{
  const f=await fixture();let quits=0;
  try {
    const control=superviseChild(f.child,{timeoutMs:1000,graceful:()=>{quits++;f.child.stdin.write('q\n');},
      graceMs:50,termGraceMs:50,killGraceMs:200});
    control.stop();const result=await control.done;
    assert.equal(quits,1);assert.deepEqual(result.signals_sent,['SIGTERM','SIGKILL']);
    assert.equal(result.close_observed,true);
  }finally{await f.cleanup();}
});

test('hung full decode times out and cannot produce a successful validation result',{timeout:5000},async()=>{
  const f=await fixture();
  try {
    const result=await superviseChild(f.child,{timeoutMs:40,termGraceMs:40,killGraceMs:200}).done;
    assert.equal(result.timed_out,true);assert.notEqual(result.code,0);
    assert.deepEqual(result.signals_sent,['SIGTERM','SIGKILL']);
  }finally{await f.cleanup();}
});

test('even unavailable kill/close returns bounded unknown cleanup instead of claiming exit',{timeout:5000},async()=>{
  const f=await fixture();f.child.kill=()=>true;
  try {
    const started=Date.now();
    const result=await superviseChild(f.child,{timeoutMs:30,termGraceMs:30,killGraceMs:30}).done;
    assert.equal(result.cleanup_timeout,true);assert.equal(result.close_observed,false);
    assert.equal(result.code,null);assert.equal(result.signal,null);assert(Date.now()-started<1000);
  }finally{await f.cleanup();}
});

test('FFmpeg self-exit close to 18 MB is marked even when the polling threshold was missed',()=>{
  assert.equal(recordingLimitEvidence(17980000,true),'self_exit_near_ffmpeg_limit');
  assert.equal(recordingLimitEvidence(17980000,false),null);
  assert.equal(recordingLimitEvidence(17000000,true),null);
  assert.equal(recordingLimitEvidence(18000000,false),'observed_at_or_above_ffmpeg_limit');
  assert.equal(recordingLimitEvidence(18000100,true),'observed_at_or_above_ffmpeg_limit');
});
test('unknown child closure, errors, and timeout cannot be successful even with exit code zero',()=>{
  const good={code:0,close_observed:true,timed_out:false};assert.equal(cleanChildExit(good),true);
  for(const patch of [{close_observed:false,cleanup_timeout:true},{timed_out:true},{error:'fixture'},{code:null}])
    assert.equal(cleanChildExit({...good,...patch}),false);
});
