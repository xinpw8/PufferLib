'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {startPacedLoop}=require('./paced_loop.cjs');

class Clock {
  constructor(){this.time=0;this.serial=0;this.queue=new Map();this.errors=[];}
  now=()=>this.time;
  setTimer=(fn,ms)=>{const id=++this.serial;this.queue.set(id,{at:this.time+ms,fn});return id;};
  clearTimer=id=>this.queue.delete(id);
  setSoon=fn=>this.setTimer(fn,0);
  clearSoon=id=>this.queue.delete(id);
  sleep=ms=>new Promise(resolve=>this.setTimer(resolve,ms));
  async advance(end){
    for(;;){
      const next=[...this.queue].sort((a,b)=>a[1].at-b[1].at||a[0]-b[0])[0];
      if(!next||next[1].at>end)break;
      this.time=next[1].at;this.queue.delete(next[0]);
      Promise.resolve(next[1].fn()).catch(error=>this.errors.push(error));
      await new Promise(resolve=>setImmediate(resolve));
    }
    this.time=end;
    assert.deepEqual(this.errors,[]);
  }
}
function legacyLoop(task,clock){
  let stopped=false,timer=null;
  function run(){if(stopped)return;timer=clock.setTimer(run,20);void task();}
  timer=clock.setTimer(run,20);
  return {stop(){stopped=true;clock.clearTimer(timer);}};
}
async function fixture(kind,{stepMs=24,frameMs=0,end=1000}={}){
  const clock=new Clock(),starts=[],ends=[];
  let busy=false,inFlight=0,maximum=0,frameDue=0;
  async function task(){
    if(busy)return;
    busy=true;starts.push(clock.now());maximum=Math.max(maximum,++inFlight);
    try{
      await clock.sleep(stepMs);
      if(frameMs&&clock.now()>=frameDue){await clock.sleep(frameMs);frameDue=clock.now()+50;}
      ends.push(clock.now());
    }finally{--inFlight;busy=false;}
  }
  const loop=kind==='legacy'?legacyLoop(task,clock):startPacedLoop(task,clock);
  await clock.advance(end);loop.stop();await clock.advance(end+100);
  return {starts,ends,maximum};
}
test('24 ms fake worker loses no 20 ms interval opportunities after completing',async()=>{
  const old=await fixture('legacy'),candidate=await fixture('candidate');
  assert.deepEqual(old.starts.slice(0,5),[20,60,100,140,180]);
  assert.deepEqual(candidate.starts.slice(0,5),[20,44,68,92,116]);
  assert.equal(old.starts.length,25);assert.equal(candidate.starts.length,41);
  assert.equal(old.maximum,1);assert.equal(candidate.maximum,1);
  assert.equal(old.ends[0],44);assert.equal(candidate.ends[0],44);
});
test('fast worker remains capped at one fixed 20 ms simulation step per 20 ms',async()=>{
  const result=await fixture('candidate',{stepMs:7});
  assert.equal(result.starts.length,50);
  assert.ok(result.starts.every((t,i)=>t===20*(i+1)));
  assert.equal(result.maximum,1);
});
test('step plus frame is serial and lateness never accumulates a catch-up burst',async()=>{
  const result=await fixture('candidate',{stepMs:24,frameMs:3});
  assert.equal(result.maximum,1);
  assert.ok(result.starts.every((t,i)=>i===0||t>=result.ends[i-1]));
  assert.ok(result.starts.every((t,i)=>i===0||t-result.starts[i-1]>=24));
});
test('stop during an in-flight request lets it complete without scheduling another',async()=>{
  const clock=new Clock();let calls=0,completed=0;
  const loop=startPacedLoop(async()=>{calls++;await clock.sleep(24);completed++;},clock);
  await clock.advance(21);assert.equal(calls,1);assert.equal(completed,0);
  loop.stop();loop.stop();await clock.advance(1000);
  assert.equal(calls,1);assert.equal(completed,1);assert.equal(clock.queue.size,0);
});
test('stop before the first tick cancels all work',async()=>{
  const clock=new Clock();let calls=0;
  const loop=startPacedLoop(async()=>{calls++;},clock);
  loop.stop();await clock.advance(1000);assert.equal(calls,0);
});
test('a 500 ms pause does not create deferred simulation work',async()=>{
  const clock=new Clock();let paused=true,calls=0;
  const loop=startPacedLoop(async()=>{if(paused)return;calls++;await clock.sleep(24);},clock);
  await clock.advance(500);assert.equal(calls,0);
  paused=false;await clock.advance(544);assert.equal(calls,2);
  paused=true;await clock.advance(1000);assert.equal(calls,2);loop.stop();
});
test('overdue work uses a cancellable I/O yield without the one millisecond timer clamp',async()=>{
  const clock=new Clock();let immediateCalls=0,timerCalls=0,calls=0;
  const loop=startPacedLoop(async()=>{calls++;await clock.sleep(24);},{
    now:clock.now,setTimer:(fn,ms)=>{timerCalls++;return clock.setTimer(fn,Math.max(1,ms));},
    clearTimer:clock.clearTimer,setSoon:fn=>{immediateCalls++;return clock.setSoon(fn);},clearSoon:clock.clearSoon,
  });
  await clock.advance(44);assert.equal(calls,2);assert.equal(immediateCalls,1);assert.equal(timerCalls,1);
  loop.stop();await clock.advance(200);assert.equal(calls,2);
});
test('stop cancels an already queued immediate successor',async()=>{
  let time=0,next=null,cancelled=0,calls=0;
  const loop=startPacedLoop(async()=>{calls++;time+=24;},{now:()=>time,
    setTimer:fn=>{next=fn;return 1;},clearTimer:()=>{},setSoon:fn=>{next=fn;return 2;},
    clearSoon:id=>{assert.equal(id,2);next=null;cancelled++;}});
  time=20;await next();assert.equal(calls,1);loop.stop();assert.equal(cancelled,1);assert.equal(next,null);
});
test('absolute deadlines remove persistent one millisecond callback lateness without advancing early',async()=>{
  const clock=new Clock(),starts=[];
  const loop=startPacedLoop(async()=>{starts.push(clock.now());await clock.sleep(18);},{...clock,
    now:clock.now,setTimer:(fn,ms)=>clock.setTimer(fn,ms+1),clearTimer:clock.clearTimer,
    setSoon:clock.setSoon,clearSoon:clock.clearSoon});
  await clock.advance(1000);loop.stop();await clock.advance(1100);
  assert.equal(starts.length,49);
  assert.deepEqual(starts.slice(0,5),[21,41,61,81,101]);
  assert.equal(starts.at(-1)-starts[0],48*20);
  assert.equal(loop.snapshot().maxLatenessMs,1);assert.equal(loop.snapshot().rebases,0);
});
test('an early timer callback cannot execute a physics task before its absolute deadline',async()=>{
  let time=0,next,waits=[];let calls=0;
  const loop=startPacedLoop(async()=>{calls++;},{now:()=>time,setTimer:(fn,ms)=>{next=fn;waits.push(ms);return 1;},clearTimer:()=>{}});
  time=19.4;await next();assert.equal(calls,0);assert.equal(waits.at(-1),1);
  time=20;await next();assert.equal(calls,1);loop.stop();
});
test('large compute overrun reports and discards wall debt without fabricating skipped simulation steps',async()=>{
  const clock=new Clock(),starts=[];let calls=0;
  const loop=startPacedLoop(async()=>{starts.push(clock.now());calls++;await clock.sleep(calls===1?200:0);},clock);
  await clock.advance(260);loop.stop();await clock.advance(300);
  assert.deepEqual(starts,[20,220,240,260]);assert.equal(calls,4);
  const info=loop.snapshot();assert.equal(info.rebases,1);assert.equal(info.discardedWallDebtMs,180);
  assert.equal(info.completed,4);assert.equal(info.maxTaskMs,200);
});
test('pause cancels future work and resume starts a fresh clock without paused-time debt',async()=>{
  const clock=new Clock(),starts=[];
  const loop=startPacedLoop(async()=>{starts.push(clock.now());await clock.sleep(5);},{...clock,startPaused:true});
  await clock.advance(100);assert.equal(starts.length,0);
  loop.resume();await clock.advance(121);assert.deepEqual(starts,[120]);
  loop.pause();await clock.advance(700);assert.deepEqual(starts,[120]);
  assert.equal(clock.queue.size,0);assert.equal(loop.snapshot().running,false);
  loop.resume();loop.resume();await clock.advance(741);assert.deepEqual(starts,[120,720,740]);
  loop.pause();await clock.advance(1000);assert.equal(starts.length,3);
  loop.stop();loop.resume();await clock.advance(2000);assert.equal(starts.length,3);
  assert.equal(loop.snapshot().rebases,0);
});
test('pause and resume during an in-flight request preserve the new deadline and never overlap',async()=>{
  const clock=new Clock(),starts=[];let active=0,maxActive=0;
  const loop=startPacedLoop(async()=>{starts.push(clock.now());maxActive=Math.max(maxActive,++active);await clock.sleep(24);active--;},clock);
  await clock.advance(21);loop.pause();await clock.advance(25);loop.resume();
  await clock.advance(46);loop.pause();await clock.advance(100);
  assert.deepEqual(starts,[20,45]);assert.equal(maxActive,1);loop.stop();
});

if(process.env.REK_PACING_EVIDENCE){
  (async()=>{
    const fs=require('node:fs');
    const scenarios=[];
    for(const spec of [{stepMs:24},{stepMs:24,frameMs:3},{stepMs:7}]){
      const result={fake_worker:true,virtual_window_ms:1000,fixed_simulation_step_ms:20,...spec};
      for(const mode of ['legacy','candidate']){
        const r=await fixture(mode,spec);
        result[mode]={starts:r.starts.length,first_start_ms:r.starts[0],last_start_ms:r.starts.at(-1),
          max_concurrent:r.maximum,first_starts_ms:r.starts.slice(0,8)};
      }
      scenarios.push(result);
    }
    fs.writeFileSync(process.env.REK_PACING_EVIDENCE,JSON.stringify({schema:'rek.viewer.pacing_fake_worker.v1',
      fake_clock:true,live_performance_measured:false,scenarios},null,2)+'\n',{flag:'wx'});
  })().catch(error=>{console.error(error);process.exitCode=1;});
}
