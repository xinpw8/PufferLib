'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {LatestFrameQueue}=require('./render_queue.cjs');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const snapshot=(tick,generation=0)=>({snapshotTick:tick,generation,qpos:Array(72).fill(tick)});
function rig(){
  let clock=0,serial=0;const timers=new Map(),calls=[],frames=[],errors=[];
  const queue=new LatestFrameQueue({now:()=>clock,
    setTimer:(fn,ms)=>{timers.set(++serial,{fn,at:clock+ms});return serial;},clearTimer:id=>timers.delete(id),
    render:value=>new Promise((resolve,reject)=>calls.push({value,resolve:()=>resolve({...value,png:'ZmFrZQ=='}),reject})),
    onFrame:(reply,value)=>frames.push(value),onError:error=>errors.push(error.message)});
  function advance(ms=0){clock+=ms;for(const [id,timer]of [...timers])if(timer.at<=clock){timers.delete(id);timer.fn();}}
  return {queue,calls,frames,errors,advance,timers};
}
test('slow renderer retains only newest immutable snapshot and never overlaps requests',async()=>{
  const r=rig(),first=snapshot(1);r.queue.offer(first);first.qpos[0]=99;r.advance();
  assert.equal(r.calls[0].value.qpos[0],1);
  for(let tick=2;tick<=100;tick++)r.queue.offer(snapshot(tick));r.advance(1000);
  assert.equal(r.calls.length,1);assert.equal(r.queue.pending.snapshot.snapshotTick,100);
  r.calls[0].resolve();await flush();r.advance();assert.equal(r.calls.length,2);
  assert.equal(r.calls[1].value.snapshotTick,100);r.calls[1].resolve();await flush();
  assert.deepEqual(r.frames.map(x=>x.snapshotTick),[1,100]);r.queue.close();
});
test('generation change discards in-flight frame and pending exact capture, then accepts reset tick zero',async()=>{
  const r=rig();r.queue.offer(snapshot(900));r.advance();
  const exact=r.queue.exact(snapshot(901));const rejected=assert.rejects(exact,/generation changed/);
  r.queue.invalidate(1);await rejected;r.queue.offer(snapshot(0,1));
  r.calls[0].resolve();await flush();r.advance();
  assert.equal(r.frames.length,0);assert.equal(r.errors.length,0);
  r.calls[1].resolve();await flush();assert.equal(r.frames[0].snapshotTick,0);assert.equal(r.frames[0].generation,1);
  r.queue.close();
});
test('paused exact capture is retained over live offers and resolves only its identified frame',async()=>{
  const r=rig();r.queue.offer(snapshot(1));r.advance();
  let done=false;const exact=r.queue.exact(snapshot(3)).then(()=>done=true);
  assert.equal(r.queue.offer(snapshot(4)),false);await flush();assert.equal(done,false);
  r.calls[0].resolve();await flush();r.advance();assert.equal(r.calls[1].value.snapshotTick,3);
  r.calls[1].resolve();await exact;assert.equal(done,true);r.queue.close();
});
test('failed renders leave queue usable, invalid snapshots fail before renderer and close drops pending work',async()=>{
  const r=rig();assert.throws(()=>r.queue.offer({...snapshot(1),qpos:[NaN]}),/Invalid/);
  r.queue.offer(snapshot(1));r.advance();r.calls[0].reject(Error('renderer unavailable'));await flush();
  assert.deepEqual(r.errors,['renderer unavailable']);r.queue.offer(snapshot(2));r.advance(50);
  r.calls[1].resolve();await flush();assert.equal(r.frames.length,1);
  r.queue.offer(snapshot(3));r.queue.close();r.advance(1000);assert.equal(r.calls.length,2);
});
test('reply identity mismatch fails exact capture without publishing another tick',async()=>{
  const errors=[],frames=[];const queue=new LatestFrameQueue({render:async()=>({png:'',generation:0,snapshotTick:99}),
    onFrame:x=>frames.push(x),onError:e=>errors.push(e.message)});
  await assert.rejects(queue.exact(snapshot(1)),/identity mismatch/);
  assert.equal(frames.length,0);assert.equal(errors.length,1);queue.close();
});
