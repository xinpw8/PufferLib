'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {heartbeatGate}=require('./heartbeat_gate.cjs');

test('50 Hz step health writes coalesce to 250 ms while every trace record remains',()=>{
  let time=0,latest=0;const health=[],trace=[];
  const heartbeat=heartbeatGate(()=>health.push({time,tick:latest}),{now:()=>time});
  for(let tick=0;tick<50;tick++){
    time=tick*20;latest=tick;trace.push({tick});heartbeat();
  }
  assert.equal(trace.length,50);assert.deepEqual(trace.map(x=>x.tick),Array.from({length:50},(_,i)=>i));
  assert.deepEqual(health,[{time:0,tick:0},{time:260,tick:13},{time:520,tick:26},{time:780,tick:39}]);
});
test('reset, snapshot, error and exit health force the current state immediately',()=>{
  let time=0,state='step';const values=[];
  const heartbeat=heartbeatGate(()=>values.push(state),{now:()=>time});
  heartbeat();
  for(const value of ['reset','snapshot','error','exit']){state=value;time++;assert.equal(heartbeat(true),true);}
  assert.deepEqual(values,['step','reset','snapshot','error','exit']);
});
test('failed write propagates and does not suppress the next attempted status',()=>{
  let fail=true,calls=0;
  const heartbeat=heartbeatGate(()=>{calls++;if(fail)throw Error('disk failed');},{now:()=>0});
  assert.throws(()=>heartbeat(),/disk failed/);fail=false;
  assert.equal(heartbeat(),true);assert.equal(calls,2);assert.equal(heartbeat(),false);
});
