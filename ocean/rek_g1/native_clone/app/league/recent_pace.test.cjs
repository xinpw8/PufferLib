'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {RecentPace}=require('./recent_pace.cjs');
test('recent pace follows actual fast and slow completions without changing cumulative time or inventing intervals',()=>{
  const pace=new RecentPace();let now=0;pace.complete(now);
  for(let i=0;i<300;i++)pace.complete(now+=20);
  assert.deepEqual([pace.snapshot().activeIntervals,pace.snapshot().activeWallMs,pace.snapshot().realTimeRatio],[250,5000,1]);
  for(let i=0;i<100;i++)pace.complete(now+=50);
  assert.deepEqual([pace.snapshot().activeIntervals,pace.snapshot().activeWallMs,pace.snapshot().realTimeRatio],[100,5000,.4]);
  for(let i=0;i<500;i++)pace.complete(now+=10);
  assert.deepEqual([pace.snapshot().activeIntervals,pace.snapshot().activeWallMs,pace.snapshot().realTimeRatio],[500,5000,2]);
});
test('pause gap is excluded and new play/reset begins a fresh measurement',()=>{
  const pace=new RecentPace();pace.complete(0);pace.complete(20);pace.breakSpan();
  pace.complete(10000);assert.equal(pace.snapshot().activeWallMs,20);
  pace.complete(10040);assert.equal(pace.snapshot().activeWallMs,60);assert.equal(pace.snapshot().activeIntervals,2);
  pace.reset();assert.equal(pace.snapshot().realTimeRatio,null);pace.complete(12000);assert.equal(pace.snapshot().activeIntervals,0);
  pace.complete(12050);assert.equal(pace.snapshot().realTimeRatio,.4);
});
test('whole long intervals and capacity limits stay explicit',()=>{
  const pace=new RecentPace({capacity:3});pace.complete(0);pace.complete(6000);
  assert.equal(pace.snapshot().activeWallMs,6000);assert.equal(pace.snapshot().simulationMs,20);
  pace.reset();for(let i=0;i<5;i++)pace.complete(i);
  assert.equal(pace.snapshot().activeIntervals,3);assert.equal(pace.snapshot().capacityLimited,true);
  assert.throws(()=>pace.complete(0),/backward/);assert.throws(()=>pace.complete(NaN),/Finite/);
});
