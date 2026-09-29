'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {PerformanceStats}=require('./performance_stats.cjs');

test('performance samples remain bounded while lifetime stalls are retained',()=>{
  const metrics=new PerformanceStats({capacity:4,observeEventLoop:false});
  for(const n of [100,2,3,4,5,6])metrics.record('record_write_ms',n);
  const value=metrics.snapshot(),span=value.spans.record_write_ms;
  assert.equal(span.count,6);assert.equal(span.totalMs,120);assert.equal(span.maxMs,100);
  assert.equal(span.recentCount,4);assert.equal(span.recentP50Ms,4);assert.equal(span.recentP99Ms,5);
  assert.equal(metrics.spans.get('record_write_ms').values.length,4);
  assert.equal(value.eventLoop,null);metrics.close();
});
test('performance samples reject invalid values and retain separate components',()=>{
  assert.throws(()=>new PerformanceStats({capacity:0}),/capacity/);
  const metrics=new PerformanceStats({observeEventLoop:false});
  for(const value of [NaN,Infinity,-1])assert.throws(()=>metrics.record('rpc_ms',value),/sample/);
  metrics.record('rpc_ms',48);metrics.record('record_write_ms',.2);
  const spans=metrics.snapshot().spans;assert.equal(spans.rpc_ms.meanMs,48);assert.equal(spans.record_write_ms.meanMs,.2);
  metrics.close();
});

test('repeated HTTP snapshots reuse bounded statistics without losing newly recorded samples',()=>{
  let clock=1000;
  const metrics=new PerformanceStats({capacity:4,observeEventLoop:false,now:()=>clock});
  metrics.record('rpc_ms',10);const first=metrics.snapshot();assert.equal(first.snapshotAgeMs,0);
  metrics.record('rpc_ms',40);clock+=249;
  const cached=metrics.snapshot();assert.equal(cached.spans,first.spans,'no quantile rebuild inside250ms');
  assert.equal(cached.spans.rpc_ms.count,1);assert.equal(cached.snapshotAgeMs,249);
  clock++;const fresh=metrics.snapshot();assert.notEqual(fresh.spans,first.spans);
  assert.equal(fresh.spans.rpc_ms.count,2);assert.equal(fresh.spans.rpc_ms.maxMs,40);assert.equal(fresh.snapshotAgeMs,0);
  metrics.record('rpc_ms',90);const forced=metrics.snapshot({force:true});
  assert.equal(forced.spans.rpc_ms.count,3);assert.equal(forced.spans.rpc_ms.maxMs,90);metrics.close();
});
