'use strict';
const {performance,monitorEventLoopDelay}=require('node:perf_hooks');

// Bounded samples measure transport/recording separately from the native RPC.
// These spans can overlap and must not be added as if they were serial work.
class PerformanceStats {
  constructor({capacity=1024,observeEventLoop=true,snapshotPeriodMs=250,now=()=>performance.now()}={}){
    if(!Number.isSafeInteger(capacity)||capacity<1)throw Error('Invalid timing capacity');
    if(!Number.isFinite(snapshotPeriodMs)||snapshotPeriodMs<=0)throw Error('Invalid timing snapshot period');
    this.capacity=capacity;this.spans=new Map();
    this.now=now;this.snapshotPeriodMs=snapshotPeriodMs;this.cached=null;this.cachedAt=-Infinity;
    this.loop=observeEventLoop?monitorEventLoopDelay({resolution:10}):null;
    this.loop?.enable();this.started=this.now();
  }
  record(name,ms){
    if(!Number.isFinite(ms)||ms<0)throw Error('Invalid timing sample');
    let span=this.spans.get(name);
    if(!span){span={values:new Float64Array(this.capacity),count:0,totalMs:0,maxMs:0};this.spans.set(name,span);}
    span.values[span.count%this.capacity]=ms;span.count++;span.totalMs+=ms;span.maxMs=Math.max(span.maxMs,ms);
  }
  snapshot({force=false}={}){
    const sampledAt=this.now();
    if(!force&&this.cached&&sampledAt-this.cachedAt<this.snapshotPeriodMs)
      return {...this.cached,snapshotAgeMs:Math.max(0,sampledAt-this.cachedAt)};
    const spans={};
    for(const [name,span] of this.spans){
      const values=Array.from(span.values.subarray(0,Math.min(span.count,this.capacity))).sort((a,b)=>a-b);
      const quantile=q=>values[Math.floor((values.length-1)*q)];
      spans[name]={count:span.count,totalMs:span.totalMs,meanMs:span.totalMs/span.count,maxMs:span.maxMs,
        recentCount:values.length,recentP50Ms:quantile(.5),recentP95Ms:quantile(.95),recentP99Ms:quantile(.99)};
    }
    const memory=process.memoryUsage();
    this.cachedAt=sampledAt;
    this.cached={schema:'rek.performance.v1',elapsedMs:sampledAt-this.started,sampleCapacity:this.capacity,
      snapshotPeriodMs:this.snapshotPeriodMs,snapshotAgeMs:0,
      spans,eventLoop:this.loop?.count?{count:this.loop.count,meanDelayMs:this.loop.mean/1e6,
        p99DelayMs:this.loop.percentile(99)/1e6,maxDelayMs:this.loop.max/1e6}:null,
      eventLoopUtilization:performance.eventLoopUtilization(),
      memory:{rssBytes:memory.rss,heapUsedBytes:memory.heapUsed,externalBytes:memory.external},
      note:`Cumulative counters and bounded recent span samples sampled at most once per${this.snapshotPeriodMs}ms unless forced. snapshotAgeMs exposes cache age. Overlapping spans are not additive. Event-loop delay includes the 10 ms monitor interval.`};
    return this.cached;
  }
  close(){this.loop?.disable();}
}
module.exports={PerformanceStats};
