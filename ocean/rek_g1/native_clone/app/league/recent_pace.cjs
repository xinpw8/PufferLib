'use strict';
// Whole observed completion-to-completion intervals only. The window retains
// the boundary interval, so coverage can exceed the requested duration.
class RecentPace {
  constructor({windowMs=5000,stepMs=20,capacity=512}={}){
    if(!Number.isFinite(windowMs)||windowMs<=0||!Number.isFinite(stepMs)||stepMs<=0||
      !Number.isSafeInteger(capacity)||capacity<1)throw Error('Invalid recent pace configuration');
    this.windowMs=windowMs;this.stepMs=stepMs;this.capacity=capacity;
    this.values=new Float64Array(capacity);this.reset();
  }
  reset(){this.head=0;this.count=0;this.wallMs=0;this.previous=null;}
  breakSpan(){this.previous=null;}
  complete(now){
    if(!Number.isFinite(now))throw Error('Finite completion clock required');
    if(this.previous!==null){
      if(now<this.previous)throw Error('Completion clock moved backward');
      const interval=now-this.previous;
      if(this.count===this.capacity)this.shift();
      this.values[(this.head+this.count)%this.capacity]=interval;this.count++;this.wallMs+=interval;
      while(this.count>1&&this.wallMs-this.values[this.head]>=this.windowMs)this.shift();
    }
    this.previous=now;
  }
  shift(){this.wallMs-=this.values[this.head];this.head=(this.head+1)%this.capacity;this.count--;}
  snapshot(){return {targetWindowMs:this.windowMs,activeIntervals:this.count,activeWallMs:this.wallMs,
    simulationMs:this.count*this.stepMs,realTimeRatio:this.wallMs>0?this.count*this.stepMs/this.wallMs:null,
    maxIntervals:this.capacity,capacityLimited:this.count===this.capacity&&this.wallMs<this.windowMs,
    scope:'Automatic play, whole completed control intervals; pauses excluded and new play span clears history.'};}
}
module.exports={RecentPace};
