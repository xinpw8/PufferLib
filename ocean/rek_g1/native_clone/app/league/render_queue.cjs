'use strict';
const {performance}=require('node:perf_hooks');

// One renderer request and one replaceable snapshot are retained. Physics never
// awaits this queue during play; explicit paused captures use exact().
class LatestFrameQueue {
  constructor({render,onFrame=()=>{},onError=()=>{},periodMs=50,
    now=()=>performance.now(),setTimer=setTimeout,clearTimer=clearTimeout}){
    this.render=render;this.onFrame=onFrame;this.onError=onError;this.periodMs=periodMs;
    this.now=now;this.setTimer=setTimer;this.clearTimer=clearTimer;
    this.generation=0;this.pending=null;this.running=false;this.timer=null;
    this.closed=false;this.lastStart=-Infinity;this.lastPublished=-1;
  }
  invalidate(generation){
    this.generation=generation;this.lastPublished=-1;this.lastStart=-Infinity;
    if(this.timer!==null)this.clearTimer(this.timer);this.timer=null;
    if(this.pending?.reject)this.pending.reject(new Error('Render generation changed'));
    this.pending=null;
  }
  snapshot(value){
    if(this.closed)throw Error('Renderer queue closed');
    if(!Number.isSafeInteger(value.generation)||value.generation!==this.generation)
      throw Error('Render generation changed');
    if(!Number.isSafeInteger(value.snapshotTick)||value.snapshotTick<0||
      !Array.isArray(value.qpos)||value.qpos.length!==72||!value.qpos.every(Number.isFinite))
      throw Error('Invalid render snapshot');
    // Optional third-person camera target; absent keeps the two-fighter overview.
    if(value.followSide!==undefined&&value.followSide!==0&&value.followSide!==1)
      throw Error('Invalid render follow side');
    const snapshot={generation:value.generation,snapshotTick:value.snapshotTick,qpos:[...value.qpos]};
    if(value.followSide!==undefined)snapshot.followSide=value.followSide;
    return snapshot;
  }
  offer(value){
    const snapshot=this.snapshot(value);
    if(this.pending?.resolve)return false;
    this.pending={snapshot};this.schedule();return true;
  }
  exact(value){
    const snapshot=this.snapshot(value);
    if(this.pending?.resolve)return Promise.reject(Error('Exact render already pending'));
    return new Promise((resolve,reject)=>{
      this.pending={snapshot,resolve,reject};
      if(this.timer!==null)this.clearTimer(this.timer);this.timer=null;
      this.schedule();
    });
  }
  schedule(){
    if(this.closed||this.running||this.timer!==null||!this.pending)return;
    const delay=this.pending.resolve?0:Math.max(0,this.periodMs-(this.now()-this.lastStart));
    this.timer=this.setTimer(()=>{this.timer=null;void this.pump();},delay);
  }
  async pump(){
    if(this.closed||this.running||!this.pending)return;
    const job=this.pending;this.pending=null;this.running=true;this.lastStart=this.now();
    try{
      const reply=await this.render(job.snapshot);
      if(reply.generation!==job.snapshot.generation||reply.snapshotTick!==job.snapshot.snapshotTick||typeof reply.png!=='string')
        throw Error('Renderer snapshot identity mismatch');
      if(this.closed||job.snapshot.generation!==this.generation)throw Error('Render generation changed');
      if(job.snapshot.snapshotTick>=this.lastPublished){
        this.onFrame(reply,{...job.snapshot,durationMs:this.now()-this.lastStart});
        this.lastPublished=job.snapshot.snapshotTick;
      }
      if(job.resolve)job.resolve(reply);
    }catch(error){
      if(!this.closed&&job.snapshot.generation===this.generation)this.onError(error);
      if(job.reject)job.reject(error);
    }finally{this.running=false;this.schedule();}
  }
  close(){this.invalidate(this.generation+1);this.closed=true;}
}
module.exports={LatestFrameQueue};
