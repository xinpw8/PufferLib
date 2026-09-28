'use strict';
// Test-only preload: replace transport before launch_logged imports it. No native
// executable, process, GPU or external service is constructed by this fixture.
const {EventEmitter}=require('node:events');
const workerPath=require.resolve('./worker.cjs');
class FakeWorker {
  constructor(options){
    this.renderOnly=options.renderOnly===true;this.closed=false;this.serial=0;
    this.child=new EventEmitter();this.child.pid=process.pid;
    this.ready=Promise.resolve({rendererOnly:this.renderOnly});this.reset();
  }
  reset(){this.state={ok:true,tick:0,phase:2,score:[0,0],roundNumber:1,terminal:0,
    fightResult:0,fightWinner:-1,roundResult:0,winner:-1,qpos:Array(72).fill(.25),
    qvel:Array(70).fill(.125),raw:Array(446).fill(0),mask:Array(66).fill(1)};}
  async request(op,args={}){
    const id=++this.serial;
    if(op==='frame')return {id,ok:true,snapshotTick:args.snapshotTick,generation:args.generation,
      png:'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2n0kAAAAASUVORK5CYII='};
    if(op==='reset')this.reset();
    if(op==='step'){
      if(args.command?.moveIndex===16)throw Error('Injected request failure');
      this.state.tick+=args.steps;this.state.qpos[0]=this.state.tick;
    }
    return {id,ok:true,state:structuredClone(this.state),rounds:[],commandEvents:[]};
  }
  async close(){this.closed=true;this.child.emit('exit',0,null);}
}
require.cache[workerPath]={id:workerPath,filename:workerPath,loaded:true,exports:{NativeWorker:FakeWorker}};
