'use strict';
const {spawn}=require('node:child_process');
const readline=require('node:readline');
const fs=require('node:fs');
const {performance}=require('node:perf_hooks');

class NativeWorker {
  constructor({executable,config,env={},logFile,renderOnly=false,spawnProcess=spawn,onTransportTiming=()=>{}}){
    this.serial=0;this.pending=new Map();this.closed=false;this.renderOnly=renderOnly;
    const log=logFile?fs.openSync(logFile,'a',0o600):'inherit';
    this.child=spawnProcess(executable,[...(renderOnly?['--render-only']:[]),'--config',config],{env:{...process.env,...env},
      stdio:['pipe','pipe',log],windowsHide:true});
    if(typeof log==='number')fs.closeSync(log);
    this.ready=new Promise((resolve,reject)=>{
      this.readyResolve=resolve;this.readyReject=reject;
      this.readyTimer=setTimeout(()=>reject(new Error('Native worker startup timeout')),60000);
    });
    readline.createInterface({input:this.child.stdout}).on('line',line=>{
      const started=performance.now();
      let value;try{value=JSON.parse(line);}catch{return;}
      onTransportTiming('json_parse_ms',performance.now()-started);
      if(value.event==='ready'){
        if(renderOnly&&value.rendererOnly!==true){
          this.fail(new Error('Native renderer role mismatch'));this.child.kill('SIGTERM');return;
        }
        clearTimeout(this.readyTimer);this.readyResolve(value);return;
      }
      if(value.event==='fatal'){this.fail(new Error(value.error));return;}
      const pending=this.pending.get(value.id);if(!pending)return;
      clearTimeout(pending.timer);this.pending.delete(value.id);
      if(value.ok)pending.resolve(value);else pending.reject(new Error(value.error||'Native worker failed'));
    });
    this.child.on('error',error=>this.fail(error));
    this.child.on('exit',(code,signal)=>this.fail(new Error(`Native worker exited (${code??signal})`)));
  }
  fail(error){
    this.closed=true;clearTimeout(this.readyTimer);this.readyReject(error);
    for(const pending of this.pending.values()){clearTimeout(pending.timer);pending.reject(error);}
    this.pending.clear();
  }
  async request(op,args={},timeout=30000){
    await this.ready;if(this.closed)throw new Error('Native worker unavailable');
    const id=++this.serial;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{
        const error=new Error(`Native ${op} timeout`);
        if(this.renderOnly){this.fail(error);this.child.kill('SIGTERM');}
        else{this.pending.delete(id);reject(error);}
      },timeout);
      this.pending.set(id,{resolve,reject,timer});
      this.child.stdin.write(JSON.stringify({id,op,...args})+'\n');
    });
  }
  async close(){
    if(this.closed)return;
    this.child.stdin.end();
    await new Promise(resolve=>{
      const timer=setTimeout(()=>{this.child.kill('SIGTERM');resolve();},3000);
      this.child.once('exit',()=>{clearTimeout(timer);resolve();});
    });
  }
}
module.exports={NativeWorker};
