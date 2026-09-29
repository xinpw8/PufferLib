'use strict';
// Only transport observation and a fixed human-side guard. Simulation is unchanged.
const fs=require('node:fs'),path=require('node:path');
const {heartbeatGate}=require('./league/heartbeat_gate.cjs');
const {framePayload}=require('./league/frame_payload.cjs');
const {PerformanceStats}=require('./league/performance_stats.cjs');
const {performance}=require('node:perf_hooks');
const root=__dirname,run=path.resolve(process.argv[2]||''),source=path.join(root,'league');
if(!process.argv[2]||!fs.existsSync(path.join(run,'identity.json')))throw Error('Prepared run directory required');
for(const key of Object.keys(process.env))if(key.startsWith('REK_'))delete process.env[key];
const trace=fs.openSync(path.join(run,'session.jsonl'),'wx',0o600);
const frameDirectory=path.join(run,'frames');fs.mkdirSync(frameDirectory,{mode:0o700});
let sequence=0,workerNumber=0,frames=0,steps=0,lastState=null,recorderFailure=null,workerFailure=null,rendererFailure=null,lastRequestError=null;
const origin=process.hrtime.bigint();
const metrics=new PerformanceStats();
function append(kind,data={}){
  try{
    const started=performance.now();
    const line=JSON.stringify({sequence:++sequence,utc:new Date().toISOString(),
      monotonicNs:process.hrtime.bigint().toString(),kind,...data})+'\n';
    const serialized=performance.now();fs.writeSync(trace,line);
    metrics.record('record_serialize_ms',serialized-started);
    metrics.record('record_write_ms',performance.now()-serialized);
  }
  catch(error){recorderFailure=error.message;throw error;}
}
const heartbeat=heartbeatGate(()=>{
  const status={utc:new Date().toISOString(),pid:process.pid,sequence,steps,frames,
    lastTick:lastState?.tick??null,lastScore:lastState?.score??null,
    ok:recorderFailure===null&&workerFailure===null&&(lastState?.ok??false),recorderFailure,workerFailure,rendererFailure,lastRequestError,
    elapsedSeconds:Number(process.hrtime.bigint()-origin)/1e9,performance:metrics.snapshot()};
  const temporary=path.join(run,'recorder-health.tmp');
  try{
    fs.writeFileSync(temporary,JSON.stringify(status)+'\n',{mode:0o600});
    fs.renameSync(temporary,path.join(run,'recorder-health.json'));
  }catch(error){recorderFailure=error.message;throw error;}
});
const inputModule=require(path.join(source,'input.cjs'));
const OriginalInput=inputModule.HumanInput;
inputModule.HumanInput=class extends OriginalInput{
  update(value){const accepted=super.update(value);append('human_input',{value,accepted});return accepted;}
  release(){super.release();append('human_release');}
  next(state,side){const command=super.next(state,side);append('resolved_input',{tick:state.tick,side,held:[...this.held],command});return command;}
};
const {NativeWorker}=require(path.join(source,'worker.cjs'));
class RecordedWorker extends NativeWorker{
  constructor(options){
    super({...options,onTransportTiming:(name,ms)=>metrics.record((options.renderOnly?'renderer_':'simulation_')+name,ms)});this.recordingId=++workerNumber;this.expectedClose=false;this.renderOnly=options.renderOnly===true;
    if(this.renderOnly)rendererFailure=null;else workerFailure=null;
    append('worker_created',{worker:this.recordingId,role:this.renderOnly?'renderer':'simulation',pid:this.child.pid,
      config:JSON.parse(fs.readFileSync(options.config,'utf8')),env:options.env});
    this.child.on('exit',(code,signal)=>{
      if(!this.expectedClose){if(this.renderOnly)rendererFailure=`Renderer exited (${code??signal})`;else workerFailure=`Native worker exited (${code??signal})`;}
      append('worker_exit',{worker:this.recordingId,code,signal,expected:this.expectedClose});heartbeat(true);
    });
    this.child.on('error',error=>{if(this.renderOnly)rendererFailure=error.message;else workerFailure=error.message;append('worker_process_error',{worker:this.recordingId,message:error.message});heartbeat(true);});
  }
  async close(){this.expectedClose=true;await super.close();}
  async request(op,args={},timeout){
    if(op==='step'&&args.humanSide!==0)throw Error('This evaluator supports the blue human side only.');
    if(op==='policy')throw Error('This evaluator keeps the recovered Bot1 opponent fixed.');
    if(op==='reset'&&lastState)append('reset_boundary',{prior:lastState,outcome:lastState.fightResult>0?'completed_match':lastState.tick>0?'unfinished_reset':'unstarted_reset'});
    const started=process.hrtime.bigint();
    append('worker_request',{worker:this.recordingId,op,args});
    try{
      const result=await super.request(op,args,timeout);
      const durationMs=Number(process.hrtime.bigint()-started)/1e6;
      metrics.record(op==='frame'?'render_rpc_ms':'native_'+op+'_rpc_ms',durationMs);
      if(op==='frame'){
        if(!this.renderOnly||result.snapshotTick!==args.snapshotTick||result.generation!==args.generation)
          throw Error('Renderer frame identity mismatch');
        const decodedAt=performance.now();
        const {png,sha256}=framePayload(result);
        metrics.record('frame_decode_hash_ms',performance.now()-decodedAt);
        const filename=String(++frames).padStart(8,'0')+'.png';
        const writtenAt=performance.now();
        try{await fs.promises.writeFile(path.join(frameDirectory,filename),png,{flag:'wx',mode:0o600});
          metrics.record('frame_write_elapsed_ms',performance.now()-writtenAt);}
        catch(error){recorderFailure=error.message;throw error;}
        append('rendered_frame',{worker:this.recordingId,requestId:result.id,durationMs,
          tick:args.snapshotTick,generation:args.generation,path:'frames/'+filename,bytes:png.length,
          sha256});
        rendererFailure=null;
      }else{
        append('worker_reply',{worker:this.recordingId,op,durationMs,result});
        if(result.state)lastState=result.state;
        if(op==='step')steps++;
      }
      if(op==='step')heartbeat();
      else if(op==='snapshot'||op==='reset')heartbeat(true);
      return result;
    }catch(error){lastRequestError={op,message:error.message,utc:new Date().toISOString()};
      if(this.renderOnly)rendererFailure=error.message;else if(this.closed)workerFailure=error.message;
      append('worker_error',{worker:this.recordingId,op,message:error.message,workerClosed:this.closed});heartbeat(true);throw error;}
  }
}
append('session_started',{schema:'rek.native_clone.session.v1',role:'human_evaluation',
  note:'Manual continuous command control and explicit paused stepping. Native adjudication, recorded inputs/states/frames. Full parity unverified.'});
const {serve}=require(path.join(source,'server.cjs'));
serve(path.join(run,'server.json'),{Worker:RecordedWorker,Renderer:RecordedWorker,intermissionMs:0,diagnostics:()=>metrics.snapshot()}).then(({server})=>{
  server.on('close',()=>{append('server_closed');fs.fsyncSync(trace);heartbeat(true);metrics.close();});
  heartbeat(true);
  setInterval(()=>{append('performance_sample',{performance:metrics.snapshot()});heartbeat(true);},5000).unref();
}).catch(error=>{append('startup_failure',{message:error.message});console.error(error);process.exitCode=1;});
