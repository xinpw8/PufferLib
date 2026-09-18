#!/usr/bin/env node
'use strict';
// Capture the isolated authentic client alongside neutral-input or policy trials.
// Native FFmpeg performs capture/encoding; no Python runtime is involved.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const crypto=require('node:crypto'),readline=require('node:readline');
const {spawn}=require('node:child_process');
const {finished}=require('node:stream/promises');

// Only child objects returned by this wrapper's spawn calls are supervised.
// A broken child or inherited pipe must not keep the wrapper alive indefinitely.
function superviseChild(child,{timeoutMs,graceful=null,graceMs=0,termGraceMs=2000,killGraceMs=2000}) {
  let timer,settled=false,stopReason=null,childError=null;const signals=[];
  let resolveDone;const done=new Promise(resolve=>{resolveDone=resolve;});
  function finish(result) {
    if(settled)return;settled=true;clearTimeout(timer);
    resolveDone({...result,...(childError?{error:childError}:{}),stop_reason:stopReason,
      timed_out:stopReason==='timeout',signals_sent:signals.slice()});
  }
  function signal(name) {
    if(child.exitCode===null&&child.signalCode===null){signals.push(name);try{child.kill(name);}catch{}}
  }
  function terminate() {
    signal('SIGTERM');timer=setTimeout(()=>{
      signal('SIGKILL');timer=setTimeout(()=>{
        for(const stream of [child.stdin,child.stdout,child.stderr])stream?.destroy();
        // No assumption that a nonresponsive child actually exited.
        child.unref();finish({code:child.exitCode,signal:child.signalCode,close_observed:false,cleanup_timeout:true});
      },killGraceMs);
    },termGraceMs);
  }
  function stop(reason='requested_stop') {
    if(settled||stopReason!==null)return;stopReason=reason;clearTimeout(timer);
    if(graceful&&child.exitCode===null&&child.signalCode===null) {
      try{graceful();}catch{}timer=setTimeout(terminate,graceMs);
    }else terminate();
  }
  child.on('error',error=>{childError=error.message;
    if(!child.pid)finish({code:null,close_observed:false});else stop('child_error');});
  child.once('close',(code,signal)=>finish({code,signal,close_observed:true}));
  timer=setTimeout(()=>stop('timeout'),timeoutMs);
  return {done,stop};
}
function recordingLimitEvidence(bytes,selfExited) {
  if(bytes>=18000000)return 'observed_at_or_above_ffmpeg_limit';
  // FFmpeg's packet-granular -fs exit can land close to the requested cap.
  return selfExited&&bytes>=18000000-65536?'self_exit_near_ffmpeg_limit':null;
}
const cleanChildExit=result=>result?.code===0&&result.close_observed===true&&!result.timed_out&&!result.error;

function captureIdentity(config) {
  const controller=config.capture_controller??'neutral_action_1';
  if(controller==='neutral_action_1') {
    if(config.worker||config.checkpoint_sha256)throw Error('policy capture requires an explicit frozen_policy controller');
    return {schema:'rek.passive_defender.capture.v1',controller,checkpoint_sha256:null,
      filename:'authentic-rek-passive-defender.mp4'};
  }
  if(controller!=='frozen_policy'||!/^[a-f0-9]{64}$/.test(config.checkpoint_sha256||'')||
      !Array.isArray(config.worker)||config.worker[2]!==config.checkpoint_sha256)
    throw Error('frozen policy capture requires a pinned worker checkpoint');
  return {schema:'rek.frozen_policy.capture.v1',controller,checkpoint_sha256:config.checkpoint_sha256,
    filename:'authentic-rek-policy-fight.mp4'};
}

function captureArgs(video,seconds=140) {
  if(!path.isAbsolute(video)||!Number.isInteger(seconds)||seconds<1||seconds>190)
    throw Error('absolute capture path and bounded duration required');
  return ['-hide_banner','-loglevel','warning','-f','x11grab','-draw_mouse','0',
    '-show_region','0','-follow_mouse','0','-framerate','20','-video_size','1280x720',
    '-i',':98.0+0,0','-t',String(seconds),'-an','-c:v','libx264','-preset','ultrafast',
    '-threads','2','-pix_fmt','yuv420p','-b:v','700k','-maxrate','700k','-bufsize','1400k',
    '-g','40','-movflags','+faststart','-fs','18000000','-n',video];
}
async function sha(file) {
  const hash=crypto.createHash('sha256');
  for await(const chunk of fs.createReadStream(file))hash.update(chunk);
  return hash.digest('hex');
}
async function main(configPath,driverPath,mediaDir) {
  if(process.platform!=='linux'||os.hostname()!=='spark-4ae3')
    throw Error('capture requires isolated Spark host');
  if(![configPath,driverPath,mediaDir].every(path.isAbsolute))throw Error('absolute paths required');
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  const identity=captureIdentity(config);
  if(!Number.isFinite(config.max_seconds)||config.max_seconds<=0||config.max_seconds>180)
    throw Error('bounded trial required');
  if(typeof config.capture_ffmpeg!=='string'||!path.isAbsolute(config.capture_ffmpeg)||
      !/^[a-f0-9]{64}$/.test(config.capture_ffmpeg_sha256||''))throw Error('pinned native FFmpeg required');
  if(await sha(config.capture_ffmpeg)!==config.capture_ffmpeg_sha256)throw Error('FFmpeg identity mismatch');
  fs.mkdirSync(mediaDir,{mode:0o700});
  const video=path.join(mediaDir,identity.filename);
  const output=[];
  const logFile=name=>{const s=fs.createWriteStream(path.join(mediaDir,name),{flags:'wx',mode:0o600});output.push(s);return s;};
  const driverOut=logFile('driver.stdout.jsonl'),driverErr=logFile('driver.stderr.txt');
  let recorder=null,recordingEnd=null,recordingResult=null,recordingControl=null,started=null,trigger=null;
  let sizeTimer=null,limitReached=false,limitEvidence=null,decodeControl=null;
  const args=captureArgs(video,Math.ceil(config.max_seconds)+10);
  function stopCapture() {
    recordingControl?.stop();
  }
  function startCapture(event) {
    if(recorder)return;
    trigger=event;started=new Date().toISOString();
    fs.writeFileSync(path.join(mediaDir,'capture-command.json'),JSON.stringify({
      command:[config.capture_ffmpeg,...args],sha256:config.capture_ffmpeg_sha256,
      trigger,utc:started,display:':98',pointer_capture:false
    },null,2)+'\n',{flag:'wx',mode:0o600});
    recorder=spawn(config.capture_ffmpeg,args,{stdio:['pipe','pipe','pipe']});
    recorder.stdin.on('error',()=>{});
    recorder.stdout.pipe(logFile('capture.stdout.txt'));recorder.stderr.pipe(logFile('capture.stderr.txt'));
    recordingControl=superviseChild(recorder,{timeoutMs:(Math.ceil(config.max_seconds)+25)*1000,
      graceful:()=>{if(recorder.stdin.writable)recorder.stdin.write('q\n');},graceMs:10000});
    recordingEnd=recordingControl.done.then(result=>{recordingResult={...result,utc:new Date().toISOString()};});
    sizeTimer=setInterval(()=>{
      if(fs.existsSync(video)&&fs.statSync(video).size>=18000000){limitReached=true;limitEvidence='observed_at_or_above_ffmpeg_limit';stopCapture();}
    },250);
    console.log(JSON.stringify({event:'capture_started',utc:started,display:':98',video}));
  }
  fs.writeFileSync(path.join(mediaDir,'driver-command.json'),JSON.stringify({
    command:[process.execPath,driverPath,configPath],host:os.hostname(),utc:new Date().toISOString(),
    driver_sha256:await sha(driverPath),config_sha256:await sha(configPath)
  },null,2)+'\n',{flag:'wx',mode:0o600});
  const driver=spawn(process.execPath,[driverPath,configPath],{stdio:['ignore','pipe','pipe']});
  // Driver may need three 10s stop/release/readback waits plus relay shutdown.
  const driverControl=superviseChild(driver,{timeoutMs:300000,termGraceMs:40000});
  driver.stderr.pipe(driverErr);
  const lines=readline.createInterface({input:driver.stdout});
  lines.on('line',line=>{
    driverOut.write(line+'\n');
    let e;try{e=JSON.parse(line);}catch{return;}
    // Both events follow native isolated/private/no-human proof in the driver.
    if(['active_private_ai_opponent','passive_defender_started'].includes(e.event))startCapture(e.event);
    if(['command_result','active_private_ai_opponent','passive_defender_started','summary','error'].includes(e.event))console.log(line);
  });
  const terminate=()=>{driverControl.stop('operator_signal');stopCapture();decodeControl?.stop('operator_signal');};
  process.once('SIGINT',terminate);process.once('SIGTERM',terminate);
  const driverResult=await driverControl.done;
  lines.close();driverOut.end();
  if(recorder){await new Promise(resolve=>setTimeout(resolve,3000));stopCapture();await recordingEnd;}
  clearInterval(sizeTimer);
  for(const stream of output)if(!stream.writableEnded)stream.end();
  await Promise.all(output.map(s=>finished(s).catch(()=>{})));
  const exists=fs.existsSync(video),bytes=exists?fs.statSync(video).size:0;
  limitEvidence??=recordingLimitEvidence(bytes,recordingResult?.close_observed===true&&recordingResult.stop_reason===null);
  limitReached||=limitEvidence!==null;
  let decodeResult=null;
  if(bytes>0&&bytes<20000000&&recordingResult?.close_observed===true){
    const decode=spawn(config.capture_ffmpeg,['-v','error','-nostdin','-i',video,'-f','null','-'],{stdio:['ignore','ignore','pipe']});
    const errors=fs.createWriteStream(path.join(mediaDir,'decode.stderr.txt'),{flags:'wx',mode:0o600});decode.stderr.pipe(errors);
    decodeControl=superviseChild(decode,{timeoutMs:30000});decodeResult=await decodeControl.done;
    if(!errors.writableEnded)errors.end();await finished(errors);
  }
  const manifest={schema:identity.schema,host:os.hostname(),
    trial_output:config.out,config_path:configPath,controller:identity.controller,checkpoint_sha256:identity.checkpoint_sha256,
    capture_source:'authentic REK framebuffer on isolated X11 :98',global_input_emitted:false,
    pointer_capture:false,synthetic_frames:false,recording_started_utc:started,trigger,
    driver_result:driverResult,recording_result:recordingResult,recording_size_limit_reached:limitReached,
    recording_size_limit_evidence:limitEvidence,
    video:exists?video:null,video_bytes:bytes,
    video_sha256:exists&&recordingResult?.close_observed===true?await sha(video):null,
    decode_result:decodeResult,delivery_status:bytes>0&&bytes<20000000&&cleanChildExit(decodeResult)?'validated_mp4':'invalid_or_missing_video',
    delivery_max_bytes_exclusive:20000000};
  fs.writeFileSync(path.join(mediaDir,'capture-manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx',mode:0o600});
  process.off('SIGINT',terminate);process.off('SIGTERM',terminate);
  console.log(JSON.stringify({event:'capture_finished',...manifest}));
  return cleanChildExit(driverResult)&&cleanChildExit(recordingResult)&&bytes>0&&bytes<20000000&&
    cleanChildExit(decodeResult)&&!limitReached?0:2;
}
module.exports={captureIdentity,captureArgs,superviseChild,recordingLimitEvidence,cleanChildExit,main};
if(require.main===module){
  if(process.argv.length!==5)throw Error('usage: record_passive_defender.cjs CONFIG DRIVER NEW_MEDIA_DIR');
  main(...process.argv.slice(2)).then(code=>{process.exitCode=code;}).catch(e=>{console.error(e.message);process.exitCode=2;});
}
