#!/usr/bin/env node
'use strict';
// Records the actual isolated X11 framebuffer alongside an existing live trial.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {spawn}=require('node:child_process'),readline=require('node:readline');
async function main(configPath,driverPath,mediaDir){
  const config=JSON.parse(fs.readFileSync(configPath));
  fs.mkdirSync(mediaDir,{recursive:false});
  const video=path.join(mediaDir,'authentic-rek-policy-fight.webm');
  const driverLog=fs.createWriteStream(path.join(mediaDir,'driver.stdout.jsonl'),{flags:'wx'});
  const stderr=fs.createWriteStream(path.join(mediaDir,'driver.stderr.txt'),{flags:'wx'});
  let recorder=null,recordingStarted=null,recordingEnd=null,recordingResult=null,policyStarted=null;
  const args=['-e','ximagesrc','display-name=:98','show-pointer=false','use-damage=false','num-buffers=6300',
    '!','video/x-raw,framerate=30/1','!','videoconvert','!','vp8enc','deadline=1','cpu-used=8','threads=4',
    'target-bitrate=6000000','keyframe-max-dist=60','!','webmmux','!','filesink',`location=${video}`];
  function startRecording(trigger){
    if(recorder)return;
    recordingStarted=new Date().toISOString();
    fs.writeFileSync(path.join(mediaDir,'capture-command.json'),JSON.stringify({command:'gst-launch-1.0',args,trigger,utc:recordingStarted},null,2)+'\n',{flag:'wx'});
    recorder=spawn('gst-launch-1.0',args,{stdio:['ignore','pipe','pipe']});
    recorder.stdout.pipe(fs.createWriteStream(path.join(mediaDir,'capture.stdout.txt'),{flags:'wx'}));
    recorder.stderr.pipe(fs.createWriteStream(path.join(mediaDir,'capture.stderr.txt'),{flags:'wx'}));
    recordingEnd=new Promise(resolve=>{
      recorder.on('error',e=>{recordingResult={error:e.message};resolve();});
      recorder.on('close',(code,signal)=>{recordingResult={code,signal,utc:new Date().toISOString()};resolve();});
    });
    console.log(JSON.stringify({event:'capture_started',utc:recordingStarted,video}));
  }
  const driver=spawn(process.execPath,[driverPath,configPath],{stdio:['ignore','pipe','pipe']});driver.stderr.pipe(stderr);
  const lines=readline.createInterface({input:driver.stdout});
  lines.on('line',line=>{
    driverLog.write(line+'\n');
    try{const e=JSON.parse(line);
      if(e.event==='command_result'&&e.command==='StartRound'&&e.status==='accepted')startRecording('native_StartRound_ack');
      if(e.event==='live_policy_started'){policyStarted=e.utc;startRecording('already_active_round');}
      if(['command_result','live_policy_started','summary','error'].includes(e.event))console.log(line);
    }catch(error){if(!(error instanceof SyntaxError))console.error(error.message);}
  });
  const driverResult=await new Promise((resolve,reject)=>{driver.on('error',reject);driver.on('close',(code,signal)=>resolve({code,signal}));});
  driverLog.end();
  if(!recorder)throw Error('No recording: the driver never reached a private fight');
  await new Promise(resolve=>setTimeout(resolve,5000));
  if(recorder.exitCode===null)recorder.kill('SIGINT');
  await recordingEnd;
  if(!fs.existsSync(video)||fs.statSync(video).size===0)throw Error('Capture did not produce a video');
  const hash=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(video))hash.update(chunk);
  const manifest={recording:'fresh_authentic_REK_client_trial',not_a_recording_of_prior_r4:true,
    capture_source:'X11 display :98; real game pixels; pointer capture disabled',
    checkpoint_sha256:config.checkpoint_sha256,config_path:configPath,trial_output:config.out,
    recording_started_utc:recordingStarted,live_policy_started_utc:policyStarted,recording_result:recordingResult,
    driver_result:driverResult,video,video_bytes:fs.statSync(video).size,video_sha256:hash.digest('hex'),
    raw_video_no_overlay:true,synthetic_video_frames:false};
  fs.writeFileSync(path.join(mediaDir,'recording-manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({event:'capture_finished',...manifest}));
  if(driverResult.code!==0||recordingResult?.code!==0)process.exitCode=2;
}
if(process.argv.length!==5)throw Error('usage: node record_live_transfer.cjs trial.config.json live_transfer_run.cjs new-media-directory');
main(...process.argv.slice(2)).catch(error=>{console.error(error.message);process.exitCode=1;});
