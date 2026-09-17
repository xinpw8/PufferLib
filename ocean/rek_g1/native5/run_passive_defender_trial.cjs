#!/usr/bin/env node
'use strict';
// Reuse the deployed private relay, without loading a policy or observation encoder.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const {spawn}=require('node:child_process');
const {finished}=require('node:stream/promises');
const FFMPEG='/home/spark-advantage/.cache/uv/archive-v0/h1CxqthI2Pr-azmQXP7WS/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2';
const FFMPEG_SHA='6bb182d0d75d23028db82e9e4f723ca69b853d055698486e6984ddb2c06fb8ce';
async function digest(file){const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(file))h.update(b);return h.digest('hex');}
async function main(basePath,out,mode){
  if(os.hostname()!=='spark-4ae3'||process.platform!=='linux')throw Error('Spark required');
  if(![basePath,out].every(path.isAbsolute)||!['safe_start','active_attach'].includes(mode))throw Error('absolute paths and explicit start mode required');
  const base=JSON.parse(fs.readFileSync(basePath,'utf8'));
  if(!Array.isArray(base.relay)||!base.relay.includes('policy-relay')||!base.relay.includes('DISPLAY=:98')||
      !/^[a-f0-9]{64}$/.test(base.relay.at(-1)))throw Error('existing pinned isolated relay required');
  if(await digest(FFMPEG)!==FFMPEG_SHA)throw Error('native capture executable hash mismatch');
  fs.mkdirSync(out,{mode:0o700});
  const write=(name,data)=>fs.writeFileSync(path.join(out,name),JSON.stringify(data,null,2)+'\n',{flag:'wx',mode:0o600});
  const config={mode,enter_private:true,max_seconds:130,out:path.join(out,'trial'),relay:base.relay,
    capture_ffmpeg:FFMPEG,capture_ffmpeg_sha256:FFMPEG_SHA};
  const configPath=path.join(out,'config.json');write('config.json',config);
  const runner=path.join(__dirname,'passive_defender_run.cjs'),recorder=path.join(__dirname,'record_passive_defender.cjs');
  write('provenance.json',{host:os.hostname(),utc:new Date().toISOString(),command:[process.execPath,...process.argv.slice(1)],
    base_config_sha256:await digest(basePath),config_sha256:await digest(configPath),
    source_sha256:Object.fromEntries(await Promise.all(['passive_defender_run.cjs','live_transfer_run.cjs',
      'record_passive_defender.cjs','audit_passive_defender.cjs','run_passive_defender_trial.cjs'].map(async n=>[n,await digest(path.join(__dirname,n))]))),
    bridge_sha256:base.relay.at(-1),policy_used:false,global_input_emitted:false});
  const command=[recorder,configPath,runner,path.join(out,'media')];write('command.json',[process.execPath,...command]);
  const stdout=fs.createWriteStream(path.join(out,'stdout.jsonl'),{flags:'wx'}),stderr=fs.createWriteStream(path.join(out,'stderr.txt'),{flags:'wx'});
  const child=spawn(process.execPath,command,{stdio:['ignore','pipe','pipe']});
  child.stdout.pipe(stdout);child.stdout.pipe(process.stdout);child.stderr.pipe(stderr);child.stderr.pipe(process.stderr);
  const stop=()=>{if(child.exitCode===null)child.kill('SIGTERM');};
  process.once('SIGINT',stop);process.once('SIGTERM',stop);
  const result=await new Promise(resolve=>{
    child.once('error',error=>resolve({code:null,error:error.message}));child.once('close',(code,signal)=>resolve({code,signal}));
  });
  process.off('SIGINT',stop);process.off('SIGTERM',stop);
  await Promise.all([finished(stdout),finished(stderr)]);write('exit.json',result);
  return result.code??2;
}
module.exports={main};
if(require.main===module){
  if(process.argv.length!==5)throw Error('usage: run_passive_defender_trial.cjs EXISTING_LIVE_CONFIG NEW_OUTPUT safe_start|active_attach');
  main(...process.argv.slice(2)).then(code=>{process.exitCode=code;}).catch(e=>{console.error(e.message);process.exitCode=2;});
}
