'use strict';
// Private configuration for the compact CUDA candidate, independent of 18768.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {League,hashFile}=require('./league.cjs');

const DEFAULT_DURATIONS=[35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103];
const FAST_SETTINGS={move_speed:[1,.01,10],yaw_speed:[1.8,.01,10],body_radius:[.22,.05,1],
  hit_speed:[.35,0,20],down_damage:[4,1,100]};
function positive(value,name){
  if(!Number.isSafeInteger(value)||value<=0)throw new Error(`Expected positive integer ${name}`);
  return value;
}
function prepare(run,executable,runtimeConfig){
  for(const [name,value] of Object.entries({run,executable,runtimeConfig}))
    if(typeof value!=='string'||!path.isAbsolute(value))throw new Error(`Absolute ${name} path required`);
  const requested=JSON.parse(fs.readFileSync(runtimeConfig,'utf8'));
  if(requested.backend!=='semantic_cuda')throw new Error('Explicit semantic_cuda runtime configuration required');
  const base={backend:'semantic_cuda',arenas:1,seed:requested.seed??73,
    round_seconds:positive(requested.round_seconds,'round_seconds'),
    locomotion_segment_ticks:positive(requested.locomotion_segment_ticks??1,'locomotion_segment_ticks'),
    move_duration_ticks:requested.move_duration_ticks??DEFAULT_DURATIONS,
    model_path:requested.model_path,assets_path:requested.assets_path,motion_features_path:requested.motion_features_path};
  if(!Number.isSafeInteger(base.seed)||base.seed<0)throw new Error('Invalid seed');
  if(!Array.isArray(base.move_duration_ticks)||base.move_duration_ticks.length!==17)
    throw new Error('Expected 17 move duration ticks');
  base.move_duration_ticks.forEach((v,i)=>positive(v,`move_duration_ticks[${i}]`));
  for(const key of ['model_path','assets_path','motion_features_path'])
    if(typeof base[key]!=='string'||!path.isAbsolute(base[key]))throw new Error(`Absolute ${key} required`);
  const fast={};
  for(const [key,[fallback,min,max]] of Object.entries(FAST_SETTINGS)){
    const value=requested.fast?.[key]??fallback;
    if(!Number.isFinite(value)||value<min||value>max)throw new Error(`Invalid fast.${key}`);
    fast[key]=value;
  }
  for(const key of Object.keys(requested.fast||{}))
    if(!Object.hasOwn(FAST_SETTINGS,key))throw new Error(`Unknown fast setting ${key}`);
  const build=path.dirname(executable);
  const identity={backend:base.backend,controlHz:50,observations:223,actions:33,encoding:'scaled_polar_xy',
    roundSeconds:base.round_seconds,locomotionSegmentTicks:base.locomotion_segment_ticks,
    moveDurationTicks:base.move_duration_ticks,fast,
    runtimeObject:hashFile(path.join(build,'fast_runtime.o')),
    assetsObject:hashFile(path.join(build,'fast_assets.o')),
    model:hashFile(base.model_path),
    assetManifest:hashFile(path.join(base.assets_path,'semantic_duel_assets_manifest.json')),
    featureManifest:hashFile(path.join(base.motion_features_path,'foot_features_manifest.json'))};
  // Runtime verifies manifest member hashes before loading; these hashes bind
  // the full asset content. Checkpoint files and controller networks stay out.
  const configHash=crypto.createHash('sha256').update(JSON.stringify(identity)).digest('hex');
  hashFile(executable); // Validate the executable exists before creating files.
  fs.mkdirSync(run,{mode:0o700});
  const leagueFile=path.join(run,'league.json');
  const league=new League({file:leagueFile});
  const backend={id:base.backend,label:'Reduced GPU candidate',available:true,configHash,executable,
    workerConfig:path.join(run,'semantic_cuda-worker.json'),logFile:path.join(run,'semantic_cuda-worker.stderr.log'),
    env:{REK_PHYSICS_BACKEND:'semantic_cuda',...Object.fromEntries(
      Object.entries(fast).map(([key,value])=>[`REK_FAST_${key.toUpperCase()}`,String(value)]))},
    runtimeNote:`Same 50 Hz CUDA environment for training and evaluation. Native CUDA policy inference. CPU kinematics renders pictures only. Experimental ${base.round_seconds}-second rounds.`,
    warning:'Reduced-model experiment: canned joint poses with approximate root movement, strike/target volumes and collision response. Full rigid-body dynamics and shipped balance policies are absent. Authentic REK motion and contact parity are unverified.'};
  fs.writeFileSync(backend.workerConfig,JSON.stringify(base,null,2)+'\n',{mode:0o600,flag:'wx'});
  league.registerPolicy({backend:backend.id,id:'scripted',label:'Scripted approach / 16 moves',
    configHash,kind:'scripted',scriptedVersion:`semantic-cuda-approach-facing-${identity.runtimeObject}`});
  const config={port:18769,leagueFile,backends:[backend],initial:{backend:backend.id,opponent:'scripted',humanSide:0}};
  const configPath=path.join(run,'server.json');
  fs.writeFileSync(configPath,JSON.stringify(config,null,2)+'\n',{mode:0o600,flag:'wx'});
  fs.writeFileSync(path.join(run,'runtime-identity.json'),JSON.stringify(identity,null,2)+'\n',{mode:0o600,flag:'wx'});
  return {configPath,configHash,port:config.port,backend:backend.id};
}
if(require.main===module){
  if(process.argv.length!==5)throw new Error('Usage: prepare_fast.cjs NEW_PRIVATE_RUN_DIRECTORY EVALUATOR_BINARY TRAIN_RUNTIME_CONFIG.json');
  process.stdout.write(JSON.stringify(prepare(...process.argv.slice(2)),null,2)+'\n');
}
module.exports={prepare};
