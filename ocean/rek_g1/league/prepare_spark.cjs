'use strict';
// Creates private runtime configuration only. Assets and checkpoints stay on Spark.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {League,hashFile}=require('./league.cjs');
const root='/home/spark-advantage';
const run=process.argv[2];
const executable=process.argv[3];
if(!run||!executable||!path.isAbsolute(run)||!path.isAbsolute(executable))throw new Error('Usage: prepare_spark.cjs PRIVATE_RUN_DIRECTORY EXECUTABLE');
fs.mkdirSync(run,{recursive:true,mode:0o700});
const assets=`${root}/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact`;
const models=`${root}/codexrook-runtime/generated/gear-sonic-g1batch8-mode0-20260909T0203Z`;
const base={arenas:4,seed:73,round_seconds:20,model_path:`${assets}/model.two_fighter_arena.xml`,
  physics_export_path:`${root}/rek-training/paired-physics-training-20260913T0423Z/model-export-capsule-diagnostic.json`,
  assets_path:assets,motion_features_path:`${root}/rek-training/gpu-runtime-20260910/motion-foot-features`,
  controller_encoder_path:`${models}/model_encoder.batch8.onnx`,controller_decoder_path:`${models}/model_decoder.batch8.onnx`};
const leagueFile=path.join(run,'league.json'),league=new League({file:leagueFile});
const backends=[
  {id:'mujoco',label:'MuJoCo · 20 s rounds',env:{REK_PHYSICS_BACKEND:'mujoco_cpu_eval',REK_ALLOW_CPU_EVALUATION:'1'}},
  {id:'puffysics',label:'Puffysics · cold joint cache · 20 s rounds',env:{REK_PHYSICS_BACKEND:'puffysics_cpu_eval',REK_ALLOW_CPU_EVALUATION:'1',REK_PUFFYSICS_STABILIZATION:'joint_cold_start'}},
];
for(const b of backends){
  const identity={physics:b.id,model:hashFile(base.model_path),export:hashFile(base.physics_export_path),
    encoder:hashFile(base.controller_encoder_path),decoder:hashFile(base.controller_decoder_path),
    roundSeconds:20,observationEncoding:'scaled_polar_xy',actions:33,
    stabilization:b.id==='puffysics'?'joint_cold_start':'none',scoring:'native_g1_candidate'};
  b.configHash=crypto.createHash('sha256').update(JSON.stringify(identity)).digest('hex');
  b.executable=executable;b.workerConfig=path.join(run,`${b.id}-worker.json`);
  b.logFile=path.join(run,`${b.id}-worker.stderr.log`);b.available=true;
  fs.writeFileSync(b.workerConfig,JSON.stringify(base,null,2)+'\n',{mode:0o600});
  league.registerPolicy({backend:b.id,id:'scripted',label:'Scripted approach / 16 moves',
    configHash:b.configHash,kind:'scripted',scriptedVersion:'native5-approach-facing-16moves-v1'});
}
const b=backends.find(x=>x.id==='puffysics');
for(const [id,label,steps,checkpoint] of [
  ['native-24k','Native PPO · 24.6k transitions',24576,`${root}/rek-training/native5-rek-20260913-v1/final-512/checkpoints/rek_native5/native5-probe/0000000000024576.bin`],
  ['native-196k','Native PPO · 196.6k transitions',196608,`${root}/rek-training/native5-rek-20260913-v1/final-4096/checkpoints/rek_native5/native5-probe/0000000000196608.bin`],
]){
  if(!fs.existsSync(checkpoint))continue;
  league.registerPolicy({backend:b.id,id,label,configHash:b.configHash,kind:'trained',checkpoint:{
    path:checkpoint,sha256:hashFile(checkpoint),format:'pufferlib-native-flat-fp32',trainingSteps:steps,
    model:{hiddenSize:256,layers:2,precision:'bf16',observationEncoding:'scaled_polar_xy',
      originalTrainingBackend:'puffysics_gpu',originalRoundSeconds:120}}});
}
const mj=backends.find(x=>x.id==='mujoco');
for(const [id,label,directory] of [
  ['combined-3m','MuJoCo PPO · 3.28M transitions','combined3276800-r1'],
  ['corrected-3m','MuJoCo PPO · continued +3.28M transitions','corrected3276800-r1'],
]){
  const checkpoint=`${root}/rek-training/training-opt-20260911/training/${directory}/0000000003276800.bin`;
  if(!fs.existsSync(checkpoint))continue;
  league.registerPolicy({backend:mj.id,id,label,configHash:mj.configHash,kind:'trained',checkpoint:{
    path:checkpoint,sha256:hashFile(checkpoint),format:'pufferlib-native-flat-fp32',trainingSteps:3276800,
    model:{hiddenSize:256,layers:2,precision:'fp32',observationEncoding:'raw223',
      recurrentResetTicks:64,legacyFastHidden:1,originalTrainingBackend:'mujoco_gpu',originalRoundSeconds:120}}});
}
const config={port:18768,leagueFile,backends,initial:{backend:'mujoco',opponent:'scripted'}};
const configPath=path.join(run,'server.json');
fs.writeFileSync(configPath,JSON.stringify(config,null,2)+'\n',{mode:0o600});
process.stdout.write(JSON.stringify({configPath,policies:backends.flatMap(b=>league.opponentOptions({backend:b.id,configHash:b.configHash}))},null,2)+'\n');
