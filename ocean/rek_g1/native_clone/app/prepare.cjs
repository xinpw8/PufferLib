'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {League}=require('./league/league.cjs');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const BATCH2_ENCODER_SHA='0e1cf37a7c1bafe870741b8a3de2560ae7a2b0cfe14c5ac8cef4d8a34d78207f';
const BATCH2_DECODER_SHA='20b49c9df1a54dc3a211d0d86c2ebe3ccc1de67883a984a7b227af76af7aacb3';
function prepare({workerTemplate,binary,runDirectory,port=18771,environmentFile}){
  if(!Number.isInteger(port)||port<1024||port>65535)throw Error('Invalid loopback port');
  const run=path.resolve(runDirectory);if(fs.existsSync(run))throw Error('Fresh run directory required');
  const worker=JSON.parse(fs.readFileSync(workerTemplate,'utf8'));
  if(worker.backend!=='mujoco'||![1,4].includes(worker.arenas))throw Error('Full MuJoCo motor worker with 1 or 4 arenas required');
  const env=JSON.parse(fs.readFileSync(environmentFile,'utf8'));
  if(env.REK_PHYSICS_BACKEND!=='mujoco_cuda'||env.REK_ALLOW_CPU_EVALUATION!=='0'||env.REK_PHYSICAL_OPPONENT!=='recovered_bot1_g1_v1')throw Error('Full physical Bot1 environment required');
  if(env.REK_MUJOCO_DETERMINISTIC_SCALAR!=='0')throw Error('Manual app expects original parallel physics path');
  worker.render_model_path=path.join(__dirname,'presentation/assets/presentation.playable.xml');
  // The presentation model is never substituted for the collision model.
  if(path.resolve(worker.model_path)===path.resolve(worker.render_model_path))throw Error('Separate collision model required');
  const filePins={};
  for(const p of [binary,workerTemplate,environmentFile,worker.model_path,worker.physics_export_path,worker.render_model_path,worker.controller_encoder_path,worker.controller_decoder_path]){
    if(!p||!fs.statSync(p).isFile())throw Error('Missing bound runtime file: '+p);filePins[path.resolve(p)]=sha(p);
  }
  if(worker.arenas===1&&(filePins[path.resolve(worker.controller_encoder_path)]!==BATCH2_ENCODER_SHA||
      filePins[path.resolve(worker.controller_decoder_path)]!==BATCH2_DECODER_SHA))
    throw Error('One arena requires the verified batch-2 encoder and decoder pair');
  const execution={arenas:worker.arenas,controllerBatch:worker.arenas*2,displayedArena:0,
    controlStepSeconds:.02,throughputUnit:'control ticks per wall second',
    aggregateArenaStepsMultiplier:worker.arenas,
    controllerPairContract:worker.arenas===1?'rek.verified_batch2_pair.v1':'existing_four_arena_path'};
  const identity={schema:'rek.native_clone.app_run.v1',filePins,worker,env,port,execution,controlsSha256:sha(path.join(__dirname,'saved-g1-bindings.json')),fullParityProven:false};
  const configHash=crypto.createHash('sha256').update(JSON.stringify(identity)).digest('hex');
  fs.mkdirSync(run,{recursive:true,mode:0o700});
  const leagueFile=path.join(run,'league.json'),league=new League({file:leagueFile});
  league.registerPolicy({backend:'mujoco',id:'bot1',label:'Bot1',kind:'scripted',configHash,scriptedVersion:'recovered_bot1_g1_v1'});
  const workerPath=path.join(run,'worker.json');
  const server={port,leagueFile,backends:[{id:'mujoco',label:'REK Native',available:true,executable:path.resolve(binary),workerConfig:workerPath,env,
    logFile:path.join(run,'worker.stderr.log'),configHash}],initial:{backend:'mujoco',opponent:'bot1',humanSide:0,roundSeconds:worker.round_seconds||120}};
  for(const [n,data] of [['worker.json',worker],['server.json',server],['identity.json',identity]])fs.writeFileSync(path.join(run,n),JSON.stringify(data,null,2)+'\n',{flag:'wx',mode:0o600});
  return {run,configHash,server:path.join(run,'server.json'),launch:['node',path.join(__dirname,'launch_logged.cjs'),run]};
}
if(require.main===module){
  const [workerTemplate,binary,runDirectory,environmentFile,port='18771']=process.argv.slice(2);
  if(!environmentFile)throw Error('Usage: node prepare.cjs worker-template.json binary fresh-run-directory environment.json [port]');
  console.log(JSON.stringify(prepare({workerTemplate,binary,runDirectory,environmentFile,port:Number(port)})));
}
module.exports={prepare};
