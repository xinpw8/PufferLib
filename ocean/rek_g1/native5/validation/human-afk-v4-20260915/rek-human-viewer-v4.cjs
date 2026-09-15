'use strict';
// Configure a new isolated viewer; preserve all prior services and runs.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const [leagueSource,sourceConfig,executable,runtimeConfig,trainingFile,checkpoint,destination,portText]=process.argv.slice(2);
for(const value of [leagueSource,sourceConfig,executable,runtimeConfig,trainingFile,checkpoint,destination])assert(path.isAbsolute(value));
const port=Number(portText);assert([18769,18770].includes(port));
const {prepare}=require(path.join(leagueSource,'prepare_fast.cjs'));
const {League,hashFile}=require(path.join(leagueSource,'league.cjs'));
const training=JSON.parse(fs.readFileSync(trainingFile,'utf8'));
assert.equal(training.status,'completed');assert.equal(training.checkpointSha256,hashFile(checkpoint));
assert(training.frozenEvaluationConditions.length>=12);
const previousConfig=JSON.parse(fs.readFileSync(sourceConfig,'utf8'));
const previousLeague=JSON.parse(fs.readFileSync(previousConfig.leagueFile,'utf8'));
const previous=previousLeague.policies['semantic_cuda/compact-v3-33m-bf16-sampled'];assert(previous);
const prepared=prepare(destination,executable,runtimeConfig);
const config=JSON.parse(fs.readFileSync(prepared.configPath,'utf8'));
const league=new League({file:config.leagueFile});
const candidate={backend:'semantic_cuda',id:'compact-v4-curriculum-bf16-sampled',label:'Compact V4 curriculum | BF16 sampled',
  configHash:prepared.configHash,kind:'trained',checkpoint:{path:checkpoint,sha256:training.checkpointSha256,
    format:'pufferlib-native-flat-fp32',trainingSteps:training.steps,model:{hiddenSize:256,layers:2,precision:'bf16',
      observationEncoding:'scaled_polar_xy',recurrentResetTicks:0,legacyFastHidden:0,actionSelection:'sampled',trainingBackend:'semantic_cuda'}}};
league.registerPolicy(candidate);
league.registerPolicy({...previous,configHash:prepared.configHash,label:'Compact V3 historical | BF16 sampled'});
config.port=port;config.initial={backend:'semantic_cuda',opponent:candidate.id,humanSide:1,roundSeconds:300};
for(const backend of config.backends){
  backend.trainingFile=trainingFile;
  backend.runtimeNote='50 Hz CUDA simulation and native CUDA policy inference. CPU kinematics renders pictures only. The selected curriculum used 20-second rounds; frozen tests cover 20-second and 300-second rounds. Human rounds default to 300 seconds. The regressed longer-round fine-tune was rejected.';
  backend.env={...backend.env,REK_FAST_SHAPING_WEIGHT:'0',REK_FAST_RANDOM_RESETS:'0',REK_FAST_OPPONENT_MODE:'scripted'};
}
fs.writeFileSync(prepared.configPath,JSON.stringify(config,null,2)+'\n',{mode:0o600});
console.log(JSON.stringify({config:prepared.configPath,configHash:prepared.configHash,initial:config.initial,port,
  checkpointSha256:training.checkpointSha256,workerSha256:hashFile(executable)}));
