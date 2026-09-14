'use strict';
const fs=require('node:fs');
const {League,hashFile}=require('./league.cjs');
const args={};
for(let i=2;i<process.argv.length;i+=2){
  if(!process.argv[i].startsWith('--')||process.argv[i+1]===undefined)throw new Error('Expected --key value');
  args[process.argv[i].slice(2)]=process.argv[i+1];
}
for(const key of ['config','backend','id','label','checkpoint','steps','encoding','precision'])
  if(!args[key])throw new Error(`Missing --${key}`);
if(!['raw223','scaled_polar_xy'].includes(args.encoding))throw new Error('Explicit valid observation encoding required');
if(!['bf16','fp32'].includes(args.precision))throw new Error('Explicit valid precision required');
if(args['action-selection']&&!['greedy','sampled'].includes(args['action-selection']))throw new Error('Invalid action selection');
const config=JSON.parse(fs.readFileSync(args.config,'utf8'));
const backend=config.backends.find(b=>b.id===args.backend);if(!backend)throw new Error('Unknown backend');
const league=new League({file:config.leagueFile});
const entry=league.registerPolicy({backend:backend.id,id:args.id,label:args.label,configHash:backend.configHash,
  kind:'trained',checkpoint:{path:args.checkpoint,sha256:hashFile(args.checkpoint),
    format:'pufferlib-native-flat-fp32',trainingSteps:Number(args.steps),model:{
      hiddenSize:Number(args.hidden||256),layers:Number(args.layers||2),precision:args.precision,
      observationEncoding:args.encoding,recurrentResetTicks:Number(args['reset-every']||0),
      legacyFastHidden:Number(args['legacy-fast-hidden']||0),
      actionSelection:args['action-selection']||'greedy',
      trainingBackend:args['training-backend']||backend.id}}});
console.log(JSON.stringify({backend:entry.backend,id:entry.id,sha256:entry.checkpoint.sha256,steps:entry.checkpoint.trainingSteps}));
