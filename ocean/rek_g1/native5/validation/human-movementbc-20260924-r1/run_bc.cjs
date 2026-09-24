'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),{spawnSync}=require('node:child_process');
const stage=__dirname,base='/home/spark-advantage/rek-training';
const binary=base+'/imitation-20260919-r1-native-bc/build-r5/bc-train';
const checkpoint=base+'/scorecredit-human-attackbc-20260924-r1/train-five-epochs/policy.bin';
const checkpointSha='5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4';
const datasetSha='b4f8261dc5e56df35520dc379c0b01f36ff14e0ccb7b7c2840e8ce4c74589403';
function hash(p){return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');}
function pin(p,h){if(hash(p)!==h)throw Error('hash mismatch: '+p);}
const mode=process.argv[2];if(process.argv.length!==3||!['--check','--run'].includes(mode))throw Error('usage: node run_bc.cjs --check|--run');
pin(binary,'febdf12d528f815ad737385208daed27d8676287b8335802da43a5442537d42d');pin(checkpoint,checkpointSha);
if(fs.statSync(checkpoint).size!==1836032)throw Error('checkpoint shape');
const dataset=path.join(stage,'dataset/human-conditional-movement.bin');pin(dataset,datasetSha);
const manifest=JSON.parse(fs.readFileSync(path.join(stage,'dataset/manifest.json')));
if(manifest.observation_schema!=='rek.native5.scaled_polar_xy.balance8_v1'||manifest.initialization_sha256!==checkpointSha||manifest.on_policy!==false||manifest.input_equivalence_claim!==false||manifest.conditional_support.join(',')!=='2,15'||manifest.training_labeled_chunks_per_epoch!==33)throw Error('dataset contract');
const out=path.join(stage,'train-five-epochs'),args=[dataset,checkpoint,checkpointSha,path.join(out,'policy.bin'),'5','.0001','128'];
const command={binary,args,observation_schema:manifest.observation_schema,checkpoint_sha256:checkpointSha,dataset_sha256:datasetSha,
 schema_enforcement:'external manifest; unchanged REKBC001 native reader checks223x33 shape, not schema string',
 training_input:'same partial balance8 view as attack BC; unknown176..183,202,204,205 zero',deployment_input:'no deployment change; existing allones full223 remains separate',
 conditional_support:[2,15],row_weights:'natural temporal occupancy of exact original movement outputs; not independent commands',
 epoch_selection:'fixed epoch5; all metrics/checkpoints retained; no development-selected epoch'};
console.log(JSON.stringify(command));if(mode==='--check')process.exit(0);
if(fs.existsSync(out))throw Error('output exists');fs.mkdirSync(out);fs.writeFileSync(path.join(out,'command.json'),JSON.stringify(command,null,2)+'\n',{flag:'wx'});
const stdout=fs.openSync(path.join(out,'stdout.jsonl'),'wx'),stderr=fs.openSync(path.join(out,'stderr.txt'),'wx');const started=new Date().toISOString(),clock=process.hrtime.bigint();let r;
try{r=spawnSync(binary,args,{stdio:['ignore',stdout,stderr],timeout:120000});}finally{fs.closeSync(stdout);fs.closeSync(stderr);}
const receipt={started_utc:started,finished_utc:new Date().toISOString(),full_process_seconds:Number(process.hrtime.bigint()-clock)/1e9,exit_code:r.status,signal:r.signal,error:r.error?.message??null,epochs:5,expected_updates:165,checkpoint_sha256:fs.existsSync(args[3])?hash(args[3]):null,game_connection:false};
fs.writeFileSync(path.join(out,'execution.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(receipt));process.exitCode=r.status===0?0:2;
