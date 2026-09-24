'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),{spawnSync}=require('node:child_process');
const stage=__dirname,base='/home/spark-advantage/rek-training';
const binary=base+'/imitation-20260919-r1-native-bc/build-r5/bc-train';
const checkpoint=base+'/balance8-onpolicy-20260924-r2/train-score-delta/policy.bin';
const checkpointSha='0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12';
function hash(p){return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');}
function pin(p,h){if(hash(p)!==h)throw Error('hash mismatch: '+p);}
const mode=process.argv[2];if(process.argv.length!==3||!['--check','--run'].includes(mode))throw Error('usage: node run_bc.cjs --check|--run');
pin(binary,'febdf12d528f815ad737385208daed27d8676287b8335802da43a5442537d42d');pin(checkpoint,checkpointSha);
if(fs.statSync(checkpoint).size!==1836032)throw Error('checkpoint shape');
const dataset=path.join(stage,'dataset/human-conditional-attack.bin');pin(dataset,'b07f599e8f38ae0cfe59c3e7e0e10fa5aee00955f0483a9f763cfb662dbbfdef');
const manifest=JSON.parse(fs.readFileSync(path.join(stage,'dataset/manifest.json')));
if(manifest.observation_schema!=='rek.native5.scaled_polar_xy.balance8_v1'||manifest.initialization_sha256!==checkpointSha||manifest.on_policy!==false||manifest.input_equivalence_claim!==false)throw Error('dataset contract');
const out=path.join(stage,'train-five-epochs'),args=[dataset,checkpoint,checkpointSha,path.join(out,'policy.bin'),'5','.0001','128'];
const command={binary,args,observation_schema:manifest.observation_schema,checkpoint_sha256:checkpointSha,dataset_sha256:manifest.dataset_sha256,
  schema_enforcement:'external manifest; unchanged REKBC001 native reader checks223x33 shape, not schema string',
  training_input:'partial observation; missing176..183,202,204,205 explicitly zero',deployment_input:'existing allones full223 unchanged',conditional_support_is_training_only:true,
  epoch_selection:'fixed epoch5; intermediate epochs retained and reported, not selected from development CE'};
console.log(JSON.stringify(command));if(mode==='--check')process.exit(0);
if(fs.existsSync(out))throw Error('output exists');fs.mkdirSync(out);
fs.writeFileSync(path.join(out,'command.json'),JSON.stringify(command,null,2)+'\n',{flag:'wx'});
const stdout=fs.openSync(path.join(out,'stdout.jsonl'),'wx'),stderr=fs.openSync(path.join(out,'stderr.txt'),'wx');
const started=new Date().toISOString(),clock=process.hrtime.bigint();let r;
try{r=spawnSync(binary,args,{stdio:['ignore',stdout,stderr],timeout:120000});}finally{fs.closeSync(stdout);fs.closeSync(stderr);}
const receipt={started_utc:started,finished_utc:new Date().toISOString(),full_process_seconds:Number(process.hrtime.bigint()-clock)/1e9,exit_code:r.status,signal:r.signal,error:r.error?.message??null,epochs:5,expected_updates:155,checkpoint_sha256:fs.existsSync(args[3])?hash(args[3]):null,game_connection:false};
fs.writeFileSync(path.join(out,'execution.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(receipt));process.exitCode=r.status===0?0:2;
