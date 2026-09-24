#!/usr/bin/env bash
# Derived frozen-evaluation dataset only; never modifies native training logs.
set -euo pipefail
umask 077
[[ $# == 0 ]] || { printf 'Usage: bash %s\n' "$0" >&2; exit 2; }
node <<'NODE'
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process'),assert=require('node:assert/strict');
const stage='/home/spark-advantage/rek-training/native-hparam-score-sweep-20260924-r1';
const native='/home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation';
const group='rek_frozen_confirmation',sourceHashes={};
const hash=data=>crypto.createHash('sha256').update(data).digest('hex');
const read=file=>{const data=fs.readFileSync(file);const digest=hash(data);if(sourceHashes[file])assert.equal(digest,sourceHashes[file],'Source changed');sourceHashes[file]=digest;return data;};
const json=file=>JSON.parse(read(file));
const write=(file,data)=>fs.writeFileSync(file,data,{flag:'wx',mode:0o600});
const put=(file,data)=>write(file,JSON.stringify(data,null,2)+'\n');
function ini(text){const out={};let section;for(const line of text.split(/\r?\n/)){const h=line.match(/^\[([^\]]+)\]$/);if(h){section=h[1];out[section]={};continue;}const m=line.match(/^([^#;=]+?)\s*=\s*(.*?)\s*$/);if(section&&m)out[section][m[1].trim()]=m[2];}return out;}
assert.equal(require('node:os').hostname(),'spark-4ae3');
assert.equal(fs.realpathSync(stage),stage);
const pins={'cache_data':'53d78b920f98f5e6a6fa009e48712b172378eb702882e9a6cd3bb19b2c0eba4d','seethestars':'13ed2ffef2e52e2ff43a6a058ea33a51cbf3d0bc096ffad7e05094dc44b024c2'};
for(const [name,digest] of Object.entries(pins))assert.equal(hash(read(native+'/'+name)),digest,'Native binary changed');
const plan=json(stage+'/plan.json'),summary=json(stage+'/confirmation-results.json'),screen=json(stage+'/screen-results.json');
const expectedIds=['control',...screen.selected.filter(id=>id!=='control')];
const expected=new Set(expectedIds.flatMap(id=>[73,947].map(seed=>id+'-s'+seed)));
assert.equal(summary.results.length,expected.size,'Incomplete confirmation summary');
const seen=new Set();
for(const row of summary.results){const key=row.arm.id+'-s'+row.seed;assert(expected.has(key)&&!seen.has(key),'Unexpected or duplicate confirmation');seen.add(key);}
const excluded=summary.results.filter(row=>row.error!==null||row.receipt.exitCode!==0).map(row=>({id:row.arm.id,seed:row.seed,error:row.error,exitCode:row.receipt.exitCode}));
const baselineRoot=path.dirname(path.dirname(path.dirname(path.dirname(plan.warm))));
const candidates=[{id:'unchanged-baseline',baseline:true,budget:0,model:plan.warm,modelSha:plan.warmSha,arm:plan.control,run:baselineRoot,label:'unchanged-baseline-confirm'},
  ...summary.results.filter(row=>row.error===null&&row.receipt.exitCode===0).map(row=>({...row,id:row.arm.id+'-s'+row.seed,baseline:false,run:stage+'/confirm-s'+row.seed+'/'+row.arm.id,label:'confirm-'+row.arm.id+'-s'+row.seed}))];
const records=[];
for(let index=0;index<candidates.length;index++){
  const row=candidates[index],evaluationDir=stage+'/evaluations/'+row.label;
  const result=json(evaluationDir+'/result.json'),metrics=result.metrics;
  assert.equal(result.modelSha,row.modelSha);assert.equal(metrics.checkpointSha256,row.modelSha);
  assert.equal(hash(read(row.model)),row.modelSha,'Checkpoint hash mismatch');
  assert.deepEqual(result.seeds,plan.confirmationSeeds);assert.equal(metrics.policySide,0);
  assert.equal(metrics.n,result.arenas*result.seeds.length);assert.equal(metrics.n,384);
  assert.deepEqual(metrics,row.baseline?summary.baseline:row.evaluation,'Summary/evaluation mismatch');
  read(evaluationDir+'/rounds.private.jsonl');
  let wall=0;for(const seed of result.seeds){const receipt=json(evaluationDir+'/seed-'+seed+'/exit.json');assert.equal(receipt.exitCode,0);assert(Number.isFinite(receipt.wallSeconds)&&receipt.wallSeconds>=0);wall+=receipt.wallSeconds;}
  const logDir=row.run+'/logs/rek_native5',names=fs.readdirSync(logDir).filter(name=>name.endsWith('.ini'));
  assert.equal(names.length,1,'Expected one original training INI');
  const sourceIni=path.join(logDir,names[0]),config=ini(read(sourceIni).toString('utf8'));
  for(const [key,field] of Object.entries({learning_rate:'lr',ent_coef:'entropy',clip_coef:'clip',vf_coef:'vf',horizon:'horizon',gamma:'gamma',gae_lambda:'lambda'}))assert.equal(Number(config.train[key]),row.arm[field],'Plan mismatch: '+key);
  assert.equal(Number(config.train.total_timesteps),row.baseline?Number(path.basename(plan.warm,'.bin')):row.budget);
  const seed=Number(config.base.seed);assert.equal(seed,row.baseline?73:row.seed);
  assert.equal(Number(config.vec.total_agents),plan.assumptions.envs);assert.equal(Number(config.train.minibatch_size),plan.assumptions.minibatch);
  assert.equal(Number(config.train.replay_ratio),plan.assumptions.replay);assert.equal(Number(config.policy.hidden_size),plan.assumptions.hidden);assert.equal(Number(config.policy.num_layers),plan.assumptions.layers);
  read(row.run+(row.baseline?'/command.txt':'/command.json'));
  const values={agent_steps:row.budget,uptime:wall,'env/score':metrics.meanOwnPoints,'env/conceded':metrics.meanConcededPoints,'env/margin':metrics.meanMargin,'env/wins':metrics.winRate,'eval/rounds':metrics.n,'eval/is_baseline':Number(row.baseline),'eval/config_index':index,'eval/train_seed':seed,'eval/frozen':1};
  assert(Object.values(values).every(Number.isFinite));assert(metrics.winRate>=0&&metrics.winRate<=1);
  const sections=['# DERIVED FROZEN POLICY EVALUATION; this is not a native training curve.','# Source: '+evaluationDir+'/result.json','# Original training INI is unchanged: '+sourceIni,'','[base]','env_name = '+group,'seed = '+seed];
  for(const section of ['train','vec','policy']){sections.push('','['+section+']');for(const [key,value] of Object.entries(config[section]))sections.push(key+' = '+value);}
  sections.push('','[metrics]',...Object.entries(values).map(([key,value])=>key+' = '+value),'');
  records.push({id:row.id,baseline:row.baseline,model:row.model,checkpointSha256:row.modelSha,sourceResult:evaluationDir+'/result.json',sourceIni,seed,values,text:sections.join('\n'),filename:String(index).padStart(3,'0')+'-'+row.id+'.ini'});
}
const root=stage+'/constellation/frozen-datasets';fs.mkdirSync(root,{recursive:true,mode:0o700});
const dataset=fs.mkdtempSync(root+'/import-'+new Date().toISOString().replace(/[-:.]/g,'')+'-');
fs.mkdirSync(dataset+'/logs/'+group,{recursive:true,mode:0o700});fs.mkdirSync(dataset+'/resources/constellation',{recursive:true,mode:0o700});
fs.cpSync(native+'/resources/shared',dataset+'/resources/shared',{recursive:true,errorOnExist:true,force:false});
for(const name of fs.readdirSync(native+'/resources/constellation'))if(/\.(?:fs|vs|rgs)$/.test(name))fs.copyFileSync(native+'/resources/constellation/'+name,dataset+'/resources/constellation/'+name,fs.constants.COPYFILE_EXCL);
for(const record of records)write(dataset+'/logs/'+group+'/'+record.filename,record.text);
write(dataset+'/plan.json',read(stage+'/plan.json'));write(dataset+'/confirmation-results.json',read(stage+'/confirmation-results.json'));
const stdout=fs.openSync(dataset+'/cache.stdout.txt','wx',0o600),stderr=fs.openSync(dataset+'/cache.stderr.txt','wx',0o600);let converted;
try{converted=cp.spawnSync(native+'/cache_data',['--full'],{cwd:dataset,stdio:['ignore',stdout,stderr],timeout:60000});}finally{fs.closeSync(stdout);fs.closeSync(stderr);}
put(dataset+'/cache-exit.json',{exitCode:converted.status,signal:converted.signal,error:converted.error?.message??null});assert.equal(converted.status,0);
const cacheFile=dataset+'/resources/constellation/experiments.ini',cache=ini(fs.readFileSync(cacheFile,'utf8'));assert.deepEqual(Object.keys(cache),[group]);
let checked=0;
for(const key of Object.keys(records[0].values)){const values=cache[group][key].split(',').map(Number);assert.equal(values.length,records.length);for(let i=0;i<records.length;i++){const original=records[i].values[key],wanted=key==='agent_steps'?Math.fround(Math.fround(original)/1e6):Math.fround(original);assert(Math.abs(values[i]-wanted)<=.000005*Math.max(1,Math.abs(wanted)),'Cache changed '+key);checked++;}}
for(const [file,digest] of Object.entries(sourceHashes))assert.equal(hash(fs.readFileSync(file)),digest,'Source changed: '+file);
put(dataset+'/provenance.json',{createdUtc:new Date().toISOString(),kind:'derived_frozen_confirmation',group,dataset,binaryPins:pins,converterArguments:['--full'],excluded,sourceHashes,cacheSha256:hash(fs.readFileSync(cacheFile)),records:records.map(({text,...record})=>({...record,derivedIniSha256:hash(text)}))});
put(dataset+'/verification.json',{derivedRows:records.length,metricValuesCompared:checked,sourceFilesUnchanged:true,policySide:0,unit:'completed simulator round',nativeCachePrecision:'float and %.6g',viewerLaunched:false,pythonRuntime:false});
write(dataset+'/viewer-command.txt','cd '+dataset+'\nDISPLAY=:98 '+native+'/seethestars\n');
write(dataset+'/README.md','# DERIVED FROZEN confirmation evaluation\n\n'+
  'This separate '+group+' group contains one aggregate row per successful configuration/training-seed pair plus the unchanged baseline. It is not a native training curve. All rows use the same confirmation seeds, 384 completed simulator rounds, and policy side 0.\n\n'+
  'env/score = mean own awarded points; env/conceded = mean opponent points; env/margin = mean point margin; env/wins = round-win fraction. No win rate is mapped into score. These are simulator rounds, not authentic REK matches.\n\n'+
  'Each train/vec/policy setting is copied from its original training INI and checked against the plan. The baseline retains its actual prior 16M-run settings and seed. agent_steps counts additional training in this sweep (baseline 0), shown in millions by the native cache. train/total_timesteps describes the recorded source training invocation. uptime is the summed measured native frozen-evaluation process time across all seeds, including both executed sides; it is not training time.\n\n'+
  'eval/config_index identifies rows; eval/train_seed and eval/is_baseline distinguish seeds and the unchanged model. Row identities, checkpoint hashes, source hashes, and lineage are in provenance.json. Original native logs remain unchanged. The existing native cache_data --full generated the cache; seethestars was not launched.\n\n'+
  'Use env/score on Y and train/learning_rate or eval/config_index on X. Select env/wins separately for round wins. The upstream env/perf default is absent. viewer-command.txt records the isolated Spark display command for later review.\n');
console.log(JSON.stringify({event:'frozen_constellation_import_verified',dataset,group,rows:records.length,excluded,viewerLaunched:false},null,2));
NODE
