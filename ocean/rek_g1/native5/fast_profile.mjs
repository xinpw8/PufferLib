import {spawnSync} from 'node:child_process';
import {writeFileSync} from 'node:fs';
import {analyze} from './mujoco_gpu/profile_training.mjs';

const [db,out]=process.argv.slice(2);
if(!db||!out)throw new Error('Usage: node fast_profile.mjs TRACE.sqlite NEW_SUMMARY.json');
const report=await analyze({db,catalog:{modules:[]},agents:512,horizon:16,requestedSteps:1048576});
const query=`SELECT COUNT(*) AS calls,SUM(k.end-k.start)/1e6 AS milliseconds,
 MAX(k.registersPerThread) AS maxRegistersPerThread,MAX(k.localMemoryPerThread) AS maxLocalMemoryPerThread
 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
 WHERE k.start>=${report.roi.startNs} AND k.end<=${report.roi.endNs} AND s.value LIKE '%fast_step(%'`;
const result=spawnSync('sqlite3',['-readonly','-json',db,query],{encoding:'utf8'});
if(result.status!==0)throw new Error(result.stderr);
const fast=JSON.parse(result.stdout)[0];
if(fast.calls!==report.roi.learnerSampleKernelCalls)throw new Error(`Fused step count ${fast.calls} differs from learner samples ${report.roi.learnerSampleKernelCalls}`);
function sql(query){const r=spawnSync('sqlite3',['-readonly','-json',db,query],{encoding:'utf8'});if(r.status!==0)throw new Error(r.stderr);return r.stdout.trim()?JSON.parse(r.stdout):[];}
const where=`k.start>=${report.roi.startNs} AND k.end<=${report.roi.endNs}`;
const streamFields='k.deviceId,k.contextId,k.streamId';
const anchors=sql(`SELECT ${streamFields},MAX(s.value LIKE '%fast_step(%') AS env,
 MAX(s.value LIKE '%sample_logits(%') AS rollout,
 MAX(s.value LIKE '%ppo_loss_compute(%' OR s.value LIKE '%mingru_scan_backward(%') AS training
 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
 WHERE ${where} GROUP BY ${streamFields}`);
if(anchors.filter(row=>row.env).length!==1)throw new Error('Need one verified environment/rollout stream');
for(const row of anchors)if((row.env||row.rollout)&&row.training)throw new Error('Training/rollout streams overlap; cannot apply dedicated-stream attribution');
const categorized=[];
for(const anchor of anchors){
  const streamWhere=`${where} AND k.deviceId=${anchor.deviceId} AND k.contextId=${anchor.contextId} AND k.streamId=${anchor.streamId}`;
  const total=sql(`SELECT COUNT(*) AS calls,SUM(k.end-k.start)/1e6 AS milliseconds,
    MAX(k.registersPerThread) AS maxRegistersPerThread,MAX(k.localMemoryPerThread) AS maxLocalMemoryPerThread
    FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE ${streamWhere}`)[0];
  let key='unclassified';
  if(anchor.training)key='ppo_training';
  else if(anchor.env&&anchor.rollout){key='learner_inference_and_rollout_io';total.calls-=fast.calls;total.milliseconds-=fast.milliseconds;}
  categorized.push({key,...total});
}
categorized.push({key:'fused_environment_step',...fast});
const byCategory=new Map();
for(const row of categorized){
  const previous=byCategory.get(row.key);
  if(previous){previous.calls+=row.calls;previous.milliseconds+=row.milliseconds;previous.maxRegistersPerThread=Math.max(previous.maxRegistersPerThread,row.maxRegistersPerThread);previous.maxLocalMemoryPerThread=Math.max(previous.maxLocalMemoryPerThread,row.maxLocalMemoryPerThread);}
  else byCategory.set(row.key,{...row});
}
report.gpu.categories=[...byCategory.values()].map(row=>({...row,
  averageMicrosecondsPerCall:row.milliseconds/row.calls*1000,
  percentOfSummedKernelTime:row.milliseconds/report.gpu.summedKernelMilliseconds*100})).sort((a,b)=>b.milliseconds-a.milliseconds);
report.gpu.unclassifiedPercentOfKernelTime=report.gpu.categories.filter(row=>row.key==='unclassified').reduce((sum,row)=>sum+row.percentOfSummedKernelTime,0);
report.gpu.streamAttribution={anchors,method:'Disjoint GPU streams identified by fast_step+sample_logits versus PPO-loss/backward anchors; exact fast_step duration removed from rollout stream.'};
report.backend='semantic_cuda_v1';report.physicsParityClaim=false;
report.roi.fusedEnvironmentKernelCalls=fast.calls;report.roi.environmentKernelsPerActionBatch=fast.calls/report.roi.learnerSampleKernelCalls;
report.gpu.attributionEvidence=[{key:'exact_fast_step_function',...fast},...anchors.map(row=>({key:'verified_disjoint_stream',...row}))];
report.interpretation.push('Fast environment attribution is the exact fast_step function. It includes both fighters, input scheduling, slider integration, contacts, points, resets and observation/pose export.');
// Older name-based attribution misses nvjet GEMMs and auxiliary backward
// kernels. Publish the verified stream decomposition rather than mixing it
// with that classifier's incompatible subcategories.
delete report.gpu.subcategories;
writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output:out,roi:report.roi,categories:report.gpu.categories,kernelBusy:report.gpu.kernelBusy,warnings:report.warnings},null,2));
