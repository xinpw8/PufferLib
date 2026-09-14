import {spawn, spawnSync} from 'node:child_process';
import {readFileSync, writeFileSync} from 'node:fs';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';

function rows(db, sql) {
  const result=spawnSync('sqlite3',['-readonly','-json',db,sql],{encoding:'utf8',maxBuffer:32*1024*1024});
  if(result.status!==0)throw new Error(`SQLite read failed: ${result.stderr}`);
  return result.stdout.trim()?JSON.parse(result.stdout):[];
}
function streamKey(row){return `${row.deviceId}/${row.contextId}/${row.streamId}`;}
function isStamp(name){return /(?:^|::)puf_stamp\(/.test(name);}
function isControllerStart(name){return /(?:^|::)pack_encoder\(/.test(name);}
function isControllerEnd(name){return /(?:^|::)apply_actions\(RobotState/.test(name);}
function isTrainingSpecific(name){return /(?:ppo_loss_(?:compute|reduce)|puff_advantage|muon_|mingru_scan_(?:forward|backward)|sum_rows_to_precision_kernel|assemble_decoder_grad|cache_imp_and_v)/.test(name);}
function isLearnerSample(name){return /(?:^|::)sample_logits\(/.test(name);}
function isGemm(name){return /(?:gemm|cutlass|cublas|wmma|gemv)/i.test(name);}

export function buildWindows(anchors){
  const training=new Map(),controller=new Map(),trainingStreams=new Set(),warnings=[];
  const pendingTrain=new Map(),pendingController=new Map();
  for(const row of anchors){
    const key=streamKey(row),name=row.name;
    if(isTrainingSpecific(name))trainingStreams.add(key);
    if(isStamp(name)){
      trainingStreams.add(key);
      const group=pendingTrain.get(key)||[];group.push(row);pendingTrain.set(key,group);
      if(group.length===3){
        const list=training.get(key)||[];
        list.push({start:group[0].start,mid:group[1].start,end:group[2].end});training.set(key,list);pendingTrain.set(key,[]);
      }
    }
    if(isControllerStart(name)){
      if(pendingController.has(key))warnings.push(`Unclosed SONIC window on stream ${key}`);
      pendingController.set(key,row);
    }
    if(isControllerEnd(name)){
      const begin=pendingController.get(key);
      if(begin){const list=controller.get(key)||[];list.push({start:begin.start,end:row.end});controller.set(key,list);pendingController.delete(key);}
      else warnings.push(`SONIC end without pack_encoder on stream ${key}`);
    }
  }
  for(const [key,group] of pendingTrain)if(group.length)warnings.push(`Incomplete PPO timestamp triplet on stream ${key}`);
  for(const key of pendingController.keys())warnings.push(`Trailing incomplete SONIC window on stream ${key}`);
  return {training,controller,trainingStreams,warnings};
}
function enclosing(windows,start,end){
  if(!windows?.length)return null;
  let low=0,high=windows.length;
  while(low<high){const mid=(low+high)>>1;if(windows[mid].start<=start)low=mid+1;else high=mid;}
  const candidate=windows[low-1];return candidate&&end<=candidate.end?candidate:null;
}
export function catalogLookup(catalog){
  const result=new Map();
  for(const module of catalog.modules||[])for(const kernel of module.kernels||[])result.set(kernel.symbol,module.name);
  return result;
}
export function classifyKernel(row,context){
  const name=row.name,key=streamKey(row);
  const symbol=name.match(/[A-Za-z_]\w*_cuda_kernel_forward/)?.[0];
  if(symbol&&context.catalog.has(symbol)){
    const module=context.catalog.get(symbol);
    const detail=/solver|update_gradient|update_constraint|linesearch|mul_m_sparse|solve_init|solve_search/.test(module)?'constraint_solver':
      /collision|broadphase|ccd_kernel/.test(module)?'collision':
      /constraint/.test(module)?'constraint_assembly':
      /derivative/.test(module)||/_next_(?:position|velocity|time|activation)_/.test(symbol)?'integration':'dynamics_and_spatial_refresh';
    return {category:'physics',detail,evidence:'exact_cached_kernel_catalog'};
  }
  if(/rek5_native::(?:rp_|rps_)/.test(name))return {category:'physics',detail:'puffysics',evidence:'native_physics_namespace'};
  if(/rek_mjgpu_.*condition|mujoco_gpu_(?:contact_counts|export_stats|clear_selected)/.test(name))
    return {category:'physics',detail:'graph_control_and_export',evidence:'native_mujoco_wrapper'};
  if(enclosing(context.controller.get(key),row.start,row.end))
    return {category:'robot_controller',detail:'sonic_network_and_io',evidence:'pack_encoder_to_robot_apply_actions_same_stream'};
  const training=enclosing(context.training.get(key),row.start,row.end);
  if(training)return {category:'ppo_training',detail:row.end<=training.mid?'rollout_relayout':'network_loss_optimizer',evidence:'three_puf_stamp_markers_same_stream'};
  if(isTrainingSpecific(name))return {category:'ppo_training',detail:'training_auxiliary',evidence:'training_only_kernel'};
  if(context.trainingStreams.has(key)&&isGemm(name))
    return {category:'ppo_training',detail:'training_stream_gemm',evidence:'dedicated_stream_has_training_only_kernel'};
  if(/^memset(?:8|16|32|64)?$/.test(name))
    return {category:'gpu_memory_maintenance',detail:'generic_memset',evidence:'driver_memset_kernel_owner_unspecified'};
  if(/(?:pack_encoder|bias_activation|fill_bias|quantize)\(/.test(name)||/\(RobotState[,)]/.test(name))
    return {category:'robot_controller',detail:'controller_or_servo',evidence:'native_controller_function_signature'};
  if(/(?:sample_logits|mingru_gate|snapshot_state|zero_term_state)\(/.test(name))
    return {category:'learner_inference',detail:'categorical_policy',evidence:'native_rollout_function'};
  if(/Rek5NativePolicy|curandState.*masks|native_policy/.test(name))
    return {category:'opponent_inference',detail:'frozen_policy',evidence:'native_policy_signature'};
  if(isGemm(name))return {category:'unclassified_gemm',detail:'shared_library_kernel',evidence:'no_verified_phase_boundary'};
  if(/Measurement|RuntimeView|RekG1|RekNative5Motion|FallView|rek5::.*(?:speeds|body_velocity|clear_pairs)/.test(name)||
      /(?:contact_facts|clear_facts|commit_hits|row_sort|reduce_hits)\(/.test(name))
    return {category:'environment_semantics',detail:'combat_motion_observation',evidence:'native_environment_signature'};
  if(/cub::.*(?:RadixSort|Scan|Reduce)/.test(name))return {category:'environment_semantics',detail:'contact_sort_reduce',evidence:'current_runtime_uses_CUB_for_contact_measurement'};
  return {category:'unclassified',detail:'needs_source_attribution',evidence:'unknown'};
}
async function streamRows(db,sql,consume){
  const process=spawn('sqlite3',['-readonly',db,sql],{stdio:['ignore','pipe','pipe']});
  let stderr='';process.stderr.setEncoding('utf8');process.stderr.on('data',chunk=>stderr+=chunk);
  const completion=new Promise((resolve,reject)=>{process.on('error',reject);process.on('close',code=>code===0?resolve():reject(new Error(`SQLite stream failed: ${stderr}`)));});
  for await(const line of createInterface({input:process.stdout,crlfDelay:Infinity}))if(line)consume(JSON.parse(line));
  await completion;
}
function aggregate(map,key,duration,row){
  const value=map.get(key)||{key,calls:0,nanoseconds:0,maxRegistersPerThread:0,maxLocalMemoryPerThread:0};
  value.calls++;value.nanoseconds+=duration;
  value.maxRegistersPerThread=Math.max(value.maxRegistersPerThread,row.registersPerThread||0);
  value.maxLocalMemoryPerThread=Math.max(value.maxLocalMemoryPerThread,row.localMemoryPerThread||0);
  map.set(key,value);
}
export async function analyze({db,catalog,agents,horizon,requestedSteps}){
  if(!Number.isInteger(agents)||agents<1||!Number.isInteger(horizon)||horizon<1)throw new Error('Positive agents and horizon required');
  const tables=new Set(rows(db,"SELECT name FROM sqlite_master WHERE type='table'").map(row=>row.name));
  for(const name of ['CUPTI_ACTIVITY_KIND_KERNEL','StringIds'])if(!tables.has(name))throw new Error(`Missing Nsight table: ${name}; require --cuda-graph-trace=node`);
  const prefix='SELECT k.start,k.end,k.deviceId,k.contextId,k.streamId,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName';
  const anchors=rows(db,`${prefix} WHERE s.value LIKE '%puf_stamp(%' OR s.value LIKE '%pack_encoder(%' OR s.value LIKE '%apply_actions(RobotState%' OR s.value LIKE '%muon_%' OR s.value LIKE '%ppo_loss_%' OR s.value LIKE '%mingru_scan_backward%' OR s.value LIKE '%sum_rows_to_precision_kernel%' ORDER BY k.deviceId,k.contextId,k.streamId,k.start,k.end`);
  const windows=buildWindows(anchors),context={...windows,catalog:catalogLookup(catalog)};
  const training=[...windows.training.entries()].flatMap(([stream,list])=>list.map(window=>({...window,stream}))).sort((a,b)=>a.start-b.start);
  if(training.length<2)throw new Error('Need at least two complete PPO epochs to exclude the first epoch and capture setup');
  if(windows.training.size!==1)throw new Error('This profile recipe currently requires one learner GPU stream');
  const start=training[0].end,end=training.at(-1).end,where=`k.start>=${start} AND k.end<=${end}`;
  const byCategory=new Map(),byDetail=new Map(),byKernel=new Map(),byEvidence=new Map();let totalNs=0,calls=0,samples=0,physicsSteps=0,newtonIterations=0;
  const query=`SELECT json_object('start',k.start,'end',k.end,'deviceId',k.deviceId,'contextId',k.contextId,'streamId',k.streamId,'name',s.value,'registersPerThread',k.registersPerThread,'localMemoryPerThread',k.localMemoryPerThread) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE ${where} ORDER BY k.deviceId,k.contextId,k.streamId,k.start,k.end`;
  await streamRows(db,query,row=>{
    const duration=row.end-row.start;if(duration<0)throw new Error('Negative kernel duration');
    const label=classifyKernel(row,context);totalNs+=duration;calls++;
    aggregate(byCategory,label.category,duration,row);aggregate(byDetail,`${label.category}/${label.detail}`,duration,row);
    aggregate(byKernel,row.name,duration,row);aggregate(byEvidence,label.evidence,duration,row);
    if(isLearnerSample(row.name))samples++;
    if(/_next_time_\w+_cuda_kernel_forward/.test(row.name)||/rek5_native::rp_step_kernel\(/.test(row.name))physicsSteps++;
    if(/linesearch_iterative__locals__kernel_\w+_cuda_kernel_forward/.test(row.name))newtonIterations++;
  });
  const expectedSamples=(training.length-1)*horizon,completeSamples=samples===expectedSamples;
  if(!completeSamples)windows.warnings.push(`Learner sample count ${samples} differs from ${expectedSamples}; training SPS unavailable`);
  const summarize=map=>[...map.values()].sort((a,b)=>b.nanoseconds-a.nanoseconds).map(({nanoseconds,...value})=>({...value,milliseconds:nanoseconds/1e6,averageMicrosecondsPerCall:nanoseconds/value.calls/1e3,percentOfSummedKernelTime:totalNs?nanoseconds/totalNs*100:null}));
  const busy=rows(db,`WITH ordered AS (SELECT deviceId,start,end,MAX(end) OVER(PARTITION BY deviceId ORDER BY start,end ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE ${where}) SELECT deviceId,SUM(MAX(0,end-MAX(start,COALESCE(previous_end,start)))) AS kernel_union_ns FROM ordered GROUP BY deviceId`);
  const api=[];
  for(const table of ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'])if(tables.has(table))api.push(...rows(db,`SELECT '${table}' AS source,s.value AS api,COUNT(*) AS calls,SUM(k.end-k.start)/1e6 AS milliseconds FROM ${table} k JOIN StringIds s ON s.id=k.nameId WHERE ${where} GROUP BY s.value ORDER BY milliseconds DESC LIMIT 20`));
  const copies=tables.has('CUPTI_ACTIVITY_KIND_MEMCPY')?rows(db,`SELECT copyKind,COUNT(*) AS calls,SUM(bytes) AS bytes,SUM(end-start)/1e6 AS milliseconds FROM CUPTI_ACTIVITY_KIND_MEMCPY k WHERE ${where} GROUP BY copyKind`):[];
  const unknown=(byCategory.get('unclassified')?.nanoseconds||0)+(byCategory.get('unclassified_gemm')?.nanoseconds||0);
  return {schemaVersion:1,measurement:'instrumented_native_headless_training',pythonExecuted:false,
    roi:{definition:'after first complete PPO epoch through final complete PPO epoch',startNs:start,endNs:end,wallSeconds:(end-start)/1e9,
      completePpoEpochs:training.length,measuredPpoEpochs:training.length-1,agents,horizon,requestedSteps,
      learnerSampleKernelCalls:samples,expectedSampleKernelCalls:expectedSamples,measuredActionTransitions:completeSamples?samples*agents:null,
      profiledSteadyStateTrainingSps:completeSamples?samples*agents/((end-start)/1e9):null,physicsBatchStepKernelCalls:physicsSteps,
      observedPhysicsSubstepsPerActionBatch:samples?physicsSteps/samples:null,newtonIterationBatchKernelCalls:newtonIterations,
      observedNewtonIterationsPerPhysicsBatchStep:physicsSteps?newtonIterations/physicsSteps:null},
    gpu:{kernelCalls:calls,summedKernelMilliseconds:totalNs/1e6,kernelBusy:busy.map(row=>({deviceId:row.deviceId,milliseconds:row.kernel_union_ns/1e6,percentOfRoiWall:row.kernel_union_ns/(end-start)*100})),
      unclassifiedPercentOfKernelTime:totalNs?unknown/totalNs*100:null,categories:summarize(byCategory),subcategories:summarize(byDetail),attributionEvidence:summarize(byEvidence),topKernels:summarize(byKernel).slice(0,40)},
    cudaApi:api,memcopies:copies,warnings:windows.warnings,
    interpretation:['Training SPS includes environment, inference, PPO, and between-epoch host gaps inside the measured interval.',
      'Nsight node tracing adds overhead. Compare speed against an unprofiled run with the same configuration.',
      'Category percentages use summed kernel durations, not wall time. Concurrent kernels can overlap.',
      'Kernel-uncovered wall time can include copies, synchronization, scheduling, or CPU work; it is not automatically CPU compute.',
      'Shared GEMMs without a verified controller/training boundary remain unclassified. Raw traces stay private.']};
}
async function main(){
  const arguments_=process.argv.slice(2),options={};
  for(let i=0;i<arguments_.length;i+=2){if(!arguments_[i]?.startsWith('--')||arguments_[i+1]===undefined)throw new Error('Use --db PATH --catalog PATH --out PATH --agents N --horizon N --requested-steps N');options[arguments_[i].slice(2)]=arguments_[i+1];}
  for(const name of ['db','catalog','out','agents','horizon','requested-steps'])if(!options[name])throw new Error(`Missing --${name}`);
  const report=await analyze({db:options.db,catalog:JSON.parse(readFileSync(options.catalog,'utf8')),agents:Number(options.agents),horizon:Number(options.horizon),requestedSteps:Number(options['requested-steps'])});
  writeFileSync(options.out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  process.stdout.write(JSON.stringify({output:options.out,roi:report.roi,categories:report.gpu.categories,unclassifiedPercent:report.gpu.unclassifiedPercentOfKernelTime,warnings:report.warnings},null,2)+'\n');
}
if(process.argv[1]===fileURLToPath(import.meta.url))main().catch(error=>{console.error(error.stack);process.exitCode=1;});
