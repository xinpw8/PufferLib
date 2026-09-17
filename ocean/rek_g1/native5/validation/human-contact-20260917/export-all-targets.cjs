#!/usr/bin/env node
'use strict';
// Numeric-only export of the fixed nine-target experiment. No raw logs or binaries.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {exact,matchSchema,parseINI,numericINI}=require('./export-results.cjs');
const CONTRACT='g1_scoring_bodyzones_1_2_3_12_13_v1';
const OLD='5989fa23e6ead72a20fa94a55ce4fb5da2db8631d4f2d2fb038a57c02f95ae85';
const tag=value=>({...value,measurement_domain:'reconstructed_ai_simulator',authentic_parity:false,target_contract:CONTRACT,primitive_target_count:9});
const num=v=>typeof v==='number'&&Number.isFinite(v),int=v=>Number.isSafeInteger(v),nat=v=>int(v)&&v>=0;
const fixed=(...v)=>x=>v.includes(x),sha=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
const arr=(n,t)=>v=>Array.isArray(v)&&v.length===n&&v.every(t),fields=(s,t)=>Object.fromEntries(s.split(' ').map(k=>[k,t]));
const nullable=v=>v===null||num(v),digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const modes={opponent_controller:fixed('recovered_bot1_v1'),observation_mode:fixed('rendered_pose_v1'),scoring_mode:fixed('recovered_hit_rules_v2'),geometry_mode:fixed('primitive_samples_v1'),contact_substeps:fixed(8)};
const eventSchemas={
 compact_mode_identity:{event:fixed('compact_mode_identity'),opponent_controller:modes.opponent_controller,observation_mode:modes.observation_mode,geometry_mode:modes.geometry_mode,contact_substeps:modes.contact_substeps,source:fixed('runtime_config')},
 scoring_identity:{event:fixed('scoring_identity'),scoring_mode:modes.scoring_mode,source:fixed('runtime_config')},
 side_result:{event:fixed('side_result'),policy_side:fixed(0,1),...fields('arenas rounds_per_arena wins losses draws points opponent_points falls opponent_falls evaluated_ticks',nat),execution_wall_seconds:num},
 behavior_result:{event:fixed('behavior_result'),policy_side:fixed(0,1),opponent:fixed('scripted'),fixture:fixed('heldout'),...fields('zero_hit_games hit_games',nat),mean_first_hit_seconds_when_hit:nullable,...fields('configured_round_seconds busy_fraction facing_fraction mean_path_length_m mean_minimum_gap_m shaping_weight',num)},
 frozen_policy_evaluation:{event:fixed('frozen_policy_evaluation'),backend:fixed('semantic_cuda'),checkpoint_sha256:sha,opponent_sha256:fixed(''),precision:fixed('bf16'),observation_encoding:fixed('scaled_polar_xy'),selection:fixed('sampled'),...fields('policy_rng_seed opponent_rng_seed wins losses draws failure_bits',nat),...fields('win_rate execution_wall_seconds round_seconds shaping_weight',num),fixture:fixed('heldout'),opponent:fixed('scripted'),...fields('environment_randomizes_seed both_sides terminal_recurrent_reset',fixed(true)),...fields('python_runtime cpu_physics physics_parity',fixed(false)),training_sps:fixed(null),...modes},
};
const configFields={
 base:'gpu_offset checkpoint_interval eval_episodes eval_agents eval_deterministic cudagraphs seed reset_every_horizon async',
 vec:'total_agents num_buffers num_threads num_policies hist_policy_hidden_size hist_policy_num_layers hist_policy_percent',
 selfplay:'enabled max_size seed opp_timeout_steps eval_pool_size eval_games',
 env:'dr num_agents num_bots seed locomotion_segment_ticks round_seconds opponent_hidden_size opponent_num_layers opponent_precision opponent_legacy_fast_hidden opponent_deterministic',
 policy:'hidden_size num_layers',
 train:'gpus total_timesteps learning_rate anneal_lr min_lr_ratio gamma gae_lambda replay_ratio clip_coef vf_coef vf_clip_coef max_grad_norm ent_coef anneal_ent_coef min_ent_coef_ratio momentum minibatch_size horizon vtrace vtrace_rho_clip vtrace_c_clip verb_eps verb_eps_anneal_start verb_eps_anneal_end',sweep:'downsample',
};
const metrics='SPS agent_steps uptime epoch env/n importance util/gpu_percent util/vram_used_gb util/vram_total_gb util/cpu_mem_gb perf/rollout perf/eval_model perf/eval_env perf/eval_copy perf/train_misc perf/train_model perf/train'.split(' ');
function run(inputArg,outputArg){
 const input=fs.realpathSync(inputArg),output=path.resolve(outputArg);
 assert(fs.statSync(input).isDirectory());assert(!fs.existsSync(output));
 const destination=path.join(fs.realpathSync(path.dirname(output)),path.basename(output)),relative=path.relative(input,destination);
 assert(relative==='..'||relative.startsWith('..'+path.sep)||path.isAbsolute(relative));
 const sources=new Map(),outputs=new Map();
 const exists=n=>fs.existsSync(path.join(input,n));
 const read=n=>{const b=fs.readFileSync(path.join(input,n)),d=digest(b);assert(!sources.has(n)||sources.get(n).sha256===d);sources.set(n,{relative_path:n,bytes:b.length,sha256:d});return b.toString('utf8');};
 const lines=n=>read(n).trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
 const json=(n,v)=>outputs.set(n,JSON.stringify(tag(v),null,2)+'\n');
 const jsonl=(n,v)=>outputs.set(n,v.map(x=>JSON.stringify(tag(x))).join('\n')+'\n');
 const exitCode=folder=>{const f=folder+'/exit-code.txt';if(!exists(f))return null;const s=read(f).trim();assert(/^\d+$/.test(s));const n=Number(s);assert(nat(n));return n;};
 const aggregate=rows=>rows.reduce((a,r)=>{a[r.winner<0?'draws':r.winner===r.policy_side?'wins':'losses']++;a.points+=r.score[r.policy_side];a.opponent_points+=r.score[1-r.policy_side];return a;},{games:rows.length,wins:0,losses:0,draws:0,points:0,opponent_points:0});
 function evaluation(folder,label,expected){
  const code=exitCode(folder);if(code!==0)return {status:code===null?'not_observed':'failed',exit_code:code};
  const parameterLines=read(folder+'/stderr.txt').split(/\r?\n/).filter(s=>s.startsWith('semantic_cuda_parameters='));assert.equal(parameterLines.length,1);
  const parameters=JSON.parse(parameterLines[0].slice('semantic_cuda_parameters='.length));assert.equal(parameters.target_contract,CONTRACT);assert.equal(parameters.target_count,9);
  const rows=lines(folder+'/matches.private.jsonl').map((r,i)=>exact(r,matchSchema,label+':'+i));assert.equal(rows.length,512);
  const events=lines(folder+'/summary.jsonl').map((r,i)=>{assert(Object.hasOwn(eventSchemas,r.event));return exact(r,eventSchemas[r.event],label+':event:'+i);});assert.equal(events.length,7);
  const keys=new Set();for(const r of rows){assert.equal(r.policy_sha256,expected);assert.equal(r.seed,200019);assert.equal(r.duration_ticks,6000);assert.equal(r.duration_seconds,120);assert.deepEqual(r.score,r.point_delta_sum);assert(r.score.every(nat));for(const [k,t] of Object.entries(modes))assert(t(r[k]));assert(r.arena>=0&&r.arena<64&&r.episode>=0&&r.episode<4);const k=[r.policy_side,r.arena,r.episode].join(':');assert(!keys.has(k));keys.add(k);}
  const finals=events.filter(r=>r.event==='frozen_policy_evaluation');assert.equal(finals.length,1);const final=finals[0],totals=aggregate(rows);assert.equal(final.checkpoint_sha256,expected);assert.equal(final.failure_bits,0);assert.equal(final.policy_rng_seed,200019);assert.equal(final.round_seconds,120);
  for(const k of ['wins','losses','draws'])assert.equal(final[k],totals[k]);assert.equal(final.win_rate,totals.wins/512);
  for(const side of [0,1]){const selected=events.filter(r=>r.event==='side_result'&&r.policy_side===side);assert.equal(selected.length,1);const a=aggregate(rows.filter(r=>r.policy_side===side));assert.equal(a.games,256);for(const k of ['wins','losses','draws','points','opponent_points'])assert.equal(selected[0][k],a[k]);}
  jsonl(label+'.matches.jsonl',rows);jsonl(label+'.events.jsonl',events);return {status:'completed',exit_code:0,checkpoint_sha256:expected,...totals,evaluation_wall_seconds:final.execution_wall_seconds};
 }
 const before=evaluation('eval-heldout512-new','frozen-before',OLD);assert.equal(before.status,'completed');assert.deepEqual([before.wins,before.losses,before.points,before.opponent_points],[510,2,75022,22294]);
 const take=(file,field,value,schema)=>{const selected=lines(file).filter(r=>r[field]===value);assert.equal(selected.length,1);return exact(selected[0],schema,file);};
 const host=take('primitive-host-test.jsonl','event','primitive_contacts_tests',{event:fixed('primitive_contacts_tests'),checks:fixed(78248),passed:fixed(true),contact_margin_m:fixed(0),static_geometry_only:fixed(true),dynamic_parity_claim:fixed(false)});
 const gpu=take('primitive-gpu-test.jsonl','event','primitive_contacts_cuda',{event:fixed('primitive_contacts_cuda'),passed:fixed(true),n:fixed(12288),checks:fixed(73728),cases_per_pair_type:fixed(2048),pair_types:v=>JSON.stringify(v)===JSON.stringify(['sphere_sphere','sphere_capsule','sphere_box','capsule_capsule','capsule_box','box_box']),positive_overlap_per_type:arr(6,v=>nat(v)&&v>0&&v<2048),...fields('host_gpu_mismatches symmetric_violations transform_violations',fixed(0)),excluded_boundary_cases:nat,boundary_filter_m:num,production_contact_margin_m:fixed(0),temporal_samples:fixed(4),...fields('python_runtime physics_stepping dynamic_parity_claim',fixed(false))});
 const catalogSchema={event:fixed('scoring_v2_catalog_fixture'),cpu_cases:fixed(6),failures:fixed(0),contact_geometry:fixed('synthetic'),impact_catalog:fixed('pinned_native')};
 const catalogHost=take('scoring-host-test.jsonl','event','scoring_v2_catalog_fixture',catalogSchema);
 const catalogGpu=take('scoring-gpu-test.jsonl','event','scoring_v2_catalog_fixture',catalogSchema);
 const adapter=take('scoring-gpu-test.jsonl','event','scoring_v2_production_adapter_fixture',{event:fixed('scoring_v2_production_adapter_fixture'),gpu_cases:fixed(12),failures:fixed(0),...fields('routes limbs v1_points v2_points',arr(6,nat)),authentic_parity:fixed(false)});
 const targetFixture=take('scoring-gpu-test.jsonl','event','native_scoring_targets_production_adapter_fixture',{event:fixed('native_scoring_targets_production_adapter_fixture'),passed:fixed(true),gpu_cases:fixed(36),target_contract:fixed(CONTRACT),primitive_target_count:fixed(9),legacy_target_count:fixed(3),isolated_hip_cases:fixed(6),contact_geometry:fixed('synthetic'),persistent_contact_rescores:fixed(0),authentic_parity:fixed(false)});
 const finite=take('assets-probe/result.jsonl','test','fast_asset_offline_fk',{test:fixed('fast_asset_offline_fk'),passed:fixed(true),checked_finite_values:fixed(698360),max_root_quaternion_norm_error:v=>num(v)&&v>=0&&v<1e-6,min_striker_proxy_radius_m:num,max_striker_proxy_radius_m:num,forbidden_cpu_physics_calls:fixed(0),python_runtime:fixed(false)});
 const targetAssets=take('assets-probe/result.jsonl','test','native_scoring_target_geometry',{test:fixed('native_scoring_target_geometry'),passed:fixed(true),target_contract:fixed(CONTRACT),primitive_target_count:fixed(9),legacy_target_count:fixed(3),striker_count:fixed(12),body_zones:v=>JSON.stringify(v)==='[3,2,2,12,12,12,13,13,13]',authentic_parity:fixed(false)});
 json('validation-tests.json',{host:tag(host),gpu:tag(gpu),catalog_host:tag(catalogHost),catalog_gpu:tag(catalogGpu),production_adapter:tag(adapter),all_targets_production_adapter:tag(targetFixture),finite_assets:tag(finite),target_assets:tag(targetAssets)});
 for(const f of ['source-sha256.txt','binary-sha256.txt'])read(f); // Hash provenance files; never copy their private paths.
 const folder='train-all-targets-r1',code=exitCode(folder);let training={status:code===null?'not_observed':'failed',exit_code:code},finalSha=null;
 if(code===0){
  const ini=parseINI(read(folder+'/logs/rek_native5/train-all-targets-r1.ini')),config={};
  for(const [section,names] of Object.entries(configFields)){assert(ini[section]);config[section]={};for(const name of names.split(' '))config[section][name]=numericINI(ini[section][name],section+'.'+name);}
  assert.equal(config.vec.total_agents,512);assert.equal(config.train.horizon,128);assert.equal(config.train.total_timesteps,67108864);assert.equal(config.env.seed,212);
  assert.deepEqual(Object.keys(ini.metrics).sort(),[...metrics].sort());const series=Object.fromEntries(metrics.map(k=>[k,ini.metrics[k].split(',').map(s=>numericINI(s.trim(),k))]));
  const count=series.agent_steps.length;assert(count>0);for(const s of Object.values(series))assert.equal(s.length,count);assert.equal(series.agent_steps.at(-1),67108864);assert(series.uptime.at(-1)>0);
  const records=Array.from({length:count},(_,i)=>({sample_index:i,...Object.fromEntries(metrics.map(k=>[k,series[k][i]]))}));
  const checkpoints=read(folder+'/checkpoint-hashes.txt').trim().split(/\r?\n/).map(s=>{const m=/^([a-f0-9]{64})\s+.*[\\/](\d{16})\.bin$/.exec(s);assert(m);return {continuation_agent_steps:Number(m[2]),sha256:m[1]};}).sort((a,b)=>a.continuation_agent_steps-b.continuation_agent_steps);
  assert.equal(checkpoints.length,17);checkpoints.forEach((r,i)=>assert.equal(r.continuation_agent_steps,i*4194304));assert.equal(checkpoints[0].sha256,OLD);finalSha=checkpoints.at(-1).sha256;
  const warm=read(folder+'/verified-warm-start.txt').trim().split(/\r?\n/).map(s=>/^([a-f0-9]{64})\s+/.exec(s)?.[1]);assert.equal(warm.length,2);warm.forEach(s=>assert.equal(s,OLD));
  const rounds=exact(JSON.parse(read(folder+'/round-summary.json')),fields('arenas completed_rounds fighter0_wins fighter1_wins ties redos unclassified fighter0_completed_points fighter1_completed_points failure_bits',nat),'training rounds');assert.equal(rounds.failure_bits,0);assert.equal(rounds.arenas,512);assert.equal(rounds.fighter0_wins+rounds.fighter1_wins+rounds.ties+rounds.redos+rounds.unclassified,rounds.completed_rounds);
  const stderr=read(folder+'/stderr.txt');const identity=prefix=>{const a=stderr.split(/\r?\n/).filter(s=>s.startsWith(prefix));assert.equal(a.length,1);return JSON.parse(a[0].slice(prefix.length));};
  const scoring=identity('semantic_cuda_scoring='),parameters=identity('semantic_cuda_parameters=');assert.equal(scoring.mode,'recovered_hit_rules_v2');assert.equal(scoring.geometry,'primitive_samples_v1');assert.equal(scoring.contact_substeps,8);assert.equal(scoring.authentic_parity,false);
  for(const k of ['opponent_controller','observation_mode','scoring_mode'])assert(modes[k](parameters[k]));assert.equal(parameters.physics_parity,false);assert.equal(parameters.target_contract,CONTRACT);assert.equal(parameters.target_count,9);
  const timing=read(folder+'/process-timing.txt'),elapsed=/^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\d+):(\d+(?:\.\d+)?)\s*$/m.exec(timing);assert(elapsed);const seconds=Number(elapsed[1])*60+Number(elapsed[2]);assert(seconds>0);
  training={status:'completed',exit_code:0,initial_checkpoint_sha256:OLD,final_checkpoint_sha256:finalSha,transitions:67108864,training_uptime_seconds:series.uptime.at(-1),mean_transitions_per_training_second:67108864/series.uptime.at(-1),process_wall_seconds:seconds,metric_samples:count,final_logged_SPS:series.SPS.at(-1),final_logged_epoch:series.epoch.at(-1),metric_samples_are_aggregates:true,final_sample_duplicate_retained:metrics.every(k=>series[k].at(-1)===series[k].at(-2))};
  json('training-config.json',{config,runtime_identity:{opponent_controller:parameters.opponent_controller,observation_mode:parameters.observation_mode,scoring_mode:parameters.scoring_mode,geometry_mode:scoring.geometry,contact_substeps:scoring.contact_substeps}});jsonl('training-metrics.jsonl',records);jsonl('checkpoint-identities.jsonl',checkpoints);json('training-rounds.json',rounds);
 }
 json('training-summary.json',training);
 const after=finalSha?evaluation('eval-heldout512-after','frozen-after',finalSha):{status:'not_observed',exit_code:null};
 json('summary.json',{frozen_before:tag(before),training:tag(training),frozen_after:tag(after),authentic_match_with_updated_checkpoint_observed:false});
 outputs.set('README.md','# Nine-target numeric experiment export\n\nAll records concern the reconstructed AI simulator; authentic REK parity is false. The target contract contains nine scoring geoms, including both hips. No authentic match with the updated checkpoint is included.\n\nFrozen evaluation files retain actual native summary events and every completed 512-match ledger. Training config contains named numeric INI fields; metrics retain all 17 logged aggregate series, including any duplicate final sample. Training steps divided by training uptime is distinct from evaluation wall time and final logged SPS. Failures or missing evaluations remain explicitly marked. The export contains no raw logs, captures, model/checkpoint binaries, credentials or private absolute paths. Source/output hashes support byte reproduction, without independently authenticating the experiment or proving transfer to REK.\n');
 json('manifest.json',{schema_version:1,exporter_sha256:digest(fs.readFileSync(__filename)),shared_exporter_sha256:digest(fs.readFileSync(path.join(__dirname,'export-results.cjs'))),sources:[...sources.values()].sort((a,b)=>a.relative_path.localeCompare(b.relative_path)),outputs:[...outputs].map(([n,s])=>({relative_path:n,bytes:Buffer.byteLength(s),sha256:digest(s)})),match_whitelist:Object.keys(matchSchema),numeric_config_whitelist:configFields,metric_whitelist:metrics});
 for(const content of outputs.values())assert(!/(?:[A-Za-z]:[\\/]|\/home\/|\\\\[0-9])/i.test(content));
 fs.mkdirSync(output);for(const [n,s] of outputs)fs.writeFileSync(path.join(output,n),s,{flag:'wx'});
 console.log(JSON.stringify(tag({event:'all_targets_export',files:outputs.size,before_games:before.games,after_games:after.games??0,training_status:training.status,after_status:after.status,verified:true})));
}
if(require.main===module){try{assert.equal(process.argv.length,4);run(process.argv[2],process.argv[3]);}catch{console.error('Export rejected: source validation or destination check failed; private details withheld.');process.exitCode=1;}}
module.exports={run};
