#!/usr/bin/env node
'use strict';

// Rebuild aggregate reports from these copied text artifacts. No training or GPU work.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const root = __dirname;
const runs = ['train-v4-close-r1', 'train-v4-pursuit-r1', 'train-v4-orientation-r1', 'train-v4-mixed-r1', 'train-v4-long-r1'];
const selectedStageCount = 4;
const warmPatchSha256 = 'f0bb1f634b62914a07417b7153a4f1e39f2fd04e8b0eff2af3df882cdd582092';
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const read = (run, name) => fs.readFileSync(path.join(root, run, name), 'utf8');
function iniParse(text) {
  const sections = {}; let current;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /^[#;]/.test(line)) continue;
    const section = line.match(/^\[([^\]]+)\]$/);
    if (section) { current = sections[section[1]] = {}; continue; }
    const value = line.match(/^([^=]+?)\s*=\s*(.*)$/);
    assert(current && value, `Invalid INI line: ${line}`);
    current[value[1].trim()] = value[2].trim();
  }
  return sections;
}
function finalMetric(ini, key) {
  const value = Number(ini.metrics[key]?.split(',').at(-1));
  assert(Number.isFinite(value), `Missing finite metric ${key}`);
  return value;
}
function hashes(text) {
  return text.split(/\r?\n/).flatMap(line => {
    const m = line.match(/^([a-f0-9]{64})\s+(.+)$/);
    return m ? [{sha256:m[1], path:m[2]}] : [];
  });
}
function finalDisplay(text, key) {
  const matches = [...text.matchAll(new RegExp(`\\b${key}\\s+(-?\\d+(?:\\.\\d+)?)`, 'g'))];
  assert(matches.length, `Missing displayed field ${key}`);
  return Number(matches.at(-1)[1]);
}
let priorSteps = 0;
const summaries = runs.map((run, stageIndex) => {
  const ini = iniParse(read(run, 'training.ini'));
  const steps = finalMetric(ini, 'agent_steps');
  const trainingSeconds = finalMetric(ini, 'uptime');
  assert.equal(steps, Number(ini.train.total_timesteps));
  assert(trainingSeconds > 0);
  const status = Number(read(run, 'exit-code.txt').trim());
  assert.equal(status, 0);
  const elapsed = read(run, 'process-timing.txt').match(/Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([\d.:]+)/);
  assert(elapsed);
  const processSeconds = elapsed[1].split(':').reduce((sum, part) => sum * 60 + Number(part), 0);
  assert(processSeconds >= trainingSeconds);
  const checkpointName = String(steps).padStart(16, '0') + '.bin';
  const checkpoint = hashes(read(run, 'checkpoint-hashes.txt')).find(item => item.path.endsWith('/' + checkpointName));
  assert(checkpoint);
  const round = JSON.parse(read(run, 'round-summary.json'));
  assert.equal(round.completed_rounds, round.fighter0_wins + round.fighter1_wins + round.ties);
  assert.equal(round.failure_bits, 0);
  const parametersLine = read(run, 'stderr.txt').split(/\r?\n/).find(line => line.startsWith('semantic_cuda_parameters='));
  assert(parametersLine);
  const parameters = JSON.parse(parametersLine.slice('semantic_cuda_parameters='.length));
  assert.equal(parameters.version, 4);
  assert.equal(parameters.round_seconds, Number(ini.env.round_seconds));
  const provenance = hashes(read(run, 'provenance.txt'));
  const build = hashes(read(run, 'build-source-manifest.txt'));
  const trainer = build.find(item => item.path.includes('/trainer/src/pufferl.cu'));
  assert(trainer);
  let initialization = {kind:'fresh', priorSteps:0, optimizer:'fresh', stepZeroReadbackVerified:false, trainerWarmStartPatchSha256:null};
  if (stageIndex) {
    assert.notEqual(ini.base.load_model_path, 'None');
    const verify = hashes(read(run, 'verified-warm-start.txt'));
    const input = hashes(read(run, 'initial-checkpoint.sha256'));
    assert.equal(verify.length, 2); assert.equal(input.length, 1);
    assert.equal(verify[0].sha256, verify[1].sha256);
    assert.equal(input[0].sha256, verify[0].sha256);
    assert(verify[1].path.endsWith('/0000000000000000.bin'));
    assert(read(run, 'stderr.txt').includes('step=0; fresh_optimizer=1; step_zero_readback='));
    initialization = {kind:'weights_only', priorSteps, sourceCheckpointSha256:input[0].sha256,
      optimizer:'fresh', recurrentState:'fresh', rng:'fresh', globalStep:'reset_to_zero', learningRateSchedule:'fresh',
      stepZeroReadbackVerified:true, stepZeroSha256:verify[1].sha256, trainerWarmStartPatchSha256:warmPatchSha256};
  } else assert.equal(ini.base.load_model_path, 'None');
  let frozenOpponent = null;
  if (ini.env.opponent_checkpoint !== 'None') {
    const line = read(run,'stderr.txt').split(/\r?\n/).find(value=>value.startsWith('native5_frozen_opponent_mix='));
    assert(line);
    const mixture = JSON.parse(line.slice('native5_frozen_opponent_mix='.length));
    assert.equal(mixture.frozen_arenas + mixture.runtime_opponent_arenas, Number(ini.vec.total_agents));
    assert.equal(mixture.seed, Number(ini.env.seed));
    assert(/^[a-f0-9]{64}$/.test(ini.env.opponent_sha256));
    assert(read(run,'stderr.txt').includes(`native5 frozen opponent: sha256=${ini.env.opponent_sha256} precision=0 deterministic=0`));
    frozenOpponent = {checkpointSha256:ini.env.opponent_sha256,precision:'bf16',actionSelection:'sampled',mixture};
  }
  const stdout = read(run, 'stdout.txt');
  const report = {
    schema:'rek.compact.training.v4', status:'completed', stage:stageIndex + 1, run,
    selection:stageIndex<selectedStageCount?'selected_lineage':'rejected_experiment',
    steps, cumulativeCurriculumSteps:steps + initialization.priorSteps,
    trainingSeconds, trainingSps:steps / trainingSeconds,
    finalRollingSps:finalMetric(ini, 'SPS'), processWallSeconds:processSeconds,
    processInclusiveSps:steps / processSeconds,
    timingScope:'Training SPS counts fighter-0 learner transitions divided by final native training-loop uptime, including rollout and PPO, excluding process startup. Final rolling SPS is separate. This is a shared Spark measurement, not an uncontended throughput ceiling.',
    metricExtraction:'The final entry of each pinned native log metric is the exact final log value; log_history_bin_mean explicitly preserves its last entry. Other entries may be bin means.',
    hardware:{host:'spark-4ae3', architecture:'aarch64', gpu:'NVIDIA GB10', backend:'semantic_cuda', policyPrecision:'bf16', checkpointStorage:'fp32'},
    checkpointSha256:checkpoint.sha256, initialization,
    runtimeParameters:parameters, frozenOpponent,
    trainingSettings:{base:{seed:ini.base.seed,reset_every_horizon:ini.base.reset_every_horizon,async:ini.base.async},
      vec:ini.vec, policy:ini.policy, train:ini.train, env:{seed:ini.env.seed,round_seconds:ini.env.round_seconds,
        opponent_checkpoint:ini.env.opponent_checkpoint === 'None' ? null : 'private',
        opponent_observation_encoding:ini.env.opponent_observation_encoding,opponent_precision:ini.env.opponent_precision,
        opponent_deterministic:ini.env.opponent_deterministic}},
    rawTrainingRoundTotals:round,
    finalDisplayedWindow:{score:finalDisplay(stdout,'score'),shapedEpisodeReturn:finalDisplay(stdout,'episode_return'),
      hits:finalDisplay(stdout,'hits'),wins:finalDisplay(stdout,'wins'),losses:finalDisplay(stdout,'losses'),draws:finalDisplay(stdout,'draws'),
      scope:'Rounded final dashboard window, not whole-run totals and not frozen evaluation.'},
    strengthConclusion:null,
    strengthNote:'Curriculum training only. Training wins, points and potential-shaped return do not establish frozen policy strength. Opponents and reset distributions are explicit; held-out evaluation is maintained separately.',
    runtimeObjectSha256:provenance.find(item => item.path.endsWith('/fast_runtime.o')).sha256,
    executableSha256:provenance.find(item => item.path.endsWith('/puffer-rek-native5')).sha256,
    patchedTrainerSourceSha256:trainer.sha256,
    evidenceFiles:Object.fromEntries(fs.readdirSync(path.join(root,run)).filter(name => name !== 'summary.json').sort().map(name =>
      [name,sha(fs.readFileSync(path.join(root,run,name)))]))
  };
  priorSteps += steps;
  fs.writeFileSync(path.join(root,run,'summary.json'),JSON.stringify(report,null,2)+'\n');
  return report;
});
for (let stage = 1;stage < summaries.length;stage++) {
  assert.equal(summaries[stage].initialization.sourceCheckpointSha256, summaries[stage-1].checkpointSha256);
  assert.equal(summaries[stage].runtimeObjectSha256,summaries[0].runtimeObjectSha256);
}
const selected = summaries.slice(0,selectedStageCount);
const cumulativeSteps = selected.reduce((sum,item)=>sum+item.steps,0);
const cumulativeTrainingSeconds = selected.reduce((sum,item)=>sum+item.trainingSeconds,0);
const cumulativeProcessSeconds = selected.reduce((sum,item)=>sum+item.processWallSeconds,0);
fs.writeFileSync(path.join(root,'summary.json'),JSON.stringify({schema:'rek.compact.curriculum.training.v4',
  stages:selected.map(item => ({run:item.run,steps:item.steps,trainingSps:item.trainingSps,trainingSeconds:item.trainingSeconds,
    processWallSeconds:item.processWallSeconds,checkpointSha256:item.checkpointSha256,summary:item.run+'/summary.json'})),
  cumulativeLearnerTransitions:cumulativeSteps,cumulativeTrainingSeconds,cumulativeTrainingSps:cumulativeSteps/cumulativeTrainingSeconds,
  cumulativeProcessSeconds,cumulativeProcessInclusiveSps:cumulativeSteps/cumulativeProcessSeconds,
  rejectedExperiments:summaries.slice(selectedStageCount).map(item=>({run:item.run,steps:item.steps,trainingSps:item.trainingSps,
    trainingSeconds:item.trainingSeconds,processWallSeconds:item.processWallSeconds,checkpointSha256:item.checkpointSha256,
    selection:'rejected_experiment',reason:'The long-round fine-tune regressed in independent frozen evaluation. It is excluded from selected-checkpoint transition and timing totals.',summary:item.run+'/summary.json'})),
  strengthConclusion:null,note:'Four selected weights-linked curriculum stages; rejected experiments are separate and excluded from selected totals. This report measures training execution and initialization, not policy strength. Frozen evaluation is maintained separately.'},null,2)+'\n');
console.log(JSON.stringify({validatedRuns:runs,trainingSps:summaries.map(item=>item.trainingSps),warmStartVerified:true}));
