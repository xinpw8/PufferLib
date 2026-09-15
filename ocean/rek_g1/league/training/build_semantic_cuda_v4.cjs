#!/usr/bin/env node
'use strict';

// Deterministic report construction from committed aggregate evidence only.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const repo = path.resolve(__dirname,'../../../..');
const trainingPath = 'ocean/rek_g1/native5/validation/semantic-fast-v4-training-20260915';
const evaluationPath = 'ocean/rek_g1/native5/validation/diverse-policy-v4-20260915/mixed-stage/summary.json';
const read = relative => JSON.parse(fs.readFileSync(path.join(repo,relative),'utf8'));
const curriculum = read(trainingPath+'/summary.json');
const stages = curriculum.stages.map(stage=>read(trainingPath+'/'+stage.summary));
const evaluation = read(evaluationPath);
const selected = '0d612521839298ffbe5783ed4fa449286a940b62a3a6768da7cc5ab248eb843b';
assert.equal(stages.length,4);
assert.equal(stages.at(-1).checkpointSha256,selected);
assert.equal(curriculum.cumulativeLearnerTransitions,704643072);
assert.equal(evaluation.shaping_weight,0);
assert.equal(evaluation.cases.length,12);
const names = new Set();
const conditions = evaluation.cases.map(result=>{
  assert(!names.has(result.name));names.add(result.name);
  assert.equal(result.checkpoint_sha256,selected);
  assert.equal(result.backend,'semantic_cuda');
  assert.equal(result.precision,'bf16');assert.equal(result.selection,'sampled');
  assert.equal(result.shaping_weight,0);assert.equal(result.failure_bits,0);
  assert.equal(result.both_sides,true);assert.equal(result.terminal_recurrent_reset,true);
  assert([20,300].includes(result.round_seconds));assert(['fixed','heldout'].includes(result.fixture));
  for(const field of ['wins','losses','draws','games','zero_hit_games','policy_points','opponent_points'])
    assert(Number.isSafeInteger(result[field]) && result[field]>=0,`Invalid ${field} for ${result.name}`);
  assert.equal(result.games,result.wins+result.losses+result.draws);assert(result.games>0);
  assert(result.zero_hit_games<=result.games);
  return {opponent:result.opponent==='checkpoint'?'Older V3 policy ('+result.opponent_sha256.slice(0,12)+')':result.opponent,
    geometry:result.fixture==='fixed'?'Fixed starts; sampled actions':'Held-out starts; seed 10001; gap 0.55–2.5 m; heading ±pi rad',
    roundSeconds:result.round_seconds,wins:result.wins,losses:result.losses,draws:result.draws,games:result.games,
    zeroHitGames:result.zero_hit_games,meanPoints:result.policy_points/result.games,
    meanOpponentPoints:result.opponent_points/result.games,totalPolicyPoints:result.policy_points,totalOpponentPoints:result.opponent_points,
    checkpointSha256:selected,precision:result.precision,actionSelection:result.selection,bothSides:true,
    opponentCheckpointSha256:result.opponent_sha256||null,shapingWeight:0,evidenceCondition:result.name};
});
assert.equal(conditions.find(item=>item.evidenceCondition==='fixed-scripted-300').wins,61);
const sources = [trainingPath+'/summary.json',evaluationPath,
  ...stages.flatMap(stage=>['summary.json','training.ini','process-timing.txt','checkpoint-hashes.txt',
    ...(stage.initialization.kind==='weights_only'?['verified-warm-start.txt']:[])].map(file=>trainingPath+'/'+stage.run+'/'+file))];
for(const file of sources)assert(fs.statSync(path.join(repo,file)).isFile());
const report = {
  status:'completed',steps:curriculum.cumulativeLearnerTransitions,
  trainingSps:curriculum.cumulativeTrainingSps,trainingSeconds:curriculum.cumulativeTrainingSeconds,
  processWallSeconds:curriculum.cumulativeProcessSeconds,benchmarkRoundSeconds:20,checkpointSha256:selected,
  trainingScope:'Selected four-stage weights-linked curriculum. SPS is cumulative learner transitions divided by the sum of native training-loop uptimes, including rollout and PPO; process time sums the four complete process measurements.',
  trainingStages:stages.map(stage=>({run:stage.run,steps:stage.steps,trainingSps:stage.trainingSps,
    trainingSeconds:stage.trainingSeconds,processWallSeconds:stage.processWallSeconds,finalRollingSps:stage.finalRollingSps,
    checkpointSha256:stage.checkpointSha256,initialization:stage.initialization,
    roundSeconds:stage.runtimeParameters.round_seconds,opponentMode:stage.runtimeParameters.opponent_mode,
    frozenOpponent:stage.frozenOpponent,rawTrainingRoundTotals:stage.rawTrainingRoundTotals,
    finalDisplayedWindow:stage.finalDisplayedWindow,settingsSource:trainingPath+'/'+stage.run+'/summary.json'})),
  frozenEvaluationProtocol:{checkpointSha256:selected,actionSelection:'sampled',precision:'bf16',observationEncoding:'scaled_polar_xy',
    shapingWeight:0,bothSides:true,terminalRecurrentReset:true,policyRngSeed:10001},
  frozenEvaluationConditions:conditions,
  note:'Completed historical native C++/CUDA training on Spark, not a live job. The selected checkpoint carries 704,643,072 learner transitions through four weights-only curriculum stages: optimizer, RNG, recurrent state, step counter and learning-rate schedule reset at each verified warm start. Combined training SPS includes rollout and PPO across 512 arenas and excludes process startup. Final mixed-stage SPS is 1,715,495.34 with additional frozen-policy inference; per-stage times are listed in the evidence. The separate 300-second fine-tune regressed and was rejected, so its transitions, timing and results are excluded from this selected checkpoint. Frozen evaluation has shaping 0 and uses the exact selected BF16 sampled policy on both sides. At 300 seconds it beat the neutral opponent 64/64 and scripted opponent 61/64, with 3 losses; longer-round strength is measured only for those fixed-start conditions. Each 20-second condition has 1,024 matches; each 300-second condition has 64. Fixed-start repetition varies action randomness, not geometry. No overall win rate is inferred across the distinct conditions, and these results do not establish human strength or authentic REK parity. Physical knockdowns remain unmodeled. Earlier V3 results and greedy inference are different protocols.',
  evidenceSources:sources
};
fs.writeFileSync(path.join(__dirname,'semantic-cuda-v4.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({checkpoint:selected,steps:report.steps,conditions:conditions.length,trainingSps:report.trainingSps,scripted300:'61 W / 3 L / 0 D'}));
