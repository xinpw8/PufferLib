'use strict';

const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {summarizeEval, compareObjectives} = require('./metrics.cjs');

const readJson = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const round = (value, places = 3) => Number.isFinite(value) ? Number(value.toFixed(places)) : null;
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const armKeys = ['id', 'name', 'lr', 'entropy', 'clip', 'vf', 'horizon', 'gamma', 'lambda', 'discountHalfLifeSeconds', 'traceHalfLifeSeconds'];
const publicArm = arm => Object.fromEntries(armKeys.filter(key => arm[key] !== undefined).map(key => [key, arm[key]]));

function publicEval(value) {
  return {
    n: value.n, policySide: 0, discardedSide1Count: value.discardedSide1Count,
    meanOwnPoints: round(value.meanOwnPoints), meanConcededPoints: round(value.meanConcededPoints),
    meanMargin: round(value.meanMargin), wins: value.wins, losses: value.losses, draws: value.draws,
    roundWinRate: round(value.winRate, 4),
    roundIndependenceStandardErrors: {ownPoints: round(value.ownPointsStandardError), margin: round(value.marginStandardError)},
    roundIndependenceWilson95: {lower: round(value.winWilson95.lower, 4), upper: round(value.winWilson95.upper, 4)},
    uncertaintyCaution: 'Round-level intervals assume independent rounds; shared seeds and repeated fixtures can violate that assumption.',
    actions: null, fullMatchWins: null,
  };
}

function readEvaluation(stage, label, expectedSha) {
  const directory = path.join(stage, 'evaluations', label);
  const recordPath = path.join(directory, 'rounds.private.jsonl');
  if (!fs.existsSync(recordPath)) return null;
  const metrics = summarizeEval(recordPath, expectedSha);
  const records = fs.readFileSync(recordPath, 'utf8').split(/\r?\n/).filter(line => line.trim())
    .map(line => JSON.parse(line)).filter(row => row.policy_side === 0);
  return {metrics, records};
}

function pairedStatistics(candidate, reference) {
  if (!candidate || !reference) return {available: false, reason: 'Paired round records unavailable'};
  const key = row => `${row.seed}:${row.arena}:${row.episode}`;
  const referenceByKey = new Map(reference.map(row => [key(row), row]));
  const seen = new Set();
  const clusters = new Map();
  for (const row of candidate) {
    const identity = key(row);
    assert(!seen.has(identity), 'Duplicate candidate pairing identity');
    seen.add(identity);
    const baseline = referenceByKey.get(identity);
    if (!baseline) continue;
    assert.equal(row.duration_ticks, baseline.duration_ticks, 'Paired evaluation durations differ');
    const own = row.score[0] - baseline.score[0];
    const conceded = row.score[1] - baseline.score[1];
    const win = Number(row.winner === 0) - Number(baseline.winner === 0);
    if (!clusters.has(row.seed)) clusters.set(row.seed, {seed: row.seed, n: 0, own: 0, conceded: 0, margin: 0, win: 0});
    const cluster = clusters.get(row.seed);
    cluster.n++;
    cluster.own += own;
    cluster.conceded += conceded;
    cluster.margin += own - conceded;
    cluster.win += win;
  }
  const groups = [...clusters.values()].sort((a, b) => a.seed - b.seed);
  const n = groups.reduce((sum, group) => sum + group.n, 0);
  if (!n) return {available: false, reason: 'No common seed/arena/episode records'};
  const keys = ['own', 'conceded', 'margin', 'win'];
  const pooled = Object.fromEntries(keys.map(name => [name, groups.reduce((sum, group) => sum + group[name], 0) / n]));
  const deltaNames = {own: 'meanOwnPoints', conceded: 'meanConcededPoints', margin: 'meanMargin', win: 'roundWinRate'};
  const deltas = value => Object.fromEntries(keys.map(name => [deltaNames[name], round(value[name], name === 'win' ? 4 : 3)]));
  let confidenceIntervals95 = null;
  let intervalNote = `Only ${groups.length} independent evaluation seed clusters. A calibrated 95% interval cannot be supported by this seed count; per-seed changes and observed ranges are reported instead.`;
  if (groups.length >= 5) {
    let state = 0x6d2b79f5;
    const random = () => {
      state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
      return (state >>> 0) / 4294967296;
    };
    const samples = Object.fromEntries(keys.map(name => [name, []]));
    for (let repeat = 0; repeat < 10000; repeat++) {
      const totals = {n: 0, own: 0, conceded: 0, margin: 0, win: 0};
      for (let draw = 0; draw < groups.length; draw++) {
        const group = groups[Math.floor(random() * groups.length)];
        totals.n += group.n;
        for (const name of keys) totals[name] += group[name];
      }
      for (const name of keys) samples[name].push(totals[name] / totals.n);
    }
    confidenceIntervals95 = Object.fromEntries(keys.map(name => {
      samples[name].sort((a, b) => a - b);
      return [deltaNames[name], {lower: round(samples[name][249]), upper: round(samples[name][9749])}];
    }));
    intervalNote = 'Deterministic percentile bootstrap of evaluation seed clusters, 10,000 resamples. Few clusters can still make coverage unreliable; paired fixtures do not guarantee identical action-dependent random draws.';
  }
  return {
    available: true, pairing: 'common evaluation seed, arena and episode; policy side 0',
    n, evaluationSeedClusters: groups.length,
    unmatchedCandidateRounds: candidate.length - n, unmatchedReferenceRounds: reference.length - n,
    differences: deltas(pooled),
    perSeed: groups.map(group => ({seed: group.seed, n: group.n, differences: deltas(Object.fromEntries(keys.map(name => [name, group[name] / group.n])))})),
    observedSeedMeanRanges: Object.fromEntries(keys.map(name => [deltaNames[name], {
      minimum: round(Math.min(...groups.map(group => group[name] / group.n))),
      maximum: round(Math.max(...groups.map(group => group[name] / group.n))),
    }])),
    confidenceIntervals95, intervalNote,
  };
}

function publicComparison(candidate, baseline) {
  const compared = compareObjectives(candidate, baseline);
  return {
    ownScoreImproved: compared.ownScoreImproved,
    marginNotRegressed: compared.marginNotRegressed,
    roundWinRateNotRegressed: compared.roundWinRateNotRegressed,
    scoreImprovedWithGuards: compared.scoreImprovedWithGuards,
    differences: {meanOwnPoints: round(compared.deltas.meanOwnPoints), meanMargin: round(compared.deltas.meanMargin), roundWinRate: round(compared.deltas.roundWinRate, 4)},
    interpretation: 'Descriptive point-estimate comparison, not a significance or promotion claim.',
  };
}

function summarizePhase(stage, phase, plan) {
  const filename = path.join(stage, phase === 'screen' ? 'screen-results.json' : 'confirmation-results.json');
  if (!fs.existsSync(filename)) return {status: 'not_complete', results: []};
  const document = readJson(filename);
  const baselineLabel = phase === 'screen' ? 'unchanged-baseline-screen' : 'unchanged-baseline-confirm';
  const baselineRecords = readEvaluation(stage, baselineLabel, plan.warmSha);
  const baseline = baselineRecords?.metrics || document.baseline;
  const labelFor = result => phase === 'screen' ? `screen-${result.arm.id}` : `confirm-${result.arm.id}-s${result.seed}`;
  const controlBySeed = new Map();
  for (const result of document.results) {
    if (result.arm.id === 'control' && !result.error && result.receipt?.exitCode === 0 && result.evaluation) {
      controlBySeed.set(result.seed, {result, evidence: readEvaluation(stage, labelFor(result), result.modelSha)});
    }
  }
  const results = document.results.map(result => {
    const good = !result.error && result.receipt?.exitCode === 0 && result.metrics && result.evaluation;
    if (!good) return {arm: publicArm(result.arm), trainingSeed: result.seed, budgetTransitions: result.budget, status: 'failed_or_incomplete', exitCode: result.receipt?.exitCode ?? null};
    const evidence = readEvaluation(stage, labelFor(result), result.modelSha);
    const evaluation = evidence?.metrics || result.evaluation;
    const control = controlBySeed.get(result.seed);
    return {
      arm: publicArm(result.arm), trainingSeed: result.seed, budgetTransitions: result.budget, status: 'completed',
      checkpointSha256: result.modelSha,
      training: {
        learnerTransitions: result.metrics.learnerTransitions,
        trainingSps: Math.round(result.metrics.trainingSps), startupInclusiveSps: Math.round(result.metrics.startupInclusiveSps),
        trainingLoopSeconds: round(result.metrics.trainingLoopSeconds), processWallSeconds: round(result.metrics.processWallSeconds),
        uptimeSource: result.metrics.uptimeSource, nativeFailureBits: result.metrics.failureBits,
        finalIniStale: result.metrics.nativeIni?.stale ?? null,
      },
      evaluation: publicEval(evaluation),
      versusUnchangedCheckpoint: publicComparison(evaluation, baseline),
      pairedVersusUnchangedCheckpoint: pairedStatistics(evidence?.records, baselineRecords?.records),
      versusMatchedTrainingControl: control ? publicComparison(evaluation, control.evidence?.metrics || control.result.evaluation) : null,
      pairedVersusMatchedTrainingControl: control ? pairedStatistics(evidence?.records, control.evidence?.records) : null,
    };
  });
  const completed = results.filter(result => result.status === 'completed');
  const throughput = completed.map(result => result.training.trainingSps);
  const ranked = [...completed].sort((a, b) => b.evaluation.meanOwnPoints - a.evaluation.meanOwnPoints || b.evaluation.roundWinRate - a.evaluation.roundWinRate || b.evaluation.meanMargin - a.evaluation.meanMargin);
  const summary = {
    status: 'completed', baseline: publicEval(baseline), results,
    completedRuns: completed.length, failedOrIncompleteRuns: results.length - completed.length,
    scoreRanking: ranked.map(result => ({arm: result.arm.id, trainingSeed: result.trainingSeed})),
    guardPassingRuns: ranked.filter(result => result.versusUnchangedCheckpoint.scoreImprovedWithGuards).map(result => ({arm: result.arm.id, trainingSeed: result.trainingSeed})),
    trainingSps: throughput.length ? {minimum: Math.min(...throughput), median: median(throughput), maximum: Math.max(...throughput)} : null,
    fullMatchWins: null, livePromotion: false,
  };
  if (phase === 'screen') {
    summary.selectedForLongerBudget = document.selected;
    summary.selectionPassedPointEstimateGuards = document.selectionPassedGuards;
    summary.selectionProvisional = true;
  } else {
    summary.byArm = [...new Set(completed.map(result => result.arm.id))].map(id => {
      const rows = completed.filter(result => result.arm.id === id);
      return {
        arm: id, independentlyTrainedCheckpoints: rows.length, trainingSeeds: rows.map(result => result.trainingSeed),
        meanOwnPointsAcrossCheckpoints: round(mean(rows.map(result => result.evaluation.meanOwnPoints))),
        meanConcededPointsAcrossCheckpoints: round(mean(rows.map(result => result.evaluation.meanConcededPoints))),
        meanMarginAcrossCheckpoints: round(mean(rows.map(result => result.evaluation.meanMargin))),
        meanRoundWinRateAcrossCheckpoints: round(mean(rows.map(result => result.evaluation.roundWinRate)), 4),
        guardPassingCheckpoints: rows.filter(result => result.versusUnchangedCheckpoint.scoreImprovedWithGuards).length,
        note: 'Descriptive average across training seeds. Reused evaluation fixtures are not additional independent evaluation seed clusters.',
      };
    });
  }
  return summary;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function buildPublicSummary(stage) {
  const plan = readJson(path.join(stage, 'plan.json'));
  return {
    schema: 'rek.native_hparam_score_sweep.public.v1', generatedUtc: new Date().toISOString(),
    objective: 'Increase own points per completed simulator round, with no point-estimate regression in score margin or round win rate.',
    executedPlan: {
      optimizer: 'Muon', initialCheckpointSha256: plan.warmSha, nativeTrainerSha256: plan.trainerSha,
      configurations: plan.arms.map(publicArm), screenBudgetTransitions: plan.screenSteps,
      screenTrainingSeeds: [73], screenEvaluationSeeds: plan.screenSeeds,
      screenArenasPerSeedPerSide: 64, screenRoundsPerArena: 1,
      largerBudgetTransitions: plan.promotionSteps, confirmationTrainingSeeds: [73, 947],
      confirmationEvaluationSeeds: plan.confirmationSeeds, confirmationArenasPerSeedPerSide: 128,
      confirmationRoundsPerArena: 1, learnerEnvironments: 512, minibatch: 8192, replayRatio: 1,
      controlHz: 50, roundSeconds: 120, hiddenSize: 256, recurrentLayers: 2,
      learningRateSchedule: 'cosine to zero over each requested run budget', entropyAnnealing: false,
      rolloutState: 'carried across horizons; reset on episode termination',
      checkpointContinuation: 'Each larger-budget run restarts from identical original weights, with fresh optimizer and recurrent state.',
      reward: plan.assumptions.reward,
      calibration: {attemptCost: .15, botAwardProbability: .25, unmeasuredKickFallProbability: 0, actionIdCorrection: true},
      selectedSide: 0, excludedSide: 1,
    },
    screen: summarizePhase(stage, 'screen', plan),
    confirmation: summarizePhase(stage, 'confirmation', plan),
    limitations: [
      'Only native GPU simulator rounds were evaluated. Full-match wins, authentic REK transfer and live promotion are unestablished.',
      'Calibration is asymmetric, so side-1 round records are validated and excluded from all policy metrics.',
      'Two screening and three confirmation evaluation seeds are too few for a reliable seed-cluster confidence interval.',
      'Common seed/arena pairing controls initial fixtures but policy-dependent execution can change random draw consumption.',
      'A score increase can accompany defensive regression; own score, conceded score, margin and round wins are separate measures.',
      'The compact environment does not model full physical balance or falls. Scoring corrections do not establish authentic parity.',
      'Native console uptime is rounded and excludes graph setup; startup-inclusive SPS also includes setup and checkpoint I/O.',
      'Per-round standard errors and Wilson intervals are descriptive under a round-independence assumption. No significance or promotion claim is made.',
    ],
  };
}

if (require.main === module) {
  assert.equal(process.argv.length, 3, 'Usage: node report.cjs STAGE_DIRECTORY');
  const stage = path.resolve(process.argv[2]);
  const summary = buildPublicSummary(stage);
  const destination = path.join(stage, 'PUBLIC_SUMMARY.json');
  fs.writeFileSync(destination, JSON.stringify(summary, null, 2) + '\n', {flag: 'wx', mode: 0o600});
  process.stdout.write(JSON.stringify({output: 'PUBLIC_SUMMARY.json', screen: summary.screen.status, confirmation: summary.confirmation.status}) + '\n');
}

module.exports = {buildPublicSummary, pairedStatistics};
