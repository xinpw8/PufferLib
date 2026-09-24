'use strict';

const fs = require('node:fs');
const path = require('node:path');

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function moments(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.length > 1
    ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
    : null;
  return {mean, standardError: variance === null ? null : Math.sqrt(variance / values.length)};
}

function wilson(wins, n) {
  const z = 1.959963984540054;
  const p = wins / n;
  const denominator = 1 + z * z / n;
  const center = (p + z * z / (2 * n)) / denominator;
  const half = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator;
  return {lower: Math.max(0, center - half), upper: Math.min(1, center + half)};
}

function summarizeEval(filename, expectedSha = null) {
  if (expectedSha !== null) check(/^[a-f0-9]{64}$/i.test(expectedSha), 'Invalid expected checkpoint SHA256');
  const records = [];
  const identities = new Set();
  const digests = new Set();
  let discardedSide1Count = 0;
  const lines = fs.readFileSync(filename, 'utf8').split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    if (!lines[index].trim()) continue;
    let record;
    try { record = JSON.parse(lines[index]); }
    catch (error) { throw new Error(`Malformed evaluation JSON at line ${index + 1}: ${error.message}`); }
    const label = `Evaluation line ${index + 1}`;
    check(record && typeof record === 'object' && !Array.isArray(record), `${label}: expected object`);
    check(record.backend === 'semantic_cuda', `${label}: unexpected backend`);
    check(record.policy_side === 0 || record.policy_side === 1, `${label}: invalid policy_side`);
    check(record.diagnostic === false && record.round_feature_override == null, `${label}: diagnostic or unspecified evaluation`);
    check(Number.isSafeInteger(record.duration_ticks) && record.duration_ticks > 0, `${label}: incomplete duration_ticks`);
    if (record.duration_seconds !== undefined) {
      check(Number.isFinite(record.duration_seconds) && record.duration_seconds > 0, `${label}: invalid duration_seconds`);
    }
    check(Array.isArray(record.score) && record.score.length === 2 && record.score.every(Number.isFinite), `${label}: nonfinite or invalid score`);
    check([-1, 0, 1].includes(record.winner), `${label}: invalid winner`);
    check([1, 2, 3].includes(record.round_result), `${label}: incomplete round_result`);
    check((record.round_result === 3) === (record.winner === -1), `${label}: inconsistent draw result`);
    for (const key of ['seed', 'arena', 'episode']) {
      check(Number.isSafeInteger(record[key]) && record[key] >= 0, `${label}: invalid ${key}`);
    }
    check(typeof record.policy_sha256 === 'string' && /^[a-f0-9]{64}$/i.test(record.policy_sha256), `${label}: invalid policy_sha256`);
    const digest = record.policy_sha256.toLowerCase();
    check(expectedSha === null || digest === expectedSha.toLowerCase(), `${label}: checkpoint SHA256 mismatch`);
    digests.add(digest);
    check(digests.size === 1, `${label}: mixed checkpoint SHA256 values`);
    if (record.failure_bits !== undefined) {
      check(record.failure_bits === 0, `${label}: native failure_bits`);
    }
    const identity = [record.seed, record.arena, record.episode, record.policy_side].join(':');
    check(!identities.has(identity), `${label}: duplicate round identity`);
    identities.add(identity);
    if (record.policy_side === 1) discardedSide1Count++;
    else records.push(record);
  }
  check(records.length > 0, 'No complete policy_side=0 evaluation rounds');
  const own = moments(records.map(record => record.score[0]));
  const conceded = moments(records.map(record => record.score[1]));
  const margin = moments(records.map(record => record.score[0] - record.score[1]));
  const wins = records.filter(record => record.winner === 0).length;
  const losses = records.filter(record => record.winner === 1).length;
  const draws = records.length - wins - losses;
  return {
    n: records.length, policySide: 0, discardedSide1Count,
    checkpointSha256: [...digests][0], meanOwnPoints: own.mean,
    meanConcededPoints: conceded.mean, meanMargin: margin.mean,
    ownPointsStandardError: own.standardError, marginStandardError: margin.standardError,
    wins, losses, draws, winRate: wins / records.length,
    winWilson95: wilson(wins, records.length), actions: null,
    failureBits: null, requiresNativeExitCheck: true,
    unit: 'completed round', fullMatchResults: null,
    uncertaintyNote: 'Standard errors and Wilson intervals treat rounds as independent; repeated arenas or shared seeds can reduce effective sample size. Actions and aggregate native failure status are unavailable in round records.',
  };
}

function durationSeconds(text) {
  let seconds = 0;
  let matched = false;
  const units = {d: 86400, h: 3600, m: 60, s: 1, ms: .001};
  for (const match of text.matchAll(/(\d+(?:\.\d+)?)\s*(ms|d|h|m|s)\b/g)) {
    seconds += Number(match[1]) * units[match[2]];
    matched = true;
  }
  return matched && Number.isFinite(seconds) ? seconds : null;
}

function nativeIniSummary(directory, expectedSteps, consoleUptime) {
  const logDirectory = path.join(directory, 'logs', 'rek_native5');
  if (!fs.existsSync(logDirectory)) return null;
  const files = fs.readdirSync(logDirectory).filter(file => file.endsWith('.ini'));
  if (files.length !== 1) return {available: false, reason: 'Expected exactly one native metrics INI', stale: null};
  const filename = path.join(logDirectory, files[0]);
  let section = '';
  const metrics = {};
  for (const line of fs.readFileSync(filename, 'utf8').split(/\r?\n/)) {
    const heading = line.match(/^\s*\[([^\]]+)\]/);
    if (heading) { section = heading[1]; continue; }
    const pair = line.match(/^\s*([^#=]+?)\s*=\s*(.*?)\s*$/);
    if (section === 'metrics' && pair) metrics[pair[1]] = pair[2];
  }
  const last = key => {
    if (metrics[key] === undefined) return null;
    const values = metrics[key].split(',').map(value => Number(value.trim()));
    return values.length && values.every(Number.isFinite) ? values.at(-1) : null;
  };
  const agentSteps = last('agent_steps');
  const uptime = last('uptime');
  const train = last('perf/train');
  const rollout = last('perf/rollout');
  return {
    available: true, path: filename, lastAgentSteps: agentSteps, lastUptimeSeconds: uptime,
    stale: agentSteps === null || uptime === null || agentSteps !== expectedSteps || Math.abs(uptime - consoleUptime) > .002,
    trainFractionOfTimedComponents: train !== null && rollout !== null && train >= 0 && rollout >= 0 && train + rollout > 0 ? train / (train + rollout) : null,
    note: 'Last saved metric bin only, potentially stale or duplicated. This is not the whole-run compute fraction and is never used for training SPS.',
  };
}

function summarizeTraining(directory, steps) {
  check(Number.isSafeInteger(steps) && steps > 0, 'Expected positive exact learner transition count');
  const stdoutPath = ['stdout.txt', 'native.stdout.txt', 'native_stdout.txt']
    .map(file => path.join(directory, file)).find(file => fs.existsSync(file));
  check(stdoutPath, 'Missing native training stdout');
  const stdout = fs.readFileSync(stdoutPath, 'utf8').replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
  const uptimes = [...stdout.matchAll(/(?:^|[\r\n])[^\r\n]*?\bUptime\s+([^\r\n]*?)(?=\s{2,}Model\b|[│|]|[\r\n]|$)/g)];
  check(uptimes.length > 0, 'Missing native final console Uptime');
  const uptime = durationSeconds(uptimes.at(-1)[1]);
  check(uptime !== null && uptime > 0, 'Invalid native final console Uptime');
  const progress = [...stdout.matchAll(/(?:^|[\r\n])\s*[│|]?\s*Steps\s+([\d.]+)([KMBT]?)/g)];
  check(progress.length > 0, 'Missing native console Steps');
  const finalProgress = progress.at(-1);
  const factor = { '': 1, K: 1e3, M: 1e6, B: 1e9, T: 1e12 }[finalProgress[2]];
  const displayedSteps = Number(finalProgress[1]) * factor;
  const decimals = finalProgress[1].split('.')[1]?.length || 0;
  const roundingTolerance = factor * 10 ** -decimals / 2;
  check(Number.isFinite(displayedSteps) && Math.abs(displayedSteps - steps) <= roundingTolerance + 1e-6, 'Final native console Steps do not match requested transitions');
  const timing = fs.readFileSync(path.join(directory, 'process-timing.txt'), 'utf8');
  const wall = timing.match(/Elapsed \(wall clock\) time[^\r\n]*?:\s*([\d:.]+)\s*$/m);
  check(wall, 'Missing process wall time');
  const wallSeconds = wall[1].split(':').map(Number).reduce((sum, value) => 60 * sum + value, 0);
  check(Number.isFinite(wallSeconds) && wallSeconds > 0, 'Invalid process wall time');
  const summaries = [...stdout.matchAll(/native5_round_summary=(\{[^\r\n]+\})/g)];
  check(summaries.length > 0, 'Missing native round summary');
  const rounds = JSON.parse(summaries.at(-1)[1]);
  const countFields = ['completed_rounds', 'fighter0_wins', 'fighter1_wins', 'ties', 'redos', 'unclassified'];
  check(countFields.every(key => Number.isSafeInteger(rounds[key]) && rounds[key] >= 0), 'Invalid training round counts');
  check(countFields.slice(1).reduce((sum, key) => sum + rounds[key], 0) === rounds.completed_rounds, 'Training round counts do not reconcile');
  const failureBits = rounds.failure_bits ?? null;
  check(failureBits === null || failureBits === 0, 'Native training failure_bits');
  return {
    run: path.basename(directory), learnerTransitions: steps,
    displayedFinalSteps: displayedSteps, displayedStepsRoundingTolerance: roundingTolerance,
    trainingLoopSeconds: uptime, trainingSps: steps / uptime,
    uptimeSource: 'final native console Uptime', uptimePrecision: 'rounded console duration',
    processWallSeconds: wallSeconds, startupInclusiveSps: steps / wallSeconds,
    trainingRoundResults: rounds, failureBits, requiresNativeExitCheck: true,
    nativeIni: nativeIniSummary(directory, steps, uptime),
    note: 'Changing-policy training rounds are not held-out evaluation. Caller must independently verify native process exit and exact completed transitions; console Steps are abbreviated.',
  };
}

function compareObjectives(candidate, baseline) {
  for (const summary of [candidate, baseline]) {
    check(summary && summary.policySide === 0 && summary.n > 0, 'Expected policy-side-0 summaries');
    check(['meanOwnPoints', 'meanMargin', 'winRate'].every(key => Number.isFinite(summary[key])), 'Invalid objective summary');
  }
  const deltas = {
    meanOwnPoints: candidate.meanOwnPoints - baseline.meanOwnPoints,
    meanMargin: candidate.meanMargin - baseline.meanMargin,
    roundWinRate: candidate.winRate - baseline.winRate,
  };
  const ownScoreImproved = deltas.meanOwnPoints > 0;
  const marginNotRegressed = deltas.meanMargin >= 0;
  const roundWinRateNotRegressed = deltas.roundWinRate >= 0;
  return {
    ownScoreImproved, marginNotRegressed, roundWinRateNotRegressed,
    scoreImprovedWithGuards: ownScoreImproved && marginNotRegressed && roundWinRateNotRegressed,
    deltas, fullMatchWinComparison: null,
    note: 'Descriptive point-estimate guard, without a significance claim. Compare matched evaluation settings. Round wins do not establish match wins.',
  };
}

module.exports = {summarizeTraining, summarizeEval, compareObjectives};
