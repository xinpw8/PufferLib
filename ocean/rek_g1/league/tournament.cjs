'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {League} = require('./league.cjs');
const {NativeWorker} = require('./worker.cjs');

const TICK_MS = 20; // Native REK semantic tick is 50 Hz.
function ensure(value, message) { if (!value) throw new Error(message); }
function safeMessage(error) { return String(error?.message || error).slice(0, 900); }
function terminalValue(value) { return Array.isArray(value) ? value.some(Boolean) : Boolean(value); }
function validScores(state) {
  return Array.isArray(state?.score) && state.score.length === 2
    && state.score.every(value => Number.isFinite(value) && value >= 0);
}
function classification(error, players, knownSide) {
  const message = safeMessage(error);
  if (/^policy:/i.test(message)) {
    const explicit = message.match(/^policy:\s*(?:side\s*[:=]?\s*)?([01])\b/i);
    const side = explicit ? Number(explicit[1]) : knownSide;
    if (side === 0 || side === 1) {
      const reason = /timeout/i.test(message) ? 'policy_timeout' : /disconnect/i.test(message) ? 'policy_disconnect'
        : /quit/i.test(message) ? 'policy_quit' : /illegal.action/i.test(message) ? 'illegal_action' : 'policy_crash';
      return {status: 'forfeit', reason, loserPolicyId: players[side], note: message};
    }
  }
  return {status: 'engine_error', reason: error?.roundRedo ? 'round_redo'
    : /nonfinite|non-finite|physics.*failure|failure.?bits/i.test(message) ? 'physics_nonfinite'
    : /exited|engine.*crash|native worker unavailable/i.test(message) ? 'engine_crash' : 'infrastructure_failure', note: message};
}
function fixtureId(backend, configHash, ids, seed) {
  return `rr-${crypto.createHash('sha256').update(JSON.stringify([backend, configHash, [...ids].sort(), seed])).digest('hex').slice(0, 32)}`;
}
function nativePolicyCommand(policy, side, seed, deterministic = true) {
  ensure(deterministic === true, 'Default tournament protocol must remain deterministic; identify sampling in checkpoint.model.actionSelection');
  if (policy.kind === 'scripted') return {side, scripted: true, checkpoint: '', deterministic: true, seed};
  const actionSelection = Object.hasOwn(policy.checkpoint.model, 'actionSelection') ? policy.checkpoint.model.actionSelection : 'greedy';
  ensure(['sampled', 'greedy'].includes(actionSelection), `Invalid actionSelection for policy ${policy.id || '(unknown)'}: expected sampled or greedy`);
  return {side, checkpoint: policy.checkpoint.path, sha256: policy.checkpoint.sha256,
    hiddenSize: policy.checkpoint.model.hiddenSize || 256, layers: policy.checkpoint.model.layers || 2,
    precision: policy.checkpoint.model.precision || 'bf16',
    observationEncoding: policy.checkpoint.model.observationEncoding,
    recurrentResetTicks: policy.checkpoint.model.recurrentResetTicks || 0,
    legacyFastHidden: policy.checkpoint.model.legacyFastHidden || 0,
    deterministic: actionSelection === 'greedy', seed};
}
function deadline(promise, milliseconds, description) {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(description)), Math.max(1, milliseconds));
  })]).finally(() => clearTimeout(timer));
}
function publicResult(match) {
  return {id: match.id, backend: match.backend, players: match.players, seed: match.seed, status: match.status,
    scores: match.scores, durationMs: match.durationMs, winnerPolicyId: match.winnerPolicyId, reason: match.reason};
}

async function runLeg({league, match, backend, workerConfig, runDirectory, timeoutMs, chunkSteps,
  deterministic, workerFactory = value => new NativeWorker(value), emit = () => {}}) {
  const legDirectory = path.join(runDirectory, match.id);
  fs.mkdirSync(legDirectory, {recursive: true});
  const configFile = path.join(legDirectory, 'worker.private.json');
  const transcript = path.join(legDirectory, 'protocol.private.jsonl');
  const record = (direction, value) => fs.appendFileSync(transcript,
    `${JSON.stringify({at: new Date().toISOString(), direction, value})}\n`, {mode: 0o600});
  fs.writeFileSync(configFile, `${JSON.stringify({...workerConfig, seed: match.seed}, null, 2)}\n`, {mode: 0o600});
  league.startMatch(match.id);
  const started = Date.now();
  let worker = null;
  let lastState = null;
  let firstTick = 0;
  let knownSide;
  let finished;
  const resultBase = {id: match.id, backend: match.backend, configHash: match.configHash,
    seed: match.seed, players: match.players};
  const elapsed = () => lastState && Number.isFinite(lastState.tick) ? Math.max(0, (lastState.tick - firstTick) * TICK_MS) : 0;
  const request = async (op, args = {}) => {
    ensure(Date.now() - started < timeoutMs, 'Tournament wall-time watchdog expired; failure ownership unknown');
    record('request', {op, ...args});
    const remaining = timeoutMs - (Date.now() - started);
    const response = await deadline(worker.request(op, args, remaining), remaining, 'Tournament wall-time watchdog expired; failure ownership unknown');
    record('response', response);
    if (response.state) {
      lastState = response.state;
      ensure(lastState.ok !== false && !lastState.failureBits, lastState.failure || 'Native physics failure bits');
      ensure(validScores(lastState), 'Native worker returned invalid or nonfinite score');
    }
    return response;
  };
  try {
    const policies = league.snapshot().policies;
    for (const id of match.players) league.verifyCheckpoint({backend: match.backend, id});
    worker = workerFactory({executable: backend.executable, config: configFile, env: backend.env || {},
      logFile: path.join(legDirectory, 'worker.stderr.log')});
    record('ready', await deadline(worker.ready, timeoutMs, 'Native worker startup timeout'));
    await request('snapshot');
    for (let side = 0; side < 2; side++) {
      knownSide = side;
      await request('policy', nativePolicyCommand(policies[`${match.backend}/${match.players[side]}`], side, match.seed, deterministic));
      knownSide = undefined;
    }
    await request('reset');
    ensure(Number.isFinite(lastState?.tick), 'Native reset did not report a simulation tick');
    firstTick = lastState.tick;
    const initialRounds = Number(lastState.completedRounds || 0);
    for (;;) {
      const response = await request('step', {action: 1, steps: chunkSteps, stopAtRound: true});
      const events = (response.rounds || []).filter(state => Number(state.completedRounds || 0) > initialRounds);
      const completed = events[0] || (Number(lastState.completedRounds || 0) > initialRounds || terminalValue(lastState.terminal) ? lastState : null);
      if (completed) {
        lastState = completed;
        ensure(lastState.ok !== false && !lastState.failureBits && validScores(lastState), 'Native terminal contains physics failure or invalid score');
        if (lastState.roundResult === 4) {
          const redo = new Error('Native round requested redo; no adjudicated win or loss'); redo.roundRedo = true; throw redo;
        }
        ensure([1, 2, 3].includes(lastState.roundResult), 'Native terminal did not identify a points, KO or tie outcome');
        const winnerSide = Number(lastState.winner);
        ensure(lastState.roundResult === 3 ? winnerSide === -1 : winnerSide === 0 || winnerSide === 1, 'Invalid native terminal winner');
        const outcome = {...resultBase, status: 'completed', scores: lastState.score, durationMs: elapsed(),
          reason: lastState.timeRemaining <= 0 ? 'time_limit' : 'round_limit', roundResult: lastState.roundResult,
          winnerSide, winnerPolicyId: winnerSide < 0 ? null : match.players[winnerSide]};
        finished = league.finishMatch(outcome);
        break;
      }
      ensure(elapsed() <= match.maxDurationMs, 'Native simulation exceeded the configured round/reset-pause budget without a terminal');
    }
  } catch (error) {
    record('failure', {message: safeMessage(error), knownSide: knownSide ?? null});
    finished = league.finishMatch({...resultBase, ...classification(error, match.players, knownSide),
      scores: validScores(lastState) ? lastState.score : null, durationMs: lastState ? elapsed() : null});
  } finally {
    if (worker) {
      // This is the evaluator process created by this leg, never another user's job.
      try { await deadline(worker.close(), 5000, 'Evaluator close timeout'); }
      catch (error) { record('cleanup_failure', {message: safeMessage(error)}); }
      finally { if (worker.child && worker.child.exitCode === null && !worker.child.killed) worker.child.kill('SIGTERM'); }
    }
  }
  emit(publicResult(finished));
  return finished;
}

async function runTournament({configPath, backendId, policyIds, seeds = [1, 2], runDirectory, timeoutMs = 180000,
  chunkSteps = 64, deterministic = true, workerFactory, emit = () => {}}) {
  ensure(path.isAbsolute(configPath) && path.isAbsolute(runDirectory), 'Configuration and private output paths must be absolute');
  ensure(Number.isSafeInteger(timeoutMs) && timeoutMs > 0, 'Invalid wall timeout');
  ensure(Number.isSafeInteger(chunkSteps) && chunkSteps > 0 && chunkSteps <= 512, 'Chunk steps must be 1..512');
  ensure(deterministic === true, 'Default tournament protocol must remain deterministic; sampled policies require explicit checkpoint.model.actionSelection and distinct IDs');
  ensure(seeds.length > 0 && seeds.every(seed => Number.isSafeInteger(seed) && seed >= 0 && seed <= 2147483647), 'Seeds must fit nonnegative native int32');
  ensure(new Set(seeds).size === seeds.length, 'Duplicate seeds');
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const backend = config.backends.find(value => value.id === backendId);
  ensure(backend && backend.available !== false, 'Requested backend unavailable');
  ensure(path.isAbsolute(backend.workerConfig) && path.isAbsolute(backend.executable), 'Backend executable and workerConfig must be absolute');
  const workerConfig = JSON.parse(fs.readFileSync(backend.workerConfig, 'utf8'));
  ensure(Number.isFinite(workerConfig.round_seconds) && workerConfig.round_seconds > 0 && workerConfig.round_seconds <= 3600,
    'Worker configuration must explicitly set round_seconds in (0, 3600]');
  const resetBudgetSeconds = Object.hasOwn(workerConfig, 'round_reset_budget_seconds')
    ? workerConfig.round_reset_budget_seconds : workerConfig.round_seconds;
  ensure(Number.isFinite(resetBudgetSeconds) && resetBudgetSeconds >= 0 && resetBudgetSeconds <= 3600,
    'round_reset_budget_seconds must be finite and in [0, 3600]');
  workerConfig.round_reset_budget_seconds = resetBudgetSeconds;
  // Physics ticks continue during reset pauses while the official round timer
  // pauses. Preserve elapsed physics time and separately bound that extra work.
  const maxDurationMs = Math.ceil((workerConfig.round_seconds + resetBudgetSeconds) * 1000) + TICK_MS;
  const league = new League({file: config.leagueFile});
  const policies = league.snapshot().policies;
  const selected = policyIds || Object.values(policies).filter(policy => policy.backend === backendId && policy.configHash === backend.configHash).map(policy => policy.id);
  ensure(selected.length >= 2 && new Set(selected).size === selected.length, 'Select at least two distinct policies');
  for (const id of selected) {
    const policy = policies[`${backendId}/${id}`];
    ensure(policy?.configHash === backend.configHash, `Unknown or incompatible policy: ${id}`);
    // Validate the immutable per-policy inference protocol before scheduling any
    // matches or launching a worker. A sampled checkpoint ID stays sampled on
    // both side assignments with the fixture seed.
    nativePolicyCommand(policy, 0, seeds[0], deterministic);
  }
  fs.mkdirSync(runDirectory, {recursive: true});
  const lockFile = `${config.leagueFile}.tournament-${fixtureId(backendId, backend.configHash, [], 0)}.lock`;
  let lock;
  try { lock = fs.openSync(lockFile, 'wx', 0o600); }
  catch (error) { if (error.code === 'EEXIST') throw new Error(`Tournament writer already active: ${lockFile}`); throw error; }
  fs.writeFileSync(lock, JSON.stringify({pid: process.pid, backend: backendId, startedAt: new Date().toISOString()}));
  try {
  const results = [];
  for (let a = 0; a < selected.length; a++) for (let b = a + 1; b < selected.length; b++) for (const seed of seeds) {
    const id = fixtureId(backendId, backend.configHash, [selected[a], selected[b]], seed);
    let state = league.snapshot();
    const old = Object.values(state.pairs).find(pair => pair.backend === backendId && pair.configHash === backend.configHash
      && pair.seed === seed && [pair.policyA, pair.policyB].sort().join('\0') === [selected[a], selected[b]].sort().join('\0'));
    ensure(!old || old.maxDurationMs === maxDurationMs || old.matchIds.every(matchId => state.matches[matchId].status !== 'scheduled'),
      'Existing scheduled fixture uses a different round/reset budget; preserve it and choose fresh seeds');
    const pair = old || league.schedulePair({id, backend: backendId, configHash: backend.configHash,
      policyA: selected[a], policyB: selected[b], seed, maxDurationMs}).pair;
    for (const matchId of pair.matchIds) {
      state = league.snapshot(); const match = state.matches[matchId];
      if (match.status === 'running') {
        const interrupted = league.finishMatch({...match, status: 'engine_error', reason: 'runner_interrupted',
          scores: null, durationMs: null, note: 'Previous runner left this match running; cause unknown, no outcome fabricated'});
        results.push(interrupted); emit(publicResult(interrupted)); continue;
      }
      if (match.status !== 'scheduled') { results.push(match); continue; }
      results.push(await runLeg({league, match, backend, workerConfig, runDirectory, timeoutMs, chunkSteps,
        deterministic, workerFactory, emit}));
    }
  }
  const summary = {backend: backendId, configHash: backend.configHash, policies: selected, seeds,
    roundSeconds: workerConfig.round_seconds, resetBudgetSeconds, maximumSimulatedDurationMs: maxDurationMs,
    matches: results.map(publicResult), standings: league.standings({backend: backendId, configHash: backend.configHash})};
  fs.writeFileSync(path.join(runDirectory, 'summary.private.json'), `${JSON.stringify(summary, null, 2)}\n`, {mode: 0o600});
  return summary;
  } finally { fs.closeSync(lock); fs.unlinkSync(lockFile); }
}

function argumentsFrom(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    ensure(argv[index].startsWith('--') && argv[index + 1], 'Arguments require --name value pairs');
    const key = argv[index].slice(2); ensure(!Object.hasOwn(values, key), `Duplicate option: ${key}`); values[key] = argv[index + 1];
  }
  const allowed = new Set(['config', 'backend', 'policies', 'seeds', 'out', 'timeout-ms', 'chunk-steps', 'deterministic']);
  for (const key of Object.keys(values)) ensure(allowed.has(key), `Unknown option: ${key}`);
  ensure(values.config && values.backend && values.out, 'Usage: node tournament.cjs --config /private/server.json --backend mujoco --out /private/results [--policies a,b] [--seeds 1,2]');
  if (values.deterministic) ensure(['true', 'false'].includes(values.deterministic), 'deterministic must be true or false');
  return {configPath: path.resolve(values.config), backendId: values.backend, runDirectory: path.resolve(values.out),
    policyIds: values.policies?.split(','), seeds: values.seeds?.split(',').map(Number),
    timeoutMs: values['timeout-ms'] ? Number(values['timeout-ms']) : undefined,
    chunkSteps: values['chunk-steps'] ? Number(values['chunk-steps']) : undefined,
    deterministic: values.deterministic !== 'false'};
}
if (require.main === module) {
  runTournament({...argumentsFrom(process.argv.slice(2)), emit: value => process.stdout.write(`${JSON.stringify(value)}\n`)})
    .then(summary => process.stdout.write(`${JSON.stringify({backend: summary.backend, matches: summary.matches.length,
      completed: summary.matches.filter(match => match.status === 'completed').length,
      forfeits: summary.matches.filter(match => match.status === 'forfeit').length,
      invalid: summary.matches.filter(match => match.status === 'engine_error').length})}\n`))
    .catch(error => { process.stderr.write(`${safeMessage(error)}\n`); process.exitCode = 1; });
}
module.exports = {runTournament, runLeg, classification, fixtureId, nativePolicyCommand, argumentsFrom};
