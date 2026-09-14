'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {League, hashFile} = require('./league.cjs');
const {runTournament, classification, argumentsFrom, nativePolicyCommand} = require('./tournament.cjs');
const CONFIG = 'c'.repeat(64);

function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'rek-tournament-test-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const league = new League({file: path.join(directory, 'league.private.json')});
  const weights = path.join(directory, 'model.bin'); fs.writeFileSync(weights, Buffer.from([1, 2, 3, 4]));
  for (const id of ['a', 'b']) league.registerPolicy({backend: 'mujoco', id, label: id, kind: 'trained', configHash: CONFIG,
    checkpoint: {path: weights, sha256: hashFile(weights), format: 'native-bf16', trainingSteps: 1024,
      model: {hiddenSize: 256, layers: 2, precision: 'bf16'}}});
  league.registerPolicy({backend: 'mujoco', id: 'script', label: 'Scripted', kind: 'scripted', configHash: CONFIG, scriptedVersion: 'v1'});
  const workerConfig = path.join(directory, 'worker.private.json');
  fs.writeFileSync(workerConfig, JSON.stringify({seed: 73, round_seconds: 20, arenas: 4}));
  const configPath = path.join(directory, 'server.private.json');
  fs.writeFileSync(configPath, JSON.stringify({leagueFile: league.file, backends: [{id: 'mujoco', configHash: CONFIG,
    executable: path.join(directory, 'fake-native-worker'), workerConfig}]}));
  return {league, directory, options: {configPath, backendId: 'mujoco', policyIds: ['a', 'b'], seeds: [1],
    runDirectory: path.join(directory, 'results'), timeoutMs: 3000, chunkSteps: 64}};
}
function initial() { return {ok: true, tick: 0, score: [0, 0], falls: [0, 0], timeRemaining: 20, terminal: 0, roundResult: 0, winner: -1, completedRounds: 0}; }
function fakeFactory(behavior, collection = []) {
  return spec => {
    const worker = {spec, ready: Promise.resolve({event: 'ready'}), requests: [], closed: false,
      state: initial(), async close() { this.closed = true; }, async request(op, args) {
        this.requests.push({op, args});
        const special = await behavior?.(this, op, args, collection.length - 1);
        if (special) return special;
        if (op === 'reset') this.state = initial();
        if (op === 'step') this.state = {...initial(), tick: 1000, score: [5, 0], timeRemaining: 0,
          terminal: 1, roundResult: 1, winner: 0, completedRounds: 1};
        return {ok: true, state: this.state, rounds: op === 'step' ? [this.state] : []};
      }};
    collection.push(worker); return worker;
  };
}

test('runs paired sides with real metadata, matched seeds, terminal scores and private logs', async t => {
  const {league, directory, options} = fixture(t); const workers = [];
  const result = await runTournament({...options, workerFactory: fakeFactory(null, workers)});
  assert.equal(workers.length, 2); assert.ok(workers.every(worker => worker.closed));
  assert.deepEqual(result.matches.map(match => match.players), [['a', 'b'], ['b', 'a']]);
  assert.ok(result.matches.every(match => match.status === 'completed' && match.durationMs === 20000));
  for (const worker of workers) {
    assert.equal(JSON.parse(fs.readFileSync(worker.spec.config, 'utf8')).seed, 1);
    const policies = worker.requests.filter(request => request.op === 'policy');
    assert.deepEqual(policies.map(request => request.args.side), [0, 1]);
    assert.ok(policies.every(request => request.args.sha256 && request.args.precision === 'bf16'));
    assert.deepEqual(worker.requests.find(request => request.op === 'step').args, {action: 1, steps: 64, stopAtRound: true});
    assert.ok(fs.statSync(path.join(path.dirname(worker.spec.config), 'protocol.private.jsonl')).size > 0);
  }
  assert.ok(league.standings({backend: 'mujoco', configHash: CONFIG}).filter(row => row.id !== 'script').every(row => row.total.wins === 1 && row.total.losses === 1));
  assert.ok(fs.existsSync(path.join(directory, 'results', 'summary.private.json')));
});

test('loads scripted mode explicitly for either fighter side', async t => {
  const {options} = fixture(t); const workers = [];
  await runTournament({...options, policyIds: ['a', 'script'], workerFactory: fakeFactory(null, workers)});
  const commands = workers.flatMap(worker => worker.requests.filter(request => request.op === 'policy').map(request => request.args));
  assert.deepEqual(commands.filter(command => command.scripted).map(command => command.side), [1, 0]);
  assert.ok(commands.filter(command => command.scripted).every(command => command.checkpoint === ''));
});

test('native KO winner overrides hit score without rewriting it', async t => {
  const {league, options} = fixture(t);
  const workerFactory = fakeFactory((worker, op) => {
    if (op !== 'step') return;
    worker.state = {...initial(), tick: 100, score: [1, 9], terminal: 1, winner: 0, roundResult: 2, completedRounds: 1};
    return {state: worker.state, rounds: [worker.state]};
  });
  const result = await runTournament({...options, workerFactory});
  assert.ok(result.matches.every(match => match.status === 'completed' && match.winnerPolicyId === match.players[0]));
  assert.deepEqual(result.matches[0].scores, [1, 9]);
  assert.equal(Object.values(league.snapshot().matches)[0].roundResult, 2);
});

test('draws remain nonwinning and redo rounds are invalid', async t => {
  const {options} = fixture(t);
  const result = await runTournament({...options, workerFactory: fakeFactory((worker, op, args, leg) => {
    if (op !== 'step') return;
    worker.state = {...initial(), tick: 1000, score: [0, 0], terminal: 1, winner: -1,
      roundResult: leg === 0 ? 3 : 4, completedRounds: 1};
    return {state: worker.state, rounds: [worker.state]};
  })});
  assert.equal(result.matches[0].status, 'completed'); assert.equal(result.matches[0].winnerPolicyId, null);
  assert.equal(result.matches[1].status, 'engine_error'); assert.equal(result.matches[1].reason, 'round_redo');
  assert.ok(result.standings.every(row => row.total.wins === 0));
});

test('explicit guilty policy crashes forfeit even while leading', async t => {
  const {options} = fixture(t);
  const result = await runTournament({...options, workerFactory: fakeFactory((worker, op) => {
    if (op !== 'step') return;
    if (worker.state.tick === 0) {
      worker.state = {...initial(), tick: 64, score: [20, 0]};
      return {state: worker.state, rounds: []};
    }
    throw new Error('policy: side=0 illegal action');
  })});
  assert.ok(result.matches.every(match => match.status === 'forfeit' && match.winnerPolicyId === match.players[1]));
  assert.ok(result.matches.every(match => match.reason === 'illegal_action'));
  assert.ok(result.matches.every(match => match.scores[0] === 20 && match.scores[1] === 0));
});

test('unknown crashes and nonfinite physics never fabricate policy wins', async t => {
  const {options} = fixture(t);
  const result = await runTournament({...options, workerFactory: fakeFactory((worker, op, args, leg) => {
    if (op === 'step') throw new Error(leg === 0 ? 'Native worker exited (139)' : 'physics nonfinite state');
  })});
  assert.ok(result.matches.every(match => match.status === 'engine_error' && match.winnerPolicyId === null));
  assert.deepEqual(result.matches.map(match => match.reason), ['engine_crash', 'physics_nonfinite']);
});

test('startup failures preserve unknown scores and release owned workers', async t => {
  const {options} = fixture(t); let closes = 0;
  const result = await runTournament({...options, workerFactory: () => ({ready: Promise.reject(new Error('Native startup failed')),
    async close() { closes++; }})});
  assert.equal(closes, 2);
  assert.ok(result.matches.every(match => match.status === 'engine_error' && match.scores === null && match.durationMs === null));
});

test('a stuck worker is bounded by wall watchdog and cannot gain wins', async t => {
  const {options} = fixture(t); let closes = 0;
  const result = await runTournament({...options, timeoutMs: 15, workerFactory: () => ({ready: new Promise(() => {}), async close() { closes++; }})});
  assert.equal(closes, 2); assert.ok(result.matches.every(match => match.status === 'engine_error'));
});

test('rerunning completed fixtures does not launch workers or duplicate outcomes', async t => {
  const {league, options} = fixture(t);
  await runTournament({...options, workerFactory: fakeFactory()});
  const revision = league.snapshot().revision;
  await runTournament({...options, workerFactory: () => { throw new Error('Must not launch'); }});
  assert.equal(league.snapshot().revision, revision);
  assert.equal(Object.keys(league.snapshot().matches).length, 2);
});

test('abandoned running fixtures become invalid instead of disappearing', async t => {
  const {league, options} = fixture(t);
  const {matches} = league.schedulePair({id: 'previous', backend: 'mujoco', configHash: CONFIG,
    policyA: 'a', policyB: 'b', seed: 1, maxDurationMs: 40020});
  league.startMatch(matches[0].id);
  const workers = [];
  const result = await runTournament({...options, workerFactory: fakeFactory(null, workers)});
  assert.equal(workers.length, 1); assert.equal(result.matches[0].reason, 'runner_interrupted');
  assert.equal(result.matches[0].scores, null);
});

test('points-winner inconsistency is invalid rather than misranked', async t => {
  const {options} = fixture(t);
  const result = await runTournament({...options, workerFactory: fakeFactory((worker, op) => {
    if (op !== 'step') return;
    worker.state = {...initial(), tick: 1000, score: [0, 2], winner: 0, roundResult: 1, terminal: 1, completedRounds: 1};
    return {state: worker.state, rounds: [worker.state]};
  })});
  assert.ok(result.matches.every(match => match.status === 'engine_error' && match.winnerPolicyId === null));
});

test('CLI and policy fault attribution reject ambiguity', () => {
  const args = argumentsFrom(['--config', '/private/config.json', '--backend', 'mujoco', '--out', '/private/results', '--seeds', '1,9']);
  assert.deepEqual(args.seeds, [1, 9]); assert.equal(args.deterministic, true);
  assert.throws(() => argumentsFrom(['--confgi', 'a']), /Unknown option/);
  assert.equal(classification(new Error('policy: inference failed'), ['a', 'b']).status, 'engine_error');
  assert.equal(classification(new Error('policy: inference failed'), ['a', 'b'], 1).loserPolicyId, 'b');
});

test('legacy native inference flags are preserved rather than guessed', () => {
  const command = nativePolicyCommand({kind: 'trained', checkpoint: {path: '/private/legacy.bin', sha256: CONFIG,
    model: {hiddenSize: 256, layers: 2, precision: 'fp32', observationEncoding: 'raw223', recurrentResetTicks: 64, legacyFastHidden: 1}}}, 1, 17, true);
  assert.equal(command.observationEncoding, 'raw223'); assert.equal(command.recurrentResetTicks, 64);
  assert.equal(command.precision, 'fp32'); assert.equal(command.legacyFastHidden, 1);
});

test('per-policy sampling overrides the default without changing the other fighter', async t => {
  const {league, options} = fixture(t);
  const source = league.snapshot().policies['mujoco/a'];
  league.registerPolicy({...source, id: 'a-sampled', label: 'A sampled', checkpoint: {...source.checkpoint,
    model: {...source.checkpoint.model, actionSelection: 'sampled'}}});
  const workers = [];
  const result = await runTournament({...options, policyIds: ['a-sampled', 'b'], workerFactory: fakeFactory(null, workers)});
  assert.deepEqual(result.matches.map(match => match.players), [['a-sampled', 'b'], ['b', 'a-sampled']]);
  assert.deepEqual(workers.map(worker => worker.requests.filter(request => request.op === 'policy').map(request => request.args.deterministic)),
    [[false, true], [true, false]]);
  assert.ok(workers.every(worker => worker.requests.filter(request => request.op === 'policy').every(request => request.args.seed === 1)));
});

test('greedy and omitted actionSelection remain deterministic; invalid enums never launch', async t => {
  const {league, options} = fixture(t);
  const source = league.snapshot().policies['mujoco/a'];
  assert.equal(nativePolicyCommand(source, 0, 1).deterministic, true);
  assert.equal(nativePolicyCommand({...source, checkpoint: {...source.checkpoint,
    model: {...source.checkpoint.model, actionSelection: 'greedy'}}}, 0, 1).deterministic, true);
  for (const actionSelection of [null, undefined, 42, {}, 'stochastic']) {
    assert.throws(() => nativePolicyCommand({...source, checkpoint: {...source.checkpoint,
      model: {...source.checkpoint.model, actionSelection}}}, 0, 1), /Invalid actionSelection/);
  }
  league.registerPolicy({...source, id: 'invalid-selection', label: 'Invalid selection test', checkpoint: {...source.checkpoint,
    model: {...source.checkpoint.model, actionSelection: 'stochastic'}}});
  let launches = 0;
  await assert.rejects(() => runTournament({...options, policyIds: ['invalid-selection', 'b'], workerFactory() { launches++; }}), /Invalid actionSelection/);
  assert.equal(launches, 0); assert.equal(Object.keys(league.snapshot().pairs).length, 0);
});

test('official 20-second terminal after reset pauses preserves 20.180 seconds of physics time', async t => {
  const {league, options} = fixture(t);
  const result = await runTournament({...options, workerFactory: fakeFactory((worker, op) => {
    if (op !== 'step') return;
    worker.state = {...initial(), tick: 1009, score: [20, 20], falls: [4, 4], timeRemaining: 0,
      terminal: 1, roundResult: 3, winner: -1, completedRounds: 1};
    return {state: worker.state, rounds: [worker.state]};
  })});
  assert.ok(result.matches.every(match => match.status === 'completed' && match.durationMs === 20180));
  assert.ok(result.matches.every(match => match.winnerPolicyId === null));
  assert.equal(result.roundSeconds, 20); assert.equal(result.resetBudgetSeconds, 20);
  assert.equal(result.maximumSimulatedDurationMs, 40020);
  assert.equal(Object.values(league.snapshot().pairs)[0].maxDurationMs, 40020);
});

test('over-budget terminals and nonterminals remain invalid with observed elapsed time', async t => {
  const {options} = fixture(t);
  const result = await runTournament({...options, workerFactory: fakeFactory((worker, op, args, leg) => {
    if (op !== 'step') return;
    worker.state = {...initial(), tick: 2002, score: [5, 0], timeRemaining: leg ? 1 : 0,
      terminal: leg ? 0 : 1, roundResult: leg ? 0 : 1, winner: leg ? -1 : 0, completedRounds: leg ? 0 : 1};
    return {state: worker.state, rounds: leg ? [] : [worker.state]};
  })});
  assert.ok(result.matches.every(match => match.status === 'engine_error' && match.durationMs === 40040));
  assert.ok(result.matches.every(match => match.winnerPolicyId === null));
});

test('reset-pause allowance accepts explicit zero and rejects invalid unbounded values', async t => {
  const {directory, options} = fixture(t);
  const file = path.join(directory, 'worker.private.json');
  for (const round_reset_budget_seconds of [-1, null, '20', 3600.1]) {
    fs.writeFileSync(file, JSON.stringify({round_seconds: 20, round_reset_budget_seconds}));
    await assert.rejects(() => runTournament({...options, workerFactory() { throw new Error('Must not launch'); }}), /round_reset_budget_seconds/);
  }
  fs.writeFileSync(file, JSON.stringify({round_seconds: 20, round_reset_budget_seconds: 0}));
  const result = await runTournament({...options, workerFactory: fakeFactory()});
  assert.equal(result.maximumSimulatedDurationMs, 20020); assert.equal(result.resetBudgetSeconds, 0);
});
