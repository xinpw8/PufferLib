'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {League, hashFile, winInterval} = require('./league.cjs');

const CONFIG = 'a'.repeat(64);
const OTHER = 'b'.repeat(64);
function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'rek-league-test-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const league = new League({file: path.join(directory, 'league.private.json')});
  return {league, directory};
}
function scripted(league, id, backend = 'mujoco', configHash = CONFIG) {
  return league.registerPolicy({backend, id, configHash, kind: 'scripted', label: id,
    scriptedVersion: 'candidate-dummy-v1'});
}
function pair(league, id = 'pair0', seed = 1, backend = 'mujoco') {
  return league.schedulePair({id, backend, configHash: CONFIG, policyA: 'a', policyB: 'b', seed, maxDurationMs: 1000});
}
function finish(league, match, overrides = {}) {
  league.startMatch(match.id);
  return league.finishMatch({id: match.id, backend: match.backend, configHash: match.configHash,
    players: match.players, seed: match.seed, status: 'completed', scores: [5, 0],
    durationMs: 1000, reason: 'time_limit', ...overrides});
}

test('registers native checkpoints with verified identity and immutable policy IDs', t => {
  const {league, directory} = fixture(t);
  const file = path.join(directory, 'weights.bin');
  fs.writeFileSync(file, Buffer.from([0, 1, 2, 3]));
  const spec = {backend: 'mujoco', id: 'trained-001', configHash: CONFIG, kind: 'trained', label: 'MuJoCo step 8192',
    checkpoint: {path: file, sha256: hashFile(file), format: 'pufferlib5-native-bf16-v1', trainingSteps: 8192,
      model: {architecture: 'mingru', hiddenSize: 256, layers: 2, observations: 223, actions: 33}}};
  const policy = league.registerPolicy(spec);
  assert.equal(policy.checkpoint.trainingSteps, 8192);
  assert.equal(league.verifyCheckpoint({backend: 'mujoco', id: policy.id}), true);
  assert.throws(() => league.registerPolicy(spec), /already registered/);
  assert.throws(() => league.registerPolicy({...spec, id: 'invalid', checkpoint: {...spec.checkpoint, sha256: OTHER}}), /SHA256 mismatch/);
  fs.writeFileSync(file, Buffer.from([9, 9]));
  assert.throws(() => league.verifyCheckpoint({backend: 'mujoco', id: policy.id}), /changed since registration/);
});

test('rejects absent or misleading trained metadata and baseline checkpoints', t => {
  const {league} = fixture(t);
  assert.throws(() => league.registerPolicy({backend: 'mujoco', id: 'a', label: 'A', configHash: CONFIG, kind: 'trained'}), /absolute path/);
  assert.throws(() => league.registerPolicy({backend: 'mujoco', id: 'a', label: 'A', configHash: CONFIG, kind: 'scripted'}), /implementation version/);
  assert.throws(() => league.registerPolicy({backend: 'mujoco', id: 'a', label: 'A', configHash: CONFIG,
    kind: 'scripted', scriptedVersion: 'v1', checkpoint: {path: 'whatever'}}), /cannot claim/);
});

test('pairs reverse sides and rank only completed valid paired legs', t => {
  const {league} = fixture(t);
  scripted(league, 'a'); scripted(league, 'b');
  const scheduled = pair(league);
  assert.deepEqual(scheduled.matches.map(match => match.players), [['a', 'b'], ['b', 'a']]);
  assert.equal(league.opponentOptions({backend: 'mujoco', configHash: CONFIG})[0].strengthStatus, 'untested');
  finish(league, scheduled.matches[0]);
  let a = league.standings({backend: 'mujoco', configHash: CONFIG}).find(row => row.id === 'a');
  assert.equal(a.total.wins, 1);
  assert.equal(a.paired.games, 0);
  assert.equal(a.rank, null);
  finish(league, scheduled.matches[1], {scores: [0, 8]});
  const standings = league.standings({backend: 'mujoco', configHash: CONFIG});
  [a] = standings;
  assert.equal(a.id, 'a'); assert.equal(a.rank, 1);
  assert.equal(a.total.wins, 2); assert.equal(a.total.pointsFor, 13);
  assert.equal(a.paired.games, 2); assert.equal(a.leftGames, 1); assert.equal(a.rightGames, 1);
  assert.equal(a.headToHead.b.wins, 2);
  assert.equal(a.strengthStatus, 'provisional');
  assert.ok(a.total.winRate95[0] > 0 && a.total.winRate95[0] < 1);
  assert.match(league.opponentOptions({backend: 'mujoco', configHash: CONFIG})[0].label, /100.0% wins, 2W\/0D\/0L/);
});

test('draws and independent backends retain separate win/loss evidence', t => {
  const {league} = fixture(t);
  for (const backend of ['mujoco', 'puffysics']) { scripted(league, 'a', backend); scripted(league, 'b', backend); }
  const mujoco = pair(league);
  mujoco.matches.forEach(match => finish(league, match, {scores: [5, 5]}));
  const puff = pair(league, 'puff0', 1, 'puffysics');
  puff.matches.forEach(match => finish(league, match, {scores: match.leg ? [0, 1] : [1, 0]}));
  const mujocoA = league.standings({backend: 'mujoco', configHash: CONFIG}).find(row => row.id === 'a');
  assert.equal(mujocoA.total.draws, 2); assert.equal(mujocoA.total.winRate, 0); assert.equal(mujocoA.total.scoreRate, 0.5);
  assert.equal(league.standings({backend: 'puffysics', configHash: CONFIG})[0].total.wins, 2);
});

test('policy quit, disconnect, timeout and crash produce losses despite a leading score', t => {
  const {league} = fixture(t);
  scripted(league, 'a'); scripted(league, 'b');
  const reasons = ['policy_quit', 'policy_disconnect', 'policy_timeout', 'policy_crash', 'illegal_action'];
  reasons.forEach((reason, index) => {
    const scheduled = pair(league, `p${index}`, index);
    scheduled.matches.forEach(match => finish(league, match, {status: 'forfeit', reason, loserPolicyId: 'a',
      scores: match.leg ? [0, 100] : [100, 0], durationMs: 15}));
  });
  const rows = league.standings({backend: 'mujoco', configHash: CONFIG});
  assert.equal(rows[0].id, 'b');
  const a = rows.find(row => row.id === 'a');
  assert.equal(a.total.losses, 10); assert.equal(a.total.forfeits, 10); assert.equal(a.paired.wins, 0);
});

test('engine failures are invalid trials without opponent wins', t => {
  const {league} = fixture(t);
  scripted(league, 'a'); scripted(league, 'b');
  const scheduled = pair(league);
  finish(league, scheduled.matches[0], {status: 'engine_error', reason: 'physics_nonfinite', scores: [100, 0]});
  finish(league, scheduled.matches[1], {scores: [0, 8]});
  const rows = league.standings({backend: 'mujoco', configHash: CONFIG});
  assert.equal(rows[0].invalidTrials, 1);
  assert.ok(rows.every(row => row.paired.games === 0 && row.rank === null));
  assert.equal(rows.find(row => row.id === 'a').total.wins, 1);
  assert.equal(rows.find(row => row.id === 'b').total.wins, 0);
});

test('duplicate fixtures/results and result identity mismatches are rejected', t => {
  const {league} = fixture(t);
  scripted(league, 'a'); scripted(league, 'b');
  const scheduled = pair(league);
  assert.throws(() => pair(league), /Duplicate pair/);
  assert.throws(() => pair(league, 'different'), /Duplicate backend/);
  const match = scheduled.matches[0];
  league.startMatch(match.id);
  const result = {...match, status: 'completed', reason: 'time_limit', durationMs: 1000, scores: [1, 0]};
  assert.throws(() => league.finishMatch({...result, backend: 'puffysics'}), /backend\/config mismatch/);
  assert.throws(() => league.finishMatch({...result, configHash: OTHER}), /backend\/config mismatch/);
  assert.throws(() => league.finishMatch({...result, players: ['b', 'a']}), /seed\/side assignment/);
  assert.throws(() => league.finishMatch({...result, seed: 12}), /seed\/side assignment/);
  assert.throws(() => league.finishMatch({...result, durationMs: 1001}), /time limit/);
  assert.throws(() => league.finishMatch({...result, winnerPolicyId: 'b', scores: [NaN, 0]}), /Invalid score/);
  const accepted = league.finishMatch({...result, winnerPolicyId: 'b'});
  assert.equal(accepted.winnerPolicyId, 'a'); // Never trust a supplied winner.
  assert.throws(() => league.finishMatch(result), /duplicate outcomes/);
  assert.equal(league.standings({backend: 'mujoco', configHash: CONFIG})[0].total.games, 1);
});

test('incompatible policy configs cannot play and incomplete matches cannot gain wins', t => {
  const {league} = fixture(t);
  scripted(league, 'a'); scripted(league, 'b', 'mujoco', OTHER);
  assert.throws(() => pair(league), /backend\/config mismatch/);
  scripted(league, 'c');
  const scheduled = league.schedulePair({id: 'ac', backend: 'mujoco', configHash: CONFIG, policyA: 'a', policyB: 'c', seed: 1, maxDurationMs: 1000});
  league.startMatch(scheduled.matches[0].id);
  const row = league.standings({backend: 'mujoco', configHash: CONFIG}).find(value => value.id === 'a');
  assert.equal(row.pending, 2); assert.equal(row.total.games, 0); assert.equal(row.rank, null);
});

test('atomic persisted updates survive reopen and reject another writer', t => {
  const {league} = fixture(t);
  scripted(league, 'a');
  const another = new League({file: league.file});
  scripted(another, 'b');
  assert.equal(Object.keys(league.snapshot().policies).length, 2);
  const before = fs.readFileSync(league.file, 'utf8');
  fs.writeFileSync(league.lock, 'active writer');
  assert.throws(() => scripted(league, 'c'), /writer already active/);
  assert.equal(fs.readFileSync(league.file, 'utf8'), before);
  fs.unlinkSync(league.lock);
  assert.throws(() => pair(league, 'bad', -1), /Invalid seed/);
  assert.equal(fs.readFileSync(league.file, 'utf8'), before);
  assert.equal(fs.existsSync(league.lock), false);
});

test('confidence and display never invent strength for untested policies', t => {
  const {league} = fixture(t);
  scripted(league, 'a');
  const option = league.opponentOptions({backend: 'mujoco', configHash: CONFIG})[0];
  assert.equal(option.rank, null); assert.equal(option.stats.winRate, null);
  assert.match(option.label, /unranked \| untested \| no matches/);
  assert.deepEqual(winInterval(0, 0), [0, 1]);
  assert.ok(winInterval(100, 100)[0] < 1);
  assert.equal(Object.hasOwn(option, 'path'), false);
});
