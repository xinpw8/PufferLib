'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {summarizeEval, summarizeTraining, compareObjectives} = require('./metrics.cjs');
const sha = 'a'.repeat(64);

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rek-native-sweep-metrics-'));
  t.after(() => fs.rmSync(dir, {recursive: true, force: true}));
  return dir;
}
function row(overrides = {}) {
  return {backend: 'semantic_cuda', policy_sha256: sha, seed: 419, arena: 0, episode: 0,
    policy_side: 0, score: [10, 4], winner: 0, round_result: 1,
    duration_ticks: 6000, duration_seconds: 120, diagnostic: false,
    round_feature_override: null, ...overrides};
}
function writeRows(dir, rows) {
  const filename = path.join(dir, 'rounds.jsonl');
  fs.writeFileSync(filename, rows.map(value => typeof value === 'string' ? value : JSON.stringify(value)).join('\n') + '\n');
  return filename;
}

test('policy side 0 statistics, sample standard errors and Wilson interval exclude side 1', t => {
  const filename = writeRows(fixture(t), [row(), row({episode: 1, score: [6, 8], winner: 1}),
    row({policy_side: 1, score: [1000, 0], winner: 0})]);
  const result = summarizeEval(filename, sha);
  assert.equal(result.n, 2);
  assert.equal(result.discardedSide1Count, 1);
  assert.equal(result.meanOwnPoints, 8);
  assert.equal(result.meanConcededPoints, 6);
  assert.equal(result.meanMargin, 2);
  assert.equal(result.ownPointsStandardError, 2);
  assert.equal(result.marginStandardError, 4);
  assert.deepEqual([result.wins, result.losses, result.draws], [1, 1, 0]);
  assert.ok(Math.abs(result.winWilson95.lower - .0945312057) < 1e-9);
  assert.ok(Math.abs(result.winWilson95.upper - .9054687943) < 1e-9);
  assert.equal(result.actions, null);
  assert.equal(result.failureBits, null);
  assert.equal(result.fullMatchResults, null);
});
test('singleton uncertainty is unknown and draws are not wins', t => {
  const result = summarizeEval(writeRows(fixture(t), [row({score: [4, 4], winner: -1, round_result: 3})]));
  assert.equal(result.ownPointsStandardError, null);
  assert.equal(result.marginStandardError, null);
  assert.equal(result.winRate, 0);
  assert.equal(result.draws, 1);
});

test('malformed JSON, nonfinite score and incomplete records are rejected', t => {
  const dir = fixture(t);
  for (const invalid of ['{broken', '{"score":[NaN,2]}', row({score: [null, 2]}),
    JSON.stringify(row()).replace('"score":[10,4]', '"score":[1e309,4]'),
    row({duration_ticks: 0}), row({round_result: 0}), row({diagnostic: true}),
    row({round_feature_override: 7}), row({winner: -1}), row({failure_bits: 1})]) {
    assert.throws(() => summarizeEval(writeRows(dir, [invalid])));
  }
});

test('side 1 records must validate before being discarded', t => {
  assert.throws(() => summarizeEval(writeRows(fixture(t), [row(), row({policy_side: 1, score: [null, 2]})])), /score/);
});

test('missing side 0, mismatched SHA and duplicate rounds fail', t => {
  const dir = fixture(t);
  assert.throws(() => summarizeEval(writeRows(dir, [row({policy_side: 1})])), /No complete policy_side=0/);
  assert.throws(() => summarizeEval(writeRows(dir, [row()]), 'b'.repeat(64)), /SHA256 mismatch/);
  assert.throws(() => summarizeEval(writeRows(dir, [row(), row({episode: 1, policy_sha256: 'b'.repeat(64)})])), /mixed checkpoint/);
  assert.throws(() => summarizeEval(writeRows(dir, [row(), row()])), /duplicate/);
});

function writeTraining(dir, uptime = '1m 02s 345ms') {
  const rounds = {completed_rounds: 10, fighter0_wins: 6, fighter1_wins: 3, ties: 1,
    redos: 0, unclassified: 0, failure_bits: 0};
  fs.writeFileSync(path.join(dir, 'stdout.txt'),
    '│ Steps 8.2K      Env 0ms 0% value 0.0 │\n│ Uptime 15ms      Model 1ms 1% kl 0.0 │\n' +
    `│ Steps 8.4M      Env 0ms 0% value 0.0 │\n│ Uptime ${uptime}      Model 1s 002ms 20% kl 0.0 │\n` +
    `native5_round_summary=${JSON.stringify(rounds)}\n`);
  fs.writeFileSync(path.join(dir, 'process-timing.txt'), 'Elapsed (wall clock) time (h:mm:ss or m:ss): 1:05.50\n');
}

test('training uses final console uptime instead of stale native INI', t => {
  const dir = fixture(t);
  writeTraining(dir);
  fs.mkdirSync(path.join(dir, 'logs', 'rek_native5'), {recursive: true});
  fs.writeFileSync(path.join(dir, 'logs', 'rek_native5', 'run.ini'),
    '[metrics]\nagent_steps = 8192, 4194304\nuptime = .015, 30\nperf/train = .1, 2\nperf/rollout = .4, 8\n');
  const result = summarizeTraining(dir, 8388608);
  assert.equal(result.trainingLoopSeconds, 62.345);
  assert.equal(result.trainingSps, 8388608 / 62.345);
  assert.equal(result.processWallSeconds, 65.5);
  assert.equal(result.nativeIni.stale, true);
  assert.equal(result.nativeIni.trainFractionOfTimedComponents, .2);
  assert.equal(result.requiresNativeExitCheck, true);
});

test('training parses millisecond and long native durations and rejects incomplete progress', t => {
  const dir = fixture(t);
  for (const [text, seconds] of [['425ms', .425], ['36s 167ms', 36.167], ['0d 1h 2m 3s', 3723]]) {
    writeTraining(dir, text);
    assert.equal(summarizeTraining(dir, 8388608).trainingLoopSeconds, seconds);
  }
  assert.throws(() => summarizeTraining(dir, 16777216), /Steps/);
  writeTraining(dir, 'unknown');
  assert.throws(() => summarizeTraining(dir, 8388608), /Uptime/);
});

test('score improvement needs nonregressing margin and round win rate', t => {
  const dir = fixture(t);
  const baseline = summarizeEval(writeRows(dir, [row()]));
  const higherScoreButWorseMargin = {...baseline, meanOwnPoints: 11, meanMargin: 5};
  assert.equal(compareObjectives(higherScoreButWorseMargin, baseline).scoreImprovedWithGuards, false);
  const higherScoreButLowerWinRate = {...baseline, meanOwnPoints: 11, winRate: .5};
  assert.equal(compareObjectives(higherScoreButLowerWinRate, baseline).scoreImprovedWithGuards, false);
  const improvement = compareObjectives({...baseline, meanOwnPoints: 11, meanMargin: 7}, baseline);
  assert.equal(improvement.scoreImprovedWithGuards, true);
  assert.equal(improvement.fullMatchWinComparison, null);
});
