'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const [inputPath, checkpoint, checkpointSha, seedText, out] = process.argv.slice(2);
assert(inputPath && checkpoint && /^[0-9a-f]{64}$/.test(checkpointSha ?? '') && out);
assert(/^\d+$/.test(seedText ?? '') && Number.isSafeInteger(Number(seedText)));
const stage = __dirname;
const oldWorker = '/home/spark-advantage/rek-training/semantic-fast-20260914-v1/live-policy-20260915/build-r1/live-policy-worker';
const newWorker = path.join(stage, 'worker-build/live-policy-worker');
const jointMask = path.join(stage, 'zero-joints.bin');
const allOnes = path.join(stage, 'all-ones.bin');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
assert.equal(hash(checkpoint), checkpointSha);
assert.equal(hash(jointMask), 'd26c5f2f7d2a7faf14f89a7290189fd8fb45223ff4a548abe7be51061ec4bf85');
const rows = fs.readFileSync(inputPath, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
assert(rows.some(x => x.type === 'step' && !x.terminal));
const isJoint = i => (i >= 13 && i <= 70) || (i >= 99 && i <= 156);
const modify = mode => rows.map(row => row.type !== 'step' ? row : {...row,
  observation: row.observation.map((x, i) => isJoint(i) ? mode === 'zero' ? 0 : x + 100 : x)});
fs.mkdirSync(out);
function run(name, binary, args, data) {
  const started = process.hrtime.bigint();
  const result = spawnSync(binary, args, {input: data.map(x => JSON.stringify(x)).join('\n') + '\n',
    encoding: 'utf8', maxBuffer: 128 * 1024 * 1024, timeout: 120000,
    env: {...process.env, REK_OBSERVATION_SCHEMA: 'rek.native5.scaled_polar_xy.v1'}});
  fs.writeFileSync(path.join(out, name + '.stdout.jsonl'), result.stdout ?? '', {flag: 'wx'});
  fs.writeFileSync(path.join(out, name + '.stderr.txt'), result.stderr ?? '', {flag: 'wx'});
  assert.equal(result.status, 0, name + ': ' + result.stderr);
  const messages = result.stdout.trim().split('\n').map(JSON.parse);
  assert(!messages.some(x => ['error', 'fatal'].includes(x.type)), name);
  const decisions = messages.filter(x => x.type !== 'ready').map(x => ({type: x.type, seq: x.seq,
    round_id: x.round_id, action: x.action, recurrent_reset: x.recurrent_reset, round_changed: x.round_changed}));
  return {messages, decisions, seconds: Number(process.hrtime.bigint() - started) / 1e9};
}
const baseArgs = [checkpoint, checkpointSha, seedText];
const baseline = run('pinned-default', oldWorker, baseArgs, rows);
const current = run('candidate-default', newWorker, [...baseArgs, 'sampled'], rows);
const ones = run('candidate-all-ones', newWorker, [...baseArgs, 'sampled', allOnes], rows);
const zero = run('candidate-zero-joints', newWorker, [...baseArgs, 'sampled', jointMask], rows);
const manual = run('candidate-explicit-zero', newWorker, [...baseArgs, 'sampled'], modify('zero'));
const perturbed = run('candidate-joints-perturbed', newWorker, [...baseArgs, 'sampled', jointMask], modify('perturb'));
assert.deepEqual(current.decisions, baseline.decisions, 'default worker changed sampled sequence');
assert.deepEqual(ones.decisions, baseline.decisions, 'all-ones worker changed sampled sequence');
assert.deepEqual(zero.decisions, manual.decisions, 'mask differs from explicitly zeroed inputs');
assert.deepEqual(zero.decisions, perturbed.decisions, 'excluded joints affect sampled sequence');
for (const result of [zero, perturbed]) {
  assert.equal(result.messages[0].feature_mask_sha256, hash(jointMask));
  assert.equal(result.messages[0].selection, 'sampled');
  assert.equal(result.messages[0].seed, Number(seedText));
}
const summary = {passed: true, native_cuda: true, training: false, game_connection: false,
  schema: 'rek.native5.scaled_polar_xy.v1', seed: Number(seedText), rows: rows.length,
  decisions: baseline.decisions.filter(x => x.type === 'action').length,
  input_sha256: hash(inputPath), checkpoint_sha256: checkpointSha,
  old_worker_sha256: hash(oldWorker), new_worker_sha256: hash(newWorker), feature_mask_sha256: hash(jointMask),
  equalities: ['old=default=all-ones', 'masked=explicit-zero=perturbed-masked'],
  run_seconds: [baseline, current, ones, zero, manual, perturbed].map(x => x.seconds)};
fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify(summary));
