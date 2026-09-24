'use strict';
const fs = require('node:fs'), path = require('node:path'), cp = require('node:child_process');
const crypto = require('node:crypto'), os = require('node:os'), assert = require('node:assert/strict');
const p = require('./plan.cjs');
const {summarizeEval, compareObjectives} = require('./metrics.cjs');
const {pairedStatistics} = require('./report.cjs');
const evaluatorSha = 'b3ded3b975eedf42557c9dd5ba0f2b7a94f18f97eb9ec136f403b68783f2a40c';
const runtimeSha = '462a394cb2d11961991ff677ea8bccf2ffe38d6172abb42789c05d43be1c6faf';
const seeds = Array.from({length: 10}, (_, index) => 30001 + index), arenas = 128;
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const put = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', {flag: 'wx', mode: 0o600});
function outputPath(stage, name) {
  assert(/^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$/.test(name), 'Use a fresh direct-child output folder name');
  const target = path.resolve(stage, name);
  assert.equal(path.dirname(target), path.resolve(stage));
  assert(!fs.existsSync(target), 'Holdout output already exists; preserved without changes');
  return target;
}
function native(dir, model, modelSha, seed, env) {
  assert.equal(sha(model), modelSha, 'Checkpoint changed');
  const exe = path.join(p.stage, 'eval-build', 'fast-policy-eval'), runtime = path.join(p.stage, 'runtime.json');
  assert.equal(sha(exe), evaluatorSha, 'Native evaluator pin mismatch');
  assert.equal(sha(runtime), runtimeSha, 'Runtime config pin mismatch');
  const availableKiB = Number(fs.readFileSync('/proc/meminfo', 'utf8').match(/^MemAvailable:\s+(\d+)/m)?.[1]);
  assert(availableKiB >= 64 * 1024 * 1024, 'Less than 64 GiB available');
  fs.mkdirSync(dir, {mode: 0o700});
  const records = path.join(dir, 'rounds.private.jsonl');
  const args = [runtime, model, modelSha, String(arenas), '1', String(seed), 'sampled', 'bf16', records];
  put(path.join(dir, 'command.json'), {executable: exe, executableSha256: evaluatorSha, args, cwd: p.previous + '/build', environment: p.environment, timeoutSeconds: 180, startedUtc: new Date().toISOString(), availableGiB: availableKiB / 1024 ** 2});
  const out = fs.openSync(path.join(dir, 'stdout.txt'), 'wx', 0o600), err = fs.openSync(path.join(dir, 'stderr.txt'), 'wx', 0o600);
  const start = performance.now(); let result;
  try { result = cp.spawnSync('/usr/bin/time', ['-v', '-o', path.join(dir, 'process-timing.txt'), 'timeout', '--signal=TERM', '--kill-after=10s', '180', exe, ...args], {cwd: p.previous + '/build', env, stdio: ['ignore', out, err], timeout: 200000}); }
  finally { fs.closeSync(out); fs.closeSync(err); }
  const receipt = {exitCode: result.status, signal: result.signal, error: result.error?.message ?? null, wallSeconds: (performance.now() - start) / 1000, finishedUtc: new Date().toISOString()};
  put(path.join(dir, 'exit.json'), receipt);
  fs.writeFileSync(path.join(dir, 'exit-code.txt'), String(result.status) + '\n', {flag: 'wx', mode: 0o600});
  assert.equal(receipt.exitCode, 0, 'Native holdout evaluation failed');
  const events = fs.readFileSync(path.join(dir, 'stdout.txt'), 'utf8').split('\n').flatMap(line => {try {return [JSON.parse(line)];} catch {return [];}});
  assert(events.some(row => row.event === 'frozen_policy_evaluation' && row.failure_bits === 0 && row.checkpoint_sha256 === modelSha && row.diagnostic === false && row.python_runtime === false && row.cpu_physics === false), 'Missing matching native completion proof');
  const metrics = summarizeEval(records, modelSha);
  assert.equal(metrics.n, arenas); assert.equal(metrics.discardedSide1Count, arenas);
  return records;
}
function evaluate(output, label, model, modelSha, env) {
  const dir = path.join(output, label); fs.mkdirSync(dir, {mode: 0o700});
  const files = [];
  for (const seed of seeds) {
    process.stdout.write(JSON.stringify({event: 'holdout_seed_start', label, seed, checkpointSha256: modelSha}) + '\n');
    files.push(native(path.join(dir, 'seed-' + seed), model, modelSha, seed, env));
  }
  const combined = path.join(dir, 'rounds.private.jsonl');
  fs.writeFileSync(combined, files.map(file => fs.readFileSync(file, 'utf8')).join(''), {flag: 'wx', mode: 0o600});
  const metrics = summarizeEval(combined, modelSha);
  assert.equal(metrics.n, 1280); assert.equal(metrics.discardedSide1Count, 1280);
  put(path.join(dir, 'metrics.json'), metrics);
  const rows = fs.readFileSync(combined, 'utf8').split(/\r?\n/).filter(line => line.trim()).map(JSON.parse).filter(row => row.policy_side === 0);
  return {metrics, rows};
}
function main() {
  assert.equal(process.argv.length, 5, 'Usage: node final-holdout.cjs NEW_STAGE_SUBFOLDER CHECKPOINT EXPECTED_SHA256');
  assert.equal(os.hostname(), 'spark-4ae3');
  const [, , name, checkpoint, suppliedSha] = process.argv;
  assert(/^[a-f0-9]{64}$/i.test(suppliedSha), 'Invalid candidate SHA256');
  const modelSha = suppliedSha.toLowerCase(), model = fs.realpathSync(checkpoint);
  const stage = fs.realpathSync(p.stage), output = outputPath(stage, name);
  assert.equal(stage, path.resolve(p.stage));
  assert.equal(sha(p.warm), p.warmSha); assert.equal(sha(model), modelSha);
  assert.notEqual(modelSha, p.warmSha, 'Candidate must differ from unchanged baseline');
  const bytes = fs.readFileSync(model); assert.equal(bytes.length, fs.statSync(p.warm).size);
  assert.equal(bytes.length % 4, 0);
  for (let offset = 0; offset < bytes.length; offset += 4) assert(Number.isFinite(bytes.readFloatLE(offset)), 'Nonfinite candidate weights');
  const archivedPlan = JSON.parse(fs.readFileSync(path.join(stage, 'plan.json'), 'utf8'));
  assert.deepEqual(p.environment, archivedPlan.environment, 'Environment changed since sweep preparation');
  assert(![...p.screenSeeds, ...p.confirmationSeeds].some(seed => seeds.includes(seed)), 'Holdout seeds overlap selection seeds');
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith('REK_'))); Object.assign(env, p.environment);
  fs.mkdirSync(output, {mode: 0o700});
  put(path.join(output, 'selection.private.json'), {fixedBeforeEvaluationUtc: new Date().toISOString(), candidate: model, candidateSha256: modelSha, baseline: p.warm, baselineSha256: p.warmSha, evaluatorSha256: evaluatorSha, runtimeSha256: runtimeSha, seeds, arenasPerSeedPerSide: arenas, roundsPerArena: 1, selection: 'sampled', precision: 'bf16', environment: p.environment});
  const baseline = evaluate(output, 'baseline', p.warm, p.warmSha, env);
  const candidate = evaluate(output, 'candidate', model, modelSha, env);
  const paired = pairedStatistics(candidate.rows, baseline.rows);
  assert.equal(paired.n, 1280); assert.equal(paired.evaluationSeedClusters, 10);
  assert.equal(paired.unmatchedCandidateRounds, 0); assert.equal(paired.unmatchedReferenceRounds, 0);
  const summary = {schema: 'rek.native_hparam_final_holdout.public.v1', completedUtc: new Date().toISOString(), evaluatorSha256: evaluatorSha, runtimeSha256: runtimeSha, seeds, arenasPerSeedPerSide: arenas, roundsPerArena: 1, selectionFixedBeforeHoldout: true, baseline: baseline.metrics, candidate: candidate.metrics, comparison: compareObjectives(candidate.metrics, baseline.metrics), paired, nativeExitCodesVerified: true, nativeFailureBits: 0, fullMatchWins: null, livePromotion: false, limitation: 'Independent postselection simulator holdout only. Policy-side-1 rounds are excluded because calibration is asymmetric. Ten-seed cluster-bootstrap intervals remain conditional on this environment and these fixtures; they cannot prove authentic REK or live match wins.'};
  put(path.join(output, 'PUBLIC_SUMMARY.json'), summary);
  process.stdout.write(JSON.stringify({event: 'holdout_complete', comparison: summary.comparison, pairedDifferences: paired.differences, confidenceIntervals95: paired.confidenceIntervals95}) + '\n');
}
if (require.main === module) {try {main();} catch (error) {console.error(error.stack); process.exitCode = 1;}}
module.exports = {outputPath, seeds, evaluatorSha, runtimeSha};
