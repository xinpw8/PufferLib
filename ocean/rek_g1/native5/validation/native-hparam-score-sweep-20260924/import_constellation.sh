#!/usr/bin/env bash
# Prepare a dataset for the existing native Constellation viewer. No GUI launch.
set -euo pipefail
umask 077
[[ $# == 0 ]] || { printf 'Usage: bash %s\n' "$0" >&2; exit 2; }
node <<'NODE'
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const cp = require('node:child_process');
const assert = require('node:assert/strict');
const os = require('node:os');
const stage = '/home/spark-advantage/rek-training/native-hparam-score-sweep-20260924-r1';
const native = '/home/spark-advantage/rek-training/normalized-sweep-20260921-r1/constellation';
const converter = native + '/cache_data';
const viewer = native + '/seethestars';
const binaryPins = {
  [converter]: '53d78b920f98f5e6a6fa009e48712b172378eb702882e9a6cd3bb19b2c0eba4d',
  [viewer]: '13ed2ffef2e52e2ff43a6a058ea33a51cbf3d0bc096ffad7e05094dc44b024c2',
};
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const sha = file => hash(fs.readFileSync(file));
const write = (file, data) => fs.writeFileSync(file, data, {flag: 'wx', mode: 0o600});
const put = (file, data) => write(file, JSON.stringify(data, null, 2) + '\n');
const metricLists = text => {
  const sections = {};
  let section;
  for (const line of text.split(/\r?\n/)) {
    const heading = line.match(/^\s*\[([^\]]+)\]\s*$/);
    if (heading) { section = heading[1]; sections[section] = {}; continue; }
    const item = line.match(/^\s*([^#;=]+?)\s*=\s*(.*?)\s*$/);
    if (section && item) sections[section][item[1].trim()] = item[2];
  }
  return sections;
};
const numericList = (raw, label) => {
  assert.equal(typeof raw, 'string', 'Missing ' + label);
  assert(raw.length, 'Empty ' + label);
  const values = raw.split(',').map(Number);
  assert(values.every(Number.isFinite), 'Nonfinite ' + label);
  return values;
};
assert.equal(os.hostname(), 'spark-4ae3', 'Run this importer on Spark');
assert.equal(fs.realpathSync(stage), stage);
for (const [file, expected] of Object.entries(binaryPins)) assert.equal(sha(file), expected, 'Native binary changed: ' + file);
const planData = fs.readFileSync(stage + '/plan.json');
const screenData = fs.readFileSync(stage + '/screen-results.json');
const plan = JSON.parse(planData);
const screen = JSON.parse(screenData);
assert.equal(plan.stage, stage);
assert.equal(plan.arms.length, 25, 'Expected the planned 25-arm screen');
assert.equal(screen.results.length, 25, 'Screen must finish before import');
const ids = new Set(plan.arms.map(arm => arm.id));
assert.equal(ids.size, 25);
assert.equal(new Set(screen.results.map(row => row.arm.id)).size, 25);
for (const result of screen.results) assert(ids.has(result.arm.id), 'Unknown result arm');
const successful = screen.results.filter(row => row.error === null && row.receipt.exitCode === 0);
const excluded = screen.results.filter(row => row.error !== null || row.receipt.exitCode !== 0)
  .map(row => ({id: row.arm.id, exitCode: row.receipt.exitCode, error: row.error,
    reason: 'Training result did not satisfy error === null and receipt.exitCode === 0'}));
assert(successful.length > 0, 'No successful completed training arms to import');
const group = 'rek_native5_score_screen';
const arms = [];
for (const id of successful.map(row => row.arm.id).sort()) {
  assert(/^(?:control|arm-\d{2})$/.test(id), 'Unexpected arm identifier');
  const armDir = stage + '/screen/' + id;
  assert.equal(fs.readFileSync(armDir + '/exit-code.txt', 'utf8').trim(), '0', 'Missing successful exit receipt: ' + id);
  const logDir = armDir + '/logs/rek_native5';
  assert.equal(fs.realpathSync(logDir), logDir, 'Unexpected log symlink');
  const filenames = fs.readdirSync(logDir).filter(name => name.endsWith('.ini')).sort();
  assert.equal(filenames.length, 1, 'Expected one native training INI for ' + id);
  const source = path.join(logDir, filenames[0]);
  assert(fs.lstatSync(source).isFile(), 'Expected regular log file');
  const data = fs.readFileSync(source);
  const sourceSha256 = hash(data);
  const raw = metricLists(data.toString('utf8')).metrics;
  assert(raw, 'Missing native metrics: ' + id);
  const metrics = Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, numericList(value, id + ':' + key)]));
  for (const key of ['agent_steps', 'uptime', 'env/score', 'env/wins', 'env/n']) assert(metrics[key], 'Missing ' + key + ': ' + id);
  const points = metrics.agent_steps.length;
  assert(Object.values(metrics).every(values => values.length === points), 'Unequal metric lengths: ' + id);
  assert.equal(sha(source), sourceSha256, 'Training log changed during preflight');
  arms.push({id, source, data, sourceSha256, metrics, points, commandSha256: sha(armDir + '/command.json')});
}
const datasets = stage + '/constellation/datasets';
fs.mkdirSync(datasets, {recursive: true, mode: 0o700});
const stamp = new Date().toISOString().replace(/[-:.]/g, '');
const dataset = fs.mkdtempSync(datasets + '/import-' + stamp + '-');
const logs = dataset + '/logs/' + group;
fs.mkdirSync(logs, {recursive: true, mode: 0o700});
fs.mkdirSync(dataset + '/resources/constellation', {recursive: true, mode: 0o700});
fs.cpSync(native + '/resources/shared', dataset + '/resources/shared', {recursive: true, errorOnExist: true, force: false});
for (const name of fs.readdirSync(native + '/resources/constellation').sort()) {
  if (/\.(?:fs|vs|rgs)$/.test(name)) fs.copyFileSync(native + '/resources/constellation/' + name, dataset + '/resources/constellation/' + name, fs.constants.COPYFILE_EXCL);
}
const records = [];
let cacheRow = 0;
for (let index = 0; index < arms.length; index++) {
  const arm = arms[index];
  const destination = logs + '/' + String(index + 1).padStart(3, '0') + '-' + arm.id + '.ini';
  write(destination, arm.data);
  assert.equal(sha(destination), arm.sourceSha256, 'Copied log differs');
  assert.equal(sha(arm.source), arm.sourceSha256, 'Source log changed');
  records.push({id: arm.id, source: arm.source, importedLog: destination, sourceSha256: arm.sourceSha256,
    commandSha256: arm.commandSha256, cachedRows: arm.points, firstCacheRow: cacheRow, lastCacheRow: cacheRow + arm.points - 1});
  cacheRow += arm.points;
}
write(dataset + '/plan.json', planData);
write(dataset + '/screen-results.json', screenData);
const stdout = fs.openSync(dataset + '/cache.stdout.txt', 'wx', 0o600);
const stderr = fs.openSync(dataset + '/cache.stderr.txt', 'wx', 0o600);
let converted;
try { converted = cp.spawnSync(converter, ['--full'], {cwd: dataset, stdio: ['ignore', stdout, stderr], timeout: 60000}); }
finally { fs.closeSync(stdout); fs.closeSync(stderr); }
put(dataset + '/cache-exit.json', {exitCode: converted.status, signal: converted.signal, error: converted.error?.message ?? null});
assert.equal(converted.status, 0, 'Native cache conversion failed');
const cacheFile = dataset + '/resources/constellation/experiments.ini';
const cache = metricLists(fs.readFileSync(cacheFile, 'utf8'));
assert.deepEqual(Object.keys(cache), [group], 'Unexpected cache groups');
let comparedValues = 0;
const expected = {};
for (const arm of arms) {
  for (const [key, values] of Object.entries(arm.metrics)) {
    if (key.includes('loss')) continue; // This is the existing native converter's rule.
    (expected[key] ||= []).push(...values);
  }
}
for (const [key, values] of Object.entries(expected)) {
  const actual = numericList(cache[group][key], 'cached ' + key);
  assert.equal(actual.length, values.length, 'Native cache dropped rows for ' + key);
  assert.equal(values.length, cacheRow, 'Input metric missing from an arm: ' + key);
  for (let i = 0; i < values.length; i++) {
    const wanted = key === 'agent_steps' ? Math.fround(Math.fround(values[i]) / 1e6) : Math.fround(values[i]);
    assert(Math.abs(actual[i] - wanted) <= .000005 * Math.max(1, Math.abs(wanted)), 'Cache metric changed: ' + key + ' row ' + i);
    comparedValues++;
  }
}
for (const record of records) {
  assert.equal(sha(record.source), record.sourceSha256, 'Original log changed after import');
  assert.equal(sha(record.importedLog), record.sourceSha256, 'Copied log changed after import');
}
assert.equal(sha(stage + '/plan.json'), hash(planData), 'Plan changed during import');
assert.equal(sha(stage + '/screen-results.json'), hash(screenData), 'Screen results changed during import');
for (const [file, expectedHash] of Object.entries(binaryPins)) assert.equal(sha(file), expectedHash, 'Native binary changed during import');
const provenance = {createdUtc: new Date().toISOString(), dataset, inputRoot: stage + '/screen', arms: records.length,
  attemptedArms: screen.results.length, excluded,
  group, converter, converterArguments: ['--full'], binaryPins, cacheFile, cacheSha256: sha(cacheFile), cachedPoints: cacheRow,
  metricMapping: 'None. env/score and env/wins retain their original independent meanings.',
  originalEnvironment: 'rek_native5', metadata: {planSha256: hash(planData), screenResultsSha256: hash(screenData)}, records};
put(dataset + '/provenance.json', provenance);
put(dataset + '/verification.json', {verifiedUtc: new Date().toISOString(), originalLogsUnchanged: true, copiedLogsByteIdentical: true,
  nativeMetricValuesCompared: comparedValues, nativePrecision: 'C float and %.6g', cachedPoints: cacheRow,
  stepUnits: 'agent_steps and train/total_timesteps in millions', omittedNativeKeys: 'The existing converter excludes metric names containing loss.'});
write(dataset + '/viewer-command.txt', 'cd ' + dataset + '\nDISPLAY=:98 ' + viewer + '\n');
write(dataset + '/README.md', '# Native Constellation: 25-arm REK score screen\n\n' +
  arms.length + ' successful completed training INIs were copied without changing their bytes from 25 attempted arms. Excluded arms and failure reasons, original paths, hashes, and cache row ranges are in provenance.json; metric checks are in verification.json.\n\n' +
  'The existing native ARM64 cache_data converter ran with --full. The existing native seethestars viewer was not launched. Node performs file checks and provenance bookkeeping; the converter and viewer do not use Python.\n\n' +
  '`env/score` is mean own awarded points in completed training rounds. `env/wins` remains the original round-win metric. No win value is remapped into score. Training curves follow a changing policy; held-out side-0 evaluations and ranking are preserved separately in screen-results.json. Simulator rounds do not establish authentic full-match wins.\n\n' +
  'The native converter uses float storage and %.6g output, expresses agent_steps and train/total_timesteps in millions, and omits metric keys containing loss. It retains all imported rows with --full.\n\n' +
  'For the viewer, use env/score on Y; use train/learning_rate, train/ent_coef, or train/horizon for hyperparameter comparisons, or agent_steps for training progress. The upstream default env/perf is absent from these logs.\n\n' +
  'viewer-command.txt records the isolated Spark display command without executing it. Before any GUI launch, its owner must check that display :98 is available for the viewer.\n');
console.log(JSON.stringify({event: 'constellation_import_verified', dataset, arms: arms.length, cachedPoints: cacheRow,
  attemptedArms: screen.results.length, excluded, cacheSha256: provenance.cacheSha256, viewerLaunched: false, pythonRuntime: false}, null, 2));
NODE
