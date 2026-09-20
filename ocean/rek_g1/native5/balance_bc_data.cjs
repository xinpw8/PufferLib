#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const crypto = require('node:crypto');
const path = require('node:path');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const check = (ok, reason) => { if (!ok) throw new Error(reason); };
const group = action => action === 1 ? 0 : action >= 2 && action <= 15 ? 1 : action >= 16 && action <= 32 ? 2 : -1;

function balance(source) {
  check(Buffer.isBuffer(source) && source.length >= 256, 'truncated header');
  check(source.toString('ascii', 0, 8) === 'REKBC001' && source.readUInt32LE(8) === 1 &&
    source.readUInt32LE(12) === 223 && source.readUInt32LE(16) === 33 && source.readUInt32LE(24) === 1056,
  'unsupported dataset');
  const rows = source.readUInt32LE(20);
  check(rows > 0 && source.length === 256 + rows * 1056, 'dataset byte length mismatch');
  const counts = [0, 0, 0];
  let heldoutLabels = 0;
  for (let row = 0; row < rows; ++row) {
    const p = 256 + row * 1056, split = source.readUInt32LE(p), action = source.readInt32LE(p + 12), weight = source.readFloatLE(p + 16);
    check(split <= 1 && action >= -1 && action <= 32 && Number.isFinite(weight) && weight >= 0, 'invalid row');
    check((action === -1) === (weight === 0), 'label weight disagreement');
    if (!weight) continue;
    if (split === 1) { heldoutLabels++; continue; }
    check(weight === 1, 'requires unweighted original training labels');
    const g = group(action); check(g >= 0, 'labeled action outside approved balance groups');
    counts[g]++;
  }
  check(counts.every(n => n > 0) && heldoutLabels > 0, 'all training groups and heldout labels required');
  const labeled = counts.reduce((a, b) => a + b, 0);
  const weights = counts.map(n => Math.fround(labeled / (3 * n)));
  const output = Buffer.from(source);
  let changed = 0;
  for (let row = 0; row < rows; ++row) {
    const p = 256 + row * 1056;
    if (source.readUInt32LE(p) === 0 && source.readFloatLE(p + 16) > 0) {
      output.writeFloatLE(weights[group(source.readInt32LE(p + 12))], p + 16); changed++;
    } else check(output.subarray(p, p + 1056).equals(source.subarray(p, p + 1056)), 'nontraining row changed');
    // Every feature, target, sequence/reset, support mask, and timestamp is byte-identical.
    check(output.subarray(p, p + 16).equals(source.subarray(p, p + 16)) &&
      output.subarray(p + 20, p + 1056).equals(source.subarray(p + 20, p + 1056)), 'nonweight bytes changed');
  }
  check(output.subarray(0, 256).equals(source.subarray(0, 256)), 'header/mask changed');
  return {output, manifest: {
    schema: 'rek.native_bc.training_group_balance.v1', source_sha256: sha(source), output_sha256: sha(output),
    feature_mask_sha256: sha(source.subarray(32, 255)), rows, changed_training_weights: changed,
    heldout_labels: heldoutLabels, heldout_bytes_unchanged: true, nonweight_bytes_unchanged: true,
    groups: counts.map((count, index) => ({name: ['neutral_action1', 'movement_actions2_15', 'attack_actions16_32'][index],
      training_count: count, weight_float32: weights[index], training_weight_mass: count * weights[index]})),
    normalization: 'equal total group mass derived only from original training counts; overall labeled mean1',
    observed_mean_weight: counts.reduce((sum, count, index) => sum + count * weights[index], 0) / labeled,
  }};
}
function run(sourcePath, expectedHash, outputPath) {
  check(/^[0-9a-f]{64}$/.test(expectedHash), 'expected source SHA256 required');
  const source = fs.readFileSync(sourcePath); check(sha(source) === expectedHash, 'source SHA256 mismatch');
  check(path.resolve(sourcePath) !== path.resolve(outputPath), 'output aliases source');
  const manifestPath = outputPath + '.manifest.json';
  check(!fs.existsSync(outputPath) && !fs.existsSync(manifestPath), 'output/manifest already exists');
  const result = balance(source);
  fs.writeFileSync(outputPath, result.output, {flag: 'wx'});
  check(sha(fs.readFileSync(outputPath)) === result.manifest.output_sha256, 'output readback mismatch');
  check(sha(fs.readFileSync(sourcePath)) === expectedHash, 'source changed during balance');
  fs.writeFileSync(manifestPath, JSON.stringify(result.manifest, null, 2) + '\n', {flag: 'wx'});
  return result.manifest;
}
if (require.main === module) {
  try {
    check(process.argv.length === 5, 'Usage: balance_bc_data.cjs SOURCE EXPECTED_SHA256 NEW_OUTPUT');
    console.log(JSON.stringify(run(...process.argv.slice(2))));
  } catch (error) { console.error(error.message); process.exitCode = 2; }
}
module.exports = {balance, run};
