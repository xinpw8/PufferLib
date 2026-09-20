'use strict';

// CPU-only diagnostic. Reads closed trajectories and their frozen behavior replay.
// It does not train, rewrite data, invoke CUDA, or apply a pass/fail performance gate.
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const GAMMA = 0.9998844821426083;
const LAMBDA = 0.9978673240629938;
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const potential = (own, other) => Math.fround((own - other) / Math.fround(5 + Math.abs(own - other)));
const reward = r => Math.fround(Math.fround(r.outcome + Math.fround(r.gamma *
  (r.terminal ? 0 : potential(r.nextOwn, r.nextOther)))) - potential(r.own, r.other));

function decode(d, b) {
  assert(d.length >= 256 && d.subarray(0, 8).toString() === 'REKRL001', 'dataset header');
  assert(b.length >= 128 && b.subarray(0, 8).toString() === 'REKBR001', 'replay header');
  const n = d.readUInt32LE(20);
  assert(n > 0 && n < 1000000 && d.length === 256 + n * 1128, 'dataset shape');
  assert(d.readUInt32LE(8) === 1 && d.readUInt32LE(12) === 223 &&
    d.readUInt32LE(16) === 33 && d.readUInt32LE(24) === 1128, 'dataset schema');
  assert(b.readUInt32LE(8) === 1 && b.readUInt32LE(12) === n &&
    b.readUInt32LE(16) === 152 && b.length === 128 + n * 152, 'replay shape');
  assert(b.subarray(24, 56).toString('hex') === sha(d), 'replay dataset digest');
  const groups = [], seen = new Set();
  for (let i = 0; i < n; ++i) {
    const p = 256 + i * 1128, q = 128 + i * 152;
    const r = {
      i, sequence: d.readUInt32LE(p + 4), weight: d.readFloatLE(p + 16),
      time: d.readDoubleLE(p + 24), dt: d.readDoubleLE(p + 1064),
      gamma: d.readFloatLE(p + 1072), lambda: d.readFloatLE(p + 1076),
      reward: d.readFloatLE(p + 1080), outcome: d.readFloatLE(p + 1084),
      own: d.readInt32LE(p + 1096), other: d.readInt32LE(p + 1100),
      nextOwn: d.readInt32LE(p + 1104), nextOther: d.readInt32LE(p + 1108),
      terminal: d.readUInt32LE(p + 1112), value: b.readFloatLE(q + 12),
    };
    assert(b.readUInt32LE(q) === i && b.readUInt32LE(q + 4) === d.readUInt32LE(p + 12), 'replay row');
    assert(Object.values(r).every(Number.isFinite), 'nonfinite input');
    assert(r.dt > 0 && r.gamma > 0 && r.gamma <= 1 && r.lambda > 0 && r.lambda <= 1, 'discount/time');
    assert((r.weight === 0 || r.weight === 1) && (r.terminal === 0 || r.terminal === 1), 'row flags');
    assert([-1, 0, 1].includes(r.outcome) && (r.terminal || r.outcome === 0), 'outcome');
    assert(Math.abs(r.reward - reward(r)) < 1e-6, 'stored reward contract');
    const fresh = !groups.length || groups.at(-1)[0].sequence !== r.sequence;
    assert(d.readUInt32LE(p + 8) === Number(fresh), 'recurrent boundary');
    if (fresh) {
      assert(!seen.has(r.sequence) && r.time === 0, 'reused round or nonzero start');
      assert(!groups.length || groups.at(-1).at(-1).terminal, 'unclosed prior round');
      seen.add(r.sequence); groups.push([]);
    } else {
      assert(!groups.at(-1).at(-1).terminal, 'interior terminal');
    }
    groups.at(-1).push(r);
  }
  assert(groups.length === d.readUInt32LE(28) && groups.at(-1).at(-1).terminal, 'round count/end');
  return groups;
}

function compute(input, gamma20 = null, lambda20 = null) {
  assert(input.length && input.at(-1).terminal, 'closed round required');
  let nextMC = 0, nextAdvantage = 0, nextValue = 0, terminalReturn = 0;
  const result = [];
  for (let i = input.length - 1; i >= 0; --i) {
    const r = { ...input[i] };
    if (gamma20 !== null) {
      assert(gamma20 > 0 && gamma20 <= 1 && lambda20 > 0 && lambda20 <= 1, 'reference discounts');
      r.gamma = Math.fround(Math.pow(gamma20, r.dt / 0.02));
      r.lambda = Math.fround(Math.pow(lambda20, r.dt / 0.02));
      r.reward = reward(r);
    }
    r.mc = r.reward + (r.terminal ? 0 : r.gamma * nextMC);
    r.advantage = r.reward + (r.terminal ? 0 : r.gamma * nextValue) - r.value +
      (r.terminal ? 0 : r.gamma * r.lambda * nextAdvantage);
    r.lambdaReturn = r.value + r.advantage;
    terminalReturn = r.terminal ? r.outcome : r.gamma * terminalReturn;
    r.telescoped = terminalReturn - potential(r.own, r.other);
    result.push(r);
    nextMC = r.mc; nextAdvantage = r.advantage; nextValue = r.value;
  }
  return result.reverse();
}

function stats(rows) {
  const n = rows.length, mean = fn => rows.reduce((sum, r) => sum + fn(r), 0) / n;
  const advantageMean = mean(r => r.advantage), mcMean = mean(r => r.mc);
  const extrema = fn => rows.reduce((a, r) => [Math.min(a[0], fn(r)), Math.max(a[1], fn(r))], [Infinity, -Infinity]);
  return {
    rows: n, old_value_mean: mean(r => r.value), mc_mean: mcMean,
    lambda_return_mean: mean(r => r.lambdaReturn),
    old_value_vs_mc_mae: mean(r => Math.abs(r.value - r.mc)),
    old_value_vs_mc_rmse: Math.sqrt(mean(r => (r.value - r.mc) ** 2)),
    lambda_return_vs_mc_mae: mean(r => Math.abs(r.lambdaReturn - r.mc)),
    lambda_return_vs_mc_rmse: Math.sqrt(mean(r => (r.lambdaReturn - r.mc) ** 2)),
    old_value_vs_lambda_return_mse: mean(r => r.advantage ** 2),
    advantage_mean: advantageMean,
    advantage_population_std: Math.sqrt(mean(r => (r.advantage - advantageMean) ** 2)),
    advantage_min_max: extrema(r => r.advantage),
    positive_advantages: rows.filter(r => r.advantage > 0).length,
    advantage_sign_disagreements_with_mc_minus_value:
      rows.filter(r => Math.sign(r.advantage) !== Math.sign(r.mc - r.value)).length,
    mc_as_advantage_population_std: Math.sqrt(mean(r => (r.mc - mcMean) ** 2)),
    mc_min_max: extrema(r => r.mc),
    telescoping_max_abs: extrema(r => Math.abs(r.mc - r.telescoped))[1],
  };
}

function report(d, b, manifest = null) {
  const groups = decode(d, b), digest = sha(d);
  if (manifest) assert(manifest.binary_sha256 === digest, 'manifest dataset digest');
  return {
    schema: 'rek.authentic_critic_cpu_reference.v1', dataset_sha256: digest,
    replay_sha256: sha(b), checkpoint_sha256: b.subarray(56, 88).toString('hex'),
    rounding: 'float32 stored/recomputed reward and discounts; float64 backward recurrence before target casts',
    configs: [
      ['stored', null, null], ['baseline_gamma_lambda', GAMMA, LAMBDA], ['baseline_gamma_lambda_1', GAMMA, 1],
    ].map(([name, gamma, lambda]) => {
      const results = groups.map(g => compute(g, gamma, lambda)), all = results.flat();
      return {
        name, reference_gamma_20ms: gamma ?? manifest?.reference_gamma_per_20ms ?? null,
        reference_lambda_20ms: lambda ?? manifest?.reference_lambda_per_20ms ?? null,
        all_rows: stats(all), actor_rows: stats(all.filter(r => r.weight > 0)),
        rounds: results.map(rows => ({
          sequence: rows[0].sequence, outcome: rows.at(-1).outcome,
          duration_seconds: rows.at(-1).time + rows.at(-1).dt,
          opening_value: rows[0].value, opening_mc: rows[0].mc,
          opening_lambda_return: rows[0].lambdaReturn,
          all_rows: stats(rows), actor_rows: stats(rows.filter(r => r.weight > 0)),
        })),
      };
    }),
  };
}

if (require.main === module) {
  const [dataset, replay, manifest, ...extra] = process.argv.slice(2);
  assert(dataset && replay && !extra.length, 'usage: node critic_reference.cjs DATASET REPLAY [MANIFEST]');
  console.log(JSON.stringify(report(fs.readFileSync(dataset), fs.readFileSync(replay),
    manifest ? JSON.parse(fs.readFileSync(manifest, 'utf8')) : null), null, 2));
}
module.exports = { GAMMA, LAMBDA, potential, reward, decode, compute, stats, report };
