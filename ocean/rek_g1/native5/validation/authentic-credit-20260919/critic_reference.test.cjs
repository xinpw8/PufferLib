'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { reward, decode, compute, stats } = require('./critic_reference.cjs');

const close = (a, b, tolerance = 1e-6) => assert(Math.abs(a - b) < tolerance, `${a} != ${b}`);
function fixture() {
  return [
    { sequence: 0, weight: 1, time: 0, dt: 0.02, gamma: 0.9, lambda: 0.8, own: 0, other: 0,
      nextOwn: 5, nextOther: 0, outcome: 0, terminal: 0, value: 10 },
    { sequence: 0, weight: 0, time: 0.02, dt: 0.02, gamma: 0.9, lambda: 0.8, own: 5, other: 0,
      nextOwn: 5, nextOther: 0, outcome: 1, terminal: 1, value: 20 },
  ].map(r => ({ ...r, reward: reward(r) }));
}
test('complete shaped return telescopes, including actor-ineligible final reward', () => {
  const rows = compute(fixture());
  close(rows[0].mc, 0.9); close(rows[1].mc, 0.5);
  close(rows[0].mc, rows[0].telescoped); close(rows[1].mc, rows[1].telescoped);
  assert.equal(stats(rows.filter(r => r.weight)).rows, 1);
});
test('lambda return retains inaccurate future critic while lambda one removes it', () => {
  const rows = compute(fixture());
  close(rows[0].advantage, -5.59); close(rows[0].lambdaReturn, 4.41);
  const mc = compute(fixture(), 0.9, 1);
  for (const r of mc) close(r.lambdaReturn, r.mc, 1e-12);
  assert.notEqual(mc[0].advantage, mc[0].mc); // MC - learned V is not the zero-V control.
});
test('population summary and terminal reset are deterministic', () => {
  const rows = compute(fixture());
  const s = stats(rows), expectedMean = (rows[0].advantage + rows[1].advantage) / 2;
  close(s.advantage_mean, expectedMean, 1e-12);
  close(s.advantage_population_std, Math.abs(rows[0].advantage - rows[1].advantage) / 2, 1e-12);
  close(rows[1].advantage, 0.5 - 20);
});
test('unclosed rounds and invalid binary headers are rejected', () => {
  assert.throws(() => compute(fixture().slice(0, 1)), /closed round/);
  assert.throws(() => decode(Buffer.alloc(0), Buffer.alloc(0)), /dataset header/);
});
