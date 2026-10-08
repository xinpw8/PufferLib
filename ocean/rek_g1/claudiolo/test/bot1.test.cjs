'use strict';
// Parity: bot1.cjs must reproduce the recovered C++ header bit-for-bit on a
// 20,000-step randomized trace (phases, binary32 timers, picked moves, RNG
// state and locomotion commands). Skips if no C++ compiler is installed.
const test = require('node:test');
const assert = require('node:assert');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const bot1 = require('../bot1.cjs');

function referenceTrace(steps) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bot1-ref-'));
  const exe = path.join(dir, 'bot1_reference');
  try { execFileSync('g++', ['-std=c++17', '-O1', path.join(__dirname, 'bot1_reference.cpp'), '-o', exe], {stdio: 'pipe'}); }
  catch (e) { return null; }
  return execFileSync(exe, [String(steps)], {maxBuffer: 1 << 28}).toString().trim().split('\n').map(JSON.parse);
}

test('bot1.cjs reproduces native_bot1.cuh exactly', t => {
  const trace = referenceTrace(20000);
  if (!trace) { t.skip('no g++'); return; }
  const s = bot1.newState(12345);
  let moves = 0, phases = new Set();
  for (const r of trace) {
    const input = {distance: r.dist, angleDegrees: r.ang, deltaSeconds: r.dt, timeSeconds: r.t, roundElapsed: r.t,
      punching: !!r.punching, opponentDown: !!r.down, ownRecovery: false, roundActive: !!r.active};
    const out = bot1.update(s, input);
    if (out.move >= 0) { moves++; bot1.attackResult(s, input, r.accepted === 1); }
    assert.strictEqual(out.move, r.move, `move at ${r.n}`);
    assert.strictEqual(out.clearPunching ? 1 : 0, r.clear, `clear at ${r.n}`);
    assert.strictEqual(s.phase, r.phase, `phase at ${r.n}`);
    assert.strictEqual(Math.fround(s.timer), Math.fround(r.timer), `timer at ${r.n}`);
    assert.strictEqual(s.rng.state >>> 0, r.rng >>> 0, `rng at ${r.n}`);
    const c = bot1.locomotion(s, input);
    for (const [k, v] of [['forward', r.f], ['strafe', r.s], ['yaw', r.y]])
      assert.ok(Math.abs(c[k] - v) <= 2e-6, `${k} at ${r.n}: ${c[k]} vs ${v}`);
    phases.add(r.phase);
  }
  assert.ok(moves > 50, 'trace exercised attacks');
  assert.ok(phases.size >= 5, 'trace exercised most phases');
});

test('Bot 1 picks from the side it sees us on', () => {
  const rng = new bot1.XorShift32(99), right = new Set(), left = new Set();
  for (let k = 0; k < 4000; k++) { right.add(bot1.pickAttack(bot1.PRIMARY_LIMB, 10, rng)); left.add(bot1.pickAttack(bot1.PRIMARY_LIMB, -10, rng)); }
  assert.deepStrictEqual([...right].sort((a, b) => a - b), [3, 4, 8, 9, 11]);
  assert.deepStrictEqual([...left].sort((a, b) => a - b), [0, 1, 2, 5, 6, 7, 10, 12, 13, 14, 15, 16]);
});

test('angle convention: positive means target on the robot right', () => {
  const own = {x: 0, y: 0, yaw: 0};
  assert.ok(bot1.angleToOpponentDegrees(own, {x: 1, y: -0.2}) > 0);
  assert.ok(bot1.angleToOpponentDegrees(own, {x: 1, y: 0.2}) < 0);
  assert.ok(bot1.facingYaw(40) < 0 && bot1.facingYaw(-40) > 0 && bot1.facingYaw(10) === 0);
});
