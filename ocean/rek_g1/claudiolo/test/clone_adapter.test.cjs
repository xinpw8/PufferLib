'use strict';
// End-to-end protocol test of adapters/clone.cjs against a surrogate-backed
// double of rek-eval-worker (same JSONL ops, state layout and masks).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {play, frameFromState, commandFor} = require('../adapters/clone.cjs');

test('frame conventions from worker qpos', () => {
  const qpos = Array(72).fill(0);
  qpos.splice(0, 7, -0.9, 0, 0.8, 1, 0, 0, 0);
  qpos.splice(36, 7, 0.9, 0, 0.8, 0, 0, 0, 1); // yaw pi
  const f = frameFromState({tick: 10, phase: 2, terminal: 0, timeRemaining: 100, qpos, score: [1, 2], falls: [0, 0],
    mask: Array(66).fill(1), raw: Array(446).fill(0)}, null, 0, '0');
  assert.ok(Math.abs(f.me.yaw) < 1e-9 && Math.abs(Math.abs(f.opp.yaw) - Math.PI) < 1e-9);
  assert.deepStrictEqual(f.points, [1, 2]);
  assert.strictEqual(f.me.down, false);
  assert.deepStrictEqual(commandFor(10, 21), {forward: -1, strafe: 0, yaw: 1, moveIndex: 1, cancelAction: false});
});

test('Claudiolo completes rounds through the worker protocol', async () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'claudiolo-clone-'));
  const summary = await play({bin: process.execPath, args: [path.join(__dirname, 'fake_worker.cjs')], out, rounds: 2,
    env: {}, maxTicks: 20000});
  assert.strictEqual(summary.rounds, 2);
  const lines = fs.readFileSync(path.join(out, 'rounds.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.strictEqual(lines.length, 2);
  assert.ok(lines.every(r => r.governor.moves > 0), 'Claudiolo struck through the adapter');
});
