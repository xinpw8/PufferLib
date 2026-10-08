'use strict';
// Calibration probes run end to end against the surrogate-backed worker double.
const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const {Worker} = require('../adapters/clone.cjs');
const {probeLocomotion, probeMove, observeBot} = require('../adapters/measure.cjs');

test('measure probes recover the double\'s locomotion and move timing', async () => {
  const w = new Worker(process.execPath, null, {}, [path.join(__dirname, 'fake_worker.cjs')]);
  await w.readyPromise;
  const fwd = await probeLocomotion(w, 'W', {forward: 1});
  assert.ok(Math.abs(fwd.steadyForward - 0.55) < 0.06, `forward ${fwd.steadyForward}`);
  assert.ok(fwd.settleSecondsAfterRelease > 0.1 && fwd.settleSecondsAfterRelease < 1.5, `settle ${fwd.settleSecondsAfterRelease}`);
  const jab = await probeMove(w, 1);
  assert.strictEqual(jab.accepted, 1);
  assert.ok(Math.abs(jab.maskClosedSeconds - (0.54 + 0.1)) < 0.08, `mask closed ${jab.maskClosedSeconds}`);
  const bot = await observeBot(w, 20);
  assert.ok(bot.botMoveStarts > 2, 'Bot 1 attacked while observed');
  w.close();
});
