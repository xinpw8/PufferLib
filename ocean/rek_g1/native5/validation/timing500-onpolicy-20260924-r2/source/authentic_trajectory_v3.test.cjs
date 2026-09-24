'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const {pack, HEADER, sha} = require('./authentic_trajectory_v3.cjs');
const base = require('./authentic_trajectory_data.cjs');
const pins = {worker: {sha256: '1'.repeat(64)}, checkpoint: {sha256: '2'.repeat(64)}, native_object: {sha256: '3'.repeat(64)}};
function row(seed = 601) {
  return {sequence: 0, reset: 1, action: 2, policyWeight: 1, valueWeight: 1,
    obs: Array(223).fill(0), mask: Array(33).fill(1), time: 0, nextTime: 0.04, dt: 0.04,
    gamma: Math.fround(0.999 ** 2), lambda: Math.fround(0.995 ** 2),
    own: 0, opponent: 0, nextOwn: 5, nextOpponent: 0, sourceSeq: 2, nextSourceSeq: 5,
    terminal: true, applied: true, outcome: 1, reward: 1, behaviorSeed: seed};
}
test('v3 identity and five actual seeds are embedded without changing observation/reward layout', () => {
  const rows = [601,602,603,604,605].map((s,i) => ({...row(s),sequence:i}));
  const identity = Buffer.from('{"test":"identity"}\n'), mask = Buffer.alloc(223,1), b = pack(rows,5,identity,mask,pins);
  assert.equal(b.length, HEADER+5*base.ROW); assert.equal(b.toString('ascii',0,8),'REKRL003');
  assert.equal(b.readUInt32LE(8),3); assert.equal(b.subarray(256,288).toString('hex'),sha(identity));
  assert.equal(b.subarray(288,320).toString('hex'),pins.worker.sha256);
  assert.equal(b.subarray(320,352).toString('hex'),pins.checkpoint.sha256);
  assert.equal(b.subarray(352,384).toString('hex'),pins.native_object.sha256);
  rows.forEach((r,i) => {
    const at=HEADER+i*base.ROW, old=base.pack([r],1).subarray(base.HEADER);
    assert.equal(b.readUInt32LE(at+20),r.behaviorSeed); assert.equal(b.readUInt32LE(at+1124),0);
    const restored=Buffer.from(b.subarray(at,at+base.ROW)); restored.fill(0,20,24); restored.fill(0,1124,1128);
    assert.deepEqual(restored,old);
  });
});
test('v3 preserves raw observations with explicit binary feature mask', () => {
  const r=row(); r.obs[13]=42; const mask=Buffer.alloc(223,1); mask[13]=0;
  const b=pack([r],1,Buffer.from('identity'),mask,pins);
  assert.equal(b[32+13],0); assert.equal(b.readFloatLE(HEADER+32+13*4),42);
});
test('v3 round seed accepts exact safe integers including high word and rejects invalid seeds', () => {
  const seed=2**40+601, b=pack([row(seed)],1,Buffer.from('identity'),Buffer.alloc(223,1),pins);
  assert.equal(b.readUInt32LE(HEADER+20),601); assert.equal(b.readUInt32LE(HEADER+1124),256);
  for(const seed of [-1,1.2,NaN,2**53]) assert.throws(() => pack([row(seed)],1,Buffer.from('x'),Buffer.alloc(223,1),pins));
});
test('v3 rejects malformed feature masks and does not weaken reward validation', () => {
  for(const mask of [Buffer.alloc(222,1),Buffer.alloc(223,2)]) assert.throws(() => pack([row()],1,Buffer.from('x'),mask,pins));
  const r=row(); r.reward=5; assert.throws(() => pack([r],1,Buffer.from('x'),Buffer.alloc(223,1),pins),/reward/);
});
module.exports={row,pins};
