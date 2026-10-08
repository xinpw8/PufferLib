'use strict';
// Unity -> Claudiolo frame conventions for the authentic bridge telemetry.
const test = require('node:test');
const assert = require('node:assert');
const {LiveFramer, unityHeading} = require('../adapters/live.cjs');
const {perceive} = require('../core.cjs');

const bones = ['pelvis', 'left_wrist_yaw_link', 'right_wrist_yaw_link', 'left_ankle_roll_link', 'right_ankle_roll_link'];
function fighter(pos, quat, hand = [0, 0, 0]) {
  const b = bones.map((n, i) => i === 1 ? [pos[0] + 0.3 + hand[0], pos[1], pos[2] + hand[2]] : [...pos]);
  return {root_position_xyz: pos, root_rotation_xyzw: quat, bone_names: bones, bone_world_positions_xyz: b,
    falling: false, fallen: false, resetting: false, motor_shutdown: false, tilt_angle: 2};
}
function source(t, me, opp, extra = {}) {
  return {local_slot: 0, phase: 1, clock: {unity_time: t}, fighters: [me, opp],
    round: {active: true, time_remaining: 100, duration: 120, number: 1, clean_hits: [3, 4], falls: [0, 1]},
    action_mask: Array(33).fill(true), input: {pending_move: false}, ...extra};
}

test('heading matches the encoder Unity->MuJoCo projection', () => {
  assert.ok(Math.abs(unityHeading([0, 0, 0, 1])) < 1e-12);
  assert.ok(Math.abs(Math.abs(unityHeading([0, 1, 0, 0])) - Math.PI) < 1e-12);
  // Positive Unity yaw about +y is clockwise from above -> negative CCW heading.
  const th = 0.3; assert.ok(Math.abs(unityHeading([0, Math.sin(th / 2), 0, Math.cos(th / 2)]) + th) < 1e-12);
});

test('facing fighters at spawn read as face to face, opponent ahead', () => {
  const fr = new LiveFramer();
  const f = fr.frame(source(10, fighter([-0.9, 0.8, 0], [0, 0, 0, 1]), fighter([0.9, 0.8, 0], [0, 1, 0, 0])));
  const P = perceive(f);
  assert.ok(Math.abs(P.d - 1.8) < 1e-9 && Math.abs(P.myBearing) < 1e-9 && Math.abs(P.oppBearing) < 1e-9);
  assert.deepStrictEqual(f.points, [3, 4]);
  assert.deepStrictEqual(f.falls, [0, 1]);
  assert.strictEqual(f.round.active, true);
});

test('Unity +z is physical left of a robot facing +x', () => {
  const fr = new LiveFramer();
  // Opponent at Unity z=+0.5: with x right, y up, z forward (left-handed),
  // facing +x puts +z on the robot's LEFT.
  const f = fr.frame(source(10, fighter([0, 0.8, 0], [0, 0, 0, 1]), fighter([1, 0.8, 0.5], [0, 1, 0, 0])));
  assert.ok(perceive(f).myBearing > 0);
});

test('limb speed from bone motion relative to the root', () => {
  const fr = new LiveFramer();
  fr.frame(source(10, fighter([0, 0.8, 0], [0, 0, 0, 1]), fighter([0.6, 0.8, 0], [0, 1, 0, 0])));
  const f = fr.frame(source(10.02, fighter([0, 0.8, 0], [0, 0, 0, 1], [0.06, 0, 0]), fighter([0.6, 0.8, 0], [0, 1, 0, 0])));
  assert.ok(Math.abs(f.me.limbSpeed - 3) < 1e-6, `limb ${f.me.limbSpeed}`);
  assert.strictEqual(f.opp.limbSpeed, 0);
});

test('falls come from tilt or the referee count when flags stay false', () => {
  const fr = new LiveFramer();
  const tipped = fighter([0.6, 0.3, 0], [0.7071, 0, 0, 0.7071]); // 90 deg about x
  let f = fr.frame(source(10, fighter([0, 0.8, 0], [0, 0, 0, 1]), tipped));
  assert.strictEqual(f.opp.down, true); assert.strictEqual(f.me.down, false);
  f = fr.frame(source(10.02, fighter([0, 0.8, 0], [0, 0, 0, 1]), fighter([0.6, 0.8, 0], [0, 1, 0, 0]),
    {referee: {available: true, slot0_count_active: true, slot1_count_active: false}}));
  assert.strictEqual(f.me.down, true); assert.strictEqual(f.opp.down, false);
});
