'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const { heading, wrap, quantiles, relativePose, analyze } = require('./compare_passive_contacts.cjs');
const { LIMBS } = require('./audit_passive_defender.cjs');
function row(t, i, segment = 0) {
  const fighter = x => ({ root_position_world_unity_xyz_m: [x, 0.7, 0],
    root_rotation_unity_xyzw: [0, 0, 0, 1], tilt_degrees: 0,
    limbs: Object.fromEntries(Object.keys(LIMBS).map(k => [k, { speed_m_s: 2, root_relative_speed_m_s: 1 }])) });
  return { event: 'kinematics', segment, sample_index: i, source_line: i + 1, qpc_ticks: 10000 + t * 1000,
    qpc_frequency_hz: 1000, utc: 'test', time_remaining_seconds: 120 - t,
    round_identity_sha256: 'a'.repeat(64), round_active: true, stream_active: true,
    neutral_check_after_warmup: true, command: { desired_action: 1, velocity_zero: true },
    actor: fighter(1), opponent: fighter(0) };
}
function event(s, opponent, actor = 0) { return { ...s, index: s.sample_index, score_delta: { actor, opponent } }; }
test('heading convention matches independently evaluated live encoder quaternion transform', () => {
  assert.equal(heading([0, 0, 0, 1]), 0);
  assert.ok(Math.abs(heading([0, Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4)]) + Math.PI / 2) < 1e-12);
  for (let i = 0; i < 1000; i++) {
    const q = [Math.sin(i), Math.cos(i * 0.7), Math.sin(i * 0.3), Math.cos(i * 0.9)];
    const n = Math.hypot(...q), m = [-q[3], q[0], q[2], q[1]].map(v => v / n);
    const expected = Math.atan2(2 * (m[0] * m[3] + m[1] * m[2]), 1 - 2 * (m[2] ** 2 + m[3] ** 2));
    assert.ok(Math.abs(wrap(heading(q) - expected)) < 1e-12);
  }
});
test('relative pose uses Unity horizontal XZ, not root height', () => {
  const s = row(0, 0); s.actor.root_position_world_unity_xyz_m = [3, 17, 4];
  const p = relativePose(s); assert.equal(p.gap_horizontal_m, 5);
  assert.deepEqual(p.defender_relative_to_opponent_heading_m, { forward: 3, lateral: 4 });
  assert.equal(p.defender_root_height_m, 17);
});
test('quantiles preserve empty/unavailable values and exact linear interpolation', () => {
  assert.equal(quantiles([null, NaN]).count, 0);
  assert.equal(quantiles([4, 0, 2, 6]).median, 3);
  assert.equal(quantiles([4, 0, 2, 6]).p90, 5.4);
});
test('pre-score snapshots never look ahead and fixed controls exclude score-adjacent motion', () => {
  const rows = Array.from({ length: 71 }, (_, i) => row(i / 10, i));
  const r = analyze(rows, [event(rows[30], 1)]);
  const c = r.score_contexts[0];
  assert.equal(c.kind, 'opponent_one_or_two_point_update_context');
  assert.equal(c.pre_window.start_sample_index, 20);
  assert.equal(c.pre_window.end_sample_index, 30);
  for (const snapshot of c.snapshots) assert.ok(snapshot.actual_seconds_before_update >= snapshot.requested_seconds_before_update);
  assert.ok(r.unscored_motion_windows.length > 0);
  for (const window of r.unscored_motion_windows) {
    assert.equal(window.label, 'no_score_observed');
    assert.ok(window.end_pose.qpc_ticks < rows[30].qpc_ticks - 1000 || window.start_pose.qpc_ticks > rows[30].qpc_ticks + 1000);
  }
});
test('five-point simultaneous changes and decreases never become ordinary-hit examples', () => {
  const rows = Array.from({ length: 31 }, (_, i) => row(i / 10, i));
  const r = analyze(rows, [event(rows[10], 5, 5), event(rows[20], -5)]);
  assert.equal(r.score_contexts[0].kind, 'five_point_update_context');
  assert.equal(r.score_contexts[1].kind, 'counter_decrease');
  assert.equal(r.comparison.ordinary_sized_opponent_updates.count, 0);
});
test('window history is bounded to its segment and unknown neutral status excludes controls', () => {
  const rows = Array.from({ length: 50 }, (_, i) => row(i / 10, i, i < 25 ? 0 : 1));
  for (const s of rows) s.command.desired_action = null;
  const r = analyze(rows, [event(rows[26], 2)]);
  assert.equal(r.score_contexts[0].pre_window.start_sample_index, 25);
  assert.equal(r.score_contexts[0].snapshots[0].pose, null);
  assert.equal(r.unscored_motion_windows.length, 0);
});
test('mismatched event source and malformed quaternions fail explicitly', () => {
  const rows = [row(0, 0), row(0.1, 1)], e = event(rows[1], 1); e.qpc_ticks++;
  assert.throws(() => analyze(rows, [e]), /score_event_sample_mismatch/);
  assert.throws(() => heading([0, 0, 0, 0]), /zero_root_quaternion/);
  assert.throws(() => analyze([rows[0], rows[0]], []), /duplicate_sample_index/);
});
