'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { PassiveAudit, analyzeFile, LIMBS, deltaKind } = require('./audit_passive_defender.cjs');

const bones = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
  'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link',
  'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link',
  'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link',
  'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
  'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'right_shoulder_pitch_link',
  'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link',
  'right_wrist_pitch_link', 'right_wrist_yaw_link'];
function sample(t, sequence, change = {}) {
  return { event: 'g1_policy_state', schema: 'rek.g1_policy_source.v1',
    round_identity_sha256: 'a'.repeat(64), local_slot: 0, observation_sequence: sequence,
    clock: { qpc_ticks: 1000000000 + Math.round(t * 10000000), qpc_frequency_hz: 10000000,
      utc: '2026-09-17T00:00:00Z' }, stream_active: true,
    input: { desired_action: 1, velocity_command_xyz: [0, 0, 0], requested_move_index: null },
    round: { active: true, clean_hits: [0, 0], time_remaining: 120 - t },
    fighters: [0, 1].map(slot => ({ root_position_xyz: [slot, 0.8, 0],
      root_rotation_xyzw: [0, 0, 0, 1], tilt_angle: 0, falling: false, fallen: false,
      bone_names: bones.slice(), bone_world_positions_xyz: bones.map((_, i) => [slot, 0.5 + i / 100, 0]) })),
    ...change };
}
function setLimb(s, slot, name, position) {
  s.fighters[slot].bone_world_positions_xyz[bones.indexOf(LIMBS[name])] = position;
}
function run(samples, options = {}) {
  const output = { kinematics: [], score_events: [], motion_intervals: [] };
  const audit = new PassiveAudit(options, (kind, value) => output[kind].push(structuredClone(value)));
  samples.forEach((s, i) => audit.consume(s, i + 1));
  return { ...output, summary: audit.finish() };
}

test('variable QPC intervals, both roles, named limb ordering and root-relative speed', () => {
  const a = sample(0, 1), b = sample(0.04, 2), c = sample(0.14, 3);
  b.fighters[1].root_position_xyz[0] += 0.08;
  for (const point of b.fighters[1].bone_world_positions_xyz) point[0] += 0.08;
  c.fighters[1] = structuredClone(b.fighters[1]);
  c.fighters[1].root_position_xyz[0] += 0.1;
  for (const point of c.fighters[1].bone_world_positions_xyz) point[0] += 0.1;
  const r = run([a, b, c]);
  assert.equal(r.kinematics[0].opponent.limbs.left_hand.speed_m_s, null);
  assert.equal(r.kinematics[1].delta_seconds, 0.04);
  assert.ok(Math.abs(r.kinematics[1].opponent.limbs.left_hand.speed_m_s - 2) < 1e-12);
  assert.ok(Math.abs(r.kinematics[2].opponent.limbs.left_hand.speed_m_s - 1) < 1e-12);
  assert.ok(r.kinematics[1].opponent.limbs.left_hand.root_relative_speed_m_s < 1e-12);
  assert.equal(r.kinematics[1].actor.root_step_m, 0);
});
test('local slot 1 reverses roles and point counters without coordinate conversion', () => {
  const a = sample(0, 1, { local_slot: 1 }), b = sample(0.1, 2, { local_slot: 1 });
  b.round.clean_hits = [2, 1];
  const r = run([a, b]);
  assert.equal(r.kinematics[0].actor.slot, 1);
  assert.equal(r.kinematics[0].opponent.slot, 0);
  assert.deepEqual(r.score_events[0].score_delta, { actor: 1, opponent: 2 });
  assert.deepEqual(r.summary.positive_point_updates, { actor: 1, opponent: 2 });
});
test('neutral checks exclude warmup, distinguish unknown/nonzero, and allow physical movement', () => {
  const a = sample(0, 1), b = sample(0.1, 2), c = sample(0.2, 3), d = sample(0.3, 4);
  a.input.desired_action = 2; a.input.velocity_command_xyz = [1, 0, 0];
  b.fighters[0].root_position_xyz[0] += 0.25;
  c.input.desired_action = null; c.input.velocity_command_xyz = null;
  d.input.velocity_command_xyz = [0, 0, 0.00001];
  const r = run([a, b, c, d], { warmupSeconds: 0.1 });
  assert.equal(r.summary.active_stream_neutral.samples, 4);
  assert.deepEqual(r.summary.after_warmup_neutral, { samples: 3, desired_action_1: 2,
    desired_action_other: 0, desired_action_unknown: 1, velocity_command_zero: 1,
    velocity_command_nonzero: 1, velocity_command_unknown: 1, both_neutral: 1 });
  assert.equal(r.summary.sampled_neutral_confirmed_after_warmup, false);
  assert.equal(r.summary.maximum_actor_root_step_m, 0.25);
  assert.equal(run([sample(0, 1), b], { warmupSeconds: 0.1 }).summary.sampled_neutral_confirmed_after_warmup, true);
});
test('motion intervals, score windows, delayed points and unscored motion use bounded labels', () => {
  const samples = Array.from({ length: 31 }, (_, i) => sample(i * 0.1, i + 1));
  for (let i = 1; i <= 30; i++) {
    setLimb(samples[i], 1, 'left_hand', [1 + (i >= 3 ? 0.6 : 0), 0.72, 0]);
    setLimb(samples[i], 1, 'right_hand', [1 + (i >= 23 ? 0.6 : 0), 0.79, 0]);
    if (i >= 8) samples[i].round.clean_hits = [0, 5];
  }
  const r = run(samples, { windowSeconds: 0.5 });
  const scored = r.motion_intervals.find(m => m.limb === 'left_hand');
  const unscored = r.motion_intervals.find(m => m.limb === 'right_hand');
  assert.equal(scored.score_observation, 'score_counter_update_observed');
  assert.deepEqual(scored.score_event_indices, [0]);
  assert.equal(unscored.score_observation, 'no_score_observed');
  assert.equal(r.score_events[0].update_kind.opponent, 'five_point_update');
  assert.equal(r.score_events[0].window.start_sample_index, 3);
  assert.equal(r.score_events[0].window.end_sample_index, 13);
  assert.equal(r.score_events[0].window.trailing_window_truncated, false);
  assert.equal(unscored.trailing_window_truncated, false);
  assert.equal(r.summary.positive_point_updates.opponent, 5);
  assert.equal(JSON.stringify(r).includes('knockout'), false);
});
test('zero-width score context includes simultaneous above-threshold motion', () => {
  const a = sample(0, 1), b = sample(0.1, 2);
  setLimb(b, 1, 'left_foot', [2, 0.56, 0]); b.round.clean_hits = [0, 2];
  const r = run([a, b], { windowSeconds: 0 });
  assert.deepEqual(r.motion_intervals[0].score_event_indices, [0]);
});
test('no derivatives or invented awards across discontinuities and counter decreases are explicit', () => {
  for (const change of [s => { s.round_identity_sha256 = 'b'.repeat(64); },
    s => { s.local_slot = 1; }, s => { s.clock.qpc_frequency_hz *= 2; },
    s => { s.observation_sequence = 1; }, s => { s.clock.qpc_ticks -= 1000000; },
    s => { s.clock.qpc_ticks += 100000000; }]) {
    const a = sample(0, 1), b = sample(0.1, 2); change(b); b.round.clean_hits = [5, 7];
    const r = run([a, b]);
    assert.equal(r.summary.segments, 2);
    assert.equal(r.score_events.length, 0);
    assert.equal(r.kinematics[1].delta_seconds, null);
  }
  const a = sample(0, 1), b = sample(0.1, 2); a.round.clean_hits = [10, 9]; b.round.clean_hits = [0, 0];
  const r = run([a, b]);
  assert.equal(r.score_events[0].update_kind.actor, 'counter_decrease');
  assert.equal(r.summary.counter_decrease_events, 1);
  assert.deepEqual(r.summary.positive_point_updates, { actor: 0, opponent: 0 });
});
test('missing limbs, unsafe clocks and invalid JSON are counted and break history', () => {
  const audit = new PassiveAudit();
  const a = sample(0, 1); delete a.fighters[1].bone_world_positions_xyz[22]; audit.consume(a);
  const b = sample(0.1, 2); b.clock.qpc_ticks = Number.MAX_SAFE_INTEGER + 1; audit.consume(b);
  audit.consumeLine('{', 3);
  audit.consume(sample(0.2, 3), 4);
  const r = audit.finish();
  assert.equal(r.invalid_records, 3); assert.equal(r.samples, 1); assert.equal(r.complete, false);
  assert.equal(r.error_counts.limb_position, 1); assert.equal(r.error_counts.qpc_ticks, 1);
});
test('acknowledgments are reported separately from sampled command coverage', () => {
  const r = run([{ event: 'hello' }, { event: 'g1_policy_action', action: 1, applied: true },
    { event: 'g1_policy_action', action: 1, applied: false }, { event: 'g1_policy_action', action: 16, applied: true }, sample(0, 1)]);
  assert.deepEqual(r.summary.acknowledgments, { action_1_applied: 1, action_1_not_applied: 1, other_actions: 1 });
  assert.equal(r.summary.after_warmup_neutral.samples, 0);
  assert.equal(r.summary.sampled_neutral_confirmed_after_warmup, false);
});
test('point sizes do not assert hit type or knockout cause', () => {
  assert.deepEqual([-1, 0, 1, 2, 3, 5, 7].map(deltaKind), ['counter_decrease', 'unchanged',
    'one_point_update', 'two_point_update', 'other_positive_update', 'five_point_update', 'other_positive_update']);
});
test('file audit streams original bytes, indexes raw lines and refuses overwrite', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rek-passive-audit-test-'));
  try {
    const input = path.join(root, 'relay.stdout.jsonl'), output = path.join(root, 'audit');
    const a = sample(0, 1), b = sample(0.1, 2); b.round.clean_hits = [1, 0];
    const bytes = JSON.stringify({ event: 'hello' }) + '\r\n' + JSON.stringify(a) + '\r\n' + JSON.stringify(b) + '\r\n';
    fs.writeFileSync(input, bytes);
    const result = await analyzeFile(input, output, { warmupSeconds: 0 });
    assert.equal(result.source.sha256, crypto.createHash('sha256').update(bytes).digest('hex'));
    assert.equal(result.source.bytes, Buffer.byteLength(bytes));
    assert.equal(result.samples, 2);
    const events = fs.readFileSync(path.join(output, 'score_events.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(events[0].source_line, 3); assert.equal(events[0].window.start_source_line, 2);
    assert.equal(events[0].window.trailing_window_truncated, true);
    await assert.rejects(analyzeFile(input, output), /output_already_exists/);
    assert.equal(fs.readFileSync(input, 'utf8'), bytes);
    const cli = spawnSync(process.execPath, [path.join(__dirname, 'audit_passive_defender.cjs'), input,
      path.join(root, 'cli'), '--warmup-seconds=0', '--window-seconds=0.5'], { encoding: 'utf8' });
    assert.equal(cli.status, 0, cli.stderr);
    assert.equal(JSON.parse(cli.stdout).samples, 2);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
