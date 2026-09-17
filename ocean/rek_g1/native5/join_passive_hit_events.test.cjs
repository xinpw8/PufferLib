'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { joinFiles } = require('./join_passive_hit_events.cjs');

function pose(t, frame, qpc, change = {}) {
  return { event: 'g1_policy_state', schema: 'rek.g1_policy_source.v1', observation_sequence: frame,
    clock: { unity_time: t, unity_frame: frame, qpc_ticks: qpc, qpc_frequency_hz: 1000,
      utc: '2026-09-17T07:00:00Z' }, round_identity_sha256: 'a'.repeat(64), local_slot: 0,
    round: { active: true, time_remaining: 120 - t, clean_hits: [0, 1] },
    input: { desired_action: 1, velocity_command_xyz: [0, 0, 0] },
    fighters: [{ root_position_xyz: [0, 0.8, 0], root_rotation_xyzw: [0, 0, 0, 1] },
      { root_position_xyz: [0.3, 0.8, 0.4], root_rotation_xyzw: [0, 1, 0, 0] }], ...change };
}
function hit(frame = 101, t = 10.04) {
  return { event: 'raw_hit_packet', raw_hit_sequence: 1, raw_protocol_sequence: 5,
    unity_frame: frame, unity_time: t, unity_unscaled_time: t + 2, monotonic_receipt_time: t + 2.01,
    wire_delivery: 'unreliable', wire_body_sha256: 'c'.repeat(64), wire_body_base64: 'AA==',
    decoded: { position_xyz: [0.15, 0.9, 0.2], surface_normal_xyz: [1, 0, 0], relative_speed: 3.25, is_kick: 0 } };
}
function score(points = 1, slot = 1, frame = 101, t = 10.04) {
  return { event: 'raw_score_packet', unity_frame: frame, unity_time: t,
    wire_delivery: 'reliable', decoded: { fighter_index: slot, new_hit_count: points, points_awarded: points } };
}
function capture(events) {
  return [{ event: 'capture_start', stopwatch_timestamp_ticks: 1000, stopwatch_frequency_hz: 1000,
    utc: '2026-09-17T07:00:00Z', pid: 32, plugin_sha256: 'b'.repeat(64), unwanted_metadata: 'omit' },
  ...events, { event: 'capture_end', stopwatch_timestamp_ticks: 3000, capture_error_count: 0,
    raw_hit_packet_count: events.filter(e => e.event === 'raw_hit_packet').length,
    raw_score_packet_count: events.filter(e => e.event === 'raw_score_packet').length }];
}
async function fixture(t, recorder, relay) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rek-hit-join-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const native = path.join(root, 'native.jsonl'), source = path.join(root, 'relay.jsonl'), out = path.join(root, 'joined');
  const nativeBytes = recorder.map(JSON.stringify).join('\r\n') + '\r\n';
  const relayBytes = (relay === '-' ? [] : relay).map(JSON.stringify).join('\n') + '\n';
  fs.writeFileSync(native, nativeBytes); fs.writeFileSync(source, relayBytes);
  const summary = await joinFiles(native, relay === '-' ? '-' : source, out);
  const rows = name => fs.readFileSync(path.join(out, name), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
  return { root, native, source, out, nativeBytes, relayBytes, summary, rows };
}
const poses = () => [pose(10, 100, 1200), pose(10.1, 102, 1739), pose(10.3, 104, 2071)];

test('variable clocks preserve raw packets and use actual Unity pose offsets without inventing hit QPC', async t => {
  const h = hit(), s = score();
  const r = await fixture(t, capture([h, s]), poses());
  const x = r.rows('hit_events.jsonl')[0], b = x.poses.candidates[0];
  assert.equal(r.summary.complete, true);
  assert.deepEqual(x.raw, h);
  assert.deepEqual(x.same_frame_score_association.score_records[0].raw, s);
  assert.equal(x.clock.qpc_ticks, null); assert.equal(x.clock.utc, null);
  assert.equal(b.prior[0].clock.qpc_ticks, 1200); assert.equal(b.next[0].clock.qpc_ticks, 1739);
  assert.ok(Math.abs(b.prior[0].pose_minus_event_unity_seconds + 0.04) < 1e-12);
  assert.ok(Math.abs(b.next[0].pose_minus_event_unity_seconds - 0.06) < 1e-12);
  assert.equal(b.next[0].pose_minus_event_qpc_seconds, null);
  assert.equal(b.prior[0].root_distance_3d_m, 0.5);
  assert.equal(b.prior[0].root_distance_ground_xz_m, 0.5);
  assert.equal(b.prior[0].heading_axis, 'root_local_positive_x_projected_to_Unity_XZ');
  assert.ok(Math.abs(b.prior[0].local_bearing_to_opponent_rad - Math.atan2(0.4, 0.3)) < 1e-12);
  assert.equal(x.same_frame_score_association.status, 'unique_same_frame_association_noncausal');
  assert.deepEqual(x.attacker_candidates.map(c => c.slot), [1]);
  assert.equal(x.attacker_candidates[0].confirmed, false);
  assert.equal(x.clip_id, null); assert.equal(x.body_zone, null);
  assert.equal(r.summary.recorder_capture.start.unwanted_metadata, undefined);
});

test('same-frame multiple hits and scores remain ambiguous, duplicates are retained and +5 separate', async t => {
  const r = await fixture(t, capture([hit(), hit(), score(), score(2, 0), score(5, 1)]), poses());
  const hits = r.rows('hit_events.jsonl');
  assert.equal(hits.length, 2);
  assert.equal(r.summary.ambiguous_same_frame_hit_records, 2);
  assert.equal(hits[0].same_frame_score_association.status, 'ambiguous_same_frame_association');
  assert.deepEqual(hits[0].same_frame_score_association.score_indices, [0, 1, 2]);
  assert.deepEqual(hits[0].same_frame_score_association.five_point_score_indices, [2]);
  assert.deepEqual(hits[0].attacker_candidates.map(c => c.slot), [1, 0]);
  assert.equal(r.rows('five_point_awards.jsonl').length, 1);
  assert.equal(r.rows('five_point_awards.jsonl')[0].association_is_causal, false);
});

test('same-frame association preserves differing packet times instead of requiring equal timestamps', async t => {
  const r = await fixture(t, capture([hit(), score(1, 1, 101, 10.045)]), poses());
  const x = r.rows('hit_events.jsonl')[0];
  assert.equal(x.same_frame_score_association.status, 'unique_same_frame_association_noncausal');
  assert.equal(x.same_frame_score_association.score_records[0].raw.unity_time, 10.045);
  assert.equal(x.clock.unity_time, 10.04);
});

test('no received hit with +5 remains unknown and never becomes a strike or knockout', async t => {
  const r = await fixture(t, capture([score(5)]), poses());
  assert.equal(r.summary.complete, true);
  assert.equal(r.summary.hit_observation, 'no_received_hit_packets_contact_unknown');
  assert.deepEqual(r.rows('hit_events.jsonl'), []);
  assert.equal(r.rows('five_point_awards.jsonl')[0].award_class, 'five_point_award_cause_unresolved');
  const withHit = await fixture(t, capture([hit(), score(5)]), poses());
  assert.deepEqual(withHit.rows('hit_events.jsonl')[0].attacker_candidates, []);
});

test('duplicate pose times and exact-frame sample retain every alternative', async t => {
  const r = await fixture(t, capture([hit()]), [pose(10, 100, 1200),
    pose(10.04, 101, 1400), pose(10.04, 101, 1401), pose(10.1, 102, 1700)]);
  const x = r.rows('hit_events.jsonl')[0];
  assert.equal(x.poses.status, 'ambiguous_duplicate_pose_time');
  assert.equal(x.poses.candidates[0].same_time.length, 2);
  assert.equal(x.poses.candidates[0].same_time[0].pose_minus_event_unity_frames, 0);
});

test('clock regression, round changes, frame contradictions and incomplete coverage are explicit', async t => {
  const replay = poses().concat(poses().map(p => ({ ...p, round_identity_sha256: 'd'.repeat(64) })));
  let r = await fixture(t, capture([hit()]), replay);
  assert.equal(r.rows('hit_events.jsonl')[0].poses.status, 'ambiguous_clock_segments');
  r = await fixture(t, capture([hit()]), [pose(10, 500, 1200), pose(10.1, 501, 1700)]);
  assert.equal(r.rows('hit_events.jsonl')[0].poses.status, 'inconsistent_unity_frame_time');
  r = await fixture(t, capture([hit(99, 9)]), poses());
  assert.equal(r.rows('hit_events.jsonl')[0].poses.status, 'partial_bracket');
  assert.equal(r.rows('hit_events.jsonl')[0].poses.candidates[0].prior.length, 0);
});

test('QPC capture bounds exclude unrelated poses and clock frequencies are never converted', async t => {
  const wrong = pose(10.02, 100, 1400); wrong.clock.qpc_frequency_hz = 2000;
  const r = await fixture(t, capture([hit()]), [pose(10, 100, 900), ...poses(), wrong, pose(10.2, 103, 3100)]);
  assert.equal(r.summary.excluded_by_capture_qpc, 3);
  assert.equal(r.summary.pose_records, 3);
  assert.equal(r.rows('hit_events.jsonl')[0].poses.status, 'bracketed');
});

test('local slot 1 maps roots correctly; native QPC if explicitly present remains measured', async t => {
  const h = { ...hit(), stopwatch_timestamp_ticks: 1300, stopwatch_frequency_hz: 1000 };
  const r = await fixture(t, capture([h]), poses().map(p => ({ ...p, local_slot: 1 })));
  const b = r.rows('hit_events.jsonl')[0].poses.candidates[0].prior[0];
  assert.equal(b.local_root.slot, 1); assert.equal(b.opponent_root.slot, 0);
  assert.equal(b.pose_minus_event_qpc_seconds, -0.1);
});

test('hashes cover exact capture bytes, output refuses overwrite and CLI works', async t => {
  const r = await fixture(t, capture([hit(), score()]), poses());
  assert.equal(r.summary.inputs.recorder.sha256, crypto.createHash('sha256').update(r.nativeBytes).digest('hex'));
  assert.equal(r.summary.inputs.relay.sha256, crypto.createHash('sha256').update(r.relayBytes).digest('hex'));
  assert.equal(r.rows('hit_events.jsonl')[0].recorder_sha256, r.summary.inputs.recorder.sha256);
  assert.equal(r.rows('hit_events.jsonl')[0].recorder_line, 2);
  await assert.rejects(joinFiles(r.native, r.source, r.out), /output_already_exists/);
  const cli = spawnSync(process.execPath, [path.join(__dirname, 'join_passive_hit_events.cjs'),
    r.native, r.source, path.join(r.root, 'cli')], { encoding: 'utf8' });
  assert.equal(cli.status, 0, cli.stderr);
  assert.equal(JSON.parse(cli.stdout).received_hit_packets, 1);
  assert.equal(fs.readFileSync(r.native, 'utf8'), r.nativeBytes);
});

test('missing footer, malformed pose and mismatched packet counts are not complete captures', async t => {
  const events = capture([hit()]);
  let r = await fixture(t, events.slice(0, -1), poses());
  assert.equal(r.summary.complete, false); assert.equal(r.summary.recorder_capture.qpc_bounds_applied, false);
  const bad = pose(10.02, 101, 1200); bad.fighters[0].root_position_xyz = null;
  r = await fixture(t, events, [bad, ...poses()]);
  assert.equal(r.summary.invalid_pose_records, 1); assert.equal(r.summary.complete, false);
  events.at(-1).raw_hit_packet_count = 7;
  r = await fixture(t, events, poses());
  assert.equal(r.summary.recorder_capture.packet_counts_match_footer, false);
});

function nativePose(t, frame, qpc, rootOnly = true) {
  const p = pose(t, frame, qpc);
  const result = { event: rootOnly ? 'root_pose_sample' : 'sample', utc: p.clock.utc,
    stopwatch_timestamp_ticks: qpc, unity_time: t, unity_frame: frame,
    unity_fixed_time: t, local_fighter_index: 0, fight_epoch: 3, round_number: 2 };
  for (const slot of [0, 1]) result[`fighter_${slot}${rootOnly ? '_root' : ''}`] = rootOnly
    ? { world_position_xyz: p.fighters[slot].root_position_xyz, world_rotation_xyzw: p.fighters[slot].root_rotation_xyzw }
    : { root_position: p.fighters[slot].root_position_xyz, root_rotation: p.fighters[slot].root_rotation_xyzw };
  if (!rootOnly) {
    result.round = { number: 2, active: true, time_remaining: 110, clean_hits: [0, 1] };
    result.input = { velocity_command: [0, 0, 0], pending_move: false, punching: false };
  }
  return result;
}

test('native-only joins root poses, keeps same-frame FixedUpdate alternatives and reports neutral observations', async t => {
  const events = [nativePose(10, 100, 1200), nativePose(10.03, 101, 1300),
    nativePose(10.04, 101, 1400, false), hit(), score(),
    { event: 'outbound_request_projection', message: 'REK_Input', velocity_command_xyz: [0, 0, 0] },
    nativePose(10.05, 101, 1500), nativePose(10.1, 102, 1700)];
  const records = capture(events); records.at(-1).client_transport_method_counts = { SendVelocityCommand: 1 };
  const r = await fixture(t, records, '-');
  assert.equal(r.summary.complete, true);
  assert.equal(r.summary.inputs.relay, null);
  assert.equal(r.summary.pose_source, 'native_root_pose_sample');
  assert.equal(r.summary.pose_records, 4);
  assert.equal(r.summary.native_observations.root_pose_samples, 4);
  assert.equal(r.summary.native_observations.compact_samples, 1);
  assert.equal(r.summary.native_observations.all_three_neutral_samples, 1);
  assert.equal(r.summary.native_observations.outbound_velocity_zero_requests, 1);
  assert.deepEqual(r.summary.recorder_capture.end.client_transport_method_counts, { SendVelocityCommand: 1 });
  const b = r.rows('hit_events.jsonl')[0].poses.candidates[0];
  assert.equal(b.same_frame.length, 2);
  assert.equal(b.prior[0].recorder_line, 3);
  assert.equal(b.next[0].clock.qpc_ticks, 1500);
  assert.equal(b.prior[0].round, null);
  assert.equal(b.prior[0].input, null);
  assert.deepEqual(b.native_round, { fight_epoch: 3, round_number: 2 });
});

test('native-only falls back to compact samples without fabricating absent root-only fields', async t => {
  const r = await fixture(t, capture([nativePose(10, 100, 1200, false), hit(), score(5),
    nativePose(10.1, 102, 1700, false)]), '-');
  assert.equal(r.summary.pose_source, 'native_sample_fallback');
  assert.equal(r.summary.native_observations.all_three_neutral_samples, 2);
  const before = r.rows('hit_events.jsonl')[0].poses.candidates[0].prior[0];
  assert.equal(before.round.time_remaining, 110);
  assert.equal(before.input.pending_move, false);
  assert.equal(before.clock.qpc_frequency_hz, 1000);
  assert.equal(before.clock.qpc_ticks, 1200);
  assert.equal(before.clock.utc, '2026-09-17T07:00:00Z');
});

test('projected root +X heading is unavailable when vertical, and coincident roots have no bearing', async t => {
  const p = poses();
  p[0].fighters[0].root_rotation_xyzw = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
  p[1].fighters[1].root_position_xyz = p[1].fighters[0].root_position_xyz.slice();
  const r = await fixture(t, capture([hit()]), p);
  const b = r.rows('hit_events.jsonl')[0].poses.candidates[0];
  assert.equal(b.prior[0].local_heading_rad, null);
  assert.equal(b.prior[0].local_bearing_to_opponent_rad, null);
  assert.equal(b.next[0].local_bearing_to_opponent_rad, null);
});
