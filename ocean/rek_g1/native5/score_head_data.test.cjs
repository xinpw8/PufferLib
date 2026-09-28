'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {buildDataset, binaryDataset, precedingGeometry, selectScores, FEATURE_ORDER} = require('./score_head_data.cjs');
const sha = 'a'.repeat(64);
const source = line => ({original_source_line_1based: line, original_source_file_sha256: sha});
function fixture() {
  const requests = [], timeline = [], pairs = [], captures = {};
  for (const round of [1, 2]) {
    for (const [i, t] of [2, 4, 5, 9, 18].entries()) {
      const line = (i + 1) * 100;
      const pose = slot => ({source: source(line - 3 + slot), monotonic_receipt_time: t - .01,
        root_position_xyz: [slot ? .5 + i * .1 : 0, 1, 0], root_quaternion_xyzw_raw: slot ? [0, 1, 0, 0] : [0, 0, 0, 1]});
      requests.push({schema: 'rek.offline.move-evidence.v1', kind: 'REK_Move', round_capture: round,
        request_id: `R${round}-M${String(i + 1).padStart(3, '0')}`, source: source(line),
        unity_realtime_since_startup: t, move_index_wire_uint8: i, local_fighter_slot_from_observed_network_index: 0,
        pose_samples: {fighter_0: {nearest_preceding: pose(0)}, fighter_1: {nearest_preceding: pose(1)}},
        last_preceding_movement_input: {kind: 'REK_Input', source: source(line - 1),
          unity_realtime_since_startup: t - .005, velocity_command_xyz: [.1 * i, 0, -.2]},
      });
    }
    for (const [i, values] of [[5, 0, 2], [7, 1, 1], [8, 0, 5], [12, 0, 1]].entries()) {
      const [time, recipient, points] = values, scoreId = `R${round}-E${String(i * 2 + 1).padStart(4, '0')}`;
      const hitId = `R${round}-E${String(i * 2 + 2).padStart(4, '0')}`;
      const line = 1000 + i * 2;
      timeline.push({schema: 'rek.offline.move-evidence.v1', timeline_event_id: scoreId, round_capture: round,
        kind: 'raw_score_packet', source: source(line), monotonic_receipt_time: time, decoded: {fighter_index: recipient, points_awarded: points}});
      timeline.push({schema: 'rek.offline.move-evidence.v1', timeline_event_id: hitId, round_capture: round,
        kind: 'raw_hit_packet', source: source(line + 1), monotonic_receipt_time: time + .001});
      pairs.push({score_event_id: scoreId, hit_event_id: hitId, round_capture: round, recorded_score_recipient: recipient,
        points, is_kick: +(points === 2), causally_attributed_request_id: null, score_original_line: line, hit_original_line: line + 1});
    }
    captures[round] = {start: 0, end: 20, sha256: sha, moves: requests.filter(r => r.round_capture === round)
      .map(r => ({line: r.source.original_source_line_1based, time: r.unity_realtime_since_startup, move: r.move_index_wire_uint8}))};
  }
  return {requests, timeline, report: {schema: 'rek.human-observation.contact-signal.v1', packet_pairs: pairs}, captures};
}
const run = f => buildDataset(f.requests, f.timeline, f.report, f.captures);
test('44 request-time features, temporal boundaries, and conservative coverage', () => {
  const d = run(fixture()); assert.equal(FEATURE_ORDER.length, 44);
  assert.deepEqual(d.training.map(r => r.request_id), ['R1-M002', 'R1-M003', 'R1-M004']);
  assert.equal(d.holdout.length, 3); assert.equal(d.excluded.length, 4);
  assert.deepEqual(d.training[0].targets, [2, 1, 1, 1]);
  assert.deepEqual(d.training[1].targets, [0, 1, 0, 1]); // Score at request time excluded; five-point award excluded.
  assert.deepEqual(d.training[2].targets, [1, 0, 1, 0]); // Exact upper boundary included.
  assert.equal(d.training[0].features[5], 2 / 3);
  assert.equal(d.training[0].features[9], .1);
  assert.equal(d.training[0].features[11], 1); assert.equal(d.training[0].features[27], 1);
});
test('future poses, future geometry, score receipt geometry, and future movements cannot enter features', () => {
  const f = fixture(), before = run(f);
  for (const r of f.requests) {
    r.pose_samples.fighter_0.nearest_following = {root_position_xyz: [999, 999, 999]};
    r.geometry_from_preceding_samples = {root_distance_xz_unity_units: 999};
    r.nearest_hit_and_score_observations = {invented: 999};
    r.future_movement = {velocity_command_xyz: [999, 999, 999]};
  }
  f.timeline.forEach(r => { r.event_receipt_geometry_from_preceding_samples = {distance: 999}; });
  assert.deepEqual(run(f), before);
});
test('falsely labelled preceding pose or command rejects future leakage', () => {
  for (const mutate of [
    r => { r.pose_samples.fighter_0.nearest_preceding.monotonic_receipt_time = r.unity_realtime_since_startup + .01; },
    r => { r.pose_samples.fighter_1.nearest_preceding.source.original_source_line_1based = 9999; },
    r => { r.last_preceding_movement_input.unity_realtime_since_startup = r.unity_realtime_since_startup + 1; },
    r => { r.last_preceding_movement_input.source.original_source_file_sha256 = 'b'.repeat(64); },
  ]) { const f = fixture(); mutate(f.requests[1]); assert.throws(() => run(f), /preceding/); }
});
test('heldout values and labels cannot affect training features or train-only scaler', () => {
  const f = fixture(), before = run(f);
  f.requests.filter(r => r.round_capture === 2).forEach(r => { r.pose_samples.fighter_1.nearest_preceding.root_position_xyz[0] += 100; });
  f.timeline.filter(r => r.round_capture === 2).forEach(r => { r.monotonic_receipt_time += .2; });
  const after = run(f);
  assert.deepEqual(before.training, after.training); assert.deepEqual(before.scaler, after.scaler);
  assert.notDeepEqual(before.holdout, after.holdout);
  assert(after.training.every(r => r.round === 1)); assert(after.holdout.every(r => r.round === 2));
});
test('five-point awards excluded, opponent strikes preserved as separate outcome, no causal action labels', () => {
  const f = fixture(), selected = selectScores(f.report, f.timeline);
  assert.equal(selected.length, 6); assert.equal(selected.filter(r => r.recipient === 1).length, 2);
  assert(selected.every(r => r.points <= 2));
  f.report.packet_pairs[0].causally_attributed_request_id = 'R1-M002'; assert.throws(() => run(f), /unverified/);
});
test('duplicate score pairs and malformed schema or raw coverage reject', () => {
  let f = fixture(); f.report.packet_pairs.push({...f.report.packet_pairs[0]}); assert.throws(() => run(f), /duplicate/);
  f = fixture(); f.report.schema = 'wrong'; assert.throws(() => run(f), /invalid_report/);
  f = fixture(); f.captures[1].moves.pop(); assert.throws(() => run(f), /coverage/);
  f = fixture(); f.requests[1].move_index_wire_uint8 = 17; assert.throws(() => run(f), /invalid_request/);
  f = fixture(); f.requests[1].pose_samples.fighter_0.nearest_preceding.root_position_xyz[0] = Infinity;
  assert.throws(() => run(f), /preceding/);
});
test('normalized quaternion sign and full rotations preserve geometry', () => {
  const f = fixture(), r = f.requests[1], before = precedingGeometry(r);
  for (const slot of [0, 1]) r.pose_samples[`fighter_${slot}`].nearest_preceding.root_quaternion_xyzw_raw =
    r.pose_samples[`fighter_${slot}`].nearest_preceding.root_quaternion_xyzw_raw.map(x => -3 * x);
  assert.deepEqual(precedingGeometry(r), before);
  assert(Math.abs(before[1] - 1) < 1e-12); assert(Math.abs(before[3] - 1) < 1e-12);
});
test('binary format includes training-only scaler once, train-first rows and binary labels', () => {
  const d = run(fixture()), b = binaryDataset(d);
  assert.equal(b.subarray(0, 8).toString('ascii'), 'REKSHP1\0');
  assert.deepEqual([8, 12, 16, 20].map(o => b.readUInt32LE(o)), [1, 44, 3, 3]);
  assert.equal(b.length, 24 + 44 * 8 + 6 * 46 * 4);
  assert.equal(b.readFloatLE(24), Math.fround(d.scaler.mean[0]));
  const offset = 24 + 44 * 8;
  assert.equal(b.readFloatLE(offset), Math.fround(d.training[0].features[0]));
  assert.deepEqual([44, 45].map(i => b.readFloatLE(offset + i * 4)), [1, 1]);
  assert(d.scaler.constant_feature_indices.includes(7)); assert.equal(d.scaler.scale[7], 1);
});
