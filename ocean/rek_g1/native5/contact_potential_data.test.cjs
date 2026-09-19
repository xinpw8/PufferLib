'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {features, selectEvents, nearestDiagnostic, fitContactPotential, canonicalJson} = require('./contact_potential_data.cjs');
const sourceHash = 'a'.repeat(64);
function source() {
  return {schema: 'rek.human-observation.contact-signal.v1', packet_pairs: [1, 2].flatMap(round =>
    Array.from({length: round === 1 ? 12 : 10}, (_, i) => ({
      score_event_id: `R${round}-E${String(i * 2 + 1).padStart(4, '0')}`,
      hit_event_id: `R${round}-E${String(i * 2 + 2).padStart(4, '0')}`,
      round_capture: round, recorded_score_recipient: 0, points: i % 2 + 1, is_kick: i % 2,
      hit_receipt_minus_score_receipt_ms: 1, causally_attributed_request_id: null,
      geometry_at_score_receipt: {distance_xz_unity_units: .4 + i * .021 + round * .01,
        bearing_relative_pelvis_positive_x_deg: -80 + i * 13 + round},
    })))};
}
test('round-one-only population scale and deterministic model identity', () => {
  const report = source(), {model, diagnostics} = fitContactPotential(report, sourceHash);
  assert.equal(model.schema, 'rek.contact_potential.v1'); assert.equal(model.sample_count, 12);
  assert.equal(diagnostics.holdout.length, 10);
  const expected = [0, 1, 2].map(j => {
    const mean = model.samples.reduce((sum, row) => sum + row[j], 0) / 12;
    return Math.sqrt(model.samples.reduce((sum, row) => sum + (row[j] - mean) ** 2, 0) / 12);
  });
  assert.deepEqual(model.feature_scale, expected);
  const {model_id, ...body} = model;
  assert.equal(model_id, crypto.createHash('sha256').update(canonicalJson(body)).digest('hex'));
  assert.deepEqual(fitContactPotential(report, sourceHash).model, model);
  assert(model.training_event_ids.every(id => id.startsWith('R1-')));
  assert(model.holdout_event_ids.every(id => id.startsWith('R2-')));
});
test('held-out geometry cannot affect anchors, scales, or model id', () => {
  const report = source(), before = fitContactPotential(report, sourceHash);
  for (const row of report.packet_pairs.filter(row => row.round_capture === 2)) {
    row.geometry_at_score_receipt.distance_xz_unity_units += 10;
    row.geometry_at_score_receipt.bearing_relative_pelvis_positive_x_deg += 45;
  }
  const after = fitContactPotential(report, sourceHash);
  assert.deepEqual(before.model, after.model);
  assert.notDeepEqual(before.diagnostics.holdout, after.diagnostics.holdout);
});
test('AI events and five-point awards do not become fitting rows', () => {
  const report = source(), before = fitContactPotential(report, sourceHash);
  report.packet_pairs.push({...report.packet_pairs[0], score_event_id: 'R1-E0801', hit_event_id: 'R1-E0802', recorded_score_recipient: 1});
  report.packet_pairs.push({...report.packet_pairs[0], score_event_id: 'R1-E0803', hit_event_id: 'R1-E0804', points: 5});
  assert.equal(selectEvents(report).length, 22);
  assert.deepEqual(before, fitContactPotential(report, sourceHash));
  assert.equal(before.model.miss_labels, 0); assert.equal(before.model.action_execution_inferred, false);
});
test('angular wrap preserves features and signed bearing orientation', () => {
  for (const angle of [-180, -91, 0, 47.25, 180, 359]) {
    const value = features(.5, angle);
    for (const offset of [-720, -360, 360, 720])
      features(.5, angle + offset).forEach((x, i) => assert(Math.abs(x - value[i]) < 1e-12));
  }
  assert(features(.5, 90)[2] > 0); assert(features(.5, -90)[2] < 0);
});
test('potential uses stated standardized nearest-anchor formula', () => {
  const result = nearestDiagnostic([3, 4, 0], [[0, 0, 0]], [1, 1, 1]);
  assert.equal(result.nearest_standardized_distance, 5); assert.equal(result.potential, -5 / 6);
  assert.equal(nearestDiagnostic([0, 0, 0], [[0, 0, 0]], [1, 1, 1]).potential, -0);
  const {model, diagnostics} = fitContactPotential(source(), sourceHash);
  diagnostics.training_leave_one_out.forEach((row, i) => {
    assert.notEqual(row.nearest_training_index, i);
    assert.notEqual(row.nearest_training_event_id, row.event_id);
    assert(row.potential >= -1 && row.potential <= 0);
  });
  assert.equal(model.unit_mapping.metre_calibration_verified, false);
  assert.equal(diagnostics.accuracy_estimated, false);
});
test('malformed schema, duplicate pairs, invalid hash, and count mismatch reject', () => {
  assert.throws(() => fitContactPotential({...source(), schema: 'wrong'}, sourceHash), /invalid_source_schema/);
  assert.throws(() => fitContactPotential(source(), 'A'.repeat(64)), /invalid_source_hash/);
  const duplicate = source(); duplicate.packet_pairs.push({...duplicate.packet_pairs[0]});
  assert.throws(() => fitContactPotential(duplicate, sourceHash), /duplicate/);
  const missing = source(); missing.packet_pairs.pop();
  assert.throws(() => fitContactPotential(missing, sourceHash), /twelve_training_ten_holdout/);
});
test('nonfinite geometry, inconsistent point/part, causal label, and invalid timing reject', () => {
  for (const mutate of [
    r => { r.geometry_at_score_receipt.distance_xz_unity_units = NaN; },
    r => { r.geometry_at_score_receipt.distance_xz_unity_units = 0; },
    r => { r.geometry_at_score_receipt.bearing_relative_pelvis_positive_x_deg = Infinity; },
    r => { r.is_kick = 1 - r.is_kick; },
    r => { r.causally_attributed_request_id = 'invented'; },
    r => { r.hit_receipt_minus_score_receipt_ms = 11; },
    r => { r.round_capture = 3; },
  ]) { const report = source(); mutate(report.packet_pairs[0]); assert.throws(() => fitContactPotential(report, sourceHash)); }
});
test('zero training scale rejects without a fallback', () => {
  const report = source(); report.packet_pairs.forEach(row => { row.geometry_at_score_receipt.distance_xz_unity_units = 1; });
  assert.throws(() => fitContactPotential(report, sourceHash), /training_scale_zero/);
  assert.throws(() => nearestDiagnostic([1, 0, 0], [[1, 0, 0]], [0, 1, 1]), /invalid_feature_scale/);
  assert.throws(() => nearestDiagnostic([1, 0, 0], [[1, 0, 0]], [1, 1, 1], 0), /nearest_training_sample_missing/);
});
