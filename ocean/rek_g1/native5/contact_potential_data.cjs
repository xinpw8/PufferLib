'use strict';

// Offline positive-only diagnostic. No game, network, action labels or negative labels.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const SOURCE_SCHEMA = 'rek.human-observation.contact-signal.v1';
const MODEL_SCHEMA = 'rek.contact_potential.v1';
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const check = (condition, reason) => { if (!condition) throw Error(reason); };
const finite = value => typeof value === 'number' && Number.isFinite(value);
const canonicalJson = value => JSON.stringify(value, function (_key, item) {
  return item && typeof item === 'object' && !Array.isArray(item)
    ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item;
});

function features(distance, bearingDegrees) {
  check(finite(distance) && distance > 0 && finite(bearingDegrees), 'invalid_receipt_geometry');
  const wrapped = ((bearingDegrees + 180) % 360 + 360) % 360 - 180;
  const radians = wrapped * Math.PI / 180;
  return [distance, Math.cos(radians), Math.sin(radians)];
}

function selectEvents(report) {
  check(report && report.schema === SOURCE_SCHEMA && Array.isArray(report.packet_pairs), 'invalid_source_schema');
  const scoreIds = new Set(), hitIds = new Set(), selected = [];
  for (const row of report.packet_pairs) {
    check(row && [1, 2].includes(row.round_capture) && [0, 1].includes(row.recorded_score_recipient), 'invalid_event_identity');
    const eventPattern = new RegExp(`^R${row.round_capture}-E[0-9]{4}$`);
    check(eventPattern.test(row.score_event_id) && eventPattern.test(row.hit_event_id) &&
      row.score_event_id !== row.hit_event_id, 'invalid_event_ids');
    check(!scoreIds.has(row.score_event_id) && !hitIds.has(row.hit_event_id) &&
      !scoreIds.has(row.hit_event_id) && !hitIds.has(row.score_event_id), 'duplicate_strike_score_pair');
    scoreIds.add(row.score_event_id); hitIds.add(row.hit_event_id);
    check(finite(row.points) && row.points >= 0, 'invalid_awarded_points');
    // Referee/KO-size awards and every nonlocal event are excluded, never relabeled.
    if (row.recorded_score_recipient !== 0 || ![1, 2].includes(row.points)) continue;
    check(row.is_kick === (row.points === 2 ? 1 : 0), 'strike_part_points_mismatch');
    check(finite(row.hit_receipt_minus_score_receipt_ms) && Math.abs(row.hit_receipt_minus_score_receipt_ms) <= 10,
      'strike_score_receipt_pair_not_verified');
    check(row.causally_attributed_request_id === null, 'source_contains_causal_action_label');
    const geometry = row.geometry_at_score_receipt;
    check(geometry && typeof geometry === 'object', 'receipt_geometry_missing');
    selected.push({event_id: row.score_event_id, hit_event_id: row.hit_event_id, round: row.round_capture,
      points: row.points, features: features(geometry.distance_xz_unity_units, geometry.bearing_relative_pelvis_positive_x_deg)});
  }
  selected.sort((a, b) => a.event_id.localeCompare(b.event_id, 'en'));
  return selected;
}

function nearestDiagnostic(feature, samples, scale, omitIndex = -1) {
  check(Array.isArray(feature) && feature.length === 3 && feature.every(finite), 'invalid_feature_vector');
  check(Array.isArray(scale) && scale.length === 3 && scale.every(x => finite(x) && x > 0), 'invalid_feature_scale');
  check(Array.isArray(samples) && samples.length > 0, 'training_samples_missing');
  let minimum = Infinity, nearestIndex = -1;
  for (let i = 0; i < samples.length; i++) {
    check(Array.isArray(samples[i]) && samples[i].length === 3 && samples[i].every(finite), 'invalid_training_sample');
    if (i === omitIndex) continue;
    const squared = feature.reduce((sum, value, j) => sum + ((value - samples[i][j]) / scale[j]) ** 2, 0);
    check(finite(squared), 'nonfinite_standardized_distance');
    if (squared < minimum) { minimum = squared; nearestIndex = i; }
  }
  check(nearestIndex >= 0, 'nearest_training_sample_missing');
  const distance = Math.sqrt(minimum);
  return {nearest_training_index: nearestIndex, nearest_standardized_distance: distance,
    potential: -distance / (1 + distance)};
}

function fitContactPotential(report, sourceSha256) {
  check(typeof sourceSha256 === 'string' && /^[a-f0-9]{64}$/.test(sourceSha256), 'invalid_source_hash');
  const events = selectEvents(report);
  const training = events.filter(row => row.round === 1), holdout = events.filter(row => row.round === 2);
  check(training.length === 12 && holdout.length === 10, 'expected_twelve_training_ten_holdout_strikes');
  const samples = training.map(row => row.features);
  const mean = [0, 1, 2].map(j => samples.reduce((sum, sample) => sum + sample[j], 0) / samples.length);
  const scale = mean.map((value, j) => Math.sqrt(samples.reduce((sum, sample) => sum + (sample[j] - value) ** 2, 0) / samples.length));
  check(scale.every(value => finite(value) && value > 0), 'training_scale_zero_or_nonfinite');
  const body = {
    schema: MODEL_SCHEMA, source_sha256: sourceSha256, fit_round: 1, sample_count: samples.length,
    feature_order: ['distance_unity_numeric', 'cos_bearing_positive_x', 'sin_bearing_positive_x'],
    feature_scale: scale, samples,
    training_event_ids: training.map(row => row.event_id), holdout_event_ids: holdout.map(row => row.event_id),
    feature_scale_estimator: 'population_standard_deviation_training_positives_only',
    bearing_semantics: 'wrap(target_bearing_minus_local_pelvis_positive_x_heading); heading=atan2(world_x,world_z); positive_world_z_toward_positive_x; degrees_to_radians',
    geometry_semantics: 'nearest_preceding_independently_received_pelvis_poses_at_score_receipt; not_server_contact_time',
    unit_mapping: {unity_units_per_candidate_unit: 1, status: 'explicit_experiment_assumption', metre_calibration_verified: false},
    potential_formula: '-sqrt(min(sum(((feature-anchor)/feature_scale)^2)))/(1+sqrt(min(sum(((feature-anchor)/feature_scale)^2))))',
    interpretation: 'positive_geometry_proximity_diagnostic_not_hit_probability',
    action_execution_inferred: false, miss_labels: 0, authentic_physics_parity: false,
  };
  const model = {...body, model_id: sha256(canonicalJson(body))};
  const diagnostic = (row, omitted = -1) => {
    const value = nearestDiagnostic(row.features, samples, scale, omitted);
    return {event_id: row.event_id, hit_event_id: row.hit_event_id, round: row.round, points: row.points,
      features: row.features, ...value, nearest_training_event_id: training[value.nearest_training_index].event_id};
  };
  const diagnostics = {
    schema: 'rek.contact_potential.diagnostics.v1', model_id: model.model_id, source_sha256: sourceSha256,
    fit_round: 1, holdout_round: 2,
    training_leave_one_out: training.map((row, i) => diagnostic(row, i)),
    holdout: holdout.map(row => diagnostic(row)),
    leave_one_out_semantics: 'omit_query_anchor_only; feature_scale_remains_full_round_one_population_scale',
    holdout_used_for_fit: false, accuracy_estimated: false, classifier_trained: false,
    limits: ['positive_only_twelve_training_ten_holdout_events', 'same_two_round_session_not_population_validation',
      'no_action_execution_or_sequence_labels', 'no_negative_examples_or_miss_rate',
      'numeric_unit_mapping_is_assumed_not_metric_calibration', 'receipt_geometry_is_not_authoritative_contact_geometry'],
  };
  return {model, diagnostics};
}

function exportFiles(sourcePath, outputDirectory) {
  const bytes = fs.readFileSync(sourcePath);
  const result = fitContactPotential(JSON.parse(bytes), sha256(bytes));
  const targets = [['contact-potential-model.json', result.model], ['contact-potential-diagnostics.json', result.diagnostics]];
  for (const [name] of targets) check(!fs.existsSync(path.join(outputDirectory, name)), 'output_already_exists');
  fs.mkdirSync(outputDirectory, {recursive: true});
  for (const [name, value] of targets) fs.writeFileSync(path.join(outputDirectory, name), JSON.stringify(value, null, 2) + '\n', {flag: 'wx'});
  return result;
}

module.exports = {features, selectEvents, nearestDiagnostic, fitContactPotential, exportFiles, canonicalJson};
if (require.main === module) {
  try {
    check(process.argv.length === 4, 'usage_contact_potential_data_SOURCE_JSON_NEW_OUTPUT_DIRECTORY');
    const {model, diagnostics} = exportFiles(...process.argv.slice(2));
    console.log(JSON.stringify({model_id: model.model_id, source_sha256: model.source_sha256,
      training: model.sample_count, holdout: diagnostics.holdout.length, feature_scale: model.feature_scale}));
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
