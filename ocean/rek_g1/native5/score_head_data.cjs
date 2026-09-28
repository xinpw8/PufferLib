'use strict';

// Offline request-time features and subsequently observed strike-score outcomes.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {canonicalJson} = require('./contact_potential_data.cjs');
const check = (ok, reason) => { if (!ok) throw Error(reason); };
const finite = x => typeof x === 'number' && Number.isFinite(x);
const vector = (x, n) => Array.isArray(x) && x.length === n && x.every(finite);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const lineOf = row => row?.source?.original_source_line_1based;
const moveTime = row => row.unity_realtime_since_startup;
const FEATURE_ORDER = ['distance_unity_numeric', 'cos_local_positive_x_bearing', 'sin_local_positive_x_bearing',
  'cos_opponent_positive_x_bearing', 'sin_opponent_positive_x_bearing', 'previous_request_elapsed_capped3_div3',
  'preceding_velocity_command_x', 'preceding_velocity_command_y', 'preceding_velocity_command_z', 'prior_requests_last3s_div10',
  ...Array.from({length: 17}, (_, i) => `current_requested_move_${i}`),
  ...Array.from({length: 17}, (_, i) => `previous_requested_move_${i}`)];
const TARGET_ORDER = ['future_local_paired_strike_points', 'future_opponent_paired_strike_points',
  'future_any_local_paired_strike_score', 'future_any_opponent_paired_strike_score'];

function readCapture(filename) {
  const bytes = fs.readFileSync(filename), first = [null, null], last = [null, null], count = [0, 0], maxGap = [0, 0];
  const moves = [], scores = new Map(); let starts = 0, ends = 0, index = 0;
  for (const text of bytes.toString('utf8').split(/\r?\n/)) {
    index++; if (!text) continue;
    const r = JSON.parse(text);
    if (r.event === 'capture_start') {
      starts++;
      check(r.schema === 'rek.private_ai.protocol.v5' && r.scope?.allowed === true &&
        r.scope.local_fighter_index === 0 && r.scope.opponent_is_ai === true, 'capture_scope_not_verified');
    }
    if (r.event === 'capture_end') { ends++; check(r.capture_error_count === 0, 'capture_has_errors'); }
    if (r.event === 'raw_bone_packet') {
      const slot = r.fighter_slot, t = r.monotonic_receipt_time;
      check([0, 1].includes(slot) && finite(t), 'invalid_raw_pose_clock');
      if (first[slot] === null) first[slot] = t;
      if (last[slot] !== null) { check(t >= last[slot], 'pose_clock_regressed'); maxGap[slot] = Math.max(maxGap[slot], t - last[slot]); }
      last[slot] = t; count[slot]++;
    }
    if (r.event === 'outbound_request_projection' && r.message === 'REK_Move')
      moves.push({line: index, time: r.unity_realtime_since_startup, move: r.move_index_wire_uint8});
    if (r.event === 'raw_score_packet') scores.set(index, {time: r.monotonic_receipt_time, decoded: r.decoded});
  }
  check(starts === 1 && ends === 1 && count.every(n => n > 1), 'incomplete_capture');
  return {sha256: hash(bytes), start: Math.max(...first), end: Math.min(...last), pose_count: count,
    maximum_pose_gap_seconds: maxGap, moves, scores};
}

function precedingGeometry(request) {
  const t = moveTime(request), poses = [0, 1].map(slot => request.pose_samples?.[`fighter_${slot}`]?.nearest_preceding);
  for (const pose of poses) check(pose && finite(pose.monotonic_receipt_time) && pose.monotonic_receipt_time <= t &&
    Number.isInteger(lineOf(pose)) && lineOf(pose) < lineOf(request) &&
    pose.source.original_source_file_sha256 === request.source.original_source_file_sha256 &&
    vector(pose.root_position_xyz, 3) && vector(pose.root_quaternion_xyzw_raw, 4), 'preceding_pose_required');
  const dx = poses[1].root_position_xyz[0] - poses[0].root_position_xyz[0];
  const dz = poses[1].root_position_xyz[2] - poses[0].root_position_xyz[2];
  const distance = Math.hypot(dx, dz); check(distance > 0, 'undefined_zero_distance_bearing');
  const heading = pose => {
    const norm = Math.hypot(...pose.root_quaternion_xyzw_raw); check(norm > 0, 'invalid_quaternion');
    const [x, y, z, w] = pose.root_quaternion_xyzw_raw.map(v => v / norm);
    const fx = 1 - 2 * (y * y + z * z), fz = 2 * (x * z - w * y);
    check(Math.hypot(fx, fz) > 0, 'undefined_horizontal_heading'); return Math.atan2(fx, fz);
  };
  const local = Math.atan2(dx, dz) - heading(poses[0]), opponent = Math.atan2(-dx, -dz) - heading(poses[1]);
  return [distance, Math.cos(local), Math.sin(local), Math.cos(opponent), Math.sin(opponent)];
}

function selectScores(report, timeline) {
  check(report?.schema === 'rek.human-observation.contact-signal.v1' && Array.isArray(report.packet_pairs), 'invalid_report_schema');
  const byId = new Map(), used = new Set(), selected = [];
  for (const event of timeline) {
    check(event.schema === 'rek.offline.move-evidence.v1' && [1, 2].includes(event.round_capture) &&
      new RegExp(`^R${event.round_capture}-E[0-9]{4}$`).test(event.timeline_event_id), 'invalid_timeline_schema');
    check(!byId.has(event.timeline_event_id), 'duplicate_timeline_event'); byId.set(event.timeline_event_id, event);
  }
  for (const pair of report.packet_pairs) {
    check([1, 2].includes(pair.round_capture) && [0, 1].includes(pair.recorded_score_recipient), 'invalid_pair_identity');
    check(!used.has(pair.score_event_id) && !used.has(pair.hit_event_id) && pair.score_event_id !== pair.hit_event_id, 'duplicate_score_hit_pair');
    used.add(pair.score_event_id); used.add(pair.hit_event_id);
    if (![1, 2].includes(pair.points)) continue; // Exclude referee/KO-size awards.
    const score = byId.get(pair.score_event_id), hit = byId.get(pair.hit_event_id);
    check(score?.kind === 'raw_score_packet' && hit?.kind === 'raw_hit_packet' &&
      score.round_capture === pair.round_capture && hit.round_capture === pair.round_capture &&
      lineOf(score) === pair.score_original_line && lineOf(hit) === pair.hit_original_line &&
      score.decoded?.fighter_index === pair.recorded_score_recipient && score.decoded.points_awarded === pair.points &&
      pair.is_kick === (pair.points === 2 ? 1 : 0) && pair.causally_attributed_request_id === null &&
      finite(score.monotonic_receipt_time) && finite(hit.monotonic_receipt_time) &&
      Math.abs(hit.monotonic_receipt_time - score.monotonic_receipt_time) <= .01, 'unverified_strike_score_pair');
    selected.push({event_id: pair.score_event_id, round: pair.round_capture, recipient: pair.recorded_score_recipient,
      points: pair.points, time: score.monotonic_receipt_time});
  }
  return selected;
}

function buildDataset(requests, timeline, report, captures) {
  check(Array.isArray(requests) && Array.isArray(timeline), 'invalid_input_arrays');
  const scores = selectScores(report, timeline), rows = [], excluded = [], ids = new Set();
  for (const round of [1, 2]) {
    const capture = captures[round]; check(capture && finite(capture.start) && finite(capture.end) && capture.end > capture.start, 'invalid_coverage');
    const ordered = requests.filter(r => r.round_capture === round).sort((a, b) => lineOf(a) - lineOf(b));
    check(ordered.length === capture.moves.length, 'raw_move_coverage_mismatch');
    for (let i = 0; i < ordered.length; i++) {
      const request = ordered[i], t = moveTime(request), move = request.move_index_wire_uint8, raw = capture.moves[i];
      check(request.schema === 'rek.offline.move-evidence.v1' && request.kind === 'REK_Move' &&
        new RegExp(`^R${round}-M[0-9]{3}$`).test(request.request_id) && !ids.has(request.request_id) &&
        request.local_fighter_slot_from_observed_network_index === 0 &&
        Number.isInteger(move) && move >= 0 && move < 17 && finite(t) &&
        request.source.original_source_file_sha256 === capture.sha256 &&
        raw.line === lineOf(request) && raw.time === t && raw.move === move, 'invalid_request_or_raw_match');
      ids.add(request.request_id);
      if (i > 0) check(t >= moveTime(ordered[i - 1]) && lineOf(request) > lineOf(ordered[i - 1]), 'request_clock_regressed');
      if (t - 3 < capture.start || t + 3 > capture.end) {
        excluded.push({request_id: request.request_id, reason: t - 3 < capture.start ? 'incomplete_history' : 'incomplete_future'}); continue;
      }
      const geometry = precedingGeometry(request), movement = request.last_preceding_movement_input;
      check(movement?.kind === 'REK_Input' && finite(movement.unity_realtime_since_startup) &&
        movement.unity_realtime_since_startup <= t && Number.isInteger(lineOf(movement)) && lineOf(movement) < lineOf(request) &&
        movement.source.original_source_file_sha256 === capture.sha256 && vector(movement.velocity_command_xyz, 3), 'preceding_movement_required');
      const previous = i ? ordered[i - 1] : null;
      const elapsed = previous ? Math.min(3, t - moveTime(previous)) / 3 : 1;
      const priorCount = ordered.slice(0, i).filter(r => moveTime(r) >= t - 3 && moveTime(r) < t).length;
      const features = [...geometry, elapsed, ...movement.velocity_command_xyz, priorCount / 10,
        ...Array.from({length: 17}, (_, j) => +(j === move)),
        ...Array.from({length: 17}, (_, j) => +(previous !== null && j === previous.move_index_wire_uint8))];
      check(vector(features, 44), 'invalid_features');
      const future = scores.filter(s => s.round === round && s.time > t && s.time <= t + 3);
      const points = [0, 1].map(slot => future.filter(s => s.recipient === slot).reduce((sum, s) => sum + s.points, 0));
      rows.push({request_id: request.request_id, round, features, targets: [...points, +(points[0] > 0), +(points[1] > 0)],
        future_event_ids: future.map(s => s.event_id)});
    }
  }
  check(ids.size === requests.length, 'unknown_request_round');
  const training = rows.filter(r => r.round === 1), holdout = rows.filter(r => r.round === 2);
  check(training.length > 1 && holdout.length > 0, 'insufficient_split_rows');
  const mean = FEATURE_ORDER.map((_, j) => training.reduce((sum, r) => sum + r.features[j], 0) / training.length);
  const populationStddev = mean.map((m, j) => Math.sqrt(training.reduce((sum, r) => sum + (r.features[j] - m) ** 2, 0) / training.length));
  const scale = populationStddev.map(x => x === 0 ? 1 : x);
  check(mean.every(finite) && scale.every(x => finite(x) && x > 0), 'invalid_training_scaler');
  return {training, holdout, excluded, scores, scaler: {mean, scale, population_stddev: populationStddev,
    constant_feature_indices: populationStddev.flatMap((x, i) => x === 0 ? [i] : []),
    estimator: 'round_one_population_standard_deviation; constant_features_scale_one'}};
}

function binaryDataset(dataset) {
  const rows = [...dataset.training, ...dataset.holdout], bytes = Buffer.alloc(24 + 44 * 8 + rows.length * 46 * 4);
  bytes.write('REKSHP1\0', 0, 'ascii'); [1, 44, dataset.training.length, dataset.holdout.length].forEach((v, i) => bytes.writeUInt32LE(v, 8 + i * 4));
  let offset = 24;
  for (const value of [...dataset.scaler.mean, ...dataset.scaler.scale, ...rows.flatMap(r => [...r.features, ...r.targets.slice(2)])]) {
    check(finite(Math.fround(value)), 'float32_overflow'); bytes.writeFloatLE(value, offset); offset += 4;
  }
  return bytes;
}

function exportFiles(requestFile, timelineFile, reportFile, rawRoundOne, rawRoundTwo, outputDirectory) {
  const sourceFiles = [requestFile, timelineFile, reportFile], bytes = sourceFiles.map(f => fs.readFileSync(f));
  const jsonl = b => b.toString('utf8').trim().split(/\r?\n/).map(s => JSON.parse(s));
  const captures = {1: readCapture(rawRoundOne), 2: readCapture(rawRoundTwo)};
  const timeline = jsonl(bytes[1]);
  for (const e of timeline.filter(r => r.kind === 'raw_score_packet')) {
    const capture = captures[e.round_capture], raw = capture?.scores.get(lineOf(e));
    check(raw && e.source.original_source_file_sha256 === capture.sha256 && raw.time === e.monotonic_receipt_time &&
      raw.decoded.fighter_index === e.decoded.fighter_index && raw.decoded.points_awarded === e.decoded.points_awarded, 'raw_score_mismatch');
  }
  const data = buildDataset(jsonl(bytes[0]), timeline, JSON.parse(bytes[2]), captures);
  const artifacts = new Map([['train.tsv', Buffer.from(data.training.map(r => [...r.features, ...r.targets].join('\t')).join('\n') + '\n')],
    ['holdout.tsv', Buffer.from(data.holdout.map(r => [...r.features, ...r.targets].join('\t')).join('\n') + '\n')],
    ['score-head-data.bin', binaryDataset(data)]]);
  const splitSummary = (rows, round) => ({round, sample_count: rows.length, request_ids: rows.map(r => r.request_id),
    positive_rows_by_recipient: [0, 1].map(j => rows.reduce((sum, r) => sum + r.targets[j + 2], 0)),
    point_target_sum_with_overlapping_windows: [0, 1].map(j => rows.reduce((sum, r) => sum + r.targets[j], 0)),
    unique_future_event_ids: [...new Set(rows.flatMap(r => r.future_event_ids))].sort(),
    unique_future_event_count_by_recipient: [0, 1].map(j => new Set(rows.flatMap(r => r.future_event_ids)
      .filter(id => data.scores.find(s => s.event_id === id).recipient === j)).size),
    future_event_memberships_including_repeats: rows.reduce((sum, r) => sum + r.future_event_ids.length, 0),
    row_future_event_ids: rows.map(r => r.future_event_ids)});
  const manifest = {schema: 'rek.score_head.request_windows.v1', feature_count: 44, feature_order: FEATURE_ORDER, target_order: TARGET_ORDER,
    source_sha256: {named_requests: hash(bytes[0]), timeline: hash(bytes[1]), contact_report: hash(bytes[2]),
      raw_round_one: captures[1].sha256, raw_round_two: captures[2].sha256},
    source_domain: 'authentic_client_human_vs_ai_observation', authentic_physics_parity: false,
    horizon_seconds: 3, history_seconds: 3, future_interval: '(request_time,request_time+3]', history_count_interval: '[request_time-3,request_time)',
    fit_round: 1, holdout_round: 2, scaler: data.scaler,
    training: splitSummary(data.training, 1), holdout: splitSummary(data.holdout, 2), excluded_requests: data.excluded,
    coverage: [1, 2].map(round => ({round, first_shared_pose_receipt_time: captures[round].start,
      last_shared_pose_receipt_time: captures[round].end, pose_counts: captures[round].pose_count,
      maximum_pose_gap_seconds: captures[round].maximum_pose_gap_seconds})),
    coverage_semantics: 'conservative_interval_shared_by_both_received_pose_streams; bounded_client_observation_not_server_event_completeness',
    geometry_semantics: 'derived_only_from_separate_nearest_preceding_pelvis_poses; normalized_xyzw_quaternion_local_positive_x',
    bearing_sign: 'atan2(target_x,target_z)-atan2(forward_x,forward_z); opponent_uses_reverse_displacement; cos_sin_in_radians',
    units: {distance: 'Unity_world_numeric_units', candidate_units_per_unity_unit: 1, mapping: 'explicit_experiment_assumption', metric_calibration_verified: false},
    feature_semantics: {request: 'issued_request_not_executed_action', previous_missing: 'one_hot_all_zero_and_elapsed_one',
      movement: 'preceding_REK_Input_velocity_command_xyz_unmodified_not_measured_velocity', request_count: 'prior_requests_only_divided_by_ten_not_clipped'},
    target_semantics: 'sum_observed_unique_paired_1_or_2_point_score_receipts_by_recipient_in_next_three_seconds; exclude_five_point_awards',
    binary_format: {magic: 'REKSHP1\\0', byte_order: 'little_endian', header_bytes: 24,
      header_u32: ['version=1', 'features=44', 'train_count', 'holdout_count'], scaler: 'float32_mean44_then_scale44',
      rows: 'train_then_holdout; float32_raw_features44_then_binary_labels2; apply_scaler_once'},
    tsv_format: 'no_header; raw_features44_then_point_targets2_then_binary_targets2; one_numeric_row_per_request',
    nonindependent_overlapping_windows: true, future_information_in_features: false, executed_move_inferred: false,
    failed_attack_labels: false, causal_effect_estimated: false, heldout_used_for_preprocessing: false,
    limitations: ['two_rounds_same_session', 'only_eight_observed_move_ids_of_seventeen', 'local_human_requests_only_opponent_requests_unobserved',
      'three_second_received_score_outcomes_not_server_contact_times', 'overlapping_windows_repeat_events', 'zero_outcome_not_failed_attack',
      'associational_prediction_not_action_causality_or_training_reward_validation'],
    artifacts: [...artifacts].map(([name, b]) => ({name, bytes: b.length, sha256: hash(b)}))};
  manifest.dataset_id = hash(canonicalJson(manifest));
  artifacts.set('manifest.json', Buffer.from(JSON.stringify(manifest, null, 2) + '\n'));
  for (const name of artifacts.keys()) check(!fs.existsSync(path.join(outputDirectory, name)), 'output_already_exists');
  fs.mkdirSync(outputDirectory, {recursive: true});
  for (const [name, b] of artifacts) fs.writeFileSync(path.join(outputDirectory, name), b, {flag: 'wx'});
  return manifest;
}

module.exports = {FEATURE_ORDER, TARGET_ORDER, precedingGeometry, selectScores, buildDataset, binaryDataset, readCapture, exportFiles};
if (require.main === module) {
  try {
    check(process.argv.length === 8, 'usage_score_head_data_REQUESTS_TIMELINE_REPORT_RAW1_RAW2_NEW_OUTPUT_DIRECTORY');
    const m = exportFiles(...process.argv.slice(2));
    console.log(JSON.stringify({dataset_id: m.dataset_id, training: m.training.sample_count, holdout: m.holdout.sample_count,
      training_positive: m.training.positive_rows_by_recipient, holdout_positive: m.holdout.positive_rows_by_recipient,
      excluded: m.excluded_requests.length, unique_training_events: m.training.unique_future_event_count_by_recipient,
      unique_holdout_events: m.holdout.unique_future_event_count_by_recipient}));
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
