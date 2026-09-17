#!/usr/bin/env node
'use strict';
// Fixed receipt-time windows, no per-event lag selection or model fitting.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const windows = [0, 0.05, 0.1, 0.25];
const digest = b => crypto.createHash('sha256').update(b).digest('hex');
const keys = (o, expected) => assert.deepEqual(Object.keys(o).sort(), expected.split(' ').sort());
const finite = n => assert(typeof n === 'number' && Number.isFinite(n));
const integer = n => { finite(n); assert(Number.isSafeInteger(n) && n >= 0); };
const vector = (a, length, check) => { assert(Array.isArray(a) && a.length === length); a.forEach(check); };
const mean = a => a.reduce((s, n) => s + n, 0) / a.length;
function run(input, summaryFile, output) {
  assert(!fs.existsSync(output), 'output must be new');
  const inputBytes = fs.readFileSync(input), summaryBytes = fs.readFileSync(summaryFile);
  const summary = JSON.parse(summaryBytes), rows = inputBytes.toString('utf8').trim().split(/\r?\n/).map(s => JSON.parse(s));
  const fullTargets = summary.target_primitives === 9;
  assert(fullTargets || summary.target_primitives === undefined);
  if (fullTargets) assert.equal(summary.core_target_primitives, 3);
  assert.equal(summary.event, 'pose_contact_replay');
  assert.equal(summary.rounds, 2); assert.equal(summary.paired_strikes, 33);
  assert.equal(summary.authoritative_contact_time, false);
  assert.equal(summary.negative_outcome_labels, 0); assert.equal(summary.round2_fit, false);
  assert.equal(summary.authentic_parity, false); assert.equal(summary.canned_motion_used, false);
  assert.equal(summary.contact_execution, 'CUDA_static_overlap');
  assert.equal(summary.cpu_physics, false); assert.equal(summary.python_runtime, false);
  assert.deepEqual(summary.unpaired_hits, [0, 0]); assert.deepEqual(summary.unpaired_scores, [4, 2]);
  assert.deepEqual(summary.lag_scan_seconds, [-0.25, 0.25]);
  assert.equal(summary.lag_step_seconds, 1 / 120); assert.equal(summary.max_pose_age_seconds, 0.075);
  assert.equal(summary.model_sha256, '6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa');
  assert.deepEqual(summary.capture_sha256, ['547cec42f700e97b9594f8d2c6df88f966052c0e0e5ce7395c4862b619fb6d7f', 'ec55c32a6e2272a8e7656d73260ca35876be2cd6c8d8d6fe8c9dfe77cd25a309']);
  const grouped = new Map(), links = [];
  let queries = 0;
  for (const row of rows) {
    if (row.event === 'link_length_check') {
      keys(row, 'event round side bone model_length_m mean_ratio min_ratio max_ratio samples');
      assert([1, 2].includes(row.round)); assert([0, 1].includes(row.side));
      assert(/^[a-z_]+$/.test(row.bone));
      for (const k of ['model_length_m', 'mean_ratio', 'min_ratio', 'max_ratio']) finite(row[k]);
      assert(row.model_length_m > 0 && row.min_ratio >= 0 && row.max_ratio >= row.min_ratio);
      assert(row.mean_ratio >= row.min_ratio - 1e-8 && row.mean_ratio <= row.max_ratio + 1e-8);
      integer(row.samples); assert(row.samples > 0); links.push(row); continue;
    }
    keys(row, 'event round hit_sequence score_sequence scorer points is_kick receipt_lag_seconds pose_sequences pose_unity_frames pose_age_seconds primitive_pair_mask part_compatible_pair_mask legacy_pair_count legacy_part_compatible_count hit_point_striker_distance_m hit_point_target_distance_m' + (fullTargets ? ' all_target_pair_count all_target_part_compatible_count target_part_compatible_counts' : ''));
    assert.equal(row.event, 'pose_contact_query');
    for (const k of ['round', 'hit_sequence', 'score_sequence', 'scorer', 'points', 'is_kick', 'primitive_pair_mask', 'part_compatible_pair_mask', 'legacy_pair_count', 'legacy_part_compatible_count']) integer(row[k]);
    assert([1, 2].includes(row.round)); assert([0, 1].includes(row.scorer)); assert([0, 1].includes(row.is_kick));
    assert.equal(row.points, row.is_kick ? 2 : 1);
    vector(row.pose_sequences, 2, integer); vector(row.pose_unity_frames, 2, integer);
    vector(row.pose_age_seconds, 2, n => { finite(n); assert(n >= 0 && n <= .075); });
    for (const k of ['hit_point_striker_distance_m', 'hit_point_target_distance_m']) { finite(row[k]); assert(row[k] >= 0); }
    finite(row.receipt_lag_seconds); assert(Math.abs(row.receipt_lag_seconds) <= .25);
    assert(Math.abs(row.receipt_lag_seconds * 120 - Math.round(row.receipt_lag_seconds * 120)) < 1e-6);
    assert(row.primitive_pair_mask < 2 ** 36 && row.part_compatible_pair_mask < 2 ** 36);
    assert.equal(BigInt(row.part_compatible_pair_mask) & BigInt(row.primitive_pair_mask), BigInt(row.part_compatible_pair_mask));
    if (fullTargets) {
      integer(row.all_target_pair_count); integer(row.all_target_part_compatible_count);
      assert(row.all_target_pair_count <= 108 && row.all_target_part_compatible_count <= row.all_target_pair_count);
      vector(row.target_part_compatible_counts, 9, n => { integer(n); assert(n <= 12); });
      assert.equal(row.target_part_compatible_counts.reduce((a, b) => a + b, 0), row.all_target_part_compatible_count);
      const pop = n => [...BigInt(n).toString(2)].filter(c => c === '1').length;
      assert(pop(row.primitive_pair_mask) <= row.all_target_pair_count);
      assert.equal(pop(row.part_compatible_pair_mask), row.target_part_compatible_counts.slice(0, 3).reduce((a, b) => a + b, 0));
    }
    assert(row.legacy_pair_count <= 18 && row.legacy_part_compatible_count <= row.legacy_pair_count);
    const key = `${row.round}:${row.hit_sequence}:${row.score_sequence}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row); queries++;
  }
  assert.equal(queries, summary.queries); assert.equal(grouped.size, 33);
  const events = [...grouped.values()].map(samples => {
    const first = samples[0], lags = new Set();
    for (const s of samples) {
      for (const k of ['round', 'hit_sequence', 'score_sequence', 'scorer', 'points', 'is_kick']) assert.equal(s[k], first[k]);
      assert(!lags.has(s.receipt_lag_seconds)); lags.add(s.receipt_lag_seconds);
    }
    assert(lags.has(0));
    const measures = windows.map(window => {
      const subset = samples.filter(s => Math.abs(s.receipt_lag_seconds) <= window + 1e-9);
      assert(subset.length > 0);
      return {
        receipt_window_half_width_seconds: window, queries: subset.length,
        primitive_overlap: subset.some(s => fullTargets ? s.all_target_pair_count > 0 : s.primitive_pair_mask > 0),
        part_compatible_primitive_overlap: subset.some(s => fullTargets ? s.all_target_part_compatible_count > 0 : s.part_compatible_pair_mask > 0),
        ...(fullTargets ? {
          core_only_primitive_overlap: subset.some(s => s.primitive_pair_mask > 0),
          core_only_part_compatible_primitive_overlap: subset.some(s => s.part_compatible_pair_mask > 0),
          target_part_compatible_overlaps: Array.from({ length: 9 }, (_, j) => subset.some(s => s.target_part_compatible_counts[j] > 0)),
        } : {}),
        legacy_sphere_overlap: subset.some(s => s.legacy_pair_count > 0),
        part_compatible_legacy_sphere_overlap: subset.some(s => s.legacy_part_compatible_count > 0),
        minimum_hit_point_striker_distance_model_m: Math.min(...subset.map(s => s.hit_point_striker_distance_m)),
        minimum_hit_point_target_distance_model_m: Math.min(...subset.map(s => s.hit_point_target_distance_m)),
      };
    });
    return { round: first.round, hit_sequence: first.hit_sequence, score_sequence: first.score_sequence,
      scorer: first.scorer, points: first.points, is_kick: first.is_kick, windows: measures };
  });
  const aggregates = [];
  for (const round of [null, 1, 2]) for (const scorer of [null, 0, 1]) {
    const selected = events.filter(e => (round === null || e.round === round) && (scorer === null || e.scorer === scorer));
    if (!selected.length) continue;
    aggregates.push({ round, scorer, observed_scoring_events: selected.length, windows: windows.map((window, i) => {
      const values = selected.map(e => e.windows[i]);
      const out = { receipt_window_half_width_seconds: window };
      for (const k of ['primitive_overlap', 'part_compatible_primitive_overlap', 'legacy_sphere_overlap', 'part_compatible_legacy_sphere_overlap']) out[k + '_events'] = values.filter(v => v[k]).length;
      if (fullTargets) {
        for (const k of ['core_only_primitive_overlap', 'core_only_part_compatible_primitive_overlap']) out[k + '_events'] = values.filter(v => v[k]).length;
        out.newly_covered_by_hip_targets_events = values.filter(v => v.part_compatible_primitive_overlap && !v.core_only_part_compatible_primitive_overlap).length;
        out.target_part_compatible_overlap_events = Array.from({ length: 9 }, (_, j) => values.filter(v => v.target_part_compatible_overlaps[j]).length);
      }
      return out;
    }) });
  }
  const linkSummary = [1, 2].flatMap(round => [0, 1].map(side => {
    const values = links.filter(r => r.round === round && r.side === side); assert(values.length > 0);
    return { round, side, tested_links: values.length, minimum_observed_length_ratio: Math.min(...values.map(r => r.min_ratio)),
      maximum_observed_length_ratio: Math.max(...values.map(r => r.max_ratio)), mean_of_per_link_mean_ratios: mean(values.map(r => r.mean_ratio)) };
  }));
  const report = {
    schema: fullTargets ? 'rek.recorded_pose_contact_diagnostic.v2' : 'rek.recorded_pose_contact_diagnostic.v1', model_sha256: summary.model_sha256,
    capture_sha256: summary.capture_sha256, query_sha256: digest(inputBytes), run_summary_sha256: digest(summaryBytes),
    paired_strikes: 33, queries, pose_counts: summary.pose_counts, unpaired_five_point_awards: summary.unpaired_scores,
    contact_execution: 'CUDA_static_overlap', cpu_physics: false, python_runtime: false,
    canned_motion_used: false, fit_on_human_rounds: false, round2_fit: false,
    pose_sampling: 'nearest_preceding_client_receipt_pose_per_fighter', max_pose_age_seconds: .075,
    lag_windows_selected_before_results: true, authoritative_contact_time: false, authentic_parity: false,
    negative_outcome_labels: 0, events, aggregates, link_length_checks: linkSummary,
    ...(fullTargets ? { target_primitives: 9, legacy_comparison_target_primitives: 3,
      target_names: ['mjgeom_3021', 'mjgeom_3285', 'mjgeom_3064', 'mjgeom_3337', 'mjgeom_3141', 'mjgeom_3399', 'mjgeom_3024', 'mjgeom_3062', 'mjgeom_3406'],
      target_zones: [3, 2, 2, 12, 12, 12, 13, 13, 13],
      target_contract: 'g1_scoring_bodyzones_1_2_3_12_13_v1' } : {}),
    limitations: [
      'Packet receipt times do not identify the server collision time; each fighter pose may have independent delay.',
      'Overlap at a nearby receipt time is a geometric diagnostic, not a reconstructed accepted scoring event.',
      'No overlap at sampled receipt poses is not a miss label or proof that the original contact was invalid.',
      'The is_kick flag limits limb category only; move attribution remains ambiguous.',
      'Distances use recovered model coordinates without fitted spatial scaling; independent physical metre calibration is absent.',
      'Collision eligibility masks, contact velocity, entry history, balance and scoring gates are not tested here.',
      'Point-to-striker and point-to-target minima within a window may occur at different poses.',
    ],
  };
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ paired_strikes: 33, queries, aggregates, link_length_checks: linkSummary }));
}
if (require.main === module) {
  try { assert.equal(process.argv.length, 5); run(...process.argv.slice(2)); }
  catch { console.error('Pose summary rejected: invalid or incomplete diagnostic inputs.'); process.exitCode = 1; }
}
module.exports = { run };
