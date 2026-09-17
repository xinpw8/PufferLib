#!/usr/bin/env node
'use strict';

// Offline context comparison. Counter changes and observed motion do not prove
// a particular collider contact, native attack identity, or rejected hit.
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const crypto = require('node:crypto');
const { LIMBS } = require('./audit_passive_defender.cjs');
const PI = Math.PI;
function check(ok, message) { if (!ok) throw new Error(message); }
function finite(n) { return typeof n === 'number' && Number.isFinite(n); }
function wrap(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }
function heading(q) {
  check(Array.isArray(q) && q.length === 4 && q.every(finite), 'invalid_root_quaternion');
  // Same Unity XYZW -> MuJoCo WXYZ convention as encode_live.cpp:55.
  const length = Math.hypot(...q); check(length > 0, 'zero_root_quaternion');
  const [x, z, y, nw] = q.map(n => n / length), w = -nw;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}
function quantiles(values) {
  const a = values.filter(finite).sort((x, y) => x - y);
  if (!a.length) return { count: 0, min: null, median: null, p90: null, max: null };
  const at = fraction => { const x = fraction * (a.length - 1), i = Math.floor(x); return a[i] + (a[Math.min(i + 1, a.length - 1)] - a[i]) * (x - i); };
  return { count: a.length, min: a[0], median: at(0.5), p90: at(0.9), max: a[a.length - 1] };
}
function relativePose(s) {
  const a = s.actor.root_position_world_unity_xyz_m, o = s.opponent.root_position_world_unity_xyz_m;
  check([a, o].every(v => Array.isArray(v) && v.length === 3 && v.every(finite)), 'invalid_root_position');
  const h = heading(s.opponent.root_rotation_unity_xyzw), ah = heading(s.actor.root_rotation_unity_xyzw);
  const dx = a[0] - o[0], dy = a[2] - o[2];
  return { sample_index: s.sample_index, source_line: s.source_line, qpc_ticks: s.qpc_ticks,
    utc: s.utc, time_remaining_seconds: s.time_remaining_seconds,
    gap_horizontal_m: Math.hypot(dx, dy),
    defender_relative_to_opponent_heading_m: { forward: Math.cos(h) * dx + Math.sin(h) * dy,
      lateral: -Math.sin(h) * dx + Math.cos(h) * dy },
    opponent_bearing_to_defender_rad: wrap(Math.atan2(dy, dx) - h),
    opponent_heading_rad: h, defender_heading_rad: ah, heading_difference_rad: wrap(ah - h),
    defender_root_height_m: a[1], opponent_root_height_m: o[1],
    defender_tilt_degrees: s.actor.tilt_degrees, opponent_tilt_degrees: s.opponent.tilt_degrees };
}
function timeDifference(a, b) {
  check(a.segment === b.segment && a.qpc_frequency_hz === b.qpc_frequency_hz, 'cross_segment_time');
  return (a.qpc_ticks - b.qpc_ticks) / a.qpc_frequency_hz;
}
function describeWindow(samples) {
  check(samples.length > 0, 'empty_window');
  const poses = samples.map(relativePose);
  const limbs = {};
  for (const name of Object.keys(LIMBS)) {
    const valid = samples.filter(s => finite(s.opponent.limbs[name]?.speed_m_s));
    let peak = null;
    for (const s of valid) if (!peak || s.opponent.limbs[name].speed_m_s > peak.opponent.limbs[name].speed_m_s) peak = s;
    limbs[name] = { world_speed_m_s: quantiles(valid.map(s => s.opponent.limbs[name].speed_m_s)),
      root_relative_speed_m_s: quantiles(valid.map(s => s.opponent.limbs[name].root_relative_speed_m_s)),
      peak_sample_index: peak?.sample_index ?? null,
      peak_time_remaining_seconds: peak?.time_remaining_seconds ?? null,
      peak_qpc_ticks: peak?.qpc_ticks ?? null };
  }
  const first = samples[0], last = samples[samples.length - 1];
  return { samples: samples.length, start_sample_index: first.sample_index, end_sample_index: last.sample_index,
    start_source_line: first.source_line, end_source_line: last.source_line,
    observed_span_seconds: timeDifference(last, first),
    gap_horizontal_m: quantiles(poses.map(p => p.gap_horizontal_m)),
    absolute_opponent_bearing_rad: quantiles(poses.map(p => Math.abs(p.opponent_bearing_to_defender_rad))),
    defender_tilt_degrees: quantiles(poses.map(p => p.defender_tilt_degrees)),
    opponent_tilt_degrees: quantiles(poses.map(p => p.opponent_tilt_degrees)), limbs };
}
function eligible(s) {
  return s.round_active === true && s.stream_active === true && s.neutral_check_after_warmup === true
    && s.command.desired_action === 1 && s.command.velocity_zero === true;
}
function analyze(samples, events, options = {}) {
  const o = { preSeconds: 1, exclusionSeconds: 1, motionSpeedMps: 1.75, ...options };
  for (const n of Object.values(o)) check(finite(n) && n > 0, 'invalid_option');
  const bySegment = new Map(), byIndex = new Map();
  for (const s of samples) {
    check(s.event === 'kinematics' && Number.isSafeInteger(s.sample_index)
      && Number.isSafeInteger(s.qpc_ticks) && finite(s.qpc_frequency_hz) && s.qpc_frequency_hz > 0, 'invalid_kinematics');
    check(!byIndex.has(s.sample_index), 'duplicate_sample_index');
    byIndex.set(s.sample_index, s);
    if (!bySegment.has(s.segment)) bySegment.set(s.segment, []);
    const segment = bySegment.get(s.segment);
    check(!segment.length || s.qpc_ticks > segment[segment.length - 1].qpc_ticks, 'nonmonotonic_segment');
    segment.push(s);
  }
  const cases = [];
  for (const event of events) {
    const center = byIndex.get(event.sample_index);
    check(center && center.qpc_ticks === event.qpc_ticks && center.segment === event.segment, 'score_event_sample_mismatch');
    check(event.score_delta && finite(event.score_delta.actor) && finite(event.score_delta.opponent), 'invalid_score_delta');
    const prior = bySegment.get(center.segment).filter(s => {
      const dt = timeDifference(center, s); return dt >= 0 && dt <= o.preSeconds;
    });
    const snapshots = [1, 0.5, 0.2, 0.1, 0].filter(seconds => seconds <= o.preSeconds).map(seconds => {
      const selected = prior.filter(s => timeDifference(center, s) >= seconds).at(-1);
      return { requested_seconds_before_update: seconds, actual_seconds_before_update: selected ? timeDifference(center, selected) : null,
        pose: selected ? relativePose(selected) : null };
    });
    const d = event.score_delta;
    const kind = d.actor < 0 || d.opponent < 0 ? 'counter_decrease'
      : d.actor === 5 || d.opponent === 5 ? 'five_point_update_context'
        : d.actor === 0 && (d.opponent === 1 || d.opponent === 2) ? 'opponent_one_or_two_point_update_context'
          : 'other_counter_update_context';
    cases.push({ event_index: event.index, kind, score_delta: d, time_remaining_seconds: center.time_remaining_seconds,
      qpc_ticks: center.qpc_ticks, qpc_frequency_hz: center.qpc_frequency_hz, utc: center.utc,
      round_identity_sha256: center.round_identity_sha256, sample_index: center.sample_index,
      neutral_eligible_samples: prior.filter(eligible).length,
      pre_window: describeWindow(prior), snapshots });
  }
  // Nonoverlapping fixed-duration bins. Exclude every bin with a counter update
  // within one second on either side; this is a comparison set, not missed hits.
  const controls = [];
  for (const [segmentId, segment] of bySegment) {
    const start = segment[0].qpc_ticks, bins = new Map();
    for (const s of segment) {
      const bin = Math.floor(((s.qpc_ticks - start) / s.qpc_frequency_hz) / o.preSeconds);
      if (!bins.has(bin)) bins.set(bin, []); bins.get(bin).push(s);
    }
    for (const [bin, rows] of bins) {
      if (rows.length < 2 || rows.some(s => !eligible(s))) continue;
      const first = rows[0], last = rows[rows.length - 1];
      if (timeDifference(last, first) < o.preSeconds * 0.8) continue;
      if (events.some(e => e.segment === segmentId
        && (e.qpc_ticks - first.qpc_ticks) / first.qpc_frequency_hz >= -o.exclusionSeconds
        && (e.qpc_ticks - last.qpc_ticks) / first.qpc_frequency_hz <= o.exclusionSeconds)) continue;
      const context = describeWindow(rows);
      if (!Object.values(context.limbs).some(l => l.world_speed_m_s.max > o.motionSpeedMps)) continue;
      controls.push({ segment: segmentId, bin, label: 'no_score_observed', start_pose: relativePose(first), end_pose: relativePose(last), context });
    }
  }
  const compact = windows => ({ count: windows.length,
    median_gap_per_window_m: quantiles(windows.map(w => w.gap_horizontal_m.median)),
    median_absolute_bearing_per_window_rad: quantiles(windows.map(w => w.absolute_opponent_bearing_rad.median)),
    peak_limb_speed_per_window_m_s: quantiles(windows.map(w => Math.max(...Object.values(w.limbs).map(l => l.world_speed_m_s.max ?? 0)))) });
  return { schema: 'rek.passive_contact_context_comparison.v1', options: o,
    total_samples: samples.length, score_contexts: cases,
    comparison: { ordinary_sized_opponent_updates: compact(cases.filter(c => c.kind === 'opponent_one_or_two_point_update_context').map(c => c.pre_window)),
      unscored_motion_windows: compact(controls.map(c => c.context)) }, unscored_motion_windows: controls,
    limits: [
      'These g1_policy_state/audit inputs supply client-rendered pose and finite-difference speed, not contact pairs, collision normals, impulse or native relative contact speed. Separate native recorder packets must be joined independently.',
      'The score-update timestamp is when the client counter changed; native contact time and replication delay are unknown.',
      'One/two/five-point update labels identify counter sizes, not a hit type, strike limb or knockout cause.',
      'The peak limb in a preceding window is a motion observation, not attribution of the score.',
      'No-score windows may contain contacts or meaningful physical interactions that did not change the observed counter.',
      'Root-relative headings use exactly the live encoder quaternion conversion; they do not establish native aiming intent.',
      'No classifier, inferred hitbox or acceptance rule is fitted. Candidate replay poses cannot guarantee repeatable contacts.',
    ] };
}
async function loadJsonl(file, consume) {
  const before = fs.statSync(file), hash = crypto.createHash('sha256');
  const input = fs.createReadStream(file); input.on('data', bytes => hash.update(bytes));
  const lines = readline.createInterface({ input, crlfDelay: Infinity }); let count = 0;
  try { for await (const line of lines) { count++; if (line.trim()) consume(JSON.parse(line), count); } }
  finally { lines.close(); input.destroy(); }
  const after = fs.statSync(file);
  check(before.size === after.size && before.mtimeMs === after.mtimeMs, 'input_changed');
  return { path: file, bytes: after.size, lines: count, sha256: hash.digest('hex') };
}
async function run(roundDirectory, destination) {
  const root = path.resolve(roundDirectory), out = path.resolve(destination);
  check(!fs.existsSync(out), 'output_already_exists');
  const samples = [], events = [], inputs = [];
  inputs.push(await loadJsonl(path.join(root, 'audit', 'kinematics.jsonl'), s => samples.push(s)));
  inputs.push(await loadJsonl(path.join(root, 'audit', 'score_events.jsonl'), e => events.push(e)));
  const report = analyze(samples, events);
  const availability = { samples: 0, fighters: [{}, {}] };
  const increment = (o, key, yes) => { o[key] = (o[key] || 0) + Number(yes); };
  inputs.push(await loadJsonl(path.join(root, 'trial', 'g1_policy_state.jsonl'), s => {
    if (s.event !== 'g1_policy_state') return;
    availability.samples++;
    for (let side = 0; side < 2; side++) {
      const f = s.fighters?.[side] || {}, a = availability.fighters[side];
      increment(a, 'root_pose_available', Array.isArray(f.root_position_xyz) && Array.isArray(f.root_rotation_xyzw));
      increment(a, 'bone_positions_available', Array.isArray(f.bone_world_positions_xyz));
      increment(a, 'last_hit_available', f.last_hit != null);
      increment(a, 'runner_move_index_available', f.runner?.current_move_index != null);
      increment(a, 'runner_motion_name_available', f.runner?.current_motion_name != null);
      increment(a, 'runner_frame_nonzero', finite(f.runner?.motion_frame_index) && f.runner.motion_frame_index !== 0);
      increment(a, 'native_joint_positions_available', f.joint_positions != null);
    }
  }));
  report.source_field_availability_by_network_slot = availability;
  report.inputs = inputs;
  fs.writeFileSync(out, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  return report;
}
module.exports = { heading, wrap, quantiles, relativePose, describeWindow, analyze, run };
if (require.main === module) {
  if (process.argv.length !== 4) { process.stderr.write('Usage: node compare_passive_contacts.cjs ROUND_DIRECTORY NEW_REPORT_JSON\n'); process.exitCode = 2; }
  else run(process.argv[2], process.argv[3]).then(r => process.stdout.write(JSON.stringify({
    report: path.resolve(process.argv[3]), samples: r.total_samples, scores: r.score_contexts.length,
    comparison: r.comparison, source_fields: r.source_field_availability_by_network_slot }) + '\n'))
    .catch(e => { process.stderr.write(e.message + '\n'); process.exitCode = 2; });
}
