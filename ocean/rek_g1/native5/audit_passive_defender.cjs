#!/usr/bin/env node
'use strict';

// Offline only: reads saved bridge JSONL. No simulator, relay, or game control.
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const crypto = require('node:crypto');
const { once } = require('node:events');

const LIMBS = Object.freeze({
  left_foot: 'left_ankle_roll_link', right_foot: 'right_ankle_roll_link',
  left_hand: 'left_wrist_yaw_link', right_hand: 'right_wrist_yaw_link',
  left_shin: 'left_knee_link', right_shin: 'right_knee_link',
});
const DEFAULTS = Object.freeze({ warmupSeconds: 1, windowSeconds: 1,
  motionSpeedMps: 1.75, maxGapSeconds: 0.25 });
const limits = [
  'Client-rendered/replicated observations, not authoritative server contact state.',
  'Limb positions are named bone origins, not collider surfaces or contact points.',
  'Variable-QPC finite differences measure displacement; impacts, interpolation and position resets can all contribute.',
  'Diagnostic speed-threshold intervals are motion, not identified attacks.',
  'no_score_observed does not mean a missed or rejected attack; native active-clip/apex and contact-enter state are unavailable.',
  'One-, two- and five-point updates describe counter increments only; batches and authoritative causes are not resolved.',
  'Neutral commands do not hold the robot stationary or disable physical response.',
  'Score/window comparisons stop at round, slot, clock, sequence and sampling-gap boundaries.',
  'Requested action and measured command checks are sampled evidence, not proof of every intervening command.',
];
function requireValue(ok, message) { if (!ok) throw new Error(message); }
function finite(n) { return typeof n === 'number' && Number.isFinite(n); }
function vector(v, n) { return Array.isArray(v) && v.length === n && v.every(finite); }
function subtract(a, b) { return a.map((x, k) => x - b[k]); }
function norm(v) { return Math.hypot(...v); }
function copyVector(v) { return v.slice(); }
function counts() { return { samples: 0, desired_action_1: 0, desired_action_other: 0,
  desired_action_unknown: 0, velocity_command_zero: 0, velocity_command_nonzero: 0,
  velocity_command_unknown: 0, both_neutral: 0 }; }
function countNeutral(c, command) {
  c.samples++;
  if (command.desired_action === 1) c.desired_action_1++;
  else if (command.desired_action === null) c.desired_action_unknown++;
  else c.desired_action_other++;
  if (command.velocity_zero === true) c.velocity_command_zero++;
  else if (command.velocity_zero === false) c.velocity_command_nonzero++;
  else c.velocity_command_unknown++;
  if (command.desired_action === 1 && command.velocity_zero === true) c.both_neutral++;
}
function deltaKind(n) {
  if (n === 0) return 'unchanged';
  if (n < 0) return 'counter_decrease';
  return n === 1 ? 'one_point_update' : n === 2 ? 'two_point_update'
    : n === 5 ? 'five_point_update' : 'other_positive_update';
}
function sampleFrom(source, sourceLine) {
  requireValue(source.schema === 'rek.g1_policy_source.v1', 'source_schema');
  const clock = source.clock || {};
  requireValue(Number.isSafeInteger(clock.qpc_ticks) && clock.qpc_ticks >= 0, 'qpc_ticks');
  requireValue(Number.isSafeInteger(clock.qpc_frequency_hz) && clock.qpc_frequency_hz > 0, 'qpc_frequency');
  requireValue(Number.isSafeInteger(source.observation_sequence) && source.observation_sequence >= 0, 'sequence');
  requireValue(typeof source.round_identity_sha256 === 'string' && /^[a-f0-9]{64}$/.test(source.round_identity_sha256), 'round_identity');
  requireValue(source.local_slot === 0 || source.local_slot === 1, 'local_slot');
  requireValue(Array.isArray(source.fighters) && source.fighters.length === 2, 'fighters');
  const fighters = source.fighters.map((f, slot) => {
    requireValue(vector(f.root_position_xyz, 3) && vector(f.root_rotation_xyzw, 4)
      && norm(f.root_rotation_xyzw) > 0 && finite(f.tilt_angle), 'root_pose');
    requireValue(Array.isArray(f.bone_names) && f.bone_names.length === 30
      && new Set(f.bone_names).size === 30 && Array.isArray(f.bone_world_positions_xyz)
      && f.bone_world_positions_xyz.length === 30, 'bones');
    const limbs = {};
    for (const [name, bone] of Object.entries(LIMBS)) {
      const index = f.bone_names.indexOf(bone);
      requireValue(index >= 0 && vector(f.bone_world_positions_xyz[index], 3), 'limb_position');
      limbs[name] = { bone, position_world_unity_xyz_m: copyVector(f.bone_world_positions_xyz[index]) };
    }
    return { slot, root_position_world_unity_xyz_m: copyVector(f.root_position_xyz),
      root_rotation_unity_xyzw: copyVector(f.root_rotation_xyzw), tilt_degrees: f.tilt_angle,
      falling_observed: typeof f.falling === 'boolean' ? f.falling : null,
      fallen_observed: typeof f.fallen === 'boolean' ? f.fallen : null, limbs };
  });
  requireValue(source.round && Array.isArray(source.round.clean_hits)
    && source.round.clean_hits.length === 2
    && source.round.clean_hits.every(n => Number.isSafeInteger(n) && n >= 0), 'scores');
  const input = source.input || {};
  const velocity = vector(input.velocity_command_xyz, 3) ? copyVector(input.velocity_command_xyz) : null;
  const command = { desired_action: Number.isInteger(input.desired_action) ? input.desired_action : null,
    velocity_command_xyz: velocity, velocity_zero: velocity ? velocity.every(n => n === 0) : null,
    requested_move_index_observed: Number.isInteger(input.requested_move_index) ? input.requested_move_index : null };
  return { source_line: sourceLine, observation_sequence: source.observation_sequence,
    qpc_ticks: clock.qpc_ticks, qpc_frequency_hz: clock.qpc_frequency_hz,
    utc: typeof clock.utc === 'string' ? clock.utc : null,
    round_identity_sha256: source.round_identity_sha256, local_slot: source.local_slot,
    round_active: source.round.active === true, stream_active: source.stream_active === true,
    time_remaining_seconds: finite(source.round.time_remaining) ? source.round.time_remaining : null,
    scores_by_slot: source.round.clean_hits.slice(), command,
    actor: fighters[source.local_slot], opponent: fighters[source.local_slot ^ 1] };
}

class PassiveAudit {
  constructor(options = {}, emit = () => {}) {
    this.options = { ...DEFAULTS, ...options };
    for (const k of Object.keys(DEFAULTS)) requireValue(finite(this.options[k])
      && this.options[k] >= (k === 'warmupSeconds' || k === 'windowSeconds' ? 0 : Number.MIN_VALUE), `option_${k}`);
    this.emit = emit;
    this.previous = null; this.segment = -1; this.ring = []; this.pendingScores = [];
    this.recentScores = []; this.activeMotion = new Map(); this.pendingMotion = [];
    this.eligibleStart = null; this.finished = false;
    this.summary = { schema: 'rek.passive_defender_audit.v1', options: this.options,
      coordinate_system: 'Unity world XYZ, Y up; root rotations XYZW',
      source_lines: 0, source_state_records: 0, samples: 0, invalid_records: 0,
      error_counts: {}, boundaries: {}, score_events: 0, counter_decrease_events: 0,
      positive_point_updates: { actor: 0, opponent: 0 },
      score_update_sizes: { actor: {}, opponent: {} },
      motion_intervals: 0, motion_intervals_no_score_observed: 0,
      active_stream_neutral: counts(), after_warmup_neutral: counts(),
      acknowledgments: { action_1_applied: 0, action_1_not_applied: 0, other_actions: 0 },
      maximum_actor_root_step_m: 0, maximum_opponent_root_step_m: 0, limits };
  }
  invalid(reason) {
    this.summary.invalid_records++;
    this.summary.error_counts[reason] = (this.summary.error_counts[reason] || 0) + 1;
    this.endSegment(); this.previous = null;
  }
  consumeLine(line, sourceLine) {
    this.summary.source_lines = sourceLine;
    if (!line.trim()) return;
    let value;
    try { value = JSON.parse(line); } catch { this.invalid('invalid_json'); return; }
    this.consume(value, sourceLine);
  }
  consume(value, sourceLine = this.summary.source_lines + 1) {
    requireValue(!this.finished, 'audit_already_finished');
    this.summary.source_lines = sourceLine;
    if (value?.event === 'g1_policy_action') {
      const a = this.summary.acknowledgments;
      if (value.action === 1) a[value.applied === true ? 'action_1_applied' : 'action_1_not_applied']++;
      else a.other_actions++;
    }
    if (value?.event !== 'g1_policy_state') return;
    this.summary.source_state_records++;
    let s;
    try { s = sampleFrom(value, sourceLine); } catch (e) { this.invalid(e.message); return; }
    let reason = null, dt = null;
    const old = this.previous;
    if (!old) reason = 'first_valid_sample';
    else if (s.round_identity_sha256 !== old.round_identity_sha256) reason = 'round_changed';
    else if (s.local_slot !== old.local_slot) reason = 'local_slot_changed';
    else if (s.qpc_frequency_hz !== old.qpc_frequency_hz) reason = 'clock_frequency_changed';
    else if (s.observation_sequence <= old.observation_sequence) reason = 'nonmonotonic_sequence';
    else {
      dt = (s.qpc_ticks - old.qpc_ticks) / s.qpc_frequency_hz;
      if (dt <= 0) reason = 'nonpositive_sample_interval';
      else if (dt > this.options.maxGapSeconds) reason = 'sample_gap';
    }
    if (reason) {
      this.endSegment(); this.segment++; this.segmentStart = s.qpc_ticks;
      this.summary.boundaries[reason] = (this.summary.boundaries[reason] || 0) + 1;
      dt = null;
    }
    s.event = 'kinematics'; s.sample_index = this.summary.samples++;
    s.segment = this.segment; s.segment_seconds = (s.qpc_ticks - this.segmentStart) / s.qpc_frequency_hz;
    s.delta_seconds = dt; s.derivative_unavailable_reason = reason;
    s.score_delta = dt === null ? null : {
      actor: s.scores_by_slot[s.local_slot] - old.scores_by_slot[s.local_slot],
      opponent: s.scores_by_slot[s.local_slot ^ 1] - old.scores_by_slot[s.local_slot ^ 1] };
    for (const role of ['actor', 'opponent']) {
      const f = s[role], before = old?.[role];
      f.root_step_m = dt === null ? null : norm(subtract(f.root_position_world_unity_xyz_m, before.root_position_world_unity_xyz_m));
      f.root_speed_m_s = dt === null ? null : f.root_step_m / dt;
      if (dt !== null) this.summary[`maximum_${role}_root_step_m`] = Math.max(this.summary[`maximum_${role}_root_step_m`], f.root_step_m);
      for (const [name, limb] of Object.entries(f.limbs)) {
        limb.root_relative_world_xyz_m = subtract(limb.position_world_unity_xyz_m, f.root_position_world_unity_xyz_m);
        limb.velocity_world_unity_xyz_m_s = dt === null ? null : subtract(limb.position_world_unity_xyz_m, before.limbs[name].position_world_unity_xyz_m).map(n => n / dt);
        limb.speed_m_s = dt === null ? null : norm(limb.velocity_world_unity_xyz_m_s);
        limb.root_relative_speed_m_s = dt === null ? null : norm(subtract(limb.root_relative_world_xyz_m, before.limbs[name].root_relative_world_xyz_m)) / dt;
      }
    }
    const eligible = s.round_active && s.stream_active;
    if (!eligible) this.eligibleStart = null;
    else if (this.eligibleStart === null) this.eligibleStart = s.qpc_ticks;
    s.neutral_check_eligible = eligible;
    s.neutral_check_after_warmup = eligible && (s.qpc_ticks - this.eligibleStart) / s.qpc_frequency_hz >= this.options.warmupSeconds;
    if (eligible) countNeutral(this.summary.active_stream_neutral, s.command);
    if (s.neutral_check_after_warmup) countNeutral(this.summary.after_warmup_neutral, s.command);

    this.advanceWindows(s);
    for (const name of Object.keys(LIMBS)) {
      const speed = s.opponent.limbs[name].speed_m_s;
      if (eligible && speed !== null && speed > this.options.motionSpeedMps) {
        let m = this.activeMotion.get(name);
        if (!m) {
          m = { event: 'opponent_limb_motion_interval', index: this.summary.motion_intervals++,
            segment: s.segment, limb: name, diagnostic_speed_threshold_m_s: this.options.motionSpeedMps,
            start_segment_seconds: old.segment_seconds, start_qpc_ticks: old.qpc_ticks,
            qpc_frequency_hz: s.qpc_frequency_hz, start_sample_index: old.sample_index,
            samples_above_threshold: 0, peak_speed_m_s: 0, peak_root_relative_speed_m_s: 0,
            score_event_indices: this.recentScores.filter(e => (old.qpc_ticks - e.ticks) / s.qpc_frequency_hz <= this.options.windowSeconds).map(e => e.index) };
          this.activeMotion.set(name, m);
        }
        m.end_segment_seconds = s.segment_seconds; m.end_qpc_ticks = s.qpc_ticks; m.end_sample_index = s.sample_index;
        m.samples_above_threshold++; m.peak_speed_m_s = Math.max(m.peak_speed_m_s, speed);
        m.peak_root_relative_speed_m_s = Math.max(m.peak_root_relative_speed_m_s, s.opponent.limbs[name].root_relative_speed_m_s);
      } else this.closeMotion(name);
    }
    if (s.score_delta && (s.score_delta.actor || s.score_delta.opponent)) this.scoreEvent(s);
    this.ring.push({ sample_index: s.sample_index, observation_sequence: s.observation_sequence,
      source_line: s.source_line, segment_seconds: s.segment_seconds, qpc_ticks: s.qpc_ticks });
    while (this.ring.length && (s.qpc_ticks - this.ring[0].qpc_ticks) / s.qpc_frequency_hz > this.options.windowSeconds) this.ring.shift();
    this.emit('kinematics', s); this.previous = s;
  }
  advanceWindows(s) {
    const t = s.segment_seconds, w = this.options.windowSeconds;
    const pending = [];
    for (const e of this.pendingScores) {
      if ((s.qpc_ticks - e.qpc_ticks) / s.qpc_frequency_hz > w) this.finishScore(e, false);
      else { e.window.end_sample_index = s.sample_index; e.window.end_source_line = s.source_line;
        e.window.last_observed_segment_seconds = t; e.window.last_observed_qpc_ticks = s.qpc_ticks; pending.push(e); }
    }
    this.pendingScores = pending;
    this.pendingMotion = this.pendingMotion.filter(m => {
      if ((s.qpc_ticks - m.end_qpc_ticks) / s.qpc_frequency_hz > w) { this.finishMotion(m, false); return false; }
      return true;
    });
    this.recentScores = this.recentScores.filter(e => (s.qpc_ticks - e.ticks) / s.qpc_frequency_hz <= w + this.options.maxGapSeconds);
  }
  scoreEvent(s) {
    const d = s.score_delta, index = this.summary.score_events++;
    if (d.actor < 0 || d.opponent < 0) this.summary.counter_decrease_events++;
    for (const role of ['actor', 'opponent']) if (d[role] > 0) {
      this.summary.positive_point_updates[role] += d[role];
      const sizes = this.summary.score_update_sizes[role]; sizes[d[role]] = (sizes[d[role]] || 0) + 1;
    }
    const start = this.ring.find(r => (s.qpc_ticks - r.qpc_ticks) / s.qpc_frequency_hz <= this.options.windowSeconds) || s;
    const e = { ...s, event: 'score_counter_update', index,
      update_kind: { actor: deltaKind(d.actor), opponent: deltaKind(d.opponent) },
      window: { requested_before_seconds: this.options.windowSeconds, requested_after_seconds: this.options.windowSeconds,
        start_sample_index: start.sample_index, end_sample_index: s.sample_index,
        start_source_line: start.source_line, end_source_line: s.source_line,
        leading_window_truncated: s.segment_seconds < this.options.windowSeconds,
        last_observed_segment_seconds: s.segment_seconds, last_observed_qpc_ticks: s.qpc_ticks } };
    this.pendingScores.push(e); this.recentScores.push({ index, ticks: s.qpc_ticks });
    for (const m of [...this.activeMotion.values(), ...this.pendingMotion]) {
      if ((m.start_qpc_ticks - s.qpc_ticks) / s.qpc_frequency_hz <= this.options.windowSeconds
        && (s.qpc_ticks - m.end_qpc_ticks) / s.qpc_frequency_hz <= this.options.windowSeconds) m.score_event_indices.push(index);
    }
  }
  closeMotion(name) {
    const m = this.activeMotion.get(name);
    if (m) { this.activeMotion.delete(name); this.pendingMotion.push(m); }
  }
  finishScore(e, boundary) {
    e.window.trailing_window_truncated = boundary
      && (e.window.last_observed_qpc_ticks - e.qpc_ticks) / e.qpc_frequency_hz < this.options.windowSeconds;
    this.emit('score_events', e);
  }
  finishMotion(m, boundary) {
    m.score_event_indices = [...new Set(m.score_event_indices)];
    m.score_observation = m.score_event_indices.length ? 'score_counter_update_observed' : 'no_score_observed';
    if (!m.score_event_indices.length) this.summary.motion_intervals_no_score_observed++;
    m.context_before_seconds = this.options.windowSeconds; m.context_after_seconds = this.options.windowSeconds;
    m.leading_window_truncated = m.start_segment_seconds < this.options.windowSeconds;
    m.trailing_window_truncated = boundary && (!this.previous
      || (this.previous.qpc_ticks - m.end_qpc_ticks) / m.qpc_frequency_hz < this.options.windowSeconds);
    this.emit('motion_intervals', m);
  }
  endSegment() {
    for (const name of this.activeMotion.keys()) this.closeMotion(name);
    for (const e of this.pendingScores) this.finishScore(e, true);
    for (const m of this.pendingMotion) this.finishMotion(m, true);
    this.pendingScores = []; this.pendingMotion = []; this.ring = []; this.recentScores = [];
    this.eligibleStart = null;
  }
  finish() {
    requireValue(!this.finished, 'audit_already_finished');
    this.endSegment(); this.finished = true;
    this.summary.segments = this.segment + 1;
    this.summary.complete = this.summary.samples > 0 && this.summary.invalid_records === 0;
    const neutral = this.summary.after_warmup_neutral;
    this.summary.sampled_neutral_confirmed_after_warmup = neutral.samples > 0
      && neutral.both_neutral === neutral.samples;
    return this.summary;
  }
}

async function analyzeFile(input, output, options = {}) {
  // Construct/validate options before creating any output.
  const pendingWrites = [];
  const audit = new PassiveAudit(options, (name, value) => {
    if (!outputs[name].write(JSON.stringify(value) + '\n')) pendingWrites.push(once(outputs[name], 'drain'));
  });
  const inputPath = path.resolve(input), outputPath = path.resolve(output);
  const before = fs.statSync(inputPath);
  requireValue(before.isFile(), 'input_not_file');
  requireValue(!fs.existsSync(outputPath), 'output_already_exists');
  fs.mkdirSync(outputPath);
  const outputs = {};
  let outputError = null;
  for (const name of ['kinematics', 'score_events', 'motion_intervals']) {
    outputs[name] = fs.createWriteStream(path.join(outputPath, name + '.jsonl'), { flags: 'wx' });
    outputs[name].on('error', error => { outputError = error; });
  }
  const source = fs.createReadStream(inputPath), hash = crypto.createHash('sha256');
  source.on('data', chunk => hash.update(chunk));
  const lines = readline.createInterface({ input: source, crlfDelay: Infinity });
  let lineNumber = 0;
  try {
    for await (const line of lines) {
      if (outputError) throw outputError;
      audit.consumeLine(line, ++lineNumber);
      if (pendingWrites.length) await Promise.all(pendingWrites.splice(0));
    }
    const summary = audit.finish();
    if (pendingWrites.length) await Promise.all(pendingWrites.splice(0));
    await Promise.all(Object.values(outputs).map(stream => new Promise((resolve, reject) => stream.end(error => error ? reject(error) : resolve()))));
    if (outputError) throw outputError;
    const after = fs.statSync(inputPath);
    requireValue(before.size === after.size && before.mtimeMs === after.mtimeMs, 'input_changed_during_audit');
    summary.source = { path: inputPath, bytes: after.size, sha256: hash.digest('hex'), modified_utc: after.mtime.toISOString() };
    summary.outputs = Object.fromEntries(Object.keys(outputs).map(name => [name, name + '.jsonl']));
    fs.writeFileSync(path.join(outputPath, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
    return summary;
  } finally { lines.close(); source.destroy(); for (const stream of Object.values(outputs)) stream.destroy(); }
}
async function main(argv) {
  requireValue(argv.length >= 2, 'Usage: node audit_passive_defender.cjs RELAY_JSONL NEW_OUTPUT_DIRECTORY [--warmup-seconds=1 --window-seconds=1 --motion-speed-mps=1.75 --max-gap-seconds=0.25]');
  const names = { 'warmup-seconds': 'warmupSeconds', 'window-seconds': 'windowSeconds',
    'motion-speed-mps': 'motionSpeedMps', 'max-gap-seconds': 'maxGapSeconds' }, options = {};
  for (const argument of argv.slice(2)) {
    const m = /^--([^=]+)=(.+)$/.exec(argument);
    requireValue(m && names[m[1]], 'unknown_option'); options[names[m[1]]] = Number(m[2]);
  }
  const summary = await analyzeFile(argv[0], argv[1], options);
  process.stdout.write(JSON.stringify({ complete: summary.complete, samples: summary.samples,
    score_events: summary.score_events, motion_intervals: summary.motion_intervals,
    summary: path.resolve(argv[1], 'summary.json') }) + '\n');
  if (!summary.complete) process.exitCode = 2;
}
module.exports = { PassiveAudit, analyzeFile, sampleFrom, deltaKind, LIMBS, DEFAULTS };
if (require.main === module) main(process.argv.slice(2)).catch(error => { process.stderr.write(error.message + '\n'); process.exitCode = 2; });
