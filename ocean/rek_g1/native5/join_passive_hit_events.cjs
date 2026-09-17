#!/usr/bin/env node
'use strict';

// Offline receive-event/pose association only. Does not connect to a game or relay.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const readline = require('node:readline');
const { once } = require('node:events');
const { heading, wrap } = require('./compare_passive_contacts.cjs');

const LIMITS = Object.freeze([
  'REK_Hit is an unreliable received effects packet. No received hit leaves contact occurrence unknown.',
  'Same-frame score association is not a causal identifier, even with one hit and one score.',
  'A scoring fighter is only an attacker candidate; hit packets contain no fighter, clip ID, body zone or rejection reason.',
  'Five-point awards are reported separately and are not classified as strikes or knockouts.',
  'Raw packets lack QPC/UTC in the deployed recorder. Unity timestamps are never converted into invented QPC or server time.',
  'Pose brackets are client-rendered samples around receipt, not authoritative contact-time geometry. No interpolation is performed.',
  'Bearing uses projected root-local +X in Unity XZ, reusing compare_passive_contacts; it is not aiming intent or controller heading.',
  'Native root/sample observations occur in FixedUpdate. Their Unity time can differ within the same rendered frame; all same-frame candidates are retained.',
]);
const finite = n => typeof n === 'number' && Number.isFinite(n);
const safe = n => Number.isSafeInteger(n) && n >= 0;
const vector = (v, n) => Array.isArray(v) && v.length === n && v.every(finite);
function check(ok, message) { if (!ok) throw new Error(message); }
function select(value, keys) { return Object.fromEntries(keys.filter(k => value[k] !== undefined).map(k => [k, value[k]])); }
function eventClock(raw) {
  return { unity_frame: raw.unity_frame ?? null, unity_time: raw.unity_time ?? null,
    unity_unscaled_time: raw.unity_unscaled_time ?? null,
    monotonic_receipt_time: raw.monotonic_receipt_time ?? null,
    qpc_ticks: raw.stopwatch_timestamp_ticks ?? raw.qpc_ticks ?? null,
    qpc_frequency_hz: raw.stopwatch_frequency_hz ?? raw.qpc_frequency_hz ?? null,
    utc: raw.utc ?? null };
}
function geometry(fighters, localSlot) {
  const local = fighters[localSlot], opponent = fighters[localSlot ^ 1];
  const d = opponent.root_position_xyz.map((n, k) => n - local.root_position_xyz[k]);
  const measuredHeading = q => {
    const n = Math.hypot(...q), [x, y, z, w] = q.map(v => v / n);
    return Math.hypot(1 - 2 * (y * y + z * z), 2 * (x * z - w * y)) > 1e-12 ? heading(q) : null;
  };
  const lh = measuredHeading(local.root_rotation_xyzw), oh = measuredHeading(opponent.root_rotation_xyzw);
  const ground = Math.hypot(d[0], d[2]);
  return { local_root: local, opponent_root: opponent,
    root_distance_3d_m: Math.hypot(...d), root_distance_ground_xz_m: ground,
    heading_axis: 'root_local_positive_x_projected_to_Unity_XZ',
    local_heading_rad: lh, opponent_heading_rad: oh,
    local_bearing_to_opponent_rad: ground > 0 && lh !== null ? wrap(Math.atan2(d[2], d[0]) - lh) : null,
    opponent_bearing_to_local_rad: ground > 0 && oh !== null ? wrap(Math.atan2(-d[2], -d[0]) - oh) : null };
}

async function streamJsonl(file, visit) {
  const absolute = path.resolve(file), before = fs.statSync(absolute);
  check(before.isFile(), 'input_not_file');
  const input = fs.createReadStream(absolute), hash = crypto.createHash('sha256');
  input.on('data', b => hash.update(b));
  const lines = readline.createInterface({ input, crlfDelay: Infinity });
  let count = 0;
  try {
    for await (const line of lines) {
      count++;
      if (!line.trim()) continue;
      let value;
      try { value = JSON.parse(line); } catch { throw new Error(`invalid_json_line_${count}`); }
      await visit(value, count);
    }
  } finally { lines.close(); input.destroy(); }
  const after = fs.statSync(absolute);
  check(before.size === after.size && before.mtimeMs === after.mtimeMs
    && before.ino === after.ino, 'input_changed_during_join');
  return { path: absolute, bytes: after.size, sha256: hash.digest('hex'), lines: count,
    modified_utc: after.mtime.toISOString() };
}

function poseFrom(source, line) {
  check(source.schema === 'rek.g1_policy_source.v1', 'source_schema');
  const c = source.clock || {};
  check(safe(c.unity_frame) && finite(c.unity_time) && safe(c.qpc_ticks)
    && safe(c.qpc_frequency_hz) && c.qpc_frequency_hz > 0, 'pose_clock');
  check(source.local_slot === 0 || source.local_slot === 1, 'local_slot');
  check(typeof source.round_identity_sha256 === 'string' && /^[a-f0-9]{64}$/.test(source.round_identity_sha256), 'round_identity');
  check(Array.isArray(source.fighters) && source.fighters.length === 2, 'fighters');
  const fighters = source.fighters.map((f, slot) => {
    check(vector(f.root_position_xyz, 3) && vector(f.root_rotation_xyzw, 4)
      && Math.hypot(...f.root_rotation_xyzw) > 0, 'root_pose');
    return { slot, root_position_xyz: f.root_position_xyz, root_rotation_xyzw: f.root_rotation_xyzw,
      tilt_angle: finite(f.tilt_angle) ? f.tilt_angle : null,
      falling: typeof f.falling === 'boolean' ? f.falling : null,
      fallen: typeof f.fallen === 'boolean' ? f.fallen : null };
  });
  return { source_kind: 'relay_g1_policy_state', relay_line: line, observation_sequence: source.observation_sequence ?? null,
    clock: c, round_identity_sha256: source.round_identity_sha256, local_slot: source.local_slot,
    round: select(source.round || {}, ['active', 'round_number', 'time_remaining', 'clean_hits']),
    input: select(source.input || {}, ['desired_action', 'velocity_command_xyz', 'requested_move_index']),
    ...geometry(fighters, source.local_slot) };
}
function nativePoseFrom(raw, line, frequency) {
  const root = raw.event === 'root_pose_sample';
  const clock = { ...eventClock(raw), qpc_frequency_hz: frequency,
    unity_fixed_time: raw.unity_fixed_time ?? null };
  check(safe(clock.unity_frame) && finite(clock.unity_time) && safe(clock.qpc_ticks)
    && safe(frequency) && frequency > 0, 'native_pose_clock');
  check(raw.local_fighter_index === 0 || raw.local_fighter_index === 1, 'native_local_slot');
  const fighters = [0, 1].map(slot => {
    const f = raw[`fighter_${slot}${root ? '_root' : ''}`] || {};
    const position = root ? f.world_position_xyz : f.root_position;
    const rotation = root ? f.world_rotation_xyzw : f.root_rotation;
    check(vector(position, 3) && vector(rotation, 4) && Math.hypot(...rotation) > 0, 'native_root_pose');
    return { slot, root_position_xyz: position, root_rotation_xyzw: rotation,
      tilt_angle: finite(f.tilt_angle) ? f.tilt_angle : null };
  });
  return { source_kind: 'recorder_' + raw.event, recorder_line: line, clock,
    round_identity_sha256: null, native_round: { fight_epoch: raw.fight_epoch ?? null,
      round_number: root ? raw.round_number ?? null : raw.round?.number ?? null },
    local_slot: raw.local_fighter_index,
    round: root ? null : select(raw.round || {}, ['number', 'active', 'time_remaining', 'clean_hits', 'falls', 'result', 'knockout']),
    input: root ? null : select(raw.input || {}, ['velocity_command', 'pending_move', 'punching']),
    ...geometry(fighters, raw.local_fighter_index) };
}

function packetKey(raw) {
  return safe(raw.unity_frame) ? String(raw.unity_frame) : null;
}
function addGroup(map, key, value) {
  if (key === null) return;
  if (!map.has(key)) map.set(key, []);
  map.get(key).push(value);
}
function boundary(a, b) {
  return a.round_identity_sha256 !== b.round_identity_sha256 || a.local_slot !== b.local_slot
    || JSON.stringify(a.native_round) !== JSON.stringify(b.native_round)
    || a.clock.qpc_frequency_hz !== b.clock.qpc_frequency_hz
    || b.clock.qpc_ticks < a.clock.qpc_ticks || b.clock.unity_time < a.clock.unity_time
    || b.clock.unity_frame < a.clock.unity_frame;
}
function poseOffset(pose, raw) {
  const c = eventClock(raw), pc = pose.clock;
  return { ...pose, pose_minus_event_unity_seconds: pc.unity_time - c.unity_time,
    pose_minus_event_unity_frames: pc.unity_frame - c.unity_frame,
    pose_minus_event_qpc_seconds: safe(c.qpc_ticks) && c.qpc_frequency_hz === pc.qpc_frequency_hz
      ? (pc.qpc_ticks - c.qpc_ticks) / pc.qpc_frequency_hz : null };
}
function bracketSegment(poses, raw, index) {
  const t = raw.unity_time;
  let left = 0, right = poses.length;
  while (left < right) {
    const mid = (left + right) >>> 1;
    if (poses[mid].clock.unity_time < t) left = mid + 1; else right = mid;
  }
  let next = left;
  while (next < poses.length && poses[next].clock.unity_time === t) next++;
  const prior = [], exact = poses.slice(left, next), after = [];
  if (left > 0) {
    const pt = poses[left - 1].clock.unity_time;
    for (let i = left - 1; i >= 0 && poses[i].clock.unity_time === pt; i--) prior.unshift(poses[i]);
  }
  if (next < poses.length) {
    const nt = poses[next].clock.unity_time;
    for (let i = next; i < poses.length && poses[i].clock.unity_time === nt; i++) after.push(poses[i]);
  }
  const contradictory = prior.some(p => p.clock.unity_frame >= raw.unity_frame)
    || exact.some(p => p.clock.unity_frame !== raw.unity_frame)
    || after.some(p => p.clock.unity_frame <= raw.unity_frame);
  return { segment: index, round_identity_sha256: poses[0].round_identity_sha256,
    native_round: poses[0].native_round ?? null,
    clock_frame_consistent: !contradictory,
    duplicated_pose_time: [prior, exact, after].some(a => a.length > 1),
    prior: prior.map(p => poseOffset(p, raw)), same_time: exact.map(p => poseOffset(p, raw)),
    next: after.map(p => poseOffset(p, raw)),
    same_frame: poses.filter(p => p.clock.unity_frame === raw.unity_frame).map(p => poseOffset(p, raw)) };
}
function brackets(segments, raw) {
  if (!packetKey(raw) || !finite(raw.unity_time)) return { status: 'event_clock_unavailable', basis: null, candidates: [] };
  const candidates = segments.flatMap((poses, i) => {
    const inside = poses[0].clock.unity_time <= raw.unity_time
      && raw.unity_time <= poses.at(-1).clock.unity_time;
    return inside || segments.length === 1 ? [bracketSegment(poses, raw, i)] : [];
  });
  const status = !candidates.length ? 'no_pose_clock_overlap'
    : candidates.length > 1 ? 'ambiguous_clock_segments'
    : !candidates[0].clock_frame_consistent ? 'inconsistent_unity_frame_time'
    : candidates[0].duplicated_pose_time ? 'ambiguous_duplicate_pose_time'
    : !candidates[0].prior.length || !candidates[0].next.length ? 'partial_bracket'
    : 'bracketed';
  return { status, basis: 'shared_client_unity_time_and_frame; capture_QPC_bounds_when_available', candidates };
}

async function writeJsonl(file, values) {
  const output = fs.createWriteStream(file, { flags: 'wx' });
  let failure = null;
  output.on('error', e => { failure = e; });
  try {
    for (const value of values) {
      if (failure) throw failure;
      if (!output.write(JSON.stringify(value) + '\n')) await once(output, 'drain');
    }
    await new Promise((resolve, reject) => output.end(e => e ? reject(e) : resolve()));
    if (failure) throw failure;
  } finally { output.destroy(); }
}

async function joinFiles(recorderFile, relayFile, outputDirectory) {
  const output = path.resolve(outputDirectory);
  check(!fs.existsSync(output), 'output_already_exists');
  const hits = [], scores = [], hitGroups = new Map(), scoreGroups = new Map();
  const nativeOnly = relayFile === '-', nativeRoots = [], nativeSamples = [], errorCounts = {};
  const nativeStats = { root_pose_samples: 0, compact_samples: 0, measured_velocity_zero_samples: 0,
    no_pending_move_samples: 0, punching_false_samples: 0, all_three_neutral_samples: 0,
    outbound_velocity_requests: 0, outbound_velocity_zero_requests: 0 };
  let nativeInvalid = 0;
  let start = null, end = null, startCount = 0, endCount = 0, errors = 0;
  const recorder = await streamJsonl(recorderFile, (raw, line) => {
    if (raw.event === 'capture_start') {
      startCount++;
      start = select(raw, ['schema', 'utc', 'stopwatch_timestamp_ticks', 'stopwatch_frequency_hz',
        'pid', 'plugin_version', 'plugin_sha256', 'game_assembly_sha256', 'global_metadata_sha256']);
    } else if (raw.event === 'capture_end') {
      endCount++;
      end = select(raw, ['utc', 'stopwatch_timestamp_ticks', 'reason', 'capture_error_count',
        'raw_hit_packet_count', 'raw_score_packet_count', 'sample_count', 'root_pose_sample_count',
        'client_transport_invocation_count', 'client_transport_method_counts']);
    } else if (raw.event === 'capture_error') errors++;
    else if (raw.event === 'raw_hit_packet' || raw.event === 'raw_score_packet') {
      const list = raw.event === 'raw_hit_packet' ? hits : scores;
      const event = { index: list.length, recorder_line: line, raw };
      list.push(event);
      addGroup(raw.event === 'raw_hit_packet' ? hitGroups : scoreGroups, packetKey(raw), event);
    }
    if (raw.event === 'sample' || raw.event === 'root_pose_sample') {
      if (raw.event === 'sample') {
        nativeStats.compact_samples++;
        const input = raw.input || {}, zero = vector(input.velocity_command, 3) && input.velocity_command.every(n => n === 0);
        nativeStats.measured_velocity_zero_samples += Number(zero);
        nativeStats.no_pending_move_samples += Number(input.pending_move === false);
        nativeStats.punching_false_samples += Number(input.punching === false);
        nativeStats.all_three_neutral_samples += Number(zero && input.pending_move === false && input.punching === false);
      } else nativeStats.root_pose_samples++;
      if (nativeOnly) {
        try { (raw.event === 'sample' ? nativeSamples : nativeRoots).push(nativePoseFrom(raw, line, start?.stopwatch_frequency_hz)); }
        catch (e) { nativeInvalid++; errorCounts[e.message] = (errorCounts[e.message] || 0) + 1; }
      }
    } else if (raw.event === 'outbound_request_projection' && raw.message === 'REK_Input') {
      nativeStats.outbound_velocity_requests++;
      nativeStats.outbound_velocity_zero_requests += Number(vector(raw.velocity_command_xyz, 3) && raw.velocity_command_xyz.every(n => n === 0));
    }
  });
  check(startCount <= 1 && endCount <= 1, 'multiple_recorder_captures');
  const clockBounded = safe(start?.stopwatch_timestamp_ticks) && safe(end?.stopwatch_timestamp_ticks)
    && safe(start?.stopwatch_frequency_hz) && start.stopwatch_frequency_hz > 0
    && end.stopwatch_timestamp_ticks >= start.stopwatch_timestamp_ticks;
  const segments = [];
  let poses = 0, excluded = 0, invalid = nativeInvalid, previous = null;
  function acceptPose(pose) {
    if (clockBounded && (pose.clock.qpc_frequency_hz !== start.stopwatch_frequency_hz
      || pose.clock.qpc_ticks < start.stopwatch_timestamp_ticks || pose.clock.qpc_ticks > end.stopwatch_timestamp_ticks)) {
      excluded++; previous = null; return;
    }
    if (!previous || boundary(previous, pose)) segments.push([]);
    segments.at(-1).push(pose); previous = pose; poses++;
  }
  let relay = null;
  if (nativeOnly) {
    for (const pose of nativeRoots.length ? nativeRoots : nativeSamples) acceptPose(pose);
  } else relay = await streamJsonl(relayFile, (raw, line) => {
    if (raw.event !== 'g1_policy_state') return;
    let pose;
    try { pose = poseFrom(raw, line); } catch (e) {
      invalid++; errorCounts[e.message] = (errorCounts[e.message] || 0) + 1; previous = null; return;
    }
    acceptPose(pose);
  });
  const packet = e => ({ index: e.index, recorder_line: e.recorder_line,
    recorder_sha256: recorder.sha256, raw: e.raw, clock: eventClock(e.raw) });
  const scoreRows = scores.map(e => ({ ...packet(e),
    award_class: e.raw.decoded?.points_awarded === 5 ? 'five_point_award_cause_unresolved' : 'other_score_award',
    received_hit_indices_same_frame: (hitGroups.get(packetKey(e.raw)) || []).map(h => h.index),
    association_is_causal: false, poses: brackets(segments, e.raw) }));
  const hitRows = hits.map(e => {
    const sameScores = scoreGroups.get(packetKey(e.raw)) || [], sameHits = hitGroups.get(packetKey(e.raw)) || [];
    const candidateScores = sameScores.filter(s => s.raw.decoded?.points_awarded > 0
      && s.raw.decoded.points_awarded !== 5 && [0, 1].includes(s.raw.decoded.fighter_index));
    return { ...packet(e), observation: 'received_hit_effects_packet',
      clip_id: null, body_zone: null, authoritative_contact_id: null,
      same_frame_score_association: { status: !sameScores.length ? 'no_score_packet_same_frame'
        : sameScores.length > 1 || sameHits.length > 1 ? 'ambiguous_same_frame_association'
        : 'unique_same_frame_association_noncausal', is_causal: false,
      hit_indices: sameHits.map(h => h.index), score_indices: sameScores.map(s => s.index),
      score_records: sameScores.map(packet),
      five_point_score_indices: sameScores.filter(s => s.raw.decoded?.points_awarded === 5).map(s => s.index) },
      attacker_candidates: [...new Set(candidateScores.map(s => s.raw.decoded.fighter_index))].map(slot => ({
        slot, basis: 'same_frame_non_five_point_scoring_fighter', confirmed: false,
        score_indices: candidateScores.filter(s => s.raw.decoded.fighter_index === slot).map(s => s.index) })),
      poses: brackets(segments, e.raw) };
  });
  const finalized = startCount === 1 && endCount === 1;
  const countMatch = finalized && end.raw_hit_packet_count === hits.length && end.raw_score_packet_count === scores.length;
  const summary = { schema: 'rek.passive_hit_pose_join.v1', inputs: { recorder, relay },
    pose_source: nativeOnly ? nativeRoots.length ? 'native_root_pose_sample' : 'native_sample_fallback' : 'relay_g1_policy_state',
    native_observations: nativeStats,
    recorder_capture: { start, end, finalized, packet_counts_match_footer: countMatch,
      error_records: errors, qpc_bounds_applied: clockBounded },
    pose_records: poses, pose_segments: segments.length, excluded_by_capture_qpc: excluded,
    invalid_pose_records: invalid, invalid_pose_reasons: errorCounts,
    received_hit_packets: hits.length, score_packets: scores.length,
    five_point_awards: scoreRows.filter(s => s.raw.decoded?.points_awarded === 5).length,
    hit_observation: hits.length ? 'received_hit_effects_observed' : 'no_received_hit_packets_contact_unknown',
    ambiguous_same_frame_hit_records: hitRows.filter(h => h.same_frame_score_association.status === 'ambiguous_same_frame_association').length,
    complete: finalized && countMatch && clockBounded && (end.capture_error_count ?? 0) === 0
      && errors === 0 && invalid === 0 && poses > 0,
    coordinate_system: 'Unity world XYZ metres, Y up; root quaternion XYZW', limits: LIMITS,
    outputs: { hits: 'hit_events.jsonl', scores: 'score_events.jsonl', five_point_awards: 'five_point_awards.jsonl' } };
  fs.mkdirSync(output);
  await writeJsonl(path.join(output, summary.outputs.hits), hitRows);
  await writeJsonl(path.join(output, summary.outputs.scores), scoreRows);
  await writeJsonl(path.join(output, summary.outputs.five_point_awards), scoreRows.filter(s => s.raw.decoded?.points_awarded === 5));
  fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
  return summary;
}

module.exports = { joinFiles, poseFrom, nativePoseFrom, brackets, eventClock, LIMITS };
if (require.main === module) {
  const args = process.argv.slice(2);
  Promise.resolve().then(() => {
    check(args.length === 3, 'Usage: node join_passive_hit_events.cjs RECORDER_JSONL RELAY_JSONL_OR_- NEW_OUTPUT_DIRECTORY');
    return joinFiles(...args);
  }).then(s => {
    process.stdout.write(JSON.stringify({ complete: s.complete, received_hit_packets: s.received_hit_packets,
      score_packets: s.score_packets, five_point_awards: s.five_point_awards,
      summary: path.resolve(args[2], 'summary.json') }) + '\n');
    if (!s.complete) process.exitCode = 2;
  }).catch(e => { process.stderr.write(e.message + '\n'); process.exitCode = 2; });
}
