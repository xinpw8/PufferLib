'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
const { spawnSync } = require('node:child_process');
const { LEGACY, SCHEMA } = require('./owned_yaw_export_evidence.cjs');
const { pack, validateRow, reward, potential, readTrial, exportData, HEADER, ROW } = require('./authentic_trajectory_data.cjs');
function row() {
  return { sequence: 0, reset: 1, action: 2, policyWeight: 1, valueWeight: 1,
    obs: Array.from({ length: 223 }, (_, i) => Math.fround(i / 10)), mask: Array(33).fill(1),
    time: 0, nextTime: 0.04, dt: 0.04, gamma: Math.fround(0.999 ** 2), lambda: Math.fround(0.995 ** 2),
    own: 0, opponent: 0, nextOwn: 1, nextOpponent: 0, sourceSeq: 2, nextSourceSeq: 5,
    terminal: true, applied: true, outcome: 1, reward: 1 };
}
test('binary has native agreed offsets and unmasked original features', () => {
  const r = row(), b = pack([r], 1);
  assert.equal(b.length, HEADER + ROW); assert.equal(b.toString('ascii', 0, 8), 'REKRL001');
  assert.equal(b.readUInt32LE(24), 1128); assert.equal(b.readUInt32LE(28), 1);
  assert.ok([...b.subarray(32, 255)].every(x => x === 1)); assert.equal(b[255], 0);
  assert.equal(b.readFloatLE(HEADER + 32 + 4 * 190), r.obs[190]);
  assert.equal(b.readFloatLE(HEADER + 924 + 4 * 2), 1);
  assert.equal(b.readDoubleLE(HEADER + 1056), 0.04); assert.equal(b.readUInt32LE(HEADER + 1112), 1);
});
test('reward/next-state changes cannot alter stored preceding observations', () => {
  const a = row(), b = { ...row(), nextOwn: 10, nextOpponent: 20, outcome: -1, reward: -1 };
  assert.deepEqual(pack([a], 1).subarray(HEADER + 32, HEADER + 924), pack([b], 1).subarray(HEADER + 32, HEADER + 924));
});
test('terminal potential removed, five-point awards enter only observed score potential', () => {
  assert.equal(potential(5, 0), 0.5);
  assert.equal(reward(0.9, 5, 0, 10, 0, true, 1), 0.5);
  assert.equal(reward(0.9, 0, 5, 0, 5, true, -1), -0.5);
  assert.equal(reward(1, 0, 0, 5, 0, false, 0), 0.5);
});
test('float32 variable gamma potential telescopes with the same discounts', () => {
  const g1 = Math.fround(0.999 ** 2), g2 = Math.fround(0.999 ** 4);
  const x = reward(g1, 0, 0, 5, 0, false, 0), y = reward(g2, 5, 0, 5, 0, true, 1);
  assert.ok(Math.abs(x + g1 * y - g1) < 1e-7);
});
test('terminal-race attempted action retained with actor weight zero', () => {
  const r = { ...row(), applied: false, policyWeight: 0 }; const b = pack([r], 1);
  assert.equal(b.readInt32LE(HEADER + 12), 2); assert.equal(b.readFloatLE(HEADER + 16), 0);
  assert.equal(b.readUInt32LE(HEADER + 1116), 0); assert.equal(b.readFloatLE(HEADER + 1120), 1);
});
test('malformed mask, missing observation, noncausal timing and nonterminal rejection fail', () => {
  for (const mutate of [r => { r.mask[2] = 0; }, r => { r.obs.pop(); }, r => { r.nextTime = 0; },
    r => { r.gamma = NaN; }, r => { r.nextOwn = -1; }, r => { r.reward = 0; },
    r => { r.terminal = false; r.outcome = 0; r.applied = false; r.policyWeight = 0; }]) {
    const r = row(); mutate(r); assert.throws(() => validateRow(r));
  }
});
function fixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'live-rek-authentic-cpu-fixture-'));
  for (const folder of ['trial', 'contact-analysis', 'referee-validation']) fs.mkdirSync(path.join(dir, folder));
  const roundId = '1'.repeat(64), checkpoint = '2'.repeat(64);
  const c = t => ({ qpc_ticks: 1000000 + t, qpc_frequency_hz: 1000000 });
  const round = (active, points) => ({ number: 1, duration: 120, time_remaining: active ? 119.7 : 0,
    active, redo: false, clean_hits: points, result: active ? 'InProgress' : 'WonByPoints', winner_index: active ? -1 : 0 });
  const source = (seq, t, active, points) => ({ event: 'g1_policy_state', observation_sequence: seq,
    round_identity_sha256: roundId, local_slot: 0, clock: c(t), round: round(active, points) });
  const req = (seq, points) => { const observation = Array(223).fill(0); observation[190] = points;
    return { type: 'step', seq, round_id: roundId, terminal: false, observation,
      observation_schema: 'rek.native5.scaled_polar_xy.v1', mask: Array(33).fill(1) }; };
  const ready = { type: 'ready', checkpoint_sha256: checkpoint, selection: 'sampled', seed: 73, precision: 'bf16',
    hidden_size: 256, num_layers: 2, observations: 223, actions: 33,
    observation_schema: 'rek.native5.scaled_polar_xy.v1' };
  const prediction = (seq, i) => ({ ...ready, type: 'action', seq, round_id: roundId, action: 2,
    legal_actions: 33, decision_index: i, recurrent_reset: i === 1, round_changed: i === 1 });
  const sent = seq => ({ type: 'policy_action', observation_sequence: seq, round_identity_sha256: roundId,
    request_id: `action-${seq}`, action: 2 });
  const ack = (seq, t, applied) => ({ ...sent(seq), event: 'g1_policy_action', clock: c(t), applied,
    reason: applied ? 'held_state_applied_locally' : 'policy_stream_not_owned' });
  const write = (name, value) => fs.writeFileSync(path.join(dir, name), JSON.stringify(value));
  const lines = (name, values) => fs.writeFileSync(path.join(dir, name), values.map(x => JSON.stringify(x)).join('\n') + '\n');
  write('trial/summary.json', { round_identity_sha256: roundId, local_slot: 0, checkpoint_sha256: checkpoint,
    predictions: 2, final_round: round(false, [1, 0]) });
  write('contact-analysis/summary.json', { completed_policy_round: true, terminal_evidence_consistent: true,
    terminal_point_counters_consistent: true, round_identity_sha256: roundId, local_slot: 0,
    checkpoint_sha256: checkpoint, score_events: 1, inputs: [] });
  write('referee-validation/live-referee-validation.json', { verification_passed: true, round_identities: [roundId], inputs: [] });
  lines('trial/worker.stdin.jsonl', [req(2, 0), req(4, 1)]);
  lines('trial/worker.stdout.jsonl', [ready, prediction(2, 1), prediction(4, 2)]);
  lines('trial/relay.stdin.jsonl', [sent(2), sent(4)]);
  lines('trial/relay.stdout.jsonl', [source(1, 0, true, [0, 0]), source(2, 10000, true, [0, 0]),
    ack(2, 15000, true), source(3, 20000, true, [1, 0]), source(4, 30000, true, [1, 0]),
    source(5, 70000, false, [1, 0]), ack(4, 71000, false)]);
  lines('trial/orchestrator.jsonl', [{ event: 'observation_unavailable', unavailable: ['derivative_warmup'] }]);
  lines('contact-analysis/score-events.jsonl', [{ index: 0, fighter_index: 0, points_awarded: 1 }]);
  return { dir, lines };
}
test('synthetic joined trial preserves omitted source, actual dt and rejected decision', async () => {
  const f = fixture(), result = await readTrial(f.dir, 7, 0.999, 0.995);
  assert.equal(result.rows.length, 2); assert.deepEqual(result.rows.map(x => x.sourceSeq), [2, 4]);
  assert.deepEqual(result.rows.map(x => x.sequence), [7, 7]); assert.deepEqual(result.rows.map(x => x.dt), [0.02, 0.04]);
  assert.equal(result.rows[1].policyWeight, 0); assert.equal(result.rows[1].reward, Math.fround(1 - potential(1, 0)));
  assert.equal(result.report.native_score_awards, 1); assert.equal(result.report.recurrent_resets, 1);
});
test('recurrent replay input duplication rejected rather than silently replaced', async () => {
  const f = fixture(); const file = path.join(f.dir, 'trial/worker.stdin.jsonl');
  fs.appendFileSync(file, fs.readFileSync(file, 'utf8').split('\n')[0] + '\n');
  await assert.rejects(readTrial(f.dir, 0, 0.999, 0.995), /duplicate worker request/);
});

test('legacy exporter rejects v2 or missing ready observation schema', async () => {
  for (const schema of ['rek.native5.scaled_polar_xy.owned_yaw_v2', undefined]) {
    const f = fixture(), file = 'trial/worker.stdout.jsonl';
    const entries = fs.readFileSync(path.join(f.dir, file), 'utf8').trim().split('\n').map(JSON.parse);
    entries[0].observation_schema = schema; f.lines(file, entries);
    await assert.rejects(readTrial(f.dir, 0, 0.999, 0.995), /unsupported ready observation schema/);
  }
});

test('legacy exporter rejects v2 or missing schema on every worker request', async () => {
  for (const schema of ['rek.native5.scaled_polar_xy.owned_yaw_v2', undefined]) {
    const f = fixture(), file = 'trial/worker.stdin.jsonl';
    const entries = fs.readFileSync(path.join(f.dir, file), 'utf8').trim().split('\n').map(JSON.parse);
    entries[1].observation_schema = schema; f.lines(file, entries);
    await assert.rejects(readTrial(f.dir, 0, 0.999, 0.995), /unsupported worker observation schema/);
  }
});
test('explicit original task-time profile is preserved without fallback defaults', async () => {
  const f = fixture(), gamma = 0.9998844821426083, lambda = 0.9978673240629938;
  const result = await readTrial(f.dir, 0, gamma, lambda);
  assert.equal(result.rows[0].gamma, Math.fround(gamma));
  assert.equal(result.rows[1].gamma, Math.fround(gamma ** 2));
  assert.equal(result.rows[0].lambda, Math.fround(lambda));
  assert.equal(result.rows[1].lambda, Math.fround(lambda ** 2));
  assert.notEqual(result.rows[0].gamma, Math.fround(0.999));
  assert.notEqual(result.rows[0].lambda, Math.fround(0.995));
});
test('explicit trial list and new destination required', async () => {
  await assert.rejects(exportData('.', '.', 0.999, 0.995, ['../escape']), /explicit unique/);
  await assert.rejects(exportData('.', '.', 0.999, 0.995, []), /explicit unique/);
  await assert.rejects(exportData('.', '.', 0.999, 0.995, ['live-r1']), /output already exists/);
});

const readEntries = (f, file) => fs.readFileSync(path.join(f.dir, file), 'utf8').trim().split('\n').map(JSON.parse);
function v2Fixture(withTerminal = false) {
  const f = fixture(), relay = readEntries(f, 'trial/relay.stdout.jsonl');
  for (const x of relay) if (x.event === 'g1_policy_state') {
    x.stream_active = x.round.active;
    x.input = {active: x.round.active, desired_action: x.round.active ? 7 : null};
  }
  const requests = readEntries(f, 'trial/worker.stdin.jsonl');
  for (const x of requests) {
    x.observation_schema = SCHEMA; x.observation[182] = x.observation[183] = 1; x.observation[187] = -1;
  }
  const responses = readEntries(f, 'trial/worker.stdout.jsonl');
  for (const x of responses) x.observation_schema = SCHEMA;
  if (withTerminal) {
    const x = structuredClone(requests.at(-1)); x.seq = 5; x.terminal = true; x.observation[187] = 0;
    requests.push(x); responses.push({type: 'terminal', seq: 5});
  }
  f.lines('trial/relay.stdout.jsonl', relay); f.lines('trial/worker.stdin.jsonl', requests);
  f.lines('trial/worker.stdout.jsonl', responses);
  f.lines('trial/encoder.stdin.jsonl', requests.map(r => relay.find(x => x.event === 'g1_policy_state' && x.observation_sequence === r.seq)));
  f.lines('trial/encoder.stdout.jsonl', [{event: 'projection_manifest', observation_schema: SCHEMA}, ...requests.map(r => {
    const source = relay.find(x => x.event === 'g1_policy_state' && x.observation_sequence === r.seq);
    return {event: 'policy_observation', ready: true, worker_request: r, provenance: {
      source_qpc_ticks: source.clock.qpc_ticks, source_qpc_frequency_hz: source.clock.qpc_frequency_hz,
      stream_active: source.stream_active, projected_busy: true, busy_projection: 'dispatched_request_v4_duration'}};
  })]);
  return f;
}
test('actual v2 export preserves recorded pre-action yaw and all non187 decision bytes', async () => {
  const f = v2Fixture(), actual = await readTrial(f.dir, 0, 0.999, 0.995, SCHEMA);
  assert.deepEqual(actual.rows.map(x => x.obs[187]), [-1, -1]);
  assert.equal(actual.rows[1].terminal, true); assert.equal(actual.rows[1].policyWeight, 0);
  assert.equal(actual.ledger[1].owned_yaw_evidence.desired_action, 7);
  // The saved sampled action is 2; its yaw must not replace desired category7.
  assert.equal(actual.rows[1].action, 2);
  const old = structuredClone(actual.rows); old.forEach(x => { x.obs[187] = 0; });
  const legacy = pack(old, 1), current = pack(actual.rows, 1, SCHEMA);
  assert.equal(current.toString('ascii', 0, 8), 'REKRL002'); assert.equal(current.readUInt32LE(8), 2);
  const restored = Buffer.from(current); legacy.copy(restored, 0, 0, 12);
  old.forEach((_, i) => restored.writeFloatLE(0, HEADER + i * ROW + 32 + 4 * 187));
  assert.deepEqual(restored, legacy);
  const output = `${f.dir}-export`, m = await exportData(path.dirname(f.dir), output, 0.999, 0.995, [path.basename(f.dir)], SCHEMA);
  assert.equal(m.origin, 'actual_recorded_v2_worker_inputs'); assert.equal(m.observation_schema, SCHEMA);
  assert.equal(m.actual_behavior_checkpoint_sha256, '2'.repeat(64));
  assert.deepEqual(fs.readFileSync(path.join(output, m.binary)), current);
});
test('actual inactive v2 terminal input permits missing intent but requires zero187 and explicit schema', async () => {
  const f = v2Fixture(true), result = await readTrial(f.dir, 0, 0.999, 0.995, SCHEMA);
  assert.equal(result.rows.length, 2); assert.equal(result.report.worker_terminal_inputs, 1);
  assert.equal(result.rows[1].obs[187], -1);
  const requests = readEntries(f, 'trial/worker.stdin.jsonl'); delete requests.at(-1).observation_schema;
  f.lines('trial/worker.stdin.jsonl', requests);
  await assert.rejects(readTrial(f.dir, 0, 0.999, 0.995, SCHEMA), /worker observation schema/);
  const g = v2Fixture(true), worker = readEntries(g, 'trial/worker.stdin.jsonl'), encoded = readEntries(g, 'trial/encoder.stdout.jsonl');
  worker.at(-1).observation[187] = 1; encoded.at(-1).worker_request.observation[187] = 1;
  g.lines('trial/worker.stdin.jsonl', worker); g.lines('trial/encoder.stdout.jsonl', encoded);
  await assert.rejects(readTrial(g.dir, 0, 0.999, 0.995, SCHEMA), /recorded owned-yaw column/);
});
test('actual v2 rejects unknown ownership, busy mismatch and reconstructed chosen-action yaw', async () => {
  for (const mutation of ['unknown', 'inactive', 'busy', 'chosen-action']) {
    const f = v2Fixture(), relay = readEntries(f, 'trial/relay.stdout.jsonl'), inputs = readEntries(f, 'trial/encoder.stdin.jsonl');
    const worker = readEntries(f, 'trial/worker.stdin.jsonl'), encoded = readEntries(f, 'trial/encoder.stdout.jsonl');
    if (mutation === 'unknown') { relay[1].input.desired_action = 0; inputs[0].input.desired_action = 0; }
    if (mutation === 'inactive') { relay[1].input.active = false; inputs[0].input.active = false; }
    if (mutation === 'busy') encoded[1].provenance.projected_busy = false;
    if (mutation === 'chosen-action') { worker[0].observation[187] = 0; encoded[1].worker_request.observation[187] = 0; }
    f.lines('trial/relay.stdout.jsonl', relay); f.lines('trial/encoder.stdin.jsonl', inputs);
    f.lines('trial/worker.stdin.jsonl', worker); f.lines('trial/encoder.stdout.jsonl', encoded);
    await assert.rejects(readTrial(f.dir, 0, 0.999, 0.995, SCHEMA));
  }
});
test('actual v2 binds full source snapshot, encoder arrays and every declared schema', async () => {
  for (const mutation of ['hash', 'observation', 'mask', 'clock', 'manifest', 'encoder', 'action', 'ready']) {
    const f = v2Fixture(), inputs = readEntries(f, 'trial/encoder.stdin.jsonl');
    const encoded = readEntries(f, 'trial/encoder.stdout.jsonl'), responses = readEntries(f, 'trial/worker.stdout.jsonl');
    if (mutation === 'hash') inputs[0].unexpected = 1;
    if (mutation === 'observation') encoded[1].worker_request.observation[10] = 1;
    if (mutation === 'mask') encoded[1].worker_request.mask[0] = 0;
    if (mutation === 'clock') encoded[1].provenance.source_qpc_ticks++;
    if (mutation === 'manifest') encoded[0].observation_schema = LEGACY;
    if (mutation === 'encoder') encoded[1].worker_request.observation_schema = LEGACY;
    if (mutation === 'action') responses[1].observation_schema = LEGACY;
    if (mutation === 'ready') responses[0].observation_schema = LEGACY;
    f.lines('trial/encoder.stdin.jsonl', inputs); f.lines('trial/encoder.stdout.jsonl', encoded); f.lines('trial/worker.stdout.jsonl', responses);
    await assert.rejects(readTrial(f.dir, 0, 0.999, 0.995, SCHEMA));
  }
});
test('v2 CLI requires exact explicit option and legacy default still rejects v2', async () => {
  const f = v2Fixture(), base = [__dirname + '/authentic_trajectory_data.cjs', path.dirname(f.dir), `${f.dir}-cli`, '0.999', '0.995', path.basename(f.dir)];
  assert.equal(spawnSync(process.execPath, base).status, 1);
  for (const flags of [['--observation-schema=unknown'], [`--observation-schema=${SCHEMA}`, `--observation-schema=${SCHEMA}`]]) {
    const result = spawnSync(process.execPath, [...base, ...flags], {encoding: 'utf8'});
    assert.equal(result.status, 1); assert.match(result.stderr, /unknown or duplicate/);
  }
  assert.equal(spawnSync(process.execPath, [...base, `--observation-schema=${SCHEMA}`]).status, 0);
  await assert.rejects(readTrial(fixture().dir, 0, 0.999, 0.995, SCHEMA), /worker observation schema/);
});
