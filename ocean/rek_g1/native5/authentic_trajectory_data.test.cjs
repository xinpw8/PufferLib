'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), os = require('node:os'), path = require('node:path');
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
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rek-authentic-cpu-fixture-'));
  for (const folder of ['trial', 'contact-analysis', 'referee-validation']) fs.mkdirSync(path.join(dir, folder));
  const roundId = '1'.repeat(64), checkpoint = '2'.repeat(64);
  const c = t => ({ qpc_ticks: 1000000 + t, qpc_frequency_hz: 1000000 });
  const round = (active, points) => ({ number: 1, duration: 120, time_remaining: active ? 119.7 : 0,
    active, redo: false, clean_hits: points, result: active ? 'InProgress' : 'WonByPoints', winner_index: active ? -1 : 0 });
  const source = (seq, t, active, points) => ({ event: 'g1_policy_state', observation_sequence: seq,
    round_identity_sha256: roundId, local_slot: 0, clock: c(t), round: round(active, points) });
  const req = (seq, points) => { const observation = Array(223).fill(0); observation[190] = points;
    return { type: 'step', seq, round_id: roundId, terminal: false, observation, mask: Array(33).fill(1) }; };
  const ready = { type: 'ready', checkpoint_sha256: checkpoint, selection: 'sampled', seed: 73, precision: 'bf16',
    hidden_size: 256, num_layers: 2, observations: 223, actions: 33 };
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
