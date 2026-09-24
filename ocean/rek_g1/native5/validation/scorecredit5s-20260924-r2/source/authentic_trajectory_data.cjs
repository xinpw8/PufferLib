'use strict';
// Offline adaptation of actual worker decisions. No game connection or physics.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const readline = require('node:readline');
const HEADER = 256, ROW = 1128, OBS = 223, ACTIONS = 33;
const {LEGACY: LEGACY_SCHEMA, SCHEMA: OWNED_SCHEMA, ownedYawEvidence} = require('./owned_yaw_export_evidence.cjs');
const BALANCE_SCHEMA = 'rek.native5.scaled_polar_xy.balance8_v1';
const requireValue = (ok, message) => { if (!ok) throw new Error(message); };
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const finite = x => Number.isFinite(x) && Number.isFinite(Math.fround(x));
function ownedMode(schema) {
  requireValue(schema === LEGACY_SCHEMA || schema === OWNED_SCHEMA || schema === BALANCE_SCHEMA, 'unsupported observation schema');
  return schema === OWNED_SCHEMA;
}
function potential(own, opponent) {
  const d = Math.fround(own - opponent);
  return Math.fround(d / Math.fround(5 + Math.abs(d)));
}
function reward(gamma, own, opponent, nextOwn, nextOpponent, terminal, outcome) {
  // Same float arithmetic and terminal potential as round_reward.h.
  return Math.fround(Math.fround(outcome + Math.fround(gamma *
    (terminal ? 0 : potential(nextOwn, nextOpponent)))) - potential(own, opponent));
}
function clock(x) {
  requireValue(Number.isSafeInteger(x?.qpc_ticks) && x.qpc_ticks > 0 &&
    Number.isSafeInteger(x.qpc_frequency_hz) && x.qpc_frequency_hz > 0, 'invalid QPC clock');
  return x;
}
async function jsonLines(file, visit) {
  const before = fs.statSync(file), hash = crypto.createHash('sha256');
  const input = fs.createReadStream(file); input.on('data', b => hash.update(b));
  let lines = 0;
  for await (const line of readline.createInterface({ input, crlfDelay: Infinity })) {
    ++lines; requireValue(line.trim().length > 0, `blank JSONL line: ${path.basename(file)}`);
    visit(JSON.parse(line), lines);
  }
  const after = fs.statSync(file);
  requireValue(before.size === after.size && before.mtimeMs === after.mtimeMs, 'input changed during read');
  return { file: path.basename(file), bytes: before.size, lines, sha256: hash.digest('hex') };
}
function json(file) {
  const b = fs.readFileSync(file);
  return { value: JSON.parse(b), input: { file: path.basename(file), bytes: b.length, sha256: sha(b) } };
}
function unique(map, key, value, label) {
  requireValue(!map.has(key), `duplicate ${label}`); map.set(key, value);
}
function bindEvidence(report, inputs) {
  for (const input of inputs) {
    const prior = report.inputs?.find(x => path.basename(x.file) === input.file);
    if (prior) requireValue(prior.sha256 === input.sha256 && prior.bytes === input.bytes,
      `analysis source binding changed: ${input.file}`);
  }
}
function validateRow(r) {
  requireValue(Number.isInteger(r.sequence) && r.sequence >= 0 && r.sequence <= 0xffffffff, 'invalid sequence');
  requireValue(r.reset === 0 || r.reset === 1, 'invalid reset');
  requireValue(Number.isInteger(r.action) && r.action >= 0 && r.action < ACTIONS, 'invalid action');
  requireValue(r.obs.length === OBS && r.obs.every(finite), 'invalid observation');
  requireValue(r.mask.length === ACTIONS && r.mask.every(x => x === 0 || x === 1) && r.mask[r.action] === 1,
    'invalid actual action mask');
  requireValue(Number.isFinite(r.time) && r.time >= 0 && Number.isFinite(r.nextTime) &&
    r.nextTime > r.time && r.dt > 0 && Math.abs(r.nextTime - r.time - r.dt) < 1e-10, 'invalid transition time');
  requireValue(r.gamma > 0 && r.gamma <= 1 && r.lambda > 0 && r.lambda <= 1, 'invalid discount');
  requireValue([r.own, r.opponent, r.nextOwn, r.nextOpponent].every(x => Number.isInteger(x) && x >= 0 && x <= 32767),
    'invalid awarded point counter');
  requireValue(r.nextOwn >= r.own && r.nextOpponent >= r.opponent, 'score counter regressed');
  requireValue(typeof r.terminal === 'boolean' && typeof r.applied === 'boolean', 'invalid outcome flags');
  requireValue(r.policyWeight === (r.applied ? 1 : 0) && r.valueWeight === 1, 'invalid loss weights');
  requireValue(r.applied || r.terminal, 'nonterminal rejected action requires separate handling');
  requireValue([-1, 0, 1].includes(r.outcome) && (r.terminal || r.outcome === 0), 'invalid terminal outcome');
  requireValue(r.reward === reward(r.gamma, r.own, r.opponent, r.nextOwn, r.nextOpponent, r.terminal, r.outcome),
    'reward contract mismatch');
  requireValue(Number.isInteger(r.sourceSeq) && r.sourceSeq >= 0 && Number.isInteger(r.nextSourceSeq) &&
    r.nextSourceSeq > r.sourceSeq && r.nextSourceSeq <= 0xffffffff, 'invalid source sequence');
}
function pack(rows, roundCount, observationSchema = LEGACY_SCHEMA) {
  const owned = ownedMode(observationSchema);
  const b = Buffer.alloc(HEADER + ROW * rows.length);
  b.write(owned ? 'REKRL002' : 'REKRL001'); b.writeUInt32LE(owned ? 2 : 1, 8); b.writeUInt32LE(OBS, 12); b.writeUInt32LE(ACTIONS, 16);
  b.writeUInt32LE(rows.length, 20); b.writeUInt32LE(ROW, 24); b.writeUInt32LE(roundCount, 28);
  b.fill(1, 32, 255); // Original behavior is unmasked. Never apply the human BC mask here.
  rows.forEach((r, i) => {
    validateRow(r); const o = HEADER + i * ROW;
    b.writeUInt32LE(0, o); b.writeUInt32LE(r.sequence, o + 4); b.writeUInt32LE(r.reset, o + 8);
    b.writeInt32LE(r.action, o + 12); b.writeFloatLE(r.policyWeight, o + 16); b.writeDoubleLE(r.time, o + 24);
    r.obs.forEach((v, j) => b.writeFloatLE(v, o + 32 + 4 * j));
    r.mask.forEach((v, j) => b.writeFloatLE(v, o + 924 + 4 * j));
    b.writeDoubleLE(r.nextTime, o + 1056); b.writeDoubleLE(r.dt, o + 1064);
    [r.gamma, r.lambda, r.reward, r.outcome].forEach((v, j) => b.writeFloatLE(v, o + 1072 + 4 * j));
    b.writeUInt32LE(r.sourceSeq, o + 1088); b.writeUInt32LE(r.nextSourceSeq, o + 1092);
    [r.own, r.opponent, r.nextOwn, r.nextOpponent].forEach((v, j) => b.writeInt32LE(v, o + 1096 + 4 * j));
    b.writeUInt32LE(Number(r.terminal), o + 1112); b.writeUInt32LE(Number(r.applied), o + 1116);
    b.writeFloatLE(r.valueWeight, o + 1120);
  });
  return b;
}
async function readTrial(directory, sequence, gamma20ms, lambda20ms, observationSchema = LEGACY_SCHEMA, behavior = null) {
  const owned = ownedMode(observationSchema);
  const balance = observationSchema === BALANCE_SCHEMA;
  const trial = path.join(directory, 'trial'), inputs = [], read = async (name, cb) =>
    inputs.push(await jsonLines(path.join(trial, name), cb));
  const summary = json(path.join(trial, 'summary.json'));
  const contact = json(path.join(directory, 'contact-analysis', 'summary.json'));
  const referee = json(path.join(directory, 'referee-validation', 'live-referee-validation.json'));
  const s = summary.value, roundId = s.round_identity_sha256, slot = s.local_slot;
  requireValue(contact.value.completed_policy_round === true && contact.value.terminal_evidence_consistent === true &&
    contact.value.terminal_point_counters_consistent === true && referee.value.verification_passed === true,
    'existing strict evidence does not establish a completed verified round');
  requireValue(contact.value.round_identity_sha256 === roundId && contact.value.local_slot === slot &&
    contact.value.checkpoint_sha256 === s.checkpoint_sha256 && referee.value.round_identities.length === 1 &&
    referee.value.round_identities[0] === roundId && [0, 1].includes(slot), 'round evidence identity mismatch');
  const requests = new Map(), predictions = new Map(), sources = new Map(), sends = new Map(), acks = new Map();
  const ownedSources = new Map(), encoderInputs = new Map(), encoders = new Map();
  const unavailable = []; let ready = null, workerTerminals = 0;
  await read('worker.stdin.jsonl', x => {
    requireValue(x.type === 'step' && x.round_id === roundId, 'unexpected worker input or round reset');
    requireValue(x.observation_schema === observationSchema, 'unsupported worker observation schema');
    unique(requests, x.seq, x, 'worker request'); if (x.terminal) ++workerTerminals;
  });
  await read('worker.stdout.jsonl', x => {
    if (x.type === 'ready') { requireValue(ready === null, 'duplicate worker ready'); ready = x; }
    else if (x.type === 'action') {
      if (owned || balance) requireValue(x.observation_schema === observationSchema, 'unsupported action observation schema');
      unique(predictions, x.seq, x, 'worker action');
    }
    else requireValue(x.type === 'terminal' && requests.get(x.seq)?.terminal === true, 'unexpected worker response');
  });
  requireValue(ready?.observation_schema === observationSchema, 'unsupported ready observation schema');
  const unmaskedHash = sha(Buffer.alloc(OBS, 1));
  requireValue(!behavior || (Number.isSafeInteger(behavior.seed) && behavior.seed >= 0 &&
    /^[a-f0-9]{64}$/.test(behavior.feature_mask_sha256)), 'invalid explicit behavior identity');
  const expectedSeed = behavior ? behavior.seed : 73;
  const featureIdentity = behavior ? (ready?.feature_mask_sha256 ?? unmaskedHash) === behavior.feature_mask_sha256 : !ready?.feature_mask_sha256;
  requireValue(ready?.checkpoint_sha256 === s.checkpoint_sha256 && ready.selection === 'sampled' && ready.seed === expectedSeed &&
    ready.precision === 'bf16' && ready.hidden_size === 256 && ready.num_layers === 2 &&
    ready.observations === OBS && ready.actions === ACTIONS && featureIdentity, 'unsupported behavior identity');
  await read('relay.stdin.jsonl', x => {
    if (x.type === 'policy_action') unique(sends, x.observation_sequence, x, 'sent action');
  });
  await read('relay.stdout.jsonl', x => {
    if (x.event === 'g1_policy_state') {
      requireValue(x.round_identity_sha256 === roundId && x.local_slot === slot, 'source identity changed');
      unique(sources, x.observation_sequence, { seq: x.observation_sequence, clock: clock(x.clock), round: x.round }, 'source');
      if (owned || balance) unique(ownedSources, x.observation_sequence, { hash: sha(JSON.stringify(x)),
        event: x.event, observation_sequence: x.observation_sequence, round_identity_sha256: x.round_identity_sha256,
        round: {active: x.round.active}, stream_active: x.stream_active, input: x.input, clock: x.clock }, 'owned source');
    } else if (x.event === 'g1_policy_action') unique(acks, x.observation_sequence, x, 'action acknowledgement');
  });
  if (owned || balance) {
    await read('encoder.stdin.jsonl', x => {
      requireValue(x.event === 'g1_policy_state' && x.round_identity_sha256 === roundId, 'unsupported encoder input');
      unique(encoderInputs, x.observation_sequence, sha(JSON.stringify(x)), 'encoder source');
    });
    let manifest = null;
    await read('encoder.stdout.jsonl', x => {
      if (x.event === 'projection_manifest') {
        requireValue(manifest === null && x.observation_schema === observationSchema, 'unsupported encoder manifest schema');
        manifest = x;
      } else {
        requireValue(x.event === 'policy_observation' && typeof x.ready === 'boolean', 'unsupported encoder response');
        if (x.ready) {
          requireValue(x.worker_request?.observation_schema === observationSchema, 'unsupported encoder observation schema');
          unique(encoders, x.worker_request.seq, x, 'encoded request');
        }
      }
    });
    requireValue(manifest !== null, 'missing encoder schema manifest');
    for (const request of requests.values()) {
      const source = ownedSources.get(request.seq);
      requireValue(source && source.hash === encoderInputs.get(request.seq), 'encoder input is not the identical relay source snapshot');
      requireValue(Array.isArray(request.observation) && request.observation.length === OBS && request.observation.every(finite) &&
        Array.isArray(request.mask) && request.mask.length === ACTIONS && request.mask.every(x => x === 0 || x === 1), 'invalid owned worker arrays');
      if (owned) ownedYawEvidence(request.observation.map(Math.fround), request.mask, request.seq, source,
        encoders.get(request.seq), request, roundId, observationSchema);
      else {
        const encoded = encoders.get(request.seq)?.worker_request;
        requireValue(encoded?.round_id === roundId && encoded.observation_schema === observationSchema &&
          JSON.stringify(encoded.observation.map(Math.fround)) === JSON.stringify(request.observation.map(Math.fround)) &&
          JSON.stringify(encoded.mask) === JSON.stringify(request.mask), 'recorded balance8 encoder/worker mismatch');
      }
    }
  }
  await read('orchestrator.jsonl', x => {
    requireValue(!['stale_prediction_discarded', 'unmatched_action_ack', 'error'].includes(x.event), 'unhandled decision path');
    if (x.event === 'observation_unavailable') unavailable.push(x.unavailable ?? [x.reason]);
  });
  bindEvidence(contact.value, inputs); bindEvidence(referee.value, inputs);
  requireValue(predictions.size > 0 && sends.size === predictions.size && acks.size === predictions.size &&
    requests.size === predictions.size + workerTerminals && s.predictions === predictions.size, 'decision count mismatch');
  const ordered = [...predictions.values()], rows = [], ledger = [];
  const allSources = [...sources.values()], terminalSources = allSources.filter(x => x.round.active === false);
  requireValue(terminalSources.length === 1, 'exactly one observed terminal state required');
  const terminal = terminalSources[0], initial = sources.get(ordered[0].seq), origin = initial.clock.qpc_ticks;
  requireValue(terminal.round.redo === false && terminal.round.result === 'WonByPoints' &&
    terminal.round.winner_index === s.final_round.winner_index, 'unsupported or inconsistent terminal');
  requireValue(initial.round.clean_hits.every(x => x === 0), 'opening points precede worker history');
  let sourcePrevious = null;
  for (const x of allSources) {
    requireValue(x.round.number === terminal.round.number && x.round.redo === false, 'round lifecycle changed');
    if (sourcePrevious) requireValue(x.seq > sourcePrevious.seq && x.clock.qpc_ticks > sourcePrevious.clock.qpc_ticks &&
      x.clock.qpc_frequency_hz === sourcePrevious.clock.qpc_frequency_hz &&
      x.round.clean_hits.every((v, j) => v >= sourcePrevious.round.clean_hits[j]), 'source chronology or score regressed');
    sourcePrevious = x;
  }
  for (let i = 0; i < ordered.length; ++i) {
    const prediction = ordered[i], request = requests.get(prediction.seq), source = sources.get(prediction.seq);
    const next = i + 1 < ordered.length ? sources.get(ordered[i + 1].seq) : terminal;
    const sent = sends.get(prediction.seq), ack = acks.get(prediction.seq);
    requireValue(source && next && request && !request.terminal && source.round.active === true, 'missing decision source');
    requireValue(prediction.round_id === roundId && request.round_id === roundId && sent.round_identity_sha256 === roundId &&
      ack.round_identity_sha256 === roundId && sent.action === prediction.action && ack.action === prediction.action &&
      ack.request_id === sent.request_id && ack.observation_sequence === prediction.seq &&
      prediction.checkpoint_sha256 === ready.checkpoint_sha256 && prediction.selection === ready.selection &&
      prediction.precision === ready.precision && prediction.decision_index === i + 1 &&
      prediction.recurrent_reset === (i === 0) && prediction.round_changed === (i === 0), 'decision identity mismatch');
    if (behavior) requireValue((prediction.feature_mask_sha256 ?? unmaskedHash) === behavior.feature_mask_sha256,
      'decision feature mask identity mismatch');
    const mask = request.mask.map(x => Number(x));
    requireValue(prediction.legal_actions === mask.reduce((a, b) => a + b, 0), 'legal mask mismatch');
    clock(ack.clock);
    requireValue(ack.clock.qpc_frequency_hz === source.clock.qpc_frequency_hz &&
      ack.clock.qpc_ticks >= source.clock.qpc_ticks, 'ack clock inconsistent');
    requireValue(ack.applied === true ? ack.clock.qpc_ticks < next.clock.qpc_ticks :
      i === ordered.length - 1 && ack.reason === 'policy_stream_not_owned' && ack.clock.qpc_ticks >= next.clock.qpc_ticks,
      'unsupported ack/next-state boundary');
    const end = i === ordered.length - 1, freq = source.clock.qpc_frequency_hz;
    const dt = (next.clock.qpc_ticks - source.clock.qpc_ticks) / freq;
    const own = source.round.clean_hits[slot], opponent = source.round.clean_hits[1 - slot];
    const nextOwn = next.round.clean_hits[slot], nextOpponent = next.round.clean_hits[1 - slot];
    requireValue(Math.fround(request.observation[190]) === own && Math.fround(request.observation[191]) === opponent,
      'worker scoreboard features disagree with its source');
    const gamma = Math.fround(Math.pow(gamma20ms, dt / 0.02));
    const lambda = Math.fround(Math.pow(lambda20ms, dt / 0.02));
    const outcome = end ? (terminal.round.winner_index === slot ? 1 : -1) : 0;
    const row = { sequence, reset: Number(i === 0), action: prediction.action,
      policyWeight: Number(ack.applied), valueWeight: 1, obs: request.observation.map(Math.fround), mask,
      time: (source.clock.qpc_ticks - origin) / freq, nextTime: (next.clock.qpc_ticks - origin) / freq, dt,
      gamma, lambda, reward: reward(gamma, own, opponent, nextOwn, nextOpponent, end, outcome), outcome,
      sourceSeq: source.seq, nextSourceSeq: next.seq, own, opponent, nextOwn, nextOpponent, terminal: end, applied: ack.applied };
    validateRow(row); rows.push(row);
    ledger.push({ sequence, decision_index: i + 1, source_sequence: source.seq, next_source_sequence: next.seq,
      request_id: sent.request_id, action: row.action, applied: ack.applied, acknowledgement_reason: ack.reason,
      source_qpc_ticks: source.clock.qpc_ticks, ack_qpc_ticks: ack.clock.qpc_ticks,
      next_qpc_ticks: next.clock.qpc_ticks, qpc_frequency_hz: freq, dt_seconds: dt,
      terminal_after: end, policy_weight: row.policyWeight, value_weight: 1,
      ...(owned ? {owned_yaw_evidence: ownedYawEvidence(row.obs, row.mask, source.seq,
        ownedSources.get(source.seq), encoders.get(source.seq), request, roundId, observationSchema)} : {}) });
  }
  const scoreInputs = [], awards = [0, 0], seenAwards = new Set();
  scoreInputs.push(await jsonLines(path.join(directory, 'contact-analysis', 'score-events.jsonl'), x => {
    requireValue(Number.isInteger(x.index) && !seenAwards.has(x.index) && [0, 1].includes(x.fighter_index) &&
      Number.isInteger(x.points_awarded) && x.points_awarded > 0, 'invalid native score award');
    seenAwards.add(x.index); awards[x.fighter_index] += x.points_awarded;
  }));
  requireValue(awards.every((v, j) => v === terminal.round.clean_hits[j]) &&
    contact.value.score_events === seenAwards.size, 'native awards do not reconcile with observed terminal');
  return { rows, ledger, report: { trial_id: path.basename(directory), sequence, round_identity_sha256: roundId,
    checkpoint_sha256: ready.checkpoint_sha256, seed: ready.seed, precision: ready.precision,
    ...(owned ? {observation_schema: observationSchema, recorded_owned_yaw_verified: true} : {}),
    ...(balance ? {observation_schema: observationSchema, recorded_balance8_all223_and_mask_verified: true} : {}),
    feature_mask: 'unmasked_original_behavior', rows: rows.length, applied: rows.filter(x => x.applied).length,
    terminal_race_rejected: rows.filter(x => !x.applied).length, recurrent_resets: 1,
    worker_terminal_inputs: workerTerminals, observed_source_count: sources.size,
    source_snapshots_not_used_by_worker: sources.size - requests.size, unavailable_observations: unavailable,
    first_source_time_remaining: allSources[0].round.time_remaining,
    maximum_decision_dt_seconds: Math.max(...rows.map(x => x.dt)),
    minimum_decision_dt_seconds: Math.min(...rows.map(x => x.dt)),
    terminal_outcome: rows.at(-1).outcome, awarded_points_by_slot: awards, local_slot: slot,
    native_score_awards: seenAwards.size, source_clock_origin_qpc_ticks: origin,
    inputs: [...inputs, { ...summary.input, file: 'trial/summary.json' },
      { ...contact.input, file: 'contact-analysis/summary.json' },
      { ...referee.input, file: 'referee-validation/live-referee-validation.json' }, ...scoreInputs] } };
}
async function exportData(root, output, gamma20ms, lambda20ms, ids, observationSchema = LEGACY_SCHEMA) {
  const owned = ownedMode(observationSchema);
  requireValue(gamma20ms > 0 && gamma20ms <= 1 && lambda20ms > 0 && lambda20ms <= 1, 'invalid reference discounts');
  requireValue(ids.length > 0 && new Set(ids).size === ids.length &&
    ids.every(x => /^live-[a-zA-Z0-9_-]+$/.test(x)), 'explicit unique trial IDs required');
  requireValue(!fs.existsSync(output), 'output already exists');
  const rounds = [];
  for (let i = 0; i < ids.length; ++i) rounds.push(await readTrial(path.join(root, ids[i]), i, gamma20ms, lambda20ms, observationSchema));
  requireValue(new Set(rounds.map(x => x.report.checkpoint_sha256)).size === 1, 'mixed behavior checkpoint');
  const rows = rounds.flatMap(x => x.rows), binary = pack(rows, rounds.length, observationSchema), mask = Buffer.alloc(OBS, 1);
  const manifest = { schema: owned ? 'rek.authentic_trajectory_dataset.owned_yaw_v2' : 'rek.authentic_trajectory_dataset.v1', created_utc: new Date().toISOString(),
    ...(owned ? {observation_schema: observationSchema, origin: 'actual_recorded_v2_worker_inputs',
      actual_behavior_checkpoint_sha256: rounds[0].report.checkpoint_sha256,
      evidence_helper_sha256: sha(fs.readFileSync(require.resolve('./owned_yaw_export_evidence.cjs')))} : {}),
    exporter_sha256: sha(fs.readFileSync(__filename)), binary: owned ? 'authentic-trajectories-owned-yaw-v2.bin' : 'authentic-trajectories.bin', binary_sha256: sha(binary),
    header_bytes: HEADER, row_bytes: ROW, observations: OBS, actions: ACTIONS, rows: rows.length,
    feature_mask_sha256: sha(mask), checkpoint_sha256: rounds[0].report.checkpoint_sha256,
    applied: rows.filter(x => x.applied).length, terminal_race_rejected: rows.filter(x => !x.applied).length,
    reference_gamma_per_20ms: gamma20ms, reference_lambda_per_20ms: lambda20ms,
    discount_rule: 'pow(reference, actual_source_to_next_source_QPC_seconds/0.02), rounded_float32',
    reward_rule: 'terminal_outcome + gamma_t*Phi(next_observed_points) - Phi(current_observed_points); terminal_Phi=0; Phi=d/(5+abs(d))',
    action_semantics: 'actual_issued_policy_action; acknowledgement proves local acceptance only; terminal-race requests preserved',
    mask_semantics: 'actual_saved_behavior_legal_action_mask; fixed_input_feature_mask_is_all_ones',
    recurrence: 'exact_worker_decision_order; fresh_worker_per_round; omitted_source_snapshots_never_inserted; no_50Hz_resampling',
    loss_semantics: 'actor_weight_zero_for_terminal_race_reject; value_weight_one; complete_terminal_bootstrap_zero',
    time_semantics: 'client_observation_QPC_time; asynchronous_receipt_transition_not_server_execution_time',
    split_semantics: 'all_explicitly_selected_rounds_are_development_training; no_claim_of_future_holdout',
    behavior_replay: 'required_before_training; original_BF16_H256_L2_seed73_unmasked_worker; chosen_actions_must_match',
    limits: [`${rounds.length} recorded rounds provide ${rounds.length} episodes, not independent evidence from every decision.`,
      'Received score changes supply rewards without inferred hit, fall, attacker, or executed-move labels.',
      'Unobserved startup precedes first worker decision and is not fabricated.',
      'Actor training-forward versus sequential behavior-forward parity remains a separate measured prerequisite.'],
    rounds: rounds.map(x => x.report) };
  fs.mkdirSync(output, { recursive: false });
  fs.writeFileSync(path.join(output, manifest.binary), binary, { flag: 'wx' });
  fs.writeFileSync(path.join(output, 'feature-mask.bin'), mask, { flag: 'wx' });
  fs.writeFileSync(path.join(output, 'transition-ledger.jsonl'), rounds.flatMap(x => x.ledger).map(x => JSON.stringify(x)).join('\n') + '\n', { flag: 'wx' });
  fs.writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  return manifest;
}
if (require.main === module) {
  const [root, output, gamma, lambda, ...args] = process.argv.slice(2);
  const flags = args.filter(x => x.startsWith('--'));
  const valid = flags.length === 0 || (flags.length === 1 && flags[0] === `--observation-schema=${OWNED_SCHEMA}`);
  const schema = flags.length === 0 ? LEGACY_SCHEMA : OWNED_SCHEMA;
  (valid ? exportData(root, output, Number(gamma), Number(lambda), args.filter(x => !x.startsWith('--')), schema)
    : Promise.reject(new Error('unknown or duplicate exporter option'))).then(x =>
    console.log(JSON.stringify({ output, rows: x.rows, applied: x.applied, rejected: x.terminal_race_rejected,
      binary_sha256: x.binary_sha256, feature_mask_sha256: x.feature_mask_sha256 })))
    .catch(e => { console.error(e.message); process.exitCode = 1; });
}
module.exports = { HEADER, ROW, OBS, ACTIONS, potential, reward, pack, validateRow, readTrial, exportData };
