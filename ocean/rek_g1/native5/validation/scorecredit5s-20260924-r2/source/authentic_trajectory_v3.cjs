'use strict';
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const base = require('./authentic_trajectory_data.cjs');
const HEADER = 384, SCHEMA = 'rek.native5.scaled_polar_xy.balance8_v1';
const check = (ok, message) => { if (!ok) throw new Error(message); };
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
function pinned(file, expected) {
  check(path.isAbsolute(file) && /^[a-f0-9]{64}$/.test(expected), 'absolute pinned artifact required');
  const bytes = fs.readFileSync(file);
  check(sha(bytes) === expected, `artifact hash mismatch: ${file}`);
  return { path: file, bytes: bytes.length, sha256: expected };
}
function pack(rows, rounds, identityBytes, mask, pins) {
  check(mask.length === base.OBS && [...mask].every(x => x === 0 || x === 1), 'invalid feature mask');
  const old = base.pack(rows, rounds, SCHEMA), out = Buffer.alloc(HEADER + rows.length * base.ROW);
  old.copy(out, 0, 0, base.HEADER); old.copy(out, HEADER, base.HEADER);
  out.write('REKRL003'); out.writeUInt32LE(3, 8); mask.copy(out, 32);
  [sha(identityBytes), pins.worker.sha256, pins.checkpoint.sha256, pins.native_object.sha256].forEach((x, i) => {
    check(/^[a-f0-9]{64}$/.test(x), 'invalid identity digest'); Buffer.from(x, 'hex').copy(out, 256 + 32 * i);
  });
  rows.forEach((r, i) => {
    check(Number.isSafeInteger(r.behaviorSeed) && r.behaviorSeed >= 0, 'invalid per-round seed');
    const seed = BigInt(r.behaviorSeed), offset = HEADER + i * base.ROW;
    out.writeUInt32LE(Number(seed & 0xffffffffn), offset + 20);
    out.writeUInt32LE(Number(seed >> 32n), offset + 1124);
  });
  return out;
}
async function exportV3(configPath, output) {
  check(!fs.existsSync(output), 'output already exists');
  const configBytes = fs.readFileSync(configPath), c = JSON.parse(configBytes);
  check(c.schema === 'rek.authentic_export_config.v3' && c.observation_schema === SCHEMA, 'unsupported export config');
  check(c.gamma_per_20ms > 0 && c.gamma_per_20ms <= 1 && c.lambda_per_20ms > 0 && c.lambda_per_20ms <= 1,
    'explicit reference discounts required');
  check(Array.isArray(c.rounds) && c.rounds.length > 0 && new Set(c.rounds.map(x => x.id)).size === c.rounds.length,
    'explicit distinct rounds required');
  const pins = {};
  for (const key of ['worker', 'checkpoint', 'native_object', 'feature_mask']) pins[key] = pinned(c[key].path, c[key].sha256);
  const mask = fs.readFileSync(pins.feature_mask.path);
  check(mask.length === base.OBS && [...mask].every(x => x === 0 || x === 1), 'invalid feature mask');
  const unmasked = mask.every(x => x === 1), rounds = [];
  let begin = 0;
  for (let sequence = 0; sequence < c.rounds.length; ++sequence) {
    const spec = c.rounds[sequence];
    check(/^[a-zA-Z0-9_-]+$/.test(spec.id) && path.isAbsolute(spec.evidence), 'invalid explicit round path');
    check(Number.isSafeInteger(spec.seed) && spec.seed >= 0, 'invalid recorded round seed');
    const capturedConfigBytes = fs.readFileSync(spec.config), capturedConfig = JSON.parse(capturedConfigBytes);
    const argv = capturedConfig.worker;
    check(Array.isArray(argv) && argv.length === 7 && argv[0] === pins.worker.path &&
      argv[1] === pins.checkpoint.path && argv[2] === pins.checkpoint.sha256 && argv[3] === String(spec.seed) &&
      capturedConfig.checkpoint_sha256 === pins.checkpoint.sha256 && capturedConfig.observation_schema === SCHEMA &&
      argv[6] === '--observation-schema=' + SCHEMA, 'captured worker invocation mismatch');
    if (argv.length === 4) check(unmasked, 'old worker invocation cannot support masked behavior');
    else check(argv[4] === 'sampled' && sha(fs.readFileSync(argv[5])) === pins.feature_mask.sha256,
      'captured sampled worker feature mask mismatch');
    const result = await base.readTrial(spec.evidence, sequence, c.gamma_per_20ms, c.lambda_per_20ms, SCHEMA,
      {seed: spec.seed, feature_mask_sha256: pins.feature_mask.sha256});
    check(result.report.checkpoint_sha256 === pins.checkpoint.sha256, 'mixed checkpoint');
    result.rows.forEach(x => { x.behaviorSeed = spec.seed; });
    Object.assign(result.report, {trial_id: spec.id, row_begin: begin, row_end: begin + result.rows.length,
      feature_mask: pins.feature_mask.sha256, worker_argv: argv,
      recorded_config: {path: spec.config, bytes: capturedConfigBytes.length, sha256: sha(capturedConfigBytes)}});
    begin += result.rows.length; rounds.push(result);
  }
  const rows = rounds.flatMap(x => x.rows);
  const identity = {schema: 'rek.authentic_behavior_identity.v3', observation_schema: SCHEMA,
    selection: 'sampled', precision: 'bf16', hidden_size: 256, num_layers: 2,
    artifacts: pins, export_config: {path: configPath, sha256: sha(configBytes)},
    exporter_sha256: sha(fs.readFileSync(__filename)), base_exporter_sha256: sha(fs.readFileSync(require.resolve('./authentic_trajectory_data.cjs'))),
    reward: 'terminal_outcome + gamma_t*Phi(next_observed_points) - Phi(current_observed_points); terminal_Phi=0; Phi=d/(5+abs(d))',
    gamma_per_20ms: c.gamma_per_20ms, lambda_per_20ms: c.lambda_per_20ms,
    discounts: 'pow(reference,actual_source_to_next_source_QPC_seconds/0.02),float32',
    rounds: rounds.map(x => x.report)};
  const identityBytes = Buffer.from(JSON.stringify(identity, null, 2) + '\n');
  const binary = pack(rows, rounds.length, identityBytes, mask, pins);
  const manifest = {schema: 'rek.authentic_trajectory_dataset.v3', observation_schema: SCHEMA,
    binary: 'authentic-trajectories-v3.bin', binary_sha256: sha(binary), identity_sha256: sha(identityBytes),
    feature_mask_sha256: pins.feature_mask.sha256, checkpoint_sha256: pins.checkpoint.sha256,
    header_bytes: HEADER, row_bytes: base.ROW, rows: rows.length, rounds: rounds.length,
    seeds: rounds.map(x => x.report.seed), applied: rows.filter(x => x.applied).length,
    terminal_race_rejected: rows.filter(x => !x.applied).length,
    reward_rule: identity.reward, discount_rule: identity.discounts,
    behavior_replay: 'required; exact recorded sampled actions; fresh native policy with actual seed per round',
    limits: ['Five selected development episodes are not an independent holdout.',
      'Acknowledgements establish local acceptance, not server execution or inferred attack credit.',
      'Full source decision order and terminal-race rejected sampled actions are preserved.',
      'No inferred fall labels or new reward bonuses are introduced.']};
  fs.mkdirSync(output);
  for (const [name, bytes] of [[manifest.binary, binary], ['behavior-identity.json', identityBytes],
    ['feature-mask.bin', mask], ['manifest.json', JSON.stringify(manifest, null, 2) + '\n'],
    ['transition-ledger.jsonl', rounds.flatMap(x => x.ledger).map(x => JSON.stringify(x)).join('\n') + '\n']])
    fs.writeFileSync(path.join(output, name), bytes, {flag: 'wx'});
  return manifest;
}
if (require.main === module) {
  check(process.argv.length === 4, 'Usage: node authentic_trajectory_v3.cjs CONFIG NEW_OUTPUT');
  exportV3(process.argv[2], process.argv[3]).then(x => console.log(JSON.stringify(x)))
    .catch(e => { console.error(e.stack); process.exitCode = 1; });
}
module.exports = {HEADER, SCHEMA, sha, pinned, pack, exportV3};
