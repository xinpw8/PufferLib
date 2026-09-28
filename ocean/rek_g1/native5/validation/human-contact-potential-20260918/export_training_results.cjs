'use strict';
// Export simulator measurements only. Checkpoints, assets, host paths, and raw
// captures stay private. Numeric trainer metrics retain repeated terminal bins.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const [stage, output] = process.argv.slice(2);
assert(stage && output && process.argv.length === 4, 'Usage: node export_training_results.cjs PRIVATE_STAGE NEW_PUBLIC_OUTPUT');
assert(!fs.existsSync(output), 'Output must be new');
const read = name => fs.readFileSync(path.join(stage, name), 'utf8');
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const hashFile = name => digest(fs.readFileSync(path.join(stage, name)));
const json = name => JSON.parse(read(name));
const rows = name => read(name).trim().split(/\r?\n/).map(line => JSON.parse(line));
const finalName = '0000000033554432.bin';
const matchKeys = new Set(('backend policy_sha256 opponent opponent_sha256 fixture seed arena episode policy_side score point_delta_sum winner round_result duration_ticks duration_seconds first_hit_seconds zero_hits busy_ticks facing_ticks gap_below_1_05m_ticks attack_commands path_length_m minimum_gap_m maximum_gap_m mean_gap_m initial_xy final_xy shaping_weight opponent_controller observation_mode scoring_mode geometry_mode contact_substeps').split(' '));
const files = new Map();
function safe(name, contents) {
  assert(!/(?:\/home\/|[A-Z]:\\|\\\\|password|bearer|authorization|access_token)/i.test(contents), `Private content in ${name}`);
  assert(!files.has(name), `Duplicate file ${name}`);
  files.set(name, contents);
}
const saveJson = (name, value) => safe(name, JSON.stringify(value, null, 2) + '\n');
function normalizedCommand(text) {
  return text.trim().split(/\s+/).map(token => {
    if (!token.includes('/home/')) return token;
    const equal = token.indexOf('=');
    const key = equal < 0 ? '' : token.slice(0, equal + 1);
    const value = equal < 0 ? token : token.slice(equal + 1);
    let replacement;
    if (key === 'REK_FAST_CONTACT_POTENTIAL=') replacement = '$CONTACT_MODEL';
    else if (key === '--env.model_path=') replacement = '$MODEL_XML';
    else if (key === '--env.physics_export_path=') replacement = '$PHYSICS_EXPORT';
    else if (key === '--env.assets_path=') replacement = '$ASSETS';
    else if (key === '--env.motion_features_path=') replacement = '$MOTION_FEATURES';
    else if (key === '--base.load_model_path=') replacement = '$INITIAL_CHECKPOINT';
    else if (key === '--base.log_dir=') replacement = '$OUTPUT/logs';
    else if (key === '--base.checkpoint_dir=') replacement = '$OUTPUT/checkpoints';
    else if (value.endsWith('/puffer-rek-native5')) replacement = '$TRAINER';
    else if (value.endsWith('/diverse-policy-eval')) replacement = '$EVALUATOR';
    else if (value.endsWith('/runtime-baseline.json')) replacement = '$RUNTIME_JSON';
    else if (value.endsWith('.bin')) replacement = '$CHECKPOINT';
    else if (value.endsWith('/matches.private.jsonl')) replacement = '$OUTPUT/matches.jsonl';
    assert(replacement, 'Unrecognized private command path');
    return key + replacement;
  }).join(' ') + '\n';
}
function trainerMetrics(name) {
  const text = read(`${name}/logs/rek_native5/${name}.ini`);
  let inMetrics = false;
  const values = {};
  for (const line of text.split(/\r?\n/)) {
    const header = line.match(/^\[([^\]]+)\]$/);
    if (header) { inMetrics = header[1] === 'metrics'; continue; }
    if (!inMetrics || !line.trim()) continue;
    const pair = line.match(/^([A-Za-z0-9_\/]+)\s*=\s*(.*)$/);
    assert(pair, 'Unexpected metric format');
    const samples = pair[2].split(',').map(Number);
    assert(samples.every(Number.isFinite), 'Nonfinite metric');
    values[pair[1]] = samples;
  }
  const count = values.agent_steps.length;
  assert(Object.values(values).every(v => v.length === count), 'Metric lengths differ');
  assert.equal(values.agent_steps.at(-1), 33554432);
  return {schema: 'rek.native_training_metrics.v1', source_log_sha256: digest(text),
    repeated_terminal_bins_are_not_independent_samples: true, values};
}
function evaluate(label, directory) {
  assert.equal(read(`${directory}/exit-code.txt`).trim(), '0');
  const summaries = rows(`${directory}/summary.jsonl`);
  const summary = summaries.find(x => x.event === 'frozen_policy_evaluation');
  assert.equal(summary.failure_bits, 0);
  assert.equal(summary.shaping_weight, 0);
  assert.equal(summary.policy_rng_seed, 200029);
  const matches = rows(`${directory}/matches.private.jsonl`);
  assert.equal(matches.length, 512);
  for (const row of matches) {
    assert(Object.keys(row).every(key => matchKeys.has(key)), 'Unexpected match key');
    assert.equal(row.shaping_weight, 0);
    assert.equal(row.seed, 200029);
    assert.equal(row.policy_sha256, summary.checkpoint_sha256);
  }
  const policyPoints = matches.reduce((n, m) => n + m.score[m.policy_side], 0);
  const opponentPoints = matches.reduce((n, m) => n + m.score[m.policy_side ^ 1], 0);
  const wins = matches.filter(m => m.winner === m.policy_side).length;
  assert.equal(wins, summary.wins);
  const matchHash = hashFile(`${directory}/matches.private.jsonl`);
  safe(`${label}-matches.jsonl`, matches.map(m => JSON.stringify(m)).join('\n') + '\n');
  safe(`${label}-summary.jsonl`, summaries.map(m => JSON.stringify(m)).join('\n') + '\n');
  safe(`${label}-eval-command.txt`, normalizedCommand(read(`${directory}/command.txt`)));
  return {label, checkpoint_sha256: summary.checkpoint_sha256, matches: matches.length,
    wins, losses: summary.losses, draws: summary.draws, policy_points: policyPoints,
    opponent_points: opponentPoints, net_points: policyPoints - opponentPoints,
    zero_hit_games: matches.filter(m => m.zero_hits).length,
    mean_first_hit_seconds: matches.reduce((n, m) => n + m.first_hit_seconds, 0) / matches.length,
    match_input_sha256: matchHash, failure_bits: summary.failure_bits};
}
const warm = evaluate('warm', 'eval-warm-heldout512');
const parity = evaluate('warm-candidate-parity', 'eval-warm-candidate-parity512');
assert.equal(warm.match_input_sha256, parity.match_input_sha256, 'Shaping-zero parity failed');
const experiments = [];
for (let index = 1; index <= 3; index++) {
  for (const kind of ['baseline', 'contact-potential']) {
    const name = `train-${kind}-r${index}`;
    const evalDir = index === 1 ? `eval-${kind}-heldout512` : `eval-${kind}-r${index}-heldout512`;
    const label = `${kind}-seed${230 + index}`;
    const summary = json(`${name}/training-summary.json`)[0];
    const metrics = trainerMetrics(name);
    assert.equal(summary.learnerTransitions, 33554432);
    assert.equal(summary.trainingRoundResults.failure_bits, 0);
    assert.equal(summary.trainingLoopSeconds, metrics.values.uptime.at(-1));
    assert.equal(summary.trainingSps, 33554432 / metrics.values.uptime.at(-1));
    const checkpoint = hashFile(`${name}/checkpoints/rek_native5/${name}/${finalName}`);
    const evaluation = evaluate(label, evalDir);
    assert.equal(checkpoint, evaluation.checkpoint_sha256);
    const initialHash = read(`${name}/verified-warm-start.txt`).trim().split(/\r?\n/).map(x => x.split(/\s+/)[0]);
    assert.equal(initialHash.length, 2);
    assert(initialHash.every(x => x === warm.checkpoint_sha256));
    const provenance = read(`${name}/provenance.txt`);
    const trainerLine = provenance.split(/\r?\n/).find(x => x.endsWith('/puffer-rek-native5'));
    assert(trainerLine, 'Missing trainer identity');
    const trainerHash = trainerLine.split(/\s+/)[0];
    saveJson(`${label}-training-summary.json`, summary);
    saveJson(`${label}-trainer-metrics.json`, metrics);
    safe(`${label}-train-command.txt`, normalizedCommand(read(`${name}/command.txt`)));
    experiments.push({kind, seed: 230 + index, trainer_sha256: trainerHash,
      initial_checkpoint_sha256: initialHash[0], checkpoint_sha256: checkpoint,
      transitions: summary.learnerTransitions, training_loop_seconds: summary.trainingLoopSeconds,
      training_sps: summary.trainingSps, process_wall_seconds: summary.processWallSeconds,
      startup_inclusive_sps: summary.startupInclusiveSps, evaluation});
  }
}
const pairs = [231, 232, 233].map(seed => {
  const baseline = experiments.find(x => x.seed === seed && x.kind === 'baseline');
  const shaped = experiments.find(x => x.seed === seed && x.kind === 'contact-potential');
  return {seed, policy_points_difference: shaped.evaluation.policy_points - baseline.evaluation.policy_points,
    opponent_points_difference: shaped.evaluation.opponent_points - baseline.evaluation.opponent_points,
    net_points_difference: shaped.evaluation.net_points - baseline.evaluation.net_points,
    wins_difference: shaped.evaluation.wins - baseline.evaluation.wins,
    training_sps_difference: shaped.training_sps - baseline.training_sps,
    training_sps_ratio: shaped.training_sps / baseline.training_sps};
});
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const comparison = {schema: 'rek.contact_potential_training_comparison.v1',
  protocol: {training_seeds: [231, 232, 233], frozen_eval_seed: 200029, frozen_matches_per_run: 512,
    initial_checkpoint_sha256: warm.checkpoint_sha256, transitions_per_training_run: 33554432,
    arenas: 512, horizon: 128, minibatch: 8192, gamma: .999, gae_lambda: .995,
    learning_rate: .0001, entropy: .01, shaping_weight_baseline: 0, shaping_weight_candidate: 1,
    shaping_disabled_for_all_frozen_evaluations: true, tuning_on_frozen_results: false,
    geometry_mode: 'primitive_samples_v1', contact_substeps: 8,
    scoring_mode: 'recovered_hit_rules_v2', opponent_controller: 'recovered_bot1_v1',
    observation_mode: 'rendered_pose_v1', round_seconds: 120,
    base_source_commit: '6ae42e5b636dbc7e6ad0c31b9819148b96e5bbcc',
    source_head_archive_sha256: hashFile('source-head.tar'), source_overlay_archive_sha256: hashFile('source-overlay.tar'),
    model_file_sha256: hashFile('source/ocean/rek_g1/native5/validation/human-contact-potential-20260918/contact-potential-model.json')},
  warm, zero_shaping_parity: {byte_identical_matches: true, match_sha256: warm.match_input_sha256},
  tests: {host_result: read('contact-tests/host-result.txt').trim(), cuda: json('contact-tests/cuda-result.json')},
  experiments, paired_differences: pairs,
  aggregate: {mean_policy_points_difference: mean(pairs.map(x => x.policy_points_difference)),
    mean_net_points_difference: mean(pairs.map(x => x.net_points_difference)),
    mean_wins_difference: mean(pairs.map(x => x.wins_difference)),
    mean_training_sps_ratio: mean(pairs.map(x => x.training_sps_ratio)),
    positive_net_point_pairs: pairs.filter(x => x.net_points_difference > 0).length},
  limitations: ['Three training seeds share one frozen evaluation fixture seed and one scripted opponent.',
    'One paired seed used the original binary for baseline; shaping-zero 512-match parity is byte-identical.',
    'Scored contact geometry is a reduced CUDA candidate; authentic REK physics parity is false.',
    'Positive-only score-receipt geometry is not executed-action supervision or hit probability.',
    'No policy is promoted or claimed to have improved authentic gameplay from these simulator results.']};
saveJson('comparison.json', comparison);
saveJson('export-manifest.json', {schema: 'rek.public_training_export.v1', files: [...files].map(([name, data]) => ({name, sha256: digest(data), bytes: Buffer.byteLength(data)}))});
fs.mkdirSync(output, {recursive: false});
for (const [name, data] of files) fs.writeFileSync(path.join(output, name), data, {flag: 'wx'});
process.stdout.write(JSON.stringify({output_files: files.size, paired_differences: pairs, aggregate: comparison.aggregate}, null, 2) + '\n');
