#!/usr/bin/env node
'use strict';

// Fixed experiment export. No directory traversal, binary copying, or arbitrary
// source strings: JSON uses exact schemas; mixed logs and INI use named selectors.
// Usage: node export-results.cjs INPUT_RESULTS_DIR NEW_OUTPUT_DIR
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const OLD = 'f87dae69a777e4ac28782bdee89b30d7434208d773be75bf97f56fba4a52b07e';
const NEW = '5989fa23e6ead72a20fa94a55ce4fb5da2db8631d4f2d2fb038a57c02f95ae85';
const DOMAIN = { measurement_domain: 'reconstructed_ai_simulator', authentic_parity: false };
const tagged = value => ({ ...DOMAIN, ...value });
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const num = value => typeof value === 'number' && Number.isFinite(value);
const int = value => Number.isSafeInteger(value);
const bool = value => typeof value === 'boolean';
const fixed = (...values) => value => values.includes(value);
const sha = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const maybeSha = value => value === '' || sha(value);
const nullableNum = value => value === null || num(value);
const array = (length, test) => value => Array.isArray(value) && value.length === length && value.every(test);
const fields = (names, test) => Object.fromEntries(names.split(' ').map(key => [key, test]));
const modes = {
  opponent_controller: fixed('recovered_bot1_v1'), observation_mode: fixed('rendered_pose_v1'),
  scoring_mode: fixed('recovered_hit_rules_v2'), geometry_mode: fixed('bounding_spheres', 'primitive_samples_v1'),
  contact_substeps: fixed(0, 1, 4, 8),
};
const matchSchema = {
  backend: fixed('semantic_cuda'), policy_sha256: sha, opponent: fixed('scripted'),
  opponent_sha256: maybeSha, fixture: fixed('heldout'),
  ...fields('seed arena episode duration_ticks busy_ticks facing_ticks gap_below_1_05m_ticks attack_commands', int),
  policy_side: fixed(0, 1), winner: fixed(-1, 0, 1), round_result: int,
  score: array(2, int), point_delta_sum: array(2, int), zero_hits: bool,
  ...fields('duration_seconds path_length_m minimum_gap_m maximum_gap_m mean_gap_m shaping_weight', num),
  first_hit_seconds: nullableNum, initial_xy: array(4, num), final_xy: array(4, num), ...modes,
};
const eventSchemas = {
  compact_mode_identity: {
    event: fixed('compact_mode_identity'), opponent_controller: modes.opponent_controller,
    observation_mode: modes.observation_mode, geometry_mode: modes.geometry_mode,
    contact_substeps: modes.contact_substeps, source: fixed('runtime_config'),
  },
  scoring_identity: { event: fixed('scoring_identity'), scoring_mode: modes.scoring_mode, source: fixed('runtime_config') },
  side_result: {
    event: fixed('side_result'), policy_side: fixed(0, 1),
    ...fields('arenas rounds_per_arena wins losses draws points opponent_points falls opponent_falls evaluated_ticks', int),
    execution_wall_seconds: num,
  },
  behavior_result: {
    event: fixed('behavior_result'), policy_side: fixed(0, 1), opponent: fixed('scripted'), fixture: fixed('heldout'),
    ...fields('zero_hit_games hit_games', int), mean_first_hit_seconds_when_hit: nullableNum,
    ...fields('configured_round_seconds busy_fraction facing_fraction mean_path_length_m mean_minimum_gap_m shaping_weight', num),
  },
  frozen_policy_evaluation: {
    event: fixed('frozen_policy_evaluation'), backend: fixed('semantic_cuda'), checkpoint_sha256: sha,
    opponent_sha256: maybeSha, precision: fixed('bf16'), observation_encoding: fixed('scaled_polar_xy'),
    selection: fixed('sampled'), ...fields('policy_rng_seed opponent_rng_seed wins losses draws failure_bits', int),
    ...fields('win_rate execution_wall_seconds round_seconds shaping_weight', num), fixture: fixed('heldout'),
    opponent: fixed('scripted'), ...fields('environment_randomizes_seed both_sides terminal_recurrent_reset', fixed(true)),
    ...fields('python_runtime cpu_physics physics_parity', fixed(false)), training_sps: fixed(null), ...modes,
  },
};
function exact(value, schema, label) {
  assert(value && typeof value === 'object' && !Array.isArray(value), `${label}: expected object`);
  assert.deepEqual(Object.keys(value).sort(), Object.keys(schema).sort(), `${label}: unknown or missing fields`);
  const result = {};
  for (const [key, test] of Object.entries(schema)) {
    assert(test(value[key]), `${label}.${key}: invalid value`);
    result[key] = value[key];
  }
  return result;
}
const lineJSON = text => text.trim().split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
function aggregate(rows) {
  const result = { games: rows.length, wins: 0, losses: 0, draws: 0, points: 0, opponent_points: 0 };
  for (const row of rows) {
    result[row.winner < 0 ? 'draws' : row.winner === row.policy_side ? 'wins' : 'losses']++;
    result.points += row.score[row.policy_side];
    result.opponent_points += row.score[1 - row.policy_side];
  }
  return result;
}
const configFields = {
  base: 'gpu_offset checkpoint_interval eval_episodes eval_agents eval_deterministic cudagraphs seed reset_every_horizon async',
  vec: 'total_agents num_buffers num_threads num_policies hist_policy_hidden_size hist_policy_num_layers hist_policy_percent',
  selfplay: 'enabled max_size seed opp_timeout_steps eval_pool_size eval_games',
  env: 'dr num_agents num_bots seed locomotion_segment_ticks round_seconds opponent_hidden_size opponent_num_layers opponent_precision opponent_legacy_fast_hidden opponent_deterministic',
  policy: 'hidden_size num_layers',
  train: 'gpus total_timesteps learning_rate anneal_lr min_lr_ratio gamma gae_lambda replay_ratio clip_coef vf_coef vf_clip_coef max_grad_norm ent_coef anneal_ent_coef min_ent_coef_ratio momentum minibatch_size horizon vtrace vtrace_rho_clip vtrace_c_clip verb_eps verb_eps_anneal_start verb_eps_anneal_end',
  sweep: 'downsample',
};
const metricNames = 'SPS agent_steps uptime epoch env/n importance util/gpu_percent util/vram_used_gb util/vram_total_gb util/cpu_mem_gb perf/rollout perf/eval_model perf/eval_env perf/eval_copy perf/train_misc perf/train_model perf/train'.split(' ');
function numericINI(value, label) {
  assert(typeof value === 'string' && /^[+-]?(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value), `${label}: invalid number`);
  const result = Number(value.replaceAll('_', ''));
  assert(num(result), `${label}: non-finite number`);
  return result;
}
function parseINI(text) {
  const sections = Object.create(null);
  let section;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || line.startsWith(';')) continue;
    const header = /^\[([^\]]+)\]$/.exec(line);
    if (header) {
      assert(!Object.hasOwn(sections, header[1]), 'duplicate INI section');
      section = sections[header[1]] = Object.create(null);
    } else {
      const entry = /^([^=]+?)\s*=\s*(.*)$/.exec(line);
      assert(section && entry, 'invalid INI entry');
      const key = entry[1].trim();
      assert(!Object.hasOwn(section, key), 'duplicate INI key');
      section[key] = entry[2].trim();
    }
  }
  return sections;
}

function run(inputArg, outputArg) {
  const input = path.resolve(inputArg), output = path.resolve(outputArg);
  assert(fs.statSync(input).isDirectory(), 'input must be a directory');
  assert(!fs.existsSync(output), 'output must be a NEW directory');
  const canonicalInput = fs.realpathSync(input);
  const canonicalOutput = path.join(fs.realpathSync(path.dirname(output)), path.basename(output));
  const relative = path.relative(canonicalInput, canonicalOutput);
  assert(relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative), 'output must be outside input');
  const sources = new Map(), outputs = new Map();
  const read = name => {
    const bytes = fs.readFileSync(path.join(input, name));
    const digest = hash(bytes);
    assert(!sources.has(name) || sources.get(name).sha256 === digest, 'source changed during export');
    sources.set(name, { relative_path: name, bytes: bytes.length, sha256: digest });
    return bytes.toString('utf8');
  };
  const putJSON = (name, value) => outputs.set(name, JSON.stringify(tagged(value), null, 2) + '\n');
  const putLines = (name, rows) => outputs.set(name, rows.map(row => JSON.stringify(tagged(row))).join('\n') + '\n');
  const events = file => lineJSON(read(file)).map((row, i) => {
    assert(Object.hasOwn(eventSchemas, row.event), `${file}:${i}: unknown event`);
    return exact(row, eventSchemas[row.event], `${file}:${i}`);
  });
  const records = [], evalRows = new Map();
  const cases = [
    ['heldout-before', 'eval-heldout512-old', OLD, 512, 510, 2, 0, 67649, 18357, 'primitive_samples_v1', 8, 200019],
    ['heldout-after', 'eval-heldout512-new', NEW, 512, 512, 0, 0, 75189, 18329, 'primitive_samples_v1', 8, 200019],
    ['geometry-spheres', 'eval-baseline-r3', OLD, 16, 16, 0, 0, 2924, 764, 'bounding_spheres', 0, 9171058],
    ['geometry-primitive-1', 'eval-primitive-1-r3', OLD, 16, 16, 0, 0, 1717, 475, 'primitive_samples_v1', 1, 9171058],
    ['geometry-primitive-4', 'eval-primitive-4-r3', OLD, 16, 16, 0, 0, 1861, 504, 'primitive_samples_v1', 4, 9171058],
    ['geometry-primitive-8', 'eval-primitive-8-r3', OLD, 16, 16, 0, 0, 1880, 513, 'primitive_samples_v1', 8, 9171058],
  ];
  for (const [name, folder, checkpoint, games, wins, losses, draws, points, opponentPoints, geometry, substeps, seed] of cases) {
    assert.equal(read(`${folder}/exit-code.txt`).trim(), '0', `${name}: failed process`);
    const rows = lineJSON(read(`${folder}/matches.private.jsonl`)).map((row, i) => exact(row, matchSchema, `${name}:${i}`));
    const summaryEvents = events(`${folder}/summary.jsonl`);
    assert.equal(summaryEvents.length, 7);
    const identity = summaryEvents.filter(row => row.event === 'compact_mode_identity');
    const scoring = summaryEvents.filter(row => row.event === 'scoring_identity');
    const finals = summaryEvents.filter(row => row.event === 'frozen_policy_evaluation');
    assert.equal(identity.length, 1); assert.equal(scoring.length, 1); assert.equal(finals.length, 1);
    const final = finals[0], total = aggregate(rows), keys = new Set();
    assert.deepEqual(total, { games, wins, losses, draws, points, opponent_points: opponentPoints });
    for (const row of rows) {
      assert.equal(row.policy_sha256, checkpoint); assert.equal(row.seed, seed);
      assert.equal(row.geometry_mode, geometry); assert.equal(row.contact_substeps, substeps);
      assert.equal(row.duration_ticks, 6000); assert.equal(row.duration_seconds, 120);
      assert.deepEqual(row.score, row.point_delta_sum);
      assert(row.score.every(value => value >= 0));
      for (const name of ['busy_ticks', 'facing_ticks', 'gap_below_1_05m_ticks']) assert(row[name] >= 0 && row[name] <= row.duration_ticks);
      assert(row.attack_commands >= 0 && row.path_length_m >= 0);
      assert(row.minimum_gap_m >= 0 && row.maximum_gap_m >= row.minimum_gap_m);
      assert(row.mean_gap_m >= row.minimum_gap_m - 1e-6 && row.mean_gap_m <= row.maximum_gap_m + 1e-6);
      assert(row.first_hit_seconds === null || (row.first_hit_seconds >= 0 && row.first_hit_seconds <= row.duration_seconds));
      assert.equal(row.zero_hits, row.first_hit_seconds === null);
      const key = `${row.policy_side}:${row.arena}:${row.episode}`;
      assert(!keys.has(key), `${name}: duplicate match`); keys.add(key);
    }
    assert.equal(final.checkpoint_sha256, checkpoint); assert.equal(final.failure_bits, 0);
    assert.equal(final.policy_rng_seed, seed); assert.equal(final.round_seconds, 120);
    for (const row of [identity[0], final]) {
      assert.equal(row.geometry_mode, geometry); assert.equal(row.contact_substeps, substeps);
    }
    for (const key of ['wins', 'losses', 'draws']) assert.equal(final[key], total[key]);
    assert.equal(final.win_rate, wins / games);
    for (const side of [0, 1]) {
      const sideEvents = summaryEvents.filter(row => row.event === 'side_result' && row.policy_side === side);
      const behavior = summaryEvents.filter(row => row.event === 'behavior_result' && row.policy_side === side);
      assert.equal(sideEvents.length, 1); assert.equal(behavior.length, 1);
      const sideRows = rows.filter(row => row.policy_side === side), actual = aggregate(sideRows);
      for (const key of ['wins', 'losses', 'draws', 'points', 'opponent_points']) assert.equal(sideEvents[0][key], actual[key]);
      assert.equal(actual.games, sideEvents[0].arenas * sideEvents[0].rounds_per_arena);
      for (const row of sideRows) {
        assert(row.arena >= 0 && row.arena < sideEvents[0].arenas);
        assert(row.episode >= 0 && row.episode < sideEvents[0].rounds_per_arena);
      }
      assert.equal(behavior[0].hit_games + behavior[0].zero_hit_games, actual.games);
      const hitRows = sideRows.filter(row => !row.zero_hits);
      assert.equal(behavior[0].hit_games, hitRows.length);
      assert.equal(behavior[0].zero_hit_games, sideRows.length - hitRows.length);
      const mean = (selected, field) => selected.reduce((total, row) => total + row[field], 0) / selected.length;
      const close = (a, b, tolerance) => assert(Math.abs(a - b) <= tolerance, `${name}: behavior aggregate mismatch`);
      if (hitRows.length) close(behavior[0].mean_first_hit_seconds_when_hit, mean(hitRows, 'first_hit_seconds'), 1e-6);
      else assert.equal(behavior[0].mean_first_hit_seconds_when_hit, null);
      close(behavior[0].mean_path_length_m, mean(sideRows, 'path_length_m'), 1e-7);
      close(behavior[0].mean_minimum_gap_m, mean(sideRows, 'minimum_gap_m'), 1e-7);
      close(behavior[0].busy_fraction, mean(sideRows, 'busy_ticks') / 6000, 1e-8);
      close(behavior[0].facing_fraction, mean(sideRows, 'facing_ticks') / 6000, 1e-8);
    }
    putLines(`${name}.matches.jsonl`, rows);
    putLines(`${name}.events.jsonl`, summaryEvents);
    records.push(tagged({ name, ...total, checkpoint_sha256: checkpoint, geometry_mode: geometry, contact_substeps: substeps, seed, execution_wall_seconds: final.execution_wall_seconds }));
    evalRows.set(name, rows);
  }
  function compareInitialCoordinates(leftName, rightName) {
    const key = row => `${row.policy_side}:${row.arena}:${row.episode}`;
    const right = new Map(evalRows.get(rightName).map(row => [key(row), row]));
    let different = 0, maximum = 0;
    for (const row of evalRows.get(leftName)) {
      assert(right.has(key(row)), 'paired match key missing');
      const delta = Math.max(...row.initial_xy.map((value, i) => Math.abs(value - right.get(key(row)).initial_xy[i])));
      different += Number(delta !== 0);
      maximum = Math.max(maximum, delta);
    }
    return { left: leftName, right: rightName, matched_keys: right.size, recorded_initial_xy_different_rows: different, maximum_absolute_coordinate_difference_m: maximum };
  }
  const initialComparisons = [compareInitialCoordinates('heldout-before', 'heldout-after')];
  assert.equal(initialComparisons[0].recorded_initial_xy_different_rows, 6);
  assert(initialComparisons[0].maximum_absolute_coordinate_difference_m < 0.001601);
  for (const suffix of ['1', '4', '8']) {
    const comparison = compareInitialCoordinates('geometry-spheres', `geometry-primitive-${suffix}`);
    assert.equal(comparison.recorded_initial_xy_different_rows, 0);
    initialComparisons.push(comparison);
  }

  const train = 'train-primitive8-r1';
  const ini = parseINI(read(`${train}/logs/rek_native5/train-primitive8-r1.ini`));
  const config = {};
  for (const [section, names] of Object.entries(configFields)) {
    assert(ini[section], `missing INI section: ${section}`);
    config[section] = {};
    for (const name of names.split(' ')) config[section][name] = numericINI(ini[section][name], `${section}.${name}`);
  }
  assert.equal(config.vec.total_agents, 512); assert.equal(config.train.horizon, 128);
  assert.equal(config.train.total_timesteps, 67108864);
  assert.deepEqual(Object.keys(ini.metrics).sort(), [...metricNames].sort(), 'unknown or missing metric series');
  const series = Object.fromEntries(metricNames.map(name => [name, ini.metrics[name].split(',').map(value => numericINI(value.trim(), name))]));
  for (const values of Object.values(series)) assert.equal(values.length, 64);
  assert.equal(series.agent_steps.at(-1), 67108864); assert.equal(series.uptime.at(-1), 54.43491005897522);
  const metricRows = Array.from({ length: 64 }, (_, i) => ({ sample_index: i, ...Object.fromEntries(metricNames.map(name => [name, series[name][i]])) }));
  putJSON('training-config.json', { config, runtime_identity: { ...modesAsValues(), geometry_mode: 'primitive_samples_v1', contact_substeps: 8 }, log_series_semantics: 'native_INI_aggregated_samples_not_raw_minibatch_events' });
  putLines('training-metrics.jsonl', metricRows);
  const checkpointRows = read(`${train}/checkpoint-hashes.txt`).trim().split(/\r?\n/).map(line => {
    const match = /^([a-f0-9]{64})\s+.*[\\/](\d{16})\.bin$/.exec(line);
    assert(match, 'invalid checkpoint identity record');
    return { continuation_agent_steps: Number(match[2]), sha256: match[1] };
  }).sort((a, b) => a.continuation_agent_steps - b.continuation_agent_steps);
  assert.equal(checkpointRows.length, 17);
  checkpointRows.forEach((row, i) => assert.equal(row.continuation_agent_steps, i * 4194304));
  assert.equal(checkpointRows[0].sha256, OLD); assert.equal(checkpointRows.at(-1).sha256, NEW);
  for (const source of ['initial-checkpoint.sha256', 'verified-warm-start.txt']) {
    const hashes = read(`${train}/${source}`).trim().split(/\r?\n/).map(line => /^([a-f0-9]{64})\s+/.exec(line)?.[1]);
    assert.equal(hashes.length, source === 'initial-checkpoint.sha256' ? 1 : 2);
    for (const digest of hashes) assert.equal(digest, OLD);
  }
  assert.equal(read(`${train}/exit-code.txt`).trim(), '0');
  const roundSchema = fields('arenas completed_rounds fighter0_wins fighter1_wins ties redos unclassified fighter0_completed_points fighter1_completed_points failure_bits', int);
  const rounds = exact(JSON.parse(read(`${train}/round-summary.json`)), roundSchema, 'training rounds');
  assert.deepEqual(rounds, { arenas: 512, completed_rounds: 10752, fighter0_wins: 10708, fighter1_wins: 42, ties: 2, redos: 0, unclassified: 0, fighter0_completed_points: 1495617, fighter1_completed_points: 376936, failure_bits: 0 });
  const timing = read(`${train}/process-timing.txt`);
  const elapsed = /^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\d+):(\d+(?:\.\d+)?)\s*$/m.exec(timing);
  assert(elapsed, 'missing process wall time');
  const processSeconds = Number(elapsed[1]) * 60 + Number(elapsed[2]);
  assert.equal(processSeconds, 55.16);
  const stderr = read(`${train}/stderr.txt`);
  const scoringLines = stderr.split(/\r?\n/).filter(line => line.startsWith('semantic_cuda_scoring='));
  assert.equal(scoringLines.length, 1);
  const trainingScoring = JSON.parse(scoringLines[0].slice('semantic_cuda_scoring='.length));
  assert.equal(trainingScoring.mode, 'recovered_hit_rules_v2'); assert.equal(trainingScoring.geometry, 'primitive_samples_v1');
  assert.equal(trainingScoring.contact_substeps, 8); assert.equal(trainingScoring.authentic_parity, false);
  const parameterLines = stderr.split(/\r?\n/).filter(line => line.startsWith('semantic_cuda_parameters='));
  assert.equal(parameterLines.length, 1);
  const trainingParameters = JSON.parse(parameterLines[0].slice('semantic_cuda_parameters='.length));
  for (const [name, value] of Object.entries(modesAsValues())) assert.equal(trainingParameters[name], value);
  assert.equal(trainingParameters.physics_parity, false);
  const trainingSummary = {
    initial_checkpoint_sha256: OLD, final_checkpoint_sha256: NEW, transitions: 67108864,
    training_uptime_seconds: series.uptime.at(-1), mean_transitions_per_training_second: 67108864 / series.uptime.at(-1),
    process_wall_seconds: processSeconds, metric_samples: metricRows.length,
    final_logged_SPS: series.SPS.at(-1), final_logged_epoch: series.epoch.at(-1),
    metric_samples_are_aggregates: true, final_sample_duplicate_retained: metricNames.every(name => series[name].at(-1) === series[name].at(-2)),
    exit_code: 0, geometry_mode: 'primitive_samples_v1', contact_substeps: 8,
  };
  putJSON('training-summary.json', trainingSummary);
  putJSON('training-rounds.json', rounds);
  putLines('checkpoint-identities.jsonl', checkpointRows);

  const gpu = exact(JSON.parse(read('gpu-contact-test.json')), {
    event: fixed('primitive_contacts_cuda'), passed: fixed(true),
    ...fields('n checks cases_per_pair_type host_gpu_mismatches symmetric_violations transform_violations excluded_boundary_cases temporal_samples', int),
    pair_types: value => JSON.stringify(value) === JSON.stringify(['sphere_sphere', 'sphere_capsule', 'sphere_box', 'capsule_capsule', 'capsule_box', 'box_box']),
    positive_overlap_per_type: array(6, int), ...fields('boundary_filter_m production_contact_margin_m', num),
    ...fields('python_runtime physics_stepping dynamic_parity_claim', fixed(false)),
  }, 'GPU test');
  for (const field of ['host_gpu_mismatches', 'symmetric_violations', 'transform_violations']) assert.equal(gpu[field], 0);
  assert.equal(gpu.n, 12288); assert.equal(gpu.checks, 73728); assert.equal(gpu.cases_per_pair_type, 2048);
  assert(gpu.positive_overlap_per_type.every(value => value > 0 && value < gpu.cases_per_pair_type));
  assert.equal(gpu.production_contact_margin_m, 0); assert.equal(gpu.temporal_samples, 4);
  const selectJSON = (file, predicate) => {
    const rows = read(file).split(/\r?\n/).filter(line => line.trim().startsWith('{')).map(line => JSON.parse(line)).filter(predicate);
    assert.equal(rows.length, 1, `${file}: missing/duplicate selected record`);
    return rows[0];
  };
  const host = exact(selectJSON('checks/stdout.txt', row => row.event === 'primitive_contacts_tests'), {
    event: fixed('primitive_contacts_tests'), checks: int, passed: fixed(true), contact_margin_m: num,
    static_geometry_only: fixed(true), dynamic_parity_claim: fixed(false),
  }, 'host test');
  const assets = exact(selectJSON('assets-probe/result.jsonl', row => row.test === 'fast_asset_offline_fk'), {
    test: fixed('fast_asset_offline_fk'), passed: fixed(true), ...fields('checked_finite_values forbidden_cpu_physics_calls', int),
    ...fields('max_root_quaternion_norm_error min_striker_proxy_radius_m max_striker_proxy_radius_m', num), python_runtime: fixed(false),
  }, 'asset finite-value test');
  assert.equal(host.checks, 78184); assert.equal(host.contact_margin_m, 0);
  assert.equal(assets.checked_finite_values, 125528); assert.equal(assets.forbidden_cpu_physics_calls, 0);
  assert(assets.max_root_quaternion_norm_error >= 0 && assets.max_root_quaternion_norm_error < 1e-6);
  assert(assets.min_striker_proxy_radius_m > 0 && assets.max_striker_proxy_radius_m >= assets.min_striker_proxy_radius_m);
  const memcheck = read('memcheck-r3.txt').trim();
  assert(/^========= COMPUTE-SANITIZER\r?\n========= ERROR SUMMARY: 0 errors$/.test(memcheck), 'memcheck did not pass cleanly');
  const memEvents = events('memcheck-r3.stdout.txt');
  const memFinal = memEvents.filter(row => row.event === 'frozen_policy_evaluation');
  assert.equal(memFinal.length, 1); assert.equal(memFinal[0].checkpoint_sha256, OLD);
  assert.equal(memFinal[0].failure_bits, 0); assert.equal(memFinal[0].geometry_mode, 'primitive_samples_v1');
  assert.equal(memFinal[0].contact_substeps, 8); assert.equal(memFinal[0].wins + memFinal[0].losses + memFinal[0].draws, 8);
  const modePass = read('mode-config-test.stdout.txt').split(/\r?\n/).filter(line => line === 'PASS explicit mode configuration, geometry defaults and boundaries, ambient override isolation, complete JSON identity, 14 invalid-mode/count rejections');
  assert.equal(modePass.length, 1);
  putJSON('validation-tests.json', {
    host_contact: tagged(host), gpu_contact: tagged(gpu), offline_asset_finite_values: tagged(assets),
    mode_configuration: tagged({ passed: true, invalid_mode_count_rejections: 14 }),
    memcheck: tagged({ tool: 'compute_sanitizer', passed: true, reported_errors: 0, evaluation: tagged(memFinal[0]) }),
  });
  putJSON('summary.json', { evaluations: records, training: tagged(trainingSummary), matched_seeds_and_match_keys_verified: true, recorded_initial_coordinate_comparisons: initialComparisons, geometry_comparison_is_action_replay: false });
  outputs.set('README.md', `# Numeric experiment export\n\nAll results describe the reconstructed AI simulator. Authentic REK parity is false. No authentic match with the new checkpoint is included.\n\n- heldout-before/after.matches.jsonl: all 512 match records per checkpoint, paired initial fixtures, 120-second rounds.\n- geometry-*.matches.jsonl: all 16 matches per geometry setting, same initial fixtures. Policy actions and subsequent states may diverge; these are not controlled action replays or contact false-positive rates.\n- *.events.jsonl: all seven native summary/identity events for each evaluation.\n- training-config.json: named numeric fields selected from the actual logged INI, plus verified scoring/geometry identity.\n- training-metrics.jsonl: all 64 native logged samples, retaining all 17 metric series and duplicate final sample. These are logged aggregates, not raw per-minibatch events. Fractional averaged agent_steps values are retained. Logged final SPS differs from whole-run transitions divided by uptime. Reported VRAM utilization describes the observed device, not isolated allocation by this task.\n- training-summary.json, training-rounds.json, checkpoint-identities.jsonl: timing, completed-round accounting and all 17 numeric checkpoint steps with SHA-256 identity. No checkpoint binaries are included.\n- validation-tests.json: host/CUDA geometry tests, finite-value asset test, mode configuration test and passing Compute Sanitizer result. Static geometry tests do not establish dynamics parity.\n- summary.json: recomputed match totals cross-checked against native summary events.\n- manifest.json: source/output byte counts and SHA-256 digests; source paths are fixed relative labels only.\n\nThe legacy opponent field scripted is a native bucket label. opponent_controller=recovered_bot1_v1 identifies the actual reconstructed controller. All output JSON records explicitly set authentic_parity=false.\n\n## Export boundary\n\nRun node ../export-results.cjs INPUT_RESULTS_DIR NEW_OUTPUT_DIR from this folder. The destination must not exist. Source JSON records require exact schemas, including fixed enum strings and numeric-array dimensions. Unknown or missing fields fail before output creation. The INI exports only fields named in configFields and metricNames in the script; all paths, unselected strings, and sweep definitions are excluded. Mixed test logs select named JSON events. Training stderr contributes only four validated scoring/geometry identity fields. Timing selects only elapsed time. Checksum lists contribute only 64-digit lowercase hashes and numeric checkpoint steps. No raw logs, commands, credentials, human capture data, game assets or model binaries are copied.\n\nSource/output hashes support byte-level reproduction of this export; they do not independently authenticate the original experiment or establish simulator-to-REK transfer.\n`);
  outputs.set('README.md', outputs.get('README.md') + '\nRecorded initial-coordinate qualification: the held-out before/after runs share seeds and all 512 match keys, but six initial_xy records differ, with maximum absolute coordinate difference 0.001600027 m. The exporter does not claim identical captured starting coordinates for these runs. All four geometry variants have identical initial_xy records. The reason for the six differences is not established by the exported data.\n');
  putJSON('manifest.json', {
    exporter_sha256: hash(fs.readFileSync(__filename)), schema_version: 1,
    sources: [...sources.values()].sort((a, b) => a.relative_path.localeCompare(b.relative_path)),
    outputs: [...outputs].map(([name, value]) => ({ relative_path: name, bytes: Buffer.byteLength(value), sha256: hash(value) })),
    match_whitelist: Object.keys(matchSchema), summary_event_whitelists: Object.fromEntries(Object.entries(eventSchemas).map(([name, schema]) => [name, Object.keys(schema)])),
    numeric_INI_whitelist: configFields, metric_whitelist: metricNames,
  });
  for (const [name, content] of outputs) {
    assert(!/(?:[A-Za-z]:[\\/]|\/home\/|\\\\[0-9])/i.test(content), `${name}: private absolute path`);
  }
  // Every parse, schema, identity, count and private-path check precedes writes.
  fs.mkdirSync(output, { recursive: false });
  for (const [name, content] of outputs) fs.writeFileSync(path.join(output, name), content, { flag: 'wx' });
  console.log(JSON.stringify(tagged({ files: outputs.size, heldout_matches: 1024, geometry_matches: 64, training_metric_samples: 64, verified: true })));
}
function modesAsValues() {
  return { opponent_controller: 'recovered_bot1_v1', observation_mode: 'rendered_pose_v1', scoring_mode: 'recovered_hit_rules_v2' };
}
if (require.main === module) {
  try {
    assert.equal(process.argv.length, 4, 'usage: node export-results.cjs INPUT_RESULTS_DIR NEW_OUTPUT_DIR');
    run(process.argv[2], process.argv[3]);
  } catch (error) {
    // Parser/AssertionError messages may contain private source excerpts.
    console.error('Export rejected: source validation or destination check failed; private source details withheld.');
    process.exitCode = 1;
  }
}
module.exports = { exact, matchSchema, parseINI, numericINI, run };
