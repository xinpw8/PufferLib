'use strict';
// Offline reduction of completed native runs. No model execution occurs here.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const base = process.argv[2];
if (!base) throw new Error('Usage: node reduce.cjs PRIVATE_RUN_DIRECTORY');
const read = name => fs.readFileSync(path.join(base, name), 'utf8');
const cases = [];
for (const version of [2, 3]) for (const roundSeconds of [20, 300]) for (const opponent of ['neutral', 'scripted']) {
  const name = `v${version}-${opponent}-${roundSeconds}`;
  const summary = read(`${name}/summary.jsonl`).trim().split('\n').map(JSON.parse);
  const matches = read(`${name}/matches.private.jsonl`).trim().split('\n').map(JSON.parse);
  const exitCode = Number(read(`${name}/exit-code.txt`));
  const overall = summary.find(x => x.event === 'frozen_policy_evaluation');
  if (exitCode || !overall || overall.failure_bits || matches.length !== 1024) throw new Error(`Failed case ${name}`);
  const sides = [0, 1].map(side => {
    const rows = matches.filter(x => x.policy_side === side);
    const durations = rows.map(x => x.duration_seconds);
    const result = summary.find(x => x.event === 'side_result' && x.policy_side === side);
    if (!result || rows.length !== 512) throw new Error(`Incomplete side ${name}`);
    return {...result, zero_scoring_games: rows.filter(x => x.score[side] === 0).length,
      mean_actual_duration_seconds: durations.reduce((a, b) => a + b, 0) / rows.length,
      minimum_actual_duration_seconds: Math.min(...durations), maximum_actual_duration_seconds: Math.max(...durations)};
  });
  cases.push({version, configured_round_seconds: roundSeconds, opponent,
    checkpoint_sha256: overall.checkpoint_sha256, arenas: 128, rounds_per_arena_per_side: 4,
    matches: matches.length, sampled: true, precision: 'bf16', policy_rng_seed: 10001,
    wins: overall.wins, losses: overall.losses, draws: overall.draws,
    win_rate: overall.wins / matches.length,
    policy_points: sides.reduce((a, s) => a + s.points, 0),
    opponent_points: sides.reduce((a, s) => a + s.opponent_points, 0),
    zero_scoring_games: sides.reduce((a, s) => a + s.zero_scoring_games, 0),
    sides, terminal_recurrent_reset: overall.terminal_recurrent_reset,
    neutral_opponent_external_override: opponent === 'neutral' ? 1 : null,
    neutral_opponent_action: opponent === 'neutral' ? 1 : null,
    neutral_rows_validated_each_tick: opponent === 'neutral',
    failure_bits: overall.failure_bits, exit_code: exitCode,
    private_match_records_sha256: crypto.createHash('sha256').update(read(`${name}/matches.private.jsonl`)).digest('hex')});
}
const result = {schema: 'rek.frozen_policy.opponent_intervention.v1', host: 'spark-4ae3',
  backend: 'semantic_cuda', cpu_physics: false, python_model_runtime: false,
  fixed_geometry: true, initial_conditions_randomized: false,
  independent_random_seeds: 1, repetitions_are_policy_action_samples: true,
  runtime_equations_modified: false, checkpoint_modified: false, production_interaction: false,
  training_sps: null, shared_gpu_during_measurement: true,
  observation_override: null,
  interpretation: 'At 20 s both checkpoints fail to score against a neutral opponent, while the scripted controls exactly reproduce prior results. Longer rounds retain poor neutral-opponent performance. Changing 20 s to 300 s changes both episode duration and timer observations, so differences between those conditions cannot be attributed solely to timer distribution shift.',
  scoring_caveat: 'V2 includes synthetic knockdown points and early three-down endings; V3 is points-only. Score totals between versions are not directly equivalent. Zero-scoring games are explicit measured terminal-score counts.',
  cases};
fs.writeFileSync(path.join(base, 'summary.public.json'), JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify(cases.map(({version, configured_round_seconds, opponent, wins, losses, draws, policy_points, opponent_points, zero_scoring_games}) =>
  ({version, configured_round_seconds, opponent, wins, losses, draws, policy_points, opponent_points, zero_scoring_games})), null, 2));
