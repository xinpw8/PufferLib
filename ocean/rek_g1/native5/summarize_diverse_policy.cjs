'use strict';
// Offline public reduction. All inference, state evolution, and per-tick
// measurements were performed in CUDA by diverse_policy_eval.cu.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = process.argv[2];
if (!root) throw new Error('Expected completed suite directory');
const loadLines = file => fs.readFileSync(file, 'utf8').trim().split('\n').map(JSON.parse);
const cases = [];
for (const ent of fs.readdirSync(root, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
  if (!ent.isDirectory() || !/^(fixed|heldout)-(neutral|scripted|retreat|strafe|checkpoint)-(20|300)$/.test(ent.name)) continue;
  const dir = path.join(root, ent.name);
  if (Number(fs.readFileSync(path.join(dir, 'exit-code.txt'), 'utf8')) !== 0) throw new Error('Failed evaluation: ' + ent.name);
  const recordsPath = path.join(dir, 'matches.private.jsonl');
  const rows = loadLines(recordsPath), logs = loadLines(path.join(dir, 'summary.jsonl'));
  const run = logs.find(x => x.event === 'frozen_policy_evaluation');
  if (!run || run.failure_bits || run.shaping_weight !== 0) throw new Error('Invalid evaluator status');
  if (run.wins + run.losses + run.draws !== rows.length) throw new Error('Terminal count mismatch');
  for (const r of rows) {
    if (r.score.some((p, s) => p !== r.point_delta_sum[s]) || r.shaping_weight !== 0) throw new Error('Raw score mismatch');
    if (r.zero_hits !== (r.point_delta_sum[r.policy_side] === 0)) throw new Error('Zero-hit mismatch');
    if ((r.first_hit_seconds === null) !== r.zero_hits) throw new Error('First-hit accounting mismatch');
    if (r.first_hit_seconds !== null && (r.first_hit_seconds <= 0 || r.first_hit_seconds > r.duration_seconds + 1e-6)) throw new Error('Invalid latency');
    for (const key of ['path_length_m', 'minimum_gap_m', 'maximum_gap_m', 'mean_gap_m']) if (!Number.isFinite(r[key]) || r[key] < 0) throw new Error('Invalid movement metric');
  }
  const sides = [0, 1].map(side => {
    const records = rows.filter(x => x.policy_side === side);
    const result = logs.find(x => x.event === 'side_result' && x.policy_side === side);
    const behavior = logs.find(x => x.event === 'behavior_result' && x.policy_side === side);
    if (!result || !behavior || result.wins + result.losses + result.draws !== records.length) throw new Error('Missing side');
    const hitRows = records.filter(x => x.first_hit_seconds !== null);
    if (behavior.zero_hit_games !== records.length - hitRows.length) throw new Error('Behavior mismatch');
    return {...result, ...behavior, games: records.length, first_hit_seconds_median_when_hit: median(hitRows.map(x => x.first_hit_seconds)),
      raw_points_per_match: result.points / records.length,
      initial_xy_distinct: new Set(records.map(x => x.initial_xy.map(v => v.toFixed(4)).join(','))).size};
  });
  cases.push({name: ent.name, ...run, games: rows.length, zero_hit_games: sides.reduce((n, s) => n + s.zero_hit_games, 0),
    policy_points: sides.reduce((n, s) => n + s.points, 0), opponent_points: sides.reduce((n, s) => n + s.opponent_points, 0), sides,
    records_sha256: crypto.createHash('sha256').update(fs.readFileSync(recordsPath)).digest('hex')});
}
function median(values) {
  if (!values.length) return null;
  values.sort((a, b) => a - b);
  const i = Math.floor(values.length / 2);
  return values.length % 2 ? values[i] : (values[i - 1] + values[i]) / 2;
}
if (cases.length !== 12) throw new Error('Expected all twelve predeclared conditions');
const result = {schema: 'rek.diverse_frozen_policy.v1', host: 'spark-4ae3',
  training_rewards_used_for_ranking: false, shaping_weight: 0, physics_parity: false,
  minimum_requirement_neutral_fixed20_all_hit: cases.find(x => x.name === 'fixed-neutral-20').zero_hit_games === 0,
  fixture_protocol: {twenty_second_cases: '128 arenas x4 rounds per side;1024 matches per condition',
    three_hundred_second_cases: '32 arenas x1 round per side;64 matches per condition',
    heldout_seed: 10001, geometry_seed_matches_policy_seed: true,
    heldout_gap_m: [.55, 2.5], heldout_heading_spread_rad: Math.PI},
  measurement_notes: [
    'Both fighter sides use the same native runtime and terminal-triggered recurrent reset.',
    'Per-side point deltas are accumulated on GPU and must equal terminal raw scores.',
    'First hit is the first positive point delta, null if absent. No timeout value is substituted for a missing hit.',
    'Root path begins at the first post-reset state each round to exclude reset teleport distance; the first20ms displacement is omitted.',
    'Facing means absolute logical-heading bearing less than0.16rad; near ticks mean root gap below1.05m, not certified attack reach.',
    'Twenty-to300-second comparisons change duration and timer observations together.',
    'Fixed-start repetitions sample policy randomness, not independent geometry.'
  ], cases};
fs.writeFileSync(path.join(root, 'summary.public.json'), JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
console.log(JSON.stringify(cases.map(x => ({case: x.name, wins: x.wins, losses: x.losses, draws: x.draws, zero_hit_games: x.zero_hit_games, policy_points: x.policy_points, opponent_points: x.opponent_points})), null, 2));
