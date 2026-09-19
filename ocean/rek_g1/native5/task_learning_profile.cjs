'use strict';

const PIN = '773f923d80e73bdc255a2ba730c918b28e416aa1';
const BASELINE = Object.freeze({control_hz: 50, horizon: 128, gamma: .999, gae_lambda: .995, ent_coef: .01});
function positive(name, value) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be finite and positive`);
  return value;
}
function halfLife(retention, hz) {
  return retention === 1 ? Infinity : Math.log(.5) / (hz * Math.log(retention));
}
function temporalProfile(options = {}) {
  const hz = positive('control_hz', options.control_hz ?? 50);
  const round = positive('round_seconds', options.round_seconds ?? 120);
  const move = positive('longest_move_seconds', options.longest_move_seconds ?? 3.16);
  const count = positive('no_recovery_count_seconds', options.no_recovery_count_seconds ?? 3);
  const discountHalfLife = positive('discount_half_life_seconds', options.discount_half_life_seconds ?? round);
  const traceHalfLife = positive('trace_half_life_seconds', options.trace_half_life_seconds ?? move + count);
  if (traceHalfLife > discountHalfLife) throw new Error('trace half life exceeds reward half life, requiring lambda > 1');
  const gamma = Math.exp(Math.log(.5) / (hz * discountHalfLife));
  const lambda = Math.exp(Math.log(.5) / (hz * traceHalfLife)) / gamma;
  const requested = positive('minimum_credit_window_seconds', options.minimum_credit_window_seconds ?? move + count);
  const minimum = Math.ceil(requested * hz);
  const horizon = options.horizon ?? 2 ** Math.ceil(Math.log2(minimum));
  if (!Number.isSafeInteger(horizon) || horizon < minimum || horizon % 8 !== 0)
    throw new Error('horizon must cover the credit window and be divisible by 8');
  const entCoef = options.ent_coef ?? BASELINE.ent_coef;
  if (!Number.isFinite(entCoef) || entCoef < 0) throw new Error('ent_coef must be finite and nonnegative');
  const terminal = options.terminal_win_points ?? 0;
  if (!Number.isFinite(terminal) || terminal < 0) throw new Error('terminal_win_points must be finite and nonnegative');
  return {
    schema: 'rek.task_learning_profile.v1', pufferlib_commit: PIN,
    readiness: 'requires_runtime_parity_and_boundary_regression', optimality_claimed: false,
    physical_seconds: {control_hz: hz, round_seconds: round, longest_move_seconds: move,
      no_recovery_count_seconds: count, minimum_credit_window_seconds: requested},
    declared_design_choices: {discount_half_life_seconds: discountHalfLife,
      trace_half_life_seconds: traceHalfLife,
      rationale: 'Reward retains one half over a round; the GAE trace retains one half over a move plus the no-recovery count. These are testable design choices.'},
    learner: {horizon, gamma, gae_lambda: lambda, ent_coef: entCoef, reset_every_horizon: 0,
      round_seconds: round, valid_transitions_with_boundary_patch: horizon,
      separate_boundary_reward_terminal_value: true,
      boundary_patch_required: true},
    baseline: {...BASELINE, rollout_seconds: BASELINE.horizon / BASELINE.control_hz,
      internal_credit_window_seconds: (BASELINE.horizon - 1) / BASELINE.control_hz,
      reward_half_life_seconds: halfLife(BASELINE.gamma, BASELINE.control_hz),
      trace_half_life_seconds: halfLife(BASELINE.gamma * BASELINE.gae_lambda, BASELINE.control_hz),
      round_reward_retention: BASELINE.gamma ** (round * BASELINE.control_hz)},
    reward: {point_scale: 1, terminal_win_points: terminal, terminal_reward_opt_in: terminal > 0,
      terminal_bonus_runtime_implemented: false, terminal_bonus_scope: 'reference contract only; runtime reward remains raw point differential',
      ko_award_points: 5, ko_in_point_delta_only: true, terminal_bonus_independent_of_win_path: true,
      changes_game_points: false,
      intrinsic_reward_enabled: false},
    exploration: {mechanism: 'masked categorical PPO entropy regularization', available_actions_max: 33,
      entropy_max_nats_full_mask: Math.log(33), entropy_max_nats_busy_yaw_only: Math.log(4),
      entropy_max_nats_reset_wait: 0, coefficient_inherited_from_baseline: options.ent_coef === undefined,
      decision_cadence_hz: hz, held_action_resampling_is_new_execution: false,
      required_diagnostics: ['legal mask size', 'entropy/log(legal count) when legal count > 1',
        'accepted attack starts per second', 'fraction busy', 'per-route accepted executions'],
      independent_intrinsic_reward: false},
    boundaries: {round_timeout: 'true episode terminal', rollout_cut: 'bootstrap and carry recurrent state',
      external_truncation: 'bootstrap final observation, reset memory only if environment resets',
      pinned_api_has_separate_truncation: false},
  };
}
function environment(profile) {
  const p = profile.learner;
  return {REK_TRAIN_GAMMA: String(p.gamma), REK_TRAIN_GAE_LAMBDA: String(p.gae_lambda),
    REK_TRAIN_ENTROPY: String(p.ent_coef), REK_FAST_SHAPING_GAMMA: String(p.gamma)};
}
if (require.main === module) {
  try {
    const args = process.argv.slice(2);
    let options = {};
    if (args.length) {
      if (args.length !== 2 || args[0] !== '--options-json') throw new Error('Usage: node task_learning_profile.cjs [--options-json JSON]');
      options = JSON.parse(args[1]);
    }
    const profile = temporalProfile(options);
    process.stdout.write(JSON.stringify({...profile, environment: environment(profile)}, null, 2) + '\n');
  } catch (error) { process.stderr.write(error.message + '\n'); process.exitCode = 2; }
}
module.exports = {temporalProfile, environment, halfLife, PIN};
