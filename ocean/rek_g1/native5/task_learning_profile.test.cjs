'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {temporalProfile, environment} = require('./task_learning_profile.cjs');
const near = (actual, expected) => assert(Math.abs(actual - expected) < 1e-10);
test('discount and combined trace are derived from declared seconds', () => {
  const p = temporalProfile();
  near(p.learner.gamma ** 6000, .5);
  near((p.learner.gamma * p.learner.gae_lambda) ** 308, .5);
  assert.equal(p.learner.horizon, 512);
  assert.equal(p.learner.reset_every_horizon, 0);
  assert.equal(p.optimality_claimed, false);
  assert.equal(p.learner.boundary_patch_required, true);
  assert.equal(p.learner.valid_transitions_with_boundary_patch, 512);
  assert.equal(environment(p).REK_TRAIN_GAMMA, environment(p).REK_FAST_SHAPING_GAMMA);
});
test('cadence changes preserve temporal meaning', () => {
  const normal = temporalProfile(), slower = temporalProfile({control_hz: 25});
  near(slower.learner.gamma, normal.learner.gamma ** 2);
  near(slower.learner.gae_lambda, normal.learner.gae_lambda ** 2);
  assert.equal(slower.learner.horizon, 256);
});
test('baseline limitation and reward scale remain explicit', () => {
  const p = temporalProfile();
  near(p.baseline.internal_credit_window_seconds, 2.54);
  assert(p.baseline.trace_half_life_seconds > 2.3 && p.baseline.trace_half_life_seconds < 2.31);
  assert(p.baseline.round_reward_retention < .0025);
  assert.equal(p.reward.terminal_reward_opt_in, false);
  assert.equal(p.reward.terminal_bonus_runtime_implemented, false);
  assert.equal(Object.hasOwn(environment(p), 'REK_FAST_TERMINAL_WIN_POINTS'), false);
  assert.equal(p.reward.intrinsic_reward_enabled, false);
  assert.equal(p.exploration.held_action_resampling_is_new_execution, false);
  assert.equal(temporalProfile({terminal_win_points: 5}).reward.terminal_reward_opt_in, true);
});
test('invalid physical or learner constraints fail before producing a profile', () => {
  for (const options of [{control_hz: 0}, {horizon: 128}, {horizon: 513},
    {trace_half_life_seconds: 121}, {ent_coef: -1}, {terminal_win_points: -1}, {round_seconds: NaN}])
    assert.throws(() => temporalProfile(options));
});
