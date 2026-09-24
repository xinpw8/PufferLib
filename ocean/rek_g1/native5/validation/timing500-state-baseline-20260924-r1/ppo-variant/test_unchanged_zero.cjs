'use strict';
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const [original, variant] = process.argv.slice(2);
const base = fs.readFileSync(original, 'utf8');
const edited = fs.readFileSync(variant, 'utf8');
assert.equal(crypto.createHash('sha256').update(base).digest('hex'), '173bfaf5c8e511ab84f55efbfee956bd54fb1bc780542667269d5c9c742dde11');
function region(text, first, after) {
  const a = text.indexOf(first), b = text.indexOf(after, a + first.length);
  assert(a >= 0 && b > a);
  return text.slice(a, b);
}
assert.equal(region(edited, 'struct AuthenticData {', '__global__ void authentic_target_slice'), region(base, 'struct AuthenticData {', '__global__ void authentic_target_slice'));
assert.equal(region(edited, '__global__ void zero_excluded_ppo_gradients', 'float number('), region(base, '__global__ void zero_excluded_ppo_gradients', 'float number('));
assert(edited.includes('if (state_baseline) {'));
assert(edited.includes('GaeTargets targets(data, replay, complete_mc, state_baseline ? &*state_baseline : nullptr);'));
assert(edited.includes('if (cross_fitted_baseline) cross_fitted_baseline_gpu_test();'));
assert(edited.includes('const bool complete_mc = complete_mc_zero_baseline || cross_fitted_baseline;'));
assert(edited.includes('require(!(complete_mc_zero_baseline && cross_fitted_baseline)'));
const prefix = ['--state-baseline=', '--state-baseline-sha256=', '--state-baseline-protocol-sha256='];
for (const flag of prefix) {
  const line = edited.split('\n').find(s => s.includes(`option.rfind("${flag}"`));
  assert(line);
  assert(line.includes(`option.substr(${flag.length})`));
}
console.log(JSON.stringify({static_checks_passed:true, original_sha256:'173bfaf5c8e511ab84f55efbfee956bd54fb1bc780542667269d5c9c742dde11', original_mc_kernel_identical:true, replay_mask_parity_loss_optimizer_region_identical:true, zero_mode_skips_baseline:true, gpu_used:false}));
