'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {prepare}=require('./prepare_authentic_ppo_kernel.cjs');
const fixture=`struct PPOGraphArgs {
    const precision_t* old_logprobs;
    const precision_t* advantages;
    const precision_t* values;
};
__global__ void ppo_loss_compute(
        float* ppo_partials, PPOKernelArgs a, PPOGraphArgs g) {
    float logratio = a.grad_values_pred[nt] - to_float(g.old_logprobs[nt]);
    float val = to_float(g.values[nt]);
    float ratio = __expf(logratio);
}

// Deterministic reduction of per-block PPO loss partials
`;
test('only names and frozen old logprob type/read change',()=>{
  const {code,provenance}=prepare(fixture);
  assert.match(code,/const float\* old_logprobs/);assert.match(code,/PPOGraphArgsFp32/);
  assert.match(code,/ppo_loss_compute_fp32/);assert.match(code,/float ratio = __expf\(logratio\);/);
  assert.equal(provenance.other_kernel_bytes_unchanged,true);
});
test('fail closed on changed source shape or ambiguous substitutions',()=>{
  assert.throws(()=>prepare(fixture.replace('const precision_t* old_logprobs','const double* old_logprobs')),/exactly one/);
  assert.throws(()=>prepare(fixture+fixture),/boundary/);
  assert.throws(()=>prepare(fixture.replace('to_float(g.old_logprobs[nt])','to_float(g.old_logprobs[nt + 1])')),/exactly one/);
});
