#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const crypto = require('node:crypto');
const sha = x => crypto.createHash('sha256').update(x).digest('hex');
const check = (ok, message) => { if (!ok) throw Error(message); };
function unique(source, pattern, replacement) {
  const matches = source.match(pattern);
  check(matches?.length === 1, 'expected exactly one native PPO adapter match: ' + pattern);
  return source.replace(pattern, replacement);
}
function prepare(source) {
  const graph = source.match(/struct PPOGraphArgs \{[\s\S]*?\n\};/g);
  const kernel = source.match(/__global__ void ppo_loss_compute\([\s\S]*?\n\}\n(?=\n\/\/ Deterministic reduction of per-block PPO loss)/g);
  check(graph?.length === 1 && kernel?.length === 1, 'native PPO source boundary mismatch');
  let g = unique(graph[0], /struct PPOGraphArgs/g, 'struct PPOGraphArgsFp32');
  g = unique(g, /const precision_t\* old_logprobs;/g, 'const float* old_logprobs;');
  g = unique(g, /const precision_t\* values;/g, 'const float* values;');
  let k = unique(kernel[0], /__global__ void ppo_loss_compute\(/g, '__global__ void ppo_loss_compute_fp32(');
  k = unique(k, /PPOGraphArgs g/g, 'PPOGraphArgsFp32 g');
  k = unique(k, /to_float\(g\.old_logprobs\[nt\]\)/g, 'g.old_logprobs[nt]');
  k = unique(k, /to_float\(g\.values\[nt\]\)/g, 'g.values[nt]');
  // Reverse the seven substitutions and prove that all native math is unchanged.
  const restoredGraph = g.replace('PPOGraphArgsFp32', 'PPOGraphArgs').replace('const float* old_logprobs;', 'const precision_t* old_logprobs;')
    .replace('const float* values;', 'const precision_t* values;');
  const restoredKernel = k.replace('ppo_loss_compute_fp32(', 'ppo_loss_compute(').replace('PPOGraphArgsFp32 g', 'PPOGraphArgs g')
    .replace(' - g.old_logprobs[nt];', ' - to_float(g.old_logprobs[nt]);').replace(' = g.values[nt];', ' = to_float(g.values[nt]);');
  check(restoredGraph === graph[0] && restoredKernel === kernel[0], 'PPO adapter changed more than names and old-logprob dtype');
  return {code: '// Generated native PPO overload: exact FP32 frozen behavior log probabilities.\n' + g + '\n\n' + k,
    provenance: {schema:'rek.authentic_ppo.fp32_old_logprob_adapter.v1',source_sha256:sha(source),
      original_graph_sha256:sha(graph[0]),original_kernel_sha256:sha(kernel[0]),
      changes:['distinct graph/kernel names','old_logprobs and old values pointers precision_t to float','old logprob/value reads without BF16 conversion'],
      other_kernel_bytes_unchanged:true}};
}
if (require.main === module) {
  try {
    check(process.argv.length === 4, 'Usage: prepare_authentic_ppo_kernel.cjs PREPARED_ALGO_CU NEW_OUTPUT_CUH');
    const [sourcePath, output] = process.argv.slice(2), result=prepare(fs.readFileSync(sourcePath,'utf8'));
    check(!fs.existsSync(output)&&!fs.existsSync(output+'.json'),'output already exists');
    fs.writeFileSync(output,result.code,{flag:'wx'});
    fs.writeFileSync(output+'.json',JSON.stringify({...result.provenance,generated_sha256:sha(result.code)},null,2)+'\n',{flag:'wx'});
  } catch (e) { console.error(e.message); process.exitCode=2; }
}
module.exports={prepare};
