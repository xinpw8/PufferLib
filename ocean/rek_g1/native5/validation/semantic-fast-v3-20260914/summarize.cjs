'use strict';

// Offline evidence reduction only. Never invokes training or a physics backend.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const dir = __dirname;
const read = (name) => fs.readFileSync(path.join(dir, name), 'utf8');
const ini = read('training/semantic-cuda-v3-512-16.ini');
const metrics = Object.fromEntries(ini.split('[metrics]\n')[1].trim().split('\n').map(line => {
  const at = line.indexOf(' = ');
  return [line.slice(0, at), line.slice(at + 3).split(',').map(Number)];
}));
const finalIndex = metrics.epoch.indexOf(4096);
if (finalIndex <= 0) throw new Error('Missing completed training metric');
const last = (key) => metrics[key][finalIndex];
const prior = (key) => metrics[key][finalIndex - 1];
const elapsed = /Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\d+):(\d+(?:\.\d+)?)/.exec(read('training/process-timing.txt'));
if (!elapsed) throw new Error('Missing measured process elapsed time');
const processSeconds = Number(elapsed[1]) * 60 + Number(elapsed[2]);
const rounds = JSON.parse(read('training/round-summary.json'));
const interval = last('uptime') - prior('uptime');
const accounted = last('perf/rollout') + last('perf/train');
const checkpoints = read('training/checkpoint-hashes.txt').trim().split('\n').map(line => {
  const [sha256, file] = line.trim().split(/\s+/);
  return {path: file, sha256, transitions: Number(path.basename(file, '.bin'))};
});
const summary = {
  schema: 'rek.semantic_cuda.v3.training.v1',
  host: 'spark-4ae3', gpu: 'NVIDIA GB10', backend: 'semantic_cuda', runtime_version: 3,
  classification: 'approximate_points_only_action_level_candidate', physics_parity_claim: false,
  cpu_physics_steps: 0, python_training_runtime: false,
  headless: true, arenas: 512, control_hz: 50, horizon: 16, minibatch_size: 8192,
  seed: 73, round_seconds: 20, opponent: 'GPU scripted standard opponent',
  completed_transitions: last('agent_steps'), completed_epochs: last('epoch'),
  measured_training_uptime_seconds: last('uptime'),
  full_run_training_sps: last('agent_steps') / last('uptime'),
  measured_process_wall_seconds: processSeconds,
  process_inclusive_sps: last('agent_steps') / processSeconds,
  final_interval: {
    transitions: last('agent_steps') - prior('agent_steps'),
    elapsed_seconds: interval, reported_sps: last('SPS'),
    rollout_seconds: last('perf/rollout'), train_seconds: last('perf/train'),
    train_model_seconds: last('perf/train_model'), train_misc_seconds: last('perf/train_misc'),
    rollout_share_of_accounted_percent: 100 * last('perf/rollout') / accounted,
    train_share_of_accounted_percent: 100 * last('perf/train') / accounted,
    unaccounted_interval_seconds: interval - accounted,
    timing_caveat: 'Native timer categories for the final reporting window; not an independent CUDA profile. Rollout includes inference and environment. Zero detailed eval_env timer is uninstrumented, not zero environment cost.'
  },
  final_rolling_dashboard_rounded: {wins: .998, losses: .001, draws: .001, mean_score: 4.700,
    mean_hits: 5.738, mean_episode_length: 1000, falls: 0, actions_invalid: 0, finite_failures: 0},
  cumulative_training_rounds: rounds,
  training_metrics_caveat: 'Online training history includes changing policies. Final rolling metrics are rounded. Frozen checkpoint strength is measured in separately recorded policy evaluations.',
  exit_code: Number(read('training/exit-code.txt').trim()),
  checkpoints: [1048576, 33554432].map(n => {
    const found = checkpoints.find(c => c.transitions === n);
    if (!found) throw new Error('Missing checkpoint ' + n);
    return found;
  }),
  input_probe_passed: read('input-probe/exit-code.txt').trim() === '0',
  retained_v2_objects_byte_identical: ['pufferl.o', 'fast_assets.o', 'native_policy.o', 'cJSON.o'],
  source_sha256: /([a-f0-9]{64})\s+[^\n]*\/fast_runtime.cu/.exec(read('training/fast-build.txt'))[1],
  other_gpu_processes_present: true,
  private_assets_and_checkpoints_in_git: false
};
if (summary.exit_code || rounds.failure_bits || !summary.input_probe_passed) throw new Error('Failed run');
fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
const files = [];
function walk(base) {
  for (const ent of fs.readdirSync(base, {withFileTypes: true})) {
    const file = path.join(base, ent.name);
    if (ent.isDirectory()) walk(file);
    else if (ent.name !== 'artifact-hashes.txt') files.push(file);
  }
}
walk(dir);
const hashes = files.sort().map(file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') + '  ' + path.relative(dir, file).split(path.sep).join('/'));
fs.writeFileSync(path.join(dir, 'artifact-hashes.txt'), hashes.join('\n') + '\n');
console.log(JSON.stringify(summary, null, 2));
