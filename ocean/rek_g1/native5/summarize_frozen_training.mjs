// Offline artifact reporting only. This program never advances or trains an environment.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const directory = path.resolve(process.argv[2] || '');
if (process.argv.length !== 3) throw new Error('Usage: node summarize_frozen_training.mjs RUN_DIRECTORY');
const text = name => fs.readFileSync(path.join(directory, name), 'utf8');
const exitCode = Number(text('exit-code.txt').trim());
if (exitCode !== 0) throw new Error(`Training did not complete successfully: ${exitCode}`);
const ini = text('logs/rek_native5/native5-frozen20s.ini');
const sections = {};
let section;
for (const line of ini.split(/\r?\n/)) {
  const heading = line.match(/^\[([^\]]+)\]$/);
  if (heading) { section = heading[1]; sections[section] = {}; continue; }
  const assignment = line.match(/^([^=]+?)\s*=\s*(.*)$/);
  if (section && assignment) sections[section][assignment[1].trim()] = assignment[2].trim();
}
const metric = key => {
  const value = sections.metrics[key];
  if (value === undefined) return null;
  const values = value.split(',').map(Number);
  if (values.some(v => !Number.isFinite(v))) throw new Error(`Nonfinite reported metric ${key}`);
  return values.at(-1);
};
const elapsed = text('process-timing.txt').match(/Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)/)?.[1];
if (!elapsed) throw new Error('Missing process wall time');
const processWallSeconds = elapsed.split(':').reduce((seconds, value) => seconds * 60 + Number(value), 0);
const rounds = JSON.parse(text('round-summary.json'));
const backend = text('stderr.txt').match(/native5 runtime: physics_backend=(\S+)/)?.[1];
if (!backend?.startsWith('puffysics_cuda')) throw new Error('Run does not identify an expected CUDA physics backend');
if (rounds.failure_bits !== 0) throw new Error('Runtime failure flags are set');
if (rounds.completed_rounds !== rounds.fighter0_wins + rounds.fighter1_wins + rounds.ties + rounds.redos + rounds.unclassified) throw new Error('Round counts do not reconcile');
const checkpoints = [];
const checkpointDirectory = path.join(directory, 'checkpoints/rek_native5/native5-frozen20s');
for (const file of fs.readdirSync(checkpointDirectory).filter(v => v.endsWith('.bin')).sort()) {
  const fullPath = path.join(checkpointDirectory, file);
  const data = fs.readFileSync(fullPath);
  if (data.length !== 1836032) throw new Error(`Unexpected checkpoint size: ${file}`);
  let nonfinite = 0;
  for (let offset = 0; offset < data.length; offset += 4) if (!Number.isFinite(data.readFloatLE(offset))) nonfinite++;
  if (nonfinite) throw new Error(`Nonfinite checkpoint parameters: ${file}, ${nonfinite}`);
  checkpoints.push({path: fullPath, transitions: Number(file.slice(0, -4)), bytes: data.length, sha256: crypto.createHash('sha256').update(data).digest('hex'), nonfinite_parameters: nonfinite});
}
const transitions = metric('agent_steps');
const uptime = metric('uptime');
const rollout = metric('perf/rollout');
const train = metric('perf/train');
const finiteLossPanels = !/\b(?:nan|[-+]?infinity|[-+]?inf)\b/i.test(text('stdout.txt'));
const config = {physics_backend:backend, stabilization:'joint_cold_start', parity_relaxed:true, arenas:Number(sections.vec.total_agents), decision_seconds:0.02, physics_seconds:0.002, physics_steps_per_decision:10, round_seconds:Number(sections.env.round_seconds), horizon:Number(sections.train.horizon), hidden_size:Number(sections.policy.hidden_size), layers:Number(sections.policy.num_layers), precision:'BF16', observation_encoding:'scaled_polar_xy', frozen_opponent_sha256:sections.env.opponent_sha256, frozen_opponent_deterministic:Number(sections.env.opponent_deterministic), seed:Number(sections.base.seed), learner_initialization:sections.base.load_model_path, python_training:false, cpu_physics_steps:0};
const report = {
  schema:'rek-native5-frozen-training-v1', directory, exit_code:exitCode, config,
  native_trainer_commit:'773f923d80e73bdc255a2ba730c918b28e416aa1',
  learner_transitions:transitions, ppo_epochs:metric('epoch'), decision_ticks_per_arena:transitions/config.arenas,
  simulated_seconds_per_arena:transitions/config.arenas*config.decision_seconds,
  process_wall_seconds:processWallSeconds, process_wall_training_sps:transitions/processWallSeconds,
  native_training_loop_seconds:uptime, native_training_loop_average_sps:transitions/uptime,
  final_reported_interval_sps:metric('SPS'),
  final_reported_interval:{rollout_seconds:rollout, ppo_seconds:train, ppo_percent_of_measured_rollout_plus_ppo:rollout>0?100*train/(rollout+train):null, note:'CUDA-graph rollout includes environment, learner inference, and frozen-opponent inference; first graph-capture timing is excluded by the pinned trainer, so a zero rollout measurement is unavailable timing, not zero work.'},
  rounds, training_fighter0_win_rate:rounds.completed_rounds?rounds.fighter0_wins/rounds.completed_rounds:null,
  average_completed_points:rounds.completed_rounds?[rounds.fighter0_completed_points/rounds.completed_rounds,rounds.fighter1_completed_points/rounds.completed_rounds]:null,
  finite_printed_loss_panels:finiteLossPanels, checkpoints,
  limitations:['Shared GPU: inspect host-and-models.txt for other processes; this is not maximum training SPS.','Round outcomes occurred while learner weights changed; these are training statistics, not held-out tournament strength.','Historical frozen policy trained as fighter0; running it as fighter1 changes the role distribution.','Cold joint caches and shortened rounds are explicit experiment changes; no REK parity or superhuman claim.']
};
process.stdout.write(JSON.stringify(report, null, 2)+'\n');
