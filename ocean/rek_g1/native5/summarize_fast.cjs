'use strict';
// Summarize actual headless trainer runs. Repeated terminal log bins are not
// independent samples and do not increase either counts or elapsed time.
const fs = require('node:fs');
const path = require('node:path');
function summarize(directory) {
  const logDirectory = path.join(directory, 'logs', 'rek_native5');
  const files = fs.readdirSync(logDirectory).filter(name => name.endsWith('.ini'));
  if (files.length !== 1) throw new Error(`Expected one trainer log in ${directory}`);
  const text = fs.readFileSync(path.join(logDirectory, files[0]), 'utf8');
  const sections = {};
  let section;
  for (const line of text.split(/\r?\n/)) {
    const heading = line.match(/^\[([^\]]+)\]$/);
    if (heading) { section = sections[heading[1]] = {}; continue; }
    const pair = line.match(/^([^#=]+?)\s*=\s*(.*)$/);
    if (pair && section) section[pair[1].trim()] = pair[2];
  }
  const metric = key => {
    const value = sections.metrics[key];
    if (!value) throw new Error(`Missing metric ${key}`);
    const values = value.split(',').map(Number);
    if (!values.every(Number.isFinite)) throw new Error(`Invalid metric ${key}`);
    return values;
  };
  const steps = metric('agent_steps');
  const uptime = metric('uptime');
  const epochs = metric('epoch');
  const total = Number(sections.train.total_timesteps);
  if (steps.at(-1) !== total || uptime.at(-1) <= 0) throw new Error('Incomplete training');
  const timing = fs.readFileSync(path.join(directory, 'process-timing.txt'), 'utf8');
  const wallMatch = timing.match(/Elapsed \(wall clock\) time .*?:\s*([\d:.]+)\s*$/m);
  if (!wallMatch) throw new Error('Missing process wall time');
  const wallSeconds = wallMatch[1].split(':').map(Number).reduce((a, b) => 60 * a + b, 0);
  const exitCode = Number(fs.readFileSync(path.join(directory, 'exit-code.txt'), 'utf8'));
  if (exitCode !== 0) throw new Error(`Failed training run ${directory}`);
  const rounds = JSON.parse(fs.readFileSync(path.join(directory, 'round-summary.json'), 'utf8'));
  const sum = rounds.fighter0_wins + rounds.fighter1_wins + rounds.ties + rounds.redos + rounds.unclassified;
  if (sum !== rounds.completed_rounds || rounds.failure_bits) throw new Error('Invalid round accounting');
  return {
    run: path.basename(directory), backend: 'semantic_cuda',
    learnerTransitions: total, simulatedSecondsPerTransition: 0.02,
    arenas: Number(sections.vec.total_agents), horizon: Number(sections.train.horizon),
    minibatch: Number(sections.train.minibatch_size), epochs: epochs.at(-1),
    trainingLoopSeconds: uptime.at(-1), trainingSps: total / uptime.at(-1),
    processWallSeconds: wallSeconds, startupInclusiveSps: total / wallSeconds,
    aggregateSimulatedSecondsPerWallSecond: 0.02 * total / uptime.at(-1),
    trainingRoundResults: rounds,
    trainingWinFraction: rounds.fighter0_wins / rounds.completed_rounds,
    heldOutWinFraction: null,
    note: 'Training counts include the changing policy. They are not held-out strength. Same 50 Hz ticks; opponent actions and physics substeps do not multiply SPS.',
  };
}
if (require.main === module) {
  if (process.argv.length < 3) throw new Error('Usage: node summarize_fast.cjs RUN_DIRECTORY ...');
  process.stdout.write(JSON.stringify(process.argv.slice(2).map(summarize), null, 2) + '\n');
}
module.exports = {summarize};
