'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = __dirname;
const summary = JSON.parse(fs.readFileSync(path.join(root, 'summary.json'), 'utf8'));
if (summary.cases.length !== 8) throw new Error('Expected eight cases');
for (const c of summary.cases) {
  const dir = path.join(root, `v${c.version}-${c.opponent}-${c.configured_round_seconds}`);
  const lines = fs.readFileSync(path.join(dir, 'summary.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const overall = lines.find(x => x.event === 'frozen_policy_evaluation');
  if (Number(fs.readFileSync(path.join(dir, 'exit-code.txt'), 'utf8')) !== 0) throw new Error('Failed run');
  for (const key of ['wins', 'losses', 'draws', 'failure_bits']) if (c[key] !== overall[key]) throw new Error('Summary mismatch: ' + key);
  if (c.wins + c.losses + c.draws !== 1024 || c.failure_bits !== 0) throw new Error('Incomplete or failed fixture');
  if (c.opponent === 'neutral' && c.configured_round_seconds === 20 && (c.policy_points !== 0 || c.zero_scoring_games !== 1024)) throw new Error('AFK control mismatch');
}
const files = [];
function walk(base) {
  for (const entry of fs.readdirSync(base, {withFileTypes: true})) {
    const file = path.join(base, entry.name);
    if (entry.isDirectory()) walk(file);
    else if (entry.name !== 'artifact-hashes.txt') {
      if (/\.bin$|\.private\.|\.o$/.test(entry.name)) throw new Error('Unexpected private or binary artifact');
      files.push(file);
    }
  }
}
walk(root);
const hashes = files.sort().map(file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') + '  ' + path.relative(root, file).split(path.sep).join('/'));
fs.writeFileSync(path.join(root, 'artifact-hashes.txt'), hashes.join('\n') + '\n');
console.log(JSON.stringify({cases: summary.cases.length, recorded_matches: summary.cases.reduce((a, c) => a + c.matches, 0), files: files.length, validation: 'passed'}));
