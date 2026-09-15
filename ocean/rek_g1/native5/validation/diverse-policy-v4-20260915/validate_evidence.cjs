'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = __dirname;
const readJson = file => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
const selected = readJson('mixed-stage/summary.json');
const rejected = readJson('long-stage/summary.json');
const getHash = run => {
  const values = new Set(run.cases.map(x => x.checkpoint_sha256));
  if (run.cases.length !== 12 || values.size !== 1) throw new Error('Incomplete frozen suite');
  return [...values][0];
};
const selectedHash = getHash(selected), rejectedHash = getHash(rejected);
const caseLogs = [], files = [];
function walk(base) {
  for (const ent of fs.readdirSync(base, {withFileTypes: true})) {
    const file = path.join(base, ent.name);
    if (ent.isDirectory()) walk(file);
    else {
      if (/\.private\.|\.bin$|\.o$/.test(ent.name)) throw new Error('Private or binary artifact in public evidence');
      if (ent.name === 'summary.jsonl') {
        const rows = fs.readFileSync(file, 'utf8').trim().split('\n').map(JSON.parse);
        const r = rows.find(x => x.event === 'frozen_policy_evaluation');
        const sides = rows.filter(x => x.event === 'side_result');
        const behavior = rows.filter(x => x.event === 'behavior_result');
        if (!r || r.failure_bits !== 0 || r.shaping_weight !== 0 || sides.length !== 2 || behavior.length !== 2) throw new Error('Invalid run: ' + file);
        if (Number(fs.readFileSync(path.join(base, 'exit-code.txt'), 'utf8')) !== 0) throw new Error('Run did not complete');
        if (sides.reduce((n, s) => n + s.wins + s.losses + s.draws, 0) !== r.wins + r.losses + r.draws) throw new Error('Counts disagree');
        caseLogs.push({file: path.relative(root, file).split(path.sep).join('/'), result: r, sides, behavior});
      }
      if (ent.name !== 'artifact-hashes.txt') files.push(file);
    }
  }
}
walk(root);
if (caseLogs.length !== 36) throw new Error('Expected 9 screening,24 full-suite,3 head-to-head conditions');
for (const [label, suite] of [['mixed-stage', selected], ['long-stage', rejected]]) for (const c of suite.cases) {
  const raw = caseLogs.find(x => x.file === `${label}/${c.name}/summary.jsonl`);
  if (!raw) throw new Error('Missing raw suite result');
  for (const key of ['wins', 'losses', 'draws', 'checkpoint_sha256']) if (raw.result[key] !== c[key]) throw new Error('Suite disagreement: ' + key);
  if (raw.behavior.reduce((n, b) => n + b.zero_hit_games, 0) !== c.zero_hit_games) throw new Error('Zero-hit disagreement');
}
const selection = {schema: 'rek.diverse_frozen_policy.selection.v1', selected: {stage: 'mixed-r1', checkpoint_sha256: selectedHash, summary: 'mixed-stage/summary.json'},
  rejected: {stage: 'long-r1', checkpoint_sha256: rejectedHash, summary: 'long-stage/summary.json'},
  reason: 'Mixed-r1 retained perfect measured neutral results and substantially stronger scripted, retreat, strafe, and older-policy performance. Long-r1 regressed under the same zero-shaping suite and lost the direct head-to-head tests.',
  checkpoint_selection_is_empirical_not_latest: true, physics_parity: false, universal_win_claim: false,
  total_packaged_conditions: caseLogs.length, total_packaged_matches: caseLogs.reduce((n, c) => n + c.result.wins + c.result.losses + c.result.draws, 0),
  head_to_head_long_vs_mixed: caseLogs.filter(x => x.file.startsWith('long-vs-mixed/')).map(x => ({condition: x.file, ...x.result})),
  marker_wait_regression_test: readJson('marker-wait-test.json')};
const selectionPath = path.join(root, 'selection.json');
fs.writeFileSync(selectionPath, JSON.stringify(selection, null, 2) + '\n');
if (!files.includes(selectionPath)) files.push(selectionPath);
const hashes = files.sort().map(file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') + '  ' + path.relative(root, file).split(path.sep).join('/'));
fs.writeFileSync(path.join(root, 'artifact-hashes.txt'), hashes.join('\n') + '\n');
console.log(JSON.stringify({conditions: caseLogs.length, matches: selection.total_packaged_matches, selectedHash, rejectedHash, status: 'passed'}));
