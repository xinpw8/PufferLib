'use strict';
const fs = require('node:fs');
const path = require('node:path');
const root = process.argv[2] ? path.resolve(process.argv[2]) : __dirname;
const results = [];
for (const stage of fs.readdirSync(root, {withFileTypes: true}).filter(x => x.isDirectory())) {
  const conditions = [];
  for (const condition of ['fixed-neutral', 'fixed-scripted', 'heldout-neutral']) {
    const dir = path.join(root, stage.name, condition);
    if (!fs.existsSync(dir)) continue;
    const lines = fs.readFileSync(path.join(dir, 'summary.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    const run = lines.find(x => x.event === 'frozen_policy_evaluation');
    const sides = lines.filter(x => x.event === 'side_result');
    const behavior = lines.filter(x => x.event === 'behavior_result');
    if (!run || run.failure_bits || run.shaping_weight !== 0 || sides.length !== 2 || behavior.length !== 2 || Number(fs.readFileSync(path.join(dir, 'exit-code.txt'), 'utf8')) !== 0) throw new Error('Invalid screen ' + stage.name + '/' + condition);
    conditions.push({condition, ...run, policy_points: sides.reduce((a, b) => a + b.points, 0), opponent_points: sides.reduce((a, b) => a + b.opponent_points, 0), zero_hit_games: behavior.reduce((a, b) => a + b.zero_hit_games, 0), sides, behavior});
  }
  if (conditions.length) {
    if (conditions.length !== 3) throw new Error('Incomplete three-condition screening stage');
    results.push({stage: stage.name, complete_diverse_suite: false, conditions});
  }
}
fs.writeFileSync(path.join(root, 'screens.json'), JSON.stringify({schema: 'rek.diverse_policy.preliminary_screens.v1', staged_results: results}, null, 2) + '\n');
console.log(JSON.stringify({stages: results.length, conditions: results.reduce((a, b) => a + b.conditions.length, 0), passed_evidence_validation: true}));
