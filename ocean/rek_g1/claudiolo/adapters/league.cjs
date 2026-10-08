#!/usr/bin/env node
'use strict';
// Claudiolo against every registered GPT checkpoint of a native league
// backend, side-reversed, fresh seeds. Reuses the league's private
// server.json (backend executable, worker config, env) and manifest.
// Claudiolo is the human side; the checkpoint runs inside the native worker
// with the same command the league tournament uses.
//
//   node adapters/league.cjs --server /private/run/server.json --backend mujoco \
//        --out /private/claudiolo-league [--policies a,b] [--seeds 1,2,3] [--rounds 1] \
//        [--round-seconds 120] [--params P.json]

const fs = require('node:fs');
const path = require('node:path');
const {play} = require('./clone.cjs');

function wilsonLower(wins, n, z = 1.96) {
  if (!n) return 0; const p = wins / n, d = 1 + z * z / n;
  return (p + z * z / (2 * n) - z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
}

function policyCommand(policy, seed) {
  if (policy.kind === 'scripted') return {scripted: true, checkpoint: '', deterministic: true, seed};
  const m = policy.checkpoint.model || {}, sel = m.actionSelection || 'greedy';
  return {checkpoint: policy.checkpoint.path, sha256: policy.checkpoint.sha256, hiddenSize: m.hiddenSize || 256,
    layers: m.layers || 2, precision: m.precision || 'bf16', observationEncoding: m.observationEncoding,
    recurrentResetTicks: m.recurrentResetTicks || 0, legacyFastHidden: m.legacyFastHidden || 0,
    deterministic: sel === 'greedy', seed};
}

async function main(a) {
  const server = JSON.parse(fs.readFileSync(a.server, 'utf8'));
  const backend = server.backends.find(b => b.id === a.backend); if (!backend) throw new Error(`no backend ${a.backend}`);
  const manifest = JSON.parse(fs.readFileSync(server.leagueFile, 'utf8'));
  let policies = Object.values(manifest.policies).filter(p => p.backend === backend.id && p.configHash === backend.configHash && p.kind === 'trained');
  if (a.policies) { const want = new Set(a.policies.split(',')); policies = policies.filter(p => want.has(p.id)); }
  const seeds = (a.seeds || '1,2,3').split(',').map(Number), rounds = +(a.rounds || 1);
  const params = a.params ? JSON.parse(fs.readFileSync(a.params, 'utf8')) : {};
  const baseConfig = JSON.parse(fs.readFileSync(backend.workerConfig, 'utf8'));
  fs.mkdirSync(a.out, {recursive: true});
  const table = {};
  for (const policy of policies) for (const seed of seeds) for (const humanSide of [0, 1]) {
    const leg = path.join(a.out, `${policy.id}-seed${seed}-claudiolo${humanSide}`);
    if (fs.existsSync(path.join(leg, 'summary.json'))) continue; // resumable
    fs.mkdirSync(leg, {recursive: true});
    const config = {...baseConfig, seed, arenas: 1, ...(a['round-seconds'] ? {round_seconds: +a['round-seconds']} : {})};
    const configFile = path.join(leg, 'worker.private.json'); fs.writeFileSync(configFile, JSON.stringify(config, null, 2) + '\n', {mode: 0o600});
    const summary = await play({bin: backend.executable, config: configFile, env: backend.env || {}, rounds, out: leg, params,
      humanSide, policy: {command: policyCommand(policy, seed)}, log: (e, d) => process.stdout.write(JSON.stringify({policy: policy.id, seed, humanSide, event: e, ...(e === 'round' ? {points: d.points, outcome: d.outcome} : {})}) + '\n')})
      .catch(error => ({error: error.message, rounds: 0, wins: 0, losses: 0, draws: 0}));
    const row = table[policy.id] ||= {legs: 0, rounds: 0, wins: 0, draws: 0, losses: 0, pointsFor: 0, pointsAgainst: 0, errors: 0};
    row.legs++; if (summary.error) { row.errors++; continue; }
    for (const k of ['rounds', 'wins', 'draws', 'losses', 'pointsFor', 'pointsAgainst']) row[k] += summary[k];
  }
  for (const row of Object.values(table)) row.wilsonLowerWin = +wilsonLower(row.wins, row.rounds).toFixed(3);
  fs.writeFileSync(path.join(a.out, 'league-summary.json'), JSON.stringify({backend: backend.id, configHash: backend.configHash, seeds, rounds, table}, null, 2) + '\n');
  console.log(JSON.stringify({event: 'league_summary', table}));
}

if (require.main === module) {
  const a = {}; const v = process.argv.slice(2); for (let i = 0; i < v.length; i += 2) a[v[i].replace(/^--/, '')] = v[i + 1];
  if (!a.server || !a.backend || !a.out) { console.error('usage: league.cjs --server S --backend B --out DIR'); process.exit(2); }
  main(a).catch(e => { console.error(e.stack || e.message); process.exitCode = 1; });
}
module.exports = {wilsonLower, policyCommand};
