#!/usr/bin/env node
'use strict';
// Robust parameter search for Claudiolo on the surrogate: a (1+lambda)
// evolution strategy whose fitness is the mean over every physics world in
// worlds.cjs (common random numbers, fixed seeds). A parameter set that only
// wins in one guessed world is worthless; the clone and rek.exe decide later.
//
//   node tune.cjs [--generations 12] [--lambda 6] [--rounds 16] [--out best.json]

const {Worker, isMainThread, parentPort, workerData} = require('node:worker_threads');
const fs = require('node:fs');
const os = require('node:os');

const SPACE = {
  strikeMax: [0.58, 0.76], strikeMin: [0.38, 0.50], standMin: [0.52, 0.72], standMax: [0.66, 0.90],
  oppHandReach: [0.62, 0.82], evadeLead: [0.05, 0.40], preemptMargin: [-0.05, 0.12],
  stepInTime: [0.3, 1.2], longMoveMinRemaining: [0.6, 2.5], aimTol: [0.10, 0.35], tooClose: [0.36, 0.48],
  minOppImpact: [0.08, 0.30], kiteLead: [3, 12], kiteTime: [10, 45],
};

function evaluate(params, rounds, seed0) {
  const S = require('./surrogate.cjs'), K = require('./core.cjs'), {WORLDS} = require('./worlds.cjs');
  let wins = 0, n = 0, diff = 0, worst = 1;
  for (const phys of Object.values(WORLDS)) {
    const s = S.summarize(S.playRounds(() => K.controller(params), {rounds, seed0, phys}));
    wins += s.wins + 0.5 * s.draws; n += s.rounds; diff += s.pointsFor - s.pointsAgainst;
    worst = Math.min(worst, (s.wins + 0.5 * s.draws) / s.rounds);
  }
  return {winRate: wins / n, worst, diffPerRound: diff / n, fitness: wins / n + 0.5 * worst + 0.002 * diff / n};
}

if (!isMainThread) {
  parentPort.postMessage(evaluate(workerData.params, workerData.rounds, workerData.seed0));
} else {
  const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => x.startsWith('--') ? [...a, [x.slice(2), arr[i + 1]]] : a, []));
  const generations = +(args.generations || 12), lambda = +(args.lambda || 6), rounds = +(args.rounds || 16), seed0 = +(args.seed || 900);
  const out = args.out || 'tuned.json';
  let rng = 12345; const rand = () => (rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648;
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const run = params => new Promise((resolve, reject) => {
    const w = new Worker(__filename, {workerData: {params, rounds, seed0}}); w.once('message', resolve); w.once('error', reject);
  });
  const pool = async list => { const res = []; const cpus = Math.max(1, os.cpus().length);
    for (let i = 0; i < list.length; i += cpus) res.push(...await Promise.all(list.slice(i, i + cpus).map(run))); return res; };
  (async () => {
    const K = require('./core.cjs');
    let best = Object.fromEntries(Object.keys(SPACE).map(k => [k, K.DEFAULTS[k]]));
    let bestScore = (await pool([best]))[0]; let sigma = 0.15;
    console.log(JSON.stringify({gen: 0, ...bestScore, params: best}));
    for (let g = 1; g <= generations; g++) {
      const kids = Array.from({length: lambda}, () => Object.fromEntries(Object.entries(best).map(([k, v]) => {
        const [lo, hi] = SPACE[k]; return [k, Math.min(hi, Math.max(lo, v + gauss() * sigma * (hi - lo)))];
      })));
      const scores = await pool(kids);
      let improved = false;
      scores.forEach((s, i) => { if (s.fitness > bestScore.fitness) { bestScore = s; best = kids[i]; improved = true; } });
      sigma = improved ? Math.min(0.3, sigma * 1.2) : Math.max(0.03, sigma * 0.8);
      console.log(JSON.stringify({gen: g, sigma: +sigma.toFixed(3), ...bestScore, params: best}));
      fs.writeFileSync(out, JSON.stringify({score: bestScore, params: best}, null, 2) + '\n');
    }
  })().catch(e => { console.error(e); process.exitCode = 1; });
}
module.exports = {evaluate, SPACE};
