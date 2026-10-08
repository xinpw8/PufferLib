'use strict';
// Fit the surrogate's free hazard/acceptance parameters so a neutral-input
// defender reproduces the authentic passive-defender cohort vs Sparring Bot 1
// (validation/passive-defender-repeat-20260917: 8 rounds, Bot strike points
// 81 = 49 hand + 16 kick packets, Bot falls 12 (incl. one double), defender
// falls 2). Coordinate descent over a few knobs; output is a phys override.
const S = require('./surrogate.cjs');

const TARGET = {botStrikePerRound: 81 / 8, botFallsPerRound: 12 / 8, ownFallsPerRound: 2 / 8, kickShare: 32 / 81};

function measure(phys, rounds = 48, seed0 = 1000) {
  const res = S.playRounds(() => S.passive, {rounds, seed0, phys});
  let kickPts = 0, strike = 0;
  for (const r of res) for (const e of r.events) if (e.type === 'hit' && e.side === 1) { strike += e.points; if (e.points === 2) kickPts += 2; }
  const sum = S.summarize(res);
  return {botStrikePerRound: sum.strikeAgainst / rounds, botFallsPerRound: sum.fallsOpp / rounds,
    ownFallsPerRound: sum.fallsOwn / rounds, kickShare: strike ? kickPts / strike : 0};
}

function loss(m) {
  return ((m.botStrikePerRound - TARGET.botStrikePerRound) / 3) ** 2 + ((m.botFallsPerRound - TARGET.botFallsPerRound) / 0.4) ** 2 +
    ((m.ownFallsPerRound - TARGET.ownFallsPerRound) / 0.15) ** 2 + ((m.kickShare - TARGET.kickShare) / 0.15) ** 2;
}

function calibrate({iterations = 3, rounds = 48, start = {}} = {}) {
  const knobs = {handAccept: [0.15, 0.9], kickAccept: [0.15, 0.9], pushSelf: [0.05, 3], pushOther: [0.0, 1.5],
    kickSelfFall: [0.0, 0.4]};
  let phys = {handAccept: 0.35, kickAccept: 0.45, pushSelf: 0.55, pushOther: 0.12, kickSelfFall: 0.10, ...start};
  let best = loss(measure(phys, rounds));
  for (let it = 0; it < iterations; it++) {
    for (const [k, [lo, hi]] of Object.entries(knobs)) {
      for (const f of [0.6, 0.8, 1.25, 1.6]) {
        const v = Math.min(hi, Math.max(lo, (phys[k] || 0.01) * f));
        const trial = {...phys, [k]: v}, l = loss(measure(trial, rounds));
        if (l < best) { best = l; phys = trial; }
      }
    }
    process.stderr.write(`iteration ${it} loss ${best.toFixed(3)} ${JSON.stringify(phys)}\n`);
  }
  return {phys, loss: best, measured: measure(phys, rounds * 2, 5000), target: TARGET};
}

if (require.main === module) console.log(JSON.stringify(calibrate(), null, 2));
module.exports = {calibrate, measure, TARGET};
