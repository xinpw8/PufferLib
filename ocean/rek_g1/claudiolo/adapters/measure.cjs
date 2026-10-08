#!/usr/bin/env node
'use strict';
// Calibration probes through the native clone worker. Measures what the
// surrogate and Claudiolo's priors only guess: keyboard walk/strafe/turn
// speeds, how long until a strike is legal after releasing translation, how
// long each move holds the attack mask closed, and what Bot 1 looks like from
// outside (root speed and limb/leg activity around its accepted moves).
//
//   node adapters/measure.cjs --worker BIN --config WORKER.json --out report.json [--bot-seconds 60]

const fs = require('node:fs');
const {Worker, frameFromState} = require('./clone.cjs');
const {MOVES} = require('../moves.cjs');

const DT = 0.02;
const neutral = {forward: 0, strafe: 0, yaw: 0, moveIndex: -1, cancelAction: false};

async function probeLocomotion(w, key, command, holdTicks = 75, releaseTicks = 90) {
  let r = await w.request({op: 'reset'}); let prev = null, st = r.state;
  const rows = [];
  // Let the countdown/round start: step neutral until the round is active.
  for (let k = 0; k < 400 && !(st.phase === 2 && !st.terminal); k++) { prev = st; st = (await w.request({op: 'step', commands: [neutral, neutral]})).state; }
  for (let k = 0; k < holdTicks + releaseTicks; k++) {
    const c = k < holdTicks ? {...neutral, ...command} : neutral;
    prev = st; st = (await w.request({op: 'step', commands: [c, neutral]})).state;
    const f = frameFromState(st, prev, 0, '0');
    rows.push({k, x: f.me.x, y: f.me.y, yaw: f.me.yaw, vx: f.me.vx, vy: f.me.vy, wz: f.me.wz,
      strikeLegal: !!(st.mask && st.mask[21])});
  }
  const hold = rows.slice(holdTicks - 25, holdTicks);
  const local = r0 => { const c = Math.cos(rows[0].yaw), s = Math.sin(rows[0].yaw); return {fwd: r0.vx * c + r0.vy * s, lat: -r0.vx * s + r0.vy * c}; };
  const steady = hold.map(local);
  const mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  const firstLegal = rows.findIndex((r0, k) => k >= holdTicks && r0.strikeLegal);
  return {key, steadyForward: mean(steady.map(v => v.fwd)), steadyLateral: mean(steady.map(v => v.lat)),
    steadyYawRate: mean(hold.map(r0 => r0.wz)),
    settleSecondsAfterRelease: firstLegal < 0 ? null : (firstLegal - holdTicks) * DT, rows};
}

async function probeMove(w, move) {
  let st = (await w.request({op: 'reset'})).state, prev = null;
  for (let k = 0; k < 400 && !(st.phase === 2 && !st.terminal); k++) { prev = st; st = (await w.request({op: 'step', commands: [neutral, neutral]})).state; }
  for (let k = 0; k < 30; k++) { prev = st; st = (await w.request({op: 'step', commands: [neutral, neutral]})).state; }
  const rows = []; let accepted = null;
  for (let k = 0; k < Math.round(MOVES[move].duration / DT) + 60; k++) {
    const c = k === 0 ? {...neutral, moveIndex: move} : neutral;
    prev = st; const reply = await w.request({op: 'step', commands: [c, neutral]}); st = reply.state;
    if (k === 0) accepted = (st.commandResults?.[0]?.accepted ?? null);
    const f = frameFromState(st, prev, 0, '0');
    rows.push({k, limb: f.me.limbSpeed, leg: f.me.legSpeed, strikeLegal: !!(st.mask && st.mask[21]), x: f.me.x});
  }
  const reopen = rows.findIndex((r0, k) => k > 2 && r0.strikeLegal);
  const peakLimb = Math.max(...rows.map(r0 => r0.limb)), peakLeg = Math.max(...rows.map(r0 => r0.leg));
  return {move: MOVES[move].name, accepted, maskClosedSeconds: reopen < 0 ? null : reopen * DT,
    tableSeconds: MOVES[move].duration, peakLimb, peakLeg, rootTravel: rows.at(-1).x - rows[0].x};
}

async function observeBot(w, seconds) {
  let st = (await w.request({op: 'reset'})).state, prev = null;
  const rows = [];
  for (let k = 0; k < Math.round(seconds / DT); k++) {
    prev = st; const reply = await w.request({op: 'step', humanSide: 0, command: neutral}); st = reply.state;
    const f = frameFromState(st, prev, 0, '0'), fb = st.commandResults?.[1] || {};
    rows.push({t: k * DT, d: Math.hypot(f.opp.x - f.me.x, f.opp.y - f.me.y), oppSpeed: Math.hypot(f.opp.vx, f.opp.vy),
      oppYawRate: Math.abs(f.opp.wz), limb: f.opp.limbSpeed, leg: f.opp.legSpeed,
      botAttempted: fb.attempted || 0, botAccepted: fb.accepted || 0, botRoute: fb.route ?? null, score: st.score});
  }
  // Activity around accepted Bot 1 moves vs. while walking.
  const starts = rows.map((r0, k) => r0.botAccepted ? k : -1).filter(k => k >= 0);
  const window = (k0, a, b) => rows.slice(Math.max(0, k0 + a), k0 + b);
  const during = starts.flatMap(k0 => window(k0, 0, 25)), walking = rows.filter(r0 => r0.oppSpeed > 0.15);
  const q = (arr, key, p) => { const v = arr.map(r0 => r0[key]).sort((a, b) => a - b); return v.length ? v[Math.floor(p * (v.length - 1))] : null; };
  return {seconds, botMoveStarts: starts.length,
    limbDuringMove: {p50: q(during, 'limb', 0.5), p90: q(during, 'limb', 0.9)}, limbWalking: {p50: q(walking, 'limb', 0.5), p90: q(walking, 'limb', 0.9)},
    legDuringMove: {p50: q(during, 'leg', 0.5), p90: q(during, 'leg', 0.9)}, legWalking: {p50: q(walking, 'leg', 0.5), p90: q(walking, 'leg', 0.9)},
    distanceAtMoveStart: starts.map(k => +rows[k].d.toFixed(3)), stillBeforeMove: starts.map(k => +window(k, -15, 0).reduce((m, r0) => Math.max(m, r0.oppSpeed), 0).toFixed(3)),
    rows};
}

async function main(a) {
  const w = new Worker(a.worker, a.config, {}); await w.readyPromise;
  const report = {worker: a.worker, config: a.config, locomotion: [], moves: [], bot: null};
  for (const [key, c] of [['W', {forward: 1}], ['S', {forward: -1}], ['A', {strafe: 1}], ['D', {strafe: -1}], ['Q', {yaw: 1}], ['E', {yaw: -1}]]) {
    const r = await probeLocomotion(w, key, c); delete r.rows; report.locomotion.push(r); console.log(JSON.stringify(r));
  }
  for (const m of [1, 2, 4, 5, 10, 0, 3]) { const r = await probeMove(w, m); report.moves.push(r); console.log(JSON.stringify(r)); }
  report.bot = await observeBot(w, +(a['bot-seconds'] || 60));
  const {rows, ...botSummary} = report.bot; console.log(JSON.stringify(botSummary));
  fs.writeFileSync(a.out, JSON.stringify(report, null, 2) + '\n');
  w.close();
}

if (require.main === module) {
  const a = {}; const v = process.argv.slice(2); for (let i = 0; i < v.length; i += 2) a[v[i].replace(/^--/, '')] = v[i + 1];
  if (!a.worker || !a.config || !a.out) { console.error('usage: measure.cjs --worker BIN --config JSON --out report.json'); process.exit(2); }
  main(a).catch(e => { console.error(e.stack || e.message); process.exitCode = 1; });
}
module.exports = {probeLocomotion, probeMove, observeBot};
