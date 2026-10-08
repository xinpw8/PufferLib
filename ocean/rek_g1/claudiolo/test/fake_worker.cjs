#!/usr/bin/env node
'use strict';
// Protocol double of rek-eval-worker backed by the surrogate, for adapter tests.
// Speaks the same JSONL: {"event":"ready"}, op reset/step/snapshot, replies
// {id, ok, state:{tick, phase, terminal, completedRounds, roundNumber,
// timeRemaining, qpos[72], qvel[70], score[2], falls[2], mask[66], raw[446],
// commandResults[2]}, rounds:[...] }.
const readline = require('node:readline');
const S = require('../surrogate.cjs');
const {MOVE_TO_CATEGORY, heldCategory} = require('../moves.cjs');

let round = null, tick = 0, completed = 0, seed = 11, feedback = [{}, {}];
function fresh() { round = new S.Round({seed: seed++}); round.botCommand = {forward: 0, strafe: 0, yaw: 0}; }
function quat(yaw) { return [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)]; }
function state() {
  const qpos = Array(72).fill(0), qvel = Array(70).fill(0), raw = Array(446).fill(0);
  round.f.forEach((f, s) => {
    const q = quat(f.yaw), z = f.down ? 0.2 : 0.8;
    qpos.splice(s * 36, 7, f.x, f.y, z, ...q);
    qvel.splice(s * 35, 6, f.vx, f.vy, 0, 0, 0, f.wz);
  });
  for (let side = 0; side < 2; side++) for (let k = 0; k < 2; k++) {
    const f = round.f[k === 0 ? side : 1 - side], base = side * 223 + k * 86;
    raw[base + 42 + 15] = f.limbSpeed / 0.45; raw[base + 42 + 0] = f.legSpeed / 0.8;
  }
  const m0 = round.mask(0), m1 = round.mask(1);
  return {ok: true, tick, phase: round.over ? 3 : 2, terminal: round.over ? 1 : 0, completedRounds: completed,
    roundNumber: completed + 1, timeRemaining: Math.max(0, round.phys.roundSeconds - round.t), qpos, qvel,
    score: [...round.points], falls: [...round.falls], mask: [...m0, ...m1].map(Number), raw, commandResults: feedback,
    winner: round.points[0] > round.points[1] ? 0 : round.points[1] > round.points[0] ? 1 : -1,
    roundResult: round.points[0] === round.points[1] ? 3 : 1, failure: null};
}
function step(command, humanSide) {
  if (humanSide !== 0) throw new Error('fake worker supports humanSide 0');
  const held = heldCategory(command.forward, command.strafe, command.yaw);
  round.applyCategory(0, held);
  feedback = [{}, {}];
  if (command.moveIndex >= 0) {
    const r = round.applyCategory(0, MOVE_TO_CATEGORY[command.moveIndex]);
    feedback[0] = {attempted: 1, accepted: r.applied ? 1 : 0, rejected: r.applied ? 0 : 1};
  }
  round.stepBot();
  for (let i = 0; i < 2; i++) round.integrate(i, round.command(i));
  round.collide(); round.strikes_(); round.resolveCount();
  round.t += round.phys.dt; tick++;
  if (round.t >= round.phys.roundSeconds - 1e-9 && !round.count) round.over = true;
}
fresh();
console.log(JSON.stringify({event: 'ready', runtimeBackend: 'surrogate-fake'}));
readline.createInterface({input: process.stdin}).on('line', line => {
  const c = JSON.parse(line); let reply = {id: c.id, ok: true};
  try {
    if (c.op === 'reset') { fresh(); tick = 0; }
    else if (c.op === 'step') {
      const rounds = [];
      for (let k = 0; k < (c.steps || 1); k++) {
        step(k ? {...c.command, moveIndex: -1} : c.command, c.humanSide || 0);
        if (round.over) { const s = state(); completed++; s.completedRounds = completed; rounds.push(s); fresh(); break; }
      }
      reply.rounds = rounds;
    } else if (c.op !== 'snapshot') throw new Error('Unknown worker operation');
    reply.state = state();
  } catch (e) { reply = {id: c.id, ok: false, error: e.message}; }
  process.stdout.write(JSON.stringify(reply) + '\n');
});
