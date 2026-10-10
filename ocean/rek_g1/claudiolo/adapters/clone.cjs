#!/usr/bin/env node
'use strict';
// Claudiolo <-> native REK clone worker (rek-eval-worker JSONL protocol).
// Claudiolo drives the worker's human side with keyboard-equivalent commands
// (forward/strafe/yaw in {-1,0,1} plus one move edge), exactly what the
// browser keyboard sends. The other side is the worker's internal opponent
// (recovered Bot 1 when configured) or a GPT checkpoint loaded with op:"policy".
//
//   node adapters/clone.cjs --worker BIN --config WORKER.json --env ENV.json --rounds 20 --out DIR
//        [--policy CKPT --sha256 HEX [--sampled]] [--params P.json] [--human-side 0]
//        [--max-ticks N]
// WORKER.json / ENV.json: the native clone's tools/worker-template.json and tools/env.json
// (REK_PHYSICAL_OPPONENT=recovered_bot1_g1_v1, 120 s rounds, match mode).
//
// Writes DIR/rounds.jsonl (one line per completed round) and DIR/summary.json.

const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const {spawn} = require('node:child_process');
const {controller} = require('../core.cjs');
const {HELD, CATEGORY_TO_MOVE} = require('../moves.cjs');

const DT = 0.02;
// Controller-order joint groups inside a 86-value native entity (q at 13+j, dq at 42+j).
const DQ = 42, LEFT_ARM = [15, 16, 17, 18], RIGHT_ARM = [22, 23, 24, 25], LEFT_LEG = [0, 3], RIGHT_LEG = [6, 9];
const ARM_LEVER = [0.45, 0.35, 0.15, 0.25], LEG_LEVER = [0.8, 0.4];

function yawFromQuat(w, x, y, z) { return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)); }
function tiltFromQuat(w, x, y, z) { // angle between body z and world z
  const zz = 1 - 2 * (x * x + y * y); return Math.acos(Math.max(-1, Math.min(1, zz)));
}
function limbSpeeds(entity) {
  if (!entity || entity.length < 71) return {limb: 0, leg: 0};
  const arm = idx => idx.reduce((a, j, k) => a + Math.abs(entity[DQ + j]) * ARM_LEVER[k], 0);
  const leg = idx => idx.reduce((a, j, k) => a + Math.abs(entity[DQ + j]) * LEG_LEVER[k], 0);
  return {limb: Math.max(arm(LEFT_ARM), arm(RIGHT_ARM)), leg: Math.max(leg(LEFT_LEG), leg(RIGHT_LEG))};
}

// Build a Claudiolo Frame for `side` from a worker state and the previous one.
function frameFromState(state, prev, side, roundKey) {
  const q = state.qpos, other = 1 - side;
  const root = s => { const o = s * 36; return {x: q[o], y: q[o + 1], z: q[o + 2], w: q[o + 3], qx: q[o + 4], qy: q[o + 5], qz: q[o + 6]}; };
  const raw = state.raw || [], mine = raw.slice(side * 223, side * 223 + 223);
  const pack = (s, entity) => {
    const r = root(s), yaw = yawFromQuat(r.w, r.qx, r.qy, r.qz);
    let vx = 0, vy = 0, wz = 0;
    if (prev) {
      const p = root.call(null, s), pq = prev.qpos, o = s * 36;
      vx = (r.x - pq[o]) / DT; vy = (r.y - pq[o + 1]) / DT;
      let dyaw = yaw - yawFromQuat(pq[o + 3], pq[o + 4], pq[o + 5], pq[o + 6]);
      while (dyaw > Math.PI) dyaw -= 2 * Math.PI; while (dyaw <= -Math.PI) dyaw += 2 * Math.PI;
      wz = dyaw / DT; void p;
    }
    const tilt = tiltFromQuat(r.w, r.qx, r.qy, r.qz), ls = limbSpeeds(entity);
    return {x: r.x, y: r.y, yaw, vx, vy, wz, z: r.z, tilt, down: r.z < 0.55 || tilt > 0.9, limbSpeed: ls.limb, legSpeed: ls.leg};
  };
  const mask = state.mask ? state.mask.slice(side * 33, side * 33 + 33).map(Boolean) : null;
  return {
    t: state.tick * DT,
    round: {active: state.phase === 2 && !state.terminal, timeRemaining: state.timeRemaining, duration: null, number: roundKey},
    points: [state.score[side], state.score[other]], falls: [state.falls[side], state.falls[other]],
    me: pack(side, mine.slice(0, 86)), opp: pack(other, mine.slice(86, 172)), mask,
  };
}

function commandFor(held, category) {
  const h = HELD[held] || HELD[1];
  const moveIndex = category >= 16 ? CATEGORY_TO_MOVE[category - 16] : -1;
  return {forward: h.forward, strafe: h.strafe, yaw: h.yaw, moveIndex, cancelAction: false};
}

class Worker {
  constructor(bin, config, env = {}, args = null) {
    this.child = spawn(bin, args || ['--config', config], {stdio: ['pipe', 'pipe', 'inherit'], env: {...process.env, ...env}});
    this.pending = new Map(); this.id = 0; this.ready = null;
    const lines = readline.createInterface({input: this.child.stdout});
    this.readyPromise = new Promise((resolve, reject) => { this.ready = {resolve, reject}; });
    lines.on('line', raw => {
      let m; try { m = JSON.parse(raw); } catch { return; }
      if (m.event === 'ready') { this.ready.resolve(m); return; }
      if (m.event === 'fatal') { this.ready.reject(new Error(m.error)); for (const p of this.pending.values()) p.reject(new Error(m.error)); return; }
      const p = this.pending.get(m.id); if (!p) return; this.pending.delete(m.id);
      if (m.ok === false) p.reject(new Error(m.error)); else p.resolve(m);
    });
    this.child.on('exit', code => { for (const p of this.pending.values()) p.reject(new Error(`worker exited ${code}`)); });
  }
  request(body) {
    const id = ++this.id;
    return new Promise((resolve, reject) => { this.pending.set(id, {resolve, reject}); this.child.stdin.write(JSON.stringify({id, ...body}) + '\n'); });
  }
  close() { this.child.stdin.end(); }
}

async function play({bin, config, env, args = null, rounds = 10, out, params = {}, humanSide = 0, policy = null, maxTicks = 200000, log = () => {}}) {
  fs.mkdirSync(out, {recursive: true});
  const roundsFile = fs.createWriteStream(path.join(out, 'rounds.jsonl'), {flags: 'a'});
  const w = new Worker(bin, config, env, args);
  const ready = await w.readyPromise; log('ready', ready);
  if (policy) {
    // Either a full native policy command (league registry fields) or a bare checkpoint.
    const command = policy.command ? {...policy.command, side: 1 - humanSide} : {side: 1 - humanSide, checkpoint: policy.path,
      sha256: policy.sha256, deterministic: !policy.sampled, hiddenSize: 256, layers: 2, precision: policy.precision || 'bf16'};
    const r = await w.request({op: 'policy', ...command});
    log('policy_loaded', {sha256: r.sha256, side: command.side});
  }
  let reply = await w.request({op: 'reset'});
  let state = reply.state, prev = null, held = 1, ctl = controller(params), done = 0, ticks = 0;
  const results = [];
  while (done < rounds && ticks < maxTicks) {
    const key = `${state.completedRounds}`;
    const frame = frameFromState(state, prev, humanSide, key);
    const category = ctl(frame);
    if (category >= 1 && category < 16) held = category;
    const command = commandFor(held, category);
    prev = state;
    reply = await w.request({op: 'step', humanSide, steps: 1, command});
    state = reply.state; ticks++;
    if (category >= 16) {
      const fb = (state.commandResults || [])[humanSide] || {};
      ctl.onResult(category, {applied: fb.accepted === 1 || fb.accepted === true});
    }
    for (const r of reply.rounds || []) {
      const me = humanSide, them = 1 - humanSide;
      const result = {round: done + 1, completedRounds: r.completedRounds, points: [r.score[me], r.score[them]],
        falls: [r.falls[me], r.falls[them]], roundResult: r.roundResult, winnerSide: r.winner,
        outcome: r.winner === me ? 'win' : r.winner === them ? 'loss' : 'draw', ticks,
        reasons: ctl.claudiolo.stats.reasons, strikes: ctl.claudiolo.stats.strikes, governor: ctl.claudiolo.gov.audit};
      roundsFile.write(JSON.stringify(result) + '\n'); results.push(result); done++;
      log('round', result);
      ctl = controller(params); held = 1; prev = null;
    }
    if (state.failure) throw new Error(`worker failure: ${state.failure}`);
  }
  w.close(); await new Promise(resolve => roundsFile.end(resolve));
  const summary = {rounds: results.length, wins: results.filter(r => r.outcome === 'win').length,
    losses: results.filter(r => r.outcome === 'loss').length, draws: results.filter(r => r.outcome === 'draw').length,
    pointsFor: results.reduce((a, r) => a + r.points[0], 0), pointsAgainst: results.reduce((a, r) => a + r.points[1], 0),
    fallsOwn: results.reduce((a, r) => a + r.falls[0], 0), fallsOpp: results.reduce((a, r) => a + r.falls[1], 0),
    humanSide, policy: policy ? {sha256: policy.sha256 || policy.command?.sha256, sampled: policy.command ? policy.command.deterministic === false : !!policy.sampled} : null, params};
  fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  return summary;
}

function parseArgs(argv) {
  const a = {}; for (let i = 0; i < argv.length; i++) { const k = argv[i]; if (!k.startsWith('--')) continue;
    const key = k.slice(2), v = argv[i + 1]; if (v === undefined || v.startsWith('--')) a[key] = true; else { a[key] = v; i++; } }
  return a;
}

if (require.main === module) {
  const a = parseArgs(process.argv.slice(2));
  if (!a.worker || !a.config || !a.out) { console.error('usage: clone.cjs --worker BIN --config JSON --out DIR [--rounds N] [--policy CKPT --sha256 HEX] [--params JSON]'); process.exit(2); }
  const params = a.params ? JSON.parse(fs.readFileSync(a.params, 'utf8')) : {};
  // --env takes the clone's tools/env.json (physics backend, recovered Bot 1 opponent, match mode).
  const env = a.env ? JSON.parse(fs.readFileSync(a.env, 'utf8')) : {};
  play({bin: a.worker, config: a.config, env, out: a.out, rounds: +(a.rounds || 10), params, humanSide: +(a['human-side'] || 0),
    policy: a.policy ? {path: a.policy, sha256: a.sha256, sampled: !!a.sampled} : null, maxTicks: +(a['max-ticks'] || 2e6),
    log: (e, d) => console.log(JSON.stringify({event: e, ...d}))})
    .then(s => console.log(JSON.stringify({event: 'summary', ...s})))
    .catch(e => { console.error(e.stack || e.message); process.exitCode = 1; });
}

module.exports = {frameFromState, commandFor, limbSpeeds, play, Worker};
