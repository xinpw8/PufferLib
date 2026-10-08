'use strict';
// Planar surrogate of a REK G1 round for developing and pre-tuning Claudiolo.
//
// NOT a parity model. What is exact: Sparring Bot 1's decision rules (bot1.cjs,
// ported from the recovered client code), the round/scoring rules
// (g1_fight_state.c: 120 s rounds, +1 hand, +2 kick, +5 to the opponent of any
// fallen G1 after a 3 s no-recovery count, then both reset to spawn), move
// durations and impact windows (moves.cjs), spawn points (+/-0.9 m) and the
// 33-category keyboard action contract. What is approximate and explicitly
// parameterized: locomotion speeds and lag, strike reach/cone/acceptance,
// body collision, and fall hazards. calibrate() fits the hazards so a
// neutral-input defender reproduces the authentic passive-defender cohort
// (8 rounds vs Sparring Bot 1: 60:91 points, about 10 Bot strike points and
// 1.5 Bot falls per round). Everything that matters is re-measured in the
// native clone and in rek.exe before any claim.

const bot1 = require('./bot1.cjs');
const {MOVES, HELD, CATEGORY_TO_MOVE} = require('./moves.cjs');

const DEFAULT_PHYS = Object.freeze({
  dt: 0.02,
  roundSeconds: 120,
  vForward: 0.55, vBackward: 0.42, vStrafe: 0.35, // m/s at unit command
  yawPerUnit: 1.0,                                 // rad/s per unit yaw command
  keyboardYawSpeed: 1.0, keyboardYawRamp: 0.5,     // recovered keyboard ramp contract
  tauLinear: 0.25, tauYaw: 0.12,                   // first-order velocity lag, s
  settleSpeed: 0.05,                               // planar speed below which a move may start
  bodyRadius: 0.17,                                // collision radius per robot, m
  handReach: 0.68, handMin: 0.42, handCone: 32,    // root-to-root m / degrees (human scored 0.42-0.68)
  kickReach: 0.78, kickMin: 0.40, kickCone: 38,
  handRecoil: 0.30, kickRecoil: 0.45,              // target velocity impulse on contact, m/s
  // calibrate.cjs v1 fit to the authentic passive-defender cohort (loss 0.157):
  handAccept: 0.448, kickAccept: 0.27,             // speed/zone gate pass probability
  pushSelf: 1.76, pushOther: 0.30,                 // fall hazard per (m/s drive into contact) per s
  kickSelfFall: 0.20, kickKnockdown: 0.06,         // per kick contact
  handKnockdown: 0.004,
  runPunchSpeed: 0.45,                             // root drive during run_and_punch, m/s
  countSeconds: 3.0, graceSeconds: 2.0,
  obsDelay: 0.04, actDelay: 0.04,                  // one-way delays, s
  spawnHalf: 0.9,
});

const wrap = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a <= -Math.PI) a += 2 * Math.PI; return a; };

class Fighter {
  constructor(side, phys) {
    this.side = side; this.phys = phys; this.spawn(); this.cooldown = {left: -9, right: -9, leftLeg: -9, rightLeg: -9};
  }
  spawn() {
    const h = this.phys.spawnHalf;
    this.x = this.side === 0 ? -h : h; this.y = 0; this.yaw = this.side === 0 ? 0 : Math.PI;
    this.vx = 0; this.vy = 0; this.wz = 0;
    this.cmd = {forward: 0, strafe: 0, yaw: 0}; this.yawRamp = 0; this.yawSign = 0;
    this.move = null; this.recoverUntil = -1; this.down = false; this.downAt = -1;
    this.limbSpeed = 0; this.legSpeed = 0;
  }
  planarSpeed() { return Math.hypot(this.vx, this.vy); }
  busy(t) { return this.move !== null || t < this.recoverUntil; }
}

// A tiny, explicit event log helps tests and diagnostics.
class Round {
  constructor({phys = {}, seed = 1, opponent = 'bot1', opponentPolicy = null} = {}) {
    this.phys = {...DEFAULT_PHYS, ...phys};
    this.f = [new Fighter(0, this.phys), new Fighter(1, this.phys)];
    this.t = 0; this.points = [0, 0]; this.falls = [0, 0]; this.strikes = [0, 0];
    this.rng = new bot1.XorShift32(seed * 2654435761 >>> 0);
    this.opponent = opponent; this.opponentPolicy = opponentPolicy;
    this.bot = bot1.newState(seed); this.botMove = -1;
    this.count = null; this.graceUntil = this.phys.graceSeconds; this.over = false;
    this.events = []; this.actQueue = []; this.obsQueue = [];
    this.held = [1, 1]; this.limbs = bot1.PRIMARY_LIMB;
  }
  random() { return this.rng.value(); }
  log(type, detail) { this.events.push({t: +this.t.toFixed(3), type, ...detail}); }

  geometry(i) {
    const a = this.f[i], b = this.f[1 - i];
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
    const bearing = wrap(Math.atan2(dy, dx) - a.yaw); // + = other robot to my LEFT
    return {d, bearing, angleDegrees: bot1.angleToOpponentDegrees(a, b)};
  }

  mask(i) {
    const a = this.f[i], m = Array(33).fill(false); m[0] = m[1] = true;
    if (this.over || a.down || this.count) return m;
    m[6] = m[7] = true;
    if (a.busy(this.t)) return m;
    for (let c = 2; c < 16; c++) m[c] = true;
    const h = HELD[this.held[i]] || HELD[1];
    const translating = h.forward !== 0 || h.strafe !== 0;
    if (!translating && a.planarSpeed() < this.phys.settleSpeed) for (let c = 16; c < 33; c++) m[c] = true;
    return m;
  }

  // Observation as the adapters will produce it (right-handed z-up, metres, radians).
  observe(i) {
    const me = this.f[i], op = this.f[1 - i];
    const pack = r => ({x: r.x, y: r.y, yaw: r.yaw, vx: r.vx, vy: r.vy, wz: r.wz, down: r.down,
      limbSpeed: r.limbSpeed, legSpeed: r.legSpeed});
    return {t: this.t, round: {active: !this.over && !this.count, timeRemaining: Math.max(0, this.phys.roundSeconds - this.t),
      duration: this.phys.roundSeconds, number: 1},
      points: [this.points[i], this.points[1 - i]], falls: [this.falls[i], this.falls[1 - i]],
      me: pack(me), opp: pack(op), mask: this.mask(i), moveBusy: me.busy(this.t)};
  }

  applyCategory(i, category) {
    const a = this.f[i];
    if (!Number.isInteger(category) || category < 0 || category > 32) throw new Error(`bad category ${category}`);
    const m = this.mask(i);
    if (!m[category]) return {applied: false, reason: 'masked'};
    if (category === 0) return {applied: true};
    if (category < 16) { this.held[i] = category; return {applied: true}; }
    const index = CATEGORY_TO_MOVE[category - 16];
    this.startMove(a, index);
    return {applied: true, move: index};
  }

  startMove(a, index) {
    a.move = {index, t0: this.t, scored: new Set()};
    a.vx *= 0.5; a.vy *= 0.5;
    this.log('move', {side: a.side, move: MOVES[index].name});
  }

  // Normalized command for fighter i this tick (held keys or Bot 1 locomotion).
  command(i) {
    const a = this.f[i];
    if (a.down || a.move || this.t < a.recoverUntil || this.count) return {forward: 0, strafe: 0, yaw: 0};
    if (i === 1 && this.opponent === 'bot1') return this.botCommand;
    const h = HELD[this.held[i]] || HELD[1];
    // Keyboard yaw ramp (recovered contract): resets on release/sign change.
    let yaw = 0;
    if (h.yaw === 0) { a.yawRamp = 0; a.yawSign = 0; }
    else {
      if (a.yawSign !== h.yaw) { a.yawRamp = 0; a.yawSign = h.yaw; }
      a.yawRamp = Math.min(1, a.yawRamp + this.phys.dt / this.phys.keyboardYawRamp);
      yaw = a.yawRamp * h.yaw * this.phys.keyboardYawSpeed;
    }
    return {forward: h.forward, strafe: h.strafe, yaw};
  }

  stepBot() {
    const b = this.f[1], g = this.geometry(1);
    const input = {distance: g.d, angleDegrees: g.angleDegrees, deltaSeconds: this.phys.dt, timeSeconds: this.t,
      roundElapsed: this.t, punching: b.move !== null, opponentDown: this.f[0].down || !!this.count,
      ownRecovery: b.down, roundActive: !this.over && !this.count};
    const out = bot1.update(this.bot, input, this.limbs);
    if (out.move >= 0) {
      const accepted = !b.down && !b.busy(this.t) && b.planarSpeed() < this.phys.settleSpeed;
      bot1.attackResult(this.bot, input, accepted);
      if (accepted) this.startMove(b, out.move); else this.log('bot_move_rejected', {move: out.move});
    }
    if (out.clearPunching && b.move) { b.move = null; b.recoverUntil = this.t; }
    this.botCommand = bot1.locomotion(this.bot, input);
  }

  integrate(i, c) {
    const a = this.f[i], p = this.phys;
    const fwd = c.forward >= 0 ? c.forward * p.vForward : c.forward * p.vBackward;
    const str = c.strafe * p.vStrafe;
    let tx = Math.cos(a.yaw) * fwd - Math.sin(a.yaw) * str, ty = Math.sin(a.yaw) * fwd + Math.cos(a.yaw) * str;
    if (a.move && MOVES[a.move.index].name === 'run_and_punch') {
      const age = this.t - a.move.t0;
      if (age > 0.5 && age < 1.9) { tx += Math.cos(a.yaw) * p.runPunchSpeed; ty += Math.sin(a.yaw) * p.runPunchSpeed; }
    }
    const kl = 1 - Math.exp(-p.dt / p.tauLinear), ky = 1 - Math.exp(-p.dt / p.tauYaw);
    a.tx = tx; a.ty = ty;
    if (a.down) { a.vx *= 0.8; a.vy *= 0.8; a.wz = 0; a.tx = a.ty = 0; return; }
    a.vx += (tx - a.vx) * kl; a.vy += (ty - a.vy) * kl;
    a.wz += (c.yaw * p.yawPerUnit - a.wz) * ky;
    a.x += a.vx * p.dt; a.y += a.vy * p.dt; a.yaw = wrap(a.yaw + a.wz * p.dt);
    // Octagon floor approximated by a 2.15 m radius wall.
    const r = Math.hypot(a.x, a.y); if (r > 2.15) { a.x *= 2.15 / r; a.y *= 2.15 / r; a.vx *= 0.3; a.vy *= 0.3; }
  }

  collide() {
    const [a, b] = this.f, p = this.phys;
    if (a.down || b.down || this.count) return;
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1e-6, min = 2 * p.bodyRadius;
    if (d >= min + 0.02) return;
    const nx = dx / d, ny = dy / d;
    const va = a.vx * nx + a.vy * ny, vb = -(b.vx * nx + b.vy * ny); // each one's speed toward the other
    if (d < min) { const push = (min - d) / 2; a.x -= nx * push; a.y -= ny * push; b.x += nx * push; b.y += ny * push; }
    // Cancel approach along the normal.
    if (va > 0) { a.vx -= va * nx; a.vy -= va * ny; }
    if (vb > 0) { b.vx += vb * nx; b.vy += vb * ny; }
    if (this.t < this.graceUntil) return;
    // Pushing effort = commanded drive toward the other body (velocity is
    // cancelled by the contact, the controller keeps driving into it).
    const ea = Math.max(0, (a.tx || 0) * nx + (a.ty || 0) * ny), eb = Math.max(0, -((b.tx || 0) * nx + (b.ty || 0) * ny));
    if (ea + eb < 0.02) return;
    const ha = (ea * p.pushSelf + eb * p.pushOther) * p.dt, hb = (eb * p.pushSelf + ea * p.pushOther) * p.dt;
    if (this.random() < ha) this.fall(0, ea >= eb ? 'push' : 'pushed');
    if (this.random() < hb) this.fall(1, eb >= ea ? 'push' : 'pushed');
  }

  strikes_() {
    const p = this.phys;
    for (let i = 0; i < 2; i++) {
      const a = this.f[i], b = this.f[1 - i];
      a.limbSpeed = 0.2; a.legSpeed = Math.min(1.2, a.planarSpeed() * 1.5);
      if (!a.move) continue;
      const mv = MOVES[a.move.index], age = this.t - a.move.t0;
      if (age >= mv.duration) { a.move = null; a.recoverUntil = this.t + (i === 1 && this.opponent === 'bot1' ? 0 : 0.1); continue; }
      a.limbSpeed = mv.kick ? 0.6 : 1.4; if (mv.kick) a.legSpeed = 1.4;
      if (a.down || b.down || this.count) continue;
      mv.impacts.forEach((e, k) => {
        if (age < e.t - 0.724 * e.lead || age > e.t + 0.724 * Math.max(e.release, 0.02)) return;
        if (e.kick) a.legSpeed = 3.5; else a.limbSpeed = 3.5;
        if (a.move.scored.has(k)) return;
        const limbKey = e.kick ? (e.side === 'left' ? 'leftLeg' : 'rightLeg') : e.side;
        if (this.t - a.cooldown[limbKey] < 0.3) return;
        const g = this.geometry(i), deg = Math.abs(g.bearing) * 180 / Math.PI;
        const reach = e.kick ? p.kickReach : p.handReach, min = e.kick ? p.kickMin : p.handMin;
        const cone = e.kick ? p.kickCone : p.handCone;
        if (g.d > reach || g.d < min || deg > cone) return;
        a.move.scored.add(k);
        const nx = (b.x - a.x) / g.d, ny = (b.y - a.y) / g.d, imp = e.kick ? p.kickRecoil : p.handRecoil;
        b.vx += nx * imp; b.vy += ny * imp; a.vx -= nx * imp * 0.2; a.vy -= ny * imp * 0.2;
        if (this.random() > (e.kick ? p.kickAccept : p.handAccept)) { this.log('contact_rejected', {side: i, move: mv.name}); return; }
        a.cooldown[limbKey] = this.t;
        const pts = e.kick ? 2 : 1; this.points[i] += pts; this.strikes[i] += pts;
        this.log('hit', {side: i, move: mv.name, points: pts, d: +g.d.toFixed(3)});
        if (this.t >= this.graceUntil) {
          if (e.kick && this.random() < p.kickSelfFall) this.fall(i, 'kick_self');
          if (this.random() < (e.kick ? p.kickKnockdown : p.handKnockdown)) this.fall(1 - i, 'knockdown');
        }
      });
    }
  }

  fall(i, cause) {
    const a = this.f[i]; if (a.down || this.over) return;
    a.down = true; a.downAt = this.t; a.move = null; this.falls[i]++;
    this.log('fall', {side: i, cause});
    if (!this.count) this.count = {until: this.t + this.phys.countSeconds, who: [i]};
    else if (!this.count.who.includes(i)) { this.count.who.push(i); this.count.until = this.t + this.phys.countSeconds; }
  }

  resolveCount() {
    if (!this.count || this.t < this.count.until) return;
    const who = this.count.who;
    if (who.length === 2) { this.points[0] += 5; this.points[1] += 5; }
    else this.points[1 - who[0]] += 5;
    this.log('count_expired', {fallen: who});
    this.count = null; this.f.forEach(f => f.spawn()); this.held = [1, 1];
    this.graceUntil = this.t + this.phys.graceSeconds;
    bot1.engage(this.bot, {distance: 1.8, angleDegrees: 0});
  }

  // controller(observation) -> category; called every tick for fighter 0.
  run(controller, {maxSeconds = this.phys.roundSeconds} = {}) {
    const p = this.phys, steps = Math.round(maxSeconds / p.dt);
    const obsLag = Math.round(p.obsDelay / p.dt), actLag = Math.max(0, Math.round(p.actDelay / p.dt));
    for (let n = 0; n < steps && !this.over; n++) {
      this.obsQueue.push(this.observe(0));
      const obs = this.obsQueue.length > obsLag ? this.obsQueue.shift() : null;
      if (obs) this.actQueue.push(controller(obs));
      if (this.actQueue.length > actLag) {
        const category = this.actQueue.shift();
        const result = this.applyCategory(0, category);
        if (controller.onResult) controller.onResult(category, result, this.t);
      }
      if (this.opponent === 'bot1') this.stepBot();
      else if (this.opponentPolicy) {
        const c = this.opponentPolicy(this.observe(1));
        this.applyCategory(1, c);
      }
      for (let i = 0; i < 2; i++) this.integrate(i, this.command(i));
      this.collide();
      this.strikes_();
      this.resolveCount();
      this.t += p.dt;
      if (this.t >= p.roundSeconds - 1e-9 && !this.count) this.over = true;
    }
    const [a, b] = this.points;
    return {points: [a, b], falls: [...this.falls], strikePoints: [...this.strikes],
      result: a > b ? 'win' : a < b ? 'loss' : 'draw', botAttempts: this.bot.attempts,
      botAccepted: this.bot.accepted, botRejected: this.bot.rejected, events: this.events};
  }
}

// Baseline controllers for calibration and comparison.
const passive = () => 1;

// GPT-like proxy: faces the opponent, closes distance, spams right hooks with
// sampled noise (matches the measured 447-right-hook move preference).
function gptLike(seed = 7) {
  const rng = new bot1.XorShift32(seed);
  return obs => {
    const dx = obs.opp.x - obs.me.x, dy = obs.opp.y - obs.me.y, d = Math.hypot(dx, dy);
    const bearing = wrap(Math.atan2(dy, dx) - obs.me.yaw);
    const r = rng.value();
    if (obs.mask[23] && d < 0.75 && Math.abs(bearing) < 0.35 && r < 0.25) return 23;
    if (Math.abs(bearing) > 0.25) return bearing > 0 ? 6 : 7;
    if (d > 0.62) return r < 0.85 ? 2 : 1;
    return r < 0.1 ? 3 : 1;
  };
}

function summarize(results) {
  const n = results.length, s = (f) => results.reduce((a, r) => a + f(r), 0);
  return {rounds: n, wins: s(r => r.result === 'win'), draws: s(r => r.result === 'draw'), losses: s(r => r.result === 'loss'),
    pointsFor: s(r => r.points[0]), pointsAgainst: s(r => r.points[1]),
    strikeFor: s(r => r.strikePoints[0]), strikeAgainst: s(r => r.strikePoints[1]),
    fallsOwn: s(r => r.falls[0]), fallsOpp: s(r => r.falls[1])};
}

function playRounds(makeController, {rounds = 20, seed0 = 1, phys = {}, opponent = 'bot1', makeOpponent = null} = {}) {
  const out = [];
  for (let k = 0; k < rounds; k++) {
    const round = new Round({phys, seed: seed0 + k, opponent, opponentPolicy: makeOpponent ? makeOpponent(seed0 + k) : null});
    out.push(round.run(makeController(seed0 + k)));
  }
  return out;
}

module.exports = {DEFAULT_PHYS, Round, Fighter, passive, gptLike, summarize, playRounds, wrap};
