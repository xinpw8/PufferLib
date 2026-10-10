'use strict';
// Claudiolo: a hand-built REK G1 fighter that plays through the keyboard
// channel only. Pure decision logic; adapters turn game telemetry into a
// Frame and Claudiolo's category (0..32) into the game's input.
//
// Frame (right-handed z-up, metres, radians, seconds):
//   {t, round:{active,timeRemaining,duration,number}, points:[me,opp], falls:[me,opp],
//    me:{x,y,yaw,vx,vy,wz,down,limbSpeed?,legSpeed?}, opp:{...same}, mask:[33]|null,
//    moveBusy?:bool}
// yaw is the heading of the robot's facing axis (G1 root-local +X).

const bot1 = require('./bot1.cjs');
const {MOVES, byName, heldCategory} = require('./moves.cjs');

const DEFAULTS = Object.freeze({
  // Perception
  stillSpeed: 0.09, stillYawRate: 0.4, limbActive: 1.2, legActive: 1.2, limbQuiet: 0.8,
  limbRel: 3.0, legRel: 2.2, baselineSeconds: 2.0, strikeStillSpeed: 0.2, lungeRange: 1.0, lungeSpeed: 99, // lunge detection off until measured in the clone
  // Geometry
  strikeMin: 0.44, strikeMax: 0.66, aimTol: 0.22, faceTol: 0.09, faceRelease: 0.035,
  tooClose: 0.42, holdDist: 0.80, standMin: 0.60, standMax: 0.76,
  oppHandReach: 0.70, oppKickReach: 0.80, evadeLead: 0.22, evadeDepth: 0, stepInTime: 0.7, strikeOnApproach: true,
  sideTargetDeg: 9, sideTolDeg: 5, sideEnable: true, sideMaxDist: 0.95,
  // Timing (seconds)
  latency: 0.06, settleTime: bot1.C.settleTime, preemptMargin: 0.04, minOppImpact: 0.15,
  oppRecovery: bot1.C.recoveryTime, ownRecovery: 0.12,
  // Tactics
  preempt: true, evade: true, punishLongMove: true, longMoveMinRemaining: 1.2,
  kiteLead: 6, kiteTime: 25, kiteDist: 1.15,
  idleApproachAfter: 2.0, learn: true, learnMinHits: 6, kickDanger: 1.45, handDanger: 0.45, kickMargin: 0.06,
  primary: 'double_uppercut', fast: 'left_jab', long: 'six_punch',
  // Governor (human-legal input)
  minChangeSeconds: 0.06, maxChangesPerSecond: 12, reactionDelay: 0,
});

const capped = d => Math.min(d, bot1.C.maxPunch);
const KICK_DURATIONS = Object.freeze(MOVES.filter(m => m.kick).map(m => capped(m.duration)).sort((a, b) => a - b));
const HAND_DURATIONS = Object.freeze(MOVES.filter(m => !m.kick).map(m => capped(m.duration)).sort((a, b) => a - b));
const translating = held => (held >= 2 && held <= 5) || held >= 8;
const wrap = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a <= -Math.PI) a += 2 * Math.PI; return a; };
const DEG = 180 / Math.PI;

function perceive(f) {
  const dx = f.opp.x - f.me.x, dy = f.opp.y - f.me.y, d = Math.hypot(dx, dy) || 1e-6;
  const ux = dx / d, uy = dy / d;
  const myBearing = wrap(Math.atan2(dy, dx) - f.me.yaw);           // + = opponent to my left
  const oppBearing = wrap(Math.atan2(-dy, -dx) - f.opp.yaw);       // + = I am to its left
  const botAngle = -oppBearing * DEG;                               // Bot convention: + = its right
  const rvx = f.opp.vx - f.me.vx, rvy = f.opp.vy - f.me.vy;
  const closing = -(rvx * ux + rvy * uy);                           // + = gap shrinking
  const oppSpeed = Math.hypot(f.opp.vx, f.opp.vy), mySpeed = Math.hypot(f.me.vx, f.me.vy);
  const oppApproach = f.opp.vx * -ux + f.opp.vy * -uy;              // opp velocity toward me
  return {d, ux, uy, myBearing, oppBearing, botAngle, closing, oppSpeed, mySpeed, oppApproach,
    oppYawRate: Math.abs(f.opp.wz || 0)};
}

// Visible-behaviour tracker for Sparring Bot 1 (and a generic threat model
// for learned opponents). It never reads hidden state: only poses, velocities
// and limb motion that the screen shows.
class OpponentTracker {
  constructor(p) { this.p = p; this.reset(); }
  reset() {
    this.limbBase = null; this.legBase = null;
    this.still = false; this.stillSince = null; this.strikeSince = null; this.strikeKick = false; this.strikeLunge = false;
    this.lastActive = null; this.quietSince = null; this.lastApproachT = null; this.phase = 'unknown';
  }
  update(f, P) {
    const p = this.p, t = f.t;
    const limb = f.opp.limbSpeed ?? 0, leg = f.opp.legSpeed ?? 0;
    // Self-calibrating thresholds: limb-speed units differ between the clone
    // (joint-velocity proxy) and rek.exe (bone motion), and walking moves the
    // legs. A strike must stand out from the opponent's own recent baseline.
    const limbThreshold = Math.max(p.limbActive, p.limbRel * (this.limbBase ?? 0));
    const legThreshold = Math.max(p.legActive, p.legRel * (this.legBase ?? 0));
    // Canned moves run with zero locomotion, so a walking robot's swinging
    // legs (or arms) are not a strike; this keeps walk onsets from reading as kicks.
    const rooted = P.oppSpeed < p.strikeStillSpeed;
    // Run-and-punch is the one canned move that drives the root: an arm swing
    // while closing fast inside striking distance is a lunge, not a walk.
    const lunge = P.d < p.lungeRange && P.oppApproach > p.lungeSpeed && limb > limbThreshold;
    const limbOn = (rooted || lunge) && limb > limbThreshold, legOn = rooted && leg > legThreshold, active = limbOn || legOn;
    if (!active && this.strikeSince === null) {
      const k = Math.min(1, (f.dt ?? 0.02) / p.baselineSeconds);
      this.limbBase = this.limbBase === null ? limb : this.limbBase + k * (limb - this.limbBase);
      this.legBase = this.legBase === null ? leg : this.legBase + k * (leg - this.legBase);
    }
    const still = P.oppSpeed < p.stillSpeed && P.oppYawRate < p.stillYawRate;
    if (still && !this.still) this.stillSince = t;
    if (!still) this.stillSince = null;
    this.still = still;
    if (P.oppApproach > 0.05) this.lastApproachT = t;
    if (active) {
      if (this.strikeSince === null) { this.strikeSince = t; this.strikeKick = legOn && (!limbOn || leg / legThreshold >= limb / limbThreshold); this.strikeLunge = lunge && !rooted; }
      this.lastActive = t; this.quietSince = null;
    } else if (this.strikeSince !== null) {
      if (this.quietSince === null) this.quietSince = t;
      if (t - this.quietSince > 0.35 && limb < Math.max(p.limbQuiet, 0.7 * limbThreshold) && leg < Math.max(p.limbQuiet, 0.7 * legThreshold)) { this.strikeSince = null; this.strikeKick = false; }
    }
    const inCone = Math.abs(P.botAngle) < bot1.C.facing, near = P.d <= bot1.C.stop + 0.3 + 0.04;
    if (this.strikeSince !== null) this.phase = 'striking';
    else if (still && (near || inCone) && P.d < 1.0) this.phase = 'settling';
    else if (P.oppApproach > 0.05) this.phase = 'approaching';
    else this.phase = still ? 'idle' : 'moving';
    return this;
  }
  // Seconds until Bot 1's next swing may start (0 if it may already be swinging).
  timeToSwing(t) {
    if (this.phase === 'striking') return 0;
    if (this.phase !== 'settling' || this.stillSince === null) return Infinity;
    return Math.max(0, this.stillSince + this.p.settleTime - t);
  }
  // Is the opponent's strike still in its hitting phase?
  dangerNow(t) {
    if (this.strikeSince === null) return false;
    const age = t - this.strikeSince;
    if (this.strikeKick) return age < this.p.kickDanger;
    return age < this.p.handDanger || (this.lastActive !== null && t - this.lastActive < 0.12);
  }
  // Conservative remaining commitment: the shortest move in the matching pool
  // (kicks or hand moves, 50 Hz table, attack phase capped at 3 s) that is
  // still consistent with how long the strike has already lasted.
  committedFor(t) {
    if (this.strikeSince === null) return 0;
    const age = t - this.strikeSince, pool = this.strikeKick ? KICK_DURATIONS : HAND_DURATIONS;
    const left = pool.filter(d => d > age).reduce((m, d) => Math.min(m, d - age), Infinity);
    return (Number.isFinite(left) ? left : 0.1) + this.p.oppRecovery;
  }
}

// Online spacing: both robots are the same G1 with the same moves, so every
// scored strike (ours or theirs) is evidence of the real reach. Claudiolo
// starts from its priors and shifts its ranges toward what actually scores.
class SpacingLearner {
  constructor(p) { this.p = p; this.hits = []; this.attempts = []; this.history = []; this.last = null; }
  observe(f, P, pendingMove) {
    const t = f.t;
    this.history.push({t, d: P.d}); while (this.history.length && this.history[0].t < t - 1.5) this.history.shift();
    if (this.last && f.points) {
      const dMe = f.points[0] - this.last[0], dOpp = f.points[1] - this.last[1];
      const closest = (w) => this.history.filter(h => h.t >= t - w).reduce((m, h) => Math.min(m, h.d), Infinity);
      if (dOpp > 0 && dOpp % 5 !== 0) this.hits.push({who: 'opp', d: closest(0.3), t, kick: dOpp % 2 === 0});
      if (dMe > 0 && dMe % 5 !== 0) {
        this.hits.push({who: 'me', d: closest(0.3), t});
        if (pendingMove) pendingMove.scored = true;
      }
    }
    this.last = f.points ? [...f.points] : null;
  }
  reach(kick = false) {
    const hs = this.hits.filter(h => kick ? h.kick : !h.kick);
    if (hs.length < (kick ? 2 : this.p.learnMinHits)) return null;
    const ds = hs.map(h => h.d).filter(Number.isFinite).sort((a, b) => a - b);
    return ds[Math.min(ds.length - 1, Math.floor(0.9 * ds.length))];
  }
}

// Human-legal input governor: keyboard vocabulary, tempo, one move in flight.
class Governor {
  constructor(p) { this.p = p; this.reset(); }
  reset() { this.held = 1; this.lastChange = -Infinity; this.changes = []; this.audit = {moves: 0, holds: 0, changes: 0, throttled: 0, masked: 0}; }
  cost(category) {
    if (category < 16) return 1;
    const keys = MOVES[require('./moves.cjs').CATEGORY_TO_MOVE[category - 16]].keys;
    return keys.includes('+') || keys.length === 2 ? 2 : 1; // Space chords and double taps are two presses
  }
  allowedNow(t, cost) {
    this.changes = this.changes.filter(x => t - x < 1);
    return t - this.lastChange >= this.p.minChangeSeconds && this.changes.length + cost <= this.p.maxChangesPerSecond;
  }
  emit(t, intent, mask) {
    if (intent.move !== undefined && intent.move !== null) {
      const c = byName(intent.move).category;
      if (mask && !mask[c]) { this.audit.masked++; }
      else if (translating(this.held)) { this.audit.masked++; } // translation still held
      else if (this.allowedNow(t, this.cost(c))) {
        this.record(t, this.cost(c)); this.audit.moves++; return c;
      } else this.audit.throttled++;
      return this.holdOrRelease(t, {translate: null, yaw: intent.yaw ?? 0}, mask);
    }
    return this.holdOrRelease(t, intent, mask);
  }
  holdOrRelease(t, intent, mask) {
    const f = intent.translate === 'W' ? 1 : intent.translate === 'S' ? -1 : 0;
    const s = intent.translate === 'A' ? 1 : intent.translate === 'D' ? -1 : 0;
    const want = heldCategory(f, s, intent.yaw || 0);
    if (want === this.held) { this.audit.holds++; return 0; }
    if (mask && !mask[want]) {
      // During a move only yaw/release are legal: fall back to the yaw part.
      const yawOnly = heldCategory(0, 0, intent.yaw || 0);
      if (yawOnly === this.held || !mask[yawOnly]) { this.audit.holds++; return 0; }
      if (!this.allowedNow(t, 1)) { this.audit.throttled++; return 0; }
      this.held = yawOnly; this.record(t, 1); return yawOnly;
    }
    if (!this.allowedNow(t, 1)) { this.audit.throttled++; return 0; }
    this.held = want; this.record(t, 1); return want;
  }
  record(t, cost) { this.lastChange = t; for (let k = 0; k < cost; k++) this.changes.push(t); this.audit.changes += cost; }
}

class Claudiolo {
  constructor(params = {}) {
    this.p = {...DEFAULTS, ...params};
    this.base = {...this.p}; this.learner = new SpacingLearner(this.p);
    this.tracker = new OpponentTracker(this.p); this.gov = new Governor(this.p);
    this.frames = []; this.lockUntil = -Infinity; this.lastMove = null; this.lastMoveT = -Infinity;
    this.roundKey = null; this.turning = 0; this.stats = {decisions: 0, strikes: {}, reasons: {}};
  }
  resetRound() { this.tracker.reset(); this.gov.reset(); this.lockUntil = -Infinity; this.turning = 0; this.frames = []; }

  delayed(frame) {
    if (!(this.p.reactionDelay > 0)) return frame;
    this.frames.push(frame);
    while (this.frames.length > 1 && this.frames[1].t <= frame.t - this.p.reactionDelay) this.frames.shift();
    const f = this.frames[0];
    return {...f, mask: frame.mask, t: frame.t};
  }

  faceYaw(P) {
    const a = Math.abs(P.myBearing);
    if (a > this.p.faceTol) this.turning = Math.sign(P.myBearing);
    else if (a < this.p.faceRelease) this.turning = 0;
    return this.turning; // +1 = Q (turn left), -1 = E
  }

  note(reason) { this.stats.reasons[reason] = (this.stats.reasons[reason] || 0) + 1; this.lastReason = reason; }

  // Returns a category 0..32.
  decide(frame) {
    this.stats.decisions++;
    const key = `${frame.round.number}`;
    if (key !== this.roundKey) { this.roundKey = key; this.resetRound(); }
    const f = this.delayed(frame), p = this.p, t = frame.t;
    f.dt = this.lastT === undefined ? 0.02 : Math.min(0.2, Math.max(0.001, t - this.lastT)); this.lastT = t;
    if (!f.round.active || f.me.down || f.opp.down) {
      this.tracker.reset(); this.note('inactive');
      return this.gov.emit(t, {translate: null, yaw: 0}, frame.mask);
    }
    const P = perceive(f), T = this.tracker.update(f, P);
    this.learner.observe(f, P, t < this.lockUntil ? this.pending : null);
    if (p.learn) this.adapt();
    const yaw = this.faceYaw(P);
    const lead = f.points[0] - f.points[1];

    // Own move in progress: only keep the desired facing.
    if (t < this.lockUntil || frame.moveBusy) { this.note('locked'); return this.gov.emit(t, {translate: null, yaw}, frame.mask); }

    // 1. Never get pushed over: break contact first.
    if (P.d < p.tooClose) { this.note('too_close'); return this.gov.emit(t, {translate: 'S', yaw}, frame.mask); }

    // 2. Ahead late: run the clock at a safe distance.
    if (lead >= p.kiteLead && f.round.timeRemaining <= p.kiteTime) {
      this.note('kite');
      if (P.d < p.kiteDist) return this.gov.emit(t, {translate: 'S', yaw}, frame.mask);
      return this.gov.emit(t, {translate: null, yaw}, frame.mask);
    }

    const attackLegal = !frame.mask || frame.mask[byName(p.primary).category];
    const aimed = Math.abs(P.myBearing) < p.aimTol;
    const inRange = P.d >= p.strikeMin && P.d <= p.strikeMax;
    const swingIn = T.timeToSwing(t);                    // Bot 1 swing start, from now
    const committed = T.committedFor(t);                 // opponent frozen at least this long
    const cone = Math.abs(P.botAngle);
    const theyReachHands = P.d < p.oppHandReach && cone < bot1.C.facing + 5;
    const theyReach = theyReachHands || (P.d < p.oppKickReach && cone < bot1.C.facing + 10);
    const strike = (move, why) => {
      if (this.translatingHeld()) { this.note('stop_to_strike'); return this.gov.emit(t, {translate: null, yaw}, frame.mask); }
      if (!attackLegal) { this.note('await_settle'); return this.gov.emit(t, {translate: null, yaw}, frame.mask); }
      const c = this.gov.emit(t, {move, yaw}, frame.mask);
      if (c >= 16) { this.onSent(move, t); this.note(`${why}_${move}`); }
      return c;
    };
    const go = (translate, why) => { this.note(why); return this.gov.emit(t, {translate, yaw}, frame.mask); };

    // 3. Punish: the opponent is locked in a canned move and cannot step or turn.
    // While its strike is still live and could reach us, get out first.
    if (committed > 0) {
      const live = T.dangerNow(t), reach = T.strikeKick ? p.oppKickReach + p.kickMargin : p.oppHandReach;
      // A lunge (run-and-punch) out-runs a backpedal: step off its line instead.
      if (live && T.strikeLunge && P.d < reach + 0.3) return go('D', 'sidestep_lunge');
      if (live && P.d < reach && cone < bot1.C.facing + 15) return go('S', T.strikeKick ? 'evade_kick' : 'evade_hands');
      if (inRange && aimed) return strike(committed >= p.longMoveMinRemaining && p.punishLongMove ? p.long : p.primary, 'punish');
      if (aimed && P.d > p.strikeMax && committed > p.stepInTime && !live) return go('W', 'step_in');
      if (P.d < p.strikeMin) return go('S', 'make_room');
      return go(null, 'watch_commit');
    }

    // 4. Pre-empt: Bot 1 has stopped in range and must wait out its settle delay.
    if (T.phase === 'settling') {
      if (inRange && aimed && p.preempt) {
        const theirImpact = swingIn + p.minOppImpact;
        const first = byName(p.primary).impacts[0].t + 2 * p.latency;
        const fast = byName(p.fast).impacts[0].t + 2 * p.latency;
        if (!theyReachHands || first + p.preemptMargin < theirImpact) return strike(p.primary, 'preempt');
        if (fast + p.preemptMargin < theirImpact) return strike(p.fast, 'preempt');
      }
      if (p.evade && theyReachHands && P.d < p.oppHandReach - p.evadeDepth && swingIn < p.evadeLead) return go('S', 'evade');
      return go(null, 'watch_settle');
    }

    // 5. It walks into our punch.
    if (T.phase === 'approaching') {
      if (inRange && aimed && p.strikeOnApproach) return strike(p.primary, 'meet');
      if (P.d < p.standMin) return go('S', 'keep_range');
      return go(null, 'wait');
    }

    // 6. Opponent idle or wandering (learned policies, Bot 1 repositioning).
    if (inRange && aimed) return strike(p.primary, 'free');
    if (p.sideEnable && P.d < p.sideMaxDist && !theyReach && Math.abs(P.myBearing) < 0.3) {
      const err = P.botAngle - p.sideTargetDeg;
      if (Math.abs(err) > p.sideTolDeg) return go(err < 0 ? 'A' : 'D', 'side');
    }
    const idleFor = T.lastApproachT === null ? Infinity : t - T.lastApproachT;
    if (P.d > p.standMax && (P.d > p.holdDist + 0.25 || idleFor > p.idleApproachAfter)) return go('W', 'approach');
    if (P.d < p.standMin) return go('S', 'keep_range');
    return go(null, 'hold');
  }

  translatingHeld() { return translating(this.gov.held); }

  adapt() {
    const q = this.learner.reach(); if (q === null) return;
    const b = this.base, p = this.p, clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
    // Shrink toward the prior: evidence weight n/(n+8) over the prior reach.
    const n = this.learner.hits.filter(h => !h.kick).length, w = n / (n + 8), prior = b.strikeMax + 0.01;
    const r = prior + w * (q - prior);
    p.strikeMax = clamp(r - 0.01, b.strikeMax - 0.08, b.strikeMax + 0.10);
    p.oppHandReach = clamp(r + 0.03, b.oppHandReach - 0.08, b.oppHandReach + 0.12);
    p.standMin = clamp(p.strikeMax - 0.06, b.standMin - 0.08, b.standMin + 0.10);
    p.standMax = clamp(r + 0.08, b.standMax - 0.06, b.standMax + 0.12);
    const k = this.learner.reach(true); if (k !== null) p.oppKickReach = clamp(k + 0.02, b.oppKickReach - 0.06, b.oppKickReach + 0.12);
  }

  onSent(move, t) {
    const m = byName(move);
    this.lockUntil = t + m.duration + this.p.ownRecovery + this.p.latency;
    this.lastMove = move; this.lastMoveT = t; this.pending = {move, t, scored: false};
    this.stats.strikes[move] = (this.stats.strikes[move] || 0) + 1;
  }

  // Adapter feedback: a move the game refused does not lock us.
  onResult(category, result) {
    if (category >= 16 && result && result.applied === false) this.lockUntil = -Infinity;
  }
}

function controller(params) {
  const c = new Claudiolo(params);
  const fn = obs => c.decide(obs);
  fn.onResult = (category, result) => c.onResult(category, result);
  fn.claudiolo = c;
  return fn;
}

module.exports = {DEFAULTS, translating, SpacingLearner, Claudiolo, OpponentTracker, Governor, perceive, controller, wrap};
