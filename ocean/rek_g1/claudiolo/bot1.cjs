'use strict';
// Exact JavaScript port of ../native5/native_bot1.cuh (recovered client-image
// Sparring Bot 1 high-level AI). Used two ways:
//   * surrogate.cjs drives its Bot 1 with it (the real decision rules);
//   * Claudiolo uses the constants and the deterministic parts as a "shadow"
//     model of what Bot 1 will do next from what is visible on screen.
// RNG is the candidate-private xorshift32 from the header; the live server's
// Unity RNG is unknown, so nothing here may assume a particular draw sequence.

const Phase = Object.freeze({Inactive: 0, Engaging: 1, Repositioning: 2, Settling: 3,
  Attacking: 4, Recovering: 5, GivingRoom: 6});
const PHASE_NAMES = ['Inactive', 'Engaging', 'Repositioning', 'Settling', 'Attacking', 'Recovering', 'GivingRoom'];

const f32 = Math.fround;
const C = Object.freeze({
  stop: f32(0.4180000126361847), facing: 35, initialDelay: f32(0.6600000262260437),
  maxEngage: f32(2.1500000953674316), minFootwork: f32(0.20000000298023224),
  settleTime: f32(0.30000001192092896), recoveryTime: f32(0.257999986410141), maxPunch: 3,
  forwardSpeed: f32(0.800000011920929), yawSpeed: 1.5, repositionChance: f32(0.05000000074505806),
  kickChance: 0.25, repositionMin: f32(0.35199999809265137), repositionMax: f32(1.4129999876022339),
  repositionBack: f32(0.15000000596046448), repositionStrafe: f32(0.4339999854564667),
});

// Primary limb per runtime move index 0..16 from the build strike catalog:
// 1 left hand, 2 right hand, 3 left leg, 4 right leg.
const PRIMARY_LIMB = Object.freeze([1, 1, 1, 2, 2, 1, 3, 3, 4, 4, 1, 2, 1, 1, 1, 1, 1]);

class XorShift32 {
  constructor(seed) { this.state = (seed >>> 0) || 0x6d2b79f5; }
  next() { let x = this.state; x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; this.state = x; return x; }
  value() { return (this.next() >>> 8) * (1 / 16777216); }
  integer(bound) {
    const b = bound >>> 0, threshold = ((-b >>> 0) % b) >>> 0;
    let x; do { x = this.next(); } while (x < threshold);
    return x % b;
  }
}

function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }

function newState(seed) {
  return {phase: Phase.Settling, timer: C.initialDelay, minTimer: 0,
    reposition: {forward: 0, strafe: 0, yaw: 0}, rng: new XorShift32(seed),
    attempts: 0, accepted: 0, rejected: 0, repositions: 0};
}

// Timers and geometry use binary32 arithmetic, as in the CUDA/C++ header.
function engage(s, i) {
  s.phase = Phase.Engaging; s.timer = C.maxEngage;
  s.minTimer = Math.abs(f32(i.angleDegrees)) < C.facing && f32(i.distance) <= f32(C.stop + f32(0.15)) ? 0 : C.minFootwork;
}

// RobotConfig.TryPickMove: reservoir draws over the assigned order.
function pickCategory(limbs, category, side, rng) {
  let all = -1, preferred = -1, nall = 0, npreferred = 0;
  for (let move = 0; move < 17; move++) {
    const limb = limbs[move]; if (!limb) continue;
    const cat = limb === 3 || limb === 4 ? 1 : 0;
    if (cat !== category) continue;
    if (rng.integer(++nall) === 0) all = move;
    const limbSide = limb === 1 || limb === 3 ? 0 : 1;
    if (limbSide === side && rng.integer(++npreferred) === 0) preferred = move;
  }
  return npreferred ? preferred : all;
}

function pickAttack(limbs, angleDegrees, rng) {
  const category = rng.value() < C.kickChance ? 1 : 0, side = angleDegrees >= 0 ? 1 : 0;
  const move = pickCategory(limbs, category, side, rng);
  return move < 0 ? pickCategory(limbs, category ^ 1, side, rng) : move;
}

function reposition(s) {
  const sign = s.rng.value() > 0.5 ? 1 : -1;
  const r = s.rng.value();
  s.reposition = {forward: -C.repositionBack, strafe: f32(-sign * C.repositionStrafe),
    yaw: f32(-f32(f32(-0.3) + f32(f32(0.6) * r)) * C.yawSpeed)};
  s.phase = Phase.Repositioning;
  s.timer = f32(C.repositionMin + f32(f32(C.repositionMax - C.repositionMin) * s.rng.value()));
  s.repositions++;
}

function attackResult(s, i, accepted) {
  if (accepted) { s.phase = Phase.Attacking; s.timer = C.maxPunch; s.accepted++; }
  else { s.rejected++; engage(s, i); }
}

// i = {distance, angleDegrees, deltaSeconds, timeSeconds, roundElapsed,
//      punching, opponentDown, ownRecovery, roundActive}
function update(s, i, limbs = PRIMARY_LIMB) {
  const out = {move: -1, clearPunching: false, unsupportedRecovery: false};
  const dt = f32(i.deltaSeconds), dist = f32(i.distance), ang = Math.abs(f32(i.angleDegrees));
  s.timer = f32(s.timer - dt);
  if (i.ownRecovery) { out.unsupportedRecovery = true; return out; }
  if (i.opponentDown && s.phase !== Phase.GivingRoom && s.phase !== Phase.Attacking) { s.phase = Phase.GivingRoom; s.timer = 1; }
  switch (s.phase) {
    case Phase.Engaging:
      s.minTimer = f32(s.minTimer - dt);
      if (s.minTimer <= 0 && ang < C.facing && dist <= f32(C.stop + f32(0.3))) { s.phase = Phase.Settling; s.timer = C.settleTime; }
      else if (s.timer <= 0) engage(s, i);
      break;
    case Phase.Repositioning: if (s.timer <= 0) engage(s, i); break;
    case Phase.Settling:
      if (s.timer > 0) { if (dist > f32(C.stop + f32(0.5)) && ang >= C.facing) engage(s, i); }
      else if (dist < f32(C.stop + f32(0.3)) || ang < C.facing) {
        if (!i.roundActive || f32(i.roundElapsed) < C.initialDelay) { engage(s, i); break; }
        out.move = pickAttack(limbs, i.angleDegrees, s.rng); s.attempts++;
        if (out.move < 0) attackResult(s, i, false);
      } else engage(s, i);
      break;
    case Phase.Attacking:
      if (!i.punching || s.timer <= 0) { out.clearPunching = true; s.phase = Phase.Recovering; s.timer = C.recoveryTime; }
      break;
    case Phase.Recovering:
      if (s.timer <= 0) { if (s.rng.value() < C.repositionChance) reposition(s); else engage(s, i); }
      break;
    case Phase.GivingRoom: if (i.opponentDown) s.timer = 1; else if (s.timer <= 0) engage(s, i); break;
    default: break;
  }
  return out;
}

function facingYaw(angle) {
  angle = f32(angle);
  if (Math.abs(angle) <= f32(C.facing * 0.5)) return 0;
  return f32(-(angle > 0 ? 1 : -1) * f32(clamp(f32(Math.abs(angle) / 45), 0, 1) * C.yawSpeed));
}

// Normalized command {forward, strafe, yaw}; same units the keyboard writes.
function locomotion(s, i) {
  if (i.ownRecovery) return {forward: 0, strafe: 0, yaw: 0};
  if (s.phase === Phase.Repositioning) return {...s.reposition};
  if (s.phase === Phase.Settling) return {forward: 0, strafe: 0, yaw: facingYaw(i.angleDegrees)};
  const dist = f32(i.distance), ang = Math.abs(f32(i.angleDegrees));
  if (s.phase === Phase.GivingRoom) return {forward: dist < 1.5 ? f32(-f32(clamp(f32(f32(1.5 - dist) * 2), 0, 1)) * f32(0.25)) : 0,
    strafe: 0, yaw: facingYaw(i.angleDegrees)};
  if (s.phase !== Phase.Engaging) return {forward: 0, strafe: 0, yaw: 0};
  const c = {forward: dist > C.stop ? f32(clamp(f32(f32(dist - C.stop) / f32(0.8)), 0, 1) * C.forwardSpeed) : 0,
    strafe: 0, yaw: facingYaw(i.angleDegrees)};
  if (ang < C.facing && dist <= f32(C.stop + f32(0.3))) {
    c.strafe = f32(-f32(Math.sin(f32(2 * f32(i.timeSeconds)))) * f32(0.15));
    if (dist > f32(C.stop * f32(0.8))) c.forward = f32(C.forwardSpeed * f32(0.3));
  }
  return c;
}

// AIOpponentController.AngleToOpponent convention (physical_bot1_geometry.h):
// positions in a right-handed z-up frame; positive = target to the robot's RIGHT.
function angleToOpponentDegrees(own, opp) {
  const dx = opp.x - own.x, dy = opp.y - own.y;
  const fx = Math.cos(own.yaw), fy = Math.sin(own.yaw);
  const targetSq = dx * dx + dy * dy;
  if (targetSq < 0.001) return 0;
  const dot = clamp((fx * dx + fy * dy) / Math.sqrt(targetSq), -1, 1);
  const cross = fy * dx - fx * dy;
  return Math.acos(dot) * 57.29578 * (cross >= 0 ? 1 : -1);
}

module.exports = {Phase, PHASE_NAMES, C, PRIMARY_LIMB, XorShift32, newState, engage, pickAttack,
  pickCategory, attackResult, update, locomotion, facingYaw, angleToOpponentDegrees, clamp};
