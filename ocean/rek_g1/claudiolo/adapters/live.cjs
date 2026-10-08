#!/usr/bin/env node
'use strict';
// Claudiolo in the authentic REK client (rek.exe) through the existing
// RekUiBridgeAgent policy stream. It replaces the encoder + CUDA policy worker
// of ../../native5/live_transfer_run.cjs with Claudiolo, and reuses that
// file's private-room proofs, round handling and one-action-in-flight pacer.
// Input path = the bridge's 33 keyboard categories (held W/S/A/D/Q/E combos
// and single move edges); no global input, no quit/disconnect actions.
//
//   node adapters/live.cjs trial.json
// trial.json: {"relay":[...command...], "out":"/new/dir", "max_seconds":150,
//              "enter_private":true, "params":{...optional Claudiolo params...}}

const fs = require('node:fs');
const path = require('node:path');
const live = require('../../native5/live_transfer_run.cjs');
const {controller} = require('../core.cjs');
const {byCategory} = require('../moves.cjs');

const wrap = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a <= -Math.PI) a += 2 * Math.PI; return a; };

// Unity rotation of root-local +X, projected to the floor. Unity (x, y-up, z)
// maps to Claudiolo's right-handed z-up (X = x, Y = z); physical left/right is
// preserved, so counter-clockwise from above stays positive.
function unityHeading(q) {
  const [x, y, z, w] = q;
  const vx = 1 - 2 * (y * y + z * z), vz = 2 * (x * z - w * y);
  return Math.atan2(vz, vx);
}

const LIMB_PATTERNS = {
  leftHand: /left.*(wrist|hand|rubber)/i, rightHand: /right.*(wrist|hand|rubber)/i,
  leftFoot: /left.*(ankle|foot|toe)/i, rightFoot: /right.*(ankle|foot|toe)/i,
};

class LiveFramer {
  constructor() { this.prev = null; this.limbIndex = null; this.vel = [{vx: 0, vy: 0, wz: 0}, {vx: 0, vy: 0, wz: 0}]; }
  indexLimbs(names) {
    const idx = {};
    for (const [k, re] of Object.entries(LIMB_PATTERNS)) {
      const hits = names.map((n, i) => re.test(n) ? i : -1).filter(i => i >= 0);
      idx[k] = hits.length ? hits[hits.length - 1] : -1; // most distal match
    }
    return idx;
  }
  frame(src) {
    const local = src.local_slot, other = 1 - local;
    const time = Number(src.clock?.unity_time);
    const fighters = src.fighters;
    if (!this.limbIndex && fighters?.[0]?.bone_names) this.limbIndex = this.indexLimbs(fighters[0].bone_names);
    const dt = this.prev ? time - this.prev.time : 0;
    const pack = (slot, k) => {
      const f = fighters[slot], p = f.root_position_xyz, x = p[0], y = p[2], yaw = unityHeading(f.root_rotation_xyzw);
      let limb = 0, leg = 0;
      if (this.prev && dt > 0.004 && dt < 0.25) {
        const pf = this.prev.src.fighters[slot], pp = pf.root_position_xyz;
        const a = 0.5, v = this.vel[k];
        v.vx += a * ((x - pp[0]) / dt - v.vx); v.vy += a * ((y - pp[2]) / dt - v.vy);
        v.wz += a * (wrap(yaw - unityHeading(pf.root_rotation_xyzw)) / dt - v.wz);
        const rel = (i) => {
          if (i < 0 || !f.bone_world_positions_xyz || !pf.bone_world_positions_xyz) return 0;
          const b = f.bone_world_positions_xyz[i], pb = pf.bone_world_positions_xyz[i];
          const dx = (b[0] - p[0]) - (pb[0] - pp[0]), dy = (b[1] - p[1]) - (pb[1] - pp[1]), dz = (b[2] - p[2]) - (pb[2] - pp[2]);
          return Math.hypot(dx, dy, dz) / dt;
        };
        const L = this.limbIndex || {};
        limb = Math.max(rel(L.leftHand ?? -1), rel(L.rightHand ?? -1));
        leg = Math.max(rel(L.leftFoot ?? -1), rel(L.rightFoot ?? -1));
      }
      const v = this.vel[k];
      const down = !!(f.falling || f.fallen || f.resetting || f.motor_shutdown) || (Number.isFinite(f.tilt_angle) && f.tilt_angle > 50);
      return {x, y, yaw, vx: v.vx, vy: v.vy, wz: v.wz, down, limbSpeed: limb, legSpeed: leg};
    };
    const r = src.round || {};
    const out = {
      t: time, round: {active: src.phase === 1 && r.active === true, timeRemaining: r.time_remaining, duration: r.duration, number: r.number},
      points: [r.clean_hits?.[local] ?? 0, r.clean_hits?.[other] ?? 0], falls: [r.falls?.[local] ?? 0, r.falls?.[other] ?? 0],
      me: pack(local, 0), opp: pack(other, 1), mask: Array.isArray(src.action_mask) ? src.action_mask.map(Boolean) : null,
      moveBusy: src.input?.pending_move === true || src.input?.move_request_pending_transport === true,
    };
    this.prev = {time, src};
    return out;
  }
}

async function run(configPath) {
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  if (!(config.max_seconds > 0 && config.max_seconds <= 600)) throw new Error('max_seconds must be 0..600');
  fs.mkdirSync(config.out, {recursive: false});
  fs.writeFileSync(path.join(config.out, 'run-config.json'), JSON.stringify(config, null, 2) + '\n', {flag: 'wx'});
  const events = fs.createWriteStream(path.join(config.out, 'orchestrator.jsonl'), {flags: 'wx'});
  const decisions = fs.createWriteStream(path.join(config.out, 'decisions.jsonl'), {flags: 'wx'});
  const log = (event, detail = {}) => { const v = {utc: new Date().toISOString(), event, ...detail}; events.write(JSON.stringify(v) + '\n'); console.log(JSON.stringify(v)); };
  let relay, leased = false, streaming = false, stopping = false, stopReason = 'not_started', nextId = 0;
  let firstRound = null, lastRound = null, roundIdentity = null, localSlot = null, opponent = null, lastSourceAt = 0;
  let sources = 0, sent = 0, applied = 0, rejected = 0;
  const pacer = new live.LiveActionPacer(), framer = new LiveFramer(), ctl = controller(config.params || {});
  let doneResolve; const done = new Promise(r => { doneResolve = r; });
  const finish = reason => { if (!stopping) { stopping = true; stopReason = reason; doneResolve(); } };
  const request = (type, fields = {}, event = 'ack') => {
    const request_id = `claudiolo-${++nextId}`;
    return live.sendAndWait(relay, {type, request_id, ...fields}, x => x.event === event && x.request_id === request_id);
  };
  const command = async name => { const ack = await request('command', {command: name}); log('command_result', {command: name, status: ack.status, reason: ack.reason});
    if (ack.status !== 'accepted') throw new Error(`${name}: ${ack.reason}`); return ack; };
  let timer, watchdog;
  try {
    relay = live.childEndpoint('relay', config.relay, config.out);
    relay.bus.on('failure', e => finish(e.message)); relay.bus.on('exit', x => finish(`relay_exit:${JSON.stringify(x)}`));
    await relay.wait(x => x.event === 'hello', 30000);
    let state = await request('get_state', {}, 'state');
    await command('AcquireExclusiveControl'); leased = true;
    if (live.betweenPrivateRounds(state)) {
      const deadline = Date.now() + 30000;
      do { await new Promise(r => setTimeout(r, 100)); state = await request('get_state', {}, 'state');
        if (Date.now() > deadline) throw new Error('round transition timeout'); } while (live.betweenPrivateRounds(state));
    }
    if (live.canExitLostPrivateSession(state)) {
      if (config.enter_private !== true) throw new Error('lost private session requires enter_private');
      await command('ExitLostG1PolicySession');
      const deadline = Date.now() + 15000;
      do { await new Promise(r => setTimeout(r, 100)); state = await request('get_state', {}, 'state');
        if (Date.now() > deadline) throw new Error('lost session exit timeout'); } while (live.privateArena(state));
    }
    state = await live.ensurePrivateArena(state, {enterPrivate: config.enter_private, command, getState: () => request('get_state', {}, 'state'), log});
    if (!state.private_ai.policy_active_gameplay_proven || state.private_ai.round_active !== true) {
      const deadline = Date.now() + 30000; let asked = false;
      while (Date.now() < deadline) {
        if (live.canRequestPrivateRound(state) && !asked) { await command('StartG1PolicyRound'); asked = true; }
        state = await request('get_state', {}, 'state');
        if (state.private_ai.policy_active_gameplay_proven && state.private_ai.round_active) break;
        await new Promise(r => setTimeout(r, 100));
      }
    }
    if (!(live.privateArena(state) && state.private_ai.policy_active_gameplay_proven === true && state.private_ai.round_active === true))
      throw new Error('active private AI scope required');
    opponent = live.botIdentity(state.private_ai); log('active_private_ai_opponent', opponent);
    relay.bus.on('message', live.guardedCallback(source => {
      if (source.event === 'g1_policy_end') { log('stream_end', source); finish(`stream_end:${source.reason}`); return; }
      if (source.event === 'g1_policy_action') {
        if (!pacer.acknowledge(source)) return;
        source.applied ? applied++ : rejected++;
        ctl.onResult(source.action, {applied: source.applied});
        return;
      }
      if (source.event !== 'g1_policy_state' || stopping) return;
      live.validateBotIdentity(source.opponent, opponent);
      sources++; lastSourceAt = Date.now();
      if (firstRound === null) { firstRound = source.round; roundIdentity = source.round_identity_sha256; localSlot = source.local_slot; }
      if (source.round_identity_sha256 !== roundIdentity) { finish('round_identity_changed'); return; }
      if (source.round) lastRound = source.round;
      if (source.round && source.round.active === false && source.round.result_value) { finish('source_round_terminal'); return; }
      const frame = framer.frame(source);
      if (!streaming) return;
      if (pacer.offer(source, Date.now())) return;
      const category = ctl(frame);
      decisions.write(JSON.stringify({seq: source.observation_sequence, t: frame.t, category,
        name: category >= 16 ? byCategory(category).name : category, reason: ctl.claudiolo.lastReason,
        d: +Math.hypot(frame.opp.x - frame.me.x, frame.opp.y - frame.me.y).toFixed(3), points: frame.points}) + '\n');
      const request_id = `act-${++nextId}`;
      pacer.sent(request_id, category, Date.now()); sent++;
      relay.send({type: 'policy_action', request_id, round_identity_sha256: source.round_identity_sha256,
        observation_sequence: source.observation_sequence, action: category});
    }, e => finish(`relay_callback:${e.message}`)));
    streaming = true;
    await command('StartG1PolicyStreamAnyAi'); log('claudiolo_started', {params: ctl.claudiolo.p});
    timer = setTimeout(() => finish('requested_duration_complete'), config.max_seconds * 1000);
    watchdog = setInterval(() => {
      if (lastSourceAt && Date.now() - lastSourceAt > 2000) finish('source_stream_missing');
      if (pacer.pending?.sent != null && Date.now() - pacer.pending.sent > 2000) finish('action_ack_missing');
    }, 250);
    process.once('SIGINT', () => finish('operator_interrupt'));
    await done;
  } catch (e) { finish(e.message); log('error', {message: e.message}); }
  finally {
    clearTimeout(timer); clearInterval(watchdog); streaming = false;
    if (leased) { try { await command('StopG1PolicyStream'); } catch (e) { log('stop_error', {message: e.message}); }
      try { await command('ReleaseExclusiveControl'); } catch (e) { log('release_error', {message: e.message}); } }
    const summary = {controller: 'claudiolo', stop_reason: stopReason, sources, sent, applied, rejected, opponent,
      local_slot: localSlot, round_identity_sha256: roundIdentity, round_outcome: live.roundOutcome(lastRound, localSlot),
      initial_round: firstRound, final_round: lastRound, governor: ctl.claudiolo.gov.audit, reasons: ctl.claudiolo.stats.reasons,
      strikes: ctl.claudiolo.stats.strikes, authentic_client: true, global_input_emitted: false};
    fs.writeFileSync(path.join(config.out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', {flag: 'wx'});
    log('summary', summary); relay?.close(); events.end(); decisions.end();
    process.exitCode = ['win', 'loss', 'draw'].includes(summary.round_outcome) ? 0 : 2;
  }
}

if (require.main === module) {
  if (process.argv.length !== 3) { console.error('usage: node adapters/live.cjs trial.json'); process.exit(2); }
  run(process.argv[2]).catch(e => { console.error(e.message); process.exitCode = 2; });
}
module.exports = {LiveFramer, unityHeading, run};
