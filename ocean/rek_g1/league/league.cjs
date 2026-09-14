'use strict';

// Private, dependency-free tournament bookkeeping. Simulation and inference
// remain in the native executables; this module never implements either.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const SCHEMA = 1;
const TERMINAL = new Set(['completed', 'forfeit', 'engine_error']);
const REASONS = {
  completed: new Set(['score_limit', 'round_limit', 'time_limit']),
  forfeit: new Set(['policy_crash', 'policy_timeout', 'policy_disconnect', 'policy_quit', 'illegal_action']),
  engine_error: new Set(['physics_nonfinite', 'engine_crash', 'runner_interrupted', 'infrastructure_failure', 'round_redo']),
};

function requireThat(test, message) {
  if (!test) throw new Error(message);
}
function identifier(value, name) {
  requireThat(typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(value), `Invalid ${name}`);
  return value;
}
function text(value, name) {
  requireThat(typeof value === 'string' && value.length > 0 && value.length <= 1024, `Invalid ${name}`);
  return value;
}
function sha(value, name) {
  requireThat(typeof value === 'string' && /^[a-f0-9]{64}$/.test(value), `Invalid ${name}`);
  return value;
}
function number(value, name, min = 0) {
  requireThat(Number.isFinite(value) && value >= min, `Invalid ${name}`);
  return value;
}
function integer(value, name, min = 0) {
  requireThat(Number.isSafeInteger(value) && value >= min, `Invalid ${name}`);
  return value;
}
function copy(value) { return JSON.parse(JSON.stringify(value)); }
function policyKey(backend, id) { return `${backend}/${id}`; }
function blank() { return {schema: SCHEMA, revision: 0, policies: {}, pairs: {}, matches: {}}; }
function hashFile(filename) {
  const hash = crypto.createHash('sha256');
  const fd = fs.openSync(filename, 'r');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  try {
    for (;;) {
      const count = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (!count) break;
      hash.update(buffer.subarray(0, count));
    }
  } finally { fs.closeSync(fd); }
  return hash.digest('hex');
}

// Binomial 95% Wilson interval for wins per valid game. Draws are non-wins.
function winInterval(wins, games) {
  if (!games) return [0, 1];
  const z = 1.959963984540054;
  const p = wins / games;
  const denominator = 1 + z * z / games;
  const center = (p + z * z / (2 * games)) / denominator;
  const half = z * Math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}
function counter() { return {games: 0, wins: 0, draws: 0, losses: 0, forfeits: 0, pointsFor: 0, pointsAgainst: 0}; }
function addResult(target, match, side) {
  target.games++;
  target.pointsFor += match.scores[side];
  target.pointsAgainst += match.scores[1 - side];
  const own = match.players[side];
  if (match.winnerPolicyId === own) target.wins++;
  else if (match.winnerPolicyId === null) target.draws++;
  else target.losses++;
  if (match.status === 'forfeit' && match.loserPolicyId === own) target.forfeits++;
}
function summarize(target) {
  return {...target, winRate: target.games ? target.wins / target.games : null,
    winRate95: winInterval(target.wins, target.games),
    scoreRate: target.games ? (target.wins + 0.5 * target.draws) / target.games : null};
}

class League {
  constructor({file}) {
    requireThat(path.isAbsolute(file), 'League file must be an absolute private runtime path');
    this.file = file;
    this.lock = `${file}.lock`;
  }

  _read() {
    if (!fs.existsSync(this.file)) return blank();
    const value = JSON.parse(fs.readFileSync(this.file, 'utf8'));
    requireThat(value.schema === SCHEMA && value.policies && value.pairs && value.matches, 'Unsupported or corrupt league manifest');
    return value;
  }

  _change(callback) {
    fs.mkdirSync(path.dirname(this.file), {recursive: true});
    let fd;
    try { fd = fs.openSync(this.lock, 'wx', 0o600); }
    catch (error) {
      if (error.code === 'EEXIST') throw new Error(`League writer already active: ${this.lock}`);
      throw error;
    }
    const temporary = `${this.file}.${process.pid}.${crypto.randomUUID()}.tmp`;
    try {
      fs.writeFileSync(fd, JSON.stringify({pid: process.pid, startedAt: new Date().toISOString()}));
      const state = this._read();
      const result = callback(state);
      state.revision++;
      const out = fs.openSync(temporary, 'wx', 0o600);
      try {
        fs.writeFileSync(out, `${JSON.stringify(state, null, 2)}\n`);
        fs.fsyncSync(out);
      } finally { fs.closeSync(out); }
      fs.renameSync(temporary, this.file);
      return copy(result);
    } finally {
      if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
      fs.closeSync(fd);
      fs.unlinkSync(this.lock);
    }
  }

  snapshot() { return copy(this._read()); }

  registerPolicy(spec) {
    const backend = identifier(spec.backend, 'backend');
    const id = identifier(spec.id, 'policy ID');
    const configHash = sha(spec.configHash, 'environment configHash');
    requireThat(spec.kind === 'scripted' || spec.kind === 'trained', 'Policy kind must be scripted or trained');
    const entry = {backend, id, label: text(spec.label, 'label'), configHash, kind: spec.kind,
      createdAt: new Date().toISOString(), checkpoint: null};
    if (spec.kind === 'trained') {
      const checkpoint = spec.checkpoint;
      requireThat(checkpoint && path.isAbsolute(checkpoint.path), 'Trained checkpoint requires an absolute path');
      requireThat(fs.statSync(checkpoint.path).isFile(), 'Checkpoint must be a file');
      const expected = sha(checkpoint.sha256, 'checkpoint SHA256');
      requireThat(hashFile(checkpoint.path) === expected, 'Checkpoint SHA256 mismatch');
      const format = text(checkpoint.format, 'explicit checkpoint format');
      requireThat(checkpoint.model && typeof checkpoint.model === 'object' && !Array.isArray(checkpoint.model), 'Checkpoint requires model architecture metadata');
      entry.checkpoint = {path: checkpoint.path, sha256: expected, format,
        trainingSteps: integer(checkpoint.trainingSteps, 'training steps'), model: copy(checkpoint.model)};
    } else {
      requireThat(!spec.checkpoint, 'Scripted opponents cannot claim a trained checkpoint');
      entry.scriptedVersion = text(spec.scriptedVersion, 'scripted implementation version');
    }
    return this._change(state => {
      const key = policyKey(backend, id);
      requireThat(!state.policies[key], `Policy already registered: ${key}; checkpoints are immutable, use a new ID`);
      state.policies[key] = entry;
      return entry;
    });
  }

  verifyCheckpoint({backend, id}) {
    const entry = this._read().policies[policyKey(backend, id)];
    requireThat(entry, 'Unknown policy');
    if (entry.kind === 'scripted') return true;
    requireThat(hashFile(entry.checkpoint.path) === entry.checkpoint.sha256, 'Checkpoint changed since registration');
    return true;
  }

  schedulePair(spec) {
    const id = identifier(spec.id, 'pair ID');
    const backend = identifier(spec.backend, 'backend');
    const configHash = sha(spec.configHash, 'environment configHash');
    const policyA = identifier(spec.policyA, 'policy A');
    const policyB = identifier(spec.policyB, 'policy B');
    const seed = integer(spec.seed, 'seed');
    const maxDurationMs = integer(spec.maxDurationMs, 'max duration milliseconds', 1);
    requireThat(policyA !== policyB, 'Tournament pair requires distinct policy IDs');
    return this._change(state => {
      requireThat(!Object.hasOwn(state.pairs, id), `Duplicate pair ID: ${id}`);
      for (const policy of [policyA, policyB]) {
        const entry = state.policies[policyKey(backend, policy)];
        requireThat(entry && entry.configHash === configHash, `Policy backend/config mismatch: ${policy}`);
      }
      const fixtureKey = JSON.stringify([backend, configHash, [policyA, policyB].sort(), seed]);
      requireThat(!Object.values(state.pairs).some(pair => pair.fixtureKey === fixtureKey), 'Duplicate backend/config/policies/seed fixture');
      const pair = {id, backend, configHash, policyA, policyB, seed, maxDurationMs, fixtureKey,
        createdAt: new Date().toISOString(), matchIds: [`${id}.0`, `${id}.1`]};
      state.pairs[id] = pair;
      pair.matchIds.forEach((matchId, leg) => {
        state.matches[matchId] = {id: matchId, pairId: id, leg, backend, configHash, seed,
          players: leg ? [policyB, policyA] : [policyA, policyB],
          status: 'scheduled', startedAt: null, finishedAt: null, maxDurationMs};
      });
      return {pair, matches: pair.matchIds.map(matchId => state.matches[matchId])};
    });
  }

  startMatch(id) {
    return this._change(state => {
      const match = state.matches[id];
      requireThat(match && match.status === 'scheduled', 'Only a scheduled match can start');
      match.status = 'running';
      match.startedAt = new Date().toISOString();
      return match;
    });
  }

  finishMatch(result) {
    requireThat(TERMINAL.has(result.status), 'Result status must be completed, forfeit or engine_error');
    requireThat(REASONS[result.status].has(result.reason), `Invalid ${result.status} reason`);
    requireThat(result.status === 'engine_error' && result.scores === null || Array.isArray(result.scores) && result.scores.length === 2, 'Scores must be [left, right], or null for an unobserved invalid trial');
    if (result.scores !== null) result.scores.forEach(score => number(score, 'score'));
    if (!(result.status === 'engine_error' && result.durationMs === null)) number(result.durationMs, 'duration milliseconds');
    return this._change(state => {
      const match = state.matches[result.id];
      requireThat(match && match.status === 'running', 'Only a running match can finish; duplicate outcomes rejected');
      // Runners echo identity to prevent feeding another backend or fixture here.
      requireThat(result.backend === match.backend && result.configHash === match.configHash, 'Result backend/config mismatch');
      requireThat(result.seed === match.seed && JSON.stringify(result.players) === JSON.stringify(match.players), 'Result seed/side assignment mismatch');
      let winnerPolicyId = null;
      let loserPolicyId = null;
      if (result.status === 'forfeit') {
        requireThat(match.players.includes(result.loserPolicyId), 'Forfeit must identify the responsible policy');
        loserPolicyId = result.loserPolicyId;
        winnerPolicyId = match.players.find(player => player !== loserPolicyId);
      } else if (result.status === 'completed') {
        requireThat(result.durationMs <= match.maxDurationMs, 'Completed match exceeded its simulated time limit');
        if (result.scores[0] !== result.scores[1]) winnerPolicyId = match.players[result.scores[0] > result.scores[1] ? 0 : 1];
        if (result.roundResult !== undefined) {
          requireThat([1, 2, 3].includes(result.roundResult), 'Only points, KO or tie is an adjudicated native round');
          requireThat(result.roundResult === 3 ? result.winnerSide === -1 : [0, 1].includes(result.winnerSide), 'Invalid native winner side');
          const nativeWinner = result.winnerSide === -1 ? null : match.players[result.winnerSide];
          requireThat(result.winnerPolicyId === undefined || result.winnerPolicyId === nativeWinner, 'Native winner policy/side mismatch');
          if (result.roundResult === 1) requireThat(nativeWinner === winnerPolicyId && nativeWinner !== null, 'Native points winner disagrees with scores');
          winnerPolicyId = nativeWinner;
        }
      }
      Object.assign(match, {status: result.status, scores: result.scores === null ? null : [...result.scores], durationMs: result.durationMs,
        reason: result.reason, winnerPolicyId, loserPolicyId, finishedAt: new Date().toISOString()});
      if (result.status === 'completed' && result.roundResult !== undefined) {
        match.roundResult = result.roundResult;
        match.winnerSide = result.winnerSide;
      }
      if (result.note !== undefined) match.note = text(result.note, 'result note');
      return match;
    });
  }

  standings({backend, configHash}) {
    const state = this._read();
    const entries = Object.values(state.policies).filter(policy => policy.backend === backend && policy.configHash === configHash);
    const rows = new Map(entries.map(policy => [policy.id, {...copy(policy), total: counter(), paired: counter(),
      headToHead: {}, pending: 0, invalidTrials: 0, leftGames: 0, rightGames: 0}]));
    for (const match of Object.values(state.matches)) {
      if (match.backend !== backend || match.configHash !== configHash) continue;
      const pair = state.pairs[match.pairId];
      const validPair = pair.matchIds.every(id => ['completed', 'forfeit'].includes(state.matches[id].status));
      match.players.forEach((id, side) => {
        const row = rows.get(id);
        if (!TERMINAL.has(match.status)) { row.pending++; return; }
        if (match.status === 'engine_error') { row.invalidTrials++; return; }
        addResult(row.total, match, side);
        if (side === 0) row.leftGames++; else row.rightGames++;
        const other = match.players[1 - side];
        if (!Object.hasOwn(row.headToHead, other)) row.headToHead[other] = counter();
        addResult(row.headToHead[other], match, side);
        if (validPair) addResult(row.paired, match, side);
      });
    }
    const result = [...rows.values()].map(row => ({...row, total: summarize(row.total), paired: summarize(row.paired),
      headToHead: Object.fromEntries(Object.entries(row.headToHead).map(([id, counts]) => [id, summarize(counts)]))}));
    // Rank only complete, side-reversed valid pairs. All single-leg outcomes,
    // including forfeits, remain visible in total and head-to-head records.
    result.sort((a, b) => b.paired.winRate95[0] - a.paired.winRate95[0]
      || (b.paired.scoreRate ?? -1) - (a.paired.scoreRate ?? -1)
      || b.paired.games - a.paired.games || a.id.localeCompare(b.id));
    let rank = 0;
    return result.map(row => ({...row, rank: row.paired.games ? ++rank : null,
      strengthStatus: row.paired.games >= 20 ? 'measured' : row.total.games ? 'provisional' : 'untested',
      rankingMethod: '95% Wilson lower bound on win rate over completed side-reversed pairs'}));
  }

  opponentOptions(query) {
    return this.standings(query).map(row => {
      const counts = row.paired.games ? row.paired : row.total;
      const rate = counts.winRate === null ? 'no matches' : `${(100 * counts.winRate).toFixed(1)}% wins, ${counts.wins}W/${counts.draws}D/${counts.losses}L`;
      const rank = row.rank === null ? 'unranked' : `#${row.rank}`;
      return {backend: row.backend, configHash: row.configHash, id: row.id, kind: row.kind,
        label: `${row.label} | ${rank} | ${row.strengthStatus} | ${rate}`, rank: row.rank,
        strengthStatus: row.strengthStatus, stats: row.total, pairedStats: row.paired,
        invalidTrials: row.invalidTrials, pending: row.pending,
        checkpoint: row.checkpoint && {sha256: row.checkpoint.sha256, format: row.checkpoint.format,
          trainingSteps: row.checkpoint.trainingSteps, model: row.checkpoint.model}};
    });
  }
}

module.exports = {League, hashFile, winInterval};
