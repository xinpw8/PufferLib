'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
function read(p) { return fs.readFileSync(p, 'utf8'); }
function summarize(stage) {
  const rows = read(path.join(stage, 'sweep/arms.tsv')).trim().split(/\r?\n/).slice(1);
  return rows.map(row => {
    const [arm, reward, criticScale, entropy, learningRate, steps] = row.split('\t');
    const dir = path.join(stage, 'sweep', arm);
    assert.equal(Number(read(path.join(dir, 'exit-code.txt'))), 0);
    const stdout = read(path.join(dir, 'stdout.txt')).replace(/\x1b\[[0-9;]*[A-Za-z]/g, '');
    const uptime = [...stdout.matchAll(/Uptime\s+([^│\r\n]+)/g)].at(-1)?.[1];
    assert.ok(uptime, `Missing final native uptime: ${arm}`);
    const factors = {ms: .001, s: 1, m: 60, h: 3600, d: 86400};
    const loopSeconds = [...uptime.matchAll(/(\d+)\s*(ms|s|m|h|d)/g)]
      .reduce((sum, m) => sum + Number(m[1]) * factors[m[2]], 0);
    assert.ok(loopSeconds > 0);
    const elapsed = read(path.join(dir, 'process-timing.txt'))
      .match(/Elapsed \(wall clock\) time .*?:\s*([\d:.]+)\s*$/m)?.[1];
    assert.ok(elapsed);
    const wallSeconds = elapsed.split(':').map(Number).reduce((sum, value) => sum * 60 + value, 0);
    const rounds = JSON.parse(read(path.join(dir, 'round-summary.json')));
    assert.equal(rounds.failure_bits, 0);
    const evalDir = path.join(stage, 'frozen-screen', arm);
    assert.equal(Number(read(path.join(evalDir, 'exit-code.txt'))), 0);
    const matches = read(path.join(evalDir, 'matches.private.jsonl')).trim().split(/\r?\n/).map(JSON.parse);
    const result = {wins: 0, losses: 0, draws: 0, points: 0, opponentPoints: 0, ownFalls: 0, opponentFalls: 0};
    for (const match of matches) {
      assert.equal(match.diagnostic, false);
      assert.equal(match.opponent_controller, 'recovered_bot1_v1');
      assert.equal(match.seed, 10001);
      assert.equal(match.duration_ticks, 6000);
      const side = match.policy_side;
      assert.ok(side === 0 || side === 1);
      result[match.winner < 0 ? 'draws' : match.winner === side ? 'wins' : 'losses']++;
      result.points += match.score[side]; result.opponentPoints += match.score[side ^ 1];
      result.ownFalls += match.falls[side]; result.opponentFalls += match.falls[side ^ 1];
    }
    assert.equal(matches.length, 512);
    assert.equal(new Set(matches.map(m => `${m.policy_side}/${m.arena}/${m.episode}`)).size, 512);
    const checkpointPath = path.join(dir, 'checkpoints/rek_native5', arm, '0000000008388608.bin');
    const checkpointSha256 = crypto.createHash('sha256').update(fs.readFileSync(checkpointPath)).digest('hex');
    assert.ok(matches.every(m => m.policy_sha256 === checkpointSha256));
    return {arm, reward, criticScale: Number(criticScale), entropy: Number(entropy), learningRate: Number(learningRate),
      steps: Number(steps), checkpointSha256,
      nativeFinalUptimeSeconds: loopSeconds, nativeUptimeResolutionSeconds: .001,
      approximateTrainingSps: Number(steps) / loopSeconds,
      processWallSeconds: wallSeconds, startupInclusiveTrainingSps: Number(steps) / wallSeconds,
      trainingRounds: rounds,
      frozen: {...result, rounds: matches.length, winFraction: result.wins / matches.length,
        meanPointMargin: (result.points - result.opponentPoints) / matches.length,
        meanCanonicalUndiscountedReward: (result.points - result.opponentPoints - result.ownFalls) / (100 * matches.length)},
      note: 'Frozen simulation screen only. Compact model has no physical fall producer. Common canonical undiscounted reward is computed from terminal points, not discounted PPO return. Final INI history can predate training completion; training SPS uses final native console uptime (1 ms display resolution), with separate complete-process timing.'};
  }).sort((a, b) => b.frozen.winFraction - a.frozen.winFraction || b.frozen.meanPointMargin - a.frozen.meanPointMargin);
}
if (require.main === module) {
  assert.equal(process.argv.length, 3, 'Usage: node summarize_normalized_sweep.cjs STAGE');
  process.stdout.write(JSON.stringify({schema: 'rek.normalized_sweep.screen.v1',
    selection: 'frozen simulation win fraction, then mean point margin; authentic validation remains required',
    results: summarize(process.argv[2])}, null, 2) + '\n');
}
module.exports = {summarize};
