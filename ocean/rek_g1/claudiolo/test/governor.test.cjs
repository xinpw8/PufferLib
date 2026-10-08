'use strict';
// Human-legal input audit: every category Claudiolo emits over full surrogate
// rounds must be keyboard vocabulary, respect the tempo cap and the game's own
// gates (no strike while translation is held or the mask forbids it).
const test = require('node:test');
const assert = require('node:assert');
const S = require('../surrogate.cjs');
const K = require('../core.cjs');
const {heldCategory, HELD, byName} = require('../moves.cjs');

function audit(params, rounds = 6, opponent = 'bot1') {
  const violations = [];
  for (let k = 0; k < rounds; k++) {
    const ctl = K.controller(params), c = ctl.claudiolo;
    let held = 1, changes = [], lastChange = -Infinity;
    const wrapped = obs => {
      const category = ctl(obs);
      if (!Number.isInteger(category) || category < 0 || category > 32) violations.push({t: obs.t, why: 'range', category});
      if (category !== 0) {
        const cost = category >= 16 ? (byName(c.lastMove || 'left_jab').keys.length === 2 || (c.lastMove && byName(c.lastMove).keys.includes('+')) ? 2 : 1) : 1;
        if (obs.t - lastChange < params.minChangeSeconds - 1e-9) violations.push({t: obs.t, why: 'tempo', category});
        changes = changes.filter(x => obs.t - x < 1); for (let n = 0; n < cost; n++) changes.push(obs.t);
        if (changes.length > params.maxChangesPerSecond) violations.push({t: obs.t, why: 'rate', category, n: changes.length});
        lastChange = obs.t;
      }
      if (category >= 16) {
        if (!obs.mask[category]) violations.push({t: obs.t, why: 'masked_move', category});
        const h = HELD[held]; if (h.forward || h.strafe) violations.push({t: obs.t, why: 'move_while_translating', category});
      }
      if (category >= 1 && category < 16) held = category;
      return category;
    };
    wrapped.onResult = ctl.onResult;
    const opts = opponent === 'bot1' ? {seed: 40 + k} : {seed: 40 + k, opponent: 'policy', opponentPolicy: S.gptLike(k)};
    new S.Round(opts).run(wrapped);
  }
  return violations;
}

test('governor keeps every input human-legal (default tempo)', () => {
  const v = audit({...K.DEFAULTS});
  assert.deepStrictEqual(v.slice(0, 5), []);
});

test('governor keeps every input human-legal with a 150 ms reaction delay and slow tempo', () => {
  const v = audit({...K.DEFAULTS, reactionDelay: 0.15, minChangeSeconds: 0.1, maxChangesPerSecond: 8}, 4, 'policy');
  assert.deepStrictEqual(v.slice(0, 5), []);
});

test('held vocabulary has no diagonal translation', () => {
  assert.strictEqual(heldCategory(1, 1, 0), 2);  // W wins over A
  assert.strictEqual(heldCategory(-1, -1, -1), 11); // SE
  assert.strictEqual(heldCategory(0, 0, 0), 1);
});
