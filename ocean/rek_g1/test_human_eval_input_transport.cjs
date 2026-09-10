// Headless unit test of the shipped inline script. No browser or OS input.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, 'human_eval_server.py'), 'utf8');
const script = source.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const listeners = new Map();
const nodes = new Map();
const moves = [];
const requests = [];
let releaseRequest;
let delayInput = false;
const state = {
  input_sequence: 0, tick: 0, held: [], last_actions: [1, 1],
  fight: { player_score: 0, opponent_score: 0, time_remaining_seconds: 120 },
  last_move_disposition: 'none', pending_move_index: null,
};
function node(id) {
  if (!nodes.has(id)) nodes.set(id, { addEventListener() {} });
  return nodes.get(id);
}
const response = value => ({ ok: true, json: async () => value });
vm.runInNewContext(script, {
  document: { getElementById: node, querySelectorAll: () => [] },
  addEventListener: (type, listener) => listeners.set(type, listener),
  setTimeout() {},
  fetch: async (url, options) => {
    if (url === '/state') return response(state);
    assert.equal(url, '/input');
    const body = JSON.parse(options.body);
    requests.push(body);
    if (body.move_index !== null) moves.push(body.move_index);
    if (delayInput) await new Promise(resolve => { releaseRequest = resolve; });
    state.input_sequence = body.sequence;
    return response({ ok: true, accepted: true });
  },
});
const settle = () => new Promise(resolve => setImmediate(resolve));
const press = code => listeners.get('keydown')({
  code, repeat: false, shiftKey: false, preventDefault() {},
});
(async () => {
  await settle();
  assert.equal(requests.length, 1); // Initial held-state clear.
  delayInput = true;
  press('KeyU');
  press('KeyI');
  press('Digit0');
  await settle();
  assert.deepEqual(moves, [7]);
  delayInput = false;
  releaseRequest();
  await settle();
  assert.deepEqual(moves, [7]); // No deferred attack requests.
  press('KeyQ');
  press('KeyI');
  await settle();
  assert.deepEqual(moves, [7, 8]);
  assert.deepEqual(requests.at(-1).held, ['Q']);
  assert.equal(requests.at(-2).move_index, null);
  console.log('PASS: attacks do not queue in transport; held Q accompanies the next edge');
})().catch(error => { console.error(error); process.exitCode = 1; });
