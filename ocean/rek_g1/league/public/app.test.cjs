'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Browser-independent DOM fixture for side selection and API payloads.
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.events = {}; this.attributes = {}; this._value = ''; }
  get options() { return this.children; }
  get cells() { return this.children; }
  get lastChild() { return this.children.at(-1); }
  get value() { return this._value; }
  set value(value) { this._value = String(value); }
  append(child) { this.children.push(child); if (this.tag === 'select' && this.children.length === 1) this._value = child.value; }
  replaceChildren() { this.children = []; this._value = ''; }
  addEventListener(name, callback) { this.events[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = value; }
  focus() {}
  async trigger(name) { return this.events[name]?.({preventDefault() {}}); }
}
async function setup(active = null, state = null, inputStatus = 200) {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  const elements = Object.fromEntries([...html.matchAll(/<([a-z][a-z0-9-]*)\b[^>]*\bid="([^"]+)"/g)].map(match => [match[2], new Element(match[1])]));
  const catalog = {backends: [{id: 'mujoco', label: 'MuJoCo', available: true}], policies: [
    {backend: 'mujoco', id: 'script', label: 'Scripted', kind: 'scripted', rank: null},
    {backend: 'mujoco', id: 'trained', label: 'Checkpoint 8192', kind: 'trained', rank: 1, wins: 4, losses: 0, draws: 0, winRate: 1},
  ], active};
  const calls = [];
  const timers = [];
  const fetch = async (url, options = {}) => {
    calls.push({url, body: options.body && JSON.parse(options.body)});
    if (url === '/api/input') return {ok: inputStatus === 200, status: inputStatus, json: async () => ({ok: inputStatus === 200})};
    if (url === '/api/select') catalog.active = JSON.parse(options.body);
    const value = url === '/api/state' ? state || {ok: true}
      : url === '/api/standings' ? [{backend: 'mujoco', standings: catalog.policies}]
      : url === '/api/catalog' || url === '/api/select' ? catalog : {ok: true};
    return {ok: true, json: async () => structuredClone(value)};
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8'), {
    document: {getElementById: id => elements[id], createElement: tag => new Element(tag),
      querySelectorAll: () => [], addEventListener() {}, hidden: false},
    window: {addEventListener() {}}, fetch, AbortSignal, setTimeout(fn, ms) { timers.push({fn, ms}); }, Date, URL,
  });
  await new Promise(setImmediate); await new Promise(setImmediate);
  return {elements, calls, catalog, poll: () => timers.find(timer => timer.ms === 100).fn()};
}

test('new trained opponent defaults human to orange; explicit blue is forwarded and labeled', async () => {
  const {elements: ui, calls} = await setup();
  assert.equal(ui['human-side'].value, '0', ui.error.textContent);
  ui.opponent.value = 'trained'; await ui.opponent.trigger('change');
  assert.equal(ui['human-side'].value, '1', ui.error.textContent);
  ui['human-side'].value = '0'; await ui['human-side'].trigger('change');
  await ui.load.trigger('click');
  const selected = calls.find(call => call.url === '/api/select').body;
  assert.deepEqual(selected, {backend: 'mujoco', opponent: 'trained', humanSide: 0});
  assert.equal(ui['blue-name'].textContent, 'YOU · BLUE');
  assert.equal(ui['orange-name'].textContent, 'Checkpoint 8192 · ORANGE');
  assert.equal(ui['control-role'].textContent, 'Play as blue');
});

test('server-confirmed human orange side controls labels without changing score indices', async () => {
  const {elements: ui} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1});
  assert.equal(ui['human-side'].value, '1');
  assert.equal(ui['orange-name'].textContent, 'YOU · ORANGE');
  assert.equal(ui['blue-name'].textContent, 'Checkpoint 8192 · BLUE');
  assert.equal(ui['control-role'].textContent, 'Play as orange');
  assert.match(ui.arena.attributes['aria-label'], /your orange robot/);
  assert.equal(ui.standings.children.length, 2);
});

test('viewer and training execution paths and short rounds are explicitly labeled', () => {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  assert.match(html, /CPU physics with CUDA policy\/controller/);
  assert.match(html, /Headless CUDA training runs separately/);
  assert.match(html, /Experimental 20-second rounds/);
});

test('compact GPU backend exposes its own execution boundary and approximation warning', async () => {
  const {elements: ui, catalog} = await setup();
  catalog.backends.push({id: 'semantic_cuda', label: 'Reduced GPU candidate', available: true,
    runtimeNote: 'Same CUDA environment for training and evaluation. CPU kinematics renders pictures only.',
    warning: 'Approximate root movement and collision response. Authentic REK parity is unverified.'});
  catalog.policies.push({backend: 'semantic_cuda', id: 'scripted', label: 'Scripted', kind: 'scripted', rank: null});
  await ui.refresh.trigger('click');
  ui.backend.value = 'semantic_cuda'; await ui.backend.trigger('change');
  await ui.load.trigger('click');
  assert.equal(ui['runtime-note'].textContent, catalog.backends[1].runtimeNote);
  assert.equal(ui['backend-warning'].textContent, catalog.backends[1].warning);
  assert.equal(ui['backend-warning'].hidden, false);
  ui.backend.value = 'mujoco'; await ui.backend.trigger('change');
  await ui.load.trigger('click');
  assert.match(ui['runtime-note'].textContent, /CPU physics/);
  assert.equal(ui['backend-warning'].hidden, true);
});

test('compact FP32 computation variants are labeled independently of checkpoint storage', async () => {
  const {elements: ui, catalog} = await setup();
  catalog.backends.push({id: 'semantic_cuda', label: 'Reduced GPU candidate', available: true});
  catalog.policies.push({backend: 'semantic_cuda', id: 'compact-33m', label: 'Compact 33.6M', kind: 'trained',
    checkpoint: {format: 'pufferlib-native-flat-fp32', model: {precision: 'fp32'}}});
  catalog.policies.push({backend: 'semantic_cuda', id: 'compact-33m-bf16-greedy', label: 'Compact 33.6M BF16 greedy', kind: 'trained',
    checkpoint: {format: 'pufferlib-native-flat-fp32', model: {precision: 'bf16'}}});
  await ui.refresh.trigger('click');
  ui.backend.value = 'semantic_cuda'; await ui.backend.trigger('change');
  ui.opponent.value = 'compact-33m'; await ui.load.trigger('click');
  assert.match(ui['active-config'].textContent, /FP32 inference variant/);
  assert.match(ui.opponent.options.find(o => o.value === 'compact-33m').textContent, /FP32 inference variant/);
  assert.doesNotMatch(ui.opponent.options.find(o => o.value === 'compact-33m-bf16-greedy').textContent, /FP32 inference variant/);
});

test('switching release is harmless but rejected held input remains visible', async () => {
  const {elements: ui} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1}, null, 409);
  await ui.backend.trigger('change'); await new Promise(setImmediate);
  assert.equal(ui.error.textContent || '', '');
  ui.arena.events.keydown({key: 'W', repeat: false, preventDefault() {}});
  await new Promise(setImmediate);
  assert.equal(ui.error.textContent, 'Input rejected (409)');
});

test('rank uses paired evidence and remains provisional even with a rank number', async () => {
  const {elements: ui, catalog} = await setup();
  Object.assign(catalog.policies[1], {rank: 1, ranked: true, strengthStatus: 'provisional',
    total: {games: 100, wins: 20, draws: 0, losses: 80, winRate: 0.2},
    paired: {games: 2, wins: 2, draws: 0, losses: 0, winRate: 1}});
  await ui.refresh.trigger('click');
  const row = ui.standings.children[0];
  assert.equal(row.cells[3].textContent, '2');
  assert.equal(row.cells[5].textContent, '0');
  assert.equal(row.cells[6].textContent, '100.0%');
  assert.equal(row.cells[7].textContent, '2 paired matches · provisional');
  const choice = ui.opponent.options.find(option => option.value === 'trained');
  assert.match(choice.textContent, /100.0% wins · 2 matches · provisional/);
});

test('fall counts remain distinct from points and match outcomes', async () => {
  const {elements: ui, poll} = await setup({backend: 'puffysics', opponent: 'trained', humanSide: 1},
    {ok: true, tick: 1000, score: [20, 20], falls: [4, 4], terminal: 1, roundResult: 3, winner: -1, timeRemaining: 0});
  await poll();
  assert.equal(ui['blue-score'].textContent, 20); assert.equal(ui['orange-score'].textContent, 20);
  assert.equal(ui['blue-falls'].textContent, 'Falls: 4'); assert.equal(ui['orange-falls'].textContent, 'Falls: 4');
  assert.equal(ui['match-status'].textContent, 'Round tied · reset to play again');
  assert.equal(ui['backend-warning'].hidden, false);
});

test('official knockout winner is shown even with fewer points, without a MuJoCo warning', async () => {
  const {elements: ui, poll} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1},
    {ok: true, tick: 100, score: [1, 9], falls: [0, 1], terminal: 1, roundResult: 2, winner: 0, timeRemaining: 18});
  await poll();
  assert.equal(ui['match-status'].textContent, 'Blue wins by knockout · reset to play again');
  assert.equal(ui['backend-warning'].hidden, true);
  assert.equal(ui['orange-falls'].textContent, 'Falls: 1');
});

test('missing fall measurements and redo do not become fabricated losses', async () => {
  const {elements: ui, poll} = await setup({backend: 'mujoco', opponent: 'script', humanSide: 0},
    {ok: true, tick: 1000, score: [0, 0], terminal: 1, roundResult: 4, winner: -1, timeRemaining: 0});
  await poll();
  assert.equal(ui['blue-falls'].textContent, 'Falls: unknown');
  assert.equal(ui['match-status'].textContent, 'Round requires replay · no winner');
});
