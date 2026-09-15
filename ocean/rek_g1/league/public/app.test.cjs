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
  focus() { this.focusCount = (this.focusCount || 0) + 1; }
  async trigger(name) { return this.events[name]?.({preventDefault() {}}); }
}
async function setup(active = null, state = null, inputStatus = 200) {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  const elements = Object.fromEntries([...html.matchAll(/<([a-z][a-z0-9-]*)\b[^>]*\bid="([^"]+)"/g)].map(match => [match[2], new Element(match[1])]));
  const catalog = {backends: [{id: 'mujoco', label: 'MuJoCo', available: true}], policies: [
    {backend: 'mujoco', id: 'script', label: 'Scripted', kind: 'scripted', rank: null},
    {backend: 'mujoco', id: 'trained', label: 'Checkpoint 8192', kind: 'trained', rank: 1, wins: 4, losses: 0, draws: 0, winRate: 1},
  ], active, humanRoundSeconds: [20, 120, 300], defaultHumanRoundSeconds: 300};
  const calls = [];
  const timers = [];
  const inputReply = typeof inputStatus === 'object' ? inputStatus : {status: inputStatus};
  const windowEvents = {}, documentEvents = {};
  const fetch = async (url, options = {}) => {
    calls.push({url, body: options.body && JSON.parse(options.body)});
    if (url === '/api/input') return {ok: inputReply.status === 200, status: inputReply.status,
      json: async () => ({ok: inputReply.status === 200, accepted: inputReply.accepted, error: inputReply.error})};
    if (url === '/api/select') catalog.active = JSON.parse(options.body);
    const value = url === '/api/state' ? {ok: true, paused: false, ...state}
      : url === '/api/standings' ? [{backend: 'mujoco', standings: catalog.policies}]
      : url === '/api/catalog' || url === '/api/select' ? catalog : {ok: true};
    return {ok: true, json: async () => structuredClone(value)};
  };
  const document = {getElementById: id => elements[id], createElement: tag => new Element(tag),
    querySelectorAll: () => [], addEventListener(name, fn) { documentEvents[name] = fn; }, hidden: false};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8'), {
    document, window: {addEventListener(name, fn) { windowEvents[name] = fn; }}, fetch, AbortSignal, setTimeout(fn, ms) { timers.push({fn, ms}); }, Date, URL,
  });
  await new Promise(setImmediate); await new Promise(setImmediate);
  return {elements, calls, catalog, inputReply, windowEvents, documentEvents, document, poll: () => timers.find(timer => timer.ms === 100).fn()};
}

test('new trained opponent defaults human to orange; explicit blue is forwarded and labeled', async () => {
  const {elements: ui, calls} = await setup();
  assert.equal(ui['human-side'].value, '0', ui.error.textContent);
  ui.opponent.value = 'trained'; await ui.opponent.trigger('change');
  assert.equal(ui['human-side'].value, '1', ui.error.textContent);
  ui['human-side'].value = '0'; await ui['human-side'].trigger('change');
  await ui.load.trigger('click');
  const selected = calls.find(call => call.url === '/api/select').body;
  assert.deepEqual(selected, {backend: 'mujoco', opponent: 'trained', humanSide: 0, roundSeconds: 300});
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

test('round and session points, recorded training and human duration are explicitly labeled', () => {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
  assert.match(html, /ROUND POINTS/);
  assert.match(html, /Session totals/);
  assert.match(html, /Recorded training run/);
  assert.match(html, /Human rounds default to 300 seconds/);
  assert.doesNotMatch(html, /Three knockdowns|Knockout awards count as points/);
});

test('round-end text follows backend metadata and does not invent compact knockouts', async () => {
  const {elements: ui, catalog} = await setup();
  const roundEndNote = 'Points decide timed rounds. Physical knockdowns are not modeled; hits do not reset positions.';
  const warning = 'Points-only reduced-model experiment. Physical knockdowns are not modeled.';
  catalog.backends.push({id: 'semantic_cuda', label: 'Reduced GPU candidate', available: true, roundEndNote, warning});
  catalog.policies.push({backend: 'semantic_cuda', id: 'scripted', label: 'Scripted', kind: 'scripted'});
  await ui.refresh.trigger('click');
  ui.backend.value = 'semantic_cuda'; await ui.backend.trigger('change');
  await ui.load.trigger('click');
  assert.ok(ui['round-protocol'].textContent.includes(roundEndNote));
  assert.doesNotMatch(ui['round-protocol'].textContent, /Three knockdowns|knockout/i);
  assert.equal(ui['backend-warning'].textContent, warning);
  ui.backend.value = 'mujoco'; await ui.backend.trigger('change');
  await ui.load.trigger('click');
  assert.match(ui['round-protocol'].textContent, /Round-end rules depend on the selected backend/);
  assert.doesNotMatch(ui['round-protocol'].textContent, /Three knockdowns|Physical knockdowns are not modeled/);
});

test('missing or blank round metadata uses generic rules rather than a knockout threshold', async () => {
  const {elements: ui, catalog} = await setup({backend: 'mujoco', opponent: 'script', humanSide: 0});
  for (const roundEndNote of [undefined, '', '   ', null]) {
    catalog.backends[0].roundEndNote = roundEndNote;
    await ui.refresh.trigger('click');
    assert.match(ui['round-protocol'].textContent, /Round-end rules depend on the selected backend/);
    assert.doesNotMatch(ui['round-protocol'].textContent, /Three knockdowns|knockout/i);
  }
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
  assert.equal(ui['match-status'].textContent, 'Round tied');
  assert.equal(ui['backend-warning'].hidden, false);
});

test('official knockout winner is shown even with fewer points, without a MuJoCo warning', async () => {
  const {elements: ui, poll} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1},
    {ok: true, tick: 100, score: [1, 9], falls: [0, 1], terminal: 1, roundResult: 2, winner: 0, timeRemaining: 18});
  await poll();
  assert.equal(ui['match-status'].textContent, 'Blue wins by knockout');
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

test('300-second human default and explicit120-second selection are sent to the server', async () => {
  const {elements: ui, calls} = await setup();
  assert.equal(ui['round-seconds'].value, '300');
  assert.deepEqual(ui['round-seconds'].options.map(o => o.value), ['20', '120', '300']);
  ui['round-seconds'].value = '120'; await ui['round-seconds'].trigger('change');
  await ui.load.trigger('click');
  assert.equal(calls.find(c => c.url === '/api/select').body.roundSeconds, 120);
  assert.equal(ui.play.textContent, 'Resume');
  assert.equal(ui.arena.attributes['data-paused'], 'true');
});

test('current round points are separate from cumulative session points and outcomes across reset', async () => {
  const state = {tick: 100, score: [1, 2], falls: [0, 0], session: {completedRounds: 3,
    bluePoints: 41, orangePoints: 17, blueWins: 2, orangeWins: 0, draws: 1,
    lastRound: {bluePoints: 10, orangePoints: 9, winner: 0, reason: 'points'}}};
  const {elements: ui, poll} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1}, state);
  await poll();
  assert.equal(ui['blue-score'].textContent, 1); assert.equal(ui['orange-score'].textContent, 2);
  assert.equal(ui['session-blue-points'].textContent, '41'); assert.equal(ui['session-orange-points'].textContent, '17');
  assert.equal(ui['session-blue-wld'].textContent, '2 W / 0 L / 1 D');
  assert.equal(ui['session-orange-wld'].textContent, '0 W / 2 L / 1 D');
  assert.match(ui['session-last'].textContent, /Blue won on points.*Blue 10 : 9 Orange/);
  await ui.reset.trigger('click'); state.score = [0, 0]; state.paused = true; await poll();
  assert.equal(ui['blue-score'].textContent, 0); assert.equal(ui['session-blue-points'].textContent, '41');
  assert.equal(ui['session-rounds'].textContent, '3 completed rounds');
});

test('missing session values remain unknown and a known empty session is explicit', async () => {
  const state = {score: [0, 0]};
  const {elements: ui, poll} = await setup({backend: 'mujoco', opponent: 'script', humanSide: 0}, state);
  await poll(); assert.equal(ui['session-blue-points'].textContent, 'Unknown');
  assert.equal(ui['session-blue-wld'].textContent, 'W / L / D unavailable');
  state.session = {completedRounds: 0, bluePoints: 0, orangePoints: 0, blueWins: 0, orangeWins: 0, draws: 0, lastRound: null};
  await poll(); assert.equal(ui['session-last'].textContent, 'No completed rounds yet.');
});

test('recorded training and frozen evaluation identify checkpoint, precision, action mode and round mismatch', async () => {
  const {elements: ui, catalog} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1, roundSeconds: 300, trainingRoundSeconds: 20});
  catalog.backends[0].training = {status: 'completed', steps: 33554432, trainingSps: 2508920,
    trainingSeconds: 13.37, processWallSeconds: 16.12, benchmarkRoundSeconds: 20, checkpointSha256: 'abc123',
    frozenEvaluation: {wins: 1007, losses: 4, draws: 13, games: 1024, actionSelection: 'sampled', precision: 'bf16'}, note: 'Recorded GPU run.'};
  catalog.policies[1].checkpoint = {sha256: 'abc123', trainingSteps: 33554432, model: {precision: 'bf16', actionSelection: 'sampled'}};
  await ui.refresh.trigger('click');
  assert.equal(ui['training-sps'].textContent, '2,508,920');
  assert.equal(ui['training-seconds'].textContent, '13.37 s');
  assert.equal(ui['training-process-seconds'].textContent, '16.12 s');
  assert.match(ui['training-summary'].textContent, /Completed run.*2,508,920 training SPS.*33,554,432 transitions/);
  assert.match(ui['training-evaluation'].textContent, /98.34% wins.*Checkpoint, precision and action mode match/);
  assert.match(ui['round-protocol'].textContent, /300 s.*20 s.*has not been measured/);
  catalog.policies[1].checkpoint.model.precision = 'fp32'; await ui.refresh.trigger('click');
  assert.match(ui['training-evaluation'].textContent, /not the loaded opponent/);
  catalog.policies[1].checkpoint.model.precision = 'bf16'; catalog.policies[1].checkpoint.model.actionSelection = 'greedy';
  await ui.refresh.trigger('click'); assert.match(ui['training-evaluation'].textContent, /not the loaded opponent/);
  catalog.policies[1].checkpoint.trainingSteps = 1048576; catalog.policies[1].checkpoint.sha256 = 'early';
  await ui.refresh.trigger('click'); assert.match(ui['training-selected'].textContent, /1,048,576 transitions/);
  assert.equal(ui['training-steps'].textContent, '33,554,432');
});

test('frozen conditions retain opponent, geometry, duration and behavioral results without an overall strength percentage', async () => {
  const {elements: ui, catalog} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1, roundSeconds: 300, trainingRoundSeconds: 20});
  catalog.policies[1].checkpoint = {sha256:'v4-sha',model:{precision:'bf16',actionSelection:'sampled'}};
  catalog.backends[0].training = {status:'completed',checkpointSha256:'v4-sha',benchmarkRoundSeconds:20,
    frozenEvaluationProtocol:{precision:'bf16',actionSelection:'sampled'},
    frozenEvaluationConditions:[
      {opponent:'neutral',geometry:'fixed starts',roundSeconds:300,wins:62,losses:0,draws:2,games:64,zeroHitGames:2,meanPoints:17.5,meanOpponentPoints:0},
      {opponent:'retreat',geometry:'held-out seed10001',roundSeconds:20,wins:3,losses:4,draws:57,games:64,zeroHitGames:57,meanPoints:0.2,meanOpponentPoints:0.3},
    ]};
  await ui.refresh.trigger('click');
  assert.equal(ui['training-conditions-wrap'].hidden,false);
  assert.equal(ui['training-conditions'].children.length,2);
  const first = ui['training-conditions'].children[0].cells;
  assert.deepEqual(first.slice(0,7).map(cell=>cell.textContent),['neutral','fixed starts','300 s','62 / 0 / 2','64','2','17.5 / 0']);
  assert.match(first[7].textContent,/sampled \/ BF16.*v4-sha.*Matches loaded checkpoint/);
  const second = ui['training-conditions'].children[1].cells;
  assert.equal(second[0].textContent,'retreat');assert.equal(second[1].textContent,'held-out seed10001');
  assert.match(ui['training-evaluation'].textContent,/2 with validated outcome counts of 2 listed/);
  assert.doesNotMatch(ui['training-evaluation'].textContent,/% wins/);
  assert.match(ui['round-protocol'].textContent,/Matching frozen tests at 300 s.*do not establish human strength/);
  assert.doesNotMatch(ui['round-protocol'].textContent,/has not been measured/);
  assert.equal(ui['round-seconds'].value,'300');
});

test('frozen condition applicability rejects checkpoint, precision and action-mode mismatches', async () => {
  const {elements: ui, catalog} = await setup({backend:'mujoco',opponent:'trained',humanSide:1,roundSeconds:300,trainingRoundSeconds:20});
  const checkpoint = {sha256:'selected',model:{precision:'bf16',actionSelection:'sampled'}};
  catalog.policies[1].checkpoint = checkpoint;
  const result = {opponent:'scripted',geometry:'fixed starts',roundSeconds:300,wins:1,losses:0,draws:0,games:1,
    checkpointSha256:'other',precision:'bf16',actionSelection:'sampled'};
  catalog.backends[0].training = {checkpointSha256:'selected',frozenEvaluationConditions:[result]};
  for (const variant of [{checkpointSha256:'other',precision:'bf16',actionSelection:'sampled'},
      {checkpointSha256:'selected',precision:'fp32',actionSelection:'sampled'},
      {checkpointSha256:'selected',precision:'bf16',actionSelection:'greedy'},
      {checkpointSha256:'selected',precision:undefined,actionSelection:undefined}]) {
    Object.assign(result,variant);await ui.refresh.trigger('click');
    assert.match(ui['training-conditions'].children[0].cells[7].textContent,/Does not establish results for the loaded opponent/);
    assert.match(ui['round-protocol'].textContent,/has not been measured in the loaded evidence/);
    assert.doesNotMatch(ui['round-protocol'].textContent,/Matching frozen tests/);
  }
});

test('invalid condition counts and absent optional metrics remain unknown and cannot certify a duration', async () => {
  const {elements: ui, catalog} = await setup({backend:'mujoco',opponent:'trained',humanSide:1,roundSeconds:300,trainingRoundSeconds:20});
  catalog.policies[1].checkpoint = {sha256:'x',model:{precision:'bf16',actionSelection:'sampled'}};
  catalog.backends[0].training = {checkpointSha256:'x',frozenEvaluationProtocol:{precision:'bf16',actionSelection:'sampled'},
    frozenEvaluationConditions:[{opponent:'neutral',geometry:'fixed',roundSeconds:300,wins:2,losses:0,draws:0,games:1,zeroHitGames:3},null]};
  await ui.refresh.trigger('click');
  const cells = ui['training-conditions'].children[0].cells;
  assert.equal(cells[3].textContent,'Invalid or missing results');assert.equal(cells[5].textContent,'Unknown');
  assert.equal(cells[6].textContent,'Unknown / Unknown');
  assert.match(ui['training-evaluation'].textContent,/0 with validated outcome counts of 2/);
  assert.doesNotMatch(ui['round-protocol'].textContent,/Matching frozen tests/);
  delete catalog.backends[0].training;await ui.refresh.trigger('click');
  assert.equal(ui['training-conditions-wrap'].hidden,true);assert.equal(ui['training-conditions'].children.length,0);
  assert.match(ui['training-evaluation'].textContent,/No verified frozen evaluation/);
});

test('actual input error JSON is shown and a later accepted input clears only that input error', async () => {
  const {elements: ui, inputReply} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1}, null,
    {status: 422, error: 'Move is unavailable during translation'});
  ui.arena.events.keydown({key: 'U', repeat: false, preventDefault() {}}); await new Promise(setImmediate);
  assert.equal(ui.error.textContent, 'Move is unavailable during translation');
  inputReply.status = 200; inputReply.error = undefined;
  ui.arena.events.keydown({key: 'I', repeat: false, preventDefault() {}}); await new Promise(setImmediate);
  assert.equal(ui.error.textContent, '');
});

test('accepted input does not clear a simulator error', async () => {
  const {elements: ui, poll} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1}, {ok: false, failure: 'GPU runtime failure'});
  await poll(); assert.equal(ui.error.textContent, 'GPU runtime failure');
  ui.arena.events.keydown({key: 'U', repeat: false, preventDefault() {}}); await new Promise(setImmediate);
  assert.equal(ui.error.textContent, 'GPU runtime failure');
});

test('input resumes before sending, arena blur only releases, and window blur or hidden page pauses', async () => {
  const {elements: ui, calls, windowEvents, documentEvents, document} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1});
  ui.arena.events.keydown({key: 'W', repeat: false, preventDefault() {}}); await new Promise(setImmediate);
  const resumeIndex = calls.findIndex(c => c.url === '/api/play' && c.body.paused === false);
  assert(resumeIndex >= 0); assert(resumeIndex < calls.findIndex(c => c.url === '/api/input'));
  const playCount = calls.filter(c => c.url === '/api/play').length;
  await ui.arena.trigger('blur'); await new Promise(setImmediate);
  assert.equal(calls.filter(c => c.url === '/api/play').length, playCount);
  windowEvents.blur(); await new Promise(setImmediate);
  assert.equal(calls.filter(c => c.url === '/api/play').at(-1).body.paused, true);
  assert.equal(ui.play.textContent, 'Resume');
  await ui.arena.trigger('pointerdown'); await new Promise(setImmediate);
  assert.equal(calls.filter(c => c.url === '/api/play').at(-1).body.paused, false);
  document.hidden = true; documentEvents.visibilitychange(); await new Promise(setImmediate);
  assert.equal(calls.filter(c => c.url === '/api/play').at(-1).body.paused, true);
});

test('paused and intermission states are explicit instead of stalled-progress warnings', async () => {
  const state = {paused: true, tick: 100, score: [0, 0], terminal: 0};
  const {elements: ui, poll} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1}, state);
  await poll(); assert.match(ui['match-status'].textContent, /^Paused/);
  assert.equal(ui.connection.textContent, 'Paused');
  Object.assign(state, {paused: false, terminal: 1, roundResult: 2, winner: 1, intermissionSeconds: 2.4});
  await poll(); assert.equal(ui['match-status'].textContent, 'Orange wins by knockout · next round in 3 s');
  assert.equal(ui.connection.textContent, 'Round intermission');
});

test('explicit Resume gives keyboard focus to the browser arena, while Pause does not', async () => {
  const {elements: ui} = await setup({backend: 'mujoco', opponent: 'trained', humanSide: 1});
  await ui.play.trigger('click'); await new Promise(setImmediate);
  assert.equal(ui.play.textContent, 'Pause'); assert.equal(ui.arena.focusCount, 1);
  await ui.play.trigger('click'); await new Promise(setImmediate);
  assert.equal(ui.play.textContent, 'Resume'); assert.equal(ui.arena.focusCount, 1);
});
