'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries(['backend', 'opponent', 'human-side', 'round-seconds', 'load', 'play', 'reset', 'refresh', 'connection', 'selection-info',
    'error', 'arena', 'frame', 'blue-score', 'orange-score', 'blue-falls', 'orange-falls', 'blue-name', 'orange-name', 'remaining', 'match-status',
    'active-config', 'tick', 'standings', 'ranking-note', 'control-role', 'backend-warning', 'runtime-note', 'round-protocol', 'focus-prompt',
    'session-blue-points', 'session-orange-points', 'session-blue-wld', 'session-orange-wld', 'session-rounds', 'session-last',
    'training-status', 'training-steps', 'training-sps', 'training-seconds', 'training-process-seconds', 'training-selected', 'training-evaluation', 'training-note', 'training-summary'].map(id => [id, $(id)]));
  let catalog = {backends: [], policies: [], active: null};
  let active = null;
  // Monotonic across page reloads for a server that rejects stale input packets.
  let seq = Date.now() * 1024;
  let changing = false;
  let lastTick = null;
  let lastAdvance = Date.now();
  let currentBlob = null;
  let stopped = false;
  let humanSideExplicit = false;
  let roundExplicit = false;
  let paused = true;
  let playRequest = null;
  let playChain = Promise.resolve();
  let playVersion = 0;
  let inputGeneration = 0;
  let inputErrorSeq = -1;
  let successfulInputSeq = -1;
  let errorKind = null;
  const keyboardHeld = new Set();
  const pointerHeld = new Set();
  const heldKeys = new Set(['W', 'A', 'S', 'D', 'Q', 'E']);
  const moveKeys = {U: 17, I: 18};

  function showError(message, kind = 'general') { errorKind = message ? kind : null; ui.error.hidden = !message; ui.error.textContent = message || ''; }
  function connection(message, state = '') { ui.connection.textContent = message; ui.connection.className = `status ${state}`; }
  async function api(url, body, extra = {}) {
    const options = {cache: 'no-store', signal: AbortSignal.timeout(15000), ...extra};
    if (body !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const response = await fetch(url, options);
    const value = await response.json();
    if (!response.ok || value.ok === false) throw new Error(value.error || value.failure || `${response.status} ${response.statusText}`);
    return value;
  }
  function held() { return [...new Set([...keyboardHeld, ...pointerHeld])].sort(); }
  function pauseDisplay() {
    ui.play.textContent = paused ? 'Resume' : 'Pause';
    ui.play.disabled = !active || changing;
    ui.arena.setAttribute('data-paused', String(paused));
    ui['focus-prompt'].textContent = paused ? 'Paused. Click the arena to play.' : 'Click the arena to enable controls';
  }
  function setPaused(value) {
    if (!active || changing) return Promise.resolve();
    if (playRequest?.value === value) return playRequest.promise;
    playVersion++;
    if (value) { inputGeneration++; release(); }
    paused = value; pauseDisplay();
    const request = {value};
    // Preserve pause/resume ordering when focus changes during a request.
    request.promise = playChain.catch(() => {}).then(() => api('/api/play', {paused: value}, {keepalive: true}))
      .catch(error => { if (playRequest === request) showError(error.message); throw error; })
      .finally(() => { if (playRequest === request) playRequest = null; });
    playRequest = request; playChain = request.promise;
    return request.promise;
  }
  function sendInput(move = null, resume = true) {
    if (!active || changing) return;
    if (!resume && paused) return;
    const packet = {held: held(), move, seq: ++seq};
    const generation = inputGeneration;
    // The native server owns move locking and the one yaw-to-attack transition.
    // The page sends edges immediately and never replays attacks on a timer.
    const playing = resume && paused ? setPaused(false) : playRequest?.promise || Promise.resolve();
    playing.then(() => {
      if (generation !== inputGeneration || changing || paused || stopped) return;
      return fetch('/api/input', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(packet), keepalive: true});
    }).then(async response => {
        if (!response) return;
        // Switching already clears server inputs. A late blur/release packet
        // can receive 409 during that transition without losing a user action.
        if (response.status === 409 && packet.held.length === 0 && packet.move === null) return;
        const value = await response.json().catch(() => ({}));
        if (!response.ok || value.ok === false) throw new Error(value.error || value.failure || `Input rejected (${response.status})`);
        if (value.accepted !== false) {
          successfulInputSeq = Math.max(successfulInputSeq, packet.seq);
          if (errorKind === 'input' && packet.seq > inputErrorSeq) showError('');
        }
      })
      .catch(error => {
        if (generation !== inputGeneration || packet.seq < successfulInputSeq || errorKind && errorKind !== 'input') return;
        inputErrorSeq = packet.seq; showError(error.message, 'input');
      });
  }
  function release() {
    keyboardHeld.clear(); pointerHeld.clear();
    document.querySelectorAll('.pressed').forEach(button => button.classList.remove('pressed'));
    sendInput(null, false);
  }
  function measuredStats(policy) {
    return policy.paired?.games ? policy.paired : policy.pairedStats?.games ? policy.pairedStats : policy.total || policy.stats || policy;
  }
  function policyLabel(policy) {
    const label = policy.label || policy.id;
    return policy.backend === 'semantic_cuda' && policy.checkpoint?.model?.precision === 'fp32'
      ? `${label} · FP32 inference variant` : label;
  }
  function policyName(policy) {
    const stats = measuredStats(policy);
    const games = Number(stats.wins || 0) + Number(stats.draws || 0) + Number(stats.losses || 0);
    const winRate = Number.isFinite(stats.winRate) ? stats.winRate : games ? Number(stats.wins || 0) / games : null;
    const rank = policy.rank != null ? `#${policy.rank}` : 'unranked';
    const evidence = games ? `${(winRate * 100).toFixed(1)}% wins · ${games} matches${policy.strengthStatus === 'measured' ? '' : ' · provisional'}` : 'untested';
    return `${policyLabel(policy)} · ${policy.kind === 'scripted' ? 'scripted' : 'RL'} · ${rank} · ${evidence}`;
  }
  function selectedPolicies() { return catalog.policies.filter(policy => policy.backend === ui.backend.value); }
  function number(value, digits = 0) {
    return Number.isFinite(value) && value >= 0 ? value.toLocaleString('en-US', {maximumFractionDigits: digits}) : 'Unknown';
  }
  function renderTraining(backend, policy) {
    const training = backend?.training;
    ui['training-summary'].textContent = training
      ? `${training.status === 'completed' ? 'Completed run' : training.status || 'Recorded run'} · ${number(training.trainingSps)} training SPS · ${number(training.steps)} transitions. Details below.`
      : 'Recorded training statistics unavailable.';
    ui['training-status'].textContent = training?.status === 'completed' ? 'Completed run' : training?.status || 'Unavailable';
    ui['training-steps'].textContent = number(training?.steps);
    ui['training-sps'].textContent = number(training?.trainingSps);
    ui['training-seconds'].textContent = Number.isFinite(training?.trainingSeconds) ? `${number(training.trainingSeconds, 2)} s` : 'Unknown';
    ui['training-process-seconds'].textContent = Number.isFinite(training?.processWallSeconds) ? `${number(training.processWallSeconds, 2)} s` : 'Unknown';
    const model = policy?.checkpoint?.model;
    ui['training-selected'].textContent = policy?.kind === 'trained'
      ? `Loaded checkpoint: ${number(policy.checkpoint?.trainingSteps)} transitions · ${model?.precision?.toUpperCase() || 'unknown precision'} inference · ${model?.actionSelection || 'greedy'} actions. Checkpoint ${policy.checkpoint?.sha256?.slice(0, 12) || 'hash unavailable'}.`
      : policy ? 'Scripted opponent selected. It has no trained checkpoint.' : 'No trained checkpoint selected.';
    const frozen = training?.frozenEvaluation;
    const known = frozen && [frozen.wins, frozen.losses, frozen.draws, frozen.games].every(v => Number.isSafeInteger(v) && v >= 0)
      && frozen.games > 0 && frozen.wins + frozen.losses + frozen.draws === frozen.games;
    if (known) {
      const matches = policy?.checkpoint?.sha256 === training.checkpointSha256 && model?.precision === frozen.precision
        && (model?.actionSelection || 'greedy') === frozen.actionSelection;
      ui['training-evaluation'].textContent = `Recorded frozen ${frozen.actionSelection} evaluation: ${frozen.wins} W / ${frozen.losses} L / ${frozen.draws} D in ${frozen.games} games (${(100 * frozen.wins / frozen.games).toFixed(2)}% wins), using ${number(training.benchmarkRoundSeconds)} s rounds.`
        + (matches ? ' Checkpoint, precision and action mode match the loaded opponent.' : ' This result belongs to the benchmark checkpoint, precision and action mode, not the loaded opponent.');
    } else ui['training-evaluation'].textContent = 'No verified frozen evaluation supplied for the recorded training run.';
    ui['training-note'].textContent = `Recorded headless training measurements, not browser frame rate or current training progress.${training?.note ? ` ${training.note}` : ''}`;
    const seconds = active?.roundSeconds;
    const trainedSeconds = active?.trainingRoundSeconds ?? training?.benchmarkRoundSeconds;
    const roundEndNote = typeof backend?.roundEndNote === 'string' && backend.roundEndNote.trim()
      ? backend.roundEndNote : 'Round-end rules depend on the selected backend.';
    ui['round-protocol'].textContent = `Human round limit: ${Number.isFinite(seconds) ? `${seconds} s` : 'not loaded'}. Training/benchmark round limit: ${Number.isFinite(trainedSeconds) ? `${trainedSeconds} s` : 'unknown'}. ${roundEndNote}`
      + (Number.isFinite(seconds) && Number.isFinite(trainedSeconds) && seconds !== trainedSeconds ? ' Strength at this longer or shorter duration has not been measured.' : '');
  }
  function renderSession(session) {
    ui['session-blue-points'].textContent = number(session?.bluePoints);
    ui['session-orange-points'].textContent = number(session?.orangePoints);
    const valid = session && [session.blueWins, session.orangeWins, session.draws].every(v => Number.isSafeInteger(v) && v >= 0);
    ui['session-blue-wld'].textContent = valid ? `${session.blueWins} W / ${session.orangeWins} L / ${session.draws} D` : 'W / L / D unavailable';
    ui['session-orange-wld'].textContent = valid ? `${session.orangeWins} W / ${session.blueWins} L / ${session.draws} D` : 'W / L / D unavailable';
    ui['session-rounds'].textContent = `${number(session?.completedRounds)} completed rounds`;
    const last = session?.lastRound;
    if (!last) { ui['session-last'].textContent = session?.completedRounds === 0 ? 'No completed rounds yet.' : 'Last result unavailable.'; return; }
    const outcome = last.winner === 0 ? 'Blue won' : last.winner === 1 ? 'Orange won' : last.reason === 'draw' ? 'Draw' : last.reason === 'replay' ? 'Replay required' : 'Winner unavailable';
    const reason = {points: 'on points', knockout: 'by knockout', draw: 'tied', replay: 'no winner', unknown: 'reason unknown'}[last.reason] || 'reason unknown';
    ui['session-last'].textContent = `Last round: ${outcome} ${reason} · Blue ${number(last.bluePoints)} : ${number(last.orangePoints)} Orange.`;
  }
  function renderRoundChoices() {
    const previous = ui['round-seconds'].value;
    const choices = Array.isArray(catalog.humanRoundSeconds) ? catalog.humanRoundSeconds : [20, 120, 300];
    ui['round-seconds'].replaceChildren();
    for (const seconds of choices.filter(v => Number.isSafeInteger(v) && v > 0)) {
      const option = document.createElement('option'); option.value = String(seconds); option.textContent = `${seconds} s (${formatTime(seconds)})`;
      ui['round-seconds'].append(option);
    }
    const preferred = roundExplicit ? Number(previous) : active?.roundSeconds ?? catalog.defaultHumanRoundSeconds ?? 300;
    if (choices.includes(preferred)) ui['round-seconds'].value = String(preferred);
    ui['round-seconds'].disabled = changing || !ui['round-seconds'].options.length;
  }
  function refreshSideSelection() {
    const policy = selectedPolicies().find(value => value.id === ui.opponent.value);
    if (!humanSideExplicit) {
      const loadedSelection = active?.backend === ui.backend.value && active?.opponent === ui.opponent.value;
      ui['human-side'].value = String(loadedSelection ? active.humanSide : policy?.kind === 'trained' ? 1 : 0);
    }
    ui['human-side'].disabled = ui.opponent.disabled || changing;
    if (!ui.opponent.disabled) {
      const humanColor = ui['human-side'].value === '1' ? 'orange' : 'blue';
      const opponentColor = humanColor === 'orange' ? 'blue' : 'orange';
      ui['selection-info'].textContent = `You: ${humanColor}. Opponent: ${opponentColor}. Load evaluation to apply these selections. Rankings stay within each backend.`;
    }
  }
  function renderOpponents(preferred) {
    ui.opponent.replaceChildren();
    for (const policy of selectedPolicies()) {
      const option = document.createElement('option'); option.value = policy.id; option.textContent = policyName(policy);
      ui.opponent.append(option);
    }
    if (selectedPolicies().some(policy => policy.id === preferred)) ui.opponent.value = preferred;
    const backend = catalog.backends.find(value => value.id === ui.backend.value);
    ui.opponent.disabled = !backend?.available || !ui.opponent.options.length || changing;
    ui.load.disabled = ui.opponent.disabled;
    ui['selection-info'].textContent = !backend?.available ? backend?.error || 'Backend unavailable.'
      : ui.opponent.options.length ? 'Choose a backend and opponent, then load the evaluation. Rankings stay within each backend.'
      : 'No selectable scripted opponent or native checkpoint is registered for this backend.';
    refreshSideSelection();
    renderStandings(catalog.policies);
  }
  function setActive(value) {
    active = value?.backend && value?.opponent ? {...value, humanSide: value.humanSide === 1 ? 1 : 0} : null;
    ui.reset.disabled = !active || changing;
    pauseDisplay();
    const backend = catalog.backends.find(item => item.id === active?.backend);
    const policy = catalog.policies.find(item => item.id === active?.opponent && item.backend === active?.backend);
    const humanColor = active?.humanSide === 1 ? 'orange' : 'blue';
    const opponentLabel = policy ? policyLabel(policy) : active?.opponent || 'OPPONENT';
    ui['active-config'].textContent = active ? `${backend?.label || active.backend} · ${opponentLabel} · You: ${humanColor}` : 'No evaluation loaded';
    ui['blue-name'].textContent = active ? active.humanSide === 0 ? 'YOU · BLUE' : `${opponentLabel} · BLUE` : 'BLUE';
    ui['orange-name'].textContent = active ? active.humanSide === 1 ? 'YOU · ORANGE' : `${opponentLabel} · ORANGE` : 'ORANGE';
    ui['control-role'].textContent = active ? `Play as ${humanColor}` : 'Choose your robot above';
    const warning = backend?.warning || (active && /puff/i.test(active.backend)
      ? 'Puffysics is experimental. Frequent robot falls have been observed in this runtime. Scores may largely reflect fall/knockout awards; wins currently provide limited evidence of combat skill.' : '');
    ui['backend-warning'].hidden = !warning;
    ui['backend-warning'].textContent = warning;
    ui['runtime-note'].textContent = backend?.runtimeNote || 'Interactive viewer: CPU physics with CUDA policy/controller. Headless CUDA training runs separately.';
    renderTraining(backend, policy);
    ui.arena.setAttribute('aria-label', `Game controls${active ? ` for your ${humanColor} robot` : ''}. Click or focus here to play. W A S D movement, Q E turn, U straight kick, I side kick.`);
  }
  async function refreshCatalog() {
    const previousBackend = ui.backend.value;
    const previousPolicy = ui.opponent.value;
    const value = await api('/api/catalog');
    catalog = {...value, backends: value.backends || [], policies: value.policies || []};
    ui.backend.replaceChildren();
    for (const backend of catalog.backends) {
      const option = document.createElement('option'); option.value = backend.id;
      option.textContent = `${backend.label || backend.id}${backend.available ? '' : ' · unavailable'}`;
      option.disabled = !backend.available; ui.backend.append(option);
    }
    const preferred = previousBackend || value.active?.backend;
    if (catalog.backends.some(backend => backend.id === preferred && backend.available)) ui.backend.value = preferred;
    else ui.backend.value = catalog.backends.find(backend => backend.available)?.id || '';
    ui.backend.disabled = changing || !catalog.backends.some(backend => backend.available);
    setActive(value.active);
    renderRoundChoices();
    renderOpponents(previousPolicy || value.active?.opponent);
    connection('Connected', 'live');
  }
  function cells(row, values) {
    values.forEach(value => { const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell); });
  }
  function renderStandings(policies) {
    const rows = policies.filter(policy => policy.backend === ui.backend.value)
      .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity) || (a.label || a.id).localeCompare(b.label || b.id));
    ui.standings.replaceChildren();
    if (!rows.length) {
      const row = document.createElement('tr'); const cell = document.createElement('td');
      cell.colSpan = 8; cell.className = 'muted'; cell.textContent = 'No policies or match results for this backend.';
      row.append(cell); ui.standings.append(row); return;
    }
    for (const policy of rows) {
      const stats = measuredStats(policy);
      const games = Number(stats.wins || 0) + Number(stats.draws || 0) + Number(stats.losses || 0);
      const rate = Number.isFinite(stats.winRate) ? stats.winRate : games ? Number(stats.wins || 0) / games : null;
      const paired = Boolean(policy.paired?.games || policy.pairedStats?.games);
      let evidence = games ? `${games} ${paired ? 'paired ' : ''}matches${policy.strengthStatus === 'measured' ? '' : ' · provisional'}` : 'Untested';
      if (policy.invalidTrials) evidence += ` · ${policy.invalidTrials} invalid`;
      if (policy.pending) evidence += ` · ${policy.pending} pending`;
      const row = document.createElement('tr');
      cells(row, [policy.rank ?? '-', policyLabel(policy), policy.kind === 'scripted' ? 'Scripted' : 'Trained RL',
        stats.wins || 0, stats.draws || 0, stats.losses || 0, rate === null ? 'Unmeasured' : `${(rate * 100).toFixed(1)}%`, evidence]);
      const interval = stats.winRate95;
      if (Array.isArray(interval) && games) row.cells[6].title = `95% win-rate interval: ${(interval[0] * 100).toFixed(1)}% to ${(interval[1] * 100).toFixed(1)}%`;
      row.lastChild.className = 'evidence'; ui.standings.append(row);
    }
  }
  async function refreshStandings() {
    const response = await api('/api/standings');
    const rows = Array.isArray(response) ? response : response.policies || response.standings || response.rows;
    if (Array.isArray(rows)) renderStandings(rows.flatMap(row => Array.isArray(row.standings)
      ? row.standings.map(policy => ({backend: row.backend, ...policy})) : [row]));
  }
  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return '--:--';
    const remaining = Math.max(0, Math.ceil(seconds));
    return `${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')}`;
  }
  function completedRoundLabel(state) {
    if (state.roundResult === 4) return 'Round requires replay · no winner';
    if (state.roundResult === 3) return 'Round tied';
    if (state.winner !== 0 && state.winner !== 1) return 'Round complete · result unavailable';
    const winner = state.winner === 0 ? 'Blue' : 'Orange';
    const method = state.roundResult === 2 ? ' by knockout' : state.roundResult === 1 ? ' on points' : '';
    return `${winner} wins${method}`;
  }
  async function stateLoop() {
    if (stopped) return;
    try {
      if (active && !changing && !document.hidden) {
        const requestedPlayVersion = playVersion;
        const state = await api('/api/state');
        if (typeof state.paused === 'boolean' && !playRequest && requestedPlayVersion === playVersion) { paused = state.paused; pauseDisplay(); }
        if (state.active) setActive({...state.active, humanSide: state.humanSide ?? state.active.humanSide});
        else if (active && (state.humanSide === 0 || state.humanSide === 1) && state.humanSide !== active.humanSide)
          setActive({...active, humanSide: state.humanSide});
        const scores = state.score || [0, 0];
        renderSession(state.session);
        ui['blue-score'].textContent = scores[0]; ui['orange-score'].textContent = scores[1];
        for (const [side, color] of ['blue', 'orange'].entries()) {
          const count = state.falls?.[side];
          ui[`${color}-falls`].textContent = `Falls: ${Number.isSafeInteger(count) && count >= 0 ? count : 'unknown'}`;
        }
        ui.remaining.textContent = formatTime(state.timeRemaining);
        ui.tick.textContent = `Tick ${state.tick ?? '?'}`;
        if (state.tick !== lastTick) { lastAdvance = Date.now(); lastTick = state.tick; }
        const terminal = Array.isArray(state.terminal) ? state.terminal.some(Boolean) : Boolean(state.terminal);
        const intermission = Number.isFinite(state.intermissionSeconds) && state.intermissionSeconds > 0;
        ui['match-status'].textContent = intermission ? `${completedRoundLabel(state)} · ${paused ? 'paused' : `next round in ${Math.ceil(state.intermissionSeconds)} s`}`
          : paused ? 'Paused · click arena or Resume' : terminal ? completedRoundLabel(state)
          : Date.now() - lastAdvance > 5000 ? 'Waiting for simulation progress' : 'Round in progress';
        if (state.failure) { showError(String(state.failure)); connection('Simulation failure', 'failed'); release(); }
        else connection(paused ? 'Paused' : intermission ? 'Round intermission' : terminal ? 'Round complete' : 'Live evaluation', 'live');
      }
    } catch (error) { showError(error.message); connection('Simulator unavailable', 'failed'); release(); }
    setTimeout(stateLoop, 100);
  }
  async function frameLoop() {
    if (stopped) return;
    try {
      if (active && !changing && !document.hidden) {
        const response = await fetch(`/frame.png?t=${Date.now()}`, {cache: 'no-store', signal: AbortSignal.timeout(5000)});
        if (response.ok && response.headers.get('content-type')?.startsWith('image/')) {
          const next = URL.createObjectURL(await response.blob());
          ui.frame.src = next;
          if (currentBlob) URL.revokeObjectURL(currentBlob);
          currentBlob = next;
        }
      }
    } catch (_) { /* The state endpoint supplies the actionable error. */ }
    setTimeout(frameLoop, 65);
  }
  ui.backend.addEventListener('change', () => { release(); renderOpponents(); });
  ui.opponent.addEventListener('change', () => { release(); refreshSideSelection(); });
  ui['human-side'].addEventListener('change', () => { release(); humanSideExplicit = true; refreshSideSelection(); });
  ui['round-seconds'].addEventListener('change', () => { release(); roundExplicit = true; });
  ui.play.addEventListener('click', () => {
    setPaused(!paused).then(() => { if (!paused && !changing && !document.hidden) ui.arena.focus({preventScroll: true}); }).catch(() => {});
  });
  ui.load.addEventListener('click', async () => {
    release(); inputGeneration++; changing = true; ui.load.disabled = true; ui.play.disabled = true; ui.reset.disabled = true; ui.backend.disabled = true; ui.opponent.disabled = true; ui['human-side'].disabled = true; ui['round-seconds'].disabled = true;
    showError(''); connection('Loading evaluation');
    try {
      await playChain.catch(() => {});
      await api('/api/select', {backend: ui.backend.value, opponent: ui.opponent.value, humanSide: Number(ui['human-side'].value), roundSeconds: Number(ui['round-seconds'].value)});
      paused = true; renderSession(null);
      await refreshCatalog(); lastTick = null; lastAdvance = Date.now();
    } catch (error) { showError(error.message); connection('Load failed', 'failed'); }
    finally { changing = false; ui.backend.disabled = !catalog.backends.some(item => item.available); renderOpponents(ui.opponent.value); renderRoundChoices(); setActive(active); }
  });
  ui.reset.addEventListener('click', async () => {
    release(); inputGeneration++; ui.reset.disabled = true;
    try { await playChain.catch(() => {}); await api('/api/reset', {}); paused = true; pauseDisplay(); showError(''); lastTick = null; lastAdvance = Date.now(); }
    catch (error) { showError(error.message); }
    finally { ui.reset.disabled = !active; }
  });
  ui.refresh.addEventListener('click', async () => {
    try { await refreshCatalog(); await refreshStandings(); showError(''); }
    catch (error) { showError(error.message); }
  });
  ui.arena.addEventListener('pointerdown', () => { ui.arena.focus({preventScroll: true}); if (paused) setPaused(false).catch(() => {}); });
  ui.arena.addEventListener('keydown', event => {
    const key = event.key.toUpperCase();
    if (!heldKeys.has(key) && !Object.hasOwn(moveKeys, key)) return;
    event.preventDefault();
    if (event.repeat || !active || changing) return;
    if (heldKeys.has(key)) { keyboardHeld.add(key); sendInput(); }
    else sendInput(moveKeys[key]);
  });
  ui.arena.addEventListener('keyup', event => {
    const key = event.key.toUpperCase();
    if (!heldKeys.has(key) && !Object.hasOwn(moveKeys, key)) return;
    event.preventDefault();
    if (keyboardHeld.delete(key)) sendInput();
  });
  ui.arena.addEventListener('blur', release);
  document.querySelectorAll('[data-held]').forEach(button => {
    const key = button.dataset.held;
    button.addEventListener('pointerdown', event => {
      event.preventDefault(); if (!active || changing) return;
      button.setPointerCapture(event.pointerId); pointerHeld.add(key); button.classList.add('pressed'); sendInput();
    });
    const up = () => { pointerHeld.delete(key); button.classList.remove('pressed'); sendInput(); };
    button.addEventListener('pointerup', up); button.addEventListener('pointercancel', up); button.addEventListener('lostpointercapture', up);
  });
  document.querySelectorAll('[data-move]').forEach(button => button.addEventListener('pointerdown', event => {
    event.preventDefault(); sendInput(Number(button.dataset.move));
  }));
  window.addEventListener('blur', () => { setPaused(true).catch(() => {}); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) setPaused(true).catch(() => {}); });
  window.addEventListener('pagehide', () => { setPaused(true).catch(() => {}); stopped = true; if (currentBlob) URL.revokeObjectURL(currentBlob); });
  refreshCatalog().then(refreshStandings).catch(error => { showError(error.message); connection('Connection failed', 'failed'); });
  stateLoop(); frameLoop();
})();
