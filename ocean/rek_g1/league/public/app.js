'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries(['backend', 'opponent', 'human-side', 'load', 'reset', 'refresh', 'connection', 'selection-info',
    'error', 'arena', 'frame', 'blue-score', 'orange-score', 'blue-falls', 'orange-falls', 'blue-name', 'orange-name', 'remaining', 'match-status',
    'active-config', 'tick', 'standings', 'ranking-note', 'control-role', 'backend-warning', 'runtime-note'].map(id => [id, $(id)]));
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
  const keyboardHeld = new Set();
  const pointerHeld = new Set();
  const heldKeys = new Set(['W', 'A', 'S', 'D', 'Q', 'E']);
  const moveKeys = {U: 17, I: 18};

  function showError(message) { ui.error.hidden = !message; ui.error.textContent = message || ''; }
  function connection(message, state = '') { ui.connection.textContent = message; ui.connection.className = `status ${state}`; }
  async function api(url, body) {
    const options = {cache: 'no-store', signal: AbortSignal.timeout(15000)};
    if (body !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const response = await fetch(url, options);
    const value = await response.json();
    if (!response.ok || value.ok === false) throw new Error(value.error || value.failure || `${response.status} ${response.statusText}`);
    return value;
  }
  function held() { return [...new Set([...keyboardHeld, ...pointerHeld])].sort(); }
  function sendInput(move = null) {
    if (!active || changing) return;
    const packet = {held: held(), move, seq: ++seq};
    // The native server owns move locking and the one yaw-to-attack transition.
    // The page sends edges immediately and never replays attacks on a timer.
    fetch('/api/input', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(packet), keepalive: true})
      .then(response => {
        // Switching already clears server inputs. A late blur/release packet
        // can receive 409 during that transition without losing a user action.
        if (response.status === 409 && packet.held.length === 0 && packet.move === null) return;
        if (!response.ok) throw new Error(`Input rejected (${response.status})`);
      })
      .catch(error => showError(error.message));
  }
  function release() {
    keyboardHeld.clear(); pointerHeld.clear();
    document.querySelectorAll('.pressed').forEach(button => button.classList.remove('pressed'));
    sendInput(null);
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
    ui['runtime-note'].textContent = backend?.runtimeNote || 'Interactive viewer: CPU physics with CUDA policy/controller. Headless CUDA training runs separately. Experimental 20-second rounds.';
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
    if (state.roundResult === 3) return 'Round tied · reset to play again';
    if (state.winner !== 0 && state.winner !== 1) return 'Round complete · result unavailable';
    const winner = state.winner === 0 ? 'Blue' : 'Orange';
    const method = state.roundResult === 2 ? ' by knockout' : state.roundResult === 1 ? ' on points' : '';
    return `${winner} wins${method} · reset to play again`;
  }
  async function stateLoop() {
    if (stopped) return;
    try {
      if (active && !changing && !document.hidden) {
        const state = await api('/api/state');
        if (state.active) setActive({...state.active, humanSide: state.humanSide ?? state.active.humanSide});
        else if (active && (state.humanSide === 0 || state.humanSide === 1) && state.humanSide !== active.humanSide)
          setActive({...active, humanSide: state.humanSide});
        const scores = state.score || [0, 0];
        ui['blue-score'].textContent = scores[0]; ui['orange-score'].textContent = scores[1];
        for (const [side, color] of ['blue', 'orange'].entries()) {
          const count = state.falls?.[side];
          ui[`${color}-falls`].textContent = `Falls: ${Number.isSafeInteger(count) && count >= 0 ? count : 'unknown'}`;
        }
        ui.remaining.textContent = formatTime(state.timeRemaining);
        ui.tick.textContent = `Tick ${state.tick ?? '?'}`;
        if (state.tick !== lastTick) { lastAdvance = Date.now(); lastTick = state.tick; }
        const terminal = Array.isArray(state.terminal) ? state.terminal.some(Boolean) : Boolean(state.terminal);
        ui['match-status'].textContent = terminal ? completedRoundLabel(state)
          : Date.now() - lastAdvance > 5000 ? 'Waiting for simulation progress' : 'Match in progress';
        if (state.failure) { showError(String(state.failure)); connection('Simulation failure', 'failed'); release(); }
        else connection(terminal ? 'Match complete' : 'Live evaluation', 'live');
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
  ui.load.addEventListener('click', async () => {
    release(); changing = true; ui.load.disabled = true; ui.reset.disabled = true; ui.backend.disabled = true; ui.opponent.disabled = true; ui['human-side'].disabled = true;
    showError(''); connection('Loading evaluation');
    try {
      await api('/api/select', {backend: ui.backend.value, opponent: ui.opponent.value, humanSide: Number(ui['human-side'].value)});
      await refreshCatalog(); lastTick = null; lastAdvance = Date.now();
      ui.arena.focus({preventScroll: true});
    } catch (error) { showError(error.message); connection('Load failed', 'failed'); }
    finally { changing = false; ui.backend.disabled = !catalog.backends.some(item => item.available); renderOpponents(ui.opponent.value); setActive(active); }
  });
  ui.reset.addEventListener('click', async () => {
    release(); ui.reset.disabled = true;
    try { await api('/api/reset', {}); showError(''); lastTick = null; lastAdvance = Date.now(); ui.arena.focus({preventScroll: true}); }
    catch (error) { showError(error.message); }
    finally { ui.reset.disabled = !active; }
  });
  ui.refresh.addEventListener('click', async () => {
    try { await refreshCatalog(); await refreshStandings(); showError(''); }
    catch (error) { showError(error.message); }
  });
  ui.arena.addEventListener('pointerdown', () => ui.arena.focus({preventScroll: true}));
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
  window.addEventListener('blur', release);
  document.addEventListener('visibilitychange', () => { if (document.hidden) release(); });
  window.addEventListener('pagehide', () => { release(); stopped = true; if (currentBlob) URL.revokeObjectURL(currentBlob); });
  refreshCatalog().then(refreshStandings).catch(error => { showError(error.message); connection('Connection failed', 'failed'); });
  stateLoop(); frameLoop();
})();
