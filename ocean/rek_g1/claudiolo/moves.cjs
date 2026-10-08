'use strict';
// Build-pinned G1 move facts (f84f1874 build), from g1_strike_catalog.c,
// native_motion_routes.c and the shared 50 Hz duration table used by
// puffer_env.cu / eval_worker.cpp / encode-live. Impact times in the catalog
// are clip seconds; real seconds = clip / playback speed.

// Policy/bridge category 16..32 -> runtime move index.
const CATEGORY_TO_MOVE = Object.freeze([6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16]);
const MOVE_TO_CATEGORY = Object.freeze(CATEGORY_TO_MOVE.reduce((a, m, i) => { a[m] = 16 + i; return a; }, []));

// Shared candidate duration table (ticks at 50 Hz), runtime move order 0..16.
const DURATION_TICKS = Object.freeze([35, 27, 31, 45, 32, 45, 157, 145, 158, 139, 134, 138, 73, 75, 68, 71, 103]);

const L1 = 1, R1 = 2, L3 = 3, R4 = 4; // limbs: left/right hand, left/right leg
// [name, keys, playbackSpeed, impacts: [clipSeconds, limb, leadClip, releaseClip]]
const RAW = [
  ['left_hook', 'I', 1.0, [[0.38, L1, 0.25, 0.25]]],
  ['left_jab', 'K', 1.26, [[0.185, L1, 0.2, 0.33]]],
  ['double_uppercut', 'Space+J', 1.82, [[0.396, L1, 0.2, 0.4], [0.735, R1, 0.2, 0.4]]],
  ['right_hook', 'O', 1.0, [[0.49, R1, 0.3, 0.3]]],
  ['right_jab', 'L', 1.25, [[0.2, R1, 0.2, 0.3]]],
  ['left_jab_right_uppercut', 'Space+L', 1.0, [[0.3, L1, 0.2, 0.0], [0.58, R1, 0.22, 0.69]]],
  ['left_side_kick', 'YY', 1.0, [[1.1, L3, 0.3, 0.5]]],
  ['left_front_kick', 'HH', 1.0, [[1.0, L3, 0.2, 0.5]]],
  ['right_side_kick', 'UU', 1.0, [[1.14, R4, 0.4, 0.15]]],
  ['right_knee', 'JJ', 1.0, [[0.65, R4, 0.4, 0.15]]],
  ['six_punch', 'Space+Y', 1.12, [[0.161, L1, 0.2, 0.1], [0.353, R1, 0.2, 0.1], [0.588, L1, 0.2, 0.1],
    [0.817, R1, 0.2, 0.1], [1.0, L1, 0.2, 0.1], [1.26, R1, 0.2, 0.1], [1.57, L1, 0.2, 0.1],
    [1.7, R1, 0.2, 0.1], [2.0, L1, 0.2, 0.1], [2.343, R1, 0.2, 0.1]]],
  ['run_and_punch', 'Space+U', 1.0, [[1.8, R1, 0.4, 0.15]]],
  ['left_right_jab', ';', 1.0, [[0.3, L1, 0.4, 0.15]]],
  ['left_right_hook', "'", 1.0, [[0.3, L1, 0.4, 0.15]]],
  ['left_hook_right_jab', 'Space+K', 1.0, [[0.3, L1, 0.4, 0.15]]],
  ['double_hook', 'Space+H', 1.0, [[0.596, L1, 0.4, 0.15], [0.843, R1, 0.4, 0.15]]],
  ['butt_smack_emote', 'Space+I', 1.0, [[0.14, L1, 0.43, 11.6]]],
];

const MOVES = Object.freeze(RAW.map(([name, keys, speed, impacts], index) => Object.freeze({
  index, name, keys, category: MOVE_TO_CATEGORY[index], speed,
  duration: DURATION_TICKS[index] / 50,
  kick: impacts[0][1] === L3 || impacts[0][1] === R4,
  impacts: Object.freeze(impacts.map(([clip, limb, lead, release]) => Object.freeze({
    t: clip / speed, limb, lead: lead / speed, release: release / speed,
    kick: limb === L3 || limb === R4, side: limb === L1 || limb === L3 ? 'left' : 'right',
  }))),
  maxPoints: impacts.reduce((p, e) => p + (e[1] === L3 || e[1] === R4 ? 2 : 1), 0),
})));

function byName(name) { const m = MOVES.find(x => x.name === name); if (!m) throw new Error(`unknown move ${name}`); return m; }
function byCategory(category) { return MOVES[CATEGORY_TO_MOVE[category - 16]]; }

// Held categories 0..15 as {forward, strafe, yaw} unit commands (A/Q positive).
const HELD = Object.freeze([
  null, [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
].map(v => v && Object.freeze({forward: v[0], strafe: v[1], yaw: v[2]})));
const HELD_NAMES = Object.freeze(['hold', 'release', 'W', 'S', 'A', 'D', 'Q', 'E', 'WQ', 'WE', 'SQ', 'SE', 'AQ', 'AE', 'DQ', 'DE']);

function heldCategory(forward, strafe, yaw) {
  // Keyboard has no diagonal translation: forward/back wins over strafe.
  const f = Math.sign(forward), s = f ? 0 : Math.sign(strafe), y = Math.sign(yaw);
  for (let c = 1; c < 16; c++) { const h = HELD[c]; if (h.forward === f && h.strafe === s && h.yaw === y) return c; }
  throw new Error('unreachable held combination');
}

module.exports = {CATEGORY_TO_MOVE, MOVE_TO_CATEGORY, DURATION_TICKS, MOVES, byName, byCategory, HELD, HELD_NAMES, heldCategory};
