'use strict';
// Public outcome-only export for the compact GPU worker. No private checkpoint paths.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const {League} = require('./league.cjs');
const {publicStanding} = require('./public_result.cjs');
const [configPath, output] = process.argv.slice(2);
assert(configPath && output,'Usage: export_compact_results.cjs PRIVATE_CONFIG NEW_OUTPUT_JSON');
const config = JSON.parse(fs.readFileSync(configPath,'utf8'));
const backend = config.backends.find(item=>item.id==='semantic_cuda');
assert(backend);
const worker = JSON.parse(fs.readFileSync(backend.workerConfig,'utf8'));
assert.equal(worker.round_seconds,20);
const league = new League({file:config.leagueFile});
const state = league.snapshot();
const matches = Object.values(state.matches).filter(match=>match.backend===backend.id && match.configHash===backend.configHash);
const result = {
  schema:'rek-compact-native-policy-league-results-v1',exportedAt:new Date().toISOString(),
  execution:'Headless native CUDA arena transitions and CUDA BF16 policy inference. Node orchestrates complete matches; CPU physics, rendering, Python and PPO updates are absent from this tournament.',
  backend:backend.id,configHash:backend.configHash,
  protocol:{roundSeconds:20,tickSeconds:0.02,geometry:'Fixed starts; seeds vary sampled policy action randomness',
    seeds:[...new Set(matches.map(match=>match.seed))].sort((a,b)=>a-b),bothSides:true,shapingWeight:0,
    ranking:'Complete side-reversed pairs ranked by the Wilson lower bound on observed wins; draws remain nonwinning.'},
  policies:league.opponentOptions({backend:backend.id,configHash:backend.configHash}),
  matches:matches.map(match=>({id:match.id,backend:match.backend,configHash:match.configHash,players:match.players,seed:match.seed,
    status:match.status,reason:match.reason,scores:match.scores,durationMs:match.durationMs,winnerPolicyId:match.winnerPolicyId,
    roundResult:match.roundResult,winnerSide:match.winnerSide})),
  standings:league.standings({backend:backend.id,configHash:backend.configHash}).map(publicStanding),
  note:'This is the named three-opponent 20-second league, not a training-speed measurement or general human-strength claim. The separate diverse frozen suite measures other starts, opponents and 300-second conditions.'
};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({matches:result.matches.length,completed:result.matches.filter(match=>match.status==='completed').length,
  standings:result.standings.map(row=>({id:row.id,rank:row.rank,wins:row.paired.wins,losses:row.paired.losses,draws:row.paired.draws}))}));
