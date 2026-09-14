'use strict';
// Export outcome evidence only. Checkpoint paths, raw observations and assets
// remain in the private league directory.
const fs=require('node:fs');
const {League}=require('./league.cjs');
const {publicStanding}=require('./public_result.cjs');
const [configPath,out]=process.argv.slice(2);
if(!configPath||!out)throw new Error('Usage: export_results.cjs PRIVATE_CONFIG OUTPUT_JSON');
const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
const league=new League({file:config.leagueFile}),snapshot=league.snapshot();
const results={schema:'rek-native-policy-league-results-v1',exportedAt:new Date().toISOString(),
  execution:'Interactive and tournament CPU physics, CUDA controller/policy inference; zero PPO updates in tournaments',
  rounds:'20-second official clock; physics time includes measured reset pauses',
  policies:config.backends.flatMap(b=>league.opponentOptions({backend:b.id,configHash:b.configHash})),
  matches:Object.values(snapshot.matches).map(m=>({id:m.id,backend:m.backend,configHash:m.configHash,
    players:m.players,seed:m.seed,status:m.status,reason:m.reason,scores:m.scores,
    durationMs:m.durationMs,winnerPolicyId:m.winnerPolicyId,roundResult:m.roundResult,
    winnerSide:m.winnerSide})),
  standings:config.backends.flatMap(b=>league.standings({backend:b.id,configHash:b.configHash})).map(publicStanding)};
fs.writeFileSync(out,JSON.stringify(results,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({exported:out,matches:results.matches.length}));
