'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto'),path=require('node:path');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
test('published neighboring drift helper is the exact archived dependency and imports without native execution',()=>{
 const helper=require.resolve('../human-attackbc-20260924-r1/compare_live_drift.cjs');
 assert.equal(sha(helper),'f2eb2f490515eb1353e51850dbadc8815f7be1559ca3384c97b803a9ac1ab07d');
 const child=require('node:child_process'),original=child.spawnSync;
 child.spawnSync=()=>{throw Error('native execution forbidden in import test');};
 try{delete require.cache[require.resolve('./compare_live_drift.cjs')];assert.equal(typeof require('./compare_live_drift.cjs').movementMetrics,'function');}
 finally{child.spawnSync=original;}
});
test('published metrics agree with execution and failed-forward negative result',()=>{
 const metrics=JSON.parse(fs.readFileSync(path.join(__dirname,'training/metrics.json')));
 const execution=JSON.parse(fs.readFileSync(path.join(__dirname,'training/execution.json')));
 assert.equal(execution.exit_code,0);assert.equal(execution.epochs,5);assert.equal(execution.expected_updates,165);
 assert.equal(execution.checkpoint_sha256,'df2ffa1ef1b2dbf97c1d62f8371e6473e33f557f6d04d125974f0646fecda67f');
 assert.equal(metrics.epochs.at(-1).train_ce,4.10309545);assert.equal(metrics.epochs.at(-1).heldout_ce,4.78090129);
 for(const split of ['training','development']){
  const forward=metrics.per_action.find(r=>r.split===split&&r.epoch===5).actions.find(r=>r.action===2);
  assert.equal(forward.recall,0);assert.equal(forward.weight,split==='training'?499:932);
 }
 const {summarize}=require('./summarize_metrics.cjs');assert.throws(()=>summarize(Buffer.from('{}\n')),/pinned native metrics source/);
});
