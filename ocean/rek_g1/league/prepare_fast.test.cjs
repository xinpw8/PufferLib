'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {prepare}=require('./prepare_fast.cjs');
function fixture(t){
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-fast-config-'));
  t.after(()=>fs.rmSync(directory,{recursive:true,force:true}));
  const input=path.join(directory,'runtime.json'),binary=path.join(directory,'rek-eval-worker');
  const value={backend:'semantic_cuda',model_path:path.join(directory,'model.xml'),
    assets_path:directory,motion_features_path:directory,round_seconds:20};
  for(const name of ['rek-eval-worker','fast_runtime.o','fast_assets.o','model.xml',
    'semantic_duel_assets_manifest.json','foot_features_manifest.json'])
    fs.writeFileSync(path.join(directory,name),`fixture ${name}`);
  fs.writeFileSync(input,JSON.stringify(value));
  return {directory,input,binary,value,save:()=>fs.writeFileSync(input,JSON.stringify(value))};
}
test('compact config preserves other ports, starts only scripted and sets every runtime parameter',t=>{
  const f=fixture(t),result=prepare(path.join(f.directory,'new-run'),f.binary,f.input);
  assert.equal(result.port,18769);
  const config=JSON.parse(fs.readFileSync(result.configPath));
  const backend=config.backends[0],worker=JSON.parse(fs.readFileSync(backend.workerConfig));
  assert.equal(worker.arenas,1);assert.equal(worker.backend,'semantic_cuda');
  assert.equal(worker.move_duration_ticks.length,17);
  assert.equal(backend.env.REK_FAST_MOVE_SPEED,'1');assert.equal(backend.env.REK_FAST_BODY_RADIUS,'0.22');
  assert.match(backend.warning,/parity are unverified/);
  const league=JSON.parse(fs.readFileSync(config.leagueFile));
  assert.deepEqual(Object.keys(league.policies),['semantic_cuda/scripted']);
  assert.throws(()=>prepare(path.join(f.directory,'new-run'),f.binary,f.input),/EEXIST/);
});
test('root-speed, timing and runtime changes each produce a distinct configuration identity',t=>{
  const f=fixture(t),hashes=[];
  const create=()=>hashes.push(prepare(path.join(f.directory,`run${hashes.length}`),f.binary,f.input).configHash);
  create();f.value.fast={move_speed:1.2};f.save();create();
  f.value.round_seconds=120;f.save();create();
  fs.appendFileSync(path.join(f.directory,'fast_runtime.o'),'changed runtime');create();
  assert.equal(new Set(hashes).size,4);
});
test('incompatible selectors and invalid behavior settings fail before creating a run',t=>{
  const f=fixture(t),run=path.join(f.directory,'invalid');
  f.value.backend='mujoco';f.save();assert.throws(()=>prepare(run,f.binary,f.input),/semantic_cuda/);
  f.value.backend='semantic_cuda';f.value.fast={move_speed:-1};f.save();
  assert.throws(()=>prepare(run,f.binary,f.input),/fast.move_speed/);
  assert.equal(fs.existsSync(run),false);
});
