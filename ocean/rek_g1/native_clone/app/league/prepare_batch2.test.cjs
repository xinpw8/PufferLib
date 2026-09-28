'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const {prepare}=require('../prepare.cjs');
const names=['REK_TEST_ENCODER_BATCH2','REK_TEST_DECODER_BATCH2','REK_TEST_ENCODER_BATCH8','REK_TEST_DECODER_BATCH8'];
const available=names.every(name=>process.env[name]);
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

test('real batch-2 preparation binds one arena, rejects wrong models, and preserves four-arena support',
  {skip:available?false:'Requires the four genuine exported ONNX paths; synthetic hashes are not substituted'},()=>{
  const [encoder2,decoder2,encoder8,decoder8]=names.map(name=>path.resolve(process.env[name]));
  for(const file of [encoder2,decoder2,encoder8,decoder8])assert.ok(fs.statSync(file).isFile());
  const actual={encoder2:sha(encoder2),decoder2:sha(decoder2),encoder8:sha(encoder8),decoder8:sha(decoder8)};
  assert.notEqual(actual.encoder2,actual.encoder8);assert.notEqual(actual.decoder2,actual.decoder8);
  console.log(JSON.stringify({event:'genuine_model_fixture',paths:{encoder2,decoder2,encoder8,decoder8},sha256:actual}));
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'rek-prepare-batch2-'));
  const binary=path.join(root,'fixture-executable'),model=path.join(root,'fixture-model.xml'),physics=path.join(root,'fixture-physics.json');
  // These files exercise preparation only. They are never executed or loaded;
  // the controller model files above are the actual published learned exports.
  fs.writeFileSync(binary,'CPU preparation fixture, not executable');fs.writeFileSync(model,'<mujoco/>');fs.writeFileSync(physics,'{}');
  const workerTemplate=path.join(root,'worker-template.json'),environmentFile=path.join(root,'environment.json');
  const worker={backend:'mujoco',arenas:1,seed:73,round_seconds:120,model_path:model,physics_export_path:physics,
    controller_encoder_path:encoder2,controller_decoder_path:decoder2};
  const env={REK_PHYSICS_BACKEND:'mujoco_cuda',REK_ALLOW_CPU_EVALUATION:'0',REK_PHYSICAL_OPPONENT:'recovered_bot1_g1_v1',REK_MUJOCO_DETERMINISTIC_SCALAR:'0'};
  const run=name=>path.join(root,name);
  function config(w=worker,e=env){fs.writeFileSync(workerTemplate,JSON.stringify(w));fs.writeFileSync(environmentFile,JSON.stringify(e));}
  function call(name){return prepare({workerTemplate,binary,environmentFile,runDirectory:run(name),port:18774});}
  try{
    config();call('one');
    const one=JSON.parse(fs.readFileSync(path.join(run('one'),'identity.json'),'utf8'));
    const prepared=JSON.parse(fs.readFileSync(path.join(run('one'),'worker.json'),'utf8'));
    assert.equal(one.execution.arenas,1);assert.equal(one.execution.controllerBatch,2);
    assert.equal(one.execution.aggregateArenaStepsMultiplier,1);assert.equal(one.execution.controlStepSeconds,.02);
    assert.equal(one.filePins[encoder2],actual.encoder2);assert.equal(one.filePins[decoder2],actual.decoder2);
    assert.equal(prepared.arenas,1);assert.equal(prepared.controller_encoder_path,encoder2);assert.equal(prepared.controller_decoder_path,decoder2);
    assert.deepEqual(one.env,env);
    for(const [label,patch] of [['wrong-encoder',{controller_encoder_path:encoder8}],['wrong-decoder',{controller_decoder_path:decoder8}],['swapped',{controller_encoder_path:decoder2,controller_decoder_path:encoder2}]]){
      config({...worker,...patch});assert.throws(()=>call(label),/verified batch-2/);assert.equal(fs.existsSync(run(label)),false);
    }
    for(const arenas of [2,0,'1']){
      config({...worker,arenas});assert.throws(()=>call('invalid-count'),/1 or 4 arenas/);assert.equal(fs.existsSync(run('invalid-count')),false);
    }
    config(worker,{...env,REK_ALLOW_CPU_EVALUATION:'1'});assert.throws(()=>call('cpu-refused'),/physical Bot1/);
    assert.equal(fs.existsSync(run('cpu-refused')),false);
    config({...worker,arenas:4,controller_encoder_path:encoder8,controller_decoder_path:decoder8});call('four');
    const four=JSON.parse(fs.readFileSync(path.join(run('four'),'identity.json'),'utf8'));
    assert.equal(four.worker.arenas,4);assert.equal(four.execution.controllerBatch,8);
    assert.equal(four.execution.aggregateArenaStepsMultiplier,4);assert.equal(four.filePins[encoder8],actual.encoder8);
    console.log(JSON.stringify({event:'prepare_contract_pass',one_arena_genuine:true,four_arena_genuine:true,
      rejected:['wrong encoder','wrong decoder','swapped models','arena2','arena0','string arena1','CPU environment'],native_executed:false}));
  }finally{
    assert.ok(root.startsWith(path.join(os.tmpdir(),'rek-prepare-batch2-')));fs.rmSync(root,{recursive:true,force:true});
  }
});
