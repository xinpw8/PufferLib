'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {EventEmitter}=require('node:events'),{PassThrough}=require('node:stream');
const {NativeWorker}=require('./worker.cjs');
function transport(renderOnly){
  const child=new EventEmitter();child.stdout=new PassThrough();child.stdin=new PassThrough();
  child.kills=[];child.kill=signal=>{child.kills.push(signal);child.emit('exit',null,signal);};
  let args;
  const worker=new NativeWorker({executable:'test',config:'fixture.json',renderOnly,
    spawnProcess:(executable,argv)=>{args=argv;return child;}});
  return {worker,child,args,reply:value=>child.stdout.write(JSON.stringify(value)+'\n')};
}
test('renderer role is requested and mismatched ready fails before any rendering request',async()=>{
  const r=transport(true);assert.deepEqual(r.args,['--render-only','--config','fixture.json']);
  const reject=assert.rejects(r.worker.ready,/role mismatch/);r.reply({event:'ready'});await reject;
  assert.equal(r.worker.closed,true);assert.deepEqual(r.child.kills,['SIGTERM']);
});
test('renderer timeout closes only that transport and cannot accumulate later queued work',async()=>{
  const render=transport(true),physics=transport(false);
  render.reply({event:'ready',rendererOnly:true});physics.reply({event:'ready'});
  await assert.rejects(render.worker.request('frame',{qpos:Array(72).fill(0),snapshotTick:0,generation:1},10),/timeout/);
  assert.equal(render.worker.closed,true);assert.deepEqual(render.child.kills,['SIGTERM']);
  await assert.rejects(render.worker.request('frame'),/unavailable/);assert.equal(render.worker.pending.size,0);
  const step=physics.worker.request('step');await Promise.resolve();physics.reply({id:1,ok:true,state:{tick:1}});
  assert.equal((await step).state.tick,1);assert.equal(physics.worker.closed,false);assert.equal(physics.child.kills.length,0);
  physics.worker.fail(Error('test cleanup'));physics.child.stdout.end();render.child.stdout.end();
});
