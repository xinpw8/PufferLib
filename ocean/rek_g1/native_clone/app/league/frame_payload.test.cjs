'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const {framePayload}=require('./frame_payload.cjs');

test('recorder and server reuse the same decoded bytes/hash without changing wire reply',()=>{
  const bytes=Buffer.from([137,80,78,71,13,10,26,10,0,1,2,3]);
  const reply={id:7,png:bytes.toString('base64'),snapshotTick:4,generation:2};
  const before=JSON.stringify(reply),recorded=framePayload(reply),published=framePayload(reply);
  assert.equal(recorded,published);assert.equal(recorded.png,published.png);
  assert.deepEqual(published.png,bytes);assert.equal(published.sha256,crypto.createHash('sha256').update(bytes).digest('hex'));
  assert.equal(JSON.stringify(reply),before);
});
test('distinct or changed replies never return an older PNG',()=>{
  const reply={png:Buffer.from('old').toString('base64')};const old=framePayload(reply);
  reply.png=Buffer.from('new').toString('base64');const next=framePayload(reply);
  assert.notEqual(old,next);assert.equal(next.png.toString(),'new');
  assert.notEqual(framePayload({...reply}),next);assert.throws(()=>framePayload({png:null}),/string required/);
});
