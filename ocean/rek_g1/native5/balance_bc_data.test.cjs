'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {balance} = require('./balance_bc_data.cjs');
function fixture() {
  const b = Buffer.alloc(256 + 7 * 1056); b.write('REKBC001');
  for (const [offset, value] of [[8,1],[12,223],[16,33],[20,7],[24,1056]]) b.writeUInt32LE(value, offset);
  b.fill(1,32,255);
  for (let row=0;row<7;row++) {
    const p=256+row*1056;b.writeUInt32LE(row>=5?1:0,p);
    b.writeInt32LE([1,1,2,21,-1,1,21][row],p+12);b.writeFloatLE(row===4?0:1,p+16);
  }
  return b;
}
test('balance modifies only labeled training weights with mean1 and equal group mass',()=>{
  const source=fixture(),before=Buffer.from(source),{output,manifest}=balance(source);
  assert.deepEqual(source,before);assert.equal(manifest.changed_training_weights,4);
  assert.deepEqual(manifest.groups.map(g=>g.training_count),[2,1,1]);
  for(const g of manifest.groups) assert.ok(Math.abs(g.training_weight_mass-4/3)<1e-6);
  assert.ok(Math.abs(manifest.observed_mean_weight-1)<1e-7);
  for(let i=0;i<source.length;i++)if(source[i]!==output[i]) {
    const row=Math.floor((i-256)/1056),within=(i-256)%1056;
    assert.ok(row>=0&&row<4&&within>=16&&within<20);
  }
  assert.deepEqual(output.subarray(256+5*1056),source.subarray(256+5*1056));
});
test('heldout label distribution cannot affect training group weights',()=>{
  const source=fixture(),changed=Buffer.from(source);changed.writeInt32LE(2,256+5*1056+12);
  changed.writeFloatLE(17,256+6*1056+16);
  assert.deepEqual(balance(source).manifest.groups,balance(changed).manifest.groups);
});
test('unweighted original and all approved training groups required',()=>{
  const source=fixture();source.writeFloatLE(2,256+16);assert.throws(()=>balance(source),/unweighted/);
  const missing=fixture();missing.writeInt32LE(1,256+2*1056+12);assert.throws(()=>balance(missing),/all training groups/);
  const hold=fixture();hold.writeInt32LE(0,256+12);assert.throws(()=>balance(hold),/outside approved/);
});
test('reject malformed layout and inconsistent unlabeled rows',()=>{
  assert.throws(()=>balance(fixture().subarray(0,100)),/truncated/);
  assert.throws(()=>balance(Buffer.concat([fixture(),Buffer.from([0])])),/byte length/);
  const source=fixture();source.writeFloatLE(1,256+4*1056+16);assert.throws(()=>balance(source),/disagreement/);
});
