'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {publicStanding}=require('./public_result.cjs');
test('public standings retain outcome evidence and exclude private checkpoint paths',()=>{
  const row={id:'policy-a',rank:1,paired:{wins:3,losses:1},checkpoint:{path:'/private/policy.bin',
    sha256:'abc',format:'native',trainingSteps:100,model:{hiddenSize:256}}};
  const output=publicStanding(row);
  assert.equal(output.checkpoint.path,undefined);assert.equal(row.checkpoint.path,'/private/policy.bin');
  assert.equal(output.paired.wins,3);assert.equal(output.checkpoint.sha256,'abc');
  assert.equal(publicStanding({id:'scripted',checkpoint:null}).checkpoint,null);
});
