'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {HumanInput,commandFor,validateCommand}=require('./input.cjs');
test('continuous diagonal and yaw survive without magnitude normalization',()=>{
  assert.deepEqual(commandFor(['W','D','Q']),{forward:1,strafe:-1,yaw:1,moveIndex:-1,cancelAction:false});
  assert.equal(commandFor(['A']).strafe,1);
  assert.deepEqual(commandFor(['W','S','A','D','Q','E']),{forward:0,strafe:0,yaw:0,moveIndex:-1,cancelAction:false});
});
test('native scheduler receives attack while moving and busy, once per edge',()=>{
  const input=new HumanInput();input.update({seq:1,held:['W','D'],move:20});
  input.update({seq:2,held:['W','D'],move:23});
  const busy={raw:Array(446).fill(1),mask:Array(66).fill(0)};
  assert.equal(input.next(busy).moveIndex,4);assert.equal(input.next(busy).moveIndex,7);assert.equal(input.next(busy).moveIndex,-1);
  assert.equal(input.next(busy).strafe,-1);
});
test('stale packets cannot resurrect held controls or edges; release clears pending',()=>{
  const input=new HumanInput();input.update({seq:3,held:['Q'],move:21});input.release();
  assert.equal(input.update({seq:2,held:['W'],move:23}),false);assert.equal(input.next({}).forward,0);assert.equal(input.next({}).moveIndex,-1);
});
test('fractional commands validated verbatim and nonfinite refused',()=>{
  const c={forward:.3,strafe:-.2,yaw:.7,moveIndex:-1,cancelAction:false};assert.deepEqual(validateCommand(c),c);
  for(const v of [NaN,Infinity,1.1])assert.throws(()=>validateCommand({...c,yaw:v}));
});
