'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {transition}=require('./derive_score_delta.cjs');
test('received one, two and five point awards keep the observed scale',()=>{
 for(const points of [1,2,5])assert.equal(transition(.04,0,0,points,0).reward,Math.fround(points/5));
 assert.equal(transition(.04,0,0,0,5).reward,-1);
 assert.equal(transition(.04,7,2,9,7).reward,Math.fround(-3/5));
});
test('same score gives zero reward without terminal bonus or potential',()=>{
 assert.equal(transition(.04,10,9,10,9).reward,0);
 assert.equal(transition(.04,5,13,5,13).reward,0);
});
test('actual interval determines five-second half-life, independent of decision count',()=>{
 assert.equal(transition(5,0,0,0,0).gamma,.5);
 assert.equal(transition(10,0,0,0,0).gamma,.25);
 assert.equal(transition(.04,0,0,0,0).gamma,Math.fround(2**(-.04/5)));
 assert.notEqual(transition(.02,0,0,0,0).gamma,transition(.04,0,0,0,0).gamma);
});
test('unclipped observed score and invalid inputs',()=>{
 assert.equal(transition(.1,0,0,20,0).reward,4);
 for(const dt of [0,-1,NaN,Infinity])assert.throws(()=>transition(dt,0,0,0,0));
 assert.throws(()=>transition(.04,2,0,1,0));
});
test('terminal rejected policy row still propagates measured score to preceding decisions',()=>{
 const rewards=[0,0,1],dt=[.02,.04,.03],weights=[1,1,0],values=Array(3);let next=0;
 for(let i=2;i>=0;i--){next=rewards[i]+(i===2?0:transition(dt[i],0,0,0,0).gamma*next);values[i]=next;}
 assert.equal(weights[2]*values[2],0);assert.equal(values[2],1);
 assert(Math.abs(values[0]-2**(-.06/5))<1e-7);assert(Math.abs(values[1]-2**(-.04/5))<1e-7);
 assert.equal(transition(.04,0,0,5,5).reward,0);
});
test('discount applies between reward-row source times and does not prediscount the current reward',()=>{
 const regular=Array(75).fill(.04),irregular=[.2,.7,1.1,1];
 const product=intervals=>intervals.reduce((p,dt)=>p*transition(dt,0,0,0,0).gamma,1);
 assert(Math.abs(product(regular)-2**(-3/5))<2e-6);assert(Math.abs(product(irregular)-2**(-3/5))<2e-7);
 assert.notEqual(2**(-2.98/5),2**(-2.96/5));
});
