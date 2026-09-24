'use strict';
const test=require('node:test'),assert=require('node:assert/strict');const {movementMetrics}=require('./compare_live_drift.cjs');
test('movement and yaw mass use original full legal distribution; conditional argmax separate',()=>{
 const requests=[{mask:Array(33).fill(1)}],rows=[{type:'drift_row',action:6,logits:Array(34).fill(0)}],m=movementMetrics(requests,rows);
 assert(Math.abs(m.mean_legal_movement_mass-14/33)<1e-15);assert.equal(m.mean_legal_action6_mass,1/33);assert.equal(m.mean_legal_action7_mass,1/33);assert.equal(m.sampled_movement,1);assert.equal(m.sampled_action6,1);assert.equal(m.conditional_unmasked_movement_argmax_counts[0],1);
});
test('runtime mask affects mass without changing unmasked direction diagnostic',()=>{
 const mask=Array(33).fill(0);mask[1]=mask[7]=1;const logits=Array(34).fill(0);logits[6]=9;
 const m=movementMetrics([{mask}],[{type:'drift_row',action:7,logits}]);assert.equal(m.mean_legal_movement_mass,.5);assert.equal(m.mean_legal_action6_mass,0);assert.equal(m.mean_legal_action7_mass,.5);assert.equal(m.conditional_unmasked_movement_argmax_counts[4],1);
});
