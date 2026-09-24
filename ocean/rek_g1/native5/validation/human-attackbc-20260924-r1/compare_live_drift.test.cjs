'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {probabilities,compare}=require('./compare_live_drift.cjs');
test('full and legal softmax retain all33 runtime classes and quantify attack mass',()=>{
 const z=Array(34).fill(0),all=Array(33).fill(1),p=probabilities(z,all);
 assert.equal(p.length,33);assert(Math.abs(p.slice(16).reduce((s,x)=>s+x,0)-17/33)<1e-15);
 const restricted=all.map((_,a)=>a===1||a===17?1:0),q=probabilities(z,restricted);assert.equal(q[1],.5);assert.equal(q[17],.5);assert.equal(q[21],0);
});
test('same frozen stream reports identity; changed conditional-neutral mass is visible',()=>{
 const requests=[{seq:9,round_id:'x',mask:Array(33).fill(1)}],base={type:'drift_row',seq:9,round_id:'x',action:21,recurrent_reset:true,logits:Array(34).fill(0)},actual=new Map([[9,{action:21}]]);
 const unchanged=compare(requests,[base],[base],actual);assert.equal(unchanged.baseline_matching_recorded_actions,1);assert.equal(unchanged.mean_legal_kl,0);assert.equal(unchanged.sampled_action_changes,0);
 const next={...base,action:17,logits:base.logits.map((z,a)=>a>=16&&a<=32?1:z)},changed=compare(requests,[base],[next],actual);
 assert(changed.mean_legal_kl>0);assert(changed.candidate_mean_legal_attack_mass>changed.baseline_mean_legal_attack_mass);assert.equal(changed.sampled_action_changes,1);
 assert.throws(()=>compare(requests,[base],[next],new Map([[9,{action:1}]])));
});
