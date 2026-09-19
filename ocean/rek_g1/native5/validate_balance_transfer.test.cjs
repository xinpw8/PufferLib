'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {validateModel,validateSessionSeparation,strata,evaluate,pool,SCHEMA}=require('./validate_balance_transfer.cjs');
const {FEATURE_NAMES,transitions,evaluateOneStep,evaluateOpenLoop}=require('./balance_transition_data.cjs');
function model(){return {lambda:.01,fit_rows:100,mean:Array(FEATURE_NAMES.length).fill(0),scale:Array(FEATURE_NAMES.length).fill(1),weights:Array.from({length:FEATURE_NAMES.length+1},()=>Array(5).fill(0))};}
function capture(){return {id:'r1',session:'new',local_slot:0,counts:[],rows:Array.from({length:55},(_,i)=>({t:i*.02,unity_time:i*.02,active:true,local_slot:0,sequence:i,
  actors:[{slot:0,state:[i*.002,0,.8,0,0],opponent:[1,0,.8,0,0],pose:Array(12).fill(0)},{slot:1,state:[1,0,.8,0,0],opponent:[i*.002,0,.8,0,0],pose:Array(12).fill(0)}],
  command:[1,0,0],requested_attack:false,requested_move_index:null,request_age:3,point_age:[3,3],falls:[0,0]}))};}
test('whole process groups remain held out even across adjacent rounds',()=>{
  const prior={session_groups:[{id:'old',origin_utc:new Date(10000).toISOString()}]};
  const c=[{id:'r1',origin:100000,pid:3},{id:'r2',origin:100090,pid:3}];
  const groups=validateSessionSeparation(c,prior);assert.equal(groups.length,1);assert.deepEqual(groups[0].captures,['r1','r2']);
  assert.throws(()=>validateSessionSeparation([{id:'leak',origin:10100,pid:99}],prior),/historical_session_overlap/);
  assert.throws(()=>validateSessionSeparation([{id:'a',origin:100000,pid:3},{id:'b',origin:100010,pid:4}],prior),/multiple_process/);
});
test('referee strata use explicit intervals despite zero visual fall counters',()=>{
  const c=capture();c.counts=[{slot:0,start:.3,end:.5},{slot:1,start:.8,end:1}];
  assert.deepEqual(strata(c,.2,.4),['all','local_count_overlap']);
  assert.deepEqual(strata(c,.4,.9),['all','local_count_overlap','opponent_count_overlap']);
  assert.deepEqual(strata(c,.55,.75),['all','no_count_overlap']);
  assert.equal(SCHEMA.no_fitting,true);
});
test('frozen model validation rejects malformed scale and changed regularization',()=>{
  validateModel(model());const a=model();a.scale[2]=0;assert.throws(()=>validateModel(a),/model_values/);
  const b=model();b.lambda=.02;assert.throws(()=>validateModel(b),/unsupported_frozen/);
});
test('constant root dynamics are predicted across an external sequence',()=>{
  const m=model();m.weights[0][0]=.1;const r=pool([evaluate(capture(),m)]);
  assert.ok(r['1'].all.ridge.planar_rmse_unity_units<1e-10);assert.ok(r['1'].all.frozen_root.planar_rmse_unity_units>.09);
});
test('open loop does not consume future measured poses or opponent roots',()=>{
  const c=capture(),m=model();m.weights[11][0]=.01;
  const before=pool([evaluate(c,m)])['1'];
  for(let i=2;i<c.rows.length;i++){c.rows[i].actors[0].pose=Array(12).fill(100);c.rows[i].actors[0].opponent=[100,100,100,0,0];}
  assert.deepEqual(pool([evaluate(c,m)])['1'],before);
});
test('timing gaps cannot be crossed by one-second rollout windows',()=>{
  const c=capture();for(let i=26;i<c.rows.length;i++)c.rows[i].t+=.5;
  assert.equal(pool([evaluate(c,model())])['1'].all.ridge.count,0);
});
test('unstratified metrics match the established evaluator exactly',()=>{
  const c=capture(),m=model();m.weights[0][0]=.07;const rows=transitions(c),actual=pool([evaluate(c,m)]);
  assert.deepEqual(actual.one_step.all,evaluateOneStep(rows,m));
  const reference=evaluateOpenLoop([c],rows,m);for(const h of ['0.1','0.25','0.5','1'])assert.deepEqual(actual[h].all,reference[h]);
});
