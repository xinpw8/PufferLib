'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {rootState,rate,advance,features,groupSessions,transitions,fitRidge,predict,evaluateOneStep,evaluateOpenLoop,highTiltWindows,requestMotion,FEATURE_NAMES,SCHEMA}=require('./balance_transition_data.cjs');
const close=(a,b,tol=1e-9)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
function capture(count=150) {
  const rows=Array.from({length:count},(_,i)=>({t:i*.02,active:true,local_slot:0,sequence:i,
    actors:[{slot:0,state:[i*.002,0,.8,0,0],opponent:[1,0,.8,0,0],pose:Array(12).fill(0)},
      {slot:1,state:[1,0,.8,0,0],opponent:[i*.002,0,.8,0,0],pose:Array(12).fill(0)}],
    command:[1,0,0],requested_attack:false,requested_move_index:null,desired_action:2,request_age:3,point_age:[10,10],points:[0,0],falls:[0,0]}));
  return {id:'test',session:'session-01',origin:100000,rows};
}
test('process launches are the split unit; adjacent rounds share a group',()=>{
  const captures=[{id:'a',origin:100000},{id:'b',origin:100090},{id:'c',origin:400000}];
  const groups=groupSessions(captures);assert.equal(groups.length,2);
  assert.equal(captures[0].session,captures[1].session);assert.notEqual(captures[0].session,captures[2].session);
  assert.throws(()=>groupSessions([{id:'missing'}]),/missing_process_clock_origin/);
});
test('root transform and measured finite differences respect Unity axes',()=>{
  assert.deepEqual(rootState({root_position_xyz:[1,2,3],root_rotation_xyzw:[0,0,0,1]}),[1,3,2,0,0]);
  const before=[0,0,.8,0,.1],now=[.02,.01,.81,.02,.12];
  const velocity=rate(before,now,.02);
  const outgoing=[1,.5,.5,1,1];const actual=advance(before,outgoing,.02);
  actual.forEach((value,i)=>close(value,now[i]));assert.ok(velocity.every(Number.isFinite));
});
test('transition exporter preserves unknown execution and rejects timing gaps',()=>{
  const c=capture(7);let rows=transitions(c);assert.equal(rows.length,5);
  assert.equal(rows[0].x.length,FEATURE_NAMES.length);close(rows[0].target[0],.1);
  assert.equal(rows[0].executed_move_index,null);
  c.rows[3].t=c.rows[2].t;rows=transitions(c);assert.equal(rows.length,3);
  c.rows[6].active=false;assert.equal(transitions(c).length,2);
  assert.equal(SCHEMA.contact_label,null);assert.equal(SCHEMA.fall_label,null);
});
test('private transition JSON round-trips without nonfinite missing-history ages',()=>{
  const c=capture(7);for(const row of c.rows)row.point_age=[Infinity,Infinity];
  const row=transitions(c)[0],decoded=JSON.parse(JSON.stringify(row));
  assert.deepEqual(decoded.point_age,[3,3]);assert.deepEqual(features(decoded),row.x);
});
test('ridge predicts held-out constant root motion and beats persistence',()=>{
  const train=transitions(capture()),held=transitions({...capture(),session:'session-02'});
  const model=fitRidge(train);const metrics=evaluateOneStep(held,model);
  assert.ok(metrics.ridge.planar_rmse_unity_units<1e-10);
  assert.ok(metrics.frozen_root.planar_rmse_unity_units>.001);
  assert.equal(model.fit_rows,train.length);assert.equal(predict(model,features(held[0])).length,5);
});
test('open loop never consumes future pose or opponent measurements',()=>{
  const c=capture(),rows=transitions(c),model=fitRidge(rows),original=evaluateOpenLoop([c],rows,model);
  // Keep roots/targets and command schedule fixed. Corrupt only future exogenous
  // pose/opponent context, while retaining each window-start context unchanged.
  // A model with a pose coefficient exposes accidental future-pose consumption.
  assert.ok(original['1'].ridge.count>0);
  const model2={mean:Array(FEATURE_NAMES.length).fill(0),scale:Array(FEATURE_NAMES.length).fill(1),
    weights:Array.from({length:FEATURE_NAMES.length+1},()=>Array(5).fill(0))};
  model2.weights[1+10][0]=.01;
  const firstOnly=rows.filter(r=>r.index>=1&&r.index<=52),short={...c,rows:c.rows.slice(0,54)};
  const baseline=evaluateOpenLoop([short],firstOnly,model2)['1'];
  for(const row of firstOnly) if(row.index!==1){row.pose=Array(12).fill(100);row.opponent=[100,100,100,0,0];row.x=features(row);}
  const changed=evaluateOpenLoop([short],firstOnly,model2)['1'];assert.deepEqual(changed,baseline);
});
test('sustained geometric tilt episodes are separate from false fall flags',()=>{
  const c=capture();for(let i=30;i<=42;i++)c.rows[i].actors[0].state[4]=1.2;
  c.rows[70].actors[0].state[4]=1.3;
  const result=highTiltWindows(c);assert.equal(result.episodes,1);assert.equal(result.local_episodes,1);
  assert.equal(result.matched_windows.length,1);assert.match(result.definition,/not referee fall labels/);
});
test('request windows deduplicate persistent request context and retain overlap warning',()=>{
  const c=capture();for(let i=1;i<c.rows.length;i++)Object.assign(c.rows[i],{
    requested_move_qpc_ticks:i<20?100:200,requested_attack:true,requested_move_index:i<20?6:7,
    request_age:(i-(i<20?1:20))*.02});
  const result=requestMotion(c);assert.equal(result.all.windows,2);
  assert.equal(result.all.newer_request_within_window,1);
  assert.equal(result.by_requested_move.length,2);
  assert.match(result.interpretation,/not proof of playback/);
});
