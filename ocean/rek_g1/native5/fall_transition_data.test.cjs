'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {buildRows,FEATURE_ORDER}=require('./fall_transition_data.cjs');
function observations({label=true,call='Slip',gap=false,moveAfter=1}={}) {
  return Array.from({length:81},(_,i)=>{
    const t=i*.05,down=label&&t>=1.2&&t<2.2;
    return {t,round:'fixture-round',active:true,slot:0,valid:!(gap&&i===25),
      states:[[0,0,.7,0,1.5],[1,0,.7,0,0]],pose:[Array(12).fill(0),Array(12).fill(0)],
      command:[0,0,0],requested_move:t>1.1?moveAfter:1,requested_time:.1,
      receipt_time:t,receipt_sequence:i+1,lifecycle:1,count_mask:down?1:0,
      call_type:down?0:4,call_faller:0,call_sequence:t<1.2?0:t<2.2?1:2,
      call_censored:false,call_name:t<1.2?null:t<2.2?call:'Knockout'};
  });
}
test('explicit count onset is one future event, repeated latch is not another event',()=>{
  const result=buildRows(observations(),'fixture');
  assert.equal(result.onsets.length,1);assert.equal(result.rows.reduce((n,r)=>n+r.y[0],0),1);
  assert.equal(result.rows.reduce((n,r)=>n+r.y[1],0),0);
  assert.equal(result.episodes.length,1);assert.equal(result.episodes[0].explicit_countout,true);
  assert.ok(Math.abs(result.episodes[0].duration-1)<1e-9);
  assert.ok(result.rows.every(r=>r.x.length===FEATURE_ORDER.length));
});
test('high observed tilt and Knockout call alone never manufacture a fall label',()=>{
  const result=buildRows(observations({label:false}),'fixture');
  assert.equal(result.onsets.length,0);assert.ok(result.rows.every(r=>r.y.every(y=>y===0)));
});
test('uncorroborated count rises censor the forecast instead of becoming negatives',()=>{
  const result=buildRows(observations({call:'Knockout'}),'fixture');
  assert.equal(result.onsets.length,0);assert.equal(result.issues.unconfirmed_count_onset,1);
  assert.ok(result.rows.every(r=>!(r.t<1.2&&r.t+.5>=1.2)));
});
test('missing observation in the future window censors its label',()=>{
  const result=buildRows(observations({gap:true}),'fixture');
  assert.ok(result.rows.every(r=>!(r.t<1.25&&r.t+.5>=1.25)));
});
test('future changed request does not leak into earlier features',()=>{
  const first=buildRows(observations({moveAfter:3}),'fixture').rows.filter(r=>r.t<=1.1);
  const second=buildRows(observations({moveAfter:14}),'fixture').rows.filter(r=>r.t<=1.1);
  assert.deepEqual(first,second);
});
