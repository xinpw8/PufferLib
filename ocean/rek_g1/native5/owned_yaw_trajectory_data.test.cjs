'use strict';
const assert=require('node:assert/strict');
const test=require('node:test');
const {HEADER,ROW,OBS,ACTIONS}=require('./authentic_trajectory_data.cjs');
const {LEGACY,desiredYaw,rowEvidence}=require('./owned_yaw_trajectory_data.cjs');
function fixture(busy=true,desired=6) {
  const b=Buffer.alloc(HEADER+ROW);b.writeUInt32LE(7,HEADER+1088);
  const obs=Array(OBS).fill(0),mask=Array(ACTIONS).fill(1);obs[182]=obs[183]=Number(busy);
  obs.forEach((v,j)=>b.writeFloatLE(v,HEADER+32+4*j));mask.forEach((v,j)=>b.writeFloatLE(v,HEADER+924+4*j));
  const request={seq:7,round_id:'round',observation_schema:LEGACY,type:'step',terminal:false,observation:obs,mask};
  const source={event:'g1_policy_state',observation_sequence:7,round_identity_sha256:'round',round:{active:true},stream_active:true,
    input:{active:true,desired_action:desired},clock:{qpc_ticks:100,qpc_frequency_hz:1000}};
  const encoded={event:'policy_observation',ready:true,worker_request:request,provenance:{source_qpc_ticks:100,
    source_qpc_frequency_hz:1000,stream_active:true,projected_busy:busy,busy_projection:'dispatched_request_v4_duration'}};
  return {b,source,encoded,worker:structuredClone(request)};
}
const run=f=>rowEvidence(f.b,0,f.source,f.encoded,f.worker,'round');
test('all owned desired categories have exact yaw and busy suppression',()=>{
  const expected=[0,0,0,0,0,1,-1,1,-1,1,-1,1,-1,1,-1];
  for(let category=1;category<=15;++category){assert.equal(desiredYaw(category),expected[category-1]);
    assert.equal(run(fixture(true,category)).owned_pending_yaw,expected[category-1]);
    assert.equal(run(fixture(false,category)).owned_pending_yaw,0);}
});
test('unknown, hold0, null and noninteger desired categories fail',()=>{
  for(const value of [0,null,undefined,16,-1,6.5]){const f=fixture();f.source.input.desired_action=value;assert.throws(()=>run(f),/desired_action/);}
});
test('busy projection must match both frozen feature values',()=>{
  const f=fixture();f.encoded.provenance.projected_busy=false;assert.throws(()=>run(f),/busy feature/);
  const g=fixture();delete g.encoded.provenance.projected_busy;assert.throws(()=>run(g),/provenance/);
});
test('active pre-action source identity, ownership, QPC and worker schema are required',()=>{
  for(const mutate of [f=>f.source.observation_sequence++,f=>f.source.stream_active=false,
    f=>f.source.round.active=false,f=>f.source.input.active=false,f=>f.encoded.provenance.source_qpc_ticks++,
    f=>f.worker.observation_schema='unknown',f=>f.worker.terminal=true]) {const f=fixture();mutate(f);assert.throws(()=>run(f));}
});
test('frozen observations, masks and positive zero187 are preserved',()=>{
  let f=fixture();f.worker.observation[12]=.1;assert.throws(()=>run(f),/observation/);
  f=fixture();f.worker.mask[0]=0;assert.throws(()=>run(f),/mask/);
  f=fixture();f.b.writeFloatLE(-0,HEADER+32+4*187);assert.throws(()=>run(f),/positive zero/);
});
test('desired yaw comes from current source, never the newly selected action',()=>{
  const f=fixture(true,7);f.worker.action=6;assert.equal(run(f).owned_pending_yaw,-1);
});
test('terminal_after and actor-weight zero do not erase active decision intent',()=>{
  const f=fixture(true,6);f.b.writeUInt32LE(1,HEADER+1112);f.b.writeFloatLE(0,HEADER+16);
  assert.equal(run(f).owned_pending_yaw,1);
});
