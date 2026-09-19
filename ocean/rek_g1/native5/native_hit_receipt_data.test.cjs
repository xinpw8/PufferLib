'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const {nativeFilenameTime,chooseNativeFile,verifyCaptureBinding,precedingPose,packetAssociation,validatePacket,validateWirePacket,receiptGeometry,associate,scan}=require('./native_hit_receipt_data.cjs');
const SHA='a'.repeat(64);
const pose=(time,frame,line)=>({source_line:line,unity_time:time,unity_unscaled_time:time,unity_frame:frame,qpc_ticks:time*1e7,
  local_slot:0,round_number:1,fighters:[{slot:0,position_xyz:[0,.8,0],rotation_xyzw:[0,0,0,1]},
    {slot:1,position_xyz:[1,.8,0],rotation_xyzw:[0,0,0,1]}]});
const hit=(line=10)=>({event:'raw_hit_packet',source_line:line,unity_time:1,unity_unscaled_time:1,unity_frame:60,
  client_fixed_tick_at_observation:500,monotonic_receipt_time:1.01,wire_body_sha256:SHA,
  decoded:{position_xyz:[.5,.9,0],surface_normal_xyz:[1,0,0],relative_speed:3,is_kick:1}});
const score=(line=11)=>({...hit(line),event:'raw_score_packet',decoded:{fighter_index:0,new_hit_count:2,points_awarded:2}});
test('capture selection is exact, unambiguous and excludes partial files',()=>{
  const name='rek-private-ai-root-motion-20260919T042827.2189090Z-pid32-71a92c83cedf4f1ea8be904a25a60cbd.jsonl';
  assert.equal(nativeFilenameTime(name),Date.parse('2026-09-19T04:28:27.218Z'));
  assert.equal(chooseNativeFile([name,name+'.partial'],'2026-09-19T04:28:27.315Z'),name);
  assert.throws(()=>chooseNativeFile([name,name.replace('71a9','81a9')],'2026-09-19T04:28:27.315Z'),/not_unique/);
});
test('native nearest-prior join excludes future line and incompatible receipt clocks',()=>{
  const a=pose(.98,59,1),b=pose(1,60,12),e=hit();
  assert.equal(precedingPose([a,b],e,'native').source_line,1);
  b.source_line=9;b.unity_unscaled_time=1.1;assert.equal(precedingPose([a,b],e,'native').source_line,1);
  b.unity_unscaled_time=1;assert.equal(precedingPose([a,b],e,'native').source_line,9);
  assert.equal(precedingPose([pose(.8,50,1)],e,'native'),null);
});
test('relay same-frame pose is not accepted as preceding a network receipt',()=>{
  const p=precedingPose([pose(.98,59,1),pose(1,60,2)],hit(),'relay');
  assert.equal(p.unity_frame,59);assert.equal(p.receipt_minus_pose_unity_seconds,1-.98);
});
test('score association requires shared frame/tick and nearby receipt time',()=>{
  assert.equal(packetAssociation(hit(),score()),true);
  assert.equal(packetAssociation(hit(),{...score(),unity_frame:61}),false);
  assert.equal(packetAssociation(hit(),{...score(),client_fixed_tick_at_observation:501}),false);
  assert.equal(packetAssociation(hit(),{...score(),monotonic_receipt_time:1.1}),false);
});
test('ambiguous score/hit associations retain unknown causal and execution labels',()=>{
  const p=[pose(.98,59,1)],a=associate('policy/test','session-03',[hit(),score(),score(12)],p,p);
  assert.equal(a.hits[0].receipt_association,'ambiguous_noncausal');assert.equal(a.hits[0].associated_score_receipt_ids.length,2);
  for(const r of [...a.hits,...a.scores])for(const key of ['attacker','defender','active_clip','executed_move_index','causal_request_id','server_clock'])assert.equal(r[key],null);
  assert.equal(a.hits[0].causal_score_event_id,null);
  const unique=associate('policy/test','session-03',[hit(),score()],p,p).hits[0];assert.equal(unique.receipt_association,'unique_same_frame_noncausal');
  assert.equal(unique.causal_score_event_id,null);
});
test('packet values and unassigned fighter-relative geometry survive export',()=>{
  validatePacket(hit());validatePacket(score());assert.throws(()=>validatePacket({...hit(),decoded:{...hit().decoded,relative_speed:NaN}}),/invalid_hit/);
  const g=receiptGeometry(hit().decoded,pose(.98,59,1));assert.equal(g.length,2);assert.equal(g[0].role,'unknown');
  assert.equal(g[0].contact_minus_root_forward_lateral_vertical[0],.5);assert.equal(g[1].contact_minus_root_forward_lateral_vertical[0],-.5);
});
test('packed wire bytes, their SHA and reported decoded fields must agree',()=>{
  const r=hit(),body=Buffer.alloc(29);[...r.decoded.position_xyz,...r.decoded.surface_normal_xyz,r.decoded.relative_speed].forEach((v,i)=>body.writeFloatLE(v,i*4));
  body.writeUInt8(r.decoded.is_kick,28);Object.assign(r,{wire_body_base64:body.toString('base64'),wire_body_bytes:29,
    wire_body_sha256:crypto.createHash('sha256').update(body).digest('hex')});validateWirePacket(r);
  assert.throws(()=>validateWirePacket({...r,wire_body_sha256:SHA}),/hash_or_size/);
  assert.throws(()=>validateWirePacket({...r,decoded:{...r.decoded,relative_speed:4}}),/disagrees_with_wire/);
  const s=score(),bytes=Buffer.alloc(7);bytes.writeUInt8(0,0);bytes.writeInt16LE(2,1);bytes.writeFloatLE(2,3);
  Object.assign(s,{wire_body_base64:bytes.toString('base64'),wire_body_bytes:7,wire_body_sha256:crypto.createHash('sha256').update(bytes).digest('hex')});validateWirePacket(s);
  assert.throws(()=>validateWirePacket({...s,decoded:{...s.decoded,fighter_index:1}}),/disagrees_with_wire/);
});
test('native/relay binding rejects wrong paired geometry and preserves clock checks',()=>{
  const native=Array.from({length:101},(_,i)=>pose(10+i*.02,100+i,i+1));
  const relay=native.map(p=>({...p,utc:new Date(1000000+p.unity_time*1000).toISOString(),qpc_frequency_hz:1e7,round_identity_sha256:SHA}));
  const start={pid:32,utc:new Date(1000000+10000).toISOString(),stopwatch_timestamp_ticks:1e8,stopwatch_frequency_hz:1e7};
  const end={stopwatch_timestamp_ticks:12e7};assert.equal(verifyCaptureBinding(native,relay,start,end).matching_anchors,3);
  const wrong=relay.map(p=>({...p,fighters:p.fighters.map(f=>({...f,position_xyz:[100,100,100]}))}));
  assert.throws(()=>verifyCaptureBinding(native,wrong,start,end),/binding_missing/);
  assert.throws(()=>verifyCaptureBinding(native,relay,{...start,stopwatch_frequency_hz:1},end),/clock_missing/);
});
test('source SHA binds exact bytes and concurrent mutation is rejected',async()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-hit-audit-')),file=path.join(directory,'fixture.jsonl'),bytes=Buffer.from('{"event":"test"}\n');
  try{
    fs.writeFileSync(file,bytes);const result=await scan(file,()=>{});
    assert.equal(result.sha256,crypto.createHash('sha256').update(bytes).digest('hex'));assert.equal(result.bytes,bytes.length);
    await assert.rejects(()=>scan(file,()=>fs.appendFileSync(file,'\n')),/source_changed_during_read/);
  }finally{fs.unlinkSync(file);fs.rmdirSync(directory);}
});
