'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const v=require('./human_command_data.cjs');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
function xml(){let body='',motor='';for(const side of ['player','opponent']){
  body+=`<body name="${side}__pelvis_0" quat="1 0 0 0">`;
  for(let i=1;i<30;i++){body+=`<body name="${side}__${v.BONES[i]}_${i}" quat="1 0 0 0"><joint name="${side}_j${i}" type="hinge" axis="0 1 0"/></body>`;motor+=`<motor joint="${side}_j${i}"/>`;}
  body+='</body>';
}return `<mujoco><worldbody>${body}</worldbody><actuator>${motor}</actuator></mujoco>`;}
const calibration=v.calibration(xml());
function wire(record,bytes){return {...record,wire_body_base64:bytes.toString('base64'),wire_body_sha256:hash(bytes)};}
function bone(n,slot=0,angle=0){
  const positions=[],rotations=[],b=Buffer.alloc(842);b[0]=slot;b[1]=30;
  for(let i=0;i<30;i++){
    positions.push(Math.fround(slot+Math.sin(n*.1)*.01),Math.fround(1+i*.01),Math.fround(i*.02));
    rotations.push(0,0,Math.fround(Math.sin((i?angle:0)/2)),Math.fround(-Math.cos((i?angle:0)/2)));
    for(let k=0;k<7;k++)b.writeFloatLE(k<3?positions[3*i+k]:rotations[4*i+k-3],2+28*i+4*k);
  }
  return wire({event:'raw_bone_packet',fighter_slot:slot,network_index:slot,bone_count:30,bone_names:v.BONES,world_positions_xyz:positions,world_rotations_xyzw:rotations,
    client_fixed_tick_at_observation:n*10,monotonic_receipt_time:10+n*.02,unity_unscaled_time:10+n*.02},b);
}
function sample(n){return {event:'sample',sample_index:n,client_fixed_tick:n*10,unity_unscaled_time:10+n*.02,phase_value:1,local_fighter_index:0,fight_epoch:8,
  round:{number:1,duration:120,time_remaining:120-n*.02,active:true,redo:false,result_value:0,winner_index:-1,clean_hits:[0,0]},fight:{result_value:0,winner_index:-1},
  fighter_0:{root_position:[0,1,0],root_rotation:[0,0,0,-1]},fighter_1:{root_position:[1,1,0],root_rotation:[0,0,0,-1]}};}
function command(seq,n,value){
  const move=Number.isInteger(value),b=Buffer.alloc(move?2:13);if(move)b[1]=value;else value.forEach((x,i)=>b.writeFloatLE(x,1+4*i));
  return wire({event:'outbound_request_projection',message:move?'REK_Move':'REK_Input',request_sequence:seq,request_only:true,server_acceptance:null,ack_observed:false,
    network_index_wire_uint8:0,client_fixed_tick_at_observation:n*10,unity_realtime_since_startup:10+n*.02+.001,
    ...(move?{move_index_wire_uint8:value}:{velocity_command_xyz:value})},b);
}
function records(change=()=>{}){
  const r=[{event:'capture_start',schema:'rek.private_ai.protocol.v5',pid:7,scope:{allowed:true,local_fighter_index:0,opponent_is_ai:true,human_in_opponent_slot:false}}];let seq=0;
  for(let n=0;n<8;n++){r.push(bone(n,0),bone(n,1),sample(n));r.push(command(++seq,n,n===3?7:n===4?[0,0,.2]:[0,0,0]));}
  change(r);
  const commands=r.filter(x=>x.event==='outbound_request_projection');
  r.push({event:'capture_end',capture_error_count:0,sample_count:r.filter(x=>x.event==='sample').length,raw_bone_packet_count:r.filter(x=>x.event==='raw_bone_packet').length,
    client_transport_invocation_count:commands.length,client_transport_method_counts:{SendMoveEvent:commands.filter(x=>x.message==='REK_Move').length,SendVelocityCommand:commands.filter(x=>x.message==='REK_Input').length}});
  return r;
}
async function withCapture(r,fn){const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-command-fixture-')),file=path.join(directory,'capture.jsonl');
  fs.writeFileSync(file,r.map(x=>JSON.stringify(x)).join('\n')+'\n');try{return await fn(file);}finally{fs.unlinkSync(file);fs.rmdirSync(directory);}}

test('all17 native move IDs map exactly; ramped/diagonal commands remain unlabeled',()=>{
  for(let move=0;move<17;move++)assert.equal(v.MOVE_ORDER[v.mapCommand({message:'REK_Move',move_index_wire_uint8:move}).action-16],move);
  for(let i=0;i<15;i++)assert.equal(v.mapCommand({message:'REK_Input',velocity_command_xyz:v.COMMANDS[i]}).action,i+1);
  assert.equal(v.mapCommand({message:'REK_Input',velocity_command_xyz:[0,0,.3]}).action,-1);
  assert.equal(v.mapCommand({message:'REK_Input',velocity_command_xyz:[1,1,0]}).action,-1);
  assert.match(v.mapCommand({message:'REK_Input',velocity_command_xyz:[0,0,0]}).reason,/not_human_release/);
});
test('calibration explicit model hierarchy/order and unsupported XML checks',()=>{
  assert.equal(calibration.joints.length,29);assert.equal(calibration.joints[0].bone,1);assert.equal(calibration.joints[28].parent,0);
  assert.throws(()=>v.calibration(xml().replace('<worldbody>','<include file="else.xml"/><worldbody>')),/nonexpanded/);
  assert.throws(()=>v.calibration(xml().replace('axis="0 1 0"','axis="0 2 0"')),/nonunit/);
  assert.throws(()=>v.calibration(xml().replace('quat="1 0 0 0"','euler="0 0 0"')),/nonexpanded/);
});
test('world to parent-local hinge projection matches known independent rotations',()=>{
  for(const angle of [-2,-.3,0,.7,2]){
    const p=v.pose(bone(1,0,angle),calibration);
    for(const q of p.q)assert.ok(Math.abs(q-angle)<1e-7,`${q} != ${angle}`);
  }
  const c={...calibration,joints:calibration.joints.map(j=>({...j,parent:j.bone===1?0:1}))};
  const p=v.pose(bone(1,0,.7),c);assert.ok(Math.abs(p.q[0]-.7)<1e-7);
  for(const q of p.q.slice(1))assert.ok(Math.abs(q)<1e-7);
});
test('raw wire body tampering fails, including pose array and move ID',()=>{
  const p=bone(1);p.world_positions_xyz[0]+=1;assert.throws(()=>v.pose(p,calibration),/wire_projection/);
  const move=command(1,1,7);move.move_index_wire_uint8=8;assert.throws(()=>v.validateCommand(move),/wire_projection/);
  const input=command(1,1,[0,0,0]);input.wire_body_sha256='0'.repeat(64);assert.throws(()=>v.validateCommand(input),/wire_hash/);
});
test('fixed observation mask distinguishes unknown busy/gates from structural constants',()=>{
  assert.deepEqual(v.FEATURE_MASK.flatMap((n,i)=>n?[]:[i]),[176,177,178,179,180,181,182,183]);
  const p=new v.Projection(),packets=[v.pose(bone(0,0),calibration),v.pose(bone(0,1),calibration)];packets.forEach((x,i)=>x.line=i+1);
  assert.equal(p.process(sample(0),packets,3).ready,false);
  const s=sample(1),out=p.process(s,packets,7);assert.equal(out.ready,true);assert.equal(out.observation[77],2);assert.equal(out.observation[71],0);assert.equal(out.observation[181],0);
  assert.equal(out.observation[86],1);assert.equal(out.observation[189],s.round.time_remaining/120);
});
test('complete ordered sequence preserves one-shot events, neutral commands and unlabeled yaw',async()=>{
  await withCapture(records(),async file=>{
    const x=await v.readCapture(file,calibration,0);assert.equal(x.ledger.length,8);assert.equal(x.summary.move_requests,1);assert.equal(x.summary.rows,7);
    const attack=x.rows.find(r=>r.action===17);assert.equal(attack.sample_index,3);assert.equal(attack.reset_before,0);
    assert.equal(attack.preceding_history.last_move,null);assert.equal(x.rows.find(r=>r.sample_index===4).preceding_history.last_move.move,7);
    assert.equal(x.rows.find(r=>r.sample_index===4).action,-1);assert.equal(x.rows.at(-1).target_reason,'right_censored_decision_interval');
    assert.equal(x.rows[0].reset_before,1);assert.ok(x.rows.slice(1).every(r=>r.reset_before===0));
    assert.ok(x.ledger.every(e=>e.server_acceptance===null&&e.human_physical_held_intent===null));
  });
});
test('current/future request changes cannot change pre-action observations',async()=>{
  const first=await withCapture(records(),file=>v.readCapture(file,calibration,0));
  const second=await withCapture(records(r=>{const index=r.findIndex(x=>x.message==='REK_Move');r[index]=command(4,3,1);}),file=>v.readCapture(file,calibration,0));
  assert.deepEqual(first.rows.map(r=>r.observation),second.rows.map(r=>r.observation));
  assert.notEqual(first.rows.find(r=>r.sample_index===3).action,second.rows.find(r=>r.sample_index===3).action);
  assert.deepEqual(first.rows.find(r=>r.sample_index===3).preceding_history,second.rows.find(r=>r.sample_index===3).preceding_history);
});
test('multiple one-shot requests are retained and ambiguous decision bin is unlabeled',async()=>{
  const r=records(a=>{const i=a.findIndex(x=>x.message==='REK_Move');a.splice(i+1,0,command(5,3,1));let n=0;for(const x of a)if(x.event==='outbound_request_projection')x.request_sequence=++n;});
  await withCapture(r,async file=>{const x=await v.readCapture(file,calibration,0);assert.equal(x.summary.move_requests,2);assert.equal(x.rows.find(r=>r.sample_index===3).target_reason,'multiple_one_shot_requests_in_50hz_interval');});
});
test('sample gap censors preceding interval and resets next usable recurrent segment',async()=>{
  const r=records(a=>{const i=a.findIndex(x=>x.event==='sample'&&x.sample_index===3);a.splice(i,1);});
  await withCapture(r,async file=>{const x=await v.readCapture(file,calibration,0);assert.equal(x.rows.find(r=>r.sample_index===2).target_reason,'gap_or_phase_boundary_after_observation');assert.equal(x.rows.find(r=>r.sample_index===5).reset_before,1);});
});
test('phase and stale pose guards refuse unavailable observations',()=>{
  const p=new v.Projection(),packets=[v.pose(bone(0,0),calibration),v.pose(bone(0,1),calibration)];packets.forEach(x=>x.line=1);
  const inactive=sample(0);inactive.round.active=false;assert.equal(p.process(inactive,packets,2).reason,'inactive_or_redo_or_wrong_slot');
  assert.equal(p.process(sample(10),packets,3).reason,'pose_age_outside_fixed_clock_guard');
});
test('request gaps, source round changes and footer count mismatch fail closed',async()=>{
  await withCapture(records(a=>{a.find(x=>x.message==='REK_Move').request_sequence=99;}),async file=>assert.rejects(v.readCapture(file,calibration,0),/request_order_or_gap/));
  await withCapture(records(a=>{a.find(x=>x.event==='sample'&&x.sample_index===3).round.number=2;}),async file=>assert.rejects(v.readCapture(file,calibration,0),/multiple_rounds/));
  const r=records();r.at(-1).sample_count++;await withCapture(r,async file=>assert.rejects(v.readCapture(file,calibration,0),/footer_count/));
});
test('REKBC001 exact binary contract, fixed mask, class support and split identity',async()=>{
  const x=await withCapture(records(),file=>v.readCapture(file,calibration,1)),b=v.binary(x.rows);
  assert.equal(b.toString('ascii',0,8),'REKBC001');assert.equal(b.length,256+x.rows.length*1056);assert.equal(b.readUInt32LE(24),1056);
  assert.deepEqual([...b.subarray(32,255)],v.FEATURE_MASK);assert.equal(b[255],0);assert.equal(b.readUInt32LE(256),1);assert.ok(b.readUInt32LE(260)>=1000000);
  for(let i=0;i<x.rows.length;i++){
    const o=256+i*1056;assert.equal(b.readInt32LE(o+12),x.rows[i].action);assert.equal(b.readFloatLE(o+16),x.rows[i].action<0?0:1);
    for(const j of v.MASKED_FIELDS)assert.equal(b.readFloatLE(o+32+4*j),0);
    for(let j=0;j<33;j++)assert.equal(b.readFloatLE(o+924+4*j),1);
  }
  assert.throws(()=>v.binary([{...x.rows[0],action:-1,weight:1}]),/binary_row/);
});
