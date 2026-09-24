'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process'),assert=require('node:assert/strict');
const columns=[9,95,72,158,202,203,204,205],selected=new Set(columns);
const schema='rek.native5.scaled_polar_xy.balance8_v1';
const stage='/home/spark-advantage/rek-training/balance8-authentic-20260924-r1';
const original='/home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1';
const model='/home/spark-advantage/codexrook-runtime/build-validation/l100-yaw-move-buffer-20260910T0448Z/semantic-duel-assets-contact/model.two_fighter_arena.xml';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const record=p=>({path:p,sha256:sha(fs.readFileSync(p)),bytes:fs.statSync(p).size});
const publish=(p,b)=>fs.writeFileSync(p,b,{flag:'wx'});
const args=['--model',model,'--projection','client_pose_projection_v1','--observation-schema',schema,'--busy-projection','dispatched_request_v4_duration'];
const datasetPath=path.join(original,'export/authentic-trajectories-v3.bin'),identityPath=path.join(original,'export/behavior-identity.json'),replayPath=path.join(original,'behavior-replay-v3.bin');
const data=fs.readFileSync(datasetPath),identityBytes=fs.readFileSync(identityPath),identity=JSON.parse(identityBytes),replay=fs.readFileSync(replayPath);
assert.equal(sha(data),'8ced592947fc1167f771d9480a0a56da3bab025d5bae292dc90308552f0bf83b');
assert.equal(sha(identityBytes),'674009772cbdbb719ff5ec7e15b4022c123cd3e61370a578a6de99aed97eb537');
assert.equal(sha(replay),'ff391c9898357023e2f4d9185a69d2aec3ae5a68feb5c3fe690ff8116b2ba0f8');
assert.equal(data.subarray(0,8).toString(),'REKRL003');assert.equal(data.readUInt32LE(20),28715);assert.equal(identity.rounds.length,5);
const out=path.join(stage,'export');fs.mkdirSync(out);
const migrated=path.join(stage,'migrated-base7561.bin');
const migration=cp.spawnSync(path.join(stage,'build/balance8-migration'),['--migrate',identity.artifacts.checkpoint.path,identity.artifacts.checkpoint.sha256,identityPath,migrated],{encoding:'utf8',env:{...process.env,CUDA_VISIBLE_DEVICES:''}});
publish(path.join(out,'migration.stdout.json'),migration.stdout);publish(path.join(out,'migration.stderr.txt'),migration.stderr);assert.equal(migration.status,0,migration.stderr);
const teacher=record(migrated),migrationReceipt=JSON.parse(fs.readFileSync(migrated+'.policy-schema.json'));
assert.equal(migrationReceipt.zeroed_weight_entries,2048);assert.equal(migrationReceipt.checkpoint_sha256,teacher.sha256);
const derived=Buffer.alloc(480+28715*1128);data.copy(derived,0,0,384);data.copy(derived,480,384);
derived.write('REKRL004',0,'ascii');derived.writeUInt32LE(4,8);Buffer.from(teacher.sha256,'hex').copy(derived,384);Buffer.from(sha(data),'hex').copy(derived,416);Buffer.from(sha(replay),'hex').copy(derived,448);
const rounds=[],ranges=Object.fromEntries(columns.map(c=>[c,{min:Infinity,max:-Infinity,nonzero:0}]));
let protectedCells=0,changedCells=0;
for(const round of identity.rounds){
 const trial=path.join(original,'evidence',round.trial_id,'trial'),rawPath=path.join(trial,'encoder.stdin.jsonl'),workerPath=path.join(trial,'worker.stdin.jsonl');
 const rawBytes=fs.readFileSync(rawPath),workerBytes=fs.readFileSync(workerPath);
 const oldInput=round.inputs.find(x=>x.file==='worker.stdin.jsonl');assert.equal(sha(workerBytes),oldInput.sha256);
 const result=cp.spawnSync(path.join(stage,'build/encode-balance8'),args,{input:rawBytes,maxBuffer:150*1024*1024,env:{...process.env,CUDA_VISIBLE_DEVICES:''}});
 assert.equal(result.status,0,result.stderr.toString());publish(path.join(out,round.trial_id+'.encoder.stdout.jsonl'),result.stdout);
 const lines=result.stdout.toString().trim().split('\n').map(JSON.parse);assert.equal(lines.shift().observation_schema,schema);
 const projected=new Map(lines.filter(x=>x.ready).map(x=>[x.worker_request.seq,x.worker_request]));
 const raw=new Map(rawBytes.toString().trim().split('\n').map(JSON.parse).filter(x=>x.observation_sequence).map(x=>[x.observation_sequence,x]));
 const worker=workerBytes.toString().trim().split('\n').map(JSON.parse).filter(x=>x.type==='step'&&!x.terminal);
 assert.equal(worker.length,round.rows);
 for(let j=0;j<worker.length;j++){
  const row=round.row_begin+j,old=worker[j],fresh=projected.get(old.seq),source=raw.get(old.seq),offset=480+row*1128,oldOffset=384+row*1128;
  assert(fresh,'missing balance8 projection '+round.trial_id+':'+old.seq);assert(source,'missing same-tick raw source');
  assert.equal(old.seq,data.readUInt32LE(oldOffset+1088));assert.equal(fresh.seq,old.seq);assert.equal(fresh.round_id,round.round_identity_sha256);assert.equal(source.round_identity_sha256,round.round_identity_sha256);
  assert.deepEqual(fresh.mask,old.mask);assert.equal(fresh.observation.length,223);
  for(let c=0;c<223;c++){
   const value=Math.fround(fresh.observation[c]);assert(Number.isFinite(value));
   const oldBytes=data.subarray(oldOffset+32+4*c,oldOffset+36+4*c),valueBytes=Buffer.alloc(4);valueBytes.writeFloatLE(value);
   const recorded=Buffer.alloc(4);recorded.writeFloatLE(old.observation[c]);assert(oldBytes.equals(recorded),'source dataset differs from actual worker observation');
   if(!selected.has(c)){assert(oldBytes.equals(valueBytes),`protected column changed ${row}:${c}`);protectedCells++;}
   else{
    assert.equal(oldBytes.readFloatLE(),0,'new cell was not structural zero in actual behavior');
    valueBytes.copy(derived,offset+32+4*c);if(!oldBytes.equals(valueBytes))changedCells++;
    ranges[c].min=Math.min(ranges[c].min,value);ranges[c].max=Math.max(ranges[c].max,value);ranges[c].nonzero+=Number(value!==0);
   }
  }
  for(let k=0;k<1128;k++)if(k<32||k>=924||!selected.has(Math.floor((k-32)/4)))assert.equal(derived[offset+k],data[oldOffset+k],'protected row byte changed');
 }
 rounds.push({trial_id:round.trial_id,sequence:round.sequence,seed:round.seed,rows:round.rows,raw_encoder_input:record(rawPath),recorded_worker_input:record(workerPath),projected_output:record(path.join(out,round.trial_id+'.encoder.stdout.jsonl'))});
 process.stdout.write(JSON.stringify({phase:'cpu_export_round',trial_id:round.trial_id,rows:round.rows,protected_cells:round.rows*215})+'\n');
}
const receipt={schema:'rek.authentic_derived_balance8_identity.v4',observation_schema:schema,recorded_behavior_identity:record(identityPath),recorded_behavior:identity.artifacts,derived_teacher:teacher,derived_teacher_was_recorded_worker:false,original_dataset:record(datasetPath),original_behavior_replay:record(replayPath),migration_receipt:record(migrated+'.policy-schema.json'),encoder:record(path.join(stage,'build/encode-balance8')),exporter:record(__filename),model:record(model),columns,protected_columns:215,protected_cells_bitwise_equal:protectedCells,changed_cells:changedCells,feature_ranges:ranges,rounds,reward:identity.reward,discounts:identity.discounts,all_other_row_bytes_preserved:true,gpu_equivalence_verified:false,training_performed:false};
const receiptBytes=Buffer.from(JSON.stringify(receipt,null,2)+'\n');Buffer.from(sha(receiptBytes),'hex').copy(derived,256);
publish(path.join(out,'derived-identity.json'),receiptBytes);publish(path.join(out,'authentic-balance8-v4.bin'),derived);
process.stdout.write(JSON.stringify({phase:'cpu_export_complete',dataset_sha256:sha(derived),identity_sha256:sha(receiptBytes),teacher_sha256:teacher.sha256,rows:28715,protected_cells:protectedCells,changed_cells:changedCells,gpu_equivalence_verified:false})+'\n');
