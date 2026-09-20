'use strict';
// Offline upgrade of already exported trajectories; never re-encodes old poses.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const readline = require('node:readline');
const {HEADER, ROW, OBS, ACTIONS} = require('./authentic_trajectory_data.cjs');
const LEGACY = 'rek.native5.scaled_polar_xy.v1';
const SCHEMA = 'rek.native5.scaled_polar_xy.owned_yaw_v2';
const COLUMN = 187;
const check = (ok, why) => { if (!ok) throw new Error(why); };
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
function desiredYaw(category) {
  check(Number.isInteger(category) && category >= 1 && category <= 15, 'active desired_action must be owned category1..15');
  return [6,8,10,12,14].includes(category) ? 1 : [7,9,11,13,15].includes(category) ? -1 : 0;
}
function sameArray(actual, expected, label, float32 = false) {
  check(Array.isArray(actual) && actual.length === expected.length && actual.every((v,i) =>
    Number.isFinite(v) && (float32 ? Math.fround(v) : v) === expected[i]), label);
}
function rowEvidence(binary, index, source, encoded, worker, identity) {
  const offset = HEADER + index * ROW, seq = binary.readUInt32LE(offset + 1088);
  check(source?.event === 'g1_policy_state' && source.observation_sequence === seq &&
    source.round_identity_sha256 === identity && source.round?.active === true && source.stream_active === true &&
    source.input?.active === true, 'missing same-source active owned state');
  check(encoded?.event === 'policy_observation' && encoded.ready === true, 'missing ready encoder provenance');
  const request = encoded.worker_request, p = encoded.provenance;
  check(request?.seq === seq && request.round_id === identity && request.observation_schema === LEGACY &&
    request.type === 'step' && request.terminal === false && worker?.seq === seq && worker.round_id === identity &&
    worker.observation_schema === LEGACY && worker.type === 'step' && worker.terminal === false,
    'worker/encoder sequence, schema, or terminal mismatch');
  check(p?.source_qpc_ticks === source.clock?.qpc_ticks &&
    p?.source_qpc_frequency_hz === source.clock?.qpc_frequency_hz && p.stream_active === true &&
    typeof p.projected_busy === 'boolean' &&
    ['dispatched_request_v4_duration','native_controller_busy'].includes(p.busy_projection), 'busy/source provenance unavailable');
  const obs = Array.from({length:OBS}, (_,j) => binary.readFloatLE(offset + 32 + 4*j));
  const mask = Array.from({length:ACTIONS}, (_,j) => binary.readFloatLE(offset + 924 + 4*j));
  sameArray(request.observation, obs, 'encoder observation differs from frozen v1 row', true);
  sameArray(worker.observation, obs, 'worker observation differs from frozen v1 row', true);
  sameArray(request.mask, mask, 'encoder mask differs from frozen v1 row');
  sameArray(worker.mask, mask, 'worker mask differs from frozen v1 row');
  check(binary.readUInt32LE(offset + 32 + 4*COLUMN) === 0, 'legacy column187 must be positive zero');
  check(obs[182] === Number(p.projected_busy) && obs[183] === Number(p.projected_busy), 'busy feature/provenance disagreement');
  const yaw = desiredYaw(source.input.desired_action);
  const value = p.projected_busy ? yaw : 0;
  return { source_sequence:seq, source_qpc_ticks:source.clock.qpc_ticks,
    desired_action:source.input.desired_action, projected_busy:p.projected_busy,
    busy_projection:p.busy_projection, owned_pending_yaw:value };
}
async function readLines(file, visit) {
  const before=fs.statSync(file), hash=crypto.createHash('sha256');
  const stream=fs.createReadStream(file);stream.on('data', b=>hash.update(b));let lines=0;
  for await(const line of readline.createInterface({input:stream,crlfDelay:Infinity})) {
    check(line.trim().length>0,'blank source record');visit(JSON.parse(line),++lines);
  }
  const after=fs.statSync(file);check(before.size===after.size && before.mtimeMs===after.mtimeMs,'input changed during upgrade');
  return {file:path.basename(file),bytes:before.size,lines,sha256:hash.digest('hex')};
}
function insert(map,key,value,label) { check(!map.has(key),`duplicate ${label}`);map.set(key,value); }
function bind(input, round) {
  const old=round.inputs.find(x=>path.basename(x.file)===input.file);
  if(old) check(old.bytes===input.bytes && old.sha256===input.sha256,`original source binding changed: ${input.file}`);
}
async function upgrade(root, originalDirectory, output, migratedCheckpointSha) {
  check(/^[0-9a-f]{64}$/.test(migratedCheckpointSha),'explicit migrated checkpoint SHA required');
  check(!fs.existsSync(output),'new output directory required');
  const manifestBytes=fs.readFileSync(path.join(originalDirectory,'manifest.json'));
  const original=JSON.parse(manifestBytes), binary=fs.readFileSync(path.join(originalDirectory,'authentic-trajectories.bin'));
  check(original.schema==='rek.authentic_trajectory_dataset.v1' && original.binary_sha256===sha(binary), 'original dataset binding mismatch');
  check(binary.subarray(0,8).toString()==='REKRL001' && binary.readUInt32LE(8)===1 &&
    binary.readUInt32LE(12)===OBS && binary.readUInt32LE(16)===ACTIONS && binary.readUInt32LE(24)===ROW &&
    binary.length===HEADER+ROW*original.rows && binary.readUInt32LE(20)===original.rows &&
    binary.readUInt32LE(28)===original.rounds.length, 'original trajectory layout mismatch');
  const upgraded=Buffer.from(binary), evidence=[], inputReports=[];let upgradedRows=0,nonzeroRows=0;
  for(const round of original.rounds) {
    check(/^live-[a-zA-Z0-9_-]+$/.test(round.trial_id),'invalid original trial ID');
    const trial=path.join(root,round.trial_id,'trial'), sources=new Map(), encoderInputs=new Map(), encoders=new Map(), workers=new Map();
    const inputs=[];
    inputs.push(await readLines(path.join(trial,'relay.stdout.jsonl'),x=>{
      if(x.event==='g1_policy_state') {
        check(x.round_identity_sha256===round.round_identity_sha256,'source round identity changed');
        insert(sources,x.observation_sequence,{hash:sha(JSON.stringify(x)),event:x.event,observation_sequence:x.observation_sequence,
          round_identity_sha256:x.round_identity_sha256,round:{active:x.round.active},stream_active:x.stream_active,
          input:{active:x.input.active,desired_action:x.input.desired_action},clock:x.clock},'source sequence');
      }
    }));
    inputs.push(await readLines(path.join(trial,'encoder.stdin.jsonl'),x=>{
      check(x.event==='g1_policy_state','unsupported encoder input');
      insert(encoderInputs,x.observation_sequence,sha(JSON.stringify(x)),'encoder source');
    }));
    inputs.push(await readLines(path.join(trial,'encoder.stdout.jsonl'),x=>{
      if(x.event==='policy_observation' && x.ready===true) insert(encoders,x.worker_request.seq,x,'encoded request');
    }));
    inputs.push(await readLines(path.join(trial,'worker.stdin.jsonl'),x=>insert(workers,x.seq,x,'worker request')));
    for(const input of inputs) bind(input,round);
    let count=0;
    for(let i=0;i<original.rows;++i) {
      const offset=HEADER+i*ROW;if(binary.readUInt32LE(offset+4)!==round.sequence)continue;
      const seq=binary.readUInt32LE(offset+1088),source=sources.get(seq);
      check(source && source.hash===encoderInputs.get(seq),'encoder input is not the identical relay source snapshot');
      const item=rowEvidence(binary,i,source,encoders.get(seq),workers.get(seq),round.round_identity_sha256);
      upgraded.writeFloatLE(item.owned_pending_yaw,offset+32+4*COLUMN);
      evidence.push({row:i,sequence:round.sequence,...item});++count;++upgradedRows;if(item.owned_pending_yaw!==0)++nonzeroRows;
    }
    check(count===round.rows,'sequence row count differs from original manifest');
    inputReports.push({trial_id:round.trial_id,sequence:round.sequence,rows:count,inputs});
  }
  check(upgradedRows===original.rows,'some original rows were not upgraded exactly once');
  upgraded.write('REKRL002');upgraded.writeUInt32LE(2,8);
  const restored=Buffer.from(upgraded);binary.copy(restored,0,0,12);
  for(let i=0;i<original.rows;++i) { const offset=HEADER+i*ROW+32+4*COLUMN;binary.copy(restored,offset,offset,offset+4); }
  check(restored.equals(binary),'upgrade changed bytes outside schema header/column187');
  const manifest={schema:'rek.authentic_trajectory_dataset.owned_yaw_v2',created_utc:new Date().toISOString(),
    observation_schema:SCHEMA,original_observation_schema:LEGACY,column:COLUMN,
    original_manifest_sha256:sha(manifestBytes),original_binary_sha256:sha(binary),binary_sha256:sha(upgraded),
    binary:'authentic-trajectories-owned-yaw-v2.bin',header_bytes:HEADER,row_bytes:ROW,rows:original.rows,
    applied:original.applied,terminal_race_rejected:original.terminal_race_rejected,
    actual_behavior_checkpoint_sha256:original.checkpoint_sha256,compatible_migrated_checkpoint_sha256:migratedCheckpointSha,
    exporter_sha256:sha(fs.readFileSync(__filename)),nonzero_upgraded_rows:nonzeroRows,
    rule:'active owned pre-action source desired category yaw, retained only during the exact saved busy projection',
    preserves:'all bytes except explicit format identity and observation column187; rewards, times, masks, actions, recurrence unchanged',
    required_next_check:'native original checkpoint/old inputs versus migrated checkpoint/new inputs; all34logits, sampledactions and chosen logprobs exact',
    terminal_semantics:'terminal observation-only worker inputs are not decision rows; terminal-race decision rows retain their active pre-action inputs',
    reference_gamma_per_20ms:original.reference_gamma_per_20ms,reference_lambda_per_20ms:original.reference_lambda_per_20ms,
    rounds:inputReports};
  fs.mkdirSync(output);fs.writeFileSync(path.join(output,manifest.binary),upgraded,{flag:'wx'});
  fs.writeFileSync(path.join(output,'upgrade-evidence.jsonl'),evidence.map(x=>JSON.stringify(x)).join('\n')+'\n',{flag:'wx'});
  fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
  return manifest;
}
if(require.main===module) {
  const [root,original,output,checkpointSha]=process.argv.slice(2);
  upgrade(root,original,output,checkpointSha).then(m=>console.log(JSON.stringify({rows:m.rows,nonzero_rows:m.nonzero_upgraded_rows,binary_sha256:m.binary_sha256,output})))
    .catch(e=>{console.error(e.message);process.exitCode=1;});
}
module.exports={COLUMN,LEGACY,SCHEMA,desiredYaw,rowEvidence,upgrade};
