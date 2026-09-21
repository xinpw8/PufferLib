#!/usr/bin/env node
'use strict';
// Read-only integration verification of the new dataset against pinned originals.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const assert = require('node:assert/strict'), readline = require('node:readline');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const retained = c => c < 172 ? c % 86 <= 9 || c % 86 >= 12 && c % 86 <= 76 :
  c >= 172 && c <= 175 || c === 184 || c === 185 || c >= 188 && c <= 191 || c >= 202 && c <= 205 || c === 217 || c === 218;
async function verify(originalDir, outputDir, rawPaths) {
  const original = fs.readFileSync(path.join(originalDir, 'human-commands.bin'));
  const output = fs.readFileSync(path.join(outputDir, 'human-observable.bin'));
  const manifest = JSON.parse(fs.readFileSync(path.join(outputDir, 'manifest.json')));
  assert.equal(hash(original), '71238da150db6e926170efdc27662f4426145aac7f69a7c93f619ece54d6cf1d');
  assert.equal(hash(output), manifest.dataset_sha256); assert.equal(output.length, original.length);
  assert.equal(manifest.observation_schema, 'rek.native5.observable_balance.v1');
  assert.deepEqual(output.subarray(0, 32), original.subarray(0, 32)); assert.equal(output[255], original[255]);
  const expectedMask = Buffer.from(Array.from({length:223}, (_, c) => +retained(c)));
  assert.deepEqual(output.subarray(32, 255), expectedMask); assert.equal([...expectedMask].reduce((a,b)=>a+b), 166);
  assert.equal(hash(expectedMask), manifest.feature_mask_sha256);
  assert.deepEqual(fs.readFileSync(path.join(outputDir, 'feature-mask.bin')), expectedMask);
  assert.deepEqual(fs.readFileSync(path.join(outputDir, 'original-feature-mask.bin')), original.subarray(32,255));
  for (const name of ['row-ledger.jsonl','command-ledger.jsonl']) assert.deepEqual(fs.readFileSync(path.join(outputDir,name)),fs.readFileSync(path.join(originalDir,name)));
  assert.deepEqual(fs.readFileSync(path.join(outputDir,'source-manifest.json')),fs.readFileSync(path.join(originalDir,'manifest.json')));
  const ledger = fs.readFileSync(path.join(outputDir,'row-ledger.jsonl'),'utf8').trim().split(/\r?\n/).map(JSON.parse);
  const projection = fs.readFileSync(path.join(outputDir,'projection-ledger.jsonl'),'utf8').trim().split(/\r?\n/).map(JSON.parse);
  assert.equal(hash(fs.readFileSync(path.join(outputDir,'projection-ledger.jsonl'))),manifest.projection_ledger_sha256);
  const sourceSamples = [new Map(), new Map()];
  for (let split=0;split<2;split++) {
    const wanted = new Set(ledger.filter(r=>r.split===split).map(r=>r.source_line));
    const stream = fs.createReadStream(rawPaths[split]), digest=crypto.createHash('sha256'); stream.on('data',b=>digest.update(b)); let line=0;
    for await (const text of readline.createInterface({input:stream,crlfDelay:Infinity})) { line++; if(wanted.has(line)){const sample=JSON.parse(text);assert.equal(sample.event,'sample');sourceSamples[split].set(line,sample);} }
    assert.equal(digest.digest('hex'), manifest.sources[split].sha256);
  }
  const counts = [0,0], labels=[0,0], kicks=[0,0], resets=[0,0]; let changed=0, history=0, maximumTiltError=0, maximumRateError=0;
  for(let i=0;i<ledger.length;i++) {
    const r=ledger[i],p=256+i*1056,s=sourceSamples[r.split].get(r.source_line); assert.ok(s);
    assert.deepEqual(output.subarray(p,p+32),original.subarray(p,p+32));
    assert.deepEqual(output.subarray(p+924,p+1056),original.subarray(p+924,p+1056));
    assert.equal(output.readUInt32LE(p),r.split); assert.equal(output.readInt32LE(p+12),r.action); assert.equal(output.readFloatLE(p+16),r.weight);
    assert.deepEqual(projection[i],{row:i,split:r.split,source_line:r.source_line,previous_source_line:r.reset_before?null:ledger[i-1].source_line,history_available:!r.reset_before});
    const obs=Array.from({length:223},(_,c)=>output.readFloatLE(p+32+c*4));
    counts[r.split]++; labels[r.split]+=r.weight>0; kicks[r.split]+=r.action===17; resets[r.split]+=r.reset_before; history+=obs[203];
    changed+=!output.subarray(p+32,p+924).equals(original.subarray(p+32,p+924));
    for(let c=0;c<223;c++){assert.ok(Number.isFinite(obs[c]));if(!retained(c))assert.equal(obs[c],0);}
    for(const c of [202,204,205])assert.equal(obs[c],0); assert.equal(obs[203],+!r.reset_before);
    for(let slot=0;slot<2;slot++) {
      const b=slot*86,f=s['fighter_'+slot],q=f.root_rotation.map(Math.fround),norm=Math.hypot(...q);
      const tilt=Math.acos(Math.max(-1,Math.min(1,1-2*((q[0]/norm)**2+(q[2]/norm)**2))))/Math.PI;
      maximumTiltError=Math.max(maximumTiltError,Math.abs(obs[b+72]-tilt));
      assert.equal(obs[b+73],Math.fround(f.root_position[1]));
      for(let c=13;c<=70;c++)assert.equal(obs[b+c],0); for(const c of [74,75])assert.equal(obs[b+c],0);
      if(!r.reset_before){
        const old=sourceSamples[r.split].get(ledger[i-1].source_line),dt=s.unity_unscaled_time-old.unity_unscaled_time;
        const expected=(Math.fround(f.root_position[1])-Math.fround(old['fighter_'+slot].root_position[1]))/dt;
        maximumRateError=Math.max(maximumRateError,Math.abs(obs[b+9]-Math.fround(expected)));
        assert.equal(obs[217+slot],s.round.clean_hits[slot]-old.round.clean_hits[slot]);
      }else{assert.equal(obs[b+9],0);assert.equal(obs[217+slot],0);}
      assert.equal(obs[190+slot],s.round.clean_hits[slot]);
    }
  }
  assert.deepEqual(counts,[5993,5992]);assert.deepEqual(labels,[4943,4990]);assert.deepEqual(kicks,[16,2]);assert.deepEqual(resets,[2,2]);
  assert.equal(changed,11985);assert.equal(history,11981);assert.ok(maximumTiltError<1e-6);assert.equal(maximumRateError,0);
  return {schema:'rek.human_observable.integration_check.v1',passed:true,rows:counts,labels,left_front_labels:kicks,segments:resets,
    changed_observation_rows:changed,history_available_rows:history,maximum_tilt_error:maximumTiltError,maximum_vertical_rate_error:maximumRateError,
    metadata_and_class_support_bytes_identical:true,ledgers_byte_identical:true,raw_hashes_verified:true,dataset_sha256:hash(output)};
}
module.exports={verify};
if(require.main===module)Promise.resolve().then(()=>{assert.equal(process.argv.length,6,'Usage: node human_observable_data.test.cjs ORIGINAL_DIR OUTPUT_DIR TRAIN_RAW HELDOUT_RAW');return verify(process.argv[2],process.argv[3],process.argv.slice(4));}).then(r=>console.log(JSON.stringify(r))).catch(e=>{console.error(e);process.exitCode=1;});
