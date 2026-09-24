'use strict';
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const sourceSha='b3b3fd0e93014a066b98c10266d329d98816eae1a7cc2df1343a8746ccb570b2';
function summarize(bytes){
 assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),sourceSha,'pinned native metrics source');
 const rows=bytes.toString().trim().split(/\r?\n/).map(JSON.parse);
 const epochs=rows.filter(r=>r.phase==='initial'||r.phase==='epoch');
 assert.deepEqual(epochs.map(r=>r.epoch),[0,1,2,3,4,5]);
 assert.equal(epochs.at(-1).updates,165);
 const details=rows.filter(r=>r.phase==='train_details'||r.phase==='heldout_details').map(r=>({
  split:r.phase==='train_details'?'training':'development',epoch:r.epoch,
  actions:r.actions.filter(a=>a.weight>0)
 }));
 return {schema:'rek.human_movement_bc_published_metrics.v1',source_stdout_sha256:sourceSha,
  native_heldout_fields_mean:'already examined development split, not an untouched final test',
  metadata:rows.filter(r=>r.phase===undefined),epochs,per_action:details,
  checkpoints:rows.filter(r=>r.phase==='epoch_saved'||r.phase==='saved'),
  omitted:'zero-weight action entries and redundant movement-category totals; no observations or checkpoint bytes'};
}
if(require.main===module){
 assert.equal(process.argv.length,4,'usage archived_native_stdout new_metrics_json');
 fs.writeFileSync(process.argv[3],JSON.stringify(summarize(fs.readFileSync(process.argv[2])),null,2)+'\n',{flag:'wx'});
}
module.exports={summarize,sourceSha};
