'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {audit,translating}=require('./audit_live_policy_quality.cjs');
test('held categories distinguish translation from neutral and yaw',()=>{
  for(let i=0;i<16;i++)assert.equal(translating(i),[2,3,4,5,8,9,10,11,12,13,14,15].includes(i));
});
test('replay counts source-mask defect without claiming counterfactual outcomes',async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rek-quality-audit-'));
  const save=(name,rows)=>fs.writeFileSync(path.join(dir,name),rows.map(x=>JSON.stringify(x)).join('\n')+'\n');
  try{
    const state=(seq,desired)=>({event:'g1_policy_state',observation_sequence:seq,input:{desired_action:desired},action_mask:Array(33).fill(true)});
    save('relay.stdout.jsonl',[state(1,8),{event:'g1_policy_action',observation_sequence:1,action:26,applied:false},state(2,6),{event:'g1_policy_action',observation_sequence:2,action:25,applied:true}]);
    const encoded=seq=>({ready:true,worker_request:{seq,observation:Array(223).fill(0),mask:Array(33).fill(1)}});
    const rows=[encoded(1),encoded(2)];save('encoder.stdout.jsonl',rows);
    const fixed=JSON.parse(JSON.stringify(rows));for(let i=16;i<33;i++)fixed[0].worker_request.mask[i]=0;
    save('corrected.jsonl',fixed);save('summary.json',[{checkpoint_sha256:'0'.repeat(64),final_round:{winner_index:1}}]);
    const result=await audit(dir,path.join(dir,'corrected.jsonl'));
    assert.equal(result.attacks_requested,2);assert.equal(result.attacks_applied,1);assert.equal(result.attacks_rejected,1);
    assert.equal(result.attack_source_advertised_while_translation_held,1);assert.equal(result.rejected_attacks_while_translation_held,1);
    assert.deepEqual(result.corrected_replay,{compared_observations:2,changed_observations:0,changed_masks:1,recorded_rejected_actions_now_masked:1,recorded_applied_actions_now_masked:0});
    assert.equal(result.final_round.winner_index,1);
    assert(Object.values(result.artifact_sha256).every(x=>/^[a-f0-9]{64}$/.test(x)));
  }finally{fs.rmSync(dir,{recursive:true});}
});
