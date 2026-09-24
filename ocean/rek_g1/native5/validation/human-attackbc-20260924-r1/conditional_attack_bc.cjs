'use strict';
const crypto=require('node:crypto');
const HEADER=256,ROW=1056,OBS=223,ACTIONS=33;
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function requireValue(ok,reason){if(!ok)throw Error(reason);}
function derive(source){
  requireValue(Buffer.isBuffer(source)&&source.length>=HEADER,'dataset_header');
  requireValue(source.subarray(0,8).toString()==='REKBC001','dataset_magic');
  requireValue(source.readUInt32LE(8)===1&&source.readUInt32LE(12)===OBS&&source.readUInt32LE(16)===ACTIONS&&source.readUInt32LE(24)===ROW,'dataset_layout');
  const count=source.readUInt32LE(20);
  requireValue(count>0&&source.length===HEADER+count*ROW,'dataset_length');
  const output=Buffer.from(source),splits=[{rows:0,labels:0,actions:{}},{rows:0,labels:0,actions:{}}];
  for(let i=0;i<count;i++){
    const p=HEADER+i*ROW,split=source.readUInt32LE(p),action=source.readInt32LE(p+12),weight=source.readFloatLE(p+16);
    requireValue(split<=1&&action>=-1&&action<ACTIONS&&Number.isFinite(weight)&&weight>=0,'row_label');
    requireValue((action===-1)===(weight===0),'row_label_weight');
    requireValue(action===-1||weight===1,'source_must_be_unweighted');
    for(let a=0;a<ACTIONS;a++)requireValue(source.readFloatLE(p+924+4*a)===1,'original_all_vocabulary_required');
    const attack=action>=16;
    output.writeInt32LE(attack?action:-1,p+12);
    output.writeFloatLE(attack?1:0,p+16);
    for(let a=0;a<ACTIONS;a++)output.writeFloatLE(a>=16?1:0,p+924+4*a);
    const stats=splits[split];stats.rows++;
    if(attack){stats.labels++;stats.actions[action]=(stats.actions[action]??0)+1;}
    requireValue(output.subarray(p,p+12).equals(source.subarray(p,p+12))&&output.subarray(p+20,p+924).equals(source.subarray(p+20,p+924)),'nonobjective_bytes_changed');
  }
  requireValue(splits.every(s=>s.labels>0),'both_splits_need_attack_labels');
  requireValue(output.subarray(0,HEADER).equals(source.subarray(0,HEADER)),'header_changed');
  return {output,receipt:{schema:'rek.conditional_attack_bc_dataset.v1',source_sha256:sha(source),dataset_sha256:sha(output),rows:count,splits,
    loss:'negative log conditional probability within policy categories16..32 on observed one-shot request rows',
    action_support:'loss normalization set only; never a runtime legality or action forcing mask',
    nonattack_rows:'retained chronologically with derived label-1 and weight0; original labels remain in source dataset/ledger',
    weights:'one per observed attack request; no per-class balancing',
    nonobjective_bytes_identical:true,feature_mask_unchanged:true,
    behavior_timing_preserved:false,timing_caveat:'shared-network updates may change total attack mass and movement even though classes0..15 have zero direct CE logit gradient'}};
}
module.exports={derive,HEADER,ROW,OBS,ACTIONS};
