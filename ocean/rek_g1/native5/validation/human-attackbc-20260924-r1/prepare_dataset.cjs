'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {derive,HEADER,ROW}=require('./conditional_attack_bc.cjs');
const COLUMNS=[9,95,72,158,202,203,204,205],UNAVAILABLE=[176,177,178,179,180,181,182,183,202,204,205];
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function check(ok,why){if(!ok)throw Error(why);}
function project(base,measured){
  check(base.length===measured.length&&base.readUInt32LE(20)===measured.readUInt32LE(20),'paired_dataset_length');
  const out=Buffer.from(base),count=base.readUInt32LE(20);let histories=0;
  for(let i=0;i<count;i++){
    const p=HEADER+i*ROW;
    check(base.subarray(p,p+32).equals(measured.subarray(p,p+32))&&base.subarray(p+924,p+ROW).equals(measured.subarray(p+924,p+ROW)),'paired_row_identity');
    const values=COLUMNS.map(c=>measured.readFloatLE(p+32+4*c));
    check(values.every(Number.isFinite),'nonfinite_balance8');
    check(values[2]>=0&&values[2]<=1&&values[3]>=0&&values[3]<=1,'tilt_range');
    check([values[4],values[5],values[6],values[7]].every(x=>x===0||x===1),'availability_range');
    check(values[4]===0&&values[6]===0&&values[7]===0,'historical_referee_must_be_unavailable');
    check(values[5]!==0||(values[0]===0&&values[1]===0),'unavailable_history_rates');histories+=values[5];
    for(const c of COLUMNS)measured.copy(out,p+32+4*c,p+32+4*c,p+36+4*c);
    for(const c of UNAVAILABLE)out.writeFloatLE(0,p+32+4*c);
    for(let c=0;c<223;c++)if(!COLUMNS.includes(c)&&!UNAVAILABLE.includes(c))check(out.subarray(p+32+4*c,p+36+4*c).equals(base.subarray(p+32+4*c,p+36+4*c)),'known_feature_changed');
  }
  for(let c=0;c<223;c++)out[32+c]=UNAVAILABLE.includes(c)?0:1;
  return {output:out,history_available_rows:histories};
}
function run(original,observable,destination){
  check(!fs.existsSync(destination),'output_exists');
  const source=fs.readFileSync(original),measured=fs.readFileSync(observable);
  check(sha(source)==='71238da150db6e926170efdc27662f4426145aac7f69a7c93f619ece54d6cf1d','original_dataset_hash');
  check(sha(measured)==='228b5696fdcb0663c15dce32b6e8bed171ed995661c3acd5d050f95402aa7c8f','measured_dataset_hash');
  const projected=project(source,measured),result=derive(projected.output),mask=result.output.subarray(32,255);
  check(result.receipt.rows===11985&&result.receipt.splits[0].rows===5993&&result.receipt.splits[1].rows===5992&&result.receipt.splits[0].labels===77&&result.receipt.splits[1].labels===109,'expected_human_rows');
  let positiveChunks=0;const sequences=[];
  for(let i=0;i<11985;i++){
    const p=HEADER+i*ROW,id=result.output.readUInt32LE(p+4);
    if(!sequences.length||sequences.at(-1).id!==id)sequences.push({id,split:result.output.readUInt32LE(p),begin:i,end:i+1});else sequences.at(-1).end=i+1;
  }
  for(const s of sequences.filter(x=>x.split===0))for(let i=s.begin;i<s.end;i+=128){let labels=0;for(let j=i;j<Math.min(i+128,s.end);j++)labels+=result.output.readFloatLE(HEADER+j*ROW+16);if(labels)positiveChunks++;}
  const manifest={...result.receipt,schema:'rek.human_conditional_attack_balance8.v1',observation_schema:'rek.native5.scaled_polar_xy.balance8_v1',
    original_dataset:{path:original,sha256:sha(source)},measured_dataset:{path:observable,sha256:sha(measured)},
    original_observations_preserved_except_balance8_and_explicit_unavailable:true,unchanged_known_columns:207,
    measured_balance8_columns:COLUMNS,zero_unavailable_columns:UNAVAILABLE,feature_mask_sha256:sha(mask),retained_features:212,
    raw_capture_sha256:['547cec42f700e97b9594f8d2c6df88f966052c0e0e5ce7395c4862b619fb6d7f','ec55c32a6e2272a8e7656d73260ca35876be2cd6c8d8d6fe8c9dfe77cd25a309'],
    provenance:'Original human command labels and complete chronology; validated Sept21 C++ root projection at identical source rows supplies8 measured/availability columns. No re-timing or interpolation.',
    partial_observation:true,on_policy:false,behavior_logprob_available:false,acknowledged_dispatch_available:false,
    history_available_rows:projected.history_available_rows,sequences,training_labeled_chunks_per_epoch:positiveChunks,
    initialization_sha256:'0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12',
    proposed_training:{epochs:5,learning_rate:.0001,horizon:128,conditional_support:[16,32],event_weight:1,class_balancing:false,optimizer:'unchanged native Muon'},
    live_feature_mask:'unchanged allones; no conditional support mask is applied live',live_feature_mask_sha256:'59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b',
    input_equivalence_claim:false,transfer_limits:['unknown temporal fields zero duringBC but observed/projected duringlive','received network bone poses versus rendered live local bones','50Hz historical chronology versus variable current live cadence','fresh referee/count unavailable duringBC','only two human rounds from one session; development holdout previously examined'],
    gpu_used:false,training_performed:false};
  fs.mkdirSync(destination,{recursive:false});fs.writeFileSync(path.join(destination,'human-conditional-attack.bin'),result.output,{flag:'wx'});fs.writeFileSync(path.join(destination,'bc-partial-feature-mask.bin'),mask,{flag:'wx'});fs.writeFileSync(path.join(destination,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({dataset_sha256:manifest.dataset_sha256,feature_mask_sha256:manifest.feature_mask_sha256,rows:manifest.rows,splits:manifest.splits,training_labeled_chunks_per_epoch:positiveChunks,sequences:sequences.length,history_available_rows:manifest.history_available_rows,gpu_used:false}));
  return manifest;
}
if(require.main===module){try{check(process.argv.length===5,'usage_original_observable_newdir');run(...process.argv.slice(2));}catch(e){console.error(e.message);process.exitCode=1;}}
module.exports={project,COLUMNS,UNAVAILABLE,run};
