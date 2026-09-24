'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const HEADER=256,ROW=1056,UNAVAILABLE=[176,177,178,179,180,181,182,183,202,204,205];
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function check(ok,why){if(!ok)throw Error(why);}
function derive(original,partial){
 check(original.length===partial.length&&original.length>=HEADER,'paired_length');
 for(const b of [original,partial])check(b.subarray(0,8).toString()==='REKBC001'&&b.readUInt32LE(8)===1&&b.readUInt32LE(12)===223&&b.readUInt32LE(16)===33&&b.readUInt32LE(24)===ROW,'dataset_layout');
 const n=original.readUInt32LE(20);check(n>0&&original.length===HEADER+n*ROW&&partial.readUInt32LE(20)===n,'row_count');
 const output=Buffer.from(partial),splits=[{rows:0,labels:0,actions:{}},{rows:0,labels:0,actions:{}}],sequences=[];
 for(let c=0;c<223;c++)check(partial[32+c]===(UNAVAILABLE.includes(c)?0:1),'partial_mask');
 for(let i=0;i<n;i++){
  const p=HEADER+i*ROW,split=original.readUInt32LE(p),seq=original.readUInt32LE(p+4),action=original.readInt32LE(p+12),weight=original.readFloatLE(p+16);
  check(split<=1&&action>=-1&&action<=32&&weight===(action<0?0:1),'original_unweighted_label');
  check(original.subarray(p,p+12).equals(partial.subarray(p,p+12))&&original.subarray(p+20,p+32).equals(partial.subarray(p+20,p+32)),'paired_chronology');
  for(let a=0;a<33;a++)check(original.readFloatLE(p+924+4*a)===1,'original_vocabulary');
  for(let c=0;c<223;c++){const v=partial.readFloatLE(p+32+4*c);check(Number.isFinite(v),'nonfinite_partial_feature');if(UNAVAILABLE.includes(c))check(v===0,'unknown_feature_not_zero');}
  const movement=action>=2&&action<=15;output.writeInt32LE(movement?action:-1,p+12);output.writeFloatLE(movement?1:0,p+16);
  for(let a=0;a<33;a++)output.writeFloatLE(a>=2&&a<=15?1:0,p+924+4*a);
  splits[split].rows++;if(movement){splits[split].labels++;splits[split].actions[action]=(splits[split].actions[action]??0)+1;}
  if(!sequences.length||sequences.at(-1).id!==seq)sequences.push({id:seq,split,begin:i,end:i+1});else sequences.at(-1).end=i+1;
  check(output.subarray(p,p+12).equals(original.subarray(p,p+12))&&output.subarray(p+20,p+32).equals(original.subarray(p+20,p+32)),'chronology_changed');
  check(output.subarray(p+32,p+924).equals(partial.subarray(p+32,p+924)),'partial_observations_changed');
 }
 check(splits.every(s=>s.labels>0),'both_splits_need_movement_labels');
 check(output.subarray(0,HEADER).equals(partial.subarray(0,HEADER)),'partial_header_changed');
 let chunks=0;for(const s of sequences.filter(s=>s.split===0))for(let i=s.begin;i<s.end;i+=128){let has=false;for(let j=i;j<Math.min(i+128,s.end);j++)has||=output.readFloatLE(HEADER+j*ROW+16)>0;if(has)chunks++;}
 return {output,receipt:{schema:'rek.human_conditional_movement_balance8.v1',observation_schema:'rek.native5.scaled_polar_xy.balance8_v1',original_dataset_sha256:sha(original),partial_reference_sha256:sha(partial),dataset_sha256:sha(output),feature_mask_sha256:sha(output.subarray(32,255)),rows:n,splits,sequences,
  all_observation_bytes_identical_to_partial_reference:true,chronology_bytes_identical_to_original:true,zero_unavailable_columns:UNAVAILABLE,
  conditional_support:[2,15],nonmovement_loss_weight:0,event_weights:'one per exact recorded movement row; temporal occupancy, not independent command starts',class_balancing:false,
  training_labeled_chunks_per_epoch:chunks,normalization_float32:Math.fround(splits[0].labels/splits[0].rows*128),
  neutral_semantics:'Original action1 is observed zero outgoing command, possibly automatic; never asserted human key release. It has no loss in this dataset.',
  partial_observation:true,on_policy:false,input_equivalence_claim:false,behavior_logprob_available:false,
  initialization_sha256:'5b19d6280700994ecddd1b2ab69f9986f55b01b06034de8b796d5e81572168a4',proposed_training:{epochs:5,learning_rate:.0001,horizon:128,optimizer:'unchanged native Muon'},
  live_feature_mask_sha256:'59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b',live_feature_mask_changed:false,
  limitations:['teaches direction conditional on a movement label, not when to initiate or stop','shared-network changes may alter attack mass and timing','50Hz human chronology versus variable live cadence','two rounds from one session; development holdout already examined'],gpu_used:false,training_performed:false}};
}
function run(originalPath,partialPath,destination){
 check(!fs.existsSync(destination),'output_exists');const original=fs.readFileSync(originalPath),partial=fs.readFileSync(partialPath);
 check(sha(original)==='71238da150db6e926170efdc27662f4426145aac7f69a7c93f619ece54d6cf1d','original_source_hash');
 check(sha(partial)==='b07f599e8f38ae0cfe59c3e7e0e10fa5aee00955f0483a9f763cfb662dbbfdef','partial_source_hash');
 const r=derive(original,partial);check(r.receipt.rows===11985&&r.receipt.splits[0].labels===1355&&r.receipt.splits[1].labels===1625&&r.receipt.sequences.length===4&&r.receipt.training_labeled_chunks_per_epoch===33,'human_counts');
 r.receipt.original_dataset_path=originalPath;r.receipt.partial_reference_path=partialPath;
 fs.mkdirSync(destination);fs.writeFileSync(path.join(destination,'human-conditional-movement.bin'),r.output,{flag:'wx'});fs.writeFileSync(path.join(destination,'manifest.json'),JSON.stringify(r.receipt,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify(r.receipt));return r.receipt;
}
if(require.main===module){try{check(process.argv.length===5,'usage_original_partial_newdir');run(...process.argv.slice(2));}catch(e){console.error(e.message);process.exitCode=1;}}
module.exports={derive,HEADER,ROW,UNAVAILABLE};
