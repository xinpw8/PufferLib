'use strict';
const fs=require('node:fs'),crypto=require('node:crypto');
const check=(ok,message)=>{if(!ok)throw new Error(message);};
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const [label,dataPath,replayPath]=process.argv.slice(2);
check(label&&dataPath,'Usage: summarize-action-probabilities LABEL DATA [REPLAY]');
const data=fs.readFileSync(dataPath),magic=data.toString('ascii',0,8),version=data.readUInt32LE(8);
check((magic==='REKRL001'&&version===1)||(magic==='REKRL002'&&version===2),'dataset format');
const n=data.readUInt32LE(20);check(data.length===256+1128*n&&data.readUInt32LE(12)===223&&data.readUInt32LE(16)===33,'dataset shape');
const replay=replayPath?fs.readFileSync(replayPath):null;
if(replay)check(replay.length===128+152*n&&replay.toString('ascii',0,8)===(version===1?'REKBR001':'REKBR002')&&replay.readUInt32LE(8)===version&&replay.subarray(24,56).toString('hex')===sha(data),'replay binding');
const categories=[17,23].map(category=>({category,legal_rows:0,selected:0,applied_selected:0,probabilities:[],attack_conditional:[],legal_attack_masses:[]}));
let selectedAttacks=0,attackLegalRows=0,applied=0,sharedLegal=0;
for(let i=0;i<n;++i){
  const o=256+i*1128,a=data.readInt32LE(o+12),accepted=data.readUInt32LE(o+1116);
  const mask=Array.from({length:33},(_,j)=>data.readFloatLE(o+924+4*j));
  check(mask.every(x=>x===0||x===1)&&mask[a]===1,'invalid saved action mask');
  selectedAttacks+=a>=16;applied+=accepted;attackLegalRows+=mask.slice(16).some(Boolean);sharedLegal+=mask[17]&&mask[23];
  let probabilities=null,attackMass=0;
  if(replay){
    const r=128+i*152;check(replay.readUInt32LE(r)===i&&replay.readUInt32LE(r+4)===a,'replay row identity');
    const logits=Array.from({length:33},(_,j)=>replay.readFloatLE(r+16+4*j));check(logits.every(Number.isFinite),'nonfinite logits');
    const maximum=Math.max(...logits.filter((_,j)=>mask[j]));
    const weights=logits.map((x,j)=>mask[j]?Math.exp(x-maximum):0),sum=weights.reduce((x,y)=>x+y,0);
    probabilities=weights.map(x=>x/sum);attackMass=probabilities.slice(16).reduce((x,y)=>x+y,0);
  }
  for(const item of categories){
    item.selected+=a===item.category;item.applied_selected+=a===item.category&&accepted===1;
    if(mask[item.category]){++item.legal_rows;if(replay){item.probabilities.push(probabilities[item.category]);item.attack_conditional.push(probabilities[item.category]/attackMass);item.legal_attack_masses.push(attackMass);}}
  }
}
function distribution(values){const sorted=[...values].sort((a,b)=>a-b),sum=values.reduce((a,b)=>a+b,0);return {mean:sum/values.length,p50:sorted[Math.floor((sorted.length-1)*.5)],p95:sorted[Math.floor((sorted.length-1)*.95)],max:sorted.at(-1),sum};}
const report={label,dataset_sha256:sha(data),schema:magic,rows:n,applied_rows:applied,attack_legal_rows:attackLegalRows,shared_17_23_legal_rows:sharedLegal,selected_attack_requests:selectedAttacks,
  replay_sha256:replay?sha(replay):null,checkpoint_sha256:replay?replay.subarray(56,88).toString('hex'):null,
  probability_method:replay?'CPU float64 legal softmax of exact native BF16 replay logits; conditional means given selection from legal attacks16..32':'probabilities unavailable without native replay',
  action_semantics:'saved sampled request; applied is local acknowledgement, not server execution or contact',
  categories:categories.map(({probabilities,attack_conditional,legal_attack_masses,...item})=>({...item,legal_fraction:item.legal_rows/n,selected_share_of_attack_requests:item.selected/selectedAttacks,
    probability_when_legal:replay?distribution(probabilities):null,probability_conditional_on_legal_attack:replay?distribution(attack_conditional):null,
    aggregate_attack_mass_weighted_conditional_probability:replay?probabilities.reduce((a,b)=>a+b,0)/legal_attack_masses.reduce((a,b)=>a+b,0):null}))};
console.log(JSON.stringify(report,null,2));
