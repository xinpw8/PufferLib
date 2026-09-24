'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict');
const trial=process.argv[2];assert(trial?.startsWith('/home/spark-advantage/rek-training/timing500-ppo-lr3e5-live-20260924-r1/lr3e5-s'));
const fd=fs.openSync(trial+'/relay.stdout.jsonl','r'),size=fs.fstatSync(fd).size,buf=Buffer.alloc(size);let offset=0;
while(offset<size){const count=fs.readSync(fd,buf,offset,size-offset,offset);assert(count>0);offset+=count;}fs.closeSync(fd);
const states=new Map(),result={utc:new Date().toISOString(),snapshot_bytes:size,applied:0,rejected:0,applied_over250ms:0,applied_over500ms:0,max_applied_age_ms:0,actions:{},armed_attacks:{},latest_round:null};
for(const line of buf.toString().split('\n')){let row;try{row=JSON.parse(line);}catch{continue;}
 if(row.event==='g1_policy_state'){states.set(row.observation_sequence,row);result.latest_round=row.round;}
 if(row.event!=='g1_policy_action')continue;
 if(!row.applied){result.rejected++;continue;}
 result.applied++;result.actions[row.action]=(result.actions[row.action]??0)+1;
 if(row.reason==='accepted_locally_and_armed')result.armed_attacks[row.action]=(result.armed_attacks[row.action]??0)+1;
 const source=states.get(row.observation_sequence);if(!source||source.round_identity_sha256!==row.round_identity_sha256)continue;
 const age=Number(BigInt(row.clock.qpc_ticks)-BigInt(source.clock.qpc_ticks))/source.clock.qpc_frequency_hz*1000;
 result.max_applied_age_ms=Math.max(result.max_applied_age_ms,age);result.applied_over250ms+=age>250;result.applied_over500ms+=age>500;
}
console.log(JSON.stringify(result));
