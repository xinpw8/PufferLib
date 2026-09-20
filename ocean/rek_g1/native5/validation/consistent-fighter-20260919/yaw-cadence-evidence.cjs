'use strict';
// File-only, narrow r21-r28 outgoing-yaw measurement. No game or GPU access.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const readline=require('node:readline');
const assert=require('node:assert/strict');
const files=[];
async function scan(file,visit){
 const hash=crypto.createHash('sha256'),stream=fs.createReadStream(file);let bytes=0,line=0;
 stream.on('data',b=>{hash.update(b);bytes+=b.length;});
 for await(const text of readline.createInterface({input:stream,crlfDelay:Infinity})){line++;if(text)visit(JSON.parse(text),line);}
 files.push({path:file,bytes,sha256:hash.digest('hex')});
}
function quantile(v,p){if(!v.length)return null;const sorted=[...v].sort((a,b)=>a-b);return sorted[Math.floor((sorted.length-1)*p)];}
function distribution(v){return {count:v.length,p50:quantile(v,.5),p95:quantile(v,.95),maximum:v.length?Math.max(...v):null};}
function runDurations(rows){
 const all=[],busy=[],free=[];let current=null;
 for(let i=0;i<rows.length;i++){
  const row=rows[i],dt=i?row.t-rows[i-1].t:0;
  if(i&&(dt<=0||dt>.25))current=null; // Incomplete run at a gap is censored.
  if(current&&row.y!==current.y){
   if(current.y){const seconds=row.t-current.t;all.push(seconds);if(current.busy)busy.push(seconds);if(current.free)free.push(seconds);}
   current=null;
  }
  if(!current)current={t:row.t,y:row.y,busy:row.busy===true,free:row.busy===false};
  else {current.busy&&=row.busy===true;current.free&&=row.busy===false;}
 }
 const describe=v=>({...distribution(v),below_100ms:v.filter(x=>x<.1).length,at_least_500ms:v.filter(x=>x>=.5).length});
 return {all:describe(all),entirely_projected_busy:describe(busy),entirely_not_busy:describe(free)};
}
function yaw(desired){return [6,8,10,12,14].includes(desired)?1:[7,9,11,13,15].includes(desired)?-1:0;}
async function sourceRows(root,n){
 const dir=path.join(root,`live-round_outcome_v1-r${n}`,'trial'),ready=new Map();
 await scan(path.join(dir,'encoder.stdout.jsonl'),j=>{if(j.ready===true&&!j.worker_request.terminal){assert.equal(j.provenance.busy_projection,'dispatched_request_v4_duration');ready.set(j.worker_request.seq,j.provenance.projected_busy);}});
 const rows=[];
 await scan(path.join(dir,'encoder.stdin.jsonl'),j=>{if(ready.has(j.observation_sequence)){
  assert.ok(Number.isInteger(j.input.desired_action)&&j.input.desired_action>=1&&j.input.desired_action<=15);
  rows.push({seq:j.observation_sequence,t:j.clock.qpc_ticks/j.clock.qpc_frequency_hz,frequency:j.clock.qpc_frequency_hz,y:yaw(j.input.desired_action),busy:ready.get(j.observation_sequence)});
 }});
 assert.equal(rows.length,ready.size);return rows;
}
async function outgoing(captures,pid,source){
 const names=fs.readdirSync(captures).filter(s=>s.includes(`-pid${pid}-`)&&s.endsWith('.jsonl'));assert.equal(names.length,1,'unique bound native capture');
 const rows=[];let frequency=null;
 await scan(path.join(captures,names[0]),j=>{
  if(j.event==='capture_start'){assert.equal(j.pid,pid);frequency=j.stopwatch_frequency_hz;assert.ok(frequency>0);}
  if(j.event==='outbound_request_projection'&&j.message==='REK_Input'&&j.network_index_source_int32===0){
   assert.equal(frequency,source[0].frequency);assert.equal(j.request_only,true);
   const command=j.velocity_command_xyz;rows.push({t:j.stopwatch_timestamp_ticks/frequency,y:Math.sign(command[2]),value:Math.abs(command[2]),translation:Math.hypot(command[0],command[1])>0});
  }
 });
 let index=-1,excluded=0;const busy=[],free=[];
 for(const row of rows){
  while(index+1<source.length&&source[index+1].t<row.t)index++;
  if(index<0||row.t-source[index].t>.05){excluded++;continue;}
  row.busy=source[index].busy;(row.busy?busy:free).push(row);
 }
 const magnitudes=rs=>distribution(rs.filter(r=>r.value>0).map(r=>r.value));
 return {pid,commands:rows.length,nonzero_yaw:magnitudes(rows),yaw_at_least_point9:rows.filter(r=>r.value>=.9).length,translation_commands:rows.filter(r=>r.translation).length,completed_sign_runs:runDurations(rows),strict_preceding_source_maximum_age_seconds:.05,busy_join_excluded:excluded,busy_commands:busy.length,busy_nonzero_yaw:magnitudes(busy),not_busy_commands:free.length,not_busy_nonzero_yaw:magnitudes(free)};
}
async function main(){
 const [root,captures,human,output]=process.argv.slice(2);assert.ok(root&&captures&&human&&output,'TRIAL_ROOT CAPTURE_DIRECTORY HUMAN_COMMAND_LEDGER NEW_JSON');
 assert.ok(!fs.existsSync(output),'fresh output required');
 const source=new Map(),retained=[];
 for(let n=21;n<=28;n++){
  const rows=await sourceRows(root,n);source.set(n,rows);let observed=0,busy=0,changes=0,busyChanges=0;
  for(let i=1;i<rows.length;i++){const a=rows[i-1],b=rows[i],dt=b.t-a.t;if(dt<=0||dt>.25)continue;observed+=dt;if(a.busy)busy+=dt;if(a.y!==b.y){changes++;if(a.busy&&b.busy)busyChanges++;}}
  retained.push({round:n,ready_rows:rows.length,observed_seconds:observed,projected_busy_seconds:busy,sign_changes:changes,busy_sign_changes:busyChanges,completed_nonzero_sign_runs:runDurations(rows)});
 }
 const native=[];
 // Ownership is a pretty-printed JSON document, unlike the measured streams.
 for(const n of [21,28]){const p=path.join(root,`live-round_outcome_v1-r${n}`,'ownership.json'),b=fs.readFileSync(p),o=JSON.parse(b);files.push({path:p,bytes:b.length,sha256:crypto.createHash('sha256').update(b).digest('hex')});native.push({round:n,...await outgoing(captures,o.pid,source.get(n))});}
 const humanYaw=[];let humanCommands=0,humanHigh=0;
 await scan(human,j=>{if(j.message==='REK_Input'){humanCommands++;const v=Math.abs(j.velocity_command_xyz[2]);if(v)humanYaw.push(v);if(v>=.9)humanHigh++;}});
 const result={measurement:'existing_recorded_yaw_cadence',retained,native,human:{commands:humanCommands,nonzero_yaw:distribution(humanYaw),yaw_at_least_point9:humanHigh},limitations:['Native prefixes prove invocation and projected request body only; server delivery, acceptance and execution remain unknown.','Busy is the saved candidate request-duration projection, not authoritative playback.','Human and policy magnitude distributions are observational and not context-matched.','Run ends at first changed sign; runs spanning >250ms gaps and the final right-censored run are excluded.'],files};
 fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({output,sha256:crypto.createHash('sha256').update(fs.readFileSync(output)).digest('hex'),rounds:retained.length}));
}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
