'use strict';
const fs=require('node:fs');
const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const paths=process.argv.slice(2);
assert.equal(paths.length,3,'OLD_DEFAULT NEW_STRIDE1 NEW_STRIDE5');
const [old,one,five]=paths.map(p=>fs.readFileSync(p));
assert.deepEqual(one,old,'stride1 changed archived default numerical outputs');
// Header20 + masks99 + float819*4 + round14*8 =3507 bytes per record.
const recordBytes=20+99+(223+446+72+70+2+2+2+1+1)*4+14*8;
assert.equal(one.length,five.length);assert.equal(one.length%recordBytes,0);
let checks=0,restricted=0,botRows=0,terminalRows=0,initialRows=0;
for(let b=0;b<one.length;b+=recordBytes){
 assert.deepEqual(five.subarray(b,b+20),one.subarray(b,b+20));
 const arena=one.readUInt32LE(b+8),tick=one.readUInt32LE(b+12),terminal=one.readUInt32LE(b+16);
 const restrict=arena!==1&&!terminal&&tick%5!==0;
 restricted+=restrict;botRows+=arena===1;terminalRows+=terminal!==0;initialRows+=tick===0;
 for(let k=0;k<99;k++){
  const learner=k<33||k>=66,action=k%33;
  const expected=restrict&&learner&&action!==0?0:one[b+20+k];
  assert.equal(five[b+20+k],expected,`mask mismatch arena${arena} tick${tick} field${k}`);checks++;
 }
 assert.deepEqual(five.subarray(b+119,b+recordBytes),one.subarray(b+119,b+recordBytes),'mask-only option changed fixed-history state/observations/rewards');
}
assert.ok(restricted&&botRows&&terminalRows&&initialRows);
console.log(JSON.stringify({test:'action_cadence_runtime_differential',passed:true,records:one.length/recordBytes,mask_checks:checks,restricted,bot_rows:botRows,terminal_rows:terminalRows,initial_rows:initialRows,default_all_bytes_equal:true,fixed_history_nonmask_all_bytes_equal:true,files:paths.map((p,i)=>({path:p,bytes:[old,one,five][i].length,sha256:crypto.createHash('sha256').update([old,one,five][i]).digest('hex')}))},null,2));
