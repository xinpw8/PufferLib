'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');
const {spawnSync}=require('node:child_process');
const [binary,checkpoint,sha]=process.argv.slice(2);
assert(binary&&checkpoint&&/^[a-f0-9]{64}$/.test(sha));
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'rek-policy-features-'));
const maskPath=path.join(temp,'mask.bin');
const mask=Buffer.alloc(223,0);mask[0]=1;mask[1]=1;
const hash=crypto.createHash('sha256').update(mask).digest('hex');
fs.writeFileSync(maskPath,mask,{flag:'wx'});
function rollout(seed,perturb=false,features=maskPath,selection='argmax') {
  const requests=Array.from({length:96},(_,seq)=>({type:'step',seq,round_id:'a'.repeat(64),
    observation_schema:'rek.native5.scaled_polar_xy.v1',terminal:false,
    observation:Array.from({length:223},(_,i)=>i<2?Math.sin(seq/17+i):perturb?Math.sin(seq+i)*3:0),
    mask:Array.from({length:33},(_,i)=>i!==0)}));
  requests.push({type:'close',seq:96});
  const run=spawnSync(binary,[checkpoint,sha,String(seed),selection,features],{
    input:requests.map(x=>JSON.stringify(x)).join('\n')+'\n',encoding:'utf8',timeout:120000,maxBuffer:4e6});
  return {run,messages:run.stdout.trim().split('\n').filter(Boolean).map(x=>JSON.parse(x))};
}
try {
  const first=rollout(73),other=rollout(819,true);
  for(const {run,messages} of [first,other]) {
    assert.equal(run.status,0,run.stderr);assert.equal(messages[0].feature_mask_sha256,hash);
    assert.equal(messages[0].selection,'argmax');
    assert.equal(messages.filter(x=>x.type==='action').length,96);
    for(const x of messages.filter(x=>x.type==='action'))assert(x.action>0&&x.action<33);
  }
  assert.deepEqual(first.messages.filter(x=>x.type==='action').map(x=>x.action),
    other.messages.filter(x=>x.type==='action').map(x=>x.action),
    'masked argmax must ignore RNG seed and excluded inputs over the entire recurrent sequence');
  const failures=[];
  for(const [name,bytes] of [['short',Buffer.alloc(222,1)],['long',Buffer.alloc(224,1)],
    ['invalid',Buffer.alloc(223,2)],['empty',Buffer.alloc(223,0)]]) {
    const file=path.join(temp,name);fs.writeFileSync(file,bytes,{flag:'wx'});
    const {run,messages}=rollout(73,false,file);
    assert.equal(run.status,2,name);assert(!messages.some(x=>x.type==='ready'||x.type==='action'),name);
    failures.push(name);
  }
  const invalid=rollout(73,false,maskPath,'random');assert.equal(invalid.run.status,2);
  console.log(JSON.stringify({test:'live-policy-selection-features',passed:true,
    native_cuda:true,authentic_game_control:false,synthetic_observations:true,
    recurrent_decisions_compared:96,feature_mask_sha256:hash,rejected_mask_cases:failures,
    checkpoint_sha256:sha}));
} finally {
  for(const file of fs.readdirSync(temp))fs.unlinkSync(path.join(temp,file));
  fs.rmdirSync(temp);
}
