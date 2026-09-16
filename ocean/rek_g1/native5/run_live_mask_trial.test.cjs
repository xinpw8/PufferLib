'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const crypto=require('node:crypto');

function fixture(t){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'rek-trial-config-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const encoder=path.join(root,'encoder'),checkpoint=path.join(root,'checkpoint');
  const driver=path.join(root,'driver.cjs'),original=path.join(root,'original.json');
  fs.writeFileSync(encoder,'unused test encoder');fs.writeFileSync(checkpoint,'test weights');
  fs.writeFileSync(driver,'process.exit(0);');
  const config={projection:'client_pose_projection_v1',enter_private:true,
    encoder:['old-encoder','--projection','client_pose_projection_v1'],
    checkpoint_sha256:'a'.repeat(64),worker:['worker','old-checkpoint','a'.repeat(64),'73'],
    relay:['relay','policy-relay','b'.repeat(64)]};
  fs.writeFileSync(original,JSON.stringify(config));
  const out=path.join(root,'out');
  return {root,config,out,checkpoint,args:[original,encoder,driver,out]};
}
const wrapper=path.join(__dirname,'run_live_mask_trial.cjs');
test('unpinned adaptation preserves checkpoint and bridge',t=>{
  const f=fixture(t),result=spawnSync(process.execPath,[wrapper,...f.args],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
  const adapted=JSON.parse(fs.readFileSync(path.join(f.out,'trial.config.json')));
  assert.deepEqual(adapted.worker,f.config.worker);assert.deepEqual(adapted.relay,f.config.relay);
});
test('explicit bridge changes only final relay argument and records both hashes',t=>{
  const f=fixture(t),bridge='c'.repeat(64);
  const result=spawnSync(process.execPath,[wrapper,...f.args,f.checkpoint,bridge],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
  const adapted=JSON.parse(fs.readFileSync(path.join(f.out,'trial.config.json')));
  assert.deepEqual(adapted.relay,[...f.config.relay.slice(0,-1),bridge]);
  assert.equal(adapted.worker[3],'73');
  assert.equal(adapted.checkpoint_sha256,crypto.createHash('sha256').update('test weights').digest('hex'));
  const p=JSON.parse(fs.readFileSync(path.join(f.out,'provenance.json')));
  assert.equal(p.bridge_sha256,bridge);assert.equal(p.original_bridge_sha256,'b'.repeat(64));
  assert.ok(p.changed_fields.includes('relay[last]'));
});
test('invalid bridge identity fails before creating a trial',t=>{
  const f=fixture(t),result=spawnSync(process.execPath,[wrapper,...f.args,f.checkpoint,'bad'],{encoding:'utf8'});
  assert.notEqual(result.status,0);assert.match(result.stderr,/Bridge SHA256/);
  assert.equal(fs.existsSync(f.out),false);
});
