'use strict';
// Preserve the protocol; optionally test an explicitly pinned checkpoint and bridge.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {spawn}=require('node:child_process');
const [original,encoder,driver,out,checkpoint,bridgeSha]=process.argv.slice(2);
if(!original||!encoder||!driver||!out||![original,encoder,driver,out].every(path.isAbsolute))
  throw Error('Usage: node run_live_mask_trial.cjs ORIGINAL_CONFIG NEW_ENCODER ORIGINAL_DRIVER NEW_OUTPUT [CHECKPOINT] [BRIDGE_SHA256]');
if(checkpoint&&!path.isAbsolute(checkpoint))throw Error('Checkpoint must be an absolute path');
if(bridgeSha&&!/^[a-f0-9]{64}$/.test(bridgeSha))throw Error('Bridge SHA256 must be lowercase hexadecimal');
const config=JSON.parse(fs.readFileSync(original,'utf8'));
if(!Array.isArray(config.encoder)||config.projection!=='client_pose_projection_v1'||config.enter_private!==true)
  throw Error('Expected existing explicit private-trial configuration');
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
fs.mkdirSync(out,{mode:0o700});
const adapted={...config,encoder:[encoder,...config.encoder.slice(1)],out:path.join(out,'trial'),max_seconds:140};
const changed=['encoder[0]','out','max_seconds'];
if(checkpoint){
  if(!Array.isArray(config.worker)||config.worker.length!==4||config.worker[2]!==config.checkpoint_sha256)
    throw Error('Expected pinned native live worker command: executable, checkpoint, SHA256, seed');
  adapted.checkpoint_sha256=hash(checkpoint);
  adapted.worker=[config.worker[0],checkpoint,adapted.checkpoint_sha256,config.worker[3]];
  changed.push('checkpoint_sha256','worker[1]','worker[2]');
}
if(bridgeSha){
  if(!Array.isArray(config.relay)||!/^[a-f0-9]{64}$/.test(config.relay.at(-1)))
    throw Error('Expected bridge SHA256 as final relay argument');
  adapted.relay=[...config.relay.slice(0,-1),bridgeSha];
  changed.push('relay[last]');
}
const adaptedPath=path.join(out,'trial.config.json');
fs.writeFileSync(adaptedPath,JSON.stringify(adapted,null,2)+'\n',{flag:'wx',mode:0o600});
fs.writeFileSync(path.join(out,'provenance.json'),JSON.stringify({host:require('node:os').hostname(),
  utc:new Date().toISOString(),command:[process.execPath,...process.argv.slice(1)],
  original_config_sha256:hash(original),adapted_config_sha256:hash(adaptedPath),encoder_sha256:hash(encoder),
  driver_sha256:hash(driver),checkpoint_sha256:adapted.checkpoint_sha256,
  original_checkpoint_sha256:config.checkpoint_sha256,
  bridge_sha256:adapted.relay?.at(-1),original_bridge_sha256:config.relay?.at(-1),
  changed_fields:changed,global_input_emitted:false},null,2)+'\n',{flag:'wx',mode:0o600});
const child=spawn(process.execPath,[driver,adaptedPath],{stdio:['ignore','pipe','pipe']});
const stdout=fs.createWriteStream(path.join(out,'stdout.jsonl'),{flags:'wx'}),stderr=fs.createWriteStream(path.join(out,'stderr.txt'),{flags:'wx'});
child.stdout.pipe(stdout);child.stdout.pipe(process.stdout);child.stderr.pipe(stderr);child.stderr.pipe(process.stderr);
let timer=setTimeout(()=>child.kill('SIGTERM'),240000);
child.on('error',e=>{console.error(e.message);process.exitCode=1;clearTimeout(timer);});
child.on('close',(code,signal)=>{clearTimeout(timer);fs.writeFileSync(path.join(out,'exit.json'),JSON.stringify({code,signal})+'\n',{flag:'wx'});process.exitCode=code??1;});
