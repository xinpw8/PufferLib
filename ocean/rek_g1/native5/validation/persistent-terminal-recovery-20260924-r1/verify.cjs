'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const requireHash=(f,h)=>{if(hash(f)!==h)throw Error('source hash mismatch: '+f);};
const handoff=path.resolve(__dirname,'../persistent-private-session-20260924-r1/handoff/live_transfer_run_masked.cjs');
requireHash(handoff,'f6d31931186da3dce3621b4a952b7e853055446c8ab5aa934060d3263cfb6136');
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'rek-terminal-recovery-'));
fs.mkdirSync(path.join(scratch,'baseline'));
const target=path.join(scratch,'live_transfer_run_masked.cjs');fs.copyFileSync(handoff,target);
function apply(p){const options={cwd:scratch,stdio:'inherit',env:{...process.env,GIT_CEILING_DIRECTORIES:path.dirname(scratch)}};
 execFileSync('git',['-c','core.autocrlf=false','apply','--check',p],options);
 execFileSync('git',['-c','core.autocrlf=false','apply',p],options);}
apply(path.resolve(__dirname,'../persistent-native-redo-20260924-r1/redo-driver.patch'));
requireHash(target,'f1f25159c5155f15ce0a6035236c684fcc4f62f8f9843c54884bebdc59a33dab');
fs.copyFileSync(target,path.join(scratch,'baseline/live_transfer_run_masked.cjs'));
apply(path.join(__dirname,'driver.patch'));
requireHash(target,'9c55723297e8abeb10ea2a98ab9cd19978e2b97285b78b8f750b04214d73ca83');
fs.copyFileSync(path.join(__dirname,'terminal-recovery.test.cjs'),path.join(scratch,'terminal-recovery.test.cjs'));
execFileSync(process.execPath,['--test','terminal-recovery.test.cjs'],{cwd:scratch,stdio:'inherit'});
console.log(JSON.stringify({cpu_only:true,synthetic_tests:8,runtime_hashes_exact:true,scratch,
 native_passive_terminal_readback_validated:false,game_or_bridge_started:false}));
