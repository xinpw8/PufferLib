'use strict';
// Reconstruct two private source variants in a fresh temporary directory and run CPU tests only.
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const base=path.resolve(__dirname,'../persistent-private-session-20260924-r1/handoff');
const pins={
 'campaign.cjs':'fe839414eca8d366297e877988b569e714ac30b6d0cd1176063ca683f3da4b8e',
 'live_transfer_run_masked.cjs':'f6d31931186da3dce3621b4a952b7e853055446c8ab5aa934060d3263cfb6136',
 'record_passive_defender.cjs':'9a29c88f30886654f5afe57e573245e2005ef790909ae656ec67114c19902e07',
 'policy_handoff.cjs':'482096b5b0a4e7473ee2f2ec3b9e96036bb4159954d560d1e23a67292d5daa72'};
const sha=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
function check(f,pin){if(sha(f)!==pin)throw Error('source hash mismatch: '+f);}
for(const [name,pin]of Object.entries(pins))check(path.join(base,name),pin);
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'rek-native-redo-'));
const redo=path.join(scratch,'persistent-native-redo-20260924-r1');
const any=path.join(scratch,'persistent-runtime-any-ai-20260924-r1');
fs.mkdirSync(redo);fs.mkdirSync(path.join(redo,'baseline'));fs.mkdirSync(any);fs.mkdirSync(path.join(any,'baseline'));
for(const name of Object.keys(pins)){
 fs.copyFileSync(path.join(base,name),path.join(redo,name));
 fs.copyFileSync(path.join(base,name),path.join(redo,'baseline',name));
}
function apply(dir,file){const patch=path.join(__dirname,file);
 const options={cwd:dir,stdio:'inherit',env:{...process.env,GIT_CEILING_DIRECTORIES:scratch}};
 execFileSync('git',['-c','core.autocrlf=false','apply','--check',patch],options);
 execFileSync('git',['-c','core.autocrlf=false','apply',patch],options);}
apply(redo,'redo-controller.patch');apply(redo,'redo-driver.patch');
check(path.join(redo,'campaign.cjs'),'5c0af77071fe561e9da70fc41c130adcebab3dc157927673bafbd7b29fbd22b7');
check(path.join(redo,'live_transfer_run_masked.cjs'),'f1f25159c5155f15ce0a6035236c684fcc4f62f8f9843c54884bebdc59a33dab');
fs.copyFileSync(path.join(__dirname,'redo.test.cjs'),path.join(redo,'redo.test.cjs'));
fs.copyFileSync(path.join(redo,'campaign.cjs'),path.join(any,'campaign.cjs'));
fs.copyFileSync(path.join(redo,'campaign.cjs'),path.join(any,'baseline','campaign.cjs'));
apply(any,'runtime-any-ai.patch');
check(path.join(any,'campaign.cjs'),'77326a4d18296af696715c6c29e2335afe212545ef03c30be5ca58750361d926');
fs.copyFileSync(path.join(__dirname,'any-ai.test.cjs'),path.join(any,'any-ai.test.cjs'));
execFileSync(process.execPath,['--test','redo.test.cjs'],{cwd:redo,stdio:'inherit'});
execFileSync(process.execPath,['--test','any-ai.test.cjs'],{cwd:any,stdio:'inherit'});
console.log(JSON.stringify({cpu_only:true,tests:22,scratch,source_hashes_verified:true,
 runtime_sources_changed:false,game_or_bridge_started:false}));
