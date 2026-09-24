'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const read=p=>fs.readFileSync(__dirname+'/'+p,'utf8');
function applySingleHunk(source,patch){
 const lines=patch.split('\n');if(lines.at(-1)==='')lines.pop();
 assert(lines[0].startsWith('--- '));assert(lines[1].startsWith('+++ '));
 const h=/^@@ -(\d+),(\d+) \+(\d+),(\d+) @@$/.exec(lines[2]);assert(h);
 assert.equal(h[1],h[3]);
 const before=[],after=[];
 for(const line of lines.slice(3)){
  assert([' ','+','-'].includes(line[0]));
  if(line[0]!=='+' )before.push(line.slice(1));
  if(line[0]!=='-' )after.push(line.slice(1));
 }
 assert.equal(before.length,Number(h[2]));assert.equal(after.length,Number(h[4]));
 const input=source.split('\n'),start=Number(h[1])-1;
 assert.deepEqual(input.slice(start,start+before.length),before);
 input.splice(start,before.length,...after);return input.join('\n');
}
const original=read('../persistent-private-session-20260924-r1/launcher/relaunch.sh');
const baseline=applySingleHunk(original,read('eventpipe-baseline.patch'));
const candidate=applySingleHunk(baseline,read('launcher.diff'));
test('published parent reconstructs the exact EventPipe baseline in memory',()=>{
 assert.equal(sha(original),'41d685616f90c1f78e70a744ee7e7059537ce33d2085cc4da87b96f8163483d7');
 assert.equal(sha(baseline),'9159f8b016e2c04975c6f2ed4d241f4cf9c211eee2f1d3ec62bda13b362c8c11');
});
test('candidate hash and complete delta are exactly one executable path',()=>{
 assert.equal(sha(candidate),'7aad20605ffe9cee07662ad047acf24afd58996ac7fef3c3e4981582543901ce');
 const oldPath='/opt/codexrook/box64/bin/box64',newPath='/opt/codexrook/box64-upstream-20260924-r1/bin/box64';
 assert.equal(candidate.split(newPath).length-1,1);assert(!candidate.includes(oldPath));
 assert.equal(candidate.replace(newPath,oldPath),baseline);
});
test('default CALLRET/BIGBLOCK, barriers, logs, EventPipe and refusal guards remain unchanged',()=>{
 assert(!candidate.includes('BOX64_DYNAREC_CALLRET='));assert(!candidate.includes('BOX64_DYNAREC_BIGBLOCK='));
 for(const text of ['BOX64_DYNAREC_STRONGMEM=2','BOX64_DYNAREC_WEAKBARRIER=0','BOX64_LOG=1','BOX64_SHOWSEGV=1',
 'BOX64_SHOWBT=1','DOTNET_EnableEventPipe=1','DOTNET_EventPipeConfig=Microsoft-Windows-DotNETRuntime:18:5',
 'DOTNET_EventPipeCircularMB=10','DOTNET_EventPipeOutputStreaming=1','DOTNET_EventPipeOutputPath=',
 'Existing REK process; launch cancelled','Existing prefix user; launch cancelled',
 '5a2edac6c586f1ea401d92e0086ebfc468dbc591e2bb11115aef056682280e7a']){
  assert(candidate.includes(text));assert(baseline.includes(text));
 }
});
