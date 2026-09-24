'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),vm=require('node:vm');
const {EventEmitter}=require('node:events'),{PassThrough}=require('node:stream');
const {cleanClose,transportClosed,validHandoff,runCaptureWrapper}=require('./policy_handoff.cjs');
const driver=require('./live_transfer_run_masked.cjs');
const source=fs.readFileSync(path.join(__dirname,'campaign.cjs'),'utf8');
const base=fs.readFileSync(path.join(__dirname,'baseline/campaign.cjs'),'utf8');
const cp='a'.repeat(64),hash='b'.repeat(64),configHash='c'.repeat(64),driverHash='d'.repeat(64);
const expected={checkpoint:cp,trialOutput:'/trial',summarySha:hash,configSha:configHash,driverSha:driverHash};
const closed=()=>({code:0,signal:null,close_observed:true,signals_sent:[],timed_out:false});
function receipt(){return {event:'policy_handoff_ready',schema:'rek.policy_handoff.v1',
  config_sha256:configHash,driver_sha256:driverHash,media_finalized:false,driver_result:closed(),
  transport:{event:'policy_endpoints_closed',schema:'rek.policy_transport_closed.v1',
    checkpoint_sha256:cp,trial_output:'/trial',summary_sha256:hash,stream_stopped:true,lease_released:true,
    endpoints:['relay','encoder','worker'].map(name=>({name,...closed()}))}};}
function tmp(){return fs.mkdtempSync(path.join(__dirname,'test-output-'));}
function extract(from,to,context={}){
  const part=source.slice(source.indexOf(from),source.indexOf(to,source.indexOf(from)));
  return vm.runInNewContext(part+'\n'+from.match(/function (\w+)/)[1],context);
}
test('receipt requires actual clean closure of exactly relay, worker and encoder',()=>{
  assert.equal(validHandoff(receipt(),expected),true);
  for(const mutate of [r=>r.transport.endpoints.pop(),r=>r.transport.endpoints.push(r.transport.endpoints[0]),
    r=>r.transport.endpoints[0].close_observed=false,r=>r.transport.endpoints[0].code=2,
    r=>r.transport.endpoints[0].signal='SIGTERM',r=>r.transport.endpoints[0].signals_sent=['SIGTERM'],
    r=>r.driver_result.close_observed=false,r=>r.driver_result.timed_out=true,
    r=>r.driver_result.code=2,r=>r.driver_result.cleanup_timeout=true,
    r=>r.transport.stream_stopped=false,r=>r.transport.lease_released=false]){
    const r=receipt();mutate(r);assert.equal(validHandoff(r,expected),false);
  }
});
test('receipt binds dataset-independent runtime identity and exact summary/config/driver bytes',()=>{
  for(const mutate of [r=>r.transport.checkpoint_sha256='e'.repeat(64),r=>r.transport.trial_output='/another',
    r=>r.transport.summary_sha256='f'.repeat(64),r=>r.config_sha256=hash,r=>r.driver_sha256=hash,
    r=>r.media_finalized=true,r=>r.schema='wrong',r=>r.transport.schema='wrong']){
    const r=receipt();mutate(r);assert.equal(validHandoff(r,expected),false);
  }
});
test('real CPU endpoint promise waits for child close after stdin is ended',async()=>{
  const dir=tmp(),endpoint=driver.childEndpoint('relay',[process.execPath,'-e',
    'process.stdout.write(JSON.stringify({ready:true})+"\\n");process.stdin.resume();process.stdin.on("end",()=>setTimeout(()=>process.exit(0),25));'],dir);
  await endpoint.wait(x=>x.ready===true);
  let settled=false;endpoint.closed.then(()=>{settled=true;});
  endpoint.close();assert.equal(settled,false);
  const result=await endpoint.closed;
  assert.equal(result.name,'relay');assert.equal(cleanClose(result),true);
});
function fakeChild(){const c=new EventEmitter();c.stdout=new PassThrough();c.stderr=new PassThrough();
  c.kill=()=>{};return c;}
test('summary or bad receipt cannot release successor before wrapper closes',async()=>{
  const child=fakeChild(),dir=tmp(),rejected=[];
  const run=runCaptureWrapper('unused',[],{stdout:path.join(dir,'out'),stderr:path.join(dir,'err'),timeoutMs:10000,
    spawnChild:()=>child,validateHandoff:r=>validHandoff(r,expected),onRejected:r=>rejected.push(r)});
  let early=false;run.policyClosed.then(()=>{early=true;});
  child.stdout.write(JSON.stringify({event:'summary',final_round:{active:false}})+'\n');
  const bad=receipt();bad.transport.endpoints[0].close_observed=false;
  child.stdout.write(JSON.stringify(bad)+'\n');
  await new Promise(r=>setImmediate(r));assert.equal(early,false);assert.equal(rejected.length,1);
  child.stdout.end();child.stderr.end();child.emit('close',2,null);
  assert.deepEqual(await run.policyClosed,{handoff:null,final:{code:2,signal:null}});
  assert.deepEqual(await run.completed,{code:2,signal:null});
});
test('valid post-close receipt releases once while final media failure stays observable',async()=>{
  const child=fakeChild(),dir=tmp();
  const run=runCaptureWrapper('unused',[],{stdout:path.join(dir,'out'),stderr:path.join(dir,'err'),timeoutMs:10000,
    spawnChild:()=>child,validateHandoff:r=>validHandoff(r,expected)});
  let mediaDone=false;run.completed.then(()=>{mediaDone=true;});
  child.stdout.write(JSON.stringify(receipt())+'\n');child.stdout.write(JSON.stringify(receipt())+'\n');
  const early=await run.policyClosed;
  assert.equal(early.handoff.event,'policy_handoff_ready');assert.equal(early.final,null);assert.equal(mediaDone,false);
  child.stdout.end();child.stderr.end();child.emit('close',2,null);
  assert.deepEqual(await run.completed,{code:2,signal:null});
});
const verdict=extract('function verdict(dir,expectedCheckpoint){','async function attempt(',{fs,path});
function summary(){return {predictions:10,applied:9,rejected:1,opponent:{client_ai_difficulty:0,sparring_bot_number:1},
  checkpoint_sha256:cp,stop_reason:'stream_end:active_round_not_observed',
  initial_round:{number:2,duration:120,time_remaining:119.7,active:true,redo:false,clean_hits:[0,0]},
  final_round:{number:2,duration:120,time_remaining:0,active:false,redo:false,clean_hits:[2,14],winner_index:1,result:'WonByPoints',result_value:1}};}
function fixture(){const dir=tmp();fs.mkdirSync(path.join(dir,'trial'));fs.mkdirSync(path.join(dir,'media'));
  fs.writeFileSync(path.join(dir,'trial/summary.json'),JSON.stringify(summary()));
  fs.writeFileSync(path.join(dir,'config.json'),'{}');return dir;}
test('terminal scoring and fair-start function is byte-identical to deployed controller',()=>{
  const part=s=>s.slice(s.indexOf('function verdict('),s.indexOf('async function attempt('));
  assert.equal(part(source),part(base));
  const dir=fixture();assert.equal(verdict(dir,cp).complete,true);
  for(const mutate of [s=>s.initial_round.time_remaining=116.99,s=>s.initial_round.clean_hits[0]=1,
    s=>s.initial_round.number=1,s=>s.final_round.redo=true,s=>s.opponent.sparring_bot_number=2,
    s=>s.checkpoint_sha256=hash,s=>s.final_round.winner_index=0]){
    const s=summary();mutate(s);fs.writeFileSync(path.join(dir,'trial/summary.json'),JSON.stringify(s));
    assert.equal(verdict(dir,cp).complete,false);
  }
});
test('direct successor requires complete native round and same frozen checkpoint',()=>{
  const successor=extract('function successorHandoff(rec,entry){','async function main(){');
  const rec={complete:true,checkpoint_sha256:cp,policy_handoff:receipt()};
  assert.equal(successor(rec,{checkpoint_sha256:cp}),rec);
  assert.equal(successor(rec,{checkpoint_sha256:hash}),null);
  rec.complete=false;assert.equal(successor(rec,{checkpoint_sha256:cp}),null);
  rec.complete=true;rec.policy_handoff=null;assert.equal(successor(rec,{checkpoint_sha256:cp}),null);
});
async function attemptFixture(finalCode,lateManifest=true,hasHandoff=true,complete=true){
  const dir=fixture();if(!complete){const s=summary();s.predictions=0;fs.writeFileSync(path.join(dir,'trial/summary.json'),JSON.stringify(s));}
  const sumPath=path.join(dir,'trial/summary.json'),sha=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
  const handoff=receipt();handoff.transport.summary_sha256=sha(sumPath);
  let finish;const completed=new Promise(r=>{finish=r;});
  const ledger={},logs=[],pendingMedia=new Set();
  const context={fs,path,process,os:{hostname:()=> 'cpu-test'},driver:__filename,recorder:__filename,
    utc:()=>new Date().toISOString(),sha,verdict,ledger,pendingMedia,mediaFailed:false,saveLedger:()=>{},
    log:(event,data)=>logs.push({event,...data}),validHandoff,
    attemptPaths:()=>({dir,config:path.join(dir,'config.json'),media:path.join(dir,'media')}),
    runCaptureWrapper:()=>({policyClosed:Promise.resolve({handoff:hasHandoff?handoff:null,final:hasHandoff?null:{code:finalCode,signal:null}}),completed})};
  const attempt=extract('async function attempt(entry,n){','function successorHandoff(',context);
  const rec=await attempt({label:'round2',order:2,policy_rng_seed:1802,checkpoint_sha256:cp},1);
  assert.equal(rec.media_status,'pending');assert.equal(rec.complete,complete);assert.equal(ledger.round2.length,1);
  if(lateManifest)fs.writeFileSync(path.join(dir,'media/capture-manifest.json'),JSON.stringify({delivery_status:'validated_mp4',video_sha256:hash}));
  finish({code:finalCode,signal:null});await Promise.all([...pendingMedia]);
  return {rec,context,logs,ledger};
}
test('late successful media updates one ledger record without changing policy verdict',async()=>{
  const x=await attemptFixture(0);assert.equal(x.rec.media_status,'finalized');assert.equal(x.rec.complete,true);
  assert.equal(x.rec.wrapper_exit,0);assert.equal(x.rec.video_sha256,hash);assert.equal(x.context.mediaFailed,false);
  assert.equal(x.ledger.round2.length,1);
});
test('late recorder error preserves completed policy proof and marks campaign failure',async()=>{
  const x=await attemptFixture(2,false);assert.equal(x.rec.media_status,'failed');assert.equal(x.rec.complete,true);
  assert.equal(x.context.mediaFailed,true);assert.equal(x.rec.wrapper_exit,2);
  assert(x.logs.some(e=>e.event==='attempt_media_finalized'&&e.policy_complete_unchanged===true));
});
test('ordinary excluded pre-policy attempt stays retryable',async()=>{
  const x=await attemptFixture(2,false,false,false);assert.equal(x.rec.complete,false);
  assert.equal(x.context.mediaFailed,false);
});
test('normal policy criterion remains and explicit runtime-only plan is bounded',()=>{
  const condition=source.match(/if\((nonwins>=3[^)]*)\)/)[1];
  const fails=new Function('nonwins','runtimeOnly','return '+condition);
  assert.equal(fails(3,false),true);assert.equal(fails(3,true),false);assert.equal(fails(2,false),false);
  assert(source.includes('planDocument.runtime_only===true'));
  assert(source.includes('plan.length<1||plan.length>4'));
});
test('first observer poll is immediate and policy path is unchanged until finally',()=>{
  assert(source.includes('read();poll=setInterval(read,1000)'));
  const current=fs.readFileSync(path.join(__dirname,'live_transfer_run_masked.cjs'),'utf8');
  const original=fs.readFileSync(path.join(__dirname,'baseline/live_transfer_run_masked.cjs'),'utf8');
  const loop=s=>s.slice(s.indexOf('async function run(configPath)'),s.indexOf('  finally {'));
  assert.equal(loop(current),loop(original));
  const wrapper=fs.readFileSync(path.join(__dirname,'record_passive_defender.cjs'),'utf8');
  assert(wrapper.indexOf("event:'policy_handoff_ready'")>wrapper.indexOf('await driverControl.done'));
  assert(wrapper.indexOf("event:'policy_handoff_ready'")<wrapper.indexOf('setTimeout(resolve,3000)'));
  assert(current.indexOf("log('policy_endpoints_closed'")>current.indexOf('await Promise.all(endpoints.map(e=>e.closed))'));
  assert(!source.includes('recycle_owned_client.sh'));
});
