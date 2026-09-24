'use strict';
const fs=require('node:fs'),readline=require('node:readline');
const {spawn}=require('node:child_process');

function cleanClose(result) {
  return result?.code===0 && result.signal===null && result.close_observed===true &&
    !result.timed_out && !result.error && !result.cleanup_timeout &&
    (!result.signals_sent || result.signals_sent.length===0);
}
function transportClosed(receipt,{checkpoint,trialOutput,summarySha}) {
  const endpoints=receipt?.endpoints;
  return receipt?.event==='policy_endpoints_closed' && receipt.schema==='rek.policy_transport_closed.v1' &&
    receipt.checkpoint_sha256===checkpoint && receipt.trial_output===trialOutput &&
    receipt.summary_sha256===summarySha && receipt.stream_stopped===true && receipt.lease_released===true &&
    Array.isArray(endpoints) && endpoints.length===3 &&
    ['relay','encoder','worker'].every(name=>endpoints.filter(e=>e.name===name).length===1) &&
    endpoints.every(cleanClose);
}
function validHandoff(receipt,expected) {
  return receipt?.event==='policy_handoff_ready' && receipt.schema==='rek.policy_handoff.v1' &&
    receipt.config_sha256===expected.configSha && receipt.driver_sha256===expected.driverSha &&
    receipt.media_finalized===false && cleanClose(receipt.driver_result) &&
    transportClosed(receipt.transport,expected);
}

// The early promise can resolve only on a validated post-close receipt. The final
// promise stays owned until media finalization and the wrapper itself have closed.
function runCaptureWrapper(cmd,args,{stdout,stderr,timeoutMs,validateHandoff,
    onRejected=()=>{},spawnChild=spawn}) {
  const out=fs.createWriteStream(stdout,{flags:'wx',mode:0o600});
  const err=fs.createWriteStream(stderr,{flags:'wx',mode:0o600});
  const child=spawnChild(cmd,args,{stdio:['ignore','pipe','pipe']});
  let resolvePolicy,resolveComplete,released=false,spawnError=null;
  const policyClosed=new Promise(resolve=>{resolvePolicy=resolve;});
  const completed=new Promise(resolve=>{resolveComplete=resolve;});
  const release=value=>{if(!released){released=true;resolvePolicy(value);}};
  child.stdout.pipe(out);child.stderr.pipe(err);
  const lines=readline.createInterface({input:child.stdout});
  lines.on('line',raw=>{
    if(released)return;
    let event;try{event=JSON.parse(raw);}catch{return;}
    if(event.event!=='policy_handoff_ready')return;
    let valid=false;try{valid=validateHandoff(event)===true;}catch{}
    if(valid)release({handoff:event,final:null});
    else onRejected(event);
  });
  const timer=setTimeout(()=>{try{child.kill('SIGTERM');}catch{}},timeoutMs);
  child.on('error',error=>{spawnError=error.message;});
  child.once('close',(code,signal)=>{
    clearTimeout(timer);lines.close();
    const final={code,signal,...(spawnError?{error:spawnError}:{})};
    release({handoff:null,final});resolveComplete(final);
  });
  return {policyClosed,completed};
}
module.exports={cleanClose,transportClosed,validHandoff,runCaptureWrapper};
