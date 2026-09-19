'use strict';
// Continue an already authenticated moogleod session on the isolated D21 desktop.
// No credentials, authentication, lease, input, game launch, or reconnect path.
const fs=require('node:fs'),net=require('node:net'),os=require('node:os'),path=require('node:path');
const {StringDecoder}=require('node:string_decoder');

const PIPE='\\\\.\\pipe\\rek-ui-bridge-v1';
const PROTOCOL='rek.ui_bridge.v1';
const PROOF='windows_native=1;station=WinSta0;desktop=RekPolicyEval;input_desktop=Default;host=D21';
const COMMAND_ID='native-authenticated-continuation';
const APPROVAL='--approved-authenticated-continuation';

function validateInvocation(args,platform=process.platform,hostname=os.hostname()) {
  const [bridgeSha256,out,authorization]=args;
  if(args.length!==3||platform!=='win32'||hostname!=='D21'||
      !/^[a-f0-9]{64}$/.test(bridgeSha256||'')||authorization!==APPROVAL)
    throw Error('Requires native D21, bridge SHA256, new output path and '+APPROVAL);
  if(!out||!path.win32.isAbsolute(out)||/(^|[\\/])onedrive[^\\/]*([\\/]|$)/i.test(out))
    throw Error('Evidence output must be an absolute path outside OneDrive');
  return {bridgeSha256,out};
}

// A supplied, unconnected socket keeps unit tests independent of the named pipe.
function runContinuation({socket,bridgeSha256,log=()=>{},timers={setTimeout,clearTimeout},
    timeoutMs=90000,pollMs=1000}) {
  if(!/^[a-f0-9]{64}$/.test(bridgeSha256||''))throw Error('Bridge SHA256 is required');
  if(!Number.isFinite(timeoutMs)||timeoutMs<=0||!Number.isFinite(pollMs)||pollMs<=0)
    throw Error('Positive timeout and polling intervals are required');
  return new Promise(resolve=>{
    let buffer='',done=false,connected=false,commandSent=false,ackReceived=false;
    let stateSequence=0,pendingState=null,nextTimer;
    const decoder=new StringDecoder('utf8');
    const deadline=timers.setTimeout(()=>finish(false,'state_or_continuation_timeout'),timeoutMs);

    function finish(passed,reason) {
      if(done)return;
      done=true;
      timers.clearTimeout(deadline);
      timers.clearTimeout(nextTimer);
      socket.destroy();
      const result={passed,reason,command_sent:commandSent,ack_received:ackReceived};
      try {log({event:'result',...result});}
      catch {result.passed=false;result.reason='evidence_write_failed';}
      resolve(result);
    }
    function record(value) {
      try {log(value);return true;}
      catch {finish(false,'evidence_write_failed');return false;}
    }
    function send(value) {
      if(done||!record({event:'request',...value}))return;
      try {socket.write(JSON.stringify(value)+'\n');}
      catch(error) {finish(false,error.code||'pipe_write_failed');}
    }
    function state() {
      if(done||pendingState!==null)return;
      pendingState='native-authenticated-state-'+(++stateSequence);
      send({type:'get_state',request_id:pendingState});
    }
    function receive(message) {
      if(!message||typeof message!=='object'||Array.isArray(message))
        return finish(false,'invalid_message');
      if(message.event==='error') {
        if(record({event:'error',request_id:message.request_id,reason:message.reason}))
          finish(false,'bridge_error:'+String(message.reason));
        return;
      }
      if(message.event==='ack'&&message.request_id===COMMAND_ID) {
        if(!commandSent||ackReceived)return;
        if(message.protocol!==PROTOCOL||message.command!=='ConfirmLoggedIn')
          return finish(false,'invalid_continuation_ack');
        ackReceived=true;
        if(!record({event:'ack',applied:message.applied,reason:message.reason,command:message.command}))return;
        if(message.applied!==true)return finish(false,'continuation_rejected:'+String(message.reason));
        state();
        return;
      }
      if(message.event!=='state'||pendingState===null||message.request_id!==pendingState)return;
      pendingState=null;
      const fg=message.foreground,account=fg?.windows_policy_account,login=message.login;
      if(!record({event:'state',request_id:message.request_id,observed_utc:message.observed_utc,
          scene:message.scene,screen:message.lobby_screen,build:message.build,foreground:fg,
          login,lease_held:message.control?.lease_held}))return;
      if(message.protocol!==PROTOCOL||message.build?.plugin_sha256!==bridgeSha256||
          fg?.isolated_session_verified!==true||fg?.isolated_session_proof!==PROOF||
          fg?.execution_surface!=='native_windows_isolated_desktop'||
          message.control?.lease_held!==false||message.control?.g1_policy_stream_running!==false)
        return finish(false,'runtime_scope_mismatch');
      if(message.scene!=='Lobby')return finish(false,'unexpected_scene');
      if(message.lobby_screen==='Home') {
        if(account?.context_fighter_name!=='moogleod'||account?.home_display_matches!==true||account?.allowed!==true)
          return finish(false,'home_account_mismatch');
        return finish(true,'moogleod_home_observed');
      }
      if(commandSent)return finish(false,'home_postcondition_not_observed');
      if(message.lobby_screen==='Intro') {
        nextTimer=timers.setTimeout(state,pollMs);
        return;
      }
      if(message.lobby_screen!=='Login')return finish(false,'unexpected_lobby_screen');
      if(login?.native_authenticated_continuation_allowed!==true||login?.already_logged_in!==true||
          login?.fighter_name_label!=='moogleod'||login?.context_fighter_name!=='moogleod'||
          login?.continuation_attempted!==false||login?.prelease_idle!==true)
        return finish(false,'human_or_unverified_login:'+String(login?.reason));
      // Set before writing: a transport failure can leave the callback outcome unknown.
      commandSent=true;
      send({type:'command',request_id:COMMAND_ID,command:'ConfirmLoggedIn'});
    }
    socket.on('connect',()=>{if(!connected&&!done){connected=true;state();}});
    socket.on('error',error=>finish(false,error.code||'pipe_error'));
    socket.on('end',()=>finish(false,'pipe_closed'));
    socket.on('close',()=>finish(false,'pipe_closed'));
    socket.on('data',chunk=>{
      if(done)return;
      buffer+=decoder.write(chunk);
      if(Buffer.byteLength(buffer,'utf8')>1048576)return finish(false,'message_bound_exceeded');
      let index;
      while(!done&&(index=buffer.indexOf('\n'))!==-1) {
        const line=buffer.slice(0,index);buffer=buffer.slice(index+1);
        let message;
        try {message=JSON.parse(line);}
        catch {return finish(false,'invalid_json');}
        receive(message);
      }
    });
  });
}

async function main(args) {
  const {bridgeSha256,out}=validateInvocation(args);
  // Exclusive creation preserves existing evidence and prevents an accidental rerun.
  const evidence=fs.openSync(out,'wx',0o600);
  try {
    const socket=new net.Socket();
    const pending=runContinuation({socket,bridgeSha256,
      log:value=>fs.writeSync(evidence,JSON.stringify({utc:new Date().toISOString(),...value})+'\n')});
    try {socket.connect(PIPE);}
    catch(error) {socket.emit('error',error);}
    const result=await pending;
    console.log(JSON.stringify({...result,out}));
    process.exitCode=result.passed?0:2;
  } finally {fs.closeSync(evidence);}
}

module.exports={runContinuation,validateInvocation,PROOF};
if(require.main===module)main(process.argv.slice(2)).catch(error=>{
  console.error(error.message);process.exitCode=2;
});
