'use strict';
(() => {
  const $=id=>document.getElementById(id),{SavedRekKeyboard,savedRekBindings}=globalThis.RekControls;
  const keyboard=new SavedRekKeyboard(),held=new Set(),moveKeys=new Map(['W','A','S','D','Q','E'].map(k=>['Key'+k,k]));
  let paused=true,ready=false,stopped=false,seq=Date.now()*1024,blob=null,lastTick=null,lastAdvance=performance.now();
  let frameETag=null,frameRequest=null;
  let inputChain=Promise.resolve(),playChain=Promise.resolve(),generation=0,changing=false;
  const showError=message=>{$('error').hidden=!message;$('error').textContent=message||'';};
  async function api(url,body){
    const r=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000),...(body===undefined?{}:
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),keepalive:true})});
    const value=await r.json();if(!r.ok)throw Error(value.error||value.failure||`HTTP ${r.status}`);return value;
  }
  function displayPause(){
    $('play').textContent=paused?'Play':'Pause';
    $('step').disabled=!ready||!paused||changing;
    $('focus-prompt').hidden=!paused&&document.activeElement===$('arena');
    $('focus-prompt').textContent=paused?'Paused. Click here to play.':'Click here to control your robot';
  }
  function sendPacket(packet){
    const g=generation;
    inputChain=inputChain.catch(()=>{}).then(()=>g===generation&&!changing?api('/api/input',packet):null)
      .catch(e=>{if(g===generation)showError(e.message);});return inputChain;
  }
  function release(){
    held.clear();keyboard.reset();generation++;
    return sendPacket({held:[],move:null,seq:++seq});
  }
  function setPaused(value){
    if(!ready||changing)return Promise.resolve();
    if(value)release();paused=value;displayPause();
    playChain=playChain.catch(()=>{}).then(()=>api('/api/play',{paused:value})).then(r=>{paused=r.paused;displayPause();})
      .catch(e=>{paused=true;displayPause();showError(e.message);});return playChain;
  }
  function send(move=null,cancelAction=false){
    if(!ready||changing)return;
    const packet={held:[...held],move,cancelAction,seq:++seq},g=generation;
    const resume=paused?setPaused(false):playChain;
    resume.then(()=>{if(!paused&&g===generation)sendPacket(packet);});
  }
  function number(v){return Number.isFinite(v)?String(v):'?';}
  function update(s){
    ready=s.ok===true&&!s.switching;paused=s.paused!==false;
    $('play').disabled=!ready||changing||s.fightResult>0;$('reset').disabled=!s.active||changing;$('cancel').disabled=!ready;
    const p=s.score||[],falls=s.falls||[],wins=s.wins||[];
    ['blue','orange'].forEach((name,i)=>{$(name+'-score').textContent=number(p[i]);$(name+'-falls').textContent=number(falls[i]);$(name+'-wins').textContent=number(wins[i]);});
    const seconds=Math.max(0,Math.ceil(s.timeRemaining||0));$('clock').textContent=`${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;
    $('round-status').textContent=globalThis.RekStateLabels.roundStatus(s);
    $('match-status').textContent=s.fightResult>0?(s.fightWinner===0?'You won the match':s.fightWinner===1?'Bot 1 won the match':`Match ended · native result ${s.fightResult}`):'Match in progress';
    const q=s.session||{};$('points').textContent=`${number(q.bluePoints)} : ${number(q.orangePoints)}`;
    $('rounds').textContent=`${number(q.completedRounds)} completed rounds`;$('wld').textContent=`${number(q.blueWins)} / ${number(q.orangeWins)} / ${number(q.draws)}`;
    $('matches').textContent=`${number(q.blueMatchWins)} / ${number(q.orangeMatchWins)}`;
    const last=q.lastRound;$('last-round').textContent=last?`${last.bluePoints} : ${last.orangePoints} · ${last.winner===0?'You won':last.winner===1?'Bot 1 won':last.reason} (${last.reason})`:'No completed round';
    const c=s.commandResults?.[s.active?.humanSide??0];
    if(c?.attempted||c?.cancelled||c?.reason)$('command-status').textContent=`Native action: ${c.accepted?'accepted':c.rejected?'rejected':c.cancelled?'cancelled':'processed'} · ${['none','invalid','recovering','punching','round inactive'][c.reason]??c.reason??''}`;
    const pace=s.pace||{},recent=pace.recent;
    $('pace').textContent=`${Number.isFinite(recent?.realTimeRatio)?`${recent.realTimeRatio.toFixed(2)}× recent (${(recent.activeWallMs/1000).toFixed(1)} s active)`:'Recent pace awaiting play'} · ${Number.isFinite(pace.realTimeRatio)?`${pace.realTimeRatio.toFixed(2)}× session`:'Session pace awaiting play'}`;
    $('timing').textContent=`20 ms control · step ${Number.isFinite(pace.lastStepMs)?pace.lastStepMs.toFixed(1):'?'} ms · frame ${Number.isFinite(pace.lastFrameMs)?pace.lastFrameMs.toFixed(1):'?'} ms · image ${s.frame?`tick ${s.frame.tick}, age ${Math.round(s.frame.ageMs)} ms`:'pending'}`;
    if(s.tick!==lastTick){lastTick=s.tick;lastAdvance=performance.now();}
    $('connection').textContent=!s.ok?(s.failure||'Loading'):paused?'Ready · paused':performance.now()-lastAdvance>3000?'Waiting for simulation':'Running';
    $('runtime-info').textContent=`Native step ${number(s.tick)} · ${globalThis.RekStateLabels.phaseLabel(s.phase)} · ties ${number(s.ties)} · redos ${number(s.redos)} · unclassified ${number(s.unclassified)} · unfinished resets ${number(q.abandonedMatches)}`;
    if(s.failure||s.renderFailure)showError(s.failure||`Renderer: ${s.renderFailure}`);displayPause();
  }
  async function stateLoop(){
    while(!stopped){try{update(await api('/api/state'));}catch(e){$('connection').textContent='Connection unavailable';showError(e.message);}
      await new Promise(r=>setTimeout(r,100));}
  }
  async function frameLoop(){
    while(!stopped){
      if(document.hidden){await new Promise(r=>setTimeout(r,250));continue;}
      const controller=new AbortController();frameRequest=controller;
      try{
        const response=await fetch('/frame.png',{cache:'no-store',
          headers:frameETag?{'If-None-Match':frameETag}:{},
          signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)])});
        if(response.ok){
          const image=await response.blob();
          if(!stopped&&!document.hidden&&!controller.signal.aborted){
            const next=URL.createObjectURL(image),old=blob;blob=next;frameETag=response.headers.get('etag');
            $('frame').src=next;if(old)URL.revokeObjectURL(old);
          }
        }
      }catch{}
      finally{if(frameRequest===controller)frameRequest=null;}
      await new Promise(r=>setTimeout(r,50));
    }
  }
  for(const binding of savedRekBindings){
    const b=document.createElement('button'),key=document.createElement('b'),label=document.createElement('small');
    key.textContent=(binding.doubleTap?'Double ':'')+binding.codes.map(c=>c.replace('Key','').replace('Semicolon',';').replace('Quote',"'")).join('+');
    label.textContent=binding.commandId.replace('move:','').replace('_processed','').replaceAll('_',' ');b.append(key,label);
    b.addEventListener('pointerdown',e=>{e.preventDefault();send(binding.category);});$('moves').append(b);
  }
  $('arena').addEventListener('pointerdown',()=>{$('arena').focus({preventScroll:true});setPaused(false);});
  $('arena').addEventListener('keydown',e=>{
    if(e.code==='Escape'){e.preventDefault();setPaused(true);return;}
    if(!moveKeys.has(e.code)&&!keyboard.recognizes(e.code))return;e.preventDefault();if(e.repeat||!ready||changing)return;
    if(moveKeys.has(e.code))held.add(moveKeys.get(e.code));const move=keyboard.press(e.code,performance.now(),e.repeat);send(move);
  });
  $('arena').addEventListener('keyup',e=>{
    if(!moveKeys.has(e.code)&&!keyboard.recognizes(e.code))return;e.preventDefault();keyboard.release(e.code);
    if(moveKeys.has(e.code)&&held.delete(moveKeys.get(e.code)))send();
  });
  $('arena').addEventListener('blur',()=>{release();displayPause();});
  $('play').addEventListener('click',()=>setPaused(!paused).then(()=>{if(!paused)$('arena').focus({preventScroll:true});}));
  $('cancel').addEventListener('click',()=>send(null,true));
  $('reset').addEventListener('click',async()=>{
    release();await setPaused(true);changing=true;showError('');
    try{await inputChain;await api('/api/reset',{});lastTick=null;paused=true;}catch(e){showError(e.message);}finally{changing=false;displayPause();}
  });
  $('step').addEventListener('click',async()=>{
    if(!paused||changing)return;changing=true;
    try{await api('/api/step',{steps:1,command:{forward:0,strafe:0,yaw:0,moveIndex:-1,cancelAction:false},frame:true});}catch(e){showError(e.message);}finally{changing=false;}
  });
  window.addEventListener('blur',()=>setPaused(true));document.addEventListener('visibilitychange',()=>{if(document.hidden){frameRequest?.abort();setPaused(true);}});
  window.addEventListener('pagehide',()=>{setPaused(true);stopped=true;frameRequest?.abort();if(blob)URL.revokeObjectURL(blob);});
  stateLoop();frameLoop();
})();
