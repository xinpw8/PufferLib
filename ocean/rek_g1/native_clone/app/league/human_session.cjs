'use strict';
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');

const HUMAN_ROUND_SECONDS=Object.freeze([20,120,300]);
const DEFAULT_HUMAN_ROUND_SECONDS=300;

function createHumanConfig(workerConfigPath,roundSeconds=DEFAULT_HUMAN_ROUND_SECONDS){
  if(!Number.isInteger(roundSeconds)||!HUMAN_ROUND_SECONDS.includes(roundSeconds))
    throw new Error('Human round duration must be 20, 120 or 300 seconds');
  const original=JSON.parse(fs.readFileSync(workerConfigPath,'utf8'));
  if(!original||typeof original!=='object'||Array.isArray(original))
    throw new Error('Native worker configuration must be an object');
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-human-'));
  const filename=path.join(directory,'worker.json');
  let closed=false;
  function close(){
    if(closed)return;
    try{fs.unlinkSync(filename);}catch(error){if(error.code!=='ENOENT')throw error;}
    // Deliberately non-recursive: an unexpected file must never be deleted.
    try{fs.rmdirSync(directory);}catch(error){if(error.code!=='ENOENT')throw error;}
    closed=true;
  }
  try{
    fs.writeFileSync(filename,JSON.stringify({...original,round_seconds:roundSeconds},null,2)+'\n',
      {flag:'wx',mode:0o600});
  }catch(error){close();throw error;}
  return {path:filename,roundSeconds,trainingRoundSeconds:Number(original.round_seconds)||120,close};
}

class HumanSession {
  constructor(){this.reset();}
  reset(){
    this.completedRounds=0;this.bluePoints=0;this.orangePoints=0;
    this.blueWins=0;this.orangeWins=0;this.draws=0;this.invalidRounds=0;
    this.lastRound=null;this.newRoundStream();
    this.completedMatches=0;this.blueMatchWins=0;this.orangeMatchWins=0;this.abandonedMatches=0;this.lastMatch=null;
  }
  newRoundStream(){this.seen=new Set();this.fightSeen=false;}
  recordFight(state){
    if(this.fightSeen||![1,2].includes(state.fightResult))return false;
    if(state.fightWinner!==0&&state.fightWinner!==1)throw Error('Native fight result requires winner0 or1');
    this.fightSeen=true;this.completedMatches++;
    if(state.fightWinner===0)this.blueMatchWins++;else this.orangeMatchWins++;
    this.lastMatch={winner:state.fightWinner,reason:state.fightResult===1?'rounds':'TKO'};return true;
  }
  abandonFight(state){
    if(this.fightSeen||state.fightResult>0||!Number.isFinite(state.tick)||state.tick<=0)return false;
    this.fightSeen=true;this.abandonedMatches++;return true;
  }
  record(nativeTerminal){
    if(!nativeTerminal||typeof nativeTerminal!=='object'||Array.isArray(nativeTerminal))
      throw new Error('Native round state must be an object');
    const {terminal,roundNumber,tick,score,roundResult,winner}=nativeTerminal;
    if(terminal===0||terminal===false)return false;
    if(terminal!==1&&terminal!==true)throw new Error('Native terminal must be 0 or 1');
    if(!Number.isSafeInteger(roundNumber)||roundNumber<1||!Number.isSafeInteger(tick)||tick<0)
      throw new Error('Native terminal requires an integer round number and tick');
    if(!Array.isArray(score)||score.length!==2||score.some(value=>!Number.isSafeInteger(value)||value<0))
      throw new Error('Native terminal requires two nonnegative integer scores');
    const key=`${roundNumber}:${tick}`;
    if(this.seen.has(key))return false;
    let reason='unknown',recordedWinner=null,valid=false;
    if((roundResult===1||roundResult===2)&&(winner===0||winner===1)){
      reason=roundResult===1?'points':'knockout';recordedWinner=winner;valid=true;
    }else if(roundResult===3){reason='draw';valid=true;}
    else if(roundResult===4)reason='replay';
    this.seen.add(key);
    this.lastRound={bluePoints:score[0],orangePoints:score[1],winner:recordedWinner,reason};
    if(valid){
      this.completedRounds++;this.bluePoints+=score[0];this.orangePoints+=score[1];
      if(recordedWinner===0)this.blueWins++;
      else if(recordedWinner===1)this.orangeWins++;
      else this.draws++;
    }else this.invalidRounds++;
    return true;
  }
  snapshot(){
    return {completedRounds:this.completedRounds,bluePoints:this.bluePoints,orangePoints:this.orangePoints,
      blueWins:this.blueWins,orangeWins:this.orangeWins,draws:this.draws,invalidRounds:this.invalidRounds,
      completedMatches:this.completedMatches,blueMatchWins:this.blueMatchWins,orangeMatchWins:this.orangeMatchWins,
      abandonedMatches:this.abandonedMatches,lastMatch:this.lastMatch?{...this.lastMatch}:null,
      lastRound:this.lastRound?{...this.lastRound}:null};
  }
}

module.exports={HUMAN_ROUND_SECONDS,DEFAULT_HUMAN_ROUND_SECONDS,createHumanConfig,HumanSession};
