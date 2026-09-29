'use strict';
// Binary tick message, little endian. public/client.js decodes the same layout.
//  0 u8  type (1)          1 u8  mode (0 idle, 1 bot, 2 pvp)   2 u8 phase   3 u8 flags
//  4 u32 tick              8 f64 server time (ms)
// 16 f32 time remaining   20 u8  round   21 u8 fight result   22 i8 fight winner   23 i8 round winner
// 24 u16 points[2]        28 u8  falls[2]  30 u8 wins[2]
// 32 u8  command flags[2] 34 u8  rejection reason[2]
// 36 f32 hold seconds remaining (countdown, intermission or match end)
// 40 f32 qpos[72]
const TICK=1,SIZE=40+72*4;
const MODES=Object.freeze({idle:0,bot:1,pvp:2});
const FLAG_TERMINAL=1,FLAG_HOLD=2;
const u8=v=>Math.max(0,Math.min(255,Math.round(Number(v)||0)));
const i8=v=>Number.isFinite(v)?Math.max(-1,Math.min(127,Math.round(v))):-1;

// Per side: bit0 attempted, bit1 accepted, bit2 rejected, bit3 cancelled.
function commandFlags(events,side){
  let flags=0,reason=0;
  for(const e of events||[])if(e.side===side){
    flags|=(e.attempted?1:0)|(e.accepted?2:0)|(e.rejected?4:0)|(e.cancelled?8:0);
    if(e.reason)reason=e.reason;
  }
  return [flags,reason];
}
function encodeTick({mode,state,events,serverTime,holdSeconds=0}){
  if(!Array.isArray(state.qpos)||state.qpos.length!==72)throw Error('72 qpos values required');
  const buffer=Buffer.alloc(SIZE),v=new DataView(buffer.buffer,buffer.byteOffset,SIZE);
  v.setUint8(0,TICK);v.setUint8(1,MODES[mode]??0);v.setUint8(2,u8(state.phase));
  v.setUint8(3,(state.terminal?FLAG_TERMINAL:0)|(holdSeconds>0?FLAG_HOLD:0));
  v.setUint32(4,Math.max(0,state.tick|0),true);v.setFloat64(8,serverTime,true);
  v.setFloat32(16,Number(state.timeRemaining)||0,true);v.setUint8(20,u8(state.roundNumber));
  v.setUint8(21,u8(state.fightResult));v.setInt8(22,i8(state.fightWinner));v.setInt8(23,i8(state.winner));
  for(let side=0;side<2;side++){
    v.setUint16(24+2*side,Math.max(0,Math.min(65535,Math.round(Number(state.score?.[side])||0))),true);
    v.setUint8(28+side,u8(state.falls?.[side]));v.setUint8(30+side,u8(state.wins?.[side]));
    const [flags,reason]=commandFlags(events,side);v.setUint8(32+side,flags);v.setUint8(34+side,u8(reason));
  }
  v.setFloat32(36,holdSeconds,true);
  for(let i=0;i<72;i++)v.setFloat32(40+4*i,state.qpos[i],true);
  return buffer;
}
function decodeTick(buffer){
  const v=new DataView(buffer.buffer,buffer.byteOffset,buffer.byteLength);
  if(buffer.byteLength!==SIZE||v.getUint8(0)!==TICK)throw Error('Not a tick message');
  const qpos=new Float32Array(72);for(let i=0;i<72;i++)qpos[i]=v.getFloat32(40+4*i,true);
  return {mode:Object.keys(MODES)[v.getUint8(1)],phase:v.getUint8(2),flags:v.getUint8(3),tick:v.getUint32(4,true),
    serverTime:v.getFloat64(8,true),timeRemaining:v.getFloat32(16,true),round:v.getUint8(20),
    fightResult:v.getUint8(21),fightWinner:v.getInt8(22),roundWinner:v.getInt8(23),
    points:[v.getUint16(24,true),v.getUint16(26,true)],falls:[v.getUint8(28),v.getUint8(29)],
    wins:[v.getUint8(30),v.getUint8(31)],commands:[v.getUint8(32),v.getUint8(33)],reasons:[v.getUint8(34),v.getUint8(35)],
    hold:v.getFloat32(36,true),qpos};
}
module.exports={TICK,SIZE,MODES,encodeTick,decodeTick,commandFlags};
