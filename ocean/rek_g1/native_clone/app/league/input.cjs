'use strict';
const KEYS=new Set(['W','S','A','D','Q','E']);
const neutral=()=>({forward:0,strafe:0,yaw:0,moveIndex:-1,cancelAction:false});
function commandFor(held){
  const k=new Set(held);
  return {forward:Number(k.has('W'))-Number(k.has('S')),
    strafe:Number(k.has('A'))-Number(k.has('D')),
    yaw:Number(k.has('Q'))-Number(k.has('E')),moveIndex:-1,cancelAction:false};
}
function validateCommand(c){
  if(!c||typeof c!=='object'||Array.isArray(c))throw Error('Command object required');
  for(const key of ['forward','strafe','yaw'])if(!Number.isFinite(c[key])||Math.abs(c[key])>1)throw Error(`${key} must be in [-1,1]`);
  if(!Number.isInteger(c.moveIndex)||c.moveIndex< -1||c.moveIndex>16)throw Error('moveIndex must be -1 through16');
  if(typeof c.cancelAction!=='boolean')throw Error('cancelAction must be boolean');
  return {forward:c.forward,strafe:c.strafe,yaw:c.yaw,moveIndex:c.moveIndex,cancelAction:c.cancelAction};
}
class HumanInput{
  constructor(){this.reset();}
  reset(){this.held=[];this.edges=[];this.seq=-1;this.disposition='none';this.cancel=false;}
  update(value){
    if(!Number.isSafeInteger(value.seq)||value.seq<=this.seq)return false;
    if(!Array.isArray(value.held)||value.held.some(k=>!KEYS.has(k)))throw Error('Invalid held key');
    if(value.move!=null&&(!Number.isInteger(value.move)||value.move<16||value.move>32))throw Error('Move category must be16 through32');
    if(value.cancelAction!=null&&typeof value.cancelAction!=='boolean')throw Error('cancelAction must be boolean');
    // Each accepted edge reaches the native scheduler exactly once. No JS
    // physics-state, translation, action-mask or busy-move suppression.
    if(value.move!=null&&this.edges.length>=32)throw Error('Input edge queue full');
    this.seq=value.seq;this.held=[...new Set(value.held)].sort();
    if(value.move!=null)this.edges.push(value.move-16);
    this.cancel=this.cancel||value.cancelAction===true;
    this.disposition='submitted';return true;
  }
  release(){this.held=[];this.edges=[];this.cancel=false;}
  next(_state,side=0){
    if(side!==0&&side!==1)throw Error('Invalid human side');
    const c=commandFor(this.held);c.moveIndex=this.edges.shift()??-1;c.cancelAction=this.cancel;this.cancel=false;
    return c;
  }
}
module.exports={HumanInput,commandFor,validateCommand,neutral};
