'use strict';

const HELD = new Map([
  ['',1],['W',2],['S',3],['A',4],['D',5],['Q',6],['E',7],
  ['QW',8],['EW',9],['QS',10],['ES',11],['AQ',12],['AE',13],['DQ',14],['DE',15],
]);
function resolveHeld(keys){
  const held=new Set(keys);
  // Opposing keys cancel independently. The 33-action runtime has no diagonal
  // translation category: prefer forward/back over strafe when both remain.
  // This is a deterministic human-adapter convention, not measured REK input
  // behavior. Yaw remains independent and combines with the chosen translation.
  for(const [a,b] of [['W','S'],['A','D'],['Q','E']]){
    if(held.has(a)&&held.has(b)){held.delete(a);held.delete(b);}
  }
  if(held.has('W')||held.has('S')){held.delete('A');held.delete('D');}
  return [...held].sort();
}
class HumanInput {
  constructor(){ this.reset(); }
  reset(){this.held=[];this.edge=null;this.pending=null;this.seq=-1;this.disposition='none';}
  update(value){
    if(!Number.isSafeInteger(value.seq)||value.seq<=this.seq)return false;
    if(!Array.isArray(value.held)||value.held.some(k=>!['W','S','A','D','Q','E'].includes(k)))
      throw new Error('Held input must contain W/S/A/D/Q/E only');
    const held=resolveHeld(value.held);
    if(value.move!=null&&(!Number.isInteger(value.move)||value.move<16||value.move>32))
      throw new Error('Move must be an attack category 16 through 32');
    this.held=held;this.seq=value.seq;
    if(value.move!=null){
      if(this.edge||this.pending)this.disposition='discarded_additional_move';
      else this.edge={category:value.move,held:[...held]};
    }
    return true;
  }
  release(){this.held=[];this.edge=null;this.pending=null;}
  next(state,side=0){
    if(side!==0&&side!==1)throw new Error('Invalid human side');
    if(!state.mask||state.mask.length!==66||!state.raw||state.raw.length!==446)throw new Error('Native state not ready');
    const mask=state.mask.slice(side*33,(side+1)*33),raw=state.raw.slice(side*223,(side+1)*223);
    const held=HELD.get(this.held.join(''));
    const edge=this.edge;this.edge=null;
    let move=this.pending||edge;
    const translation=keys=>keys.some(k=>['W','S','A','D'].includes(k));
    let preferred=held,fallback=held;
    if(move){
      if(raw[182]!==0||translation(this.held)||translation(move.held)||raw[79]!==0){
        this.pending=null;move=null;this.disposition='discarded_busy_move';
      }else{preferred=move.category;fallback=1;}
    }
    const action=[preferred,fallback,0,1].find(k=>mask[k]===1);
    if(action===undefined)throw new Error('No legal native action');
    if(move){
      if(action===move.category){this.pending=null;this.disposition='accepted';}
      else if(this.pending||move.held.some(k=>k==='Q'||k==='E')){
        this.pending=move;this.disposition='buffered_yaw_interruption';
      }else{this.pending=null;this.disposition='discarded_masked_move';}
    }
    return action;
  }
}
module.exports={HumanInput};
