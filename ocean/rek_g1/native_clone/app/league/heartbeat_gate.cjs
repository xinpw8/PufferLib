'use strict';
const {performance}=require('node:perf_hooks');

// Health is a replaceable latest status. Input/state events remain synchronous
// append-only records. No delayed health jobs or status objects accumulate.
function heartbeatGate(write,{periodMs=250,now=()=>performance.now()}={}){
  let last=-Infinity;
  return function heartbeat(force=false){
    const current=now();
    if(!force&&current-last<periodMs)return false;
    write();last=current;return true;
  };
}
module.exports={heartbeatGate};
