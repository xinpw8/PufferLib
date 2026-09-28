'use strict';
const {performance}=require('node:perf_hooks');

// At most one invocation is pending. Late work starts its successor after it
// completes; elapsed timer opportunities do not accumulate a catch-up backlog.
function startPacedLoop(task,{periodMs=20,now=()=>performance.now(),
  setTimer=setTimeout,clearTimer=clearTimeout}={}){
  let stopped=false,timer=null;
  async function run(){
    timer=null;
    if(stopped)return;
    const started=now();
    try{await task();}
    finally{
      if(!stopped)timer=setTimer(run,Math.max(0,periodMs-(now()-started)));
    }
  }
  timer=setTimer(run,periodMs);
  return {stop(){stopped=true;if(timer!==null)clearTimer(timer);timer=null;}};
}
module.exports={startPacedLoop};
