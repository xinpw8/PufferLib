'use strict';
const {performance}=require('node:perf_hooks');

// One invocation and one scheduled callback maximum. Deadlines advance by the
// fixed period, independently of callback lateness. Large wall-clock debt is
// explicitly rebased; no physics step is fabricated, batched, or skipped here.
function startPacedLoop(task,{periodMs=20,now=()=>performance.now(),
  setTimer=setTimeout,clearTimer=clearTimeout,setSoon=setImmediate,clearSoon=clearImmediate,
  startPaused=false,maxDebtMs=100}={}){
  if(!(Number.isFinite(periodMs)&&periodMs>0&&Number.isFinite(maxDebtMs)&&maxDebtMs>=periodMs))
    throw Error('Invalid pacing period or debt limit');
  let stopped=false,active=!startPaused,timer=null,immediate=false,running=false,epoch=0;
  let deadline=now()+periodMs;
  const stats={invocations:0,completed:0,lateStarts:0,totalLatenessMs:0,maxLatenessMs:0,
    totalTaskMs:0,maxTaskMs:0,rebases:0,discardedWallDebtMs:0};
  function cancel(){if(timer!==null)(immediate?clearSoon:clearTimer)(timer);timer=null;}
  function schedule(){
    if(stopped||!active||running||timer!==null)return;
    const remaining=deadline-now();immediate=remaining<=0;
    timer=immediate?setSoon(run):setTimer(run,Math.ceil(remaining));
  }
  async function run(){
    timer=null;
    if(stopped||!active)return;
    // Timer precision is not an authorization to advance simulated time early.
    if(now()<deadline){schedule();return;}
    running=true;const generation=epoch,started=now(),late=Math.max(0,started-deadline);
    stats.invocations++;if(late>0)stats.lateStarts++;
    stats.totalLatenessMs+=late;stats.maxLatenessMs=Math.max(stats.maxLatenessMs,late);
    try{await task();}
    finally{
      const finished=now(),cost=finished-started;running=false;stats.completed++;
      stats.totalTaskMs+=cost;stats.maxTaskMs=Math.max(stats.maxTaskMs,cost);
      if(!stopped&&active&&generation===epoch){
        deadline+=periodMs;
        const debt=finished-deadline;
        if(debt>maxDebtMs){stats.rebases++;stats.discardedWallDebtMs+=debt;deadline=finished;}
      }
      schedule();
    }
  }
  schedule();
  return {
    pause(){if(active){active=false;epoch++;}cancel();},
    resume(){if(stopped||active)return;active=true;epoch++;deadline=now()+periodMs;schedule();},
    stop(){stopped=true;active=false;epoch++;cancel();},
    snapshot(){return {...stats,periodMs,maxDebtMs,active,running,
      meanTaskMs:stats.completed?stats.totalTaskMs/stats.completed:null,
      meanLatenessMs:stats.invocations?stats.totalLatenessMs/stats.invocations:null};},
  };
}
module.exports={startPacedLoop};
