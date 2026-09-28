'use strict';
(function(root){
  const phases=Object.freeze(['Idle','Countdown','Fighting','Between rounds','Match complete']);
  const phaseLabel=phase=>phases[phase]??'Phase unknown';
  function roundStatus(state){
    return `Round ${Number.isFinite(state.roundNumber)?state.roundNumber:'?'} · ${phaseLabel(state.phase)}${state.paused&&state.phase!==4?' · paused':''}`;
  }
  const api={phaseLabel,roundStatus};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.RekStateLabels=api;
})(globalThis);
