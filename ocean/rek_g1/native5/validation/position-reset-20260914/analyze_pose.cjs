'use strict';
// Read-only postprocessing of the isolated probe's native snapshots.
const fs=require('node:fs'),readline=require('node:readline');
const filename=process.argv[2];if(!filename)throw new Error('Protocol JSONL path required');
const trials=new Map();
function quaternionDegrees(a,b){
  const dot=a.reduce((sum,value,i)=>sum+value*b[i],0);
  const norm=Math.hypot(...a)*Math.hypot(...b);
  return 2*Math.acos(Math.min(1,Math.abs(dot/norm)))*180/Math.PI;
}
(async()=>{
  for await(const line of readline.createInterface({input:fs.createReadStream(filename)})){
    const row=JSON.parse(line);if(!row.trial?.startsWith('far_')||row.direction!=='response'||!row.state)continue;
    const s=row.state,o=s.raw.slice(223),pose={tick:s.tick,busy:!!o[183],route:o[179],z:o[2],q:o.slice(3,7),yaw:2*Math.atan2(o[175],o[172])};
    let trial=trials.get(row.trial);
    if(!trial){trial={name:row.trial,maximum_height_step_m:0,maximum_orientation_step_deg:0,maximum_logical_yaw_step_rad:0,handoffs:[],seen:false};trials.set(row.trial,trial);}
    const previous=trial.previous;trial.previous=pose;
    if(!previous)continue;
    if(pose.busy)trial.seen=true;
    if(!trial.seen)continue;
    const height=Math.abs(pose.z-previous.z),rotation=quaternionDegrees(pose.q,previous.q);
    const yaw=Math.abs(Math.atan2(Math.sin(pose.yaw-previous.yaw),Math.cos(pose.yaw-previous.yaw)));
    trial.maximum_height_step_m=Math.max(trial.maximum_height_step_m,height);
    trial.maximum_orientation_step_deg=Math.max(trial.maximum_orientation_step_deg,rotation);
    trial.maximum_logical_yaw_step_rad=Math.max(trial.maximum_logical_yaw_step_rad,yaw);
    if(pose.busy!==previous.busy)trial.handoffs.push({kind:pose.busy?'entry':'exit',tick:pose.tick,height_step_m:height,
      orientation_step_deg:rotation,previous_root_z_m:previous.z,next_root_z_m:pose.z});
    if(previous.route>=7&&pose.route<7)trial.handoffs.push({kind:'pose_to_idle',tick:pose.tick,height_step_m:height,
      orientation_step_deg:rotation,previous_root_z_m:previous.z,next_root_z_m:pose.z});
  }
  const result=[...trials.values()].map(({previous,seen,...trial})=>trial);
  console.log(JSON.stringify({trials:result,maximum_height_step_m:Math.max(...result.map(t=>t.maximum_height_step_m)),
    maximum_orientation_step_deg:Math.max(...result.map(t=>t.maximum_orientation_step_deg)),
    maximum_exit_height_step_m:Math.max(...result.flatMap(t=>t.handoffs.filter(h=>h.kind==='exit').map(h=>h.height_step_m))),
    maximum_exit_orientation_step_deg:Math.max(...result.flatMap(t=>t.handoffs.filter(h=>h.kind==='exit').map(h=>h.orientation_step_deg))),
    maximum_pose_to_idle_height_step_m:Math.max(...result.flatMap(t=>t.handoffs.filter(h=>h.kind==='pose_to_idle').map(h=>h.height_step_m))),
    maximum_pose_to_idle_orientation_step_deg:Math.max(...result.flatMap(t=>t.handoffs.filter(h=>h.kind==='pose_to_idle').map(h=>h.orientation_step_deg)))},null,2));
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
