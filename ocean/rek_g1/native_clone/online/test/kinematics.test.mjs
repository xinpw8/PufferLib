// Browser kinematics against MuJoCo mj_kinematics for recorded poses.
// Run: node --test test/kinematics.test.mjs SCENE_DIR QPOS.f32 (SCENE_DIR has check.json from export_scene)
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {forwardKinematics,geomFrame,interpolateQpos} from '../public/scene.js';

const [sceneDir,qposFile]=process.argv.slice(2).length?process.argv.slice(2):[process.env.REK_SCENE_DIR,process.env.REK_CHECK_QPOS];
const scene=JSON.parse(fs.readFileSync(sceneDir+'/scene.json'));
const reference=JSON.parse(fs.readFileSync(sceneDir+'/check.json'));
const raw=fs.readFileSync(qposFile),rows=[];
for(let r=0;r<raw.length/288;r++)rows.push(Array.from({length:72},(_,i)=>raw.readFloatLE(r*288+4*i)));
const matrix=([w,x,y,z])=>[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)];

test('visible geom frames match MuJoCo for every recorded pose',()=>{
  let worstPos=0,worstRot=0;
  rows.forEach((qpos,r)=>{
    const frames=forwardKinematics(scene,qpos);
    for(const geom of scene.geoms){
      const {pos,quat}=geomFrame(frames,geom),m=matrix(quat);
      for(let k=0;k<3;k++)worstPos=Math.max(worstPos,Math.abs(pos[k]-reference[r].xpos[3*geom.id+k]));
      for(let k=0;k<9;k++)worstRot=Math.max(worstRot,Math.abs(m[k]-reference[r].xmat[9*geom.id+k]));
    }
  });
  console.log(JSON.stringify({poses:rows.length,geoms:scene.geoms.length,worstPositionM:worstPos,worstMatrixEntry:worstRot}));
  assert.ok(worstPos<1e-5,`position error ${worstPos}`);assert.ok(worstRot<1e-5,`rotation error ${worstRot}`);
});
test('interpolation endpoints reproduce both snapshots',()=>{
  const a=rows[3],b=rows[4],start=interpolateQpos(scene,a,b,0),end=interpolateQpos(scene,a,b,1);
  for(let i=0;i<72;i++){assert.ok(Math.abs(start[i]-a[i])<1e-6);assert.ok(Math.abs(Math.abs(end[i])-Math.abs(b[i]))<1e-6);}
});
