import test from 'node:test';
import assert from 'node:assert/strict';
import {ChaseCamera,CHASE} from '../public/scene.js';

// Fighter 0 at (x, y) facing `yaw`; fighter 1 two metres ahead of it.
function pose(x,y,yaw=0,z=.8){
  const q=new Float64Array(72);
  q[0]=x;q[1]=y;q[2]=z;q[3]=Math.cos(yaw/2);q[6]=Math.sin(yaw/2);
  q[36]=x+2;q[37]=y;q[38]=.8;q[39]=0;q[42]=1;return q;
}
const dt=1/60;
for(const lockOn of [true,false]){
  test(`${lockOn?'lock-on':'follow'}: a strike lunge inside the dead zone does not move the eye`,()=>{
    const cam=new ChaseCamera(),start=cam.update(pose(0,0),0,dt,{lockOn,snap:true});
    let eye=start.eye;
    for(let i=0;i<30;i++)eye=cam.update(pose(.35*Math.sin(Math.PI*i/30),0),0,dt,{lockOn}).eye;
    assert.ok(Math.hypot(eye[0]-start.eye[0],eye[1]-start.eye[1])<1e-9);
  });
  test(`${lockOn?'lock-on':'follow'}: height is fixed while the fighter bobs or falls`,()=>{
    const cam=new ChaseCamera();cam.update(pose(0,0),0,dt,{lockOn,snap:true});
    for(const z of [.8,.7,.3,.9])assert.equal(cam.update(pose(0,0,0,z),0,dt,{lockOn}).eye[2],CHASE.eyeZ);
  });
}
test('follow: turning never exceeds the capped rate and small yaw swings are ignored',()=>{
  const cam=new ChaseCamera();let prev=cam.update(pose(0,0,0),0,dt,{snap:true});
  const heading=v=>Math.atan2(v.target[1]-v.eye[1],v.target[0]-v.eye[0]);
  // ±11.5 degree gait/strike swings inside the band: only the slow recentre (0.24 degrees here).
  for(let i=0;i<60;i++){const v=cam.update(pose(0,0,.2*Math.sin(i/5)),0,dt);assert.ok(Math.abs(heading(v))*180/Math.PI<.5);}
  let max=0;prev=cam.update(pose(0,0,0),0,dt);
  for(let i=0;i<720;i++){
    const v=cam.update(pose(0,0,Math.PI/2),0,dt);max=Math.max(max,Math.abs(heading(v)-heading(prev))/dt);prev=v;
    // Within 4 s the view is inside the band; the rest is the slow recentre.
    if(i===240)assert.ok(Math.PI/2-heading(v)<=CHASE.bandDeg*Math.PI/180+.01,'inside the band after 4 s');
  }
  assert.ok(max*180/Math.PI<=CHASE.maxTurnDegPerSecond+1e-6,`turn rate ${max*180/Math.PI}`);
  assert.ok(Math.abs(heading(prev)-Math.PI/2)<3*Math.PI/180,'settles behind the new facing within 12 s');
});
test('a round-reset teleport cuts instead of flying across the arena',()=>{
  const cam=new ChaseCamera();cam.update(pose(1.5,1.5),0,dt,{lockOn:true,snap:true});
  const v=cam.update(pose(-.9,0),0,dt,{lockOn:true});
  assert.ok(Math.abs(v.target[0]+.9)<1e-9&&Math.abs(v.target[1])<1e-9);
});
test('lock-on cuts to the reverse angle when the fighters pass each other',()=>{
  const cam=new ChaseCamera(),heading=v=>Math.atan2(v.target[1]-v.eye[1],v.target[0]-v.eye[0]);
  cam.update(pose(0,0),0,dt,{lockOn:true,snap:true});
  // Opponent now behind fighter 0 (it walked past).
  const passed=pose(0,0);passed[36]=-2;let v;
  for(let i=0;i<Math.ceil((CHASE.cutAfterSeconds+.1)/dt);i++)v=cam.update(passed,0,dt,{lockOn:true});
  assert.ok(Math.abs(Math.abs(heading(v))-Math.PI)<CHASE.bandDeg*Math.PI/180+.01,`heading ${heading(v)}`);
});
