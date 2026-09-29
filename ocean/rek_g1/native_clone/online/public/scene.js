// Scene math shared by the browser client and its Node tests. No DOM or three.js.
// Quaternions are MuJoCo order [w, x, y, z]; the world is z-up.

export function quatMul(a,b){
  return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
}
export function quatRotate(q,v){
  const [w,x,y,z]=q,tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);
  return [v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx];
}
export function quatNormalize(q){
  const n=Math.hypot(q[0],q[1],q[2],q[3]);return n>0?[q[0]/n,q[1]/n,q[2]/n,q[3]/n]:[1,0,0,0];
}
function axisAngle(axis,angle){const s=Math.sin(angle/2);return [Math.cos(angle/2),axis[0]*s,axis[1]*s,axis[2]*s];}

// mj_kinematics for free, ball, slide and hinge joints. Returns body frames.
export function forwardKinematics(scene,qpos){
  const n=scene.bodies.length,xpos=new Array(n),xquat=new Array(n);
  xpos[0]=[0,0,0];xquat[0]=[1,0,0,0];
  for(let b=1;b<n;b++){
    const body=scene.bodies[b],p=body.parent,off=quatRotate(xquat[p],body.pos);
    let pos=[xpos[p][0]+off[0],xpos[p][1]+off[1],xpos[p][2]+off[2]],quat=quatMul(xquat[p],body.quat);
    for(let j=body.jntadr;j<body.jntadr+body.jntnum;j++){
      const joint=scene.joints[j],a=joint.qposadr;
      if(joint.type===0){pos=[qpos[a],qpos[a+1],qpos[a+2]];quat=quatNormalize([qpos[a+3],qpos[a+4],qpos[a+5],qpos[a+6]]);continue;}
      const r=quatRotate(quat,joint.pos),anchor=[pos[0]+r[0],pos[1]+r[1],pos[2]+r[2]];
      if(joint.type===2){
        const axis=quatRotate(quat,joint.axis),d=qpos[a]-scene.qpos0[a];
        pos=[pos[0]+axis[0]*d,pos[1]+axis[1]*d,pos[2]+axis[2]*d];continue;
      }
      quat=quatMul(quat,joint.type===1?quatNormalize([qpos[a],qpos[a+1],qpos[a+2],qpos[a+3]])
        :axisAngle(joint.axis,qpos[a]-scene.qpos0[a]));
      const v=quatRotate(quat,joint.pos);pos=[anchor[0]-v[0],anchor[1]-v[1],anchor[2]-v[2]];
    }
    xpos[b]=pos;xquat[b]=quatNormalize(quat);
  }
  return {xpos,xquat};
}
export function geomFrame(frames,geom){
  const q=frames.xquat[geom.body],o=quatRotate(q,geom.pos),p=frames.xpos[geom.body];
  return {pos:[p[0]+o[0],p[1]+o[1],p[2]+o[2]],quat:quatMul(q,geom.quat)};
}

// Snapshot interpolation: positions and hinges linear, free-joint rotations slerped.
function slerp(a,b,t){
  let dot=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3],s=1;
  if(dot<0){dot=-dot;s=-1;}
  if(dot>.9995)return quatNormalize(a.map((v,i)=>v+(s*b[i]-v)*t));
  const theta=Math.acos(dot),k0=Math.sin((1-t)*theta)/Math.sin(theta),k1=s*Math.sin(t*theta)/Math.sin(theta);
  return a.map((v,i)=>k0*v+k1*b[i]);
}
export function interpolateQpos(scene,a,b,t){
  const out=new Float64Array(a.length);
  for(let i=0;i<a.length;i++)out[i]=a[i]+(b[i]-a[i])*t;
  for(const joint of scene.joints){
    const k=joint.type===0?joint.qposadr+3:joint.type===1?joint.qposadr:-1;if(k<0)continue;
    const q=slerp([a[k],a[k+1],a[k+2],a[k+3]],[b[k],b[k+1],b[k+2],b[k+3]],t);
    for(let i=0;i<4;i++)out[k+i]=q[i];
  }
  return out;
}

// Third-person view behind one fighter, as in the native renderer: 3 m behind the
// pelvis, 1 m above, 10 degrees down, heading held inside a 20-degree band and
// recentred with a 1 s time constant of simulated time.
export const FOLLOW={behind:3,height:1,pitchDeg:10,deadbandDeg:20,recentreSeconds:1,minHeadingNorm:.2};
export function fighterHeading(qpos,side){
  const k=36*side+3,w=qpos[k],x=qpos[k+1],y=qpos[k+2],z=qpos[k+3];
  const numerator=2*(y*x+z*w),denominator=1-2*(z*z+y*y);
  return Math.hypot(numerator,denominator)<FOLLOW.minHeadingNorm?null:Math.atan2(numerator,denominator);
}
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
export class FollowHeading{
  constructor(){this.valid=false;}
  // `seconds` is simulated time; a jump backwards or a new side snaps the view.
  update(qpos,side,seconds){
    let target=fighterHeading(qpos,side);
    const continuous=this.valid&&side===this.side&&seconds>=this.seconds;
    if(!continuous){
      if(target===null){const dx=qpos[36*(1-side)]-qpos[36*side],dy=qpos[36*(1-side)+1]-qpos[36*side+1];target=dx*dx+dy*dy>0?Math.atan2(dy,dx):0;}
      this.heading=target;
    }else if(target!==null){
      const band=FOLLOW.deadbandDeg*Math.PI/180;let error=wrap(target-this.heading);
      if(Math.abs(error)>band){this.heading+=error-Math.sign(error)*band;error=Math.sign(error)*band;}
      this.heading=wrap(this.heading+error*(1-Math.exp(-(seconds-this.seconds)/FOLLOW.recentreSeconds)));
    }
    this.valid=true;this.side=side;this.seconds=seconds;return this.heading;
  }
}
export function followCamera(qpos,side,heading){
  const p=[qpos[36*side],qpos[36*side+1],qpos[36*side+2]],pitch=FOLLOW.pitchDeg*Math.PI/180;
  return {eye:[p[0]-FOLLOW.behind*Math.cos(heading),p[1]-FOLLOW.behind*Math.sin(heading),p[2]+FOLLOW.height],
    target:[p[0],p[1],p[2]+FOLLOW.height-FOLLOW.behind*Math.tan(pitch)]};
}
// The native renderer's two-fighter overview (azimuth 130, elevation -60).
export function overviewCamera(qpos){
  const dx=qpos[0]-qpos[36],dy=qpos[1]-qpos[37],distance=Math.max(3.8,2.5+1.2*Math.hypot(dx,dy));
  const target=[(qpos[0]+qpos[36])/2,(qpos[1]+qpos[37])/2,Math.max(.9,(qpos[2]+qpos[38])/2)];
  const az=130*Math.PI/180,el=-60*Math.PI/180,f=[Math.cos(el)*Math.cos(az),Math.cos(el)*Math.sin(az),Math.sin(el)];
  return {eye:[target[0]-distance*f[0],target[1]-distance*f[1],target[2]-distance*f[2]],target};
}

// Static arena meshes whose wall plane the eye has crossed are hidden.
export const CUTAWAY={marginM:.4,minRadiusM:1};
export function cutawayWalls(scene,positions){
  const walls=[];
  for(const geom of scene.geoms){
    if(geom.body!==0||geom.mesh<0)continue;
    const mesh=scene.meshes[geom.mesh];let sx=0,sy=0;
    for(let i=0;i<mesh.vertices;i++){
      const v=quatRotate(geom.quat,[positions[mesh.offset+3*i],positions[mesh.offset+3*i+1],positions[mesh.offset+3*i+2]]);
      sx+=v[0];sy+=v[1];
    }
    const cx=geom.pos[0]+sx/mesh.vertices,cy=geom.pos[1]+sy/mesh.vertices,radius=Math.hypot(cx,cy);
    if(radius>=CUTAWAY.minRadiusM)walls.push({id:geom.id,ux:cx/radius,uy:cy/radius,radius});
  }
  return walls;
}
export function hiddenWalls(walls,eye){
  return new Set(walls.filter(w=>eye[0]*w.ux+eye[1]*w.uy>w.radius-CUTAWAY.marginM).map(w=>w.id));
}

// Maps server time to local time from the fastest recent packet and picks a
// playback delay just above the recent arrival jitter.
export class Playout{
  constructor({window=150,minDelay=25,maxDelay=150}={}){this.window=window;this.minDelay=minDelay;this.maxDelay=maxDelay;this.samples=[];this.snapshots=[];}
  push(snapshot,arrival){
    this.samples.push(arrival-snapshot.serverTime);if(this.samples.length>this.window)this.samples.shift();
    const last=this.snapshots.at(-1);
    // A reset (new match) starts a fresh buffer instead of blending across the teleport.
    if(last&&snapshot.tick<last.tick)this.snapshots=[];
    else if(last&&snapshot.serverTime<=last.serverTime)return;
    this.snapshots.push(snapshot);if(this.snapshots.length>64)this.snapshots.shift();
  }
  offset(){return Math.min(...this.samples);}
  delay(){
    const offset=this.offset(),jitter=this.samples.map(s=>s-offset).sort((a,b)=>a-b);
    const p95=jitter[Math.floor(.95*(jitter.length-1))]||0;
    return Math.min(this.maxDelay,Math.max(this.minDelay,p95+8));
  }
  // Returns [a, b, t] bracketing server time `at`, or the newest snapshot alone.
  sample(now){
    const s=this.snapshots;if(!s.length)return null;
    const at=now-this.offset()-this.delay();
    if(at>=s.at(-1).serverTime||s.length===1)return [s.at(-1),s.at(-1),0];
    if(at<=s[0].serverTime)return [s[0],s[0],0];
    let i=s.length-1;while(i>0&&s[i-1].serverTime>at)i--;
    const a=s[i-1],b=s[i];return [a,b,(at-a.serverTime)/(b.serverTime-a.serverTime)];
  }
}
