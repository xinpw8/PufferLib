// Browser client: renders the server's simulation locally and sends key changes.
import * as THREE from './vendor/three.module.min.js';
import {forwardKinematics,interpolateQpos,ChaseCamera,OverviewCamera,cutawayWalls,hiddenWalls,Playout} from './scene.js';

const $=id=>document.getElementById(id);
const MODES=['idle','bot','pvp'],REASONS=['','invalid','recovering','busy punching','round not active'];
const PHASES=['Idle','Countdown','Fighting','Between rounds','Match complete'];

// ---- Rendering -------------------------------------------------------------
THREE.ColorManagement.enabled=false; // MuJoCo shades raw texture and colour values.
const canvas=$('view');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true,powerPreference:'high-performance'});
renderer.outputColorSpace=THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
const world=new THREE.Scene();world.background=new THREE.Color(0,0,0);
const camera=new THREE.PerspectiveCamera(45,1,.03,120);camera.up.set(0,0,1);
const headlight=new THREE.DirectionalLight(0xffffff,1);world.add(headlight,headlight.target);
function resize(){renderer.setSize(innerWidth,innerHeight,false);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();}
addEventListener('resize',resize);resize();

let scene=null,bodies=[],geomObjects=new Map(),walls=[];
async function loadScene(){
  const [description,bin]=await Promise.all([
    fetch('/scene/scene.json').then(r=>{if(r.status===401)throw Error('login');return r.json();}),
    fetch('/scene/scene.bin').then(r=>r.arrayBuffer())]);
  scene=description;const floats=new Float32Array(bin),loader=new THREE.TextureLoader();
  const textures=await Promise.all(scene.textures.map(t=>loader.loadAsync('/scene/'+t.file).then(texture=>{
    texture.flipY=false;texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.colorSpace=THREE.NoColorSpace;
    texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());texture.needsUpdate=true;return texture;})));
  const geometries=scene.meshes.map(m=>{
    const g=new THREE.BufferGeometry(),n=m.vertices;
    g.setAttribute('position',new THREE.BufferAttribute(floats.subarray(m.offset,m.offset+3*n),3));
    g.setAttribute('normal',new THREE.BufferAttribute(floats.subarray(m.offset+3*n,m.offset+6*n),3));
    if(m.uv)g.setAttribute('uv',new THREE.BufferAttribute(floats.subarray(m.offset+6*n,m.offset+8*n),2));
    g.computeBoundingSphere();return g;
  });
  const materials=new Map();
  function material(geom){
    const m=scene.materials[geom.mat],isDefault=[.5,.5,.5,1].every((v,i)=>Math.abs(geom.rgba[i]-v)<1e-6);
    const rgba=m&&isDefault?m.rgba:geom.rgba,key=`${geom.mat}|${rgba.join(',')}`;
    if(!materials.has(key)){
      const color=new THREE.Color(rgba[0],rgba[1],rgba[2]),s=m?.specular??.5,e=m?.emission??0;
      const map=m&&m.tex>=0?textures[m.tex]:null;
      if(map&&(m.texrepeat[0]!==1||m.texrepeat[1]!==1)){map.repeat.set(m.texrepeat[0],m.texrepeat[1]);}
      materials.set(key,new THREE.MeshPhongMaterial({color,map,specular:new THREE.Color(s,s,s),shininess:(m?.shininess??.5)*128,
        emissive:color.clone().multiplyScalar(e),transparent:rgba[3]<1,opacity:rgba[3]}));
    }
    return materials.get(key);
  }
  bodies=scene.bodies.map((_,b)=>{const g=b?new THREE.Group():world;if(b)world.add(g);return g;});
  for(const geom of scene.geoms){
    if(geom.mesh<0)continue;
    const mesh=new THREE.Mesh(geometries[geom.mesh],material(geom));
    mesh.position.set(...geom.pos);mesh.quaternion.set(geom.quat[1],geom.quat[2],geom.quat[3],geom.quat[0]);
    mesh.castShadow=geom.body!==0;mesh.receiveShadow=true;
    bodies[geom.body].add(mesh);geomObjects.set(geom.id,mesh);
  }
  walls=cutawayWalls(scene,floats);
  // Lights: MuJoCo's headlight rides with the camera; model lights are spot lights.
  // Intensities carry a factor of pi to match fixed-function lighting.
  const h=scene.headlight;
  world.add(new THREE.AmbientLight(new THREE.Color(...h.ambient),Math.PI));
  headlight.color.setRGB(...h.diffuse);headlight.intensity=Math.PI*(h.active?1:0);
  for(const l of scene.lights){
    const light=new THREE.SpotLight(new THREE.Color(...l.diffuse),Math.PI,0,Math.PI/4,.6,0);
    light.position.set(...l.pos);light.target.position.set(l.pos[0]+l.dir[0],l.pos[1]+l.dir[1],l.pos[2]+l.dir[2]);
    light.castShadow=true;light.shadow.mapSize.set(2048,2048);light.shadow.bias=-.0005;
    world.add(light,light.target);
  }
}

// ---- Network ---------------------------------------------------------------
const playout=new Playout();
let ws=null,seat=-1,lobby=null,latest=null,rtt=null,retry=0,ticks=[],connectedOnce=false;
function decodeTick(buffer){
  const v=new DataView(buffer);if(v.byteLength!==328||v.getUint8(0)!==1)return null;
  const qpos=new Float32Array(72);for(let i=0;i<72;i++)qpos[i]=v.getFloat32(40+4*i,true);
  return {mode:MODES[v.getUint8(1)],phase:v.getUint8(2),terminal:(v.getUint8(3)&1)!==0,tick:v.getUint32(4,true),
    serverTime:v.getFloat64(8,true),timeRemaining:v.getFloat32(16,true),round:v.getUint8(20),fightResult:v.getUint8(21),
    fightWinner:v.getInt8(22),points:[v.getUint16(24,true),v.getUint16(26,true)],falls:[v.getUint8(28),v.getUint8(29)],
    wins:[v.getUint8(30),v.getUint8(31)],commands:[v.getUint8(32),v.getUint8(33)],reasons:[v.getUint8(34),v.getUint8(35)],
    hold:v.getFloat32(36,true),qpos};
}
function connect(){
  ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);ws.binaryType='arraybuffer';
  $('connection').textContent='Connecting';
  ws.onopen=()=>{retry=0;connectedOnce=true;$('connection').textContent='Connected';};
  ws.onmessage=event=>{
    if(event.data instanceof ArrayBuffer){
      const tick=decodeTick(event.data);if(!tick)return;
      playout.push(tick,performance.now());
      if(!latest||tick.tick!==latest.tick){ticks.push(performance.now());}
      latest=tick;updateHud(tick);return;
    }
    const m=JSON.parse(event.data);
    if(m.t==='lobby'){lobby=m;const before=seat;seat=m.seat;if(before!==seat)releaseKeys();updateLobby();}
    else if(m.t==='pong'){const sample=performance.now()-m.c;rtt=rtt===null?sample:rtt*.7+sample*.3;}
    else if(m.t==='error')flash(m.error,false);
  };
  ws.onclose=async()=>{
    seat=-1;$('connection').textContent='Disconnected · retrying';
    const r=await fetch('/scene/scene.json',{method:'HEAD',cache:'no-store'}).catch(()=>null);
    if(r&&r.status===401){location.href='/login';return;}
    setTimeout(connect,Math.min(5000,500*2**retry++));
  };
}
setInterval(()=>{if(ws?.readyState===1)ws.send(JSON.stringify({t:'ping',c:performance.now()}));},1000);
setInterval(()=>{
  const cutoff=performance.now()-2000;ticks=ticks.filter(t=>t>cutoff);
  $('ping').textContent=rtt===null?'ping –':`ping ${Math.round(rtt)} ms`;
  $('rate').textContent=latest&&latest.hold===0&&latest.mode!=='idle'?`sim ${Math.round(ticks.length/2)}/s · buffer ${Math.round(playout.samples.length?playout.delay():0)} ms`:'sim –';
},500);

// ---- Input -----------------------------------------------------------------
const MOVE_KEYS={KeyW:'W',KeyA:'A',KeyS:'S',KeyD:'D',KeyQ:'Q',KeyE:'E'};
let keyboard=null,seq=Date.now()*1024;const held=new Set();
const controlling=()=>ws?.readyState===1&&seat>=0&&lobby&&(lobby.mode==='pvp'||lobby.mode==='bot'&&seat===0);
function sendInput(move=null,cancelAction=false){
  if(!controlling())return;
  ws.send(JSON.stringify({t:'input',seq:++seq,held:[...held],move,cancelAction}));
}
function releaseKeys(){held.clear();keyboard?.reset();if(ws?.readyState===1)ws.send(JSON.stringify({t:'release'}));}
addEventListener('keydown',e=>{
  if(e.target instanceof HTMLInputElement)return;
  if(e.code==='KeyV'&&!e.repeat){toggleView();return;}
  const known=MOVE_KEYS[e.code]||e.code==='KeyX'||keyboard?.recognizes(e.code);if(!known)return;
  e.preventDefault();if(e.repeat)return;
  if(MOVE_KEYS[e.code])held.add(MOVE_KEYS[e.code]);
  const move=keyboard?.recognizes(e.code)?keyboard.press(e.code,performance.now(),false):null;
  sendInput(move,e.code==='KeyX');
});
addEventListener('keyup',e=>{
  if(!(MOVE_KEYS[e.code]||keyboard?.recognizes(e.code)))return;
  e.preventDefault();keyboard?.release(e.code);
  if(MOVE_KEYS[e.code]&&held.delete(MOVE_KEYS[e.code]))sendInput();
});
addEventListener('blur',releaseKeys);
document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseKeys();});
canvas.addEventListener('pointerdown',()=>canvas.focus({preventScroll:true}));

// ---- HUD -------------------------------------------------------------------
// Lock-on (behind you, facing the opponent) is the calmest view; follow tracks your facing.
const VIEWS=['lock-on','follow','overview'];let view=0;
function toggleView(){view=(view+1)%VIEWS.length;$('toggle-view').textContent=`View: ${VIEWS[view]}`;}
$('toggle-view').addEventListener('click',toggleView);
$('toggle-help').addEventListener('click',()=>{const h=$('help');h.hidden=!h.hidden;$('toggle-help').setAttribute('aria-expanded',String(!h.hidden));});
$('fight-bot').addEventListener('click',()=>{if(ws?.readyState===1)ws.send(JSON.stringify({t:'bot'}));canvas.focus({preventScroll:true});});
let flashTimer=0;
function flash(text,good){
  const el=$('attack');el.textContent=text;el.className=`attack show ${good?'good':'bad'}`;
  clearTimeout(flashTimer);flashTimer=setTimeout(()=>{el.className='attack';},900);
}
const nameOf=side=>lobby?.mode==='bot'&&side===1?'Bot 1':lobby?.players?.[side]?.name??(side===1?'Waiting…':'—');
function updateLobby(){
  for(const side of [0,1]){$(`name-${side}`).textContent=nameOf(side);$(`name-${side}`).parentElement.classList.toggle('you',seat===side);}
  $('fight-bot').hidden=!lobby.canFightBot;
  if(lobby.failure)$('connection').textContent=lobby.failure;
  updateOverlay();
}
function updateOverlay(){
  const o=$('overlay'),title=$('overlay-title'),text=$('overlay-text'),t=latest,hold=t?.hold??0;
  if(!scene){o.hidden=false;return;}
  if(!lobby){o.hidden=false;title.textContent='Connecting';text.textContent='';return;}
  const you=seat,opponent=lobby.mode==='bot'?'Bot 1':nameOf(1-Math.max(0,you));
  let show=true;
  if(lobby.mode==='idle'){
    if(you<0){title.textContent='Game full';text.textContent='You are watching. You will take a seat when a player leaves.';}
    else if(t?.fightResult>0&&lobby.players.filter(Boolean).length===1){title.textContent=t.fightWinner===0?'You beat Bot 1':'Bot 1 wins';
      text.textContent='Fight Bot 1 again, or wait here: a second player who logs in starts a match against you.';}
    else{title.textContent='Waiting for an opponent';text.textContent='Share this page. When a second player logs in, the match starts automatically.';}
  }else if(hold>0&&t.fightResult>0){
    const winner=t.fightWinner===you?'You win':`${t.fightWinner===0?nameOf(0):nameOf(1)} wins`;
    title.textContent=you<0?`${nameOf(t.fightWinner)} wins`:winner;
    text.textContent=lobby.mode==='pvp'?`Next match in ${Math.ceil(hold)} s`:`Back to the lobby in ${Math.ceil(hold)} s`;
  }else if(hold>0&&t.terminal){title.textContent='Round over';text.textContent=`Next round in ${Math.ceil(hold)} s`;}
  else if(hold>0){title.textContent=`${Math.ceil(hold)}`;text.textContent=you<0?'Match starting':`You versus ${opponent}. Click the arena, then fight.`;}
  else show=false;
  o.hidden=!show;
}
function updateHud(t){
  for(const side of [0,1]){$(`points-${side}`).textContent=t.points[side];$(`wins-${side}`).textContent=t.wins[side];$(`falls-${side}`).textContent=t.falls[side];}
  const s=Math.max(0,Math.ceil(t.timeRemaining));$('clock').textContent=`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
  $('round-status').textContent=t.mode==='idle'?'Lobby':`Round ${t.round} · ${PHASES[t.phase]??'–'}`;
  if(seat>=0&&t.mode!=='idle'){
    const flags=t.commands[seat];
    if(flags&2)flash('Attack',true);else if(flags&4)flash(`Rejected: ${REASONS[t.reasons[seat]]||'busy'}`,false);
  }
  updateOverlay();
}

// ---- Frame loop ------------------------------------------------------------
const chase=new ChaseCamera(),overviewCam=new OverviewCamera();let lastFrame=performance.now(),lastTick=-1,hidden=new Set();
function draw(now){
  requestAnimationFrame(draw);
  const dt=Math.min(.1,(now-lastFrame)/1000);lastFrame=now;
  const sample=scene&&playout.sample(now);if(!sample){renderer.render(world,camera);return;}
  const [a,b,t]=sample,qpos=a===b?a.qpos:interpolateQpos(scene,a.qpos,b.qpos,t);
  const frames=forwardKinematics(scene,qpos);
  for(let i=1;i<bodies.length;i++){const p=frames.xpos[i],q=frames.xquat[i];bodies[i].position.set(p[0],p[1],p[2]);bodies[i].quaternion.set(q[1],q[2],q[3],q[0]);}
  const side=seat>=0&&VIEWS[view]!=='overview'?seat:-1,snap=a.tick<lastTick;lastTick=b.tick;
  // A new match cuts; the cameras ease everything else.
  const pose=side>=0?chase.update(qpos,side,dt,{lockOn:VIEWS[view]==='lock-on',snap}):overviewCam.update(qpos,dt,{snap});
  camera.position.set(...pose.eye);camera.lookAt(...pose.target);
  headlight.position.set(...pose.eye);headlight.target.position.set(...pose.target);
  const next=side>=0?hiddenWalls(walls,pose.eye):new Set();
  for(const id of hidden)if(!next.has(id))geomObjects.get(id).visible=true;
  for(const id of next)geomObjects.get(id).visible=false;hidden=next;
  renderer.render(world,camera);
}

// ---- Start -----------------------------------------------------------------
(async()=>{
  keyboard=new globalThis.RekControls.SavedRekKeyboard();
  for(const binding of globalThis.RekControls.savedRekBindings){
    const button=document.createElement('button'),key=document.createElement('b'),label=document.createElement('small');
    key.textContent=(binding.doubleTap?'Double ':'')+binding.codes.map(c=>c.replace('Key','').replace('Semicolon',';').replace('Quote',"'")).join('+');
    label.textContent=binding.commandId.replace('move:','').replace('_processed','').replaceAll('_',' ');button.append(key,label);
    button.addEventListener('pointerdown',e=>{e.preventDefault();sendInput(binding.category);});$('moves').append(button);
  }
  try{await loadScene();}catch(error){if(error.message==='login'){location.href='/login';return;}$('overlay-title').textContent='Could not load the arena';$('overlay-text').textContent=error.message;return;}
  connect();requestAnimationFrame(draw);canvas.focus({preventScroll:true});
})();
