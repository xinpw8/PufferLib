'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {SavedRekKeyboard,savedRekBindings}=require('./public/controls.js');
const {HumanInput}=require('./input.cjs');
const crypto=require('node:crypto');
const manifestBytes=fs.readFileSync(path.join(__dirname,'../validation/semantic_duel_assets_manifest.json'));
assert.equal(crypto.createHash('sha256').update(manifestBytes).digest('hex'),'7d4719a3ca1e9e5a8faf571bc3c5c70e2b4c9e841be34303fa0a3f47d3692a28');
const nativeAssets=JSON.parse(manifestBytes);
// Join named saved commands to the original clip roles, independently of category order.
const commandRoles={
  'move:left_hook_processed':'left_hook','move:left_jab_processed':'left_jab',
  'move:double_uppercut_processed':'double_uppercut','move:right_hook_processed':'right_hook',
  'move:right_jab_processed':'right_jab','move:left_jab_right_uppercut_processed':'left_jab_right_uppercut',
  'move:left_side_kick_processed':'kick_left_side','move:left_front_kick_processed':'kick_left_front',
  'move:right_side_kick_processed':'kick_right_side','move:right_knee_processed':'kick_right_knee',
  'move:6_punch_processed':'six_punch','move:run_and_punch_processed':'run_and_punch',
  'move:left_right_jab_processed':'left_right_jab','move:left_right_hook_processed':'left_right_hook',
  'move:left_hook_right_jab_processed':'left_hook_right_jab','move:double_hook_processed':'double_hook',
  'move:butt_smack_emote_processed':'butt_smack_emote'
};
function originalMove(commandId){
  const clips=nativeAssets.clips.filter(c=>c.role===commandRoles[commandId]);assert.equal(clips.length,1,commandId);
  const routes=nativeAssets.routes.filter(r=>r.npz_path_id===clips[0].npz_path_id&&r.runtime_move_index!==null);
  assert.equal(routes.length,1,commandId);return routes[0].runtime_move_index;
}
test('all seventeen saved bindings produce corresponding one-shot native move index',()=>{
  const saved=JSON.parse(fs.readFileSync(path.join(__dirname,'../saved-g1-bindings.json'),'utf8'));
  assert.deepEqual(savedRekBindings,saved.bindings);
  for(const row of savedRekBindings){
    const keys=new SavedRekKeyboard();let move=null;
    for(const code of row.codes)move=keys.press(code,100);
    if(row.doubleTap){assert.equal(move,null);for(const c of row.codes)keys.release(c);for(const c of row.codes)move=keys.press(c,200);}
    assert.equal(move,row.category,row.commandId);
    const input=new HumanInput();input.update({seq:1,held:['W','D'],move});
    assert.deepEqual(input.next(),{forward:1,strafe:-1,yaw:0,moveIndex:originalMove(row.commandId),cancelAction:false},row.commandId);
    assert.equal(input.next().moveIndex,-1);
  }
});
test('all seventeen button categories select the named original clip without keyboard reinterpretation',()=>{
  assert.equal(Object.keys(commandRoles).length,17);
  const seen=new Set();
  for(const row of savedRekBindings){
    const input=new HumanInput();input.update({seq:1,held:[],move:row.category});
    const actual=input.next().moveIndex;assert.equal(actual,originalMove(row.commandId),row.commandId);seen.add(actual);
  }
  assert.equal(seen.size,17);
});
test('reported HH, UU and L gestures reach front kick, side kick and jab in the original registry',()=>{
  for(const [code,doubleTap,expected] of [['KeyH',true,7],['KeyU',true,8],['KeyL',false,4]]){
    const keyboard=new SavedRekKeyboard();let move=keyboard.press(code,100);
    if(doubleTap){assert.equal(move,null);keyboard.release(code);move=keyboard.press(code,200);}
    const input=new HumanInput();input.update({seq:1,held:[],move});assert.equal(input.next().moveIndex,expected,code);
    assert.equal(input.next().moveIndex,-1,'one shot');
  }
});
test('saved chords take priority and double taps require a released edge within300ms',()=>{
  for(const row of savedRekBindings.filter(x=>x.codes.includes('Space'))){
    const k=new SavedRekKeyboard();assert.equal(k.press('Space',10),null);
    assert.equal(k.press(row.codes.find(x=>x!=='Space'),20),row.category);assert.equal(k.pending,null);
  }
  for(const dt of [0,299,300,300.01]){
    const k=new SavedRekKeyboard();assert.equal(k.press('KeyY',100),null);
    assert.equal(k.press('KeyY',101,true),null);assert.equal(k.press('KeyY',102),null);k.release('KeyY');
    assert.equal(k.press('KeyY',100+dt),dt<=300?16:null);k.reset();assert.equal(k.press('KeyY',2000),null);
  }
});
