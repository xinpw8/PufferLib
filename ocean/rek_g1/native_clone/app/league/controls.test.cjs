'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {SavedRekKeyboard,savedRekBindings}=require('./public/controls.js');
const {HumanInput}=require('./input.cjs');
test('all seventeen saved bindings produce corresponding one-shot native move index',()=>{
  const saved=JSON.parse(fs.readFileSync(path.join(__dirname,'../saved-g1-bindings.json'),'utf8'));
  assert.deepEqual(savedRekBindings,saved.bindings);
  for(const row of savedRekBindings){
    const keys=new SavedRekKeyboard();let move=null;
    for(const code of row.codes)move=keys.press(code,100);
    if(row.doubleTap){assert.equal(move,null);for(const c of row.codes)keys.release(c);for(const c of row.codes)move=keys.press(c,200);}
    assert.equal(move,row.category,row.commandId);
    const input=new HumanInput();input.update({seq:1,held:['W','D'],move});
    assert.deepEqual(input.next(),{forward:1,strafe:-1,yaw:0,moveIndex:row.category-16,cancelAction:false});
    assert.equal(input.next().moveIndex,-1);
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
