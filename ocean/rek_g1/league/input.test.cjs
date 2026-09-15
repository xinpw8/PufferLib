'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {HumanInput}=require('./input.cjs');
function state(){return {mask:Array(66).fill(1),raw:Array(446).fill(0)};}
test('held inputs persist and release',()=>{const i=new HumanInput(),s=state();
  i.update({seq:1,held:['W','Q']});assert.equal(i.next(s),8);assert.equal(i.next(s),8);
  i.update({seq:2,held:[]});assert.equal(i.next(s),1);});
test('attack during attack is discarded',()=>{const i=new HumanInput(),s=state();s.raw[182]=1;
  i.update({seq:1,held:[],move:17});assert.equal(i.next(s),1);s.raw[182]=0;assert.equal(i.next(s),1);});
test('one yaw interruption buffers without stacking',()=>{const i=new HumanInput(),s=state();s.mask[17]=0;
  i.update({seq:1,held:['Q'],move:17});assert.equal(i.next(s),1);assert.equal(i.pending.category,17);
  i.update({seq:2,held:['Q'],move:18});s.mask[17]=1;assert.equal(i.next(s),17);
  assert.equal(i.next(s),6);assert.equal(i.pending,null);});
test('translation prevents attack buffering',()=>{const i=new HumanInput(),s=state();
  i.update({seq:1,held:['W','E'],move:17});assert.equal(i.next(s),9);
  i.update({seq:2,held:[]});assert.equal(i.next(s),1);});
test('no quit action and stale sequence ignored',()=>{const i=new HumanInput();
  assert.throws(()=>i.update({seq:1,held:[],move:33}));
  i.update({seq:2,held:['D']});assert.equal(i.update({seq:1,held:[]}),false);});
test('orange human uses orange busy state and legal mask',()=>{const i=new HumanInput(),s=state();
  s.raw[182]=1;s.mask[17]=0;
  i.update({seq:1,held:[],move:17});assert.equal(i.next(s,1),17);
  s.raw[223+182]=1;i.update({seq:2,held:[],move:18});assert.equal(i.next(s,1),1);
  assert.throws(()=>i.next(s,2));});

const KEYS=['W','S','A','D','Q','E'];
function chord(bits){return KEYS.filter((_,index)=>bits&(1<<index));}
function expectedCategory(bits){
  const axis=(positive,negative)=>Number(Boolean(bits&positive))-Number(Boolean(bits&negative));
  const forward=axis(1,2),strafe=axis(4,8),yaw=axis(16,32);
  const translation=forward>0?1:forward<0?2:strafe>0?3:strafe<0?4:0;
  return [[1,2,3,4,5],[6,8,10,12,14],[7,9,11,13,15]][yaw>0?1:yaw<0?2:0][translation];
}
const TRANSLATION=new Set([2,3,4,5,8,9,10,11,12,13,14,15]);

test('all 64 held chords resolve on both sides independent of order and duplicates',()=>{
  for(let bits=0;bits<64;bits++)for(const side of [0,1]){
    const held=chord(bits),expected=expectedCategory(bits);
    for(const keys of [held,[...held].reverse(),[...held,...held]]){
      const i=new HumanInput(),s=state();
      assert.equal(i.update({seq:1,held:keys}),true);
      assert.equal(i.next(s,side),expected,`${keys.join('+')} side${side}`);
      assert.equal(i.next(s,side),expected,'held command persists without another packet');
    }
  }
});

test('all 4096 chord transitions resolve without stale movement',()=>{
  const s=state();
  for(let before=0;before<64;before++)for(let after=0;after<64;after++){
    const i=new HumanInput();i.update({seq:1,held:chord(before)});i.next(s);
    i.update({seq:2,held:chord(after)});assert.equal(i.next(s),expectedCategory(after));
    i.release();assert.equal(i.next(s),1);
  }
});

test('diagonals use forward/back priority, opposing axes cancel and yaw is independent',()=>{
  const i=new HumanInput(),s=state();
  const sequence=[
    [['W','A'],2],[['W','A','Q'],8],[['W','S','A','Q'],12],
    [['A','D','Q'],6],[['W','Q','E'],2],[['W','S','A','D','Q','E'],1],
    [['S','D','E'],11],[['D','E'],15],[[],1],
  ];
  sequence.forEach(([held,action],index)=>{
    i.update({seq:index,held});assert.equal(i.next(s),action);
  });
});

test('all chords retain legal action masks on both fighter sides',()=>{
  for(let bits=0;bits<64;bits++)for(const side of [0,1]){
    const i=new HumanInput(),s=state(),category=expectedCategory(bits);
    s.mask[side*33+category]=0;i.update({seq:1,held:chord(bits)});
    assert.equal(i.next(s,side),0,'masked held command falls back to continue');
  }
});

test('all chords preserve attack locking and only resolved yaw can buffer',()=>{
  for(let bits=0;bits<64;bits++){
    const i=new HumanInput(),s=state(),held=chord(bits),category=expectedCategory(bits);
    const translating=TRANSLATION.has(category),yawOnly=category===6||category===7;
    i.update({seq:1,held,move:17});
    assert.equal(i.next(s),translating?category:17);
    assert.equal(i.pending,null);
    s.raw[182]=1;i.update({seq:2,held,move:18});
    assert.equal(i.next(s),category);assert.equal(i.pending,null);
    s.raw[182]=0;i.update({seq:3,held:[]});assert.equal(i.next(s),1,'busy attack never replays');

    i.reset();s.mask[17]=0;i.update({seq:1,held,move:17});
    assert.equal(i.next(s),translating?category:1);
    assert.equal(i.pending?.category??null,yawOnly?17:null);
    i.update({seq:2,held,move:18});s.mask[17]=1;
    assert.equal(i.next(s),translating?category:yawOnly?17:18);
    assert.equal(i.next(s),category,'additional attack must not remain stacked');
  }
});

test('rejected payloads do not mutate held input, sequence or pending attack',()=>{
  const i=new HumanInput(),s=state();s.mask[17]=0;
  i.update({seq:3,held:['Q'],move:17});i.next(s);
  for(const packet of [{seq:4,held:['F']},{seq:4,held:'W'},{seq:4,held:[],move:33}]){
    assert.throws(()=>i.update(packet));assert.equal(i.seq,3);
    assert.deepEqual(i.held,['Q']);assert.equal(i.pending.category,17);
  }
  assert.equal(i.update({seq:2,held:['W','A']}),false);
  assert.deepEqual(i.held,['Q']);
});
