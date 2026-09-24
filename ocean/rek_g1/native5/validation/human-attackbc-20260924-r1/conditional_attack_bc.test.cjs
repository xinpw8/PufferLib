'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {derive,HEADER,ROW,ACTIONS}=require('./conditional_attack_bc.cjs');
const {project,COLUMNS,UNAVAILABLE}=require('./prepare_dataset.cjs');
function fixture(){
  const b=Buffer.alloc(HEADER+6*ROW);b.write('REKBC001');
  for(const [at,v] of [[8,1],[12,223],[16,33],[20,6],[24,ROW]])b.writeUInt32LE(v,at);
  b.fill(1,32,255);
  for(let i=0;i<6;i++){
    const p=HEADER+i*ROW,a=[1,17,-1,2,21,17][i];
    b.writeUInt32LE(i<3?0:1,p);b.writeUInt32LE(i<3?11:22,p+4);b.writeUInt32LE(i%3===0?1:0,p+8);
    b.writeInt32LE(a,p+12);b.writeFloatLE(a<0?0:1,p+16);b.writeDoubleLE(i%3*.02,p+24);
    for(let c=0;c<223;c++)b.writeFloatLE(i+c/223,p+32+4*c);
    for(let a=0;a<ACTIONS;a++)b.writeFloatLE(1,p+924+4*a);
  }return b;
}
test('only attack event rows retain labels; every recurrent row and metadata remains',()=>{
  const b=fixture(),before=Buffer.from(b),r=derive(b);assert.deepEqual(b,before);
  assert.equal(r.output.length,b.length);assert.deepEqual(r.receipt.splits.map(s=>[s.rows,s.labels]),[[3,1],[3,2]]);
  assert.deepEqual(r.receipt.splits[0].actions,{'17':1});
  for(let i=0;i<6;i++){
    const p=HEADER+i*ROW;
    assert.deepEqual(r.output.subarray(p,p+12),b.subarray(p,p+12));
    assert.deepEqual(r.output.subarray(p+20,p+924),b.subarray(p+20,p+924));
    assert.equal(r.output.readInt32LE(p+12),[-1,17,-1,-1,21,17][i]);
    assert.equal(r.output.readFloatLE(p+16),[0,1,0,0,1,1][i]);
  }
});
test('conditional support is exactly16..32 on every row; feature mask untouched',()=>{
  const b=fixture(),r=derive(b);assert.deepEqual(r.output.subarray(0,HEADER),b.subarray(0,HEADER));
  for(let i=0;i<6;i++)for(let a=0;a<33;a++)assert.equal(r.output.readFloatLE(HEADER+i*ROW+924+4*a),a>=16?1:0);
});
test('weighted, unsupported, malformed, or missing-split attack sources are rejected',()=>{
  for(const mutate of [b=>b.writeFloatLE(2,HEADER+ROW+16),b=>b.writeFloatLE(0,HEADER+924),b=>b.writeInt32LE(34,HEADER+12),b=>b.writeInt32LE(1,HEADER+ROW+12)]){
    const b=fixture();mutate(b);assert.throws(()=>derive(b));
  }
  assert.throws(()=>derive(fixture().subarray(0,-1)));
});
test('conditional CE has zero nonattack gradient and is invariant to nonattack logits/common attack shift',()=>{
  function ref(z,target,w=1){
    const m=Math.max(...z.slice(16)),ex=z.slice(16).map(v=>Math.exp(v-m)),sum=ex.reduce((a,b)=>a+b,0);
    const grad=Array(33).fill(0);for(let a=16;a<33;a++)grad[a]=w*(ex[a-16]/sum-Number(a===target));
    return {ce:w*(Math.log(sum)+m-z[target]),grad};
  }
  const z=Array.from({length:33},(_,a)=>a/11),r=ref(z,17);
  assert.deepEqual(r.grad.slice(0,16),Array(16).fill(0));assert.ok(Math.abs(r.grad.reduce((a,b)=>a+b,0))<1e-14);
  const variant=z.map((v,a)=>a<16?v+1000:v+7),s=ref(variant,17);
  assert.ok(Math.abs(r.ce-s.ce)<1e-14);for(let a=0;a<33;a++)assert.ok(Math.abs(r.grad[a]-s.grad[a])<1e-14);
  assert.ok(ref(z,17,0).grad.every(x=>x===0));
  const epsilon=1e-5;for(let a=0;a<33;a++){
    const hi=z.slice(),lo=z.slice();hi[a]+=epsilon;lo[a]-=epsilon;
    assert.ok(Math.abs((ref(hi,17).ce-ref(lo,17).ce)/(2*epsilon)-r.grad[a])<1e-9);
  }
});
test('balance8 overlay changes only8 specified channels and explicitly unknown11 fields',()=>{
  const b=fixture(),m=Buffer.from(b);
  for(let i=0;i<6;i++)for(let k=0;k<8;k++)m.writeFloatLE([i?-.3:0,i?.6:0,.25,.5,0,i?1:0,0,0][k],HEADER+i*ROW+32+4*COLUMNS[k]);
  const p=project(b,m);assert.equal(p.history_available_rows,5);
  for(let i=0;i<6;i++)for(let c=0;c<223;c++){
    const at=HEADER+i*ROW+32+4*c;
    if(UNAVAILABLE.includes(c))assert.equal(p.output.readFloatLE(at),0);
    else if(COLUMNS.includes(c))assert.deepEqual(p.output.subarray(at,at+4),m.subarray(at,at+4));
    else assert.deepEqual(p.output.subarray(at,at+4),b.subarray(at,at+4));
  }
  for(let c=0;c<223;c++)assert.equal(p.output[32+c],UNAVAILABLE.includes(c)?0:1);
});
test('projection rejects row mismatch, unsupported referee freshness and missing-history rates',()=>{
  for(const mutate of [m=>m.writeUInt32LE(123,HEADER+4),m=>m.writeFloatLE(1,HEADER+32+4*202),m=>m.writeFloatLE(.3,HEADER+32+4*9)]){
    const b=fixture(),m=Buffer.from(b);for(let i=0;i<6;i++)for(const c of COLUMNS)m.writeFloatLE(0,HEADER+i*ROW+32+4*c);
    mutate(m);assert.throws(()=>project(b,m));
  }
});
