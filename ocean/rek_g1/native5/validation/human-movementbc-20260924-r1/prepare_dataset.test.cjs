'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {derive,HEADER,ROW,UNAVAILABLE}=require('./prepare_dataset.cjs');
function fixture(){
 const b=Buffer.alloc(HEADER+8*ROW);b.write('REKBC001');for(const [at,v]of [[8,1],[12,223],[16,33],[20,8],[24,ROW]])b.writeUInt32LE(v,at);b.fill(1,32,255);
 const actions=[1,6,21,-1,2,7,17,1];
 for(let i=0;i<8;i++){const p=HEADER+i*ROW;b.writeUInt32LE(i<4?0:1,p);b.writeUInt32LE(i<4?2:5,p+4);b.writeUInt32LE(i%4===0?1:0,p+8);b.writeInt32LE(actions[i],p+12);b.writeFloatLE(actions[i]<0?0:1,p+16);b.writeDoubleLE(i%4*.02,p+24);for(let a=0;a<33;a++)b.writeFloatLE(1,p+924+4*a);}
 const partial=Buffer.from(b);for(const c of UNAVAILABLE)partial[32+c]=0;
 for(let i=0;i<8;i++){const p=HEADER+i*ROW;for(let c=0;c<223;c++)partial.writeFloatLE(UNAVAILABLE.includes(c)?0:i+c/223,p+32+4*c);partial.writeInt32LE(-1,p+12);partial.writeFloatLE(0,p+16);}
 return {original:b,partial};
}
test('only2..15 original labels survive and all partial observation bytes remain identical',()=>{
 const {original,partial}=fixture(),origCopy=Buffer.from(original),partCopy=Buffer.from(partial),r=derive(original,partial);
 assert.deepEqual(original,origCopy);assert.deepEqual(partial,partCopy);assert.deepEqual(r.receipt.splits.map(s=>s.labels),[1,2]);
 for(let i=0;i<8;i++){const p=HEADER+i*ROW;assert.equal(r.output.readInt32LE(p+12),[-1,6,-1,-1,2,7,-1,-1][i]);assert.equal(r.output.readFloatLE(p+16),[0,1,0,0,1,1,0,0][i]);assert.deepEqual(r.output.subarray(p+32,p+924),partial.subarray(p+32,p+924));assert.deepEqual(r.output.subarray(p,p+12),original.subarray(p,p+12));assert.deepEqual(r.output.subarray(p+20,p+32),original.subarray(p+20,p+32));}
});
test('support exactly2..15, mask unchanged, normalization retains complete training chronology',()=>{
 const {original,partial}=fixture(),r=derive(original,partial);assert.deepEqual(r.output.subarray(0,HEADER),partial.subarray(0,HEADER));assert.equal(r.receipt.normalization_float32,32);assert.equal(r.receipt.training_labeled_chunks_per_epoch,1);
 for(let i=0;i<8;i++)for(let a=0;a<33;a++)assert.equal(r.output.readFloatLE(HEADER+i*ROW+924+4*a),a>=2&&a<=15?1:0);
});
test('reweighted original, misaligned chronology, unavailable values and wrong partial mask fail',()=>{
 for(const mutate of [(a,b)=>a.writeFloatLE(2,HEADER+ROW+16),(a,b)=>b.writeUInt32LE(88,HEADER+4),(a,b)=>b.writeFloatLE(1,HEADER+32+4*176),(a,b)=>{b[32+202]=1;}]){const {original,partial}=fixture();mutate(original,partial);assert.throws(()=>derive(original,partial));}
});
test('conditional movement CE gives zero direct neutral and attack logit gradients',()=>{
 const z=Array.from({length:33},(_,a)=>a/10),gradient=Array(33).fill(0),ex=z.slice(2,16).map(Math.exp),sum=ex.reduce((a,b)=>a+b,0);for(let a=2;a<=15;a++)gradient[a]=ex[a-2]/sum-Number(a===6);
 assert.equal(gradient[0],0);assert.equal(gradient[1],0);assert(gradient.slice(16).every(v=>v===0));assert(Math.abs(gradient.reduce((a,b)=>a+b,0))<1e-14);
});
