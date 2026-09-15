#!/usr/bin/env node
'use strict';
// Convert an observed X11 XWD framebuffer to PNG. No rendered/synthetic state.
const fs=require('node:fs'),zlib=require('node:zlib');
const source=fs.readFileSync(process.argv[2]);
const word=i=>source.readUInt32BE(i*4);
if(word(1)!==7||word(2)!==2||![24,32].includes(word(11)))throw Error('unsupported XWD layout');
const width=word(4),height=word(5),stride=word(12),offset=word(0)+word(19)*12;
const pixelBytes=word(11)/8;
if(width>8192||height>8192||offset+height*stride>source.length)throw Error('invalid XWD size');
const rows=Buffer.alloc((width*3+1)*height);
const masks=[word(14),word(15),word(16)];
const shifts=masks.map(m=>{let n=0;while(n<32&&((m>>>n)&1)===0)n++;return n;});
for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const p=offset+y*stride+x*pixelBytes;
  const pixel=word(7)===0?source.readUIntLE(p,pixelBytes):source.readUIntBE(p,pixelBytes);
  for(let c=0;c<3;c++)rows[y*(width*3+1)+1+x*3+c]=Math.round(((pixel&masks[c])>>>shifts[c])*255/(masks[c]>>>shifts[c]));
}
function crc32(b){let c=0xffffffff;for(const v of b){c^=v;for(let n=0;n<8;n++)c=(c>>>1)^((c&1)?0xedb88320:0);}return(c^0xffffffff)>>>0;}
function chunk(type,data){const b=Buffer.alloc(data.length+12);b.writeUInt32BE(data.length,0);b.write(type,4);data.copy(b,8);b.writeUInt32BE(crc32(b.subarray(4,-4)),b.length-4);return b;}
const header=Buffer.alloc(13);header.writeUInt32BE(width,0);header.writeUInt32BE(height,4);header[8]=8;header[9]=2;
fs.writeFileSync(process.argv[3],Buffer.concat([Buffer.from('89504e470d0a1a0a','hex'),chunk('IHDR',header),chunk('IDAT',zlib.deflateSync(rows,{level:1})),chunk('IEND',Buffer.alloc(0))]),{flag:'wx'});
console.log(JSON.stringify({width,height,pixelBytes,reportedBitsPerPixel:word(11),source:'X11 observed framebuffer',input_emitted:false}));
