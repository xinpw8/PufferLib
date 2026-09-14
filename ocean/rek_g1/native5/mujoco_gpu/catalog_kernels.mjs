// Build-time inspection of existing Warp-generated CUDA, using Node only.
// Generated kernels and PTX stay outside the repository. Nothing executes here.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
export function parameters(text){
  let nesting=0,start=0;const output=[];
  for(let i=0;i<=text.length;i++){
    const ch=text[i];if(ch==='<')nesting++;if(ch==='>')nesting--;
    if(nesting<0)throw new Error('Invalid CUDA template nesting');
    if(i===text.length || (ch===','&&nesting===0)){
      const declaration=text.slice(start,i).trim();start=i+1;
      if(!declaration)continue;
      const match=declaration.match(/^([\s\S]*\S)\s+([A-Za-z_]\w*)$/);
      if(!match)throw new Error(`Unsupported kernel declaration: ${declaration}`);
      output.push({type:match[1].replace(/\s+/g,' '),name:match[2]});
    }
  }
  if(nesting!==0)throw new Error('Unterminated CUDA template');
  return output;
}
export function kernels(source){
  const output=[];
  const pattern=/extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{/g;
  for(const match of source.matchAll(pattern))output.push({symbol:match[1],parameters:parameters(match[2])});
  return output;
}
export function blockDimension(source){
  const matches=[...source.matchAll(/^\s*#define\s+WP_TILE_BLOCK_DIM\s+(\d+)\s*$/gm)];
  if(matches.length!==1)throw new Error('Expected one generated WP_TILE_BLOCK_DIM');
  const value=Number(matches[0][1]);
  if(value<1||value>1024)throw new Error('Invalid generated CUDA block dimension');
  return value;
}
export function sharedBytes(symbol,metadata){
  const value=metadata[`${symbol}_smem_bytes`];
  if(!Number.isSafeInteger(value)||value<0)throw new Error(`Missing shared-memory metadata: ${symbol}`);
  return value;
}
export function catalog(cache,architecture='sm121'){
  const modules=[];
  for(const entry of fs.readdirSync(cache,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){
    if(!entry.isDirectory()||!entry.name.startsWith('wp_'))continue;
    const directory=path.join(cache,entry.name);
    const source=path.join(directory,`${entry.name}.cu`);
    const ptx=path.join(directory,`${entry.name}.${architecture}.ptx`);
    const meta=path.join(directory,`${entry.name}.meta`);
    if(!fs.existsSync(source)||!fs.existsSync(ptx)||!fs.existsSync(meta))continue;
    const bytes=fs.readFileSync(source),functions=kernels(bytes.toString('utf8'));
    if(!functions.length)continue;
    const metaBytes=fs.readFileSync(meta),metadata=JSON.parse(metaBytes);
    for(const fn of functions)fn.sharedBytes=sharedBytes(fn.symbol,metadata);
    modules.push({name:entry.name,source,ptx,meta,blockDim:blockDimension(bytes.toString('utf8')),sourceSha256:hash(bytes),ptxSha256:hash(fs.readFileSync(ptx)),metaSha256:hash(metaBytes),kernels:functions});
  }
  if(!modules.length)throw new Error('No complete generated CUDA/PTX modules found');
  return {schema:'rek-native-mujoco-kernel-catalog-v1',architecture,modules};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const [cache,out,architecture]=process.argv.slice(2);
  if(!cache||!out)throw new Error('Usage: node catalog_kernels.mjs CACHE OUTPUT_JSON [sm121]');
  const result=catalog(cache,architecture);
  fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});
  console.log(JSON.stringify({modules:result.modules.length,kernels:result.modules.reduce((n,m)=>n+m.kernels.length,0),out}));
}
