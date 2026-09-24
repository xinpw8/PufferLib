'use strict';
const fs=require('node:fs'),cp=require('node:child_process'),assert=require('node:assert/strict');
const stage='/home/spark-advantage/rek-training/timing500-ppo-refresh-live-20260924-r1',controller=Number(process.argv[2]);
assert(Number.isInteger(controller)&&controller>1);
const expected=['node',stage+'/root-campaign/campaign.cjs'];
function active(){try{return JSON.stringify(fs.readFileSync('/proc/'+controller+'/cmdline').toString().split('\0').filter(Boolean))===JSON.stringify(expected);}catch{return false;}}
assert(active());const log=fs.openSync(stage+'/resource-snapshots.jsonl','wx',0o600);let samples=0;
const timer=setInterval(()=>{
 if(!active()||samples>=1800){clearInterval(timer);fs.closeSync(log);console.log(JSON.stringify({finished_utc:new Date().toISOString(),samples}));return;}
 const utc=new Date().toISOString();
 const gpu=cp.spawnSync('nvidia-smi',['--query-gpu=utilization.gpu,clocks.sm,temperature.gpu','--format=csv,noheader,nounits'],{encoding:'utf8',timeout:2000});
 const jobs=samples%5===0?cp.spawnSync('nvidia-smi',['--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],{encoding:'utf8',timeout:2000}):null;
 const memory=fs.readFileSync('/proc/meminfo','utf8').split('\n').filter(x=>/^(MemAvailable|SwapFree):/.test(x));
 fs.writeSync(log,JSON.stringify({utc,sample:samples++,gpu_exit:gpu.status,gpu:gpu.stdout?.trim(),jobs_exit:jobs?.status??null,jobs:jobs?.stdout?.trim()??null,memory,loadavg:fs.readFileSync('/proc/loadavg','utf8').trim(),observation_only:true})+'\n');
},2000);
