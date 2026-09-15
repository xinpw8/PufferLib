#!/usr/bin/env node
'use strict';
// Local read-only delivery of one authentic game recording and its evidence.
const http=require('node:http'),fs=require('node:fs'),path=require('node:path');
const directory=path.resolve(process.argv[2]),port=Number(process.argv[3]||18771);
const routes={
  '/fight.webm':['authentic-rek-policy-fight.webm','video/webm'],
  '/manifest.json':['recording-manifest.json','application/json'],
  '/result.json':['measured-report.json','application/json']
};
const page=`<!doctype html><meta charset="utf-8"><title>Authentic REK policy fight</title>
<style>body{margin:16px auto;max-width:1280px;background:#11151b;color:#e9edf3;font:16px system-ui}h1{font-size:24px}p{color:#c1cad6}video{display:block;width:100%;max-height:calc(100vh - 245px);min-height:180px;object-fit:contain;background:#000}button,a{padding:10px 15px;margin:8px 8px 8px 0;color:#e9edf3;background:#263347;border:1px solid #526174;border-radius:6px;display:inline-block}button{cursor:pointer}</style>
<h1>Authentic REK: trained V4 policy vs Sparring Bot 1</h1>
<p>Fresh recorded trial on Spark. Actual game pixels, no generated frames or visual overlay. The earlier 15–32 trial was not recorded.</p>
<video id="fight" controls preload="metadata" src="/fight.webm"></video>
<div><button id="play">Play</button><button id="pause">Pause</button><button id="start">Fight starts</button><button id="middle">Middle of fight</button><button id="end">Round finish</button></div>
<p id="status">Loading recording…</p><a href="/fight.webm" download="authentic-rek-policy-fight.webm">Download video</a><a href="/manifest.json">Recording identity</a><a href="/result.json">Measured result</a>
<script>const v=document.getElementById('fight'),s=document.getElementById('status');
function update(){s.textContent=v.videoWidth+' × '+v.videoHeight+' · '+v.currentTime.toFixed(1)+' / '+(Number.isFinite(v.duration)?v.duration.toFixed(1):'?')+' seconds';}
v.addEventListener('loadedmetadata',()=>{v.currentTime=10;update()});v.addEventListener('timeupdate',update);v.addEventListener('seeked',update);
document.getElementById('play').onclick=()=>v.play();document.getElementById('pause').onclick=()=>v.pause();
document.getElementById('start').onclick=()=>{v.pause();v.currentTime=10};document.getElementById('middle').onclick=()=>{v.pause();v.currentTime=60};document.getElementById('end').onclick=()=>{v.pause();v.currentTime=Math.max(0,v.duration-2)};
v.addEventListener('error',()=>s.textContent='Video playback error '+v.error?.code);
</script>`;
http.createServer((req,res)=>{
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405);res.end();return;}
  const url=new URL(req.url,'http://localhost');
  if(url.pathname==='/'){res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});res.end(req.method==='HEAD'?undefined:page);return;}
  const route=routes[url.pathname];if(!route){res.writeHead(404);res.end();return;}
  const file=path.join(directory,route[0]);let stat;try{stat=fs.statSync(file);}catch{res.writeHead(404);res.end();return;}
  let start=0,end=stat.size-1,status=200;const headers={'Content-Type':route[1],'Accept-Ranges':'bytes','Cache-Control':'no-cache'};
  if(req.headers.range){const m=/^bytes=(\d+)-(\d*)$/.exec(req.headers.range);if(!m){res.writeHead(416,{'Content-Range':'bytes */'+stat.size});res.end();return;}
    start=Number(m[1]);end=m[2]?Math.min(Number(m[2]),end):end;
    if(!Number.isSafeInteger(start)||start>end){res.writeHead(416,{'Content-Range':'bytes */'+stat.size});res.end();return;}
    status=206;headers['Content-Range']='bytes '+start+'-'+end+'/'+stat.size;
  }
  headers['Content-Length']=end-start+1;res.writeHead(status,headers);if(req.method==='HEAD'){res.end();return;}
  const stream=fs.createReadStream(file,{start,end});stream.on('error',()=>res.destroy());res.on('close',()=>stream.destroy());stream.pipe(res);
}).listen(port,'127.0.0.1',()=>console.log(JSON.stringify({url:'http://127.0.0.1:'+port+'/',directory,read_only:true})));
