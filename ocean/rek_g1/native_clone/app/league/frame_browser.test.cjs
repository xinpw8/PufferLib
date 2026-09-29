'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function browser(hidden=false){
  const elements=new Map(),windowEvents=new Map(),documentEvents=new Map(),timers=[],requests=[],apiRequests=[],images=[],revoked=[];
  const element=()=>({hidden:false,textContent:'',listeners:new Map(),append(){},addEventListener(type,fn){this.listeners.set(type,fn);},focus(){}});
  const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
  const document={hidden,activeElement:get('arena'),getElementById:get,createElement:element,
    addEventListener:(type,fn)=>documentEvents.set(type,fn)};
  const window={addEventListener:(type,fn)=>windowEvents.set(type,fn)};
  const state={ok:true,paused:true,tick:0,phase:2,score:[0,0],active:{humanSide:0},pace:{},session:{}};
  const context=vm.createContext({document,window,performance,Date,AbortSignal,AbortController,
    URL:{createObjectURL:value=>{images.push(value);return 'blob:'+images.length;},revokeObjectURL:value=>revoked.push(value)},
    setTimeout:(fn,ms)=>{timers.push({fn,ms});return timers.length;},
    fetch:(url,options={})=>{
      if(url!=='/frame.png'){apiRequests.push(url);return Promise.resolve({ok:true,json:async()=>url==='/api/state'?state:{paused:true}});}
      return new Promise((resolve,reject)=>{
        const job={url,options,resolve,reject};requests.push(job);
        options.signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true});
      });
    }});
  for(const file of ['controls.js','state_labels.js','app.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'public',file),'utf8'),context,{filename:file});
  function wake(ms){const index=timers.findIndex(t=>t.ms===ms);assert(index>=0,'scheduled timer '+ms);timers.splice(index,1)[0].fn();}
  return {requests,apiRequests,images,revoked,get,document,wake,state,
    visibility:value=>{document.hidden=value;documentEvents.get('visibilitychange')();},
    close:()=>windowEvents.get('pagehide')()};
}
const response=(status,etag,blob)=>({ok:status===200,status,headers:{get:name=>name==='etag'?etag:null},blob});
test('actual browser loop sends last ETag and preserves the displayed image on a bodyless304',async()=>{
  const f=browser();try{
    await flush();assert.equal(f.requests.length,1);assert.equal(f.requests[0].options.headers['If-None-Match'],undefined);
    f.requests[0].resolve(response(200,'"generation1-tick0"',async()=>({id:'first'})));await flush();
    assert.equal(f.images.length,1);assert.equal(f.get('frame').src,'blob:1');
    f.wake(50);await flush();assert.equal(f.requests[1].options.headers['If-None-Match'],'"generation1-tick0"');
    let readBody=false;f.requests[1].resolve(response(304,'"generation1-tick0"',async()=>{readBody=true;throw Error('304 has no body');}));await flush();
    assert.equal(readBody,false);assert.equal(f.images.length,1);assert.equal(f.revoked.length,0);
    f.wake(50);await flush();f.requests[2].resolve(response(200,'"generation1-tick2"',async()=>({id:'second'})));await flush();
    assert.equal(f.images.length,2);assert.deepEqual(f.revoked,['blob:1']);assert.equal(f.get('frame').src,'blob:2');
  }finally{f.close();await flush();}
});
test('hidden browser never requests frames, aborts in-flight image and resumes with its last displayed ETag',async()=>{
  const f=browser(true);try{
    await flush();assert.equal(f.requests.length,0);
    for(let i=0;i<4;i++){f.wake(250);await flush();}assert.equal(f.requests.length,0);
    const polls=f.apiRequests.filter(x=>x==='/api/state').length;f.wake(100);await flush();
    assert.equal(f.apiRequests.filter(x=>x==='/api/state').length,polls+1,'hidden state heartbeat is unchanged');
    f.visibility(false);f.wake(250);await flush();assert.equal(f.requests.length,1);
    f.requests[0].resolve(response(200,'"one"',async()=>({id:'first'})));await flush();f.wake(50);await flush();
    assert.equal(f.requests.length,2);f.visibility(true);await flush();assert.equal(f.requests[1].options.signal.aborted,true);
    f.wake(50);await flush();for(let i=0;i<4;i++){f.wake(250);await flush();}assert.equal(f.requests.length,2);
    f.visibility(false);f.wake(250);await flush();assert.equal(f.requests.length,3);
    assert.equal(f.requests[2].options.headers['If-None-Match'],'"one"');
  }finally{f.close();await flush();assert.equal(f.requests.at(-1).options.signal.aborted,true);}
});
test('image body finishing after visibility change cannot replace the displayed snapshot',async()=>{
  const f=browser();try{
    await flush();let finishBody;
    f.requests[0].resolve(response(200,'"late"',()=>new Promise(resolve=>finishBody=resolve)));await flush();
    f.visibility(true);finishBody({id:'late'});await flush();assert.equal(f.images.length,0);
    f.wake(50);await flush();f.visibility(false);f.wake(250);await flush();
    assert.equal(f.requests[1].options.headers['If-None-Match'],undefined,'unpublished image never advances validator');
  }finally{f.close();await flush();}
});
test('browser distinguishes recent active pace from the cumulative session pace',async()=>{
  const f=browser(true);try{
    await flush();f.state.pace={realTimeRatio:.89,recent:{realTimeRatio:.423,activeWallMs:5010}};
    f.wake(100);await flush();assert.equal(f.get('pace').textContent,'0.42× recent (5.0 s active) · 0.89× session');
  }finally{f.close();await flush();}
});
