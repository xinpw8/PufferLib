'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const net=require('node:net');
const {once}=require('node:events');
const {serve}=require('./server.cjs');

test('isolated HTTP endpoint accepts all held chords and rejects malformed controls',async()=>{
  // Empty backend list and no initial selection guarantee no native worker,
  // simulator, GPU job or live match is touched. Use a private ephemeral port.
  const reservation=net.createServer();reservation.listen(0,'127.0.0.1');
  await once(reservation,'listening');const port=reservation.address().port;
  await new Promise(resolve=>reservation.close(resolve));
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-input-http-'));
  const configPath=path.join(directory,'config.json');
  fs.writeFileSync(configPath,JSON.stringify({port,backends:[],leagueFile:path.join(directory,'league.json')}));
  let instance;
  try{
    instance=await serve(configPath);
    if(!instance.server.listening)await once(instance.server,'listening');
    const origin=`http://127.0.0.1:${port}`;
    const post=async packet=>{
      const response=await fetch(`${origin}/api/input`,{method:'POST',
        headers:{'Content-Type':'application/json',Origin:origin},body:JSON.stringify(packet)});
      return {status:response.status,body:await response.json()};
    };
    const keys=['W','S','A','D','Q','E'];let seq=0;
    for(let bits=0;bits<64;bits++){
      const held=keys.filter((_,index)=>bits&(1<<index));
      for(const move of [null,17,18]){
        const response=await post({seq:++seq,held,move});
        assert.equal(response.status,200,`${held.join('+')}: ${JSON.stringify(response.body)}`);
        assert.deepEqual(response.body,{ok:true,accepted:true});
      }
    }
    assert.deepEqual(await post({seq:1,held:[]}),{status:200,body:{ok:true,accepted:false}});
    for(const packet of [{seq:++seq,held:['F']},{seq:++seq,held:'W'},
      {seq:++seq,held:[],move:33},{seq:++seq,held:[],move:17.5}]){
      const response=await post(packet);assert.equal(response.status,400);
      assert.equal(typeof response.body.error,'string');
    }
    assert.equal(fs.existsSync(path.join(directory,'league.json')),false,'no league mutation');
  }finally{
    if(instance){const closed=once(instance.server,'close');await instance.close();await closed;}
    fs.unlinkSync(configPath);fs.rmdirSync(directory);
  }
});
