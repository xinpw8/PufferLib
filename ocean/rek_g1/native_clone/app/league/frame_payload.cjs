'use strict';
const crypto=require('node:crypto');
const cache=new WeakMap();

// Recorder and server receive the same reply object. A weak key shares one
// decode/hash without retaining completed replies or changing wire JSON.
function framePayload(reply){
  if(typeof reply?.png!=='string')throw Error('Frame PNG string required');
  const prior=cache.get(reply);
  if(prior&&prior.encoded===reply.png)return prior.value;
  const png=Buffer.from(reply.png,'base64');
  const value={png,sha256:crypto.createHash('sha256').update(png).digest('hex')};
  cache.set(reply,{encoded:reply.png,value});
  return value;
}
module.exports={framePayload};
