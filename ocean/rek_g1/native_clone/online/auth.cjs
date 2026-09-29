'use strict';
const crypto=require('node:crypto');

const COOKIE='rek_auth';
const TOKEN_DAYS=7;

// scrypt$N$r$p$salt$hash, base64url fields.
function hashPassword(password,{N=16384,r=8,p=1}={}){
  if(typeof password!=='string'||password.length<8)throw Error('Password must be at least 8 characters');
  const salt=crypto.randomBytes(16);
  const hash=crypto.scryptSync(password,salt,32,{N,r,p});
  return ['scrypt',N,r,p,salt.toString('base64url'),hash.toString('base64url')].join('$');
}
function verifyPassword(password,stored){
  if(typeof password!=='string'||typeof stored!=='string')return false;
  const [kind,N,r,p,salt,hash]=stored.split('$');
  if(kind!=='scrypt'||!salt||!hash)return false;
  const expected=Buffer.from(hash,'base64url');
  const actual=crypto.scryptSync(password,Buffer.from(salt,'base64url'),expected.length,{N:+N,r:+r,p:+p});
  return crypto.timingSafeEqual(actual,expected);
}

function sign(secret,payload){return crypto.createHmac('sha256',secret).update(payload).digest('base64url');}
function issueToken(secret,name,now=Date.now()){
  const payload=[now+TOKEN_DAYS*86400000,crypto.randomBytes(9).toString('base64url'),Buffer.from(name).toString('base64url')].join('.');
  return payload+'.'+sign(secret,payload);
}
// Returns the player name for a valid, unexpired token, else null.
function verifyToken(secret,token,now=Date.now()){
  if(typeof token!=='string')return null;
  const parts=token.split('.');if(parts.length!==4)return null;
  const payload=parts.slice(0,3).join('.'),expected=Buffer.from(sign(secret,payload)),actual=Buffer.from(parts[3]);
  if(actual.length!==expected.length||!crypto.timingSafeEqual(actual,expected))return null;
  if(!(Number(parts[0])>now))return null;
  try{return Buffer.from(parts[2],'base64url').toString('utf8');}catch{return null;}
}
function readCookie(header,name=COOKIE){
  for(const part of String(header||'').split(';')){
    const index=part.indexOf('=');if(index<0)continue;
    if(part.slice(0,index).trim()===name)return decodeURIComponent(part.slice(index+1).trim());
  }
  return null;
}
function cleanName(value){
  const name=String(value||'').replace(/[^\p{L}\p{N} _.-]/gu,'').trim().slice(0,20);
  return name||'Player';
}

// Failed logins per client address; blocks after `limit` failures in `windowMs`.
class LoginLimiter{
  constructor({limit=8,windowMs=15*60000}={}){this.limit=limit;this.windowMs=windowMs;this.failures=new Map();}
  blocked(key,now=Date.now()){
    const list=(this.failures.get(key)||[]).filter(t=>now-t<this.windowMs);
    if(list.length)this.failures.set(key,list);else this.failures.delete(key);
    return list.length>=this.limit;
  }
  fail(key,now=Date.now()){const list=this.failures.get(key)||[];list.push(now);this.failures.set(key,list);}
  clear(key){this.failures.delete(key);}
}

module.exports={COOKIE,TOKEN_DAYS,hashPassword,verifyPassword,issueToken,verifyToken,readCookie,cleanName,LoginLimiter};
