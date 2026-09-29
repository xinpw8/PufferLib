'use strict';
// Writes a private server config. A new random password is printed once and
// saved to --password-file (mode 0600); only its scrypt hash enters the config.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {hashPassword}=require('./auth.cjs');

function args(argv){
  const out={};for(let i=0;i<argv.length;i+=2){if(!argv[i].startsWith('--'))throw Error('Expected --name value pairs');out[argv[i].slice(2)]=argv[i+1];}
  for(const key of ['out','password-file','binary','worker-config','server-json','scene-dir','log-dir','port'])
    if(!out[key])throw Error(`--${key} is required`);
  return out;
}
function main(){
  const a=args(process.argv.slice(2));
  if(fs.existsSync(a.out))throw Error('Config already exists; remove it deliberately to rotate the password');
  const password=a.password||crypto.randomBytes(9).toString('base64url');
  const server=JSON.parse(fs.readFileSync(a['server-json'],'utf8'));
  const config={port:Number(a.port),host:'127.0.0.1',allowedHosts:(a.hosts||'').split(',').filter(Boolean),
    secret:crypto.randomBytes(32).toString('hex'),passwordHash:hashPassword(password),
    binary:path.resolve(a.binary),workerConfig:path.resolve(a['worker-config']),env:server.backends[0].env,
    roundSeconds:Number(a['round-seconds']||120),sceneDir:path.resolve(a['scene-dir']),logDir:path.resolve(a['log-dir'])};
  fs.writeFileSync(a.out,JSON.stringify(config,null,2)+'\n',{flag:'wx',mode:0o600});
  fs.writeFileSync(a['password-file'],password+'\n',{flag:'wx',mode:0o600});
  console.log(JSON.stringify({config:path.resolve(a.out),passwordFile:path.resolve(a['password-file']),port:config.port,allowedHosts:config.allowedHosts}));
}
main();
