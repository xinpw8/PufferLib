'use strict';
// Change the shared password: node set_password.cjs CONFIG.json [--logout-all]
// Reads the new password from stdin so it never appears in shell history or ps.
// --logout-all also rotates the cookie secret. Restart the server afterwards.
const fs=require('node:fs');
const crypto=require('node:crypto');
const readline=require('node:readline');
const {hashPassword}=require('./auth.cjs');

async function main(){
  const [configPath,flag]=process.argv.slice(2);
  if(!configPath||flag&&flag!=='--logout-all')throw Error('Usage: node set_password.cjs CONFIG.json [--logout-all]');
  const config=JSON.parse(fs.readFileSync(configPath,'utf8'));
  process.stderr.write('New password (at least 8 characters): ');
  const password=(await new Promise(resolve=>readline.createInterface({input:process.stdin}).once('line',resolve))).trim();
  config.passwordHash=hashPassword(password);
  if(flag==='--logout-all')config.secret=crypto.randomBytes(32).toString('hex');
  const tmp=configPath+'.tmp';fs.writeFileSync(tmp,JSON.stringify(config,null,2)+'\n',{mode:0o600});fs.renameSync(tmp,configPath);
  process.stderr.write(`\nSaved.${flag?' All sessions signed out.':''} Restart the server to apply.\n`);
  process.exit(0);
}
main().catch(error=>{process.stderr.write(error.message+'\n');process.exit(1);});
