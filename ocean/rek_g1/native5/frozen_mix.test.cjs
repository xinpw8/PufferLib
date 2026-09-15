'use strict';
// Host-only compilation of the exact startup helpers from puffer_env.cu.
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'puffer_env.cu'),'utf8');
const helpers=source.split('// BEGIN REK_FROZEN_MIX_HOST_HELPERS')[1]?.split('// END REK_FROZEN_MIX_HOST_HELPERS')[0];
assert(helpers,'Exact production helpers must exist');
const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-frozen-mix-'));
const input=path.join(directory,'test.cpp'),binary=path.join(directory,'test');
const test=`
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <assert.h>
#include <stdio.h>
${helpers}
int main(){
  double f=-1;
  assert(rek_native5_parse_frozen_fraction(NULL,&f)&&f==1);
  for(const char* s: {"0","0.25","1","1e-12"})assert(rek_native5_parse_frozen_fraction(s,&f));
  for(const char* s: {"","-0.1","1.1","NaN","inf","0.2junk","1e-999"})assert(!rek_native5_parse_frozen_fraction(s,&f));
  uint8_t rows[1024],copy[1024],changed[1024];
  for(int n: {1,2,7,512})for(double fraction: {0.,1e-12,.25,.5,.99,1.}){
    memset(rows,7,sizeof(rows));
    int count=rek_native5_frozen_override_rows(rows,n,73,fraction),observed=0;
    int expected=fraction==1?n:(int)floor(n*fraction);if(fraction>0&&!expected)expected=1;
    assert(count==expected);
    for(int a=0;a<n;a++){assert(rows[2*a]==0);assert(rows[2*a+1]<=1);observed+=rows[2*a+1];}
    assert(observed==count);
    assert(rek_native5_frozen_override_rows(copy,n,73,fraction)==count);
    assert(memcmp(rows,copy,n*2)==0);
  }
  assert(rek_native5_frozen_override_rows(rows,512,73,.5)==256);
  assert(rek_native5_frozen_override_rows(changed,512,74,.5)==256);
  assert(memcmp(rows,changed,sizeof(rows))!=0);
  assert(rek_native5_frozen_override_rows(rows,0,73,.5)==-1);
  assert(rek_native5_frozen_override_rows(NULL,1,73,.5)==-1);
  assert(rek_native5_frozen_override_rows(rows,2,73,1.1)==-1);
  assert(rek_native5_frozen_override_rows(rows,2,73,NAN)==-1);
  puts("frozen mix helpers passed: exact counts, boundaries, tiny batches, seed replay, learner preservation");
}
`;
try{
  fs.writeFileSync(input,'#include <initializer_list>\n'+test);
  const build=spawnSync(process.env.CXX||'g++',['-std=c++17','-Wall','-Wextra','-Werror',input,'-o',binary],{encoding:'utf8'});
  assert.equal(build.status,0,build.error?.message||build.stderr);
  const run=spawnSync(binary,[],{encoding:'utf8'});assert.equal(run.status,0,run.stderr);process.stdout.write(run.stdout);
}finally{
  for(const p of [input,binary])if(fs.existsSync(p))fs.unlinkSync(p);
  fs.rmdirSync(directory);
}
