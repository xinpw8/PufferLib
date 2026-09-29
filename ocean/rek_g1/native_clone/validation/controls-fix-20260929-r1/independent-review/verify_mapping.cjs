'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const fixture=JSON.parse(fs.readFileSync(path.join(__dirname,'EXPECTED-MOVES.json'),'utf8'));
const corrected=process.argv[2]||'C:/rekagent/work/rek-controls-fix-20260929-r1/app/league/input.cjs';
const previous=process.argv[3]||'C:/rekagent/work/rek-playback-app-20260928-r7/app/league/input.cjs';
function inspect(source){
  const {HumanInput}=require(source),mismatches=[];
  for(const row of fixture.bindings){
    const input=new HumanInput();input.update({seq:1,held:[],move:row.category});
    const actual=input.next().moveIndex;
    if(actual!==row.nativeMoveIndex)mismatches.push({commandId:row.commandId,category:row.category,expected:row.nativeMoveIndex,actual});
    assert.equal(input.next().moveIndex,-1,row.commandId+' one-shot');
  }
  return {source,bindingsChecked:fixture.bindings.length,mismatches};
}
const old=inspect(previous),fixed=inspect(corrected);
assert.equal(old.mismatches.length,10,'The unmodified old input boundary must reproduce all ten independently expected failures');
assert.equal(fixed.mismatches.length,0,'Every corrected boundary must match the independently named original clip');
console.log(JSON.stringify({schema:'rek.controls.independent_regression.v1',pass:true,old,fixed},null,2));
