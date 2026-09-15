'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {HUMAN_ROUND_SECONDS,DEFAULT_HUMAN_ROUND_SECONDS,createHumanConfig,HumanSession}=require('./human_session.cjs');

function configFixture(t,roundSeconds=20){
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'rek-human-test-'));
  const filename=path.join(directory,'original.json');
  const original={backend:'semantic_cuda',model_path:path.join(directory,'model.xml'),
    assets_path:path.join(directory,'assets'),motion_features_path:path.join(directory,'features'),
    round_seconds:roundSeconds,arenas:4,seed:73,move_duration_ticks:[35,27,31],fast:{move_speed:1}};
  const bytes=JSON.stringify(original)+'\n';fs.writeFileSync(filename,bytes);
  t.after(()=>{fs.unlinkSync(filename);fs.rmdirSync(directory);});
  return {directory,filename,original,bytes};
}
function terminal(overrides={}){
  return {terminal:1,roundNumber:1,tick:1000,score:[8,3],roundResult:1,winner:0,...overrides};
}

test('human durations are explicit and default to 300 seconds',()=>{
  assert.deepEqual(HUMAN_ROUND_SECONDS,[20,120,300]);assert.equal(DEFAULT_HUMAN_ROUND_SECONDS,300);
  assert.equal(Object.isFrozen(HUMAN_ROUND_SECONDS),true);
});

test('human config changes only round_seconds and never modifies original assets or config',t=>{
  const f=configFixture(t);
  for(const duration of HUMAN_ROUND_SECONDS){
    const config=createHumanConfig(f.filename,duration);
    try{
      assert.notEqual(config.path,f.filename);assert.equal(path.isAbsolute(config.path),true);
      assert.match(path.basename(path.dirname(config.path)),/^rek-human-/);
      assert.equal(config.roundSeconds,duration);assert.equal(config.trainingRoundSeconds,20);
      assert.deepEqual(JSON.parse(fs.readFileSync(config.path,'utf8')),{...f.original,round_seconds:duration});
      assert.equal(fs.readFileSync(f.filename,'utf8'),f.bytes);
      if(process.platform!=='win32')assert.equal(fs.statSync(config.path).mode&0o777,0o600);
    }finally{config.close();config.close();}
    assert.equal(fs.existsSync(config.path),false);assert.equal(fs.existsSync(path.dirname(config.path)),false);
  }
  const defaultConfig=createHumanConfig(f.filename);
  try{assert.equal(defaultConfig.roundSeconds,300);}finally{defaultConfig.close();}
});

test('human config reports legacy training duration default without rewriting source',t=>{
  const f=configFixture(t,0),config=createHumanConfig(f.filename,120);
  try{assert.equal(config.trainingRoundSeconds,120);assert.equal(fs.readFileSync(f.filename,'utf8'),f.bytes);}
  finally{config.close();}
});

test('invalid durations and malformed configs are rejected',t=>{
  const f=configFixture(t);
  for(const duration of [null,'300',0,-1,20.5,60,301,NaN,Infinity])
    assert.throws(()=>createHumanConfig(f.filename,duration),/duration/);
  assert.equal(fs.readFileSync(f.filename,'utf8'),f.bytes);
  for(const value of ['null','[]','5','"config"']){
    fs.writeFileSync(f.filename,value);assert.throws(()=>createHumanConfig(f.filename),/object/);
  }
});

test('human config cleanup never recursively removes an unexpected sibling',t=>{
  const f=configFixture(t),config=createHumanConfig(f.filename);
  const sibling=path.join(path.dirname(config.path),'preserve.txt');fs.writeFileSync(sibling,'preserve');
  assert.throws(()=>config.close());assert.equal(fs.readFileSync(sibling,'utf8'),'preserve');
  assert.equal(fs.existsSync(config.path),false);assert.equal(fs.readFileSync(f.filename,'utf8'),f.bytes);
  fs.unlinkSync(sibling);config.close();assert.equal(fs.existsSync(path.dirname(config.path)),false);
});

test('session records native points winners and deduplicates exact terminal snapshots',()=>{
  const session=new HumanSession(),result=terminal();
  assert.equal(session.record({terminal:0}),false);assert.equal(session.record(result),true);
  assert.equal(session.record({...result}),false);
  assert.deepEqual(session.snapshot(),{completedRounds:1,bluePoints:8,orangePoints:3,
    blueWins:1,orangeWins:0,draws:0,invalidRounds:0,
    lastRound:{bluePoints:8,orangePoints:3,winner:0,reason:'points'}});
});

test('KO winner remains authoritative even with fewer points',()=>{
  const session=new HumanSession();
  session.record(terminal({score:[12,5],roundResult:2,winner:1}));
  const result=session.snapshot();assert.equal(result.blueWins,0);assert.equal(result.orangeWins,1);
  assert.deepEqual(result.lastRound,{bluePoints:12,orangePoints:5,winner:1,reason:'knockout'});
});

test('draws count as valid rounds while replay and unknown results are invalid',()=>{
  const session=new HumanSession();
  session.record(terminal({roundResult:3,winner:-1,score:[4,4]}));
  session.record(terminal({roundNumber:2,tick:2000,roundResult:4,winner:-1,score:[100,200]}));
  assert.deepEqual(session.snapshot().lastRound,{bluePoints:100,orangePoints:200,winner:null,reason:'replay'});
  session.record(terminal({roundNumber:3,tick:3000,roundResult:99,winner:0,score:[300,400]}));
  const result=session.snapshot();
  assert.equal(result.completedRounds,1);assert.equal(result.draws,1);assert.equal(result.invalidRounds,2);
  assert.equal(result.blueWins,0);assert.equal(result.orangeWins,0);
  assert.equal(result.bluePoints,4);assert.equal(result.orangePoints,4);
  assert.deepEqual(result.lastRound,{bluePoints:300,orangePoints:400,winner:null,reason:'unknown'});
});

test('missing valid native winner does not become a score-derived win or draw',()=>{
  const session=new HumanSession();
  session.record(terminal({winner:-1}));
  session.record(terminal({roundNumber:2,tick:2000,roundResult:2,winner:null}));
  assert.equal(session.snapshot().completedRounds,0);assert.equal(session.snapshot().invalidRounds,2);
  assert.equal(session.snapshot().draws,0);assert.equal(session.snapshot().bluePoints,0);
});

test('new native round streams preserve scoreboard while reset clears the session',()=>{
  const session=new HumanSession(),first=terminal();session.record(first);
  const before=session.snapshot();session.newRoundStream();assert.deepEqual(session.snapshot(),before);
  assert.equal(session.record(first),true,'same native round/tick can belong to a new worker generation');
  assert.equal(session.snapshot().completedRounds,2);assert.equal(session.snapshot().bluePoints,16);
  assert.equal(session.record(first),false);
  session.reset();assert.deepEqual(session.snapshot(),{completedRounds:0,bluePoints:0,orangePoints:0,
    blueWins:0,orangeWins:0,draws:0,invalidRounds:0,lastRound:null});
  assert.equal(session.record(first),true);
});

test('snapshot is detached and malformed terminal states do not mutate session',()=>{
  const session=new HumanSession();session.record(terminal());const before=session.snapshot();
  const copied=session.snapshot();copied.bluePoints=999;copied.lastRound.winner=1;
  assert.deepEqual(session.snapshot(),before);
  for(const invalid of [null,[],{terminal:'1'},terminal({roundNumber:0}),terminal({tick:NaN}),
    terminal({score:[1]}),terminal({score:[1,-1]}),terminal({score:[1,2.5]})]){
    assert.throws(()=>session.record(invalid));assert.deepEqual(session.snapshot(),before);
  }
  const good=terminal({roundNumber:2,tick:2000});assert.equal(session.record(good),true);
});
