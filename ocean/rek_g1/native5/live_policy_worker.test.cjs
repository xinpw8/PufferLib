'use strict';
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const readline = require('node:readline');

async function main() {
  const [mode, binary, checkpoint, sha, selection='sampled'] = process.argv.slice(2);
  const observationSchema=process.env.REK_OBSERVATION_SCHEMA??'rek.native5.scaled_polar_xy.v1';
  assert(['protocol', 'gpu'].includes(mode));
  assert(binary);
  if (mode === 'gpu') assert(checkpoint && /^[0-9a-f]{64}$/.test(sha));
  const child = spawn(binary, mode === 'gpu' ? [checkpoint, sha, '73',selection] : [], {stdio:['pipe','pipe','pipe']});
  let stderr = '', exited = false;
  const exit = new Promise((resolve,reject) => { child.on('error',reject); child.on('exit',(code,signal)=>{exited=true;resolve({code,signal});}); });
  child.stderr.on('data', x => { stderr += x; });
  const pending = [], received = [], transcript = [];
  readline.createInterface({input:child.stdout}).on('line',line => {
    const item = JSON.parse(line);
    transcript.push(item);
    if (pending.length) pending.shift()(item); else received.push(item);
  });
  const next = () => new Promise((resolve,reject) => {
    if (received.length) return resolve(received.shift());
    const timer = setTimeout(() => reject(new Error('worker response timeout')),120000);
    pending.push(x => { clearTimeout(timer); resolve(x); });
  });
  const request = async x => { child.stdin.write(typeof x === 'string' ? x+'\n' : JSON.stringify(x)+'\n'); return next(); };
  let assertions = 0, seq = 0;
  const check = (condition,message) => { assert(condition,message); assertions++; };
  const roundA = 'a'.repeat(64), roundB = 'b'.repeat(64);
  const step = (n=seq+1, round=roundA) => ({type:'step',seq:n,round_id:round,
    observation_schema:observationSchema,observation:Array(223).fill(0),
    mask:Array.from({length:33},(_,i)=>i===1),terminal:false});
  const bad = async (x, code) => {
    const result = await request(x); check(result.type==='error',code+' not rejected');
    check(result.code===code,`expected ${code}, got ${result.code}`);
    check(!Object.hasOwn(result,'action') && result.action_available===false,code+' emitted action');
  };
  try {
    const ready = await next();
    check(ready.type===(mode==='gpu'?'ready':'protocol_ready'),'startup type');
    if (mode==='gpu') {
      check(ready.checkpoint_sha256===sha,'exact checkpoint');
      check(ready.native_cuda===true && ready.environment_stepping===false,'GPU only, no environment');
      check(ready.precision==='bf16' && ready.selection===selection,'inference mode');
      check(ready.feature_mask_sha256==='','default observation features retained');
      check(ready.observation_schema===observationSchema,'exact observation schema');
    } else check(ready.inference_available===false,'CPU parser cannot infer');
    const verifyAction = (r, expected) => {
      check(r.type===(mode==='gpu'?'action':'validated'),'step response');
      check(r.seq===seq,'sequence echo');
      if(mode==='gpu') {
        check(r.action===expected,'one-hot mask action');
        check(r.checkpoint_sha256===sha,'action checkpoint identity');
        check(Number.isFinite(r.gpu_ms)&&r.gpu_ms>=0,'GPU timing');
      } else check(!Object.hasOwn(r,'action'),'parser never emits action');
      check(Number.isFinite(r.latency_ms)&&r.latency_ms>=0,'latency');
    };
    let r = await request(step(++seq)); verifyAction(r,1);
    check(r.recurrent_reset && r.round_changed && r.round_id===roundA,'first round reset');
    r = await request(step(++seq)); verifyAction(r,1); check(!r.recurrent_reset&&!r.round_changed,'same round retained');
    await bad(step(seq),'stale_seq');
    const cases = [
      [x=>x.observation.pop(),'observation_shape'],
      [x=>x.observation.push(0),'observation_shape'],
      [x=>{x.observation[3]=null;},'observation_value'],
      [x=>{x.observation[3]='0';},'observation_value'],
      [x=>{x.observation[3]=4e38;},'observation_value'],
      [x=>x.mask.pop(),'mask_shape'],
      [x=>{x.mask[0]=2;},'mask_value'],
      [x=>{x.mask[0]='1';},'mask_value'],
      [x=>{x.mask.fill(0);},'empty_action_mask'],
      [x=>{x.terminal=1;},'terminal_boolean_required'],
      [x=>{x.round_id='bad';},'invalid_round_id'],
      [x=>{x.round_id='A'.repeat(64);},'invalid_round_id'],
      [x=>{x.observation_schema='unknown';},'observation_schema_mismatch'],
      [x=>{delete x.observation;},'missing_field'],
      [x=>{x.extra=true;},'unsupported_field'],
      [x=>{x.seq=-1;},'invalid_seq'],
      [x=>{x.seq=0.5;},'invalid_seq'],
      [x=>{x.seq=2**53;},'invalid_seq'],
      [x=>{x.type='unknown';},'unknown_type'],
    ];
    for(const [mutate,code] of cases) { const x=step();mutate(x);await bad(x,code); }
    const crossSchema=step();crossSchema.observation_schema=observationSchema==='rek.native5.scaled_polar_xy.v1'?'rek.native5.scaled_polar_xy.owned_yaw_v2':'rek.native5.scaled_polar_xy.v1';
    await bad(crossSchema,'observation_schema_mismatch');
    if(observationSchema==='rek.native5.scaled_polar_xy.owned_yaw_v2'){
      for(const yaw of [-1,0,1]){const input=step(++seq);input.observation[187]=yaw;const accepted=await request(input);verifyAction(accepted,1);}
      const unknown=step();unknown.observation[187]=.5;await bad(unknown,'owned_yaw_intent_value');
      const badTerminal=step();badTerminal.terminal=true;badTerminal.observation[187]=1;await bad(badTerminal,'owned_yaw_intent_value');
    }
    if(observationSchema==='rek.native5.observable_balance.v1'){
      for(const [index,value,code] of [[187,1,'observable_balance_padding'],[71,.5,'observable_balance_availability'],
        [72,1.1,'observable_balance_tilt'],[13,.5,'observable_balance_missing_joints'],
        [42,.5,'observable_balance_missing_rates'],[204,1,'observable_balance_missing_referee']]){
        const input=step();input.observation[index]=value;await bad(input,code);
      }
      const measured=step(++seq);measured.observation[72]=.75;measured.observation[158]=.25;
      measured.observation[202]=1;measured.observation[204]=1;
      verifyAction(await request(measured),1);
    }
    await bad('{broken','invalid_json_object');
    await bad('[]','invalid_json_object');
    await bad(JSON.stringify(step()).replace('"observation":[0,','"observation":[1e999,'),'observation_value');
    await bad(JSON.stringify(step()).replace('"type":"step"','"type":"step","type":"step"'),'duplicate_field');
    await bad(JSON.stringify(step())+'\u0000','embedded_nul');
    await bad(' '.repeat(65537),'line_too_long');
    const forced = step(++seq); forced.mask.fill(0);forced.mask[17]=1;
    r=await request(forced);verifyAction(r,17);check(!r.recurrent_reset,'errors do not reset state or consume seq');
    r=await request({type:'reset',seq:++seq,round_id:roundA});
    check(r.type==='reset' && r.recurrent_reset && !Object.hasOwn(r,'action'),'explicit reset is action-free');
    r=await request(step(++seq));verifyAction(r,1);check(r.recurrent_reset&&!r.round_changed,'next action acknowledges reset');
    const terminal=step(++seq);terminal.terminal=true;terminal.mask.fill(0);
    r=await request(terminal);check(r.type==='terminal'&&r.recurrent_reset&&!Object.hasOwn(r,'action'),'terminal is action-free');
    r=await request(step(++seq,roundB));verifyAction(r,1);check(r.recurrent_reset&&r.round_changed&&r.round_id===roundB,'new measured round resets');
    r=await request({type:'close',seq:++seq});check(r.type==='closed'&&!Object.hasOwn(r,'action'),'close');
    child.stdin.end();
    const result=await exit;check(result.code===0&&!result.signal,'clean exit');
    const timings=transcript.filter(x=>x.type==='action').map(x=>x.latency_ms).sort((a,b)=>a-b);
    console.log(JSON.stringify({test:'live-policy-worker',mode,selection,synthetic_observations:true,authentic_game_control:false,
      assertions,responses:transcript.length,actions:timings.length,checkpoint_sha256:mode==='gpu'?sha:null,
      latency_ms:timings.length?{min:timings[0],median:timings[Math.floor(timings.length/2)],max:timings.at(-1)}:null,
      exit_code:result.code,stderr,transcript},null,2));
  } finally { if(!exited) child.kill('SIGTERM'); }
}
main().catch(e=>{console.error(e.stack||e);process.exitCode=1;});
