'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const here=__dirname,source=fs.readFileSync(path.join(here,'source/relaunch.sh'));
const parent=fs.readFileSync(path.join(here,'../persistent-private-session-20260924-r1/launcher/relaunch.sh'));
const receipt=JSON.parse(fs.readFileSync(path.join(here,'receipts/runtime-result.json')));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
test('exact baseline and CALLRET0 launcher pins',()=>{
 assert.equal(sha(parent),'41d685616f90c1f78e70a744ee7e7059537ce33d2085cc4da87b96f8163483d7');
 assert.equal(sha(source),'452876dff3e6f29284bce16c0ea7c53eedd558dfb743b6fda8c47e7623bfc400');
});
test('only one CALLRET assignment differs from pinned parent',()=>{
 const lines=source.toString().split(/(?<=\n)/),added=lines.filter(l=>l.includes('BOX64_DYNAREC_CALLRET'));
 assert.equal(added.length,1);assert.match(added[0],/BOX64_DYNAREC_CALLRET=0/);
 assert.equal(lines.filter(l=>!l.includes('BOX64_DYNAREC_CALLRET')).join(''),parent.toString());
});
test('both completed outcomes have real zero-score starts and native terminal matches',()=>{
 for(const a of [receipt.attempts[0],receipt.attempts[4]]){
  assert.deepEqual(a.first_encoder_source.round.clean_hits,[0,0]);assert(a.first_encoder_source.round.time_remaining>=117);
  assert.deepEqual(a.final_round,a.native_terminal.round);assert.equal(a.final_round.duration,120);
  assert.equal(a.final_round.active,false);assert.equal(a.final_round.time_remaining,0);assert.equal(a.final_round.redo,false);
  assert.equal(a.media.delivery_status,'validated_mp4');assert(a.media.video_bytes<20000000);
  assert.equal(a.media.driver_result.code,0);assert(a.endpoints_closed.stream_stopped&&a.endpoints_closed.lease_released);
  assert(a.endpoints_closed.endpoints.every(e=>e.code===0&&e.close_observed&&e.signals_sent.length===0));
 }
});
test('redo, crash and unsupported-pair exclusions are preserved',()=>{
 const [,redo,crash,mixed]=receipt.attempts;
 assert.equal(redo.predictions,0);assert.equal(redo.observed_redo.round.duration,30);assert.equal(redo.observed_redo.round.redo,true);
 assert.equal(crash.final_round.active,true);assert.equal(crash.final_round.time_remaining,85.6994);assert.deepEqual(crash.final_round.clean_hits,[5,3]);
 assert.equal(mixed.predictions,0);assert.match(mixed.stop_reason,/unsupported_pairing/);
});
test('recovered final result does not rewrite original four-label plan or ledger',()=>{
 assert.equal(receipt.planned_round_count,4);assert.equal(receipt.original_ledger_completed,1);assert.equal(receipt.original_ledger_rewritten,false);
 assert.equal(receipt.retirement.summary_sha256,receipt.attempts[4].summary_source.sha256);
 assert.equal(receipt.retirement.wrapper_exit_code,0);assert.equal(receipt.retirement.wrapper_state,'Z');
 assert.equal(receipt.retirement.game_signalled,false);assert.equal(receipt.retirement.stale_lock_preserved,true);
 assert.deepEqual(receipt.totals,{complete:2,wins:1,ties:1,losses:0,points:[23,21],incomplete:3});
});
test('same policy/schema/mask and no policy acceptance claim',()=>{
 for(const a of receipt.attempts){assert.equal(a.checkpoint_sha256,'f147bdc358261e272c58e70167cd8b0b9e4953806c417d8891d94a3ad083ac84');
  assert.equal(a.observation_schema,'rek.native5.scaled_polar_xy.balance8_v1');assert.equal(a.feature_mask_sha256,'59158bfdf9ddb9a38686f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b');}
 assert.equal(receipt.policy_acceptance_claim,false);assert.equal(receipt.native_score_packet_audit,false);
});
