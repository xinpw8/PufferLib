'use strict';
const assert = require('node:assert/strict');
const {initialSource} = require('./selection.cjs');
const SCHEMA = 'rek.controlled_policy_coverage.v1';
const INPUTS = ['trial/summary.json', 'contact-analysis/summary.json', 'trial/relay.stdout.jsonl', 'trial/worker.stdin.jsonl'];
const near = (a, b) => assert(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-9, 'coverage clocks disagree');
function compactSource(x) {
  return {observation_sequence:x.observation_sequence, round_identity_sha256:x.round_identity_sha256,
    local_slot:x.local_slot, stream_active:x.stream_active, global_input_emitted:x.global_input_emitted,
    clock:x.clock, round:x.round};
}
function deriveCoverage(summary, contact, sources, acks, firstDecisionSeq) {
  assert.equal(summary.controlled_startup_validated, true, 'explicit controlled startup required');
  assert(sources.length > 0);
  const first = sources[0], controlled = sources.find(x => x.stream_active === true);
  const readiness = sources.find(x => x.observation_sequence === summary.startup_readiness?.observation_sequence);
  const start = initialSource(summary, first, controlled, readiness);
  assert.equal(contact.round_identity_sha256, first.round_identity_sha256);
  assert.equal(contact.local_slot, first.local_slot);
  assert.equal(contact.checkpoint_sha256, summary.checkpoint_sha256);
  assert.equal(contact.terminal_evidence_consistent, true);
  assert.equal(contact.terminal_point_counters_consistent, true);
  assert.equal(contact.native_capture_complete, true);
  assert.equal(contact.initial_observation_within_first_second, true);
  assert(['win','loss','draw'].includes(contact.outcome));
  assert(typeof contact.completed_policy_round === 'boolean');
  const decision = sources.find(x => x.observation_sequence === firstDecisionSeq);
  assert(decision && decision.stream_active === true && decision.observation_sequence >= start.observation_sequence,
    'first decision precedes controlled source');
  const terminals = sources.filter(x => x.round.active === false);
  assert.equal(terminals.length, 1);
  const terminal = terminals[0];
  assert.equal(terminal, sources.at(-1));
  assert.deepEqual(terminal.round, summary.final_round);
  let previous = null;
  for (const x of sources) {
    assert.equal(x.round_identity_sha256, first.round_identity_sha256);
    assert.equal(x.local_slot, first.local_slot);
    assert.equal(x.round.number, first.round.number);
    assert.equal(x.round.redo, false);
    assert(Number.isFinite(x.clock.unity_time));
    assert(Number.isSafeInteger(x.clock.qpc_ticks));
    assert.equal(x.clock.qpc_frequency_hz, first.clock.qpc_frequency_hz);
    if (previous) {
      assert(x.observation_sequence > previous.observation_sequence && x.clock.qpc_ticks > previous.clock.qpc_ticks);
      assert(x.clock.unity_time >= previous.clock.unity_time);
      assert(x.round.clean_hits.every((v, i) => v >= previous.round.clean_hits[i]));
    }
    if (x.observation_sequence <= firstDecisionSeq) assert.deepEqual(x.round.clean_hits, [0,0], 'score precedes first decision');
    previous = x;
  }
  const applied = acks.filter(x => x.applied === true);
  assert(applied.length >= 2);
  for (const ack of applied) {
    assert.equal(ack.round_identity_sha256, first.round_identity_sha256);
    assert(ack.observation_sequence >= firstDecisionSeq, 'applied action precedes first decision');
    assert(Number.isFinite(ack.clock?.unity_time));
  }
  const clocks = applied.map(x => x.clock.unity_time).sort((a,b) => a-b);
  const maxGap = Math.max(...clocks.slice(1).map((x,i) => x-clocks[i]));
  const firstGap = clocks[0] - start.clock.unity_time;
  const terminalGap = terminal.clock.unity_time - clocks.at(-1);
  const old = contact.policy_control_coverage;
  assert.equal(old.applied_action_returns, applied.length);
  assert.equal(old.tolerance_seconds, 1);
  near(old.first_applied_seconds_after_first_observation, clocks[0] - first.clock.unity_time);
  near(old.maximum_applied_action_gap_seconds, maxGap);
  near(old.terminal_seconds_after_last_applied, terminalGap);
  assert(firstGap >= 0 && firstGap <= 1, 'first controlled source to first ACK exceeds original bound');
  assert(maxGap <= 1, 'applied ACK gap exceeds original bound');
  assert(terminalGap >= 0 && terminalGap <= 1, 'terminal ACK gap exceeds original bound');
  return {round_identity_sha256:first.round_identity_sha256, local_slot:first.local_slot,
    checkpoint_sha256:summary.checkpoint_sha256, legacy_completed_policy_round:contact.completed_policy_round,
    legacy_first_applied_seconds_after_first_observation:old.first_applied_seconds_after_first_observation,
    controlled_policy_interval_complete:true, native_round_duration_seconds:start.round.duration,
    first_raw_sequence:first.observation_sequence, readiness_sequence:readiness.observation_sequence,
    controlled_source_sequence:start.observation_sequence, first_decision_sequence:firstDecisionSeq,
    actual_control_start_time_remaining_seconds:start.round.time_remaining,
    native_round_elapsed_before_control_seconds:start.round.duration-start.round.time_remaining,
    raw_to_control_qpc_seconds:(start.clock.qpc_ticks-first.clock.qpc_ticks)/first.clock.qpc_frequency_hz,
    controlled_to_first_applied_seconds:firstGap, maximum_applied_action_gap_seconds:maxGap,
    terminal_seconds_after_last_applied:terminalGap, applied_action_returns:applied.length,
    tolerance_seconds:1, startup_observed_counters_zero_through_first_decision:true,
    basis:'Observed local applied acknowledgements: unchanged UnityTime interval bounds, QPC startup identity and reward discounts. Server acceptance unknown; native duration is not claimed as policy-control duration or a wall-clock coverage guarantee.'};
}
function receipt(coverage, inputs) {
  assert.equal(inputs.length, INPUTS.length);
  for (const name of INPUTS) {
    const p = inputs.find(x => x.file === name);
    assert(p && Number.isSafeInteger(p.bytes) && p.bytes > 0 && /^[a-f0-9]{64}$/.test(p.sha256), 'missing coverage input binding');
  }
  return {schema:SCHEMA, coverage, inputs};
}
function validateReceipt(value, coverage, inputs) {
  assert.equal(value.schema, SCHEMA);
  const expected = receipt(coverage, inputs);
  assert.deepEqual(value, expected, 'controlled coverage receipt or source binding changed');
  return value.coverage;
}
module.exports = {SCHEMA, compactSource, deriveCoverage, receipt, validateReceipt};
