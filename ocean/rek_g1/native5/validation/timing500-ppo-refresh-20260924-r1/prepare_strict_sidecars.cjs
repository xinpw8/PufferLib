'use strict';
// CPU-only derivation from explicitly selected completed captures. No game or bridge access.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const selectionApi=require('./selection.cjs');
const coverageApi=require('./controlled_policy_coverage.cjs');
const selection=selectionApi.loadSelection(process.argv[2]);
const nativeSelection=selectionApi.loadNativeSelection(selection);
const stage=selection.stage;
const nativeRoot = '/home/spark-advantage/codexrook-runtime/wineprefix/drive_c/rekagent/evidence/runtime/rek-private-ai-protocol-v7';
const moduleRoot = '/home/spark-advantage/rek-training/referee-bridge-spark-20260924-r1';
const sourceRoot = '/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/source/ocean/rek_g1/native5';
const referee = require(path.join(moduleRoot, 'validate_live_referee.cjs'));
const receipts = require(path.join(moduleRoot, 'native_hit_receipt_data.cjs'));
const contacts = require(path.join(sourceRoot, 'analyze_live_contacts.cjs'));
const selected=selection.rounds;
const checkpoint=selection.checkpoint.sha256;
const check = (condition, message) => { if (!condition) throw Error(message); };
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', {flag: 'wx', mode: 0o600});
const provenance = file => { const bytes = fs.readFileSync(file); return {file, bytes: bytes.length, sha256: sha(bytes)}; };

async function prepare(spec) {
  const label=spec.id;
  const original=path.join(spec.directory,'trial');
  const originalSummaryPath = path.join(original, 'summary.json');
  const originalBytes = fs.readFileSync(originalSummaryPath);
  const summary = JSON.parse(originalBytes);
  check(fs.readFileSync(path.join(spec.directory, 'wrapper.exit-code.txt'), 'utf8').trim() === '0', label + ':wrapper_not_completed');
  check(summary.checkpoint_sha256 === checkpoint && summary.authentic_client === true, label + ':checkpoint_or_client_mismatch');
  check(summary.final_round?.active === false && summary.final_round.time_remaining === 0 &&
    summary.final_round.result === 'WonByPoints' && summary.final_round.redo === false, label + ':not_completed_points_round');
  selectionApi.completedSummary(summary,selection);
  let first = null, last = null, controlled=null, readinessSource=null, sources = 0, terminalSources = 0;
  const coverageSources=[], coverageAcks=[];
  let firstDecisionSeq=null;
  const workerInputPath=path.join(original,'worker.stdin.jsonl');
  const workerInput=await receipts.scan(workerInputPath,row=>{
    if(row.type==='step'&&!row.terminal&&firstDecisionSeq===null)firstDecisionSeq=row.seq;
  });
  const relayPath = path.join(original, 'relay.stdout.jsonl');
  const relaySource = await receipts.scan(relayPath, row => {
    if(row.event==='g1_policy_action')coverageAcks.push(row);
    if (row.event !== 'g1_policy_state') return;
    coverageSources.push(coverageApi.compactSource(row));
    check(row.schema === 'rek.g1_policy_source.v1', label + ':source_schema');
    if (!first) first = row;
    if(row.observation_sequence===summary.startup_readiness?.observation_sequence)readinessSource=row;
    if(!controlled&&row.stream_active===true)controlled=row;
    check(row.round_identity_sha256 === first.round_identity_sha256 && row.local_slot === first.local_slot &&
      row.round.number === first.round.number && row.round.redo === false, label + ':source_identity_changed');
    if (last) check(row.observation_sequence > last.observation_sequence && row.clock.qpc_ticks > last.clock.qpc_ticks &&
      row.clock.qpc_frequency_hz === last.clock.qpc_frequency_hz &&
      row.round.clean_hits.every((v, i) => v >= last.round.clean_hits[i]), label + ':source_chronology_regressed');
    if (row.round.active === false) terminalSources++;
    last = row; sources++;
  });
  check(first && last && terminalSources === 1 && /^[a-f0-9]{64}$/.test(first.round_identity_sha256) &&
    [0, 1].includes(first.local_slot), label + ':source_identity_or_terminal_missing');
  const initial=selectionApi.initialSource(summary,first,controlled,readinessSource);
  check(same(last.round, summary.final_round), label + ':summary_source_round_mismatch');
  check(first.round.clean_hits.every(v => v === 0), label + ':initial_points_nonzero');
  for (const key of ['round_identity_sha256', 'local_slot']) {
    if (Object.hasOwn(summary, key)) check(summary[key] === first[key], label + ':existing_summary_identity_conflict');
  }
  const nativePin=nativeSelection.rounds.find(r=>r.label===label)?.native_source;
  check(nativePin,label+':explicit_native_capture_missing');
  const nativeFile=nativePin.file,nativeName=path.basename(nativeFile);
  const scores = [], counts = {raw_hit_packet: 0, raw_score_packet: 0};
  let captureEnd = null, captureEnds = 0, captureStarts = 0, captureErrors = 0;
  const nativeSource = await receipts.scan(nativeFile, (row, line) => {
    if (row.event === 'capture_start') captureStarts++;
    if (row.event === 'capture_end') { captureEnd = row; captureEnds++; }
    if (row.event === 'capture_error') captureErrors++;
    if (row.event !== 'raw_score_packet' && row.event !== 'raw_hit_packet') return;
    receipts.validateWirePacket(row); counts[row.event]++;
    if (row.event === 'raw_score_packet') {
      check(Number.isInteger(row.decoded.points_awarded) && row.decoded.points_awarded > 0, label + ':nonpositive_or_noninteger_award');
      scores.push({index: scores.length, native_source_line: line, wire_body_sha256: row.wire_body_sha256,
        wire_body_base64: row.wire_body_base64, ...row.decoded});
    }
  });
  check(nativeSource.bytes===nativePin.bytes&&nativeSource.sha256===nativePin.sha256,label+':explicit_native_capture_hash_mismatch');
  check(captureStarts === 1 && captureEnds === 1 && captureErrors === 0 && captureEnd.capture_error_count === 0 &&
    captureEnd.raw_hit_packet_count === counts.raw_hit_packet && captureEnd.raw_score_packet_count === counts.raw_score_packet,
    label + ':incomplete_native_capture');
  const totals = [0, 0];
  for (const score of scores) {
    totals[score.fighter_index] += score.points_awarded;
    check(totals[score.fighter_index] === score.new_hit_count, label + ':native_score_counter_mismatch');
  }
  check(same(totals, last.round.clean_hits), label + ':native_terminal_points_mismatch');

  const destination = path.join(stage, 'evidence', label), trial = path.join(destination, 'trial');
  check(!fs.existsSync(destination), label + ':destination_exists');
  fs.mkdirSync(trial, {recursive: true, mode: 0o700});
  for (const entry of fs.readdirSync(original, {withFileTypes: true})) {
    check(entry.isFile(), label + ':unexpected_trial_entry');
    if (entry.name !== 'summary.json') fs.symlinkSync(path.join(original, entry.name), path.join(trial, entry.name));
  }
  fs.writeFileSync(path.join(trial, 'summary.original.json'), originalBytes, {flag: 'wx', mode: 0o600});
  const augmented = {...summary, round_identity_sha256: first.round_identity_sha256, local_slot: first.local_slot};
  write(path.join(trial, 'summary.json'), augmented);
  const identity = {schema: 'rek.authentic_summary_identity_derivation.v1', original_summary: provenance(originalSummaryPath),
    preserved_summary: 'trial/summary.original.json', augmented_summary: provenance(path.join(trial, 'summary.json')),
    added_fields: ['round_identity_sha256', 'local_slot'].filter(k => !Object.hasOwn(summary, k)),
    derivation: 'All relay snapshots agree on round identity and actor slot; exact passive readiness binds forward in QPC/frame/sequence to the first controlled source under the unchanged driver117.5/117second bounds; summary initial round equals that controlled source, terminal equals last source. Actual timer values preserved.',
    initial_source_sequence:initial.observation_sequence, earliest_raw_source_sequence:first.observation_sequence,
    readiness_source_sequence:readinessSource?.observation_sequence??null,initial_time_remaining_seconds:initial.round.time_remaining,
    native_round_elapsed_before_control_seconds:initial.round.duration-initial.round.time_remaining,
    recorded_passive_qpc_seconds:(initial.clock.qpc_ticks-first.clock.qpc_ticks)/initial.clock.qpc_frequency_hz,
    readiness_to_control_qpc_seconds:readinessSource?(initial.clock.qpc_ticks-readinessSource.clock.qpc_ticks)/initial.clock.qpc_frequency_hz:null,
    relay_source: {...relaySource, file: relayPath}, sources, terminal_sources: terminalSources,
    round_identity_sha256: first.round_identity_sha256, local_slot: first.local_slot};
  write(path.join(destination, 'summary-identity-derivation.json'), identity);
  write(path.join(destination, 'native-wire-validation.json'), {schema: 'rek.completed_native_wire_validation.v1',
    native_source: {...nativeSource, file: nativeFile}, validator: provenance(path.join(moduleRoot, 'native_hit_receipt_data.cjs')),
    counts, decoded_wire_fields_match: true, capture_complete: true, terminal_awarded_points_by_slot: totals,
    all_cumulative_score_counters_match: true, score_receipts: scores, causal_labels_created: 0});
  const contact = await contacts.analyzeFiles(trial, nativeFile, path.join(destination, 'contact-analysis'));
  const ref = await referee.run(trial, nativeFile, path.join(destination, 'referee-validation'));
  const coverage=coverageApi.deriveCoverage(augmented,contact,coverageSources,coverageAcks,firstDecisionSeq);
  const coverageReceipt=coverageApi.receipt(coverage,[
    {...provenance(path.join(trial,'summary.json')),file:'trial/summary.json'},
    {...provenance(path.join(destination,'contact-analysis/summary.json')),file:'contact-analysis/summary.json'},
    {...relaySource,file:'trial/relay.stdout.jsonl'},
    {...workerInput,file:'trial/worker.stdin.jsonl'}].map(({file,bytes,sha256})=>({file,bytes,sha256})));
  write(path.join(destination,'controlled-policy-coverage.json'),coverageReceipt);
  check(coverage.controlled_policy_interval_complete === true && contact.terminal_evidence_consistent === true &&
    contact.terminal_point_counters_consistent === true && contact.native_capture_complete === true &&
    same(contact.reconciled_full_round_points_by_slot, totals) && contact.score_events === scores.length,
    label + ':strict_contact_checks_failed');
  check(ref.verification_passed === true && same(ref.round_identities, [first.round_identity_sha256]) &&
    ref.inputs.find(p => path.basename(p.file) === nativeName)?.sha256 === nativeSource.sha256 &&
    ref.inputs.find(p => path.basename(p.file) === 'relay.stdout.jsonl')?.sha256 === relaySource.sha256,
    label + ':strict_referee_checks_failed');
  check(contact.inputs.find(p => p.file === nativeName)?.sha256 === nativeSource.sha256 &&
    contact.inputs.find(p => p.file === 'relay.stdout.jsonl')?.sha256 === relaySource.sha256 &&
    sha(fs.readFileSync(originalSummaryPath)) === sha(originalBytes), label + ':inputs_changed_between_passes');
  const out = {label, destination, round_identity_sha256: first.round_identity_sha256, local_slot: first.local_slot,
    checkpoint_sha256: checkpoint, source_observations: sources, completed_policy_round: contact.completed_policy_round,
    controlled_policy_interval_complete:coverage.controlled_policy_interval_complete,
    controlled_policy_coverage:provenance(path.join(destination,'controlled-policy-coverage.json')),
    native_capture_complete: contact.native_capture_complete, referee_verification_passed: ref.verification_passed,
    score_events: scores.length, points: totals, outcome: contact.outcome, native_source: {...nativeSource, file: nativeFile},
    original_summary_sha256: sha(originalBytes), contact_summary: provenance(path.join(destination, 'contact-analysis', 'summary.json')),
    referee_validation: provenance(path.join(destination, 'referee-validation', 'live-referee-validation.json'))};
  console.log(JSON.stringify(out));
  return out;
}

async function main() {
  selectionApi.pinArtifacts(selection);
  for(const spec of selected){check(fs.readFileSync(path.join(spec.directory,'wrapper.exit-code.txt'),'utf8').trim()==='0',spec.id+':wrapper_not_closed');selectionApi.completedSummary(JSON.parse(fs.readFileSync(path.join(spec.directory,'trial/summary.json'))),selection);}
  const evidence = path.join(stage, 'evidence');
  check(!fs.existsSync(evidence), 'evidence_destination_exists');
  fs.mkdirSync(evidence, {recursive: true, mode: 0o700});
  const moduleFiles = [__filename, require.resolve('./controlled_policy_coverage.cjs'),require.resolve('./selection.cjs'),path.join(sourceRoot, 'analyze_live_contacts.cjs'), path.join(sourceRoot, 'join_passive_hit_events.cjs'),
    path.join(sourceRoot, 'compare_passive_contacts.cjs'), path.join(moduleRoot, 'native_hit_receipt_data.cjs'),
    path.join(moduleRoot, 'native_referee_data.cjs'), path.join(moduleRoot, 'validate_live_referee.cjs')];
  const modules = moduleFiles.map(provenance), rounds = [];
  for (const spec of selected) rounds.push(await prepare(spec));
  check(new Set(rounds.map(x=>x.round_identity_sha256)).size===rounds.length,'duplicate_actual_round_identity');
  for (const p of modules) check(provenance(p.file).sha256 === p.sha256, 'validator_changed_during_derivation');
  write(path.join(evidence, 'manifest.json'), {schema: 'rek.scorecredit5s_onpolicy_strict_evidence.v1', created_utc: new Date().toISOString(),
    selection:provenance(process.argv[2]),native_capture_selection:provenance(selection.native_capture_selection.path), minimum_completed_rounds:selection.minimum_completed_rounds, checkpoint_sha256: checkpoint, modules, rounds, original_files_changed: false,
    summary_derivation: 'Original summary bytes preserved; only missing immutable source identity fields added to derived summaries.',
    verification: 'Existing contact/referee analyzer outputs retained verbatim; explicit controlled-policy coverage receipt retains legacy result and rederives unchanged1s bounds from actual controlled start; raw hit/score wire bodies independently validated.',
    no_game_connection: true, no_gpu_execution: true, no_runtime_changes: true});
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
