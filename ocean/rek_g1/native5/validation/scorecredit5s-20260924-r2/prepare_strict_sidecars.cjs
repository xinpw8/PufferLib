'use strict';
// CPU-only derivation from five completed, immutable captures. No game or bridge access.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const campaign = '/home/spark-advantage/rek-training/balance8-live-20260924-r1';
const stage = '/home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2';
const nativeRoot = '/home/spark-advantage/codexrook-runtime/wineprefix/drive_c/rekagent/evidence/runtime/rek-private-ai-protocol-v7';
const moduleRoot = '/home/spark-advantage/rek-training/referee-bridge-spark-20260924-r1';
const sourceRoot = '/home/spark-advantage/rek-training/f7-action-id-fix-20260924-r1/source/ocean/rek_g1/native5';
const referee = require(path.join(moduleRoot, 'validate_live_referee.cjs'));
const receipts = require(path.join(moduleRoot, 'native_hit_receipt_data.cjs'));
const contacts = require(path.join(sourceRoot, 'analyze_live_contacts.cjs'));
const selected = ['balance8-s901-retry4', 'balance8-s902-retry3', 'balance8-s903-retry3', 'balance8-s904-retry3', 'balance8-s905-retry3'];
const checkpoint = '9a875c347b512bb46dde25481be75d276ea1b4c0799886b30aaf03be6ae1c5ce';
const check = (condition, message) => { if (!condition) throw Error(message); };
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', {flag: 'wx', mode: 0o600});
const provenance = file => { const bytes = fs.readFileSync(file); return {file, bytes: bytes.length, sha256: sha(bytes)}; };

async function prepare(label) {
  const original = path.join(campaign, label, 'trial');
  const originalSummaryPath = path.join(original, 'summary.json');
  const originalBytes = fs.readFileSync(originalSummaryPath);
  const summary = JSON.parse(originalBytes);
  check(fs.readFileSync(path.join(campaign, label, 'wrapper.exit-code.txt'), 'utf8').trim() === '0', label + ':wrapper_not_completed');
  check(summary.checkpoint_sha256 === checkpoint && summary.authentic_client === true, label + ':checkpoint_or_client_mismatch');
  check(summary.final_round?.active === false && summary.final_round.time_remaining === 0 &&
    summary.final_round.result === 'WonByPoints' && summary.final_round.redo === false, label + ':not_completed_points_round');
  let first = null, last = null, sources = 0, terminalSources = 0;
  const relayPath = path.join(original, 'relay.stdout.jsonl');
  const relaySource = await receipts.scan(relayPath, row => {
    if (row.event !== 'g1_policy_state') return;
    check(row.schema === 'rek.g1_policy_source.v1', label + ':source_schema');
    if (!first) first = row;
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
  check(same(first.round, summary.initial_round) && same(last.round, summary.final_round), label + ':summary_source_round_mismatch');
  check(first.round.clean_hits.every(v => v === 0), label + ':initial_points_nonzero');
  for (const key of ['round_identity_sha256', 'local_slot']) {
    if (Object.hasOwn(summary, key)) check(summary[key] === first[key], label + ':existing_summary_identity_conflict');
  }
  const nativeName = receipts.chooseNativeFile(fs.readdirSync(nativeRoot), first.clock.utc);
  const nativeFile = path.join(nativeRoot, nativeName);
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
    derivation: 'All completed relay policy-source snapshots agree on round identity and actor slot; original summary initial/final round objects equal first/last snapshots.',
    relay_source: {...relaySource, file: relayPath}, sources, terminal_sources: terminalSources,
    round_identity_sha256: first.round_identity_sha256, local_slot: first.local_slot};
  write(path.join(destination, 'summary-identity-derivation.json'), identity);
  write(path.join(destination, 'native-wire-validation.json'), {schema: 'rek.completed_native_wire_validation.v1',
    native_source: {...nativeSource, file: nativeFile}, validator: provenance(path.join(moduleRoot, 'native_hit_receipt_data.cjs')),
    counts, decoded_wire_fields_match: true, capture_complete: true, terminal_awarded_points_by_slot: totals,
    all_cumulative_score_counters_match: true, score_receipts: scores, causal_labels_created: 0});
  const contact = await contacts.analyzeFiles(trial, nativeFile, path.join(destination, 'contact-analysis'));
  const ref = await referee.run(trial, nativeFile, path.join(destination, 'referee-validation'));
  check(contact.completed_policy_round === true && contact.terminal_evidence_consistent === true &&
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
    native_capture_complete: contact.native_capture_complete, referee_verification_passed: ref.verification_passed,
    score_events: scores.length, points: totals, outcome: contact.outcome, native_source: {...nativeSource, file: nativeFile},
    original_summary_sha256: sha(originalBytes), contact_summary: provenance(path.join(destination, 'contact-analysis', 'summary.json')),
    referee_validation: provenance(path.join(destination, 'referee-validation', 'live-referee-validation.json'))};
  console.log(JSON.stringify(out));
  return out;
}

async function main() {
  const evidence = path.join(stage, 'evidence');
  check(!fs.existsSync(evidence), 'evidence_destination_exists');
  fs.mkdirSync(evidence, {recursive: true, mode: 0o700});
  const moduleFiles = [__filename, path.join(sourceRoot, 'analyze_live_contacts.cjs'), path.join(sourceRoot, 'join_passive_hit_events.cjs'),
    path.join(sourceRoot, 'compare_passive_contacts.cjs'), path.join(moduleRoot, 'native_hit_receipt_data.cjs'),
    path.join(moduleRoot, 'native_referee_data.cjs'), path.join(moduleRoot, 'validate_live_referee.cjs')];
  const modules = moduleFiles.map(provenance), rounds = [];
  for (const label of selected) rounds.push(await prepare(label));
  for (const p of modules) check(provenance(p.file).sha256 === p.sha256, 'validator_changed_during_derivation');
  write(path.join(evidence, 'manifest.json'), {schema: 'rek.balance8_onpolicy_strict_evidence.v1', created_utc: new Date().toISOString(),
    campaign, checkpoint_sha256: checkpoint, modules, rounds, original_files_changed: false,
    summary_derivation: 'Original summary bytes preserved; only missing immutable source identity fields added to derived summaries.',
    verification: 'Existing contact/referee analyzer outputs retained verbatim; raw hit/score wire bodies independently validated.',
    no_game_connection: true, no_gpu_execution: true, no_runtime_changes: true});
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
