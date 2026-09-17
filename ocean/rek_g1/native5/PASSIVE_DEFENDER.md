# Authentic private-AI passive defender

This experiment holds the player's high-level command at action 1 (release all
held movement). No policy, checkpoint, observation encoder, movement command or
attack command runs. Native balance, collisions, recoil, falls and resets remain
active. A neutral command does not fix the robot's position or orientation.

Execution is restricted to the owned Spark Wine display `:98`, with the existing
native private/no-human, known AI identity, exact G1 pairing and exclusive-lease
checks. Any known bot number is acceptable. Windows receives no input. This
uses the native recorder and bridge; the later observer-binding fix is
documented in the repeat-validation report.

## One bounded collection

`run_passive_defender_trial.cjs EXISTING_LIVE_CONFIG NEW_OUTPUT safe_start`
copies only the pinned relay command from an existing live-evaluation config.
It creates a fresh configuration, launches the neutral runner and records an
H.264 MP4 of the authentic client. It never loads that config's policy weights.
Use `active_attach` explicitly to collect the remainder of an already-running
round. Such a recording is identified as a partial initial round.

Use `follow_on` for a subsequent collection when the native arena is already
proven private, solo and against a known AI. It attaches active play or waits
up to 45 seconds for a verified native round transition. Actionable private
Idle/loss states use the existing start/recovery commands. It sends no inputs
during the transition wait, pins the bot and fighter slots, and aborts on a
scope change. Its collection cap is 130 seconds. It records whether attachment
occurred partway through a round; it does not claim first-tick coverage. This
mode has offline fixture coverage and a completed active-attach live trial
(R12). Its transition-wait branch has not yet completed a live trial.

The production command runs through `wsl -e bash -lc 'ssh spark ...'`. The
wrapper currently selects a 130-second collection cap and a verified existing
native ARM64 FFmpeg executable on Spark. It records at 20 frames/s, 700 kb/s,
without audio, pointer capture, mouse following or focus changes. Video delivery
requires a successful full decode and a file size strictly below 20,000,000
bytes. Capture timing is logged; exact video-frame/QPC synchronization is not
asserted.

`safe_start` uses the game's existing private-entry and ready commands. The
runner ends at the first observed terminal round, scope change or timeout. It
does not repeatedly restart matches. A successful result requires observed
terminal state, exclusively neutral post-acknowledgment command samples, and
verified stream stop/control-lease release. A duration cap remains an incomplete
experiment. Source and local command acknowledgments are kept separately from
unknown authoritative server acceptance.

## Recorded data

- `trial/relay.stdout.jsonl`: unchanged bridge messages, including all source
  observations and action acknowledgments.
- `trial/g1_policy_state.jsonl`: all source frames, including frames skipped
  while an acknowledgment is pending.
- `trial/relay.stdin.jsonl`: every command sent to the bridge.
- `trial/{initial-state,active-state,final-state,summary,provenance,exit}.json`:
  scope, outcome, ownership cleanup and source identity.
- `media/authentic-rek-passive-defender.mp4` and `capture-manifest.json`: footage,
  capture command, hash, size and decode result.

Each source contains both fighters' root positions/orientations, 30 bone-local
rotations and world positions, observed state flags, native round counters and
monotonic QPC timestamps. These are live client-rendered/replicated observations.
Retain raw traces privately; do not publish game binaries or private assets.

## Offline analysis

Run `audit_passive_defender.cjs RELAY_JSONL NEW_AUDIT_DIRECTORY` after collection
finishes. The analyzer refuses to overwrite an existing result or analyze a
source that changes during the audit.

It emits `kinematics.jsonl`, `score_events.jsonl`, `motion_intervals.jsonl` and
`summary.json`. Derivatives use measured timestamp differences, with continuity
broken across invalid samples, excessive gaps, round/slot changes or clock
changes. Named ankle, wrist and knee positions and speeds are recorded for both
robots, including root-relative speed. Score updates index one second of raw
observations on either side, with truncated windows identified.

Motion windows use an explicitly diagnostic limb-speed threshold. A window
marked `no_score_observed` is not a confirmed failed attack. Walking, recoil,
reset interpolation or another movement can exceed the threshold. Coincident
scoring does not establish causal limb attribution. Five-point counter updates
are labeled by size, without inventing a KO cause.

Native opponent move identity, authoritative strike phase, collision manifolds
and score-rejection reasons are unavailable in the current stream. Raw poses
and video support further analysis, but this recorder does not claim to resolve
those missing fields or demonstrate simulator parity.

`compare_passive_contacts.cjs ROUND_DIRECTORY NEW_REPORT_JSON` compares measured
root geometry and limb motion around score updates with nonoverlapping unscored
motion windows. Root heading uses the encoder's projected root-local +X axis
in the Unity XZ plane. It is not a measured aiming axis or an attack identity.
Unscored motion windows are controls, not confirmed missed attacks.

### Existing native hit-message recorder

The installed `RekEvidenceRecorder` 0.7.2 independently records the received
`FightCoordinator.OnHitReceived` and `OnScoreReceived` messages. Read-only packet
copies preserve world contact position, surface normal, relative speed and the
native `is_kick` flag, plus score recipient and awarded points. The deployed
recorder's capture gate currently requires Sparring Bot 1; the passive relay's
broader any-known-bot support does not change that gate.

`join_passive_hit_events.cjs RECORDER_JSONL RELAY_JSONL_OR_- NEW_OUTPUT_DIRECTORY`
joins these receipts to measured pose brackets. Use `-` for native recorder
root-pose samples when there is no relay trace. Outputs are `hit_events.jsonl`,
`score_events.jsonl`, separate `five_point_awards.jsonl`, and `summary.json`.
Raw packet data, hashes, receipt clocks, pose clocks, timing offsets, neutral
observations and ambiguous associations are retained privately.

Hit receipts confirm that the client received a native hit-effects event.
They do not identify the canned move, strike limb, victim, rejection reason or
authoritative server impact time. Same-frame score associations are explicitly
noncausal. The hit channel is unreliable, so a missing received packet does not
prove absence of contact. Five-point score awards are not counted as strikes.
FixedUpdate poses can have different Unity times within one rendered frame;
the joiner preserves all alternatives and flags this timing inconsistency.
Successful file processing does not establish action-position repeatability.

## Tests

Use Node's test runner on `passive_defender_run.test.cjs`,
`audit_passive_defender.test.cjs`, `record_passive_defender.test.cjs` and the
existing `live_transfer_run.test.cjs`, plus `compare_passive_contacts.test.cjs`
and `join_passive_hit_events.test.cjs`. Fixtures cover private/human scope,
neutral-only dispatch, round/identity changes, terminal versus timeout,
cleanup, variable observation intervals, score/motion windows and bounded
pointer-free capture arguments.

## Executed validation

See `validation/passive-defender-20260917/README.md` for the completed authentic
private-AI round: 5,674 live source samples, neutral player commands, final score
5:12, verified control release and an 11.02 MB MP4. Earlier interrupted attempts
are retained separately. Both Windows and Spark passed all 61 tests.

See `validation/passive-defender-repeat-20260917/README.md` for the user's
annotations, eight approximately 120 s captures with 65 hit receipts, a
separate short capture, geometry counterexamples, client-exit diagnostics and
the later R11/R12 paired collections. The observability report identifies the
missing opponent attack label. None of these proves exact action-position
repeatability.
