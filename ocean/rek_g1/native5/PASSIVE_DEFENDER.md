# Authentic private-AI passive defender

This experiment holds the player's high-level command at action 1 (release all
held movement). No policy, checkpoint, observation encoder, movement command or
attack command runs. Native balance, collisions, recoil, falls and resets remain
active. A neutral command does not fix the robot's position or orientation.

Execution is restricted to the owned Spark Wine display `:98`, with the existing
native private/no-human, known AI identity, exact G1 pairing and exclusive-lease
checks. Any known bot number is acceptable. Windows receives no input. The game
and installed bridge do not need modification for this experiment.

## One bounded collection

`run_passive_defender_trial.cjs EXISTING_LIVE_CONFIG NEW_OUTPUT safe_start`
copies only the pinned relay command from an existing live-evaluation config.
It creates a fresh configuration, launches the neutral runner and records an
H.264 MP4 of the authentic client. It never loads that config's policy weights.
Use `active_attach` explicitly to collect the remainder of an already-running
round. Such a recording is identified as a partial initial round.

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

## Tests

Use Node's test runner on `passive_defender_run.test.cjs`,
`audit_passive_defender.test.cjs`, `record_passive_defender.test.cjs` and the
existing `live_transfer_run.test.cjs`. Fixtures cover private/human scope,
neutral-only dispatch, round/identity changes, terminal versus timeout,
cleanup, variable observation intervals, score/motion windows and bounded
pointer-free capture arguments.

## Executed validation

See `validation/passive-defender-20260917/README.md` for the completed authentic
private-AI round: 5,674 live source samples, neutral player commands, final score
5:12, verified control release and an 11.02 MB MP4. Earlier interrupted attempts
are retained separately. Both Windows and Spark passed all 61 tests.
