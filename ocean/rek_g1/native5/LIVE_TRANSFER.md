# Frozen checkpoint into the authentic REK client

The execution target is the isolated REK game copy on Spark, under Wine 11.13
and X11 display `:98`. The Windows desktop receives no synthetic input.

Data flow:

```
Authentic REK client poses, round counters and native input state
  -> verified local named-pipe relay
  -> native C++ client-pose observation encoder
  -> persistent CUDA/BF16 MinGRU checkpoint inference
  -> sequenced native held-command / move request
  -> authentic REK client and its normal server connection
```

`live_transfer_run.cjs` orchestrates JSONL transport. It contains no simulator,
physics implementation, policy inference, or Python dependency. Environment
stepping and neural inference are not performed by this Node control process.
The encoder loads the private model's hinge calibration once; it never steps
MuJoCo. `live_policy_worker.cu` runs the selected frozen network on the GPU.

## Representation and limits

`client_pose_projection_v1` maps received roots and bone rotations to the
selected checkpoint's 223-feature schema. Its manifest distinguishes measured
fields, mathematical projections, structural training constants, and
unavailable quantities. See [encoder documentation](live_transfer/README.md).

For visual-only network clients, authoritative attack completion is unknown.
The opt-in `dispatched_request_v4_duration` projection uses the training
environment's move durations after a native send returns. It is a timing
estimate, and does not establish server acceptance or playback completion.
Actual REK still handles physics, damage and command eligibility.

The action space contains no quit, disconnect or matchmaking action. Private
Sparring Bot 1 entry is setup orchestration. A loss can be exited only after an
observed inactive loss prompt. Every gameplay request is tied to its source
round and observation, with a single action in flight. Older telemetry is
discarded after each game acknowledgment. Windows input is unavailable.

## Running and inspecting a trial

Build the native encoder, worker and bridge using their checked-in build
instructions. Preserve the existing installed game and deploy only into the
owned isolated copy. Adapt `live_transfer_config.example.json` to the deployed
binary hashes and use a new output directory for each trial.

```
node live_transfer_run.cjs trial.config.json
node summarize_live_transfer.cjs /absolute/path/to/trial-output
```

The output preserves all process stdin, stdout and stderr, projected
observations, native action acknowledgments, checkpoint identity and round
counters. `measured-report.json` includes artifact hashes and latency metrics.
Its additive `awarded_points` field labels initial/final native `CleanHits`
totals as cumulative integer awarded points, including referee awards. Raw
`initial_round`/`final_round.clean_hits` fields are preserved. These totals do
not count strike events or identify award causes. See
[counter provenance](live_transfer/SCORE_COUNTER_PROVENANCE.md).
Raw traces and proprietary game/model assets remain private. Only sanitized
test results and source code belong in the repository.

For a frozen-policy trial with a video, add `capture_controller: "frozen_policy"`
and the pinned native `capture_ffmpeg` / `capture_ffmpeg_sha256` fields to the
config, then run on Spark:

```sh
node record_passive_defender.cjs /absolute/config.json /absolute/live_transfer_run.cjs /new/media-directory
node summarize_live_transfer.cjs /absolute/trial-directory
node analyze_live_contacts.cjs /absolute/trial-directory /absolute/matching-native-capture.jsonl /new/contact-analysis-directory
```

The video wrapper preserves neutral-defender compatibility and emits a distinct
frozen-policy manifest and MP4 filename. Each video must decode successfully and
be smaller than 20,000,000 bytes. Use a fresh output directory for each round.
The native capture and policy stream must refer to the same measured round.
The contact analysis keeps requests, local dispatch returns, received effects,
awarded points and pose context separate. It does not turn missing hit packets
into misses or label the latest requested move as the executed attack.

Pair this actual-client evaluation with the headless training/evaluation report
after contact-model changes. A startup failure, partial round or transport test
does not supply an authentic-game win rate. Live decision rate and video frame
rate are separate from headless training SPS.

Live decision frequency is bounded by the running game's cadence. It must not
be reported as headless training SPS. Training win rates in the candidate do
not establish a live REK win rate.
