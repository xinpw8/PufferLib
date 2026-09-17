# Passive defender against authentic private AI, 2026-09-17 UTC

The experiment sends only action 1, which releases held player controls. The
native robot controller, collisions, recoil and balance continue running. The
player is not frozen or anchored. No checkpoint, encoder or learned policy is
used. Both fighters' live replicated poses and native score counters are saved.

Execution host: `spark-4ae3`, Linux ARM64. The installed authentic REK client runs
in the existing isolated Wine display `:98`. Windows receives no input. These
measurements describe that client/runtime; Windows parity is not asserted.

## Executed attempts

| Attempt | Result | Observed player commands | Video |
| --- | --- | --- | --- |
| round-r1 | Private entry exceeded the original 45 s wait; cleanup verified | None | None |
| round-r2 | 4,146 samples; game exited before round completion | 4,145 applied neutral commands; zero other actions | Validated 7,562,241-byte MP4 |
| round-r3 | Relaunched game exited during startup; relay connection timed out | None | None |
| round-r4 | Observed terminal round; cleanup verified; exit 0 | 5,671 applied neutral commands; zero other actions | Validated 11,023,171-byte MP4 |

The private arena loaded asynchronously after round-r1's deadline. The passive
runner now permits a 120 s entry wait. The learned driver retains its 45 s
default. The game exits remain unexplained: bounded log inspection found no
fatal marker or OOM evidence. They are retained as failures.

## Completed round: round-r4

The subsequent owned-client launch remained available. The private AI round was
observed from 119.666595 s remaining through the terminal `WonByPoints` state.
Collection lasted 119.843 s and recorded 5,674 source samples. Every observed
player desired action was 1 with zero velocity command. No player attack was
requested. This includes a terminal sample; the analyzer counts 5,673 samples
with the stream and round both active.

Of 5,673 neutral requests, 5,671 applied locally. One was rejected as stale and
one as `policy_stream_not_owned` during terminal shutdown. Both rejections are
retained. All 5,672 post-acknowledgment source samples passed the full neutral
check. Native stop and lease release were accepted and independently read back.
No automatic rematch was started after the requested experiment.

Final score was player 5, AI 12. Eight observed score updates comprise a
simultaneous +5 to both players and seven further +1 AI awards:

| Source sample index | Remaining time (s) | Player delta | AI delta | Score |
| --- | --- | --- | --- | --- |
| 2555 | 64.947340 | 5 | 5 | 5:5 |
| 3454 | 46.142788 | 0 | 1 | 5:6 |
| 3536 | 44.542480 | 0 | 1 | 5:7 |
| 3910 | 36.839855 | 0 | 1 | 5:8 |
| 4043 | 34.139526 | 0 | 1 | 5:9 |
| 5191 | 10.019797 | 0 | 1 | 5:10 |
| 5271 | 8.419604 | 0 | 1 | 5:11 |
| 5344 | 6.919508 | 0 | 1 | 5:12 |

Counter increments are observations, not established contact/KO causes. The
analyzer found zero invalid records and one sampling gap exceeding 0.25 s. It
breaks derivative and score-window continuity at that gap. There were 644
diagnostic limb-motion intervals, 588 without a contextual score update. These
remain motion counts, not attack or miss counts.

The MP4 is H.264, 1280 x 720, 20 frames/s, 125.100 s. Full decoding succeeded,
and the final 5:12 score was visually checked. SHA-256:
`77f89eecd98881718d89db7769cf088b6d797efbe47f9e7d62a062f86c2d5fc5`.
Raw relay SHA-256:
`6671bd913a528ed971e4d8247dd7dff76fb6dfb2a6ff44060db5786ab617d543`.
Both hashes were verified after copying to the physical server.

## First usable recording: round-r2

All 4,146 observed desired actions equal 1 and all observed velocity commands
equal `[0, 0, 0]`. All 4,145 post-acknowledgment samples also satisfy the stricter
neutral check, including no pending move/special/e-stop request. The opponent
was native Sparring Bot 1. Score changed from 0:0 to 0:7 before interruption.

| Source sample index | Remaining round time (s) | AI point increase | AI total |
| --- | --- | --- | --- |
| 1668 | 86.186295 | 1 | 1 |
| 1895 | 81.583565 | 1 | 2 |
| 2385 | 71.578735 | 1 | 3 |
| 2866 | 61.857582 | 2 | 5 |
| 3964 | 39.535300 | 2 | 7 |

The analyzer completed with zero invalid records. Each score event references
one second of surrounding raw observations on either side. The diagnostic
1.75 m/s limb-motion threshold produced 482 limb intervals, including 390 with
no score update in their contextual window. These counts are not attack or
miss counts. Walking, recoil and resets can exceed the threshold; simultaneous
scoring does not identify the causal limb. Opponent move IDs, authoritative
strike phases, contact manifolds and rejection reasons remain unavailable.

The MP4 is H.264, 1280 x 720, 20 frames/s, 99.100 s and fully decodes. Video
includes capture overhead beyond the 85.944 s collection interval. No exact
video-frame/QPC synchronization is claimed. Its SHA-256 is
`497eb8de758df8f331491d487490d15adaefdcbcf97d108844ed1dff77cd22d0`.
The raw relay SHA-256 is
`a6fe74decf59fc1f7c2cce8e77a7a737b8d10e1ff65a692682eaa33756a40cf6`.

`round-r2.audit.json` reports successful offline analysis; this does not imply
a completed live experiment. `round-r2.summary.json` records exit code 2,
incomplete round, relay EOF and unavailable cleanup acknowledgment after the
game disappeared. The capture manifest separately identifies valid media.

## Reproduction and saved evidence

Collection command on Spark, invoked through Windows PowerShell and WSL SSH:

```sh
node /home/spark-advantage/rek-training/passive-defender-20260917-r1/source/ocean/rek_g1/native5/run_passive_defender_trial.cjs \
  /home/spark-advantage/rek-training/policy-quality-20260916-r1/live-any-ai-r4/trial.config.json \
  /home/spark-advantage/rek-training/passive-defender-20260917-r1/NEW_ATTEMPT safe_start
```

The new output directory must not exist. Full commands, stdout, stderr, raw
observations, hashes, per-run configuration, score windows and video remain in:

- Spark: `/home/spark-advantage/rek-training/passive-defender-20260917-r1/`
- Physical server: `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-17\passive-defender-r1\`

Only aggregate results, source and tests are committed here. Raw bone traces,
video, game binaries, credentials and policy weights are not committed.

Per-attempt provenance identifies the executed source hashes. R2 used the
original 45 s entry wait. R4 used the extended entry wait. Child-process timeout
escalation was added to the recording wrapper after these captures and tested
separately; neither recording is retroactively attributed to that revision.

## Verification and deployed source

The four Node test files passed 61/61 on both Windows and Spark. Complete test
stdout/stderr is included here. Coverage includes neutral-only real-child relay
fixtures, private/no-human identity checks, terminal cleanup, delayed private
entry, variable-QPC score analysis and owned-child timeout escalation. Linux
fixtures actually ignore SIGTERM to exercise the forced-stop path. None sends
input to REK.

The final implementation is deployed under
`/home/spark-advantage/rek-training/passive-defender-20260917-r1/source-final/`.
Use that source directory for subsequent collections. The earlier `source/`
snapshot is preserved with the archived R4 evidence. No further match or
training job was started after the completed passive experiment.
