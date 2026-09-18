# Authentic policy performance test

Candidate checkpoint: `dd335d696ab0f2ae891b448d174e3615d834cd5b8fd4cf8cffbae6c5126f4c46`.
This is the frozen checkpoint after nine-target contact training, not an earlier
checkpoint or the neutral-input defender.

## Required paired measurements

The next two private-AI rounds use the same frozen checkpoint and record:

- Terminal result, local fighter slot, opponent bot identity and awarded points.
- Each requested move, local dispatch return, distance and facing at its source observation.
- Received native score and hit-effect packets, with adjacent measured poses.
- CUDA inference latency, live observation/decision rates and capture continuity.
- Authentic framebuffer MP4, with each file strictly below 20,000,000 bytes.

Repeated requests and native dispatch returns do not establish server execution.
An absent hit packet does not label a miss. Point awards and received strike
effects are counted separately. Candidate temporal associations are reported
with their uncertainty, without fitting definitive move hit regions from them.

The native `CleanHits` counter was verified as cumulative integer awarded points,
including referee awards. See [source provenance](../../live_transfer/SCORE_COUNTER_PROVENANCE.md).

## Attempt on 18 September 2026 UTC

The normal isolated Spark client was launched at 00:14:47 UTC with the pinned
game and recorder/bridge DLLs. Wine reported three `GL_OUT_OF_MEMORY` buffer
allocation errors before bridge readiness. A read-only state request returned
`relay_closed_before_state`, code 3. No private arena was entered and no policy
input was sent. The owned process was terminated at 00:16:20 UTC, exit 143.

The isolated NVIDIA GL context subsequently reported 91,927 MB total available
memory and 31 MB currently available dedicated memory. The separate
`qwen38-flash` container owns the approximately 91 GB `sglang` allocation.
Earlier successful REK runs used the same Wine D3D11 backend and emulated GTX
470 reporting, so that GPU name alone does not establish a rendering regression.
No unrelated workload was stopped.

A per-process Mesa/llvmpipe fallback was also attempted. Native GL capability
was verified, but Wine stalled before Unity initialization or a bridge endpoint.
The game exited 143 at 00:26:40 UTC. Seven remaining launch-owned helpers were
identified and stopped individually. The isolated X server and unrelated model
service remain running. No relay or game process from these attempts remains.

There are **zero completed authentic evaluation rounds for this checkpoint in
this attempt**. No authentic win rate, hit rate or transfer success is claimed.
Its earlier 510/512 simulator wins and approximately 988,653 training SPS remain
simulator measurements, separate from this test.

The native CUDA worker passed 121 protocol/inference assertions using synthetic
observations, with five masked decisions and median latency 0.162177 ms. These
are inference diagnostics, not game-performance or training-throughput results.
A subsequent validation attempt failed its startup assertion; its stderr is
preserved. A following direct startup again returned the expected pinned CUDA
worker identity. Resource contention is a candidate explanation, not a proven
cause of that transient assertion failure.

After the software-renderer attempt was cleaned up, the saved repeat validation
passed all 121 assertions, with median latency 0.276338 ms and maximum
1.953534 ms across five synthetic masked decisions. This sample is too small to
characterize a sustained control-rate distribution.

## Reproduction and evidence

### Historical authentic-policy check

The new analyzer was also run against two existing complete private-AI policy
rounds, both using the **older** checkpoint
`f87dae69a777e4ac28782bdee89b30d7434208d773be75bf97f56fba4a52b07e`.

| Historical trial | Policy : AI points | Outcome | Attack requests / native dispatch returns | Local score events |
| --- | ---: | --- | ---: | ---: |
| `live-bot1-rendered-r3` | 4 : 14 | Loss | 78 / 78 | 4 |
| `live-any-ai-r4` | 5 : 17 | Loss | 67 / 67 | 5 |

All awarded points reconcile with the terminal counters. Each local score event
awarded one point. Native captures have zero capture errors and correlate with
the corresponding policy streams through 1,107 and 1,090 concurrent pose/clock
anchors. No missing effect packet was labeled a miss. Dividing score events by
requested attacks would not establish an executed-attack hit rate.

These are observed failures of that older policy, not measurements of the new
checkpoint. The [numeric historical report](results/authentic-historical-policy-summary.json)
includes control coverage, all move-request counts and request distance/bearing
ranges. Those ranges are not asserted successful-contact regions.

The final capture, driver, score-summary and contact-analysis regression suite
passed 56 tests locally. The contact analyzer's 16 tests also passed on Spark,
and both historical rounds were processed with the final analyzer source.

### Current attempt

Private Spark test root:
`/home/spark-advantage/rek-training/primitive-live-eval-20260918-r1`.
It contains two prepared configs, checkpoint/encoder/worker/driver hashes,
native-worker diagnostics and orchestration test output. Preparation is not
reported as a completed fight.

Readiness evidence:
`/home/spark-advantage/rek-training/primitive-contact-20260917-r1/authentic-readiness-20260918-r1`.
Runtime logs are under
`/home/spark-advantage/codexrook-runtime/live-transfer-20260915/primitive-readiness-20260918-r1`.

`record_passive_defender.cjs` now accepts an explicit
`capture_controller: "frozen_policy"` with the pinned worker/checkpoint, and
labels that capture accordingly. The existing neutral-input mode is preserved.
Both modes enforce the existing MP4 size and decode checks. The capture and
live-driver tests passed 36/36 on Windows and Spark.

The private archive on the physical file server contains 99 files, 1,346,158 bytes:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-17\primitive-live-eval-20260918-r1`.
The 78 startup/configuration files and 21 historical-analysis files were all
SHA-256 compared with their local copies.

Only numeric summaries, analysis code and tests are published. Checkpoints,
game binaries, raw game captures and account material remain private.
