# V4 native human-worker AFK verification

The selected mixed-curriculum checkpoint was exercised through the actual
native human-evaluator worker on Spark, using isolated processes. Each case
used one arena, sampled BF16 inference, Philox seed 73, fixed initial geometry,
zero shaping weight, and external neutral category 1 on the other fighter.
Every case reached its first official terminal with zero runtime failures
and zero falls. No live HTTP endpoint or Windows input was used.

Checkpoint SHA-256:
`0d612521839298ffbe5783ed4fa449286a940b62a3a6768da7cc5ab248eb843b`.

| Policy side | Round length | Actual points for/against | First contact | Policy path |
| --- | ---: | ---: | ---: | ---: |
| Blue | 20 s | 78 / 0 | 2.92 s | 1.6583 m |
| Orange | 20 s | 69 / 0 | 3.40 s | 1.8271 m |
| Blue | 300 s | 533 / 0 | 3.10 s | 4.8865 m |
| Orange | 300 s | 530 / 0 | 3.06 s | 5.1773 m |

The harness checks every native tick, asserts that the stationary fighter's
action is 1, and verifies that actual policy contact-point observations equal
the score increase. It records first hit, actions, attack entries, root travel,
facing and terminal results. This establishes AFK scoring through the worker
used for human evaluation, consistent with the separate batched AFK result.
It is not a claim of identical trajectories between different inference batch
sizes, general human-level strength, or parity with authentic REK.

## Remaining alignment limitation

Facing within 0.16 rad was 92.5%, 6.4%, 0.513%, and 0.440% respectively.
This metric uses logical control heading, not the animated torso quaternion.
Uniform 5 Hz samples show median absolute bearings of 0.0633, 0.3874, 0.5956,
and 0.6157 rad. In the orange 300 s case, only 55.4% of samples had the opponent
within the forward 180-degree sector. Sustained frontal alignment is therefore
not established by these successful scoring results.

## Reproduction

The source harness is `native5/fast_worker_afk_probe.cjs`. It accepts the exact
worker executable, base asset configuration, checkpoint, expected hash, and a
new output directory. It asserts the worker's V4 startup identity, then runs
both policy sides at 20 and 300 s without changing any live service.

`provenance.json` records the exact command, source/executable/configuration
hashes and complete evaluation environment. `summary.json` contains compact
results, the first scoring event, and SHA-256 hashes of every complete trace.
The complete per-tick-derived traces and detailed score-event summary remain
in the private Spark directory named there. No checkpoint weights, game
assets, proprietary binaries, or joint arrays are included in this evidence.
The process exited zero. Its wall time includes worker startup, native
inference, diagnostic transfers and Node protocol handling, and is not
training SPS.

## Deployment observation

`deployment.json` records a read-only check at 2026-09-15T05:57:30.555Z.
Localhost 18769 served Node PID 151625 and native worker child 151637, using
the verified V4 worker and selected checkpoint. The active human side was
orange, with a 300 s round, paused at tick zero and score 0:0. All 12 frozen
evaluation conditions were present. Existing listeners on 18766, 18768 and
8899 remained separate and running. The prior viewer and isolated test Node
processes were absent.

`rek-human-viewer-v4.cjs` and `rek-viewer-v4-service.sh` are byte-identical
copies of the deployment helper sources, with hashes recorded in the JSON.
They preserve historical deployment provenance, not a general-purpose restart
instruction. The service script verifies the exact old process identity and
paused state before stopping it. This evidence check did not execute either
helper, stop a process, or send gameplay input.
