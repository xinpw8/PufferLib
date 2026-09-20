# Balance and count observability

Read-only review, 2026-09-20 UTC. The current policy already receives rendered
pelvis height and full root orientation. It does not receive the separately
validated referee count state. This distinction is supported by actual worker
inputs from r10, r11 and r12; it is not a claim that adding features improves
fighting. No encoder/runtime change, GPU work, training or client interaction
was performed for this report.

## Current 223-feature contract

Indices are zero-based; paired entries are own / opponent. Source references
were rechecked against `live_transfer/encode_live.cpp:140`, `:175`, `:203`,
`:208`, `:212`, `:222` and `:228` when this report was written.

| Indices | Current meaning |
|---|---|
| 2, 73 / 88, 159 | Measured rendered pelvis height, duplicated within each entity |
| 3..6 / 89..92 | Full measured root quaternion, including orientation changes associated with tilt |
| 7, 8, 12 / 93, 94, 98 | Derived horizontal local velocity and yaw rate |
| 9..11 / 95..97 | Vertical velocity and other angular-rate slots fixed at zero |
| 13..70 / 99..156 | Projected bone hinge angles and receipt-time finite-difference rates |
| 71, 72, 79 / 157, 158, 165 | Explicit down, tilt and fallen slots fixed at zero |
| 77 / 163 | Literal 2, not measured floor contacts |
| 189, 190, 191 | Remaining round time /120, own awarded points, opponent awarded points |
| 192, 193; 202..205 | Fall totals and down-state slots fixed at zero |
| 217, 218 | Observation-window awarded-point increases, including referee awards |
| Referee count and elapsed count seconds | Present in source telemetry, absent from policy inputs |

Thus zero explicit fall features do not establish an upright fighter. The
rendered quaternion/height remain observable. Root height is reported below
in Unity coordinate units; no new standing-height ratio or physical fall
threshold is inferred.

## Measured examples

r10 won 10:6 without a count or five-point award. It provides no own-count
example. Both r10 and r11 used the unmasked observation contract.

r11 won 11:9: non-five-point awards were 6:9, with one local +5. The opponent's
received count began at round elapsed 71.17996 s. There were 141 actual worker
decisions during that count, with received count seconds 0, 1 and 2. Opponent
height ranged from 0.11710437 to 0.15417170, with source tilt 74.235916 to
120.45766 degrees. Its height and full quaternion entered the policy, while
explicit down/tilt/fall-count/count-state slots remained zero and floor slots
remained 2. Native packet validation measured a 2.9917118 s count ending in a
received Knockout/+5 call. The score changed from 3:6 to 8:6 at worker sequence
3617, 2.9363378 s after the first count observation. Count-clear/Knockout was
first observed 54.9368 ms later. The two receipt streams are asynchronous.

r12 lost 8:10 despite leading non-five-point awards 8:5. Its own count produced
the opponent's +5. The pose was already changing before the received count:

| Worker sequence | Relative to first own-count observation | Own height | Source tilt |
|---|---:|---:|---:|
| 813 | -0.503089 s | 0.52443710 | 43.11744 degrees |
| 828 | -0.192930 s | 0.09734136 | 85.13155 degrees |
| 838 | 0 s | 0.114509106 | 92.79706 degrees |

These worker inputs contained the corresponding height and full quaternion,
not the scalar source tilt. The first own-count observation was at round
elapsed 18.24965 s, score 0:0. Across 145 actual worker decisions, explicit
down/tilt/falls/count-state remained zero. A fresh move-10 request used sequence
905, 1.3671407 s after the first count observation, with the recorded own count
active, height 0.100891456 and tilt 95.10352 degrees. Native dispatch returned
1.4021348 s after the first count packet receipt. This does not establish
execution or contact, and that later request cannot explain the earlier count
onset. The +5 scoreboard change arrived after 2.9510328 s; count-clear/Knockout
followed 70.9340 ms later. Strict validation measured a 3.0195863 s count.

## Proposed versioned correction, not implemented

The smallest explicit count-state extension can retain 223 dimensions:

| Currently unused zero column | Proposed meaning |
|---|---|
| 187 | Validated received referee state available |
| 206 | Own received count active |
| 207 | Opponent received count active |
| 215 | Received quantized elapsed count seconds |
| 216, optional | Receipt age in seconds |

All five columns were zero in all 11,600 r10/r11 worker inputs. The encoder
sets them to zero at line 212; native `raw_value` has no nonzero cases for them
(`fast_runtime.cu:523`). Do not reuse 208: the native contract assigns it reset
elapsed time.

Use a new semantic schema version. To preserve initial actor/value behavior,
zero only the corresponding first-layer input columns and retain every other
parameter, then verify frozen-policy replay compatibility. Join each recorded
worker decision to its same-source validated referee receipt, retaining the
frozen behavior probabilities. Never backfill from later receipts. Unavailable
state must remain distinguishable from a measured inactive count. The packet's
elapsed integer is not an exact remaining countout deadline.

Authentic trajectory training can consume these measured features. Compact
native training cannot yet generate their causal transitions: it uses canned
root-height/orientation frames and planar motion, with no balance/contact
producer that causes actual falls, counts or recovery
(`fast_runtime.cu:459`, `:488`). Its appropriate availability value would be
zero. Do not manufacture falls from height, tilt or hit counts, or add a
live-only attack gate. Count features clarify an already-received referee
phase; they do not establish prevention of the initiating fall or recovery.

## Evidence

Private root: `C:/rekagent/work/consistent-fighter-20260919-r1/`. Within
`live-round_outcome_v1-r10`, `-r11` and `-r12`, the review joins
`trial/encoder.stdin.jsonl`, `trial/encoder.stdout.jsonl` and
`trial/worker.stdin.jsonl` by observation sequence. Request timing comes from
`contact-analysis/attacks.jsonl`; packet matching and count episodes come from
`referee-validation/live-referee-validation.json`. Both r11/r12 strict referee
validations passed. Their native recordings are bound by these SHA256 hashes:

- r11: `d1c0e7135e88acad05d3b772403d742b279e65b0e65f94bfffd9afba4ce63a4a`
- r12: `6a69ee0b860ff30c3c9c8dee854a77f0fcfd72b17741985399a14bd51f73e110`

Reviewed encoder SHA256:
`4f5bd932b61889b5de6e4b2e0db2b9a4f5a695652c53d2f4f95b68a4810fd54d`.
Reviewed compact runtime SHA256:
`07c22ca1d4255db9940dc9e39a2ff0f1d1368491dfda4e3bbc17a09a61c11273`.
Earlier boundaries remain documented in
[action-interface diagnosis](../reward-objective-20260919/action-interface-diagnosis.md)
and [external balance validation](../balance-transfer-windows-20260919/README.md).
