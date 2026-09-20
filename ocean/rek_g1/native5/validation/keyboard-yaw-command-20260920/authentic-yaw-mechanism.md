# Authentic yaw behavior, r37-r42

The reset-trained policy produced a few longer turns, but the six development
rounds do not show a general increase in sustained turning. Median retained
yaw-sign duration and median nonzero outgoing yaw remained effectively unchanged.
This measurement describes commands and their timing; it does not identify the
cause of the match results.

All six Windows clients were closed before reading their streams. Alternating
legacy rounds r37/r39/r41 used checkpoint
`93180341b685c99549d9f55693a56b8cf42dfc45857790d77d43920d35219ba3`;
reset rounds r38/r40/r42 used
`85808a6731f4f40f5756800a9edf93faa5c312aa7d5f7468cafdd4ae3a5c381f`.
Both policies used the same live bridge, encoder and sampling configuration.
The training command-model difference is documented in [README.md](README.md).

## Command behavior

| Round | Training mode | Retained sign-run median, ms | Retained runs >=500 ms | Nonzero outgoing yaw median | Nonzero yaw p95 | Commands with absolute yaw >=0.9 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r37 | Legacy | 34.437 | 0 | 0.068801 | 0.236076 | 0 |
| r38 | Reset | 34.542 | 0 | 0.068653 | 0.217053 | 0 |
| r39 | Legacy | 35.088 | 0 | 0.069401 | 0.219594 | 0 |
| r40 | Reset | 33.705 | 2 | 0.068736 | 0.269103 | 37 |
| r41 | Legacy | 34.486 | 0 | 0.069138 | 0.243535 | 0 |
| r42 | Reset | 34.575 | 1 | 0.069728 | 0.263138 | 9 |

| Pooled measurement | Legacy | Reset |
| --- | ---: | ---: |
| Ready source rows | 17,489 | 17,473 |
| Retained nonzero sign runs | 5,225 | 5,346 |
| Retained sign-run median / p95, ms | 34.689 / 120.745 | 34.317 / 118.023 |
| Retained runs below 100 ms | 4,737 | 4,916 |
| Retained runs >=500 ms | 0 | 3 |
| Outgoing movement commands | 19,733 | 19,565 |
| Commands with nonzero yaw | 13,179 | 12,935 |
| Absolute nonzero outgoing yaw median / p95 | 0.069123 / 0.236674 | 0.069034 / 0.245635 |
| Absolute nonzero outgoing yaw p95 outside projected busy | 0.252600 | 0.409484 |
| Outgoing commands with absolute yaw >=0.9 | 0 | 46 |
| Native completed sign-run median, ms | 34.542 | 34.141 |

The three retained runs >=500 ms are also visible as three completed native
request-sign runs. None is entirely inside projected-busy time. Reset's larger
outside-busy upper tail is concentrated in r40, whose p95 is 0.807633; reset
r38 has p95 0.172583 and no near-full-yaw commands. A win in r38 therefore does
not require the newly observed long-turn tail. Continued short sign runs remain
common under both policies.

## Attack-facing context

Existing validated contact summaries contain the following pre-request geometry.
No category17 request occurred, so their `remaining_attacks` group includes all
attack requests in this cohort.

| Round | Attack requests | Median gap, captured Unity units | Median absolute rendered-root bearing, degrees |
| --- | ---: | ---: | ---: |
| r37, legacy | 82 | 0.652589 | 83.5185 |
| r38, reset | 90 | 0.839140 | 46.5961 |
| r39, legacy | 93 | 0.776984 | 37.8145 |
| r40, reset | 66 | 0.721273 | 63.2000 |
| r41, legacy | 78 | 0.729683 | 52.2569 |
| r42, reset | 78 | 0.797086 | 49.5253 |

The reset arm does not uniformly reduce this bearing proxy. Round r40 has a
larger median than preceding legacy r39, despite its longer outgoing-yaw tail.
Distances and combat states differ, and rendered pelvis/root +X can differ from
controller heading during canned motion. These are descriptive snapshots,
without a demonstrated distance calibration, contact attribution or matched-state
comparison. Geometry quantiles retain the existing summaries' linear interpolation
at `(N-1)*q`, unlike the yaw tool's lower-order statistic.

The [outcome report](../consistent-fighter-20260919/yaw-command-development-r37-r42.md)
records reset 2W/1L, 46:36 points, versus legacy 0W/3L, 26:55. Both arms received
16 non-five-point points. Reset's total-score advantage includes more five-point
awards and does not establish improved ordinary strike scoring.

## Measurement and limits

The private adaptation preserves the existing
[yaw-cadence-evidence.cjs](../consistent-fighter-20260919/yaw-cadence-evidence.cjs)
formulas. Desired held categories determine retained yaw, so category0 does not
create a false release. Runs end at the first changed sign; gaps above 250 ms,
nonincreasing timestamps and final right-censored runs are excluded. Quantiles
use sorted index `floor((N-1)*p)`. Arm quantiles pool individual within-round
samples without joining runs across round boundaries.

Native `REK_Input` prefixes are joined to the strictly preceding ready source
within 50 ms. Six prefixes per arm lack that source and remain only in overall
native statistics. Busy is the saved `dispatched_request_v4_duration` projection,
which can differ from the bridge's local busy lifecycle. Retained snapshots can
miss changes between ready observations; native prefixes measure outgoing
requests rather than a continuous physical trajectory. The first observed sign
starts a run, matching the original tool, so an already active initial run can
be left-censored. Longer or busier rounds contribute more samples to pooled
quantiles.

Three rounds per arm cannot establish a stable treatment effect. Opponent
trajectories, attack mix, positions and rendered frame timing differ. Invocation
and projected request bodies do not establish server delivery, execution,
physical heading response or contact. No human recording was read.

## Retained evidence

Private directory:
`C:\rekagent\work\consistent-fighter-20260919-r1\yaw-command-mechanism-r1`.
`measurement-r1.json` contains per-round and per-arm results plus exact paths,
byte counts and SHA256 values for every result, ownership, encoder-input,
encoder-output and native-capture file consumed.

- Measurement SHA256: `e4a430c64290e1891fba7f22f1137942bc529b86b48aa1f55cd69a8a528106e3`.
- Private `measure-closed-r37-r42.cjs` SHA256: `5bd75ce4b3a66e40ff179d581afa2ac526242587de9b2b3b8b25022ab13cfc2f`.
- `attack-context-r1.json` SHA256: `190327a1d9248207a69c08e0e970557fd75b92631d38fc49e4c3d197ca7a843a`; it preserves all six existing derived-summary paths and SHA256 values plus its collector hash.

Native capture SHA256 values, bound to each round's owned PID:

| Round / PID | SHA256 |
| --- | --- |
| r37 / 5892 | `2af6c358954307f8d64dde5d9d0b7cde87deaa68935deb86e84a79bd41b35d38` |
| r38 / 17272 | `f37e3ef045a3ccb789b98529f34e2330ab3c8da15029b7bab3c6cea72cd5988b` |
| r39 / 256488 | `3fcc5b96f44206a61657be04abfcb385d4f5964abafac664cc83eca994966445` |
| r40 / 353040 | `42ccb0506178f7d3e1b729185172b7f484dcc3100e85d02181341e24e4a9a857` |
| r41 / 373992 | `2776234af9720342fe4dc2fe3296ed4e709552109ef3c659beb9699ba3c7dba6` |
| r42 / 392048 | `5c7d8b899bd9bed21f237dbaaf018c3e2daaee33eb4648935f2beb7cd2ab8934` |
