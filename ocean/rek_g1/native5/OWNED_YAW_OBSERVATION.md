# Owned pending yaw observation, version 2

This additive interface exposes previously hidden owned command intent. It does
not change physics, attack duration, action legality, rewards, or the meaning of
native dispatch acknowledgments. No improved fighting result is established by
the software tests or throughput measurement below.

## Contract

Legacy remains the default: `rek.native5.scaled_polar_xy.v1`, 223 observations,
33 actions, column 187 zero. Explicit opt-in is
`rek.native5.scaled_polar_xy.owned_yaw_v2`.

| Condition | Column 178, unchanged | New column 187 |
|---|---|---|
| Busy, owned retained yaw +1 or -1 | 0 | Owned desired yaw |
| Not busy | Existing held yaw | 0 |
| Terminal | Existing terminal encoding | 0 |
| Compact native Bot-controlled row | Existing encoding | 0 |

For active live v2 sources, `stream_active` and a valid owned `desired_action`
from 1 through 15 are required. Categories 6/8/10/12/14 yield +1;
7/9/11/13/15 yield -1; the others yield zero. Policy output 0 means retain the
existing owned command. It is not a valid source desired category. Explicit
release 1 clears owned yaw. Neither transient native input zero nor an
unacknowledged prior request is used to invent a release or retained intent.

Busy uses the existing declared projection, with its existing uncertainty and
duration. The new feature is neither measured angular velocity nor proof of
server execution. The actor receives observations but does not receive its
sampled action as a recurrent input; memory alone does not reliably reconstruct
the hidden stochastic held command.

The live encoder takes `--observation-schema SCHEMA`. The CUDA worker and compact
training runtime take `REK_OBSERVATION_SCHEMA=SCHEMA`. Driver configuration,
encoder readiness, worker readiness, and each encoded request must agree.
Unknown schemas fail closed. The training wrapper records and forwards this
environment variable. V2 compact training rejects non-`semantic_cuda` backends
and every frozen-opponent checkpoint, including one with an unspecified schema.
Legacy frozen opponents therefore cannot silently receive new observations.

## Measured motivation

Same-checkpoint development rounds r21/r22/r23 had 2,235 / 2,778 / 2,780
pre-action rows with busy and nonzero owned desired yaw while legacy column 178
was zero, approximately 46.32 / 57.34 / 57.07 seconds. Their ordinary awarded
points were 12:5 / 1:9 / 4:9. Move-3 requests numbered 67 / 110 / 88, with median
absolute encoded opponent bearings 32.57 / 78.20 / 85.73 degrees over all attacks.
The poor rounds contained 52 / 44 attack requests at projected distance below
1 received-coordinate unit but at least 90 degrees from the encoded forward
direction. This uses the existing 1:1 coordinate projection; physical metre
calibration has not been established.

These measurements establish hidden owned state and poorly aligned requests in
the existing observation coordinates. They do not establish the native
controller's physical forward axis, server acceptance, move execution, causal
hit attribution, or a causal explanation for round outcomes. The compact
runtime still lacks authentic yaw-during-attack dynamics, balance, knockdown,
and countout dynamics. See the earlier
[action-interface diagnosis](validation/reward-objective-20260919/action-interface-diagnosis.md).

## Migration and replay requirement

`owned_yaw_trajectory_data.cjs` upgrades historical v1 decision rows only from
the exact same-source pre-action owned desired category and saved busy
projection. New `REKRL002`/`REKBR002` version-2 files retain the existing layouts;
legacy readers reject them. Rewards, timing, action support, recurrence, and
chosen actions remain unchanged. Missing provenance fails rather than being
filled from future actions.

Migration copies the flat FP32 checkpoint and zeros only the encoder input
weights for column 187. Original v1 data must have positive zero in that column.
The original checkpoint is preserved. For the r21-r23 upgrade, all 17,558 rows
passed exact native comparison of all 34 logits, chosen-action log probability,
and sampled action; 7,793 rows gained nonzero intent. This is an initialization
equivalence check, not an optimizer or fighting comparison.

## Software validation, 2026-09-20

- Live encoder: 4,786 assertions plus 145 hinge projection cases, no simulation.
- Paired CUDA compact-runtime fixture: 1,603 checks. Arena state is byte-identical;
  all observation columns except 187 are equal. Attack, yaw, hold, release,
  terminal, native Bot-owned, and nonbusy cases are covered.
- CPU schema/backend/frozen-opponent contract: 126 checks.
- Native worker: 126 legacy GPU assertions and 150 v2 GPU assertions; v2 parser
  130 assertions. Mixed schemas and invalid intent values are rejected.
- Existing selection/feature-mask regression: 96 recurrent decisions compared.
- Existing CUDA auto-reset regression: 592 checks, 12 graph replays.
- Driver: 37 tests, including per-frame schema rejection before any action.
- Private helper-copy configuration: 40 checks, including the prepared control
  and treatment configurations; no orchestration executed.
  The original helper SHA-256 remains
  `89b6e3740a6c3484def1e17930fbf6529acec22b199babf6b068cc8161592342`.
  Its copy differs only in candidate encoder/schema merge support; the test
  reconstructs the original text exactly outside that function and call.

Shared-header hashes are recorded by encoder, worker, runtime, and full-trainer
builds. The full patched native Puffer5 trainer compiled and linked successfully.

## Full training throughput, performance-only

One explicitly authorized v2 run used the same 33,554,432-transition workload,
512 arenas, horizon 512, minibatch 8,192, seed 419, and hyperparameters as the
[preceding measurement](validation/native-training-throughput-20260920/README.md).
Only the opt-in observation contract and its exactly migrated initialization
were changed. Both INI files matched the preceding build byte-for-byte.

| Measurement | Result |
|---|---:|
| Trainer-loop time | 36.2613492012 s |
| Whole-training throughput | 925,349.7936 transitions/s |
| Process wall time | 36.95 s |
| Startup-inclusive throughput | 908,103.7077 transitions/s |
| Existing rollout-graph timer | 27.9569250494 s |
| Existing training model timer | 7.6244838759 s |
| Existing training miscellaneous timer | 0.1597995195 s |
| Loop outside recorded CUDA components | 0.5201407564 s |
| Exit status / runtime failure bits | 0 / 0 |

These are existing timers, with the same initial-bin weighting and repeated-final
bin exclusion documented in the preceding report. Graph mode does not record a
separate inference/environment/copy split. Residual time is not assigned to a
specific subsystem. One run provides no uncertainty estimate or speed regression
claim. The performance-only output is not promoted for authentic evaluation.

The run logged the explicit v2 schema and verified step-zero readback against the
requested migrated checkpoint. The compact training wins are not held-out
strength. No game process or Windows input was involved in this measurement.

| Artifact | SHA-256 |
|---|---|
| Tested live worker | `321a24d6871799c724f8652b00efebba5e35017e4962304ab61c8c19b06ef1a1` |
| Tested live encoder | `24356728e6e4b6b0f8e652d1d5733b52dc80ecd181930fb35b5f36a57a3675c0` |
| Full native trainer | `a8790d8251796df02f9cb7b382abcfd8621344612a984743a4b808d536e8eff7` |
| Migrated initialization | `9c0cefb9776c6e30b120ae9feeae94756833c653eb9de1fd508f25c554064f55` |
| Performance-only final checkpoint | `ffecc1660d3721e396473130284cf4806230536affd9eef8ceaa38bbf52035f7` |
| Extended private helper | `469ec2192bf22266b73e74b8f3285e0030d0e01a9416117879c25e167972e5f2` |

Private Spark root:
`/home/spark-advantage/rek-training/owned-yaw-observation-20260920-r1`.
Tested inference builds are `worker-build` and `encoder-build`; their immutable
source snapshot is `source/`. `source-r2/` additionally includes the CPU training
binding helper and final driver fixture; it built `full-trainer-build/`.
The performance run is `train-performance-only-v2-r1/`.

Private Windows mirror:
`C:\rekagent\work\consistent-fighter-20260919-r1\owned-yaw-observation\spark-results`.
All 807 files, totaling 26,638,457 bytes, passed independent source/destination
SHA-256 comparison. Sibling files preserve the performance command, offline timer
aggregation, helper configuration test, and copy manifest. No checkpoint,
proprietary asset, account record, or raw recording is included here.
