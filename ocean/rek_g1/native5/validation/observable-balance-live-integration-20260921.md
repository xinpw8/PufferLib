# Observable balance live encoder, 2026-09-21

The separate native `live_transfer/encode_observable_balance.cpp` implements
`rek.native5.observable_balance.v1` through the shared `observable_balance.h`
projection. Existing `encode_live.cpp`, its binary, its build script, and its
default observation schema are unchanged. Existing weights are incompatible with
this opt-in schema. This work did not train or evaluate a policy, connect to a
game, dispatch input, or step physics.

## Contract

The adapter transforms measured Unity roots into the common frame, then passes
the current and actual preceding observation to the shared projection. Root
height, quaternion-derived tilt, finite-difference horizontal/vertical velocity,
wrapped heading rate, received awarded points and their deltas are available.
First observations and invalid time intervals have explicit history availability
zero. Same-round body displacements remain in the history. No fall cause,
contact attribution, floor height or server-current state is inferred.

Both live and physical adapters use `joint_pose_available=0` until their joint
correspondence is established. Joint angles and rates are zero padding with
explicit availability flags. The manifest has 223 indexed field descriptions and
a numeric 223-element structural mask retaining 166 schema positions; structural
retention does not establish per-sample joint availability.

Received referee state requires the existing 33-byte packet hash/base64 binding,
decoded byte/slot-bit agreement, source round/redo agreement, positive matching
QPC frequency, receipt age at most 0.5 seconds, monotonic receipt/lifecycle
identity, stable repeated receipt contents, and stable latched call identity.
Unavailable receipts have explicit null fields and count availability zero.
The live adapter does not independently read the native recorder. Its manifest
states that authenticated bridge transport owns process binding and that the
adapter must restart for a new producer process.

Action gating retains the declared `dispatched_request_v4_duration` busy
projection and intersects its restrictions with the source transport mask,
held-translation restrictions, and the shared action-cadence contract. Provenance
preserves the source mask description and states that server playback acceptance
is unknown. This does not establish physical/live action-mask parity.

## Native verification

Fresh Spark build uses GCC for cJSON and C++17 g++ with warnings as errors,
linking OpenSSL for SHA256. Dynamic dependencies exclude Python, Torch, MuJoCo
and CUDA. Native black-box tests passed **92 cases and 17,262 assertions**,
covering geometry, history/reset boundaries, count bits, point counters,
receipt/call integrity, unavailable fields, action masks, cadence, and explicit
schema selection.

Final binary:

`/home/spark-advantage/rek-training/observable-balance-live-20260921-r1/build-r3/encode-observable-balance`

SHA256: `6eb359ba5b356881a081da4cf853d83027cb0beb8fece1e7ad487044c5db8a07`.

Build and test from the repository root:

```sh
bash ocean/rek_g1/native5/live_transfer/build_observable_encoder.sh NEW_BUILD_DIRECTORY
node ocean/rek_g1/native5/live_transfer/observable_encoder_test.cjs NEW_BUILD_DIRECTORY/encode-observable-balance PRIVATE_MODEL_XML
```

Invocation requires explicit opt-in:

```sh
encode-observable-balance --model PRIVATE_MODEL_XML \
  --projection client_pose_projection_v1 \
  --observation-schema rek.native5.observable_balance.v1 \
  --busy-projection dispatched_request_v4_duration
```

The model file is hashed for provenance; this adapter does not load it as a
physics model. The optional `--action-stride` accepts 1 or 5; default is 1.

## Authentic recorded-input replay

The encoder replayed recorded stdin from private rounds r64 and r65. Every source
record exactly matched a canonical JSON observation in the previously validated
relay. The original relay and independent native capture hashes were recomputed
and matched their validation receipts. Native capture PIDs were 66048 and 248816.

| Check | r64 | r65 |
|---|---:|---:|
| Recorded observations / ready outputs | 5,643 / 5,643 | 5,714 / 5,714 |
| History-available observations | 5,642 | 5,713 |
| Fresh received referee observations | 5,643 | 5,714 |
| Observations with nonzero measured tilt | 5,643 | 5,714 |
| Observations with nonzero vertical finite difference | 5,642 | 5,713 |
| Legacy-comparable masks, all exactly equal | 5,642 | 5,713 |

All output features were finite. Measured heights, awarded points, relative count
bits, joint padding/availability, structural padding and source-mask restrictions
passed per-record checks. Each round gained exactly its first observation, where
the new contract explicitly marks history unavailable. This mask comparison uses
the recorded default stride 1. Both actual native manifests passed the current
Windows driver's exported `validateEncoderReady` with the explicit balance schema.

These replays establish the live observation connection. They provide no new
policy performance measurement, authenticated live deployment test, or dynamics
fidelity claim.

## Private evidence preservation

NAS directory:

`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\observable-balance-live-r1`

`observable-balance-live-and-bindings-r1.tar.gz` contains 53 selected completed
files: final encoder source/build/tests/replay, input hash/native-validation
anchors, and the separate root worker binding build/protocol/GPU-test receipts
with six selected binding source files. Exact Windows driver source/test copies
and actual-manifest validation are preserved alongside it. Raw replay inputs and
binaries remain private.

Archive size: 48,167,055 bytes. SHA256:
`65bd63563c7c0146daf475669ce1cd079403a5de934d9dcb29d6cdb547520bb4`.

NAS receipt SHA256:
`451e16ea4d640bb454e888a2b65e8da678a9f43267f948aba458f91e6567100d`.

Source hashes before/after packaging, tar content comparison, Spark/local/NAS
archive hashes, and each NAS copy hash passed. All destination files were new;
there were no overwrites. Superseded encoder builds, duplicate compressed inputs,
the unrelated full worker source tree, and active Windows trials were excluded.
