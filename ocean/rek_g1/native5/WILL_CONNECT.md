# Attack-reach diagnostic

The native evaluator and custom browser viewer now show separate **Left jab**
and **Right jab** reach estimates. The source is the user-supplied
`will_connect.h`, with reviewed numerical corrections. All numbered attachment
copies were byte-identical. Originals, plots, review and test evidence are
preserved on the physical server under
`rek-evidence/2026-09-27/will-connect-integration-r1`.

This is an **uncalibrated geometric diagnostic**. The supplied shoulder offsets,
reach, timing, auto-aim, glove size, upright hurtboxes and clean-contact speed
are guesses. Actual REK scoring includes other body zones and additional rules.
Green means predicted contact under those assumptions. It does not establish a
scored point, calibrated success probability or official environment parity.

## Data path

`RekNative5Snapshot` -> `will_connect_diagnostics.h` -> `will_connect_json.h`
-> `state.willConnect` -> browser HUD.

Inputs are both fighters' root XYZ/WXYZ, world linear velocities, root-local
angular velocities, legal-action masks and round/terminal/failure flags from
one synchronized snapshot. Root-local +X is the established G1 facing axis.
Angular velocity is rotated into world space and converted to the derivative
of horizontal facing, including root tilt. No finite differences or future
frames are required.

Outputs carry schema `rek.will_connect.diagnostic.v1`, `calibration: placeholder`,
`guardMode: unavailable`, `officialScoringValidated: false`, the input body
values, and per-attack outcome, lamp, signed gap margin in m, contact time in s,
relative contact speed in m/s, predicted point, target and exclusion reason.
Invalid/unavailable metrics serialize as null.

Native registry positions 5/8 correspond to catalog moves 1/4 and categorical
actions **21/24**. The supplied U/I aliases are retained only in the standalone
core's original API. The viewer uses semantic names. Its existing U/I kick
bindings remain intact; the saved official custom profile separately uses K/L
for jabs. No keyboard or controller mapping changes.

Inactive rounds, terminal/failing snapshots, malformed masks, unavailable
attacks, invalid poses, root tilt above the proxy's assumed 0.35 rad limit,
and overlap of its placeholder torso capsules suppress estimates. That tilt
limit is an applicability assumption, not an official fall threshold.
The simplified `semantic_cuda` backend is explicitly unsupported: its qvel
does not implement the physical free-joint velocity contract. The browser also
hides stale running results, failed connections and terminal results.

## Geometry corrections and limits

- Auto-aim now accounts for attacker translation during extension, restoring
  invariance when the same constant velocity is added to both fighters.
- Nonfinite/invalid input returns INVALID/OFF; sweep work is capped at 256
  intervals before integer conversion.
- A curved-path chord intersection cannot invent a contact time without a
  true inside endpoint for refinement.
- Broad rejection respects a configured amber band.

Turning still approximates a curved fist trajectory with chords and can miss
near-tangent contacts. A negative chord margin can coexist with MISS/AMBER.
The model assumes constant root velocities and a prescribed extension, omits
articulated punch dynamics, and receives no guard-glove data in this adapter.

The existing 223-float policy input, action space, reward, motor controller,
physics and runtime state are untouched. These values are not fed into a
policy or used to award points. The attached raylib HUD was replaced with
equivalent browser presentation because this evaluator renders through EGL.

## Build and checks

Normal evaluator builds include the feature through `eval_worker.cpp` and pin
the three diagnostic headers in `build_eval.sh`'s hash receipt. The checked
Spark binary is compiled and linked from the corrected current runtime plus
pinned existing physical objects. It was not executed or substituted into a
running service.

For CPU-only archived snapshots:

```sh
gcc -std=c99 -O2 -c vendor/cJSON.c -o /tmp/rek-wc-json.o
g++ -std=c++17 -O2 ocean/rek_g1/native5/will_connect_snapshot.cpp /tmp/rek-wc-json.o -o /tmp/rek-wc-snapshot
/tmp/rek-wc-snapshot < snapshots.jsonl > predictions.jsonl
```

Each input line has `qpos[72]`, `qvel[70]`, binary `mask[66]`, `phase`,
`terminal` (0/1), and `failureBits`. The CLI accepts physical MuJoCo snapshots
only. Keep source hashes and per-line identities alongside derived output.
`test_will_connect_replay.py` does that for the archived continuous-response
format; because those traces lack action masks, its eligibility assumptions
are explicitly synthetic.

Verification on 27 September 2026:

- Original C suite: 34 passed; corrective regression suite: 30 passed. Both
  also passed address/undefined-behavior sanitizers.
- C++ snapshot tests cover quaternion conversion, masks, both sides, state
  exclusions, common-velocity invariance and JSON nulls.
- Two closed simulator traces: 908 snapshots, 3,632 estimates, all MISS at
  their separated positions. No attacks or actual hit labels are present;
  this is a data-path check, not accuracy validation.
- Browser suite: 33 passed. Synthetic desktop/mobile screenshots reviewed.
- Spark native evaluator compiled and linked successfully. No gameplay,
  live UI input, service replacement or policy training occurred.

Calibration requires causal samples before actual jab requests, verified move
identity and pose age, measured glove paths/timing and target geometry, and
independent whole-match validation against received contact/scoring events.
Preserve point awards, falls and other attacks as distinct outcomes when
attributing results. Do not tune on held-out matches or promote geometric
contact estimates into scoring labels.
