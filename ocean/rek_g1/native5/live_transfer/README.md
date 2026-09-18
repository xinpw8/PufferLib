# Native live observation projection

`encode-live` translates actual REK client telemetry into the selected V4
policy's 223-value `scaled_polar_xy` observation and starts with the source's
33-action transport mask. An optional declared request-duration projection
further restricts that mask during projected attacks. Retained held translation
also closes attack entries, even when the visual client's instantaneous velocity
has been cleared. This matches the training action contract and only removes
eligibility; source restrictions are never relaxed. It does not step MuJoCo,
the compact candidate, or any other
physics environment. MuJoCo is used once at startup to compile the private
recovered XML and read body rest orientations, hinge axes, and actuator order.
CUDA policy inference is a separate process. Game execution remains in REK.

This is the explicit `client_pose_projection_v1` transfer experiment. It is not
an authoritative server-state reconstruction. Both the startup manifest and
each projected result identify the projection. The CLI refuses to run without
the explicit projection argument.

## Build and JSON-lines interface

```sh
bash ocean/rek_g1/native5/live_transfer/build_encoder.sh /private/build
/private/build/encode-live --model /private/model.two_fighter_arena.xml \
  --projection client_pose_projection_v1 \
  --busy-projection dispatched_request_v4_duration
```

There is no Python dependency. The startup manifest contains a per-index source
inventory for all 223 fields. Subsequent stdin lines are `g1_policy_state`
records from the bridge's `rek.g1_policy_source.v1` schema. Every stdout record
is flushed immediately. The orchestrator forwards only the nested
`worker_request` of a `ready:true` result to the CUDA policy worker:

```json
{"event":"policy_observation","ready":true,"projection":"client_pose_projection_v1","worker_request":{"type":"step","seq":2,"round_id":"64 lowercase hex characters","observation_schema":"rek.native5.scaled_polar_xy.v1","observation":[223],"mask":[33],"terminal":false},"provenance":{}}
```

The array shorthand above indicates lengths, not valid request contents.
`ready:false` contains an `unavailable` reason and no worker request. Required
measurements are never replaced with zero when absent. Source sequence numbers
must increase across the process. A restarted producer requires restarting both
encoder and worker. `{"type":"reset"}` clears derivative and hit-proxy history
without rewinding the sequence check. `{"type":"close"}` closes the encoder.

The first sample and every new round warm up pose differences. Normal inference
requires a second sample with positive producer-QPC delta at most 250 ms. Gaps
rewarm the derivative history. A terminal record produces `terminal:true`; the
worker acknowledges and clears recurrent state without selecting an action.
Its observation values are not consumed for inference. The next round again
requires fresh pose samples. Control lease acquisition, action age validation,
game input dispatch, and round restarts belong to the separate bridge and
orchestrator. This executable has no input-dispatch capability.

## Feature mapping and limitations

| Slots | Source and interpretation |
| --- | --- |
| Each fighter 0..8, 12 | Measured root position/quaternion; derived horizontal local velocity and yaw velocity from consecutive poses. Unity position `(x,y,z)` becomes `(x,z,y)`; quaternion XYZW becomes WXYZ `(-w,x,z,y)`. |
| Each fighter 13..41 | Client bone-local rotation projected onto each recovered hinge axis relative to its model rest orientation. This is a pose projection, not a measured server joint coordinate. |
| Each fighter 42..70 | Wrapped finite difference of those projected angles divided by observed time. |
| Each fighter 71,72,73,77,79 | Measured root height at 73. Other slots preserve the V4 definitions: zero down/tilt flags and literal contact feature 2. V4 has no knockdown transition; these fields were constant in training. Actual fallen/tilt/contact values remain in provenance and the raw source log. |
| Each fighter 9..11,74..76,78,80..85 | Explicit zero constants for features the V4 candidate does not model. A constant does not assert a zero measurement in the authentic game. |
| 86,87 | Opponent root distance and relative bearing/pi replace opponent absolute X/Y. |
| 172..183 | Measured actor yaw, owned desired held category, native settled predicates, declared busy projection, and move route. Outside owned control the observed velocity-command sign supplies the held projection. Route uses an exact native move association if supplied; otherwise an acknowledged client request is labeled as a request projection. Unknown busy route blocks inference. |
| 184..195 | Local slot, semantic phase mapping, episode-local constant 1, duration/remaining time divided by 120, measured cumulative integer awarded-point totals from `round.clean_hits`, V4 constant-zero falls, terminal winner. Authentic RoundActive phase 1 maps to candidate active phase 2; a completed round maps to candidate terminal phase 4. Native totals include clean-strike and referee awards. The encoder's values and binary are unchanged. |
| 196..201 | Observation-window last-hit proxy from an awarded-point counter increase and the opposite fighter's maximum observed foot/wrist/knee speed. Referee awards can also increase this counter. Neither a clean strike, the striking limb nor actual impact velocity is established by this proxy. History begins at the stream window; initial false means no proxy hit observed in that window. |
| 202..205 | V4 constant-zero down-state slots; authentic fallen flags remain in provenance. |
| 209..214 | Explicit candidate reset-duration feature 0.5, then measured native round/fight result and winner enums. The 0.5 is not an assertion about an authentic reset duration. |
| 217,218,221,222 | Observed awarded-point counter increases and their sum since the previous sample, retaining the raw `clean_hits` field name. These are point deltas, not counts of strike events. Multiple authentic ticks between samples are not reconstructed. |
| Other trailing slots | Explicit unused V4 zero constants, enumerated in the manifest. |

The encoder also reports the largest off-axis component of each fighter's
projected joint rotations. This is a diagnostic of the pose projection, not a
fitted correction or server-ground-truth residual. Quaternion sign changes are
handled by the wrapped hinge difference; global root positions are not recentered
or rescaled. Local actor slot and absolute root position remain policy inputs.

The [counter provenance note](SCORE_COUNTER_PROVENANCE.md) records the native
integer accumulation and packet assignment. Existing encoder-manifest wording
that calls these clean-hit counts or says the scoreboard formula is unknown is
stale pending a metadata-only rebuild. This documentation
and summary correction does not change inference behavior or the live contract.

The native source currently cannot identify `SonicPolicyRunner.currentMotion`
as a named RobotConfig move: it is a different native type. The bridge therefore
emits a null move identity. Its `requested_move_index` records the latest
acknowledged client command. Visual-only clients also lack a usable native busy
flag: local `IsPunching` stays false while the server drives playback. The raw
`action_busy` therefore remains null and the server's acceptance stays unknown.

The optional `--busy-projection dispatched_request_v4_duration` argument selects
a bounded client-request timing projection. It requires the exact request QPC
timestamp and the native send method having returned. The lock lasts the
selected V4 move's configured duration from that request timestamp. At 50 Hz,
move indices 0..16 use ticks
`35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103`, exactly as in
`puffer_env.cu` and `eval_worker.cpp`. This table and its origin are emitted in
the startup manifest. It is a candidate definition, not a measurement of
authentic server playback duration. During this projected lock, only categories
0,1,6,7 remain eligible, intersected with the raw source mask. Thus repeated
attacks and translations cannot stack during the declared lock. Measured death
or a new round cancels it. The same move can start a new lock only with a new
request timestamp. Raw local flags and the derived lock are separately logged.
Without this opt-in, unavailable native busy state prevents inference.

## Validation

```sh
/private/build/encoder-test /private/model.two_fighter_arena.xml
/private/build/encode-live --model /private/model.two_fighter_arena.xml \
  --projection client_pose_projection_v1 --self-test
```

The native test checks 145 mathematical hinge cases and 890 observation/JSON
assertions, including all held categories with transient zero native velocity,
source-mask restriction preservation, warmup, missing-field rejection, timestamp bounds,
duplicates, terminal handoff, route provenance, request-duration locks and mask
restrictions, measured death, repeated identical requests, and reset history. It uses
synthetic source poses constructed from the private model calibration and
prints only aggregate results. No joint arrays or private model are published.
These tests verify the adapter mathematics and contract. Actual client samples
and closed-loop behavior must be validated separately before claiming a working
transfer or authentic-game performance.

Coordinate and bone-order sources are the existing recovered model report,
`rek_unity_to_mujoco_arena_calibration.json`, and
`ocean/rek/evidence/windows/RekEvidenceRecorder/RecorderContract.cs`. The exact
source schema and action dispatch contract are documented in
`ocean/rek/evidence/windows/RekUiBridgeAgent/G1_POLICY_STREAM.md`.
