# Optional recovered contact rules

`REK_FAST_SCORING=v4_spheres` is the unchanged default. The explicit alternative
is `REK_FAST_SCORING=recovered_hit_rules_v1`. Unknown values fail startup.
Both retain the compact V4 movement, action, pose, geometry and round models.
This is a scoring-mode change, not a new physics backend or an authentic-parity
claim. Configuration and evaluation must record the selected mode.

Recovered mode requires the actual asset manifest build fingerprint and strike
source SHA, all route identities/configuration fields, and clip identities,
frame counts and sample rates to match the recovered catalog. The loader bakes
the catalog's 29 events and per-route event offsets/counts. The original
`g1_hit_detector.c` apex code is included verbatim in a namespaced inline
host/device helper; no replacement apex approximation is used.

For each existing compact swept contact-enter candidate, recovered mode uses:

- Maximum relative sphere-center speed among intersecting pelvis/torso/head
  target proxies, threshold 1.75 m/s. This is a declared kinematic proxy.
- Source-clip cursor reconstructed from the bake's playback rate and bounds,
  rounded and tested against the recovered limb-specific apex ramp, minimum 0.2.
- 0.30000001192092896 s per-striker-body cooldown, retained across moves.
- One accepted score per invocation/apex identity, shared across that fighter's
  striker bodies. A new invocation receives a new ID.
- One point for a hand strike, two for foot/shin strikes. Rewards use weighted
  own points minus opponent points; logged hit_count counts accepted contacts
  without weighting. Existing optional potential shaping remains separate.

The compact assumptions remain explicit: constant upright/no balance dynamics,
conservative sphere geometry, per-limb union contact latch reset at move start,
no collision blocking, native manifold or contact-point velocity. Last-hit
history remains an accepted-score proxy, not the native knockdown-attribution
gate. No authentic server contact fields are invented.

## Executed tests

Private artifacts: `/home/spark-advantage/rek-training/policy-quality-20260916-r1/contact-audit/recovered-build-r4`.
The test links the preserved V4 object under renamed APIs and its original
asset loader under a separate name. The candidate is a separate runtime.
Both receive identical actions selected by the same frozen checkpoint from
the preserved V4 observations, so scoring cannot perturb this matched replay.

Twelve deterministic host cases and twelve GPU cases passed: threshold reject
and equality acceptance, wrong limb, out-of-window/inactive apex, same-body
cooldown, cooldown persistence across invocations, duplicate apex, new
invocation, independent fighter, hand=1 and foot/shin=2. A 64-arena, 1000-tick,
seed-10001 scripted-opponent replay then passed:

| Check | Result |
| --- | --- |
| Default vs preserved V4 raw observations and rewards | Bitwise identical each tick |
| Default and recovered qpos/action masks vs preserved V4 | Bitwise identical each tick |
| Preserved/default weighted points | 2249 : 556 |
| Recovered weighted points, same actions/poses | 1197 : 514 |
| Policy wins under preserved score | 64/64 |
| Shadow wins under recovered score | 56/64 |

These shadow outcomes are not a closed-loop new-mode evaluation or a forecast
of authentic REK performance. The checkpoint still observes old scores in this
test. Fresh closed-loop evaluation and retraining must explicitly select the
new mode. No training was performed by this test.

The aggregated `fast_assets.o` contains the C11 catalog and static-route host
objects via `ld -r`. Existing consumers still link exactly the four usual
objects: fast_runtime, fast_assets, native_policy, cJSON. Public runtime ABI and
checkpoint layout are unchanged. The build script is `build_recovered_scoring.sh`.

Verified object SHA-256 values:

```
fast_runtime.o 5e0ed440c8f954ec0933616b98f39e9d4ad73b27ff83801a36ea460a7064dd87
fast_assets.o 5a50dfcb222ff2875820e6cfd389f3aa2f8396b9a07b04970bb8bb5ff605664f
test executable 61dd0fa60b56447f29cedf082cc29e41ca0ad847b542f63b1436df6590b2fa1d
```
