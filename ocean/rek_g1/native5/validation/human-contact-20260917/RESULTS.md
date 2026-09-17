# Primitive-contact training continuation, 17 September 2026

## Result

A native CUDA PPO continuation completed on `spark-4ae3` with recovered individual collision primitives. It processed **67,108,864 transitions in 54.434910 s**, or **1,232,828 training SPS**, using 512 arenas and horizon 128. Process wall time including startup and exit was 55.16 s. No Python runtime or CPU physics stepping was used; CPU asset loading/FK, launch orchestration and file I/O remain.

The frozen checkpoint evaluation used 64 arenas, four rounds per arena on each side, randomized fixtures, seed 200019, sampled BF16 inference, 120-second rounds and the reconstructed Bot 1 controller:

| Checkpoint | Wins / losses / draws | Policy points | Opponent points |
| --- | --- | ---: | ---: |
| Before continuation | 510 / 2 / 0 | 67,649 | 18,357 |
| After continuation | 512 / 0 / 0 | 75,189 | 18,329 |

This is simulator evaluation. **No authentic REK match has been run with the new checkpoint.** The observed 512/512 result does not establish a perfect population win rate, superhuman play, or transfer to REK.

The [numeric result bundle](results/README.md) contains all 1,024 held-out match records, all 64 geometry-comparison matches, 64 native training metric samples, checkpoint identities and verification results. The [exporter](export-results.cjs) checks schemas and recomputes totals before writing the public subset. Private captures, game assets and checkpoint binaries are excluded.

The before/after runs share the seed and all 512 match keys. Six recorded `initial_xy` rows differ by at most 0.001600027 m. They must not be described as byte-identical recorded starting positions. The four geometry variants have identical recorded initial coordinates. See the result bundle for the measurement qualification.

## Human evidence

Both completed Windows human rounds were retained: 186 named move requests, 22 local strike/score pairs, 11 AI strike/score pairs and six additional five-point awards without paired strike effects. The unique same-client-frame pairs are useful observations, not a server action/causality acknowledgement.

The measured local successful-contact separation was 0.454 to 0.683 Unity units for one-point contacts and 0.416 to 0.681 for two-point contacts. These are sample extrema across different moves and poses, **not validated hit boundaries**. Unity-unit to metre calibration was not independently measured. The native G1 ground forward convention resolves to horizontal pelvis-local +X.

Four relatively isolated request/contact candidates were identified. Four other strikes contradict the nominal limb category of the most recent request, so latest-request labeling is unsafe. Absence of a hit packet was not converted into a miss. These rounds were not fitted to PPO or used as imitation labels. Round two remains reserved for future replay validation.

Private human report:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-17\human-gameplay-win10-0731\training-signal-analysis\contact-signal-report.md`

## Runtime change

The old compact runtime replaced each limb or target group with an enclosing sphere. The opt-in `REK_FAST_GEOMETRY=primitive_samples_v1` mode retains:

- Four distinct authored 0.005 m radius spheres per foot.
- One oriented box per hand.
- One capsule per shin.
- Pelvis and torso boxes plus the head capsule.

There are 12 striker primitives against three targets per fighter. The existing enclosing spheres serve only as conservative broadphase tests in this mode. Static narrowphase covers all six sphere/capsule/box pairs, including exact piecewise-quadratic segment-to-box distance and full 15-axis box SAT. Production contact margin is zero.

`REK_FAST_CONTACT_SUBSTEPS=8` samples rigid-pose interpolation inside each 0.02 s environment step. This leaves the control rate at 50 Hz and performs eight contact samples, not eight integrated physics steps. It is not continuous collision detection. Pose interpolation cannot restore controller tracking, contact deflection or missing dynamics.

Legacy mode remains the default. Runtime JSON fields are `fast.geometry_mode` and `fast.contact_substeps`; explicit configuration overrides ambient values, and evaluation records include the geometry identity.

## Paired geometry experiment

Same initial checkpoint, seed 9171058, four arenas and two rounds on each side:

| Geometry | Policy points | Opponent points | Wins |
| --- | ---: | ---: | ---: |
| Legacy enclosing spheres | 2,924 | 764 | 16/16 |
| Individual primitives, one sample | 1,717 | 475 | 16/16 |
| Individual primitives, four samples | 1,861 | 504 | 16/16 |
| Individual primitives, eight samples | 1,880 | 513 | 16/16 |

Eight-sample policy points are 35.7% lower than the sphere model. This establishes a material behavioral/scoring change, not a measured false-positive contact rate: changed scores and observations can change subsequent policy actions. Four versus eight samples also differ; strict temporal convergence has not been established.

## Verification and integration failure

- 78,184 native host checks passed, including an independent double-precision distance oracle, rotations, degeneracies and motion interpolation. Address/undefined-behavior sanitizers also passed on WSL.
- 12,288 Spark GPU cases produced 73,728 matching host/device comparisons across all six primitive pair types. A test-only 0.0002 m boundary filter excluded 24 numerically ambiguous cases; production uses no margin.
- Recovered-asset baking passed with zero calls to CPU physics stepping/forward dynamics.
- The first integrated CUDA build failed with an invalid local-memory read before geometry processing. PTX treated a by-value View descriptor parameter as local storage without copying it. Explicit grid-constant kernel descriptors and const-reference device helper arguments eliminated the failing path. The failed builds and sanitizer output were retained.
- The rebuilt legacy mode exactly reproduced the earlier per-side scores and behavior summary.
- Compute Sanitizer reported zero errors for a bounded eight-sample integrated run.
- Training reported zero failure bits. Its step-zero checkpoint exactly matched the requested warm start. Optimizer state was newly initialized.

The initial 256-arena baseline allocation failure was also preserved. Its cause was not established. Later 512-arena training and 64-arena evaluation succeeded; unrelated GPU jobs were left untouched.

## Paths and reproduction

Private Spark experiment: `/home/spark-advantage/rek-training/primitive-contact-20260917-r1`

Final trainer: `/home/spark-advantage/rek-training/primitive-contact-20260917-r1/build-r3/puffer-rek-native5`

Final evaluator: `/home/spark-advantage/rek-training/primitive-contact-20260917-r1/eval-r3/diverse-policy-eval`

Checkpoint: `/home/spark-advantage/rek-training/primitive-contact-20260917-r1/train-primitive8-r1/checkpoints/rek_native5/train-primitive8-r1/0000000067108864.bin`

Checkpoint SHA-256: `5989fa23e6ead72a20fa94a55ce4fb5da2db8631d4f2d2fb038a57c02f95ae85`

The experiment directory contains complete commands, stdout/stderr, trainer INI metrics, build identities, checkpoints and per-match JSONL. `train-primitive8-r1/command.txt` is the exact training invocation. Initial and final frozen comparisons are `eval-heldout512-old` and `eval-heldout512-new`.

For a fresh source stage, build with `build_fast.sh NEW_BUILD`, then use `run-checks.sh STAGE` for the paired checks against the pinned prior private checkpoint. The checked-in JSON files configure one, four and eight contact samples. A training continuation requires the explicit geometry flags above; no default was silently changed.

## Next empirical work

Replay the recorded bone poses and score events through the contact detector, reserving round two as holdout. Use the measured positive events and preserve ambiguous action attribution. This checks geometry against actual observed motion rather than assuming canned poses coincide with the client.

Then evaluate the frozen checkpoint against an actual private-arena REK AI, recording score, confirmed hits, facing error, distance and accepted actions. The compact environment still lacks observed root displacement during attacks, physical balance/falls/KO and authoritative contact-point velocity/contact-enter timing. Additional PPO cannot establish those missing mechanics. Authentic transfer remains the criterion.
