# Recorded human-pose contact replay

Executed on `spark-4ae3` on 17 September 2026. Both completed human rounds were used, without fitting either round. The same fixed diagnostic was applied to round two.

**First-pass limitation discovered after execution:** native scoring also accepts body zones 12 and 13, the left and right hips. This pass only tests the three original core targets. Its results are retained as a diagnostic of that incomplete candidate, not a complete reconstruction of native eligible contacts. The next replay must include the recovered hip-zone geoms. The pinned XML has zero `margin` and `gap` on all 91 geoms.

## Finding

The current geometric reconstruction does not yet explain all observed scoring contacts. Bone dimensions agree with the recovered model, but sparse receipt-time poses do not establish the original contact time or scoring decision.

| Window about score-packet receipt | Events with any primitive overlap | Events with kick/punch-compatible primitive overlap |
| --- | ---: | ---: |
| At receipt | 5 / 33 | 3 / 33 |
| Within plus/minus 50 ms | 19 / 33 | 17 / 33 |
| Within plus/minus 100 ms | 20 / 33 | 17 / 33 |
| Within plus/minus 250 ms | 22 / 33 | 19 / 33 |

The denominator comprises 22 local-player and 11 AI strike/score pairs. The six five-point awards without associated hit packets were kept separate. Each pair is unique within its observed client frame and fixed tick. No absent hit packet was assigned a miss label.

Round one had 1/18 compatible overlaps at receipt and 10/18 within 250 ms. Round two had 2/15 and 9/15 respectively. Expanding the receipt window increases opportunities to find overlap; it does not establish the server latency or identify the accepted contact. The limb-category flag does not identify the active move.

For comparison, the legacy enclosing spheres produced 21/33 compatible overlaps at receipt and 24/33 within 250 ms. This positive-only sample cannot measure false positives, select the better scoring model, or justify enlarged colliders.

## Method

The native [replay program](../../pose_contact_replay.cu) reads each captured bone's actual world position and quaternion. It does not use canned motion clips, forward-kinematic joint reconstruction, fitted root offsets or fitted spatial scale. The verified conversion is Unity position `(x,y,z)` to MuJoCo `(x,z,y)` and Unity quaternion `(x,y,z,w)` to MuJoCo `(-w,x,z,y)`.

Each compiled collider's local offset and orientation are transformed using its captured body world pose. Source/export comparison verified all 74 fighter geoms against the captured asset probe, with local dimension/pose differences below `1e-12`. The test uses 12 striker primitives against the opposing pelvis, torso and head primitives. Contact predicates run in CUDA, using the production primitive-overlap implementation with zero margin. CPU work loads JSON/model metadata and prepares rigid transforms; no CPU physics is stepped and no Python is executed.

The query times were fixed before viewing results: score receipt plus `k/120` seconds, for integer `k` from -30 through 30. At each query, the latest preceding received pose is selected independently for each fighter, rejecting ages over 75 ms. This is a discrete received-pose scan, not interpolation or continuous collision detection. A denser query grid can select the same received pose repeatedly.

Across both rounds, each fighter supplied 3,598 bone packets per round. All 28 nonzero parent-to-child rest lengths were checked against every available pose. Observed/model length ratios ranged from **0.999978845 to 1.00003443**. This supports consistent captured/model scale; independent physical metre calibration was not performed.

## Execution and reproducibility

- 33 strike/score pairs; zero unpaired hit packets; unpaired score counts `[4, 2]`.
- 2,013 GPU geometry queries.
- Process exit code 0.
- Compute Sanitizer: zero errors.
- Normal and sanitizer runs produced byte-identical query files and stdout.
- Query-file SHA-256: `9c45c4c518b09ea00ac6504b0984aa8a174b55eb5151feb5e3ef61e63a5f2eae`.
- Model and both capture hashes are pinned in [run-pose-contact-replay.sh](run-pose-contact-replay.sh) and [pose-contact-summary.json](pose-contact-summary.json).

Private Spark run: `/home/spark-advantage/rek-training/human-pose-replay-20260917-r1/run-r2`. Commands, stdout/stderr, input/output hashes and the failed initial linker attempt remain preserved. The initial attempt passed the versioned MuJoCo library directly to `nvcc`; passing it through `-Xlinker` corrected that build invocation. No replay algorithm changes were needed.

[summarize-pose-contact.cjs](summarize-pose-contact.cjs) exports the fixed-window aggregate and all 33 event summaries. It publishes no raw bone poses, proprietary assets, captures or checkpoint binaries. Point-to-striker and point-to-target window minima are separate diagnostics and may occur at different sampled times.

## Consequence for training

The candidate policy's 512/512 simulator wins remain an internal result. This replay does not validate its scoring model against authentic REK. The immediate correction is to include all recovered eligible target zones. Then recover or constrain the relationship between received motion, contact eligibility and accepted scoring before treating contact geometry as established. Per-fighter pose delay, temporal undersampling, native contact-entry rules and active-action attribution remain relevant unknowns. Missing balance, attack root motion and knockout dynamics also remain.

The isolated Spark client was relaunched for authentic evaluation. It stalled before bridge initialization and logged a Wine graphics-buffer `GL_OUT_OF_MEMORY` error. The process remained alive, with no usable scene state. No arena entry or policy input was sent, and Windows input was untouched. Exact startup causality remains unresolved. No new authentic match is included in these results.
