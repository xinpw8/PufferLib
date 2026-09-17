# Correcting the missing G1 scoring targets

The recorded-pose test exposed a concrete omission in the compact environment: it only included three core target geoms. Native scoring also accepts the left and right hip zones, each mapped to three additional colliders in the G1 asset.

## Source-derived correction

Native `BodyZoneExtensions.IsScoring` and `HitDetector.TryScoreContact` accept body zones 1, 2, 3, 12 and 13. The G1 `importData.bodyZones` mapping resolves to nine target geoms. Its head-shaped capsule is attached to the torso and has zone 2; no separate zone-1 body is present. Six parsed `BodyPartTag` entries confirm the existing 12 striker geoms were already complete.

| Target | Geom suffix | Zone | Shape |
| --- | --- | ---: | --- |
| Pelvis | mjgeom_3021 | 3 | Box |
| Torso | mjgeom_3285 | 2 | Box |
| Head-shaped torso geom | mjgeom_3064 | 2 | Capsule |
| Left hip pitch | mjgeom_3337 | 12 | Box |
| Left hip roll | mjgeom_3141 | 12 | Box |
| Left hip yaw | mjgeom_3399 | 12 | Capsule |
| Right hip pitch | mjgeom_3024 | 13 | Box |
| Right hip roll | mjgeom_3062 | 13 | Box |
| Right hip yaw | mjgeom_3406 | 13 | Capsule |

The shared [target catalog](../../native_contact_geometry.h) is used by the CUDA environment and recorded-pose replay. All target geoms have enabled cross-fighter contact masks. The pinned model has zero margin and gap on all 91 geoms. Neither collider enlargement nor fitted angle/distance thresholds were introduced.

Private source evidence: `IsilDump/REKApp/REKApp/BodyZoneExtensions.txt`, `HitDetector.txt`, `mujoco_asset_probe_v8.json` and `fight_semantics_components_v2.json`. Build fingerprint: `f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659`. Model and capture hashes are included in the numeric reports.

## Measured contact replay

Same 33 observed strike/score pairs, both human rounds, same recorded poses, same 2,013 fixed queries. No fitted parameters were introduced.

| Receipt-time window | Original three targets: compatible overlap | Corrected nine targets: compatible overlap |
| --- | ---: | ---: |
| At receipt | 3 / 33 | 7 / 33 |
| Plus/minus 50 ms | 17 / 33 | 29 / 33 |
| Plus/minus 100 ms | 17 / 33 | 30 / 33 |
| Plus/minus 250 ms | 19 / 33 | 31 / 33 |

At 250 ms, round one increased from 10/18 to 17/18; round two increased from 9/15 to 14/15. Local-player events account for 20/22 corrected overlaps and AI events for 11/11. Missing hip targets therefore explain a substantial part of the earlier reconstruction gap.

Two events still have no compatible overlap in the sampled window: round one hit sequence 12 / score sequence 15, and round two hit sequence 5 / score sequence 7. Their minimum recorded hit-point distances to compatible striker shapes were approximately 0.01894 and 0.00852 model metres. This does not establish a needed collision margin. Temporal undersampling, per-fighter delay and authoritative timing remain unresolved.

The normal and Compute Sanitizer runs exited 0, with zero sanitizer errors and byte-identical outputs. A paired regression checked all 2,013 queries and 112 link-length rows: original core masks, legacy sphere counts and striker distances stayed exactly identical; target distances could only decrease. There were 231 queries with additional compatible pairs.

Private corrected run: `/home/spark-advantage/rek-training/human-pose-replay-20260917-r1/run-r4`. Query SHA-256: `66fca98e5c3a6f613fc743f6816caa223bb997834b7e41f960830b17c3a1f9fa`. The preceding out-of-memory attempt is preserved as `run-r3`. The successful retry followed cleanup of this task's stalled REK launch; unrelated processes were untouched.

[All-target event results](pose-contact-all-targets-summary.json) retain every event and fixed window. [First-pass results](pose-contact-summary.json) remain unchanged. See [the method and limitations](POSE-CONTACT.md).

## Runtime scope

The opt-in `primitive_samples_v1` branch now evaluates all nine target geoms. Asset baking verifies both fighters' geom/body/type associations and dimensions. Runtime diagnostics identify `target_contract=g1_scoring_bodyzones_1_2_3_12_13_v1` and effective target count 9. Legacy enclosing-sphere behavior retains its three targets and remains the default. Previous training/evaluation results predate this correction.

This establishes a source-backed target omission and an empirical improvement on observed poses. It does not establish server-time contact reconstruction, scoring acceptance, negative-example accuracy, motion/knockout parity or policy transfer to authentic REK.

## Runtime verification

- 78,248 host geometry checks and 73,728 host/device comparisons passed.
- The production CUDA scoring adapter passed the existing 12 acceptance fixtures and 36 new isolated-target/persistent-contact checks. All six hip targets score in primitive mode; legacy mode retains only its original core targets. Persistent contact does not score a second time in these fixtures.
- The recovered asset probe verified all nine targets, primitive dimensions and orientation matrices, with zero CPU physics calls.
- Rebuilt legacy mode reproduced the original 16/16 wins and 2,924:764 score exactly.
- Frozen checkpoint `5989fa23e6ead72a20fa94a55ce4fb5da2db8631d4f2d2fb038a57c02f95ae85`, evaluated with all nine targets, scored 75,022:22,294 over 512 games: 510 wins, two losses, zero draws, zero zero-hit games and zero failure bits. The evaluation used the same seed 200019 and 64 arenas, four 120-second rounds on each side. Execution time was 12.65 s; this is evaluation timing, not training SPS.

The earlier three-target result for this checkpoint was 512/512 and 75,189:18,329. Correcting collision eligibility therefore changes the policy's measured performance. The authentic REK outcome remains unmeasured.
