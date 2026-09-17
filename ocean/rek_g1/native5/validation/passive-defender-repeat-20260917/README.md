# Passive private-AI contact evidence, 2026-09-17

Four finalized native captures contain **31 received hit packets: 24 non-kick and 7 kick**, plus 35 score packets. This establishes repeated receipt of native hit effects during neutral-defender observations. It does not identify a canned AI clip or prove an exact action-position combination reliably causes contact.

[aggregate.json](aggregate.json) contains the checked counts, source paths and SHA256 values, per-hit native receipt clocks and decoded geometry, pose availability, and incomplete attempts. [human_annotations.json](human_annotations.json) preserves the user's R4 observations separately.

## Verified capture results

| Capture | Received hits, non-kick / kick | Score packets | Separate +5 awards | Awarded points, slots 0 / 1 | Neutral compact samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4 | 7 / 0 | 9 | 2 | 5 / 12 | 6,003 / 6,003 |
| R5 | 5 / 1 | 7 | 1 | 5 / 7 | 6,004 / 6,004 |
| Native follow-up round 2 | 11 / 2 | 13 | 0 | 0 / 15 | 6,006 / 6,006 |
| Native follow-up round 3 | 1 / 4 | 6 | 1 | 5 / 9 | 5,999 / 5,999 |

Every capture finalized on `scope_exit:round_not_active`, reports zero capture errors, and has packet counts matching its footer. Independent rehashing found no input-hash or count mismatch. All 24,012 compact samples show velocity `[0,0,0]`, `pending_move=false`, and `punching=false`. All 27,074 recorded outbound velocity requests are zero. Capture transport-method counts contain only `SendVelocityCommand`, with no recorded `SendMove` invocation. These observations leave the defender's physics active. The two native-only follow-ups lack a paired runner/relay record of a fresh neutral request.

All 31 hit packets have one same-Unity-frame score packet: 24 opponent-slot +1 awards and seven opponent-slot +2 awards. The `is_kick` field is respectively 0 and 1 in those records. Even a unique same-frame association is **noncausal**: the hit message has no fighter ID or shared authoritative contact ID. Scoring slot 1 is an attacker candidate, not proven attribution. The four +5 awards have no same-frame received hit; their causes are not classified here. The unreliable hit channel prevents interpreting an absent packet as proof of absent contact.

## Clocks and geometry

Every hit preserves the native Unity frame, Unity time, unscaled time, realtime receipt time, contact position, surface normal and relative speed. **None has per-hit QPC, UTC, or server impact time.** Root samples retain their measured QPC separately; no clock conversion or interpolation supplies a missing hit timestamp.

R4/R5 use the saved G1 source poses. All 13 hits have one prior sample, one next sample and one same-frame sample, with join status `bracketed`. Native-only follow-ups use `root_pose_sample`: all 18 hits retain prior/next samples and 8 or 9 same-frame roots, but report `inconsistent_unity_frame_time`. Native pose sampling occurs in FixedUpdate while packet receipt uses the rendered-frame clock. These are useful client pose alternatives, not an exact impact pose. Their sub-2 ms Unity-time offsets must not be described as impact-time precision.

Distances use Unity world XYZ in metres, with ground-plane distance in XZ. Bearing reuses the live-encoder-compatible projection of **root-local +X onto Unity XZ**. This is neither Unity +Z forward nor a measured controller target heading or aiming intention. Full positions and XYZW root quaternions remain in the join artifacts.

The saved contact-context reports give concrete reasons not to reduce contact acceptance to a distance or limb-speed threshold:

- R5 has a one-second `no_score_observed` motion window around 90.9 to 89.9 seconds remaining. Its initial root gap is 0.386 m and opponent bearing is 0.087 rad; a named bone origin reaches 8.506 m/s. Its first ordinary score context, around 110.1 seconds remaining, instead has a 0.638 m root gap. These are different evolving trajectories, not matched action trials or a rejected-contact label.
- R4 contains a `no_score_observed` window with median root gap 0.097 m and substantial tilt, and another with peak named-bone-origin speed 9.899 m/s. Close roots and fast visible motion alone do not identify a scored strike. Bone origins are not collider surfaces.

Known AI canned clip IDs, strike limb/side, target body zone, rejected attempts, authoritative collision pairs/manifolds and server impact time remain unavailable in these inputs. No exact action-position repeatability proof, inferred hitbox model or policy-improvement claim follows from this set.

## Incomplete attempts

| Attempt | Saved outcome | Collection and limitation |
| --- | --- | --- |
| R6 | Incomplete; safe-start required native Idle | 0 sources, 0 action requests; cleanup verified |
| R7 | Incomplete; `relay_exit` | 6.968 s, 249 sources, 249/249 neutral requests applied; cleanup unverified in trial summary |
| R8 | Incomplete; `relay_exit` | 9.810 s, 327 sources, 326 applied neutral requests, one stale-observation rejection; cleanup unverified in trial summary |
| R9 | Not started | Docker exec exited with code 29 at 07:22:28 UTC before the capture-disabled runner began; local config exists but no R9 trial directory |

R7/R8 are partial observations and are excluded from the four-capture totals. [runtime-exits.json](runtime-exits.json) separately records owned Docker exec exits with codes 5 and 29, distinct from the relay's zero exit status. The latest startup lasted 16 seconds and exited before testing. The underlying cause and code meanings remain unknown; the last logged stage does not establish a fault location. The native-only follow-up captures are separate artifacts and must not be relabelled as completed R6/R7 runner attempts.

## Exact provenance

All paths below are under `C:\rekagent\work\passive-defender-20260917-r1\`:

- `round-r4\native-hit-join-r2\{summary.json,hit_events.jsonl,score_events.jsonl,five_point_awards.jsonl}`
- `round-r5\native-hit-join-r2\{summary.json,hit_events.jsonl,score_events.jsonl,five_point_awards.jsonl}`
- `native-followup-round2\{summary.json,hit_events.jsonl,score_events.jsonl,five_point_awards.jsonl}`
- `native-followup-round3\{summary.json,hit_events.jsonl,score_events.jsonl,five_point_awards.jsonl}`
- R4/R5 `contact-context-r1.json`, and R6/R7/R8 `trial\summary.json`.

The four original recorder files are in that root's `native\` directory:

| Capture | Native recorder filename | SHA256 |
| --- | --- | --- |
| R4 | `rek-private-ai-root-motion-20260917T044648.8582898Z-pid32-a96b4bb32bb6423db2d6b5139be5b4e9.jsonl` | `0d3825e596ddba7a6c1b3c8095e1b980e9685ce40638b3014a807c9b1265b4b3` |
| R5 | `rek-private-ai-root-motion-20260917T070323.7099459Z-pid32-1b7328dddfe641a69ba4e8cc12a94529.jsonl` | `16295a3b3700e08c61fa5c126a4b223821dd4e6637c26d5d790096f556039c21` |
| Follow-up 2 | `rek-private-ai-root-motion-20260917T070533.4813127Z-pid32-6df2daa4c8b744f0a35a3f0fab1b7173.jsonl` | `179e226a3fa5a9ad19c6faa34d58808a0c756ddd33ffd9f9c1142f78379320fc` |
| Follow-up 3 | `rek-private-ai-root-motion-20260917T070746.9276754Z-pid32-b3024a47b2fc4ebd9fd16fceb719f6cc.jsonl` | `a9dd4e34a7eb2d1efeb052d6daccb0501217aa7e63cdde61ec03354c8a560bea` |

R4/R5 r2 joins use their respective `trial\g1_policy_state.jsonl` inputs. Their hit records match r1 exactly after excluding relay line numbers; r1 used `relay.stdout.jsonl`. The human annotation correctly hashes the original R4 relay stdout, so it intentionally differs from the r2 filtered-source hash. Both provenance paths were checked. The recorder DLL identity in all four captures is `a19f619c83eeecf9c6ccf79adf339be1f7f1cca8e3cd622f80616f268aaffa95`.

Saved six-file test outputs verify **90/90 passed on Windows and 90/90 on Spark**, both with empty stderr. Both local copies were read and hashed. These tests validate the offline/runner tooling, not causal contact attribution or repeatability.

The physical-server archive is `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-17\passive-defender-repeat-r1`. The initial 165 files, 1,036,564,524 bytes, were verified source-to-destination by SHA256; `verified-sha256.json` records that copy. Final reports, runtime diagnostics and the terminal-frame screenshot are subsequent additions.

R5's fully decoded MP4 is 11,033,570 bytes, below the 20,000,000-byte sharing
limit. Its SHA256 is `cef3eb985f3fdf7e1c60b1d1aec114db6390aa3fa95c329ec5dbe209866b66c6`.
The frame at video time 120 s was visually checked: round 1, 0:00 remaining,
player 5, Sparring Bot 1 score 7. This is a video clock, not a claimed exact
mapping to a native hit receipt. The screenshot SHA256 is
`c17d7cb2a7c0e89d47aa39de704b0d08ec7ba419e02aa0293cd43dfc4a331c80`.

The updated offline tools and explicit `follow_on` runner are staged at
`/home/spark-advantage/rek-training/passive-defender-20260917-r1/source-repeat-r1/`.
All live attempts described here used the earlier `source-final/` snapshot.
The new follow-on behavior passed fixtures on both hosts but has not been
validated by a completed live collection. Neither source change is a new
hitbox or a training reward change.
