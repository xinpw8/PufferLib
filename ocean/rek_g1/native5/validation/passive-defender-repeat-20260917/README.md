# Passive private-AI contact evidence, 2026-09-17

Across the eight-capture cohort and the two later paired native captures,
**77 hit receipts have been checked: 56 non-kick and 21 kick**. A separate
29.9 s capture adds three non-kick receipts. Named attack-position repeatability
remains unproven. These runs use the authentic pinned Steam client on isolated
Spark Wine display `:98`; they are not accepted Windows parity traces. Windows
received no input.

Eight finalized, approximately 120 s native captures contain **65 received hit packets: 49 non-kick and 16 kick**, plus 79 score packets and 14 separate +5 awards. The original four captures and four post-startup-recovery captures are separate cohorts below. A further finalized 29.9 s capture contains three non-kick receipts and is excluded from those totals. This establishes repeated receipt of native hit effects during neutral-defender observations. It does not identify a canned AI clip or prove an exact action-position combination reliably causes contact.

[aggregate.json](aggregate.json) contains the checked counts, source paths and SHA256 values, per-hit native receipt clocks and decoded geometry, pose availability, and incomplete attempts. [human_annotations.json](human_annotations.json) preserves the user's R4 observations separately.

The later R11/R12 paired collections are documented separately in
[paired-recovery.md](paired-recovery.md) and are excluded from the eight-capture
aggregate above. Both reached native terminal state with neutral inputs and
verified lease release. The inspected opponent-action observability boundary is
documented in [opponent-action-observability.md](opponent-action-observability.md).

## Original capture results

| Capture | Received hits, non-kick / kick | Score packets | Separate +5 awards | Awarded points, slots 0 / 1 | Neutral compact samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| R4 | 7 / 0 | 9 | 2 | 5 / 12 | 6,003 / 6,003 |
| R5 | 5 / 1 | 7 | 1 | 5 / 7 | 6,004 / 6,004 |
| Native follow-up round 2 | 11 / 2 | 13 | 0 | 0 / 15 | 6,006 / 6,006 |
| Native follow-up round 3 | 1 / 4 | 6 | 1 | 5 / 9 | 5,999 / 5,999 |

Every original capture finalized on `scope_exit:round_not_active`, reports zero capture errors, and has packet counts matching its footer. Independent rehashing found no input-hash or count mismatch. All 24,012 compact samples show velocity `[0,0,0]`, `pending_move=false`, and `punching=false`. All 27,074 recorded outbound velocity requests are zero. Capture transport-method counts contain only `SendVelocityCommand`, with no recorded `SendMove` invocation. These observations leave the defender's physics active. The two native-only follow-ups lack a paired runner/relay record of a fresh neutral request.

All 31 original hit packets have one same-Unity-frame score packet: 24 opponent-slot +1 awards and seven opponent-slot +2 awards. The `is_kick` field is respectively 0 and 1 in those records. Even a unique same-frame association is **noncausal**: the hit message has no fighter ID or shared authoritative contact ID. Scoring slot 1 is an attacker candidate, not proven attribution. The four +5 awards have no same-frame received hit; their causes are not classified here. The unreliable hit channel prevents interpreting an absent packet as proof of absent contact.

## Recovery captures, extension v1

These native-only captures followed the isolated lazy-preload startup recovery described below. They are separate from R6/R7/R8/R9 and preserve the original four-capture totals in `cohort_totals.pre_recovery`.

| Recovery capture, UTC start | QPC duration, s | Received hits, non-kick / kick | Score packets | Separate +5 awards | Awarded points, slots 0 / 1 | Neutral compact samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 1, 08:09:50.509 | 119.862 | 8 / 3 | 13 | 2 | 10 / 14 | 6,001 / 6,001 |
| Round 2, 08:12:00.299 | 119.850 | 2 / 1 | 5 | 2 | 10 / 4 | 5,997 / 5,997 |
| Round 3, 08:14:10.032 | 119.996 | 6 / 2 | 11 | 3 | 10 / 15 | 6,004 / 6,004 |
| Round 4, 08:16:23.416 | 119.978 | 9 / 3 | 15 | 3 | 15 / 15 | 6,003 / 6,003 |

The recovery cohort adds **34 received hits: 25 non-kick and 9 kick**, 44 score packets and ten separate +5 awards. All 24,005 compact samples meet the same three neutral conditions, and all 28,370 recorded velocity requests are zero. Each footer contains only `SendVelocityCommand`. All four finalize on `scope_exit:round_not_active` with zero capture errors and matching footer packet counts. Streaming rehashes and independent raw-record counts match the join inputs; every raw decoded hit and score record matches its joined copy. These are sampled neutral-state and request observations, without paired runner/relay proof of a fresh neutral command.

Each of the 34 recovery hits has one noncausal same-frame score association: 25 slot-1 +1 awards and nine slot-1 +2 awards. The ten +5 awards have no same-frame hit packet. Their causes remain unclassified. Across the eight approximately 120 s captures, all 48,017 compact samples are neutral and all 55,444 recorded velocity requests are zero. There is no attack-identity or exact action-position repeatability result.

The separate `recovery-native-short1` capture runs from 08:18:33.2718815 to 08:19:03.1710875 UTC, lasting 29.8992073 s by native QPC. It finalizes on `scope_exit:round_not_active` with zero capture errors, three non-kick hits, three slot-1 +1 scores and no +5. All 1,499 compact samples are neutral and all 1,770 recorded velocity requests are zero. Its three same-frame associations remain noncausal. The footer establishes scope exit; the reason this active interval was shorter is unknown. It is retained under `short_captures` and excluded from full-capture totals. No active `.partial` file was copied or read for this extension. Other finalized recordings exist but have not been processed here and are not counted. R11 is a separate collection.

## Clocks and geometry

Every hit preserves the native Unity frame, Unity time, unscaled time, realtime receipt time, contact position, surface normal and relative speed. **None has per-hit QPC, UTC, or server impact time.** Root samples retain their measured QPC separately; no clock conversion or interpolation supplies a missing hit timestamp.

R4/R5 use the saved G1 source poses. All 13 hits have one prior sample, one next sample and one same-frame sample, with join status `bracketed`. Native-only follow-ups use `root_pose_sample`: all 18 original hits retain prior/next samples and 8 or 9 same-frame roots, but report `inconsistent_unity_frame_time`. All 34 recovery hits and three short-capture hits also retain prior/next and same-frame roots with that status. One recovery-round-1 hit has 14 same-frame roots; the others have 8 or 9. Native pose sampling occurs in FixedUpdate while packet receipt uses the rendered-frame clock. These are useful client pose alternatives, not an exact impact pose. Their sub-2 ms Unity-time offsets must not be described as impact-time precision.

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

R7/R8 are partial observations and are excluded from the eight-capture totals. [runtime-exits.json](runtime-exits.json) separately records owned Docker exec exits with codes 5 and 29, distinct from the relay's zero exit status. The R9 startup lasted 16 seconds and exited before testing. The underlying cause and code meanings remain unknown; the last logged stage does not establish a fault location. The native-only follow-up captures are separate artifacts and must not be relabelled as completed R6/R7 runner attempts.

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

Recovery joins are in `recovery-native-round1` through `recovery-native-round4`, with the separate short capture in `recovery-native-short1`. Each directory contains the same four join artifacts listed above. The join command was `node ocean/rek_g1/native5/join_passive_hit_events.cjs NATIVE_JSONL - NEW_OUTPUT_DIRECTORY`; `-` selects measured native root samples without a relay input. The join tool SHA256 is `1a58b2867c825e2a8bd57c234f7052f47c84f0ea1fd0b81c504c3dbb26b408ee`. Per-output hashes are in the aggregate. All recovery headers retain the same recorder DLL identity above.

| Recovery capture | Native recorder filename under `native\` | SHA256 |
| --- | --- | --- |
| Round 1 | `rek-private-ai-root-motion-20260917T080950.5090999Z-pid32-bc8d98cf78034f4ca834cb8cdac85569.jsonl` | `c2c62c4e24255584bc0de80ad79105dc7c8fdb10029accba30faed4a412a9036` |
| Round 2 | `rek-private-ai-root-motion-20260917T081200.2989358Z-pid32-29c09e0abb0f4bdda08c99b137ff45ad.jsonl` | `3cb48cfc68f261aba58f738127a24c74b6cb5204afd372881aa91c458b7053a6` |
| Round 3 | `rek-private-ai-root-motion-20260917T081410.0315950Z-pid32-60e29d0332bb40ce8da07991b393a66c.jsonl` | `0432a0bfd71ada991f82a208811c2a95d3ca3370da07e5bb0bfafb650d58ad07` |
| Round 4 | `rek-private-ai-root-motion-20260917T081623.4161467Z-pid32-02c810b1a203427294fa77800d9f62e7.jsonl` | `29e02a20b0b98bd074b4199ebe9776d9e08d282636f818140cfaf8e886a3ce67` |
| Short 1 | `rek-private-ai-root-motion-20260917T081833.2718815Z-pid32-b85054f60aea42d1ad1b95692d45b3a8.jsonl` | `a0dfac958b054d18973fa59cb8c56539fecbd053cfeec5110243421ccc258e83` |

The last three finalized files were copied from `/home/spark-advantage/codexrook-runtime/wineprefix/drive_c/rekagent/evidence/runtime/rek-private-ai-protocol-v7/` using exact filenames through `ssh -n` and tar. Their remote hashes equal the local hashes. All five recovery files were independently streamed to verify local hashes, packet content and neutral observations. They are later additions and are not covered by the initial 165-file archive verification below. This report does not assert that these recovery additions have been archived.

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
R5 through R9 used the earlier `source-final/` snapshot. R10 through R12 used
`source-repeat-r1/`; R12 completed with `follow_on` attached to an already-active
round. Its native-transition wait branch remains fixture-tested only. Neither
source change is a new hitbox or a training reward change.

## Startup recovery diagnostics

Later isolated startup tests are recorded in `runtime-exits.json`; none is
included in the contact totals. A base-client launch without Doorstop survived
85 s, and BepInEx core with no plugins in a private game copy survived 103 s.
Both were stopped deliberately. Adding the two required, unchanged plugins
then failed after 18 s with status 5. This ruled out removal of the legacy
bridge as a sufficient recovery step.

In that same copy, changing only `[IL2CPP] PreloadIL2CPPInteropAssemblies` from
`true` to `false` passed startup: recorder 0.7.2 armed, UI bridge 0.4.9 listened,
and chainloader startup completed. The client remained running at 08:03:08 UTC,
132 s after launch. This supports a lazy-preload workaround; a race, a specific
plugin defect and permanent stability remain unproven. The pinned BepInEx
implementation makes this option skip its eager parallel assembly load sweep
while retaining normal dependency resolution. [Implementation](https://raw.githubusercontent.com/BepInEx/BepInEx/6abdba47eeebe08552282e7a58ef0f4a9ab60b62/Runtimes/Unity/BepInEx.Unity.IL2CPP/Il2CppInteropManager.cs).

Only the diagnostic copy's configuration changed. The original game and plugin
files remain intact. The old `RekAgentBridge` is omitted from the copy because
the current recorder and UI bridge are independent of its registry-redirection,
synthetic input-device and TCP-controller behavior. No credentials or registry
entries were migrated. The native welcome-back screen was already signed in;
the existing `ConfirmLoggedIn` command selected its visible Let's Go action.
No authentication challenge or account-login automation was needed.

## Paired collection after observer fix

R10 requested a native private arena and ready state but sent zero policy
actions. Read-only observations could bind the runtime session before ready;
a subsequent transient eligibility failure then invalidated its route proof.
The precise field at that invalidation is unknown. R10 released its lease and
the independent native recorder continued capturing the recovery cohort.

The revised bridge makes observation callers nonbinding while preserving all
current private/no-human, AI identity, endpoint and already-bound session
checks. Control entry still binds. It passed 440 protocol cases, a pipe
roundtrip, and independent source review. The built v0.4.9 DLL hash is
`6e15abbde6fdcd07f5ac71c632fd67cc2553d0a5cc35de34c9c81603c9880276`.

The old isolated client survived 1,653 s and was deliberately stopped after
its active capture finalized. Only the diagnostic copy received the revised
bridge; its previous DLL and startup logs were preserved. The first bounded
wait mistakenly included historical partial files and expired without stopping
the game. The corrected wait selected files newer than this client's launch.
No historical partial files were changed.

R11 passed the previously failing private-ready transition and completed a
119.823 s collection with 5,649 neutral sources. R12's 110.596 s collection
attached to native round 2 with about 110.50 s remaining and also reached
terminal state, with 5,297 neutral sources. Both verified stream stop and lease
release. Their native hit joins and video hashes are retained separately.
These validate collector operation; they do not prove a causal hitbox model.

The physical-server archive also contains all ten finalized files from the
08:00:56 old-client launch, totaling 1,787,259,355 bytes. The manifest
`old-runtime-all-finalized-sha256.json` verifies Spark, local and archive hashes;
unprocessed files are not counted as hit results. The subsequent
`paired-and-build-sha256.json` verifies 178 files, 742,191,245 bytes, including
R11/R12 traces and MP4s, paired joins, bridge build/source artifacts and a
timestamped runtime-log snapshot. A snapshot of a running client's logs is not
its complete lifetime output. Existing archive files were preserved.
