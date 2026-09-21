# Opt-in kinematic body-cvel scoring

Implemented and verified on 2026-09-21 UTC. `REK_FAST_CONTACT_VELOCITY=body_cvel_v1` supplies each entered geometry pair's own body-relative speed to the recovered score calculation. The default remains `legacy_sphere_proxy_v1`. No observation, actor, sampler, live bridge, or reward arithmetic changed.

This corrects the supplied velocity quantity and its pair attribution within the compact kinematic runtime. It does not reproduce controller response, contact impulses, recovery, or balance dynamics. The first authentic development round, r58, subsequently passed existing strict validation but lost 5:12. No fighting improvement is established.

## Contract and implementation

The recovered native getter reads the linear triplet `cvel[6 * body + 3..5]`. This is a body twist referenced to its root subtree COM, not collider-centre velocity. The earlier [source investigation](../contact-velocity-contract-20260920/README.md) and [independent composition probe](../contact-velocity-probe-20260920/README.md) preserve the derivation and version evidence. Static inspection identifies the installed native library as MuJoCo 3.7.0; that does not prove the module used by earlier client processes or the server version.

The opt-in asset load runs `mj_differentiatePos`, `mj_kinematics`, `mj_comPos`, and `mj_comVel` on each route's canonical normalized poses and actual incoming 20 ms edge. Both fighter trees are baked separately. Loop frame zero uses the last frame; non-loop frame zero uses itself. Identical coordinates are explicitly zero-rate. No simulation stepping, force solve, or controller inference occurs.

`FastFrame` remains 1,664 bytes. A separate optional 360-byte `FastBodyVelocityFrame` contains two sets of 14 body-linear triplets and two root subtree COM vectors. Six striker limbs and nine target geometries map to those body slots; the two torso geometries share one body velocity. The 1,768-frame asset uses 636,480 additional bytes, uploaded once only in the opt-in mode. The default has no optional device allocation.

For baked triplet `L`, root subtree COM `C`, modeled yaw `theta`, planar root velocity `v`, and yaw rate `omega`, device composition is:

```text
L_world = Rz(theta) L + (vx, vy, 0)
          + (0, 0, omega) cross Rz(theta) C
```

An unchanged old/current frame suppresses the clip-rate term while retaining modeled root translation and yaw. Current end-of-tick rates are used for entries found by the existing contact samples. This is an explicit temporal approximation, including route-start reconciliation and contact entries earlier in the tick.

The mode requires `geom_pair_v1`, `primitive_samples_v1`, and `recovered_hit_rules_v2`. Each entered geometry pair gets its own relative-speed norm. A fast persistent or unrelated target cannot supply speed to a newly entered slow pair. A speed rejection does not consume body cooldown or per-move dedup; a later qualifying pair can score. Existing apex eligibility, geometry history, cooldown, and dedup remain in force. Only quantity documentation and a parameter name changed in `recovered_contact_rules.cuh`; score arithmetic is unchanged.

`FastAssets` grows internally. Rebuild runtime and assets together. Retaining the one-argument source API does not make old `fast_runtime.o` objects binary-compatible with the new class layout. Archived comparisons used the complete archived executable, never mixed objects.

## Executed verification

| Check | Result |
|---|---|
| CPU helper | 52 checks passed |
| CPU real-asset bake and composition | 159,175 checks passed; 49,504 canonical and 297,024 transformed body triplets |
| CPU full-FP32 pair-speed error | Maximum 2.6060708613e-6 m/s; bound 1e-5 m/s |
| CUDA composition | 1,080 cases; maximum component error 4.76837158e-7 m/s and speed error 9.53674316e-7 m/s |
| Production scoring call | 16 checks passed, including persistent-fast/new-slow separation and later qualifying contact |
| Existing contact-entry fixture | 16 checks passed |
| Existing apex/contact-history equivalence | 18,470 checks passed |
| Existing autoreset/reward regression | 592 checks passed |
| Recovered scoring regression | 6 CPU catalog, 12 GPU production, and 36 isolated-target checks passed |
| Default allocation and incompatible mode | Null default optional pointer, exactly one opt-in extra allocation, incompatible combination rejected |

The CPU test covers 7 loop incoming edges and 17 zero edges. The largest transformed component error was 1.6850046993e-6 m/s. Default and explicit-false asset loads leave optional storage empty and retain legacy frame bytes, routes, mappings, initial poses, and provenance.

CUDA verification ran from `2026-09-21T00:59:49.879106492Z` to `2026-09-21T00:59:54.386916102Z`, exit 0. Default and geometry-pair runtime snapshots were byte-identical to the frozen apex executable. Explicit legacy velocity matched the pair snapshot. Each probe had 666 records and 1,110 checks, with snapshot SHA256 `d23177ac2344ddaf476cc369bcfd36d03b33da0a0e6666ccd7267eb91e8341a0`. The body-cvel snapshot also matched on this short fixture; this does not establish equal training dynamics or efficacy.

## Matched full training measurement

One fresh run used the same f3 parent and 33,554,432-transition workload as the prior apex run, with the velocity mode as the treatment. Settings: 512 arenas, horizon 512, minibatch 8,192, seed 419, learning rate 1e-4, entropy 0.01, gamma 0.9998844821426083, lambda 0.9978673240629938, 120 s rounds, keyboard-reset yaw, stride 1, legacy unmasked 223-column observations, geometry-pair entry, 8 contact samples, recovered Bot1, randomized initial gap/heading, and round-outcome reward. There was no authentic-data optimizer update in this step.

The frozen private `run-training.sh` ran after verification. Start/end: `2026-09-21T01:01:01.550985598Z` / `2026-09-21T01:01:48.675716335Z`; exit 0. The training loop took 46.3135905266 s: **724,505.09 learner transitions/s**. Process time was 47.01 s, or 713,772.22 transitions/s including startup. These are complete training measurements, not pooled arena-step rates.

The changing training policy completed 5,120 simulated rounds: 4,585 wins, 479 losses, 56 ties; points 337,613:174,580; failure bits 0. These are training counts against the compact opponent, with no held-out strength interpretation.

| Artifact | SHA256 |
|---|---|
| Unchanged f3 parent / step-zero readback | `f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4` |
| Final 33,554,432-transition checkpoint | `08f717ff7945ea0a75779ca5e48ed99bbbf244956c5753e1ffa3136f04435249` |
| Fully rebuilt trainer | `a3e7515523a750a952fccd68968fdf1e67031efb616c2003b2224b6b9ba7b2ed` |
| Runtime object | `479ceaf336560c2bfd19b12f665f649bccf8c6b664b6d1759373367e2b97eff6` |
| Matching assets object | `c0d588e41046f89e121333af0afd56d341dce3262eab9dae65855213f3aa6685` |
| Frozen training runner | `088547e3c6c83778cbdea3fdddf5b9ba3251e1439f3876958fa89661f3fb23a3` |

## Reproduction and private evidence

Spark stage: `/home/spark-advantage/rek-training/body-cvel-20260920-r1`. `source-base.tar` is the `ec11a14d2f024317938cbe7ffe96ea1ee32a99b8` source base; `source-runtime-delta-r1.tar` adds this implementation. The frozen `source-runtime-r1`, `build-runtime.sh`, and `verify-runtime.sh` contain the full build and verification commands. `run-training.sh` invokes:

```sh
bash "$native/run_diverse_training.sh" "$build" "$output" 33554432 512 512 120 "$initial"
```

Exact environment, expanded native command, configuration, stdout/stderr, process timing, summaries, and all three checkpoint hashes are retained under `train-body_cvel_v1-r1`. The final checkpoint is `checkpoints/rek_native5/train-body_cvel_v1-r1/0000000033554432.bin` within that directory.

Source/build/verification and completed training were mirrored to `C:\rekagent\work\consistent-fighter-20260919-r1\body-cvel-r1\spark-results-runtime-r1`. All 2,195 source-manifest hashes passed both remote post-archive readback and local extracted readback. The compressed evidence is 26,389,891 bytes, SHA256 `9db986baa083377853678d516adc7ebe595caeefacf2667b42ff260c49f9022f`; manifest SHA256 `99e31fdde6a61cbaefca9da5761a7d1ee8b0bb4dbbb479b1be207fa7b80947cf`. CPU results and both development revisions remain separately preserved in `spark-results-cpu-r2` under the same private parent.

The report does not publish model assets, raw client records, or checkpoints. The source model SHA256 is `6cec7d81b69187bfdf2429d71b6288ebb999b5359ecaab7990721323b21722aa`; pinned MuJoCo 3.7.0 Linux library SHA256 is `ef77a7d7d1e5a83197674170b296d0ea072376db89edbf161cc0637508d7642f`.

## Remaining boundary

The producer still differentiates prerecorded route poses and adds modeled base motion. It cannot generate a controller's force response, support/contact changes caused by impacts, or consequent fall/recovery/countout transitions. Those missing causal dynamics are the strongest source-backed parity gap; another training sweep alone cannot establish or repair them. The subsequent authentic cohort must be assessed separately, without attributing score awards to particular requests unless the native evidence supplies that identity.

## NAS archive

The completed private folder and archive script are preserved at `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\native-body-cvel-r2`: 2,287 files, 128,181,930 bytes. Each source-before, NAS readback, and source-after SHA256 matched. `archive-manifest.json` SHA256 is `1a9f0b1caa70147fef25c46bd9168b781adfb7efbb31e885cc5489eaa8aa3bbc`. Sources and existing destinations were preserved. The partial first attempt, `native-body-cvel-r1`, and its transcript remain separately retained: receipt generation stopped on a hidden `.gitignore` size lookup; the fresh successful archive used the minimal `Get-Item -Force` correction.
