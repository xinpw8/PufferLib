# Matched joint-feature exclusion experiment

Status: native training and inference verification completed; authentic Bot 1 evaluation is running. No improvement or acceptance is established.

The hypothesis is that differences between canned training joints and projected live joints impair transfer. This is a hypothesis, not an established cause. The same mask excludes both fighters' joint positions and velocities in training and live inference. It preserves 107 of 223 inputs, including root geometry, orientation, velocity, actor command projections and score. It also removes an indirect view of opponent attacks, which could hurt performance.

## Input contract

Schema stays `rek.native5.scaled_polar_xy.v1`. The mask is 223 raw bytes, each zero or one. Zero-based exclusions are `13..70` and `99..156`. The input dimension and network are unchanged. Zero denotes an excluded feature, not a measurement.

Training uses existing `REK_POLICY_FEATURE_MASK` support after scaling for the learner and both actor rows. Inference uses the explicit last worker argument. The live driver verifies the mask hash in both the ready message and every prediction. No action restrictions, forced attacks or cooldowns were added. The encoder still validates and projects bones before masking. Its busy/history projections remain approximations.

```text
worker CHECKPOINT CHECKPOINT_SHA256 SEED sampled zero-joints.bin
```

## Executed native tests and training

- Shared mask CPU tests: 58,312 checks, 257 vectors, retained values bit-preserved, malformed masks rejected.
- Native worker protocol: 112 assertions; compilation passed.
- Actual GPU replay: 5,728 recorded decisions in each of six native runs. Sampled action/reset sequences match for old/default/all-ones workers and for masked/explicitly-zeroed/perturbed-masked inputs. See `gpu-replay-summary.json`.
- Native CUDA training: 16,777,216 transitions in 18.672583 seconds, 898,494.65 full-process SPS, exit zero, zero failure bits. This includes initialization, rollout, PPO, checkpoint writes and shutdown. No Python training runtime or CPU physics stepping.
- On-policy training outcomes: 2,102 wins, 396 losses, 62 ties. These are neither held-out evaluation nor authentic-game wins.

The training run uses the same original F7 warm-start, seeds, optimizer, reward and corrected calibration settings as the preceding no-kick-prior candidate, adding only the feature mask. The warm-start originally used all features, so adaptation to changed inputs is part of this experiment.

## Authentic evaluation protocol

Frozen policy seeds 701 through 720, target 18 wins in 20 full rounds, stop at the third nonwin. Actual private sparring Bot 1 / difficulty 0 on isolated Spark X98, original request-duration encoder, no live attack gate. Incomplete attempts remain separate. Windows input is untouched. MP4 capture remains under 20 MB per file.

Compared with the preceding cohort, the client is now recycled after each counted round, with no policy stream or lease active. This avoids waiting through an unobserved automatically started next round and tests a mitigation for repeated later-round crashes. It is an infrastructure change, not proof the crashes are fixed.

After this cohort began, the next-launch script was changed to `BOX64_DYNAREC_WEAKBARRIER=0` while retaining `STRONGMEM=2`. The installed Box64 source documents regular barriers at zero versus weak barriers at the default one. The original launcher is preserved, and each launch archives its script. This is an explicit runtime accuracy experiment with potential FPS cost; performance must be reported by runtime cohort. It does not change the CUDA trainer. Crash cause remains unknown.

## Identities

| Artifact | SHA256 |
| --- | --- |
| Final checkpoint | `1b1b8424be9b677e29a9e1bc58e42ed2241bbeae9e34068b21dff5942505e31b` |
| Joint mask | `d26c5f2f7d2a7faf14f89a7290189fd8fb45223ff4a548abe7be51061ec4bf85` |
| Native worker | `976ea82b86e77d9695b51156bf71b1629fdf0e490c3aa5d2f1751da1c696f832` |
| Corrected native trainer | `1754a66278059fdaa207e02d7231bf2a8d7dcc235880b2d055c4d628a344072a` |

Spark training root: `/home/spark-advantage/rek-training/joint-mask-transfer-20260924-r1`.

Spark live root: `/home/spark-advantage/rek-training/f7-joint-mask-live-20260924-r1`.

The launcher retains exact private asset/configuration paths for reproducibility on the authorized host. Those assets, checkpoints and game binaries are not included in this publication.
