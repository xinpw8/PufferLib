# Persistent geom-pair contact entry

An isolated opt-in correction replaces the compact per-limb contact latch with persistent **geom-pair** history. One matched 33,554,432-transition native CUDA training run completed successfully. Authentic evaluation is pending; this report establishes implementation behavior and training throughput, not fighting improvement or physical parity.

## Source contract and change

The recovered `ContactTrackingManager.OnMujocoPostStep` uses the unordered key `(min(geomA,geomB)<<32)|max(geomA,geomB)`. It resolves bodies afterward. Multiple manifold points on one geom pair aggregate together; two colliders attached to the same body remain different pairs. Private evidence: `ContactTrackingManager.txt`, assembly lines 2192–2209 and 2296–2350; decoded calls at 2901/2906 and key construction at 3015–3016. Dump SHA256: `ae143ecfc0e2ecb1c3c47ccfbdb4c88d969d2445aa95d059b2f8608a1524aac2`.

The [existing bake](../../fast_assets.cpp) verifies 12 striker geoms and [nine scoring target geoms](../../native_contact_geometry.h), including two distinct torso colliders on the same body. Their Cartesian product contains 108 directed cross-fighter pairs per fighter. The pair identity does not depend on the attack, limb union or body-zone number.

`REK_FAST_CONTACT_ENTRY=geom_pair_v1` requires `recovered_hit_rules_v2` and `primitive_samples_v1`. Its [host/device helper](../../contact_entry.h) and [runtime integration](../../fast_runtime.cu) implement:

- Two 64-bit words covering all 108 pairs, updated through idle as well as attacks.
- No history reset at attack start. Initial round overlap is seeded without fabricating an entry.
- Sample-by-sample enter/exit transitions. Only endpoint overlap remains latched, so a transient crossing can exit within the same tick.
- Evaluation at t=0 reconciles the existing instantaneous canned-route switch. It introduces no interpolation between unrelated clips and no physical blending model.
- Fresh entry plus valid strike intent is required before the existing apex, body cooldown and invocation-dedup checks.

Unset mode, or explicit `legacy_limb_union_v1`, retains the default behavior. Unknown modes are rejected. The new mode affects contact eligibility for both fighters; the recovered Bot1 command/controller is unchanged.

The existing maximum sphere-center finite-difference speed proxy remains unchanged, including its per-limb aggregation over intersecting targets. Native scoring actually reads the contacted bodies' `cvel` linear components. This is a separate unresolved velocity disparity. Contact timing is still sampled at eight intervals per 20 ms tick, with the existing end-of-tick intent/apex evaluation. Canned pose discontinuities, balance dynamics and authoritative contact reconstruction remain unmodeled.

## Verification

- [CPU fixture](../../test_contact_entry.cpp): 447 checks, including every pair index through 107, unknown-mode rejection, persistent overlap, distinct same-body colliders, separation/reentry and sampled enter/exit.
- [Production CUDA fixture](../../test_contact_entry_runtime.cu): 16 checks. Persistent overlap across a later move beyond cooldown scores zero under the new mode; the corresponding legacy fixture scores again. Idle contact followed by an attack does not manufacture entry. Reset seeding, pair changes, transient crossings and actual `advance_fighter` history retention pass.
- Existing recovered-scoring fixture: six host acceptance cases and 48 CUDA acceptance/target/persistent-contact cases pass.
- Existing real-asset ABI fixture: archived keyboard-reset build versus current default produced identical 666 snapshots, 2,335,662 bytes. SHA256 `d23177ac2344ddaf476cc369bcfd36d03b33da0a0e6666ccd7267eb91e8341a0`. The opt-in smoke also passed 1,110 checks with zero failures. This short separated scenario has identical outputs; the synthetic fixture demonstrates the intended scoring difference.

GPU verification ran once, 2026-09-21 00:08:56.620549086–00:08:58.909746454 UTC, exit 0. CPU compilation ran 00:07:53.145711316–00:08:42.900253056 UTC. One outer shell quoting failure occurred before build execution and is preserved in `orchestration-attempts.txt`; no failed native build, GPU retry or training retry occurred. Unrelated resident process 2325 was untouched.

## Matched native training

Comparator: the existing [keyboard-reset stride1 run](../keyboard-yaw-command-20260920/README.md), checkpoint `85808a6731f4f40f5756800a9edf93faa5c312aa7d5f7468cafdd4ae3a5c381f`. Both start from original `f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4`, with 512 arenas, horizon 512, minibatch 8192, 120 s rounds, environment seed 419, base seed 73, learning rate 0.0001, entropy 0.01, gamma 0.9998844821426083 and lambda 0.9978673240629938. Both use keyboard-reset yaw, action stride1, the legacy 223-feature schema, recovered Bot1, eight contact samples and round-outcome reward with no spatial shaping. Only contact-entry mode changes. The previous comparator was retained; no new control run was performed.

| Measurement | Existing legacy-entry comparator | Geom-pair treatment |
| --- | ---: | ---: |
| Learner transitions | 33,554,432 | 33,554,432 |
| Training-loop seconds | 36.120334 | 45.493099 |
| Full-training transitions/s | 928,962.398 | 737,571.912 |
| Whole-process seconds | 36.82 | 46.20 |
| Startup-inclusive transitions/s | 911,309.940 | 726,286.407 |

The treatment is 20.6% lower in loop throughput in this comparison. SPS counts the same 50 Hz learner transitions, including rollout and optimization. It is not environment-only throughput. Additional persistent contact evaluation costs work even outside attacks.

Training ran once from 2026-09-21 00:09:31.321615485 to 00:10:17.633946862 UTC. Native and wrapper exits were 0, failure bits 0, and the step-zero checkpoint exactly matched the parent. Changing-policy compact totals: 4,493 wins, 578 losses, 49 ties; 355,849:215,160 points. These training totals are not held-out strength or authentic REK outcomes.

Identities:

- Native executable: `58ebcd3d0447dae0b1ca9c2a9cec9c1f9c68b43f88dc8700e8950e3ac19cf979`.
- Runtime source: `7dffca7f1e2ce7cedd1af2ca13d27916dbe0f78da469945978903de4693e422b`.
- Entry helper: `d4c769d5b3e2282a838c65d261fb03c4760f6b19d832d5637abcde4f70e7adca`.
- Final checkpoint: `07fdce5f1d14833103189ad424ff2c6f25d37166e259a9e3ba9230680cb61766`.

Private checkpoint: `/home/spark-advantage/rek-training/contact-entry-20260920-r1/train-geom_pair_v1-r1/checkpoints/rek_native5/train-geom_pair_v1-r1/0000000033554432.bin`.

The live config `authentic-contact-entry-trained-v1.json` changes only checkpoint identity/path and its descriptive name relative to the comparator. Encoder, worker, seed73, sampled inference, unmasked legacy223 schema and stride1 are unchanged. Config SHA256: `f7d8f3c226467bc2f95182e0979a247d71e491cf3f5f856447e0f3052e6eaef8`. No headless contact-entry flag is applied to the authentic client.

## Reproduction

Use the existing native toolchain and private asset inputs. From the repository root on Spark, choose fresh build/output paths:

```bash
native=$PWD/ocean/rek_g1/native5
stage=/home/spark-advantage/rek-training/CONTACT_ENTRY_FRESH_RUN
mkdir "$stage"
g++ -std=c++17 -O3 "$native/test_contact_entry.cpp" -o "$stage/test-contact-entry"
"$stage/test-contact-entry"
bash "$native/build_fast.sh" "$stage/build"
mujoco=/home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco
for source in test_contact_entry_runtime.cu validation-quality/scoring_v2_test.cu; do
  name=$(basename "$source" .cu)
  /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_121 -I"$native" \
    "$native/$source" "$stage/build/fast_assets.o" "$stage/build/cJSON.o" \
    -L"$mujoco" -Xlinker=-rpath -Xlinker="$mujoco" \
    -l:libmujoco.so.3.7.0 -lcrypto -o "$stage/$name"
done
"$stage/test_contact_entry_runtime"
"$stage/scoring_v2_test" --gpu
unset REK_POLICY_FEATURE_MASK REK_FAST_CONTACT_POTENTIAL REK_TRAIN_OPPONENT_CHECKPOINT REK_FROZEN_OPPONENT_FRACTION
unset REK_FAST_SHAPING_GAMMA REK_FAST_SHAPING_TARGET REK_FAST_SHAPING_BEARING_WEIGHT REK_FAST_REWARD_GAMMA
export REK_OBSERVATION_SCHEMA=rek.native5.scaled_polar_xy.v1 REK_POLICY_ACTION_STRIDE=1
export REK_FAST_YAW_COMMAND=keyboard_reset_v1 REK_FAST_CONTACT_ENTRY=geom_pair_v1
export REK_FAST_REWARD=round_outcome_v1 REK_FAST_SCORING=recovered_hit_rules_v2
export REK_FAST_GEOMETRY=primitive_samples_v1 REK_FAST_CONTACT_SUBSTEPS=8
export REK_FAST_OPPONENT=recovered_bot1_v1 REK_FAST_OBSERVATION=rendered_pose_v1 REK_FAST_OPPONENT_MODE=scripted
export REK_FAST_RANDOM_RESETS=1 REK_FAST_RESET_GAP_MIN=.55 REK_FAST_RESET_GAP_MAX=2.5
export REK_FAST_RESET_HEADING_SPREAD_RAD=3.14159265 REK_FAST_SHAPING_WEIGHT=0
export REK_TRAIN_SEED=419 REK_TRAIN_LEARNING_RATE=.0001 REK_TRAIN_MINIBATCH=8192 REK_TRAIN_ENTROPY=.01
export REK_TRAIN_GAMMA=.9998844821426083 REK_TRAIN_GAE_LAMBDA=.9978673240629938
initial=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin
bash "$native/run_diverse_training.sh" "$stage/build" "$stage/train-geom-pair" 33554432 512 512 120 "$initial"
node "$native/summarize_fast.cjs" "$stage/train-geom-pair"
```

The executed private stage preserves `build-and-test.sh`, `verify-runtime.sh`, `run-training.sh`, exact expanded training command, stdout/stderr, timing files, checkpoints, object/executable hashes and all three ABI snapshot files. `verify-runtime.sh` links the existing `test_action_cadence.cu` against archived/current runtime objects, executes the fixed identical schedule and compares default bytes with `cmp`; no new launch harness was introduced.

Verified Windows mirror: `C:\rekagent\work\consistent-fighter-20260919-r1\contact-entry-r1\spark-results`, 78 files, 32,603,781 bytes. All copied hashes match Spark, and remote before/after manifests are identical. SHA256 manifest: `b77b648aa14b7159bc34a30e946acd0ed834f4eb153814ca991bb77e7630e8e1`; verification JSON: `28f6e6e7964923bd287c96522ffb28f921f4994ad2c5467d6e08aeb5e4a952e9`. No proprietary assets or checkpoints are included in this repository.
