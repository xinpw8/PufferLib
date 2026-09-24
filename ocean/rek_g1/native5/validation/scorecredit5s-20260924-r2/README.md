# Balance8 authentic on-policy score-delta candidate

Private stage: /home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2.
Local review mirror: C:\rekagent\work\balance8-onpolicy-20260924-r2.
Production source and prior artifacts were not modified.

## Data and verified behavior

The selected completed 120-second non-redo private Bot1 rounds are s901-retry4, s902-retry3, s903-retry3, s904-retry3, and s905-retry3. Outcomes were 5:13, 9:23, 13:3, 10:9, and 5:10: 2 wins, 3 losses, 42:58 points. These are five development episodes.

All 15,078 decisions use actual recorded balance8 inputs, masks, chronology, and seeds 901 through 905. No 50 Hz resampling, feature replacement, weight migration, or inferred action causality is applied. Observed per-round decision rates were 26.104, 24.422, 25.731, 24.428, and 25.342 decisions/s. Interval range across the cohort was 0.0273387 to 0.1582515 seconds.

The 15,073 applied decisions retain actor weight 1. Each round's final rejected request retains actor weight 0 and remains in sequence for recurrent replay and return propagation. Original complete native score packets and received referee evidence were checked with the existing validators. Only missing summary round identity and actor slot fields were derived into private sidecars; original summaries were preserved.

Actual recorded behavior checkpoint:
9a875c347b512bb46dde25481be75d276ea1b4c0799886b30aaf03be6ae1c5ce

Worker:
52741aed67037073bff5ecb21af63864550cba93a316dfe59a41b8b50126d7b2

Native policy object:
4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c

All-ones feature mask:
59158bfdf9ddb9a386f62aac4a5c96357d4d7fe26c03262cf0abea3ca46b1b

Schema: rek.native5.scaled_polar_xy.balance8_v1.

Worker logs contain sampled actions, identities, and recurrent-reset information. Original logits were not logged. Native replay reconstructed logits, values, and FP32 sampler log probabilities from the exact checkpoint/native object. All 15,078 sampled actions matched, with zero mismatches and full native process wall time 5.58 seconds. This is actual behavior replay, without a migrated teacher. PPO sequential teacher and batched numerical checks still execute before any optimizer update.

## Separate reward versions

export/authentic-trajectories-v3.bin preserves the established terminal outcome plus discounted score-potential reward and its original metadata. SHA256:
c5fd52c40f3f01c4f45cee1aafb912b9c6b7aca5bef16e625b9bf371f6e78367

score-delta-5s/authentic-score-delta-v5.bin is the explicitly selected new contract:

    r_i = ((nextOwn - own) - (nextOpponent - opponent)) / 5
    gamma_i = float32(2^(-actual_QPC_dt_i / 5))
    G_i = r_i + (terminal_i ? 0 : gamma_i * G_(i+1))
    advantage_i = return_i = G_i

There is no terminal win bonus, score potential, reward clipping, guessed fall probability, or reward from Slip/count flags. The actual received five-point awards enter through the observed score counters. Existing terminal outcome fields remain provenance metadata and are not added to rewards. Complete-MC mode is required; VF coefficient is zero. Original lambda cells are retained but complete-MC uses lambda 1.

Only gamma and reward cells changed within each trajectory row. All 223 observations, action legality masks, chosen actions, seeds, resets, timestamps, score counters, terminal flags, and loss weights are bitwise preserved. Native reader checks both the new reward and five-second discount for every row.

Undiscounted reward sums are -1.6, -2.8, +2.0, +0.2, and -1.0, within FP32 summation tolerance, equal to each observed net point margin divided by 5.

The recurrence discounts between reward-row source times. It does not prediscount the reward by its own interval. Fixed receipt time therefore retains up to one observation interval of timing quantization across different decision cadences. Terminal rejected-row reward remains available to earlier returns even though that row's actor gradient has weight zero.

V5 dataset SHA256:
a28c938abe35794497ef9e36577574af28064fb669a1dc2f6d560aa0878567b1

V5 identity SHA256:
03b3c21a698c850a1471f086d5116e7507a52beef7787ad128e24f6ddc4510ac

Verified V5 behavior replay SHA256:
b1b119b277f11cf8f702730445091ce11bf4fa60ff0f2a7697c1b70f5ea9bf1e

## Minimal implementation changes

- Private authentic_trajectory_data.cjs accepts the explicit balance8 schema and verifies same-tick encoder/source and all-223 worker equality.
- Private authentic_trajectory_v3.cjs accepts the recorded seven-argument worker invocation, including its explicit schema.
- derive_score_delta.cjs produces the separate version-5 reward/discount dataset from immutable baseline rows.
- Private authentic_trajectory.h decodes REKRL005/REKBR005 and validates score-delta rewards and actual-time discounts.
- Private replay_authentic_behavior.cu adds the explicit balance8 schema and version-5 container. Native inference/sampling are unchanged.
- Private authentic_ppo.cu accepts recorded balance8 identity versions 3/5, reports the actual behavior checkpoint, and requires complete-MC for version 5. PPO loss, optimizer, recurrence, native BF16 handling, and existing distributional acceptance budgets are unchanged.

Tests: 21 existing JS exporter tests, 6 focused score-delta JS tests, baseline native reader checks, and 15,085 native score-delta checks passed. Tests cover unchanged row bytes, point totals, seed identity, terminal rejected weighting, score scale/sign, simultaneous awards, no-score zero reward, actual-time discount, invalid inputs, and corrupted native reward/discount cells.

## Reproduction and execution

Executed CPU preparation:

    node prepare_strict_sidecars.cjs
    node source/authentic_trajectory_v3.cjs export-config.json /home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2/export
    node derive_score_delta.cjs /home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2
    bash build_score_delta.sh

Executed native replay only:

    bash /home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2/run_score_delta.sh replay --run

Executed by the root agent after review:

    bash /home/spark-advantage/rek-training/balance8-onpolicy-20260924-r2/run_score_delta.sh train --run

Training configuration was one epoch, learning rate 1e-5, horizon 128, PPO clip 0.2, VF clip 0.2, VF coefficient 0, entropy 0.001, complete MC zero baseline, and explicit distributional BF16 acceptance with the true immutable old log probabilities. No old denominator was recomputed after updates. Training completed 120 optimizer updates, exit 0, in 65.68 seconds of full native process wall time. An unrelated wan-i2v GPU job was running concurrently; this is a contended end-to-end duration, not an isolated throughput benchmark. CUDA target recurrence matched the CPU reference with zero maximum error. Initial sequential teacher logit/value errors were zero; initial clipping was zero. Post-update mean legal KL was 0.000495710549716, maximum 0.00848360970981, and clipped fraction 0.000132643586683.

Trained checkpoint SHA256: 0daf90a96442d541d38d9dd4f8fe765917c3c766ca80019855cfd7d358d9fe12.

The existing pinned worker subsequently processed the first 512 real s901 requests one at a time with this checkpoint. All legality, seed, schema, checkpoint, mask, sequence, and recurrent-reset checks passed. Request-to-action wall latency was median 0.475428 ms, p95 0.685669 ms, maximum 2.193842 ms. Native GPU time was median 0.108928 ms, p95 0.187040 ms, maximum 0.320800 ms. The smoke used no encoder, bridge, game, environment steps, or optimizer updates. The unrelated job was not touched.

Root reported the prospective scorecredit5s live cohort launched at 2026-09-24 08:42:43 UTC, with seeds 1001 through 1020, the existing 18/20 target, and stopping after three nonwins. No live result is included at publication cutoff. Training and smoke checks do not establish improved Bot1 win rate.

The launcher checks exact artifact hashes, refuses existing output directories, and logs the command, UTC bounds, complete native process wall time, exit code, and native stdout/stderr. Re-execution requires a fresh output stage; prior evidence is not overwritten. Compact receipts and test output are mirrored in receipts/.

## Publication layout and existing dependencies

This folder is an isolated source/evidence variant. It does not install changes into production native5 source. The seven files in source/ are the changed or new adapter/reader/replay/PPO test files. Patches in patches/ are relative to a materialized source/ directory; their exact frozen baseline locations and hashes are recorded in SOURCE-MANIFEST.json. No binaries, checkpoints, raw game captures, or proprietary assets are included.

Unchanged source dependencies are referenced by path and SHA256 in DEPENDENCIES.json, principally from ../balance8-authentic-20260924-r1/source and ../authentic-ppo-noprior-20260924-r1/source. Two unchanged exporter test/helper files are referenced from ../../. To construct the 19-file source set without copying those dependencies into this publication:

    node materialize_source.cjs NEW_SOURCE_DIRECTORY

The materializer checks dependency hashes, requires a new directory, and mechanically copies the seven overrides plus twelve unchanged dependencies. Its output was tested with all 27 JS tests passing. Native CPU decoding and GPU replay results above use the same source bytes.

The exact executed build script also references the existing Spark prepared Puffer5 kernel directory at /home/spark-advantage/rek-training/authentic-ppo-noprior-20260924-r1/distributional-candidate/ppo-build/source, existing native-policy/device headers in the prior balance8 Spark source tree, and the pinned native_policy.o. These are reused build dependencies; the CUDA toolkit, cuBLAS, cuRAND, OpenSSL, and existing NCCL headers are required. See the prior publication's build instructions for kernel preparation. The evidence validators referenced by prepare_strict_sidecars.cjs remain at their recorded Spark paths. They were reused without edits.

Training stdout/time/command receipts are in receipts/train-score-delta/. The smoke summary is receipts/worker-smoke-512.json. Raw 512-request smoke input/output and individual latency arrays remain in the private Spark worker-smoke-512 directory. Original five-round raw capture archive locations are recorded in the preceding balance8 publication.
