# Left-front kick exploration checkpoint

This is an offline data-collection intervention in a new copy of the frozen f3 control actor. On its recorded r24/r26/r28 histories, native BF16 inference increased LEFT_FRONT category17 from **0.0820596% to 9.9943156% of legal attack probability mass**. The candidate generated 29 counterfactual kick requests. No fight, optimizer, reward modification, worker change, or behavior-replay publication occurred in this experiment. Fighting efficacy is unknown.

## Representable intervention

The registered native architecture contains an encoder `[256,223]`, a bias-free decoder `[34,256]`, and two recurrent `[768,256]` matrices, totaling 459,008 saved FP32 parameters. `decoder_row_exploration.cu` derives their locations from the actual pinned allocator registration, including the decoder pointer and shapes. There is no decoder bias slot.

Only the existing category17 decoder row is changed:

```text
w17_new = (1 - alpha) * w17_original + alpha * w23_original
alpha = 0.80835148601383922
```

Category23 is RIGHT_HOOK. This produces state-dependent logit interpolation, not a constant bias. The permitted row occupies float indices `[61440,61696)`. All other 458,752 parameters, including the entire encoder, recurrent matrices, other actor rows, and value row `[65536,65792)`, remain bitwise identical. The original file is preserved. The sampler, legal masks, observation schema v1, 223 unmasked features, recurrent resets, and native logprob calculation are unchanged.

The CPU target is `sum(p17) / sum(sum(p16..p32)) = 0.1`, with legal-mask softmax at each recorded state. All attack-legal states in this cohort also permit category17. It is an aggregate probability-mass target, not a per-state minimum or a quota of actual attacks. Human action labels and contact outcomes were not used to select alpha.

| Alpha | Predicted kick share of attack mass |
| --- | ---: |
| 0 | 0.0820596% |
| 0.25 | 0.2566395% |
| 0.5 | 1.0767343% |
| 0.75 | 6.3247704% |
| 1 | 40.2184377% |
| 0.80835148601383922 | 10.0000000% |

These CPU predictions interpolate the existing exact BF16 replay logits in float64. Saving blended FP32 weights and casting them for native BF16 inference introduces additional rounding, which the single native verification measured directly.

## Verification

CPU self-tests passed: registered bias-free layout, identity and full-donor endpoints, midpoint preservation, invalid alpha rejection, seven targeted parameter-corruption rejections including the value row, masked-kick zero probability, normalized softmax, and target calibration. The total negative-check count is 11. The CPU path makes no CUDA calls.

One native fixed-history verification covered all **17,226 decisions in three rounds**, with 426 attack-legal states. It used the exact live-worker-linked native object, BF16, batch1, hidden256, two recurrent layers, and seed73 restarted per recorded worker sequence.

- Original sampled actions, all 34 logits, and native-order sampled logprobs matched the saved behavior replay bitwise for every row.
- All 33 candidate non17 outputs, including value, matched original outputs bitwise for every row. Checkpoint differences are confined to the 256 allowed decoder weights.
- All candidate sampled requests were legal; 48 requests changed. Candidate totals were 29 LEFT_FRONT and 216 RIGHT_HOOK requests, compared with original totals of 0 and 242 respectively.
- Candidate kick probability sum was 31.9907331; total attack probability sum was 320.0892835. Their ratio was 0.099943155654153987. The corresponding original sums were 0.2589251 and 315.5329121.
- Maximum category17 logit discrepancy from CPU interpolation was 0.0847635906. Sampled native logprobs remained finite, ranging from -8.31196785 to -0.00004196167.

The verification ran once under a 120 s timeout and exited 0. UTC start/end were `2026-09-20T03:59:16.785642571Z` and `2026-09-20T03:59:27.427961331Z`; process wall time 10.61 s, user 5.73 s, system 3.83 s. GPU ownership was released immediately afterward. CPU calibration took 0.16 s.

The counterfactual requests are diagnostic draws on the original observations, not executed actions or collected behavior. New actions will change subsequent observations and legal opportunities. The measured mass does not guarantee the same rate in new fights, successful execution/contact, stable kicks, trips, or improved scores. No counterfactual output is emitted in `REKBR001`/`REKBR002` format.

## Exact identities and private artifacts

Remote stage: `/home/spark-advantage/rek-training/left-front-exploration-20260920-r1`.

| Artifact | SHA256 |
| --- | --- |
| Original f3 control checkpoint | `f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4` |
| New `candidate-r1/left-front-exploration.bin` | `6e4100648c17c06142d8fe4f997b6e1e7fd958dd2f0d3a90a7de367f370ab245` |
| r24/r26/r28 v1 trajectory | `06bf216a33e57c2bd22cddeb23a914f41e3f6bf31ebd77a19a4b60cd15cef96e` |
| Exact original behavior replay | `6f21784c9e48aa2787c61ce28416ebc1f94a16c7efe8ab90bf0501a6e5f7ae9d` |
| `build-r1/decoder-row-exploration` | `9a6fa93f4017082dd4266dce6db51ccb3ef5c8c696d3760549fb9a1d081fefed` |
| `decoder_row_exploration.cu` | `8f15c823058ae2b804c4d5e39b31eb01bc47ea968e991fd2988a73ec33d680c3` |
| `build_decoder_row_exploration.sh` | `c6e4ff251deb52c6627065e77db4ce0a203be73825f3e8259e5990865a8593c0` |
| Exact linked `native_policy.o` | `4ada3de760b5a00f7bb3d6592cd2da4ca48a4a196e774d40e6d220a57d1f574c` |
| Prepared `puffer5_bc_core.cuh` | `d3e07e6c5f376584543cdba457d582d6674183a0efb29c0755f236adcf284454` |
| Prepared `algo.cu` | `8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92` |
| `candidate-r1/calibration.json` | `1023fc00da9f9c72357ee74dac7e28c0a3cebd6d8cf5500d1fd094eecd4eeabc` |
| `verification-r1/verification.json` | `34d9c0e963010478953b452e89c223aefe4007f840b8aec087824babf58f3577` |

The linked object is `/home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o`, the same path and hash recorded by the actual `owned-yaw-observation-20260920-r1/worker-build` link provenance. It was not substituted with a full-trainer build object. The all-row original replay check independently verifies its behavior here.

Complete staged sources, executable, new checkpoint, command scripts, stdout/stderr, timings, and hashes were mirrored to `C:\rekagent\work\consistent-fighter-20260919-r1\left-front-exploration-r1\spark-results`: 40 files, 4,586,505 bytes including the manifest. All 39 manifested files passed SHA256 readback on Spark and in the local extracted copy. The 2,921,534-byte compressed transfer has SHA256 `df78932501f204b114c281c85032bc807ee2eb64f6ddbd04c2e691be27ad6251`; `artifacts.sha256` has SHA256 `1c738657dd2173d875e6cdb03d4dc62f8cddd963b70239b6c8b912a92a026c3b`. No checkpoint or raw trajectory is included in the public repository.

The completed mirror and archive script were additionally copied to fresh physical-server directory `\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-19\consistent-fighter-r1\left-front-exploration-r1`: 41 copied files, 4,591,811 bytes. The existing archive pattern refused an existing destination, verified every source hash before and after copying, read every NAS copy back, and rechecked the final source file set. All checks passed; sources were preserved. NAS `archive-manifest.json` SHA256 is `54cdf7572e6fdf8ec3076b6ebc39f427a1c5f2b6c0cca31e475244ed71993277`. Manifest and transcript are additional archive metadata files.

## Prepared collection status

The coordinating task prepared private `authentic-left-front-exploration-v1.json`, SHA256 `dfb8119374c52d3783a48a41c5f536a9631fff1f367b66e0546ba2ffed95f17e`. Only its name and checkpoint path/hash differ from the f3 control configuration; encoder, worker, seed73, native legality, reward, and 20 ms cadence are preserved. Three fresh development rounds, r30/r31/r32, were predeclared for this frozen exploration checkpoint. No trial had started when this report was prepared. Actual export, exact native behavior replay, and outcome-based PPO fine-tuning are subsequent work requiring separate execution. No promotion or frozen acceptance result is claimed.

A subsequent native worker smoke test loaded the exact candidate, reported BF16/native CUDA, seed73 and `environment_stepping:false`, then closed on the supplied protocol command. Exit code was 0 with empty stderr. This verifies checkpoint loading and protocol readiness, without a game connection. Its separate private logs are `left-kick-worker-smoke.stdout.jsonl` and `left-kick-worker-smoke.stderr.txt` in the coordinating stage; they are not part of the earlier 41-file archive.

## Reproduction

Use a fresh stage/output. These are the executed commands, with paths factored into shell variables. Full shell traces and `/usr/bin/time -v` output are in the private artifact stage.

```bash
stage=/home/spark-advantage/rek-training/left-front-exploration-20260920-r1
cohort=/home/spark-advantage/rek-training/authentic-owned-yaw-cohort-20260920-r1
original=/home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/matched-gae-r1/train-control-v1/ppo.bin
sha=f3fabe8ca1b781844d2e314daef7fc47423046425aa373b618a5f50cc2a075b4
bash "$stage/source/build_decoder_row_exploration.sh" \
  /home/spark-advantage/rek-training/owned-yaw-migration-20260920-r1/build-r1 \
  /home/spark-advantage/rek-training/semantic-fast-20260914-v1/build-v4/native_policy.o \
  "$stage/build-r1"
# Build invokes --cpu-self-test, without GPU execution.
mkdir "$stage/candidate-r1"
"$stage/build-r1/decoder-row-exploration" --calibrate \
  "$cohort/data-v1/authentic-trajectories.bin" \
  "$cohort/replay-v1-r1/behavior-replay.bin" "$original" "$sha" \
  "$stage/candidate-r1/left-front-exploration.bin"
timeout 120 "$stage/build-r1/decoder-row-exploration" --verify \
  "$cohort/data-v1/authentic-trajectories.bin" \
  "$cohort/replay-v1-r1/behavior-replay.bin" "$original" "$sha" \
  "$stage/candidate-r1/left-front-exploration.bin" 0.80835148601383922
```

Build and candidate creation reject existing output paths. Verification checks the candidate bytes against the exact specified row transform before any GPU allocation. Both modes require the dataset/replay/checkpoint binding to agree and use the existing strict v1 trajectory loader.
