# Compact runtime V3: build, input checks, native training

Executed on `spark-4ae3`, NVIDIA GB10, beginning `2026-09-15T04:57:23Z` for training. This is the points-only action-level candidate. Synthetic hit-damage knockdowns, their +5 bonus, and their position resets were removed. Knockdowns remain unmodeled; authentic REK physics parity is not claimed.

## Results

| Measurement | Observed result |
| --- | ---: |
| Completed headless native CUDA training transitions | 33,554,432 |
| Arenas / horizon / minibatch | 512 / 16 / 8,192 |
| Training uptime | 13.751488 s |
| Full-run training SPS, transitions / training uptime | 2,440,058 |
| Whole process wall time | 14.36 s |
| Process-inclusive SPS, transitions / whole process wall time | 2,336,660 |
| Final reporting-window SPS | 2,458,145 |
| Completed training rounds | 33,280 |
| Cumulative training W / L / D | 21,129 / 6,121 / 6,030 |
| Cumulative completed-round points, learner / script | 93,267 / 43,290 |
| Failure bits / exit code | 0 / 0 |

The final rolling dashboard rounded wins/losses/draws to `0.998 / 0.001 / 0.001`. These are online training statistics with a changing policy. Frozen evaluation is separate. Training covered 20 s episodes against the GPU scripted opponent, seed 73, native BF16 PufferLib policy and PPO. Other pre-existing GPU processes were present, as recorded in `training/provenance.txt`; this is not an exclusive-device benchmark.

Final-window native timers account for 183.243 ms rollout and 245.026 ms training in a 503.222 ms interval. Their accounted-time shares are 42.787% and 57.213%; another 74.953 ms is not assigned by those timers. Rollout combines inference and environment stepping. The detailed environment timer is uninstrumented, so its zero value is not a zero-cost claim. No new Nsight profile was taken for V3.

## Build isolation

`source-v3`, `build-v3`, `eval-build-v3`, `input-probe-v3`, and `train-v3-33m` are new directories below `/home/spark-advantage/rek-training/semantic-fast-20260914-v1`. V2 artifacts were preserved. Only `fast_runtime.o` was recompiled for the trainer. `pufferl.o`, `fast_assets.o`, `native_policy.o`, `cJSON.o`, and both trainer INIs were verified byte-identical to V2. Current evaluator sources were compiled separately against the V3 runtime objects.

Training and state evolution use native C++/CUDA. The offline asset baker uses native MuJoCo kinematics to recover clip geometry; it does not step CPU physics. No Python interpreter was invoked. Native NCCL library discovery uses an existing package directory whose path contains `python3.12`; this is a shared-library location, not Python execution. ELF dependencies are saved.

## Input and deterministic contact checks

`input-probe/stdout.txt` records passing checks for:

- Episode-local observation feature 186 remains 1 across repeated round resets, while the diagnostic round counter increments.
- Held Q/E pauses during an attack and resumes afterward; desired yaw can be changed or released.
- Extra attack and translation inputs during a locked attack are discarded.
- The tested attack lasts exactly 157 ticks and translation must settle before attack entry.
- Two 6,400-tick contact replays are bit-identical in qpos, complete six rounds, and record 197 player points with zero modeled falls.

The separate position-regression evidence is under `../position-reset-20260914/`; this directory contains the build, common input regression, and training evidence only.

## Checkpoints

Private checkpoint directory: `/home/spark-advantage/rek-training/semantic-fast-20260914-v1/train-v3-33m/checkpoints/rek_native5/semantic-cuda-v3-512-16/`.

| Transitions | Filename | SHA256 |
| ---: | --- | --- |
| 1,048,576 | `0000000001048576.bin` | `e041aa11932fd9898631d6429c7bfaf3250491d5cbc2f0515ab5e66361066c1f` |
| 33,554,432 | `0000000033554432.bin` | `e209edd0d1301170c5253f622f5f11aabccb57daf7fd4117983942bccb989fc6` |

Checkpoint files, proprietary clips/models, and executables are not included in these repository artifacts.

## Reproduction and evidence

`reproduce.sh` contains the exact incremental build and matched training commands and refuses to overwrite the V3 build/output directories. It requires the existing private assets and V2 native build. `training/command.txt` records the executed training command; `build/commands.txt` records the build with shell tracing. Each phase preserves stdout, stderr, exit status, hashes, and process timing.

`summary.json` is derived from the raw artifacts by `node summarize.cjs`. That script is offline evidence reduction only. `artifact-hashes.txt` hashes the sanitized local evidence files. It does not start training, render, or step any environment.
