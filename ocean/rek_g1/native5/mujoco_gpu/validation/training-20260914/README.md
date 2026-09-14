# Native MuJoCo GPU training result

Executed on `spark-4ae3`, NVIDIA GB10, through Windows PowerShell, WSL and
`ssh spark`. Exit code: **0**. Physics, controller inference, learner inference,
rollout storage and PPO execute in native CUDA. No Python or CPU physics
fallback was invoked. Native host code still handles setup, graph submission,
synchronization, reporting and checkpoints.

| Measurement | Result |
|---|---:|
| Learner transitions | 1,048,576 |
| Arenas / robots | 512 / 1,024 |
| PPO epochs / horizon | 128 / 16 |
| Mean training SPS | 6,020 |
| Complete process wall time | 175.97 s |
| SPS including startup and shutdown | 5,959 |
| Completed rounds | 946 |
| Learner wins / losses / draws | 261 / 367 / 318 |
| Learner / opponent completed points | 1,987 / 2,255 |
| Runtime failure bits | 0 |

An SPS unit is one learner transition. Opponent actions, physics substeps and
replayed optimizer samples are not counted as additional transitions.

Round totals cover the changing policy during training. They are not an
independent evaluation of the final checkpoint. The terminal console's `wins`
value is a recent logging sample and must not replace the cumulative totals.
No superhuman-strength or gameplay-parity conclusion follows from this run.

This is one configuration with other GPU workloads present. It is not a maximum
throughput result. The console's aggregate rollout timer dominates; its nested
Model/Env timers do not instrument this CUDA path and their displayed zeros do
not mean physics is free. Kernel-level profiling is reported separately.

The final private checkpoint is under the run directory in `result.json`:
`checkpoints/rek_native5/native5-mujoco_cuda-scripted-20s/0000000001048576.bin`.
Its hash is recorded. No checkpoint, game binary, controller weights or full
private state arrays are published here.

Included files contain the complete command, stdout/stderr, process timing,
exit status, round summary, checkpoint/input hashes and native link dependencies.
Model/config artifacts and complete asset hashes remain in the private run
directory. This run uses the corrected reset-contact refresh and GPU mask guard.

The recorded command inherited the following native backend inputs. Later
runner revisions include these values directly in the command record:

```sh
export REK_MUJOCO_KERNEL_CATALOG=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/kernel-catalog-v3.json
export REK_MUJOCO_CONDITIONAL_PTX=/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/build-native-v2/mujoco-conditional.ptx
export REK_MUJOCO_CONDITIONAL_SHA256=2300284dfc4f4560234ef6ffe8a3fab8c918a1abc8ffcdcf270e180042837b21
```

The process inventory in `host-and-models.txt` includes unrelated pre-existing
Python applications. Those processes were preserved and are outside this
native training execution path. A listed historical opponent checkpoint was
hashed by the old runner but was not loaded in this scripted-opponent run.
