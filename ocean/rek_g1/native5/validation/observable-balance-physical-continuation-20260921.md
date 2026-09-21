# Physical observable-balance continuation, 2026-09-21

The bounded warm-start run completed **4,194,304 new learner transitions and
16 updates**, with runtime and wrapper exit codes 0. Its final checkpoint is
`a11ace1655cf1f3176dc0732e07fef47a6d3e8b32177a5da47cd075ac4a231bd`.
This is changing-policy physical simulator training against
`CandidateApproachDummy`. It does not establish improved authentic REK strength.

## Initialization and executed configuration

The input was the completed [original physical training run](observable-balance-physical-training-20260921.md)
checkpoint `390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310`.
No behavioral-cloning checkpoint or critic transformation was used. The copied
input and native step-zero readback both have that exact hash and are
byte-identical. The final checkpoint differs.

This was weights-only initialization with fresh optimizer state, RNG, recurrent
state and step count. It was not a full-state training resume. The architecture
is 223 observations, 33 actions and two 256-wide recurrent layers, with 459,008
FP32 master parameters and BF16 policy execution. The trainer's literal
`--base.load_model_path` was checked against the generated source; its warm-load
branch checks the file size and publishes the exact step-zero checkpoint.

The clean native binary and physical environment were unchanged. Startup
receipts confirm `mujoco_cuda`, native SONIC inference, CPU physics disabled,
no Python runtime, `rek.native5.observable_balance.v1` and
`normalized_points_falls_v1`. The schema is bound externally to the checkpoint
hash, since the flat weight file has no embedded observation metadata. Root
origin/quaternion and preceding-observation history are populated; shared joint
pose/rate mapping remains unavailable. Native count state is available.

Configuration: 512 arenas / 1,024 fighters; horizon 512; minibatch 8,192;
4,194,304 transitions; replay ratio 1; 120 s rounds; 50 Hz actions with stride 1;
base and environment seeds 419; initial learning rate 0.0001 with cosine
annealing; entropy coefficient 0.001; gamma 0.9998844821426083;
GAE lambda 0.9978673240629938. Each arena executed 8,192 learner steps,
or 163.84 simulated seconds. The 1,200 s process timeout was not reached.

The opponent remains the existing deterministic approach/backoff/turn and
cycling-attack dummy using raw inspection observations. It is neither the
recovered Bot 1 controller nor a self-play policy. No loss, reward, controller,
opponent or physics changes were made for this continuation.

## Complete-run results

| Measurement | Actual result |
| --- | ---: |
| New learner transitions / updates | 4,194,304 / 16 |
| Completed rounds | 512 |
| Learner wins / losses / draws | 232 / 242 / 38 |
| Completed-round points, learner : opponent | 5,410 : 5,538 |
| Redos / unclassified results | 0 / 0 |
| Runtime failure bits | 0 |
| Lifetime confirmed falls, learner : opponent | 956 : 1,120 |
| Lifetime awarded points, learner : opponent | 6,817 : 7,352 |
| Reward saturations | 0 |
| Complete native trainer uptime | 700.117 s |
| Transitions / native trainer uptime | 5,990.86 SPS |
| Whole-process elapsed time | 709.57 s |
| Transitions / whole-process elapsed time | 5,911.05 SPS |

Completed-round counters exclude unfinished rounds. Lifetime fall and point
counters cover all executed runtime transitions, including unfinished rounds.
Fall counts are actual native `BECAME_FALLEN` events, not inferred pose labels.
Reward normalization remained fixed at 0.01, with own confirmed-fall penalty
-0.01, bounds [-1,1], no terminal bonus and no saturation events.

Native uptime includes every completed rollout, PPO update and periodic
checkpoint write. Its clock starts after initial construction and is shifted
to exclude CUDA graph-capture construction. Whole-process time includes those
costs and final shutdown. Display resolutions are 1 ms and 0.01 s respectively.
These are complete-training throughput measurements, not physics-only SPS or
the final dashboard window's SPS.

The original run recorded 204 wins, 274 losses and 34 draws in its 512 completed
training rounds. The new totals describe a different changing-policy training
cohort. They are not a frozen paired evaluation and cannot establish stronger
authentic play. Authentic evaluation is tracked separately.

## Reproduction and preservation

Spark stage:
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1`.

Completed output: `train-warm-physical-continue-r1`.
Final checkpoint relative path:
`train-warm-physical-continue-r1/checkpoints/rek_native5/warm-physical-continue-r1/0000000004194304.bin`.

Executed wrapper command:

```sh
bash /home/spark-advantage/rek-training/physical-observable-balance-20260921-r1/run-balance-physical-warmstart.sh \
  /home/spark-advantage/rek-training/physical-observable-balance-20260921-r1/train-balance-physical-r1/checkpoints/rek_native5/balance-physical-r1/0000000004194304.bin \
  390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310 \
  physical-continue-r1
```

The exact expanded native command and environment are preserved in `command.txt`.
The archive retains initial weights, step-zero and all 16 update checkpoints,
their hash-bound observation-schema sidecars, source receipts, raw logs,
per-update metrics, timing, scripts, the clean binary and build provenance.
All referenced build/provenance/checkpoint hashes and checkpoint schema sidecars
were reverified successfully. No additional trainer, inference or game process
was launched by the monitoring and collection work.

| Artifact | SHA256 |
| --- | --- |
| Clean trainer | `bd696ecc152c8b326bd6891bdf60ee97f7b0fa523f39e98696fe21708165185d` |
| Executed warm-start runner | `b81c0a0dc343c38c66f764e0f5c44e840aefc5f72c3485b0b4834b62708141a8` |
| Final checkpoint | `a11ace1655cf1f3176dc0732e07fef47a6d3e8b32177a5da47cd075ac4a231bd` |
| Training archive | `92e933a572a5399ce3b3751acd15c54d50c5fdcdfeb0be1601f6d9b2099468b5` |
| Aggregate result | `ec61af6d17590b42b72344d0b8d6c6bcbdf54224bac35c29783413f5dd96b47d` |
| NAS receipt | `3fe9cf029da1d62a0bc7c39712a3f855ed5eb6d9b98955fa90c5e386c078efa0` |

Fresh private NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\training-continuation-r1`.

The 32,809,159-byte archive contains 74 selected files plus aggregate and inventory
sidecars. Source hashes before/after packaging, tar comparison, and Spark/local/NAS
archive hashes passed. The original training archive was reverified and all
destination files were new. No existing evidence was overwritten. No proprietary
assets, binaries, checkpoints or raw captures are published here.

## Prepared controlled learning-rate follow-ups

A separate runner was prepared without launching another GPU job:
`run-balance-physical-lr-followup.sh`, SHA256
`17e90c448e3ba3230aaaa2f813bf25ea3b0a74824887cb4ffa5ad2698b6e50b2`.
It accepts only initial learning rates `.001` and `.015`, fixes initialization
to the original `390007...` checkpoint, and preserves all other physical run
settings and seeds. It creates fresh output labels, records the selected rate
in command/schema provenance, and supports CPU-only `--check-only` validation.
Both allowed rates passed non-launching checks; `.003` and `.0001` were rejected.
The original fresh-start and warm-start runners remain unchanged. The follow-up
preparation is separate from this completed run and its training archive; it is
not evidence that either higher-rate experiment has run or improved outcomes.
