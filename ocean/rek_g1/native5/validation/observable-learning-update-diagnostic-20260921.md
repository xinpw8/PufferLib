# Observable-balance parameter update diagnostic, 2026-09-21

The saved checkpoints demonstrate nonzero actor, critic and recurrent updates,
including changes visible after BF16 rounding. They do not establish improved
fighting strength. This diagnostic used CPU-only Node on Spark, read existing
files, and performed no policy inference, GPU work or game interaction.

## Measured changes

Each checkpoint contains 459,008 little-endian FP32 master weights. The verified
layout is encoder `[256,223]`, combined actor/value decoder `[34,256]` with the
value row last, and two recurrent matrices `[768,256]`. Counts below are exact
changed float32 values; numerical and bitwise counts agree. Delta norms use
float64 accumulation. Relative delta is `||after-before||2 / ||before||2`.

The physical comparison starts after epoch 1 and ends after epoch 16, covering
3,932,160 additional transitions and 480 optimizer steps. The continuation
comparison starts at its saved step zero and ends at step 2,359,296, covering
nine rollout epochs and 288 optimizer steps. Later continuation checkpoints
were deliberately excluded from this fixed diagnostic.

| Block / total weights | Physical changed | Physical delta L2 | Relative delta | Continuation changed | Continuation delta L2 | Relative delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder / 57,088 | 11,008 | 0.055843 | 0.4268% | 11,008 | 0.051960 | 0.3971% |
| Actor decoder / 8,448 | 8,448 | 0.047082 | 1.4275% | 8,448 | 0.048185 | 1.4608% |
| Value decoder / 256 | 256 | 0.014333 | 2.4305% | 256 | 0.008167 | 1.3948% |
| Recurrent layer 0 / 196,608 | 196,603 | 0.094556 | 0.5913% | 196,606 | 0.087943 | 0.5499% |
| Recurrent layer 1 / 196,608 | 196,608 | 0.200496 | 1.2549% | 196,607 | 0.186011 | 1.1642% |
| All / 459,008 | 412,923 | 0.233838 | 0.8880% | 412,925 | 0.217767 | 0.8270% |

All input weights were finite. After round-to-nearest, ties-to-even BF16
conversion, 272,578 physical-comparison weights and 276,418 continuation weights
still differ. This tests weight representability, not policy-output equivalence.
Encoder changes occupy exactly 43 input columns, with all 256 weights changing
in each; 180 columns have zero delta. Zero changes alone do not distinguish
structural padding from available features that remained zero during training.
The JSON retains before/after norms, RMS and maximum deltas, and column counts.

Continuation step zero is byte-identical to the physical final checkpoint.
The smoke initialization `d8dc477...` used base seed 73, whereas physical
training used seed 419. A smoke-initial-to-physical-final comparison confounds
initialization and training, so it is excluded from the learning table.

## Actual pinned trainer semantics

Inspected stage root:
`/home/spark-advantage/rek-training/physical-observable-balance-20260921-r1`.
The build manifest pins PufferLib commit
`773f923d80e73bdc255a2ba730c918b28e416aa1`; the inspected files are the actual
patched build inputs in `trainer-build-r1/trainer/src/pufferl.cu`,
`trainer-build-r1/trainer/src/algo.cu` and
`trainer-build-r1/trainer/config/default.ini`, rather than the newer repository
trainer.

- Pinned default learning rate is **0.015**. The physical run overrides it with
  **0.0001**, 150 times smaller. The default's existence does not establish its
  suitability for this physical task.
- There are 32 minibatch optimizer steps per rollout epoch, hence 512 across
  the complete 16-epoch physical run. Learning rate is
  `0.0001 * 0.5 * (1 + cos(pi*t/16))`, for `t=0..15`: final-epoch rate
  approximately `9.60736e-7`, mean rate `5.3125e-5`. A weight-only continuation
  starts a fresh optimizer and learning-rate schedule.
- Muon globally clips gradients, uses FP32 momentum, L2-normalizes each matrix
  update, applies five Newton-Schulz iterations and aspect scaling, then updates
  FP32 master weights. BF16 forward weights are refreshed after each minibatch.
  Thus reward scale 0.01 does not imply an optimizer step scaled by 0.01.
- GAE is accumulated in float32 and stored in BF16. PPO consumes raw advantages
  without mean/std normalization. Entropy remains 0.001; value coefficient is
  0.5. Reward scale can change their relative contributions even with Muon
  normalization. Their actual gradient contributions were not measured here.
- Dashboard losses use `%.3f`. The INI writer explicitly excludes every
  `loss/*` field. Exact KL, policy loss and value loss therefore cannot be
  recovered from these saved logs. Displayed `0.000` does not prove exact zero.

These results exclude completely stalled updates and an FP32-master/BF16-cast
no-op. They justify controlled learning-rate comparisons; they do not identify
a numerical bug. The next planned arms use 0.001 and 0.015 with controlled
initialization, data budget and evaluation. Their results must establish
whether larger updates help. Full-precision loss/KL logging is needed to
resolve the current display gap. This diagnostic launched neither arm.

## Reproduction and private preservation

Private authored script and runner:
`C:\rekagent\work\observable-learning-diagnostic-20260921-r1\measure-weight-updates.cjs`
and `run-diagnostic.ps1`. The recorded command is:

```text
wsl.exe -e bash -lc "ssh spark node < /mnt/c/rekagent/work/observable-learning-diagnostic-20260921-r1/measure-weight-updates.cjs"
```

Exit code 0, Node v18.19.1 on Linux arm64. The script checks expected input
hashes, checkpoint sizes and finite values, tests its BF16 rounding and norm
calculation, then rehashes all inputs after measurement. It emits aggregate
JSON and source excerpts only. Checkpoint or trainer binaries are not copied.
The runner refuses to overwrite existing evidence outputs.

Private NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-observable-balance-r1\update-diagnostic-r1`.
The existing project directory was verified before creating this fresh
subfolder. All nine files passed source-before/source-after/destination hash
comparison; no existing file was overwritten.

| Artifact | SHA256 |
| --- | --- |
| CPU diagnostic script | `5fe79600da65b4b9fcc1d4543c9c01e415358ec1d6223023cb847c005a74cd7a` |
| Aggregate result JSON | `dcdaf5588b015b763e7217446a76b0a6706549f81a0c5e5cce5e046e33037088` |
| NAS copy receipt | `1024e32eb0554e64bca4851f4c68c7f6212879c4fd4a117a7557f381da475055` |
| Actual trainer binary | `bd696ecc152c8b326bd6891bdf60ee97f7b0fa523f39e98696fe21708165185d` |
| Actual `pufferl.cu` | `ae71826468701bf19691548555c1a2d354f8795065bb3bc7fb6e5e2e2b0eb378` |
| Actual `algo.cu` | `8a514cb8dd12d49b79cbd5afe7298875b6f0ca0491270bb19a8696bd527f4d92` |
| Physical epoch 1 | `bc8567fe6b6623563bb99e88b5d02fd3ba8264f4df05b7c4fe84550ba963c72f` |
| Physical final / continuation step zero | `390007e256574d2fc5e1100eab4da048fccfec2532bd5e5a59bdcc19cf326310` |
| Continuation step 2,359,296 | `df5a10303f15c9fbc933914442d38c5c1244167f30e3e13f357b5d0c974c6680` |

Exact checkpoint paths, all eleven input hashes, source excerpts and execution
timestamps are in `result.json`, `input-hashes.sha256` and `command.json`.
No private weights, binaries or raw captures are published in this report.
