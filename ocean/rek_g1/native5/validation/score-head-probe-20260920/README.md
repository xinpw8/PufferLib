# Native CUDA observational score-head measurement

The existing standalone score-head draft was executed unchanged on 2026-09-20.
It fits the training round, but the preregistered epoch-500 comparison shows no
seed-average held-out BCE or Brier improvement over a constant training-prior
baseline for either output. MLP16 retains modest local-score ranking signal;
its probability calibration worsens with training. This does not establish
useful reward shaping, behavior cloning, policy pretraining, or fighting gains.

The original untracked source files and the older validation directory were
preserved byte-exact. Only this new report was added to the repository. The head
was not connected to the simulator, policy, reward, or live client.

## Frozen experiment

The existing CLI runs three models with seeds 11, 29, and 73: a five-feature
geometry logistic model, a full 44-feature logistic model, and a 44-to-16-to-2
tanh MLP. Each receives 500 full-batch SGD updates, learning rate 0.03 and L2
coefficient 0.01. The objective is mean binary cross-entropy over examples and
both outputs plus half L2 times squared non-bias weights. Epoch 500 was the
preregistered primary result; epoch 100 was a diagnostic snapshot. There was
no held-out seed, epoch, architecture, or hyperparameter selection.

Exactly two rounds from one human session are used. Round ID 1 supplies 72
training windows, with 44 local-positive and 21 opponent-positive windows.
Round ID 2 supplies 104 held-out windows, with 44 local-positive and 27
opponent-positive windows. Each binary output asks whether any eligible paired
one/two-point client score receipt for that recipient appears in the next
three seconds. Five-point awards are excluded. These are received-score
outcomes, not causal labels for execution of the current requested attack.

Features contain preceding geometry, preceding movement commands, request
history, and the current requested move. The current request is a deliberate
conditioning input. Training-round means and population standard deviations
are supplied by the existing dataset and applied once on CUDA. The constant
baseline predicts the training fractions 44/72 and 21/72 for every example.

## Held-out results

All values below are at epoch 500. Lower BCE and Brier are better; higher AUROC
is better. The baseline uses training prevalence, never held-out prevalence.

| Model | Seed | Local BCE | Local Brier | Local AUROC | Opponent BCE | Opponent Brier | Opponent AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant training prior | fixed | 0.7532 | 0.2794 | 0.5000 | 0.5752 | 0.1932 | 0.5000 |
| Geometry logistic | 11 | 0.7748 | 0.2893 | 0.4299 | 0.6242 | 0.2146 | 0.2828 |
| Geometry logistic | 29 | 0.7751 | 0.2894 | 0.4311 | 0.6222 | 0.2138 | 0.2939 |
| Geometry logistic | 73 | 0.7776 | 0.2906 | 0.4201 | 0.6222 | 0.2138 | 0.2886 |
| Full logistic | 11 | 1.1013 | 0.3550 | 0.5030 | 0.7430 | 0.2364 | 0.2915 |
| Full logistic | 29 | 1.1048 | 0.3554 | 0.5087 | 0.7417 | 0.2371 | 0.2819 |
| Full logistic | 73 | 1.0926 | 0.3538 | 0.5152 | 0.7286 | 0.2334 | 0.2900 |
| Full MLP16 | 11 | 0.9420 | 0.3391 | 0.5598 | 0.6176 | 0.2090 | 0.3497 |
| Full MLP16 | 29 | 0.8024 | 0.2904 | 0.6205 | 0.5384 | 0.1787 | 0.6845 |
| Full MLP16 | 73 | 0.9210 | 0.3323 | 0.6072 | 0.5855 | 0.1979 | 0.5527 |

The three-seed arithmetic means are summaries, not ensemble predictions:

| Model | Local BCE / Brier / AUROC | Opponent BCE / Brier / AUROC |
| --- | --- | --- |
| Geometry logistic | 0.7758 / 0.2898 / 0.4270 | 0.6229 / 0.2141 / 0.2884 |
| Full logistic | 1.0996 / 0.3547 / 0.5090 | 0.7378 / 0.2356 / 0.2878 |
| Full MLP16 | 0.8885 / 0.3206 / 0.5958 | 0.5805 / 0.1952 / 0.5289 |

Every epoch-500 local model is worse than the prior in BCE and Brier. MLP seed
29 improves both opponent probability metrics, but the other two MLP seeds do
not; selecting that seed afterward would violate the frozen comparison.
Full logistic training AUROC averages 0.8801 local and 0.8836 opponent while
held-out AUROC is 0.5090 and 0.2878. Fitting this round is therefore not evidence
of transfer.

The diagnostic epoch-100 MLP snapshot has local held-out mean BCE 0.7133,
Brier 0.2596, and AUROC 0.6069, better than the local prior. Its opponent BCE
0.6274 and Brier 0.2175 are worse than the opponent prior. At epoch 500 the MLP
predicts local-positive probability 0.7089 on average despite an observed
held-out fraction of 0.4231. The early local signal and later deterioration
are compatible with overfitting and prevalence shift; they do not establish
that epoch 100 would generalize to a new session. Epoch-100 weights were not
saved by the unchanged executable, and no rerun or post-hoc selection occurred.

## Execution and numerical checks

Execution started at `2026-09-20T02:33:45.112673637Z` and finished at
`2026-09-20T02:33:45.589685982Z`; process wall time was 0.47 s, exit status 0.
The GPU was released immediately afterward. The nine fits plus their snapshot
measurements took approximately 0.009-0.040 s each. This tiny supervised probe
is not a policy-throughput benchmark.

The unchanged built-in CUDA self-test passed 856 finite-difference gradient
comparisons, maximum absolute error `1.588227683e-8`, three loss-reduction
checks, and 352 feature-scaler comparisons. All forward predictions, feature
scaling, gradients, reductions, and SGD updates for learned models ran on
CUDA. File parsing, metric calculation, the constant-prior calculation, and
the numerical test reference ran on the host. The eight existing dataset CPU
tests also passed. Independent code review found no blocking gradient or
metric defect.

## Reproduction and private evidence

The preserved experiment stage is
`/home/spark-advantage/rek-training/score-head-probe-20260920-r1` on Spark.
Its `source/` and `data/` contain byte-identical copies of the original draft
and exported data. The local evidence copy is
`C:\rekagent\work\score-head-probe-20260920-r1`.
These are experiment artifacts, not installed runtime components or a NAS
archive claim. Commands are issued from Windows through WSL SSH to Spark.

Given those preserved inputs and fresh output names, the exact native commands
are:

```bash
stage=/home/spark-advantage/rek-training/score-head-probe-20260920-r1
REK_NATIVE5_CUDA=/usr/local/cuda REK_CUDA_ARCH=sm_121 \
  bash "$stage/source/build_score_head_probe.sh" "$stage/build"
"$stage/build/score-head-probe" --run \
  "$stage/data/score-head-data.bin" "$stage/data/manifest.json" \
  "$stage/run-default-r1"
```

Existing build and run output directories are deliberately refused. Both
commands above were already executed in this stage; reproduction requires
fresh destinations. The compiler was official CUDA 13.0.88 targeting `sm_121`.

| Artifact | SHA-256 |
| --- | --- |
| Original probe source | `f14fe4e044184e5a3b053720a0754ad031433c87d9f77f956537392d06a26d27` |
| Original build script | `faad7d875bd4959cc8f639b33294be18721d9415b5528ceedf3f5b607b110617` |
| Dataset binary | `d874a739fef538c81b0bc8a60c443ec8de2053867d9c78bee8866df709c3560b` |
| Preregistration | `aac51e9cf1161c237bf0e13852101088f4da030bdf9bee11f32a2d0fa121e1b5` |
| Native executable | `6b16b4d5503fb61307d616369bbb3221551dab506464267acaf66a125f6b8ed8` |
| Complete report | `53fcbdc0b356e34a70b96c661a013d7af2884175b257c7c9bd6f85d0a99a3359` |

`run-default-r1/report.json` retains every seed, both epoch snapshots, both
splits, and all class metrics. Nine private parameter files are alongside it.
`run-default-r1.command.txt`, UTC start/finish files, `.time.txt`, `.stdout.txt`,
`.stderr.txt`, `.exit-code.txt`, and `.hashes.sha256` retain execution evidence.
All 19 listed result/log files were hash-verified after fetching to Windows;
all nine original source/data files were verified unchanged after execution.

## Transfer limits and usable conclusion

The 176 windows do not constitute 176 independent outcomes: there are only
18 distinct eligible training score events and 14 held-out events, repeatedly
included by overlapping three-second windows. Both rounds share one human
session. Only eight of 17 move IDs occur across that session, opponent
requests are unobserved, and zero labels mean no eligible received score in
the window. They do not mean that a requested attack executed and failed.

This head predicts two observational outcomes from 44 request-conditioned
features. It is neither an action policy nor a compatible initialization for
the existing recurrent 223-observation/33-action policy checkpoint. Its
parameter format is standalone `REKSHPW1`; no native reward/runtime consumer
was added. Outcome correlation cannot be substituted for causal action value.

The concrete result is that native CUDA fitting is feasible and fast, while
useful probability prediction is not established across the held-out round.
The limited local ranking signal warrants keeping the existing offline probe
as a diagnostic. These measurements do not support using this head as an
environment reward or treating it as evidence that behavioral pretraining
improves authentic fighting.
