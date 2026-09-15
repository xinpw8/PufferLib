# V4 curriculum training execution

These artifacts record four completed native C++/CUDA training stages on
Spark. All use 512 arenas, 50 Hz learner decisions, 20-second rounds,
horizon 128, minibatch 8192, gamma 0.999 and GAE lambda 0.995. The first three
oppose a neutral robot; the fourth uses mixed opponents. These training
results do not establish frozen strength against a human or authentic REK.

The close-range stage starts fresh, samples root gaps of 0.55 to 1 m and
heading offsets up to 0.2 rad, and uses potential shaping weight 1. The
pursuit stage loads its final checkpoint, samples root gaps of 1.25 to 2.5 m
and heading offsets up to 0.35 rad, and uses shaping weight 10. Both retain
the same raw points-only combat dynamics. The orientation stage broadens
heading offsets to pi radians. The mixed stage assigns 128 arenas to a frozen
V3 policy and 384 to the runtime mixture of neutral, scripted, retreat and
strafe opponents. Runtime mixture weights are 0.25 each; realized assignment
varies. Per-stage parameters, actual frozen-row counts, opponent hashes and
complete native training settings are recorded in each summary.

The separate `train-v4-long-r1` experiment continues from the mixed checkpoint
for 134,217,728 further transitions in 300-second rounds with shaping zero.
Independent frozen evaluation regressed, so this candidate is rejected. Its
raw training artifacts, exact initialization and process status remain here,
but its transitions and times are excluded from the selected curriculum's
704,643,072 transitions and combined SPS. `summary.json` lists it under
`rejectedExperiments`, never under selected `stages`.

## Initialization

The actual archived trainer lacked a training initial-model load path.
`pufferlib5_initial_model.patch` adds weights-only loading before rollout.
Its SHA-256 at the verified pursuit build is
`f0bb1f634b62914a07417b7153a4f1e39f2fd04e8b0eff2af3df882cdd582092`.
The pursuit step-zero checkpoint exactly matches its source checkpoint:
`93f767f6fcc72738cb5e310ef351f8d3f3635a2e8d70a6129d88857b408fa48a`.
`verified-warm-start.txt` records both hashes after a successful bytewise
comparison. Optimizer, RNG, recurrent state, global step and learning-rate
schedule restart; this is not full-state resume.
Each later stage's source hash and step-zero readback are also checked
against the preceding stage's final checkpoint. Cumulative transition counts
sum this verified weights-linked chain; they do not imply optimizer-state
continuity.

## Measurement

`summary.json` and the per-stage reports are generated from copied raw text
artifacts with `node rebuild-summaries.cjs`. The script checks exit codes,
checkpoint hashes, warm-start linkage, final step counts, round accounting
and runtime identity before reporting results. It performs no training.

Training SPS is final learner transitions divided by final native loop
uptime, including rollout and PPO. The INI's final metric element is retained
exactly by the pinned trainer, even when earlier elements are downsampled.
Final rolling SPS and process-inclusive SPS have separate fields. Runtime
startup and asset preprocessing are outside the native loop and inside the
whole-process measurement. The shared Spark was not reserved exclusively for
these jobs. No throughput ceiling is claimed.

The first three stages measured about 2.72 to 2.74 million training
transitions/s. The mixed stage measured about 1.72 million transitions/s and
includes an additional frozen native policy inference on all 512 opponent
rows, with 128 outputs used. This extra computation is included in its SPS.

Whole-run raw round scores and outcomes are separate from the final rounded
dashboard's potential-shaped episode return. Internal GPU model/environment
timers remain zero in this trainer route and cannot be used as a breakdown.
Neither whole-run training wins nor the last dashboard window is a frozen
policy test. The corresponding held-out tests are recorded separately in
`../diverse-policy-v4-20260915/`.

The build-source manifests describe their earlier compile step. Their
`runtime_validation=not_run` field is preserved as originally emitted; the
later training command, process exit and round summaries establish the
executed runs. Checkpoints, raw game assets and private binaries are not
included here. Provenance contains target-host paths and hashes.
