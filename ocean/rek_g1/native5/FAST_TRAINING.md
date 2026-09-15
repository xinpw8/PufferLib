# Action-level GPU training

The `semantic_cuda` candidate treats locomotion and attacks as timed combat
actions. The policy does not generate torques or learn balance. Every action
tick advances 0.02 simulated seconds without waiting for wall time. For example,
the configured 157-tick move still lasts 3.14 simulated seconds.

`fast_runtime.cu` advances a batch of two-fighter arenas in one CUDA kernel.
Root movement, held inputs, move phase, swept strike contacts, points,
knockdowns, resets, observations and action masks are GPU-resident. The native
PufferLib 5.0 learner performs policy inference, rollout storage and PPO in
CUDA. Python and Torch are absent from this execution path. CPU initialization,
asset loading, CUDA submission, logging and checkpoint I/O remain ordinary host
responsibilities. The human viewer additionally uses CPU forward kinematics
to render pictures; it never integrates CPU physics.

Actual poses, strike geometry and move durations are baked from the installed
local assets. The reduced root dynamics, contact response and knockdown rules
are explicit modeling choices. Read [FAST_RUNTIME.md](FAST_RUNTIME.md) and
[FAST_ASSETS.md](FAST_ASSETS.md) for the retained data and approximations.
Successful training in this candidate does not establish authentic REK parity.

## Current V2 result

After fixing the session-counter observation described below, a fresh policy
trained for 33,554,432 transitions at 512 arenas in **13.374 seconds** of trainer
uptime: **2,508,920 training SPS**, or **2,400,174 startup-inclusive SPS**.
The same 50 Hz action clock and configured move durations remain in effect.
This is about 417 times the earlier approximately 6,020 SPS native articulated
MuJoCo result, with the explicit reduced-model differences documented here.

Frozen BF16 sampled evaluation then produced **1,007 wins, 4 losses and 13
draws in 1,024 matches (98.34% wins)** against the same GPU scripted opponent.
This used 128 arenas, four rounds per arena on each side, with no diagnostic
observation override. Side 0 had 504 W / 3 L / 5 D; side 1 had 503 W / 1 L / 8 D.
All runtime failure bits were zero. Greedy evaluation won 21:1 in each of the
two fixed side assignments. Those two deterministic fixtures do not establish
a general win-probability confidence interval.

The environment starts from fixed geometry; sampled policy RNG provides
trajectory variation. These results measure this scripted matchup, not human
strength or authentic-game parity. Final checkpoint SHA256:
`0fa325324083023d69f6e5b489792880e423b5b7c7bd06387eaf2c6007018fe2`.

Current timing evidence:
[V2 training](validation/semantic-fast-20260914/v2/training-v2-summary.json).

The separate 60-match, side-reversed league ranked the 33.6M sampled checkpoint
first with 19 W / 0 L / 21 D in its 40 games. It went 19 W / 0 L / 1 D against
the script, but **all 20 matches against the earlier checkpoint drew 0:0**.
The current strategy relies on an opponent approaching. It has not demonstrated
initiative against passive opponents or competent self-play. Training speed is
resolved at this operating point; general fighting strength is not.

The live V2 human evaluator is at `http://127.0.0.1:18769/`, with the 33.6M
BF16-sampled opponent selected and the human on orange. It also offers the
earlier checkpoint, the script and explicitly separate greedy variants.

## V1 measured headless training on Spark

Host: `spark-4ae3`, NVIDIA GB10, CUDA target `sm_121`, 2026-09-14.
Same pinned native trainer commit `773f923d80e73bdc255a2ba730c918b28e416aa1`,
459,008 parameters, two 256-wide recurrent layers, horizon 16, replay ratio 1.
Each reduced-model benchmark completed 33,554,432 learner transitions.

| Arenas | Minibatch | PPO epochs | Training loop | Training SPS | Startup-inclusive SPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8,192 | 4,096 | 13.303 s | 2,522,325 | 2,412,252 |
| 2,048 | 32,768 | 1,024 | 11.419 s | 2,938,464 | 2,786,913 |
| 8,192 | 131,072 | 256 | 9.971 s | 3,365,272 | 3,144,745 |

Training SPS is total learner transitions divided by the trainer's final
uptime, including rollouts and optimization. Startup-inclusive SPS divides
by the measured process wall time, including asset loading and final saving.
Opponent actions, rendered frames and physics substeps do not multiply the
reported transition count. Repeated padded terminal bins in Puffer's INI log
are not counted as independent samples.

The previous native articulated MuJoCo benchmark achieved about 6,020 training
SPS at 512 arenas, horizon 16, minibatch 8,192. The compact 512-arena result is
about 419 times faster. This comparison changes the dynamics model, controller
cost and scripted opponent implementation. It is not an equivalent-physics
solver speedup. The prior run also used only 1,048,576 transitions. A compact
run with that same transition count measured 2,467,940 training SPS.

These are observed operating points, not a maximum-throughput claim. Other
host workloads were preserved. Increasing arenas also increased minibatch
size and reduced optimizer-update count here, so this is not an isolated
arena-count experiment. The 512-arena run learned substantially faster per
transition under these settings. Its final reported training window had
99.6% wins versus the GPU script, while the entire changing-policy run had
27,735 wins, 6,867 losses and 4,787 draws. Neither figure is a held-out result.

### Round-counter distribution bug

Independent frozen BF16 sampled evaluation of the 33.6M checkpoint initially
won 642 of 1,024 matches (62.70%) against the same GPU script. The native
Puffer evaluator also reported approximately 61%, ruling out the standalone
policy loader as the main explanation. That native evaluation subsequently
failed during report formatting because `env/perf` is absent, so its process
is preserved as a diagnostic and is not counted as a successful evaluation.

V1 mistakenly exposed its cumulative session round counter as observation 186.
Late training consequently received a large counter, while fresh evaluation
started at one. A controlled GPU observation intervention held the same
checkpoint, seed, opponent, geometry and BF16 sampled inference fixed:

| Observation 186 | Wins | Losses | Draws | Win fraction |
| ---: | ---: | ---: | ---: | ---: |
| Constant 1 | 294 | 67 | 151 | 57.42% |
| Constant 64 | 510 | 0 | 2 | 99.61% |

These 512-match interventions are diagnostics and never enter policy rankings.
They establish a causal dependence on irrelevant session history. Independent
20-second round episodes require this policy feature to remain episode-local.
V2 sets it to one while retaining cumulative diagnostic counters for reporting.
Its checkpoints must be retrained and assigned a new runtime configuration
identity. V1 binaries, checkpoints and all measurements remain preserved.

## Run and inspect

Run these on the authorized Spark host via the existing WSL SSH connection.
Dependencies and private model/motion paths follow the existing native setup.
Every output directory must be new, preserving previous evidence.

```sh
bash ocean/rek_g1/native5/build_fast.sh /absolute/new-fast-build
bash ocean/rek_g1/native5/run_fast_probe.sh /absolute/new-fast-build /absolute/new-probe
bash ocean/rek_g1/native5/fast_input_probe.sh /absolute/new-fast-build /absolute/new-input-probe
bash ocean/rek_g1/native5/run_fast_training.sh /absolute/new-fast-build /absolute/new-training 33554432 512 16
node ocean/rek_g1/native5/summarize_fast.cjs /absolute/new-training
```

The run wrapper records exact command, stdout, stderr, return code, host,
runtime/asset hashes, configuration, round accounting, elapsed time and
checkpoint hashes. Checkpoints and source assets remain private on Spark.
The native checkpoint file stores FP32 weights; native training uses BF16
inference. Evaluation must select `precision: "bf16"` explicitly. Storage
dtype alone does not identify inference precision.

The matching human evaluator is independent of existing ports 18766/18768:

```sh
REK_EVAL_RUNTIME=semantic_cuda bash ocean/rek_g1/native5/build_eval.sh /absolute/new-fast-build /absolute/new-fast-eval
node ocean/rek_g1/league/prepare_fast.cjs /private/new-league /absolute/new-fast-eval/rek-eval-worker /private/runtime-config.json
node ocean/rek_g1/league/server.cjs /private/new-league/server.json
```

Forward loopback port 18769 over the existing SSH connection. Register only
compatible compact checkpoints under this backend/configuration. The
evaluator reuses the exact trainer runtime objects, so the selected trained
policy and the human act in the same reduced model. The 33-category action
space contains no quit, disconnect or match-reset action.

## Evidence

- [Training summaries and full logs](validation/semantic-fast-20260914/)
- [Asset bake verification](validation/fast-assets-20260914/)
- [V2 native evaluator verification](validation/semantic-fast-eval-v2-20260914/result.json)
- [Frozen policy and league results](validation/compact-policy-v2-20260914/README.md)
- [Earlier articulated MuJoCo training](mujoco_gpu/validation/training-20260914/result.json)

The GPU probes verify exact configured attack duration, held translation,
yaw interruption/resumption, rejection of stacked attacks, translation
settling, deterministic contact/down outcomes, completed rounds and CUDA graph
timing. They validate this implementation's rules, not authentic-game parity.
