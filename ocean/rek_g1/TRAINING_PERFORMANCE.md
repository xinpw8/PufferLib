# CUDA training optimization, 2026-09-11

The optimized runtime retains the existing Sonic controller, all 33 action
categories, held-input semantics, native combat rules, 500 Hz MuJoCo-Warp
physics, 50 Hz policy decisions, and native Puffer PPO/Muon learner.
It is still a semantic candidate. These optimization tests do not establish
control equivalence with authentic REK or superiority to human players.

## Implemented changes

1. `gpu_reset_forward_gate.py` conditionally executes reset-only forward
   recomputation when at least one arena resets. Previously it ran at all
   eleven reset boundaries per control tick, even with an empty reset mask.
   The predicate and branch remain on CUDA. Selected worlds use the same
   forward operation, reader restoration, solver settings and timing.
   Persistent, owned scratch supports the conditional CUDA graph body;
   ordinary eager calls retain the original behavior.
2. `gpu_combat_measurement_fused.cu` fuses live-contact validation, integer
   bookkeeping, contact-enter history and candidate packing. It preserves
   directed contact processing order and the original floating-point
   velocity/norm operations. Native hit scoring and fall/referee rules are
   unchanged. The synthetic measurement-only benchmark improved 3.47 times;
   that ratio is not an end-to-end training result.
3. `evaluate_gpu_dummy.py` evaluates frozen native checkpoints through the
   same rollout path, without PPO updates. It journals completed rounds and
   verifies identical policy hashes before and after evaluation. Action 32
   remains an emote and is now excluded from attack-facing statistics.
4. The external-GPU MinGRU PPO scan now consumes the same per-observation
   terminal mask as rollout inference. Forward recomputation resets recurrent
   state at each episode boundary, and backward propagation stops there.
   This corrects a training inconsistency; it changes neither the environment
   transition nor its reward. Legacy null-mask and custom-network paths retain
   their existing behavior.

## Runtime configuration

Build the contact library explicitly on Spark, outside the training loop:

```sh
python ocean/rek_g1/build_gpu_combat_measurement.py \
  --output /absolute/new/output/libmeasurement.so
```

The builder uses CUDA `sm_121`, writes compiler/source/library hashes and
compiler output to a build manifest, and preserves existing destinations.
It uses the CUDA toolkit's standard headers only. No proprietary asset or
controller is bundled with this source.

Add these fields to the existing, asset-pinned GPU duel JSON configuration:

```json
{
  "conditional_reset_forward": true,
  "fused_combat_library": "/absolute/new/output/libmeasurement.so"
}
```

All other configuration fields are retained. Existing configurations default
to the reference path, permitting direct comparisons. The measured optimized
training runs explicitly enable both fields. Unsupported conditional graphs,
changed scratch allocation layouts, and incompatible libraries raise errors;
there is no automatic CPU fallback.

Train against the current human-evaluation scripted opponent:

```sh
python ocean/rek_g1/train_gpu_duel.py \
  --gpu-duel-config /absolute/optimized-config.json \
  --opponent candidate-dummy --total-agents 1024 \
  --horizon 64 --minibatch-size 4096 --total-timesteps 3276800 \
  --load-checkpoint /absolute/initial.bin \
  --run-dir /absolute/new/training-run --output /absolute/new/report.json
```

The physical fighter count is 1,024; only the 512 learner rows enter PPO.
Learner SPS must not be confused with physical-fighter SPS or 2 ms physics
substeps. Asset/runtime dependency paths must already be configured as for
the reference CUDA runtime.

Rebuild the native `pufferlib._C` extension from the terminal-corrected source
as well. Updating Python files alone leaves the old recurrent learner in use.
The verified Spark deployment uses:

* Python stage: `/home/spark-advantage/rek-training/training-opt-20260911/optimized/ocean/rek_g1`
* Native package: `/home/spark-advantage/rek-training/gpu-runtime-20260910/native-terminal-reset-v1`
* Configuration: `/home/spark-advantage/rek-training/training-opt-20260911/config-combined.json`
* Final trained weights: `/home/spark-advantage/rek-training/training-opt-20260911/training/corrected3276800-r1/0000000003276800.bin`

Run commands, full build manifests and dependency paths are preserved in
`C:/rekagent/evidence/rek-training-opt-20260911/commands/` and
`C:/rekagent/evidence/rek-training-opt-20260911/native-terminal-reset-v1/`.
The policy path identifies a completed training artifact, not an accepted
stronger fighter.

## Matched training measurements

These measurements predate the recurrent terminal correction below. Both
versions used the same original learner extension, so they remain a matched
runtime comparison. They are not corrected-learner throughput measurements.

Three alternating runs per version used 512 learner arenas, 1,024 physical
fighters, 262,144 learner steps, horizon 64 and minibatch 4,096. Every process
started from the same checkpoint and native configuration. The model,
controller, scripted opponent and native learner extension were identical.

| Version | Run 1 learner SPS | Run 2 learner SPS | Run 3 learner SPS | Pooled learner SPS |
| --- | ---: | ---: | ---: | ---: |
| Original | 3,000.57 | 3,015.41 | 2,955.03 | 2,990.12 |
| Conditional reset plus contact fusion | 6,006.30 | 5,897.48 | 6,017.60 | 5,973.30 |

The pooled improvement is **1.99768 times**, calculated as total learner
steps divided by total training wall time within each group. Physical-fighter
SPS is twice learner SPS for this fixed-opponent experiment. These runs each
cover 10.24 simulated seconds per arena and complete no rounds; they measure
throughput, not fighting improvement. Setup and final checkpoint saving are
excluded, while rollout, PPO and scheduled reporting are included.

Physics, controller inference, opponent logic and learning stay on CUDA.
Native CPU environments and environment worker threads remain disabled.
CPU graph submission, synchronization, initialization and file/report I/O
remain present. Unrelated resident Spark processes were preserved, so these
repetitions do not exclude every possible source of external contention.

## Behavioral verification

The contact adapter passed CPU and CUDA differential fixtures, including
graph replay, contact history, invalid transactions and floating-point
velocities, with zero tolerance on consumed values. The reset-body diagnostic
passed false, mixed and full reset masks over 20 repeats with the unchanged
paired-repeat numerical criterion.

The complete environment replay used four arenas, 256 control ticks and four
repeats each of original A, original B and the combined candidate. The arenas
covered idle, held movement followed by an accepted E-to-kick interruption,
a close-range kick, and an explicitly synthetic fall measurement driving
the unchanged native five-point KO and mixed physical reset.

Both original references must have independent captured graphs. Reusing one
original graph for A and B while allocating another graph for the candidate
confounded the initial comparison. The independent-graph control passed all
seven numerical criteria without changing their thresholds. Body-position
RMS differences were 0.03880 m between original references and 0.03813 m
between original A and the candidate over the complete interacting trace.
At the first step in the earlier trace, original and candidate maximum qpos
differences were both approximately 4.5e-7 in mixed qpos units.

The strict all-fields-exact diagnostic still reports failure, because the
original engine itself varies in two transient fall fields. Across four
repeats, A-B and A-candidate each have three fall-phase and six fall-event
element differences. Candidate-B is exactly equal on all 26 discrete fields.
The observed event variants both occur in unchanged original runs. This
supports the explicit optimized configuration within the observed reference
variation; it does not establish bitwise deterministic trajectories.
The failure report is preserved, with separate pairwise analysis.

This short replay did not exercise a physical scored hit or a terminal round.
Its first requested Q-to-front-kick input was rejected by the original action
mask and recorded as neutral. Those events are not represented as covered.

## Native recurrent terminal correction

External rollout already reset MinGRU state before inferring an observation
whose terminal flag is set. PPO previously replayed that sequence without its
terminal mask. Consequently, unchanged policy weights could produce different
action log probabilities during rollout and training, causing artificial PPO
KL and clipping when a minibatch sequence crossed a round boundary.

`src/pufferlib.cu` now selects the terminal row alongside that minibatch's
observation row, including when no action mask exists, and binds the persistent
CUDA mask before graph capture. `src/models.cu` applies the reset before the
corresponding zero-based observation in every MinGRU layer, including backward
checkpoint recomputation. Gradients cannot cross a reset. The masked path also
uses the finite analytic initial-state gradient at zero initial state.
Mask allocation is restricted to external-GPU MinGRU training.

Executed checks:

* An independent CPU finite-difference fixture passed 14 reset/initial-state
  cases, with maximum gradient absolute error 4.643e-10. It also checked
  terminal-row transpose/selection alignment and zero gradient across resets.
* The actual CUDA scan on GB10 passed 42 forward/backward cases at sequence
  lengths 4, 8 and 64, plus six terminal-mask mutations through a captured
  graph. Maximum output error against the oracle was 2.917e-6; maximum combined
  input-gradient error was 3.211e-6. Future gradients did not cross a reset.
  These executed results use FP32; they do not validate BF16.
* A full native-extension negative control used identical synthetic policy
  weights, learning rate zero, horizon 8 and CUDA graphs. The original extension
  reproduced the mismatch. The corrected extension reported `kl=0` and
  `clipfrac=0` in all seven cases, with unchanged checkpoint hashes and finite
  losses. This is a policy-replay regression, not a fighting evaluation.

| Reset before zero-based observation | Original KL | Original clip fraction | Corrected KL / clip fraction |
| --- | ---: | ---: | ---: |
| None, initial capture and later replay | 0 | 0 | 0 / 0 |
| 2, 4, 6 | 0.02332247 | 0.265625 | 0 / 0 |
| 4, at the scan checkpoint | 0.00837249 | 0.140625 | 0 / 0 |
| 0 | 0 | 0 | 0 / 0 |
| 7, final observation | 0.00739300 | 0.093750 | 0 / 0 |
| 3, 4, adjacent resets | 0.01068324 | 0.171875 | 0 / 0 |

The table uses native `kl`, not the separate signed `old_kl` estimator. In the
corrected cases, that estimator retains floating-point residuals around 1e-7.
This does not claim bitwise equality of all floating-point intermediates.
The original extension SHA-256 is
`9f030f49defb26d175ee917c7747c9e3b067b156ff728651a0c5fc87f470f431`;
the corrected full extension is
`f6ad77a5e0ce151894b6d1f9945a9e5374ba294761e030eaf2de03cbcaafdfdc`.
Evidence: `commands/recurrent-reset-cpu-review-001/stdout.txt`, `scan.json`,
and `native-terminal-replay/{original,fixed}.json` under the evidence directory.

## First long run and frozen policy comparison

The completed first long run, `combined3276800-r1`, used the original learner
extension before the terminal correction. It executed 3,276,800 learner steps
in 582.309 s at 5,627.25 learner SPS, with 512 arenas, horizon 64 and 100 epochs.
This is 128 simulated seconds per arena and 6,553,600 physical-fighter steps.
It recorded zero CPU physics steps and CPU controller inferences. Its 512
completed training rounds yielded 411 wins, 84 losses and 17 ties, or 80.27%
wins. Those outcomes mix changing policy weights and cannot evaluate the final
checkpoint as a frozen policy.

Separate frozen evaluations used 128 arenas, seed 73, the same initial qpos,
6,400 control ticks per arena (128 simulated seconds), stochastic categorical
sampling, and the same candidate dummy. Both reset recurrent state every
64-step horizon, made zero PPO updates, and verified unchanged policy hashes.
Each completed 128 rounds; ongoing rounds at cutoff are excluded from the
completed-round statistics. All-step action and contact counts include the
unfinished rounds.

| Frozen policy metric | Initial | First trained |
| --- | ---: | ---: |
| Completed-round wins / losses / ties | 91 / 33 / 4 | 82 / 35 / 11 |
| Round win percentage | 71.09% | 64.06% |
| Learner points per completed round | 11.90625 | 11.92969 |
| Opponent points per completed round | 7.39844 | 8.71094 |
| Native discrete move starts, all steps | 4,235 | 5,442 |
| Arena scored hits, all steps | 379 | 553 |
| Facing opponent within 30 degrees | 38.75% | 32.83% |
| Attack-request samples facing opponent | 43.90% | 40.51% |
| Mean horizontal opponent range | 1.06353 m | 1.03060 m |

The frozen win rate fell 7.03125 percentage points. This comparison provides
no evidence of improved fighting performance. One seed and 128 completed
rounds per policy do not establish the cause of that difference or a reliable
multi-seed effect. In particular, the identified recurrent bug cannot by
itself explain the observed policy-quality difference without corrected-run
evaluation. More scored contacts do not imply more wins; those contacts are
arena totals, and points include referee awards. Neither policy was tested
against authentic Bot 1 or humans. No superhuman or authentic-REK-parity claim
is supported.

The initial policy SHA-256 is
`21b0c0e88c6fade10313add9d269b54c1d43fd81276c35247de9c49be7630cea`.
The first trained policy SHA-256 is
`7575182d6b4c2bf0fa7fa558e9210cea2d0c7a8832fd4d279335eb646dfffd77`.
Evidence is in `combined3276800-r1/reports/combined3276800-r1.json`,
`frozen-first-comparison.json`, and the two
`eval-{initial,trained}256-ticks6400-v1/evaluations/` report directories.
Their instrumented evaluation throughput is not training SPS.

The training budget supplies approximately one complete 120 s round per
arena. First round completions appeared at epoch 94 of 100, when the cosine
learning-rate schedule had reached approximately 1.20% of its initial value.
Reward is immediate learner-minus-opponent point change, without a terminal
win, facing or range reward. The first frozen comparison shows essentially
unchanged learner scoring and increased opponent scoring. These observations
identify limitations of the budget and objective; they do not isolate a
hyperparameter as the cause or justify inventing a successful policy result.

### Corrected learner throughput

The corrected extension completed `corrected262144-r1`, a 262,144-step run
with the same initial weights and settings as the short comparisons above:
43.3893 s and 6,041.68 learner SPS. This preserves the measured performance
improvement. It is a single additional check, not another three-pair estimate.
The CUDA stream envelope attributed 99.543% to rollout, 0.417% to PPO and
0.039% to reporting. Host timings overlap these intervals. CPU physics and
controller execution counts remained zero.

### Corrected learner long run

`corrected3276800-r1` loaded the first trained checkpoint and completed an
additional 3,276,800 steps in 580.382 s, or 5,645.94 learner SPS. Both long
runs together provide 6,553,600 learner steps of policy training. This run
restored policy weights only, with fresh optimizer and environment state.
It completed 512 training rounds: 361 wins, 126 losses and 25 ties. Mean
completed-round points were 12.82422 for the learner and 8.55664 for the
opponent. These are changing-policy training statistics, not a frozen test.
The report again records zero CPU physics/controller execution and no CPU
environment workers. Physics settings, opponent and action semantics were
unchanged.

The final policy SHA-256 is
`87d9ef33f2893ad8b369783e8fecbd95afc4a4793cce8d701adf0423b933a544`.
Evidence is in `corrected3276800-r1/reports/corrected3276800-r1.json`.

The final frozen evaluation completed the same 128 rounds, starting from the
same states, seed, configuration and inference implementation as both earlier
evaluations. It made zero policy updates and preserved its checkpoint hash.

| Frozen policy metric | Initial | First trained | Corrected, further trained |
| --- | ---: | ---: | ---: |
| Wins / losses / ties | 91 / 33 / 4 | 82 / 35 / 11 | 82 / 38 / 8 |
| Round win percentage | 71.09% | 64.06% | 64.06% |
| Learner points per completed round | 11.90625 | 11.92969 | 12.06250 |
| Opponent points per completed round | 7.39844 | 8.71094 | 8.36719 |
| Facing opponent within 30 degrees | 38.75% | 32.83% | 29.55% |
| Attack-request samples facing opponent | 43.90% | 40.51% | 44.60% |
| Mean horizontal opponent range | 1.06353 m | 1.03060 m | 0.99839 m |

The additional run did not establish improved fighting ability. The final
win-rate Wilson 95% interval is 55.45% to 71.85%, compared with 62.72% to
78.24% initially, under an independent-round assumption. This one-seed test
does not establish a robust negative effect either. Additional training and
the recurrent correction changed together, so the sequence cannot isolate
the correction's effect on fighting quality. The correctness claim comes
from the separate unchanged-weight native replay and gradient tests.

The final policy is retained as a training artifact; it is not promoted as a
stronger fighter. Evidence is in `frozen-corrected-comparison.json` and
`eval-corrected256-ticks6400-v1/evaluations/eval-corrected256-ticks6400-v1/report.json`.

## Remaining timing breakdown

The corrected long run's CUDA stream envelope attributed 99.576% to rollout,
0.390% to PPO, 0.029% to reporting and 0.003% to periodic checkpointing.
These are stream intervals, including waits and submission gaps, rather than
hardware kernel-busy or occupancy measurements.

A separate 512-arena, 50-tick neutral-versus-dummy component probe measured:

| Component | Percentage of instrumented environment graph interval |
| --- | ---: |
| Physics steps | 75.657% |
| Combat measurement and referee | 10.788% |
| Controller inference | 4.054% |
| Reset forward selection | 2.984% |
| State and observations | 1.890% |
| Semantic motion | 1.697% |
| Actuator drive | 1.353% |
| Unattributed | 1.577% |

The component denominator excludes PPO and wrapper/opponent work. Timing
nodes perturb this diagnostic; these percentages cannot be added to the
long training percentages. Its separate uninstrumented neutral probe reached
7,546.00 learner control steps/s, which is not training SPS. Evidence is
`combined-components512-50ticks.json`.

The deployed choice is conditional reset forward plus contact fusion with
the corrected native learner. Increasing arena count or optimizing PPO does
not address the measured dominant cost. The remaining large target is the
articulated-physics step. Solver/kernel changes need their own trajectory
comparisons; neither reducing physics frequency nor substituting an
unvalidated engine is included in the reported speedup.

## Rejected approaches

Dense Jacobians are unsupported by this installed backend for the model's
70 velocity degrees of freedom; its limit is 60. The default sparse solver
was retained. Increasing beyond 512 arenas and reducing constraint capacity
had already shown no useful training throughput gain. No solver iteration
limit, tolerance, contact capacity or simulation timestep was reduced.

Runtime assets, checkpoints, raw traces, commands and complete output remain
in the external evidence directory `C:/rekagent/evidence/rek-training-opt-20260911`.
They are not distributed with the repository.
