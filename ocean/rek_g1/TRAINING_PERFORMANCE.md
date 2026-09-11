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

## Matched training measurements

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

## Rejected approaches

Dense Jacobians are unsupported by this installed backend for the model's
70 velocity degrees of freedom; its limit is 60. The default sparse solver
was retained. Increasing beyond 512 arenas and reducing constraint capacity
had already shown no useful training throughput gain. No solver iteration
limit, tolerance, contact capacity or simulation timestep was reduced.

Runtime assets, checkpoints, raw traces, commands and complete output remain
in the external evidence directory `C:/rekagent/evidence/rek-training-opt-20260911`.
They are not distributed with the repository.
