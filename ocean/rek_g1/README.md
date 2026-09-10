# REK Unitree G1 semantic environment

This directory contains the native, state-based Unitree G1 duel candidate for
PufferLib. It is executable on the DGX Spark and no longer exposes direct
actuator actions. It is not yet accepted as a control-equivalent REK clone.
Acceptance requires held-out REK action sequences whose trajectory and event
errors are no greater than REK's repeated-run variance.

## Runtime boundary

### CUDA candidate

`gpu_semantic_duel.py` runs controller inference, motion scheduling, MuJoCo
Warp physics, contact measurement, fall/referee state, rewards, and observations
on CUDA. Model parsing, asset hash verification, and initial uploads occur once
on the host. The step path does not run CPU physics or CPU controller inference.
Python still orchestrates graph launches; this does not mean zero CPU usage.

`GpuDuelConfig` requires the model and asset-manifest hashes, exact-batch
controller manifest and source bundle, baked motion-foot features, and compiled
motion/combat library paths. Asset payloads remain external. Build the native
libraries with `build_g1_motion_cuda.sh` and `build_g1_combat_cuda.sh`, using
separate output directories. The controller batch must contain complete pairs.
Call `capture_step()` before stepping the environment.

`verify_gpu_duel.py --config CONFIG.json --scenario approach-kick --out REPORT.json`
runs the complete GPU candidate and saves observations, applied/requested
actions, masks, motion starts, reference roots, body positions, and arena clocks
in a hashed NPZ trace. `turn-kick` checks a move after held Q; `front-kick`
isolates one kick and accepts an explicitly reported fixture starting distance.
Probe throughput is environment control-step throughput, not training SPS.

`gpu_native_puffer.py` connects these CUDA buffers to the original native
PufferLib policy, prioritized replay, PPO, Muon optimizer, and checkpoint
format. External GPU mode allocates no CPU environments or environment worker
threads. `train_gpu_duel.py` requires an explicit positive training-step count
and create-new run/report paths. Its report records measured training SPS,
host CPU time, completed-round combat metrics, precision, and checkpoint and
extension hashes. The default reward transform for this path is no clipping;
the legacy training path retains its existing default.

Native shared-policy self-play statistics do not measure superiority to a
human or a frozen opponent. Fighter rows are interleaved within arenas, so a
contiguous frozen-policy bank split alone is not an opponent evaluation.

`train_gpu_duel.py --opponent candidate-dummy` instead trains each even player
row against the human evaluator's fixed `CandidateApproachDummy` on the odd
row. Its approach, facing, range thresholds and 16-move cycle are unchanged;
this opponent is not authentic REK Bot 1. The CUDA implementation is compared
directly with the original Python decisions. Opponent samples never enter PPO.
`--total-agents` remains the physical fighter count, so 256 fighters produce
128 learners. Reports distinguish learner SPS from physical fighter SPS.

`benchmark_cpu_dummy.py` is the explicitly labeled CPU reference benchmark:
native CPU simulation and Sonic inference, the same scripted opponent, and
native CUDA Puffer learning. Its per-step downloads/uploads and worker IPC are
intentional baseline costs, not part of the production CUDA environment.
`profile_gpu_duel.py` separately measures an uninstrumented environment run and
an instrumented CUDA graph. Training phase intervals and graph component
intervals have different denominators; host and GPU durations overlap and must
not be added. Physics, controller, opponent decisions and metric accumulation
stay on CUDA in the production path. Host launches and report/checkpoint I/O
still use the CPU.

Completed Spark measurements on 2026-09-10, against the evaluator's fixed
dummy, illustrate the current performance limitation:

| Run | Arenas / learners | Learner steps | Learner SPS | Physical fighter SPS |
| --- | ---: | ---: | ---: | ---: |
| CPU simulation baseline | 4 | 4,096 | 473.03 | 946.06 |
| CUDA simulation, matched initial weights and rollout settings | 4 | 4,096 | 126.05 | 252.10 |
| CUDA simulation training smoke | 128 | 8,192 | 2,260.60 | 4,521.20 |

After the loop-boundary correction, matched 131,072-learner-step runs measured
1,926.12 learner SPS at 128 arenas and 3,152.39 at 512 arenas (68.05 s versus
41.58 s). Both used the same initial policy, horizon 64, minibatch 4,096,
scripted opponent, unmodified physics settings and original native CUDA
learner backend. Four times the arena count yielded 1.63665 times throughput.
The larger controller bundle passed exact CPU-ORT batch equivalence checks.
Per-arena simulated duration differs at a fixed total sample budget, so these
runs do not compare policy quality. Neither completed a round. CPU physics and
controller execution counts were zero; host submission still consumed about
92% of one CPU core. The rollout occupied 99.8% of the measured CUDA stream
envelope, with PPO updates accounting for 0.13 to 0.20%.

The 128-arena run covers only 1.28 simulated seconds per arena. It is a
throughput/training smoke test, with no completed rounds. A separate 50-tick
environment component profile attributed 71.30% of its CUDA stream interval
to physics plus reset forward recomputation, 17.93% to combat measurement and
referee work, and 1.82% to controller inference. These are instrumented stream
intervals, not hardware kernel-utilization percentages or an end-to-end PPO
breakdown. The runtime still simulates both articulated bodies at 500 Hz;
placing this workload on CUDA does not make it a lightweight semantic simulator.

A separate exclusive Nsight Systems capture of ten 128-arena control ticks
recorded 104,522 kernels and 4,190 device-to-device copies. There were no
host-to-device or device-to-host copies in the measured range. Kernels were
active during 84.31% of the 559.31 ms range; any GPU work was active during
86.80%. This measures activity, not SM occupancy or arithmetic utilization.
On the host, CUDA runtime calls covered 99.43% of the range on one thread:
30 graph launches took 444.96 ms and the final synchronization took 110.27 ms.
These host intervals overlap GPU work and cannot be added to it. The trace
localizes host cost to driver submission/backpressure and synchronization;
it does not separate driver spinning from other driver work. Instrumentation
changes timing, so these values are not an uninstrumented SPS benchmark.
A cleaned five-tick component smoke separately measured 31.11% physics step,
34.81% reset-state forward refresh, and 22.25% combat work in its instrumented
graph interval. The reset refresh currently runs even with an empty reset mask.
An experimental conditional skip was excluded because its mixed-reset
additional-error bound exceeded measured baseline repeat variation. No solver
tolerances or iteration limits were reduced. A whole-horizon capture prototype
was also excluded after CUDA rejected executable graph nesting. Both failed
experiments remain archived outside the working source tree.

The longer requested 1,572,864-learner-step run failed a native motion scheduler
check with status 311 after its last successful 393,216-step progress report.
Its roughly 1,916 learner SPS was an intermediate measurement, not a completed
training result. No final policy or completed-run report was produced for that
attempt. The short completed measurements above remain separate results.
An independently reproduced reverse-loop frame-boundary bug that can produce
311 has since been corrected and tested on CPU and CUDA. Its connection to the
original training incident is unconfirmed: a synchronized rerun passed 589,824
steps before being deliberately interrupted, preserving its periodic weights.

Training now saves policy-only checkpoints every 16 epochs by default after a
successful status check (`--checkpoint-every 0` disables this). These do not
contain optimizer or environment state. An exception produces `failure.json`,
individual state arrays and a diagnostic-only failed policy in the run
directory, where capture remains possible. `--check-every-step` adds explicit
synchronization for fault reproduction and invalidates throughput comparisons.

Fixed-opponent reports include action-category counts, native discrete-move
starts, completed-round wins and points, arena scored contacts, active-round
facing within 30 degrees, and horizontal root separation in metre bins. Facing
and range are sampled after each control step. Points include referee awards;
arena hit counts do not identify which player scored. Training statistics alone
do not establish an improvement over a frozen initial policy or a human.
Attributed-contact counts mean knockdown-attribution-qualified contacts;
scoring uses a separate filter, so scored hits can exceed attributed contacts.

The GPU loader performs the same per-clip heading normalization as
`g1_semantic_assets.c`. Active actuator controls use the native joint-limit
clamp without clipping filter history or retained fall/reset controls. The
contact adapter samples every 2 ms and preserves directed contact ordering.

GPU execution does not establish authentic REK parity. The candidate retains
the public-family controller identity and round-terminal episode limitation
described below. A successful kick or held-input probe is a regression result,
not evidence that all moves or trajectories match authentic REK.

### Shared state and action contract

One arena contains two physical G1 robots in one MuJoCo model. Puffer rows are
arena-major, player then opponent. Every 50 Hz outer action advances ten 2 ms
physics steps. Both fighters' controls are staged before the shared-contact
step, so neither row receives an ordering advantage.

The native observation ABI is schema 4 with 223 binary32 values per robot:

* 86 values for self state;
* 86 values for opponent state;
* 12 command, heading, route, and composer values;
* 39 score, fall, referee, round, and tick-event values.

The one-head categorical action ABI has 33 categories:

* continue the currently scheduled command;
* neutral and the validated W, S, A, D, Q, and E held combinations;
* all 17 build-pinned discrete routes in registry order 6, 7, 8, 9, 0 through
  5, then 10 through 16.

Held translation and yaw are represented every controller tick. Q or E can be
held with a translation input. Translation blocks a new discrete move until
the locomotion transition settles. An accepted move suppresses effective yaw while
the desired yaw hold and ramp state continue. Neutral, Q, and E remain
selectable during a move to update that desired state without restarting the
move. Retaining and advancing the ramp through a move is provisional candidate
semantics. It has not been recovered as current REK runtime behavior. No F
binding is present because the installed keyboard asset has no identified F
locomotion field.

## Implemented REK semantics

The runtime contains native motion composition, batched ONNX policy execution,
shared-arena MuJoCo contacts, build-pinned fall measurement, hit filtering,
strike attribution, scoring, referee counts, round timing, paired terminal
events, and arena-local episode reset. Its score-delta reward is an explicit
training contract, not a recovered REK reward.

Pinned static evidence and recovered code establish these fall and reset facts:

* the selected serialized G1 prefab has null prone and supine recovery clips;
* a committed fall suspends Sonic policy and motion progression while MuJoCo
  continues under the last joint targets with exact binary32 0.1 retention of
  the live proportional gain, damping gain, and effort limit;
* `ResetBothToSpawn` gives 2.0 s of fall-detection grace, applies the root spawn
  poses immediately, and completes joint and controller reset one 2 ms fixed
  update later;
* local `ResetAfterFall` instead gives 0.5 s of fall-detection grace.

The user-observed L100 behavior has no get-up action. This candidate selects
`CanGetUp=false`, validates that fact in every native observation, and uses the
3.0 s no-recovery referee branch. Repeated controlled capture is still required
before timing parity is accepted. Staged
actuator gains are inputs to the 0.1 retention rule. They are not represented
as measurements of the current service's live gain table.

## Completed-round metrics

The Puffer episode boundary is the recovered round-end event, not the end of a
Best-of-3 fight. The following user statistics are published once per completed
arena round from its side-0 row. Publishing both fighter rows would make every
decisive self-play round appear to have a pooled 50 percent win rate. `n` is
therefore the number of completed rounds, and each reported value is averaged
over those rounds:

* `side0_round_win_rate` and `side1_round_win_rate` use the terminal
  `round_winner_index`;
* `round_tie_rate` records terminal `round_result=3`, while
  `round_redo_result_rate` records `round_result=4` and `redo_round_rate`
  records that the completed round itself started as a redo. These are separate
  states and none is treated as a win or loss. The current state machine does
  not emit result 4, so that diagnostic remains zero for current-runtime runs;
* `side0_points_per_round`, `side1_points_per_round`,
  `side0_falls_per_round`, and `side1_falls_per_round` are the terminal round
  state;
* `scored_hits_per_round` counts arena contacts accepted by every scoring gate,
  and `attributed_contacts_per_round` counts the independent aggressor-strike
  attribution gate;
* `elapsed_seconds_per_round` and `semantic_steps_per_round` measure the 50 Hz
  Puffer episode through its terminal control step;
* `ko_round_rate` records the terminal `knockout_occurred` flag, including a
  double-KO tie.

Points are not hit counts. One accepted hand hit awards one point, one accepted
kick awards two, and referee outcomes can award three or five points without a
scored hit. The runtime exposes accepted-hit and attributed-contact counts only
for the complete arena, so the metrics do not invent per-fighter hit counts.
Likewise, no fight-win statistic is emitted: the current Puffer reset starts a
new round-one fight after every round terminal, before a multi-round fight
winner can be measured.

## Asset and authority boundary

`g1_semantic_assets.c` accepts only the generated semantic bundle whose model,
clip, idle-reference, size, and SHA-256 identities match its compiled manifest.
The ONNX encoder and decoder are supplied at process start and must pass the
compiled identity gate used for this candidate family. Proprietary game
binaries, extracted NPZ archives, and ONNX payloads are not linked into or
committed with the Puffer extension.

The exact encoder and decoder weights used by the current REK service are not
known. The executable candidate therefore remains classified as the validated
public GEAR-SONIC family unless a separately captured, hash-pinned current-build
identity proves otherwise. Shape compatibility alone is not identity.

## Evidence boundary

The Windows recorder is the authority for actual REK behavior. Spark is the
target for the stripped-down native environment and Puffer training. The
current held-input replay is diagnostic because its source fight lacks a
matched initial physical state, the opponent action timeline, action
acknowledgements, and a repeated-run variance envelope. It cannot accept or
reject parity.

Useful entry points are:

* `binding.c` for the Puffer extension and required process inputs;
* `semantic_duel_runtime.c` for the 50 Hz semantic and 500 Hz combat loop;
* `gear_sonic_native_duel.c` for batched policy and shared MuJoCo execution;
* `native_puffer_extension_smoke.py` for deterministic ABI and runtime checks;
* `human_eval_server.py` and `run_human_eval_spark.sh` for isolated human
  control and an append-only applied-action trace;
* `held_trace_candidate_replay.py` for the measured Windows trace replay;
* `POLICY_ADAPTER_DESIGN.md` and `NATIVE_PUFFER_BINDING.md` for detailed
  contracts and remaining gates.

## Human evaluation and acceptance

The human evaluator binds only to `127.0.0.1`. Browser key events control row 0
inside the Spark process and never become operating-system keyboard, mouse, or
gamepad events. Row 1 is an explicitly labeled deterministic candidate dummy.
It is not a reconstruction of REK Bot 1. Rows 2 through 7 remain neutral so the
native eight-row batch contract is satisfied. The evaluator requires the exact
native action-mask pointer. It will not replace a hidden legality fact with a
local guess.

W, S, A, D, Q, and E are held inputs. All 17 discrete routes are selectable by
direct evaluator shortcuts. U and I retain evaluator convenience aliases for
the user-confirmed left-front and right-side moves. The static build bindings
are displayed separately and are not claimed as measured input timing.
An attack pressed during another attack or held translation is discarded.
Only an attack interrupting held Q/E can remain pending while yaw settles.
That pending slot cannot be replaced or followed by queued attacks. These input
rules implement the user's observed contract; they do not establish REK parity.
Every accepted browser input and every applied 50 Hz action is written to a
create-new JSONL trace. Each control-step record includes the exact row actions,
action-selection reasons, terminal bits, and the first arena's 223-value
binary32 observation in little-endian base64. Runtime binaries and assets are
accepted only at caller-supplied SHA-256 identities.

The evaluator is a measurement instrument, not an acceptance result. The gold
standard is a paired human evaluation in which authentic REK and this isolated
environment start from matched state and receive the same tick-indexed held and
discrete input trace. Each individual movement and move must be compared for
input latency, duration, root trajectory, orientation, fall and recovery state,
contact and hit events, and round events. Acceptance requires held-out errors no
larger than authentic REK's own repeated-run variance. A visual resemblance,
successful training run, or single 50 Hz trace cannot satisfy that gate.

`run_human_eval_spark.sh` requires explicit paths and expected hashes through
its `REK_G1_*` environment variables, plus a create-new
`REK_G1_HUMAN_EVAL_TRACE` path. `REK_G1_HUMAN_EVAL_PYTHON`,
`REK_G1_HUMAN_EVAL_PORT`, and `REK_G1_HUMAN_EVAL_PHYSICS_WORKERS` are optional.
