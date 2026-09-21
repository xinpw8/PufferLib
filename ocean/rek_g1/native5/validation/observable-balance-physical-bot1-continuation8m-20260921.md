# Recovered G1 Bot 1: 8M physical continuation

The continuation completed 8,388,608 additional learner transitions and 32
rollout/training iterations on 2026-09-21. Native process and wrapper receipt
exit codes were zero. Final checkpoint:
`ef85a01b207d033417d8e1bc9d2b32b9eba610e784180dc7dd287a894acf2f4b`.
The completed training rounds recorded 599 wins, 378 losses and 47 draws out
of 1,024, with zero redos, unclassified outcomes, runtime failure bits or reward
saturations. These measurements describe a changing policy in the physical
simulator. Authentic frozen-checkpoint evaluation remains separate.

## Configuration and initialization

The input is the final checkpoint from the
[previous recovered-Bot1 run](observable-balance-physical-bot1-training-20260921.md):
`7c34eaa9f00c1ee97f8cbd8abf894d773d8a838547a10e496bcb5980de821c84`.
The copied input and native step-zero checkpoint are byte-identical to it.
Initialization loads weights only; optimizer, RNG, recurrent state and step
count restart. The critic is retained without preprocessing.

The recorded commands were compared mechanically. Only the run ID, input and
output paths, and transition budget differ from the previous Bot 1 run.
Environment, native executable, reused trainer object, source/build manifests,
reward, opponent, observations, seeds and other hyperparameters match.

| Setting | Value |
| --- | --- |
| Backend | `mujoco_cuda`; native SONIC inference; CPU physics disabled; no Python runtime |
| Observation / action shape | `rek.native5.observable_balance.v1`; 223 observations; 33 actions |
| Policy | Two recurrent layers, width 256 |
| Parallel simulation | 512 arenas, 1,024 fighters, one learner per arena |
| Transition budget | 8,388,608 new learner transitions |
| Horizon / minibatch / replay ratio | 512 / 8,192 / 1 |
| Rollout iterations / minibatch optimizer steps | 32 / 1,024 |
| Round duration / action rate | 120 s / 50 Hz, stride 1 |
| Exposure per arena | 16,384 learner steps, 327.68 simulated seconds |
| Base seed / environment seed | 419 / 419 |
| Initial learning rate | 0.015, cosine annealing, minimum ratio 0 |
| Entropy coefficient | 0.001, constant |
| Gamma / GAE lambda | 0.9998844821426083 / 0.9978673240629938 |
| PPO clip / value clip / value coefficient | 0.2 / 0.2 / 0.5 |
| Gradient norm limit / optimizer momentum | 0.5 / 0.95 |
| V-trace / self-play / reset each horizon | Disabled / disabled / disabled |
| Checkpoint interval | Every rollout iteration, including initial step zero |

Reward is `normalized_points_falls_v1`: fixed scale 0.01 applied to own awarded
point delta minus opponent point delta minus own confirmed `BECAME_FALLEN`
events. The own-fall term is therefore -0.01 per event. Bounds are [-1, 1],
terminal bonus is zero, and no saturation occurred.

The opponent remains `recovered_bot1_g1_v1`, with source-derived G1 tactics and
continuous native commands. Its candidate Update/dispatch/FixedUpdate cadence
is once per 50 Hz tick, with candidate-private xorshift32 RNG. Counted resets
preserve tactical, RNG and recovery state; round resets deactivate/reactivate
the controller while retaining RNG and recovery fields. Current G1 get-up
clips are null and motor-shutdown hold is unmodeled. The bot's zero categorical
inspection slot does not identify the directly issued native command.

Observation history uses the preceding 50 Hz observation and resets only at
explicit reset or episode boundaries. Root origin/quaternion binding is
verified. Shared joint pose/rate mapping remains unavailable. Authentic Unity
RNG, server cadence, complete controller behavior and physical trajectory
parity remain unverified; the runtime reports `authentic_parity=false`.

Doubling the new-transition budget also stretches cosine learning-rate decay
from 16 to 32 rollout updates. Together with changed starting weights and fresh
optimizer/RNG state, this prevents attributing differences solely to additional
sample count. This is one continuation with one seed.

## Training measurements with comparable denominators

The continuation's completed-round points total 16,179 : 13,496. Lifetime
awarded points total 21,970 : 17,677 and confirmed falls total 2,131 : 3,248.
Learner values precede opponent values in every pair.

Completed-round points exclude unfinished rounds. Lifetime counters cover all
executed runtime transitions, including warmup if present and incomplete
rounds. Their rate denominator below is the declared learner-transition
budget, rather than an independently measured runtime-step counter.

| Measurement | Previous Bot 1 run | Continuation |
| --- | ---: | ---: |
| Declared new learner transitions | 4,194,304 | 8,388,608 |
| Completed training rounds | 512 | 1,024 |
| Completed-round learner win fraction | 40.625% | 58.496% |
| Completed points per round, learner : opponent | 16.287 : 17.709 | 15.800 : 13.180 |
| Lifetime confirmed falls per million declared learner transitions | 451.088 : 364.780 | 254.035 : 387.192 |
| Lifetime awarded points per million declared learner transitions | 2,671.719 : 2,980.947 | 2,619.028 : 2,107.263 |
| Aggregate normalized learner reward per million declared learner transitions | -7.603 | 2.577 |

The training win fraction increased while learner falls and opponent points
decreased on the stated normalized denominators. Learner completed points per
round also decreased. These observations support evaluating the final policy;
they do not establish that a frozen policy has improved, that the effect will
replicate, or that authentic Sparring Bot 1 performance will follow. Unequal
raw totals are not treated as a strength comparison. The previous checkpoint's
r114..r117 authentic results remain a separate development cohort.

## Complete-run timing

| Timing or throughput | Previous Bot 1 run | Continuation |
| --- | ---: | ---: |
| Native trainer uptime | 714.890 s | 1,487.292 s |
| Whole-process elapsed time | 724.03 s | 1,496.52 s |
| Complete-training native SPS | 5,867.062 | 5,640.189 |
| Whole-process SPS | 5,792.998 | 5,605.410 |

The continuation's native display was `24m 47s 292ms`; process wall time was
`24:56.52`. Process CPU user/system times were 660.67 / 815.04 s. The 2,400 s
timeout was not reached. Native uptime includes all rollouts, PPO updates and
checkpoint writes, excluding initial construction and CUDA graph capture.
Whole-process time includes initialization and shutdown. Display resolutions
are 0.001 s and 0.01 s respectively. Each SPS value divides the full declared
transition budget by the corresponding full-run time. Dashboard windows and
the intermediate metrics section in the saved INI are not completion timing.

## Verification and private preservation

The collector verified 33 checkpoints at steps 0 through 8,388,608 in exact
262,144-step increments. Each is 1,836,032 bytes and contains finite FP32
weights. Checkpoint schema sidecars match their hashes, initialization,
observation shape, reward, learning rate, opponent and trainer identity. The
final checkpoint differs from the initial one. All 839 selected provenance,
source-tree, build-artifact and checkpoint-manifest records matched, with no
malformed manifest entries. Completed outcome counters reconcile to 1,024.

An independent CPU-only check read the archived command, saved configuration,
raw round and lifetime counters, exit codes and timing; recomputed the rates
above; and checked the summary's checkpoint ledger and provenance records.
All 127 selected archive files and both aggregate/inventory sidecars were
individually rehashed without extraction, with exact byte-size agreement.
All ten preserved local/NAS files and the receipt copy also matched. The NAS
directory contains those ten files plus the receipt.

Spark stage:
`/home/spark-advantage/rek-training/physical-bot1-integration-20260921-r1`.
Run directory: `train-bot1-continue8m-r1`.
Final checkpoint relative path:
`train-bot1-continue8m-r1/checkpoints/rek_native5/bot1-continue8m-r1/0000000008388608.bin`.
Exact environment/native arguments and resolved settings are retained in the
run's `command.txt` and `logs/rek_native5/bot1-continue8m-r1.ini`.

| Artifact | SHA256 |
| --- | --- |
| Executed runner | `49adff962e2473863f7eeb5cecf5b45bfaccc05c9cc97ddd8857bcd2264fcea4` |
| Native trainer executable | `61dea9fd819e24f132ace54eaeff7cfb4520a3d79fd6b791f4cb79cc010b8aa4` |
| Reused trainer object | `254363eac4668a25ff8e3cc1cd4c3226b4216ef2046a0a9bb1ce3651caa26959` |
| Exact command | `a365356af8910ce97876968872b6484cfd14d799c7dcac66b3daa3a567b0d695` |
| Saved configuration | `154b477ac68617a1d818f33c476da8fec6d0d9c174ba67ec389130fb77919404` |
| Final checkpoint | `ef85a01b207d033417d8e1bc9d2b32b9eba610e784180dc7dd287a894acf2f4b` |
| Training archive | `147c77343d2d470ea080aeecc74e252fda8bc98136a429c9d962ca5fdc95e05a` |
| Aggregate summary | `5788438007a46b650269aaaa479e317bab7a36667c968da0e22eaae3603ec11e` |
| NAS receipt | `6b6402e4384ef84c95d422adb21d8b4478a375c90793857769ffbc6b55722cc6` |

The 60,763,303-byte archive is retained in fresh private NAS directory:
`\\192.168.0.19\MyShare\pufferlib\rek-evidence\2026-09-21\physical-bot1-continuation8m-r1`.
Local collection:
`C:\rekagent\work\physical-bot1-integration-20260921-r1\continuation8m-tools\completed-r1`.
The receipt records matching source-before/source-after/copy hashes and no
overwrites. Selected source, binaries, checkpoints, sidecars and raw training
logs remain private. Proprietary game/model/motion assets are hash-referenced
and excluded. The separate integration probe remains in stage `run-r2`.
Collection, independent checking and preservation launched no physics,
trainer, inference or game process.

## Registered authentic evaluation

At `2026-09-21T18:37:18.8968009Z`, the final checkpoint was registered for a new
prospective cohort starting at r118: 20 completed 120 s rounds against private
Sparring Bot 1, difficulty 0, with at least 18 wins required. Sampled selection,
worker seed 73, no feature mask and the observable-balance schema stay fixed.
Every valid completed win, loss or draw counts. The third nonwin makes 18/20
impossible and stops the cohort; reaching 18 wins early does not stop it.
Any incomplete attempt, evidence-validation failure, configuration mismatch,
opponent mismatch or duration mismatch stops further launches and preserves
the evidence. No retuning, replacement rounds or pooling with prior checkpoints
is permitted within this cohort.

Plan directory:
`C:\rekagent\work\observable-balance-live-eval-20260921-r1\cohort-continue8m-r1`.
Frozen candidate configuration SHA256:
`9e8a8f126ff21af1cd54498274aa8845684e3296991be525280ab12b22b81880`.
Cohort runner SHA256:
`871a91a3506034a2a5ae9b6df26861a77d70c32a19c5996adb076ae5cbe325b5`.

Authentic results and closed-round evidence belong in the
[evaluation report](observable-balance-authentic-evaluation-20260921.md).
The training completion reported here does not assert that the authentic
criterion has been met.
