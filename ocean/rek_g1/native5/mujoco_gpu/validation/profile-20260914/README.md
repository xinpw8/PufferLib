# Native headless training profile, 2026-09-14

Actual host: `spark-4ae3`, NVIDIA GB10. Native PufferLib 5.0 training used the
`build-native-v2` executable with the MuJoCo CUDA backend, scripted opponent,
512 agents, horizon 16, minibatch 8,192, and 65,536 requested transitions.
Training and profiling exited 0. No Python or CPU physics was run.

The SQLite analysis validated 112 learner action-sampling calls and 1,120
physics batch steps inside the measured interval. This is 10 physics substeps
per action batch. It observed 9,183 Newton iteration batch calls, averaging
8.199 per physics batch step. These counts include real conditional-loop
kernel execution, rather than graph-launch totals alone.

## Measured throughput and denominator

The first complete PPO epoch was excluded to remove graph capture and initial
setup. The remaining seven epochs contain 57,344 action transitions over
9.7495384 s of profile timeline: **5,881.7 profiled training SPS**. This interval
includes environment execution, policy inference, PPO, and between-epoch gaps.
The complete profiling command took 17.34 s including setup and trace handling.
The separate 1,048,576-transition unprofiled run is the throughput reference;
this shorter instrumented run does not measure tracing overhead independently.

Nsight Systems 2025.3.2 used `--cuda-graph-trace=node`. Raw traces were exported
with native `nsys export --type=sqlite`, then queried with SQLite and Node.
No `nsys stats` or Python report scripts were used.

| GPU kernel category | Summed duration | Share of summed kernel time |
| --- | ---: | ---: |
| Physics | 6,734.287 ms | 75.785% |
| Combat, motion, observations, contact measurement | 1,369.188 ms | 15.408% |
| Generic GPU memory clearing | 460.674 ms | 5.184% |
| Robot controller and servo | 298.420 ms | 3.358% |
| PPO training | 9.657 ms | 0.109% |
| Identified learner-inference kernels | 3.433 ms | 0.039% |
| Remaining kernels and ambiguous shared GEMMs | 10.337 ms | 0.116% |

Shares divide by summed kernel durations, not wall time. The interval union
of this process's GPU kernels covers 91.130% of the measured wall interval.
This is target-kernel timeline coverage, not an SM-utilization measurement.
Copies, synchronization, scheduling gaps, and CPU activity can occupy the
remainder. CUDA API durations overlap GPU execution and must not be added to
kernel durations as independent work.

## Main costs and engineering implication

`update_gradient_JTCJ_sparse` alone consumes 27.458% of summed kernel time.
Blocked Cholesky consumes 9.690%; sparse JTDAJ assembly 5.180%; the largest
contact radix-sort kernel 4.852%; iterative line search 4.049%. The full Newton
solver category consumes 58.658% of kernel time. These are the
measured optimization targets. PPO and policy-network changes cannot materially
improve end-to-end throughput at their present share.

SONIC GEMMs were attributed through GPU windows from `pack_encoder` through
`apply_actions(RobotState,...)` on the same stream. PPO windows use the three
actual `puf_stamp` GPU markers per epoch. Physics entries match the hashed
cached-kernel catalog. Generic clears are kept separate because their owning
buffer is not identified from the generic kernel name. Unknown shared GEMMs
remain explicitly unclassified.

The short profile completed no 20-second rounds and reports failure_bits 0.
It provides performance evidence, not policy-strength evidence. Other existing
GPU workloads were recorded and left running; this was not an exclusive-GPU
maximum-throughput benchmark.

## Files

`summary-final.json` contains complete aggregate categories, subcategories,
kernel counts, top kernels, API costs, transfer counts, and attribution evidence.
`commands.txt` and `analysis-final-command.txt` preserve the executed commands.
`analysis-final-hashes.txt` identifies the final analyzer and original private
trace files. Logs and native run configuration are included under `run/`.

The original `.nsys-rep`, SQLite export, checkpoints, models, and PTX stay on
Spark under
`/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/train65536-native-profile-r1/`.
No raw trace, policy weights, private model, or PTX is included here.
