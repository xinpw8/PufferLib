# Native MuJoCo GPU full-step probe

`full_step_probe.cpp` exercises the actual cached MuJoCo-Warp CUDA kernels
through the native CUDA Driver API. It runs the complete position/contact,
constraint, velocity/force, sparse mass factorization, Newton solve, and
implicitfast integration sequence. Newton convergence executes in a GPU
conditional WHILE graph. The executable does not run Python or CPU physics.
The MuJoCo shared library parses the XML and supplies model constants.

The probe defaults to four worlds, zero controls, a 0.002 s timestep, 512 contact
slots per world, and 512 constraint rows per world. It checks every diagnostic
step and reports milestones at 1, 10, and 1,000 steps. The optional final
argument selects a bounded 1-, 10-, or 1,000-step test. Unlike a training hot
loop, this diagnostic intentionally reads state and capacity counters after
every step so an earlier overflow cannot be hidden by later counter resets.

```sh
./full_step_probe.sh NEW_OUTPUT REK_MODEL_XML KERNEL_CATALOG_JSON 1000
```

When sources are staged outside the checkout, set `REK_NATIVE5_ROOT` to the
directory containing `vendor/cJSON.c`. The script records build/run commands,
stdout, stderr, exit status, host identity, source and input hashes, dynamic
library dependencies, process timing, and complete final qpos/qvel/time arrays.
It preserves inputs and does not copy game binaries into the repository.

## Executed result, 2026-09-14

On `spark-4ae3`, NVIDIA GB10, `full-step-1000-r1` passed:

| Measurement | Observed |
| --- | ---: |
| Worlds and steps | 4 worlds, 1,000 steps each |
| Simulated duration | 1.9999814 s |
| Peak broadphase pairs, all worlds | 118 |
| Peak contacts, all worlds | 112 |
| Peak constraint rows per world | 169 |
| Peak sparse constraint nonzeros per world | 1,106 |
| Peak CCD pairs per geometry-pair type | 32 |
| Maximum quaternion norm error | 2.78699689e-7 |
| Maximum clock error | 1.86916441e-5 s |
| Accumulated GPU event duration | 857.203712 ms |
| Diagnostic wall time | 1.06154833 s |

Every sampled state was finite. Contact/constraint/CCD capacities stayed within
bounds. Explicit export refresh preserved qpos, qvel, and simulation time.
The zero-control robots fell and settled, exercising both primitive and convex
contacts. The 1- and 10-step probes passed first. A preceding attempt failed
before any physics step because a solver kernel prefix was ambiguous; that
failure was preserved and the exact cached entry was selected before rerunning.

These results establish a functioning native full-step execution path. They do
not establish a trained fighting policy, gameplay parity, a CPU-reference
comparison, or training throughput. `training_sps` is deliberately `null`.
Training throughput requires the controller, action/observation pipeline,
rollout storage, policy inference, and optimizer to be measured together.

The separate bounded `full_step_memcheck.sh` run executed four worlds through
ten full steps under NVIDIA Compute Sanitizer 2025.3.1. It exited0 and reported
`ERROR SUMMARY: 0 errors`. Logs are in `full-step-memcheck-10-r1`.
