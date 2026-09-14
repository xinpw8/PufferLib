# Native MuJoCo CUDA integration

The pinned REK arena's MuJoCo dynamics can run from a standalone C++ program.
Its state, collision detection, constraint assembly, Newton solve and implicitfast
integration stay on the GPU. Native PufferLib 5.0 supplies policy inference,
rollouts and PPO. Python, Torch and the Warp host runtime are absent from this
execution path. Native host code performs model compilation, allocation, graph
construction and launch, reporting and checkpoint I/O.

The integration loads existing Warp-generated CUDA/PTX from Spark. The offline
Node catalog utility reads source signatures and metadata without executing
Python. Kernel arguments, block dimensions and dynamic shared memory are bound
once. Conditional CUDA graph nodes control solver convergence without a CPU
condition read. This is specialized to the available REK model and cached kernels,
not a general replacement for every MuJoCo-Warp configuration.

## Executed validation

See [complete-step results](validation/full-step-20260914/README.md) and
[Newton solver results](solver_smoke.md).

On Spark GB10, four arenas completed 1,000 steps each, including actual convex
contacts, with finite state and no capacity overflow. A separate ten-step CUDA
memory check reported zero errors. These establish native execution for the
tested workload; they do not establish REK gameplay parity or policy strength.

The corrected native trainer completed **1,048,576 learner transitions** at
512 arenas and **6,020 mean training SPS**, with 946 completed rounds and zero
reported runtime failures. Physics, inference, rollout and PPO ran on GPU.
See [training evidence](validation/training-20260914/README.md) for complete
commands, timings and win/loss totals. Final-checkpoint strength and maximum
throughput have not been established.

The [native training profile](validation/profile-20260914/README.md) attributes
75.79% of summed GPU kernel time to physics, including 58.66% to the Newton
solver. PPO accounts for 0.11%. These are measured kernel-duration shares,
not estimates from the trainer's uninstrumented nested timers.

## Build and select

From the repository's native5 directory, `REK_NATIVE5_ENABLE_MUJOCO_GPU=1`
enables this backend when invoking `build_native.sh NEW_BUILD --build-runtime`.
The build compiles C++/CUDA and creates `NEW_BUILD/mujoco-conditional.ptx`.

Select `REK_PHYSICS_BACKEND=mujoco_cuda` and provide:

- `REK_MUJOCO_KERNEL_CATALOG`: catalog produced by `catalog_kernels.mjs`.
- `REK_MUJOCO_CONDITIONAL_PTX`: the build's native conditional helper.
- `REK_MUJOCO_CONDITIONAL_SHA256`: its SHA-256.

The headless runner accepts `REK_TRAINING_BACKEND=mujoco_cuda` and
`REK_TRAINING_OPPONENT=scripted`. Its default remains the existing Puffysics
frozen-opponent run. Neither training selection enables CPU evaluation.

Cached source/PTX, original assets, checkpoints and full private state logs stay
outside this repository. Reproduction requires the matching model, controller,
motion assets and CUDA kernel artifacts. No proprietary binaries are included.
