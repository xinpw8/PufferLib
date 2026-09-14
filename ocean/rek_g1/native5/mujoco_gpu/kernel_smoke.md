# Native loading of an existing MuJoCo-Warp CUDA kernel

Executed on `spark-4ae3`, NVIDIA GB10, on 2026-09-14. No Python interpreter,
Python imports, Torch runtime, Warp runtime library, or CPU physics step ran.
The host executable was compiled with `g++`; its runtime dependencies are
CUDA Driver, OpenSSL for hashing, and standard C/C++ libraries.

`native_module.{h,cpp}` loads the existing cached PTX bytes through
`cuModuleLoadDataEx`, resolves an entry point, and launches it asynchronously.
The caller owns the CUDA context, stream, argument storage, and GPU buffers.
Loading verifies an optional SHA-256. Launching allocates and copies nothing.
The shared `warp_abi.h` describes by-value arguments. The smoke build verifies
all field offsets and sizes against the installed Warp 1.12.0 C++ headers.

## Executed result

The actual cached `_next_velocity_d66c2f53_cuda_kernel_forward` entry performs
MuJoCo-Warp's velocity integration: velocity plus scaled acceleration times
timestep. The test uses synthetic input buffers and checks its analytical
result; it does not run a separate CPU physics engine.

- 512 worlds, 70 velocity coordinates each.
- 107,520 result values checked across direct launches and graph replay.
- Maximum absolute error against the analytical fixture: `1.4275428839027882e-8`.
- Broadcast and per-world timesteps, padded row strides, and unchanged input
  buffers verified. Hash mismatch and nonexistent entry points rejected.
- 1,000 CUDA-graph replays: `4.10028791 ms` by CUDA events in the final run.
- Exit code `0`.

This timing covers one cached integrator kernel. It is neither a full MuJoCo
step nor an environment rollout or training-SPS measurement. It establishes
that Python is unnecessary for loading and executing these existing kernels.
Correct model packing and the entire ordered forward/solver/integration
schedule still require separate implementation and validation.

The tested PTX SHA-256 is
`332d80eed8635021b2fa7f4651d3ea909f50d32e169549260c73e6e4db8d5d19`.
It remains in the private Spark cache under
`wp_mujoco_warp._src.forward_16fbac8/`. No generated kernel, model, or proprietary
asset is included here.

## Required launch detail

That cached module declares `WP_TILE_BLOCK_DIM=32` at source line 2. An initial
128-thread launch failed with an illegal device-memory access. The corrected
32-thread launch passed. Generated shared tile storage uses this compile-time
block dimension even for this nominally elementwise kernel. The full scheduler
must preserve each module's actual block dimension and shared-memory metadata.
The failed and passing stderr/exit records are retained.

## Reproduction

```sh
bash kernel_smoke.sh NEW_OUTPUT EXISTING_FORWARD_PTX WARP_NATIVE_HEADER_DIRECTORY \
  _next_velocity_d66c2f53_cuda_kernel_forward
```

This particular smoke expects the recorded 32-thread forward-module entry and
its six-argument signature. The loader itself supports arbitrary cached CUDA
entry points and does not infer their argument schema or launch geometry.
Commands, host identity, source/header/module hashes, ELF dependencies, output,
stderr, and wall timing are in `validation/kernel-smoke-20260914/smoke-final/`.
