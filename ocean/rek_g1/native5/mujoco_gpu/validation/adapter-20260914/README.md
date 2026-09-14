# Native MuJoCo CUDA adapter probe

Passed on `spark-4ae3` with the feature-enabled `build-native-v2`.

The probe constructs the real native5 `Physics` adapter from the pinned XML
and existing prepared actuator-parameter export. It executes ten 0.002 s
MuJoCo GPU substeps across four arenas. The XML's original geometry is used;
the export filename does not select its Puffysics capsule geometry here.

Linker wrappers abort if the adapter calls `mj_step`, `mj_forward`, or
`mj_kinematics` on the CPU. All three call counts remained zero, including
startup. The executable has no Python, Torch, or Warp runtime dependency.
CPU model parsing, native host launch orchestration, and diagnostic readbacks
still exist. The process CPU-time fields are not measurements of CPU physics.

The selected reset restored arena 0 while the other three arenas' qpos,
qvel, and clocks remained bit-identical. An empty reset mask preserved all
integration state. The GPU mask-any conditional skips reset refresh work
when no arena is selected. Reset refresh regenerates contacts from the current
pose and preserves persistent failure flags.

This 0.020 s adapter fixture had no contacts. Active contact execution is
covered separately by the native full-step 1,000-step diagnostic. This is
neither a training-SPS measurement nor evidence of winning, long-run control
stability, or REK gameplay parity.

`commands.sh` records the invocation. The reusable harness is
[`physics_mujoco_gpu_probe.sh`](../../../physics_mujoco_gpu_probe.sh).
Raw compiler output, object hashes, dependency records, timing, stdout and
stderr remain at the host path in `provenance.json`. This folder contains
sanitized evidence only, without model contents, full states, or binaries.
