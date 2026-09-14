# Native GPU full-step validation

Actual host: `spark-4ae3`, NVIDIA GB10, 2026-09-14.

- `full-step-1-r1`: preserved pre-execution failure from an ambiguous cached solver entry prefix.
- `full-step-1-r2`: corrected entry, four worlds, one full step passed.
- `full-step-10-r1`: four worlds, ten full steps passed, including contacts.
- `full-step-1000-r1`: four worlds, 1,000 full steps passed, including convex contacts and Newton solving.
- `full-step-memcheck-10-r1`: four worlds, ten full steps under NVIDIA Compute Sanitizer 2025.3.1 memcheck; exit0 and `ERROR SUMMARY: 0 errors`.

Every probe used native CUDA kernels without Python or CPU physics stepping.
No contact, constraint, or CCD overflow was observed. This is correctness
validation, not a training benchmark or gameplay-parity claim.

Commands, host identity, hashes, timings, exit statuses, stdout and stderr are
retained here. Full qpos/qvel/time arrays were removed from these public result
copies; complete diagnostic output remains on the named Spark host under
`/home/spark-advantage/rek-training/native-mujoco-gpu-20260914/`.
No model, checkpoint, game binary, or private game-state dump is included.

See [probe design and findings](../../full_step_probe.md).
