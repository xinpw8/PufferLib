# Puffysics backend experiment

Status: experimental, not a training environment and not accepted REK parity.
Production `gpu_duel_physics.py` and the human evaluator are unchanged.

The final v8 motion probes fail: walking produces nonfinite states near
1.94 s; the left-front kick stops on an EPA initialization failure near
2.47 s. These are unresolved defects. Do not adopt this backend for training.

This prototype imports the current G1 candidate's compiled MuJoCo model into
Puffysics and retains the same existing Sonic encoder/decoder, motion clips,
observation preparation, action history, and 250 Hz actuator-target filter.
The controller runs at 50 Hz. Physics runs at 500 Hz on CUDA, with one thread
per arena and one native kernel launch per 2 ms physics step. Each arena has
two robots. Model parsing, initialization and evidence serialization are host
operations; the rollout does not call a CPU physics step or CPU inference.

The compiled adapter preserves all 60 massive bodies, 91 primitive shapes,
58 hinges and two floating roots. It maps displaced/rotated inertial frames,
joint axes, limits, armature, PD force limits, native position targets, root
orientation and angular-velocity conventions. Private model exports and motion
or controller binaries are deliberately excluded from this directory.

## Measured results

Executed on `spark-4ae3`, NVIDIA GB10, with CUDA 13.0 targeting `sm_121`.

- Two-second idle rollout completed with eight robots, no nonfinite states,
  no collision/solver failure bits, and no sampled fall-proxy crossings.
- Maximum root-position error versus the current MuJoCo candidate was 0.111 m;
  RMS error was 0.0774 m. This is not a parity pass.
- Fixed-pose mass-factor caching reduced that rollout from 120.508 s to
  53.765 s. All recorded qpos, qvel, controller actions, targets and simulation
  times were bitwise identical between corrected uncached and cached CUDA runs.
- Final v8 repeated 512-arena, 32-control-tick CUDA-graph measurements give 870
  arena SPS for this prototype versus 10,062 for the CUDA MuJoCo candidate. See
  `EXPERIMENT.md`. SPS counts arena control steps, not
  individual robot decisions or physics substeps. Policy training, reward
  computation and semantic combat are absent from this benchmark.

The reference here is the existing MuJoCo candidate. Authentic REK recordings
remain necessary for the project's actual acceptance criterion.

## Remaining physics differences

- MuJoCo `implicitfast` integration and contact `solref`/`solimp` are not reproduced.
- Predictive hard hinge stops differ from MuJoCo soft limits.
- Moving Coulomb joint friction is included; static joint friction is absent.
- Cylinder pairs use a single contact point instead of a MuJoCo manifold.
- Full MuJoCo parent/weld collision-filter equivalence remains unverified.
- The reference-motion probe does not implement combat scheduling, hit/KO
  scoring, opponent strategy, rewards or policy optimization.

## Build and run on Spark

Run these from this directory on the CUDA host. Set `PYTHONPATH` to the parent
`ocean/rek_g1` directory and the installed runtime dependencies. Reuse the
existing private GPU-duel configuration containing `model`, `model_sha256`,
`assets`, `assets_sha256`, `controller_manifest` and `controller_source`.
Export the pinned model with `export_rek_compiled_model.py`; its CLI requires
the model path, candidate source directory, private motion-asset root and
asset-manifest digest. `ADAPTER_SCHEMA.md` describes the generated data.

```sh
nvcc -std=c++17 -O2 -arch=sm_121 --shared -Xcompiler=-fPIC \
  -DRP_USE_ART_CACHE=1 puffysics_native.cu -o librek_puffysics.so

python run_rek_puffysics.py --config /path/to/private-gpu-duel-config.json \
  --backend puffysics --library ./librek_puffysics.so \
  --model-export /path/to/private-model-export.json --solver-mode 1 \
  --role idle --steps 100 --warmup 3 --out /path/to/new-run-directory

python run_rek_puffysics.py --config /path/to/private-gpu-duel-config.json \
  --backend mujoco --role idle --steps 100 --warmup 3 \
  --out /path/to/new-reference-directory
```

Omitting `--eager` enables CUDA graph replay. Solver mode 0 is diagnostic only
and omits armature in free integration. Only one native handle may exist per
process because model parameters are stored in CUDA constant memory.

`compare_traces.py` produces numerical comparisons. `render_comparison.py`
reconstructs drawable geometry from recorded qpos, without advancing either
simulation. `--format gif` avoids an external video encoder. Wall hiding and
fighter colors are render-only changes, never changes to the source model.

## Regression tests

CPU analytic fixtures validate world-COM dynamics, joint armature, contact
response, cylinder geometry and the fixed-pose cache. These are small correctness
tests, not CPU training. The exporter and C-ABI packing tests also run on CPU;
packing requires an explicit private export fixture. CUDA rollout evidence is
required in addition to these tests.

The engine is vendored under its original MIT license. See `UPSTREAM.md` for
the pinned source and local modifications.
