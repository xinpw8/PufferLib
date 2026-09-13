# Experimental full semantic training adapter

The adapter supplies the complete existing G1 semantic duel with Puffysics
physics. It retains the same Sonic controller, motion scheduler, 223-float
observations, 33-category action head, contact-based combat measurement,
referee/reset logic, opponent wrapper and native PufferLib learner. The paired
training harness owns policy inference, rollouts and PPO optimization.

The production Python files and engine headers are unchanged. A compile guard
in the native prototype wrapper permits the supported `B3_ART_CONTACTS=0`
configuration and rejects mode 1 when articulated code is absent. Default
builds retain the existing stepping paths.
`puffysics_semantic_native.cu` includes `puffysics_native.cu` and adds state
export and selected-world generalized-state reconstruction. The factory
temporarily supplies this physics class while constructing `GpuSemanticDuel`,
then restores its original factory.

See [STANDARD_COLLISION_REPRO.md](STANDARD_COLLISION_REPRO.md) for the exact
upstream/fork distinction and the independent-contact build. The compiled
library exposes its contact setting through `rp_art_contacts_enabled()`;
old libraries without that export report the setting as unknown. Selecting
mode 0 alone does not disable articulated contact response.

## Source and build

The reproducible source stage starts at repository commit
`73c918b6`, with the four new adapter/verification files added separately:

- `puffysics_semantic_native.cu`
- `puffysics_semantic_physics.py`
- `puffysics_training_duel.py`
- `verify_puffysics_semantic.py`

The CUDA translation unit includes the existing v8 prototype solver and its
pinned engine headers. Source and library digests should accompany benchmark
reports because new adapter files are additional to that committed baseline.

Build on Spark, from this directory:

```sh
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_121 --shared \
  -Xcompiler=-fPIC -Xptxas=-v -DRP_USE_ART_CACHE=1 \
  puffysics_semantic_native.cu -o librek_puffysics_semantic_v1.so
```

The original `rp_step_kernel` compiled with 196 registers and a 186,736-byte
stack per thread, matching the original v8 compilation. Added export kernels
use 16 to 40 registers. The selected-state reconstruction kernel uses 128
registers and a 28,320-byte stack. Compilation and CPU model mapping happen
before training timing.

Factory signature:

```python
create_training_duel(config, *, library, export_path, solver_mode=1)
```

Both native solver routes are available. Pass `solver_mode=0` to run standard
Puffysics `b3_step`; `solver_mode=1` runs the experimental articulated-body
`rp_art_step` route used by the initial adapter experiment. The compiled CUDA
library already contains both paths. Solver selection changes no CUDA code.
Reports identify the actual step function and solver mode. MuJoCo parity is
not a requirement for these engine diagnostics. Mode 0 does not include rotor
armature in free integration; this limitation is reported without blocking
training.

Import it from `puffysics_training_duel`. Add this directory and its parent
`ocean/rek_g1` to `PYTHONPATH`, along with the existing native learner and
controller dependencies. Use `conditional_reset_forward=False` in both paired
configurations because that optimization depends on MuJoCo Warp internals.
Existing fused combat measurement and deferred observation packing are
supported and can remain enabled in both arms.

## What is measured and preserved

All body and geometry transforms derive from current native B3 bodies and
the compiled source model's fixed frame maps. Spatial velocities use the
source model's root-subtree centre of mass as their origin. This preserves
the existing observation and hit-velocity reader conventions.

Contacts are the native solver's actual manifold points. The export records
their shape IDs, arena IDs, signed separation, position and orthonormal frame.
Points are packed in deterministic arena, shape-pair and point order. Contact
positions correspond to collision detection before the 2 ms integration;
body transforms and velocities correspond to the end of that step. No contact
force, collision event, score or reward is invented by the adapter.

Selected-world forward reconstructs native rigid poses and velocities from
the existing generalized state using hinge frames and anchors. It clears
solver warm starts in selected worlds and preserves cumulative failure
counters. Unselected native worlds remain untouched. The original reset
module still controls the order of root resets, joint resets and retained
free-root velocities.

The exposed empty reset-work tensors represent absent MuJoCo-specific
scratch arrays. They are cleared by the unchanged reset module and are never
used as measurements. Actual native solver warm starts are cleared inside
selected-world reconstruction.

## Verification on NVIDIA GB10

The verification report is:

`/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/reports/semantic-adapter-verification-v1.json`

The tested library is:

`/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/source/ocean/rek_g1/puffysics_prototype/librek_puffysics_semantic_v1.so`

The report passed on four arenas with zero CPU physics steps:

| Check | Result |
| --- | ---: |
| Twelve physics steps against original v8, identical controls | Bitwise equal |
| Maximum original-v8 state error | 0 |
| All 25 published tensors after zero-mask forward | Bitwise unchanged |
| Unselected qpos/qvel and all clocks after selected reset | Bitwise unchanged |
| Generalized qpos reset round-trip maximum error | 2.39e-7 |
| Generalized qvel reset round-trip maximum error | 1.31e-8 |
| Body/geometry matrix maximum error against CPU MuJoCo kinematics | 6.46e-6 |
| Spatial velocity maximum error against CPU MuJoCo kinematics | 1.80e-5 |
| Full idle reset maximum qpos error | 1.94e-7 |
| Counted-reset free velocity error | 0 |
| Counted-reset joint qpos maximum magnitude | 1.79e-7 |
| Counted-reset joint qvel maximum magnitude | 0 |
| Actual exported manifold points after reset fixture | 12 |
| Nonfinite, solver failure or contact-capacity failures | 0 |

The numerical reset/kinematics acceptance tolerance was 2e-5. The CPU
comparison calls only `mj_kinematics`, `mj_comPos` and `mj_comVel`, with no
CPU physics integration. The unchanged observation assembler, reset module,
original combat measurement and fused combat measurement all constructed
and sampled successfully. This is adapter verification, not a training
throughput result or a policy-quality result.

The standard `b3_step` mode 0 bridge also passed the same four-arena
verification. Its report is
`/home/spark-advantage/rek-training/paired-physics-training-20260913T0423Z/reports/semantic-adapter-verification-standard-mode0-v1.json`.
Twelve steps matched the original v8 mode 0 library bitwise. All 25 checked
tensors remained unchanged for a zero-mask forward, selected resets retained
unselected state, and standard/fused combat measurements sampled successfully.
No nonfinite state, solver failure or contact-capacity failure occurred.
The kinematic mapping and reset checks verify this bridge's state conventions;
they do not require either engine to follow MuJoCo trajectories.

## Limits

- Both standard solver mode 0 and experimental articulated solver mode 1 are
  supported. Use `verify_puffysics_semantic.py --solver-mode 0` to verify the
  standard-solver bridge against the original v8 library running mode 0.
- MuJoCo implicitfast integration and contact solref/solimp softness are not
  reproduced by the native engine.
- Mode 1 predictive hard joint stops differ from MuJoCo soft limits. Mode 0
  retains the standard Puffysics joint-constraint solver and omits rotor
  armature from free integration.
- Moving Coulomb joint friction is implemented; static joint friction is not.
- Native cylinder contacts use a single contact point rather than a MuJoCo
  manifold. Complete MuJoCo collision-filter equivalence remains unverified.
- The bridge exports all native manifold points into a capacity of 128 points
  per arena in aggregate. A count beyond that capacity causes explicit
  invalidation. Native manifold-capacity or collision-solver failures also
  invalidate the run and are not cleared by selected resets.
- Semantic state export adds four launches around each original physics
  kernel. Selected forward adds four launches, including reconstruction.
  A benchmark must include this necessary adapter work in environment time.
- Agreement with another candidate engine does not establish authentic REK
  parity. Training throughput and policy quality require separate evidence.
