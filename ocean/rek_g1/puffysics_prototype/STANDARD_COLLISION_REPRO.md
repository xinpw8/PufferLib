# Portable box/cylinder collision reproducer

`reproduce_standard_collision.cu` reproduces one failed collision query from
the standard `b3_step` training run. It needs only the bundled engine headers
and a C++ compiler. CUDA is optional. It contains the two captured primitive
operands directly, with no dependency on robot models, motion assets,
controller weights, MuJoCo, PufferLib or a training checkpoint.

Both CPU and NVIDIA GB10 CUDA execution reproduce the captured result:

```json
{"status":16,"count":0,"gjk_iterations":6,"epa_iterations":0,"separation":null,"normal":[0,1,0]}
```

Status 16 is this fork's `B3_COLLISION_DEGENERATE`. No contact was returned.
The normal is the initialized diagnostic value, not a valid contact normal.
The capture came from arena 140, original shape IDs 14 and 40, type IDs 2
and 3, in `puffysics-standard-h16-r1/report.json`. Those IDs are not needed
to run the reproducer. The source preserves the exact float32 body and shape
transforms without quaternion renormalization.

## Build and run

Run from `ocean/rek_g1/puffysics_prototype`:

```sh
g++ -std=c++17 -O2 -x c++ reproduce_standard_collision.cu -o collision_cpu
./collision_cpu --expect-status 16

nvcc -std=c++17 -O2 -arch=sm_121 reproduce_standard_collision.cu -o collision_cuda
./collision_cuda --expect-status 16
```

Use a CUDA architecture supported by the target GPU when testing elsewhere.
Both commands returned exit 0 on Spark, meaning the expected failure was
reproduced. Without `--expect-status`, a nonzero collision status returns
exit 1. A CUDA/runtime or argument error returns exit 2.

After changing the collision implementation in a separate engine checkout,
rebuild against those headers and run with `--expect-status 0`. That checks
whether this input completes without a reported collision failure. It does
not independently establish contact accuracy or broader engine stability.

Optional CPU trace:

```sh
g++ -std=c++17 -O2 -DB3_CONVEX_TRACE -x c++ \
  reproduce_standard_collision.cu -o collision_trace
./collision_trace --expect-status 16
```

The captured trace reaches GJK iteration 5 with a four-vertex simplex,
distance `7.20843309e-06` and gap `0.854764879`. The implementation then takes
the `count == 4` degeneracy branch in `b3_convex.cuh`, before any EPA iteration.
This localizes the observed failure; it is not an implemented fix.

## Provenance and attribution

The bundled engine derives from public Puffysics commit
`4f6653cc52da92c3bb4972c6f00b6c733f5c2dc9`:

[Pinned upstream source](https://github.com/michaelthompsonx-lab/puffysics/tree/4f6653cc52da92c3bb4972c6f00b6c733f5c2dc9)

The local engine import and experiment were committed in this repository as
`2da294a00532993e2baa1b9eb3f11ca7a69f995c`. `UPSTREAM.md` lists the local
changes. Most relevant here, finite-cylinder support and the entire
`b3_convex.cuh` GJK/EPA path were added locally. The inspected pinned public
engine supports spheres, capsules and boxes and has no `B3_CYLINDER` path.

Therefore this reproduces a failure in the local finite-cylinder extension,
reached through the standard stepping entrypoint. It does not demonstrate a
bug in unmodified public Puffysics or in a contributor's current branch.
The two-shape query is useful to anyone implementing or reviewing cylinder
support, without requiring access to the original training environment.

The tested source SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `reproduce_standard_collision.cu` | `47db6a5a2b5d2813714a0f57a552392b3cdcd40cb26aecbe4cfdef73ede4ab75` |
| `engine/puffysics.cuh` | `c20f1a22454a75d76fa22186fd7f14e852cef292c1316f317bd2a1fb1eb898ea` |
| `engine/b3_convex.cuh` | `7dce3e2e880a64dce3db0b1ab50e49f86de114f117209e75f5e8346f2b1f871a` |

The reproducer disables articulated contact response at compile time because
it calls only `b3_collide_pair`. The identical failure on CPU and CUDA shows
that this query does not require the PPO learner, actuator forces, state
export, articulated response or GPU execution to reproduce.

## Independent-contact training configuration

Pinned upstream defaults `B3_ART_CONTACTS` to 1 and explicitly supports
setting it to 0. With 0, the standard step uses independent contact impulses
instead of articulated contact response. The native wrapper now guards its
experimental `rp_art_step` path so this supported configuration can compile,
and rejects solver mode 1 when that path is absent. Default builds retain
their existing mode 0 and mode 1 paths.

```sh
nvcc -std=c++17 -O2 -arch=sm_121 --shared -Xcompiler=-fPIC -Xptxas=-v \
  -DB3_ART_CONTACTS=0 puffysics_semantic_native.cu \
  -o librek_puffysics_independent_v1.so
```

The library reports its actual setting through `rp_art_contacts_enabled()`.
The semantic facade records this setting; older libraries without that API
report it as unknown. This build has no fixed-pose cache and compiled the
physics kernel with 128 registers and a 28,304-byte stack, compared with
196 registers and a 186,736-byte stack in the earlier cache-enabled build.

Its SHA-256 is
`7e0d4bb0546445f8b91236aaba9fdd5391b52c7f46683b73a0b0155a9d361911`.
Build stdout, stderr and exit status are saved under the Spark experiment's
`reports/independent-build-v1.*` files. A four-arena, twelve-step capsule
smoke passed, with no contacts yet in that 24 ms interval; its report is
`reports/independent-capsule-smoke-v1.json`. This is not a training result.

Replacing cylinders with enclosing capsules bypasses the local GJK/EPA path.
Keeping radius `r` and half-length `h` gives a capsule centre-segment half-
length `h` and total axial extent `2*(h+r)`, compared with `2*h` for the
cylinder. This intentionally changes geometry. The wrapper still contains
local model/force mapping, OR-style collision filtering and diagnostics, so
this configuration must not be described as an unmodified upstream engine.
