# Engine provenance and modifications

Source: https://github.com/michaelthompsonx-lab/puffysics

Pinned source commit: `4f6653cc52da92c3bb4972c6f00b6c733f5c2dc9`

The original MIT license is preserved in `engine/LICENSE`.

Local experimental modifications:

1. Native finite-cylinder support, AABB and contact queries, plus explicit
   collision-capacity and convergence diagnostics. No proxy shapes replace
   the eight cylinders in the candidate model.
2. Optional MuJoCo OR-style collision bit filtering.
3. Joint-armature hook in the articulated inertia and contact response.
4. Three independently tested world-COM ABA corrections: remove the spurious
   linear gyroscopic force, restore the parent-angular-velocity contribution
   to the child COM acceleration bias, and propagate reduced articulated
   inertia times the acceleration bias in the backward pass.
5. Optional torsional-friction disabling for the source model's `condim=3`.
6. Skip exactly zero impulse solves.
7. Experimental fixed-pose factor cache in `b3_art_cached.cuh`, selected with
   `RP_USE_ART_CACHE`. Contact iteration counts and impulse ordering are retained.
8. Exact first-failing collision-query capture for independent CPU/CUDA replay.
   GJK simplex edge differences are formed after promotion to double, avoiding
   cancellation reproduced in an actual walking trace.
9. Retain certified separating support-plane bounds across GJK iterations.
   This resolves captured near-touch failures and exact-touch regression cases
   without increasing tolerances or iteration budgets. Penetrating cases still
   invoke EPA.

`b3_art.cuh` is the corrected uncached comparator. The cached header is an
explicit separate experimental copy. These files are not claimed to match
MuJoCo dynamics or authentic REK trajectories.
