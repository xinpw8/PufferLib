# Native GPU Newton solver fixture

The 2026-09-14 Spark run in `validation/solver-smoke-20260914/` passed.
The executable loads the existing MuJoCo-Warp CUDA modules directly through
the CUDA Driver API. There is no Python interpreter, Torch, Warp runtime
library, or CPU physics evolution in this test.

The real REK model supplies 70 velocity DOFs and a GPU-computed mass matrix.
Four independent worlds contain the model's 58 joint-friction constraints.
The fixture supplies nonzero warm-start accelerations with zero velocities
and zero external generalized forces. Its stationary solution is zero
acceleration. The largest final acceleration was `1.4156103134155273e-7`.

The initialization contains 24 operations; the iteration contains 20.
The original Newton stopping condition drives a CUDA conditional WHILE
graph node. Eight graph replays all completed in one Newton iteration for
each world. The persistent unfinished-world counter is reset by a device
copy on every replay. There are no host condition reads during a solve.

This fixture verifies native kernel ABI binding, tiled launches, dynamic
shared memory, the sparse mass-matrix solve, friction constraints, line
search and graph reuse. It does not test collision detection, a complete
physics step, gameplay parity, policy training, or training SPS.

`solver_schedule.cpp` follows the installed MuJoCo-Warp `solver.py` routines
`init_context`, `_linesearch`, `_update_constraint`, `_update_gradient`,
`_solver_iteration`, and `_solve`. The cached implementation supports this
model's sparse Newton solver, elliptic contacts, 80 padded DOFs and 50
iterative line-search iterations. Other configurations are rejected before
launch. The inner line search and outer Newton solve keep their original
convergence checks.

To reproduce from the staged native source, run `solver_smoke.sh` with a new
output directory, the existing private REK XML and the verified kernel
catalog. Exact commands, source/model/catalog/executable hashes and runtime
dependencies are retained in the validation directory. Model assets and
compiled cached kernels remain outside the repository.
