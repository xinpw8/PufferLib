"""Opt-in CUDA capacity diagnostics for the pinned MuJoCo-Warp backend.

The installed backend prints overflow warnings but exposes no sticky flag.
Its sparse nonzero allocator is local to make_constraint. A source-checked
diagnostic copy appends one observer call before that local counter expires.
Original constraint kernels, their order, and all physics settings are retained.
"""

import ast
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import textwrap

import torch
import warp as wp


CONSTRAINT_SOURCE_SHA256 = "1f90b0620633cdedf6bf2997b7fad412d956ee996a30d72688a10fed539cf68d"


@wp.kernel
def _capacity_high_water(
    nefc: wp.array[int],
    nacon: wp.array[int],
    ncollision: wp.array[int],
    efc_nnz: wp.array[int],
    sparse: bool,
    njmax: int,
    naconmax: int,
    njmax_nnz: int,
    counters: wp.array[int],
):
    world = wp.tid()
    constraints = nefc[world]
    wp.atomic_max(counters, 0, constraints)
    if constraints > njmax:
        wp.atomic_or(counters, 5, 1)
    if sparse:
        nonzeros = efc_nnz[world]
        wp.atomic_max(counters, 3, nonzeros)
        if nonzeros > njmax_nnz:
            wp.atomic_or(counters, 5, 2)
    if world == 0:
        wp.atomic_max(counters, 1, nacon[0])
        wp.atomic_max(counters, 2, ncollision[0])
        wp.atomic_add(counters, 4, 1)
        if nacon[0] > naconmax:
            wp.atomic_or(counters, 5, 4)
        if ncollision[0] > naconmax:
            wp.atomic_or(counters, 5, 8)


class GpuPhysicsCapacityMonitor:
    """Diagnostic-only high-water counters; no host reads until snapshot()."""

    def __init__(self, physics):
        from mujoco_warp._src import constraint

        self.physics = physics
        self.constraint_module = constraint
        source_path = Path(constraint.__file__)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_hash != CONSTRAINT_SOURCE_SHA256:
            raise RuntimeError(f"capacity diagnostic requires inspected constraint source: {source_hash}")
        self.original = constraint.make_constraint
        unwrapped = inspect.unwrap(self.original)
        source = textwrap.dedent(inspect.getsource(unwrapped))
        tree = ast.parse(source)
        function = tree.body[0]
        if not isinstance(function, ast.FunctionDef) or function.name != "make_constraint":
            raise RuntimeError("unexpected constraint function source")
        if any(isinstance(node, ast.Return) for node in ast.walk(function)):
            raise RuntimeError("constraint function gained early returns")
        observer = ast.parse("_rek_capacity_observe(m, d, efc_nnz)").body[0]
        function.body.append(observer)
        ast.fix_missing_locations(tree)
        namespace = dict(unwrapped.__globals__)
        namespace["_rek_capacity_observe"] = self.record_constraints
        exec(compile(tree, str(source_path) + "[capacity-observer]", "exec"), namespace)
        self.instrumented = namespace["make_constraint"]
        self.counters = torch.zeros(6, device=physics.qpos.device, dtype=torch.int32)
        self.counter_array = wp.from_torch(self.counters, dtype=wp.int32)

    @contextmanager
    def capture_hook(self):
        """Scope the copied function to eager setup or graph construction.

        The captured graph retains the observer kernels after the Python module
        is restored. This context is not a GPU timing measurement.
        """
        if self.constraint_module.make_constraint is not self.original:
            raise RuntimeError("another constraint diagnostic hook is active")
        self.constraint_module.make_constraint = self.instrumented
        try:
            yield self
        finally:
            self.constraint_module.make_constraint = self.original

    def record_constraints(self, model, data, efc_nnz):
        if data is not self.physics.data:
            return
        wp.launch(
            _capacity_high_water, dim=data.nworld,
            inputs=[data.nefc, data.nacon, data.ncollision, efc_nnz,
                    model.is_sparse, data.njmax, data.naconmax, data.njmax_nnz],
            outputs=[self.counter_array], stream=self.physics.stream,
        )

    def reset(self):
        """Explicit diagnostic boundary, ordered on the physics stream."""
        stream = torch.cuda.ExternalStream(self.physics.stream.cuda_stream, device=self.counters.device)
        with torch.cuda.stream(stream):
            self.counters.zero_()

    def snapshot(self, *, fail_on_overflow=True):
        """One synchronization/download boundary; reject any truncated run."""
        self.physics.wp.synchronize_stream(self.physics.stream)
        nefc, nacon, ncollision, nnz, observations, flags = self.counters.cpu().tolist()
        data = self.physics.data
        result = {
            "schema": "rek.g1_cuda_physics_capacity_high_water.v1",
            "constraint_source_sha256": CONSTRAINT_SOURCE_SHA256,
            "arenas": data.nworld,
            "observed_constraint_builds": observations,
            "high_water": {"constraints_per_arena": nefc, "contacts_aggregate": nacon,
                           "broadphase_pairs_aggregate": ncollision, "sparse_nnz_per_arena": nnz},
            "capacity": {"constraints_per_arena": data.njmax, "contacts_aggregate": data.naconmax,
                         "sparse_nnz_per_arena": data.njmax_nnz},
            "overflow_flags": flags,
            "overflow": {"constraints": bool(flags & 1), "sparse_nnz": bool(flags & 2),
                         "contacts": bool(flags & 4), "broadphase": bool(flags & 8)},
            "limits": "Per-constraint-build peaks include every captured forward inside each physics substep and reset refresh. Sparse nnz is the actual temporary allocation counter, including allocations refused by capacity guards. This instrumented diagnostic is not an SPS benchmark or a guarantee for untested states.",
        }
        if fail_on_overflow and flags:
            raise RuntimeError(f"CUDA physics capacity overflow; reject diagnostic: {result}")
        if not observations:
            raise RuntimeError("capacity diagnostic observed no constraint builds")
        return result
