"""Opt-in conditional reset forwarding with a shared forced-reference path.

The default physics class is unchanged. Construct this gate before capture,
assign ``physics.forward_selected = gate.forward_selected``, then reset the
environment. Eager calls retain the original unconditional implementation.
``force_forward`` is a device Boolean for paired diagnostics: forcing true and
letting a nonempty reset mask select true execute the same captured body and
use the same scratch addresses. No timestep, solver, or contact order changes.
"""

from __future__ import annotations


class PersistentScratch:
    """Record exact Warp allocation sizes, then replay owned stable pointers."""

    def __init__(self, allocate):
        self.allocate = allocate
        self.buffers = []
        self.sizes = []
        self.cursor = None
        self.deleter = lambda pointer, size: None

    def alloc(self, size):
        if self.cursor is None:
            value = self.allocate(size)
            self.buffers.append(value)
            self.sizes.append(size)
            return value.data_ptr()
        index = self.cursor
        if index >= len(self.sizes) or self.sizes[index] != size:
            raise RuntimeError("Warp scratch allocation sequence changed")
        self.cursor += 1
        return self.buffers[index].data_ptr()

    def free(self, pointer, size):
        # Buffers remain owned until all graphs using this object are destroyed.
        pass

    def begin_replay(self):
        if self.cursor is not None:
            raise RuntimeError("scratch replay is already active")
        self.cursor = 0

    def end_replay(self):
        cursor = self.cursor
        self.cursor = None
        if cursor != len(self.sizes):
            raise RuntimeError("Warp scratch allocation count changed")


class PersistentWarpCall:
    """Setup-only pinned scratch for one fixed Warp call sequence.

    Preparation executes the operation. Its caller must reset physical state
    afterwards. The allocator is restored on every exit and never changed by
    CUDA graph replay. This requires the inspected Warp private allocator API.
    """

    def __init__(self, physics, operation):
        import torch

        self.physics = physics
        self.operation = operation
        if physics.stream.is_capturing:
            raise RuntimeError("persistent call setup cannot run during capture")
        if not hasattr(physics.device, "current_allocator"):
            raise RuntimeError("installed Warp lacks current_allocator")
        self.scratch = PersistentScratch(
            lambda size: torch.empty(size, dtype=torch.uint8, device=physics.qpos.device)
        )
        original = physics.device.current_allocator
        caller = torch.cuda.current_stream(physics.qpos.device)
        stream = torch.cuda.ExternalStream(physics.stream.cuda_stream, device=physics.qpos.device)
        stream.wait_stream(caller)
        try:
            with torch.cuda.stream(stream), physics.wp.ScopedDevice(physics.device), physics.wp.ScopedStream(physics.stream, sync_enter=False):
                physics.device.current_allocator = self.scratch
                operation()
                stream.synchronize()
                with physics.wp.ScopedCapture(stream=physics.stream) as capture:
                    self.capture()
                self.validation_graph = capture.graph
        finally:
            physics.device.current_allocator = original
            caller.wait_stream(stream)

    def capture(self):
        physics = self.physics
        if not physics.stream.is_capturing:
            raise RuntimeError("persistent call is capture-only")
        original = physics.device.current_allocator
        self.scratch.begin_replay()
        try:
            physics.device.current_allocator = self.scratch
            with physics.wp.ScopedDevice(physics.device), physics.wp.ScopedStream(physics.stream, sync_enter=False):
                self.operation()
        finally:
            physics.device.current_allocator = original
            self.scratch.end_replay()

    def snapshot_scratch(self):
        """Diagnostic-only device copies, performed outside the timed loop."""
        return tuple(value.clone() for value in self.scratch.buffers)

    def restore_scratch(self, snapshots):
        if len(snapshots) != len(self.scratch.buffers):
            raise ValueError("scratch snapshot length differs")
        for value, saved in zip(self.scratch.buffers, snapshots):
            if value.shape != saved.shape or value.dtype != saved.dtype or value.device != saved.device:
                raise ValueError("scratch snapshot layout differs")
            value.copy_(saved)


class ResetForwardGate:
    """Skip the native forward body only when every selected arena is false."""

    def __init__(self, physics):
        import torch

        if not physics.wp.is_conditional_graph_supported():
            raise RuntimeError("CUDA conditional graphs are unavailable")
        physics._validate_selected_forward_backend()
        self.physics = physics
        self.original = physics.forward_selected
        self.body = PersistentWarpCall(
            physics, lambda: physics.mjw.forward(physics.model, physics.data)
        )
        from warp._src.context import runtime
        if not runtime.core.wp_cuda_graph_check_conditional_body(self.body.validation_graph.graph):
            raise RuntimeError(runtime.get_error_string())
        device = physics.qpos.device
        self.force_forward = torch.zeros((), dtype=torch.bool, device=device)
        self._condition_bool = torch.zeros((), dtype=torch.bool, device=device)
        self._condition_int = torch.zeros(1, dtype=torch.int32, device=device)
        self._condition_wp = physics.wp.from_torch(self._condition_int, dtype=physics.wp.int32)

    def forward_selected(self, mask):
        import torch

        physics = self.physics
        if mask.shape != (physics.arenas,) or mask.device != physics.qpos.device or mask.dtype != torch.bool:
            raise ValueError("forward mask must be one CUDA Boolean per arena")
        physics._validate_selected_forward_backend()
        if not physics.stream.is_capturing:
            # Warp capture_if would download its predicate outside capture.
            # Preserve unconditional eager behavior instead.
            return self.original(mask)
        for name, values in physics.reset_reader_fields.items():
            physics._reset_reader_backup[name].copy_(values)
        torch.any(mask, out=self._condition_bool)
        self._condition_bool.logical_or_(self.force_forward)
        self._condition_int.copy_(self._condition_bool)
        physics.wp.capture_if(
            self._condition_wp, on_true=self.body.capture, stream=physics.stream
        )
        for name, values in physics.reset_reader_fields.items():
            selected = mask.reshape(-1, *([1] * (values.ndim - 1)))
            values.copy_(torch.where(selected, values, physics._reset_reader_backup[name]))

    def metadata(self):
        return {
            "schema": "rek.reset_forward_gate.v1",
            "opt_in": True,
            "hot_path_host_predicate_reads": 0,
            "eager_forward": "original unconditional implementation",
            "captured_predicate": "any(reset_mask) or force_forward",
            "scratch_allocations": len(self.body.scratch.sizes),
            "scratch_bytes": sum(self.body.scratch.sizes),
            "equivalence_established": False,
        }
