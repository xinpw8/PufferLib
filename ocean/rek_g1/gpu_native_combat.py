"""CUDA-native REK G1 fall, hit, referee, and combat state bridge.

Call ``begin_tick`` once before each 20 ms controller interval, then call
``post_step`` after every individual 2 ms MuJoCo Warp step. Composer state and
active route IDs are sampled before the scheduler advances the composer.
No hot-path method transfers tensors to the host or synchronizes CUDA.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gpu_combat_measurement import (
    GpuCombatMeasurementBatch,
    GpuFallMeasurementBatch,
    RekG1GpuCombatMeasurement,
)
from gpu_native_motion import Composer


P = ct.c_void_p
Z = ct.c_size_t


@dataclass(frozen=True)
class GpuNativeCombatOutputs:
    """Live CUDA output buffers, updated in place by the bridge."""

    fall: torch.Tensor
    fight: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    dampened: torch.Tensor
    begin_reset: torch.Tensor
    complete_reset: torch.Tensor
    episode_reset: torch.Tensor
    input_reset: torch.Tensor
    clear_contacts: torch.Tensor
    tick_fall_events: torch.Tensor
    tick_signals: torch.Tensor
    tick_referee_calls: torch.Tensor
    tick_score_delta: torch.Tensor
    tick_attributed_contacts: torch.Tensor
    tick_scored_contacts: torch.Tensor
    statuses: torch.Tensor


class GpuNativeCombat:
    """Own persistent native combat/fall state for a CUDA duel batch."""

    def __init__(
        self,
        library: str | Path,
        measurement: RekG1GpuCombatMeasurement,
        motion: Any,
        active_route_ids: torch.Tensor,
    ) -> None:
        self.measurement = measurement
        self.motion = motion
        self.composers = motion.composers.tensor
        self.active_route_ids_source = active_route_ids
        self.device = measurement.device
        self.arenas = measurement.arena_count
        self.rows = self.arenas * 2
        self.candidate_capacity = measurement.contact_capacity * 2
        if self.device.type != "cuda":
            raise ValueError("native combat requires CUDA measurement tensors")
        if tuple(self.composers.shape) != (self.rows, ct.sizeof(Composer)) \
                or self.composers.dtype != torch.uint8 \
                or self.composers.device != self.device \
                or not self.composers.is_contiguous():
            raise ValueError("motion composers must be contiguous native CUDA structs")
        if tuple(active_route_ids.shape) != (self.rows,) \
                or active_route_ids.dtype != torch.int32 \
                or active_route_ids.device != self.device:
            raise ValueError("active_route_ids must be CUDA int32 rows")
        self.active_route_ids = torch.empty(
            self.rows, dtype=torch.int32, device=self.device)

        library_path = Path(library)
        if not library_path.is_file():
            raise FileNotFoundError(library_path)
        self.library = ct.CDLL(str(library_path))
        self._configure_abi()
        state_size = self.library.rek_g1_cuda_native_combat_state_size()
        contact_size = self.library.rek_g1_cuda_native_combat_contact_size()
        event_size = self.library.rek_g1_cuda_native_combat_impact_event_size()
        event_count = self.library.rek_g1_cuda_native_combat_impact_event_count()
        if state_size != 252 or contact_size != 120 or event_size != 20 \
                or event_count != 29:
            raise RuntimeError("native combat CUDA ABI size mismatch")

        self.states = torch.empty(
            (self.arenas, state_size), dtype=torch.uint8, device=self.device)
        self.packed_contacts = torch.empty(
            (self.candidate_capacity, contact_size),
            dtype=torch.uint8,
            device=self.device,
        )
        self.impact_events = torch.empty(
            (event_count, event_size), dtype=torch.uint8, device=self.device)
        self.route_event_offsets = torch.empty(
            24, dtype=torch.int32, device=self.device)
        self.route_event_counts = torch.empty_like(self.route_event_offsets)

        self.tick_fall_events = torch.zeros(
            self.rows, dtype=torch.uint32, device=self.device)
        self.tick_signals = torch.zeros(
            self.arenas, dtype=torch.uint32, device=self.device)
        self.tick_referee_calls = torch.zeros_like(self.tick_signals)
        self.tick_score_delta = torch.zeros(
            self.rows, dtype=torch.int32, device=self.device)
        self.tick_attributed_contacts = torch.zeros_like(self.tick_signals)
        self.tick_scored_contacts = torch.zeros_like(self.tick_signals)
        self._arena_terminals = torch.zeros(
            self.arenas, dtype=torch.uint8, device=self.device)
        self.dampened = torch.zeros(
            self.rows, dtype=torch.bool, device=self.device)
        self.begin_reset = torch.zeros(
            self.arenas, dtype=torch.bool, device=self.device)
        self.complete_reset = torch.zeros_like(self.begin_reset)
        self.episode_reset = torch.zeros_like(self.begin_reset)
        self.input_reset = torch.zeros(
            self.rows, dtype=torch.bool, device=self.device)
        self.clear_contacts = torch.zeros_like(self.begin_reset)
        self.statuses = torch.empty(
            self.arenas, dtype=torch.int32, device=self.device)
        self.fall = torch.empty(
            (self.rows, 15), dtype=torch.float32, device=self.device)
        self.fight = torch.empty(
            (self.rows, 39), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty(
            self.rows, dtype=torch.float32, device=self.device)
        self.terminals = torch.empty_like(self.rewards)
        self.outputs = GpuNativeCombatOutputs(
            fall=self.fall,
            fight=self.fight,
            rewards=self.rewards,
            terminals=self.terminals,
            dampened=self.dampened,
            begin_reset=self.begin_reset,
            complete_reset=self.complete_reset,
            episode_reset=self.episode_reset,
            input_reset=self.input_reset,
            clear_contacts=self.clear_contacts,
            tick_fall_events=self.tick_fall_events,
            tick_signals=self.tick_signals,
            tick_referee_calls=self.tick_referee_calls,
            tick_score_delta=self.tick_score_delta,
            tick_attributed_contacts=self.tick_attributed_contacts,
            tick_scored_contacts=self.tick_scored_contacts,
            statuses=self.statuses,
        )

        with torch.cuda.device(self.device):
            self._check_launch(self.library.rek_g1_cuda_native_combat_upload_catalog(
                self.impact_events.data_ptr(),
                self.route_event_offsets.data_ptr(),
                self.route_event_counts.data_ptr(),
            ))
        self.reset()

    def _configure_abi(self) -> None:
        for name in (
            "rek_g1_cuda_native_combat_state_size",
            "rek_g1_cuda_native_combat_contact_size",
            "rek_g1_cuda_native_combat_impact_event_size",
            "rek_g1_cuda_native_combat_impact_event_count",
        ):
            function = getattr(self.library, name)
            function.argtypes = []
            function.restype = Z
        upload = self.library.rek_g1_cuda_native_combat_upload_catalog
        upload.argtypes = [P, P, P]
        upload.restype = ct.c_int
        init = self.library.rek_g1_cuda_native_combat_init
        init.argtypes = [P, P, Z, P]
        init.restype = ct.c_int
        begin = self.library.rek_g1_cuda_native_combat_begin_tick
        begin.argtypes = [P] * 11 + [Z, P]
        begin.restype = ct.c_int
        post = self.library.rek_g1_cuda_native_combat_post_step
        post.argtypes = [P] * 31 + [Z, Z, P]
        post.restype = ct.c_int
        self._deferred_post_step = getattr(
            self.library, "rek_g1_cuda_native_combat_post_step_deferred", None)
        if self._deferred_post_step is not None:
            self._deferred_post_step.argtypes = post.argtypes
            self._deferred_post_step.restype = ct.c_int
        observe = self.library.rek_g1_cuda_native_combat_observe
        observe.argtypes = [P] * 16 + [Z, P]
        observe.restype = ct.c_int

    @property
    def stream(self) -> int:
        return torch.cuda.current_stream(self.device).cuda_stream

    @staticmethod
    def _check_launch(status: int) -> None:
        if status:
            raise RuntimeError(f"CUDA launch failed with runtime code {status}")

    @staticmethod
    def _require_tensor(
        tensor: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        name: str,
    ) -> None:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype \
                or tensor.device != device or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous {dtype} on {device} with shape {shape}")

    def _validate_measurement(self, batch: GpuCombatMeasurementBatch) -> None:
        fall = batch.fall
        hits = batch.hits
        expected = (
            (fall.floats, (self.rows, 5), torch.float32, "fall.floats"),
            (fall.integers, (self.rows, 7), torch.int64, "fall.integers"),
            (fall.valid, (self.rows,), torch.bool, "fall.valid"),
            (hits.integers, (self.candidate_capacity, 12), torch.int64,
             "hits.integers"),
            (hits.floats, (self.candidate_capacity, 13), torch.float32,
             "hits.floats"),
            (hits.candidate_valid, (self.candidate_capacity,), torch.bool,
             "hits.candidate_valid"),
            (hits.candidate_order, (self.candidate_capacity,), torch.int64,
             "hits.candidate_order"),
            (hits.candidate_offsets, (self.arenas,), torch.int64,
             "hits.candidate_offsets"),
            (hits.candidate_counts, (self.arenas,), torch.int64,
             "hits.candidate_counts"),
            (hits.arena_scan_valid, (self.arenas,), torch.bool,
             "hits.arena_scan_valid"),
            (hits.arena_time_seconds, (self.arenas,), torch.float32,
             "hits.arena_time_seconds"),
        )
        for tensor, shape, dtype, name in expected:
            self._require_tensor(tensor, shape, dtype, self.device, name)

    def reset(self) -> GpuNativeCombatOutputs:
        """Initialize all native states and recalibrate measured fall geometry."""
        self.measurement.reset()
        with torch.cuda.device(self.device):
            self._check_launch(self.library.rek_g1_cuda_native_combat_init(
                self.states.data_ptr(), self.statuses.data_ptr(),
                self.arenas, self.stream,
            ))
        self.tick_fall_events.zero_()
        self.tick_signals.zero_()
        self.tick_referee_calls.zero_()
        self.tick_score_delta.zero_()
        self.tick_attributed_contacts.zero_()
        self.tick_scored_contacts.zero_()
        self._arena_terminals.zero_()
        self.dampened.zero_()
        self.begin_reset.zero_()
        self.complete_reset.zero_()
        self.episode_reset.zero_()
        self.input_reset.zero_()
        self.clear_contacts.zero_()
        fall = self.measurement.sample_fall(torch.zeros(
            self.rows, dtype=torch.int64, device=self.device))
        return self.observe(fall)

    def begin_tick(self) -> GpuNativeCombatOutputs:
        """Clear tick accumulators and apply any prior round-end episode reset."""
        with torch.cuda.device(self.device):
            self._check_launch(self.library.rek_g1_cuda_native_combat_begin_tick(
                self.states.data_ptr(),
                self.tick_fall_events.data_ptr(),
                self.tick_signals.data_ptr(),
                self.tick_referee_calls.data_ptr(),
                self.tick_score_delta.data_ptr(),
                self.tick_attributed_contacts.data_ptr(),
                self.tick_scored_contacts.data_ptr(),
                self._arena_terminals.data_ptr(),
                self.episode_reset.data_ptr(),
                self.input_reset.data_ptr(),
                self.statuses.data_ptr(),
                self.arenas,
                self.stream,
            ))
        self.measurement.clear_arena_contacts(self.episode_reset)
        return self.outputs

    def post_step(
        self,
        batch: GpuCombatMeasurementBatch,
        *,
        pack_observation: bool = True,
    ) -> GpuNativeCombatOutputs:
        """Advance exact native state after one individual 2 ms physics step.

        Deferred packing is an opt-in for consumers that use only state/reset
        events until a later observe call. Its native entrypoint preserves all
        validation and status latches, including the reset-completion branch.
        The default continues to support the original deployed library ABI.
        Failed arenas can retain older diagnostic packed values; their latched
        status invalidates the whole run. No accepted result may use those
        values. Native reset/referee state and valid arenas are unaffected.
        """
        if not isinstance(pack_observation, bool):
            raise ValueError("pack_observation must be a Boolean")
        native_post_step = self.library.rek_g1_cuda_native_combat_post_step
        if not pack_observation:
            native_post_step = self._deferred_post_step
            if native_post_step is None:
                raise RuntimeError("deferred observation packing requires the opt-in native combat library")
        self._validate_measurement(batch)
        fall = batch.fall
        hits = batch.hits
        self.active_route_ids.copy_(self.active_route_ids_source)
        pointers = (
            self.states,
            fall.floats,
            fall.integers,
            fall.valid,
            hits.integers,
            hits.floats,
            hits.candidate_valid,
            hits.candidate_order,
            hits.candidate_offsets,
            hits.candidate_counts,
            hits.arena_scan_valid,
            hits.arena_time_seconds,
            self.composers,
            self.active_route_ids,
            self.impact_events,
            self.route_event_offsets,
            self.route_event_counts,
            self.packed_contacts,
            self.tick_fall_events,
            self.tick_signals,
            self.tick_referee_calls,
            self.tick_score_delta,
            self.tick_attributed_contacts,
            self.tick_scored_contacts,
            self._arena_terminals,
            self.input_reset,
            self.dampened,
            self.begin_reset,
            self.complete_reset,
            self.clear_contacts,
            self.statuses,
        )
        with torch.cuda.device(self.device):
            self._check_launch(native_post_step(
                *(tensor.data_ptr() for tensor in pointers),
                self.arenas,
                self.candidate_capacity,
                self.stream,
            ))
        self.measurement.clear_arena_contacts(self.clear_contacts)
        return self.observe(fall) if pack_observation else self.outputs

    def sample_and_post_step(
        self,
        physics_substep_index: int,
        can_get_up: torch.Tensor,
    ) -> GpuNativeCombatOutputs:
        """Measure and consume one 2 ms state without a host synchronization."""
        return self.post_step(self.measurement.sample(
            physics_substep_index, can_get_up))

    def observe(
        self,
        fall: GpuFallMeasurementBatch,
    ) -> GpuNativeCombatOutputs:
        """Refresh fall15/fight39 and reward/terminal outputs without state change."""
        self._require_tensor(
            fall.floats, (self.rows, 5), torch.float32,
            self.device, "fall.floats")
        self._require_tensor(
            fall.integers, (self.rows, 7), torch.int64,
            self.device, "fall.integers")
        self._require_tensor(
            fall.valid, (self.rows,), torch.bool,
            self.device, "fall.valid")
        pointers = (
            self.states,
            fall.floats,
            fall.integers,
            fall.valid,
            self.tick_fall_events,
            self.tick_signals,
            self.tick_referee_calls,
            self.tick_score_delta,
            self.tick_attributed_contacts,
            self.tick_scored_contacts,
            self._arena_terminals,
            self.fall,
            self.fight,
            self.rewards,
            self.terminals,
            self.statuses,
        )
        with torch.cuda.device(self.device):
            self._check_launch(self.library.rek_g1_cuda_native_combat_observe(
                *(tensor.data_ptr() for tensor in pointers),
                self.arenas,
                self.stream,
            ))
        return self.outputs

    def check_status(self) -> None:
        """Synchronize explicitly and raise if any arena latched an error."""
        statuses = self.statuses.cpu()
        failing = torch.nonzero(statuses, as_tuple=False).flatten()
        if failing.numel():
            details = [
                (int(index), int(statuses[index]))
                for index in failing.tolist()
            ]
            raise RuntimeError(f"native combat arena failures: {details}")
