"""Capture and explain GPU kick-contact acceptance without changing combat.

The probe runs the existing approach/front-kick policy through CUDA graph
replay by default, or eagerly when requested. It copies a bounded per-arena
view of the already packed native contacts into CUDA scratch buffers after
every 2 ms substep, archives each graph replay on-device, and performs one host
transfer after the run. Offline replay follows g1_hit_detector.c gate order
and verifies its attribution, score, and point totals against the native CUDA
counters before reporting any rejected gate.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch

from gpu_semantic_duel import GpuDuelConfig, GpuSemanticDuel


PHYSICS_SUBSTEPS = 10
CONTACT_BYTES = 120
IMPACT_EVENT_BYTES = 20
SPEED_THRESHOLD_MPS = np.float32(1.75)
ATTRIBUTION_APPROACH_MPS = np.float32(2.0)
COOLDOWN_SECONDS = np.float32(0.30000001192092896)
APEX_MIN_RAMP = np.float32(0.20000000298023224)
SCORING_ZONES = {1, 2, 3, 12, 13}
KICK_PARTS = {2, 8}
ZONE_NAMES = {
    0: "unknown", 1: "head", 2: "torso", 3: "pelvis",
    4: "left_shoulder", 5: "right_shoulder",
    6: "left_elbow", 7: "right_elbow",
    8: "left_wrist", 9: "right_wrist",
    10: "left_fist", 11: "right_fist",
    12: "left_hip", 13: "right_hip",
    14: "left_knee", 15: "right_knee",
    16: "left_ankle", 17: "right_ankle",
}
PART_NAMES = {1: "hand", 2: "foot", 8: "shin"}
SIDE_NAMES = {0: "left", 1: "right"}


class ImpactEvent(ct.Structure):
    _fields_ = [
        ("impact_time_seconds", ct.c_float),
        ("lead_time_seconds", ct.c_float),
        ("release_time_seconds", ct.c_float),
        ("gain_boost", ct.c_float),
        ("limb", ct.c_int32),
    ]


class StrikeIntent(ct.Structure):
    _fields_ = [
        ("impact_events", ct.c_void_p),
        ("impact_event_count", ct.c_size_t),
        ("clip_cursor_frames", ct.c_float),
        ("clip_fps", ct.c_float),
        ("move_id", ct.c_int32),
        ("action_playing", ct.c_uint8),
        ("layer_active", ct.c_uint8),
        ("layer_loop", ct.c_uint8),
    ]


class HitContact(ct.Structure):
    _fields_ = [
        ("strike_intent", StrikeIntent),
        ("striker_body_position_world", ct.c_float * 3),
        ("target_body_position_world", ct.c_float * 3),
        ("striker_body_linear_velocity_world", ct.c_float * 3),
        ("target_body_linear_velocity_world", ct.c_float * 3),
        ("relative_speed_mps", ct.c_float),
        ("time_seconds", ct.c_float),
        ("striker_part", ct.c_int32),
        ("striker_side", ct.c_int32),
        ("target_zone", ct.c_int32),
        ("striker_fighter", ct.c_uint32),
        ("target_fighter", ct.c_uint32),
        ("striker_body_slot", ct.c_uint32),
        ("is_enter", ct.c_uint8),
        ("round_active", ct.c_uint8),
        ("striker_upright", ct.c_uint8),
        ("target_upright", ct.c_uint8),
        ("target_standing", ct.c_uint8),
    ]


if ct.sizeof(ImpactEvent) != IMPACT_EVENT_BYTES \
        or ct.sizeof(HitContact) != CONTACT_BYTES:
    raise RuntimeError("native contact diagnostic ABI mismatch")


def load_config(path: Path) -> GpuDuelConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "model", "assets", "controller_manifest", "controller_source",
        "motion_features", "motion_library", "combat_library",
    ):
        values[key] = Path(values[key])
    values["move_duration_ticks"] = tuple(values["move_duration_ticks"])
    return GpuDuelConfig(**values)


class PackedContactTrace:
    """Fixed-size CUDA trace supporting eager calls and captured graph replay."""

    def __init__(self, env: GpuSemanticDuel, ticks: int, per_arena: int):
        self.env = env
        self.ticks = ticks
        self.per_arena = per_arena
        self.record_calls = 0
        self.archived_ticks = 0
        device = env.actions.device
        arenas, rows = env.arenas, env.rows
        self.contacts = torch.empty(
            (ticks, PHYSICS_SUBSTEPS, arenas, per_arena, CONTACT_BYTES),
            dtype=torch.uint8, device=device)
        self.valid = torch.empty(
            (ticks, PHYSICS_SUBSTEPS, arenas, per_arena),
            dtype=torch.bool, device=device)
        self.counts = torch.empty(
            (ticks, PHYSICS_SUBSTEPS, arenas),
            dtype=torch.int64, device=device)
        self.attributed = torch.empty(
            (ticks, PHYSICS_SUBSTEPS, arenas),
            dtype=torch.uint32, device=device)
        self.scored = torch.empty_like(self.attributed)
        self.points = torch.empty(
            (ticks, PHYSICS_SUBSTEPS, rows),
            dtype=torch.int32, device=device)
        self.begin_reset = torch.empty(
            (ticks, PHYSICS_SUBSTEPS, arenas),
            dtype=torch.bool, device=device)
        self.complete_reset = torch.empty_like(self.begin_reset)
        self.episode_reset = torch.empty_like(self.begin_reset)
        self.terminal = torch.empty(
            (ticks, PHYSICS_SUBSTEPS, arenas),
            dtype=torch.uint8, device=device)
        self._current_contacts = torch.empty(
            (PHYSICS_SUBSTEPS, arenas, per_arena, CONTACT_BYTES),
            dtype=torch.uint8, device=device)
        self._current_valid = torch.empty(
            (PHYSICS_SUBSTEPS, arenas, per_arena),
            dtype=torch.bool, device=device)
        self._current_counts = torch.empty(
            (PHYSICS_SUBSTEPS, arenas), dtype=torch.int64, device=device)
        self._current_attributed = torch.empty(
            (PHYSICS_SUBSTEPS, arenas), dtype=torch.uint32, device=device)
        self._current_scored = torch.empty_like(self._current_attributed)
        self._current_points = torch.empty(
            (PHYSICS_SUBSTEPS, rows), dtype=torch.int32, device=device)
        self._current_begin_reset = torch.empty(
            (PHYSICS_SUBSTEPS, arenas), dtype=torch.bool, device=device)
        self._current_complete_reset = torch.empty_like(
            self._current_begin_reset)
        self._current_episode_reset = torch.empty_like(
            self._current_begin_reset)
        self._current_terminal = torch.empty(
            (PHYSICS_SUBSTEPS, arenas), dtype=torch.uint8, device=device)
        self._slots = torch.arange(
            per_arena, dtype=torch.int64, device=device).expand(arenas, -1)
        self._indices = torch.empty_like(self._slots)

    def begin_tick(self, tick: int) -> None:
        if tick != self.archived_ticks or self.record_calls % PHYSICS_SUBSTEPS:
            raise RuntimeError("contact trace tick boundary is inconsistent")

    def record(self, batch, output) -> None:
        substep = self.record_calls % PHYSICS_SUBSTEPS
        counts = batch.hits.candidate_counts
        offsets = batch.hits.candidate_offsets
        self._current_counts[substep].copy_(counts)
        torch.add(offsets[:, None], self._slots, out=self._indices)
        self._indices.clamp_(0, self.env.combat.packed_contacts.shape[0] - 1)
        torch.index_select(
            self.env.combat.packed_contacts,
            0,
            self._indices.reshape(-1),
            out=self._current_contacts[substep].reshape(-1, CONTACT_BYTES),
        )
        valid = self._slots < counts[:, None]
        self._current_valid[substep].copy_(valid)
        self._current_contacts[substep].masked_fill_(~valid[..., None], 0)
        self._current_attributed[substep].copy_(output.tick_attributed_contacts)
        self._current_scored[substep].copy_(output.tick_scored_contacts)
        self._current_points[substep].copy_(output.tick_score_delta)
        self._current_begin_reset[substep].copy_(output.begin_reset)
        self._current_complete_reset[substep].copy_(output.complete_reset)
        self._current_episode_reset[substep].copy_(output.episode_reset)
        self._current_terminal[substep].copy_(self.env.combat._arena_terminals)
        self.record_calls += 1

    def archive_tick(self, tick: int) -> None:
        if tick != self.archived_ticks or self.record_calls % PHYSICS_SUBSTEPS:
            raise RuntimeError("contact trace archive boundary is inconsistent")
        for destination, source in (
            (self.contacts, self._current_contacts),
            (self.valid, self._current_valid),
            (self.counts, self._current_counts),
            (self.attributed, self._current_attributed),
            (self.scored, self._current_scored),
            (self.points, self._current_points),
            (self.begin_reset, self._current_begin_reset),
            (self.complete_reset, self._current_complete_reset),
            (self.episode_reset, self._current_episode_reset),
            (self.terminal, self._current_terminal),
        ):
            destination[tick].copy_(source)
        self.archived_ticks += 1

    def finish(self) -> None:
        if self.archived_ticks != self.ticks \
                or self.record_calls % PHYSICS_SUBSTEPS:
            raise RuntimeError("contact trace is incomplete")

    def download(self) -> dict[str, np.ndarray]:
        self.finish()
        return {
            "packed_contacts": self.contacts.cpu().numpy(),
            "contact_valid": self.valid.cpu().numpy(),
            "contact_counts": self.counts.cpu().numpy(),
            "attributed_cumulative": self.attributed.cpu().numpy(),
            "scored_cumulative": self.scored.cpu().numpy(),
            "points_cumulative": self.points.cpu().numpy(),
            "begin_reset": self.begin_reset.cpu().numpy(),
            "complete_reset": self.complete_reset.cpu().numpy(),
            "episode_reset": self.episode_reset.cpu().numpy(),
            "terminal": self.terminal.cpu().numpy(),
        }


class HitReplayState:
    def __init__(self) -> None:
        self.last_score_time = np.zeros((2, 6), dtype=np.float32)
        self.cooldown_seen = np.zeros((2, 6), dtype=np.bool_)
        self.scored_move_id = np.zeros(2, dtype=np.int32)
        self.scored_apex_mask = np.zeros(2, dtype=np.uint32)
        self.scored_move_seen = np.zeros(2, dtype=np.bool_)

    def reset(self) -> None:
        self.__init__()


def f32(value) -> np.float32:
    return np.float32(value)


def event_ramp(event: ImpactEvent, clip_time: np.float32) -> np.float32:
    delta = f32(clip_time - f32(event.impact_time_seconds))
    if delta > 0:
        release = f32(event.release_time_seconds)
        if release <= 0 or delta > release:
            return f32(0)
        ramp = f32(f32(1) - f32(delta / release))
    else:
        lead = max(f32(0.0001), f32(event.lead_time_seconds))
        if delta < -lead:
            return f32(0)
        ramp = f32(f32(delta / lead) + f32(1))
    doubled = f32(ramp + ramp)
    squared = f32(ramp * ramp)
    return f32(squared * f32(f32(3) - doubled))


def limb_matches(limb: int, part: int, side: int) -> bool:
    limb_is_kick = limb in (3, 4)
    limb_side = 0 if limb in (1, 3) else 1 if limb in (2, 4) else -1
    return side in (0, 1) and limb_side == side \
        and limb_is_kick == (part in KICK_PARTS)


def attribution(contact: HitContact) -> tuple[bool, float, float]:
    direction = np.empty(3, dtype=np.float32)
    norm_squared = f32(0)
    for axis in range(3):
        direction[axis] = f32(
            contact.target_body_position_world[axis]
            - contact.striker_body_position_world[axis])
        norm_squared = f32(
            norm_squared + f32(direction[axis] * direction[axis]))
    if not (norm_squared > 0) or not np.isfinite(norm_squared):
        return False, 0.0, 0.0
    inverse_norm = f32(f32(1) / np.sqrt(norm_squared))
    striker_approach = f32(0)
    target_approach = f32(0)
    for axis in range(3):
        direction[axis] = f32(direction[axis] * inverse_norm)
        striker_approach = f32(striker_approach + f32(
            f32(contact.striker_body_linear_velocity_world[axis])
            * direction[axis]))
        target_approach = f32(target_approach - f32(
            f32(contact.target_body_linear_velocity_world[axis])
            * direction[axis]))
    accepted = bool(
        bool(contact.target_standing)
        and striker_approach >= ATTRIBUTION_APPROACH_MPS
        and striker_approach > target_approach
    )
    return accepted, float(striker_approach), float(target_approach)


def intent_apex(
    contact: HitContact,
    events: list[ImpactEvent],
    event_base: int,
) -> tuple[int | None, float, float | None, str | None]:
    intent = contact.strike_intent
    if not intent.action_playing or not intent.layer_active or intent.layer_loop:
        return None, 0.0, None, "inactive_strike_intent"
    if not math.isfinite(intent.clip_cursor_frames) \
            or not math.isfinite(intent.clip_fps) or intent.clip_fps <= 0 \
            or intent.clip_cursor_frames < 0 or not intent.impact_events \
            or intent.impact_event_count == 0:
        return None, 0.0, None, "invalid_or_empty_strike_intent"
    address = int(intent.impact_events)
    byte_offset = address - event_base
    if byte_offset < 0 or byte_offset % IMPACT_EVENT_BYTES:
        return None, 0.0, None, "impact_event_pointer_out_of_catalog"
    first = byte_offset // IMPACT_EVENT_BYTES
    count = int(intent.impact_event_count)
    if first + count > len(events):
        return None, 0.0, None, "impact_event_range_out_of_catalog"

    cursor = float(intent.clip_cursor_frames)
    cursor_floor = math.floor(cursor)
    fraction = cursor - cursor_floor
    rounded = cursor_floor
    if fraction > 0.5 or (fraction == 0.5 and cursor_floor % 2):
        rounded += 1
    clip_time = f32(f32(rounded) / f32(intent.clip_fps))
    maximum = f32(0)
    matched_event = False
    for index in range(count):
        event = events[first + index]
        if event.limb == 0 or not limb_matches(
                event.limb, contact.striker_part, contact.striker_side):
            continue
        matched_event = True
        ramp = event_ramp(event, clip_time)
        maximum = max(maximum, ramp)
        if ramp >= APEX_MIN_RAMP:
            return index, float(ramp), float(clip_time), None
    reason = "no_limb_matching_impact_event" if not matched_event \
        else "apex_ramp_below_threshold"
    return None, float(maximum), float(clip_time), reason


def process_contact(
    state: HitReplayState,
    contact: HitContact,
    events: list[ImpactEvent],
    event_base: int,
) -> dict:
    accepted_attribution = False
    striker_approach = target_approach = 0.0
    result = {
        "relative_speed_mps": float(contact.relative_speed_mps),
        "time_seconds": float(contact.time_seconds),
        "striker": int(contact.striker_fighter),
        "target": int(contact.target_fighter),
        "striker_slot": int(contact.striker_body_slot),
        "part": PART_NAMES.get(contact.striker_part, str(contact.striker_part)),
        "side": SIDE_NAMES.get(contact.striker_side, str(contact.striker_side)),
        "target_zone": ZONE_NAMES.get(contact.target_zone, str(contact.target_zone)),
        "round_active": bool(contact.round_active),
        "striker_upright": bool(contact.striker_upright),
        "target_upright": bool(contact.target_upright),
        "target_standing": bool(contact.target_standing),
        "clip_cursor_frames": float(contact.strike_intent.clip_cursor_frames),
        "clip_fps": float(contact.strike_intent.clip_fps),
        "move_id": int(contact.strike_intent.move_id),
        "action_playing": bool(contact.strike_intent.action_playing),
        "layer_active": bool(contact.strike_intent.layer_active),
        "layer_loop": bool(contact.strike_intent.layer_loop),
        "attribution_accepted": False,
        "score_accepted": False,
        "points": 0,
        "score_gate": None,
    }
    if not contact.is_enter:
        result["score_gate"] = "not_contact_enter"
        return result
    if f32(contact.relative_speed_mps) < SPEED_THRESHOLD_MPS:
        result["score_gate"] = "relative_speed_below_threshold"
        return result

    accepted_attribution, striker_approach, target_approach = attribution(contact)
    result.update({
        "attribution_accepted": accepted_attribution,
        "striker_approach_mps": striker_approach,
        "target_approach_mps": target_approach,
    })
    failed = []
    if contact.target_zone not in SCORING_ZONES:
        failed.append("target_zone_not_scoring")
    if not contact.round_active:
        failed.append("round_inactive")
    if not contact.striker_upright:
        failed.append("striker_not_upright")
    if not contact.target_upright:
        failed.append("target_not_upright")
    if failed:
        result["score_gate"] = "+".join(failed)
        return result

    apex_index, apex_ramp, clip_time, apex_failure = intent_apex(
        contact, events, event_base)
    result["clip_time_seconds"] = clip_time
    result["apex_ramp"] = apex_ramp
    result["apex_event_index"] = apex_index
    if apex_failure:
        result["score_gate"] = apex_failure
        return result

    fighter = int(contact.striker_fighter)
    slot = int(contact.striker_body_slot)
    elapsed = f32(f32(contact.time_seconds) - state.last_score_time[fighter, slot])
    if state.cooldown_seen[fighter, slot] and elapsed < COOLDOWN_SECONDS:
        result["cooldown_elapsed_seconds"] = float(elapsed)
        result["score_gate"] = "per_body_cooldown"
        return result
    bit_index = min(int(apex_index), 30)
    apex_bit = np.uint32(1 << bit_index)
    move_id = int(contact.strike_intent.move_id)
    if state.scored_move_seen[fighter] \
            and state.scored_move_id[fighter] == move_id \
            and state.scored_apex_mask[fighter] & apex_bit:
        result["score_gate"] = "move_apex_already_scored"
        return result

    state.last_score_time[fighter, slot] = f32(contact.time_seconds)
    state.cooldown_seen[fighter, slot] = True
    if not state.scored_move_seen[fighter] \
            or state.scored_move_id[fighter] != move_id:
        state.scored_move_id[fighter] = move_id
        state.scored_apex_mask[fighter] = 0
        state.scored_move_seen[fighter] = True
    state.scored_apex_mask[fighter] |= apex_bit
    result["score_accepted"] = True
    result["points"] = 2 if contact.striker_part in KICK_PARTS else 1
    result["score_gate"] = "accepted"
    return result


def parse_impact_events(raw: np.ndarray) -> list[ImpactEvent]:
    return [ImpactEvent.from_buffer_copy(row.tobytes()) for row in raw]


def analyze(
    arrays: dict[str, np.ndarray],
    impact_events_raw: np.ndarray,
    impact_event_base: int,
    per_arena: int,
) -> dict:
    counts = arrays["contact_counts"]
    ticks, substeps, arenas = counts.shape
    overflow = np.argwhere(counts > per_arena)
    if overflow.size:
        return {
            "conclusive": False,
            "reason": "contact_trace_capacity_exceeded",
            "first_overflow_tick_substep_arena": overflow[0].tolist(),
            "maximum_contacts_in_one_arena_substep": int(counts.max()),
        }
    events = parse_impact_events(impact_events_raw)
    states = [HitReplayState() for _ in range(arenas)]
    records = []
    mismatches = []
    arena_totals = [dict(attributed=0, scored=0, points=[0, 0]) for _ in range(arenas)]
    for tick in range(ticks):
        for arena in range(arenas):
            if arrays["episode_reset"][tick, 0, arena]:
                states[arena].reset()
            expected_attributed = 0
            expected_scored = 0
            expected_points = [0, 0]
            terminal_latched = False
            for substep in range(substeps):
                complete = bool(arrays["complete_reset"][tick, substep, arena])
                if not terminal_latched and not complete:
                    count = int(counts[tick, substep, arena])
                    for slot in range(count):
                        if not arrays["contact_valid"][tick, substep, arena, slot]:
                            mismatches.append({
                                "tick": tick, "substep": substep, "arena": arena,
                                "kind": "missing_valid_contact_slot", "slot": slot,
                            })
                            continue
                        contact = HitContact.from_buffer_copy(
                            arrays["packed_contacts"][tick, substep, arena, slot].tobytes())
                        result = process_contact(
                            states[arena], contact, events, impact_event_base)
                        result.update({
                            "tick": tick, "substep": substep,
                            "arena": arena, "contact_order": slot,
                        })
                        if result["attribution_accepted"]:
                            expected_attributed += 1
                            arena_totals[arena]["attributed"] += 1
                        if result["score_accepted"]:
                            expected_scored += 1
                            arena_totals[arena]["scored"] += 1
                            fighter = result["striker"]
                            expected_points[fighter] += result["points"]
                            arena_totals[arena]["points"][fighter] += result["points"]
                        if result["attribution_accepted"] or result["score_accepted"]:
                            records.append(result)
                observed = (
                    int(arrays["attributed_cumulative"][tick, substep, arena]),
                    int(arrays["scored_cumulative"][tick, substep, arena]),
                    arrays["points_cumulative"][
                        tick, substep, arena * 2:arena * 2 + 2].astype(int).tolist(),
                )
                expected = (expected_attributed, expected_scored, expected_points.copy())
                if observed != expected:
                    mismatches.append({
                        "tick": tick, "substep": substep, "arena": arena,
                        "kind": "native_counter_replay_mismatch",
                        "expected": expected, "observed": observed,
                    })
                if arrays["begin_reset"][tick, substep, arena]:
                    states[arena].reset()
                terminal_latched = terminal_latched \
                    or bool(arrays["terminal"][tick, substep, arena])

    unscored_attributed = [
        record for record in records
        if record["attribution_accepted"] and not record["score_accepted"]
    ]
    gates = {}
    for record in unscored_attributed:
        gate = record["score_gate"]
        gates[gate] = gates.get(gate, 0) + 1
    return {
        "conclusive": not mismatches,
        "native_counter_replay_mismatches": mismatches[:20],
        "arena_totals": arena_totals,
        "attributed_or_scored_contacts": records,
        "unscored_attributed_contacts": unscored_attributed,
        "unscored_attributed_gate_counts": gates,
        "maximum_contacts_in_one_arena_substep": int(counts.max()),
    }


def run(config: GpuDuelConfig, steps: int, per_arena: int, eager: bool):
    setup_start = time.perf_counter()
    env = GpuSemanticDuel(config)
    with torch.cuda.stream(env.stream):
        trace = PackedContactTrace(env, steps, per_arena)
        actions = torch.ones((env.rows, 1), dtype=torch.int64, device=config.device)
        requested = torch.empty((steps, env.rows), dtype=torch.int64, device=config.device)
        applied = torch.empty_like(requested)
        original_post_step = env.combat.post_step

        def traced_post_step(batch):
            output = original_post_step(batch)
            trace.record(batch, output)
            return output

        env.combat.post_step = traced_post_step
    if not eager:
        # Python executes the wrapper during capture, placing ten fixed
        # substep snapshots in _current_*. Graph replay refreshes those same
        # buffers; archive_tick then copies them to the next history slot.
        env.capture_step()
    setup_seconds = time.perf_counter() - setup_start
    start = time.perf_counter()
    for tick in range(steps):
        trace.begin_tick(tick)
        actions.fill_(1)
        position = env.observations[0::2, :3]
        opponent = env.observations[0::2, 86:89]
        distance = (position[:, :2] - opponent[:, :2]).square().sum(-1).sqrt()
        kick_ready = env.action_mask[0::2, 17] != 0
        busy = env.observations[0::2, 183] != 0
        command = torch.where(
            distance > 0.80, 2, torch.where(kick_ready, 17, 1))
        actions[0::2, 0] = torch.where(busy, 0, command)
        requested[tick].copy_(actions[:, 0])
        legal = env.action_mask.gather(1, actions) != 0
        neutral_or_continue = torch.where(env.action_mask[:, :1] != 0, 0, 1)
        actions.copy_(torch.where(legal, actions, neutral_or_continue))
        applied[tick].copy_(actions[:, 0])
        if eager:
            env.stream.wait_stream(torch.cuda.current_stream(config.device))
            with torch.cuda.stream(env.stream):
                env.actions.copy_(actions[:, 0])
                env._step_impl()
            torch.cuda.current_stream(config.device).wait_stream(env.stream)
        else:
            env.step(actions)
        trace.archive_tick(tick)
    env.stream.synchronize()
    elapsed = time.perf_counter() - start
    env.check_status()
    arrays = trace.download()
    arrays["requested_actions"] = requested.cpu().numpy()
    arrays["actions"] = applied.cpu().numpy()
    impact_events = env.combat.impact_events.cpu().numpy()
    event_base = env.combat.impact_events.data_ptr()
    analysis = analyze(arrays, impact_events, event_base, per_arena)
    report = {
        "schema": "rek.g1_gpu_kick_contact_diagnostic.v1",
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "arenas": env.arenas,
        "robots": env.rows,
        "steps": steps,
        "simulated_seconds": steps * 0.02,
        "setup_seconds": setup_seconds,
        "wall_seconds": elapsed,
        "per_arena_trace_capacity": per_arena,
        "cuda_graph": not eager,
        "combat_library_sha256": hashlib.sha256(
            config.combat_library.read_bytes()).hexdigest(),
        "analysis": analysis,
    }
    arrays["impact_events"] = impact_events
    arrays["impact_event_base"] = np.array(event_base, dtype=np.uint64)
    env.close()
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=210)
    parser.add_argument("--per-arena", type=int, default=32)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.per_arena < 1:
        parser.error("per-arena contact trace capacity must be positive")
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)
    report, arrays = run(
        load_config(args.config), args.steps, args.per_arena, args.eager)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.out.with_suffix(".npz")
    with trace_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["trace_path"] = str(trace_path)
    report["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["analysis"]["conclusive"]:
        raise RuntimeError("offline hit replay did not match native counters")


if __name__ == "__main__":
    main()
