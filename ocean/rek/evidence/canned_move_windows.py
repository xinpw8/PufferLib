#!/usr/bin/env python3
"""Extract compact, request-aligned T800 client-observation windows.

The recorder observes a client transport helper before the request is sent.
Consequently every emitted window is a candidate response to a move request,
not evidence that the server accepted or executed that move.  The skeleton is
the client's visual state sampled in FixedUpdate, not a server-tick trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import deque
from pathlib import Path
from typing import Any


SCHEMA = "rek.t800_canned_move_windows.v2"
RAW_SCHEMA = "rek.private_ai.client_fixed.v3"
GAME_ASSEMBLY_SHA256 = (
    "6bd006d9c16ddb2b55d60f4df106a8fdbd2fef04603acc6492239d579a73d412"
)
GLOBAL_METADATA_SHA256 = (
    "e73d6bc53abf099af09f6d3ce5880c855694a8c7b48d6031e836da6215b5b6bd"
)

MOVE_NAMES = {
    2: "skill",
    3: "youbiantui",
    4: "left_light_attack",
    5: "right_light_attack",
    9: "right_shoryuken_lm",
    10: "front_kick_L",
}


class WindowError(ValueError):
    pass


def require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise WindowError(f"{label}: expected {expected!r}, got {actual!r}")


def _validated_bones(
    robot: dict[str, Any], expected_count: int, label: str
) -> dict[str, Any]:
    bones = robot.get("bones")
    if not isinstance(bones, dict):
        raise WindowError(f"{label} has no bone record")
    require(bones.get("count"), expected_count, f"{label} bone count")
    for field, width in (
        ("world_positions_xyz", 3),
        ("world_rotations_xyzw", 4),
        ("local_rotations_xyzw", 4),
    ):
        values = bones.get(field)
        if not isinstance(values, list):
            raise WindowError(f"{label} {field} is not an array")
        require(len(values), expected_count * width, f"{label} {field} length")
    return bones


def compact_sample(
    sample: dict[str, Any], fighter: int, fighter_bone_count: int,
    opponent_bone_count: int, snapshot_advanced: bool,
) -> dict[str, Any]:
    robot = sample.get(f"fighter_{fighter}")
    opponent = sample.get(f"fighter_{1 - fighter}")
    if not isinstance(robot, dict) or not isinstance(opponent, dict):
        raise WindowError("sample does not contain both fighter records")
    bones = _validated_bones(robot, fighter_bone_count, f"fighter_{fighter}")
    _validated_bones(
        opponent, opponent_bone_count, f"fighter_{1 - fighter}"
    )
    transport = sample.get("transport_observation") or {}
    round_state = sample.get("round") or {}
    return {
        "tick": sample.get("client_fixed_tick"),
        "unity_fixed_time": sample.get("unity_fixed_time"),
        "snapshot_sequence": transport.get("fight_state_snapshot_sequence"),
        "remote_snapshot_advanced_since_prior_client_sample": snapshot_advanced,
        "observation_semantics": (
            "client_visual_state_sampled_in_FixedUpdate; interpolation_between_"
            "remote_authoritative_snapshots_is_possible"
        ),
        "round": {
            "time_remaining": round_state.get("time_remaining"),
            "clean_hits": round_state.get("clean_hits"),
            "falls": round_state.get("falls"),
            "result": round_state.get("result"),
        },
        "fighter": {
            "falling": robot.get("falling"),
            "fallen": robot.get("fallen"),
            "dampened": robot.get("dampened"),
            "resetting": robot.get("resetting"),
            "tilt_angle": robot.get("tilt_angle"),
            "floor_contact_count": robot.get("floor_contact_count"),
            "root_position": robot.get("root_position"),
            "root_rotation": robot.get("root_rotation"),
            "root_linear_velocity": robot.get("root_linear_velocity"),
            "root_angular_velocity": robot.get("root_angular_velocity"),
            "bone_world_positions_xyz": bones.get("world_positions_xyz"),
            "bone_world_rotations_xyzw": bones.get("world_rotations_xyzw"),
            "bone_local_rotations_xyzw": bones.get("local_rotations_xyzw"),
        },
        "opponent": {
            "falling": opponent.get("falling"),
            "fallen": opponent.get("fallen"),
            "root_position": opponent.get("root_position"),
            "root_rotation": opponent.get("root_rotation"),
            "root_linear_velocity": opponent.get("root_linear_velocity"),
            "root_angular_velocity": opponent.get("root_angular_velocity"),
        },
    }


def _relative(sample: dict[str, Any], action_tick: int, fixed_delta: float) -> dict[str, Any]:
    result = dict(sample)
    result["relative_time_s"] = (int(sample["tick"]) - action_tick) * fixed_delta
    return result


def extract(
    raw_path: Path,
    output_path: Path,
    *,
    fighter: int | None = None,
    pre_ms: float = 200.0,
    post_ms: float = 2200.0,
    stride_ms: float = 20.0,
) -> dict[str, Any]:
    if fighter is not None and fighter not in (0, 1):
        raise WindowError("fighter must be 0 or 1")
    if pre_ms < 0 or post_ms <= 0 or stride_ms <= 0:
        raise WindowError("window durations and stride are invalid")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    digest = hashlib.sha256()
    start = None
    end = None
    fixed_delta = None
    pre_ticks = post_ticks = stride_ticks = None
    recent: deque[dict[str, Any]] = deque()
    windows: list[dict[str, Any]] = []
    sample_count = 0
    expected_tick = 0
    local_fighter = None
    opponent_fighter = None
    fighter_bone_count = None
    opponent_bone_count = None
    prior_snapshot_sequence = None

    with raw_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WindowError(f"invalid JSON at line {line_number}") from exc
            event = record.get("event")
            if event == "capture_start":
                if start is not None:
                    raise WindowError("multiple capture_start records")
                start = record
                require(start.get("schema"), RAW_SCHEMA, "raw schema")
                require(start.get("machine"), "D21", "capture host")
                require(start.get("game_assembly_sha256"), GAME_ASSEMBLY_SHA256,
                        "GameAssembly SHA-256")
                require(start.get("global_metadata_sha256"), GLOBAL_METADATA_SHA256,
                        "global metadata SHA-256")
                scope = start.get("scope")
                if not isinstance(scope, dict):
                    raise WindowError("capture scope is absent")
                require(scope.get("allowed"), True, "capture scope allowed")
                require(scope.get("opponent_is_ai"), True, "AI opponent scope")
                require(scope.get("opponent_slot_is_ai"), True,
                        "AI opponent slot scope")
                require(scope.get("human_in_opponent_slot"), False,
                        "human opponent scope")
                local_fighter = scope.get("local_fighter_index")
                if local_fighter not in (0, 1):
                    raise WindowError("local_fighter_index must be 0 or 1")
                opponent_fighter = 1 - local_fighter
                require(scope.get("opponent_slot"), opponent_fighter,
                        "opponent slot")
                if fighter is not None:
                    require(fighter, local_fighter,
                            "requested fighter versus local command issuer")
                fighter = local_fighter
                fighter_bones = start.get(f"fighter_{fighter}_bones")
                opponent_bones = start.get(f"fighter_{opponent_fighter}_bones")
                if not isinstance(fighter_bones, list) or not fighter_bones:
                    raise WindowError("local fighter bone names are absent")
                if not isinstance(opponent_bones, list) or not opponent_bones:
                    raise WindowError("opponent fighter bone names are absent")
                if not all(isinstance(name, str) and name for name in fighter_bones):
                    raise WindowError("local fighter bone names are malformed")
                if not all(isinstance(name, str) and name for name in opponent_bones):
                    raise WindowError("opponent fighter bone names are malformed")
                fighter_bone_count = len(fighter_bones)
                opponent_bone_count = len(opponent_bones)
                fixed_delta = float(start.get("fixed_delta_time"))
                if fixed_delta <= 0:
                    raise WindowError("fixed_delta_time is not positive")
                pre_ticks = round(pre_ms / (1000.0 * fixed_delta))
                post_ticks = round(post_ms / (1000.0 * fixed_delta))
                stride_ticks = max(1, round(stride_ms / (1000.0 * fixed_delta)))
            elif event == "sample":
                if start is None or fixed_delta is None:
                    raise WindowError("sample precedes capture_start")
                tick = record.get("client_fixed_tick")
                require(tick, expected_tick, "contiguous client fixed tick")
                expected_tick += 1
                sample_count += 1
                require(record.get("local_fighter_index"), local_fighter,
                        "sample local fighter index")
                snapshot_sequence = (record.get("transport_observation") or {}).get(
                    "fight_state_snapshot_sequence"
                )
                if not isinstance(snapshot_sequence, int):
                    raise WindowError("sample has no integer snapshot sequence")
                snapshot_advanced = (
                    prior_snapshot_sequence is None
                    or snapshot_sequence != prior_snapshot_sequence
                )
                if (prior_snapshot_sequence is not None
                        and snapshot_sequence < prior_snapshot_sequence):
                    raise WindowError("fight state snapshot sequence regressed")
                prior_snapshot_sequence = snapshot_sequence
                sample = compact_sample(
                    record, fighter, int(fighter_bone_count),
                    int(opponent_bone_count), snapshot_advanced,
                )
                recent.append(sample)
                while recent and int(recent[0]["tick"]) < tick - int(pre_ticks):
                    recent.popleft()
                for window in windows:
                    action_tick = int(window["request"]["tick"])
                    relative_tick = tick - action_tick
                    if 0 <= relative_tick <= int(post_ticks):
                        if relative_tick % int(stride_ticks) == 0:
                            if not window["samples"] or window["samples"][-1]["tick"] != tick:
                                window["samples"].append(
                                    _relative(sample, action_tick, fixed_delta)
                                )
            elif event == "client_transport_method_invoked":
                if record.get("method") != "SendMoveEvent":
                    continue
                if start is None or fixed_delta is None:
                    raise WindowError("move request precedes capture_start")
                input_record = record.get("input") or {}
                require(input_record.get("network_index"), local_fighter,
                        "move request network index")
                move_index = input_record.get("pending_move_index")
                if move_index not in MOVE_NAMES:
                    raise WindowError(f"request names unmapped move index {move_index!r}")
                action_tick = int(record.get("client_fixed_tick_at_observation"))
                before = [
                    _relative(sample, action_tick, fixed_delta)
                    for sample in recent
                    if (action_tick - int(sample["tick"])) % int(stride_ticks) == 0
                ]
                windows.append({
                    "window_index": len(windows),
                    "semantic_role": "client_request_aligned_candidate_response",
                    "accepted": {"state": "unknown"},
                    "executed": {"state": "unknown"},
                    "request": {
                        "tick": action_tick,
                        "unity_frame": record.get("unity_frame"),
                        "unity_unscaled_time": record.get("unity_unscaled_time"),
                        "client_transport_invocation_sequence": record.get(
                            "client_transport_invocation_sequence"
                        ),
                        "method_invocation_sequence": record.get(
                            "method_invocation_sequence"
                        ),
                        "network_index": input_record.get("network_index"),
                        "move_index": move_index,
                        "move_name": MOVE_NAMES[move_index],
                        "velocity_command": input_record.get("velocity_command"),
                        "punching": input_record.get("punching"),
                        "recovering": input_record.get("recovering"),
                        "provenance": record.get("provenance"),
                    },
                    "samples": before,
                })
            elif event == "capture_end":
                if end is not None:
                    raise WindowError("multiple capture_end records")
                end = record

    if start is None or end is None:
        raise WindowError("raw trace is incomplete")
    require(end.get("sample_count"), sample_count, "capture_end sample_count")
    require(end.get("capture_error_count"), 0, "capture error count")
    declared_moves = (end.get("client_transport_method_counts") or {}).get(
        "SendMoveEvent", 0
    )
    require(declared_moves, len(windows), "capture_end move request count")

    for window in windows:
        samples = window["samples"]
        window["sample_count"] = len(samples)
        window["first_relative_time_s"] = (
            samples[0]["relative_time_s"] if samples else None
        )
        window["last_relative_time_s"] = (
            samples[-1]["relative_time_s"] if samples else None
        )
        action_tick = int(window["request"]["tick"])
        first_tick = int(samples[0]["tick"]) if samples else None
        last_tick = int(samples[-1]["tick"]) if samples else None
        pre_complete = (
            first_tick is not None and first_tick <= action_tick - int(pre_ticks)
        )
        post_complete = (
            last_tick is not None and last_tick >= action_tick + int(post_ticks)
        )
        window["coverage"] = {
            "pre_complete": pre_complete,
            "post_complete": post_complete,
            "left_censored": not pre_complete,
            "right_censored": not post_complete,
        }

    document = {
        "schema": SCHEMA,
        "semantic_limit": (
            "SendMoveEvent prefix observation proves a client request only; "
            "server acceptance and move execution remain unknown. Samples are "
            "2 ms client FixedUpdate observations of remote-authoritative visual "
            "state, not server ticks or isolated canned-move payloads; repeated "
            "snapshot sequences identify samples between approximately 10 Hz "
            "fight-state snapshots, where client visual interpolation is possible"
        ),
        "source": {
            "path": str(raw_path.resolve()),
            "sha256": digest.hexdigest(),
            "bytes": raw_path.stat().st_size,
            "game_assembly_sha256": start["game_assembly_sha256"],
            "global_metadata_sha256": start["global_metadata_sha256"],
            "plugin_sha256": start.get("plugin_sha256"),
            "server_tick_available": start.get("server_tick_available"),
        },
        "capture": {
            "fighter": fighter,
            "opponent": opponent_fighter,
            "sparring_bot_number": (start.get("scope") or {}).get("sparring_bot_number"),
            "fixed_delta_time": fixed_delta,
            "raw_sample_count": sample_count,
            "move_request_count": len(windows),
            "end_reason": end.get("reason"),
            "fighter_bones": start.get(f"fighter_{fighter}_bones"),
            "opponent_bones": start.get(f"fighter_{opponent_fighter}_bones"),
        },
        "window": {
            "pre_ms": pre_ms,
            "post_ms": post_ms,
            "stride_ms": stride_ms,
            "pre_ticks": pre_ticks,
            "post_ticks": post_ticks,
            "stride_ticks": stride_ticks,
        },
        "move_request_counts": [
            {"move_index": move_index, "move_name": MOVE_NAMES[move_index],
             "count": sum(item["request"]["move_index"] == move_index for item in windows)}
            for move_index in sorted({item["request"]["move_index"] for item in windows})
        ],
        "windows": windows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary path {temporary}")
    try:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return document


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fighter", type=int, choices=(0, 1), default=None,
        help="optional assertion; command issuer is derived from capture scope",
    )
    parser.add_argument("--pre-ms", type=float, default=200.0)
    parser.add_argument("--post-ms", type=float, default=2200.0)
    parser.add_argument("--stride-ms", type=float, default=20.0)
    args = parser.parse_args()
    result = extract(
        args.raw, args.out, fighter=args.fighter, pre_ms=args.pre_ms,
        post_ms=args.post_ms, stride_ms=args.stride_ms,
    )
    print(f"move_request_count: {result['capture']['move_request_count']}")
    print(f"move_request_counts: {json.dumps(result['move_request_counts'], sort_keys=True)}")
    print("acceptance_and_execution: unknown")
    print(f"output: {args.out.resolve()}")
    print(f"sha256: {sha256_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
