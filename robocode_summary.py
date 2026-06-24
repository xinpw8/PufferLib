#!/usr/bin/env python3
"""Print scripted-bot eval perf from eval_robocode.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print checkpoint scripted-bot eval perf from a robocode eval JSON."
    )
    parser.add_argument("json_path", type=Path)
    return parser.parse_args()


def eval_perf(record: dict[str, Any]) -> Any:
    eval_result = record.get("eval") or {}
    scores = eval_result.get("scores") or {}
    for key in ("Perf", "perf", "env/perf"):
        if key in scores:
            return scores[key]

    stdout = eval_result.get("stdout") or ""
    for line in stdout.replace("\r", "\n").splitlines():
        if line.strip().lower().startswith("perf:"):
            return line.split(":", 1)[1].strip()
    return None


def load_log(data: dict[str, Any], json_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    log_path_value = data.get("log_path")
    if not log_path_value:
        return {}, {}

    log_path = Path(log_path_value)
    if not log_path.is_absolute():
        log_path = (json_path.parent / log_path).resolve()

    try:
        with log_path.open() as f:
            log_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}, {}

    metrics = log_data.get("metrics", {})
    return log_data, metrics if isinstance(metrics, dict) else {}


def step_scale(data: dict[str, Any], log_data: dict[str, Any]) -> int:
    for source in (data, log_data):
        value = source.get("step_scale") or source.get("world_size")
        try:
            scale = int(value)
        except (TypeError, ValueError):
            continue
        if scale > 0:
            return scale
    return 1


def nearest_log_index(metrics: dict[str, Any], agent_step: int | float | None) -> int | None:
    steps = metrics.get("agent_steps")
    if agent_step is None or not isinstance(steps, list) or not steps:
        return None

    def distance(index: int) -> float:
        try:
            return abs(float(steps[index]) - float(agent_step))
        except (TypeError, ValueError):
            return float("inf")

    index = min(range(len(steps)), key=distance)
    return None if distance(index) == float("inf") else index


def metric_value_at(metrics: dict[str, Any], key: str, index: int | None) -> Any:
    value = metrics.get(key)
    if index is None:
        return None
    if isinstance(value, list) and index < len(value):
        return value[index]
    return None


def scaled_agent_steps(record: dict[str, Any], scale: int) -> int | None:
    value = record.get("checkpoint_agent_steps")
    if value is None:
        value = record.get("checkpoint_step")
        if value is None:
            return None
        value = value * scale

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    args = parse_args()
    with args.json_path.open() as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        print("No records found")
        return 1

    log_data, log_metrics = load_log(data, args.json_path)
    scale = step_scale(data, log_data)

    print(f"{'ckpt_step':>16}  {'agent_steps':>16}  {'nearest_log_step':>16}  {'bot_perf':>12}")
    print(f"{'-' * 16}  {'-' * 16}  {'-' * 16}  {'-' * 12}")
    for record in records:
        ckpt_step = record.get("checkpoint_step")
        agent_steps = scaled_agent_steps(record, scale)
        log_index = nearest_log_index(log_metrics, agent_steps)
        log_agent_steps = metric_value_at(log_metrics, "agent_steps", log_index)
        if log_agent_steps is None:
            log_agent_steps = record.get("log_agent_steps")
        perf = eval_perf(record)

        ckpt_s = format_int(ckpt_step)
        agent_s = format_int(agent_steps)
        log_step_s = format_int(log_agent_steps)
        perf_s = "" if perf is None else str(perf)
        print(f"{ckpt_s:>16}  {agent_s:>16}  {log_step_s:>16}  {perf_s:>12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
