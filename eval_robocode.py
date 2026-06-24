#!/usr/bin/env python3
"""Sample robocode checkpoints for a run and evaluate against the scripted bot.

Example:
    python eval_robocode.py 1782158510638
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
METRIC_KEY_TOKENS = ("score", "winrate", "draw", "perf")
METRIC_RE = re.compile(
    rf"\b(?P<key>[A-Za-z][A-Za-z0-9_./-]*)\s*(?:=|:|\s)\s*(?P<value>{FLOAT_RE})",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate evenly sampled robocode checkpoints against the scripted bot."
    )
    parser.add_argument(
        "run",
        help="Run id, or path to checkpoints/robocode/<run_id>.",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to logs/robocode/<run_id>_eval_samples.json.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run pufferlib.pufferl.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for each eval command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve samples and log values without running eval.",
    )
    parser.add_argument(
        "--bot-policy",
        type=int,
        default=None,
        help="Optional env.bot_policy override for the scripted bot.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to each pufferl eval command.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_run(run: str, checkpoint_dir: Path, log_dir: Path) -> tuple[str, Path, Path]:
    run_path = Path(run)
    if run_path.exists():
        checkpoint_run_dir = run_path.resolve()
        run_id = checkpoint_run_dir.name
    else:
        run_id = run
        checkpoint_run_dir = (checkpoint_dir / "robocode" / run_id).resolve()

    log_path = (log_dir / "robocode" / f"{run_id}.json").resolve()
    return run_id, checkpoint_run_dir, log_path


def checkpoint_step(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def sorted_checkpoints(checkpoint_run_dir: Path) -> list[Path]:
    checkpoints = [p for p in checkpoint_run_dir.glob("*.bin") if p.is_file()]
    return sorted(
        checkpoints,
        key=lambda p: (
            checkpoint_step(p) is None,
            checkpoint_step(p) if checkpoint_step(p) is not None else p.stat().st_mtime,
            p.name,
        ),
    )


def evenly_sample(items: list[Path], samples: int) -> list[Path]:
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if len(items) <= samples:
        return items
    if samples == 1:
        return [items[0]]

    last = len(items) - 1
    indices = [round(i * last / (samples - 1)) for i in range(samples)]
    return [items[i] for i in indices]


def max_logged_agent_step(metrics: dict[str, Any]) -> float | None:
    steps = metrics.get("agent_steps")
    if not isinstance(steps, list) or not steps:
        return None
    values = []
    for step in steps:
        try:
            values.append(float(step))
        except (TypeError, ValueError):
            pass
    return max(values) if values else None


def checkpoint_agent_step(path: Path, step_scale: int) -> int | None:
    step = checkpoint_step(path)
    return step * step_scale if step is not None else None


def filter_logged_checkpoints(
    checkpoints: list[Path], step_scale: int, max_agent_step: float | None
) -> tuple[list[Path], int]:
    if max_agent_step is None:
        return checkpoints, 0

    kept = []
    ignored = 0
    for checkpoint in checkpoints:
        agent_step = checkpoint_agent_step(checkpoint, step_scale)
        if agent_step is not None and agent_step > max_agent_step:
            ignored += 1
        else:
            kept.append(checkpoint)
    return kept or checkpoints, ignored


def load_log(log_path: Path) -> dict[str, Any]:
    with log_path.open() as f:
        data = json.load(f)
    return data


def log_step_scale(log_data: dict[str, Any]) -> int:
    world_size = log_data.get("world_size", 1)
    try:
        scale = int(world_size)
    except (TypeError, ValueError):
        return 1
    return max(scale, 1)


def metric_value_at(metrics: dict[str, Any], key: str, index: int | None) -> Any:
    value = metrics.get(key)
    if index is None:
        return None
    if isinstance(value, list) and index < len(value):
        return value[index]
    return None


def nearest_log_index(metrics: dict[str, Any], step: int | None) -> int | None:
    steps = metrics.get("agent_steps")
    if step is None or not isinstance(steps, list) or not steps:
        return None

    def distance(index: int) -> float:
        try:
            return abs(float(steps[index]) - float(step))
        except (TypeError, ValueError):
            return float("inf")

    index = min(range(len(steps)), key=distance)
    return None if distance(index) == float("inf") else index


def clean_output(output: str) -> str:
    return ANSI_RE.sub("", output).replace("\r", "\n")


def parse_eval_scores(output: str) -> dict[str, Any]:
    clean = clean_output(output)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    score_lines = [
        line
        for line in lines
        if any(token in line.lower() for token in METRIC_KEY_TOKENS)
    ]
    metrics: dict[str, float] = {}
    for match in METRIC_RE.finditer(clean):
        key = match.group("key")
        if not any(token in key.lower() for token in METRIC_KEY_TOKENS):
            continue
        try:
            metrics[key] = float(match.group("value"))
        except ValueError:
            pass

    return {
        "metrics": metrics,
        "score_lines": score_lines,
    }


def eval_command(
    python: str,
    checkpoint: Path,
    bot_policy: int | None,
    extra_args: list[str],
) -> list[str]:
    command = [
        python,
        "-m",
        "pufferlib.pufferl",
        "eval",
        "robocode",
        "--load-model-path",
        str(checkpoint),
        "--selfplay.enabled",
        "0",
        "--env.num-agents",
        "1",
        "--env.num-bots",
        "1",
    ]
    if bot_policy is not None:
        command.extend(["--env.bot-policy", str(bot_policy)])
    command.extend(extra_args)
    return command


def run_eval(command: list[str], cwd: Path, timeout: float | None) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    parsed = parse_eval_scores(proc.stdout)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "scores": parsed["metrics"],
        "score_lines": parsed["score_lines"],
    }


def build_record(
    checkpoint: Path,
    log_metrics: dict[str, Any],
    step_scale: int,
    command: list[str],
    cwd: Path,
    timeout: float | None,
    dry_run: bool,
) -> dict[str, Any]:
    step = checkpoint_step(checkpoint)
    agent_step = step * step_scale if step is not None else None
    log_index = nearest_log_index(log_metrics, agent_step)
    record: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": step,
        "checkpoint_agent_steps": agent_step,
        "step_scale": step_scale,
        "log_index": log_index,
        "log_agent_steps": metric_value_at(log_metrics, "agent_steps", log_index),
        "eval_command": command,
    }

    if dry_run:
        record["eval"] = None
    else:
        record["eval"] = run_eval(command, cwd=cwd, timeout=timeout)
    return record


def main() -> int:
    args = parse_args()
    root = repo_root()
    checkpoint_dir = (root / args.checkpoint_dir).resolve()
    log_dir = (root / args.log_dir).resolve()
    run_id, checkpoint_run_dir, log_path = resolve_run(args.run, checkpoint_dir, log_dir)

    if not checkpoint_run_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint run directory: {checkpoint_run_dir}")
    checkpoints = sorted_checkpoints(checkpoint_run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No top-level .bin checkpoints in {checkpoint_run_dir}")

    if log_path.is_file():
        log_data = load_log(log_path)
        log_metrics = log_data.get("metrics", {})
        if not isinstance(log_metrics, dict):
            raise ValueError(f"Log has no metrics object: {log_path}")
    else:
        log_data = {}
        log_metrics = {}
    step_scale = log_step_scale(log_data)
    max_agent_step = max_logged_agent_step(log_metrics)
    checkpoints, ignored_after_log = filter_logged_checkpoints(
        checkpoints, step_scale, max_agent_step)

    sampled = evenly_sample(checkpoints, args.samples)

    output_path = args.output
    if output_path is None:
        output_path = log_dir / "robocode" / f"{run_id}_eval_samples.json"
    elif not output_path.is_absolute():
        output_path = (root / output_path).resolve()

    records = []
    for checkpoint in sampled:
        command = eval_command(args.python, checkpoint, args.bot_policy, args.extra_args)
        print(f"Evaluating {checkpoint.name} vs scripted bot", flush=True)
        records.append(
            build_record(
                checkpoint=checkpoint,
                log_metrics=log_metrics,
                step_scale=step_scale,
                command=command,
                cwd=root,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
        )

    result = {
        "run_id": run_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "log_path": str(log_path),
        "requested_samples": args.samples,
        "selected_samples": len(sampled),
        "ignored_checkpoints_after_log": ignored_after_log,
        "step_scale": step_scale,
        "eval_target": {
            "type": "scripted_bot",
            "env.num_agents": 1,
            "env.num_bots": 1,
            "env.bot_policy": args.bot_policy,
        },
        "dry_run": args.dry_run,
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    failures = [
        record
        for record in records
        if record["eval"] is not None and record["eval"]["returncode"] != 0
    ]
    print(f"Wrote {output_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
