"""Summarize executed solver diagnostics without pooling different workloads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from summarize_training_comparison import aggregate, comparable


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(root):
    runs = []
    for path in sorted(root.glob("puffysics-*/report.json")):
        if not any(part in path.parent.name for part in ("standard-", "capsule-", "independent-")):
            continue
        report = json.loads(path.read_text())
        failure = report.get("physics_failure_evidence", {})
        status_counts = Counter(row[2] for row in failure.get("per_arena", []) if row[2])
        runs.append({
            "label": path.parent.name, "report": identity(path),
            "status": report["status"], "training": report.get("training"),
            "timing_contract": report["timing_contract"],
            "warmup_updates": report["arguments"]["warmup_updates"],
            "measured_updates": report["measured"]["verified_updates"],
            "learner_transitions": report["measured"]["verified_learner_transitions"],
            "training_sps": report["measured"]["training_learner_transitions_per_second"],
            "training_wall_seconds": report["measured"]["timing"]["wall_seconds"],
            "warmup_wall_seconds": report["warmup"]["timing"]["wall_seconds"],
            "weights_changed": report["checkpoints"].get("measured_weights_changed"),
            "failure_stages": [error["stage"] for error in report["errors"]],
            "physics_failure": {key: value for key, value in failure.items() if key != "per_arena"},
            "nonzero_solver_status_counts": dict(status_counts),
            "initial_checkpoint_sha256": report["checkpoints"]["initial"]["sha256"],
            "benchmark_source": report["sources"]["benchmark"],
        })
    traces = []
    for path in sorted(root.glob("puffysics-*-nsys-summary.json")):
        if not any(part in path.name for part in ("standard-", "capsule-", "independent-")):
            continue
        trace = json.loads(path.read_text())
        traces.append({"report": identity(path), "sum_kernel_ms": trace["sum_kernel_ms"],
                       "kernel_calls": trace["kernel_calls"], "top_kernels": trace["kernels"][:5]})
    groups = {}
    for name, pattern in (
        ("independent_capsule_512_h16", "puffysics-independent-capsule-h16-r*/report.json"),
        ("independent_capsule_4096_h16", "puffysics-independent-capsule-4096-h16-v*/report.json"),
        ("standard_cold_512_h8", "puffysics-standard-cold-h8-v2-r*/report.json"),
    ):
        paths = sorted(root.glob(pattern))
        if len(paths) != 3:
            raise ValueError("expected three repetitions for " + name)
        reports = [json.loads(path.read_text()) for path in paths]
        if any(comparable(report) != comparable(reports[0]) for report in reports):
            raise ValueError("repetition inputs differ for " + name)
        groups[name] = aggregate(paths)
        groups[name]["measurement_regime"] = reports[0]["timing_contract"]["measurement_regime"]
    return {"schema": "rek.puffysics_solver_diagnostics.v1", "runs": runs, "traces": traces, "groups": groups,
            "limits": ["Cold-start eight-tick runs are not sustained throughput.",
                       "Capsule substitutions change geometry and make no parity claim.",
                       "Mode 0 selects b3_step in the modified REK fork; it does not imply unmodified upstream.",
                       "Failure flags are never cleared to inflate SPS; failed runs retain null SPS."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{key: run[key] for key in ("label", "status", "training_sps")} for run in result["runs"]], indent=2))
