"""CPU audit of the diagnostic on immutable closed physics traces.

These fixtures have physical qpos/qvel but no action masks. All-on masks and
active phase are synthetic eligibility assumptions, explicitly recorded below.
This checks plumbing/finite outputs, not attack accuracy or actual eligibility.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--wsl-distribution")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, identities, sources = [], [], []
    for trace in args.trace:
        meta_path, binary = trace / "response.json", trace / "response.f32"
        metadata = json.loads(meta_path.read_text())
        assert metadata["complete"] and metadata["dtype"] == "float32_little_endian"
        fields, stride = {}, 0
        for name, count in metadata["fields"]:
            fields[name] = (stride, count)
            stride += count * 4
        arenas, ticks = metadata["arenas"], metadata["ticks"]
        assert fields["post_qpos"][1] == 72 * arenas
        assert fields["post_qvel"][1] == 70 * arenas
        assert binary.stat().st_size == ticks * stride
        before = sha(binary)
        with binary.open("rb") as stream:
            for tick in range(ticks):
                frame = stream.read(stride)
                for arena in range(arenas):
                    p = struct.unpack_from("<72f", frame, fields["post_qpos"][0] + arena * 72 * 4)
                    v = struct.unpack_from("<70f", frame, fields["post_qvel"][0] + arena * 70 * 4)
                    inputs.append(dict(qpos=p, qvel=v, mask=[1] * 66, phase=2, terminal=0, failureBits=0))
                    identities.append(dict(trace=str(trace), tick=tick, arena=arena))
        assert sha(binary) == before
        sources.append(dict(path=str(binary), sha256=before, metadata_sha256=sha(meta_path),
                            ticks=ticks, arenas=arenas, bytes=binary.stat().st_size))
    text = "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n" for row in inputs)
    (args.output / "snapshots.jsonl").write_text(text)
    command = [args.executable]
    if args.wsl_distribution:
        command = ["wsl.exe", "-d", args.wsl_distribution, "--", *command]
    run = subprocess.run(command, input=text, text=True, capture_output=True, timeout=60)
    (args.output / "stderr.txt").write_text(run.stderr)
    (args.output / "predictions.jsonl").write_text(run.stdout)
    assert run.returncode == 0, run.stderr
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(rows) == len(inputs)
    outcomes, reasons = collections.Counter(), collections.Counter()
    for identity, row in zip(identities, rows):
        assert row["schema"] == "rek.will_connect.diagnostic.v1"
        assert row["calibration"] == "placeholder" and not row["officialScoringValidated"]
        assert row["guardMode"] == "unavailable" and len(row["fighters"]) == 2
        for side, fighter in enumerate(row["fighters"]):
            assert fighter["side"] == side and len(fighter["attacks"]) == 2
            for attack, category in zip(fighter["attacks"], (21, 24)):
                assert attack["action"] == category
                outcomes[attack["outcome"]] += 1
                if attack["reason"]:
                    reasons[attack["reason"]] += 1
                if not attack["available"]:
                    assert attack["light"] == "off" and attack["margin"] is None
                if attack["outcome"] == "hit":
                    assert attack["light"] == "green" and attack["contactTime"] >= 0
                    assert attack["relativeSpeed"] >= 0
    for index, identity in enumerate(identities):
        identity["prediction_index"] = index
    (args.output / "identities.json").write_text(json.dumps(identities, indent=2) + "\n")
    report = dict(schema="rek.will_connect.closed_trace_audit.v1", sources=sources,
                  command=command, exit_code=run.returncode, snapshots=len(rows),
                  attack_estimates=len(rows) * 4, outcomes=dict(outcomes), reasons=dict(reasons),
                  eligibility="synthetic all-on masks and active phase; not present in source trace",
                  guards="unavailable", official_scoring_validated=False,
                  physics_steps_executed=0, policy_updates=0,
                  files={p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("snapshots", "attack_estimates", "outcomes", "reasons")}))


if __name__ == "__main__":
    main()
