"""Explicit offline nvcc build for the opt-in fused contact measurement.

The source includes only C++ standard headers and cuda_runtime.h. No project
include directories, engine headers, game assets, or GPU execution are needed.
The runtime loader does not call this helper or compile code during stepping.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time


SOURCE = Path(__file__).with_name("gpu_combat_measurement_fused.cu")
FLAGS = ("-std=c++17", "-shared", "-Xcompiler=-fPIC", "-O2", "--fmad=false", "-arch=sm_121")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") \
            else hashlib.sha256(stream.read()).hexdigest()


def _compiler(value: str | Path) -> Path:
    found = shutil.which(str(value))
    if found is None:
        raise FileNotFoundError(f"nvcc was not found: {value}")
    return Path(found).resolve(strict=True)


def build(output: str | Path, *, nvcc: str | Path = "/usr/local/cuda/bin/nvcc",
          manifest: str | Path | None = None) -> dict:
    output = Path(output).resolve()
    if output.suffix != ".so":
        raise ValueError("output must explicitly name a .so library")
    manifest_path = Path(manifest).resolve() if manifest is not None \
        else output.with_name(output.name + ".build.json")
    source = SOURCE.resolve(strict=True)
    if manifest_path == output:
        raise ValueError("manifest and library paths must differ")
    for destination in (output, manifest_path):
        if destination.exists():
            raise FileExistsError(destination)
    compiler = _compiler(nvcc)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "rek.g1_fused_combat_build.v1",
        "status": "failed",
        "host": platform.node(),
        "host_architecture": platform.machine(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": sha256(source)},
        "compiler": {"path": str(compiler), "sha256": sha256(compiler)},
        "flags": list(FLAGS),
        "architecture": "sm_121",
        "working_directory": str(source.parent),
        "requested_output": str(output),
        "manifest_path": str(manifest_path),
        "library": None,
        "gpu_execution": False,
        "dependencies": "C++ standard headers and CUDA toolkit headers/runtime; no project headers",
    }
    started = time.perf_counter()
    error = None
    try:
        version_command = [str(compiler), "--version"]
        version = subprocess.run(version_command, cwd=source.parent, capture_output=True, text=True)
        report["compiler"].update(version_command=version_command, version_exit_code=version.returncode,
                                  version_stdout=version.stdout, version_stderr=version.stderr)
        if version.returncode:
            raise RuntimeError(f"nvcc --version failed with exit code {version.returncode}")
        with tempfile.TemporaryDirectory(prefix=".rek-combat-build-", dir=output.parent) as staging:
            compiled = Path(staging) / output.name
            command = [str(compiler), *FLAGS, str(source), "-o", str(compiled)]
            report["command"] = command
            completed = subprocess.run(command, cwd=source.parent, capture_output=True, text=True)
            report.update(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"nvcc build failed with exit code {completed.returncode}")
            if not compiled.is_file() or compiled.stat().st_size == 0:
                raise RuntimeError("nvcc returned success without a nonempty library")
            if sha256(source) != report["source"]["sha256"]:
                raise RuntimeError("measurement source changed during compilation")
            # Exclusive creation preserves any pre-existing destination, also
            # if another process created it while the compiler was running.
            with compiled.open("rb") as src, output.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            report["library"] = {"path": str(output), "sha256": sha256(output),
                                  "bytes": output.stat().st_size}
            report["status"] = "built"
    except Exception as exc:
        error = exc
        report["error"] = str(exc)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if error is not None:
        raise RuntimeError(f"{error}; build record: {manifest_path}") from error
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new explicitly requested .so path")
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--manifest", type=Path, help="new JSON path; default OUTPUT.so.build.json")
    args = parser.parse_args()
    result = build(args.output, nvcc=args.nvcc, manifest=args.manifest)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
