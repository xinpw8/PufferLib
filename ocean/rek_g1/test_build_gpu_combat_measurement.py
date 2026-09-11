"""CPU-only builder regression; mocks nvcc and never compiles or uses CUDA."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import build_gpu_combat_measurement as builder


class BuildMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.directory = Path(self.workspace.name)
        self.compiler = self.directory / "nvcc"
        self.compiler.write_bytes(b"mock compiler identity")
        self.output = self.directory / "explicit.so"
        self.calls = []

    def run_compiler(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "nvcc fixture", "")
        Path(command[command.index("-o")+1]).write_bytes(b"fixture library")
        return subprocess.CompletedProcess(command, 0, "compiler stdout", "compiler stderr")

    def build(self, **kwargs):
        with patch.object(builder, "_compiler", return_value=self.compiler), \
             patch.object(builder.subprocess, "run", side_effect=self.run_compiler):
            return builder.build(self.output, nvcc=self.compiler, **kwargs)

    def test_explicit_flags_output_manifest_and_hashes(self):
        result = self.build()
        self.assertEqual(result["status"], "built")
        self.assertFalse(result["gpu_execution"])
        self.assertEqual(result["architecture"], "sm_121")
        self.assertEqual(result["flags"], list(builder.FLAGS))
        self.assertEqual(result["command"][1:7], list(builder.FLAGS))
        self.assertEqual(result["compiler"]["sha256"], builder.sha256(self.compiler))
        self.assertEqual(result["source"]["sha256"], builder.sha256(builder.SOURCE))
        self.assertEqual(result["library"]["sha256"], builder.sha256(self.output))
        self.assertEqual(result["stdout"], "compiler stdout")
        self.assertEqual(result["stderr"], "compiler stderr")
        self.assertEqual(json.loads(Path(result["manifest_path"]).read_text()), result)
        self.assertEqual(len(self.calls), 2)
        self.assertNotIn("shell", self.calls[1][1])
        self.assertFalse(Path(result["command"][-1]).exists())

    def test_existing_library_is_preserved_without_compiler_call(self):
        self.output.write_bytes(b"existing user library")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.output.read_bytes(), b"existing user library")
        self.assertFalse(self.calls)

    def test_existing_manifest_is_preserved_without_compiler_call(self):
        manifest = self.directory / "report.json"
        manifest.write_text("existing record")
        with self.assertRaises(FileExistsError):
            self.build(manifest=manifest)
        self.assertEqual(manifest.read_text(), "existing record")
        self.assertFalse(self.calls)

    def test_compiler_failure_records_error_without_fallback(self):
        def failed(command, **kwargs):
            self.calls.append(command)
            return subprocess.CompletedProcess(command, 0 if command[-1] == "--version" else 7,
                                               "nvcc fixture", "explicit failure")
        with patch.object(builder, "_compiler", return_value=self.compiler), \
             patch.object(builder.subprocess, "run", side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                builder.build(self.output, nvcc=self.compiler)
        result = json.loads(self.output.with_name("explicit.so.build.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertIsNone(result["library"])
        self.assertFalse(self.output.exists())
        self.assertEqual(len(self.calls), 2)

    def test_success_without_library_is_rejected(self):
        with patch.object(builder, "_compiler", return_value=self.compiler), \
             patch.object(builder.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaisesRegex(RuntimeError, "without a nonempty library"):
                builder.build(self.output)
        self.assertFalse(self.output.exists())

    def test_non_library_output_and_overlapping_manifest_rejected(self):
        with self.assertRaises(ValueError):
            builder.build(self.directory / "artifact.txt")
        with self.assertRaises(ValueError):
            builder.build(self.output, manifest=self.output)


if __name__ == "__main__":
    unittest.main()
