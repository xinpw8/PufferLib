#!/usr/bin/env python3
"""Focused contract tests for the CUDA GEAR-SONIC controller wrapper."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
import tempfile
import unittest

import gpu_controller


class ManifestContractTests(unittest.TestCase):
    def _manifest(self, report: object) -> Path:
        directory = tempfile.TemporaryDirectory(prefix="gpu-controller-manifest-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / gpu_controller.MANIFEST_FILENAME
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_accepts_positive_explicit_batch_without_rek_claim(self) -> None:
        path = self._manifest(
            {
                "schema": gpu_controller.MANIFEST_SCHEMA,
                "batch_size": 16,
                "rek_parity_claim": False,
            }
        )
        self.assertEqual(gpu_controller._batch_size_from_manifest(path), 16)

    def test_rejects_rek_parity_claim(self) -> None:
        path = self._manifest(
            {
                "schema": gpu_controller.MANIFEST_SCHEMA,
                "batch_size": 8,
                "rek_parity_claim": True,
            }
        )
        with self.assertRaisesRegex(
            gpu_controller.GpuControllerError, "must not claim REK parity"
        ):
            gpu_controller._batch_size_from_manifest(path)

    def test_capture_path_has_no_host_tensor_extraction(self) -> None:
        source = "\n".join(
            (
                inspect.getsource(gpu_controller._PinnedG1Encoder.forward),
                inspect.getsource(gpu_controller._PinnedG1Decoder.forward),
                inspect.getsource(gpu_controller.CapturedGearSonicController.replay),
            )
        )
        for forbidden in (".cpu(", ".item(", ".tolist(", ".numpy("):
            self.assertNotIn(forbidden, source)

    def test_rejects_boolean_batch_size(self) -> None:
        path = self._manifest(
            {
                "schema": gpu_controller.MANIFEST_SCHEMA,
                "batch_size": True,
                "rek_parity_claim": False,
            }
        )
        with self.assertRaisesRegex(
            gpu_controller.GpuControllerError, "must be an integer"
        ):
            gpu_controller._batch_size_from_manifest(path)


if __name__ == "__main__":
    unittest.main()
