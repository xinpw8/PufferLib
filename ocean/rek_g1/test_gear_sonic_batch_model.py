import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gear_sonic_batch_model as batch_model


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ExplicitBatchBundleInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "source"
        self.output = root / "output"
        self.source.mkdir()
        self.output.mkdir()
        self.batch_size = 8
        self.specs = {}
        models = {}

        for filename, original_spec in batch_model.MODEL_SPECS.items():
            source_payload = f"source:{filename}".encode("ascii")
            output_payload = f"batch:{self.batch_size}:{filename}".encode("ascii")
            (self.source / filename).write_bytes(source_payload)
            output_file = f"{Path(filename).stem}.batch{self.batch_size}.onnx"
            (self.output / output_file).write_bytes(output_payload)
            spec = dict(original_spec)
            spec["sha256"] = _sha256(source_payload)
            self.specs[filename] = spec
            tensor_hash = _sha256(f"tensor:{filename}".encode("ascii"))
            models[filename] = {
                "source_path": "/ignored/source/path",
                "source_sha256": spec["sha256"],
                "output_path": "/ignored/output/path",
                "output_file": output_file,
                "output_sha256": _sha256(output_payload),
                "mutations": [],
                "equivalence": {
                    "providers": ["CPUExecutionProvider"],
                    "seed": 20260908,
                    "probe_contract": (
                        "g1_mode_zero_with_only_active_encoder_channels_nonzero"
                        if filename == "model_encoder.onnx"
                        else "finite_distinct_decoder_rows"
                    ),
                    "absolute_tolerance": 0.0,
                    "max_absolute_difference": 0.0,
                    "minimum_pairwise_output_linf": 0.125,
                    "exact_float32_equal": True,
                    "expected_sha256_float32_le": tensor_hash,
                    "actual_sha256_float32_le": tensor_hash,
                },
            }

        self.manifest = {
            "schema": batch_model.SCHEMA,
            "classification": "public_family_candidate",
            "rek_parity_claim": False,
            "batch_size": self.batch_size,
            "models": models,
            "limits": [],
        }
        self._write_manifest()
        patcher = mock.patch.object(batch_model, "MODEL_SPECS", self.specs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_manifest(self) -> None:
        (self.output / batch_model.MANIFEST_FILENAME).write_text(
            json.dumps(self.manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_accepts_exact_manifest_and_ignores_recorded_absolute_paths(self) -> None:
        result = batch_model.inspect_explicit_batch_bundle(
            self.source, self.output, self.batch_size
        )
        self.assertEqual(result["batch_size"], self.batch_size)
        for filename, record in result["models"].items():
            self.assertEqual(
                Path(record["output_path"]).parent,
                self.output.resolve(),
            )
            self.assertEqual(record["source_sha256"], self.specs[filename]["sha256"])

    def test_rejects_tampered_output_bytes(self) -> None:
        output_file = self.manifest["models"]["model_decoder.onnx"]["output_file"]
        (self.output / output_file).write_bytes(b"tampered")
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "output identity"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size
            )

    def test_rejects_direct_symlink_model_path(self) -> None:
        source = self.source / "model_encoder.onnx"
        backing = self.source / "encoder-backing.onnx"
        source.replace(backing)
        try:
            source.symlink_to(backing)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "non-symlink file"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size
            )

    def test_rejects_nonzero_equivalence_tolerance(self) -> None:
        self.manifest["models"]["model_encoder.onnx"]["equivalence"][
            "absolute_tolerance"
        ] = 1e-6
        self._write_manifest()
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "tolerance"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size
            )

    def test_rejects_equivalence_probe_without_output_diversity(self) -> None:
        self.manifest["models"]["model_encoder.onnx"]["equivalence"][
            "minimum_pairwise_output_linf"
        ] = 0.0
        self._write_manifest()
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "not distinct"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size
            )

    def test_rejects_missing_equivalence_tensor_hashes(self) -> None:
        equivalence = self.manifest["models"]["model_encoder.onnx"]["equivalence"]
        equivalence["expected_sha256_float32_le"] = None
        equivalence["actual_sha256_float32_le"] = None
        self._write_manifest()
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "hash is invalid"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size
            )

    def test_rejects_bitwise_mismatch_with_zero_numeric_difference(self) -> None:
        equivalence = self.manifest["models"]["model_encoder.onnx"]["equivalence"]
        equivalence["actual_sha256_float32_le"] = "0" * 64
        self._write_manifest()
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "hashes differ"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size
            )

    def test_rejects_missing_equivalence_execution_identity(self) -> None:
        equivalence = self.manifest["models"]["model_decoder.onnx"]["equivalence"]
        equivalence["providers"] = []
        self._write_manifest()
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "providers are invalid"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size
            )

    def test_rejects_environment_count_mismatch(self) -> None:
        with self.assertRaisesRegex(batch_model.BatchRewriteError, "size differs"):
            batch_model.inspect_explicit_batch_bundle(
                self.source, self.output, self.batch_size + 1
            )


if __name__ == "__main__":
    unittest.main()
