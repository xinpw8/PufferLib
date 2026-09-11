import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import gear_sonic_batch_model as batch_model


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _scalar_pairwise_output_linf(values: np.ndarray) -> float | None:
    distances = [
        float(np.max(np.abs(
            values[left].astype(np.float64) - values[right].astype(np.float64)
        )))
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return min(distances) if distances else None


class PairwiseOutputDiversityTests(unittest.TestCase):
    def assertExactReference(self, values: np.ndarray) -> None:
        expected = _scalar_pairwise_output_linf(values)
        for block_rows in (1, 2, 7, 64, 128, 256):
            with self.subTest(shape=values.shape, block_rows=block_rows):
                actual = batch_model._minimum_pairwise_output_linf(
                    values, block_rows=block_rows
                )
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertEqual(actual.hex(), expected.hex())

    def test_random_rows_match_scalar_float64_result_exactly(self) -> None:
        rng = np.random.default_rng(20260910)
        for shape in ((2, 1), (17, 29), (129, 64), (257, 29)):
            self.assertExactReference(rng.standard_normal(shape, dtype=np.float32))

    def test_noncontiguous_rows_match_reference(self) -> None:
        rng = np.random.default_rng(71)
        values = rng.standard_normal((262, 128), dtype=np.float32)[::2, ::2]
        self.assertFalse(values.flags.c_contiguous)
        self.assertExactReference(values)

    def test_duplicate_across_blocks_and_partial_final_block_is_detected(self) -> None:
        values = np.arange(257 * 29, dtype=np.float32).reshape(257, 29)
        values[-1] = values[0]
        self.assertExactReference(values)
        self.assertEqual(batch_model._minimum_pairwise_output_linf(values), 0.0)

    def test_closest_pair_on_block_boundary_or_final_block_is_visited(self) -> None:
        values = (10 * np.arange(259, dtype=np.float32))[:, None]
        for left, right in ((127, 128), (0, 258), (256, 258)):
            with self.subTest(left=left, right=right):
                fixture = values.copy()
                fixture[right] = fixture[left] + np.float32(0.125)
                self.assertExactReference(fixture)
                self.assertEqual(
                    batch_model._minimum_pairwise_output_linf(fixture), 0.125
                )

    def test_batch_one_and_empty_batch_have_no_pairs(self) -> None:
        for rows in (0, 1):
            self.assertExactReference(np.zeros((rows, 29), dtype=np.float32))

    def test_extreme_finite_values_are_promoted_before_subtraction(self) -> None:
        maximum = np.finfo(np.float32).max
        tiny = np.nextafter(np.float32(0), np.float32(1))
        values = np.asarray([
            [maximum, maximum, maximum],
            [-maximum, -maximum, -maximum],
            [np.nextafter(maximum, np.float32(0)), maximum, maximum],
            [tiny, -tiny, np.finfo(np.float32).tiny],
            [-tiny, tiny, -np.finfo(np.float32).tiny],
            [0.0, -0.0, 0.0],
        ], dtype=np.float32)
        with np.errstate(over="raise", invalid="raise"):
            self.assertExactReference(values)
            self.assertExactReference(values[:2])
        self.assertExactReference(np.asarray([[0.0], [-0.0]], dtype=np.float32))

    def test_invalid_block_size_is_rejected(self) -> None:
        values = np.zeros((2, 29), dtype=np.float32)
        for block_rows in (True, 0, -1, 1.5):
            with self.subTest(block_rows=block_rows), self.assertRaises(ValueError):
                batch_model._minimum_pairwise_output_linf(values, block_rows=block_rows)

    def test_invalid_output_rank_or_dtype_is_rejected(self) -> None:
        for values in (np.zeros(2, dtype=np.float32), np.zeros((2, 29), dtype=np.float64)):
            with self.assertRaises(ValueError):
                batch_model._minimum_pairwise_output_linf(values)

    def test_verify_equivalence_preserves_report_and_duplicate_rejection(self) -> None:
        output_width = int(batch_model.MODEL_SPECS["model_decoder.onnx"]["output_width"])
        actual_rows = []
        duplicate = False

        def run(_outputs, feeds):
            values = next(iter(feeds.values()))
            result = np.zeros((len(values), output_width), dtype=np.float32)
            if not duplicate:
                result[:] = values[:, :output_width]
            if len(values) > 1:
                actual_rows.append(result)
            return [result]

        session = mock.Mock()
        session.run.side_effect = run
        runtime = mock.Mock()
        runtime.InferenceSession.return_value = session
        arguments = dict(
            source=Path("model_decoder.onnx"), target=Path("batch.onnx"),
            batch_size=129, providers=["CPUExecutionProvider"], seed=71, atol=0.0,
        )
        with mock.patch.dict("sys.modules", {"onnxruntime": runtime}):
            report = batch_model.verify_equivalence(**arguments)
            self.assertEqual(
                report["minimum_pairwise_output_linf"].hex(),
                _scalar_pairwise_output_linf(actual_rows[-1]).hex(),
            )
            self.assertTrue(report["exact_float32_equal"])
            self.assertEqual(report["max_absolute_difference"], 0.0)
            self.assertEqual(
                report["expected_sha256_float32_le"],
                report["actual_sha256_float32_le"],
            )
            duplicate = True
            with self.assertRaisesRegex(batch_model.BatchRewriteError, "indistinguishable"):
                batch_model.verify_equivalence(**arguments)


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
