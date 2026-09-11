from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import generate_g1_model_identity_header as target


class ModelIdentityHeaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "source"
        self.batch = self.root / "batch"
        self.sources.mkdir()
        self.batch.mkdir()
        self.encoder_source = b"source-encoder\n"
        self.decoder_source = b"source-decoder\n"
        self.encoder_output = b"batch-encoder\n"
        self.decoder_output = b"batch-decoder\n"
        (self.sources / "model_encoder.onnx").write_bytes(self.encoder_source)
        (self.sources / "model_decoder.onnx").write_bytes(self.decoder_source)
        (self.batch / "model_encoder.batch8.onnx").write_bytes(self.encoder_output)
        (self.batch / "model_decoder.batch8.onnx").write_bytes(self.decoder_output)
        self.manifest = self.batch / target.MANIFEST_FILENAME
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": target.SCHEMA,
                    "classification": target.CLASSIFICATION,
                    "rek_parity_claim": False,
                    "batch_size": 8,
                    "models": {},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        self.encoder_source_hash = hashlib.sha256(self.encoder_source).hexdigest()
        self.decoder_source_hash = hashlib.sha256(self.decoder_source).hexdigest()
        self.encoder_output_hash = hashlib.sha256(self.encoder_output).hexdigest()
        self.decoder_output_hash = hashlib.sha256(self.decoder_output).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def inspection(self) -> dict[str, object]:
        return {
            "manifest_path": str(self.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "models": {
                "model_encoder.onnx": {
                    "source_sha256": self.encoder_source_hash,
                    "output_sha256": self.encoder_output_hash,
                },
                "model_decoder.onnx": {
                    "source_sha256": self.decoder_source_hash,
                    "output_sha256": self.decoder_output_hash,
                },
            },
        }

    def spec_patch(self) -> mock._patch_dict:
        return mock.patch.dict(
            target.MODEL_SPECS,
            {
                "model_encoder.onnx": {"sha256": self.encoder_source_hash},
                "model_decoder.onnx": {"sha256": self.decoder_source_hash},
            },
            clear=True,
        )

    def equivalence(self, filename: str) -> dict[str, object]:
        tensor_hash = hashlib.sha256(f"tensor:{filename}".encode("ascii")).hexdigest()
        return {
            "providers": list(target.BUILD_GATE_EQUIVALENCE_PROVIDERS),
            "seed": target.BUILD_GATE_EQUIVALENCE_SEED,
            "probe_contract": (
                "g1_mode_zero_with_only_active_encoder_channels_nonzero"
                if filename == "model_encoder.onnx"
                else "finite_distinct_decoder_rows"
            ),
            "absolute_tolerance": target.BUILD_GATE_EQUIVALENCE_ATOL,
            "max_absolute_difference": 0.0,
            "minimum_pairwise_output_linf": 0.125,
            "exact_float32_equal": True,
            "expected_sha256_float32_le": tensor_hash,
            "actual_sha256_float32_le": tensor_hash,
        }

    def rewrite_matching_output(
        self,
        source: Path,
        output: Path,
        batch_size: int,
    ) -> dict[str, object]:
        self.assertEqual(batch_size, 8)
        supplied = self.batch / output.name
        output.write_bytes(supplied.read_bytes())
        return {
            "source_path": str(source),
            "output_path": str(output),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }

    def verify_matching_output(
        self,
        source: Path,
        output: Path,
        batch_size: int,
        *,
        providers: tuple[str, ...],
        seed: int,
        atol: float,
    ) -> dict[str, object]:
        self.assertEqual(output.parent, self.batch.resolve())
        self.assertEqual(batch_size, 8)
        self.assertEqual(providers, target.BUILD_GATE_EQUIVALENCE_PROVIDERS)
        self.assertEqual(seed, target.BUILD_GATE_EQUIVALENCE_SEED)
        self.assertEqual(atol, target.BUILD_GATE_EQUIVALENCE_ATOL)
        return self.equivalence(source.name)

    def test_validated_record_generates_path_free_header(self) -> None:
        output = self.root / "build" / "g1_model_identity_generated.h"
        with self.spec_patch(), mock.patch.object(
            target,
            "inspect_explicit_batch_bundle",
            return_value=self.inspection(),
        ) as inspect, mock.patch.object(
            target,
            "rewrite_model",
            side_effect=self.rewrite_matching_output,
        ) as rewrite, mock.patch.object(
            target,
            "verify_equivalence",
            side_effect=self.verify_matching_output,
        ) as verify:
            record = target.generate_header(self.sources, self.manifest, output)
        inspect.assert_called_once_with(self.sources.resolve(), self.batch.resolve(), 8)
        self.assertEqual(rewrite.call_count, 2)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(record["batch_size"], 8)
        self.assertFalse(record["rek_parity_claim"])
        self.assertFalse(record["current_steam_authority"])
        self.assertFalse(record["training_enabled"])
        self.assertEqual(
            record["encoder"]["independent_rewrite_sha256"],
            self.encoder_output_hash,
        )
        self.assertTrue(record["encoder"]["equivalence"]["exact_float32_equal"])
        header = output.read_text(encoding="ascii")
        self.assertIn(self.encoder_source_hash, header)
        self.assertIn(self.encoder_output_hash, header)
        self.assertIn(str(len(self.encoder_output)), header)
        self.assertNotIn(str(self.root), header)
        self.assertNotIn(".onnx", header)

    def test_manifest_claims_cannot_authorize_an_arbitrary_output_graph(self) -> None:
        output = self.root / "rejected.h"

        def rewrite_different_output(
            source: Path,
            rewritten: Path,
            batch_size: int,
        ) -> dict[str, object]:
            del source, batch_size
            rewritten.write_bytes(b"arbitrary-shape-compatible-output\n")
            return {
                "output_sha256": hashlib.sha256(rewritten.read_bytes()).hexdigest()
            }

        with self.spec_patch(), mock.patch.object(
            target,
            "inspect_explicit_batch_bundle",
            return_value=self.inspection(),
        ), mock.patch.object(
            target,
            "rewrite_model",
            side_effect=rewrite_different_output,
        ), mock.patch.object(target, "verify_equivalence") as verify:
            with self.assertRaisesRegex(
                target.ModelIdentityBuildError,
                "differs from the independent dimension-only rewrite",
            ):
                target.generate_header(self.sources, self.manifest, output)
        verify.assert_not_called()
        self.assertFalse(output.exists())

    def test_manifest_claims_cannot_bypass_live_inference(self) -> None:
        output = self.root / "rejected.h"
        with self.spec_patch(), mock.patch.object(
            target,
            "inspect_explicit_batch_bundle",
            return_value=self.inspection(),
        ), mock.patch.object(
            target,
            "rewrite_model",
            side_effect=self.rewrite_matching_output,
        ), mock.patch.object(
            target,
            "verify_equivalence",
            side_effect=target.BatchRewriteError("injected runtime mismatch"),
        ):
            with self.assertRaisesRegex(
                target.ModelIdentityBuildError,
                "independent model gate failed: injected runtime mismatch",
            ):
                target.generate_header(self.sources, self.manifest, output)
        self.assertFalse(output.exists())

    def test_nonexact_live_equivalence_record_is_rejected(self) -> None:
        output = self.root / "rejected.h"
        nonexact = self.equivalence("model_encoder.onnx")
        nonexact["exact_float32_equal"] = False
        with self.spec_patch(), mock.patch.object(
            target,
            "inspect_explicit_batch_bundle",
            return_value=self.inspection(),
        ), mock.patch.object(
            target,
            "rewrite_model",
            side_effect=self.rewrite_matching_output,
        ), mock.patch.object(
            target,
            "verify_equivalence",
            return_value=nonexact,
        ):
            with self.assertRaisesRegex(
                target.ModelIdentityBuildError,
                "equivalence is not exact float32",
            ):
                target.generate_header(self.sources, self.manifest, output)
        self.assertFalse(output.exists())

    def test_validator_source_disagreement_is_rejected(self) -> None:
        inspection = self.inspection()
        inspection["models"]["model_encoder.onnx"]["source_sha256"] = "0" * 64
        with self.spec_patch(), mock.patch.object(
            target,
            "inspect_explicit_batch_bundle",
            return_value=inspection,
        ):
            with self.assertRaisesRegex(
                target.ModelIdentityBuildError,
                "validated source hash changed",
            ):
                target.generate_header(
                    self.sources,
                    self.manifest,
                    self.root / "rejected.h",
                )

    def test_odd_robot_batch_is_rejected_before_inspection(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["batch_size"] = 7
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        with mock.patch.object(target, "inspect_explicit_batch_bundle") as inspect:
            with self.assertRaisesRegex(
                target.ModelIdentityBuildError,
                "positive fighter pair count",
            ):
                target.generate_header(
                    self.sources,
                    self.manifest,
                    self.root / "rejected.h",
                )
        inspect.assert_not_called()

    def test_symlink_manifest_is_rejected(self) -> None:
        link = self.root / target.MANIFEST_FILENAME
        try:
            link.symlink_to(self.manifest)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        with self.assertRaisesRegex(
            target.ModelIdentityBuildError,
            "regular, non-symlink file",
        ):
            target.generate_header(self.sources, link, self.root / "rejected.h")


if __name__ == "__main__":
    unittest.main()
