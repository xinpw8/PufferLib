#!/usr/bin/env python3
"""Focused tests for the native Gear-Sonic ONNX Runtime adapter."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import onnxruntime as ort

from verify_gear_sonic_ort_probe import _comparison


ENCODER_INPUT_WIDTH = 1762
ENCODER_OUTPUT_WIDTH = 64
DECODER_INPUT_WIDTH = 994
DECODER_OUTPUT_WIDTH = 29
ONNXRUNTIME_VERSION = "1.22.1"
SOURCE_DIRECTORY = Path(__file__).resolve().parent


class GearSonicOrtBatch(ctypes.Structure):
    _fields_ = [
        ("api", ctypes.c_void_p),
        ("environment", ctypes.c_void_p),
        ("session_options", ctypes.c_void_p),
        ("encoder", ctypes.c_void_p),
        ("decoder", ctypes.c_void_p),
        ("cpu_memory", ctypes.c_void_p),
        ("batch_size", ctypes.c_size_t),
    ]


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is required")
    path = Path(value).absolute()
    if not path.is_file():
        raise RuntimeError(f"{name} is not a file: {path}")
    return path


def _probe_values(batch_size: int, start: int, stop: int, salt: int) -> np.ndarray:
    rows = np.arange(batch_size, dtype=np.uint64)[:, None]
    columns = np.arange(start, stop, dtype=np.uint64)[None, :]
    values = (rows * 131 + columns * 17 + np.uint64(salt * 29)) % 2003
    return (values.astype(np.float32) - np.float32(1001.0)) / np.float32(317.0)


def _all_zero(value: ctypes.Structure) -> bool:
    return bytes(value) == bytes(ctypes.sizeof(value))


class GearSonicOrtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.encoder_path = _required_path("GEAR_SONIC_ENCODER")
        cls.decoder_path = _required_path("GEAR_SONIC_DECODER")
        cls.include_directory = _required_path(
            "GEAR_SONIC_ORT_INCLUDE"
        ).parent
        cls.ort_library = _required_path("GEAR_SONIC_ORT_LIBRARY")
        cls.batch_size = int(os.environ["GEAR_SONIC_BATCH_SIZE"])
        if cls.batch_size <= 0:
            raise RuntimeError("GEAR_SONIC_BATCH_SIZE must be positive")
        if ort.__version__ != ONNXRUNTIME_VERSION:
            raise RuntimeError(
                f"expected onnxruntime {ONNXRUNTIME_VERSION}, got {ort.__version__}"
            )

        cls.temporary_directory = tempfile.TemporaryDirectory(
            prefix="gear-sonic-ort-test-"
        )
        cls.shared_library = Path(cls.temporary_directory.name) / "gear_sonic_ort.so"
        compiler = os.environ.get("CC", "cc")
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O2",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Wconversion",
                "-Wsign-conversion",
                "-Werror",
                "-isystem",
                str(cls.include_directory),
                str(SOURCE_DIRECTORY / "gear_sonic_ort.c"),
                str(cls.ort_library),
                "-lm",
                f"-Wl,-rpath,{cls.ort_library.parent}",
                "-o",
                str(cls.shared_library),
            ],
            check=True,
        )
        cls.library = ctypes.CDLL(cls.shared_library)
        cls.library.gear_sonic_ort_open.argtypes = [
            ctypes.POINTER(GearSonicOrtBatch),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        cls.library.gear_sonic_ort_open.restype = ctypes.c_int
        cls.library.gear_sonic_ort_open_from_memory.argtypes = [
            ctypes.POINTER(GearSonicOrtBatch),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        cls.library.gear_sonic_ort_open_from_memory.restype = ctypes.c_int
        cls.library.gear_sonic_ort_encode.argtypes = [
            ctypes.POINTER(GearSonicOrtBatch),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        cls.library.gear_sonic_ort_encode.restype = ctypes.c_int
        cls.library.gear_sonic_ort_decode.argtypes = list(
            cls.library.gear_sonic_ort_encode.argtypes
        )
        cls.library.gear_sonic_ort_decode.restype = ctypes.c_int
        cls.library.gear_sonic_ort_close.argtypes = [
            ctypes.POINTER(GearSonicOrtBatch)
        ]
        cls.library.gear_sonic_ort_close.restype = None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    @classmethod
    def _open(
        cls, batch_size: int | None = None, decoder_path: Path | None = None
    ) -> tuple[GearSonicOrtBatch, ctypes.Array[ctypes.c_char], int]:
        batch = GearSonicOrtBatch()
        error = ctypes.create_string_buffer(1024)
        opened = cls.library.gear_sonic_ort_open(
            ctypes.byref(batch),
            os.fsencode(cls.encoder_path),
            os.fsencode(decoder_path or cls.decoder_path),
            cls.batch_size if batch_size is None else batch_size,
            error,
            len(error),
        )
        return batch, error, opened

    @classmethod
    def _open_from_memory(
        cls,
    ) -> tuple[
        GearSonicOrtBatch,
        ctypes.Array[ctypes.c_char],
        int,
        ctypes.Array[ctypes.c_ubyte],
        ctypes.Array[ctypes.c_ubyte],
    ]:
        encoder_payload = cls.encoder_path.read_bytes()
        decoder_payload = cls.decoder_path.read_bytes()
        encoder = (ctypes.c_ubyte * len(encoder_payload)).from_buffer_copy(
            encoder_payload
        )
        decoder = (ctypes.c_ubyte * len(decoder_payload)).from_buffer_copy(
            decoder_payload
        )
        batch = GearSonicOrtBatch()
        error = ctypes.create_string_buffer(1024)
        opened = cls.library.gear_sonic_ort_open_from_memory(
            ctypes.byref(batch),
            encoder,
            len(encoder),
            decoder,
            len(decoder),
            cls.batch_size,
            error,
            len(error),
        )
        return batch, error, opened, encoder, decoder

    def test_exact_outputs(self) -> None:
        batch, error, opened = self._open()
        self.assertEqual(opened, 1, error.value.decode())
        self.assertEqual(error.value, b"")
        try:
            encoder_input = np.zeros(
                (self.batch_size, ENCODER_INPUT_WIDTH), dtype=np.float32
            )
            encoder_input[:, 0] = np.float32(1.0)
            encoder_input[:, 4:584] = _probe_values(
                self.batch_size, 4, 584, 1
            )
            encoder_input[:, 601:661] = _probe_values(
                self.batch_size, 601, 661, 2
            )
            decoder_input = _probe_values(
                self.batch_size, 0, DECODER_INPUT_WIDTH, 3
            )
            actual_tokens = np.empty(
                (self.batch_size, ENCODER_OUTPUT_WIDTH), dtype=np.float32
            )
            actual_actions = np.empty(
                (self.batch_size, DECODER_OUTPUT_WIDTH), dtype=np.float32
            )
            float_pointer = ctypes.POINTER(ctypes.c_float)
            self.assertEqual(
                self.library.gear_sonic_ort_encode(
                    ctypes.byref(batch),
                    encoder_input.ctypes.data_as(float_pointer),
                    actual_tokens.ctypes.data_as(float_pointer),
                    error,
                    len(error),
                ),
                1,
                error.value.decode(),
            )
            self.assertEqual(
                self.library.gear_sonic_ort_decode(
                    ctypes.byref(batch),
                    decoder_input.ctypes.data_as(float_pointer),
                    actual_actions.ctypes.data_as(float_pointer),
                    error,
                    len(error),
                ),
                1,
                error.value.decode(),
            )

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            encoder = ort.InferenceSession(
                str(self.encoder_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            decoder = ort.InferenceSession(
                str(self.decoder_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            expected_tokens = np.asarray(
                encoder.run(["encoded_tokens"], {"obs_dict": encoder_input})[0],
                dtype=np.float32,
            )
            expected_actions = np.asarray(
                decoder.run(["action"], {"obs_dict": decoder_input})[0],
                dtype=np.float32,
            )
            self.assertEqual(actual_tokens.tobytes(), expected_tokens.tobytes())
            self.assertEqual(actual_actions.tobytes(), expected_actions.tobytes())
        finally:
            self.library.gear_sonic_ort_close(ctypes.byref(batch))
        self.assertTrue(_all_zero(batch))

    def test_exact_batch_shape_is_required(self) -> None:
        batch, error, opened = self._open(self.batch_size + 1)
        self.assertEqual(opened, 0)
        self.assertIn(b"exact batch shape mismatch", error.value)
        self.assertTrue(_all_zero(batch))
        self.library.gear_sonic_ort_close(ctypes.byref(batch))

    def test_memory_sessions_do_not_reopen_or_retain_model_paths(self) -> None:
        batch, error, opened, encoder_bytes, decoder_bytes = (
            self._open_from_memory()
        )
        self.assertEqual(opened, 1, error.value.decode())
        self.assertEqual(error.value, b"")
        try:
            ctypes.memset(encoder_bytes, 0, len(encoder_bytes))
            ctypes.memset(decoder_bytes, 0, len(decoder_bytes))
            encoder_input = np.zeros(
                (self.batch_size, ENCODER_INPUT_WIDTH), dtype=np.float32
            )
            encoder_input[:, 0] = np.float32(1.0)
            encoder_input[:, 4:584] = _probe_values(
                self.batch_size, 4, 584, 11
            )
            encoder_input[:, 601:661] = _probe_values(
                self.batch_size, 601, 661, 12
            )
            actual = np.empty(
                (self.batch_size, ENCODER_OUTPUT_WIDTH), dtype=np.float32
            )
            float_pointer = ctypes.POINTER(ctypes.c_float)
            self.assertEqual(
                self.library.gear_sonic_ort_encode(
                    ctypes.byref(batch),
                    encoder_input.ctypes.data_as(float_pointer),
                    actual.ctypes.data_as(float_pointer),
                    error,
                    len(error),
                ),
                1,
                error.value.decode(),
            )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            expected = np.asarray(
                ort.InferenceSession(
                    str(self.encoder_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                ).run(["encoded_tokens"], {"obs_dict": encoder_input})[0],
                dtype=np.float32,
            )
            self.assertEqual(actual.tobytes(), expected.tobytes())
        finally:
            self.library.gear_sonic_ort_close(ctypes.byref(batch))
        self.assertTrue(_all_zero(batch))

    def test_reopen_rejected_without_mutating_live_session(self) -> None:
        batch, error, opened = self._open()
        self.assertEqual(opened, 1, error.value.decode())
        try:
            before = bytes(batch)
            self.assertEqual(
                self.library.gear_sonic_ort_open(
                    ctypes.byref(batch),
                    os.fsencode(self.encoder_path),
                    os.fsencode(self.decoder_path),
                    self.batch_size,
                    error,
                    len(error),
                ),
                0,
            )
            self.assertEqual(error.value, b"open: batch is already initialized")
            self.assertEqual(bytes(batch), before)

            encoder_payload = self.encoder_path.read_bytes()
            decoder_payload = self.decoder_path.read_bytes()
            encoder = (ctypes.c_ubyte * len(encoder_payload)).from_buffer_copy(
                encoder_payload
            )
            decoder = (ctypes.c_ubyte * len(decoder_payload)).from_buffer_copy(
                decoder_payload
            )
            self.assertEqual(
                self.library.gear_sonic_ort_open_from_memory(
                    ctypes.byref(batch),
                    encoder,
                    len(encoder),
                    decoder,
                    len(decoder),
                    self.batch_size,
                    error,
                    len(error),
                ),
                0,
            )
            self.assertEqual(error.value, b"open: batch is already initialized")
            self.assertEqual(bytes(batch), before)

            encoder_input = np.zeros(
                (self.batch_size, ENCODER_INPUT_WIDTH), dtype=np.float32
            )
            encoder_input[:, 0] = np.float32(1.0)
            tokens = np.empty(
                (self.batch_size, ENCODER_OUTPUT_WIDTH), dtype=np.float32
            )
            float_pointer = ctypes.POINTER(ctypes.c_float)
            self.assertEqual(
                self.library.gear_sonic_ort_encode(
                    ctypes.byref(batch),
                    encoder_input.ctypes.data_as(float_pointer),
                    tokens.ctypes.data_as(float_pointer),
                    error,
                    len(error),
                ),
                1,
                error.value.decode(),
            )
            self.assertTrue(np.isfinite(tokens).all())
        finally:
            self.library.gear_sonic_ort_close(ctypes.byref(batch))
        self.assertTrue(_all_zero(batch))

    def test_memory_open_rejects_empty_model_array(self) -> None:
        decoder_payload = self.decoder_path.read_bytes()
        decoder = (ctypes.c_ubyte * len(decoder_payload)).from_buffer_copy(
            decoder_payload
        )
        batch = GearSonicOrtBatch()
        error = ctypes.create_string_buffer(1024)
        self.assertEqual(
            self.library.gear_sonic_ort_open_from_memory(
                ctypes.byref(batch),
                None,
                0,
                decoder,
                len(decoder),
                self.batch_size,
                error,
                len(error),
            ),
            0,
        )
        self.assertEqual(error.value, b"open: null argument")
        self.assertTrue(_all_zero(batch))

    def test_failed_second_session_is_cleaned_up(self) -> None:
        missing = Path(self.temporary_directory.name) / "missing-decoder.onnx"
        for _ in range(3):
            batch, error, opened = self._open(decoder_path=missing)
            self.assertEqual(opened, 0)
            self.assertIn(b"create decoder session", error.value)
            self.assertTrue(_all_zero(batch))

    def test_invalid_and_uninitialized_arguments(self) -> None:
        error = ctypes.create_string_buffer(1024)
        self.assertEqual(
            self.library.gear_sonic_ort_open(
                None,
                os.fsencode(self.encoder_path),
                os.fsencode(self.decoder_path),
                self.batch_size,
                error,
                len(error),
            ),
            0,
        )
        self.assertEqual(error.value, b"open: null argument")

        batch = GearSonicOrtBatch()
        one_byte_error = ctypes.create_string_buffer(1)
        self.assertEqual(
            self.library.gear_sonic_ort_open(
                ctypes.byref(batch),
                os.fsencode(self.encoder_path),
                os.fsencode(self.decoder_path),
                0,
                one_byte_error,
                len(one_byte_error),
            ),
            0,
        )
        self.assertEqual(one_byte_error.raw, b"\x00")
        self.assertTrue(_all_zero(batch))

        input_value = ctypes.c_float(0.0)
        output_value = ctypes.c_float(0.0)
        self.assertEqual(
            self.library.gear_sonic_ort_encode(
                ctypes.byref(batch),
                ctypes.byref(input_value),
                ctypes.byref(output_value),
                error,
                len(error),
            ),
            0,
        )
        self.assertEqual(error.value, b"run: uninitialized argument")
        self.library.gear_sonic_ort_close(None)
        self.library.gear_sonic_ort_close(ctypes.byref(batch))
        self.library.gear_sonic_ort_close(ctypes.byref(batch))

    def test_nonfinite_and_overflow_inputs_are_rejected(self) -> None:
        batch, error, opened = self._open()
        self.assertEqual(opened, 1, error.value.decode())
        float_pointer = ctypes.POINTER(ctypes.c_float)
        try:
            encoder_input = np.zeros(
                (self.batch_size, ENCODER_INPUT_WIDTH), dtype=np.float32
            )
            encoder_input[0, 0] = np.float32(np.nan)
            tokens = np.full(
                (self.batch_size, ENCODER_OUTPUT_WIDTH),
                np.float32(123.5),
                dtype=np.float32,
            )
            before = tokens.tobytes()
            self.assertEqual(
                self.library.gear_sonic_ort_encode(
                    ctypes.byref(batch),
                    encoder_input.ctypes.data_as(float_pointer),
                    tokens.ctypes.data_as(float_pointer),
                    error,
                    len(error),
                ),
                0,
            )
            self.assertEqual(error.value, b"run: non-finite input")
            self.assertEqual(tokens.tobytes(), before)

            batch.batch_size = ctypes.c_size_t(-1).value
            self.assertEqual(
                self.library.gear_sonic_ort_encode(
                    ctypes.byref(batch),
                    encoder_input.ctypes.data_as(float_pointer),
                    tokens.ctypes.data_as(float_pointer),
                    error,
                    len(error),
                ),
                0,
            )
            self.assertEqual(error.value, b"run: tensor size overflow")
            batch.batch_size = self.batch_size
        finally:
            self.library.gear_sonic_ort_close(ctypes.byref(batch))

    def test_verifier_uses_bitwise_float32_equality(self) -> None:
        result = _comparison(
            np.array([0.0], dtype="<f4"), np.array([-0.0], dtype="<f4")
        )
        self.assertTrue(result["array_equal"])
        self.assertFalse(result["bitwise_equal"])
        self.assertNotEqual(
            result["actual_sha256_float32_le"],
            result["expected_sha256_float32_le"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
