"""Masked native MinGRU kernel differential; CPU oracle tests need no GPU.

Run the actual CUDA checks explicitly with --library and optional --legacy-library.
The oracle is a sequential mathematical recurrence. It does not substitute for
authentic REK validation, and this diagnostic never runs an environment.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import unittest

import torch


def sequential_scan(combined, state, inputs, terminals):
    outputs = []
    hidden_size = state.shape[-1]
    for tick in range(inputs.shape[1]):
        if terminals is not None:
            state = torch.where(terminals[:, tick, None] != 0, 0, state)
        hidden, gate, projection = combined[:, tick].split(hidden_size, dim=-1)
        gate = torch.sigmoid(gate)
        target = torch.where(hidden >= 0, hidden + 0.5, torch.sigmoid(hidden))
        state = (1 - gate) * state + gate * target
        projection = torch.sigmoid(projection)
        outputs.append(projection * state + (1 - projection) * inputs[:, tick])
    return torch.stack(outputs, dim=1), state


def oracle(combined, state, inputs, terminals, grad_out, grad_next):
    combined, state, inputs = [
        value.detach().cpu().double().requires_grad_(True)
        for value in (combined, state, inputs)
    ]
    mask = None if terminals is None else terminals.detach().cpu()
    out, next_state = sequential_scan(combined, state, inputs, mask)
    loss = (out * grad_out.detach().cpu().double()).sum()
    loss = loss + (next_state * grad_next.detach().cpu().double()).sum()
    grads = torch.autograd.grad(loss, (combined, state, inputs))
    return tuple(value.detach() for value in (out, next_state, *grads))


class NativeScan:
    def __init__(self, library: Path):
        self.path = library.resolve()
        self.library = ctypes.CDLL(str(self.path))
        self.library.rek_scan_precision_bytes.restype = ctypes.c_int
        self.library.rek_scan_supports_terminals.restype = ctypes.c_int
        self.library.rek_scan_run.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 15
        self.library.rek_scan_run.restype = ctypes.c_int
        self.dtype = {4: torch.float32, 2: torch.bfloat16}[
            self.library.rek_scan_precision_bytes()
        ]

    def buffers(self, combined, state, inputs):
        batch, ticks, hidden = inputs.shape
        return (
            torch.empty_like(inputs), torch.empty_like(state),
            torch.empty((batch, ticks + 1, hidden), device=inputs.device),
            torch.empty((batch, ticks + 1, hidden), device=inputs.device),
            torch.empty((batch, ticks + 1, hidden), device=inputs.device),
            torch.empty_like(combined), torch.empty_like(state), torch.empty_like(inputs),
        )

    def run(self, combined, state, inputs, terminals, grad_out, grad_next, buffers):
        batch, ticks, hidden = inputs.shape
        pointers = (combined, state, inputs, terminals, *buffers, grad_out, grad_next)
        status = self.library.rek_scan_run(
            batch, ticks, hidden,
            *(None if tensor is None else tensor.data_ptr() for tensor in pointers),
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"native scan returned CUDA status {status}")
        return (buffers[0], buffers[1], buffers[5], buffers[6], buffers[7])


def compare(actual, expected, dtype, label):
    # Float checkpoints retain the pre-existing scan's float32 accumulation.
    # BF16 additionally rounds its tensor outputs and gradients.
    atol, rtol = (0.025, 0.025) if dtype == torch.bfloat16 else (0.00015, 0.00015)
    errors = {}
    for name, got, want in zip(
        ("output", "next_state", "grad_combined", "grad_state", "grad_input"), actual, expected
    ):
        got = got.detach().cpu().double()
        if not bool(torch.isfinite(got).all()):
            raise AssertionError(f"{label}/{name}: nonfinite native result")
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol, msg=f"{label}/{name}")
        errors[name] = float((got - want).abs().max())
    return errors


def run_cuda_regression(library: Path, legacy_library: Path | None = None):
    native = NativeScan(library)
    if not native.library.rek_scan_supports_terminals():
        raise ValueError("candidate library has no terminal support")
    legacy = None if legacy_library is None else NativeScan(legacy_library)
    if legacy is not None and native.dtype != legacy.dtype:
        raise ValueError("legacy and candidate precision must match")
    generator = torch.Generator(device="cpu").manual_seed(7391)
    cases = []
    for ticks in (4, 8, 64):
        batch, hidden = 3, 7
        tensors = [
            torch.randn(shape, generator=generator).mul_(0.6).to("cuda", native.dtype)
            for shape in ((batch, ticks, 3 * hidden), (batch, ticks, hidden),
                          (batch, ticks, hidden), (batch, hidden))
        ]
        combined, inputs, grad_out, grad_next = tensors
        for zero_initial in (False, True):
            state = torch.full((batch, hidden), 0 if zero_initial else 0.7,
                               dtype=native.dtype, device="cuda")
            patterns = {
                "none": [], "t0": [0], "interior": [1], "final": [ticks - 1],
                "checkpoint_edge": [min(4, ticks - 1)],
                "before_checkpoint": [3], "adjacent": [0, 1, ticks - 1],
            }
            for name, resets in patterns.items():
                mask = torch.zeros((batch, ticks), device="cuda", dtype=native.dtype)
                # Different reset placement for each row catches accidental broadcasting.
                for row in range(batch):
                    for tick in resets:
                        mask[row, (tick + row) % ticks] = 1
                buffers = native.buffers(combined, state, inputs)
                label = f"T{ticks}/{name}/h0_{0 if zero_initial else 0.7}"
                actual = native.run(combined, state, inputs, mask, grad_out, grad_next, buffers)
                expected = oracle(combined, state, inputs, mask, grad_out, grad_next)
                cases.append({"case": label, "errors": compare(actual, expected, native.dtype, label)})
                if not resets:
                    unmasked = native.run(combined, state, inputs, None, grad_out, grad_next,
                                          native.buffers(combined, state, inputs))
                    for index, (got, old) in enumerate(zip(actual, unmasked)):
                        if zero_initial and index == 3:
                            continue  # Legacy log(0)/0 initial-state gradient is undefined.
                        if not torch.equal(got, old):
                            raise AssertionError(f"{label}: null-mask arithmetic changed at output {index}")
                    if legacy is not None:
                        old = legacy.run(combined, state, inputs, None, grad_out, grad_next,
                                         legacy.buffers(combined, state, inputs))
                        for index, (got, before) in enumerate(zip(actual, old)):
                            if zero_initial and index == 3:
                                continue
                            if not torch.equal(got, before):
                                raise AssertionError(f"{label}: legacy binary differs at output {index}")

    # Fixed pointers, changing terminal contents must work after CUDA graph capture.
    ticks = 8
    combined = torch.randn((3, ticks, 21), generator=generator).to("cuda", native.dtype)
    inputs = torch.randn((3, ticks, 7), generator=generator).to("cuda", native.dtype)
    state = torch.zeros((3, 7), device="cuda", dtype=native.dtype)
    grad_out = torch.randn((3, ticks, 7), generator=generator).to("cuda", native.dtype)
    grad_next = torch.ones_like(state)
    mask = torch.zeros((3, ticks), device="cuda", dtype=native.dtype)
    buffers = native.buffers(combined, state, inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        native.run(combined, state, inputs, mask, grad_out, grad_next, buffers)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = native.run(combined, state, inputs, mask, grad_out, grad_next, buffers)
    graph_cases = []
    for name, resets in (("none", []), ("t0", [0]), ("interior", [3]),
                         ("checkpoint", [4]), ("final", [7]), ("all", list(range(8)))):
        mask.zero_()
        for tick in resets:
            mask[:, tick] = 1
        graph.replay()
        expected = oracle(combined, state, inputs, mask, grad_out, grad_next)
        graph_cases.append({"case": name, "errors": compare(actual, expected, native.dtype, name)})
    # Future-only loss cannot cross the reset before observation 4.
    mask.zero_()
    mask[:, 4] = 1
    grad_out[:, :4].zero_()
    graph.replay()
    if torch.count_nonzero(actual[2][:, :4]).item() or torch.count_nonzero(actual[3]).item():
        raise AssertionError("gradient crossed a reset boundary")
    return {
        "stage": "passed", "dtype": str(native.dtype), "device": torch.cuda.get_device_name(),
        "library": str(native.path), "library_sha256": hashlib.sha256(native.path.read_bytes()).hexdigest(),
        "legacy_library": None if legacy is None else str(legacy.path),
        "legacy_library_sha256": None if legacy is None else hashlib.sha256(legacy.path.read_bytes()).hexdigest(),
        "cases": cases, "captured_mask_mutations": graph_cases,
        "future_gradient_crosses_reset": False,
        "legacy_zero_initial_grad_state_exception": "masked path uses the finite analytic limit",
        "checkpoint_horizons": [4, 8, 64],
    }


class SequentialOracleTests(unittest.TestCase):
    def test_terminal_resets_before_same_observation(self):
        combined = torch.zeros((1, 4, 3), dtype=torch.float64)
        inputs = torch.zeros((1, 4, 1), dtype=torch.float64)
        state = torch.zeros((1, 1), dtype=torch.float64)
        mask = torch.tensor([[0, 0, 1, 0]])
        actual, last = sequential_scan(combined, state, inputs, mask)
        torch.testing.assert_close(actual.flatten(), torch.tensor([.125, .1875, .125, .1875], dtype=torch.float64))
        self.assertEqual(last.item(), .375)

    def test_future_loss_stops_at_reset(self):
        combined = torch.zeros((1, 8, 3), dtype=torch.float64)
        inputs = torch.zeros((1, 8, 1), dtype=torch.float64)
        mask = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0]])
        grad_out = torch.ones_like(inputs)
        grad_out[:, :4] = 0
        actual = oracle(combined, torch.ones((1, 1)), inputs, mask, grad_out, torch.ones((1, 1)))
        self.assertEqual(torch.count_nonzero(actual[2][:, :4]).item(), 0)
        self.assertEqual(actual[3].item(), 0)

    def test_zero_initial_derivative_is_finite_and_nonzero_without_reset(self):
        combined = torch.zeros((1, 4, 3), dtype=torch.float64)
        inputs = torch.zeros((1, 4, 1), dtype=torch.float64)
        actual = oracle(combined, torch.zeros((1, 1)), inputs, torch.zeros((1, 4)),
                        torch.ones_like(inputs), torch.ones((1, 1)))
        self.assertAlmostEqual(actual[3].item(), .53125)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--legacy-library", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.library is None:
        unittest.main(argv=[__file__])
    else:
        report = run_cuda_regression(args.library, args.legacy_library)
        payload = json.dumps(report, indent=2) + "\n"
        if args.report:
            args.report.write_text(payload, encoding="utf-8")
        print(payload, end="")
