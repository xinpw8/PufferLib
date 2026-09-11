"""CUDA-resident inference for the pinned GEAR-SONIC controller graphs.

The controller accepts only explicit-batch bundles already validated by
``gear_sonic_batch_model``. Model parsing and reconstruction happen once during
startup. Every inference input, model tensor, and output must remain on CUDA.

This module establishes graph identity and numerical agreement with the
recorded ONNX Runtime candidate. It does not establish parity with REK.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


MANIFEST_FILENAME = "explicit_batch_manifest.json"
MANIFEST_SCHEMA = "rek.g1_gear_sonic_explicit_batch.v1"
VALIDATED_ONNX_VERSION = "1.22.0"
VALIDATED_TORCH_VERSION = "2.9.1a0+gitd38164a"
ENCODER_EXECUTION_CONTRACT = "g1_mode_zero"

EXPECTED_OPERATOR_COUNTS: Mapping[str, Counter[str]] = {
    "encoder": Counter(
        {
            "Constant": 97,
            "Reshape": 25,
            "Mul": 19,
            "Gemm": 15,
            "Sigmoid": 12,
            "Slice": 11,
            "Unsqueeze": 8,
            "Add": 7,
            "Concat": 7,
            "Sub": 6,
            "Shape": 5,
            "Cast": 4,
            "ConstantOfShape": 3,
            "Equal": 3,
            "Where": 3,
            "Expand": 3,
            "Tanh": 3,
            "Round": 3,
            "Div": 3,
            "Gather": 1,
            "ScatterND": 1,
            "Transpose": 1,
            "ReduceSum": 1,
        }
    ),
    "decoder": Counter(
        {
            "Constant": 15,
            "MatMul": 7,
            "Add": 7,
            "Sigmoid": 6,
            "Mul": 6,
            "Slice": 3,
            "Unsqueeze": 2,
            "Concat": 1,
            "Squeeze": 1,
        }
    ),
}

MODEL_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "model_encoder.onnx": {
        "role": "encoder",
        "source_sha256": (
            "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
        ),
        "input_name": "obs_dict",
        "input_width": 1762,
        "output_name": "encoded_tokens",
        "output_width": 64,
    },
    "model_decoder.onnx": {
        "role": "decoder",
        "source_sha256": (
            "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"
        ),
        "input_name": "obs_dict",
        "input_width": 994,
        "output_name": "action",
        "output_width": 29,
    },
}


class GpuControllerError(RuntimeError):
    """The CUDA controller runtime violated its pinned contract."""


@dataclass(frozen=True)
class GpuModelIdentity:
    """Identity and I/O contract for one converted graph."""

    role: str
    path: Path
    sha256: str
    input_name: str
    input_width: int
    output_name: str
    output_width: int


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GpuControllerError(f"{label} must be an object")
    return value


def _batch_size_from_manifest(manifest_path: Path) -> int:
    try:
        report = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            "explicit-batch manifest",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GpuControllerError(
            "explicit-batch manifest is not valid UTF-8 JSON"
        ) from exc
    if report.get("schema") != MANIFEST_SCHEMA:
        raise GpuControllerError("explicit-batch manifest schema mismatch")
    if report.get("rek_parity_claim") is not False:
        raise GpuControllerError("explicit-batch manifest must not claim REK parity")
    batch_size = report.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise GpuControllerError("explicit-batch size must be an integer")
    if batch_size < 1:
        raise GpuControllerError("explicit-batch size must be positive")
    return batch_size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_dependencies() -> tuple[Any, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise GpuControllerError(
            "torch and onnx are required for the CUDA controller"
        ) from exc
    return torch, onnx


def _tensor_shape(value: Any) -> tuple[int | str, ...]:
    shape: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(str(dimension.dim_param))
        else:
            shape.append("")
    return tuple(shape)


def _validate_onnx_contract(
    onnx: Any,
    model: Any,
    identity: GpuModelIdentity,
    batch_size: int,
) -> None:
    if [(entry.domain, entry.version) for entry in model.opset_import] != [("", 13)]:
        raise GpuControllerError(f"{identity.role} ONNX opset mismatch")
    graph_inputs = list(model.graph.input)
    graph_outputs = list(model.graph.output)
    if len(graph_inputs) != 1 or graph_inputs[0].name != identity.input_name:
        raise GpuControllerError(f"{identity.role} ONNX input mismatch")
    if len(graph_outputs) != 1 or graph_outputs[0].name != identity.output_name:
        raise GpuControllerError(f"{identity.role} ONNX output mismatch")
    if graph_inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise GpuControllerError(f"{identity.role} ONNX input must be float32")
    if graph_outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise GpuControllerError(f"{identity.role} ONNX output must be float32")
    expected_input = (batch_size, identity.input_width)
    expected_output = (batch_size, identity.output_width)
    if _tensor_shape(graph_inputs[0]) != expected_input:
        raise GpuControllerError(f"{identity.role} ONNX input shape mismatch")
    if _tensor_shape(graph_outputs[0]) != expected_output:
        raise GpuControllerError(f"{identity.role} ONNX output shape mismatch")
    operators = Counter(node.op_type for node in model.graph.node)
    if operators != EXPECTED_OPERATOR_COUNTS[identity.role]:
        raise GpuControllerError(f"{identity.role} ONNX operator graph mismatch")


def _require_validated_versions(torch: Any, onnx: Any) -> None:
    versions = {
        "torch": str(torch.__version__),
        "onnx": str(onnx.__version__),
    }
    expected = {
        "torch": VALIDATED_TORCH_VERSION,
        "onnx": VALIDATED_ONNX_VERSION,
    }
    if versions != expected:
        raise GpuControllerError(
            f"unvalidated CUDA controller dependency versions: {versions!r}"
        )


def _float_tensor(
    numpy_helper: Any,
    tensor: Any,
    name: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    values = np.asarray(numpy_helper.to_array(tensor))
    if values.dtype != np.float32:
        raise GpuControllerError(f"{name} must contain float32")
    if tuple(values.shape) != expected_shape:
        raise GpuControllerError(
            f"{name} shape must be {expected_shape}, got {tuple(values.shape)}"
        )
    return torch.from_numpy(values.copy())


def _initializers(model: Any) -> Mapping[str, Any]:
    result = {tensor.name: tensor for tensor in model.graph.initializer}
    if len(result) != len(model.graph.initializer):
        raise GpuControllerError("ONNX initializer names must be unique")
    return result


def _constants(onnx: Any, model: Any) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        tensors = [
            attribute.t
            for attribute in node.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        ]
        if len(tensors) != 1 or node.name in result:
            raise GpuControllerError("ONNX Constant node contract mismatch")
        result[node.name] = tensors[0]
    return result


class _PinnedG1Encoder(torch.nn.Module):
    """Capture-safe active G1 branch of the pinned multi-mode encoder."""

    def __init__(self, onnx: Any, numpy_helper: Any, model: Any) -> None:
        super().__init__()
        tensors = _initializers(model)
        layer_shapes = (
            (2048, 640),
            (1024, 2048),
            (512, 1024),
            (512, 512),
            (64, 512),
        )
        for index, (source_layer, shape) in enumerate(
            zip((0, 2, 4, 6, 8), layer_shapes, strict=True)
        ):
            weight_name = f"module.encoders.g1.module.{source_layer}.weight"
            bias_name = f"module.encoders.g1.module.{source_layer}.bias"
            self.register_buffer(
                f"weight_{index}",
                _float_tensor(numpy_helper, tensors[weight_name], weight_name, shape),
            )
            self.register_buffer(
                f"bias_{index}",
                _float_tensor(
                    numpy_helper, tensors[bias_name], bias_name, (shape[0],)
                ),
            )

        constants = _constants(onnx, model)
        quantizers = (
            ("quantizer_offset", "/quantizer/Constant_1"),
            ("quantizer_scale", "/quantizer/Constant_2"),
            ("quantizer_half", "/quantizer/Constant_3"),
            ("quantizer_divisor", "/quantizer/Constant_4"),
        )
        for target_name, source_name in quantizers:
            self.register_buffer(
                target_name,
                _float_tensor(
                    numpy_helper, constants[source_name], source_name, (32,)
                ),
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        batch_size = observation.shape[0]
        positions = observation[:, 4:584].reshape(batch_size, 10, 58)
        rotations = observation[:, 601:661].reshape(batch_size, 10, 6)
        value = torch.cat((positions, rotations), dim=-1).reshape(batch_size, 640)
        for index in range(5):
            value = torch.nn.functional.linear(
                value,
                getattr(self, f"weight_{index}"),
                getattr(self, f"bias_{index}"),
            )
            if index < 4:
                value = value * torch.sigmoid(value)
        value = value.reshape(batch_size, 2, 1, 32)
        value = torch.tanh(value + self.quantizer_offset)
        value = value * self.quantizer_scale
        value = value - self.quantizer_half
        value = value + (torch.round(value) - value)
        value = value / self.quantizer_divisor
        return value.reshape(batch_size, 64).float()


class _PinnedG1Decoder(torch.nn.Module):
    """Capture-safe pinned dynamic G1 action decoder."""

    def __init__(self, numpy_helper: Any, model: Any) -> None:
        super().__init__()
        tensors = _initializers(model)
        layer_shapes = (
            (994, 2048),
            (2048, 2048),
            (2048, 1024),
            (1024, 1024),
            (1024, 512),
            (512, 512),
            (512, 29),
        )
        for index, (source_layer, shape) in enumerate(
            zip((0, 2, 4, 6, 8, 10, 12), layer_shapes, strict=True)
        ):
            weight_name = f"onnx::MatMul_{136 + index}"
            bias_name = f"module.decoders.g1_dyn.module.{source_layer}.bias"
            self.register_buffer(
                f"weight_{index}",
                _float_tensor(numpy_helper, tensors[weight_name], weight_name, shape),
            )
            self.register_buffer(
                f"bias_{index}",
                _float_tensor(
                    numpy_helper, tensors[bias_name], bias_name, (shape[1],)
                ),
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        token = observation[:, :64].unsqueeze(1)
        history = observation[:, 64:].unsqueeze(1)
        value = torch.cat((token, history), dim=-1)
        for index in range(7):
            value = getattr(self, f"bias_{index}") + torch.matmul(
                value, getattr(self, f"weight_{index}")
            )
            if index < 6:
                value = value * torch.sigmoid(value)
        return value[:, :, :29].squeeze(1)


class CapturedGearSonicController:
    """Static-address CUDA graph for one encoder and decoder invocation."""

    def __init__(
        self,
        controller: "GearSonicGpuController",
        graph: Any,
        encoder_observation: torch.Tensor,
        decoder_observation: torch.Tensor,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        self.controller = controller
        self.graph = graph
        self.encoder_observation = encoder_observation
        self.decoder_observation = decoder_observation
        self.tokens = tokens
        self.actions = actions

    def replay(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch the captured pair without copying or synchronizing tensors."""

        self.graph.replay()
        return self.tokens, self.actions


class GearSonicGpuController:
    """The exact-batch GEAR-SONIC encoder and decoder resident on CUDA."""

    def __init__(
        self,
        *,
        batch_size: int,
        encoder_identity: GpuModelIdentity,
        decoder_identity: GpuModelIdentity,
        device: str = "cuda",
        require_validated_versions: bool = True,
    ) -> None:
        torch_runtime, onnx = _runtime_dependencies()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise GpuControllerError("batch size must be an integer")
        if batch_size < 1:
            raise GpuControllerError("batch size must be positive")
        if require_validated_versions:
            _require_validated_versions(torch_runtime, onnx)
        if not torch_runtime.cuda.is_available():
            raise GpuControllerError("CUDA is unavailable")
        resolved_device = torch_runtime.device(device)
        if resolved_device.type != "cuda":
            raise GpuControllerError("the GEAR-SONIC GPU controller requires CUDA")
        if resolved_device.index is None:
            resolved_device = torch_runtime.device(
                "cuda", torch_runtime.cuda.current_device()
            )
        if torch_runtime.get_float32_matmul_precision() != "highest":
            raise GpuControllerError("float32 matmul precision must be highest")
        if bool(torch_runtime.backends.cuda.matmul.allow_tf32):
            raise GpuControllerError("TF32 must be disabled for controller inference")

        self._torch = torch_runtime
        self.batch_size = batch_size
        self.device = resolved_device
        self.encoder_identity = encoder_identity
        self.decoder_identity = decoder_identity
        self.encoder = self._load_module(onnx, encoder_identity)
        self.decoder = self._load_module(onnx, decoder_identity)
        self._verify_residency(self.encoder, "encoder")
        self._verify_residency(self.decoder, "decoder")

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        source_bundle: Path,
        *,
        device: str = "cuda",
        require_validated_versions: bool = True,
    ) -> "GearSonicGpuController":
        """Validate an explicit-batch bundle, then load it on CUDA."""

        manifest_path = Path(manifest_path).resolve(strict=True)
        if manifest_path.name != MANIFEST_FILENAME:
            raise GpuControllerError(
                f"manifest filename must be {MANIFEST_FILENAME!r}"
            )
        batch_size = _batch_size_from_manifest(manifest_path)
        try:
            import gear_sonic_batch_model as batch_model
        except ImportError as exc:
            raise GpuControllerError(
                "gear_sonic_batch_model must be importable"
            ) from exc
        try:
            inspection = batch_model.inspect_explicit_batch_bundle(
                Path(source_bundle), manifest_path.parent, batch_size
            )
        except Exception as exc:
            raise GpuControllerError(
                f"explicit-batch bundle validation failed: {exc}"
            ) from exc

        identities: dict[str, GpuModelIdentity] = {}
        records = _mapping(inspection.get("models"), "validated models")
        for filename, contract in MODEL_CONTRACTS.items():
            record = _mapping(records.get(filename), f"models.{filename}")
            source_sha256 = record.get("source_sha256")
            if source_sha256 != contract["source_sha256"]:
                raise GpuControllerError(f"{filename} source identity mismatch")
            identities[str(contract["role"])] = GpuModelIdentity(
                role=str(contract["role"]),
                path=Path(str(record["output_path"])),
                sha256=str(record["output_sha256"]),
                input_name=str(contract["input_name"]),
                input_width=int(contract["input_width"]),
                output_name=str(contract["output_name"]),
                output_width=int(contract["output_width"]),
            )
        return cls(
            batch_size=batch_size,
            encoder_identity=identities["encoder"],
            decoder_identity=identities["decoder"],
            device=device,
            require_validated_versions=require_validated_versions,
        )

    def _load_module(self, onnx: Any, identity: GpuModelIdentity) -> Any:
        try:
            if _sha256_file(identity.path) != identity.sha256:
                raise GpuControllerError(f"{identity.role} ONNX SHA-256 mismatch")
            model = onnx.load(identity.path)
            onnx.checker.check_model(model)
            _validate_onnx_contract(onnx, model, identity, self.batch_size)
            from onnx import numpy_helper

            module = (
                _PinnedG1Encoder(onnx, numpy_helper, model)
                if identity.role == "encoder"
                else _PinnedG1Decoder(numpy_helper, model)
            ).eval()
            del model
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.to(device=self.device)
            gc.collect()
            return module
        except GpuControllerError:
            raise
        except Exception as exc:
            raise GpuControllerError(
                f"failed to reconstruct {identity.role} graph: {exc}"
            ) from exc

    def _verify_residency(self, module: Any, role: str) -> None:
        tensors = [*module.parameters(), *module.buffers()]
        if not tensors:
            raise GpuControllerError(f"{role} contains no resident tensors")
        wrong_device = [str(tensor.device) for tensor in tensors if not tensor.is_cuda]
        if wrong_device:
            raise GpuControllerError(
                f"{role} contains non-CUDA tensors: {wrong_device[:3]!r}"
            )
        wrong_index = [
            str(tensor.device)
            for tensor in tensors
            if tensor.device.index != self.device.index
        ]
        if wrong_index:
            raise GpuControllerError(
                f"{role} contains tensors on another CUDA device: {wrong_index[:3]!r}"
            )

    def _validate_input(self, value: Any, identity: GpuModelIdentity) -> Any:
        torch = self._torch
        if not isinstance(value, torch.Tensor):
            raise GpuControllerError(f"{identity.role} input must be a torch tensor")
        if value.device != self.device:
            raise GpuControllerError(
                f"{identity.role} input must be on {self.device}, got {value.device}"
            )
        if value.dtype != torch.float32:
            raise GpuControllerError(f"{identity.role} input must be float32")
        expected = (self.batch_size, identity.input_width)
        if tuple(value.shape) != expected:
            raise GpuControllerError(
                f"{identity.role} input shape must be {expected}, got {tuple(value.shape)}"
            )
        if not value.is_contiguous():
            raise GpuControllerError(f"{identity.role} input must be contiguous")
        return value

    def _infer(self, module: Any, value: Any, identity: GpuModelIdentity) -> Any:
        torch = self._torch
        value = self._validate_input(value, identity)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            output = module(value)
        expected = (self.batch_size, identity.output_width)
        if output.device != self.device:
            raise GpuControllerError(f"{identity.role} output left CUDA")
        if output.dtype != torch.float32:
            raise GpuControllerError(f"{identity.role} output is not float32")
        if tuple(output.shape) != expected:
            raise GpuControllerError(
                f"{identity.role} output shape must be {expected}, got {tuple(output.shape)}"
            )
        return output

    def encode(self, encoder_observation: Any) -> Any:
        """Encode one CUDA-resident explicit batch into 64-value tokens."""

        return self._infer(self.encoder, encoder_observation, self.encoder_identity)

    def decode(self, decoder_observation: Any) -> Any:
        """Decode one CUDA-resident explicit batch into 29 raw actions."""

        return self._infer(self.decoder, decoder_observation, self.decoder_identity)

    def capture_pair(
        self,
        encoder_observation: torch.Tensor,
        decoder_observation: torch.Tensor,
        *,
        warmup: int = 3,
    ) -> CapturedGearSonicController:
        """Capture a static-address encoder and decoder pair for replay."""

        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise GpuControllerError("CUDA graph warmup must be a positive integer")
        encoder_observation = self._validate_input(
            encoder_observation, self.encoder_identity
        )
        decoder_observation = self._validate_input(
            decoder_observation, self.decoder_identity
        )
        torch_runtime = self._torch
        current_stream = torch_runtime.cuda.current_stream(self.device)
        warmup_stream = torch_runtime.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch_runtime.cuda.stream(warmup_stream):
            for _ in range(warmup):
                self.encode(encoder_observation)
                self.decode(decoder_observation)
        current_stream.wait_stream(warmup_stream)
        torch_runtime.cuda.synchronize(self.device)

        graph = torch_runtime.cuda.CUDAGraph()
        with torch_runtime.cuda.graph(graph):
            tokens = self.encode(encoder_observation)
            actions = self.decode(decoder_observation)
        return CapturedGearSonicController(
            self,
            graph,
            encoder_observation,
            decoder_observation,
            tokens,
            actions,
        )

    def resident_bytes(self) -> int:
        """Return storage bytes held by CUDA parameters and buffers."""

        modules = (self.encoder, self.decoder)
        return sum(
            tensor.numel() * tensor.element_size()
            for module in modules
            for tensor in [*module.parameters(), *module.buffers()]
        )
