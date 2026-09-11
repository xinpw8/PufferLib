"""PyTorch realization of PufferLib's native linear-MinGRU policy.

The parameter layout matches ``policy_weights_create`` in ``src/models.cu``:
encoder weight, fused decoder weight, then one MinGRU matrix per layer.  This
is the layout used by native flat ``.bin`` checkpoints.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class NativePufferPolicy(torch.nn.Module):
    """Bias-free PufferLib policy with native flat-checkpoint compatibility."""

    def __init__(
        self,
        observation_size: int,
        act_sizes: tuple[int, ...] | list[int],
        *,
        hidden_size: int = 256,
        num_layers: int = 2,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.act_sizes = tuple(int(size) for size in act_sizes)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        if self.observation_size <= 0 or self.hidden_size <= 0:
            raise ValueError("observation_size and hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not self.act_sizes or any(size <= 1 for size in self.act_sizes):
            raise ValueError("act_sizes must contain discrete heads of size > 1")
        self.action_size = sum(self.act_sizes)

        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("NativePufferPolicy requires a CUDA device")

        self.encoder_weight = torch.nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.observation_size,
                dtype=torch.float32,
                device=requested_device,
            )
        )
        self.decoder_weight = torch.nn.Parameter(
            torch.empty(
                self.action_size + 1,
                self.hidden_size,
                dtype=torch.float32,
                device=requested_device,
            )
        )
        self.mingru_weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.empty(
                        3 * self.hidden_size,
                        self.hidden_size,
                        dtype=torch.float32,
                        device=requested_device,
                    )
                )
                for _ in range(self.num_layers)
            ]
        )
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.encoder_weight.device

    @property
    def native_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.native_parameters())

    @property
    def native_checkpoint_bytes(self) -> int:
        return 4 * self.native_parameter_count

    def native_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return (
            self.encoder_weight,
            self.decoder_weight,
            *tuple(self.mingru_weights),
        )

    def native_layout(self) -> tuple[dict[str, Any], ...]:
        names = ("encoder.weight", "decoder.weight") + tuple(
            f"mingru.weights.{index}" for index in range(self.num_layers)
        )
        offset = 0
        result = []
        for name, parameter in zip(names, self.native_parameters(), strict=True):
            count = parameter.numel()
            result.append(
                {
                    "name": name,
                    "shape": tuple(parameter.shape),
                    "float_offset": offset,
                    "float_count": count,
                }
            )
            offset += count
        return tuple(result)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_normal_(self.encoder_weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.decoder_weight, nonlinearity="linear")
        for weight in self.mingru_weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="linear")

    @staticmethod
    def _g(value: torch.Tensor) -> torch.Tensor:
        return torch.where(value >= 0, value + 0.5, value.sigmoid())

    @staticmethod
    def _log_g(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            value >= 0,
            torch.log(F.relu(value) + 0.5),
            -F.softplus(-value),
        )

    @staticmethod
    def _highway(
        original: torch.Tensor,
        recurrent: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        gate = projection.sigmoid()
        return gate * recurrent + (1.0 - gate) * original

    @staticmethod
    def _heinsen_scan(
        log_coefficients: torch.Tensor,
        log_values: torch.Tensor,
    ) -> torch.Tensor:
        accumulated = log_coefficients.cumsum(dim=1)
        return (
            accumulated
            + (log_values - accumulated).logcumsumexp(dim=1)
        ).exp()

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor]:
        state_device = self.device if device is None else torch.device(device)
        if state_device != self.device:
            raise ValueError("policy state must use the policy CUDA device")
        return (
            torch.zeros(
                self.num_layers,
                int(batch_size),
                self.hidden_size,
                dtype=torch.float32,
                device=state_device,
            ),
        )

    def _decode(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = F.linear(hidden, self.decoder_weight)
        logits = decoded[..., : self.action_size]
        values = decoded[..., self.action_size]
        return logits, values

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError("eval observations must be [agents, observation_size]")
        if observations.device != self.device or observations.dtype != torch.float32:
            raise ValueError("eval observations must be float32 on the policy CUDA device")
        if len(state) != 1 or state[0].shape != (
            self.num_layers,
            observations.shape[0],
            self.hidden_size,
        ):
            raise ValueError("invalid MinGRU evaluation state")

        hidden = F.linear(observations, self.encoder_weight)
        recurrent_state = state[0]
        next_state = []
        for index, weight in enumerate(self.mingru_weights):
            candidate, gate, projection = F.linear(hidden, weight).chunk(3, dim=-1)
            recurrent = torch.lerp(
                recurrent_state[index],
                self._g(candidate),
                gate.sigmoid(),
            )
            hidden = self._highway(hidden, recurrent, projection)
            next_state.append(recurrent)
        logits, values = self._decode(hidden)
        return logits, values, (torch.stack(next_state, dim=0),)

    def forward_train(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 3 or observations.shape[2] != self.observation_size:
            raise ValueError(
                "training observations must be [segments, horizon, observation_size]"
            )
        if observations.device != self.device or observations.dtype != torch.float32:
            raise ValueError("training observations must be float32 on policy CUDA device")

        hidden = F.linear(observations, self.encoder_weight)
        for weight in self.mingru_weights:
            candidate, gate, projection = F.linear(hidden, weight).chunk(3, dim=-1)
            log_coefficients = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(candidate)
            recurrent = self._heinsen_scan(log_coefficients, log_values)
            hidden = self._highway(hidden, recurrent, projection)
        return self._decode(hidden)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_train(observations)

    def flat_native(self) -> torch.Tensor:
        """Return a CUDA flat tensor in native checkpoint order."""
        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in self.native_parameters()]
        )

    def load_native_checkpoint(self, path: str | os.PathLike[str]) -> str:
        """Load one native flat checkpoint and return its SHA-256 digest."""
        checkpoint = Path(path)
        payload = checkpoint.read_bytes()
        if len(payload) != self.native_checkpoint_bytes:
            raise ValueError(
                f"checkpoint size mismatch: expected {self.native_checkpoint_bytes} "
                f"bytes, got {len(payload)}"
            )
        host = torch.frombuffer(bytearray(payload), dtype=torch.float32)
        device_flat = host.to(device=self.device, non_blocking=False)
        offset = 0
        with torch.no_grad():
            for parameter in self.native_parameters():
                count = parameter.numel()
                parameter.copy_(device_flat[offset : offset + count].view_as(parameter))
                offset += count
        return hashlib.sha256(payload).hexdigest()

    def save_native_checkpoint(self, path: str | os.PathLike[str]) -> str:
        """Write the native flat format at an explicit checkpoint boundary."""
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = self.flat_native().to(device="cpu").contiguous().numpy().tobytes()
        if len(payload) != self.native_checkpoint_bytes:
            raise RuntimeError("serialized native checkpoint has an invalid size")
        temporary = checkpoint.with_name(checkpoint.name + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, checkpoint)
        return hashlib.sha256(payload).hexdigest()
