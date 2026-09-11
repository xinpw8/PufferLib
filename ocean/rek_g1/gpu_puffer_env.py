"""CUDA-resident environment boundary for the REK G1 Puffer policy.

The wrapped environment owns stable observation, reward, terminal, and action
mask tensors.  Policy actions enter ``step`` on the same CUDA device.  Neither
this adapter nor its metric hook copies tensors to the host while stepping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch


class GpuMetricPlugin(Protocol):
    """Metric hook whose update stays on device and snapshot may synchronize."""

    def reset(self) -> None: ...

    def update(
        self,
        observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> None: ...

    def snapshot(self, *, clear: bool = True) -> Mapping[str, float]: ...


class CudaTensorEnvironment(Protocol):
    """Minimum in-place environment interface consumed by the adapter."""

    observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    action_mask: torch.Tensor

    def reset(self) -> None: ...

    def step(self, actions: torch.Tensor) -> None: ...


class CudaTensorEnvAdapter:
    """Validate and expose a stable, CUDA-only Puffer environment interface.

    The environment must enqueue all work on the current PyTorch CUDA stream,
    or establish equivalent stream dependencies internally.  Buffers are
    mutated in place. Replacing a buffer during reset or step is rejected.
    """

    def __init__(
        self,
        env: CudaTensorEnvironment,
        act_sizes: Sequence[int],
        *,
        metric_plugins: Sequence[GpuMetricPlugin] = (),
    ) -> None:
        self.env = env
        self.act_sizes = tuple(int(size) for size in act_sizes)
        if not self.act_sizes or any(size <= 1 for size in self.act_sizes):
            raise ValueError("act_sizes must contain discrete heads of size > 1")

        self.observations = env.observations
        self.rewards = env.rewards
        self.terminals = env.terminals
        self.action_mask = env.action_mask
        self.metric_plugins = tuple(metric_plugins)

        self._validate_buffers()
        self.total_agents = self.observations.shape[0]
        self.obs_size = self.observations.shape[1]
        self.num_atns = len(self.act_sizes)
        self.action_mask_size = sum(self.act_sizes)
        self.device = self.observations.device
        self._buffer_ids = self._current_buffer_ids()

        for plugin in self.metric_plugins:
            plugin_device = getattr(plugin, "device", self.device)
            plugin_device = torch.device(plugin_device)
            if plugin_device.type != "cuda" or (
                plugin_device.index is not None
                and plugin_device.index != self.device.index
            ):
                raise ValueError("metric plugin must use the environment CUDA device")

    def _current_buffer_ids(self) -> tuple[int, int, int, int]:
        return tuple(
            id(value)
            for value in (
                self.env.observations,
                self.env.rewards,
                self.env.terminals,
                self.env.action_mask,
            )
        )

    def _validate_buffers(self) -> None:
        observations = self.observations
        rewards = self.rewards
        terminals = self.terminals
        action_mask = self.action_mask

        for name, value in (
            ("observations", observations),
            ("rewards", rewards),
            ("terminals", terminals),
            ("action_mask", action_mask),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device.type != "cuda":
                raise ValueError(f"{name} must be CUDA-resident")
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        if observations.ndim != 2 or observations.dtype != torch.float32:
            raise ValueError("observations must be contiguous CUDA float32 [agents, obs]")
        total_agents = observations.shape[0]
        if total_agents <= 0 or observations.shape[1] <= 0:
            raise ValueError("observations must have positive dimensions")
        if rewards.shape != (total_agents,) or rewards.dtype != torch.float32:
            raise ValueError("rewards must be contiguous CUDA float32 [agents]")
        if terminals.shape != (total_agents,) or terminals.dtype != torch.float32:
            raise ValueError("terminals must be contiguous CUDA float32 [agents]")
        if action_mask.shape != (total_agents, sum(self.act_sizes)):
            raise ValueError(
                "action_mask must be [agents, sum(act_sizes)] on the CUDA device"
            )
        if action_mask.dtype not in (torch.bool, torch.uint8):
            raise TypeError("action_mask must use bool or uint8")
        if not (
            observations.device
            == rewards.device
            == terminals.device
            == action_mask.device
        ):
            raise ValueError("all environment buffers must use one CUDA device")

    def _assert_stable_buffers(self) -> None:
        if self._current_buffer_ids() != self._buffer_ids:
            raise RuntimeError(
                "CUDA environment replaced a buffer; reset and step must mutate in place"
            )

    def reset(self, *, reset_metrics: bool = True) -> None:
        result = self.env.reset()
        if result is not None:
            raise RuntimeError("CUDA environment reset must mutate buffers and return None")
        self._assert_stable_buffers()
        if reset_metrics:
            for plugin in self.metric_plugins:
                plugin.reset()

    def step(self, actions: torch.Tensor) -> None:
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.shape != (self.total_agents, self.num_atns):
            raise ValueError(
                f"actions must have shape ({self.total_agents}, {self.num_atns})"
            )
        if actions.dtype not in (torch.int32, torch.int64):
            raise TypeError("categorical actions must use int32 or int64")
        if actions.device != self.device or not actions.is_contiguous():
            raise ValueError("actions must be contiguous on the environment CUDA device")

        result = self.env.step(actions)
        if result is not None:
            raise RuntimeError("CUDA environment step must mutate buffers and return None")
        self._assert_stable_buffers()
        for plugin in self.metric_plugins:
            plugin.update(self.observations, self.terminals)

    def log(self, *, clear_metrics: bool = True) -> dict[str, Any]:
        """Collect host-visible results at an explicit reporting boundary."""
        result: dict[str, Any] = {}
        env_log = getattr(self.env, "log", None)
        if env_log is not None:
            values = env_log()
            if values is not None:
                result.update(dict(values))

        for plugin in self.metric_plugins:
            values = dict(plugin.snapshot(clear=clear_metrics))
            overlap = result.keys() & values.keys()
            if overlap:
                names = ", ".join(sorted(overlap))
                raise RuntimeError(f"duplicate environment metric keys: {names}")
            result.update(values)
        return result

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def make_rek_g1_cuda_env(
    env: CudaTensorEnvironment,
    *,
    additional_metric_plugins: Sequence[GpuMetricPlugin] = (),
) -> CudaTensorEnvAdapter:
    """Attach the canonical completed-round metric collector to a G1 CUDA env."""
    try:
        from .gpu_metrics import RekG1GpuMetricCollector
    except ImportError:
        from gpu_metrics import RekG1GpuMetricCollector

    observations = env.observations
    if not isinstance(observations, torch.Tensor) or observations.ndim != 2:
        raise ValueError("REK G1 observations must be a rank-two tensor")
    collector = RekG1GpuMetricCollector(
        observations.shape[0],
        observations.device,
    )
    return CudaTensorEnvAdapter(
        env,
        (33,),
        metric_plugins=(collector, *tuple(additional_metric_plugins)),
    )
