"""Native PufferLib runtime for a caller-owned CUDA environment.

This is the production training bridge. The policy, rollout storage,
prioritized replay, PPO loss, Muon optimizer, and checkpoint format remain in
``pufferlib._C``. Python drives only the external environment boundary. Every
hot-path buffer is CUDA-resident and every operation is enqueued without a
per-step host transfer or device synchronization.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .gpu_puffer_env import CudaTensorEnvAdapter, make_rek_g1_cuda_env
except ImportError:
    from gpu_puffer_env import CudaTensorEnvAdapter, make_rek_g1_cuda_env


_REQUIRED_NATIVE_API = (
    "create_pufferl",
    "external_rollout_begin",
    "external_rollout_step",
    "external_actions_to_int32",
    "external_rollout_finish",
    "train",
    "log",
    "eval_log",
    "save_weights",
    "load_weights",
    "close",
)


def _native_backend():
    try:
        from pufferlib import _C
    except ImportError as error:
        raise RuntimeError(
            "the compiled native PufferLib CUDA backend is required"
        ) from error
    return _C


def _resolved_cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("external native training requires a CUDA environment")
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


class NativeExternalGpuPuffer:
    """Drive native Puffer PPO against stable caller-owned CUDA tensors."""

    def __init__(
        self,
        args: dict[str, Any],
        env: CudaTensorEnvAdapter,
        *,
        backend: Any | None = None,
        reward_clip: float | None = None,
        reset_environment: bool = True,
    ) -> None:
        if not isinstance(env, CudaTensorEnvAdapter):
            raise TypeError("env must be a validated CudaTensorEnvAdapter")
        self.env = env
        self.backend = backend if backend is not None else _native_backend()
        missing = [name for name in _REQUIRED_NATIVE_API if not hasattr(self.backend, name)]
        if missing:
            raise RuntimeError(
                "native PufferLib backend lacks external GPU API: "
                + ", ".join(missing)
            )

        self.args = deepcopy(args)
        vec = self.args.setdefault("vec", {})
        train = self.args.setdefault("train", {})
        configured_agents = int(vec.get("total_agents", env.total_agents))
        if configured_agents != env.total_agents:
            raise ValueError(
                f"vec.total_agents={configured_agents} does not match "
                f"CUDA environment rows={env.total_agents}"
            )
        configured_buffers = int(vec.get("num_buffers", 1))
        if configured_buffers != 1:
            raise ValueError("external GPU training currently requires vec.num_buffers=1")

        device_index = _resolved_cuda_index(env.device)
        configured_gpu = int(self.args.get("gpu_id", device_index))
        if configured_gpu != device_index:
            raise ValueError(
                f"gpu_id={configured_gpu} does not match environment cuda:{device_index}"
            )

        clip = 0.0 if reward_clip is None else float(reward_clip)
        if clip < 0.0 or clip != clip or clip == float("inf"):
            raise ValueError("reward_clip must be finite and nonnegative")

        vec.update(
            {
                "total_agents": env.total_agents,
                "num_buffers": 1,
                "num_threads": 0,
                "external_gpu": 1,
                "external_observations_ptr": env.observations.data_ptr(),
                "external_rewards_ptr": env.rewards.data_ptr(),
                "external_terminals_ptr": env.terminals.data_ptr(),
                "external_action_mask_ptr": env.action_mask.data_ptr(),
            }
        )
        train["reward_clip"] = clip
        self.args["gpu_id"] = device_index
        self.args.setdefault("world_size", 1)
        self.args.setdefault("rank", 0)
        self.args.setdefault("nccl_id", b"")

        if reset_environment:
            env.reset()
        # Native init-time graph warmup reads these external buffers from its
        # own streams. Establish their initial contents once before binding.
        torch.cuda.synchronize(env.device)
        self.pufferl = self.backend.create_pufferl(self.args)
        self._validate_native_layout()
        self.stream = torch.cuda.Stream(device=env.device)
        self.actions = torch.empty(
            (env.total_agents, env.num_atns),
            dtype=torch.int32,
            device=env.device,
        )
        self.reward_clip = clip
        self._closed = False

    def _validate_native_layout(self) -> None:
        native = self.pufferl
        if not bool(native.external_gpu):
            raise RuntimeError("native backend did not enter external GPU mode")
        if int(native.total_agents) != self.env.total_agents:
            raise RuntimeError("native agent count does not match CUDA environment")
        if int(native.obs_size) != self.env.obs_size:
            raise RuntimeError("compiled native observation size does not match environment")
        if tuple(int(value) for value in native.act_sizes) != self.env.act_sizes:
            raise RuntimeError("compiled native action heads do not match environment")
        if int(native.num_atns) != self.env.num_atns:
            raise RuntimeError("compiled native action count does not match environment")
        native_mask = int(native.action_mask_size)
        if native_mask not in (0, self.env.action_mask_size):
            raise RuntimeError("compiled native action mask size does not match environment")
        if int(native.native_env_count) != 0 or bool(native.has_env_threads):
            raise RuntimeError("external native mode allocated CPU environments or workers")
        if int(native.gpu_observations_ptr) != self.env.observations.data_ptr():
            raise RuntimeError("native observation pointer is not the caller-owned buffer")
        if int(native.gpu_rewards_ptr) != self.env.rewards.data_ptr():
            raise RuntimeError("native reward pointer is not the caller-owned buffer")
        if int(native.gpu_terminals_ptr) != self.env.terminals.data_ptr():
            raise RuntimeError("native terminal pointer is not the caller-owned buffer")
        if native_mask and int(native.gpu_action_mask_ptr) != self.env.action_mask.data_ptr():
            raise RuntimeError("native action-mask pointer is not the caller-owned buffer")

        compiled_env = getattr(self.backend, "env_name", None)
        requested_env = self.args.get("env_name")
        if compiled_env is not None and requested_env != compiled_env:
            raise RuntimeError(
                f"native backend was compiled for {compiled_env}, not {requested_env}"
            )

    def _current_stream_pointer(self) -> int:
        return int(self.stream.cuda_stream)

    def rollouts(self) -> None:
        """Collect one native rollout horizon without leaving the CUDA device."""
        if self._closed:
            raise RuntimeError("native external trainer is closed")
        caller_stream = torch.cuda.current_stream(self.env.device)
        self.stream.wait_stream(caller_stream)
        stream = self._current_stream_pointer()
        with torch.cuda.stream(self.stream):
            self.backend.external_rollout_begin(self.pufferl, stream)
            for step in range(int(self.pufferl.hypers.horizon)):
                self.backend.external_rollout_step(self.pufferl, step, stream)
                self.backend.external_actions_to_int32(
                    self.pufferl, self.actions.data_ptr(), stream
                )
                self.env.step(self.actions)
            self.backend.external_rollout_finish(self.pufferl, stream)
        caller_stream.wait_stream(self.stream)

    def train(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("native external trainer is closed")
        caller_stream = torch.cuda.current_stream(self.env.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            result = dict(self.backend.train(self.pufferl))
        caller_stream.wait_stream(self.stream)
        return result

    def log(self, *, clear_metrics: bool = True) -> dict[str, Any]:
        result = dict(self.backend.log(self.pufferl))
        result["env"] = self.env.log(clear_metrics=clear_metrics)
        result["reward_transform"] = (
            "none" if self.reward_clip == 0.0 else "symmetric_clamp"
        )
        result["reward_clip"] = self.reward_clip
        return result

    def eval_log(self, *, clear_metrics: bool = True) -> dict[str, Any]:
        result = dict(self.backend.eval_log(self.pufferl))
        result["env"] = self.env.log(clear_metrics=clear_metrics)
        result["reward_transform"] = (
            "none" if self.reward_clip == 0.0 else "symmetric_clamp"
        )
        result["reward_clip"] = self.reward_clip
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "rek-g1-native-external-gpu-v1",
            "backend": "pufferlib-native-cuda",
            "environment": {
                "compiled_name": getattr(self.backend, "env_name", None),
                "total_agents": self.env.total_agents,
                "observation_size": self.env.obs_size,
                "action_sizes": list(self.env.act_sizes),
                "action_mask_size": int(self.pufferl.action_mask_size),
                "cuda_device": str(self.env.device),
                "external_buffers": True,
                "native_cpu_environment_count": int(self.pufferl.native_env_count),
                "native_environment_threads": bool(self.pufferl.has_env_threads),
            },
            "learner": {
                "native_policy": True,
                "native_prioritized_replay": True,
                "native_ppo": True,
                "native_muon": True,
                "native_checkpoint_layout": True,
            },
            "reward": {
                "transform": (
                    "none" if self.reward_clip == 0.0 else "symmetric_clamp"
                ),
                "clip_magnitude": self.reward_clip,
                "legacy_clipping_enabled": self.reward_clip > 0.0,
            },
        }

    def save_weights(self, path: str | Path) -> dict[str, Any]:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save_weights(self.pufferl, str(checkpoint))
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest = self.manifest()
        manifest["checkpoint"] = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": digest,
            "global_step": int(self.pufferl.global_step),
        }
        manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def load_weights(self, path: str | Path) -> None:
        self.backend.load_weights(self.pufferl, str(Path(path)))

    def num_params(self) -> int:
        return int(self.pufferl.num_params())

    @property
    def global_step(self) -> int:
        return int(self.pufferl.global_step)

    @property
    def epoch(self) -> int:
        return int(self.pufferl.epoch)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.backend.close(self.pufferl)
        finally:
            self.env.close()
            self._closed = True


def create_rek_g1_native_puffer(
    args: dict[str, Any],
    env: Any,
    **kwargs: Any,
) -> NativeExternalGpuPuffer:
    """Attach canonical REK metrics and construct the native CUDA trainer."""
    return NativeExternalGpuPuffer(args, make_rek_g1_cuda_env(env), **kwargs)
