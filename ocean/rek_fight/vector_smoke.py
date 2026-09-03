#!/usr/bin/env python3
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import time

import torch


def load_local_extension():
    candidates = sorted((Path.cwd() / "pufferlib").glob("_C*.so"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one local extension, found {candidates}")
    spec = importlib.util.spec_from_file_location("_C", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, candidates[0]


_C, EXTENSION_PATH = load_local_extension()


class CudaPtr:
    def __init__(self, pointer, shape):
        self.__cuda_array_interface__ = {
            "data": (pointer, False),
            "shape": shape,
            "typestr": "<f4",
            "version": 2,
        }


def cpu_tensor(pointer, shape):
    count = 1
    for extent in shape:
        count *= extent
    array = (ctypes.c_float * count).from_address(pointer)
    return torch.frombuffer(array, dtype=torch.float32).reshape(shape)


def make_actions(total_agents, device):
    actions = torch.ones(
        (total_agents, 4), dtype=torch.float32, device=device
    )
    actions[:, 3] = 0.0
    actions[0::2, 0] = 2.0
    actions[1::2, 0] = 0.0
    actions[0::4, 2] = 2.0
    actions[1::4, 2] = 0.0
    actions[0::8, 3] = 1.0
    actions[4::8, 3] = 4.0
    return actions.contiguous()


def run(mode, total_agents, num_threads, steps, warmup):
    gpu = mode == "gpu"
    args = {
        "vec": {
            "total_agents": total_agents,
            "num_buffers": 1,
            "num_threads": num_threads,
        },
        "env": {
            "max_steps": 1500,
            "fall_height": 0.5,
            "fall_up_z": 0.5,
            "root_stabilizer_scale": 1.0,
        },
    }
    vec = _C.create_vec(args, int(gpu))
    try:
        if _C.env_name != "rek_fight":
            raise RuntimeError(f"unexpected env: {_C.env_name}")
        if _C.precision_bytes != 4:
            raise RuntimeError(
                f"training requires float32, precision_bytes={_C.precision_bytes}"
            )
        if vec.obs_size != 173 or vec.num_atns != 4:
            raise RuntimeError(
                f"unexpected schema: obs={vec.obs_size}, atns={vec.num_atns}"
            )
        if list(vec.act_sizes) != [3, 3, 3, 7]:
            raise RuntimeError(f"unexpected action sizes: {vec.act_sizes}")

        device = "cuda" if gpu else "cpu"
        if gpu:
            observations = torch.as_tensor(
                CudaPtr(vec.gpu_obs_ptr, (total_agents, vec.obs_size))
            )
            rewards = torch.as_tensor(
                CudaPtr(vec.gpu_rewards_ptr, (total_agents,))
            )
            terminals = torch.as_tensor(
                CudaPtr(vec.gpu_terminals_ptr, (total_agents,))
            )
        else:
            observations = cpu_tensor(
                vec.obs_ptr, (total_agents, vec.obs_size)
            )
            rewards = cpu_tensor(vec.rewards_ptr, (total_agents,))
            terminals = cpu_tensor(vec.terminals_ptr, (total_agents,))

        actions = make_actions(total_agents, device)
        step = vec.gpu_step if gpu else vec.cpu_step
        vec.reset()
        for _ in range(warmup):
            step(actions.data_ptr())
        if gpu:
            torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(steps):
            step(actions.data_ptr())
        if gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        finite = bool(
            torch.isfinite(observations).all().item()
            and torch.isfinite(rewards).all().item()
            and torch.isfinite(terminals).all().item()
        )
        terminal_domain = bool(
            ((terminals == 0.0) | (terminals == 1.0)).all().item()
        )
        if not finite or not terminal_domain:
            raise RuntimeError(
                f"invalid output: finite={finite}, terminal_domain={terminal_domain}"
            )

        matches = total_agents // 2
        return {
            "mode": mode,
            "steps": steps,
            "total_agents": total_agents,
            "matches": matches,
            "num_threads": num_threads,
            "elapsed_seconds": elapsed,
            "vector_steps_per_second": steps / elapsed,
            "match_steps_per_second": steps * matches / elapsed,
            "agent_steps_per_second": steps * total_agents / elapsed,
            "obs_size": vec.obs_size,
            "num_atns": vec.num_atns,
            "act_sizes": list(vec.act_sizes),
            "outputs_finite": finite,
            "terminal_domain_valid": terminal_domain,
            "cuda_device": torch.cuda.get_device_name() if gpu else None,
        }
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--total-agents", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument(
        "--mode", choices=("cpu", "gpu", "both"), default="both"
    )
    args = parser.parse_args()
    if args.total_agents <= 0 or args.total_agents % 2:
        parser.error("--total-agents must be positive and even")
    if args.steps <= 0 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup non-negative")

    modes = ("cpu", "gpu") if args.mode == "both" else (args.mode,)
    results = [
        run(
            mode,
            args.total_agents,
            args.num_threads,
            args.steps,
            args.warmup,
        )
        for mode in modes
    ]
    print(
        json.dumps(
            {
                "schema": "rek_fight.vector_smoke.v1",
                "compiled_env": _C.env_name,
                "compiled_with_cuda": bool(_C.gpu),
                "precision_bytes": _C.precision_bytes,
                "extension_path": str(EXTENSION_PATH.resolve()),
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
