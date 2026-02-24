## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import io
import sys
import glob
import ast
import time
import random
import shutil
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque
import multiprocessing as mp
from copy import deepcopy

import numpy as np

import torch
from torch import func
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import PufferLib C++ backend. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

LN10_BY_400 = np.log(10) / 400.0

def _torch_load(path, map_location='cpu', allow_pickle=False):
    if allow_pickle:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't expose weights_only.
            return torch.load(path, map_location=map_location)
    return torch.load(path, map_location=map_location)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def _get_trained_state_dict(policy_fp32, muon):
    """Read trained weights from Muon's contiguous weight buffer.

    Muon's init_contiguous_weights() maps parameters into a flat buffer via set_data().
    We read directly from weight_buffer for robustness, using named_parameters()
    only for the names and shapes (which are always correct).
    """
    weight_buffer = muon.weight_buffer
    state_dict = {}
    offset = 0
    for name, param in policy_fp32.named_parameters():
        size = param.numel()
        state_dict[name] = weight_buffer.narrow(0, offset, size).view(param.shape).detach().clone()
        offset += size
    return state_dict

class SelfplayManager:
    """Manages opponent pool, quality-weighted sampling, ELO, and swap tracking for selfplay."""
    def __init__(self, pufferl_cpp, total_agents, num_buffers, n_slots=6):
        self.pufferl_cpp = pufferl_cpp
        self.total_agents = int(total_agents)
        self.num_buffers = int(num_buffers)
        self.block_size = self.total_agents // self.num_buffers
        self.cut = int(0.8 * self.block_size)
        self.tail = self.block_size - self.cut
        self.n_slots = n_slots
        self._rot_idx = 0

        # Opponent pool storage (CPU state dicts)
        self.opponent_pool = {}           # policy_id -> CPU state_dict
        self.opponent_pool_ids = []
        self.saved_policy_count = 0
        self.max_opponent_history = 300

        # ELO tracking
        self.elos = [1000.0]  # index 0 = learner, rest = opponents (baseline 1000)

        # Quality-weighted sampling
        self.opponent_qualities = {}
        self.quality_lr = 0.01

        # Swap tracking (epoch-level)
        self.cur_opp_id = 0
        self.done_since_swap = 0
        self.swap_quota = int(total_agents * 0.2)

    def save_snapshot(self, policy_fp32, muon, epoch):
        """Save current policy to opponent pool. Called every 10 epochs."""
        if epoch == 0 or epoch % 10 != 0:
            return
        snapshot = {k: v.cpu() for k, v in _get_trained_state_dict(policy_fp32, muon).items()}
        self.saved_policy_count += 1
        pid = self.saved_policy_count
        self.opponent_pool[pid] = snapshot
        start_q = max(self.opponent_qualities.values()) if self.opponent_qualities else 0.0
        self.opponent_qualities[pid] = start_q
        self.opponent_pool_ids.append(pid)
        self.elos.append(self.elos[0])
        # Evict oldest
        if len(self.opponent_pool_ids) > self.max_opponent_history:
            oldest = self.opponent_pool_ids.pop(0)
            self.opponent_pool.pop(oldest, None)

    def sample_opponent(self):
        """Epsilon-greedy opponent sampling: 30% uniform random, 70% quality-weighted."""
        if not self.opponent_pool_ids:
            return 0
        ids = np.array(self.opponent_pool_ids, dtype=np.int32)
        if np.random.random() < 0.3:
            return int(np.random.choice(ids))
        qs = np.array([self.opponent_qualities.get(p, 0.0) for p in ids])
        probs = softmax(qs)
        return int(np.random.choice(ids, p=probs))

    def load_opponent(self, policy_id):
        """Load opponent weights into C++ pool and activate."""
        if policy_id == 0 or policy_id not in self.opponent_pool:
            return
        slot = self._rot_idx
        weights = self.opponent_pool[policy_id]
        _C.load_opponent_weights(self.pufferl_cpp, weights, slot, int(policy_id))
        _C.set_active_opponent(self.pufferl_cpp, slot, policy_id)
        self.cur_opp_id = policy_id
        self._rot_idx = (slot + 1) % self.n_slots

    def state_dict(self):
        return {
            'total_agents': int(self.total_agents),
            'num_buffers': int(self.num_buffers),
            'block_size': int(self.block_size),
            'cut': int(self.cut),
            'tail': int(self.tail),
            'n_slots': int(self.n_slots),
            'rot_idx': int(self._rot_idx),
            'saved_policy_count': int(self.saved_policy_count),
            'max_opponent_history': int(self.max_opponent_history),
            'opponent_pool_ids': [int(pid) for pid in self.opponent_pool_ids],
            'opponent_qualities': {int(k): float(v) for k, v in self.opponent_qualities.items()},
            'elos': [float(v) for v in self.elos],
            'quality_lr': float(self.quality_lr),
            'cur_opp_id': int(self.cur_opp_id),
            'done_since_swap': int(self.done_since_swap),
            'swap_quota': int(self.swap_quota),
            'slot_policy_ids': [int(v) for v in _C.get_opponent_slot_policy_ids(self.pufferl_cpp)],
            'active_opponent_slot': int(self.pufferl_cpp.active_opponent_slot),
            'opponent_pool': {
                int(pid): {k: v.detach().cpu() for k, v in weights.items()}
                for pid, weights in self.opponent_pool.items()
            },
        }

    def load_state_dict(self, state):
        if not state:
            return

        self._rot_idx = int(state.get('rot_idx', self._rot_idx))
        self.saved_policy_count = int(state.get('saved_policy_count', self.saved_policy_count))
        self.max_opponent_history = int(state.get('max_opponent_history', self.max_opponent_history))
        self.opponent_pool_ids = [int(pid) for pid in state.get('opponent_pool_ids', self.opponent_pool_ids)]
        self.opponent_qualities = {
            int(k): float(v) for k, v in state.get('opponent_qualities', self.opponent_qualities).items()
        }
        self.elos = [float(v) for v in state.get('elos', self.elos)]
        self.quality_lr = float(state.get('quality_lr', self.quality_lr))
        self.cur_opp_id = int(state.get('cur_opp_id', self.cur_opp_id))
        self.done_since_swap = int(state.get('done_since_swap', self.done_since_swap))
        self.swap_quota = int(state.get('swap_quota', self.swap_quota))

        restored_pool = {}
        for pid, weights in state.get('opponent_pool', {}).items():
            restored_pool[int(pid)] = {k: v.cpu() for k, v in weights.items()}
        self.opponent_pool = restored_pool

        slot_policy_ids = [int(v) for v in state.get('slot_policy_ids', [])]
        if slot_policy_ids:
            for slot, policy_id in enumerate(slot_policy_ids[:self.n_slots]):
                if policy_id <= 0:
                    continue
                weights = self.opponent_pool.get(policy_id)
                if weights is None:
                    continue
                _C.load_opponent_weights(self.pufferl_cpp, weights, slot, policy_id)

            active_slot = int(state.get('active_opponent_slot', -1))
            if 0 <= active_slot < len(slot_policy_ids):
                _C.set_active_opponent(self.pufferl_cpp, active_slot, int(slot_policy_ids[active_slot]))
                self.cur_opp_id = int(slot_policy_ids[active_slot])
            elif self.cur_opp_id > 0:
                self.load_opponent(self.cur_opp_id)

    def update_from_rollout(self, rollouts):
        """Process rollout terminals/rewards for swap tracking and ELO updates.
        Called once per epoch after evaluate()."""
        terminals = rollouts.terminals.cpu()  # [horizon, total_agents]
        rewards = rollouts.rewards.cpu()      # [horizon, total_agents]

        total_tail_done = 0
        total_tail_wins = 0
        total_tail_losses = 0

        for buf in range(self.num_buffers):
            start = buf * self.block_size + self.cut
            end = (buf + 1) * self.block_size
            tail_terms = terminals[:, start:end]
            tail_rewards = rewards[:, start:end]
            done_mask = tail_terms > 0.5
            n_done = done_mask.sum().item()
            total_tail_done += n_done
            total_tail_wins += ((tail_rewards > 0) & done_mask).sum().item()
            total_tail_losses += ((tail_rewards < 0) & done_mask).sum().item()

        # ELO update (only from decisive games: wins+losses, excluding draws)
        decisive = total_tail_wins + total_tail_losses
        if decisive > 0 and self.cur_opp_id > 0:
            elos = np.array(self.elos, dtype=np.float64)
            score = total_tail_wins / decisive
            r1 = elos[0]
            # Find opponent ELO index
            if self.cur_opp_id < len(elos):
                r2 = elos[self.cur_opp_id]
            else:
                r2 = r1
            expected = 1.0 / (1.0 + np.exp((r2 - r1) * LN10_BY_400))
            k = 32.0
            delta = k * (score - expected)
            elos[0] += delta
            if self.cur_opp_id < len(elos):
                elos[self.cur_opp_id] -= delta
            self.elos = elos.tolist()

        # Quality update: decrease quality of opponents we beat
        if total_tail_wins > 0 and self.cur_opp_id in self.opponent_qualities:
            pool_ids = np.array(self.opponent_pool_ids)
            qs = np.array([self.opponent_qualities.get(pid, 0.0) for pid in pool_ids])
            probs = softmax(qs)
            try:
                idx = np.where(pool_ids == self.cur_opp_id)[0][0]
                pi = probs[idx]
            except IndexError:
                pi = 1.0
            N = len(pool_ids)
            decrement = self.quality_lr / (N * max(pi, 1e-6))
            self.opponent_qualities[self.cur_opp_id] -= decrement * total_tail_wins

        # Swap tracking
        self.done_since_swap += total_tail_done
        if self.done_since_swap >= self.swap_quota and len(self.opponent_pool_ids) > 0:
            self.done_since_swap = 0
            new_id = self.sample_opponent()
            if new_id != 0:
                self.load_opponent(new_id)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME
)
# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)

# DEBUG FLAG IS A BUG. FUCK THIS DO NOT NOT NOT ENABLE
#torch.autograd.set_detect_anomaly(True)
#torch._dynamo.config.capture_scalar_outputs = True

class PuffeRL:
    def __init__(self, config, vec_config, env_config, policy_config, logger=None, verbose=True):
        # Reproducibility
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        minibatch_size = config['minibatch_size']
        horizon = config['horizon']
        total_agents = vec_config['total_agents']
        batch_size = horizon * total_agents
        self.batch_size = batch_size

        if (minibatch_size % horizon) != 0:
            raise pufferlib.APIUsageError(
                f'minibatch_size {minibatch_size} must be divisible by horizon {horizon}')

        if (minibatch_size > batch_size):
            minibatch_size = batch_size
            print(f'WARNING: minibatch_size {minibatch_size} > total_agents {total_agents} * horizon {horizon}. Reducing it for you.')

            #raise pufferlib.APIUsageError(f'minibatch_size {minibatch_size} must be <= '
            #    f'horizon {horizon} * total_agents {total_agents} ({batch_size})')

        # Logging
        self.logger = logger
        self.pufferl_cpp = _C.create_pufferl(config, vec_config, env_config, policy_config)
        self.rollouts = self.pufferl_cpp.rollouts

        # Selfplay
        self.selfplay = self.pufferl_cpp.selfplay
        self.selfplay_mgr = None
        if self.selfplay:
            n_slots = _C.get_opponent_pool_size(self.pufferl_cpp)
            self.selfplay_mgr = SelfplayManager(
                self.pufferl_cpp, vec_config['total_agents'],
                vec_config['num_buffers'], n_slots)

        # Initializations
        self.config = config
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.utilization = {}
        self.profile = defaultdict(float)
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}
        self.verbose = verbose

        self.policy_fp32 = self.pufferl_cpp.policy_fp32

        # Dashboard
        self.model_size = sum(p.numel() for p in self.policy_fp32.parameters() if p.requires_grad)
        self.start_time = time.time()
        self.print_dashboard(clear=True)

    def _sync_bf16_from_fp32(self):
        bf16 = self.pufferl_cpp.policy_bf16
        fp32 = self.pufferl_cpp.policy_fp32
        if bf16 is fp32:
            return
        with torch.no_grad():
            for p_bf16, p_fp32 in zip(bf16.parameters(), fp32.parameters()):
                p_bf16.data.copy_(p_fp32.data)

    def _load_named_weights_into_muon(self, state_dict):
        wb = self.pufferl_cpp.muon.weight_buffer
        offset = 0
        with torch.no_grad():
            for name, param in self.policy_fp32.named_parameters():
                if name not in state_dict:
                    raise KeyError(f"Missing parameter '{name}' in checkpoint")
                size = param.numel()
                src = state_dict[name].view(-1).to(device=wb.device, dtype=wb.dtype)
                wb.narrow(0, offset, size).copy_(src)
                offset += size
        self._sync_bf16_from_fp32()

    def _checkpoint_dir(self):
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], self.config["env"], run_id)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _save_full_trainer_state(self, run_id, model_name):
        muon_state = {}
        for k, v in self.pufferl_cpp.muon.state_dict().items():
            if torch.is_tensor(v):
                muon_state[k] = v.detach().cpu()
            else:
                muon_state[k] = v

        state = {
            'schema_version': 2,
            'run_id': run_id,
            'model_name': model_name,
            'global_step': int(self.global_step),
            'epoch': int(self.epoch),
            'last_log_step': int(self.last_log_step),
            'saved_time': float(time.time()),
            'pufferl_cpp_state': {
                'epoch': int(self.pufferl_cpp.epoch),
                'train_warmup': int(self.pufferl_cpp.train_warmup),
                'rng_seed': int(self.pufferl_cpp.rng_seed),
                'rng_offset': self.pufferl_cpp.rng_offset.detach().cpu(),
                'active_opponent_slot': int(self.pufferl_cpp.active_opponent_slot),
            },
            'muon_state': muon_state,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            'env_state': _C.get_env_state(self.pufferl_cpp),
        }
        if self.selfplay_mgr:
            state['selfplay_state'] = self.selfplay_mgr.state_dict()
        return state

    def _restore_full_trainer_state(self, state):
        muon_state = state.get('muon_state')
        if muon_state:
            with torch.no_grad():
                self.pufferl_cpp.muon.load_state_dict(muon_state)
            self._sync_bf16_from_fp32()
        elif 'policy_state_dict' in state:
            self._load_named_weights_into_muon(state['policy_state_dict'])

        self.global_step = int(state.get('global_step', self.global_step))
        self.epoch = int(state.get('epoch', self.epoch))
        self.last_log_step = int(state.get('last_log_step', self.global_step))
        self.last_log_time = time.time()
        self.start_time = time.time()

        cpp_state = state.get('pufferl_cpp_state', {})
        if 'epoch' in cpp_state:
            self.pufferl_cpp.epoch = int(cpp_state['epoch'])
        else:
            self.pufferl_cpp.epoch = int(self.epoch)
        if 'train_warmup' in cpp_state:
            self.pufferl_cpp.train_warmup = int(cpp_state['train_warmup'])
        if 'rng_seed' in cpp_state:
            self.pufferl_cpp.rng_seed = int(cpp_state['rng_seed'])
        if 'rng_offset' in cpp_state:
            with torch.no_grad():
                self.pufferl_cpp.rng_offset.copy_(
                    cpp_state['rng_offset'].to(
                        device=self.pufferl_cpp.rng_offset.device,
                        dtype=self.pufferl_cpp.rng_offset.dtype,
                    )
                )
        if 'active_opponent_slot' in cpp_state:
            self.pufferl_cpp.active_opponent_slot = int(cpp_state['active_opponent_slot'])

        rng_state = state.get('rng_state', {})
        if 'python' in rng_state:
            random.setstate(rng_state['python'])
        if 'numpy' in rng_state:
            np.random.set_state(rng_state['numpy'])
        if 'torch_cpu' in rng_state:
            torch.set_rng_state(rng_state['torch_cpu'])
        cuda_states = rng_state.get('torch_cuda_all')
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)

        env_state = state.get('env_state')
        if isinstance(env_state, dict) and env_state:
            _C.set_env_state(self.pufferl_cpp, env_state)

        if self.selfplay_mgr and 'selfplay_state' in state:
            self.selfplay_mgr.load_state_dict(state['selfplay_state'])

    def _restore_legacy_state(self, state_path, explicit_model_path=None):
        state = _torch_load(state_path, map_location='cpu', allow_pickle=True)
        model_name = state.get('model_name')

        if explicit_model_path is None:
            if model_name is None:
                raise RuntimeError(f"Legacy trainer state missing model_name: {state_path}")
            model_path = os.path.join(os.path.dirname(state_path), model_name)
        else:
            model_path = explicit_model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint model not found for resume: {model_path}")

        weights = _torch_load(model_path, map_location='cpu')
        self._load_named_weights_into_muon(weights)

        self.global_step = int(state.get('global_step', state.get('agent_step', self.global_step)))
        self.epoch = int(state.get('update', self.epoch))
        self.last_log_step = int(self.global_step)
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.pufferl_cpp.epoch = int(self.epoch)

    def _resolve_resume_target(self, load_path):
        if load_path is None:
            return None, None

        if load_path == 'latest':
            root = os.path.join(self.config['data_dir'], self.config["env"])
            full = glob.glob(os.path.join(root, '*/trainer_state_full.pt'))
            if full:
                latest = max(full, key=os.path.getmtime)
                return latest, 'full'
            models = glob.glob(os.path.join(root, '*/model_*.pt'))
            if models:
                latest_model = max(models, key=os.path.getmtime)
                sibling_full = os.path.join(os.path.dirname(latest_model), 'trainer_state_full.pt')
                if os.path.exists(sibling_full):
                    return sibling_full, 'full'
                sibling_legacy = os.path.join(os.path.dirname(latest_model), 'trainer_state.pt')
                if os.path.exists(sibling_legacy):
                    return sibling_legacy, 'legacy_model'
                return latest_model, 'model'
            raise FileNotFoundError(f'No checkpoints found under {root}')

        path = os.path.expanduser(load_path)
        if os.path.isdir(path):
            full = os.path.join(path, 'trainer_state_full.pt')
            if os.path.exists(full):
                return full, 'full'
            legacy = os.path.join(path, 'trainer_state.pt')
            if os.path.exists(legacy):
                return legacy, 'legacy'
            models = glob.glob(os.path.join(path, 'model_*.pt'))
            if models:
                return max(models, key=os.path.getmtime), 'model'
            raise FileNotFoundError(f'No checkpoint files found in {path}')

        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint path not found: {path}')

        basename = os.path.basename(path)
        if basename == 'trainer_state_full.pt':
            return path, 'full'
        if basename == 'trainer_state.pt':
            return path, 'legacy'
        if basename.startswith('model_') and basename.endswith('.pt'):
            sibling_full = os.path.join(os.path.dirname(path), 'trainer_state_full.pt')
            if os.path.exists(sibling_full):
                return sibling_full, 'full'
            sibling_legacy = os.path.join(os.path.dirname(path), 'trainer_state.pt')
            if os.path.exists(sibling_legacy):
                return sibling_legacy, 'legacy_model'
            return path, 'model'
        if basename.endswith('.pt'):
            return path, 'full_or_model'

        raise RuntimeError(f'Unsupported checkpoint path: {path}')

    def load_training_state(self, load_path):
        resolved, mode = self._resolve_resume_target(load_path)
        if resolved is None:
            return

        if mode == 'full':
            state = _torch_load(resolved, map_location='cpu', allow_pickle=True)
            self._restore_full_trainer_state(state)
            print(f'Resumed full trainer state from {resolved}')
            return

        if mode == 'legacy':
            self._restore_legacy_state(resolved)
            print(f'Resumed legacy trainer state from {resolved}')
            return

        if mode == 'legacy_model':
            model_guess = max(
                glob.glob(os.path.join(os.path.dirname(resolved), 'model_*.pt')),
                key=os.path.getmtime,
            )
            self._restore_legacy_state(resolved, explicit_model_path=model_guess)
            print(f'Resumed legacy trainer state from {resolved} with model {model_guess}')
            return

        if mode == 'model':
            weights = _torch_load(resolved, map_location='cpu')
            self._load_named_weights_into_muon(weights)
            print(f'Loaded model weights from {resolved} (optimizer/trainer state not restored)')
            return

        if mode == 'full_or_model':
            data = _torch_load(resolved, map_location='cpu', allow_pickle=True)
            if isinstance(data, dict) and ('muon_state' in data or 'pufferl_cpp_state' in data):
                self._restore_full_trainer_state(data)
                print(f'Resumed full trainer state from {resolved}')
            elif isinstance(data, dict):
                self._load_named_weights_into_muon(data)
                print(f'Loaded model weights from {resolved} (optimizer/trainer state not restored)')
            else:
                raise RuntimeError(f'Unrecognized checkpoint format at {resolved}')
            return

        raise RuntimeError(f'Unhandled resume mode: {mode}')

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def evaluate(self):
        _C.rollouts(self.pufferl_cpp)
        self.global_step += self.batch_size

    def train(self):
        # Selfplay: process rollout for swap tracking / ELO before training
        if self.selfplay_mgr:
            self.selfplay_mgr.update_from_rollout(self.rollouts)

        _C.train(self.pufferl_cpp)
        logs = None
        self.epoch += 1

        # Selfplay: save policy snapshot for opponent pool
        if self.selfplay_mgr:
            self.selfplay_mgr.save_snapshot(self.policy_fp32, self.pufferl_cpp.muon, self.epoch)
        done_training = self.global_step >= self.config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.6:
            torch.cuda.synchronize()
            logs = _C.log_environments(self.pufferl_cpp)
            self.stats = logs
            self.losses = _C.log_losses(self.pufferl_cpp)
            self.profile = _C.log_profile(self.pufferl_cpp)
            self.utilization = _C.log_utilization(self.pufferl_cpp)
            logs = self.write_logs(logs)

            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step

        if self.epoch % self.config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def write_logs(self, logs):
        if not self.logger:
            return

        config = self.config
        device = config['device']
        agent_steps = int(self.global_step * config['gpus'])
        selfplay_logs = {}
        if self.selfplay_mgr:
            mgr = self.selfplay_mgr
            selfplay_logs = {
                'selfplay/learner_elo': mgr.elos[0],
                'selfplay/pool_size': len(mgr.opponent_pool_ids),
                'selfplay/cur_opponent': mgr.cur_opp_id,
                'selfplay/snapshots_saved': mgr.saved_policy_count,
            }
        logs = {
            'SPS': int(self.sps * config['gpus']),
            'environment/SPS': int(self.sps * config['gpus']),
            'agent_steps': int(agent_steps * config['gpus']),
            'uptime': self.uptime,
            'epoch': int(self.epoch * config['gpus']),
            #'learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in logs.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v for k, v in self.profile.items()},
            **selfplay_logs,
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        model_path = self.save_checkpoint()
        # Clear Python references to C++ tensors BEFORE calling C++ close
        self.rollouts = None
        self.policy_fp32 = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminals = None

        torch.cuda.synchronize()
        _C.close(self.pufferl_cpp)
        self.pufferl_cpp = None

        # Clear cuBLAS workspaces that accumulate per-stream
        # This is the only way to check for memleaks. May not
        # be strictly necessary for normal training.
        torch.cuda.empty_cache()
        torch._C._cuda_clearCublasWorkspaces()

        if not self.logger:
            return

        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'],
            self.config["env"], f'{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if not self.logger:
            return

        run_id = self.logger.run_id
        path = self._checkpoint_dir()

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if not os.path.exists(model_path):
            torch.save(_get_trained_state_dict(self.policy_fp32, self.pufferl_cpp.muon), model_path)

        state = {
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)

        full_state_path = os.path.join(path, 'trainer_state_full.pt')
        full_state = self._save_full_trainer_state(run_id, model_name)
        torch.save(full_state, full_state_path + '.tmp')
        os.replace(full_state_path + '.tmp', full_state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[white]', b1='[bright_cyan]', b2='[bright_white]'):
        if not self.verbose:
            return

        config = self.config
        sps = self.sps * config['gpus']
        agent_steps = self.global_step * config['gpus']
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=18)
        table.add_column(justify="right", width=12)

        table.add_row(
            f'{b1}PufferLib {b2}4.0 {idx[0]*" "}:blowfish:',
            f'{c1}GPU: {b2}{self.utilization.get("gpu_util", 0):.0f}{c2}%',
            f'{c1}VRAM: {b2}{self.utilization.get("vram_used_gb", 0):.1f}{c2}/{b2}{self.utilization.get("vram_total_gb", 0):.0f}{c2}G',
            f'{c1}RAM: {b2}{self.utilization.get("cpu_mem_gb", 0):.1f}{c2}G',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps']*config['gpus'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{c2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{c2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{c2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{c2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{c2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{c2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{c2}Remaining', remaining)

        delta = profile['rollout'] + profile['train']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf2('Evaluate', b1, delta, profile['rollout'], b2, c2))
        p.add_row(*fmt_perf2('  GPU', b2, delta, profile['eval_gpu'], b2, c2))
        p.add_row(*fmt_perf2('  Env', b2, delta, profile['eval_env'], b2, c2))
        p.add_row(*fmt_perf2('Train', b1, delta, profile['train'], b2, c2))
        p.add_row(*fmt_perf2('  Misc', b2, delta, profile['train_misc'], b2, c2))
        p.add_row(*fmt_perf2('  Forward', b2, delta, profile['train_forward'], b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        display_stats = dict(self.stats or self.last_stats)
        if self.selfplay_mgr:
            display_stats['elo'] = self.selfplay_mgr.elos[0]

        for metric, value in display_stats.items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    if seconds < 1:
        return f"{b2}{seconds*1000:.0f}{c2}ms"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def fmt_perf2(name, color, delta_ref, elapsed, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*elapsed/delta_ref - 1e-5)
    return f'{color}{name}', duration(elapsed, b2, c2), f'{b2}{percent:2d}{c2}%'

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

class Logger:
    def __init__(self, args, load_id=None, resume='allow'):
        train_args = args['train']

        self.run_id = str(int(1000*time.time()))
        root = os.path.join(train_args['data_dir'], 'logs', args['env_name'])
        if not os.path.exists(root):
            os.makedirs(root)

        self.path = os.path.join(root, self.run_id + '.json')
        self.logs = {'data': []}
        for k, v in pufferlib.unroll_nested_dict(train_args):
            self.logs[k] = v

        self.wandb = None
        if args['wandb']:
            import wandb
            wandb.init(
                id=load_id or wandb.util.generate_id(),
                project=args['wandb_project'],
                group=args['wandb_group'],
                allow_val_change=True,
                save_code=False,
                resume=resume,
                config=args,
                tags = [args['tag']] if args['tag'] is not None else [],
                settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
            )
            self.wandb = wandb
            self.run_id = wandb.run.id
            self.should_upload_model = not args['no_model_upload']

       
    def log(self, logs, step):
        self.logs['data'].append(logs)

        if self.wandb:
            self.wandb.log(logs, step=step)

    def log_cost(self, cost):
        self.logs['cost'] = cost

    def upload_model(self, model_path):
        if not self.wandb:
            return

        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path, early_stop):
        self.logs['early_stop'] = early_stop
        import json
        with open(self.path, 'w') as f:
            json.dump(self.logs, f)

        if not self.wandb:
            return
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.run.summary['early_stop'] = early_stop
        self.wandb.finish()

    def download(self):
        assert self.wandb, 'No wandb run'
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'

def _train_rank(env_name, args=None, logger=None, verbose=True, early_stop_fn=None):
    """Worker function for multi-GPU training. Runs on each GPU."""

    if args:
        torch.cuda.set_device(args['train']['rank'])

    args = args or load_config(env_name)

    train_config = dict(**args['train'])
    train_config['env_name'] = args['env_name']

    vec_config = args['vec']
    env_config = args['env']
    policy_config = args['policy']
    pufferl = PuffeRL(train_config, vec_config, env_config, policy_config, logger, verbose)

    resume_path = args.get('load_model_path')
    if resume_path is not None:
        pufferl.load_training_state(resume_path)

    if train_config['profile']:
        _C.profiler_start()

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20*train_config['total_timesteps'], 100_000_000)
    all_logs = []

    while pufferl.global_step < train_config['total_timesteps']:
        pufferl.evaluate()
        logs = pufferl.train()

        if logs is None:
            continue

        should_stop_early = False
        if early_stop_fn is not None:
            should_stop_early = early_stop_fn(logs)

            # This is hacky, but need to see if threshold looks reasonable
            if 'early_stop_threshold' in logs:
                pufferl.logger.log({'environment/early_stop_threshold': logs['early_stop_threshold']}, logs['agent_steps'])

        if pufferl.global_step > logging_threshold:
            all_logs.append(logs)

        if should_stop_early:
            if train_config['profile']:
                _C.profiler_stop()
            model_path = pufferl.close()
            pufferl.logger.log_cost(pufferl.uptime)
            pufferl.logger.close(model_path, early_stop=True)
            return pufferl, all_logs

    if train_config['profile']:
        _C.profiler_stop()

    pufferl.print_dashboard()

    if not logger:
        model_path = pufferl.close()

    return pufferl, all_logs


def train(env_name, args=None, logger=None, verbose=True, early_stop_fn=None):
    if args is None:
        args = load_config(env_name)

    num_gpus = args['train']['gpus']

    nccl_id_path = f'/tmp/puffer_nccl_{os.getpid()}'
    if os.path.exists(nccl_id_path):
        os.remove(nccl_id_path)

    # Set shared config
    args['train']['world_size'] = num_gpus
    args['train']['nccl_id_path'] = nccl_id_path

    args['train']['total_timesteps'] /= num_gpus
    args['train']['minibatch_size'] /= num_gpus
    args['vec']['total_agents'] /= num_gpus
    args['vec']['num_threads'] /= num_gpus

    # Spawn workers for ranks 1..N-1
    ctx = mp.get_context('spawn')
    procs = []
    for rank in range(1, num_gpus):
        worker_args = deepcopy(args)
        worker_args['train']['rank'] = rank
        p = ctx.Process(target=_train_rank, args=(env_name, worker_args, None, False, early_stop_fn))
        p.start()
        procs.append(p)

    # Run rank 0 on main process
    torch.cuda.set_device(0)

    args['train']['rank'] = 0

    if logger is None:
        logger = Logger(args)

    pufferl, all_logs = _train_rank(env_name, args=args, logger=logger, verbose=True)

    for p in procs:
        p.join()

    if os.path.exists(nccl_id_path):
        os.remove(nccl_id_path)


    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    uptime = pufferl.uptime
    agent_steps = pufferl.global_step
    logs = {}
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        pufferl.evaluate()
        if i == 0 or i % 32 != 0:
            continue

        torch.cuda.synchronize()
        logs = _C.log_environments(pufferl.pufferl_cpp)
        pufferl.stats = logs

        if logs:
            break

    logs['uptime'] = uptime
    logs['agent_steps'] = agent_steps
    logs = pufferl.write_logs(logs)

    all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.log_cost(uptime)
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs

def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')

def _sweep_worker(env_name, q_host, q_worker, device):
    while True:
        #print("Worker waiting")
        args = q_worker.get()
        #print("Worker got data")
        args['train']['device'] = device
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            all_logs = train(env_name, args=args, verbose=False)
        except Exception:
            import traceback
            traceback.print_exc()

        #all_logs = [{'foo': 0}]
        #print("Worker ran experiment")
        q_host.put(all_logs)
        #print("Worker submitted result")

def multisweep(args=None, env_name=None):
    args = args or load_config(env_name)
    sweep_gpus = args['sweep_gpus']
    if sweep_gpus == -1:
        sweep_gpus = torch.cuda.device_count()

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'

    from multiprocessing import Process, Queue, set_start_method
    from copy import deepcopy

    host_queues = []
    worker_queues = []
    workers = []
    worker_args = []
    set_start_method('spawn')
    for i in range(sweep_gpus):
        q_host = Queue()
        q_worker = Queue()
        w = Process(
            target=_sweep_worker,
            args=(env_name, q_host, q_worker, f'cuda:{i}')
        )
        w.start()
        host_queues.append(q_host)
        worker_queues.append(q_worker)
        args = deepcopy(args)
        worker_args.append(args)

    for w in range(sweep_gpus):
        args = worker_args[w]
        sweep.suggest(args)
        total_timesteps = args['train']['total_timesteps']
        worker_queues[w].put(args)

    runs = 0

    suggestion = deepcopy(args)
    while runs < args['max_runs']:
        for w in range(sweep_gpus):
            args = worker_args[w]
            if host_queues[w].empty():
                continue

            all_logs = host_queues[w].get(timeout=0)
            if not all_logs:
                continue

            all_logs = [e for e in all_logs if target_key in e]
            scores = downsample([log[target_key] for log in all_logs], points_per_run)
            times = downsample([log['uptime'] for log in all_logs], points_per_run)
            steps = downsample([log['agent_steps'] for log in all_logs], points_per_run)
            #costs = np.stack([times, steps], axis=1)
            costs = times
            timesteps = [log['agent_steps'] for log in all_logs]
            timesteps = downsample(timesteps, points_per_run)
            for score, cost, timestep in zip(scores, costs, timesteps):
                args['train']['total_timesteps'] = timestep
                sweep.observe(args, score, cost)

            runs += 1

            sweep.suggest(args)
            worker_queues[w].put(args)

def paretosweep(args=None, env_name=None):
    args = args or load_config(env_name)
    sweep_gpus = args['sweep_gpus']
    if sweep_gpus == -1:
        sweep_gpus = torch.cuda.device_count()

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    total_timesteps = args['sweep']['train'].pop('total_timesteps')
    mmin = total_timesteps['min']
    mmax = total_timesteps['max']
    all_timesteps = np.geomspace(mmin, mmax, sweep_gpus)
    # You hardcoded buffer size to 5 instead of 10 for this
    sweeps = [sweep_cls(args['sweep']) for _ in range(sweep_gpus)]
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'

    from multiprocessing import Process, Queue, set_start_method
    from copy import deepcopy

    host_queues = []
    worker_queues = []
    workers = []
    worker_args = []
    set_start_method('spawn')
    for i in range(sweep_gpus):
        q_host = Queue()
        q_worker = Queue()
        w = Process(
            target=_sweep_worker,
            args=(env_name, q_host, q_worker, f'cuda:{i}')
        )
        w.start()
        host_queues.append(q_host)
        worker_queues.append(q_worker)
        args = deepcopy(args)
        worker_args.append(args)

    for w in range(sweep_gpus):
        args = worker_args[w]
        sweeps[w].suggest(args)
        args['train']['total_timesteps'] = all_timesteps[w]
        worker_queues[w].put(args)

    runs = 0

    suggestion = deepcopy(args)
    while runs < args['max_runs']:
        for w in range(sweep_gpus):
            args = worker_args[w]
            if host_queues[w].empty():
                continue

            all_logs = host_queues[w].get(timeout=0)
            if not all_logs:
                continue

            all_logs = [e for e in all_logs if target_key in e]
            scores = downsample([log[target_key] for log in all_logs], points_per_run)
            times = downsample([log['uptime'] for log in all_logs], points_per_run)
            steps = downsample([log['agent_steps'] for log in all_logs], points_per_run)
            #costs = np.stack([times, steps], axis=1)
            costs = times
            timesteps = [log['agent_steps'] for log in all_logs]
            timesteps = downsample(timesteps, points_per_run)
            for score, cost, timestep in zip(scores, costs, timesteps):
                args['train']['total_timesteps'] = timestep
                sweeps[w].observe(args, score, cost)

            runs += 1

            sweeps[w].suggest(args)
            args['train']['total_timesteps'] = all_timesteps[w]
            worker_queues[w].put(args)

    print('Done')

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    args['no_model_upload'] = True  # Uploading trained model during sweep crashed wandb

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    running_target_buffer = deque(maxlen=30)

    def stop_if_perf_below(logs):
        if any("losses/" in k and np.isnan(v) for k, v in logs.items()):
            logs['is_loss_nan'] = True
            return True

        if method != 'Protein':
            return False

        if ('uptime' in logs and target_key in logs):
            metric_val, cost = logs[target_key], logs['uptime']
            running_target_buffer.append(metric_val)
            target_running_mean = np.mean(running_target_buffer)
            
            # If metric distribution is percentile, threshold is also logit transformed
            threshold = sweep.get_early_stop_threshold(cost)
            print(f'Threshold: {threshold} at cost {cost}')
            logs['early_stop_threshold'] = max(threshold, -5)  # clipping for visualization

            if sweep.should_stop(max(target_running_mean, metric_val), cost):
                logs['is_loss_nan'] = False
                return True
        return False

    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # In the first run, skip sweep and use the train args specified in the config
        if i > 0:
            sweep.suggest(args)

        all_logs = train(env_name, args=args, early_stop_fn=stop_if_perf_below)
        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        is_final_loss_nan = all_logs[-1].get('is_loss_nan', False)
        if is_final_loss_nan:
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f'Saved {len(weights)} weights to {path}')

def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    '''
    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['policy'])
    '''
    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['wandb']:
            path = Logger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandbrun')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=1200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='puffer4')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--sweep-gpus', type=int, default=-1, help='multigpu sweeps')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.add_argument('--profile', action='store_true', help='Enable nsys profiling (use with nsys --capture-range=cudaProfilerApi)')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=auto_type if value == 'auto' else type(value)
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, export] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == 'train':
        train(env_name=env_name)
    elif mode == 'eval':
        eval(env_name=env_name)
    elif mode == 'sweep':
        sweep(env_name=env_name)
    elif mode == 'multisweep':
        multisweep(env_name=env_name)
    elif mode == 'paretosweep':
        paretosweep(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
