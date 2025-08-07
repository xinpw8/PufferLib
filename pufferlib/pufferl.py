# # puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full details
# # This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# # Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

# import warnings
# warnings.filterwarnings('error', category=RuntimeWarning)

# import os
# import sys
# import glob
# import ast
# import time
# import random
# import shutil
# import math
# import argparse
# import importlib
# import configparser
# from threading import Thread
# from collections import defaultdict, deque

# import numpy as np
# import psutil

# import torch
# import torch.distributed
# from torch.distributed.elastic.multiprocessing.errors import record
# import torch.utils.cpp_extension

# import pufferlib
# from pufferlib import sweep
# from pufferlib import vector
# from pufferlib import pytorch
# try:
#     from pufferlib import _C
# except ImportError:
#     raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

# import rich
# import rich.traceback
# from rich.table import Table
# from rich.console import Console
# from rich_argparse import RichHelpFormatter
# rich.traceback.install(show_locals=False)

# import signal # Aggressively exit on ctrl+c
# signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# # Assume advantage kernel has been built if CUDA compiler is available
# ADVANTAGE_CUDA = shutil.which("nvcc") is not None


# class PuffeRL:
#     def __init__(self, config, vecenv, policy, logger=None):
#         # Backend perf optimization
#         torch.set_float32_matmul_precision('high')
#         torch.backends.cudnn.deterministic = config['torch_deterministic']
#         torch.backends.cudnn.benchmark = True

#         # Reproducibility
#         seed = config['seed']
#         #random.seed(seed)
#         #np.random.seed(seed)
#         #torch.manual_seed(seed)

#         # Vecenv info
#         vecenv.async_reset(seed)
#         obs_space = vecenv.single_observation_space
#         atn_space = vecenv.single_action_space
#         total_agents = vecenv.num_agents
#         self.total_agents = total_agents

#         # Experience
#         if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
#             raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
#         elif config['batch_size'] == 'auto':
#             config['batch_size'] = total_agents * config['bptt_horizon']
#         elif config['bptt_horizon'] == 'auto':
#             config['bptt_horizon'] = config['batch_size'] // total_agents

#         batch_size = config['batch_size']
#         horizon = config['bptt_horizon']
#         segments = batch_size // horizon
#         self.segments = segments
        
#         print(f"[PUFFERL_DEBUG] Configuration check:")
#         print(f"  num_envs: {vecenv.num_envs}")
#         print(f"  total_agents: {total_agents}")
#         print(f"  batch_size: {batch_size}")
#         print(f"  bptt_horizon: {horizon}")
#         print(f"  segments: {segments}")
        
#         if total_agents > segments:
#             raise pufferlib.APIUsageError(
#                 f'Total agents {total_agents} > segments {segments}. batch_size={batch_size}, horizon={horizon}, num_envs={vecenv.num_envs}. Increase batch_size or decrease bptt_horizon.'
#             )

#         device = config['device']
#         self.observations = torch.zeros(segments, horizon, *obs_space.shape,
#             dtype=pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
#             pin_memory=device == 'cuda' and config['cpu_offload'],
#             device='cpu' if config['cpu_offload'] else device)
#         self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
#             dtype=pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
#         self.values = torch.zeros(segments, horizon, device=device)
#         self.logprobs = torch.zeros(segments, horizon, device=device)
#         self.rewards = torch.zeros(segments, horizon, device=device)
#         self.terminals = torch.zeros(segments, horizon, device=device)
#         self.truncations = torch.zeros(segments, horizon, device=device)
#         self.ratio = torch.ones(segments, horizon, device=device)
#         self.importance = torch.ones(segments, horizon, device=device)
#         self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
#         self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
#         self.free_idx = total_agents

#         # LSTM
#         if config['use_rnn']:
#             n = vecenv.agents_per_batch
#             h = policy.hidden_size
#             self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
#             self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

#         # Minibatching & gradient accumulation
#         minibatch_size = config['minibatch_size']
#         max_minibatch_size = config['max_minibatch_size']
#         self.minibatch_size = min(minibatch_size, max_minibatch_size)
#         if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
#             raise pufferlib.APIUsageError(
#                 f'minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly')

#         if batch_size < minibatch_size:
#             raise pufferlib.APIUsageError(
#                 f'batch_size {batch_size} must be >= minibatch_size {minibatch_size}'
#             )

#         self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
#         self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
#         self.minibatch_segments = self.minibatch_size // horizon 
#         if self.minibatch_segments * horizon != self.minibatch_size:
#             raise pufferlib.APIUsageError(
#                 f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
#             )

#         # Torch compile
#         self.uncompiled_policy = policy
#         self.policy = policy
#         if config['compile']:
#             self.policy = torch.compile(policy, mode=config['compile_mode'], fullgraph=config['compile_fullgraph'])

#         # Optimizer
#         if config['optimizer'] == 'adam':
#             optimizer = torch.optim.Adam(
#                 self.policy.parameters(),
#                 lr=config['learning_rate'],
#                 betas=(config['adam_beta1'], config['adam_beta2']),
#                 eps=config['adam_eps'],
#             )
#         elif config['optimizer'] == 'muon':
#             from heavyball import ForeachMuon
#             warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
#             import heavyball.utils
#             heavyball.utils.compile_mode = config['compile_mode'] if config['compile'] else None
#             optimizer = ForeachMuon(
#                 self.policy.parameters(),
#                 lr=config['learning_rate'],
#                 betas=(config['adam_beta1'], config['adam_beta2']),
#                 eps=config['adam_eps'],
#             )
#         else:
#             raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

#         self.optimizer = optimizer
        
#         # Restore optimizer state if resuming
#         if hasattr(self, '_resume_optimizer_state') and self._resume_optimizer_state is not None:
#             try:
#                 self.optimizer.load_state_dict(self._resume_optimizer_state)
#                 print("[Resume] Restored optimizer state")
#             except Exception as e:
#                 print(f"[Resume] Warning: Could not restore optimizer state: {e}")

#         # Logging
#         self.logger = logger
#         if logger is None:
#             self.logger = NoLogger(config)

#         # Learning rate scheduler
#         epochs = max(1, config['total_timesteps'] // config['batch_size'])
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#         self.total_epochs = epochs

#         # Automatic mixed precision
#         precision = config['precision']
#         self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
#         if precision not in ('float32', 'bfloat16'):
#             raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

#         # Initializations
#         self.config = config
#         self.vecenv = vecenv
#         self.epoch = 0
#         self.global_step = 0
#         self.last_log_step = 0
#         self.last_log_time = time.time()
#         self.start_time = time.time()
#         self.utilization = Utilization()
#         self.profile = Profile()
        
#         # Handle resume state if provided
#         if '_resume_trainer_state' in config:
#             resume_state = config['_resume_trainer_state']
#             self.epoch = resume_state.get('epoch', 0)
#             self.global_step = resume_state.get('global_step', 0)
#             print(f"[Resume] Restored training state - epoch: {self.epoch}, global_step: {self.global_step}")
#             # Store optimizer state for later restoration
#             self._resume_optimizer_state = resume_state.get('optimizer_state_dict', None)
#         self.stats = defaultdict(list)
#         self.last_stats = defaultdict(list)
#         self.losses = {}

#         # Dashboard
#         self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
#         self.print_dashboard(clear=True)

#     @property
#     def uptime(self):
#         return time.time() - self.start_time

#     @property
#     def sps(self):
#         if self.global_step == self.last_log_step:
#             return 0

#         return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

#     def _validate_pufferl_chess_observations(self, obs_tensor, location):
#         """COLOR MONITORING: Validates chess observations at pufferl.py level"""
#         if obs_tensor is None or obs_tensor.numel() == 0:
#             print(f"[MONITOR_FATAL] Pufferl.py: Empty observation tensor at {location}")
#             print(f"  Training loop received empty observations")
#             print(f"  FIX: Check vector.py recv() or environment generation")
#             exit(1)
            
#         # Check if this looks like chess observations
#         if obs_tensor.shape[-1] == 1537:  # Chess detected (sparse format)
#             batch_size = obs_tensor.shape[0]
            
#             # Validate tensor properties
#             if obs_tensor.dtype not in [torch.float32, torch.float64]:
#                 print(f"[MONITOR_FATAL] Pufferl.py: Invalid tensor dtype at {location}")
#                 print(f"  Expected float32/float64, got {obs_tensor.dtype}")
#                 print(f"  FIX: Check tensor conversion in vector.py or chess.py")
#                 exit(1)
                
#             # Validate chess content
#             board_sums = obs_tensor[:, :1472].sum(dim=1)
#             num_legal_moves = obs_tensor[:, 1472]
            
#             if (board_sums < 1.0).any() or (num_legal_moves < 0).any() or (num_legal_moves > 64).any():
#                 invalid_indices = torch.where((board_sums < 1.0) | (num_legal_moves < 0) | (num_legal_moves > 64))[0]
#                 print(f"[MONITOR_FATAL] Pufferl.py: Invalid chess observations at {location}")
#                 print(f"  Invalid batch indices: {invalid_indices.tolist()}")
#                 print(f"  Board sums: {board_sums}")
#                 print(f"  Num legal moves: {num_legal_moves}")
#                 print(f"  FIX: Check observation pipeline from chess.h through vector.py")
#                 exit(1)
                
#             # print(f"[MONITOR_OK] Pufferl.py: Chess observations valid at {location} "
#             #       f"(batch={batch_size}, device={obs_tensor.device}, "
#             #       f"board_range=[{board_sums.min():.1f},{board_sums.max():.1f}], "
#             #       f"legal_moves_range=[{num_legal_moves.min():.0f},{num_legal_moves.max():.0f}])")

#     def _validate_pufferl_chess_actions(self, action_tensor, location):
#         """COLOR MONITORING: Validates chess actions at pufferl.py level"""
#         if action_tensor is None or action_tensor.numel() == 0:
#             print(f"[MONITOR_FATAL] Pufferl.py: Empty action tensor at {location}")
#             print(f"  Training loop generated empty actions")
#             print(f"  FIX: Check policy.forward_eval() or pytorch.sample_logits()")
#             exit(1)
            
#         # Validate action range (chess UCI actions: 0-1967)
#         if (action_tensor < 0).any() or (action_tensor >= 1968).any():
#             invalid_mask = (action_tensor < 0) | (action_tensor >= 1968)
#             invalid_actions = action_tensor[invalid_mask]
#             print(f"[MONITOR_FATAL] Pufferl.py: Invalid chess actions at {location}")
#             print(f"  Invalid actions (first 10): {invalid_actions[:10].tolist()}")
#             print(f"  Valid range: [0, 1967] for UCI chess actions")
#             print(f"  FIX: Check action space or policy output in models.py/torch.py")
#             exit(1)
            
#         # print(f"[MONITOR_OK] Pufferl.py: Chess actions valid at {location} "
#         #       f"(shape={action_tensor.shape}, range=[{action_tensor.min()},{action_tensor.max()}])")

#     def evaluate(self):
#         profile = self.profile
#         epoch = self.epoch
#         profile('eval', epoch)
#         profile('eval_misc', epoch, nest=True)

#         config = self.config
#         device = config['device']

#         if config['use_rnn']:
#             for k in self.lstm_h:
#                 self.lstm_h[k].zero_()
#                 self.lstm_c[k].zero_()

#         self.full_rows = 0
#         recv_timeout_count = 0
#         max_recv_timeouts = 5
        
#         while self.full_rows < self.segments:
#             profile('env', epoch)
            
#             # Add timeout to recv call to prevent infinite hangs
#             try:
#                 import signal
                
#                 def recv_timeout_handler(signum, frame):
#                     raise TimeoutError("vecenv.recv() timed out")
                
#                 signal.signal(signal.SIGALRM, recv_timeout_handler)
#                 signal.alarm(30)  # 30 second timeout for recv
                
#                 try:
#                     print(f"[PUFFERL DEBUG] Calling vecenv.recv() at step {self.global_step}")
#                     o, r, d, t, info, env_id, mask = self.vecenv.recv()
#                     print(f"[PUFFERL DEBUG] vecenv.recv() returned successfully")
                    
#                     # Check if recv returned valid data
#                     if o is None:
#                         recv_timeout_count += 1
#                         print(f"[PUFFERL_ERROR] vecenv.recv() returned None observations ({recv_timeout_count}/{max_recv_timeouts})")
#                         signal.alarm(0)  # Cancel timeout before continuing
#                         if recv_timeout_count >= max_recv_timeouts:
#                             raise RuntimeError(f"vecenv.recv() returned None {max_recv_timeouts} times - training cannot continue")
#                         continue
                    
#                     # Handle potential None values from recv
#                     if mask is None and o is not None:
#                         mask = np.ones(len(o), dtype=bool) if hasattr(o, '__len__') else np.ones(1, dtype=bool)
#                     if d is None:
#                         d = np.zeros_like(mask, dtype=bool)
#                     if t is None:
#                         t = np.zeros_like(mask, dtype=bool)
#                     if r is None:
#                         r = np.zeros_like(mask, dtype=np.float32)
#                     if env_id is None:
#                         env_id = list(range(len(o)))
#                     if info is None:
#                         info = {}
                    
#                     recv_timeout_count = 0  # Reset counter on successful recv
#                 finally:
#                     signal.alarm(0)  # Cancel timeout
                    
#             except TimeoutError:
#                 recv_timeout_count += 1
#                 print(f"[PUFFERL_ERROR] vecenv.recv() timed out ({recv_timeout_count}/{max_recv_timeouts})")
                
#                 if recv_timeout_count >= max_recv_timeouts:
#                     raise RuntimeError(f"vecenv.recv() timed out {max_recv_timeouts} times - training cannot continue")
                
#                 # Try to continue with next iteration
#                 continue

#             profile('eval_misc', epoch)
#             env_id = slice(env_id[0], env_id[-1] + 1)

#             done_mask = d + t # TODO: Handle truncations separately
#             self.global_step += int(mask.sum())

#             profile('eval_copy', epoch)
#             o = torch.as_tensor(o)
            
#             # --- COLOR MONITORING: Validate observations entering training loop ---
#             self._validate_pufferl_chess_observations(o, "evaluate() recv")
            
#             o_device = o.to(device)#, non_blocking=True)
#             r = torch.as_tensor(r).to(device)#, non_blocking=True)
#             d = torch.as_tensor(d).to(device)#, non_blocking=True)

#             profile('eval_forward', epoch)
#             with torch.no_grad(), self.amp_context:
#                 state = dict(
#                     reward=r,
#                     done=d,
#                     env_id=env_id,
#                     mask=mask,
#                 )

#                 if config['use_rnn']:
#                     state['lstm_h'] = self.lstm_h[env_id.start]
#                     state['lstm_c'] = self.lstm_c[env_id.start]

#                 # Debug observation for puzzles
#                 if hasattr(self.vecenv, 'puzzle_mode') and self.global_step % 100 == 0:
#                     print(f"\n[OBS DEBUG] Step {self.global_step}:")
#                     print(f"  Observation shape: {o_device.shape}")
#                     # Check action mask (last 65 values)
#                     if o_device.shape[-1] >= 1537:  # Chess observation size
#                         mask_start = 1472
#                         action_mask = o_device[0, mask_start:mask_start+65].cpu().numpy()
#                         print(f"  Action mask (first 10): {action_mask[:10]}")
#                         # Check if h4h7 is legal
#                         h4h7_idx = 1395 // 64  # Which sparse index
#                         h4h7_bit = 1395 % 64   # Which bit in that index
#                         if h4h7_idx < len(action_mask) - 1:  # -1 for count
#                             print(f"  h4h7 sparse index: {h4h7_idx}, expecting bit {h4h7_bit}")
#                             if int(action_mask[h4h7_idx + 1]) & (1 << h4h7_bit):
#                                 print(f"  h4h7 IS LEGAL according to mask!")
#                             else:
#                                 print(f"  h4h7 NOT legal according to mask")
#                     import sys
#                     sys.stdout.flush()
                
#                 logits, value = self.policy.forward_eval(o_device, state)
#                 action, logprob, _ = pytorch.sample_logits(logits)
                
#                 # Debug action generation for puzzles
#                 if hasattr(self.vecenv, 'puzzle_mode') and self.global_step % 100 == 0:
#                     # Get the raw action value
#                     action_val = action.cpu().numpy()[0] if action.numel() > 0 else -1
#                     # Get top 5 actions from logits
#                     probs = torch.softmax(logits[0], dim=-1)
#                     top5_probs, top5_actions = torch.topk(probs, 5)
#                     print(f"\n[ACTION DEBUG] Step {self.global_step}:")
#                     print(f"  Selected action: {action_val}")
#                     print(f"  Top 5 actions: {top5_actions.cpu().numpy()} with probs {top5_probs.cpu().numpy()}")
#                     # Check if h4h7 (which should be action 1395) is in the top actions
#                     h4h7_action = 1395  # This is h4h7 in UCI mapping
#                     if h4h7_action < len(probs):
#                         h4h7_prob = probs[h4h7_action].item()
#                         print(f"  h4h7 (action {h4h7_action}) probability: {h4h7_prob:.6f}")
#                     import sys
#                     sys.stdout.flush()
                
#                 # --- COLOR MONITORING: Validate actions from policy ---
#                 self._validate_pufferl_chess_actions(action, "evaluate() policy output")

#             profile('eval_copy', epoch, nest=True)
#             with torch.no_grad():
#                 if config['use_rnn']:
#                     self.lstm_h[env_id.start] = state['lstm_h']
#                     self.lstm_c[env_id.start] = state['lstm_c']

#                 # Fast path for fully vectorized envs
#                 l = self.ep_lengths[env_id.start].item()
#                 batch_rows = slice(self.ep_indices[env_id.start].item(), 1+self.ep_indices[env_id.stop - 1].item())
                
#                 # DEBUG: Print shapes to understand mismatch
#                 print(f"[DEBUG] batch_rows: {batch_rows}, env_id: {env_id}")
#                 print(f"[DEBUG] o_device.shape: {o_device.shape if 'o_device' in locals() else 'N/A'}")
#                 print(f"[DEBUG] o.shape: {o.shape}")
#                 print(f"[DEBUG] self.observations[batch_rows, l].shape: {self.observations[batch_rows, l].shape}")

#                 if config['cpu_offload']:
#                     self.observations[batch_rows, l] = o.clone()
#                 else:
#                     self.observations[batch_rows, l] = o_device

#                 self.actions[batch_rows, l] = action
#                 self.logprobs[batch_rows, l] = logprob
#                 self.rewards[batch_rows, l] = r
#                 self.terminals[batch_rows, l] = d.float()
#                 self.values[batch_rows, l] = value.flatten()

#                 # Note: We are not yet handling masks in this version
#                 self.ep_lengths[env_id] += 1
#                 if l+1 >= config['bptt_horizon']:
#                     num_full = env_id.stop - env_id.start
#                     self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
#                     self.ep_lengths[env_id] = 0
#                     self.free_idx += num_full
#                     self.full_rows += num_full

#                 action = action.cpu().numpy()
#                 if isinstance(logits, torch.distributions.Normal):
#                     action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

#                 # --- COLOR MONITORING: Validate numpy actions before sending to env ---
#                 if action.size > 0 and all(0 <= a < 1968 for a in action.flat):
#                     pass
#                     # print(f"[MONITOR_OK] Pufferl.py: Sending valid chess actions to env "
#                     #       f"(shape={action.shape}, range=[{action.min()},{action.max()}])")
#                 elif action.size > 0:
#                     invalid_actions = [a for a in action.flat if not (0 <= a < 1968)]
#                     print(f"[MONITOR_FATAL] Pufferl.py: Invalid actions being sent to environment!")
#                     print(f"  Invalid actions (first 10): {invalid_actions[:10]}")
#                     print(f"  Valid range: [0, 1967] for UCI chess actions")
#                     print(f"  FIX: Check action processing in evaluate() method")
#                     exit(1)

#             profile('eval_misc', epoch)
#             if info and self.global_step % 100 == 0:
#                 print(f"[PUFFERL DEBUG] Processing info, length: {len(info)}")
#                 if info:
#                     print(f"[PUFFERL DEBUG] First info keys: {list(info[0].keys()) if isinstance(info[0], dict) else 'Not a dict'}")
#             for i in info:
#                 for k, v in pufferlib.unroll_nested_dict(i):
#                     if isinstance(v, np.ndarray):
#                         v = v.tolist()
#                     elif isinstance(v, (list, tuple)):
#                         self.stats[k].extend(v)
#                     else:
#                         self.stats[k].append(v)
#                     # Debug puzzle stats specifically
#                     if k in ['puzzle_attempts', 'puzzle_wrong_moves', 'puzzle_success_rate']:
#                         print(f"[STATS ADDED] {k}={v}, total in buffer: {len(self.stats[k])}")
                    
#                     # Debug monitoring for puzzle stats
#                     if k == 'puzzle_success_rate' and self.global_step % 1000 == 0:
#                         print(f"[PUFFERL DEBUG] Step {self.global_step}: puzzle_success_rate = {v}")
#                         print(f"[PUFFERL DEBUG] Current stats buffer size: {len(self.stats[k])}")
#                         fflush_stdout = __import__('sys').stdout.flush
#                         fflush_stdout()

#             profile('env', epoch)
#             print(f"[PUFFERL DEBUG] Calling vecenv.send() with action shape {action.shape}")
#             self.vecenv.send(action)
#             print(f"[PUFFERL DEBUG] vecenv.send() completed")

#         profile('eval_misc', epoch)
#         self.free_idx = self.total_agents
#         self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
#         self.ep_lengths.zero_()
#         profile.end()
#         return self.stats

#     @record
#     def train(self):
#         profile = self.profile
#         epoch = self.epoch
#         profile('train', epoch)
#         losses = defaultdict(float)
#         config = self.config
#         device = config['device']

#         b0 = config['prio_beta0']
#         a = config['prio_alpha']
#         clip_coef = config['clip_coef']
#         vf_clip = config['vf_clip_coef']
#         anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
#         self.ratio[:] = 1

#         for mb in range(self.total_minibatches):
#             profile('train_misc', epoch, nest=True)
#             self.amp_context.__enter__()

#             shape = self.values.shape
#             advantages = torch.zeros(shape, device=device)
#             advantages = compute_puff_advantage(self.values, self.rewards,
#                 self.terminals, self.ratio, advantages, config['gamma'],
#                 config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

#             profile('train_copy', epoch)
#             adv = advantages.abs().sum(axis=1)
#             prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
#             prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
#             idx = torch.multinomial(prio_probs, self.minibatch_segments)
#             mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta
#             mb_obs = self.observations[idx]
#             mb_actions = self.actions[idx]
#             mb_logprobs = self.logprobs[idx]
#             mb_rewards = self.rewards[idx]
#             mb_terminals = self.terminals[idx]
#             mb_truncations = self.truncations[idx]
#             mb_ratio = self.ratio[idx]
#             mb_values = self.values[idx]
#             mb_returns = advantages[idx] + mb_values
#             mb_advantages = advantages[idx]

#             profile('train_forward', epoch)
#             if not config['use_rnn']:
#                 mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

#             state = dict(
#                 action=mb_actions,
#                 lstm_h=None,
#                 lstm_c=None,
#             )

#             logits, newvalue = self.policy(mb_obs, state)
#             actions, newlogprob, entropy = pytorch.sample_logits(logits, action=mb_actions)

#             profile('train_misc', epoch)
#             newlogprob = newlogprob.reshape(mb_logprobs.shape)
#             logratio = newlogprob - mb_logprobs
#             ratio = logratio.exp()
#             self.ratio[idx] = ratio.detach()

#             with torch.no_grad():
#                 old_approx_kl = (-logratio).mean()
#                 approx_kl = ((ratio - 1) - logratio).mean()
#                 clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

#             adv = advantages[idx]
#             adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
#                 ratio, adv, config['gamma'], config['gae_lambda'],
#                 config['vtrace_rho_clip'], config['vtrace_c_clip'])
#             adv = mb_advantages
#             adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

#             # Losses
#             pg_loss1 = -adv * ratio
#             pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
#             pg_loss = torch.max(pg_loss1, pg_loss2).mean()

#             newvalue = newvalue.view(mb_returns.shape)
#             v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
#             v_loss_unclipped = (newvalue - mb_returns) ** 2
#             v_loss_clipped = (v_clipped - mb_returns) ** 2
#             v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

#             entropy_loss = entropy.mean()

#             loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
#             self.amp_context.__enter__() # TODO: AMP needs some debugging

#             # This breaks vloss clipping?
#             self.values[idx] = newvalue.detach().float()

#             # Logging
#             profile('train_misc', epoch)
#             losses['policy_loss'] += pg_loss.item() / self.total_minibatches
#             losses['value_loss'] += v_loss.item() / self.total_minibatches
#             losses['entropy'] += entropy_loss.item() / self.total_minibatches
#             losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
#             losses['approx_kl'] += approx_kl.item() / self.total_minibatches
#             losses['clipfrac'] += clipfrac.item() / self.total_minibatches
#             losses['importance'] += ratio.mean().item() / self.total_minibatches

#             # Learn on accumulated minibatches
#             profile('learn', epoch)
#             loss.backward()
#             if (mb + 1) % self.accumulate_minibatches == 0:
#                 torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()

#         # Reprioritize experience
#         profile('train_misc', epoch)
#         if config['anneal_lr']:
#             self.scheduler.step()

#         y_pred = self.values.flatten()
#         y_true = advantages.flatten() + self.values.flatten()
#         var_y = y_true.var()
#         if var_y == 0:
#             explained_var = float('nan')
#         else:
#             explained_var = (1 - (y_true - y_pred).var() / var_y).item()
#         losses['explained_variance'] = explained_var

#         profile.end()
#         logs = None
#         self.epoch += 1
#         done_training = self.global_step >= config['total_timesteps']
#         if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
#             logs = self.mean_and_log()
#             self.losses = losses
#             self.print_dashboard()
#             self.stats = defaultdict(list)
#             self.last_log_time = time.time()
#             self.last_log_step = self.global_step
#             profile.clear()

#         if self.epoch % config['checkpoint_interval'] == 0 or done_training:
#             self.save_checkpoint()
#             self.msg = f'Checkpoint saved at update {self.epoch}'

#         return logs

#     def mean_and_log(self):
#         config = self.config
        
#         # Debug monitoring before averaging
#         if 'puzzle_success_rate' in self.stats:
#             print(f"\n[MEAN_AND_LOG DEBUG] Before averaging:")
#             print(f"  puzzle_success_rate values: {self.stats['puzzle_success_rate'][:10]}...")  # First 10
#             print(f"  Total values: {len(self.stats['puzzle_success_rate'])}")
            
#         for k in list(self.stats.keys()):
#             v = self.stats[k]
#             try:
#                 v = np.mean(v)
#             except:
#                 del self.stats[k]

#             self.stats[k] = v
            
#         # Debug monitoring after averaging
#         if 'puzzle_success_rate' in self.stats:
#             print(f"  After averaging: puzzle_success_rate = {self.stats['puzzle_success_rate']}")
#             fflush_stdout = __import__('sys').stdout.flush
#             fflush_stdout()

#         device = config['device']
#         agent_steps = int(dist_sum(self.global_step, device))
#         logs = {
#             'SPS': dist_sum(self.sps, device),
#             'agent_steps': agent_steps,
#             'uptime': time.time() - self.start_time,
#             'epoch': int(dist_sum(self.epoch, device)),
#             'learning_rate': self.optimizer.param_groups[0]["lr"],
#             **{f'environment/{k}': v for k, v in self.stats.items()},
#             **{f'losses/{k}': v for k, v in self.losses.items()},
#             **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
#             #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
#             #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
#             #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
#         }

#         if torch.distributed.is_initialized():
#            if torch.distributed.get_rank() != 0:
#                self.logger.log(logs, agent_steps)
#                return logs
#            else:
#                return None

#         self.logger.log(logs, agent_steps)
#         return logs

#     def close(self):
#         import time as time_module
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] PuffeRL.close() called")
        
#         vecenv_close_start = time_module.time()
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] About to close vecenv...")
#         self.vecenv.close()
#         vecenv_close_time = time_module.time() - vecenv_close_start
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] Vecenv close took {vecenv_close_time:.2f}s")
        
#         util_stop_start = time_module.time()
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] About to stop utilization...")
#         self.utilization.stop()
#         util_stop_time = time_module.time() - util_stop_start
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] Utilization stop took {util_stop_time:.2f}s")
        
#         checkpoint_start = time_module.time()
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] About to save checkpoint...")
#         model_path = self.save_checkpoint()
#         checkpoint_time = time_module.time() - checkpoint_start
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] Checkpoint save took {checkpoint_time:.2f}s")
        
#         copy_start = time_module.time()
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] About to copy model file...")
#         run_id = self.logger.run_id
#         path = os.path.join(self.config['data_dir'], f'{run_id}.pt')
#         shutil.copy(model_path, path)
#         copy_time = time_module.time() - copy_start
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] Model copy took {copy_time:.2f}s")
        
#         print(f"[PUFFERL_DEBUG] [{time_module.strftime('%H:%M:%S')}] PuffeRL.close() completed")
#         return path
    

#     def save_checkpoint(self):
#         if torch.distributed.is_initialized():
#            if torch.distributed.get_rank() != 0:
#                return
 
#         run_id = self.logger.run_id
#         path = os.path.join(self.config['data_dir'], run_id)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         model_name = f'model_{self.epoch:06d}.pt'
#         model_path = os.path.join(path, model_name)
#         if os.path.exists(model_path):
#             return model_path

#         torch.save(self.uncompiled_policy.state_dict(), model_path)

#         state = {
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'global_step': self.global_step,
#             'agent_step': self.global_step,
#             'update': self.epoch,
#             'model_name': model_name,
#             'run_id': run_id,
#         }
#         state_path = os.path.join(path, 'trainer_state.pt')
#         torch.save(state, state_path + '.tmp')
#         os.rename(state_path + '.tmp', state_path)
#         return model_path

#     def print_dashboard(self, clear=False, idx=[0],
#             c1='[cyan]', c2='[white]', b1='[bright_cyan]', b2='[bright_white]'):
#         config = self.config
#         sps = dist_sum(self.sps, config['device'])
#         agent_steps = dist_sum(self.global_step, config['device'])
#         if torch.distributed.is_initialized():
#            if torch.distributed.get_rank() != 0:
#                return
 
#         profile = self.profile
#         console = Console()
#         dashboard = Table(box=rich.box.ROUNDED, expand=True,
#             show_header=False, border_style='bright_cyan')
#         table = Table(box=None, expand=True, show_header=False)
#         dashboard.add_row(table)

#         table.add_column(justify="left", width=30)
#         table.add_column(justify="center", width=12)
#         table.add_column(justify="center", width=12)
#         table.add_column(justify="center", width=13)
#         table.add_column(justify="right", width=13)

#         table.add_row(
#             f'{b1}PufferLib {b2}3.0 {idx[0]*" "}:blowfish:',
#             f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
#             f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
#             f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
#             f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
#         )
#         idx[0] = (idx[0] - 1) % 10
            
#         s = Table(box=None, expand=True)
#         remaining = 'A hair past a freckle'
#         if sps != 0:
#             remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

#         s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
#         s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
#         s.add_row(f'{c2}Env', f'{b2}{config["env"]}')
#         s.add_row(f'{c2}Params', abbreviate(self.model_size, b2, c2))
#         s.add_row(f'{c2}Steps', abbreviate(agent_steps, b2, c2))
#         s.add_row(f'{c2}SPS', abbreviate(sps, b2, c2))
#         s.add_row(f'{c2}Epoch', f'{b2}{self.epoch}')
#         s.add_row(f'{c2}Uptime', duration(self.uptime, b2, c2))
#         s.add_row(f'{c2}Remaining', remaining)

#         delta = profile.eval['buffer'] + profile.train['buffer']
#         p = Table(box=None, expand=True, show_header=False)
#         p.add_column(f"{c1}Performance", justify="left", width=10)
#         p.add_column(f"{c1}Time", justify="right", width=8)
#         p.add_column(f"{c1}%", justify="right", width=4)
#         p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
#         p.add_row(*fmt_perf('  Forward', c2, delta, profile.eval_forward, b2, c2))
#         p.add_row(*fmt_perf('  Env', c2, delta, profile.env, b2, c2))
#         p.add_row(*fmt_perf('  Copy', c2, delta, profile.eval_copy, b2, c2))
#         p.add_row(*fmt_perf('  Misc', c2, delta, profile.eval_misc, b2, c2))
#         p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
#         p.add_row(*fmt_perf('  Forward', c2, delta, profile.train_forward, b2, c2))
#         p.add_row(*fmt_perf('  Learn', c2, delta, profile.learn, b2, c2))
#         p.add_row(*fmt_perf('  Copy', c2, delta, profile.train_copy, b2, c2))
#         p.add_row(*fmt_perf('  Misc', c2, delta, profile.train_misc, b2, c2))

#         l = Table(box=None, expand=True, )
#         l.add_column(f'{c1}Losses', justify="left", width=16)
#         l.add_column(f'{c1}Value', justify="right", width=8)
#         for metric, value in self.losses.items():
#             l.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')

#         monitor = Table(box=None, expand=True, pad_edge=False)
#         monitor.add_row(s, p, l)
#         dashboard.add_row(monitor)

#         table = Table(box=None, expand=True, pad_edge=False)
#         dashboard.add_row(table)
#         left = Table(box=None, expand=True)
#         right = Table(box=None, expand=True)
#         table.add_row(left, right)
#         left.add_column(f"{c1}User Stats", justify="left", width=20)
#         left.add_column(f"{c1}Value", justify="right", width=10)
#         right.add_column(f"{c1}User Stats", justify="left", width=20)
#         right.add_column(f"{c1}Value", justify="right", width=10)
#         i = 0

#         if self.stats:
#             self.last_stats = self.stats

#         for metric, value in (self.stats or self.last_stats).items():
#             try: # Discard non-numeric values
#                 int(value)
#             except:
#                 continue

#             u = left if i % 2 == 0 else right
#             u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
#             i += 1
#             if i == 100:
#                 break

#         if clear:
#             console.clear()

#         with console.capture() as capture:
#             console.print(dashboard)

#         print('\033[0;0H' + capture.get())

# def compute_puff_advantage(values, rewards, terminals,
#         ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
#     '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
#     nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
#     compile the fast version.'''

#     device = values.device
#     if not ADVANTAGE_CUDA:
#         values = values.cpu()
#         rewards = rewards.cpu()
#         terminals = terminals.cpu()
#         ratio = ratio.cpu()
#         advantages = advantages.cpu()

#     torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
#         ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

#     if not ADVANTAGE_CUDA:
#         return advantages.to(device)

#     return advantages


# def abbreviate(num, b2, c2):
#     if num < 1e3:
#         return str(num)
#     elif num < 1e6:
#         return f'{num/1e3:.1f}K'
#     elif num < 1e9:
#         return f'{num/1e6:.1f}M'
#     elif num < 1e12:
#         return f'{num/1e9:.1f}B'
#     else:
#         return f'{num/1e12:.2f}T'

# def duration(seconds, b2, c2):
#     seconds = int(seconds)
#     h = seconds // 3600
#     m = (seconds % 3600) // 60
#     s = seconds % 60
#     return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

# def fmt_perf(name, color, delta_ref, prof, b2, c2):
#     percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
#     return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

# def dist_sum(value, device):
#     if not torch.distributed.is_initialized():
#         return value

#     tensor = torch.tensor(value, device=device)
#     torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
#     return tensor.item()

# def dist_mean(value, device):
#     if not torch.distributed.is_initialized():
#         return value

#     return dist_sum(value, device) / torch.distributed.get_world_size()

# class Profile:
#     def __init__(self, frequency=5):
#         self.profiles = defaultdict(lambda: defaultdict(float))
#         self.frequency = frequency
#         self.stack = []

#     def __iter__(self):
#         return iter(self.profiles.items())

#     def __getattr__(self, name):
#         return self.profiles[name]

#     def __call__(self, name, epoch, nest=False):
#         if epoch % self.frequency != 0:
#             return

#         #if torch.cuda.is_available():
#         #    torch.cuda.synchronize()

#         tick = time.time()
#         if len(self.stack) != 0 and not nest:
#             self.pop(tick)

#         self.stack.append(name)
#         self.profiles[name]['start'] = tick

#     def pop(self, end):
#         profile = self.profiles[self.stack.pop()]
#         delta = end - profile['start']
#         profile['elapsed'] += delta
#         profile['delta'] += delta

#     def end(self):
#         #if torch.cuda.is_available():
#         #    torch.cuda.synchronize()

#         end = time.time()
#         for i in range(len(self.stack)):
#             self.pop(end)

#     def clear(self):
#         for prof in self.profiles.values():
#             if prof['delta'] > 0:
#                 prof['buffer'] = prof['delta']
#                 prof['delta'] = 0

# class Utilization(Thread):
#     def __init__(self, delay=1, maxlen=20):
#         super().__init__()
#         self.cpu_mem = deque([0], maxlen=maxlen)
#         self.cpu_util = deque([0], maxlen=maxlen)
#         self.gpu_util = deque([0], maxlen=maxlen)
#         self.gpu_mem = deque([0], maxlen=maxlen)
#         self.stopped = False
#         self.delay = delay
#         self.start()

#     def run(self):
#         while not self.stopped:
#             self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
#             mem = psutil.virtual_memory()
#             self.cpu_mem.append(100*mem.active/mem.total)
#             if torch.cuda.is_available():
#                 # Monitoring in distributed crashes nvml
#                 if torch.distributed.is_initialized():
#                    time.sleep(self.delay)
#                    continue

#                 self.gpu_util.append(torch.cuda.utilization())
#                 free, total = torch.cuda.mem_get_info()
#                 self.gpu_mem.append(100*(total-free)/total)
#             else:
#                 self.gpu_util.append(0)
#                 self.gpu_mem.append(0)

#             time.sleep(self.delay)

#     def stop(self):
#         self.stopped = True

# def downsample(arr, m):
#     if len(arr) < m:
#         return arr

#     if m == 0:
#         return [arr[-1]]

#     orig_arr = arr
#     last = arr[-1]
#     arr = arr[:-1]
#     arr = np.array(arr)
#     n = len(arr)
    
#     # If array is smaller than m after removing last element, just return original
#     if n < m:
#         return orig_arr
    
#     n = (n//m)*m
#     # If n becomes 0, we can't reshape, so return original array
#     if n == 0:
#         return orig_arr
        
#     arr = arr[-n:]
#     downsampled = arr.reshape(m, -1).mean(axis=1)
#     return np.concatenate([downsampled, [last]])

# class NoLogger:
#     def __init__(self, args):
#         self.run_id = str(int(100*time.time()))

#     def log(self, logs, step):
#         pass

#     def close(self, model_path):
#         pass

# class NeptuneLogger:
#     def __init__(self, args, load_id=None, mode='async'):
#         import neptune as nept
#         neptune_name = args['neptune_name']
#         neptune_project = args['neptune_project']
#         neptune = nept.init_run(
#             project=f"{neptune_name}/{neptune_project}",
#             capture_hardware_metrics=False,
#             capture_stdout=False,
#             capture_stderr=False,
#             capture_traceback=False,
#             with_id=load_id,
#             mode=mode,
#             tags = [args['tag']] if args['tag'] is not None else [],
#         )
#         self.run_id = neptune._sys_id
#         self.neptune = neptune
#         for k, v in pufferlib.unroll_nested_dict(args):
#             neptune[k].append(v)

#     def log(self, logs, step):
#         for k, v in logs.items():
#             self.neptune[k].append(v, step=step)

#     def close(self, model_path):
#         self.neptune['model'].track_files(model_path)
#         self.neptune.stop()

#     def download(self):
#         self.neptune["model"].download(destination='artifacts')
#         return f'artifacts/{self.run_id}.pt'
 
# class WandbLogger:
#     def __init__(self, args, load_id=None, resume='allow'):
#         import wandb
#         wandb.init(
#             id=load_id or wandb.util.generate_id(),
#             project=args['wandb_project'],
#             group=args['wandb_group'],
#             allow_val_change=True,
#             save_code=False,
#             resume=resume,
#             config=args,
#             tags = [args['tag']] if args['tag'] is not None else [],
#         )
#         self.wandb = wandb
#         self.run_id = wandb.run.id

#     def log(self, logs, step):
#         self.wandb.log(logs, step=step)

#     def close(self, model_path):
#         artifact = self.wandb.Artifact(self.run_id, type='model')
#         artifact.add_file(model_path)
#         self.wandb.run.log_artifact(artifact)
#         self.wandb.finish()

#     def download(self):
#         artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
#         data_dir = artifact.download()
#         model_file = max(os.listdir(data_dir))
#         return f'{data_dir}/{model_file}'
 
# def train(env_name, args=None, vecenv=None, policy=None, logger=None):
#     import os  # Ensure os is available in this scope
#     args = args or load_config(env_name)
#     vecenv = vecenv or load_env(env_name, args)
#     policy = policy or load_policy(args, vecenv)

#     # Stockfish is already enabled in the Chess environment constructor
#     # No need to enable it again here for regular training
#     if env_name == 'puffer_chess' and not args['env'].get('self_play', False):
#         print(f"[Chess] Stockfish opponent already enabled in environment (ELO={args['env'].get('stockfish_elo', 900)}, search_ms={args['env'].get('stockfish_search_ms', 10)})")

#     # Assume TorchRun DDP is used if LOCAL_RANK is set
#     if 'LOCAL_RANK' in os.environ:
#         world_size = int(os.environ.get('WORLD_SIZE', 1))
#         print("World size", world_size)
#         master_addr = os.environ.get('MASTER_ADDR', 'localhost')
#         master_port = os.environ.get('MASTER_PORT', '29500')
#         local_rank = int(os.environ["LOCAL_RANK"])
#         print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
#         torch.cuda.set_device(local_rank)
#         args['train']['device'] = torch.cuda.current_device()
#         torch.distributed.init_process_group(backend='nccl', world_size=world_size)
#         policy = policy.to(local_rank)
#         model = torch.nn.parallel.DistributedDataParallel(
#             policy, device_ids=[local_rank], output_device=local_rank
#         )
#         if hasattr(policy, 'lstm'):
#             #model.lstm = policy.lstm
#             model.hidden_size = policy.hidden_size

#         model.forward_eval = policy.forward_eval
#         policy = model.to(local_rank)

#     if args['neptune']:
#         logger = NeptuneLogger(args)
#     elif args['wandb']:
#         logger = WandbLogger(args)

#     train_config = dict(**args['train'], env=env_name)
#     pufferl = PuffeRL(train_config, vecenv, policy, logger)

#     # Initialize frozen policy for chess self-play
#     if 'chess' in env_name:
#         try:
#             # Initial frozen policy setup for all chess environments
#             policy_updates_applied = 0
            
#             # Save initial policy to shared location for multiprocessing workers
#             import tempfile
#             policy_file = os.path.join(tempfile.gettempdir(), 'puffer_chess_policy.pth')
#             import torch
#             torch.save(pufferl.policy.state_dict(), policy_file)
            
#             if hasattr(pufferl.vecenv, 'envs'):
#                 for env in pufferl.vecenv.envs:
#                     if hasattr(env, 'episode_per_color') and env.episode_per_color:
#                         env.update_frozen_policy(pufferl.policy)
#                         policy_updates_applied += 1
#             elif hasattr(pufferl.vecenv, 'notify') and hasattr(pufferl.vecenv, 'num_workers'):
#                 # For multiprocessing backend, use notify mechanism
#                 pufferl.vecenv.notify()
#                 total_games = pufferl.vecenv.num_workers * pufferl.vecenv.envs_per_worker
#                 policy_updates_applied = total_games
#             elif hasattr(pufferl.vecenv, 'episode_per_color') and pufferl.vecenv.episode_per_color:
#                 pufferl.vecenv.update_frozen_policy(pufferl.policy)
#                 policy_updates_applied += 1
#             elif hasattr(pufferl.vecenv, 'driver_env') and hasattr(pufferl.vecenv.driver_env, 'episode_per_color') and pufferl.vecenv.driver_env.episode_per_color:
#                 pufferl.vecenv.driver_env.update_frozen_policy(pufferl.policy)
#                 policy_updates_applied += 1
            
#             if policy_updates_applied > 0:
#                 print(f"[Chess Self-Play] Initial frozen policy set for {policy_updates_applied} environments")
#             else:
#                 print(f"[Chess Self-Play] No episode-per-color environments found - using dual-agent mode")
#         except Exception as e:
#             print(f"[Chess Self-Play] Failed to initialize frozen policy: {e}")
#             import traceback
#             traceback.print_exc()

#     all_logs = []
#     episode_count = 0
#     import time as time_module
#     while pufferl.global_step < train_config['total_timesteps']:
#         if pufferl.global_step % 100000 == 0:  # Log every 100K steps
#             print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Step {pufferl.global_step} / {train_config['total_timesteps']}")
        
#         eval_start = time_module.time()
#         print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] About to call evaluate() at step {pufferl.global_step}")
#         pufferl.evaluate()
#         eval_time = time_module.time() - eval_start
#         print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Evaluate() completed in {eval_time:.2f}s")
        
#         train_step_start = time_module.time()
#         print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] About to call train() at step {pufferl.global_step}")
#         logs = pufferl.train()
#         train_step_time = time_module.time() - train_step_start
#         print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Train() completed in {train_step_time:.2f}s, step now {pufferl.global_step}")
        
#         # Update frozen policy for episode-per-color chess self-play
#         if 'chess' in env_name:
#             episode_count += 1
#             # Get frozen policy update frequency from config args
#             update_frequency = args['env'].get('frozen_policy_update_frequency', 50)
            
#             if episode_count % update_frequency == 0:
#                 try:
#                     print(f"[Chess Debug] Policy update triggered at epoch {episode_count} (frequency: every {update_frequency} epochs)")
#                     # Schedule policy update for all chess environments
#                     policy_updates_scheduled = 0
                    
#                     # Save policy to shared location for multiprocessing workers
#                     import tempfile
#                     import os
#                     policy_file = os.path.join(tempfile.gettempdir(), 'puffer_chess_policy.pth')
#                     import torch
#                     torch.save(pufferl.policy.state_dict(), policy_file)
                    
#                     if hasattr(pufferl.vecenv, 'envs'):
#                         # print(f"[Chess Debug] Path 1: Found {len(pufferl.vecenv.envs)} environments")
#                         for env in pufferl.vecenv.envs:
#                             if hasattr(env, 'episode_per_color') and env.episode_per_color:
#                                 env.update_frozen_policy(pufferl.policy)
#                                 policy_updates_scheduled += 1
#                     elif hasattr(pufferl.vecenv, 'notify') and hasattr(pufferl.vecenv, 'num_workers'):
#                         total_games = pufferl.vecenv.num_workers * 512  # 4 workers * 512 games each
#                         # print(f"[Chess Debug] Using multiprocessing notify mechanism for {pufferl.vecenv.num_workers} workers ({total_games} total games)")
#                         # For multiprocessing backend, use notify mechanism
#                         pufferl.vecenv.notify()
#                         # Count all workers (each manages 512 games)
#                         policy_updates_scheduled = pufferl.vecenv.num_workers
#                     elif hasattr(pufferl.vecenv, 'episode_per_color') and pufferl.vecenv.episode_per_color:
#                         print(f"[Chess Debug] Path 2: Using vecenv directly")
#                         pufferl.vecenv.update_frozen_policy(pufferl.policy)
#                         policy_updates_scheduled += 1
#                     elif hasattr(pufferl.vecenv, 'driver_env') and hasattr(pufferl.vecenv.driver_env, 'episode_per_color') and pufferl.vecenv.driver_env.episode_per_color:
#                         print(f"[Chess Debug] Path 3: Using driver_env only (workers won't get updates!)")
#                         pufferl.vecenv.driver_env.update_frozen_policy(pufferl.policy)
#                         policy_updates_scheduled += 1
                    
#                     if policy_updates_scheduled > 0:
#                         print(f"[Chess Self-Play] Policy updated for {policy_updates_scheduled} environments (epoch {episode_count})")
#                 except Exception as e:
#                     print(f"[Chess Self-Play] Failed to update policy: {e}")
#                     import traceback
#                     traceback.print_exc()

#         if logs is not None:
#             if pufferl.global_step > 0.20*train_config['total_timesteps']:
#                 all_logs.append(logs)

#     # Final eval. You can reset the env here, but depending on
#     # your env, this can skew data (i.e. you only collect the shortest
#     # rollouts within a fixed number of epochs)
#     print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Training loop completed, starting final evaluation...")
#     final_eval_start = time_module.time()
#     i = 0
#     stats = {}
#     while i < 32 or not stats:
#         stats = pufferl.evaluate()
#         i += 1
#     final_eval_time = time_module.time() - final_eval_start
#     print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Final evaluation completed in {final_eval_time:.2f}s")

#     logging_start = time_module.time()
#     logs = pufferl.mean_and_log()
#     if logs is not None:
#         all_logs.append(logs)
#     logging_time = time_module.time() - logging_start
#     print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Final logging completed in {logging_time:.2f}s")

#     dashboard_start = time_module.time()
#     pufferl.print_dashboard()
#     dashboard_time = time_module.time() - dashboard_start
#     print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Dashboard print completed in {dashboard_time:.2f}s")
    
#     close_start = time_module.time()
#     model_path = pufferl.close()
#     close_time = time_module.time() - close_start
#     print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] PuffeRL close completed in {close_time:.2f}s")
    
#     # Clean up shared memory files for chess
#     cleanup_start = time_module.time()
#     if 'chess' in env_name:
#         try:
#             import tempfile
            
#             policy_sync_dir = os.path.join(tempfile.gettempdir(), 'puffer_chess_policies')
#             if os.path.exists(policy_sync_dir):
#                 for policy_file in glob.glob(os.path.join(policy_sync_dir, '*')):
#                     try:
#                         os.remove(policy_file)
#                     except:
#                         pass
#                 try:
#                     os.rmdir(policy_sync_dir)
#                 except:
#                     pass
#             print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Chess cleanup completed")
#         except Exception as e:
#             print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Failed to clean up shared policy files: {e}")
#     cleanup_time = time_module.time() - cleanup_start
#     print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] File cleanup completed in {cleanup_time:.2f}s")
    
#     # Don't close logger yet if we're in a sweep - let sweep handle cleanup
#     logger_close_start = time_module.time()
#     if not args.get('_sweep_mode', False):
#         print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Non-sweep mode: closing logger and pufferl...")
#         pufferl.logger.close(model_path)
#         pufferl.close()
#         logger_close_time = time_module.time() - logger_close_start
#         print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Logger/pufferl close completed in {logger_close_time:.2f}s")
#         return all_logs
#     else:
#         # In sweep mode, return both logs and pufferl instance for proper cleanup
#         logger_close_time = time_module.time() - logger_close_start
#         print(f"[TRAIN_DEBUG] [{time_module.strftime('%H:%M:%S')}] Sweep mode: returning pufferl for cleanup (took {logger_close_time:.2f}s)")
#         return all_logs, pufferl

# def train_selfplay(env_name='puffer_chess', config=None, use_engine_opponent=False, engine_depth=2, engine_path=None, engine_elo=1320, engine_search_ms=10):
#     """Training loop for self-play.

#     For chess, when self_play is enabled in config, uses native dual-agent self-play
#     where white and black are separate RL agents with shared network weights.
#     """

#     args = config or load_config(env_name)
#     device = args['train']['device']
    
#     # For chess, check if self_play is enabled in config
#     if env_name == 'puffer_chess':
#         config_self_play = args['env'].get('self_play', False)
#         print(f"[DEBUG] config_self_play = {config_self_play}")
#         print(f"[DEBUG] args['env'] = {args['env']}")
        
#         if config_self_play:
#             # Native dual-agent self-play - use environment directly like NMMO/MOBA
#             print("[Chess] Using native dual-agent self-play mode")
#             vecenv = load_env(env_name, args)
#             policy = load_policy(args, vecenv)
            
#             # Add env name so PuffeRL.print_dashboard can display it
#             train_config = dict(**args['train'], env=env_name)

#             # Set up logging
#             logger = None
#             if args['neptune']:
#                 logger = NeptuneLogger(args)
#             elif args['wandb']:
#                 logger = WandbLogger(args)

#             pufferl = PuffeRL(train_config, vecenv, policy, logger)
            
#             while pufferl.global_step < args['train']['total_timesteps']:
#                 pufferl.evaluate()
#                 pufferl.train()
                
#                 # Periodically save checkpoints
#                 if pufferl.epoch % 100 == 0:
#                     pufferl.save_checkpoint()
            
#             return pufferl.close()
#         else:
#             # Fall back to wrapper approach for backward compatibility
#             args['env']['self_play'] = not use_engine_opponent
#     else:
#         # For non-chess environments, use original logic
#         args['env']['self_play'] = not use_engine_opponent

#     # Original wrapper-based approach for chess when self_play is not in config,
#     # or for other environments
#     from pufferlib.ocean.chess.selfplay_wrapper import ChessSelfPlayWrapper

#     base_env = load_env(env_name, args)
#     policy = load_policy(args, base_env)

#     if use_engine_opponent:
#         # Native Stockfish integration – enable once per VecEnv.
#         from pufferlib.ocean.chess import binding

#         # The base_env can be a driver-specific wrapper (Serial, Ray, etc.).
#         # Attempt to locate the underlying C handle robustly.
#         c_vec = getattr(base_env, 'c_envs', None)
#         if c_vec is None and hasattr(base_env, 'driver_env') and base_env.driver_env is not None:
#             c_vec = getattr(base_env.driver_env, 'c_envs', None)

#         if c_vec is None:
#             raise pufferlib.APIUsageError('Failed to locate native Chess VecEnv handle for Stockfish toggle.')

#         # Toggle Stockfish for all sub-envs (black side only) – optional custom binary path
#         if engine_path:
#             binding.vec_enable_stockfish_black(c_vec, engine_path, engine_elo, engine_search_ms)
#         else:
#             # Let C++ auto-detect the Stockfish binary
#             binding.vec_enable_stockfish_black(c_vec, None, engine_elo, engine_search_ms)

#         # No additional Python plumbing needed – the C++ core now generates
#         # every black reply internally, so the plain environment is ready.
#         vecenv = base_env
#     else:
#         vecenv = ChessSelfPlayWrapper(base_env, policy, device=device)
    
#     # Add env name so PuffeRL.print_dashboard can display it
#     train_config = dict(**args['train'], env=env_name)

#     # Train with shared policy
#     if args['neptune']:
#         logger = NeptuneLogger(args)
#     elif args['wandb']:
#         logger = WandbLogger(args)
#     else:
#         logger = None

#     pufferl = PuffeRL(train_config, vecenv, policy, logger)
    
#     while pufferl.global_step < args['train']['total_timesteps']:
#         # During evaluation, both players use the current policy
#         pufferl.evaluate()
        
#         # During training, we train on games played between
#         # the current policy (both sides)
#         pufferl.train()
        
#         # Periodically save checkpoints
#         if pufferl.epoch % 100 == 0:
#             pufferl.save_checkpoint()
    
#     return pufferl.close()

# # ----------------------------------------------------------------------
# # Helper: arena-style self-play evaluation for Chess
# # ----------------------------------------------------------------------

# def evaluate_chess_self_play(policy, vecenv, args, num_games):
#     """Play `num_games` self-play games using `policy` on both sides and
#     return a dict with win/draw/loss counts and an approximate Elo delta.

#     The Elo estimate assumes a logistic model with draw=0.5."""

#     device = args['train']['device']
#     wins = draws = losses = 0

#     move_cap = args.get('move_limit', 1024)

#     start_time = time.time()

#     for game_idx in range(num_games):
#         obs, _ = vecenv.reset()

#         # Fresh recurrent state for each episode if the network is RNN-based
#         state = {}
#         if args['train']['use_rnn']:
#             hdim = policy.hidden_size
#             n_agents = vecenv.num_agents
#             state = {
#                 'lstm_h': torch.zeros(n_agents, hdim, device=device),
#                 'lstm_c': torch.zeros(n_agents, hdim, device=device),
#             }

#         done = np.array([False])
#         ply = 0
#         while not done.any():
#             with torch.no_grad():
#                 ob_t = torch.as_tensor(obs).to(device)
#                 logits, _ = policy.forward_eval(ob_t, state)
#                 action, _, _ = pytorch.sample_logits(logits)
#                 action = action.cpu().numpy().reshape(vecenv.action_space.shape)

#             obs, reward, done, _, _ = vecenv.step(action)

#             ply += 1
#             if ply >= move_cap:
#                 # Hard draw after move_cap half-moves to avoid pathological games
#                 done[...] = True
#                 reward[...] = 0.0  # Draw

#         # Terminal – outcome from white's perspective
#         final_r = reward[0]
#         if final_r > 1e-4:
#             wins += 1
#         elif final_r < -1e-4:
#             losses += 1
#         else:
#             draws += 1

#         # Progress output every game
#         elapsed = time.time() - start_time
#         eta = (elapsed / (game_idx + 1)) * (num_games - game_idx - 1)
#         print(f"[Eval] Game {game_idx+1}/{num_games} finished in {ply} ply. "
#               f"Score so far W/D/L: {wins}/{draws}/{losses}. ETA {eta:.1f}s", flush=True)

#     total = wins + draws + losses
#     score = (wins + 0.5 * draws) / total if total > 0 else 0.5
#     elo = 0.0
#     if 0.0 < score < 1.0:
#         elo = -400.0 * math.log10(1.0 / score - 1.0)

#     return {
#         'wins': wins,
#         'draws': draws,
#         'losses': losses,
#         'elo': elo,
#     }

# def eval(env_name, args=None, vecenv=None, policy=None):
#     args = args or load_config(env_name)
#     # args['render_mode'] = 'raylib'
#     # args['env']['render_mode'] = 'raylib'
#     args['vec'] = dict(backend='Serial', num_envs=1)
#     args['env']['self_play'] = True                      # C++ core toggles

#     # Automatically load the most recent checkpoint if the caller did not
#     # specify a concrete path via --load-model-path.  This makes
#     #   $ puffer eval puffer_chess
#     # work out-of-the-box after training without having to hunt for the
#     # checkpoint filename.
#     if args.get('load_model_path') is None:
#         args['load_model_path'] = 'latest'

#     # plain Chess env (already vectorised Serial)
#     base_env = load_env(env_name, args)
#     policy    = policy or load_policy(args, base_env)

#     # wrap only if wanted
#     if args.get('chess_self_play', True):
#         from pufferlib.ocean.chess.selfplay_wrapper import ChessSelfPlayWrapper
#         vecenv = ChessSelfPlayWrapper(base_env, policy,
#                                       device=args['train']['device'])
#     else:
#         vecenv = base_env
        
#     def is_serial(v):
#         return isinstance(v, vector.Serial)

#     # single guard, no duplicate afterwards
#     if not (is_serial(vecenv) or
#             (isinstance(vecenv, ChessSelfPlayWrapper) and is_serial(vecenv.env))):
#         raise pufferlib.APIUsageError('eval requires Serial vector env')

#     # ---------------------------------------------------------------
#     # Headless arena-style evaluation for self-play chess
#     # ---------------------------------------------------------------
#     eval_games = args.get('eval_games', 0)
#     if eval_games:
#         results = evaluate_chess_self_play(policy, vecenv, args, eval_games)
#         print(f"Self-play evaluation over {eval_games} games → "
#               f"Wins: {results['wins']} | Draws: {results['draws']} | "
#               f"Losses: {results['losses']} | Estimated Elo Δ: {results['elo']:.1f}")
#         return

#     # ---------- nothing else changes below ----------
#     ob, info = vecenv.reset()
#     driver = vecenv.driver_env                # forwarded by wrapper
#     num_agents = vecenv.observation_space.shape[0]
#     device = args['train']['device']

#     state = {}
#     if args['train']['use_rnn']:
#         state = dict(
#             lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
#             lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
#         )

#     # Special low-tech console renderer for quick debugging when evaluating
#     # the standalone chess environment.
#     if env_name == 'puffer_chess':
#         max_moves = 200
#         moves = 0
#         while moves < max_moves:
#             render = driver.render()          # string from C++ render()
#             print('\033[0;0H' + render)       # always print; forget raylib here

#             with torch.no_grad():
#                 ob_t = torch.as_tensor(ob).to(device)
#                 logits, _ = policy.forward_eval(ob_t, state)
#                 action, _, _ = pytorch.sample_logits(logits)
#                 action = action.cpu().numpy().reshape(vecenv.action_space.shape)

#             ob, rew, done, trunc, _ = vecenv.step(action)
#             moves += 1
#             if done.any(): break  
    
    
#     frames = []
#     while True:
#         render = driver.render()
#         if len(frames) < args['save_frames']:
#             frames.append(render)

#         # Screenshot Ocean envs with F12, gifs with control + F12
#         if driver.render_mode == 'ansi':
#             print('\033[0;0H' + render + '\n')
#             time.sleep(1/args['fps'])
#         elif driver.render_mode == 'rgb_array':
#             import cv2
#             render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
#             cv2.imshow('frame', render)
#             cv2.waitKey(1)
#             time.sleep(1/args['fps'])

#         with torch.no_grad():
#             ob = torch.as_tensor(ob).to(device)
#             logits, value = policy.forward_eval(ob, state)
#             action, logprob, _ = pytorch.sample_logits(logits)
#             action = action.cpu().numpy().reshape(vecenv.action_space.shape)

#         if isinstance(logits, torch.distributions.Normal):
#             action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

#         ob = vecenv.step(action)[0]

#         if len(frames) > 0 and len(frames) == args['save_frames']:
#             import imageio
#             imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
#             frames.append('Done')

#     if env_name == 'puffer_chess' and args['render_mode'] == 'auto':
#         args['render_mode'] = args['env']['render_mode'] = 'ansi'

# def sweep(args=None, env_name=None):
#     args = args or load_config(env_name)
#     if not args['wandb'] and not args['neptune']:
#         raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')

#     # Store original args to avoid corruption between runs
#     import copy
#     original_args = copy.deepcopy(args)
    
#     method = args['sweep'].pop('method')
#     try:
#         import pufferlib.sweep as sweep_module
#         sweep_cls = getattr(sweep_module, method)
#     except:
#         raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

#     sweep = sweep_cls(args['sweep'])
#     points_per_run = args['sweep']['downsample']
#     target_key = f'environment/{args["sweep"]["metric"]}'
#     # Track environment and pufferl instance for proper cleanup
#     current_vecenv = None
#     current_pufferl = None
    
#     for i in range(args['max_runs']):
#         import time as time_module
#         run_start_time = time_module.time()
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Starting run {i+1}/{args['max_runs']}")
        
#         # Clean up previous run's resources
#         if current_pufferl is not None:
#             cleanup_start = time_module.time()
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Closing previous run's pufferl...")
#             try:
#                 # Don't save checkpoint or close logger - just close the environment
#                 # Set a flag to avoid cleanup loops
#                 if hasattr(current_pufferl, 'vecenv'):
#                     current_pufferl.vecenv = None  # Prevent vecenv.close() in pufferl.close()
#                 current_pufferl.close()
#                 del current_pufferl
#                 current_pufferl = None
#                 cleanup_time = time_module.time() - cleanup_start
#                 print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] PuffeRL cleanup took {cleanup_time:.2f}s")
#             except Exception as e:
#                 cleanup_time = time_module.time() - cleanup_start
#                 print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Failed to close previous pufferl after {cleanup_time:.2f}s: {e}")
        
#         if current_vecenv is not None:
#             cleanup_start = time_module.time()
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Closing previous run's vecenv...")
#             try:
#                 # More defensive cleanup - check if vecenv is still valid before closing
#                 if hasattr(current_vecenv, 'close') and callable(getattr(current_vecenv, 'close', None)):
#                     current_vecenv.close()
#                 del current_vecenv
#                 current_vecenv = None
#                 cleanup_time = time_module.time() - cleanup_start
#                 print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Vecenv cleanup took {cleanup_time:.2f}s")
#             except Exception as e:
#                 cleanup_time = time_module.time() - cleanup_start
#                 print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Failed to close previous vecenv after {cleanup_time:.2f}s: {e}")
#                 # Force cleanup by setting to None even if close failed
#                 current_vecenv = None
        
#         # Clean up any stale policy files between runs for double_buffered_chess
#         if 'double_buffered_chess' in env_name:
#             import tempfile
#             import os
#             policy_file = os.path.join(tempfile.gettempdir(), 'puffer_chess_policy.pth')
#             if os.path.exists(policy_file):
#                 try:
#                     os.remove(policy_file)
#                     print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Cleaned up stale policy file for run {i+1}")
#                 except Exception as e:
#                     print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Failed to clean policy file: {e}")
        
#         # Force garbage collection and multiprocessing cleanup
#         gc_start = time_module.time()
#         import gc
#         gc.collect()
#         gc_time = time_module.time() - gc_start
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Garbage collection took {gc_time:.2f}s")
        
#         # Shorter sleep for multiprocessing cleanup - 5 seconds was excessive
#         if i > 0 and 'double_buffered_chess' in env_name:
#             sleep_start = time_module.time()
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Waiting 2 seconds for multiprocessing cleanup...")
#             time.sleep(2)
#             sleep_time = time_module.time() - sleep_start
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Sleep completed in {sleep_time:.2f}s")
        
#         setup_start = time_module.time()
        
#         # CRITICAL FIX: Reload base configuration for each run to avoid parameter corruption
#         base_args = load_config(env_name)
#         # Copy over sweep-specific settings
#         args = base_args
#         args['wandb'] = base_args.get('wandb', False) or original_args.get('wandb', False)
#         args['neptune'] = base_args.get('neptune', False) or original_args.get('neptune', False)
#         args['sweep'] = original_args['sweep']
        
#         seed = time.time_ns() & 0xFFFFFFFF
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
        
#         # Debug: Print args before sweep.suggest
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Before sweep.suggest - num_envs: {args['env']['num_envs']}, batch_size: {args['train']['batch_size']}")
        
#         sweep.suggest(args)
        
#         # Debug: Print args after sweep.suggest
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] After sweep.suggest - num_envs: {args['env']['num_envs']}, batch_size: {args['train']['batch_size']}")
        
#         total_timesteps = args['train']['total_timesteps']
#         setup_time = time_module.time() - setup_start
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Run setup took {setup_time:.2f}s")
        
#         # Mark that we're in sweep mode so train function returns pufferl instance
#         args['_sweep_mode'] = True
        
#         # Validate parameters before creating environment
#         def validate_parameters(args):
#             batch_size = args['train']['batch_size']
#             bptt_horizon = args['train']['bptt_horizon']
#             num_envs = args['env']['num_envs']
            
#             # Basic sanity checks
#             if batch_size <= 0 or bptt_horizon <= 0 or num_envs <= 0:
#                 return False, "Batch size, BPTT horizon, and num_envs must be positive"
            
#             # Check if configuration is likely to cause resource issues
#             total_agents = num_envs * 2  # Chess has 2 agents per env
#             segments = batch_size // bptt_horizon
            
#             if total_agents > segments:
#                 return False, f"Total agents {total_agents} > segments {segments}. This will cause training to hang."
            
#             # Memory usage estimate (rough heuristic)
#             estimated_memory_gb = (batch_size * bptt_horizon * 8 * 4) / (1024**3)  # 8 obs dims, 4 bytes per float
#             if estimated_memory_gb > 8:  # Arbitrary threshold
#                 return False, f"Estimated memory usage {estimated_memory_gb:.1f}GB exceeds reasonable limits"
            
#             return True, "Valid"
        
#         # Validate parameters
#         is_valid, validation_msg = validate_parameters(args)
#         if not is_valid:
#             print(f"[SWEEP_ERROR] [{time_module.strftime('%H:%M:%S')}] Run {i+1} has invalid parameters: {validation_msg}")
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Skipping to next run")
#             continue
        
#         # Create fresh environment and policy for this run
#         env_start = time_module.time()
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Creating fresh environment for run {i+1}")
        
#         try:
#             current_vecenv = load_env(env_name, args)
#             policy = load_policy(args, current_vecenv)
#         except Exception as e:
#             print(f"[SWEEP_ERROR] [{time_module.strftime('%H:%M:%S')}] Failed to create environment for run {i+1}: {e}")
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Skipping to next run")
#             continue
            
#         env_time = time_module.time() - env_start
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Environment creation took {env_time:.2f}s")
        
#         # Create fresh logger for this run to avoid state conflicts
#         logger = None
#         if args['neptune']:
#             logger = NeptuneLogger(args)
#         elif args['wandb']:
#             logger = WandbLogger(args)
        
#         # Train with the fresh environment and capture PuffeRL instance for cleanup
#         train_start = time_module.time()
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Starting training for run {i+1}")
        
#         # Add robust error handling and timeout for training
#         train_result = None
#         training_failed = False
#         timeout_seconds = 600  # 10 minute timeout per run
        
#         try:
#             import signal
            
#             def timeout_handler(signum, frame):
#                 raise TimeoutError(f"Training run {i+1} timed out after {timeout_seconds} seconds")
            
#             # Set up timeout
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(timeout_seconds)
            
#             try:
#                 train_result = train(env_name, args=args, vecenv=current_vecenv, policy=policy, logger=logger)
#             finally:
#                 signal.alarm(0)  # Cancel timeout
                
#         except TimeoutError as e:
#             print(f"[SWEEP_ERROR] [{time_module.strftime('%H:%M:%S')}] {e}")
#             training_failed = True
#         except Exception as e:
#             print(f"[SWEEP_ERROR] [{time_module.strftime('%H:%M:%S')}] Training run {i+1} failed with error: {e}")
#             training_failed = True
        
#         train_time = time_module.time() - train_start
        
#         if training_failed:
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Training run {i+1} failed after {train_time:.2f}s, skipping to next run")
#             # Clean up failed run
#             if current_pufferl is not None:
#                 try:
#                     current_pufferl.close()
#                 except:
#                     pass
#             if current_vecenv is not None:
#                 try:
#                     current_vecenv.close()
#                 except:
#                     pass
#             continue  # Skip to next run
            
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Training completed in {train_time:.2f}s")
        
#         if isinstance(train_result, tuple):
#             all_logs, current_pufferl = train_result
#         else:
#             all_logs = train_result
#             current_pufferl = None
            
#         processing_start = time_module.time()
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Total logs: {len(all_logs)}")
#         if len(all_logs) > 0:
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Sample log keys: {list(all_logs[0].keys())}")
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Looking for target_key: {target_key}")
        
#         all_logs = [e for e in all_logs if target_key in e]
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Filtered logs with target key: {len(all_logs)}")
        
#         if len(all_logs) == 0:
#             print(f"[SWEEP_ERROR] [{time_module.strftime('%H:%M:%S')}] No logs found with target key '{target_key}' - sweep cannot continue!")
#             # Clean up before returning
#             if current_pufferl is not None:
#                 try:
#                     current_pufferl.close()
#                 except:
#                     pass
#             if current_vecenv is not None:
#                 try:
#                     current_vecenv.close()
#                 except:
#                     pass
#             return
            
#         scores = downsample([log[target_key] for log in all_logs], points_per_run)
#         costs = downsample([log['uptime'] for log in all_logs], points_per_run)
#         timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Downsampled: {len(scores)} scores, {len(costs)} costs")
#         for score, cost, timestep in zip(scores, costs, timesteps):
#             args['train']['total_timesteps'] = timestep
#             sweep.observe(args, score, cost)

#         # Prevent logging final eval steps as training steps
#         args['train']['total_timesteps'] = total_timesteps
#         processing_time = time_module.time() - processing_start
#         total_run_time = time_module.time() - run_start_time
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Log processing took {processing_time:.2f}s")
#         print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Total run {i+1} time: {total_run_time:.2f}s")
    
#     # Final cleanup
#     import time as time_module
#     final_cleanup_start = time_module.time()
#     print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Starting final cleanup...")
    
#     if current_pufferl is not None:
#         try:
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Final cleanup: closing pufferl...")
#             # Close logger first
#             model_path = current_pufferl.save_checkpoint()
#             current_pufferl.logger.close(model_path)
#             # Then close pufferl
#             current_pufferl.close()
#         except Exception as e:
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Failed final pufferl cleanup: {e}")
    
#     if current_vecenv is not None:
#         try:
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Final cleanup: closing vecenv...")
#             current_vecenv.close()
#         except Exception as e:
#             print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Failed final vecenv cleanup: {e}")
    
#     final_cleanup_time = time_module.time() - final_cleanup_start
#     print(f"[SWEEP_DEBUG] [{time_module.strftime('%H:%M:%S')}] Final cleanup took {final_cleanup_time:.2f}s")

# def profile(args=None, env_name=None, vecenv=None, policy=None):
#     args = load_config()
#     vecenv = vecenv or load_env(env_name, args)
#     policy = policy or load_policy(args, vecenv)

#     train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])
#     pufferl = PuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

#     import torchvision.models as models
#     from torch.profiler import profile, record_function, ProfilerActivity
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#         with record_function("model_inference"):
#             for _ in range(10):
#                 stats = pufferl.evaluate()
#                 pufferl.train()

#     print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
#     prof.export_chrome_trace("trace.json")

# def export(args=None, env_name=None, vecenv=None, policy=None):
#     args = args or load_config(env_name)
#     vecenv = vecenv or load_env(env_name, args)
#     policy = policy or load_policy(args, vecenv)

#     weights = []
#     for name, param in policy.named_parameters():
#         weights.append(param.data.cpu().numpy().flatten())
#         print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
#     path = f'{args["env_name"]}_weights.bin'
#     weights = np.concatenate(weights)
#     weights.tofile(path)
#     print(f'Saved {len(weights)} weights to {path}')

# def autotune(args=None, env_name=None, vecenv=None, policy=None):
#     package = args['package']
#     module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
#     env_module = importlib.import_module(module_name)
#     env_name = args['env_name']
#     make_env = env_module.env_creator(env_name)
#     vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
# def load_env(env_name, args):
#     package = args['package']
#     module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
#     env_module = importlib.import_module(module_name)
#     make_env = env_module.env_creator(env_name)
    
#     # Filter env_kwargs to only include parameters accepted by the environment
#     env_kwargs = args['env'].copy()
    
#     # For chess environment, filter out moves_per_episode if using base Chess class
#     if env_name == 'puffer_chess' and 'moves_per_episode' in env_kwargs:
#         # moves_per_episode is only used by DoubleBufferedChess wrapper
#         print(f"[DEBUG] Removing moves_per_episode from chess env_kwargs (only used in double_buffered_chess)")
#         env_kwargs.pop('moves_per_episode')
    
#     # Also remove frozen_policy_update_frequency as it's not a constructor param
#     if 'frozen_policy_update_frequency' in env_kwargs:
#         env_kwargs.pop('frozen_policy_update_frequency')
    
#     print(f"[DEBUG] load_env: env_kwargs = {env_kwargs}")
#     return vector.make(make_env, env_kwargs=env_kwargs, **args['vec'])

# def load_policy(args, vecenv):
#     package = args['package']
#     module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
#     env_module = importlib.import_module(module_name)

#     device = args['train']['device']
#     policy_cls = getattr(env_module.torch, args['policy_name'])
#     policy = policy_cls(vecenv, **args['policy'])

#     rnn_name = args['rnn_name']
#     if rnn_name is not None:
#         rnn_cls = getattr(env_module.torch, args['rnn_name'])
#         policy = rnn_cls(vecenv, policy, **args['rnn'])

#     policy = policy.to(device)

#     load_id = args['load_id']
#     if load_id is not None:
#         if args['neptune']:
#             path = NeptuneLogger(args, load_id, mode='read-only').download()
#         elif args['wandb']:
#             path = WandbLogger(args, load_id).download()
#         else:
#             raise pufferlib.APIUsageError('No run id provided for eval')

#         state_dict = torch.load(path, map_location=device)
#         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#         policy.load_state_dict(state_dict)

#     # Handle resume flag - automatically sets load_model_path to latest
#     if args.get('resume', False):
#         args['load_model_path'] = 'latest'

#     load_path = args['load_model_path']
#     if load_path == 'latest':
#         # Look for checkpoints in standard experiments directory
#         checkpoint_patterns = [
#             "experiments/*.pt",
#             "*.pt",
#             "checkpoints/*.pt", 
#             "models/*.pt"
#         ]
#         all_checkpoints = []
#         for pattern in checkpoint_patterns:
#             all_checkpoints.extend(glob.glob(pattern))
        
#         if all_checkpoints:
#             load_path = max(all_checkpoints, key=os.path.getctime)
#             print(f"[Resume] Loading latest checkpoint: {load_path}")
#         else:
#             print("[Resume] No checkpoints found - starting from scratch")
#             load_path = None

#     if load_path is not None:
#         state_dict = torch.load(load_path, map_location=device)
#         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
#         try:
#             policy.load_state_dict(state_dict)
#             print(f"[Resume] Loaded model weights from {load_path}")
#         except RuntimeError as e:
#             if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
#                 print(f"[Resume] Architecture mismatch detected in {load_path}")
#                 print(f"[Resume] This likely means the model was saved with a different architecture")
#                 print(f"[Resume] Starting fresh training instead of resuming")
#                 print(f"[Resume] Error: {str(e)[:200]}...")
#                 # Don't load the state dict, but don't fail either - just start fresh
#                 return policy
#             else:
#                 # Re-raise other errors
#                 raise e
        
#         # For resume, also try to load training state if available
#         if args.get('resume', False):
#             try:
#                 # Look for trainer state in the same directory
#                 checkpoint_dir = os.path.dirname(load_path)
#                 state_path = os.path.join(checkpoint_dir, 'trainer_state.pt')
#                 if os.path.exists(state_path):
#                     trainer_state = torch.load(state_path, map_location=device)
#                     print(f"[Resume] Found trainer state at {state_path}")
#                     # Store trainer state in args for use during training initialization
#                     args['_resume_trainer_state'] = trainer_state
#                 else:
#                     print(f"[Resume] No trainer state found at {state_path} - optimizer will start fresh")
#             except Exception as e:
#                 print(f"[Resume] Warning: Could not load trainer state: {e}")

#     return policy

# def load_config(env_name):
#     parser = argparse.ArgumentParser(
#         description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
#         ' demo options. Shows valid args for your env and policy',
#         formatter_class=RichHelpFormatter, add_help=False)
#     parser.add_argument('--load-model-path', type=str, default=None,
#         help='Path to a pretrained checkpoint')
#     parser.add_argument('--load-id', type=str,
#         default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
#     parser.add_argument('--resume', action='store_true',
#         help='Resume training from the latest checkpoint (loads model + training state)')
#     parser.add_argument('--render-mode', type=str, default='auto',
#         choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
#     parser.add_argument('--save-frames', type=int, default=0)
#     parser.add_argument('--gif-path', type=str, default='eval.gif')
#     parser.add_argument('--fps', type=float, default=15)
#     parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
#     parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
#     parser.add_argument('--wandb-project', type=str, default='pufferlib')
#     parser.add_argument('--wandb-group', type=str, default='debug')
#     parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
#     parser.add_argument('--neptune-name', type=str, default='pufferai')
#     parser.add_argument('--neptune-project', type=str, default='ablations')
#     parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
#     parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
#     parser.add_argument('--eval-games', type=int, default=0,
#         help='Number of self-play evaluation games to run in headless mode (chess only)')
#     parser.add_argument('--move-limit', type=int, default=1024,
#         help='Maximum ply (half-moves) per game during headless chess evaluation; draws after this')
#     args = parser.parse_known_args()[0]

#     # Load defaults and config
#     puffer_dir = os.path.dirname(os.path.realpath(__file__))
#     puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
#     puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
#     if env_name == 'default':
#         p = configparser.ConfigParser()
#         p.read(puffer_default_config)
#     else:
#         for path in glob.glob(puffer_config_dir, recursive=True):
#             p = configparser.ConfigParser()
#             p.read([puffer_default_config, path])
#             if env_name in p['base']['env_name'].split(): break
#         else:
#             raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

#     # Dynamic help menu from config
#     def auto_type(value):
#         """Type inference for numeric args that use 'auto' as a default value"""
#         if value == 'auto': return value
#         if value.isnumeric(): return int(value)
#         return float(value)

#     for section in p.sections():
#         for key in p[section]:
#             try:
#                 value = ast.literal_eval(p[section][key])
#             except:
#                 value = p[section][key]

#             fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
#             parser.add_argument(
#                 fmt.replace('_', '-'),
#                 default=value,
#                 type=auto_type if value == 'auto' else type(value)
#             )

#     parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
#         action='help', help='Show this help message and exit')

#     # Unpack to nested dict
#     parsed = vars(parser.parse_args())
#     args = defaultdict(dict)
#     for key, value in parsed.items():
#         next = args
#         for subkey in key.split('.'):
#             prev = next
#             next = next.setdefault(subkey, {})

#         prev[subkey] = value

#     args['train']['use_rnn'] = args['rnn_name'] is not None
#     return args

# def main():
#     err = 'Usage: puffer [train, eval, sweep, autotune, profile, export, selfplay] [env_name] [optional args]. --help for more info'
#     if len(sys.argv) < 3:
#         raise pufferlib.APIUsageError(err)

#     mode = sys.argv.pop(1)
#     env_name = sys.argv.pop(1)
#     if mode == 'train':
#         train(env_name=env_name)
#     elif mode == 'eval':
#         eval(env_name=env_name)
#     elif mode == 'sweep':
#         sweep(env_name=env_name)
#     elif mode == 'autotune':
#         autotune(env_name=env_name)
#     elif mode == 'profile':
#         profile(env_name=env_name)
#     elif mode == 'export':
#         export(env_name=env_name)
#     elif mode == 'selfplay':
#         train_selfplay(env_name=env_name)
#     else:
#         raise pufferlib.APIUsageError(err)

# if __name__ == '__main__':
#     main()










## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
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

import numpy as np
import psutil

import torch
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
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Assume advantage kernel has been built if CUDA compiler is available
ADVANTAGE_CUDA = shutil.which("nvcc") is not None

class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config['seed']
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience
        if config['batch_size'] == 'auto' and config['bptt_horizon'] == 'auto':
            raise pufferlib.APIUsageError('Must specify batch_size or bptt_horizon')
        elif config['batch_size'] == 'auto':
            config['batch_size'] = total_agents * config['bptt_horizon']
        elif config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = config['batch_size'] // total_agents

        batch_size = config['batch_size']
        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        device = config['device']
        self.observations = torch.zeros(segments, horizon, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents

        # LSTM
        if config['use_rnn']:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f'minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly')

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f'batch_size {batch_size} must be >= minibatch_size {minibatch_size}'
            )

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}'
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        if config['compile']:
            self.policy = torch.compile(policy, mode=config['compile_mode'])
            self.policy.forward_eval = torch.compile(policy, mode=config['compile_mode'])
            pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, mode=config['compile_mode'])

        # Optimizer
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
            )
        elif config['optimizer'] == 'muon':
            from heavyball import ForeachMuon
            warnings.filterwarnings(action='ignore', category=UserWarning, module=r'heavyball.*')
            import heavyball.utils
            heavyball.utils.compile_mode = config['compile_mode'] if config['compile'] else None
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
            )
        else:
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config['total_timesteps'] // config['batch_size']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = contextlib.nullcontext()
        if config.get('amp', True) and config['device'] == 'cuda':
            self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def evaluate(self, count_steps: bool = True):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']

        if config['use_rnn']:
            for k in self.lstm_h:
                self.lstm_h[k] = torch.zeros(self.lstm_h[k].shape, device=device)
                self.lstm_c[k] = torch.zeros(self.lstm_c[k].shape, device=device)

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t # TODO: Handle truncations separately
            if count_steps:
                self.global_step += int(mask.sum())

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)#, non_blocking=True)
            r = torch.as_tensor(r).to(device)#, non_blocking=True)
            d = torch.as_tensor(d).to(device)#, non_blocking=True)

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state['lstm_h'] = self.lstm_h[env_id.start]
                    state['lstm_c'] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1+self.ep_indices[env_id.stop - 1].item())

                if config['cpu_offload']:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l+1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile('env', epoch)
            self.vecenv.send(action)

        profile('eval_misc', epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile('train_misc', epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(self.values, self.rewards,
                self.terminals, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            profile('train_copy', epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            adv = advantages[idx]
            adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
                ratio, adv, config['gamma'], config['gae_lambda'],
                config['vtrace_rho_clip'], config['vtrace_c_clip'])
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            self.amp_context.__enter__() # TODO: AMP needs some debugging

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.total_minibatches
            losses['value_loss'] += v_loss.item() / self.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.total_minibatches
            losses['importance'] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses['explained_variance'] = explained_var.item()

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            'SPS': dist_sum(self.sps, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               self.logger.log(logs, agent_steps)
               return logs
           else:
               return None

        # Ensure strictly increasing step values for external loggers (e.g., Neptune)
        if agent_steps <= self.last_log_step:
            return None
        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        try:
            self.utilization.join(timeout=2)
        except Exception:
            pass
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.rename(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[white]', b1='[bright_cyan]', b2='[bright_white]'):
        config = self.config
        sps = dist_sum(self.sps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
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
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0 {idx[0]*" "}:blowfish:',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = 'A hair past a freckle'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{c2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{c2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{c2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{c2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{c2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{c2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{c2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf('  Forward', c2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf('  Env', c2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf('  Copy', c2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', c2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf('  Forward', c2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf('  Learn', c2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf('  Copy', c2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', c2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')

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

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version.'''

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return str(num)
    elif num < 1e6:
        return f'{num/1e3:.1f}K'
    elif num < 1e9:
        return f'{num/1e6:.1f}M'
    elif num < 1e12:
        return f'{num/1e9:.1f}B'
    else:
        return f'{num/1e12:.2f}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        if epoch % self.frequency != 0:
            return

        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['elapsed'] += delta
        profile['delta'] += delta

    def end(self):
        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.daemon = True
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        try:
            while not self.stopped:
                self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
                mem = psutil.virtual_memory()
                self.cpu_mem.append(100*mem.active/mem.total)
                if torch.cuda.is_available():
                    # Monitoring in distributed crashes nvml
                    if torch.distributed.is_initialized():
                       time.sleep(self.delay)
                       continue

                    try:
                        self.gpu_util.append(torch.cuda.utilization())
                        free, total = torch.cuda.mem_get_info()
                        self.gpu_mem.append(100*(total-free)/total)
                    except Exception:
                        self.gpu_util.append(0)
                        self.gpu_mem.append(0)
                else:
                    self.gpu_util.append(0)
                    self.gpu_mem.append(0)

                time.sleep(self.delay)
        except Exception:
            pass

    def stop(self):
        self.stopped = True

def downsample(arr, m):
    if len(arr) < m:
        return arr

    if m == 0:
        return [arr[-1]]

    orig_arr = arr
    last = arr[-1]
    arr = arr[:-1]
    arr = np.array(arr)
    n = len(arr)
    n = (n//m)*m
    arr = arr[-n:]
    downsampled = arr.reshape(m, -1).mean(axis=1)
    return np.concatenate([downsampled, [last]])

class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100*time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
        pass

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        self.args = args
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def close(self, model_path):
        # Only upload artifacts when explicitly enabled to avoid sweep slowdowns
        upload_flag = False
        try:
            upload_flag = bool(int(self.args.get('train', {}).get('upload_model_artifact', 0)))
        except Exception:
            upload_flag = bool(self.args.get('train', {}).get('upload_model_artifact', False))

        if upload_flag and model_path is not None:
            self.neptune['model'].track_files(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'
 
class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
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
        )
        self.wandb = wandb
        self.run_id = wandb.run.id

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def close(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'
 
def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = dict(**args['train'], env=env_name)
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    all_logs = []
    while pufferl.global_step < train_config['total_timesteps']:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20*train_config['total_timesteps']:
                all_logs.append(logs)

    # Optional final eval for reporting, disabled by default for sweeps
    final_eval_iters = int(train_config.get('final_eval_iters', 0))
    for _ in range(final_eval_iters):
        pufferl.evaluate(count_steps=False)

    # Only log a final summary if we advanced beyond the last logged step
    if pufferl.global_step > pufferl.last_log_step:
        logs = pufferl.mean_and_log()
        if logs is not None:
            all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
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
            frames.append('Done')

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sweep.suggest(args)
        total_timesteps = args['train']['total_timesteps']
        all_logs = train(env_name, args=args)
        all_logs = [e for e in all_logs if target_key in e]
        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)
        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace("trace.json")

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
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

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
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

    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['rnn'])

    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
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

def load_config(env_name):
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='xinpw8')
    parser.add_argument('--neptune-project', type=str, default='chess')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    args = parser.parse_known_args()[0]

    # Load defaults and config
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

    # Dynamic help menu from config
    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    for section in p.sections():
        for key in p[section]:
            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=puffer_type(p[section][key]),
                type=puffer_type
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

    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info'
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
    elif mode == 'autotune':
        autotune(env_name=env_name)
    elif mode == 'profile':
        profile(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()