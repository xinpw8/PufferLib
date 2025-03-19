from pdb import set_trace as T
import numpy as np

import os
import random
import psutil
import time

from threading import Thread
from collections import defaultdict, deque

import rich
from rich.console import Console
from rich.table import Table
import torch
import torch.distributed as dist

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

torch.set_float32_matmul_precision('high')

# Fast Cython advantage functions
#from c_advantage import rewards_and_masks, compute_gae
from c_advantage import compute_gae

import torch
from torch.utils.cpp_extension import load

# Compile the CUDA kernel
cuda_module = load(
    name='advantage_kernel',
    sources=['c_advantage.cu'],
    verbose=True
)

def compute_advantages(
    reward_block: torch.Tensor,  # [num_steps, horizon]
    reward_mask: torch.Tensor,   # [num_steps, horizon]
    values_mean: torch.Tensor,   # [num_steps, horizon]
    values_std: torch.Tensor,    # [num_steps, horizon]
    buf: torch.Tensor,          # [num_steps, horizon]
    dones: torch.Tensor,        # [num_steps]
    rewards: torch.Tensor,      # [num_steps]
    advantages: torch.Tensor,   # [num_steps]
    bounds: torch.Tensor,       # [num_steps]
    horizon: int
):
    assert all(t.is_cuda for t in [reward_block, reward_mask, values_mean, values_std, 
                                  buf, dones, rewards, advantages, bounds]), "All tensors must be on GPU"
    
    # Ensure contiguous memory
    tensors = [reward_block, reward_mask, values_mean, values_std, buf, dones, rewards, advantages]
    for t in tensors:
        t.contiguous()

    num_steps = rewards.shape[0]
    
    # Precompute vstd_min and vstd_max
    vstd_max = values_std.max().item()
    vstd_min = values_std.min().item()

    # Launch kernel
    threads_per_block = 256
    blocks = (num_steps + threads_per_block - 1) // threads_per_block
    
    cuda_module.advantage_kernel(
        reward_block,
        reward_mask,
        values_mean,
        values_std,
        buf,
        dones,
        rewards,
        advantages,
        bounds,
        num_steps,
        horizon,
        vstd_min,
        vstd_max,
    )
    
    torch.cuda.synchronize()
    return advantages

def create(config, vecenv, policy, optimizer=None, wandb=None, neptune=None):
    seed_everything(config.seed, config.torch_deterministic)
    profile = Profile()
    losses = make_losses()

    utilization = Utilization()
    msg = f'Model Size: {abbreviate(count_params(policy))} parameters'
    print_dashboard(config.env, utilization, 0, 0, profile, losses, {}, msg, clear=True)

    vecenv.async_reset(config.seed)
    obs_shape = vecenv.single_observation_space.shape
    obs_dtype = vecenv.single_observation_space.dtype
    atn_shape = vecenv.single_action_space.shape
    atn_dtype = vecenv.single_action_space.dtype
    total_agents = vecenv.num_agents

    lstm = policy.recurrent if hasattr(policy, 'recurrent') else None
    policy.hidden_size = 128
    experience = Experience(config.batch_size, config.bptt_horizon,
        config.minibatch_size, policy.hidden_size, obs_shape, obs_dtype,
        atn_shape, atn_dtype, config.cpu_offload, config.device, lstm, total_agents,
        use_e3b=config.use_e3b, e3b_coef=config.e3b_coef,
        use_p3o=config.use_p3o, p3o_horizon=config.p3o_horizon
    )

    uncompiled_policy = policy
    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode, fullgraph=config.compile_fullgraph)

    optimizer = torch.optim.Adam(policy.parameters(),
        lr=config.learning_rate, eps=1e-5)

    return pufferlib.namespace(
        config=config,
        vecenv=vecenv,
        policy=policy,
        uncompiled_policy=uncompiled_policy,
        optimizer=optimizer,
        experience=experience,
        profile=profile,
        losses=losses,
        wandb=wandb,
        neptune=neptune,
        global_step=0,
        epoch=0,
        stats=defaultdict(list),
        msg=msg,
        last_log_time=0,
        utilization=utilization,
        use_p3o=config.use_p3o,
        p3o_horizon=config.p3o_horizon,
        use_e3b=config.use_e3b,
        e3b_coef=config.e3b_coef,
        e3b_norm=config.e3b_norm,
    )

@pufferlib.utils.profile
def evaluate(data):
    profile = data.profile
    with profile.eval_misc:
        config = data.config
        experience = data.experience
        policy = data.policy
        infos = defaultdict(list)
        lstm_h = experience.lstm_h
        lstm_c = experience.lstm_c

    while not experience.full:
        with profile.env:
            o, r, d, t, info, env_id, mask = data.vecenv.recv()

            # Zero-copy indexing for contiguous env_id
            if config.env_batch_size == 1:
                gpu_env_id = cpu_env_id = slice(env_id[0], env_id[-1] + 1)
            else:
                cpu_env_id = env_id
                gpu_env_id = torch.as_tensor(env_id).to(config.device, non_blocking=True)

        with profile.eval_misc:
            done_mask = d + t
            data.global_step += mask.sum()

        with profile.eval_copy:
            if data.use_e3b and done_mask.any():
                done_idxs = env_id[done_mask]
                experience.e3b_inv[done_idxs] = experience.e3b_orig[done_idxs]

            o = torch.as_tensor(o)
            o_device = o.to(config.device, non_blocking=True)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)

            if lstm_h is not None:
                h = lstm_h[0, gpu_env_id]
                c = lstm_c[0, gpu_env_id]

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_forward, torch.no_grad():
            if data.use_p3o:
                if lstm_h is not None:
                    (logits, value_mean, value_std), hidden, (h, c) = policy(o_device, (h, c))
                else:
                    (logits, value_mean, value_std), hidden = policy(o_device)
            else:
                if lstm_h is not None:
                    (logits, value), hidden, (h, c) = policy(o_device, (h, c))
                else:
                    (logits, value), hidden = policy(o_device)
                value = value.flatten()

            actions, logprob, _ = pufferlib.pytorch.sample_logits(logits, is_continuous=policy.is_continuous)

            if data.use_e3b:
                e3b = experience.e3b_inv[env_id]
                phi = hidden.detach()        
                u = phi.unsqueeze(1) @ e3b
                b = u @ phi.unsqueeze(2)
                experience.e3b_inv[env_id] -= (u.mT @ u) / (1 + b)
                experience.e3b_inv[done_mask] = experience.e3b_orig[done_mask]
                e3b_reward = b.squeeze()

                if experience.e3b_mean is None:
                    experience.e3b_mean = e3b_reward.mean()
                    experience.e3b_std = e3b_reward.std()
                else:
                    w = data.e3b_norm
                    experience.e3b_mean = (1-w)*e3b_reward.mean() + w*experience.e3b_mean
                    experience.e3b_std = (1-w)*e3b_reward.std() + w*experience.e3b_std

                e3b_reward = (e3b_reward - experience.e3b_mean) / (experience.e3b_std + 1e-6)
                e3b_reward = config.e3b_coef*e3b_reward
                r += e3b_reward.cpu()

            # Clip rewards
            r = torch.clamp(r, -1, 1)

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_copy, torch.no_grad():
            if lstm_h is not None:
                lstm_h[:, gpu_env_id] = h
                lstm_c[:, gpu_env_id] = c

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_copy:
            o = o if config.cpu_offload else o_device

            if data.use_p3o:
                actions = experience.store(o, value_mean, value_std.detach(), actions, logprob, r, d, cpu_env_id, mask)
            else:
                actions = experience.store(o, value, None, actions, logprob, r, d, cpu_env_id, mask)

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_misc:
            for i in info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    infos[k].append(v)

        with profile.env:
            data.vecenv.send(actions)

    with profile.eval_misc:
        for k, v in infos.items():
            if '_map' in k:
                if data.wandb is not None:
                    data.stats[f'Media/{k}'] = data.wandb.Image(v[0])
                    continue
                elif data.neptune is not None:
                    # TODO: Add neptune image logging
                    pass

            if isinstance(v, np.ndarray):
                v = v.tolist()
            try:
                iter(v)
            except TypeError:
                data.stats[k].append(v)
            else:
                data.stats[k] += v

    # TODO: Better way to enable multiple collects
    data.experience.ptr = 0
    data.experience.step = 0
    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    with profile.train_copy:
        idxs = experience.sort_training_data()
        dones = experience.dones[idxs]
        rewards = experience.rewards[idxs]

    with profile.train_misc:
        if config.use_p3o:
            reward_block = experience.reward_block
            mask_block = experience.mask_block
            values_mean = experience.values_mean[idxs]
            values_std = experience.values_std[idxs]
            advantages = experience.advantages

            # Note: This function gets messed up by computing across
            # episode bounds. Because we store experience in a flat buffer,
            # bounds can be crossed even after handling dones. This prevent
            # our method from scaling to longer horizons. TODO: Redo the way
            # we store experience to avoid this issue
            vstd_min = values_std.min().item()
            vstd_max = values_std.max().item()
            reward_block = experience.reward_block
            mask_block = experience.mask_block
            torch.cuda.synchronize()

            advantages = compute_advantages(reward_block, mask_block, values_mean, values_std,
                    experience.buf, dones, rewards, advantages, experience.bounds,
                    config.p3o_horizon)
            advantages = advantages.cpu().numpy()
            torch.cuda.synchronize()
                
            experience.flatten_batch(advantages, reward_block, mask_block)
            torch.cuda.synchronize()
        else:
            values_np = experience.values[idxs].to('cpu', non_blocking=True).numpy()
            dones_np = dones.to('cpu', non_blocking=True).numpy()
            rewards_np = rewards.to('cpu', non_blocking=True).numpy()
            torch.cuda.synchronize()
            advantages_np = compute_gae(dones_np, values_np,
            rewards_np, config.gamma, config.gae_lambda)
            experience.flatten_batch(advantages_np)

    # Optimizing the policy and value network
    total_minibatches = experience.num_minibatches * config.update_epochs
    mean_pg_loss, mean_v_loss, mean_entropy_loss = 0, 0, 0
    mean_old_kl, mean_kl, mean_clipfrac = 0, 0, 0
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(experience.num_minibatches):
            with profile.train_misc:
                obs = experience.b_obs[mb]
                obs = obs.to(config.device)
                atn = experience.b_actions[mb]
                log_probs = experience.b_logprobs[mb]
                adv = experience.b_advantages[mb]

                if config.use_p3o:
                    val_mean = experience.b_values_mean[mb]
                    val_std = experience.b_values_std[mb]
                    rew_block = experience.b_reward_block[mb]
                    mask_block = experience.b_mask_block[mb]
                else:
                    val = experience.b_values[mb]
                    ret = experience.b_returns[mb]

                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_forward:
                if data.use_p3o:
                    if experience.lstm_h is not None:
                        (logits, newvalue_mean, newvalue_std), lstm_state = data.policy.forward_train(obs, lstm_state)
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        flat_obs = obs.reshape(-1, *data.vecenv.single_observation_space.shape)
                        logits, newvalue_mean, newvalue_std = data.policy.forward_train(flat_obs, lstm_state)
                else:
                    if experience.lstm_h is not None:
                        (logits, newvalue), lstm_state = data.policy.forward_train(obs, lstm_state)
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        flat_obs = obs.reshape(-1, *data.vecenv.single_observation_space.shape)
                        logits, newvalue = data.policy.forward_train(flat_obs)
                    newvalue = newvalue.flatten()

                actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits,
                    action=atn, is_continuous=data.policy.is_continuous)

                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                logratio = newlogprob - log_probs.reshape(-1)
                ratio = logratio.exp()

                # TODO: Only do this if we are KL clipping? Saves 1-2% compute
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

                adv = adv.reshape(-1)
                if config.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if config.use_p3o:
                    newvalue_mean = newvalue_mean.view(-1, config.p3o_horizon)
                    newvalue_std = newvalue_std.view(-1, config.p3o_horizon)
                    newvalue_var = torch.square(newvalue_std)
                    criterion = torch.nn.GaussianNLLLoss(reduction='none')
                    v_loss = criterion(newvalue_mean, rew_block, newvalue_var)
                    #TODO: Count mask and sum
                    # There is going to have to be some sort of norm here.
                    # Right now, learning works at different horizons, but you need
                    # to retune hyperparameters. Ideally, horizon should be a stable
                    # param that zero-shots the same hypers

                    # Faster than masking
                    v_loss = (v_loss*mask_block).sum() / mask_block.sum()
                elif config.clip_vloss:
                    v_loss_unclipped = (newvalue - ret) ** 2
                    v_clipped = val + torch.clamp(
                        newvalue - val,
                        -config.vf_clip_coef,
                        config.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

            with profile.learn:
                data.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)
                data.optimizer.step()
                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                losses.policy_loss += pg_loss.item() / total_minibatches
                losses.value_loss += v_loss.item() / total_minibatches
                losses.entropy += entropy_loss.item() / total_minibatches
                losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                losses.approx_kl += approx_kl.item() / total_minibatches
                losses.clipfrac += clipfrac.item() / total_minibatches

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    with profile.train_misc:
        if config.anneal_lr:
            frac = 1.0 - data.global_step / config.total_timesteps
            lrnow = frac * config.learning_rate
            data.optimizer.param_groups[0]["lr"] = lrnow

        if config.use_p3o:
            y_pred = experience.values_mean
            y_true = experience.reward_block
        else:
            y_pred = experience.values
            y_true = experience.returns

        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        #losses.explained_variance = explained_var.item()
        data.epoch += 1

        done_training = data.global_step >= config.total_timesteps
        # TODO: beter way to get episode return update without clogging dashboard
        # TODO: make this appear faster
        logs = None
        if done_training or profile.update(data):
            logs = mean_and_log(data)
            print_dashboard(config.env, data.utilization, data.global_step, data.epoch,
                profile, data.losses, data.stats, data.msg)
            data.stats = defaultdict(list)

        #print('MEAN', experience.b_values_mean.mean(0).mean(0))
        #print('STD', torch.exp(experience.b_values_logstd).mean(0).mean(0))

        if data.epoch % config.checkpoint_interval == 0 or done_training:
            save_checkpoint(data)
            data.msg = f'Checkpoint saved at update {data.epoch}'

        torch.cuda.synchronize()

    return logs

def compute_pg_loss(log_probs, newlogprob, adv, clip_coef):
    logratio = newlogprob - log_probs.reshape(-1)
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    adv = adv.view(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Policy loss
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(
        ratio, 1 - clip_coef, 1 + clip_coef
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    return pg_loss, approx_kl, old_approx_kl, clipfrac

def dist_sum(value, device):
    if not dist.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not dist.is_initialized():
        return value

    return dist_sum(value, device) / dist.get_world_size()

def mean_and_log(data):
    for k in list(data.stats.keys()):
        v = data.stats[k]
        try:
            v = np.mean(v)
        except:
            del data.stats[k]

        data.stats[k] = v

    device = data.config.device

    sps = dist_sum(data.profile.SPS, device)
    agent_steps = int(dist_sum(data.global_step, device))
    epoch = int(dist_sum(data.epoch, device))
    learning_rate = data.optimizer.param_groups[0]["lr"]
    environment = {k: dist_mean(v, device) for k, v in data.stats.items()}
    losses = {k: dist_mean(v, device) for k, v in data.losses.items()}
    performance = {k: dist_sum(v, device) for k, v in data.profile}

    logs = {
        'SPS': sps,
        'agent_steps': agent_steps,
        'epoch': epoch,
        'learning_rate': learning_rate,
        **{f'environment/{k}': v for k, v in environment.items()},
        **{f'losses/{k}': v for k, v in losses.items()},
        **{f'performance/{k}': v for k, v in performance.items()},
    }

    if dist.is_initialized() and dist.get_rank() != 0:
        return logs

    if data.wandb is not None:
        data.last_log_time = time.time()
        data.wandb.log(logs)
    elif data.neptune is not None:
        data.last_log_time = time.time()
        for k, v in logs.items():
            data.neptune[k].append(v, step=agent_steps)

    return logs

def close(data):
    data.vecenv.close()
    data.utilization.stop()
    config = data.config
    if data.wandb is not None:
        artifact_name = f"{config.exp_id}_model"
        artifact = data.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(data)
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)
        data.wandb.finish()
    elif data.neptune is not None:
        data.neptune.stop()

class Profile:
    SPS: ... = 0
    uptime: ... = 0
    remaining: ... = 0
    eval_time: ... = 0
    env_time: ... = 0
    eval_forward_time: ... = 0
    eval_copy_time: ... = 0
    eval_misc_time: ... = 0
    train_time: ... = 0
    train_forward_time: ... = 0
    learn_time: ... = 0
    train_copy_time: ... = 0
    train_misc_time: ... = 0
    custom_time: ... = 0
    def __init__(self):
        self.start = time.time()
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_copy = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_copy = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
        self.custom = pufferlib.utils.Profiler()
        self.prev_steps = 0

    def __iter__(self):
        yield 'SPS', self.SPS
        yield 'uptime', self.uptime
        yield 'remaining', self.remaining
        yield 'eval_time', self.eval_time
        yield 'env_time', self.env_time
        yield 'eval_forward_time', self.eval_forward_time
        yield 'eval_copy_time', self.eval_copy_time
        yield 'eval_misc_time', self.eval_misc_time
        yield 'train_time', self.train_time
        yield 'train_forward_time', self.train_forward_time
        yield 'learn_time', self.learn_time
        yield 'train_copy_time', self.train_copy_time
        yield 'train_misc_time', self.train_misc_time
        yield 'custom_time', self.custom_time

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self, data, interval_s=1):
        global_step = data.global_step
        if global_step == 0:
            return True

        uptime = time.time() - self.start
        if uptime - self.uptime < interval_s:
            return False

        self.SPS = (global_step - self.prev_steps) / (uptime - self.uptime)
        self.prev_steps = global_step
        self.uptime = uptime

        self.remaining = (data.config.total_timesteps - global_step) / self.SPS
        self.eval_time = data._timers['evaluate'].elapsed
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_copy_time = self.eval_copy.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = data._timers['train'].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_copy_time = self.train_copy.elapsed
        self.train_misc_time = self.train_misc.elapsed
        self.custom_time = self.custom.elapsed
        return True

def make_losses():
    return pufferlib.namespace(
        policy_loss=0,
        value_loss=0,
        entropy=0,
        old_approx_kl=0,
        approx_kl=0,
        clipfrac=0,
        explained_variance=0,
    )

class Experience:
    '''Flat tensor storage and array views for faster indexing'''
    def __init__(self, batch_size, bptt_horizon, minibatch_size, hidden_size,
                 obs_shape, obs_dtype, atn_shape, atn_dtype, cpu_offload=False,
                 device='cuda', lstm=None, lstm_total_agents=0,
                 use_e3b=False, e3b_coef=0.1, use_p3o=False, p3o_horizon=32):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == 'cuda' and cpu_offload
        obs_device = device if not pin else 'cpu'
        self.obs=torch.zeros(batch_size, *obs_shape, dtype=obs_dtype,
            pin_memory=pin, device=device if not pin else 'cpu')
        self.actions=torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, device=device)
        self.logprobs=torch.zeros(batch_size, device=device)
        self.rewards=torch.zeros(batch_size, device=device)
        self.dones=torch.zeros(batch_size, device=device)
        self.truncateds=torch.zeros(batch_size, device=device)

        self.use_e3b = use_e3b
        if use_e3b:
            self.e3b_inv = 10*torch.eye(hidden_size).repeat(lstm_total_agents, 1, 1).to(device)
            self.e3b_orig = self.e3b_inv.clone()
            self.e3b_mean = None
            self.e3b_std = None

        self.use_p3o = use_p3o
        self.p3o_horizon = p3o_horizon
        if use_p3o:
            self.values_mean=torch.zeros(batch_size, p3o_horizon, device=device)
            self.values_std=torch.zeros(batch_size, p3o_horizon, device=device)
            self.reward_block = torch.zeros(batch_size, p3o_horizon, dtype=torch.float32, device=device)
            self.mask_block = torch.ones(batch_size, p3o_horizon, dtype=torch.float32, device=device)
            self.buf = torch.zeros(batch_size, p3o_horizon, dtype=torch.float32, device=device)
            self.advantages = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self.bounds = torch.zeros(batch_size, dtype=torch.int32, device=device)
        else:
            self.values = torch.zeros(batch_size, device=device)

        self.sort_keys = np.zeros((batch_size, 3), dtype=np.int32)
        self.sort_keys[:, 0] = np.arange(batch_size)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError('batch_size must be divisible by minibatch_size')

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError('minibatch_size must be divisible by bptt_horizon')

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.p3o_horizon = p3o_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, value_mean, value_std, action, logprob, reward, done, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = np.where(mask)[0]
        num_indices = indices.size
        end = ptr + num_indices
        dst = slice(ptr, end)

        # Zero-copy indexing for contiguous env_id
        if num_indices == mask.size and isinstance(env_id, slice):
            gpu_inds = cpu_inds = slice(0, min(self.batch_size - ptr, num_indices))
        else:
            cpu_inds = indices[:self.batch_size - ptr]
            gpu_inds = torch.as_tensor(indices).to(self.obs.device, non_blocking=True)

 
        obs = obs.to(self.obs.device, non_blocking=True)

        if self.obs.device.type == 'cuda':
            self.obs[dst] = obs[gpu_inds]
        else:
            self.obs[dst] = obs[cpu_inds]

        if self.use_p3o:
            self.values_mean[dst] = value_mean[gpu_inds]
            self.values_std[dst] = value_std[gpu_inds]
        else:
            self.values[dst] = value_mean[gpu_inds]

        self.actions[dst] = action[gpu_inds]
        self.logprobs[dst] = logprob[gpu_inds]
        self.rewards[dst] = reward[gpu_inds]
        self.dones[dst] = done[gpu_inds]

        if isinstance(env_id, slice):
            self.sort_keys[dst, 1] = np.arange(cpu_inds.start, cpu_inds.stop, dtype=np.int32)
        else:
            self.sort_keys[dst, 1] = env_id[cpu_inds]

        self.sort_keys[dst, 2] = self.step
        self.ptr = end
        self.step += 1

        return action.cpu().numpy()

    def sort_training_data(self):
        idxs = np.lexsort((self.sort_keys[:, 2], self.sort_keys[:, 1]))
        self.b_idxs_obs = torch.as_tensor(idxs.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(1,0,-1)).to(self.obs.device).long()
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(
            self.num_minibatches, self.minibatch_size)
        self.sort_keys[:, 1:] = 0
        return idxs

    def flatten_batch(self, advantages_np, reward_block=None, mask_block=None):
        advantages = torch.as_tensor(advantages_np).to(self.device, non_blocking=True)
        self.b_advantages = advantages.reshape(
            self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size)

        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)[b_idxs].contiguous()
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)[b_idxs]
        self.b_dones = self.dones.to(self.device, non_blocking=True)[b_idxs]
        self.b_obs = self.obs[self.b_idxs_obs]

        if self.use_p3o:
            self.reward_block = torch.as_tensor(reward_block).to(self.device)
            self.b_reward_block = self.reward_block.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon, self.p3o_horizon
                ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size, self.p3o_horizon)

            b_mask_block = torch.as_tensor(mask_block).to(self.device)
            self.b_mask_block = b_mask_block.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon, self.p3o_horizon
                ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size, self.p3o_horizon)

            self.b_values_mean = self.values_mean.to(self.device, non_blocking=True)[b_flat]
            self.b_values_std = self.values_std.to(self.device, non_blocking=True)[b_flat]
        else:
            self.b_values = self.values.to(self.device, non_blocking=True)[b_flat]
            self.returns = advantages + self.values # Check sorting of values here
            self.b_returns = self.b_advantages + self.b_values # Check sorting of values here

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque(maxlen=maxlen)
        self.cpu_util = deque(maxlen=maxlen)
        self.gpu_util = deque(maxlen=maxlen)
        self.gpu_mem = deque(maxlen=maxlen)

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*free/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def save_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.epoch:06d}.pt'
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return model_path

    torch.save(data.uncompiled_policy, model_path)

    state = {
        'optimizer_state_dict': data.optimizer.state_dict(),
        'global_step': data.global_step,
        'agent_step': data.global_step,
        'update': data.epoch,
        'model_name': model_name,
        'exp_id': config.exp_id,
    }
    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)
    return model_path

def try_load_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_id)
    if not os.path.exists(path):
        print('No checkpoints found. Assuming new experiment')
        return

    trainer_path = os.path.join(path, 'trainer_state.pt')
    resume_state = torch.load(trainer_path, weights_only=False)
    model_path = os.path.join(path, resume_state['model_name'])
    data.policy.uncompiled.load_state_dict(model_path, map_location=config.device)
    data.optimizer.load_state_dict(resume_state['optimizer_state_dict'])
    print(f'Loaded checkpoint {resume_state["model_name"]}')

def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)

def rollout(env_creator, env_kwargs, policy_cls, rnn_cls, agent_creator, agent_kwargs,
        backend, render_mode='auto', model_path=None, device='cuda'):

    if render_mode != 'auto':
        env_kwargs['render_mode'] = render_mode

    # We are just using Serial vecenv to give a consistent
    # single-agent/multi-agent API for evaluation
    env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs, backend=backend)

    if model_path is None:
        agent = agent_creator(env, policy_cls, rnn_cls, agent_kwargs).to(device)
    else:
        agent = torch.load(model_path, map_location=device, weights_only=False)

    #e3b_inv = 10*torch.eye(agent.hidden_size).repeat(env_kwargs['num_envs'], 1, 1).to(device)
    e3b_inv = None

    ob, info = env.reset()
    driver = env.driver_env
    os.system('clear')

    state = (None, None)
    num_agents = env.observation_space.shape[0]
    if hasattr(agent, 'recurrent'):
        shape = (num_agents, agent.hidden_size)
        state = (torch.zeros(shape).to(device), torch.zeros(shape).to(device))

    frames = []
    tick = 0
    value = [0]
    intrinsic = [0]
    intrinsic_mean = None
    intrinsic_std = None
    while tick <= 200000:
        if tick % 1 == 0:
            #render = driver.render(overlay=float(intrinsic[0]))
            render = driver.render()
            if driver.render_mode == 'ansi':
                print('\033[0;0H' + render + '\n')
                time.sleep(0.05)
            elif driver.render_mode == 'rgb_array':
                frames.append(render)
                import cv2
                render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', render)
                cv2.waitKey(1)
                time.sleep(1/24)
            elif driver.render_mode in ('human', 'raylib') and render is not None:
                frames.append(render)

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            if hasattr(agent, 'recurrent'):
                (logits, value), hidden, (h, c) = agent(ob, state)
            else:
                action, _, value, _, e3b, intrinsic = agent(ob)

            action, logprob, _ = pufferlib.pytorch.sample_logits(logits, is_continuous=agent.is_continuous)
            action = action.cpu().numpy().reshape(env.action_space.shape)

        ob, reward = env.step(action)[:2]
        reward = reward.mean()
        if tick % 128 == 0:
            print(f'Reward: {reward:.4f}, Tick: {tick}')
        tick += 1

    # Save frames as gif
    if frames:
        import imageio
        os.makedirs('../docker', exist_ok=True) or imageio.mimsave('../docker/eval.gif', frames, fps=15, loop=0)

def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

ROUND_OPEN = rich.box.Box(
    "╭──╮\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = '[bright_cyan]'
c2 = '[white]'
c3 = '[cyan]'
b1 = '[bright_cyan]'
b2 = '[bright_white]'

def abbreviate(num):
    if num < 1e3:
        return f'{b2}{num:.0f}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}k'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}m'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}b'
    else:
        return f'{b2}{num/1e12:.1f}{c2}t'

def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100*time/uptime - 1e-5)
    return f'{c1}{name}', duration(time), f'{b2}{percent:2d}%'

# TODO: Add env name to print_dashboard
def print_dashboard(env_name, utilization, global_step, epoch,
        profile, losses, stats, msg, clear=False, max_stats=[0]):
    console = Console()
    if clear:
        console.clear()

    dashboard = Table(box=ROUND_OPEN, expand=True,
        show_header=False, border_style='bright_cyan')

    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)
    cpu_percent = np.mean(utilization.cpu_util)
    dram_percent = np.mean(utilization.cpu_mem)
    gpu_percent = np.mean(utilization.gpu_util)
    vram_percent = np.mean(utilization.gpu_mem)
    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=13)
    table.add_column(justify="right", width=13)
    table.add_row(
        f':blowfish: {c1}PufferLib {b2}2.0.0',
        f'{c1}CPU: {c3}{cpu_percent:.1f}%',
        f'{c1}GPU: {c3}{gpu_percent:.1f}%',
        f'{c1}DRAM: {c3}{dram_percent:.1f}%',
        f'{c1}VRAM: {c3}{vram_percent:.1f}%',
    )
        
    s = Table(box=None, expand=True)
    s.add_column(f"{c1}Summary", justify='left', vertical='top', width=16)
    s.add_column(f"{c1}Value", justify='right', vertical='top', width=8)
    s.add_row(f'{c2}Environment', f'{b2}{env_name}')
    s.add_row(f'{c2}Agent Steps', abbreviate(global_step))
    s.add_row(f'{c2}SPS', abbreviate(profile.SPS))
    s.add_row(f'{c2}Epoch', abbreviate(epoch))
    s.add_row(f'{c2}Uptime', duration(profile.uptime))
    s.add_row(f'{c2}Remaining', duration(profile.remaining))

    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf('Evaluate', profile.eval_time, profile.uptime))
    p.add_row(*fmt_perf('  Forward', profile.eval_forward_time, profile.uptime))
    p.add_row(*fmt_perf('  Env', profile.env_time, profile.uptime))
    p.add_row(*fmt_perf('  Copy', profile.eval_copy_time, profile.uptime))
    p.add_row(*fmt_perf('  Misc', profile.eval_misc_time, profile.uptime))
    p.add_row(*fmt_perf('Train', profile.train_time, profile.uptime))
    p.add_row(*fmt_perf('  Forward', profile.train_forward_time, profile.uptime))
    p.add_row(*fmt_perf('  Learn', profile.learn_time, profile.uptime))
    p.add_row(*fmt_perf('  Copy', profile.train_copy_time, profile.uptime))
    p.add_row(*fmt_perf('  Misc', profile.train_misc_time, profile.uptime))
    p.add_row(*fmt_perf('  Custom', profile.custom_time, profile.uptime))

    l = Table(box=None, expand=True, )
    l.add_column(f'{c1}Losses', justify="left", width=16)
    l.add_column(f'{c1}Value', justify="right", width=8)
    for metric, value in losses.items():
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
    for metric, value in stats.items():
        try: # Discard non-numeric values
            int(value)
        except:
            continue

        u = left if i % 2 == 0 else right
        u.add_row(f'{c2}{metric}', f'{b2}{value:.3f}')
        i += 1
        if i == 30:
            break

    for i in range(max_stats[0] - i):
        u = left if i % 2 == 0 else right
        u.add_row('', '')

    max_stats[0] = max(max_stats[0], i)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    table.add_row(f' {c1}Message: {c2}{msg}')

    with console.capture() as capture:
        console.print(dashboard)

    print('\033[0;0H' + capture.get())
