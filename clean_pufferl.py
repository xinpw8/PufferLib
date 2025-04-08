from pdb import set_trace as T
import numpy as np

import os
import random
import psutil
import time

from threading import Thread
from collections import defaultdict, deque
from contextlib import nullcontext

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
#from c_advantage import compute_gae

import torch
from torch.utils.cpp_extension import load


# Compile the CUDA kernel
cuda_module = load(
    name='compute_gae',
    sources=['pufferlib.cu'],
    verbose=True
)

def compute_gae(
        values: torch.Tensor,     # [num_steps, horizon]
        rewards: torch.Tensor,    # [num_steps, horizon]
        dones: torch.Tensor,      # [num_steps, horizon]
        gamma: float,
        gae_lambda: float,
        ):

    num_steps = values.shape[0]
    horizon = values.shape[1]
    advantages = torch.zeros(num_steps, horizon, dtype=torch.float32, device=values.device)

    for t in [values, rewards, dones, advantages]:
        assert t.ndim == 2
        assert t.shape[0] == num_steps
        assert t.shape[1] == horizon
        t.contiguous()
        assert t.is_cuda, "All tensors must be on GPU"
    
   
    cuda_module.compute_gae(
        values,
        rewards,
        dones,
        advantages,
        gamma,
        gae_lambda,
        num_steps,
        horizon,
    )
   
    torch.cuda.synchronize()
    return advantages


def compute_advantages(
    reward_block: torch.Tensor, # [num_steps, horizon]
    reward_mask: torch.Tensor,  # [num_steps, horizon]
    values_mean: torch.Tensor,  # [num_steps, horizon]
    values_std: torch.Tensor,   # [num_steps, horizon]
    buf: torch.Tensor,          # [num_steps, horizon]
    dones: torch.Tensor,        # [num_steps]
    rewards: torch.Tensor,      # [num_steps]
    advantages: torch.Tensor,   # [num_steps]
    bounds: torch.Tensor,       # [num_steps]
    vstd_max: float,
    puf: float,
    horizon: int
):
    assert all(t.is_cuda for t in [reward_block, reward_mask, values_mean, values_std, 
                                  buf, dones, rewards, advantages, bounds]), "All tensors must be on GPU"
    
    # Ensure contiguous memory
    tensors = [reward_block, reward_mask, values_mean, values_std, buf, dones, rewards, advantages, bounds]
    for t in tensors:
        t.contiguous()
        assert t.is_cuda

    num_steps = rewards.shape[0]
    
    # Precompute vstd_min and vstd_max
    #vstd_max = values_std.max().item()
    #vstd_min = values_std.min().item()

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
        vstd_max,
        puf,
        horizon,
    )
    
    torch.cuda.synchronize()
    return advantages

def create(config, vecenv, policy, optimizer=None, wandb=None, neptune=None):
    seed_everything(config.seed, config.torch_deterministic)
    losses = make_losses()

    utilization = Utilization()
    msg = f'Model Size: {abbreviate(count_params(policy))} parameters'

    vecenv.async_reset(config.seed)
    obs_shape = vecenv.single_observation_space.shape
    obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[vecenv.single_observation_space.dtype]
    atn_shape = vecenv.single_action_space.shape
    atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[vecenv.single_action_space.dtype]
    total_agents = vecenv.num_agents

    on_policy_rows = config.batch_size // config.bptt_horizon
    off_policy_rows = config.replay_factor*config.batch_size // config.bptt_horizon
    experience_rows = on_policy_rows + off_policy_rows

    pin = config.device == 'cuda' and config.cpu_offload
    obs_device = config.device if not pin else 'cpu'
    experience = pufferlib.namespace(
        obs=torch.zeros(experience_rows, config.bptt_horizon, *obs_shape,
            dtype=obs_dtype, pin_memory=pin, device='cpu' if pin else config.device),
        actions=torch.zeros(experience_rows, config.bptt_horizon, *atn_shape,
            dtype=atn_dtype, device=config.device),
        logprobs=torch.zeros(experience_rows, config.bptt_horizon, device=config.device),
        rewards=torch.zeros(experience_rows, config.bptt_horizon, device=config.device),
        dones=torch.zeros(experience_rows, config.bptt_horizon, device=config.device),
        truncateds=torch.zeros(experience_rows, config.bptt_horizon, device=config.device),
    )
    stored_indices = torch.zeros(experience_rows, device=config.device, dtype=torch.int32)
    ep_lengths = torch.zeros(total_agents, device=config.device, dtype=torch.int32)
    ep_indices = torch.arange(total_agents, device=config.device, dtype=torch.int32)
    free_idx = total_agents

    assert free_idx <= experience_rows
    if config.use_e3b:
        experience.e3b_inv = torch.eye(policy.hidden_size).repeat(total_agents, 1, 1).to(config.device) / config.e3b_lambda
        experience.e3b_orig = experience.e3b_inv.clone()
        experience.e3b_mean = None
        experience.e3b_std = None

    if config.use_diayn:
        # TODO: Check shapes
        experience.diayn_archive = torch.nn.functional.one_hot(torch.arange(config.diayn_archive), config.diayn_archive).to(config.device).float()
        experience.diayn_skills = torch.randint(0, config.diayn_archive, (total_agents,), dtype=torch.long, device=config.device)
        experience.diayn_batch = torch.zeros(experience_rows, dtype=torch.long, device=config.device)

    if config.use_p3o:
        batch_size = config.batch_size
        p3o_horizon = config.p3o_horizon
        device = config.device
        experience.values_mean=torch.zeros(batch_size, p3o_horizon, device=device)
        experience.values_std=torch.zeros(batch_size, p3o_horizon, device=device)
        experience.reward_block = torch.zeros(batch_size, p3o_horizon, dtype=torch.float32, device=device)
        experience.mask_block = torch.ones(batch_size, p3o_horizon, dtype=torch.float32, device=device)
        experience.buf = torch.zeros(batch_size, p3o_horizon, dtype=torch.float32, device=device)
        experience.advantages = torch.zeros(batch_size, dtype=torch.float32, device=device)
        experience.bounds = torch.zeros(batch_size, dtype=torch.int32, device=device)
        experience.vstd_max = 1.0
    else:
        experience.values = torch.zeros(experience_rows, config.bptt_horizon, device=config.device)

    lstm_h = None
    lstm_c = None
    if isinstance(policy, torch.nn.LSTM):
        assert total_agents > 0
        shape = (policy.num_layers, total_agents, policy.hidden_size)
        lstm_h = torch.zeros(shape).to(config.device)
        lstm_c = torch.zeros(shape).to(config.device)

    minibatch_size = min(config.minibatch_size, config.max_minibatch_size)
    num_minibatches = config.batch_size / minibatch_size
    if num_minibatches != int(num_minibatches):
        raise ValueError('batch_size must be divisible by minibatch_size')
    else:
        num_minibatches = int(num_minibatches)

    minibatch_rows = minibatch_size / config.bptt_horizon
    if minibatch_rows != int(minibatch_rows):
        raise ValueError('minibatch_size must be divisible by bptt_horizon')
    else:
        minibatch_rows = int(minibatch_rows)

    uncompiled_policy = policy
    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode, fullgraph=config.compile_fullgraph)

    assert config.optimizer in ('adam', 'muon', 'kron')
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps
        )
    elif config.optimizer == 'muon':
        from heavyball import ForeachMuon
        import heavyball.utils
        #heavyball.utils.compile_mode = "reduce-overhead"
        optimizer = ForeachMuon(
            policy.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps
        )
    elif config.optimizer == 'kron':
        from heavyball import ForeachPSGDKron
        import heavyball.utils
        #heavyball.utils.compile_mode = "reduce-overhead"
        optimizer = ForeachPSGDKron(
            policy.parameters(),
            lr=config.learning_rate,
            precond_lr=config.precond_lr,
            beta=config.adam_beta1,
        )

    epochs = config.total_timesteps // config.batch_size
    assert config.scheduler in ('linear', 'cosine')
    if config.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = None if config.precision == 'float32' else torch.amp.GradScaler()

    amp_context = nullcontext()
    if config.precision != 'float32':
        amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, config.precision))

    profile = Profile(amp_context)
    print_dashboard(config.env, utilization, 0, 0, profile, losses, {}, msg, clear=True)

    return pufferlib.namespace(
        config=config,
        vecenv=vecenv,
        policy=policy,
        uncompiled_policy=uncompiled_policy,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
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
        puf=config.puf,
        use_diayn=config.use_diayn,
        diayn_archive=config.diayn_archive,
        diayn_coef=config.diayn_coef,
        # Do we use these?
        ptr=0,
        step=0,
        lstm_h=lstm_h,
        lstm_c=lstm_c,
        minibatch_rows=minibatch_rows,
        num_minibatches=num_minibatches,
        stored_indices=stored_indices,
        ep_lengths=ep_lengths,
        ep_indices=ep_indices,
        free_idx=free_idx,
        on_policy_rows=on_policy_rows,
        off_policy_rows=off_policy_rows,
        experience_rows=experience_rows,
        device=config.device,
    )

@pufferlib.utils.profile
def evaluate(data):
    profile = data.profile
    with profile.eval_misc:
        config = data.config
        experience = data.experience
        policy = data.policy
        infos = defaultdict(list)
        lstm_h = data.lstm_h
        lstm_c = data.lstm_c

    while not full(data):
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

            if data.use_diayn:
                idxs = env_id[done_mask]
                if len(idxs) > 0:
                    z_idxs = torch.randint(0, experience.diayn_archive.shape[0], (done_mask.sum(),)).to(config.device)
                    experience.diayn_skills[idxs] = z_idxs

        with profile.eval_copy:
            if data.use_e3b and done_mask.any():
                done_idxs = env_id[done_mask]
                experience.e3b_inv[done_idxs] = experience.e3b_orig[done_idxs]


            o = torch.as_tensor(o)
            o_device = o.to(config.device, non_blocking=True)
            r = torch.as_tensor(r).to(config.device, non_blocking=True)
            d = torch.as_tensor(d).to(config.device, non_blocking=True)

            h = None
            c = None
            if lstm_h is not None:
                h = lstm_h[0, gpu_env_id]
                c = lstm_c[0, gpu_env_id]

        with profile.eval_forward, torch.no_grad():
            state = pufferlib.namespace(
                reward=r,
                done=d,
                env_id=gpu_env_id,
                mask=mask,
                lstm_h=h,
                lstm_c=c,
            )

            if data.use_diayn:
                z_idxs = experience.diayn_skills[env_id]
                z = experience.diayn_archive[z_idxs]
                state.diayn_z_idxs = z_idxs
                state.diayn_z = z

            logits, value = policy(o_device, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits, is_continuous=policy.is_continuous)

            if data.use_diayn:
                diayn_policy = policy if lstm_h is None else policy.policy
                q = diayn_policy.diayn_discriminator(state.hidden).squeeze()
                r_diayn = torch.log_softmax(q, dim=-1).gather(-1, z_idxs.unsqueeze(-1)).squeeze()
                r += config.diayn_coef*r_diayn# - np.log(1/data.diayn_archive)
                state.diayn_z = z
                state.diayn_z_idxs = z_idxs

            if data.use_e3b:
                e3b = experience.e3b_inv[env_id]
                phi = state.hidden.detach()        
                u = phi.unsqueeze(1) @ e3b
                b = u @ phi.unsqueeze(2)
                experience.e3b_inv[env_id] -= (u.mT @ u) / (1 + b)
                done_inds = env_id[done_mask]
                experience.e3b_inv[done_inds] = experience.e3b_orig[done_inds]
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
                r += e3b_reward

            # Clip rewards
            r = torch.clamp(r, -1, 1)

        with profile.eval_copy, torch.no_grad():
            if lstm_h is not None:
                lstm_h[:, gpu_env_id] = state.lstm_h
                lstm_c[:, gpu_env_id] = state.lstm_c

            o = o if config.cpu_offload else o_device
            actions = store(data, state, o, o_device, value, action, logprob, r, d, gpu_env_id, mask)

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
    data.ptr = 0
    data.step = 0
    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    with profile.train_copy:
        #idxs = experience.sort_training_data()
        #dones = experience.dones[idxs]
        #rewards = experience.rewards[idxs]
        dones = experience.dones
        rewards = experience.rewards

    # TODO: Beter place for this
    data.free_idx = 0
    data.ep_lengths.zero_()

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

            mask_block.zero_()
            experience.buf.zero_()
            reward_block.zero_()
            r_mean = rewards.mean().item()
            r_std = rewards.std().item()
            advantages.zero_()
            experience.bounds.zero_()

            '''
            if data.epoch == 0:
                values_std[:] = r_std
                with torch.no_grad():
                    data.policy.policy.value_logstd[:] = np.log(r_std)
            '''

            # TODO: Rename vstd to r_std
            advantages = compute_advantages(reward_block, mask_block, values_mean, values_std,
                    experience.buf, dones, rewards, advantages, experience.bounds,
                    r_std, data.puf, config.p3o_horizon)

            horizon = torch.where(values_std[0] > 0.95*r_std)[0]
            horizon = horizon[0].item()+1 if len(horizon) else 1
            if horizon < 16:
                horizon = 16

            advantages = advantages.cpu().numpy()
            torch.cuda.synchronize()

            experience.flatten_batch(advantages, reward_block, mask_block)
            torch.cuda.synchronize()

    # Optimizing the policy and value network
    total_minibatches = data.num_minibatches * config.update_epochs
    mean_pg_loss, mean_v_loss, mean_entropy_loss = 0, 0, 0
    mean_old_kl, mean_kl, mean_clipfrac = 0, 0, 0
    cross_entropy = torch.nn.CrossEntropyLoss()
    accumulate_minibatches = max(1, config.minibatch_size // config.max_minibatch_size)
    for epoch in range(config.update_epochs):
        advantages = compute_gae(experience.values, experience.rewards, experience.dones, config.gamma, config.gae_lambda)
        n_samples = config.minibatch_size // config.bptt_horizon
        batch = sample(data, advantages, n_samples)

        with profile.train_misc:
            state = pufferlib.namespace(
                action=batch.actions,
                lstm_h=None,
                lstm_c=None,
            )
            if config.use_diayn:
                z_idxs = batch.diayn_z_idxs

            if config.use_p3o:
                val_mean = batch.values_mean
                val_std = batch.values_std
                rew_block = batch.reward_block
                mask_block = batch.mask_block
            else:
                val = batch.values.flatten()

        with profile.train_forward:
            if not isinstance(data.policy, torch.nn.LSTM):
                batch.obs = batch.obs.reshape(-1, *data.vecenv.single_observation_space.shape)

            logits, newvalue = data.policy.forward_train(batch.obs, state)
            lstm_h = state.lstm_h
            lstm_c = state.lstm_c
            if lstm_h is not None:
                lstm_h = lstm_h.detach()
            if lstm_c is not None:
                lstm_c = lstm_c.detach()

            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits,
                action=batch.actions, is_continuous=data.policy.is_continuous)

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.train_misc:
            logratio = newlogprob - batch.logprobs.reshape(-1)
            ratio = logratio.exp()

            # TODO: Only do this if we are KL clipping? Saves 1-2% compute
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

            adv = batch.advantages.reshape(-1)
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
                newvalue_mean = newvalue.mean.view(-1, config.p3o_horizon)
                newvalue_std = newvalue.std.view(-1, config.p3o_horizon)
                newvalue_var = torch.square(newvalue_std)
                criterion = torch.nn.GaussianNLLLoss(reduction='none')
                #v_loss = criterion(newvalue_mean[:, :32], rew_block[:, :32], newvalue_var[:, :32])
                v_loss = criterion(newvalue_mean, rew_block, newvalue_var)
                v_loss = v_loss[:, :(horizon+3)]
                mask_block = mask_block[:, :(horizon+3)]
                #v_loss[:, horizon:] = 0
                #v_loss = (v_loss * mask_block).sum(axis=1)
                #v_loss = (v_loss - v_loss.mean().item()) / (v_loss.std().item() + 1e-8)
                #v_loss = v_loss.mean()
                v_loss = v_loss[mask_block.bool()].mean()
                #TODO: Count mask and sum
                # There is going to have to be some sort of norm here.
                # Right now, learning works at different horizons, but you need
                # to retune hyperparameters. Ideally, horizon should be a stable
                # param that zero-shots the same hypers

                # Faster than masking
                #v_loss = (v_loss*mask_block[:, :32]).sum() / mask_block[:, :32].sum()
                #v_loss = (v_loss*mask_block).sum() / mask_block.sum()
                #v_loss = v_loss[mask_block.bool()].mean()
            elif config.clip_vloss:
                newvalue = newvalue.flatten()
                ret = batch.returns.flatten()
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
                newvalue = newvalue.flatten()
                v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - config.ent_coef*entropy_loss + v_loss*config.vf_coef

            with profile.custom:
                if config.use_diayn:
                    diayn_discriminator = data.policy.diayn_discriminator if hasattr(data.policy, 'diayn_discriminator') else data.policy.policy.diayn_discriminator
                    q = diayn_discriminator(state.hidden).squeeze()
                    diayn_loss = cross_entropy(q, z_idxs)
                    loss += config.diayn_loss_coef*diayn_loss

        with profile.learn:
            if data.scaler is None:
                loss.backward()
            else:
                data.scaler.scale(loss).backward()

            if data.scaler is not None:
                data.scaler.unscale_(data.optimizer)

            with torch.no_grad():
                grads = torch.cat([p.grad.flatten() for p in data.policy.parameters()])
                grad_var = grads.var(0).mean() * config.minibatch_size
                data.msg = f'Gradient variance: {grad_var.item():.3f}'

            if (epoch + 1) % accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)

                if data.scaler is None:
                    data.optimizer.step()
                else:
                    data.scaler.step(data.optimizer)
                    data.scaler.update()

                data.optimizer.zero_grad()

        # Reprioritize experience
        advantages = compute_gae(experience.values, experience.rewards, experience.dones, config.gamma, config.gae_lambda)
 
        n_samples = data.off_policy_rows
        exp = sample(data, advantages, n_samples)
        for k, v in experience.items():
            v[data.on_policy_rows:] = exp[k]

        with profile.train_misc:
            losses.policy_loss += pg_loss.item() / total_minibatches
            losses.value_loss += v_loss.item() / total_minibatches
            losses.entropy += entropy_loss.item() / total_minibatches
            losses.old_approx_kl += old_approx_kl.item() / total_minibatches
            losses.approx_kl += approx_kl.item() / total_minibatches
            losses.clipfrac += clipfrac.item() / total_minibatches
            losses.grad_var += grad_var.item() / total_minibatches

            if data.use_diayn:
                losses.diayn_loss += diayn_loss.item() / total_minibatches

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    with profile.train_misc:
        if config.anneal_lr:
            data.scheduler.step()

        if config.use_p3o:
            y_pred = experience.values_mean
            y_true = experience.reward_block
        else:
            y_pred = experience.values.flatten()

            # Probably not updated
            y_true = advantages.flatten() + experience.values.flatten()

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

    return logs

def full(data):
    return data.free_idx >= data.on_policy_rows

def store(data, state, cpu_obs, gpu_obs, value, action, logprob, reward, done, env_id, mask):
    # Mask learner and Ensure indices do not exceed batch size
    exp = data.experience
    ptr = data.ptr
    indices = np.where(mask)[0]
    num_indices = indices.size
    end = ptr + num_indices
    dst = slice(ptr, end)

    # Zero-copy indexing for contiguous env_id
    '''
    if num_indices == mask.size and isinstance(env_id, slice):
        gpu_inds = cpu_inds = slice(0, min(self.batch_size - ptr, num_indices))
    else:
        cpu_inds = indices[:self.batch_size - ptr]
        gpu_inds = torch.as_tensor(cpu_inds).to(self.obs.device, non_blocking=True)
    '''

    batch_rows = data.ep_indices[env_id]
    l = data.ep_lengths[env_id]

    if exp.obs.device.type == 'cuda':
        exp.obs[batch_rows, l] = gpu_obs
    else:
        exp.obs[batch_rows, l] = cpu_obs

    if isinstance(env_id, slice):
        data.stored_indices[batch_rows] = torch.arange(env_id.start, env_id.stop, device=data.device).int()
    else:
        data.stored_indices[batch_rows] = env_id


    if data.use_diayn:
        data.diayn_batch[dst] = state.diayn_z_idxs[gpu_inds]

    if data.use_p3o:
        exp.values_mean[dst] = value.mean[gpu_inds]
        exp.values_std[dst] = value.std[gpu_inds]
    else:
        exp.values[batch_rows, l] = value.flatten()

    exp.actions[batch_rows, l] = action
    exp.logprobs[batch_rows, l] = logprob
    exp.rewards[batch_rows, l] = reward.to(exp.rewards.device) # ???
    exp.dones[batch_rows, l] = done.float().to(exp.dones.device) # ???

    l += 1
    data.ep_lengths[env_id] = l
    full = l >= data.config.bptt_horizon
    num_full = full.sum()
    if num_full > 0:
        if isinstance(env_id, slice):
            env_id = torch.arange(env_id.start, env_id.stop, device=data.device).int()

        full_ids = env_id[full]
        data.ep_indices[full_ids] = data.free_idx + torch.arange(num_full, device=data.device).int()
        data.ep_lengths[full_ids] = 0
        data.free_idx += num_full

    data.step += 1
    return action.cpu().numpy()

def sample(data, advantages, n, reward_block=None, mask_block=None):
    exp = data.experience
    idx = torch.multinomial(advantages.abs().sum(axis=1), n)
    output = {k: v[idx] for k, v in exp.items()}

    if data.use_p3o:
        output['reward_block'] = reward_block[idx]
        output['mask_block'] = mask_block[idx]
        output['values_mean'] = exp.values_mean[idx]
        output['values_std'] = exp.values_std[idx]
    else:
        output['values'] = exp.values[idx]
        output['advantages'] = advantages[idx]
        output['returns'] = advantages[idx] + exp.values[idx]

    if data.use_diayn:
        output['diayn_z_idxs'] = exp.diayn_batch[idx]
        output['diayn_z'] = exp.diayn_skills[idx]

    return pufferlib.namespace(**output)



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
    def __init__(self, amp_context):
        self.start = time.time()
        self.env = pufferlib.utils.Profiler()
        # TODO: Figure out which of these need amp
        self.eval_forward = pufferlib.utils.Profiler(amp_context=amp_context)
        self.eval_copy = pufferlib.utils.Profiler(amp_context=amp_context)
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler(amp_context=amp_context)
        self.learn = pufferlib.utils.Profiler()
        self.train_copy = pufferlib.utils.Profiler(amp_context=amp_context)
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
        diayn_loss=0,
        grad_var=0,
    )

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

    state = pufferlib.namespace(
        lstm_h=None,
        lstm_c=None,
    )

    num_agents = env.observation_space.shape[0]
    if hasattr(agent, 'recurrent'):
        shape = (num_agents, agent.hidden_size)
        state.lstm_h = torch.zeros(shape).to(device)
        state.lstm_c = torch.zeros(shape).to(device)

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
            logits, value = agent(ob, state)
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
