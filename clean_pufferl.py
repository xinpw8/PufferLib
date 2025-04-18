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
from torch.utils.cpp_extension import load

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

def create(config, vecenv, policy, optimizer=None, wandb=None, neptune=None):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    torch.set_float32_matmul_precision('high')
    if config.seed is not None:
        torch.manual_seed(config.seed)

    ext = 'cu' if 'cuda' in config.device else 'cpp'
    puffer_cuda = load(
        name='puffer_cuda',
        sources=[f'pufferlib.{ext}'],
        verbose=True
    )
    compute_gae = puffer_cuda.compute_gae
    compute_vtrace = puffer_cuda.compute_vtrace
    compute_puff_advantage = puffer_cuda.compute_puff_advantage

    losses = pufferlib.namespace(
        policy_loss=0,
        value_loss=0,
        entropy=0,
        old_approx_kl=0,
        approx_kl=0,
        clipfrac=0,
        explained_variance=0,
        diayn_loss=0,
        grad_var=0,
        importance=0,
    )

    utilization = Utilization()
    msg = f'Model Size: {abbreviate(count_params(policy))} parameters'

    vecenv.async_reset(config.seed)
    total_agents = vecenv.num_agents
    obs_shape = vecenv.single_observation_space.shape
    atn_shape = vecenv.single_action_space.shape
    obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[vecenv.single_observation_space.dtype]
    atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[vecenv.single_action_space.dtype]
    on_policy_rows = config.batch_size // config.bptt_horizon
    off_policy_rows = int(config.replay_factor*config.batch_size // config.bptt_horizon)
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
        ratio = torch.ones(experience_rows, config.bptt_horizon, device=config.device),
    )
    ep_uses = torch.zeros(experience_rows, device=config.device, dtype=torch.int32)
    #stored_indices = torch.zeros(experience_rows, device=config.device, dtype=torch.int32)
    ep_lengths = torch.zeros(total_agents, device=config.device, dtype=torch.int32)
    ep_indices = torch.arange(total_agents, device=config.device, dtype=torch.int32)
    free_idx = total_agents
    assert free_idx <= experience_rows

    diayn_skills = None
    if config.use_diayn:
        diayn_skills = torch.randint(
            0, config.diayn_archive, (total_agents,), dtype=torch.long, device=config.device)
        experience.diayn_batch = torch.zeros(experience_rows, config.bptt_horizon,
            dtype=torch.long, device=config.device)

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

    if config.use_vtrace or config.use_puff_advantage:
        experience.importance = torch.ones(experience_rows, config.bptt_horizon, device=config.device)

    lstm_h = None
    lstm_c = None
    if isinstance(policy, torch.nn.LSTM):
        assert total_agents > 0
        shape = (policy.num_layers, total_agents, policy.hidden_size)
        lstm_h = torch.zeros(shape).to(config.device)
        lstm_c = torch.zeros(shape).to(config.device)

    minibatch_size = min(config.minibatch_size, config.max_minibatch_size)
    uncompiled_policy = policy
    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode, fullgraph=config.compile_fullgraph)

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
    elif config.optimizer == 'muon':
        from heavyball import ForeachMuon
        import heavyball.utils
        #heavyball.utils.compile_mode = "reduce-overhead"
        optimizer = ForeachMuon(
            policy.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,

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
    else:
        raise ValueError(f'Unknown optimizer: {config.optimizer}')

    epochs = config.total_timesteps // config.batch_size
    assert config.scheduler in ('linear', 'cosine')
    if config.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_context = nullcontext()
    scaler = None
    if config.precision != 'float32':
        amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, config.precision))
        scaler = torch.amp.GradScaler()

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
        puf=config.puf,
        use_diayn=config.use_diayn,
        diayn_coef=config.diayn_coef,
        # Do we use these?
        ptr=0,
        step=0,
        lstm_h=lstm_h,
        lstm_c=lstm_c,
        #stored_indices=stored_indices,
        ep_uses=ep_uses,
        ep_lengths=ep_lengths,
        ep_indices=ep_indices,
        free_idx=free_idx,
        on_policy_rows=on_policy_rows,
        off_policy_rows=off_policy_rows,
        experience_rows=experience_rows,
        device=config.device,
        minibatch_size=minibatch_size,
        compute_gae=compute_gae,
        compute_vtrace=compute_vtrace,
        compute_puff_advantage=compute_puff_advantage,
        diayn_skills=diayn_skills,
        total_agents=total_agents,
        total_epochs=epochs,
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

    while data.free_idx < data.on_policy_rows:
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
                state.diayn_z = data.diayn_skills[env_id]

            logits, value = policy(o_device, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits, is_continuous=policy.is_continuous)

            '''
            if data.use_diayn:
                diayn_policy = policy if lstm_h is None else policy.policy
                q = diayn_policy.diayn_discriminator(logits).squeeze()
                r_diayn = torch.log_softmax(q, dim=-1).gather(-1, state.diayn_z.unsqueeze(-1)).squeeze()
                r += config.diayn_coef*r_diayn# - np.log(1/data.diayn_archive)
            '''

            # Clip rewards
            r = torch.clamp(r, -1, 1)

        with profile.eval_copy, torch.no_grad():
            if lstm_h is not None:
                lstm_h[:, gpu_env_id] = state.lstm_h
                lstm_c[:, gpu_env_id] = state.lstm_c

            o = o if config.cpu_offload else o_device
            actions = store(data, state, o, value, action, logprob, r, d, gpu_env_id, mask)

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

    data.free_idx = 0
    data.ep_indices = torch.arange(data.total_agents, device=config.device, dtype=torch.int32)
    data.ep_lengths.zero_()
    data.ep_uses.zero_()

    # Process infos correctly
    processed_infos = defaultdict(list)
    if info: # Check if the outer list is not empty
        list_of_logs = info[0] # Get the inner list [{log1}, {log2}, ...]
        if list_of_logs: # Check if the inner list is not empty
            for log_dict in list_of_logs: # Iterate through each actual log dictionary
                # Only process dicts from envs that completed an episode (n > 0)
                if log_dict and isinstance(log_dict, dict) and log_dict.get('n', 0) > 0:
                    for k, v in pufferlib.utils.unroll_nested_dict(log_dict):
                        # The C code 'add_log' ensures n is 1 per finished episode log,
                        # so we don't need to divide return/length by n here.
                        processed_infos[k].append(v)

    # Update stats with processed infos
    for k, v_list in processed_infos.items():
        if '_map' in k:
            if data.wandb is not None:
                data.stats[f'Media/{k}'] = data.wandb.Image(v_list[0]) # Assuming first image is representative
                continue
            elif data.neptune is not None:
                # TODO: Add neptune image logging
                pass

        # Handle potential non-iterables or numpy arrays before extending
        current_list = data.stats[k]
        for item in v_list:
             if isinstance(item, np.ndarray):
                 current_list.extend(item.tolist())
             elif hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
                 current_list.extend(item)
             else:
                 current_list.append(item)

    return data.stats, info # Return original info for potential downstream use if needed

@pufferlib.utils.profile
def train(data):
    config = data.config
    profile = data.profile
    experience = data.experience
    losses = data.losses

 
    '''
    with profile.custom:
        if config.use_diayn:
            diayn_policy = data.policy.policy
            obs = experience.obs[:, ::8]
            q = diayn_policy.discrim_forward(obs)
            z_idxs = experience.diayn_batch[:, 0]
            q = q.view(-1, q.shape[-1])
            diayn_r = (torch.argmax(q, 1) == z_idxs).float()
            experience.rewards[:, -1] += 1.0*diayn_r
            print('DIAYN acc: ', diayn_r.mean())
    '''

    total_minibatches = int(config.update_epochs*config.batch_size/data.minibatch_size)
    accumulate_minibatches = max(1, config.minibatch_size // config.max_minibatch_size)
    n_samples = data.minibatch_size // config.bptt_horizon
    for mb in range(total_minibatches):
        with profile.train_misc:
            if config.use_p3o:
                # Note: This function gets messed up by computing across
                # episode bounds. Because we store experience in a flat buffer,
                # bounds can be crossed even after handling dones. This prevent
                # our method from scaling to longer horizons. TODO: Redo the way
                # we store experience to avoid this issue
                vstd_min = experience.values_std.min().item()
                vstd_max = experience.values_std.max().item()

                data.mask_block.zero_()
                data.buf.zero_()
                data.reward_block.zero_()
                data.bounds.zero_()

                r_mean = experience.rewards.mean().item()
                r_std = experience.rewards.std().item()

                # TODO: Rename vstd to r_std
                advantages = compute_advantages(
                    experience.reward_block, experience.mask_block,
                    experience.values_mean, experience.values_std,
                    experience.buf, experience.dones, experience.rewards,
                    experience.bounds, r_std, data.puf, config.p3o_horizon
                )

                horizon = torch.where(experience.values_std[0] > 0.95*r_std)[0]
                horizon = horizon[0].item()+1 if len(horizon) else 1
                if horizon < 16:
                    horizon = 16

                advantages = advantages.cpu().numpy()
                torch.cuda.synchronize()
            elif config.use_vtrace:
                importance = advantages = torch.zeros(experience.values.shape, device=config.device).to(config.device)
                vs = torch.zeros(experience.values.shape, device=config.device)
                data.compute_vtrace(experience.values, experience.rewards, experience.dones,
                    experience.ratio, vs, advantages, config.gamma, config.vtrace_rho_clip, config.vtrace_c_clip)
            elif config.use_puff_advantage:
                importance = advantages = torch.zeros(experience.values.shape, device=config.device).to(config.device)
                vs = torch.zeros(experience.values.shape, device=config.device)
                data.compute_puff_advantage(experience.values, experience.rewards, experience.dones,
                    experience.ratio, vs, advantages, config.gamma, config.gae_lambda, config.vtrace_rho_clip, config.vtrace_c_clip)
            else:
                importance = advantages = data.compute_gae(experience.values, experience.rewards,
                    experience.dones, config.gamma, config.gae_lambda)

        with profile.train_copy:
            batch = sample(data, importance, n_samples)

        loss = 0
        with profile.custom:
            if config.use_diayn:
                pass
                '''
                diayn_policy = data.policy.policy
                obs = batch.obs[:, ::8]
                q = diayn_policy.discrim_forward(obs)
                z_idxs = batch.diayn_z[:, 0]
                q = q.view(-1, q.shape[-1])
                diayn_loss = torch.nn.functional.cross_entropy(q, z_idxs)
                loss += config.diayn_loss_coef*diayn_loss
                '''

                '''
                with torch.no_grad():
                    batch.advantages *= diayn_r.unsqueeze(1).expand_as(batch.advantages)
                '''

                '''
                rewards = experience.rewards.clone()
                rewards[batch.idx, -1] += diayn_r
                advantages = data.compute_gae(experience.values, rewards,
                    experience.dones, config.gamma, config.gae_lambda)
                batch.advantages = advantages[batch.idx]
                '''


        with profile.train_misc:
            state = pufferlib.namespace(
                action=batch.actions,
                lstm_h=None,
                lstm_c=None,
            )

            if config.use_diayn:
                state.diayn_z = batch.diayn_z.reshape(-1)

        with profile.train_forward:
            if not isinstance(data.policy, torch.nn.LSTM):
                batch.obs = batch.obs.reshape(-1, *data.vecenv.single_observation_space.shape)

            # TODO: Currently only returning traj shaped value as a hack
            logits, newvalue = data.policy.forward_train(batch.obs, state)

            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits,
                action=batch.actions, is_continuous=data.policy.is_continuous)

            if config.use_diayn:
                N = 1
                batch_logits = state.batch_logits[:, ::N]
                batch_logits = torch.nn.functional.log_softmax(batch_logits, dim=-1)
                mask = torch.nn.functional.one_hot(batch.actions[:, ::N], batch_logits.shape[-1]).bool()
                #batch_logits = mask*batch_logits
                batch_logits = batch_logits.view(batch_logits.shape[0], -1)
                diayn_policy = data.policy.policy
                q = diayn_policy.discrim_forward(batch_logits)
                z_idxs = batch.diayn_z[:, 0]
                q = q.view(-1, q.shape[-1])
                diayn_loss = torch.nn.functional.cross_entropy(q, z_idxs)
                loss += config.diayn_loss_coef*diayn_loss
 
        with profile.train_misc:
            newlogprob = newlogprob.reshape(batch.logprobs.shape)
            logratio = newlogprob - batch.logprobs
            ratio = logratio.exp()
            experience.ratio[batch.idx] = ratio

            # TODO: Only do this if we are KL clipping? Saves 1-2% compute
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

            if config.use_vtrace or config.use_puff_advantage:
                with torch.no_grad():
                    adv = advantages[batch.idx]
                    vs = vs[batch.idx]
                    if config.use_vtrace:
                        data.compute_vtrace(batch.values, batch.rewards, batch.dones,
                            ratio, vs, adv, config.gamma, config.vtrace_rho_clip, config.vtrace_c_clip)
                    elif config.use_puff_advantage:
                        data.compute_puff_advantage(batch.values, batch.rewards, batch.dones,
                            ratio, vs, adv, config.gamma, config.gae_lambda, config.vtrace_rho_clip, config.vtrace_c_clip)

                    #advantages[batch.idx] = adv
                    #importance[batch.idx] = adv

            adv = batch.advantages
            if config.norm_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            adv = adv * batch.prio

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
                v_loss = criterion(newvalue_mean, batch.reward_block, newvalue_var)
                v_loss = v_loss[:, :(horizon+3)]
                mask_block = mask_block[:, :(horizon+3)]
                v_loss = v_loss[mask_block.bool()].mean()
            elif config.clip_vloss:
                newvalue = newvalue#.flatten()
                ret = batch.returns#.flatten()
                v_loss_unclipped = (newvalue - ret) ** 2
                val = batch.values#.flatten()
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
            loss += pg_loss - config.ent_coef*entropy_loss + v_loss*config.vf_coef

        # This breaks vloss clipping?
        with torch.no_grad():
            experience.values[batch.idx] = newvalue

        with profile.learn:
            if data.scaler is not None:
                loss = data.scaler.scale(loss)

            loss.backward()

            if data.scaler is not None:
                data.scaler.unscale_(data.optimizer)

            # TODO: Delete?
            with torch.no_grad():
                grads = torch.cat([p.grad.flatten() for p in data.policy.parameters()])
                grad_var = grads.var(0).mean() * config.minibatch_size
                data.msg = f'Gradient variance: {grad_var.item():.3f}'

            if (mb + 1) % accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)

                # TODO: Can remove scaler if only using bf16
                if data.scaler is None:
                    data.optimizer.step()
                else:
                    data.scaler.step(data.optimizer)
                    data.scaler.update()

                data.optimizer.zero_grad()


        with profile.train_misc:
            losses.policy_loss += pg_loss.item() / total_minibatches
            losses.value_loss += v_loss.item() / total_minibatches
            losses.entropy += entropy_loss.item() / total_minibatches
            losses.old_approx_kl += old_approx_kl.item() / total_minibatches
            losses.approx_kl += approx_kl.item() / total_minibatches
            losses.clipfrac += clipfrac.item() / total_minibatches
            losses.grad_var += grad_var.item() / total_minibatches
            losses.importance += ratio.mean().item() / total_minibatches

            if data.use_diayn:
                losses.diayn_loss += diayn_loss.item() / total_minibatches

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break

    # Reprioritize experience
    data.max_uses = data.ep_uses.max().item()
    data.mean_uses = data.ep_uses.float().mean().item()
    if config.replay_factor > 0:
        advantages = torch.zeros(experience.values.shape, device=config.device).to(config.device)
        vs = torch.zeros(experience.values.shape, device=config.device)
        data.compute_puff_advantage(experience.values, experience.rewards, experience.dones,
            experience.ratio, vs, advantages, config.gamma, config.gae_lambda, config.vtrace_rho_clip, config.vtrace_c_clip)

        exp = sample(data, advantages, data.off_policy_rows, method='random')
        for k, v in experience.items():
            v[data.on_policy_rows:] = exp[k]

    #print(advantages[:data.on_policy_rows].mean(), advantages[data.on_policy_rows:].mean())
    experience.ratio[:data.on_policy_rows] = 1

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

        logs = None
        done_training = data.global_step >= config.total_timesteps
        if done_training or profile.update(data):
            logs = mean_and_log(data)
            print_dashboard(config.env, data.utilization, data.global_step, data.epoch,
                profile, data.losses, data.stats, data.msg)
            data.stats = defaultdict(list)

        for k in losses:
            losses[k] = 0

        if data.epoch % config.checkpoint_interval == 0 or done_training:
            save_checkpoint(data)
            data.msg = f'Checkpoint saved at update {data.epoch}'

    return logs

def store(data, state, obs, value, action, logprob, reward, done, env_id, mask):
    exp = data.experience
    batch_rows = data.ep_indices[env_id]
    l = data.ep_lengths[env_id]

    if isinstance(env_id, slice):
        env_id = torch.arange(env_id.start, env_id.stop, device=data.device).int()

    #data.stored_indices[batch_rows] = env_id

    exp.obs[batch_rows, l] = obs
    exp.actions[batch_rows, l] = action
    exp.logprobs[batch_rows, l] = logprob
    exp.rewards[batch_rows, l] = reward
    exp.dones[batch_rows, l] = done.float()

    if data.use_p3o:
        exp.values_mean[batch_rows, l] = value.mean
        exp.values_std[batch_rows, l] = value.std
    else:
        exp.values[batch_rows, l] = value.flatten()

    if data.use_diayn:
        exp.diayn_batch[batch_rows, l] = state.diayn_z
        #idxs = env_id[done]
        #if len(idxs) > 0:
        #    z_idxs = torch.randint(0, data.config.diayn_archive, (done.sum(),)).to(data.device)
        #    data.diayn_skills[idxs] = z_idxs

    # TODO: Handle masks!!
    #indices = np.where(mask)[0]
    #data.ep_lengths[env_id[mask]] += 1
    data.ep_lengths[env_id] += 1
    full = data.ep_lengths[env_id] >= data.config.bptt_horizon
    num_full = full.sum()
    if num_full > 0:
        full_ids = env_id[full]
        data.ep_indices[full_ids] = data.free_idx + torch.arange(num_full, device=data.device).int()
        data.ep_lengths[full_ids] = 0
        data.free_idx += num_full

    data.step += 1
    return action.cpu().numpy()

def sample(data, advantages, n, reward_block=None, mask_block=None, method='prio'):
    exp = data.experience
    if method == 'topk':
        _, idx = torch.topk(advantages.abs().sum(axis=1), n)
    elif method == 'prio':
        adv = advantages.abs().sum(axis=1)
        probs = adv**data.config.prio_alpha
        probs = (probs + 1e-6)/(probs.sum() + 1e-6)
        idx = torch.multinomial(probs, n)
    elif method == 'multinomial':
        idx = torch.multinomial(advantages.abs().sum(axis=1) + 1e-6, n)
    elif method == 'random':
        idx = torch.randint(0, advantages.shape[0], (n,), device=data.device)
    else:
        raise ValueError(f'Unknown sampling method: {method}')


    data.ep_uses[idx] += 1
    output = {k: v[idx] for k, v in exp.items()}
    output['idx'] = idx

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
        output['diayn_z'] = exp.diayn_batch[idx]

    output['prio'] = 1
    if method == 'prio':
        beta = data.config.prio_beta0 + (1 - data.config.prio_beta0)*data.config.prio_alpha*data.epoch/data.total_epochs
        output['prio'] = (((1/len(probs)) * (1/probs[idx]))**beta).unsqueeze(1).expand_as(output['advantages'])

    return pufferlib.namespace(**output)

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

    agent_steps = int(dist_sum(data.global_step, device))
    logs = {
        'SPS': dist_sum(data.profile.SPS, device),
        'agent_steps': agent_steps,
        'epoch': int(dist_sum(data.epoch, device)),
        'learning_rate': data.optimizer.param_groups[0]["lr"],
        'max_uses': data.max_uses,
        'mean_uses': data.mean_uses,
        **{f'environment/{k}': dist_mean(v, device) for k, v in data.stats.items()},
        **{f'losses/{k}': dist_mean(v, device) for k, v in data.losses.items()},
        **{f'performance/{k}': dist_sum(v, device) for k, v in data.profile},
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

def save_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.epoch:06d}.pt'
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return model_path

    torch.save(data.uncompiled_policy.state_dict(), model_path)

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
    data.policy.uncompiled.load_state_dict(
        torch.load(model_path, weights_only=True), map_location=config.device)
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

    agent = agent_creator(env, policy_cls, rnn_cls, agent_kwargs).to(device)
    if model_path is not None:
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    ob, info = env.reset()
    driver = env.driver_env
    os.system('clear')

    state = pufferlib.namespace(
        lstm_h=None,
        lstm_c=None,
        diayn_z=torch.arange(env.num_agents, dtype=torch.long, device=device) % 4
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
        if tick > 1000 and tick % 1 == 0:
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
