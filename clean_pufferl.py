# TODO: Add information
# - Help menu
# - Docs link
#python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 clean_pufferl.py --env puffer_nmmo3 --mode train
#from torch.distributed.elastic.multiprocessing.errors import record
#@record

import os
import random
import psutil
import time
import configparser
import argparse
import shutil
import glob
import uuid
import ast

from threading import Thread
from collections import defaultdict, deque
from contextlib import nullcontext

import numpy as np

import torch

import torch.distributed
import torch.utils.cpp_extension

import pufferlib
import pufferlib.pytorch
import pufferlib.sweep
import pufferlib.vector
from pufferlib import _C

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import rich
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter
import rich.traceback
rich.traceback.install(show_locals=False)

c1 = '[cyan]'
c2 = '[white]'
b1 = '[bright_cyan]'
b2 = '[bright_white]'


class CleanPuffeRL:
    def __init__(self, config, vecenv, policy, neptune=False, wandb=False):
        # Backend perf optimization
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience buffer
        device = config['device']
        batch_size = config['batch_size']

        if config['bptt_horizon'] == 'auto':
            config['bptt_horizon'] = batch_size // total_agents

        horizon = config['bptt_horizon']
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f'Total agents {total_agents} <= segments {segments}'
            )

        self.ep_uses = torch.zeros(segments, device=device, dtype=torch.int32)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents
        experience = dict(
            obs=torch.zeros(segments, horizon, *obs_space.shape,
                dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
                pin_memory=device == 'cuda' and config['cpu_offload'],
                device='cpu' if config['cpu_offload'] else device),
            actions=torch.zeros(segments, horizon, *atn_space.shape,
                dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype], device=device),
            values = torch.zeros(segments, horizon, device=device),
            logprobs=torch.zeros(segments, horizon, device=device),
            rewards=torch.zeros(segments, horizon, device=device),
            dones=torch.zeros(segments, horizon, device=device),
            truncateds=torch.zeros(segments, horizon, device=device),
            ratio = torch.ones(segments, horizon, device=device),
        )
        self.experience = experience

        if config['use_vtrace'] or config['use_puff_advantage']:
            experience['importance'] = torch.ones(segments, horizon, device=device)

        # LSTM
        # TODO: This breaks compile
        if config['use_rnn']:
            # TODO: Doesn't exist in native envs
            # TODO: Replace slice with env idx or similar
            n = vecenv.agents_per_batch
            self.lstm_h = {i*n: torch.zeros(n, policy.hidden_size, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, policy.hidden_size, device=device) for i in range(total_agents//n)}


        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']
        max_minibatch_size = config['max_minibatch_size']
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size % max_minibatch_size != 0 and max_minibatch_size % minibatch_size != 0:
            # todo: better error
            raise pufferlib.APIUsageError(
                f'max_minibatch_size {max_minibatch_size} must be a multiple of minibatch_size {minibatch_size}'
            )

        self.accumulate_minibatches = max(1, config['minibatch_size'] // config['max_minibatch_size'])
        self.total_minibatches = int(config['update_epochs'] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon 
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f'minibatch_size {self.minibatch_size} must be divisible by horizon {horizon}'
            )

        # Torch compile
        self.uncompiled_policy = policy
        if config['compile']:
            policy = torch.compile(policy, mode=config['compile_mode'], fullgraph=config['compile_fullgraph'])

        self.policy = policy

        # Optimizer
        # TODO: **optim params
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
            )
        elif config['optimizer'] == 'muon':
            from heavyball import ForeachMuon
            import heavyball.utils
            heavyball.utils.compile_mode = config['compile_mode'] if config['compile'] else None
            optimizer = ForeachMuon(
                policy.parameters(),
                lr=config['learning_rate'],
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_eps'],
            )
        else:
            raise ValueError(f'Unknown optimizer: {config["optimizer"]}')

        self.optimizer = optimizer

        # Learning rate scheduler
        epochs = config['total_timesteps'] // config['batch_size']
        self.total_epochs = epochs
        if config['scheduler'] == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        elif config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            raise pufferlib.APIUsageError(f'Unknown scheduler: {config["scheduler"]}')

        self.scheduler = scheduler

        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Logging
        self.neptune = neptune
        self.wandb = wandb
        if neptune:
            self.neptune = init_neptune(args, tag=config['tag'])
            self.run_id = self.neptune._sys_id
            for k, v in pufferlib.unroll_nested_dict(args):
                self.neptune[k].append(v)
        elif wandb:
            self.wandb = init_wandb(args, tag=config['tag'])
            self.run_id = self.wandb.run.id
        else:
            self.run_id = int(random.random() * 1e8)

        # Profiling
        self.uptime = 0
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile(['eval', 'env', 'eval_forward', 'eval_copy', 'eval_misc', 'train', 'train_forward',
            'learn', 'train_copy', 'train_misc', 'custom'], frequency=5)

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.global_step = 0
        self.epoch = 0
        self.stats = defaultdict(list) # TODO: can this be set in eval and handle accum differently?

        # Dashboard
        self.losses = {}
        num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.msg = f'Model Size: {abbreviate(num_params)} parameters'
        self.print_dashboard(clear=True)


    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        experience = self.experience
        policy = self.policy
        device = config['device']

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            # TODO: Port to vecenv
            env_id = slice(env_id[0], env_id[-1] + 1)

            # TODO: Handle truncations
            done_mask = d + t
            self.global_step += mask.sum()

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device, non_blocking=True)
            r = torch.as_tensor(r).to(device, non_blocking=True)
            d = torch.as_tensor(d).to(device, non_blocking=True)

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

                logits, value = policy(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits, is_continuous=policy.is_continuous)
                r = torch.clamp(r, -1, 1)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                o = o if config['cpu_offload'] else o_device
                actions = self.store(state, o, value, action, logprob, r, d, env_id, mask)

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
            self.vecenv.send(actions)

        profile('eval_misc', epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        self.ep_uses.zero_()
        profile.end()
        return self.stats

    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)

        config = self.config
        experience = self.experience
        losses = defaultdict(float)
        device = config['device']

        for mb in range(self.total_minibatches):
            profile('train_misc', epoch, nest=True)
            self.amp_context.__enter__()

            loss = 0
            shape = experience['values'].shape
            if config['use_vtrace']:
                importance = advantages = torch.zeros(shape, device=device)
                vs = torch.zeros(shape, device=device)
                self.compute_vtrace(experience['values'], experience['rewards'],
                    experience['dones'], experience['ratio'], vs, advantages,
                    config['gamma'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
            elif config['use_puff_advantage']:
                importance = advantages = torch.zeros(shape, device=device)
                vs = torch.zeros(shape, device=device)

                # TODO: Eliminate
                n = (shape[0]//256)*256
                torch.ops.pufferlib.compute_puff_advantage(experience['values'][:n], experience['rewards'][:n],
                    experience['dones'][:n], experience['ratio'][:n], vs[:n], advantages[:n], config['gamma'],
                    config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
            else:
                importance = advantages = self.compute_gae(experience['values'], experience['rewards'],
                    experience['dones'], config['gamma'], config['gae_lambda'])

            profile('train_copy', epoch)
            batch = self.sample(importance, self.minibatch_segments)

            profile('train_forward', epoch)
            state = dict(
                action=batch['actions'],
                lstm_h=None,
                lstm_c=None,
            )

            if not config['use_rnn']:
                batch['obs'] = batch['obs'].reshape(-1, *self.vecenv.single_observation_space.shape)

            # TODO: Currently only returning traj shaped value as a hack
            logits, newvalue = self.policy.forward_train(batch['obs'], state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits,
                action=batch['actions'], is_continuous=self.policy.is_continuous)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(batch['logprobs'].shape)
            logratio = newlogprob - batch['logprobs']
            ratio = logratio.exp()
            experience['ratio'][batch['idx']] = ratio # TODO: Experiment with this

            # TODO: Only do this if we are KL clipping? Saves 1-2% compute
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            # TODO: Do you need to do this? Policy hasn't changed
            if config['use_vtrace'] or config['use_puff_advantage']:
                with torch.no_grad():
                    adv = advantages[batch['idx']]
                    vs = vs[batch['idx']]
                    if config['use_vtrace']:
                        self.compute_vtrace(batch['values'], batch['rewards'], batch['dones'],
                            ratio, vs, adv, config['gamma'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
                    elif config['use_puff_advantage']:
                        torch.ops.pufferlib.compute_puff_advantage(batch['values'], batch['rewards'], batch['dones'],
                            ratio, vs, adv, config['gamma'], config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            adv = batch['advantages']
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Prioritized replay
            adv = adv * batch['prio']

            # Policy loss
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(
                ratio, 1 - config['clip_coef'], 1 + config['clip_coef']
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            ret = batch['returns']
            newvalue = newvalue.view(ret.shape)
            v_loss_unclipped = (newvalue - ret) ** 2
            val = batch['values']
            v_clipped = val + torch.clamp(
                newvalue - val,
                -config['vf_clip_coef'],
                config['vf_clip_coef'],
            )
            v_loss_clipped = (v_clipped - ret) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            # Entropy loss
            entropy_loss = entropy.mean()

            # Total loss
            loss += pg_loss - config['ent_coef']*entropy_loss + v_loss*config['vf_coef']
            self.amp_context.__enter__()

            # This breaks vloss clipping?
            with torch.no_grad():
                experience['values'][batch['idx']] = newvalue.float()

            profile('learn', epoch)
            loss.backward()

            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

                profile('train_misc', epoch)
                losses['policy_loss'] += pg_loss.item() / self.total_minibatches
                losses['value_loss'] += v_loss.item() / self.total_minibatches
                losses['entropy'] += entropy_loss.item() / self.total_minibatches
                losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
                losses['approx_kl'] += approx_kl.item() / self.total_minibatches
                losses['clipfrac'] += clipfrac.item() / self.total_minibatches
                losses['importance'] += ratio.mean().item() / self.total_minibatches

        # Reprioritize experience
        profile('train_misc', epoch)
        self.max_uses = self.ep_uses.max().item()
        self.mean_uses = self.ep_uses.float().mean().item()
        experience['ratio'][:] = 1

        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = experience['values'].flatten()
        # TODO: Probably not updated
        y_true = advantages.flatten() + experience['values'].flatten()

        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses['explained_variance'] = explained_var.item()

        profile.end()
        profile.clear()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() - self.start_time - self.uptime > 1:
            self.uptime = time.time() - self.start_time
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs

    def store(self, state, obs, value, action, logprob, reward, done, env_id, mask):
        config = self.config
        exp = self.experience

        # Fast path for fully vectorized envs
        l = self.ep_lengths[env_id.start].item()
        batch_rows = slice(self.ep_indices[env_id.start].item(), 1+self.ep_indices[env_id.stop - 1].item())

        exp['obs'][batch_rows, l] = obs
        exp['actions'][batch_rows, l] = action
        exp['logprobs'][batch_rows, l] = logprob
        exp['rewards'][batch_rows, l] = reward
        exp['dones'][batch_rows, l] = done.float()
        exp['values'][batch_rows, l] = value.flatten()

        # TODO: Handle masks!!
        #indices = np.where(mask)[0]
        #data.ep_lengths[env_id[mask]] += 1
        self.ep_lengths[env_id] += 1
        if l+1 >= config['bptt_horizon']:
            num_full = env_id.stop - env_id.start
            self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
            self.ep_lengths[env_id] = 0
            self.free_idx += num_full
            self.full_rows += num_full

        return action.cpu().numpy()

    def sample(self, advantages, n, reward_block=None, mask_block=None, method='prio'):
        config = self.config
        exp = self.experience
        if method == 'topk':
            _, idx = torch.topk(advantages.abs().sum(axis=1), n)
        elif method == 'prio':
            adv = advantages.abs().sum(axis=1)
            probs = adv**config['prio_alpha']
            probs = torch.nan_to_num(probs, 0, 0, 0)
            probs = (probs + 1e-6)/(probs.sum() + 1e-6)
            idx = torch.multinomial(probs, n)
        elif method == 'multinomial':
            idx = torch.multinomial(advantages.abs().sum(axis=1) + 1e-6, n)
        elif method == 'random':
            idx = torch.randint(0, advantages.shape[0], (n,), device=config['device'])
        else:
            raise ValueError(f'Unknown sampling method: {method}')

        self.ep_uses[idx] += 1
        output = {k: v[idx] for k, v in exp.items()}
        output['idx'] = idx
        output['values'] = exp['values'][idx]
        output['advantages'] = advantages[idx]
        output['returns'] = advantages[idx] + exp['values'][idx]

        output['prio'] = 1
        if method == 'prio':
            b0 = config['prio_beta0']
            a = config['prio_alpha']
            beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
            output['prio'] = (((1/len(probs)) * (1/probs[idx]))**beta).unsqueeze(1).expand_as(output['advantages'])

        return dict(**output)

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
            #'SPS': dist_sum(self.profile.SPS, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            'learning_rate': self.optimizer.param_groups[0]["lr"],
            'max_uses': self.max_uses,
            'mean_uses': self.mean_uses,
            **{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            **{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            **{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return logs

        if self.wandb:
            self.wandb.log(logs)
        elif self.neptune:
            for k, v in logs.items():
                self.neptune[k].append(v, step=agent_steps)

        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        path = os.path.join(self.config['data_dir'], f'{self.run_id}.pt')
        shutil.copy(model_path, path)
        if self.wandb:
            artifact = self.wandb.Artifact(self.run_id, type='model')
            artifact.add_file(path)
            self.wandb.run.log_artifact(artifact)
            self.wandb.finish()
        elif self.neptune:
            self.neptune['model'].track_files(path)
            self.neptune.stop()

    def save_checkpoint(self):
        path = os.path.join(self.config['data_dir'], self.run_id)
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.epoch:06d}.pt'
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
            'run_id': self.run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.rename(state_path + '.tmp', state_path)
        return model_path

    def try_load_checkpoint(self):
        config = self.config
        path = os.path.join(config['data_dir'], self.run_id)
        if not os.path.exists(path):
            print('No checkpoints found. Assuming new experiment')
            return

        trainer_path = os.path.join(path, 'trainer_state.pt')
        resume_state = torch.load(trainer_path, weights_only=False)
        model_path = os.path.join(path, resume_state['model_name'])
        self.policy.uncompiled.load_state_dict(
            torch.load(model_path, weights_only=True), map_location=config['device'])
        self.optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        print(f'Loaded checkpoint {resume_state["model_name"]}')

    def print_dashboard(self, clear=False, max_stats=[0]):
        utilization = self.utilization
        profile = self.profile
        config = self.config
        console = Console()
        if clear:
            console.clear()

        dashboard = Table(box=rich.box.ROUNDED, expand=True,
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
            f':blowfish: {b1}PufferLib {b2}2.0.0',
            f'{c1}CPU: {b2}{cpu_percent:.1f}{c2}%',
            f'{c1}GPU: {b2}{gpu_percent:.1f}{c2}%',
            f'{c1}DRAM: {b2}{dram_percent:.1f}{c2}%',
            f'{c1}VRAM: {b2}{vram_percent:.1f}{c2}%',
        )
            
        s = Table(box=None, expand=True)
        SPS = 0
        delta = profile.eval['delta'] + profile.train['delta']
        remaining = 'A hair past a freckle'
        if delta != 0:
            SPS = config['batch_size'] / delta
            remaining = duration((config['total_timesteps'] - self.global_step)/SPS)

        uptime = time.time() - self.start_time
        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{c2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{c2}Steps', abbreviate(self.global_step))
        s.add_row(f'{c2}SPS', abbreviate(SPS))
        s.add_row(f'{c2}Epoch', abbreviate(self.epoch))
        s.add_row(f'{c2}Uptime', duration(uptime))
        s.add_row(f'{c2}Remaining', remaining)

        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval))
        p.add_row(*fmt_perf('  Forward', c2, delta, profile.eval_forward))
        p.add_row(*fmt_perf('  Env', c2, delta, profile.env))
        p.add_row(*fmt_perf('  Copy', c2, delta, profile.eval_copy))
        p.add_row(*fmt_perf('  Misc', c2, delta, profile.eval_misc))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train))
        p.add_row(*fmt_perf('  Forward', c2, delta, profile.train_forward))
        p.add_row(*fmt_perf('  Learn', c2, delta, profile.learn))
        p.add_row(*fmt_perf('  Copy', c2, delta, profile.train_copy))
        p.add_row(*fmt_perf('  Misc', c2, delta, profile.train_misc))
        if 'custom' in profile.profiles:
            p.add_row(*fmt_perf('  Custom', c2, uptime, profile.custom))

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
        for metric, value in self.stats.items():
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
        table.add_row(f' {c1}Message: {c2}{self.msg}')

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

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
    def __init__(self, keys, frequency=1):
        self.stack = []
        self.frequency = frequency
        self.profiles = {k:
            dict(
                start = 0,
                buffer = 0,
                delta = 0,
                elapsed = 0,
                calls = 0,
            ) for k in keys
        }

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        if epoch % self.frequency != 0:
            return

        torch.cuda.synchronize()
        tick = time.time()

        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['buffer'] += delta
        profile['elapsed'] += delta
        profile['calls'] += 1

    def end(self):
        torch.cuda.synchronize()
        end = time.time()

        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for v in self.profiles.values():
            if v['buffer'] != 0:
                v['delta'] = v['buffer']

            v['buffer'] = 0

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
            self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*(total-free)/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

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

def fmt_perf(name, color, delta_ref, prof):
    percent = 0 if delta_ref == 0 else int(100*prof['delta']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed']), f'{b2}{percent:2d}{c2}%'


def init_wandb(args, id=None, resume=True, tag=None):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        group=args['wandb_group'],
        allow_val_change=True,
        save_code=False,
        resume=resume,
        config=args,
        tags=[tag] if tag is not None else [],
    )
    return wandb

def init_neptune(args, id=None, resume=True, tag=None, mode="async"):
    import neptune
    import neptune.exceptions
    try:
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        run = neptune.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            tags=[tag] if tag is not None else [],
            mode=mode,
        )
    except neptune.exceptions.NeptuneConnectionLostException:
        print("couldn't connect to neptune, logging in offline mode")
        return init_neptune(args, name, id, resume, tag, mode="offline")
    return run

def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'])
    args['rnn']['input_size'] = policy.hidden_size
    args['rnn']['hidden_size'] = policy.hidden_size
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])

    return policy.to(args['train']['device'])

def downsample_linear(arr, m):
    n = len(arr)
    x_old = np.linspace(0, 1, n)  # Original indices normalized
    x_new = np.linspace(0, 1, m)  # New indices normalized
    return np.interp(x_new, x_old, arr)

def experiment(vecenv, policy, args):
    train_config = dict(**args['train'], env=env_name, tag=args['tag'])
    pufferl = CleanPuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

    all_logs = []
    while pufferl.global_step < train_config['total_timesteps']:
        pufferl.evaluate()
        logs = pufferl.train()
        if logs is not None:
            all_logs.append(logs)

    vecenv.async_reset(train_config['seed'])
    for _ in range(10):
        stats = pufferl.evaluate()

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.close()
    return all_logs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--env', '--environment', type=str,
        default='puffer_squared', help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train',
        choices='train eval sweep autotune profile'.split())
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
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    args = parser.parse_known_args()[0]

    # Load defaults and config
    for path in glob.glob('config/**/*.ini', recursive=True):
        p = configparser.ConfigParser()
        p.read(['config/default.ini', path])
        if args.env in p['base']['env_name'].split(): break
    else:
        raise pufferlib.APIUsageError('No config for env_name {}'.format(args.env))

    # Dynamic help menu from config
    for section in p.sections():
        for key in p[section]:
            try:
                value = ast.literal_eval(p[section][key])
            except:
                value = p[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(fmt.replace('_', '-'), default=value)

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    env_name = parsed.pop('env')
    args = {}
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    # Dynamically import environment and policy
    import importlib
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    policy_cls = getattr(env_module.torch, args['policy_name'])
    rnn_name = args['rnn_name']
    rnn_cls = None
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)

    if args['mode'] == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
        exit(0)

    args['train']['use_rnn'] = rnn_cls is not None
    env_name = args['env_name']
    device = args['train']['device']

    if args['mode'] == 'sweep':
        if not args['wandb'] and not args['neptune']:
            raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')

        method = args['sweep'].pop('method')
        try:
            sweep_cls = getattr(pufferlib.sweep, method)
        except:
            raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

        sweep = sweep_cls(args['sweep'])
        target_key = f'environment/{args["sweep"]["metric"]}'
        total_timesteps = args['train']['total_timesteps']
        for i in range(args['max_runs']):
            seed = time.time_ns() & 0xFFFFFFFF
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sweep.suggest(args)

            vecenv = pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])
            policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
            all_logs = experiment(vecenv, policy, args)

            scores = downsample_linear([log[target_key] for log in all_logs], 10)
            costs = downsample_linear([log['uptime'] for log in all_logs], 10)
            timesteps = downsample_linear([log['agent_steps'] for log in all_logs], 10)

            for score, cost, timestep in zip(scores, costs, timesteps):
                args['train']['total_timesteps'] = timestep
                sweep.observe(args, score, cost)

            # Prevent logging final eval steps as training steps
            args['train']['total_timesteps'] = total_timesteps

        exit(0)

    if args['mode'] == 'eval':
        args['vec'] = dict(backend='Serial', num_envs=1)
        
    vecenv = pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)

    load_id = args['load_id']
    if load_id is not None:
        if args['mode'] not in ('train', 'eval'):
            raise pufferlib.APIUsageError('load_id requires mode to be train or eval')

        if args['neptune']:
            import neptune
            neptune_name = args['neptune_name']
            neptune_project = args['neptune_project']
            run = neptune.init_run(
                project=f"{neptune_name}/{neptune_project}",
                with_id=load_id, mode="read-only")
            data_dir = 'artifacts'
            run["model"].download(destination=data_dir)
        elif args['wandb']:
            run = init_wandb(args, load_id, resume='must')
            artifact = run.use_artifact(f'{load_id}:latest')
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        policy.load_state_dict(torch.load(f'{data_dir}/{load_id}.pt', map_location=device))

    if args['load_model_path'] is not None:
        policy.load_state_dict(torch.load(
            args['load_model_path'], map_location=args['train']['device']))

    if args['mode'] == 'train':
        experiment(vecenv, policy, args)
    elif args['mode'] == 'eval':
        ob, info = vecenv.reset()
        driver = vecenv.driver_env
        num_agents = vecenv.observation_space.shape[0]
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

        frames = []
        while True:
            render = driver.render()
            if len(frames) < args['save_frames']:
                frames.append(render)

            # TODO: Frames from raylib
            if driver.render_mode == 'ansi':
                print('\033[0;0H' + render + '\n')
                time.sleep(1/args['fps'])
            elif driver.render_mode == 'rgb_array':
                import cv2
                render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', render)
                cv2.waitKey(1)
                time.sleep(1/args['fps'])

            with torch.no_grad():
                ob = torch.as_tensor(ob).to(args['train']['device'])
                logits, value = policy(ob, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(
                    logits, is_continuous=policy.is_continuous)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            ob = vecenv.step(action)[0]

            if len(frames) > 0 and len(frames) == args['save_frames']:
                import imageio
                imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
                frames.append('Done')
    elif args['mode'] == 'profile':
        import torch
        import torchvision.models as models
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                for _ in range(10):
                    stats = pufferl.evaluate()
                    pufferl.train()

        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
        prof.export_chrome_trace("trace.json")
