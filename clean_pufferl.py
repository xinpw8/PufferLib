##############################################################################
# clean_pufferl.py
#
# A "critic-free" GRPO implementation inspired by the references you provided.
# This removes the typical value function and GAE logic, uses group-relative
# rewards as advantages, and applies a clipped policy gradient with a KL penalty.
#
# Please note:
#  - For many multi-step RL game environments, you often rely on temporal
#    credit assignment (e.g. a "critic" or multi-step returns). Purely
#    using final outcomes or single-step rewards as "group" signals
#    may suffice in some tasks but could be suboptimal in more complex
#    settings. Still, the code below follows the GRPO references that
#    remove the separate critic and rely on group-based relative rewards.
#  - We treat the entire collected batch as one "group." Each sample's
#    advantage is (r_i - mean(r)) / std(r) across the batch. If you
#    prefer per-episode or smaller groupings, you can adjust accordingly.
#  - We keep "old_policy" to impose KL regularization and (optionally)
#    ratio clipping (PPO-style) for stable updates.
##############################################################################

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

import pufferlib
import pufferlib.utils
import pufferlib.pytorch

torch.set_float32_matmul_precision('high')


def create(config, vecenv, policy, optimizer=None, wandb=None):
    """
    Create training context for GRPO.

    Key differences from PPO:
      - No critic references or GAE.
      - We'll store an 'old_policy' to compute KL and old logprobs for ratio.
      - The rest of the library (env steps, etc.) can remain mostly unchanged.
    """
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

    # This GRPO example does not rely on a value function or GAE
    # (critic-free). We won't store or compute "values" at all.
    # We'll store transitions for logprobs, rewards, etc.
    # If you want a multi-step approach or partial critics, you can add them.
    lstm = policy.lstm if hasattr(policy, 'lstm') else None
    experience = Experience(
        config.batch_size,
        config.bptt_horizon,
        config.minibatch_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        atn_dtype,
        config.cpu_offload,
        config.device,
        lstm,
        total_agents
    )

    # Build a reference (old) policy for KL and ratio
    old_policy = type(policy)(policy.policy)
    old_policy.load_state_dict(policy.state_dict())
    old_policy.to(config.device)
    old_policy.eval()

    uncompiled_policy = policy
    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)

    return pufferlib.namespace(
        config=config,
        vecenv=vecenv,
        policy=policy,
        old_policy=old_policy,  # reference
        uncompiled_policy=uncompiled_policy,
        optimizer=optimizer,
        experience=experience,
        profile=profile,
        losses=losses,
        wandb=wandb,
        global_step=0,
        epoch=0,
        stats=defaultdict(list),
        msg=msg,
        last_log_time=0,
        utilization=utilization,
    )

@pufferlib.utils.profile
def evaluate(data):
    """
    Same as PPO: We just run the current policy in the environment,
    store transitions (obs, action, logprob, reward, done), ignoring
    any "value" since we are critic-free.
    """
    config, profile, experience = data.config, data.profile, data.experience

    with profile.eval_misc:
        policy = data.policy
        infos = defaultdict(list)
        lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

    while not experience.full:
        with profile.env:
            o, r, d, t, info, env_id, mask = data.vecenv.recv()
            env_id = env_id.tolist()

        with profile.eval_misc:
            data.global_step += sum(mask)
            o = torch.as_tensor(o).to(config.device)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)

        with profile.eval_forward, torch.no_grad():
            if lstm_h is not None:
                h = lstm_h[:, env_id]
                c = lstm_c[:, env_id]
                actions, logprob, entropy, _, (h, c) = policy(o, (h, c))
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c
            else:
                actions, logprob, entropy, _ = policy(o)

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_misc:
            actions_np = actions.cpu().numpy()
            mask = torch.as_tensor(mask)
            # Store in buffer
            experience.store(
                obs=o,  # (already on device)
                action=actions_np,
                logprob=logprob.cpu(),
                reward=r.cpu(),
                done=d.cpu(),
                env_id=env_id,
                mask=mask
            )

            for i in info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    infos[k].append(v)

        with profile.env:
            data.vecenv.send(actions_np)

    with profile.eval_misc:
        for k, v in infos.items():
            if '_map' in k and data.wandb is not None:
                data.stats[f'Media/{k}'] = data.wandb.Image(v[0])
                continue
            if isinstance(v, np.ndarray):
                v = v.tolist()
            try:
                iter(v)
            except TypeError:
                data.stats[k].append(v)
            else:
                data.stats[k] += v

    experience.ptr = 0
    experience.step = 0
    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    """
    Minimal, critic-free GRPO:
     - Copy current policy -> old_policy
     - Compute 'group advantage' = (r_i - mean(r)) / (std(r)+1e-8)
     - ratio = exp(new_logprob - old_logprob)
     - clipped PG loss plus a KL penalty
     - no value function update
    """
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    # Copy policy -> old_policy once per iteration
    data.old_policy.load_state_dict(data.policy.state_dict())
    data.old_policy.eval()

    with profile.train_misc:
        idxs = experience.sort_training_data()
        # For critic-free: no GAE or value-based approach
        # We have stored immediate rewards in experience.
        # Let's define "group advantage" simply as:
        #    A_i = (r_i - mean(r)) / (std(r) + 1e-8)
        # across the entire batch (the "group").
        # If your environment returns multiple steps,
        # this is an extremely naive approach but matches
        # the "critic-free" principle.

        rew_np = experience.rewards_np[idxs]
        # The entire group's distribution of rewards
        mean_r = rew_np.mean()
        std_r = rew_np.std() + 1e-8
        advantages_np = (rew_np - mean_r) / std_r

        # Flatten the batch for indexing (like PPO),
        # but the advantage is simply your group advantage
        experience.flatten_batch(advantages_np)

    total_minibatches = experience.num_minibatches * config.update_epochs
    for epoch in range(config.update_epochs):
        lstm_state = None
        for mb in range(experience.num_minibatches):
            with profile.train_misc:
                obs = experience.b_obs[mb].to(config.device)
                atn = experience.b_actions[mb]
                adv = experience.b_advantages[mb]
                old_logprob = experience.b_logprobs[mb]
                # shape them properly
                adv = adv.reshape(-1)
                old_logprob = old_logprob.reshape(-1)

            with profile.train_forward:
                # Current policy forward
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, _, lstm_state = data.policy(
                        obs, state=lstm_state, action=atn
                    )
                    # old/reference policy logprobs
                    with torch.no_grad():
                        _, oldlogprob_batch, _, _, _ = data.old_policy(
                            obs, state=None, action=atn
                        )
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, _ = data.policy(obs, action=atn)
                    with torch.no_grad():
                        _, oldlogprob_batch, _, _ = data.old_policy(obs, action=atn)

                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                # advantage normalization (again) if desired:
                if config.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ratio = exp( logπ_new - logπ_old )
                ratio = (newlogprob - oldlogprob_batch).exp()

                # clip the ratio if using PPO-style clipping
                #  (From the reference doc, "Step 5: Update with Clipping.")
                #  If you don't want clipping, set clip_coef=0.0
                clip_coef = getattr(config, "clip_coef", 0.2)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss_1 = -adv * ratio
                pg_loss_2 = -adv * clipped_ratio
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                # KL penalty with old policy
                # A simple approximation: KL ~ E[ old_logprob - new_logprob ]
                approx_kl = (oldlogprob_batch - newlogprob).mean()
                kl_coef = getattr(config, "kl_coef", 1.0)

                # Entropy bonus
                entropy_loss = entropy.mean()
                # Final GRPO objective
                loss = pg_loss - config.ent_coef * entropy_loss + kl_coef * approx_kl

            with profile.learn:
                data.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)
                data.optimizer.step()
                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                losses.policy_loss += pg_loss.item() / total_minibatches
                losses.entropy += entropy_loss.item() / total_minibatches
                losses.kl += approx_kl.item() / total_minibatches

    # If you want to anneal LR
    with profile.train_misc:
        if config.anneal_lr:
            frac = 1.0 - data.global_step / config.total_timesteps
            lrnow = frac * config.learning_rate
            data.optimizer.param_groups[0]["lr"] = lrnow

        data.epoch += 1
        done_training = data.global_step >= config.total_timesteps
        if done_training or profile.update(data):
            mean_and_log(data)
            print_dashboard(config.env, data.utilization, data.global_step,
                            data.epoch, profile, data.losses, data.stats, data.msg)
            data.stats = defaultdict(list)

        # Save model if needed
        if data.epoch % config.checkpoint_interval == 0 or done_training:
            save_checkpoint(data)
            data.msg = f'Checkpoint saved at update {data.epoch}'


def mean_and_log(data):
    for k in list(data.stats.keys()):
        v = data.stats[k]
        try:
            v = np.mean(v)
        except:
            del data.stats[k]
            continue
        data.stats[k] = v

    if data.wandb is None:
        return

    data.last_log_time = time.time()
    logs = {
        '0verview/SPS': data.profile.SPS,
        '0verview/agent_steps': data.global_step,
        '0verview/epoch': data.epoch,
        '0verview/learning_rate': data.optimizer.param_groups[0]["lr"],
    }
    logs.update({f'environment/{k}': v for k, v in data.stats.items()})
    logs.update({f'losses/{k}': v for k, v in data.losses.items()})
    for k, v in data.profile:
        logs[f'performance/{k}'] = v
    data.wandb.log(logs)


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


##############################################################################
# HELPER CLASSES (unchanged or lightly adapted from PPO)
##############################################################################

class Profile:
    SPS: ... = 0
    uptime: ... = 0
    remaining: ... = 0
    eval_time: ... = 0
    env_time: ... = 0
    eval_forward_time: ... = 0
    eval_misc_time: ... = 0
    train_time: ... = 0
    train_forward_time: ... = 0
    learn_time: ... = 0
    train_misc_time: ... = 0

    def __init__(self):
        self.start = time.time()
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
        self.prev_steps = 0

    def __iter__(self):
        yield 'SPS', self.SPS
        yield 'uptime', self.uptime
        yield 'remaining', self.remaining
        yield 'eval_time', self.eval_time
        yield 'env_time', self.env_time
        yield 'eval_forward_time', self.eval_forward_time
        yield 'eval_misc_time', self.eval_misc_time
        yield 'train_time', self.train_time
        yield 'train_forward_time', self.train_forward_time
        yield 'learn_time', self.learn_time
        yield 'train_misc_time', self.train_misc_time

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
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = data._timers['train'].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed
        return True


def make_losses():
    """
    We remove references to value_loss and explained_variance
    because this is a critic-free approach.
    """
    return pufferlib.namespace(
        policy_loss=0,
        entropy=0,
        kl=0,
    )


class Experience:
    """
    Minimal storage for GRPO:
      - obs
      - actions
      - logprobs
      - rewards
      - no values stored
      - done flags
    We still handle batch/bptt logic similarly to your library,
    but we do not do GAE or store any value predictions.
    """

    def __init__(
        self,
        batch_size,
        bptt_horizon,
        minibatch_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        atn_dtype,
        cpu_offload=False,
        device='cuda',
        lstm=None,
        lstm_total_agents=0
    ):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == 'cuda' and cpu_offload

        self.obs = torch.zeros(batch_size, *obs_shape, dtype=obs_dtype,
                               pin_memory=pin, device=(device if not pin else 'cpu'))
        self.actions = torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, pin_memory=pin)
        self.logprobs = torch.zeros(batch_size, pin_memory=pin)
        self.rewards = torch.zeros(batch_size, pin_memory=pin)
        self.dones = torch.zeros(batch_size, pin_memory=pin)
        self.truncateds = torch.zeros(batch_size, pin_memory=pin)  # not used here

        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)

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
        self.minibatch_size = minibatch_size
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, action, logprob, reward, done, env_id, mask):
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[: self.batch_size - ptr]
        end = ptr + len(indices)

        self.obs[ptr:end] = obs[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob[indices]
        self.rewards_np[ptr:end] = reward[indices]
        self.dones_np[ptr:end] = done[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])

        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(sorted(range(len(self.sort_keys)),
                                 key=self.sort_keys.__getitem__))
        # shape the indexes for BFS-based rollout
        self.b_idxs_obs = torch.as_tensor(
            idxs.reshape(self.minibatch_rows, self.num_minibatches, self.bptt_horizon)
                .transpose(1, 0, -1)
        ).to(self.obs.device).long()

        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(self.num_minibatches, self.minibatch_size)
        self.sort_keys = []
        return idxs

    def flatten_batch(self, advantages_np):
        """
        In PPO code, we typically combine advantage with stored values, etc.
        Here, we only need advantage. We'll store it in b_advantages.
        """
        advantages = torch.as_tensor(advantages_np).to(self.device)
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat

        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.actions[b_idxs].to(self.device, non_blocking=True)
        self.b_logprobs = self.logprobs[b_idxs].to(self.device, non_blocking=True)
        self.b_dones = self.dones[b_idxs].to(self.device, non_blocking=True)

        self.b_advantages = advantages.reshape(
            self.minibatch_rows, self.num_minibatches, self.bptt_horizon
        ).transpose(0, 1).reshape(self.num_minibatches, self.minibatch_size)


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
            self.cpu_util.append(100 * psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)
            if torch.cuda.is_available():
                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * free / total)
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
    resume_state = torch.load(trainer_path, map_location=config.device)
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

    env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs, backend=backend)

    if model_path is None:
        agent = agent_creator(env, policy_cls, rnn_cls, agent_kwargs).to(device)
    else:
        agent = torch.load(model_path, map_location=device)

    ob, info = env.reset()
    driver = env.driver_env
    os.system('clear')
    state = None

    frames = []
    tick = 0
    while tick <= 2000:
        if tick % 1 == 0:
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
                time.sleep(1 / 24)
            elif driver.render_mode in ('human', 'raylib') and render is not None:
                frames.append(render)

        with torch.no_grad():
            ob_t = torch.as_tensor(ob).to(device)
            if hasattr(agent, 'lstm'):
                actions, _, _, _, state = agent(ob_t, state)
            else:
                actions, logprob, ent, _ = agent(ob_t)

            actions_np = actions.cpu().numpy().reshape(env.action_space.shape)

        ob, reward, done, info = env.step(actions_np)[:4]
        if tick % 128 == 0:
            print(f'Reward: {reward.mean():.4f}, Tick: {tick}')
        tick += 1

    if frames:
        import imageio
        os.makedirs('../docker', exist_ok=True)
        imageio.mimsave('../docker/eval.gif', frames, fps=15, loop=0)


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
    if h:
        return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s"
    elif m:
        return f"{b2}{m}{c2}m {b2}{s}{c2}s"
    else:
        return f"{b2}{s}{c2}s"


def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100 * time / uptime - 1e-5)
    return f'{c1}{name}', duration(time), f'{b2}{percent:2d}%'

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
    p.add_row(*fmt_perf('  Misc', profile.eval_misc_time, profile.uptime))
    p.add_row(*fmt_perf('Train', profile.train_time, profile.uptime))
    p.add_row(*fmt_perf('  Forward', profile.train_forward_time, profile.uptime))
    p.add_row(*fmt_perf('  Learn', profile.learn_time, profile.uptime))
    p.add_row(*fmt_perf('  Misc', profile.train_misc_time, profile.uptime))

    l = Table(box=None, expand=True)
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
        try:
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