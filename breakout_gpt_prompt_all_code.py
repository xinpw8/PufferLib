let me provide the all of the files, just so you can understand...

# c_breakout.pyx
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

cimport numpy as cnp
from libc.math cimport pi, sin, cos
from libc.stdlib cimport rand
import numpy as np

cdef:
    int num_agents
    cnp.ndarray observations
    cnp.ndarray actions
    cnp.ndarray rewards
    cnp.ndarray episodic_returns
    cnp.ndarray terminals
    cnp.ndarray truncations
    cnp.ndarray image_sign
    # cnp.ndarray flat_sign

cdef class CBreakout:
    cdef:
        float[:, :, :] observations  # 3D array (num_agents, obs_size, flattened_image_size + flat_size)
        unsigned char[:] dones
        float[:] rewards
        int[:] scores
        float[:] episodic_returns
        int obs_size
        int num_agents
        int[:] timesteps
        int[:] image_sign
        # int[:] flat_sign

    def __init__(self, 
                # float dt, 
                cnp.ndarray observations, 
                cnp.ndarray rewards, 
                cnp.ndarray scores, 
                cnp.ndarray episodic_returns, 
                cnp.ndarray dones, 
                int num_agents,
                int obs_size,
    ):
        cdef int agent_idx
        self.image_sign = np.zeros(num_agents, dtype=np.int32)
        # self.flat_sign = np.zeros(num_agents, dtype=np.int32)

        self.observations = observations
        self.rewards = rewards
        self.scores = scores
        self.episodic_returns = episodic_returns
        self.dones = dones
        self.obs_size = obs_size
        self.num_agents = num_agents

        for agent_idx in range(self.num_agents):
            self.reset(agent_idx)

    cdef void compute_observations(self, agent_idx):
        cdef int j
        cdef int k

        for agent_idx in range(self.num_agents):

            # Generate the 25-element flattened image observation (original image is 5x5)
            for j in range(25):
                self.observations[agent_idx, j] = np.random.randn()

            '''
            continue to perform the following:
            self.observation = {
            'image': np.random.randn(5, 5).astype(np.float32),
            'flat': np.random.randint(-1, 2, (5,), dtype=np.int8),
            }
            self.image_sign = np.sum(self.observation['image']) > 0
            self.flat_sign = np.sum(self.observation['flat']) > 0
            '''

            # # continue to perform the above commented block, but in cython
            # # generate flat observation (5 elements, flat)
            # for j in range(25, 30):
            #     self.observations[agent_idx, j] = np.random.randint(-1, 2, size=(5,), dtype=np.int8)            
            
            # now, calculate the signs
            # image_sum = 0.0
            image_sum = np.sum(self.observations[agent_idx, :25])
            # for j in range(25):
            #     image_sum += self.observations[agent_idx, j]
            if image_sum > 0:
                self.image_sign[agent_idx] = 1
            else:
                self.image_sign[agent_idx] = 0

            # # flat_sum = 0
            # flat_sum = np.sum(self.observations[agent_idx, 25:])
            # # for j in range(25, 30):
            # #     flat_sum += self.observations[agent_idx, j]
            # if flat_sum > 0:
            #     self.flat_sign[agent_idx] = 1
            # else:
            #     self.flat_sign[agent_idx] = 0
            
    cdef void reset(self, int agent_idx):
        # returns image_sign and flat_sign (0 or 1) for each agent
        self.compute_observations(agent_idx)
        self.dones[agent_idx] = 0


        # self.scores[agent_idx] = 0
        # self.dones[agent_idx] = 0


    def step(self, cnp.ndarray[unsigned char, ndim=1] actions):
        cdef int action
        cdef int agent_idx = 0

        self.rewards[:] = 0.0
        
        self.scores[agent_idx] = 0
        self.dones[agent_idx] = 0


        for agent_idx in range(self.num_agents):
            action = actions[agent_idx]
            if self.image_sign[agent_idx] == action:
                self.rewards[agent_idx] += 0.5
            # if self.flat_sign[agent_idx] == action:
            #     self.rewards[agent_idx] += 0.5

        # the rest of the method is as follows:
        # info = dict(score=reward)
        # return self.observation, reward, True, False, info
        # write it in this cython version
        return self.observations, self.rewards, self.dones, self.scores

# breakout.py
import os

import gymnasium
import numpy as np

from raylib import rl, colors

import pufferlib
from pufferlib.environments.ocean import render
from pufferlib.environments.ocean.breakout.c_breakout import CBreakout


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PufferBreakout(pufferlib.PufferEnv):
    def __init__(
        self,
        report_interval: int = 1,
        num_agents: int = 1,
        # render_mode: str = "rgb_array",
    ) -> None:

        self.report_interval = report_interval

        self.c_env: CBreakout | None = None
        self.tick = 0
        self.reward_sum = 0
        self.score_sum = 0
        self.episodic_returns = np.zeros(num_agents, dtype=np.float32)
        self.dones = np.zeros(num_agents, dtype=np.uint8)
        self.scores = np.zeros(num_agents, dtype=np.int32)

        # This block required by advanced PufferLib env spec
        self.obs_size = 5*5 + 5  # image_size + flat_size
        low = 0
        high = 1
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, shape=(5, 5), dtype=np.float32
        )
        # self.observation_space = gymnasium.spaces.Tuple(
        #     gymnasium.spaces.Box(low=low, high=high, shape=(5, 5), dtype=np.float32),
        #     gymnasium.spaces.Box(low=low, high=high, shape=(5,), dtype=np.int8),
        # )
        
        
        self.action_space = gymnasium.spaces.Discrete(1)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = num_agents
        # self.render_mode = render_mode
        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations=np.zeros(
                (self.num_agents, *self.observation_space.shape), dtype=np.float32
            ),
            rewards=np.zeros(self.num_agents, dtype=np.float32),
            terminals=np.zeros(self.num_agents, dtype=bool),
            truncations=np.zeros(self.num_agents, dtype=bool),
            masks=np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint8)

        # if render_mode == "ansi":
        #     self.client = render.AnsiRender()
        # elif render_mode == "rgb_array":
        #     self.client = render.RGBArrayRender()
        # elif render_mode == "human":
        #     self.client = RaylibClient(
        #         self.width,
        #         self.height,
        #         self.num_brick_rows,
        #         self.num_brick_cols,
        #         self.brick_positions,
        #         self.ball_width,
        #         self.ball_height,
        #         self.brick_width,
        #         self.brick_height,
        #         self.fps,
        #     )
        # else:
        #     raise ValueError(f"Invalid render mode: {render_mode}")

    def step(self, actions):
        self.actions[:] = actions

        # if self.render_mode == "human" and self.human_action is not None:
        #     self.actions[0] = self.human_action
        # elif self.render_mode == "human":
        #     self.actions[0] = 0

        self.c_env.step(self.actions)

        info = {}
        self.reward_sum += self.buf.rewards.mean()
        self.score_sum += self.scores.mean()

        if self.tick % self.report_interval == 0:
            info["episodic_return"] = self.episodic_returns.mean()
            info["reward"] = self.reward_sum / self.report_interval

            self.reward_sum = 0
            self.score_sum = 0
            self.tick = 0

        self.tick += 1

        return (
            self.buf.observations,
            self.buf.rewards,
            self.buf.terminals,
            self.buf.truncations,
            info,
        )

    def reset(self, seed=None):
        if self.c_env is None:
            self.c_env = CBreakout(
                # dt=self.dt,
                observations=self.buf.observations,
                rewards=self.buf.rewards,
                scores=self.scores,
                episodic_returns=self.episodic_returns,
                dones=self.dones,
                num_agents=self.num_agents,
                obs_size=self.obs_size,
            )

        return self.buf.observations, {}
    

    def close(self):
        pass

    def _calculate_scores(self, action):
        score = 0
        for agent_idx in range(self.num_agents):
            self.scores[agent_idx] = 0
            if self.image_sign == action['image']:
                reward += 0.5
            # if self.flat_sign == action['flat']:
            #     reward += 0.5
            
            
            
    def render(self):
        pass

# breakout.ini
[base]
package = ocean
env_name = breakout

[train]
total_timesteps = 100_000_000

num_envs = 2
num_workers = 2
env_batch_size = 1
device = cpu
render_mode = human
checkpoint_interval = 50

batch_size = 65536
minibatch_size = 8192

anneal_lr = false
bptt_horizon = 8
clip_coef = 0.10944657790889258
clip_vloss = true
ent_coef = 0.007053685467058537
gae_lambda = 0.9462698603300184
gamma = 0.9348332880396868
learning_rate = 0.00031617638387428646
max_grad_norm = 0.7705206871032715
norm_adv = true
update_epochs = 3
vf_clip_coef = 0.06908694316063399
vf_coef = 0.46530283591543886

[sweep.metric]
goal = maximize
name = environment/episodic_return

# environment.py (this is a factory file, NOT the env code)
import pufferlib.emulation
import pufferlib.postprocess


def make_breakout(num_envs=1, render_mode="rgb_array"):
    from .breakout import breakout

    # return breakout.PufferBreakout(render_mode=render_mode)
    return breakout.PufferBreakout()

...

MAKE_FNS = {
    "breakout": make_breakout,
    "moba": make_moba,
    "foraging": make_foraging,
    "predator_prey": make_predator_prey,
    "group": make_group,
    "puffer": make_puffer,
    "snake": make_snake,
    "continuous": make_continuous,
    "squared": make_squared,
    "bandit": make_bandit,
    "memory": make_memory,
    "password": make_password,
    "stochastic": make_stochastic,
    "multiagent": make_multiagent,
    "spaces": make_spaces,
    "performance": make_performance,
    "performance_empiric": make_performance_empiric,
}


def env_creator(name="squared"):
    if name in MAKE_FNS:
        return MAKE_FNS[name]
    else:
        raise ValueError(f"Invalid environment name: {name}")

# default.ini
[base]
package = None
env_name = None
policy_name = Policy
rnn_name = None

...


# models.py (this is the Policy that is used)
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Default(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.encoder = nn.Linear(np.prod(
            env.single_observation_space.shape), hidden_size)

        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        if self.is_multidiscrete:
            action_nvec = env.single_action_space.nvec
            self.decoder = nn.ModuleList([pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, n), std=0.01) for n in action_nvec])
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        return torch.relu(self.encoder(observations.float())), None

    def decode_actions(self, hidden, lookup, concat=True):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        value = self.value_head(hidden)
        if self.is_multidiscrete:
            actions = [dec(hidden) for dec in self.decoder]
            return actions, value
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = hidden.shape[0]
            return probs, value

        actions = self.decoder(hidden)
        return actions, value


# cleanrl.py (the core of the PPO implementation - gets actions and such - very efficiently written and fast)
from typing import List, Union

import torch
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

import pufferlib.models


# taken from torch.distributions.Categorical
def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

# taken from torch.distributions.Categorical
def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)

def sample_logits(logits: Union[torch.Tensor, List[torch.Tensor]],
        action=None, is_continuous=False):
    is_discrete = isinstance(logits, torch.Tensor)
    if is_continuous:
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)

        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    elif is_discrete:
        normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
        logits = [logits]
    else: # not sure what else it could be
        normalized_logits = [l - l.logsumexp(dim=-1, keepdim=True) for l in logits]


    if action is None:
        action = torch.stack([torch.multinomial(logits_to_probs(l), 1).squeeze() for l in logits])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)
    logprob = torch.stack([log_prob(l, a) for l, a in zip(normalized_logits, action)]).T.sum(1)
    logits_entropy = torch.stack([entropy(l) for l in normalized_logits]).T.sum(1)

    if is_discrete:
        return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)

    return action.T, logprob, logits_entropy


class Policy(torch.nn.Module):
    '''Wrap a non-recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.is_continuous = hasattr(policy, 'is_continuous') and policy.is_continuous

    def get_value(self, x, state=None):
        _, value = self.policy(x)
        return value

    def get_action_and_value(self, x, action=None):
         logits, value = self.policy(x)
         action, logprob, entropy = sample_logits(logits, action, self.is_continuous)
         return action, logprob, entropy, value

    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)



# demo.py (there is a lot more in this file - i just included the basic parts that run the environment files)
import configparser
import argparse
import shutil
import glob
import uuid
import ast
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import clean_pufferl
   
def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args['train']['device'])


...

def train(args, make_env, policy_cls, rnn_cls, wandb, eval_frac=0.1, elos={'model_random.pt': 1000}):
    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    elif args['vec'] == 'native':
        vec = pufferlib.vector.Native
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray).')

    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=args['env'],
        num_envs=args['train']['num_envs'],
        num_workers=args['train']['num_workers'],
        batch_size=args['train']['env_batch_size'],
        zero_copy=args['train']['zero_copy'],
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)

    if env_name == 'moba':
        import torch
        os.makedirs('moba_elo', exist_ok=True)
        torch.save(policy, os.path.join('moba_elo', 'model_random.pt'))

    train_config = pufferlib.namespace(**args['train'], env=env_name,
        exp_id=args['exp_id'] or env_name + '-' + str(uuid.uuid4())[:8])
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    uptime = data.profile.uptime
    steps_evaluated = 0
    steps_to_eval = int(args['train']['total_timesteps'] * eval_frac)
    batch_size = args['train']['batch_size']
    while steps_evaluated < steps_to_eval:
        stats, _ = clean_pufferl.evaluate(data)
        steps_evaluated += batch_size

    clean_pufferl.mean_and_log(data)


# vector.py 
import numpy as np
import time
import psutil

from pufferlib import namespace
from pufferlib.environment import PufferEnv
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.exceptions import APIUsageError
from pufferlib.namespace import Namespace
import pufferlib.spaces
import gymnasium

RESET = 0
STEP = 1
SEND = 2
RECV = 3
CLOSE = 4
MAIN = 5
INFO = 6

def recv_precheck(vecenv):
    if vecenv.flag != RECV:
        raise APIUsageError('Call reset before stepping')

    vecenv.flag = SEND

def send_precheck(vecenv, actions):
    if vecenv.flag != SEND:
        raise APIUsageError('Call (async) reset + recv before sending')

    actions = np.asarray(actions)
    if not vecenv.initialized:
        vecenv.initialized = True
        if not vecenv.action_space.contains(actions):
            raise APIUsageError('Actions do not match action space')

    vecenv.flag = RECV
    return actions

def reset(vecenv, seed=42):
    vecenv.async_reset(seed)
    obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
    return obs, infos

def step(vecenv, actions):
    actions = np.asarray(actions)
    vecenv.send(actions)
    obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
    return obs, rewards, terminals, truncations, infos # include env_ids or no?

def joint_space(space, n):
    if isinstance(space, pufferlib.spaces.Discrete):
        return gymnasium.spaces.MultiDiscrete([space.n] * n)
    elif isinstance(space, pufferlib.spaces.MultiDiscrete):
        return gymnasium.spaces.Box(low=0,
            high=np.repeat(space.nvec[None] - 1, n, axis=0),
            shape=(n, len(space)), dtype=space.dtype)
    elif isinstance(space, pufferlib.spaces.Box):
        return gymnasium.spaces.Box(
            low=np.repeat(space.low[None], n, axis=0),
            high=np.repeat(space.high[None], n, axis=0),
            shape=(n, *space.shape), dtype=space.dtype)
    else:
        raise ValueError(f'Unsupported space: {space}')


class Serial:
    reset = reset
    step = step

    @property
    def num_envs(self):
        return self.agents_per_batch
 
    def __init__(self, env_creators, env_args, env_kwargs, num_envs, **kwargs):
        self.envs = [creator(*args, **kwargs) for (creator, args, kwargs)
            in zip(env_creators, env_args, env_kwargs)]

        self.driver_env = driver = self.envs[0]
        self.emulated = self.driver_env.emulated
        check_envs(self.envs, self.driver_env)
        self.agents_per_env = [env.num_agents for env in self.envs]
        self.agents_per_batch = sum(self.agents_per_env)
        self.num_agents = sum(self.agents_per_env)
        self.single_observation_space = driver.single_observation_space
        self.single_action_space = driver.single_action_space
        self.action_space = joint_space(self.single_action_space, self.agents_per_batch)
        self.observation_space = joint_space(self.single_observation_space, self.agents_per_batch)
        self.agent_ids = np.arange(self.num_agents)
        self.initialized = False
        self.flag = RESET
        self.buf = None

    def _assign_buffers(self, buf):
        '''Envs handle their own data buffers'''
        ptr = 0
        self.buf = buf
        for i, env in enumerate(self.envs):
            end = ptr + self.agents_per_env[i]
            env.buf = namespace(
                observations=buf.observations[ptr:end],
                rewards=buf.rewards[ptr:end],
                terminals=buf.terminals[ptr:end],
                truncations=buf.truncations[ptr:end],
                masks=buf.masks[ptr:end]
            )
            ptr = end

    def async_reset(self, seed=42):
        self.flag = RECV
        seed = make_seeds(seed, len(self.envs))

        if self.buf is None:
            self.buf = namespace(
                observations = np.zeros(
                    (self.agents_per_batch, *self.single_observation_space.shape),
                    dtype=self.single_observation_space.dtype),
                rewards = np.zeros(self.agents_per_batch, dtype=np.float32),
                terminals = np.zeros(self.agents_per_batch, dtype=bool),
                truncations = np.zeros(self.agents_per_batch, dtype=bool),
                masks = np.ones(self.agents_per_batch, dtype=bool),
            )
            self._assign_buffers(self.buf)

        infos = []
        for env, s in zip(self.envs, seed):
            ob, i = env.reset(seed=s)
               
            if i:
                infos.append(i)

        self.infos = infos

    def send(self, actions):
        if not actions.flags.contiguous:
            actions = np.ascontiguousarray(actions)

        actions = send_precheck(self, actions)
        rewards, dones, truncateds, self.infos = [], [], [], []
        ptr = 0
        for idx, env in enumerate(self.envs):
            end = ptr + self.agents_per_env[idx]
            atns = actions[ptr:end]
            if env.done:
                o, i = env.reset()
                buf = self.buf
            else:
                o, r, d, t, i = env.step(atns)

            if i:
                self.infos.append(i)

            ptr = end

    def recv(self):
        recv_precheck(self)
        buf = self.buf
        return (buf.observations, buf.rewards, buf.terminals, buf.truncations,
            self.infos, self.agent_ids, buf.masks)

    def close(self):
        for env in self.envs:
            env.close()

def _worker_process(env_creators, env_args, env_kwargs, num_envs,
        num_workers, worker_idx, send_pipe, recv_pipe, shm):

    envs = Serial(env_creators, env_args, env_kwargs, num_envs)
    obs_shape = envs.single_observation_space.shape
    obs_dtype = envs.single_observation_space.dtype
    atn_shape = envs.single_action_space.shape
    atn_dtype = envs.single_action_space.dtype

    # Environments read and write directly to shared memory
    shape = (num_workers, envs.num_agents)
    atn_arr = np.ndarray((*shape, *atn_shape),
        dtype=atn_dtype, buffer=shm.actions)[worker_idx]
    buf = namespace(
        observations=np.ndarray((*shape, *obs_shape),
            dtype=obs_dtype, buffer=shm.observations)[worker_idx],
        rewards=np.ndarray(shape, dtype=np.float32, buffer=shm.rewards)[worker_idx],
        terminals=np.ndarray(shape, dtype=bool, buffer=shm.terminals)[worker_idx],
        truncations=np.ndarray(shape, dtype=bool, buffer=shm.truncateds)[worker_idx],
        masks=np.ndarray(shape, dtype=bool, buffer=shm.masks)[worker_idx],
    )
    buf.masks[:] = True
    envs._assign_buffers(buf)

    semaphores=np.ndarray(num_workers, dtype=np.uint8, buffer=shm.semaphores)
    start = time.time()
    while True:
        sem = semaphores[worker_idx]
        if sem >= MAIN:
            if time.time() - start > 0.5:
                time.sleep(0.01)
            continue

        start = time.time()
        if sem == RESET:
            seeds = recv_pipe.recv()
            _, infos = envs.reset(seed=seeds)
        elif sem == STEP:
            _, _, _, _, infos = envs.step(atn_arr)
        elif sem == CLOSE:
            print("closing worker", worker_idx)
            send_pipe.send(None)
            break

        if infos:
            semaphores[worker_idx] = INFO
            send_pipe.send(infos)
        else:
            semaphores[worker_idx] = MAIN


def make(env_creator_or_creators, env_args=None, env_kwargs=None, backend=Serial, num_envs=1, **kwargs):
    if num_envs < 1:
        raise APIUsageError('num_envs must be at least 1')
    if num_envs != int(num_envs):
        raise APIUsageError('num_envs must be an integer')

    if 'num_workers' in kwargs:
        num_workers = kwargs['num_workers']
        # TODO: None?
        envs_per_worker = num_envs / num_workers
        if envs_per_worker != int(envs_per_worker):
            raise APIUsageError('num_envs must be divisible by num_workers')

        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
            if batch_size is None:
                batch_size = num_envs

            #if batch_size % envs_per_worker != 0:
            #    raise APIUsageError(
            #        'batch_size must be divisible by (num_envs / num_workers)')
        
 
    if env_args is None:
        env_args = []

    if env_kwargs is None:
        env_kwargs = {}

    if not isinstance(env_creator_or_creators, (list, tuple)):
        env_creators = [env_creator_or_creators] * num_envs
        env_args = [env_args] * num_envs
        env_kwargs = [env_kwargs] * num_envs

    if len(env_creators) != num_envs:
        raise APIUsageError('env_creators must be a list of length num_envs')
    if len(env_args) != num_envs:
        raise APIUsageError('env_args must be a list of length num_envs')
    if len(env_kwargs) != num_envs:
        raise APIUsageError('env_kwargs must be a list of length num_envs')

    for i in range(num_envs):
        if not callable(env_creators[i]):
            raise APIUsageError('env_creators must be a list of callables')
        if not isinstance(env_args[i], (list, tuple)):
            raise APIUsageError('env_args must be a list of lists or tuples')
        if not isinstance(env_kwargs[i], (dict, Namespace)):
            raise APIUsageError('env_kwargs must be a list of dictionaries')

    # Keeps batch size consistent when debugging with Serial backend
    if backend is Serial and 'batch_size' in kwargs:
        num_envs = kwargs['batch_size']

    # TODO: Check num workers is not greater than num envs. This results in
    # different Serial vs Multiprocessing behavior

    # Sanity check args
    for k in kwargs:
        if k not in ['num_workers', 'batch_size', 'zero_copy','backend']:
            raise APIUsageError(f'Invalid argument: {k}')

    # TODO: First step action space check
    
    return backend(env_creators, env_args, env_kwargs, num_envs, **kwargs)

def make_seeds(seed, num_envs):
    if isinstance(seed, int):
        return [seed + i for i in range(num_envs)]

    err = f'seed {seed} must be an integer or a list of integers'
    if isinstance(seed, (list, tuple)):
        if len(seed) != num_envs:
            raise APIUsageError(err)

        return seed

    raise APIUsageError(err)

def check_envs(envs, driver):
    valid = (PufferEnv, GymnasiumPufferEnv, PettingZooPufferEnv)
    if not isinstance(driver, valid):
        raise APIUsageError(f'env_creator must be {valid}')

    driver_obs = driver.single_observation_space
    driver_atn = driver.single_action_space
    for env in envs:
        if not isinstance(env, valid):
            raise APIUsageError(f'env_creators must be {valid}')
        obs_space = env.single_observation_space
        if obs_space != driver_obs:
            raise APIUsageError(f'\n{obs_space}\n{driver_obs} obs space mismatch')
        atn_space = env.single_action_space
        if atn_space != driver_atn:
            raise APIUsageError(f'\n{atn_space}\n{driver_atn} atn space mismatch')


# clean_pufferl.py

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

# Fast Cython GAE implementation
#import pyximport
#pyximport.install(setup_args={"include_dirs": np.get_include()})
from c_gae import compute_gae


def create(config, vecenv, policy, optimizer=None, wandb=None):
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

    lstm = policy.lstm if hasattr(policy, 'lstm') else None
    experience = Experience(config.batch_size, config.bptt_horizon,
        config.minibatch_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
        config.cpu_offload, config.device, lstm, total_agents)

    uncompiled_policy = policy

    if config.compile:
        policy = torch.compile(policy, mode=config.compile_mode)

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
        global_step=0,
        epoch=0,
        stats=defaultdict(list),
        msg=msg,
        last_log_time=0,
        utilization=utilization,
    )

@pufferlib.utils.profile
def evaluate(data):
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

            o = torch.as_tensor(o)
            o_device = o.to(config.device)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)

        with profile.eval_forward, torch.no_grad():
            # TODO: In place-update should be faster. Leaking 7% speed max
            # Also should be using a cuda tensor to index
            if lstm_h is not None:
                h = lstm_h[:, env_id]
                c = lstm_c[:, env_id]
                actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c
            else:
                actions, logprob, _, value = policy(o_device)

            if config.device == 'cuda':
                torch.cuda.synchronize()

        with profile.eval_misc:
            value = value.flatten()
            actions = actions.cpu().numpy()
            mask = torch.as_tensor(mask)# * policy.mask)
            o = o if config.cpu_offload else o_device
            experience.store(o, value, actions, logprob, r, d, env_id, mask)

            for i in info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    infos[k].append(v)

        with profile.env:
            data.vecenv.send(actions)

    with profile.eval_misc:
        # Moves into models... maybe. Definitely moves.
        # You could also just return infos and have it in demo
        if 'pokemon_exploration_map' in infos:
            for pmap in infos['pokemon_exploration_map']:
                if not hasattr(data, 'pokemon_map'):
                    import pokemon_red_eval
                    data.map_updater = pokemon_red_eval.map_updater()
                    data.pokemon_map = pmap

                data.pokemon_map = np.maximum(data.pokemon_map, pmap)

            if len(infos['pokemon_exploration_map']) > 0:
                rendered = data.map_updater(data.pokemon_map)
                data.stats['Media/exploration_map'] = data.wandb.Image(rendered)

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

    # TODO: Better way to enable multiple collects
    data.experience.ptr = 0
    data.experience.step = 0
    return data.stats, infos

@pufferlib.utils.profile
def train(data):
    config, profile, experience = data.config, data.profile, data.experience
    data.losses = make_losses()
    losses = data.losses

    with profile.train_misc:
        idxs = experience.sort_training_data()
        dones_np = experience.dones_np[idxs]
        values_np = experience.values_np[idxs]
        rewards_np = experience.rewards_np[idxs]
        # TODO: bootstrap between segment bounds
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
                val = experience.b_values[mb]
                adv = experience.b_advantages[mb]
                ret = experience.b_returns[mb]

            with profile.train_forward:
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, newvalue, lstm_state = data.policy(
                        obs, state=lstm_state, action=atn)
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = data.policy(
                        obs.reshape(-1, *data.vecenv.single_observation_space.shape),
                        action=atn,
                    )

                if config.device == 'cuda':
                    torch.cuda.synchronize()

            with profile.train_misc:
                logratio = newlogprob - log_probs.reshape(-1)
                ratio = logratio.exp()

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
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
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

        y_pred = experience.values_np
        y_true = experience.returns_np
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        losses.explained_variance = explained_var
        data.epoch += 1

        done_training = data.global_step >= config.total_timesteps
        # TODO: beter way to get episode return update without clogging dashboard
        # TODO: make this appear faster
        if profile.update(data):
            mean_and_log(data)
            print_dashboard(config.env, data.utilization, data.global_step, data.epoch,
                profile, data.losses, data.stats, data.msg)
            data.stats = defaultdict(list)

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

        data.stats[k] = v

    if data.wandb is None:
        return

    data.last_log_time = time.time()
    data.wandb.log({
        '0verview/SPS': data.profile.SPS,
        '0verview/agent_steps': data.global_step,
        '0verview/epoch': data.epoch,
        '0verview/learning_rate': data.optimizer.param_groups[0]["lr"],
        **{f'environment/{k}': v for k, v in data.stats.items()},
        **{f'losses/{k}': v for k, v in data.losses.items()},
        **{f'performance/{k}': v for k, v in data.profile},
    })


...


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
    def __init__(self, batch_size, bptt_horizon, minibatch_size, obs_shape, obs_dtype, atn_shape, atn_dtype,
                 cpu_offload=False, device='cuda', lstm=None, lstm_total_agents=0):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == 'cuda' and cpu_offload
        obs_device = device if not pin else 'cpu'
        self.obs=torch.zeros(batch_size, *obs_shape, dtype=obs_dtype,
            pin_memory=pin, device=device if not pin else 'cpu')
        self.actions=torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, pin_memory=pin)
        self.logprobs=torch.zeros(batch_size, pin_memory=pin)
        self.rewards=torch.zeros(batch_size, pin_memory=pin)
        self.dones=torch.zeros(batch_size, pin_memory=pin)
        self.truncateds=torch.zeros(batch_size, pin_memory=pin)
        self.values=torch.zeros(batch_size, pin_memory=pin)

        #self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

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

    def store(self, obs, value, action, logprob, reward, done, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[:self.batch_size - ptr]
        end = ptr + len(indices)
 
        self.obs[ptr:end] = obs.to(self.obs.device)[indices]
        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(sorted(
            range(len(self.sort_keys)), key=self.sort_keys.__getitem__))
        self.b_idxs_obs = torch.as_tensor(idxs.reshape(
                self.minibatch_rows, self.num_minibatches, self.bptt_horizon
            ).transpose(1,0,-1)).to(self.obs.device).long()
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(
            self.num_minibatches, self.minibatch_size)
        self.sort_keys = []
        return idxs

    def flatten_batch(self, advantages_np):
        advantages = torch.as_tensor(advantages_np).to(self.device)
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_advantages = advantages.reshape(self.minibatch_rows,
            self.num_minibatches, self.bptt_horizon).transpose(0, 1).reshape(
            self.num_minibatches, self.minibatch_size)
        self.returns_np = advantages_np + self.values_np
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values