import os

import gymnasium
import numpy as np

from raylib import rl, colors

import pufferlib
from pufferlib.environments.ocean import render
from pufferlib.environments.ocean.spaces_cy.c_spaces_cy import CSpacesCy


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SpacesCy(pufferlib.PufferEnv):
    def __init__(
        self,
        report_interval: int = 1,
        num_agents: int = 1,
    ) -> None:

        self.report_interval = report_interval

        self.c_env: CSpacesCy | None = None
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
        
        # self.observation_space = gymnasium.spaces.Box(
        #     low=low, high=high, shape=(5, 5), dtype=np.float32
        # )
        # self.action_space = gymnasium.spaces.Discrete(1)
                
        self.observation_space = gymnasium.spaces.Dict(
            gymnasium.spaces.Box(low=low, high=high, shape=(5, 5), dtype=np.float32),
            gymnasium.spaces.Box(low=low, high=high, shape=(5,), dtype=np.int8),
        )      
        self.action_space = gymnasium.spaces.MultiDiscrete([2, 2])
        
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = num_agents
        self.emulated = None
        self.done = False
        buf_observations = np.ascontiguousarray(np.zeros((self.num_agents, *self.observation_space.shape), dtype=np.float32))
        self.buf = pufferlib.namespace(buf_observations=buf_observations,
            rewards=np.zeros(self.num_agents, dtype=np.float32),
            terminals=np.zeros(self.num_agents, dtype=bool),
            truncations=np.zeros(self.num_agents, dtype=bool),
            masks=np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint8)

    def step(self, actions):
        self.actions[:] = actions

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
            self.c_env = CSpacesCy(
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
            if self.flat_sign == action['flat']:
                reward += 0.5
                        
    def render(self):
        pass

def test_performance(timeout=20, atn_cache=1024, num_envs=400):
    tick = 0

    import time

    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f"SPS: %f", num_envs * tick / (time.time() - start))


if __name__ == "__main__":
    # Run with c profile
    from cProfile import run

    num_envs = 100
    env = SpacesCy(num_agents=num_envs)
    env.reset()
    actions = np.random.randint(0, 9, (1024, num_envs))
    test_performance(20, 1024, num_envs)
    # exit(0)

    run("test_performance(20)", "stats.profile")
    import pstats
    from pstats import SortKey

    p = pstats.Stats("stats.profile")
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)

    # test_performance(10)
