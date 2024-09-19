import numpy as np
import gymnasium
import pufferlib
from .cy_ocean_cy import COceanCy

class OceanCyEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=2048):
        super().__init__()

        self.num_envs = num_envs
        
        self.buf = pufferlib.namespace(
            image_observations=np.zeros((self.num_envs, 5, 5), dtype=np.float32),
            flat_observations=np.zeros((self.num_envs, 5), dtype=np.int8),
            actions=np.zeros((self.num_envs, 2), dtype=np.uint32),
            rewards=np.zeros((self.num_envs, 1), dtype=np.float32),
            dones=np.zeros((self.num_envs, 1), dtype=np.uint8),
            scores=np.zeros((self.num_envs, 1), dtype=np.int32)
        )

        # Create the observation and action spaces
        self.observation_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.float32),
            'flat': gymnasium.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.int8),
        })
        self.action_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Discrete(2),
            'flat': gymnasium.spaces.Discrete(2),
        })

        self.c_envs = [COceanCy(self.buf.image_observations[i:i+1], 
                                self.buf.flat_observations[i:i+1],
                                self.buf.actions[i:i+1],
                                self.buf.rewards[i:i+1],
                                self.buf.dones[i:i+1],
                                self.buf.scores[i:i+1],
                                num_agents=1)
                       for i in range(self.num_envs)]

    def reset(self, seed=None):
        for env in self.c_envs:
            env.reset()

        observations = {
            'image': self.buf.image_observations,
            'flat': self.buf.flat_observations
        }

        return observations, {}

    def step(self, actions):
        if isinstance(actions, dict):
            self.buf.actions[0][0] = actions['image']
            self.buf.actions[0][1] = actions['flat']
            self.c_envs[0].step()

        terminated = self.buf.dones.copy()
        truncated = np.zeros_like(terminated)
        
        return (
            {'image': self.buf.image_observations, 
             'flat': self.buf.flat_observations},
            self.buf.rewards,
            terminated,
            truncated,
            {}
        )

def make_ocean_cy(num_envs=2048):
    return OceanCyEnv(num_envs=num_envs)
