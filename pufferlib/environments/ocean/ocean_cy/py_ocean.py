# import numpy as np
# import gymnasium
# import pufferlib
# from .cy_ocean_cy import COceanCy

# class OceanCyEnv(pufferlib.PufferEnv):
#     def __init__(self, num_envs=1):
#         super().__init__()

#         self.num_envs = num_envs

#         # Create the buffers for the environment
#         self.image_observations = np.zeros((self.num_envs, 5, 5), dtype=np.float32)
#         self.flat_observations = np.zeros((self.num_envs, 5), dtype=np.int8)
#         self.actions = np.zeros((self.num_envs, 2), dtype=np.uint32)  # Actions now 2D
#         self.rewards = np.zeros((self.num_envs, 1), dtype=np.float32)  # 2D arrays
#         self.dones = np.zeros((self.num_envs, 1), dtype=np.uint8)  # 2D arrays
#         self.scores = np.zeros((self.num_envs, 1), dtype=np.int32)  # 2D arrays

#         # Create the observation and action spaces
#         self.observation_space = gymnasium.spaces.Dict({
#             'image': gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32),
#             'flat': gymnasium.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.int8),
#         })
#         self.action_space = gymnasium.spaces.Dict({
#             'image': gymnasium.spaces.Discrete(2),
#             'flat': gymnasium.spaces.Discrete(2),
#         })

        
#         # Initialize Cython environment for each environment
#         self.c_envs = [COceanCy(self.image_observations[i:i+1],  # 3D array (num_envs, 5, 5)
#                                 self.flat_observations[i:i+1],        # 2D array (num_envs, 5)
#                                 self.actions[i:i+1],              # 2D array (num_envs, 2)
#                                 self.rewards[i:i+1],              # 2D array (num_envs, 1)
#                                 self.dones[i:i+1],                # 2D array (num_envs, 1)
#                                 self.scores[i:i+1],               # 2D array (num_envs, 1)
#                                 num_agents=1,
#                                 )
#                     for i in range(self.num_envs)]


#     def reset(self, seed=None):      
#         if seed is not None:
#             np.random.seed(seed)
#             for i, env in enumerate(self.c_envs):
#                 env.set_seed(seed + i)
        
#         for env in self.c_envs:
#             env.reset()

#         # Log the initial observation
#         observation, info = self._get_observation(), {}
#         # print(f'Initial observation after reset (with test values): {observation}')
#         return observation, info


#     def step(self, actions):
#         # Check if actions is a single dictionary (single-agent case)
#         if isinstance(actions, dict):
#             # Update the actions directly for the single agent
#             self.actions[0][0] = actions['image']
#             self.actions[0][1] = actions['flat']
#             self.c_envs[0].step()  # Step the single agent environment
#         else:
#             # Multi-agent case (if you have multiple agents in the future)
#             for i, env in enumerate(self.c_envs):
#                 self.actions[i][0] = actions[i]['image']
#                 self.actions[i][1] = actions[i]['flat']
#                 env.step()

#         # Return the observations, rewards, dones, etc.
#         return self._get_observation(), self.rewards.copy(), self.dones.copy(), {}, {}


#     def _get_observation(self):
#         # Log the current observations to check if they are non-zero
#         # print(f'Current image observations: {self.image_observations}')
#         # print(f'Current flat observations: {self.flat_observations}')
        
#         # Return the current observations from the buffers
#         return {
#             'image': self.image_observations.copy(),
#             'flat': self.flat_observations.copy(),
#         }

# def make_ocean_cy(num_envs=1):
#     return OceanCyEnv(num_envs=num_envs)



### version below uses buffer and pufferlib namespace
import numpy as np
import gymnasium
import pufferlib
from .cy_ocean_cy import COceanCy

class OceanCyEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=1):
        super().__init__()

        self.num_envs = num_envs

        # Initialize the buffer with necessary fields
        self.buf = pufferlib.namespace(
            image_observations=np.zeros((self.num_envs, 5, 5), dtype=np.float32),
            flat_observations=np.zeros((self.num_envs, 5), dtype=np.float32),
            actions=np.zeros((self.num_envs, 2), dtype=np.uint32),
            rewards=np.zeros((self.num_envs, 1), dtype=np.float32),
            dones=np.zeros((self.num_envs, 1), dtype=np.uint8),
            scores=np.zeros((self.num_envs, 1), dtype=np.int32)
        )

        # Create the observation and action spaces
        self.observation_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.float32),
            'flat': gymnasium.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32),
        })
        self.action_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Discrete(2),
            'flat': gymnasium.spaces.Discrete(2),
        })

        # Initialize Cython environment for each environment
        self.c_envs = [COceanCy(self.buf.image_observations[i:i+1],  # Use buffer for observations
                                self.buf.flat_observations[i:i+1],
                                self.buf.actions[i:i+1],
                                self.buf.rewards[i:i+1],
                                self.buf.dones[i:i+1],
                                self.buf.scores[i:i+1],
                                num_agents=1)
                       for i in range(self.num_envs)]

    def reset(self, seed=None):
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            for i, env in enumerate(self.c_envs):
                env.set_seed(seed + i)

        # Reset each Cython environment
        for env in self.c_envs:
            env.reset()

        # Combine the observations into a single dictionary
        observations = {
            'image': self.buf.image_observations,
            'flat': self.buf.flat_observations
        }

        # Return the observations and an empty info dictionary
        return observations, {}

    def step(self, actions):
        # Check if actions is a single dictionary (single-agent case)
        if isinstance(actions, dict):
            self.buf.actions[0][0] = actions['image']
            self.buf.actions[0][1] = actions['flat']
            self.c_envs[0].step()
        else:
            # Multi-agent case
            for i, env in enumerate(self.c_envs):
                self.buf.actions[i][0] = actions[i]['image']
                self.buf.actions[i][1] = actions[i]['flat']
                env.step()

        # Assuming your environment terminates after a single step
        terminated = self.buf.dones.copy()
        truncated = np.zeros_like(terminated)  # Set truncated to False (or relevant flag)

        # Return the observations, rewards, termination, truncation, and info
        return (
            {'image': self.buf.image_observations, 'flat': self.buf.flat_observations},  # observations
            self.buf.rewards,  # rewards
            terminated,        # terminated
            truncated,         # truncated (added)
            {}                 # info
        )

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        if seed is not None:
            np.random.seed(seed)
            for i, env in enumerate(self.c_envs):
                env.set_seed(seed + i)

def make_ocean_cy(num_envs=1):
    return OceanCyEnv(num_envs=num_envs)
