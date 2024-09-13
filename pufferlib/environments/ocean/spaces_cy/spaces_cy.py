# import os

# import gymnasium
# import numpy as np

# from raylib import rl, colors

# import pufferlib
# from pufferlib.environments.ocean import render
# from pufferlib.environments.ocean.spaces_cy.c_spaces_cy import CSpacesCy

# from pufferlib.emulation import GymnasiumPufferEnv, emulate_observation_space, emulate_action_space
# from pufferlib.namespace import Namespace
# from pufferlib import namespace
# from pufferlib.namespace import namespace, dataclass, Namespace

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)


# class SpacesCy(pufferlib.PufferEnv):
#     def __init__(
#         self,
#         report_interval: int = 1,
#         num_agents: int = 1,
#     ) -> None:

#         self.report_interval = report_interval

#         self.c_env: CSpacesCy | None = None
#         self.tick = 0
#         self.reward_sum = 0
#         self.score_sum = 0
#         self.episodic_returns = np.zeros(num_agents, dtype=np.float32)
#         self.dones = np.zeros(num_agents, dtype=np.uint8)
#         self.scores = np.zeros(num_agents, dtype=np.int32)
#         self.num_agents = num_agents
#         self.emulated = None
#         self.done = False
        
#         # This block required by advanced PufferLib env spec
#         self.obs_size = 5*5 + 5  # image_size + flat_size
#         low = 0
#         high = 1
        
#         # randomly put something in here just to init this
#         self.buf = pufferlib.namespace(obs_size=self.obs_size)
#         # init these because they keep crashing it
#         self.buf.buf_image_observations = np.zeros((self.num_agents, 5, 5), dtype=np.float32)
#         self.buf.buf_flat_observations = np.zeros((self.num_agents, 5), dtype=np.int8)
        
#         # # emulated observation space
#         # self.observation_space = gymnasium.spaces.Box(
#         #     low=0, high=255, shape=(self.obs_size,), dtype=np.uint8
#         # )
        
        
#         # self.observation_space = gymnasium.spaces.Box(
#         #     low=low, high=high, shape=(5, 5), dtype=np.float32
#         # )
#         # self.action_space = gymnasium.spaces.Discrete(1)
#         # self.action_space = gymnasium.spaces.Discrete(2)
                        
#         # self.observation_space = gymnasium.spaces.Dict({
#         #     'image': gymnasium.spaces.Box(low=low, high=high, shape=(5, 5), dtype=np.float32),
#         #     'flat': gymnasium.spaces.Box(low=low, high=high, shape=(5,), dtype=np.int8),
#         # })      
#         # self.action_space = gymnasium.spaces.MultiDiscrete([2, 2])
        

#         # Define your raw observation and action spaces
#         flat_space = gymnasium.spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8)
#         image_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5, 5), dtype=np.float32)

#         raw_observation_space = gymnasium.spaces.Dict({
#             'flat': flat_space,
#             'image': image_space,
#         })
#         raw_action_space = gymnasium.spaces.Discrete(2)

#         # Emulate observation and action spaces
#         self.observation_space, self.obs_dtype = emulate_observation_space(raw_observation_space)
#         self.action_space, self.atn_dtype = emulate_action_space(raw_action_space)      
        
          
#         # Initialize separate buffers for image and flat observations
#         self.buf_image_observations = np.zeros((self.num_agents, 5, 5), dtype=np.float32)
#         self.buf_flat_observations = np.zeros((self.num_agents, 5), dtype=np.int8)    
        
#         self.single_observation_space = self.observation_space
#         self.single_action_space = self.action_space

#         # self.observation_space: Dict('flat': Box(0, 1, (5,), int8), 'image': Box(0.0, 1.0, (5, 5), float32))
#         # observation space is a Dict with two keys: 'flat' and 'image'
#         # 'flat' is a Box(0, 1, (5,), int8)
#         # 'image' is a Box(0.0, 1.0, (5, 5), float32)

#         self.buf = pufferlib.namespace(
#             buf_image_observations=self.buf_image_observations,
#             buf_flat_observations=self.buf_flat_observations, # added for handling of Dict observation space
#             rewards=np.zeros(self.num_agents, dtype=np.float32),
#             terminals=np.zeros(self.num_agents, dtype=bool),
#             truncations=np.zeros(self.num_agents, dtype=bool),
#             masks=np.ones(self.num_agents, dtype=bool),
#         )
#         self.actions = np.zeros(self.num_agents, dtype=np.uint8)
        
#         print(f"self.buf contents: {vars(self.buf)}")

#     def step(self, actions):
#         self.actions[:] = actions

#         self.c_env.step(self.actions)

#         info = {}
#         self.reward_sum += self.buf.rewards.mean()
#         self.score_sum += self.scores.mean()

#         if self.tick % self.report_interval == 0:
#             info["episodic_return"] = self.episodic_returns.mean()
#             info["reward"] = self.reward_sum / self.report_interval

#             self.reward_sum = 0
#             self.score_sum = 0
#             self.tick = 0

#         self.tick += 1

#         # Concatenate the 'image' and 'flat' observations into a single array
#         concatenated_observations = np.concatenate([
#             self.buf.buf_image_observations.reshape(self.num_agents, -1),
#             self.buf.buf_flat_observations
#         ], axis=1)


#         return (
#             # self.buf.observations,
#             concatenated_observations, # so, self.buf isn't used here? is that right??
#             self.buf.rewards,
#             self.buf.terminals,
#             self.buf.truncations,
#             info,
#         )

#     def reset(self, seed=None):
#         if self.c_env is None:
#             self.c_env = CSpacesCy(
#                 buf_image_observations=self.buf.buf_image_observations,
#                 buf_flat_observations=self.buf.buf_flat_observations,
#                 # observations=self.buf.observations,
#                 rewards=self.buf.rewards,
#                 scores=self.scores,
#                 episodic_returns=self.episodic_returns,
#                 dones=self.dones,
#                 num_agents=self.num_agents,
#                 obs_size=self.obs_size,
#             )


#         concatenated_observations = np.concatenate([
#             self.buf.buf_image_observations.reshape(self.num_agents, -1),
#             self.buf.buf_flat_observations
#         ], axis=1)

#         # hopefully not using self.buf.observations here is correct as well...
#         return concatenated_observations, {}

        # return self.buf.observations, {}
    
# # 2nd version - not working
# import os
# import gymnasium
# import numpy as np
# from raylib import rl, colors
# import pufferlib
# from pufferlib.environments.ocean import render
# from pufferlib.environments.ocean.spaces_cy.c_spaces_cy import CSpacesCy
# from pufferlib.emulation import GymnasiumPufferEnv, emulate_observation_space, emulate_action_space
# from pufferlib.namespace import Namespace
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# class SpacesCy(pufferlib.PufferEnv):
#     def __init__(self, report_interval: int = 1, num_agents: int = 1) -> None:
#         self.report_interval = report_interval
#         self.c_env: CSpacesCy | None = None
#         self.tick = 0
#         self.reward_sum = 0
#         self.score_sum = 0
#         self.episodic_returns = np.zeros(num_agents, dtype=np.float32)
#         self.dones = np.zeros(num_agents, dtype=np.uint8)
#         self.scores = np.zeros(num_agents, dtype=np.int32)
#         self.num_agents = num_agents
#         self.emulated = None
#         self.done = False
        
#         # This block required by advanced PufferLib env spec
#         self.obs_size = 5*5 + 5  # image_size + flat_size
#         low = 0
#         high = 1
        
#         # Initialize separate buffers for image and flat observations
#         self.buf_image_observations = np.zeros((self.num_agents, 5, 5), dtype=np.float32)
#         self.buf_flat_observations = np.zeros((self.num_agents, 5), dtype=np.int8)    
        
#         # Define your raw observation and action spaces
#         flat_space = gymnasium.spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8)
#         image_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5, 5), dtype=np.float32)
#         raw_observation_space = gymnasium.spaces.Dict({
#             'flat': flat_space,
#             'image': image_space,
#         })
#         raw_action_space = gymnasium.spaces.Discrete(2)

#         # Emulate observation and action spaces
#         self.observation_space, self.obs_dtype = emulate_observation_space(raw_observation_space)
#         self.action_space, self.atn_dtype = emulate_action_space(raw_action_space)      
        
#         self.single_observation_space = self.observation_space
#         self.single_action_space = self.action_space

#         # # Initialize the buf object with the necessary attributes
#         # self.buf = Namespace(
#         #     buf_image_observations=self.buf_image_observations,
#         #     buf_flat_observations=self.buf_flat_observations,
#         #     rewards=np.zeros(self.num_agents, dtype=np.float32),
#         #     terminals=np.zeros(self.num_agents, dtype=bool),
#         #     truncations=np.zeros(self.num_agents, dtype=bool),
#         #     masks=np.ones(self.num_agents, dtype=bool),
#         # )
#         # self.actions = np.zeros(self.num_agents, dtype=np.uint8)
            
#         self.buf = {
#             'buf_image_observations': self.buf_image_observations,
#             'buf_flat_observations': self.buf_flat_observations,
#             'rewards': np.zeros(self.num_agents, dtype=np.float32),
#             'terminals': np.zeros(self.num_agents, dtype=bool),
#             'truncations': np.zeros(self.num_agents, dtype=bool),
#             'masks': np.ones(self.num_agents, dtype=bool),
#     }        
#         print(f"self.buf contents: {self.buf}")




#     def step(self, actions):
#         self.actions[:] = actions
#         self.c_env.step(self.actions)
#         info = {}
#         self.reward_sum += self.buf.rewards.mean()
#         self.score_sum += self.scores.mean()

#         if self.tick % self.report_interval == 0:
#             info["episodic_return"] = self.episodic_returns.mean()
#             info["reward"] = self.reward_sum / self.report_interval
#             self.reward_sum = 0
#             self.score_sum = 0
#             self.tick = 0

#         self.tick += 1

#         # Concatenate the 'image' and 'flat' observations into a single array
#         concatenated_observations = np.concatenate([
#             self.buf.buf_image_observations.reshape(self.num_agents, -1),
#             self.buf.buf_flat_observations
#         ], axis=1)

#         return (
#             concatenated_observations,
#             self.buf.rewards,
#             self.buf.terminals,
#             self.buf.truncations,
#             info,
#         )

#     def reset(self, seed=None):
#         if self.c_env is None:
#             self.c_env = CSpacesCy(
#                 buf_image_observations=self.buf.buf_image_observations,
#                 buf_flat_observations=self.buf.buf_flat_observations,
#                 rewards=self.buf.rewards,
#                 scores=self.scores,
#                 episodic_returns=self.episodic_returns,
#                 dones=self.dones,
#                 num_agents=self.num_agents,
#                 obs_size=self.obs_size,
#             )

#         concatenated_observations = np.concatenate([
#             self.buf.buf_image_observations.reshape(self.num_agents, -1),
#             self.buf.buf_flat_observations
#         ], axis=1)

#         return concatenated_observations, {}






#     def close(self):
#         pass

#     def _calculate_scores(self, action):
#         score = 0
#         for agent_idx in range(self.num_agents):
#             self.scores[agent_idx] = 0
#             if self.image_sign == action['image']:
#                 reward += 0.5
#             if self.flat_sign == action['flat']:
#                 reward += 0.5
                        
#     def render(self):
#         pass

# def test_performance(timeout=20, atn_cache=1024, num_envs=400):
#     tick = 0

#     import time

#     start = time.time()
#     while time.time() - start < timeout:
#         atns = actions[tick % atn_cache]
#         env.step(atns)
#         tick += 1

#     print(f"SPS: %f", num_envs * tick / (time.time() - start))


# if __name__ == "__main__":
#     # Run with c profile
#     from cProfile import run

#     num_envs = 100
#     env = SpacesCy(num_agents=num_envs)
#     env.reset()
#     actions = np.random.randint(0, 9, (1024, num_envs))
#     test_performance(20, 1024, num_envs)
#     # exit(0)

#     run("test_performance(20)", "stats.profile")
#     import pstats
#     from pstats import SortKey

#     p = pstats.Stats("stats.profile")
#     p.sort_stats(SortKey.TIME).print_stats(25)
#     exit(0)

#     # test_performance(10)



import os
import gymnasium
import numpy as np
from raylib import rl, colors
import pufferlib
from pufferlib.environments.ocean import render
from pufferlib.environments.ocean.spaces_cy.c_spaces_cy import CSpacesCy
from pufferlib.emulation import GymnasiumPufferEnv, emulate_observation_space, emulate_action_space, make_buffer
from pufferlib.namespace import namespace
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



class SpacesCy(pufferlib.PufferEnv):
    def __init__(self, report_interval: int = 1, num_agents: int = 1) -> None:
        self.report_interval = report_interval
        self.c_env: CSpacesCy | None = None
        self.tick = 0
        self.reward_sum = 0
        self.score_sum = 0
        self.episodic_returns = np.zeros(num_agents, dtype=np.float32)
        self.dones = np.zeros(num_agents, dtype=np.uint8)
        self.scores = np.zeros(num_agents, dtype=np.int32)
        self.num_agents = num_agents
        self.emulated = None
        self.done = False

        # Initialize the actions array to store agent actions
        self.actions = np.zeros((self.num_agents, 2), dtype=np.uint8)

        # This block required by advanced PufferLib env spec
        self.obs_size = 5*5 + 5  # image_size + flat_size

        # Initialize separate buffers for image and flat observations
        self.buffero_image_observations = np.zeros((self.num_agents, 5, 5), dtype=np.float32)
        self.buffero_flat_observations = np.zeros((self.num_agents, 5), dtype=np.int8)

        # Define your raw observation and action spaces
        flat_space = gymnasium.spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8)
        image_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(5, 5), dtype=np.float32)
        raw_observation_space = gymnasium.spaces.Dict({
            'flat': flat_space,
            'image': image_space,
        })
        raw_action_space = gymnasium.spaces.MultiDiscrete([2, 2])

        # Emulate observation and action spaces
        self.observation_space, self.obs_dtype = emulate_observation_space(raw_observation_space)
        self.action_space, self.atn_dtype = emulate_action_space(raw_action_space)
        
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        # self.num_agents = 1

        self.is_obs_emulated = self.single_observation_space is not self.observation_space
        self.is_atn_emulated = self.single_action_space is not self.action_space
        self.emulated = pufferlib.namespace(
            observation_dtype = self.observation_space.dtype,
            emulated_observation_dtype = self.obs_dtype,
        )
        
        self.obs, self.obs_struct = make_buffer(
            self.single_observation_space.dtype, self.obs_dtype)
        
        
        # Initialize buf using a dictionary directly
        self.buffero = {
            'buffero_image_observations': self.buffero_image_observations,
            'buffero_flat_observations': self.buffero_flat_observations,
            'observations': np.zeros((self.num_agents, self.obs_size), dtype=np.uint8),
            'rewards': np.zeros(self.num_agents, dtype=np.float32),
            'terminals': np.zeros(self.num_agents, dtype=bool),
            'truncations': np.zeros(self.num_agents, dtype=bool),
            'masks': np.ones(self.num_agents, dtype=bool),
        }

        print(f"self.buffero contents after initialization: {self.buffero}")
        print(f'self.emulated: {self.emulated}, self.obs: {self.obs}, self.obs_struct: {self.obs_struct}')
        
        # Print the content of self.buffero to verify that it's initialized properly
        # print(f"self.buffero contents after initialization: {self.buffero}")

    def reset(self, seed=None):
        # Check if self.buffero has the expected structure before calling CSpacesCy
        # print(f"self.buffero contents before creating CSpacesCy: {self.buffero}")
        
        if self.c_env is None:
            self.c_env = CSpacesCy(
                buffero_image_observations=self.buffero['buffero_image_observations'],
                buffero_flat_observations=self.buffero['buffero_flat_observations'],
                rewards=self.buffero['rewards'],
                scores=self.scores,
                episodic_returns=self.episodic_returns,
                dones=self.dones,
                num_agents=self.num_agents,
                emulated=self.emulated,
            )

        concatenated_observations = np.concatenate([
            self.buffero['buffero_image_observations'].reshape(self.num_agents, -1),
            self.buffero['buffero_flat_observations']
        ], axis=1)

        self.buffero['observations'][:] = concatenated_observations  # Store concatenated obs in observations

        return concatenated_observations, {}

    def step(self, actions):
        self.actions[:, :] = actions
        # print(f'self.actions: {self.actions}')
        self.c_env.step(self.actions)

        info = {}
        self.reward_sum += self.buffero['rewards'].mean()
        self.score_sum += self.scores.mean()

        if self.tick % self.report_interval == 0:
            info["episodic_return"] = self.episodic_returns.mean()
            info["reward"] = self.reward_sum / self.report_interval
            self.reward_sum = 0
            self.score_sum = 0
            self.tick = 0

        self.tick += 1

        # Concatenate the 'image' and 'flat' observations into a single array
        concatenated_observations = np.concatenate([
            self.buffero['buffero_image_observations'].reshape(self.num_agents, -1),
            self.buffero['buffero_flat_observations']
        ], axis=1)

        self.buffero['observations'][:] = concatenated_observations  # Store concatenated obs in observations

        return (
            concatenated_observations,
            self.buffero['rewards'],
            self.buffero['terminals'],
            self.buffero['truncations'],
            info,
        )


    def close(self):
        pass

    def _calculate_scores(self, action):
        score = 0
        for agent_idx in range(self.num_agents):
            self.scores[agent_idx] = 0
            if self.image_sign == action['image']:
                score += 0.5
            if self.flat_sign == action['flat']:
                score += 0.5

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
    from cProfile import run
    num_envs = 100
    env = SpacesCy(num_agents=num_envs)
    env.reset()
    actions = np.random.randint(0, 9, (1024, num_envs))
    test_performance(20, 1024, num_envs)
    run("test_performance(20)", "stats.profile")
    import pstats
    from pstats import SortKey
    p = pstats.Stats("stats.profile")
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)