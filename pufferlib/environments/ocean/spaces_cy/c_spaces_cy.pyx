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

cdef class CSpacesCy:
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
        cdef float[:, :] obs = self.observations[agent_idx, :, :]  # Use a 2D memoryview slice

        # Use a loop to manually assign the random values
        cdef int i, j
        cdef float value
        for i in range(5):
            for j in range(5):
                value = np.random.randn()  # Generate a random float
                obs[i, j] = value  # Assign it explicitly

        # Calculate the image sign (sum over the entire 5x5 observation)
        image_sum = np.sum(obs)
        self.image_sign[agent_idx] = 1 if image_sum > 0 else 0


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


        # # flat_sum = 0
        # flat_sum = np.sum(self.observations[agent_idx, 25:])
        # # for j in range(25, 30):
        # #     flat_sum += self.observations[agent_idx, j]
        # if flat_sum > 0:
        #     self.flat_sign[agent_idx] = 1
        # else:
        #     self.flat_sign[agent_idx] = 0

        for i in range(25, 30):
            value = np.random.randint(-1, 2)
            obs[i] = value
        
        # Calculate the flat sign (sum over the entire 5-element flat observation)
        flat_sum = np.sum(obs[25:])
        self.flat_sign[agent_idx] = 1 if flat_sum > 0 else 0
            
    cdef void reset(self, int agent_idx):
        # returns image_sign and flat_sign (0 or 1) for each agent
        self.compute_observations(agent_idx)
        self.dones[agent_idx] = 0

        # self.scores[agent_idx] = 0


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
            if self.flat_sign[agent_idx] == action:
                self.rewards[agent_idx] += 0.5

        # the rest of the method is as follows:
        # info = dict(score=reward)
        # return self.observation, reward, True, False, info
        # write it in this cython version
        return self.observations, self.rewards, self.dones, self.scores