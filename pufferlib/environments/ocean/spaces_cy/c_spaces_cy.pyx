#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
#include <numpy/arrayobject.h>
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
        float[:, :, :] buffero_image_observations  # 3D array (num_agents, obs_size, flattened_image_size + flat_size) (num_agents, 5, 5)
        signed char[:, :] buffero_flat_observations  # Change 'int[:, :]' to 'signed char[:, :]' to match np.int8
        unsigned char[:] dones
        float[:] rewards
        int[:] scores
        float[:] episodic_returns
        int num_agents
        int[:] image_sign
        int[:] flat_sign

    def __init__(self, 
                cnp.ndarray buffero_image_observations,
                cnp.ndarray buffero_flat_observations,
                cnp.ndarray rewards, 
                cnp.ndarray scores, 
                cnp.ndarray episodic_returns, 
                cnp.ndarray dones, 
                int num_agents,
    ):
        self.image_sign = np.zeros(num_agents, dtype=np.int32)
        self.flat_sign = np.zeros(num_agents, dtype=np.int32)

        self.buffero_image_observations = buffero_image_observations
        self.buffero_flat_observations = buffero_flat_observations
        self.rewards = rewards
        self.scores = scores
        self.episodic_returns = episodic_returns
        self.dones = dones
        self.num_agents = num_agents

        for agent_idx in range(self.num_agents):
            self.reset(agent_idx)
            
        # This function initializes the NumPy C API
        self.init_numpy()

    def init_numpy(self):
        cnp.import_array()

    cdef void compute_observations(self, int agent_idx):
        cdef float[:, :] image_obs = self.buffero_image_observations[agent_idx, :, :]  # Image buffer
        cdef signed char[:] flat_obs = self.buffero_flat_observations[agent_idx, :]  # Flat buffer
        cdef int i, j

        # Generate image observations
        for i in range(5):
            for j in range(5):
                image_obs[i, j] = np.random.randn()

        # Calculate the image sign (sum over the entire 5x5 observation)
        image_sum = np.sum(image_obs)
        self.image_sign[agent_idx] = 1 if image_sum > 0 else 0

        # Generate flat observations (ensure you stay within the bounds)
        for i in range(5):  # flat_obs has 5 elements
            flat_obs[i] = np.random.randint(-1, 2)

        # Calculate the flat sign (sum over the entire flat observation)
        flat_sum = np.sum(flat_obs)
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

        cdef int i, j, k
        cdef int flat_dim = self.buffero_flat_observations.shape[1]
        cdef int image_dim_1 = self.buffero_image_observations.shape[1]
        cdef int image_dim_2 = self.buffero_image_observations.shape[2]

        # Prepare space for the flattened observations
        cdef float[:, :] concatenated_observations = np.zeros(
            (self.num_agents, image_dim_1 * image_dim_2 + flat_dim),
            dtype=np.float32
        )

        # Concatenate manually
        for agent_idx in range(self.num_agents):
            # Flatten the image observations manually
            for i in range(image_dim_1):
                for j in range(image_dim_2):
                    concatenated_observations[agent_idx, i * image_dim_2 + j] = self.buffero_image_observations[agent_idx, i, j]

            # Append the flat observations
            for k in range(flat_dim):
                concatenated_observations[agent_idx, image_dim_1 * image_dim_2 + k] = self.buffero_flat_observations[agent_idx, k]

        # Process actions and rewards
        for agent_idx in range(self.num_agents):
            action = actions[agent_idx]
            if self.image_sign[agent_idx] == action:
                self.rewards[agent_idx] += 0.5
            if self.flat_sign[agent_idx] == action:
                self.rewards[agent_idx] += 0.5

        return concatenated_observations, self.rewards, self.dones, self.scores



        # # info = dict(score=reward)

        # return self.buffero_observations, self.rewards, self.dones, self.scores