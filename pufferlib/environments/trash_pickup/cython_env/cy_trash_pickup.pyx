cimport numpy as cnp
from libc.stdlib cimport malloc, free  # Import malloc and free
from libc.stdint cimport uint64_t
import os

cdef extern from "trash_pickup.h":
    ctypedef struct CTrashPickupEnv:
        float* observations
        int* actions
        float* rewards
        unsigned char* dones

    void initialize_env(CTrashPickupEnv* env, int grid_size, int num_agents, int num_trash, int num_bins, int max_steps)
    void reset_env(CTrashPickupEnv* env)
    void step_env(CTrashPickupEnv* env)
    bint is_episode_over(CTrashPickupEnv* env)
    void get_observations(CTrashPickupEnv* env)
    void free_env(CTrashPickupEnv* env)

cdef class CyTrashPickupEnv:
    cdef CTrashPickupEnv* env
    cdef int obs_size
    cdef int num_agents

    def __init__(self, int grid_size=10, int num_agents=3, int num_trash=15, int num_bins=2, int max_steps=300):
        # Allocate memory for the CTrashPickupEnv struct
        self.env = <CTrashPickupEnv*>malloc(sizeof(CTrashPickupEnv))
        if self.env == NULL:
            raise MemoryError("Failed to allocate memory for CTrashPickupEnv")

        initialize_env(self.env, grid_size, num_agents, num_trash, num_bins, max_steps)
        self.num_agents = num_agents

        # Calculate observation size
        self.obs_size = 3 + grid_size * grid_size

        # Allocate memory for observations, actions, rewards, and dones arrays in the env struct
        cdef int total_obs_size = self.num_agents * self.obs_size
        self.env.observations = <float*>malloc(total_obs_size * sizeof(float))
        self.env.actions = <int*>malloc(self.num_agents * sizeof(int))
        self.env.rewards = <float*>malloc(self.num_agents * sizeof(float))
        self.env.dones = <unsigned char*>malloc(self.num_agents * sizeof(unsigned char))

        if not (self.env.observations and self.env.actions and self.env.rewards and self.env.dones):
            self.__dealloc__()  # Free allocated memory
            raise MemoryError("Failed to allocate memory for environment arrays")

    def reset(self):
        reset_env(self.env)
        return self.get_observations()

    def step(self, actions):
        # Set actions
        cdef int i
        for i in range(self.num_agents):
            self.env.actions[i] = actions[f'agent_{i}']

        # Step the environment
        step_env(self.env)

        observations = self.get_observations()
        rewards = [self.env.rewards[i] for i in range(self.num_agents)]
        dones = [bool(self.env.dones[i]) for i in range(self.num_agents)]
        infos = [{} for _ in range(self.num_agents)]  # You can add additional info if needed

        return observations, rewards, dones, infos

    def get_observations(self):
        # Return observations as a list of dicts for each agent
        observations = {}
        cdef int i, j
        for i in range(self.num_agents):
            obs = {}
            idx = i * self.obs_size
            obs['agent_position'] = [self.env.observations[idx], self.env.observations[idx + 1]]
            obs['carrying_trash'] = self.env.observations[idx + 2]
            grid_data = []
            for j in range(idx + 3, idx + self.obs_size):
                grid_data.append(self.env.observations[j])
            obs['grid'] = grid_data  # You may need to reshape this into a 2D array
            observations[f"agent_{i}"] = obs
        return observations

    def __dealloc__(self):
        if self.env is not NULL:
            free_env(self.env)
            free(self.env.observations)
            free(self.env.actions)
            free(self.env.rewards)
            free(self.env.dones)
            free(self.env)
