# cy_puffer_enduro.pyx
# cython: language_level=3

cimport numpy as cnp
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
from libc.time cimport time
from random import SystemRandom

rng = SystemRandom()

cdef extern from "enduro.h":
    # Structures
    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        float reward
        float step_rew_car_passed_no_crash
        float stay_on_road_reward
        float crashed_penalty
        float passed_cars
        float passed_by_enemy
        int cars_to_pass
        float days_completed
        float days_failed
        float collisions_player_vs_car
        float collisions_player_vs_road

    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx

    ctypedef struct Enduro:
        float* observations
        int* actions
        float* rewards
        unsigned char* terminals
        unsigned char* truncateds
        LogBuffer* log_buffer
        size_t obs_size
        int num_envs

    ctypedef struct Client:
        float width
        float height
        Enduro gameState

    # Function prototypes
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    Log aggregate_and_clear(LogBuffer* logs)
    void init(Enduro* env, int seed, int env_index)
    void c_reset(Enduro* env)
    void c_step(Enduro* env)
    void c_render(Client* client, Enduro* env)
    Client* make_client(Enduro* env)
    void close_client(Client* client, Enduro* env)

# Define Cython wrapper class
cdef class CyEnduro:
    cdef:
        Enduro* envs
        LogBuffer* logs
        Client* client
        int num_envs

    def __init__(self,
                 float[:, :] observations,
                 int[:] actions,
                 float[:] rewards,
                 cnp.uint8_t[:] terminals,
                 cnp.uint8_t[:] truncateds,
                 int num_envs):

        cdef int i
        cdef long t
        self.num_envs = num_envs

        # Allocate memory for environments
        self.envs = <Enduro*>calloc(num_envs, sizeof(Enduro))
        if not self.envs:
            raise MemoryError("Failed to allocate memory for environments")

        # Allocate memory for logs
        self.logs = allocate_logbuffer(num_envs)
        if not self.logs:
            free(self.envs)
            raise MemoryError("Failed to allocate memory for logs")

        # Generate a unique seed using high-resolution time and environment index
        from time import time as py_time  # Python time module for high-resolution time

        for i in range(num_envs):
            unique_seed = rng.randint(0, 2**32 - 1) & 0x7FFFFFFF
            memset(&self.envs[i], 0, sizeof(Enduro))
            self.envs[i].observations = &observations[i, 0]
            self.envs[i].actions = &actions[i]
            self.envs[i].rewards = &rewards[i]
            self.envs[i].terminals = &terminals[i]
            self.envs[i].truncateds = &truncateds[i]
            self.envs[i].log_buffer = self.logs
            self.envs[i].obs_size = observations.shape[1]

            #if i % 100 == 0:
            #    print(f"Initializing environment #{i} with seed {unique_seed}")

            init(&self.envs[i], unique_seed, i)

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        if not self.client:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(&self.envs[0])
            os.chdir(cwd)

        c_render(self.client, &self.envs[0])

    def close(self):
        if self.client:
            close_client(self.client, &self.envs[0])
        if self.envs:
            free(self.envs)
        if self.logs:
            free_logbuffer(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
