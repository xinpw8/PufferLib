# py_enduro_clone.pyx

cimport numpy as cnp
from libc.stdlib cimport calloc, free
from libc.stdint cimport int32_t
import os

cdef extern from "enduro_clone.h":
    
    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        float days_completed
        float days_failed
        float collisions_player_vs_car
        float collisions_player_vs_road
        float reward
        
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
        Log log

    LogBuffer* allocate_logbuffer(int size)
    Log aggregate_and_clear(LogBuffer* logs)
    void init(Enduro* env)
    void reset(Enduro* env)
    void c_step(Enduro* env)

cdef class CyEnduro:
    cdef:
        Enduro* envs
        LogBuffer* logs
        Log log
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
                float[:] rewards, unsigned char[:] terminals,
                unsigned char[:] truncateds, int num_envs):

        self.num_envs = num_envs
        self.envs = <Enduro*> calloc(num_envs, sizeof(Enduro))
        self.logs = allocate_logbuffer(num_envs)

        cdef int i
        cdef Enduro* env
        for i in range(num_envs):
            env = &self.envs[i]
            env.observations = &observations[i, 0]
            env.actions = &actions[i]
            env.rewards = &rewards[i]
            env.terminals = &terminals[i]
            env.truncateds = &truncateds[i]
            env.log_buffer = self.logs
            init(env)
            
    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])
            
    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])
            
    def close(self):
        free(self.logs)
        free(self.envs)
        
    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return {
            "episode_return": log.episode_return,
            "episode_length": log.episode_length,
            "score": log.score,
            "days_completed": log.days_completed,
            "days_failed": log.days_failed,
            "collisions_player_vs_car": log.collisions_player_vs_car,
            "collisions_player_vs_road": log.collisions_player_vs_road,
            "reward": log.reward,
            "passed_cars": log.episode_return
        }