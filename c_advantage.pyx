# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False


import numpy as np
cimport numpy as cnp
from libc.string cimport memset, memcpy

def rewards_and_masks(float[:, :] reward_block, float[:, :] reward_mask,
        float[:] dones, float[:] rewards, int horizon):
    cdef int num_steps = len(rewards)
    memset(&reward_mask[0, 0], 0, num_steps * horizon * sizeof(float))
    cdef int i, j, t
    for i in range(num_steps):
        for j in range(horizon):
            t = i + j

            if t >= num_steps - 1:
                break

            if dones[t]:
                break

            reward_block[i, j] = rewards[t+1]
            reward_mask[i, j] = 1.0

def fast_rewards_and_masks(float[:, :] reward_block, float[:, :] reward_mask,
        float[:] dones, float[:] rewards, int horizon):
    cdef int num_steps = len(rewards)
    cdef int i, h 
    for i in range(num_steps):
        h = horizon
        if i + h >= num_steps:
            h = num_steps - i - 1

        memcpy(&reward_block[i, 0], &rewards[i+1], h * sizeof(float))

def compute_gae(cnp.ndarray dones, cnp.ndarray values,
        cnp.ndarray rewards, float gamma, float gae_lambda):
    '''Fast Cython implementation of Generalized Advantage Estimation (GAE)'''
    cdef int num_steps = len(rewards)
    cdef cnp.ndarray advantages = np.zeros(num_steps, dtype=np.float32)
    cdef float[:] c_advantages = advantages
    cdef float[:] c_dones = dones
    cdef float[:] c_values = values
    cdef float[:] c_rewards = rewards

    cdef float lastgaelam = 0
    cdef float nextnonterminal, delta
    cdef int t, t_cur, t_next
    for t in range(num_steps-1):
        t_cur = num_steps - 2 - t
        t_next = num_steps - 1 - t
        nextnonterminal = 1.0 - c_dones[t_next]
        delta = c_rewards[t_next] + gamma * c_values[t_next] * nextnonterminal - c_values[t_cur]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        c_advantages[t_cur] = lastgaelam

    return advantages
