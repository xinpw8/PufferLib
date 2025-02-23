# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp

def rewards_and_masks(float[:] dones, float[:] rewards, int horizon):
    '''Fast Cython implementation of Generalized Advantage Estimation (GAE)'''
    cdef int num_steps = len(rewards)
    cdef cnp.ndarray reward_block = np.zeros((num_steps, horizon), dtype=np.float32)
    cdef cnp.ndarray reward_mask = np.zeros((num_steps, horizon), dtype=np.float32)
    cdef float[:, :] c_reward_block = reward_block
    cdef float[:, :] c_reward_mask = reward_mask

    cdef float nextnonterminal,
    cdef int i, j, t
    for i in range(num_steps):
        for j in range(horizon):
            t = i + j

            if t >= num_steps:
                break

            if dones[t]:
                break

            c_reward_block[i, j] = rewards[t+1]
            c_reward_mask[i, j] = 1.0

    return reward_block, reward_mask


