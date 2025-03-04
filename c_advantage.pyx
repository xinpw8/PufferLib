# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.string cimport memset, memcpy
from libc.math cimport fmaxf, fminf, expf
import cython


@cython.profile(True)
def rewards_and_masks(
        float[:, :] reward_block,
        float[:, :] reward_mask,
        float[:, :] values_mean,
        float[:, :] values_std,
        float[:, :] buf,
        float[:] dones,
        float[:] rewards,
        float[:] advantages,
        int[:] bounds,
        int horizon
    ):

    cdef int num_steps = len(rewards)
    memset(&reward_mask[0, 0], 0, num_steps * horizon * sizeof(float))
    memset(&advantages[0], 0, num_steps * sizeof(float))

    cdef:
        float vstd_max = -1e10
        float vstd_min = 1e10
        int i, j, k, t
        float r

    for i in range(num_steps):
        k = 0
        for j in range(horizon):
            t = i + j

            if dones[t]:
                break

            if t >= num_steps - 1:
                break

            k += 1

            reward_block[i, j] = rewards[t+1]
            reward_mask[i, j] = 1.0

            # Store value std in buffer
            #vstd = values_logstd[i, j]
            #vstd = vstd if vstd < 10 else 10
            #vstd = vstd if vstd > -10 else -10
            #vstd = expf(vstd)
            vstd = values_std[i, j]
            buf[i, j] = vstd

            # Online max and min
            vstd_max = vstd_max if vstd_max > vstd else vstd
            vstd_min = vstd_min if vstd_min < vstd else vstd

        bounds[i] = k

    cdef float delta = vstd_min - vstd_max
    if delta == 0:
        for i in range(num_steps):
            k = bounds[i]
            for j in range(k):
                advantages[i] += (reward_block[i, j] - values_mean[i, j])

        return

    cdef float adv_scale, adv_sum
    for i in range(num_steps):
        k = bounds[i]
        adv_sum = 0
        for j in range(k):
            adv_scale = (vstd_max - buf[i, j]) / delta
            adv_scale = adv_scale if adv_scale > 0.05 else 0.05
            adv_scale = adv_scale if adv_scale < 1 else 1
            buf[i, j] = adv_scale
            adv_sum += adv_scale

        for j in range(k):
            advantages[i] += buf[i, j]/adv_sum * (reward_block[i, j] - values_mean[i, j])
 
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
