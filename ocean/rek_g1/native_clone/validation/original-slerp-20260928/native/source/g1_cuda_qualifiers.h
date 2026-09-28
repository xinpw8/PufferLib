#pragma once

/* Compile the same state-machine source for either the host or CUDA device. */
#if defined(REK_G1_CUDA_DEVICE)
#define REK_G1_FN __device__
#define REK_G1_CONSTANT __device__ __constant__
#define REK_G1_FIGHT_CONFIG_F84F1874 REK_G1_CUDA_FIGHT_CONFIG_F84F1874
#define REK_G1_FALL_CONFIG_F84F1874 REK_G1_CUDA_FALL_CONFIG_F84F1874
#else
#define REK_G1_FN
#define REK_G1_CONSTANT
#endif

#if defined(__cplusplus)
#define REK_G1_ZERO_INIT {}
#define REK_G1_STATIC_ASSERT static_assert
#else
#define REK_G1_ZERO_INIT {0}
#define REK_G1_STATIC_ASSERT _Static_assert
#endif
