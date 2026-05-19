#ifndef PUFFERLIB_PRECISION_H
#define PUFFERLIB_PRECISION_H

#ifdef __CUDACC__
#include <cuda_bf16.h>

#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

#endif

#endif // PUFFERLIB_PRECISION_H
